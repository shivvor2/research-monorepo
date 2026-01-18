"""
PyTorch Lightning module for training with two optimizers.

This module provides a LightningModule that supports training with two
optimizers (e.g., Muon for weight matrices, AdamW for embeddings) with
independent scheduling for each.

Key Features:
    - Manual optimization for multiple optimizer support
    - Independent LR and param_group scheduling per optimizer
    - Gradient accumulation with proper loss scaling
    - Configurable gradient clipping
    - Automatic param_group value updates (momentum, etc.)

Design Decisions:
    1. Uses manual optimization (self.automatic_optimization = False) because
       Lightning's automatic optimization doesn't support multiple optimizers
       with different stepping patterns.

    2. Optimizer and scheduler configs are passed via dependency injection
       (OptimizerConfig dataclasses) for flexibility and serializability.

    3. Parameter partitioning uses name-based patterns (target_modules) rather
       than shape-based heuristics because embeddings are 2D but shouldn't
       use Muon.

    4. Logging uses Lightning's standard `self.log()` interface, which is
       logger-agnostic. Users configure their preferred logger (WandB,
       TensorBoard, etc.) at the Trainer level.

    5. Checkpointing is handled by Lightning callbacks, not by this module.
       Users configure checkpointing at the Trainer level.

Example:
    Basic usage::

        from research_lib.training import (
            DualOptimizerModule,
            TrainingConfig,
            default_muon_config,
            default_adam_config,
        )

        module = DualOptimizerModule(
            model=my_model,
            training_config=TrainingConfig(total_steps=10000),
            matrix_optimizer_config=default_muon_config(),
            vector_optimizer_config=default_adam_config(),
            target_modules=["attn", "mlp", "qkv", "c_fc", "c_proj"],
        )

        # Configure logging and checkpointing at Trainer level
        from lightning.pytorch.loggers import WandbLogger
        from lightning.pytorch.callbacks import ModelCheckpoint

        trainer = L.Trainer(
            max_steps=10000,
            logger=WandbLogger(project="my-project"),
            callbacks=[ModelCheckpoint(dirpath="checkpoints/", monitor="val/loss")],
        )
        trainer.fit(module, train_dataloader)

Possible Extensions:
    - Support for different loss functions via constructor argument
    - Hooks for custom training logic (e.g., EMA updates)
    - Integration with learning rate finders

See Also:
    - :mod:`research_lib.training.configs` for configuration dataclasses
    - :mod:`research_lib.training.param_utils` for parameter partitioning
    - :mod:`research_lib.training.scheduling` for param_group scheduling
"""

from typing import Any, Dict, List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..configs import OptimizerConfig, TrainingConfig
from ..param_utils import partition_parameters, summarize_partition
from ..scheduling import (
    get_current_lr,
    get_param_group_value,
    update_param_group_schedules,
)


class DualOptimizerModule(L.LightningModule):
    """Lightning module for training with two optimizers.

    This module handles:
        - Parameter partitioning based on target_modules
        - Dual optimizer setup (matrix optimizer + vector optimizer)
        - Independent LR scheduling per optimizer
        - Custom param_group scheduling (momentum, etc.)
        - Gradient accumulation with proper scaling
        - Gradient clipping

    The module is model-agnostic and can be used with any nn.Module that
    accepts input_ids and returns logits (e.g., GPT, BERT, encoder-decoder).

    Attributes:
        model: The neural network model to train.
        training_config: Global training configuration.
        matrix_optimizer_config: Config for the matrix/weight optimizer (e.g., Muon).
        vector_optimizer_config: Config for the vector/embedding optimizer (e.g., AdamW).
        target_modules: Patterns for selecting matrix optimizer parameters.

    Note:
        This module uses manual optimization. The training_step handles:
        1. Forward pass and loss computation
        2. Scaled backward pass (for gradient accumulation)
        3. Gradient clipping (if configured)
        4. Optimizer stepping (when accumulation is complete)
        5. Scheduler stepping and param_group updates

    Note:
        Logging and checkpointing are configured at the Trainer level, not here.
        This module uses `self.log()` which routes to whatever logger is configured.
    """

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        matrix_optimizer_config: OptimizerConfig,
        vector_optimizer_config: OptimizerConfig,
        target_modules: List[str],
    ):
        """Initialize the Lightning module.

        Args:
            model: The model to train. Can be any nn.Module with a forward
                method that returns logits.
            training_config: Global training parameters (total_steps, grad_accum, etc.)
            matrix_optimizer_config: Configuration for the optimizer handling
                parameters matching target_modules (typically Muon).
            vector_optimizer_config: Configuration for the optimizer handling
                all other parameters (typically AdamW).
            target_modules: List of substrings to match against parameter names.
                Parameters containing any of these patterns use matrix_optimizer.
                All other parameters use vector_optimizer.
                Pass empty list [] for single-optimizer (vector only) training.

        Example:
            >>> module = DualOptimizerModule(
            ...     model=MyModel(),
            ...     training_config=TrainingConfig(total_steps=10000),
            ...     matrix_optimizer_config=default_muon_config(),
            ...     vector_optimizer_config=default_adam_config(),
            ...     target_modules=["attn", "mlp"],
            ... )
        """
        super().__init__()

        # CRITICAL: Enable manual optimization for multiple optimizers
        self.automatic_optimization = False

        # Store configs (not hyperparameters to avoid serialization issues with classes)
        self.training_config = training_config
        self.matrix_optimizer_config = matrix_optimizer_config
        self.vector_optimizer_config = vector_optimizer_config
        self.target_modules = target_modules

        # Store model
        self.model = model

        # Track optimization step count (different from global_step with grad accum)
        self._optimizer_step_count = 0

        # Save hyperparameters for logging (exclude model and configs with classes)
        self.save_hyperparameters(
            ignore=["model", "matrix_optimizer_config", "vector_optimizer_config"]
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        """Configure optimizers and LR schedulers.

        Returns:
            Tuple of (optimizers_list, schedulers_list).
        """
        # Partition parameters
        matrix_params, vector_params = partition_parameters(
            self.model, self.target_modules
        )

        # Log partition summary
        if self.trainer is not None:
            try:
                if self.trainer.is_global_zero:
                    summary = summarize_partition(self.model, self.target_modules)
                    self.print(summary)
            except AttributeError:
                # Trainer not fully initialized (e.g., during testing)
                pass

        # Build optimizers
        optimizers = []
        schedulers = []

        # Matrix optimizer (e.g., Muon) - skip if no target params
        if matrix_params:
            matrix_opt = self.matrix_optimizer_config.build_optimizer(matrix_params)
            matrix_sch = self.matrix_optimizer_config.build_scheduler(
                matrix_opt, self.training_config
            )
            optimizers.append(matrix_opt)
            schedulers.append(matrix_sch)
        else:
            # No matrix params - still need placeholders for indexing
            optimizers.append(None)
            schedulers.append(None)

        # Vector optimizer (e.g., AdamW) - skip if no other params
        if vector_params:
            vector_opt = self.vector_optimizer_config.build_optimizer(vector_params)
            vector_sch = self.vector_optimizer_config.build_scheduler(
                vector_opt, self.training_config
            )
            optimizers.append(vector_opt)
            schedulers.append(vector_sch)
        else:
            optimizers.append(None)
            schedulers.append(None)

        # Filter out None values
        optimizers = [o for o in optimizers if o is not None]
        schedulers = [s for s in schedulers if s is not None]

        return optimizers, schedulers

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute a single training step.

        Handles gradient accumulation, optimizer stepping, and scheduling.

        Args:
            batch: Dictionary containing 'input_ids' and optionally 'labels'.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            The unscaled loss tensor for logging.
        """
        # Get optimizers and schedulers
        opts = self.optimizers()
        schs = self.lr_schedulers()

        # Handle single optimizer case
        if not isinstance(opts, list):
            opts = [opts]
        if not isinstance(schs, list):
            schs = [schs]

        # Determine if we should step this batch
        grad_accum = self.training_config.grad_accum_steps
        should_step = (batch_idx + 1) % grad_accum == 0

        # Extract inputs
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # Forward pass
        logits = self.forward(input_ids)

        # Compute loss (causal LM: predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum

        # Backward pass
        self.manual_backward(scaled_loss)

        # Step optimizers if accumulation complete
        if should_step:
            # Gradient clipping
            if self.training_config.gradient_clip_val > 0:
                for opt in opts:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.training_config.gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )

            # Step all optimizers
            for opt in opts:
                opt.step()
                opt.zero_grad()

            # Step all schedulers
            for sch in schs:
                sch.step()

            # Update custom param_group schedules (momentum, etc.)
            self._update_custom_schedules(opts)

            # Increment step counter
            self._optimizer_step_count += 1

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/optimizer_step", float(self._optimizer_step_count), on_step=True
        )

        # Log LRs
        for i, opt in enumerate(opts):
            lr = get_current_lr(opt)
            self.log(f"train/lr_{i}", lr, on_step=True)

            # Log momentum if present
            momentum = get_param_group_value(opt, "momentum")
            if momentum is not None:
                self.log(f"train/momentum_{i}", momentum, on_step=True)

        return loss

    def _update_custom_schedules(self, optimizers: List[Optimizer]) -> None:
        """Update custom param_group schedules for all optimizers.

        Args:
            optimizers: List of optimizers to update.
        """
        step = self._optimizer_step_count
        total_steps = self.training_config.total_steps

        # Determine which config goes with which optimizer
        matrix_params, vector_params = partition_parameters(
            self.model, self.target_modules
        )

        opt_idx = 0

        # Matrix optimizer schedules
        if matrix_params and opt_idx < len(optimizers):
            update_param_group_schedules(
                optimizers[opt_idx],
                self.matrix_optimizer_config.scheduler_config,
                step,
                total_steps,
            )
            opt_idx += 1

        # Vector optimizer schedules
        if vector_params and opt_idx < len(optimizers):
            update_param_group_schedules(
                optimizers[opt_idx],
                self.vector_optimizer_config.scheduler_config,
                step,
                total_steps,
            )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute a single validation step.

        Args:
            batch: Dictionary containing 'input_ids' and optionally 'labels'.
            batch_idx: Index of the current batch.

        Returns:
            The validation loss tensor.
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        with torch.no_grad():
            logits = self.forward(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_train_start(self) -> None:
        """Log model information at the start of training."""
        # Log parameter counts
        matrix_params, vector_params = partition_parameters(
            self.model, self.target_modules
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        matrix_count = sum(p.numel() for p in matrix_params)
        vector_count = sum(p.numel() for p in vector_params)

        self.log("model/total_params", float(total_params))
        self.log("model/matrix_params", float(matrix_count))
        self.log("model/vector_params", float(vector_count))
