"""
PyTorch Lightning module for training with two optimizers.

This module provides a LightningModule that supports training with two
optimizers (e.g., Muon for weight matrices, AdamW for embeddings) with
independent scheduling for each.

Design Decisions:
    1. Uses manual optimization (self.automatic_optimization = False) because
       Lightning's automatic optimization doesn't support multiple optimizers
       with different stepping patterns.

    2. **Trainer is the config surface**: Training loop params (max_steps,
       accumulate_grad_batches) are read from Trainer at runtime, not from
       custom config classes. Gradient clipping is the exception because
       Lightning disables it when manual optimization happens

    3. **Late binding**: ParamSchedulers are created in on_fit_start() after
       the Trainer is attached and optimizers exist.

    4. **Proper checkpointing**: Scheduler state is saved/restored via
       on_save_checkpoint/on_load_checkpoint hooks.

    5. GradAccumSchedule optionally overrides trainer.accumulate_grad_batches
       with warning on conflict.

    6. Parameter partitioning uses name-based patterns (target_modules) rather
       than shape-based heuristics.

Example:
    Basic usage::

        from research_lib.training import (
            DualOptimizerModule,
            default_muon_config,
            default_adamw_config,
        )

        muon_opt, muon_sched = default_muon_config()
        adamw_opt, adamw_sched = default_adamw_config()

        module = DualOptimizerModule(
            model=my_model,
            matrix_optimizer_config=muon_opt,
            vector_optimizer_config=adamw_opt,
            matrix_schedule_config=muon_sched,
            vector_schedule_config=adamw_sched,
            matrix_target_modules=["attn", "mlp", "qkv", "c_fc", "c_proj"],
        )

        trainer = L.Trainer(
            max_steps=10000,
            accumulate_grad_batches=4,
            gradient_clip_val=1.0,
            logger=WandbLogger(project="my-project"),
            callbacks=[ModelCheckpoint(dirpath="checkpoints/", monitor="val/loss")],
        )
        trainer.fit(module, train_dataloader)

    Custom loss function via subclassing::

        class DistillationModule(DualOptimizerModule):
            def __init__(self, teacher_model, alpha=0.5, **kwargs):
                super().__init__(**kwargs)
                self.teacher = teacher_model
                self.alpha = alpha

            def compute_loss(self, model_output, batch):
                ce_loss = super().compute_loss(model_output, batch)
                with torch.no_grad():
                    teacher_logits = self.teacher(batch["input_ids"])
                kl_loss = F.kl_div(
                    F.log_softmax(model_output, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                )
                return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

See Also:
    - :mod:`research_lib.training.configs` for OptimizerConfig and ScheduleConfig
    - :mod:`research_lib.training.presets` for default_muon_config, etc.
    - :mod:`research_lib.training.param_utils` for parameter partitioning
    - :mod:`research_lib.training.scheduling` for ParamScheduler
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch.optim import Optimizer

from ..configs import (
    GradAccumSchedule,
    OptimizerConfig,
    ScheduleConfig,
    build_optimizer,
)
from ..param_utils import partition_parameters, summarize_partition
from ..scheduling import ParamScheduler
from ..scheduling.utils import get_current_lr, get_param_value


class DualOptimizerModule(L.LightningModule):
    """Lightning module for training with two optimizers.

    Training parameters (max_steps, accumulate_grad_batches) are read from the
    Trainer at runtime. Do not duplicate them in configs.

    Gradient clipping must be configured on this
    module because Lightning disables automatic clipping when manual
    optimization is enabled.

    This module handles:
        - Parameter partitioning based on target_modules
        - Dual optimizer setup (matrix optimizer + vector optimizer)
        - Independent scheduling per optimizer via ParamScheduler
        - Gradient accumulation (from Trainer or GradAccumSchedule override)
        - Gradient clipping (from Trainer)

    Partitioning Logic:
        You can specify which parameters belong to which optimizer using EITHER
        `matrix_target_modules` OR `vector_target_modules` (mutually exclusive).

        - If `matrix_target_modules` is set: Matched params → Matrix Optimizer,
          remaining params → Vector Optimizer.
        - If `vector_target_modules` is set: Matched params → Vector Optimizer,
          remaining params → Matrix Optimizer.
        - If neither is set: All params → Vector Optimizer (single optimizer mode).

    Attributes:
        model: The neural network model to train.
        matrix_optimizer_config: Config for the matrix/weight optimizer (e.g., Muon).
        vector_optimizer_config: Config for the vector/embedding optimizer (e.g., AdamW).
        matrix_schedule_config: Parameter schedules for matrix optimizer.
        vector_schedule_config: Parameter schedules for vector optimizer.
        grad_accum_schedule: Optional override for trainer.accumulate_grad_batches.
        gradient
        grad_clip_val: Value in which the gradient is clipped
        grad_clip_algo: Algorithm used for clipping, either "norm" or "value"

    Note:
        Lightning's ``global_step`` counter increments on each ``optimizer.step()``
        call. With multiple optimizers, this would cause ``max_steps`` to be reached
        prematurely. This module avoids that by stepping secondary optimizers
        directly, so ``max_steps`` corresponds to actual training steps.
    """

    def __init__(
        self,
        model: nn.Module,
        matrix_optimizer_config: OptimizerConfig,
        vector_optimizer_config: OptimizerConfig,
        matrix_schedule_config: ScheduleConfig,
        vector_schedule_config: ScheduleConfig,
        matrix_target_modules: Optional[List[str]] = None,
        vector_target_modules: Optional[List[str]] = None,
        grad_accum: Optional[int] = None,
        grad_accum_schedule: Optional[GradAccumSchedule] = None,
        grad_clip_val: Optional[Union[int, float]] = None,
        grad_clip_algo: str = "norm",
    ) -> None:
        """Initialize the dual optimizer module.

        Args:
            model: The model to train. Can be any nn.Module with a forward
                method that returns logits.
            matrix_optimizer_config: Optimizer config for matrix params.
            vector_optimizer_config: Optimizer config for vector params.
            matrix_schedule_config: Schedule config for matrix optimizer.
            vector_schedule_config: Schedule config for vector optimizer.
            matrix_target_modules: Patterns for parameters that MUST go to the
                matrix optimizer. Remainder goes to vector optimizer.
            vector_target_modules: Patterns for parameters that MUST go to the
                vector optimizer. Remainder goes to matrix optimizer.
            grad_accum: Constant gradient accumulation factor. Mutually exclusive
                with grad_accum_schedule. If neither is provided, defaults to 1.
            grad_accum_schedule: Step-based gradient accumulation schedule.
                Mutually exclusive with grad_accum.
            grad_clip_val: Optional gradient-clipping value. Default: None.
            grad_clip_algo: "norm" or "value". Default: "norm".

        Raises:
            ValueError: If both matrix_target_modules and vector_target_modules provided.
            ValueError: If both grad_accum and grad_accum_schedule provided.

        Example:
            >>> muon_opt, muon_sched = default_muon_config()
            >>> adamw_opt, adamw_sched = default_adamw_config()
            >>> # Simple constant accumulation
            >>> module = DualOptimizerModule(
            ...     model=MyModel(),
            ...     matrix_optimizer_config=muon_opt,
            ...     vector_optimizer_config=adamw_opt,
            ...     matrix_schedule_config=muon_sched,
            ...     vector_schedule_config=adamw_sched,
            ...     matrix_target_modules=["attn", "mlp"],
            ...     grad_accum=4,
            ... )
            >>> # Or with a schedule
            >>> module = DualOptimizerModule(
            ...     ...
            ...     grad_accum_schedule=GradAccumSchedule({0: 1, 1000: 4}),
            ... )

        Note:
            Since this module uses manual optimization (required for multiple
            optimizers), Lightning's ``Trainer(accumulate_grad_batches=N)`` is
            NOT supported. Use ``grad_accum`` or ``grad_accum_schedule`` instead.
        """
        super().__init__()

        # CRITICAL: Enable manual optimization for multiple optimizers
        self.automatic_optimization = False

        # Store model
        self.model = model

        # Store configs (pure data, no runtime dependencies)
        self.matrix_optimizer_config = matrix_optimizer_config
        self.vector_optimizer_config = vector_optimizer_config
        self.matrix_schedule_config = matrix_schedule_config
        self.vector_schedule_config = vector_schedule_config

        # Validate target module args (mutually exclusive)
        if matrix_target_modules is not None and vector_target_modules is not None:
            raise ValueError(
                "Cannot specify both `matrix_target_modules` and `vector_target_modules`. "
                "Please select one targeting strategy."
            )

        if vector_target_modules is not None:
            self.target_modules = vector_target_modules
            self._target_strategy = "vector"
        else:
            # Default to matrix strategy (empty list if None provided)
            self.target_modules = (
                matrix_target_modules if matrix_target_modules is not None else []
            )
            self._target_strategy = "matrix"

        # Validate grad accum args (mutually exclusive)
        if grad_accum is not None and grad_accum_schedule is not None:
            raise ValueError(
                "Cannot specify both `grad_accum` and `grad_accum_schedule`. "
                "Use `grad_accum` for constant accumulation, or "
                "`grad_accum_schedule` for step-based schedules."
            )

        # Build grad accum schedule
        if grad_accum_schedule is not None:
            self._grad_accum_schedule = grad_accum_schedule
        elif grad_accum is not None:
            if grad_accum < 1:
                raise ValueError(f"grad_accum must be >= 1, got {grad_accum}")
            self._grad_accum_schedule = GradAccumSchedule({0: grad_accum})
        else:
            # Default: no accumulation
            self._grad_accum_schedule = GradAccumSchedule({0: 1})

        # Runtime state (initialized in on_fit_start)
        self._optimizer_step_count = 0
        self._matrix_scheduler: Optional[ParamScheduler] = None
        self._vector_scheduler: Optional[ParamScheduler] = None
        self._has_matrix_params = False
        self._has_vector_params = False

        # Pending scheduler states from checkpoint (loaded in on_fit_start)
        self._pending_scheduler_states: Optional[Dict[str, Any]] = None

        # Gradient clipping
        self.gradient_clip_value = grad_clip_val
        self.gradient_clip_algorithm = grad_clip_algo

        # Save hyperparameters for logging (exclude model and configs with classes)
        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self) -> List[Optimizer]:
        """Configure optimizers based on parameter partitioning.

        Returns:
            List of optimizers. May contain 1 or 2 optimizers depending on
            parameter partitioning.
        """
        # Partition parameters
        matched_params, other_params = partition_parameters(
            self.model, self.target_modules
        )

        # Route parameters based on strategy
        if self._target_strategy == "vector":
            # Matches -> Vector, Others -> Matrix
            vector_params = matched_params
            matrix_params = other_params
        else:
            # Matches -> Matrix, Others -> Vector (Default)
            matrix_params = matched_params
            vector_params = other_params

        self._has_matrix_params = len(matrix_params) > 0
        self._has_vector_params = len(vector_params) > 0

        # Log partition summary on rank 0
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

        if self._has_matrix_params:
            matrix_opt = build_optimizer(self.matrix_optimizer_config, matrix_params)
            optimizers.append(matrix_opt)

        if self._has_vector_params:
            vector_opt = build_optimizer(self.vector_optimizer_config, vector_params)
            optimizers.append(vector_opt)

        return optimizers

    def on_fit_start(self) -> None:
        """Initialize ParamSchedulers after trainer is attached.

        This is where late binding happens:
        - Get total_steps from trainer.estimated_stepping_batches
        - Create ParamSchedulers for each optimizer
        - Restore scheduler states if resuming from checkpoint
        """

        # Get total steps from trainer (this is the key late binding)
        total_steps = self.trainer.estimated_stepping_batches

        # Get optimizers
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        # Create ParamSchedulers (binding schedules to optimizers)
        # Extract schedules from ScheduleConfig and pass to ParamScheduler
        opt_idx = 0

        if self._has_matrix_params:
            self._matrix_scheduler = ParamScheduler(
                optimizer=opts[opt_idx],
                global_schedules=self.matrix_schedule_config.global_schedules,
                total_steps=total_steps,
                group_overrides=self.matrix_schedule_config.group_overrides,
            )
            opt_idx += 1

        if self._has_vector_params:
            self._vector_scheduler = ParamScheduler(
                optimizer=opts[opt_idx],
                global_schedules=self.vector_schedule_config.global_schedules,
                total_steps=total_steps,
                group_overrides=self.vector_schedule_config.group_overrides,
            )

        # Restore scheduler states from checkpoint if available
        if self._pending_scheduler_states is not None:
            if (
                self._matrix_scheduler is not None
                and self._pending_scheduler_states.get("matrix") is not None
            ):
                self._matrix_scheduler.load_state_dict(
                    self._pending_scheduler_states["matrix"]
                )
            if (
                self._vector_scheduler is not None
                and self._pending_scheduler_states.get("vector") is not None
            ):
                self._vector_scheduler.load_state_dict(
                    self._pending_scheduler_states["vector"]
                )
            self._pending_scheduler_states = None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save scheduler states to checkpoint.

        Args:
            checkpoint: The checkpoint dictionary to save state into.
        """
        checkpoint["optimizer_step_count"] = self._optimizer_step_count
        checkpoint["matrix_scheduler_state"] = (
            self._matrix_scheduler.state_dict() if self._matrix_scheduler else None
        )
        checkpoint["vector_scheduler_state"] = (
            self._vector_scheduler.state_dict() if self._vector_scheduler else None
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load scheduler states from checkpoint.

        Note:
            Actual scheduler state restoration happens in on_fit_start()
            because schedulers don't exist yet at this point.

        Args:
            checkpoint: The checkpoint dictionary to load state from.
        """
        self._optimizer_step_count = checkpoint.get("optimizer_step_count", 0)
        self._pending_scheduler_states = {
            "matrix": checkpoint.get("matrix_scheduler_state"),
            "vector": checkpoint.get("vector_scheduler_state"),
        }

    def _get_current_grad_accum(self) -> int:
        """Get current gradient accumulation factor.

        Returns:
            The gradient accumulation factor for the current step.
            Uses grad_accum_schedule if provided, otherwise uses Trainer value.
        """
        return self._grad_accum_schedule.get_accum(self._optimizer_step_count)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            **kwargs: Additional arguments passed to the model.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.model(input_ids, **kwargs)

    def compute_loss(
        self,
        model_output: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the training loss.

        Override this method for custom loss functions (e.g., distillation,
        contrastive learning, auxiliary losses, non-LM tasks).

        The default implementation assumes causal language modeling:
        cross-entropy loss on next-token prediction with shifted logits/labels.

        Args:
            model_output: Output from self.forward(). For the default implementation,
                this should be logits of shape (batch_size, seq_len, vocab_size).
            batch: The full batch dictionary. Default implementation uses
                batch["labels"] if present, otherwise batch["input_ids"] as labels.

        Returns:
            Scalar loss tensor.

        Example:
            Subclass for distillation::

                class DistillationModule(DualOptimizerModule):
                    def __init__(self, teacher, alpha=0.5, **kwargs):
                        super().__init__(**kwargs)
                        self.teacher = teacher
                        self.alpha = alpha

                    def compute_loss(self, model_output, batch):
                        ce_loss = super().compute_loss(model_output, batch)
                        with torch.no_grad():
                            teacher_out = self.teacher(batch["input_ids"])
                        kl_loss = compute_kl_divergence(model_output, teacher_out)
                        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        """
        logits = model_output

        # Prefer Labels keys, fallback to input_ids (auto_regressive)
        if "labels" in batch:
            labels = batch["labels"]
        elif "input_ids" in batch:
            labels = batch["input_ids"]
        else:
            raise KeyError("Batch must contain 'labels' or 'input_ids' for Causal LM")

        # Causal LM: predict next token (shift by 1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute a single training step.

        Handles gradient accumulation, optimizer stepping, and scheduling.

        Args:
            batch: Dictionary containing at minimum 'input_ids'. May also contain
                'labels' and other keys accessible in compute_loss().
            batch_idx: Index of the current batch within the epoch.

        Returns:
            The unscaled loss tensor for logging.
        """
        # Get optimizers
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        # Determine if we should step this batch
        grad_accum = self._get_current_grad_accum()
        should_step = (batch_idx + 1) % grad_accum == 0

        # Forward pass
        model_output = self.forward(**batch)

        # Compute loss via (overridable) method
        loss = self.compute_loss(model_output, batch)

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum

        # Backward pass
        self.manual_backward(scaled_loss)

        # Step optimizers if accumulation complete
        if should_step:
            # Gradient clipping via trainer
            if self.gradient_clip_value is not None and self.gradient_clip_value > 0:
                for opt in opts:
                    self.clip_gradients(
                        opt,  # lightning supports passing list of optimizers
                        gradient_clip_val=self.gradient_clip_value,
                        gradient_clip_algorithm=self.gradient_clip_algorithm,
                    )

            # Step all optimizers - only first one should increment global_step
            for i, opt in enumerate(opts):
                if i == 0:
                    opt.step()  # This increments global_step
                else:
                    # Access underlying optimizer directly to avoid double-counting
                    opt.optimizer.step()
                opt.zero_grad()

            # Step param schedulers
            if self._matrix_scheduler is not None:
                self._matrix_scheduler.step()
            if self._vector_scheduler is not None:
                self._vector_scheduler.step()

            # Increment step counter
            self._optimizer_step_count += 1

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grad_accum", float(grad_accum), on_step=True)
        self.log(
            "train/optimizer_step", float(self._optimizer_step_count), on_step=True
        )

        # Log LRs and momentum
        for i, opt in enumerate(opts):
            lr = get_current_lr(opt)
            self.log(f"train/lr_{i}", lr, on_step=True)

            # Log momentum if present
            momentum = get_param_value(opt, "momentum")
            if momentum is not None:
                self.log(f"train/momentum_{i}", momentum, on_step=True)

        return loss

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

        with torch.no_grad():
            model_output = self.forward(input_ids)
            loss = self.compute_loss(model_output, batch)

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
        matched_params, other_params = partition_parameters(
            self.model, self.target_modules
        )

        if self._target_strategy == "vector":
            matrix_params = other_params
            vector_params = matched_params
        else:
            matrix_params = matched_params
            vector_params = other_params

        total_params = sum(p.numel() for p in self.model.parameters())
        matrix_count = sum(p.numel() for p in matrix_params)
        vector_count = sum(p.numel() for p in vector_params)

        self.log("model/total_params", float(total_params))
        self.log("model/matrix_params", float(matrix_count))
        self.log("model/vector_params", float(vector_count))
