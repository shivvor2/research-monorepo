"""
PyTorch Lightning modules for training with custom optimizer configurations.

This subpackage provides LightningModule implementations for various
optimizer configurations:

- :class:`DualOptimizerModule`: Training with two optimizers (e.g., Muon + AdamW)

These modules integrate with:
- The configuration system in :mod:`research_lib.training.configs`
- The scheduling utilities in :mod:`research_lib.training.scheduling`
- The presets in :mod:`research_lib.training.presets`

Design Philosophy:
    - **Trainer is the config surface**: Training loop params (max_steps,
      gradient_clip_val, accumulate_grad_batches) come from the Trainer
    - **Late binding**: ParamSchedulers created in on_fit_start() after Trainer attached
    - **Proper checkpointing**: Scheduler state saved/restored via Lightning hooks

Note on Logging and Checkpointing:
    These modules use Lightning's standard `self.log()` interface and do not
    hardcode any specific logging or checkpointing backends. Users configure
    their preferred systems at the Trainer level::

        from lightning.pytorch.loggers import WandbLogger
        from lightning.pytorch.callbacks import ModelCheckpoint

        logger = WandbLogger(project="my-project")
        checkpoint = ModelCheckpoint(dirpath="checkpoints/", monitor="val/loss")

        trainer = L.Trainer(logger=logger, callbacks=[checkpoint])
        trainer.fit(module, dataloader)

Example:
    Basic usage with presets::

        from research_lib.training.modules import DualOptimizerModule
        from research_lib.training.presets import default_muon_config, default_adamw_config

        muon_opt, muon_sched = default_muon_config(lr=0.02)
        adamw_opt, adamw_sched = default_adamw_config(lr=3e-4)

        module = DualOptimizerModule(
            model=my_model,
            matrix_optimizer_config=muon_opt,
            vector_optimizer_config=adamw_opt,
            matrix_schedule_config=muon_sched,
            vector_schedule_config=adamw_sched,
            matrix_target_modules=["attn", "mlp"],
        )

        trainer = L.Trainer(
            max_steps=100000,
            accumulate_grad_batches=4,
            gradient_clip_val=1.0,
        )
        trainer.fit(module, train_dataloader)
"""

from .dual_optimizer import DualOptimizerModule

__all__ = [
    "DualOptimizerModule",
]
