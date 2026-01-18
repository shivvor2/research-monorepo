"""
PyTorch Lightning modules for training with custom optimizer configurations.

This subpackage provides LightningModule implementations for various
optimizer configurations:

- :class:`DualOptimizerModule`: Training with two optimizers (e.g., Muon + AdamW)
- :class:`SingleOptimizerModule`: Training with a single optimizer (planned)

These modules integrate with the configuration system in :mod:`research_lib.training.configs`
and the scheduling utilities in :mod:`research_lib.training.scheduling`.

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

    For HuggingFace-style checkpointing, use a custom callback or save manually
    after training::

        trainer.fit(module, dataloader)
        module.model.save_pretrained("final_model")
"""

from .dual_optimizer import DualOptimizerModule

__all__ = [
    "DualOptimizerModule",
]
