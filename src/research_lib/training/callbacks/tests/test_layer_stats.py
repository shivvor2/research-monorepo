"""
Tests for the layer-statistics callback.

Target: src/research_lib/training/callbacks/layer_stats.py
Run with: pytest src/research_lib/training/callbacks/tests/test_layer_stats.py -v
"""

import lightning as L
import pytest
import torch
import torch.nn as nn

from research_lib.training.callbacks.layer_stats import (
    LayerStatsCallback,
    StatsProvider,
    mean_summary,
)


class ToyProvider(nn.Module):
    """A minimal StatsProvider: records the batch mean while collecting."""

    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.collect_stats = False
        self.latest_stats = None
        self.offset = offset

    def forward(self, x):
        if self.collect_stats:
            self.latest_stats = {"m": x.detach().float().mean() + self.offset}
        return x


class ToyLM(L.LightningModule):
    """Two providers, routed in a fixed order through the batch."""

    def __init__(self):
        super().__init__()
        self.p0 = ToyProvider(offset=0.0)
        self.p1 = ToyProvider(offset=10.0)
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.p1(self.p0(x))

    def training_step(self, batch, batch_idx):
        return self.weight * batch[0].sum()

    def validation_step(self, batch, batch_idx):
        self(batch[0])

    def test_step(self, batch, batch_idx):
        self(batch[0])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def make_data(n=8, size=4):
    # Batches of size 2 over an arange tensor: batch means are known.
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.arange(n * size, dtype=torch.float32).reshape(n, size)
        ),
        batch_size=2,
    )


def make_trainer(callbacks, max_steps=2):
    return L.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=max_steps,  # the fit stops mid-epoch otherwise
        limit_val_batches=1.0,
        limit_test_batches=1.0,
    )


class TestStatsProvider:
    def test_multistate_gated_delta_satisfies_the_protocol(self):
        from research_lib.layers.attention import MultistateGatedDelta

        layer = MultistateGatedDelta(hidden_size=32, num_heads=2, head_dim=16)
        assert isinstance(layer, StatsProvider)

    def test_toy_provider_satisfies_the_protocol(self):
        assert isinstance(ToyProvider(), StatsProvider)
        assert not isinstance(nn.Linear(2, 2), StatsProvider)


class SkippingProvider(nn.Module):
    """A StatsProvider that fills its stats on the first two forwards only
    (a provider is allowed to skip batches: e.g. a layer whose mode does
    not touch it every forward)."""

    def __init__(self):
        super().__init__()
        self.collect_stats = False
        self.latest_stats = None
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.collect_stats and self.calls <= 2:
            self.latest_stats = {"m": torch.tensor(float(self.calls))}
        return x


class SkippingLM(L.LightningModule):
    """One provider; validation runs the data through it."""

    def __init__(self):
        super().__init__()
        self.p = SkippingProvider()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.p(x)

    def training_step(self, batch, batch_idx):
        return self.weight * batch[0].sum()

    def validation_step(self, batch, batch_idx):
        self(batch[0])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


class TestLayerStatsCallback:
    def test_stats_averaged_and_logged_under_right_names(self):
        callback = LayerStatsCallback(prefix="mixer")
        trainer = make_trainer([callback])
        module = ToyLM()
        data = make_data()
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        # The whole arange tensor's mean, averaged over the epoch's batch
        # means: layer0 = mean of batch means (no offset), layer1 = +10.
        expected = torch.arange(8 * 4, dtype=torch.float32).mean().item()
        assert metrics["mixer/layer0/m"] == pytest.approx(expected)
        assert metrics["mixer/layer1/m"] == pytest.approx(expected + 10.0)

    def test_provider_that_skips_batches_is_not_resnapshotted(self):
        """``latest_stats`` is cleared at each epoch start and after every
        snapshot, so a provider that does not fill its stats in a batch is
        never snapshotted with a stale value."""
        callback = LayerStatsCallback(prefix="skip")
        trainer = make_trainer([callback])
        module = SkippingLM()
        data = make_data()  # 4 validation batches
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        # The provider fills stats at calls 1 and 2 only (values 1 and 2):
        # the epoch's mean is (1 + 2) / 2 = 1.5. Without the resets,
        # batches 3 and 4 would re-snapshot the stale value 2 -> 1.75.
        assert metrics["skip/layer0/m"] == pytest.approx(1.5)

    def test_test_epoch_also_collects(self):
        callback = LayerStatsCallback(prefix="ts")
        trainer = make_trainer([callback])
        module = ToyLM()
        data = make_data()
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        trainer.test(module, dataloaders=data)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        expected = torch.arange(8 * 4, dtype=torch.float32).mean().item()
        assert metrics["ts/layer0/m"] == pytest.approx(expected)
        assert metrics["ts/layer1/m"] == pytest.approx(expected + 10.0)

    def test_collection_on_only_during_validation_and_test_epochs(self):
        observed = []

        class _Spy(L.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                observed.append(("train", pl_module.p0.collect_stats))

            def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
            ):
                observed.append(("val_batch", pl_module.p0.collect_stats))

        callback = LayerStatsCallback()
        trainer = make_trainer([callback, _Spy()])  # spy runs after the stats callback
        module = ToyLM()
        data = make_data()
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        stages = {}
        for stage, flag in observed:
            stages.setdefault(stage, set()).add(flag)
        assert stages["train"] == {False}  # never on during training steps
        assert stages["val_batch"] == {True}  # on during the validation epoch
        # And disabled again at the epoch end.
        assert module.p0.collect_stats is False
        assert module.p1.collect_stats is False

    def test_custom_summarizer(self):
        callback = LayerStatsCallback(
            prefix="custom",
            summarize=lambda key, t: {f"{key}_max": float(t.max())},
        )
        trainer = make_trainer([callback])
        module = ToyLM()
        data = make_data()
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        assert "custom/layer0/m_max" in metrics
        assert "custom/layer1/m_max" in metrics

    def test_mean_summary_is_the_default(self):
        assert mean_summary("k", torch.tensor([1.0, 2.0, 3.0])) == {"k": 2.0}

    def test_no_providers_is_a_no_op(self):
        class _Empty(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

            def training_step(self, batch, batch_idx):
                return self.linear(batch[0]).sum()

            def validation_step(self, batch, batch_idx):
                pass  # no providers: nothing to collect

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        callback = LayerStatsCallback()
        trainer = make_trainer([callback])
        module = _Empty()
        data = make_data()
        trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
        assert not any(k.startswith("mixer/") for k in trainer.callback_metrics)
