"""
Tests for the multi-state GDN-2 synthetic-recall training module.

Target: src/research_lib/training/modules/mgdn2_synthetic_recall.py
Run with: pytest src/research_lib/training/tests/test_mgdn2_synthetic_recall_module.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.data.synthetic import (
    MQARConfig,
    OverwriteConfig,
    SyntheticRecallDataModule,
    generate_mqar,
    generate_overwrite,
)
from research_lib.layers.attention import MultistateGatedDelta
from research_lib.training.configs import ScheduleConfig
from research_lib.training.losses import AnswerLogits
from research_lib.training.modules.mgdn2_synthetic_recall import (
    MGDN2SyntheticRecallModule,
)
from research_lib.training.param_utils import partition_parameters_with_names
from research_lib.training.presets import build_weight_decay_configs
from research_lib.training.scheduling import WarmupStableDecaySchedule

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VECTOR_TARGET_MODULES = [
    "embedding",  # token embedding
    "lm_head",  # untied LM head
    r".*norm.*",  # every norm weight (block norms, final norm, output RMSNorm)
    "A_log",  # gated-delta log-decay (per head)
    "dt_bias",  # gated-delta decay bias (per channel)
    "lambda_mixer",  # every λ parameter (all modes)
    "conv",  # depthwise conv filters (short conv, BaseConv)
    r".*\.bias$",  # every bias term
]


def tiny_mqar_config() -> MQARConfig:
    return MQARConfig(vocab_size=64, seq_len=32, num_kv_pairs=(2, 4))


def tiny_overwrite_config() -> OverwriteConfig:
    return OverwriteConfig(
        vocab_size=64, seq_len=32, num_keys=(2, 4), block_size=4, distant_min_blocks=2
    )


def tiny_model(layout: str = "single", dtype=None) -> ZoologyLM:
    """Zoology harness with a BaseConv block and one gated-delta mixer."""
    mixers = zoology_mixers(
        2,
        16,
        make_mixer=lambda: MultistateGatedDelta(
            hidden_size=16,
            num_heads=1,
            head_dim=16,
            layout=layout,
            t_max=32,
            dtype=dtype,
        ),
        dtype=dtype,
    )
    return ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16), mixers, dtype=dtype)


def lr_schedule(peak: float = 1e-3) -> ScheduleConfig:
    return ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                warmup_frac=0.0,
                cooldown_frac=1.0,
                min_value=0.0,
                max_value=peak,
                decay_type="cosine",
            )
        ]
    )


def make_module(
    layout: str = "single",
    slice_values=(2, 4),
    with_classes=None,
    loss_on="answers",
    answer_head=True,
    dtype=None,
    decay_scheme: str = "grouped",
) -> MGDN2SyntheticRecallModule:
    matrix_config, vector_config = build_weight_decay_configs(
        decay_scheme, weight_decay=0.1, betas=(0.9, 0.95)
    )
    return MGDN2SyntheticRecallModule(
        model=tiny_model(layout, dtype=dtype),
        slice_values=slice_values,
        with_classes=with_classes,
        loss_on=loss_on,
        answer_head=answer_head,
        matrix_optimizer_config=matrix_config,
        vector_optimizer_config=vector_config,
        matrix_schedule_config=lr_schedule(),
        vector_schedule_config=lr_schedule(),
        vector_target_modules=VECTOR_TARGET_MODULES,
    )


def make_datamodule(
    task="mqar", batch_size=8, num_workers=0, train_seed=0
) -> SyntheticRecallDataModule:
    config = tiny_mqar_config() if task == "mqar" else tiny_overwrite_config()
    return SyntheticRecallDataModule(
        task=task,
        task_config=config,
        batch_size=batch_size,
        num_val_per_slice=4,
        num_test_per_slice=4,
        train_seed=train_seed,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForwardAndLoss:
    """Test 1: forward ignores extra keys; loss conventions."""

    def test_forward_ignores_extra_keys(self):
        """forward(**batch) on a full overwrite batch equals model(input_ids)."""
        # answer_head=False pins the full-logits path: with the default
        # answer_head=True, forward returns AnswerLogits instead.
        module = make_module(layout="ws", answer_head=False)
        batch = generate_overwrite(tiny_overwrite_config(), num_examples=4, seed=0)
        expected = module.model(batch["input_ids"])
        out = module.forward(**batch)
        torch.testing.assert_close(out, expected)

    def test_compute_loss_on_answers_uses_labels(self):
        """'answers' loss changes when answer logits change, not elsewhere."""
        module = make_module(loss_on="answers")
        batch = generate_mqar(tiny_mqar_config(), num_examples=4, seed=0)
        logits = module.model(batch["input_ids"])
        base = module.compute_loss(logits, batch).item()
        # Perturb non-answer logits only: loss must not change.
        logits2 = logits.clone()
        labels = batch["labels"]
        non_answers = labels[:, 1:] == -100  # shifted non-answer positions
        logits2[:, :-1][non_answers] += 10.0
        perturbed = module.compute_loss(logits2, batch).item()
        assert base == pytest.approx(perturbed)
        # Perturb one answer's logit for a non-target token: loss must change
        # (a constant shift over the whole vocab would be softmax-invariant).
        answers = labels[:, 1:] != -100
        logits3 = logits.clone()
        b = int(answers.nonzero()[0][0])
        t = int(answers.nonzero()[0][1])
        target = int(labels[b, t + 1])
        channel = (target + 1) % logits.shape[-1]
        logits3[b, t, channel] += 1.0
        assert base != pytest.approx(module.compute_loss(logits3, batch).item())

    def test_loss_on_all_differs_from_answers(self):
        """Full-sequence CE differs from answer-only CE on the same batch."""
        batch = generate_mqar(tiny_mqar_config(), num_examples=4, seed=0)
        logits = make_module(loss_on="answers").model(batch["input_ids"])
        m_answers = make_module(loss_on="answers")
        m_all = make_module(loss_on="all")
        answers_loss = float(m_answers.compute_loss(logits, batch))
        all_loss = float(m_all.compute_loss(logits, batch))
        assert answers_loss != pytest.approx(all_loss)
        # 'all' ignores labels entirely (uses input_ids).
        labels_permuted = batch["labels"].clone()
        labels_permuted[:, 5:] = -100
        batch2 = dict(batch)
        batch2["labels"] = labels_permuted
        loss_all = float(m_all.compute_loss(logits, batch2))
        loss_all_again = float(m_all.compute_loss(logits, batch))
        assert loss_all == pytest.approx(loss_all_again)

    def test_invalid_loss_on_rejected(self):
        with pytest.raises(ValueError):
            make_module(loss_on="everywhere")


class TestWeightDecayGroups:
    """Test 4: the two AdamW groups partition the parameters correctly."""

    @pytest.mark.parametrize("layout", ["single", "ws", "fs", "ll"])
    def test_partitioning(self, layout):
        model = tiny_model(layout)
        vector_params, matrix_params = partition_parameters_with_names(
            model, VECTOR_TARGET_MODULES
        )
        vector_names = {name for name, _ in vector_params}
        matrix_names = {name for name, _ in matrix_params}
        all_names = vector_names | matrix_names
        assert all_names == {n for n, _ in model.named_parameters()}
        assert not (vector_names & matrix_names)

        for name, param in model.named_parameters():
            is_lambda = "lambda_mixer" in name
            is_vector_kind = (
                "A_log" in name
                or "dt_bias" in name
                or "lambda_mixer" in name
                or "norm" in name
                or name.endswith(".bias")
                or name.startswith(("embedding.", "lm_head."))
                or (param.dim() != 2 and not is_lambda and "conv" in name)
            )
            if is_vector_kind:
                assert name in vector_names, name
            elif param.dim() == 2:
                assert name in matrix_names, name
        # λ mixer parameters (including its 2-D weights) are all vector-side.
        if layout != "single":
            assert any("lambda_mixer" in n and ".weight" in n for n in vector_names)

    def test_optimizer_groups_created(self):
        """A real fit builds two AdamW optimizers with the right decay."""
        import lightning as L

        module = make_module(layout="ws")
        dm = make_datamodule("mqar", batch_size=8)
        trainer = L.Trainer(
            max_steps=1,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(module, datamodule=dm)
        optimizers = module.optimizers()
        assert len(optimizers) == 2
        matrix_opt, vector_opt = optimizers
        assert matrix_opt.defaults["weight_decay"] == 0.1
        assert vector_opt.defaults["weight_decay"] == 0.0


class TestCosineSchedule:
    """The cosine LR peaks at step 0, halves at the midpoint, decays to 0 at max_steps."""

    def test_cosine_lr_values(self):
        import lightning as L  # noqa: F401  (import check only)

        total = 100
        peak = 3e-3
        schedule = WarmupStableDecaySchedule(
            param_name="lr",
            warmup_frac=0.0,
            cooldown_frac=1.0,
            min_value=0.0,
            max_value=peak,
            decay_type="cosine",
        )
        assert schedule(0, total) == pytest.approx(peak)
        assert schedule(total // 2, total) == pytest.approx(peak / 2, rel=1e-6)
        assert schedule(total, total) == pytest.approx(0.0, abs=1e-9)

    def test_scheduler_binds_to_optimizer(self):
        from research_lib.training.scheduling import ParamScheduler

        total = 50
        peak = 1e-2
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[
                WarmupStableDecaySchedule(
                    param_name="lr",
                    warmup_frac=0.0,
                    cooldown_frac=1.0,
                    min_value=0.0,
                    max_value=peak,
                    decay_type="cosine",
                )
            ],
            total_steps=total,
        )
        # step() applies the value for the current count, then increments:
        # the optimizer's LR after k calls is the schedule value at step k-1.
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(peak)
        for _ in range(total // 2):
            scheduler.step()
        mid = optimizer.param_groups[0]["lr"]
        assert peak / 2 - peak * 0.02 <= mid <= peak / 2 + peak * 0.02
        # After total + 1 calls the applied value is f(total) = 0 exactly.
        for _ in range(total + 1 - 1 - total // 2):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
class TestSmokeTraining:
    """Test 2 (slow): 100 CPU steps reduce the answer loss."""

    def test_loss_decreases(self):
        import lightning as L

        module = make_module(layout="ws", slice_values=(2, 4))
        dm = make_datamodule("mqar", batch_size=8)
        losses = []

        class _Recorder(L.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                value = outputs["loss"] if isinstance(outputs, dict) else outputs
                losses.append(float(value))

        trainer = L.Trainer(
            max_steps=100,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[_Recorder()],
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(module, datamodule=dm)
        assert len(losses) == 100
        head = sum(losses[:10]) / 10
        tail = sum(losses[-10:]) / 10
        assert tail < head, (head, tail)


class TestValidationLogging:
    """Test 3: validation logs the RecallMetrics keys."""

    @pytest.fixture
    def fitted_mqar(self):
        import lightning as L

        module = make_module(layout="ws", slice_values=(2, 4))
        dm = make_datamodule("mqar", batch_size=8)
        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[],
            num_sanity_val_steps=0,
            val_check_interval=2,
        )
        trainer.fit(module, datamodule=dm)
        return trainer, module

    @pytest.fixture
    def fitted_overwrite(self):
        import lightning as L

        module = make_module(layout="ws", slice_values=(2, 4), with_classes=None)
        dm = make_datamodule("overwrite", batch_size=8)
        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[],
            num_sanity_val_steps=0,
            val_check_interval=2,
        )
        trainer.fit(module, datamodule=dm)
        return trainer, module

    def test_mqar_logs_slices(self, fitted_mqar):
        trainer, module = fitted_mqar
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        assert "val/acc" in metrics
        assert "val/acc/slice=2" in metrics
        assert "val/acc/slice=4" in metrics

    def test_overwrite_logs_class_and_distance_keys(self, fitted_overwrite):
        trainer, module = fitted_overwrite
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        assert "val/acc" in metrics
        assert any("/class=" in k for k in metrics)
        assert any(k.startswith("val/leak/class=") for k in metrics)
        # The write-distance bins are logged too (overwrite batches carry
        # write_distance; RecallMetrics bins them by default).
        assert any(k.startswith("val/acc/dist=") for k in metrics), sorted(metrics)
        assert any(k.startswith("val/leak/dist=") for k in metrics), sorted(metrics)
        assert any(
            k.startswith("val/acc/slice=") and "/dist=" in k for k in metrics
        ), sorted(metrics)

    def test_test_step_logs_test_metrics(self, fitted_mqar):
        trainer, module = fitted_mqar
        dm = make_datamodule("mqar", batch_size=8)
        trainer.test(module, datamodule=dm)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        assert "test/acc" in metrics
        assert "test/acc/slice=2" in metrics


@pytest.mark.slow
class TestAnswerHeadTraining:
    """100 fp64 CPU steps, one run per head mode, same seed and batches:
    the loss trajectories agree at every step."""

    @staticmethod
    def _run(answer_head: bool) -> list[float]:
        import lightning as L

        # Identical weights and batches for both runs (the trajectories
        # agree only when the two builds start from identical weights):
        # repeat the RNG state before building the module.
        torch.manual_seed(0)
        module = make_module(
            layout="ws",
            slice_values=(2, 4),
            answer_head=answer_head,
            dtype=torch.float64,
        )
        dm = make_datamodule("mqar", batch_size=8)
        losses = []

        class _Recorder(L.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                value = outputs["loss"] if isinstance(outputs, dict) else outputs
                losses.append(float(value))

        trainer = L.Trainer(
            max_steps=100,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[_Recorder()],
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(module, datamodule=dm)
        return losses

    def test_trajectories_agree(self):
        losses_true = self._run(answer_head=True)
        losses_false = self._run(answer_head=False)
        assert len(losses_true) == len(losses_false) == 100
        for step, (a, b) in enumerate(zip(losses_true, losses_false, strict=True)):
            assert a == pytest.approx(b, rel=1e-8), (step, a, b)


class TestAnswerHead:
    """The answer-head module path: equivalence with the full head and
    the loss_on switch."""

    @staticmethod
    def _validate(module, dm) -> dict[str, float]:
        import lightning as L

        trainer = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.validate(module, datamodule=dm)
        return {k: float(v) for k, v in trainer.callback_metrics.items()}

    @pytest.mark.parametrize("task", ["mqar", "overwrite"])
    @pytest.mark.parametrize("layout", ["single", "ll"])
    def test_answer_head_module_equivalence(self, task, layout):
        """answer_head=True vs False: same training loss (a
        training_step-equivalent forward + compute_loss) and the same
        validation metrics, in fp64."""
        batch = (
            generate_mqar(tiny_mqar_config(), num_examples=4, seed=0)
            if task == "mqar"
            else generate_overwrite(tiny_overwrite_config(), num_examples=8, seed=0)
        )
        # Two independent builds must start from identical weights, so
        # reseed before each; the module equivalence is exact only when
        # the trunk weights match.
        torch.manual_seed(0)
        m_true = make_module(layout=layout, answer_head=True, dtype=torch.float64)
        torch.manual_seed(0)
        m_false = make_module(layout=layout, answer_head=False, dtype=torch.float64)

        out_true = m_true.forward(**batch)
        out_false = m_false.forward(**batch)
        assert isinstance(out_true, AnswerLogits)
        assert isinstance(out_false, torch.Tensor)
        loss_true = m_true.compute_loss(out_true, batch)
        loss_false = m_false.compute_loss(out_false, batch)
        assert loss_true.detach() == pytest.approx(loss_false.detach(), rel=1e-12)

        dm = make_datamodule(task, batch_size=8)
        metrics_true = self._validate(m_true, dm)
        metrics_false = self._validate(m_false, dm)
        assert metrics_true.keys() == metrics_false.keys()
        for key in metrics_true:
            if key.endswith("/loss"):
                assert metrics_true[key] == pytest.approx(metrics_false[key], rel=1e-12)
            else:
                assert metrics_true[key] == metrics_false[key], key

    def test_loss_on_all_returns_full_logits(self):
        """loss_on="all" keeps the full head even with answer_head=True:
        forward returns (B, T, V) logits and the full-sequence CE."""
        module = make_module(loss_on="all", answer_head=True)
        batch = generate_mqar(tiny_mqar_config(), num_examples=4, seed=0)
        out = module.forward(**batch)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 32, 64)
        loss = module.compute_loss(out, batch)
        expected = F.cross_entropy(
            out[:, :-1].reshape(-1, 64), batch["input_ids"][:, 1:].reshape(-1)
        )
        assert loss.detach() == pytest.approx(expected.detach())
        # labels are ignored under loss_on="all": passing them changes nothing.
        out2 = module.forward(batch["input_ids"], labels=batch["labels"])
        assert isinstance(out2, torch.Tensor)
        torch.testing.assert_close(out2, out)

    def test_answer_head_requires_hidden_states_and_head(self):
        """answer_head=True raises TypeError naming the missing method."""
        matrix_config, vector_config = build_weight_decay_configs(
            "grouped", weight_decay=0.1, betas=(0.9, 0.95)
        )

        class _NoHead(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(TypeError, match="hidden_states"):
            MGDN2SyntheticRecallModule(
                model=_NoHead(),
                slice_values=(2, 4),
                matrix_optimizer_config=matrix_config,
                vector_optimizer_config=vector_config,
                matrix_schedule_config=lr_schedule(),
                vector_schedule_config=lr_schedule(),
            )


@pytest.mark.slow
class TestResume:
    """Test 5 (integration, slow): the datamodule's stream position survives a
    real Lightning checkpoint with DataLoader workers."""

    class _IndexRecorder:
        """Lightning callback factory recording batch_index + input hash."""

        def __init__(self):
            import lightning as L

            self.records = []
            self._cb = None

            class _CB(L.Callback):
                def __init__(cb_self):
                    super().__init__()
                    cb_self.owner = None

                def on_train_batch_start(cb_self, trainer, pl_module, batch, batch_idx):
                    rec = cb_self.owner.records
                    rec.append(
                        (
                            int(batch["batch_index"]),
                            batch["input_ids"].clone(),
                        )
                    )

            cb = _CB()
            cb.owner = self
            self._cb = cb

        def callback(self):
            return self._cb

        def indices(self):
            return [i for i, _ in self.records]

        def batches(self, indices):
            return {i: ids for i, ids in self.records if i in set(indices)}

    @staticmethod
    def _trainer(max_steps, cb, ckpt_dir):
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint

        callbacks = [cb.callback()]
        if ckpt_dir is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(ckpt_dir), every_n_train_steps=3, save_top_k=-1
                )
            )
        return L.Trainer(
            max_steps=max_steps,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )

    def test_exact_stream_resumption(self, tmp_path):
        # Full run: 6 steps.
        dm_full = make_datamodule(batch_size=8, num_workers=2)
        module_full = make_module(layout="ws")
        rec_full = self._IndexRecorder()
        trainer_full = self._trainer(6, rec_full, None)
        trainer_full.fit(module_full, datamodule=dm_full)
        r_full = rec_full.indices()
        assert r_full == [0, 1, 2, 3, 4, 5]

        # Interrupted run: 3 steps, checkpoint at step 3.
        dm_a = make_datamodule(batch_size=8, num_workers=2)
        module_a = make_module(layout="ws")
        rec_a = self._IndexRecorder()
        ckpt_dir = tmp_path / "ckpts"
        trainer_a = self._trainer(3, rec_a, ckpt_dir)
        trainer_a.fit(module_a, datamodule=dm_a)
        r_a = rec_a.indices()
        assert r_a == [0, 1, 2]
        ckpt = sorted(ckpt_dir.glob("*.ckpt"))
        assert len(ckpt) == 1

        # Resumed run: a fresh module and datamodule, continue to step 6.
        dm_b = make_datamodule(batch_size=8, num_workers=2)
        module_b = make_module(layout="ws")
        rec_b = self._IndexRecorder()
        trainer_b = self._trainer(6, rec_b, None)
        trainer_b.fit(
            module_b, datamodule=dm_b, ckpt_path=str(ckpt[0]), weights_only=False
        )
        r_b = rec_b.indices()
        assert r_b == [3, 4, 5]

        # The resumed batches are bit-identical to the full run's.
        full_b = rec_full.batches({3, 4, 5})
        resumed_b = rec_b.batches({3, 4, 5})
        for index in (3, 4, 5):
            torch.testing.assert_close(full_b[index], resumed_b[index])
