"""
Tests for the answer-position loss helpers.

Target: src/research_lib/training/losses.py
Run with: pytest src/research_lib/training/tests/test_losses.py -v

fp64 on CPU: the answer-position head must give the same loss, the same
gradients for every parameter and the same metrics as the full head in
exact arithmetic; these tests check the equivalence at fp64 rounding
level, on real batches from both synthetic tasks, for both layer
patterns and the ``single`` and ``ll`` layouts.
"""

import pytest
import torch
import torch.nn.functional as F

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.data.synthetic import (
    MQARConfig,
    OverwriteConfig,
    generate_mqar,
    generate_overwrite,
)
from research_lib.layers.attention import MultistateGatedDelta
from research_lib.training.losses import answer_cross_entropy, answer_position_logits

IGNORE = -100


def tiny_model(pattern: str = "zoology", layout: str = "single") -> ZoologyLM:
    """fp64 Zoology harness: BaseConv + one gated-delta mixer.

    Vocab 64, d_model 16, 1 head, t_max 32 (the spec's tiny model).
    """
    mixers = zoology_mixers(
        2,
        16,
        make_mixer=lambda: MultistateGatedDelta(
            hidden_size=16, num_heads=1, head_dim=16, layout=layout, t_max=32
        ),
        pattern=pattern,
    )
    return ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16), mixers).double()


def mqar_batch() -> dict[str, torch.Tensor]:
    """A real MQAR batch: T = 32, D in {2, 4}."""
    return generate_mqar(
        MQARConfig(vocab_size=64, seq_len=32, num_kv_pairs=(2, 4)),
        num_examples=4,
        seed=0,
    )


def overwrite_batch() -> dict[str, torch.Tensor]:
    """A real overwrite batch: T = 32, K in {2, 4}, all four classes."""
    return generate_overwrite(
        OverwriteConfig(
            vocab_size=64,
            seq_len=32,
            num_keys=(2, 4),
            block_size=4,
            distant_min_blocks=2,
        ),
        num_examples=8,
        seed=0,
    )


def full_shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The full-path loss: shifted CE with ``ignore_index`` (the parent's
    ``DualOptimizerModule.compute_loss``)."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
    )


def make_batch(task: str) -> dict[str, torch.Tensor]:
    return mqar_batch() if task == "mqar" else overwrite_batch()


class TestAnswerPositionLogits:
    """The helper selects exactly the answer rows, in row-major order."""

    @pytest.mark.parametrize("pattern", ["zoology", "mixer_x2"])
    @pytest.mark.parametrize("layout", ["single", "ll"])
    @pytest.mark.parametrize("task", ["mqar", "overwrite"])
    def test_selects_exactly_the_answer_rows(self, pattern, layout, task):
        torch.manual_seed(0)
        model = tiny_model(pattern, layout)
        batch = make_batch(task)
        hidden = model.hidden_states(batch["input_ids"])
        answers = answer_position_logits(hidden, model.head, batch["labels"])

        mask = batch["labels"][:, 1:] != IGNORE
        torch.testing.assert_close(answers.mask, mask, rtol=0, atol=0)
        # The rows of head(hidden)[:, :-1][mask], in row-major order.
        torch.testing.assert_close(
            answers.logits, model.head(hidden)[:, :-1][mask], atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            answers.targets, batch["labels"][:, 1:][mask], atol=0, rtol=0
        )
        assert answers.logits.shape == (int(mask.sum()), 64)
        assert answers.targets.shape == (int(mask.sum()),)

    def test_batch_mismatch_rejected(self):
        hidden = torch.zeros(2, 32, 16)
        labels = torch.zeros(3, 32, dtype=torch.long)
        with pytest.raises(ValueError, match="batch mismatch"):
            answer_position_logits(hidden, lambda x: x, labels)


class TestLossEquivalence:
    """The answer loss equals the full shifted CE, rtol=1e-12."""

    @pytest.mark.parametrize("pattern", ["zoology", "mixer_x2"])
    @pytest.mark.parametrize("layout", ["single", "ll"])
    @pytest.mark.parametrize("task", ["mqar", "overwrite"])
    def test_loss_equivalence(self, pattern, layout, task):
        torch.manual_seed(0)
        model = tiny_model(pattern, layout)
        batch = make_batch(task)
        full_logits = model(batch["input_ids"])
        answers = answer_position_logits(
            model.hidden_states(batch["input_ids"]), model.head, batch["labels"]
        )
        loss_answers = answer_cross_entropy(answers)
        loss_full = full_shifted_ce(full_logits, batch["labels"])
        assert loss_answers.detach() == pytest.approx(loss_full.detach(), rel=1e-12)


class TestGradientEquivalence:
    """Every parameter's gradient agrees, atol=rtol=1e-10."""

    @pytest.mark.parametrize("pattern", ["zoology", "mixer_x2"])
    @pytest.mark.parametrize("layout", ["single", "ll"])
    @pytest.mark.parametrize("task", ["mqar", "overwrite"])
    def test_gradient_equivalence(self, pattern, layout, task):
        torch.manual_seed(0)
        model = tiny_model(pattern, layout)
        batch = make_batch(task)

        # Answer path: head evaluated only at the answer positions.
        answers = answer_position_logits(
            model.hidden_states(batch["input_ids"]), model.head, batch["labels"]
        )
        answer_cross_entropy(answers).backward()
        grads_answer = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()

        # Full path: the head on every position, ignore_index CE.
        full_logits = model(batch["input_ids"])
        full_shifted_ce(full_logits, batch["labels"]).backward()

        for name, param in model.named_parameters():
            torch.testing.assert_close(
                param.grad, grads_answer[name], atol=1e-10, rtol=1e-10
            )
