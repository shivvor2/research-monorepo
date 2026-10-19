"""Answer-position loss helpers for causal LMs with ``-100`` labels.

Under ``loss_on: answers`` the training loss is the mean cross-entropy
over the answer positions :math:`A = \\{(b, t) : \\text{labels}[b, t+1]
\\ne -100\\}` of the LM head applied to the final hidden state.  Positions
outside :math:`A` appear neither in the sum nor in its denominator, and
the full path already gives them exactly zero gradient
(``ignore_index``).  The head (no bias) acts on each position separately,
nothing after it mixes positions, and the metrics read only positions in
:math:`A`.  So computing the head only on :math:`A` gives the same loss,
the same gradients for every parameter and the same metrics, in exact
arithmetic.  In floating point the two paths differ at rounding level
(different matmul shapes and reduction order), so runs with the
answer-position head and with the full head are not bit-identical;
``training.answer_head`` is therefore part of the run identity.  The
trunk (embedding, mixers, final norm) still runs on every position.

The helpers are not task-specific: any causal LM with ``-100`` labels
can use them.

Run tests with::

    pytest src/research_lib/training/tests/test_losses.py -v
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_IGNORE_INDEX = -100
"""The label value reserved for non-answer positions (repo convention)."""


@dataclass(frozen=True)
class AnswerLogits:
    """Next-token logits at the answer positions only.

    Attributes:
        logits: ``(N, V)`` logits of the head at the answer positions,
            in row-major order (``(b, t)`` sorted by ``b`` then ``t``).
        targets: ``(N,)`` target token ids, ``labels[:, 1:][mask]``.
        mask: ``(B, T - 1)`` bool, ``labels[:, 1:] != ignore_index``
            (the answer positions under the repo shift convention).
    """

    logits: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


def answer_position_logits(
    hidden: torch.Tensor,
    head: Callable[[torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> AnswerLogits:
    """Evaluate ``head`` only where the shifted label is not ``ignore_index``.

    Args:
        hidden: Final hidden states of shape ``(B, T, d)`` (after the
            final norm; the trunk runs on every position).
        head: The LM head, ``(..., d) -> (..., V)`` (e.g.
            ``ZoologyLM.head``).
        labels: Unshifted labels of shape ``(B, T)``; position ``p`` is
            predicted from position ``p - 1``, so the answer positions
            are ``labels[:, 1:] != ignore_index``.
        ignore_index: The label value marking non-answer positions.

    Returns:
        :class:`AnswerLogits` with ``logits`` of shape ``(N, V)``, the
        rows ``head(hidden)[:, :-1][mask]`` in row-major order, and
        ``targets == labels[:, 1:][mask]``.

    Shape:
        hidden: ``(B, T, d)``, labels: ``(B, T)``; outputs ``(N, V)``,
        ``(N,)`` and ``(B, T - 1)``.
    """
    if hidden.shape[0] != labels.shape[0] or hidden.shape[1] != labels.shape[1]:
        raise ValueError(
            f"batch mismatch: hidden {tuple(hidden.shape)} vs labels "
            f"{tuple(labels.shape)}"
        )
    mask = labels[:, 1:] != ignore_index
    logits = head(hidden[:, :-1][mask])
    targets = labels[:, 1:][mask]
    return AnswerLogits(logits=logits, targets=targets, mask=mask)


def answer_cross_entropy(answers: AnswerLogits) -> torch.Tensor:
    """Mean cross-entropy over the answer rows.

    Equals the shifted ``F.cross_entropy(full_logits, labels,
    ignore_index=...)`` of the full logits in exact arithmetic: the mean
    is over the same ``N`` rows and the ignored rows contribute zero.

    Args:
        answers: :class:`AnswerLogits` from
            :func:`answer_position_logits`.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(answers.logits, answers.targets)
