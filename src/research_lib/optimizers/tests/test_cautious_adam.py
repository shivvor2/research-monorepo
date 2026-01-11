"""
Test suite for CautiousAdamW optimizer.

Run with: pytest src/research_lib/optimizers/tests/test_cautious_adam.py -v
"""

import pytest
import torch
import torch.nn as nn

from ..cautious_adamw import CautiousAdamW


class TestCautiousAdamW:

    def test_instantiation(self):
        """Test simple instantiation matching standard API."""
        params = [torch.nn.Parameter(torch.randn(10))]
        opt = CautiousAdamW(params, lr=1e-3)
        assert len(opt.param_groups) == 1
        assert opt.defaults["weight_decay"] == 1e-2

    def test_cautious_logic_shrinking(self):
        """
        Verify WD is applied when gradient shrinks the parameter.
        Param > 0, Gradient > 0 (Adam Step < 0).
        Param * Step < 0 (Aligned for shrinkage).
        """
        # Param = 1.0
        p = torch.nn.Parameter(torch.tensor([1.0]))
        # Grad = 1.0 (Positive grad -> -Step)
        p.grad = torch.tensor([1.0])

        # High WD to make it obvious
        lr = 0.1
        wd = 1.0
        opt = CautiousAdamW(
            [p], lr=lr, weight_decay=wd, betas=(0.0, 0.0)
        )  # beta1=0 for immediate update

        # Step:
        # m = 1.0, v = 1.0
        # step = -1.0 * lr = -0.1
        # mask = (1.0 * -0.1) < 0 -> True
        # WD update: p -= lr * wd * p = 0.1 * 1.0 * 1.0 = 0.1
        # Adam update: p += step = -0.1
        # Total: 1.0 - 0.1 - 0.1 = 0.8

        opt.step()

        # Check logic: 1.0 -> 0.8
        torch.testing.assert_close(p.data, torch.tensor([0.8]))

    def test_cautious_logic_growing(self):
        """
        Verify WD is SKIPPED when gradient grows the parameter.
        Param > 0, Gradient < 0 (Adam Step > 0).
        Param * Step > 0 (Opposed).
        """
        # Param = 1.0
        p = torch.nn.Parameter(torch.tensor([1.0]))
        # Grad = -1.0 (Negative grad -> +Step)
        p.grad = torch.tensor([-1.0])

        lr = 0.1
        wd = 1.0
        opt = CautiousAdamW([p], lr=lr, weight_decay=wd, betas=(0.0, 0.0))

        # Step:
        # m = -1.0, v = 1.0
        # step = -(-1.0) * lr = +0.1
        # mask = (1.0 * 0.1) < 0 -> False
        # WD update: 0
        # Adam update: p += step = +0.1
        # Total: 1.0 + 0.1 = 1.1

        opt.step()

        torch.testing.assert_close(p.data, torch.tensor([1.1]))

    def test_standard_adam_convergence(self):
        """Ensure it still optimizes a simple function."""
        x = torch.nn.Parameter(torch.tensor([10.0]))
        opt = CautiousAdamW([x], lr=1.0)

        # Minimize x^2
        for _ in range(100):
            opt.zero_grad()
            loss = x**2
            loss.backward()
            opt.step()

        assert x.abs() < 0.1

    def test_device_compatibility(self):
        """Test it runs on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        p = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
        p.grad = torch.tensor([1.0], device="cuda")
        opt = CautiousAdamW([p], lr=0.1)

        opt.step()
        assert p.device.type == "cuda"
