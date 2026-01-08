"""
Comprehensive test suite for ModdedNanoGPT architecture.

Target: src/research_lib/architectures/modded_nanogpt_base.py
Run with: pytest src/research_lib/architectures/tests/test_modded_nanogpt_base.py -v
"""

import math
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import (
    ModdedNanoGPT,
    TransformerBlock,
)


class TestModdedNanoGPT:

    @pytest.fixture
    def tiny_config(self):
        """
        Returns a minimal configuration for fast testing.
        """
        return NanoGPTConfig(
            vocab_size=100,
            block_size=32,
            n_layer=2,
            n_embd=64,
            n_head=4,
            ff_dim=128,
            bias=True,
            dropout=0.1,
            padding_idx=0,
        )

    @pytest.fixture
    def model(self, tiny_config):
        torch.manual_seed(42)
        return ModdedNanoGPT(tiny_config)

    # =========================================================================
    # Initialization & Configuration Tests
    # =========================================================================

    def test_initialization(self, model, tiny_config):
        """Test that the model initializes with correct dimensions."""
        assert isinstance(model, ModdedNanoGPT)
        assert len(model.blocks) == tiny_config.n_layer
        assert model.embedding.num_embeddings == tiny_config.vocab_size
        assert model.embedding.embedding_dim == tiny_config.n_embd

        # Check output head dimensions
        assert model.output.in_features == tiny_config.n_embd
        assert model.output.out_features == tiny_config.vocab_size

    def test_total_params(self, model):
        """Sanity check on parameter count to ensure layers aren't missing."""
        # Approx calculation:
        # Embed: 100*64
        # Blocks (x2):
        #   Attn: 4 * (64*64) (in_proj + out_proj) + biases
        #   FF: (64*128 + 128*64) + biases
        #   Norms: 64 * 2
        # Head: 64*100
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        # Rough lower bound check
        assert total_params > 10000

    # =========================================================================
    # Forward Pass & Shape Tests
    # =========================================================================

    def test_forward_pass_shape(self, model, tiny_config):
        """Test standard forward pass output shape."""
        batch_size = 4
        seq_len = tiny_config.block_size

        # Create random integer inputs within vocab range
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        output = model(x)

        # Expected: [Batch, Seq, Vocab]
        assert output.shape == (batch_size, seq_len, tiny_config.vocab_size)
        assert not torch.isnan(output).any(), "Output contains NaNs"

    def test_variable_sequence_length(self, model, tiny_config):
        """Test that model handles sequence lengths smaller than block_size."""
        x = torch.randint(0, tiny_config.vocab_size, (2, 10))  # Seq len 10
        output = model(x)
        assert output.shape == (2, 10, tiny_config.vocab_size)

    def test_padding_idx_behavior(self, model, tiny_config):
        """Test that padding index 0 produces zero embeddings (if config dictates)."""
        # Note: nn.Embedding handles padding_idx, ensuring gradients are zero there
        x = torch.zeros((1, 5), dtype=torch.long)  # All padding

        # Hook to check embedding output
        embeddings = model.embedding(x)

        # If padding_idx is set in config and passed to nn.Embedding
        if tiny_config.padding_idx is not None:
            # The embedding vector at padding_idx should be zeros if initialized that way,
            # but standard nn.Embedding initializes it to 0.
            # Let's check gradients instead.
            output = model(x)
            loss = output.sum()
            loss.backward()
            assert model.embedding.weight.grad[tiny_config.padding_idx].sum() == 0.0

    # =========================================================================
    # Logic & Causality Tests
    # =========================================================================

    def test_causal_masking(self, model, tiny_config):
        """
        Crucial GPT Test: Ensure prediction at t only depends on tokens 0..t.
        We change the token at t+1 and ensure output at t does not change.
        """
        model.eval()
        B, T = 2, 10
        x = torch.randint(0, tiny_config.vocab_size, (B, T))

        with torch.no_grad():
            out1 = model(x)

            # Modify the last token
            x_modified = x.clone()
            x_modified[:, -1] = (x_modified[:, -1] + 1) % tiny_config.vocab_size

            out2 = model(x_modified)

            # Check outputs.
            # The output at position T-2 (second to last) should depend on 0..T-2.
            # It should NOT depend on T-1 (the last token).
            # Therefore out1[:, :-1, :] should equal out2[:, :-1, :]

            # We use a slightly higher tolerance for float math
            assert torch.allclose(
                out1[:, :-1, :], out2[:, :-1, :], atol=1e-5
            ), "Causality violation: Changing future token affected past predictions."

            # The last position SHOULD change
            assert not torch.allclose(
                out1[:, -1, :], out2[:, -1, :]
            ), "Sanity check failed: Changing input didn't change output at all."

    def test_logit_clipping(self, model):
        """Test that TanhSoftCapping is effectively limiting logits."""
        # Force large weights to trigger potential large logits
        with torch.no_grad():
            model.output.weight.data *= 10.0

        x = torch.randint(0, 10, (1, 5))
        logits = model(x)

        # Soft cap value is defaults to 30.0 in TanhSoftCapping
        # We check if values are within reasonable bounds (e.g., < 31)
        assert (
            logits.abs().max() < 31.0
        ), "Logit soft capping failed to constrain values."

    # =========================================================================
    # Optimization & Training Tests
    # =========================================================================

    def test_gradient_propagation(self, model, tiny_config):
        """Test that gradients flow through the entire network."""
        model.train()
        x = torch.randint(0, tiny_config.vocab_size, (2, 8))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 8))

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, tiny_config.vocab_size), targets.view(-1)
        )
        loss.backward()

        # Check gradients exist for key components
        assert model.embedding.weight.grad is not None
        assert model.output.weight.grad is not None
        assert model.blocks[0].attn.out_proj.weight.grad is not None

        # Ensure gradients are not all zero (which would imply broken graph)
        assert model.output.weight.grad.abs().sum() > 0

    def test_overfit_single_batch(self, model, tiny_config):
        """Integration test: Can the model overfit a single batch? (Sanity check for learning capability)."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        x = torch.randint(0, tiny_config.vocab_size, (4, 8))
        y = x  # Auto-regressive task: predict self (simplified for test)

        initial_loss = None

        # Simple training loop
        for i in range(20):
            optimizer.zero_grad()
            logits = model(x)
            # Shift targets: logits[0:-1] predicts x[1:]
            # Standard GPT training alignment
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Model failed to learn: Init {initial_loss} -> Final {final_loss}"
        assert (
            final_loss < 2.0
        ), "Model failed to converge significantly on trivial task."

    # =========================================================================
    # System & Compilation Tests
    # =========================================================================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, tiny_config):
        """Test model functionality on GPU."""
        model = ModdedNanoGPT(tiny_config).cuda()
        x = torch.randint(0, tiny_config.vocab_size, (2, 8)).cuda()

        output = model(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 8, tiny_config.vocab_size)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skipping compile test (requires CUDA availiability and Linux environment)",
    )
    def test_torch_compile(self, tiny_config):
        """
        Test torch.compile compatibility.
        Note: This can be slow on the first run.
        """
        # Reset dynamo to ensure clean state
        torch._dynamo.reset()

        model = ModdedNanoGPT(tiny_config).cuda()
        model = torch.compile(model)

        x = torch.randint(0, tiny_config.vocab_size, (2, 8)).cuda()

        # Run forward pass (compilation happens here)
        try:
            output = model(x)
            assert output.shape == (2, 8, tiny_config.vocab_size)
        except Exception as e:
            pytest.fail(f"torch.compile failed: {e}")
