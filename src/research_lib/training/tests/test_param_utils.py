"""Tests for parameter selection and partitioning utilities."""

import warnings

import pytest
import torch
import torch.nn as nn

from research_lib.training.param_utils import (
    ParameterCounts,
    count_parameters,
    count_parameters_by_status,
    partition_parameters,
    partition_parameters_multi,
    partition_parameters_multi_with_names,
    partition_parameters_with_names,
    select_parameters,
    select_parameters_with_names,
    summarize_partition,
)


class SimpleModel(nn.Module):
    """Simple model for testing partitioning."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.attn_q = nn.Linear(64, 64, bias=False)
        self.attn_k = nn.Linear(64, 64, bias=False)
        self.attn_v = nn.Linear(64, 64, bias=False)
        self.attn_out = nn.Linear(64, 64, bias=False)
        self.mlp_fc1 = nn.Linear(64, 256, bias=False)
        self.mlp_fc2 = nn.Linear(256, 64, bias=False)
        self.ln = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 100, bias=False)


class OverlapModel(nn.Module):
    """Model with parameter names that can match multiple patterns."""

    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(64, 64, bias=False)
        self.mlp_proj = nn.Linear(64, 64, bias=False)
        self.attn_mlp_shared = nn.Linear(64, 64, bias=False)  # Matches both!
        self.other = nn.Linear(64, 64, bias=False)


# =============================================================================
# Tests for Primitives: select_parameters
# =============================================================================


class TestSelectParameters:
    """Tests for select_parameters primitive."""

    def test_select_matching_params(self):
        """Test selecting parameters that match patterns."""
        model = SimpleModel()
        params = select_parameters(model, ["attn"])

        # Should select attn_q, attn_k, attn_v, attn_out
        assert len(params) == 4

    def test_select_multiple_patterns(self):
        """Test selecting with multiple patterns."""
        model = SimpleModel()
        params = select_parameters(model, ["attn", "mlp"])

        # attn: 4, mlp: 2
        assert len(params) == 6

    def test_select_empty_patterns(self):
        """Test that empty patterns returns empty list."""
        model = SimpleModel()
        params = select_parameters(model, [])

        assert len(params) == 0

    def test_select_no_matches(self):
        """Test that non-matching patterns returns empty list."""
        model = SimpleModel()
        params = select_parameters(model, ["nonexistent"])

        assert len(params) == 0

    def test_select_respects_requires_grad(self):
        """Test that frozen params are excluded by default."""
        model = SimpleModel()
        model.attn_q.weight.requires_grad = False

        params = select_parameters(model, ["attn"], require_grad=True)

        # Should exclude attn_q
        assert len(params) == 3

    def test_select_includes_frozen_when_flag_false(self):
        """Test that frozen params are included when require_grad=False."""
        model = SimpleModel()
        model.attn_q.weight.requires_grad = False

        params = select_parameters(model, ["attn"], require_grad=False)

        # Should include all 4 attn params
        assert len(params) == 4


class TestSelectParametersWithNames:
    """Tests for select_parameters_with_names primitive."""

    def test_returns_names(self):
        """Test that names are returned alongside params."""
        model = SimpleModel()
        named_params = select_parameters_with_names(model, ["attn"])

        assert len(named_params) == 4
        for name, param in named_params:
            assert isinstance(name, str)
            assert "attn" in name
            assert isinstance(param, torch.Tensor)


# =============================================================================
# Tests for Convenience Functions: partition_parameters
# =============================================================================


class TestPartitionParameters:
    """Tests for partition_parameters convenience function."""

    def test_partition_attn_and_mlp(self):
        """Test partitioning with attn and mlp targets."""
        model = SimpleModel()
        target_patterns = ["attn", "mlp"]

        target_params, other_params = partition_parameters(model, target_patterns)

        # attn_q, attn_k, attn_v, attn_out, mlp_fc1, mlp_fc2 = 6 params
        assert len(target_params) == 6

        # embed, ln.weight, ln.bias, lm_head = 4 params
        assert len(other_params) == 4

    def test_partition_empty_targets(self):
        """Test that empty targets puts all params in other."""
        model = SimpleModel()
        target_patterns = []

        target_params, other_params = partition_parameters(model, target_patterns)

        assert len(target_params) == 0
        assert len(other_params) == 10  # All params

    def test_partition_all_match(self):
        """Test when all params match targets."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        )
        target_patterns = ["0", "1"]

        target_params, other_params = partition_parameters(model, target_patterns)

        assert len(other_params) == 0
        assert len(target_params) == 4  # 2 weights + 2 biases

    def test_partition_respects_requires_grad(self):
        """Test that frozen params are excluded when require_grad=True."""
        model = SimpleModel()
        model.embed.weight.requires_grad = False

        target_params, other_params = partition_parameters(
            model, ["attn", "mlp"], require_grad=True
        )

        # embed should not appear in either list
        total_params = len(target_params) + len(other_params)
        assert total_params == 9  # 10 - 1 frozen

    def test_no_double_assignment(self):
        """Test that parameters are not assigned to both groups."""
        model = SimpleModel()
        target_params, other_params = partition_parameters(model, ["attn"])

        target_ids = set(id(p) for p in target_params)
        other_ids = set(id(p) for p in other_params)

        # No overlap
        assert len(target_ids & other_ids) == 0


class TestPartitionParametersWithNames:
    """Tests for partition_parameters_with_names convenience function."""

    def test_returns_names(self):
        """Test that names are returned alongside params."""
        model = SimpleModel()
        target_named, other_named = partition_parameters_with_names(model, ["attn"])

        for name, param in target_named:
            assert isinstance(name, str)
            assert isinstance(param, torch.Tensor)
            assert "attn" in name

        for name, param in other_named:
            assert isinstance(name, str)
            assert isinstance(param, torch.Tensor)
            assert "attn" not in name


# =============================================================================
# Tests for Multi-Group Partitioning
# =============================================================================


class TestPartitionParametersMulti:
    """Tests for partition_parameters_multi convenience function."""

    def test_three_way_split(self):
        """Test three-way partitioning."""
        model = SimpleModel()
        pattern_groups = [
            ["attn"],  # Group 0
            ["mlp"],  # Group 1
            [],  # Group 2: catch-all
        ]

        groups = partition_parameters_multi(model, pattern_groups)

        assert len(groups) == 3
        assert len(groups[0]) == 4  # attn params
        assert len(groups[1]) == 2  # mlp params
        assert len(groups[2]) == 4  # embed, ln, lm_head

    def test_catch_all_only(self):
        """Test with only a catch-all group."""
        model = SimpleModel()
        pattern_groups = [[]]

        groups = partition_parameters_multi(model, pattern_groups)

        assert len(groups) == 1
        assert len(groups[0]) == 10  # All params

    def test_no_catch_all(self):
        """Test without a catch-all group (some params unassigned)."""
        model = SimpleModel()
        pattern_groups = [
            ["attn"],
            ["mlp"],
            # No catch-all!
        ]

        groups = partition_parameters_multi(model, pattern_groups)

        assert len(groups) == 2
        assert len(groups[0]) == 4  # attn
        assert len(groups[1]) == 2  # mlp
        # embed, ln, lm_head are unassigned!

    def test_overlap_first_match_wins(self):
        """Test that first match wins with overlap_strategy='first'."""
        model = OverlapModel()
        pattern_groups = [
            ["attn"],  # Group 0
            ["mlp"],  # Group 1
            [],  # Group 2: catch-all
        ]

        # Should warn about attn_mlp_shared matching both groups
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            groups = partition_parameters_multi(
                model, pattern_groups, overlap_strategy="first", warn_on_overlap=True
            )

            # Check warning was raised
            assert len(w) == 1
            assert "attn_mlp_shared" in str(w[0].message)

        # attn_mlp_shared should be in group 0 (first match)
        group0_names = [
            name
            for name, _ in partition_parameters_multi_with_names(
                model, pattern_groups, warn_on_overlap=False
            )[0]
        ]
        assert any("attn_mlp_shared" in name for name in group0_names)

    def test_overlap_no_warning_when_disabled(self):
        """Test that warning can be disabled."""
        model = OverlapModel()
        pattern_groups = [["attn"], ["mlp"], []]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            partition_parameters_multi(
                model, pattern_groups, overlap_strategy="first", warn_on_overlap=False
            )

            # No warning should be raised
            assert len(w) == 0

    def test_overlap_error_strategy(self):
        """Test that overlap_strategy='error' raises ValueError."""
        model = OverlapModel()
        pattern_groups = [["attn"], ["mlp"], []]

        with pytest.raises(ValueError, match="attn_mlp_shared"):
            partition_parameters_multi(model, pattern_groups, overlap_strategy="error")

    def test_empty_pattern_groups(self):
        """Test with empty pattern_groups list."""
        model = SimpleModel()
        groups = partition_parameters_multi(model, [])

        assert groups == []


# =============================================================================
# Tests for Counting Utilities
# =============================================================================


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters(self):
        """Test counting total parameters."""
        params = [
            torch.randn(10, 20),  # 200
            torch.randn(5, 5),  # 25
            torch.randn(100),  # 100
        ]

        assert count_parameters(params) == 325

    def test_count_empty_list(self):
        """Test counting empty list returns 0."""
        assert count_parameters([]) == 0


class TestCountParametersByStatus:
    """Tests for count_parameters_by_status function."""

    def test_all_trainable(self):
        """Test counting when all params are trainable."""
        model = nn.Linear(10, 10)
        counts = count_parameters_by_status(model)

        assert counts.total == 110  # 10*10 + 10
        assert counts.trainable == 110
        assert counts.frozen == 0
        assert counts.trainable_ratio == 1.0
        assert counts.frozen_ratio == 0.0

    def test_mixed_trainable_frozen(self):
        """Test counting with mixed trainable/frozen params."""
        model = SimpleModel()
        model.embed.weight.requires_grad = False

        counts = count_parameters_by_status(model)

        embed_size = 100 * 64  # 6400
        assert counts.frozen == embed_size
        assert counts.trainable == counts.total - embed_size

    def test_all_frozen(self):
        """Test counting when all params are frozen."""
        model = nn.Linear(10, 10)
        for param in model.parameters():
            param.requires_grad = False

        counts = count_parameters_by_status(model)

        assert counts.total == 110
        assert counts.trainable == 0
        assert counts.frozen == 110


class TestParameterCounts:
    """Tests for ParameterCounts dataclass."""

    def test_ratios(self):
        """Test ratio calculations."""
        counts = ParameterCounts(total=100, trainable=75, frozen=25)

        assert counts.trainable_ratio == 0.75
        assert counts.frozen_ratio == 0.25

    def test_zero_total(self):
        """Test ratios with zero total (edge case)."""
        counts = ParameterCounts(total=0, trainable=0, frozen=0)

        assert counts.trainable_ratio == 0.0
        assert counts.frozen_ratio == 0.0


# =============================================================================
# Tests for Summary Utilities
# =============================================================================


class TestSummarizePartition:
    """Tests for summarize_partition function."""

    def test_summary_contains_key_info(self):
        """Test that summary contains expected information."""
        model = SimpleModel()
        target_patterns = ["attn", "mlp"]

        summary = summarize_partition(model, target_patterns)

        # Check key sections are present
        assert "Parameter Partition Summary" in summary
        assert "Target patterns: ['attn', 'mlp']" in summary
        assert "Parameter Counts:" in summary
        assert "Total:" in summary
        assert "Trainable:" in summary
        assert "Frozen:" in summary
        assert "Matrix optimizer (target):" in summary
        assert "Vector optimizer (other):" in summary

    def test_summary_shows_percentages(self):
        """Test that summary shows percentages."""
        model = SimpleModel()
        summary = summarize_partition(model, ["attn"])

        assert "%" in summary

    def test_summary_shows_frozen_params(self):
        """Test that summary shows frozen parameters."""
        model = SimpleModel()
        model.embed.weight.requires_grad = False

        summary = summarize_partition(model, ["attn"])

        assert "Frozen parameters" in summary
        assert "embed.weight" in summary

    def test_summary_truncates_long_lists(self):
        """Test that parameter lists are truncated."""
        # Create model with many parameters
        model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(20)])

        summary = summarize_partition(model, ["0", "1"], max_params_to_show=5)

        assert "... and" in summary
        assert "more" in summary
