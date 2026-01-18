"""Tests for parameter selection and partitioning utilities."""

import warnings

import pytest
import torch
import torch.nn as nn

from research_lib.training.param_utils import (
    ParameterCounts,
    _is_regex_pattern,
    _matches_pattern,
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
    """Simple model for testing partitioning.

    Structure:
        - embed: Embedding(100, 64) -> 1 param (embed.weight)
        - layers.0.attn: Linear(64, 64, bias=False) -> 1 param (layers.0.attn.weight)
        - layers.0.mlp: Linear(64, 64, bias=False) -> 1 param (layers.0.mlp.weight)
        - ln: LayerNorm(64) -> 2 params (ln.weight, ln.bias)
        - lm_head: Linear(64, 100, bias=False) -> 1 param (lm_head.weight)

    Total: 6 parameters
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        # Use nested structure to test suffix matching properly
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(64, 64, bias=False),
                        "mlp": nn.Linear(64, 64, bias=False),
                    }
                )
            ]
        )
        self.ln = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 100, bias=False)


class OverlapModel(nn.Module):
    """Model with parameter names that can match multiple regex patterns.

    Structure (all Linear with bias=False, so 1 param each):
        - attn_proj.weight
        - mlp_proj.weight
        - attn_mlp_shared.weight
        - other.weight

    Total: 4 parameters

    Note: With PEFT-style suffix matching:
        - "attn_proj" matches attn_proj.weight
        - "mlp_proj" matches mlp_proj.weight
        - For overlap testing, use regex patterns like ".*attn.*" and ".*mlp.*"
    """

    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(64, 64, bias=False)
        self.mlp_proj = nn.Linear(64, 64, bias=False)
        self.attn_mlp_shared = nn.Linear(64, 64, bias=False)  # Matches both regex!
        self.other = nn.Linear(64, 64, bias=False)


# =============================================================================
# Tests for regex matching helpers
# =============================================================================


class TestPatternMatching:
    """Tests for PEFT-style pattern matching."""

    def test_suffix_matching_exact(self):
        """Test that exact name matches."""
        assert _matches_pattern("attn", "attn") is True
        assert _matches_pattern("q_proj", "q_proj") is True

    def test_suffix_matching_with_module_path(self):
        """Test suffix matching with module path prefix and .weight suffix."""
        # Pattern should match when it's a component in the path
        assert _matches_pattern("layers.0.attn.weight", "attn") is True
        assert (
            _matches_pattern("model.layers.0.self_attn.q_proj.weight", "q_proj") is True
        )
        assert (
            _matches_pattern("encoder.layer.0.attention.self.query.weight", "query")
            is True
        )

    def test_suffix_matching_partial_no_match(self):
        """Test that partial matches (pattern is substring but not component) don't match."""
        # "attn" should NOT match "attn_proj" because "attn" != "attn_proj"
        assert _matches_pattern("layers.0.attn_proj.weight", "attn") is False
        assert _matches_pattern("layers.0.attn_output.weight", "attn") is False

    def test_suffix_matching_component_match(self):
        """Test that pattern matches as a path component."""
        # "attention" as a component should match
        assert _matches_pattern("encoder.attention.self.weight", "attention") is True
        # But "attn" should not match "attention"
        assert _matches_pattern("encoder.attention.self.weight", "attn") is False

    def test_regex_pattern_substring(self):
        """Test regex patterns for substring matching."""
        # Regex .*attn.* should match anything containing "attn"
        assert _matches_pattern("layers.0.attn_proj.weight", ".*attn.*") is True
        assert _matches_pattern("self_attn.query.weight", ".*attn.*") is True
        assert (
            _matches_pattern("model.attention.output.weight", ".*attention.*") is True
        )
        # Note: ".*attn.*" does NOT match "attention" (no substring "attn")
        assert _matches_pattern("model.attention.output.weight", ".*attn.*") is False

    def test_regex_pattern_specific(self):
        """Test specific regex patterns."""
        # Match only layers 0-5
        pattern = r"layers\.[0-5]\.attn\.weight"
        assert _matches_pattern("layers.0.attn.weight", pattern) is True
        assert _matches_pattern("layers.5.attn.weight", pattern) is True
        assert _matches_pattern("layers.6.attn.weight", pattern) is False
        assert _matches_pattern("layers.10.attn.weight", pattern) is False

    def test_regex_pattern_fullmatch(self):
        """Test that regex uses fullmatch, not search."""
        # This pattern should only match the exact structure
        pattern = r"layers\.\d+\.attn\.weight"
        assert _matches_pattern("layers.0.attn.weight", pattern) is True
        assert (
            _matches_pattern("model.layers.0.attn.weight", pattern) is False
        )  # Has prefix
        assert (
            _matches_pattern("layers.0.attn.weight.extra", pattern) is False
        )  # Has suffix

    def test_is_regex_pattern(self):
        """Test detection of regex metacharacters."""
        # Simple strings - no regex
        assert _is_regex_pattern("attn") is False
        assert _is_regex_pattern("q_proj") is False
        assert _is_regex_pattern("self_attn") is False

        # Regex patterns - contain metacharacters
        assert _is_regex_pattern(".*attn.*") is True
        assert _is_regex_pattern(r"layers\.\d+") is True
        assert _is_regex_pattern("attn|mlp") is True
        assert _is_regex_pattern("(attn)") is True
        assert _is_regex_pattern("attn$") is True
        assert _is_regex_pattern("^attn") is True


# =============================================================================
# Tests for Primitives: select_parameters
# =============================================================================


class TestSelectParameters:
    """Tests for select_parameters primitive."""

    def test_select_matching_params(self):
        """Test selecting parameters that match patterns."""
        model = SimpleModel()
        # Suffix matching: "attn" matches "layers.0.attn.weight"
        params = select_parameters(model, ["attn"])
        assert len(params) == 1  # layers.0.attn.weight

    def test_select_multiple_patterns(self):
        """Test selecting with multiple patterns."""
        model = SimpleModel()
        params = select_parameters(model, ["attn", "mlp"])
        assert len(params) == 2  # layers.0.attn.weight, layers.0.mlp.weight

    def test_select_with_regex(self):
        """Test selecting with regex patterns."""
        model = SimpleModel()
        # Regex to match layer params (anything under layers.*)
        params = select_parameters(model, [r"layers\..*"])
        assert len(params) == 2  # attn and mlp weights

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
        # Freeze the attention layer
        for param in model.layers[0]["attn"].parameters():
            param.requires_grad = False

        params = select_parameters(model, ["attn", "mlp"], require_grad=True)
        # Only mlp should be selected (attn is frozen)
        assert len(params) == 1

    def test_select_includes_frozen_when_flag_false(self):
        """Test that frozen params are included when require_grad=False."""
        model = SimpleModel()
        for param in model.layers[0]["attn"].parameters():
            param.requires_grad = False

        params = select_parameters(model, ["attn", "mlp"], require_grad=False)
        assert len(params) == 2


class TestSelectParametersWithNames:
    """Tests for select_parameters_with_names primitive."""

    def test_returns_names(self):
        """Test that names are returned alongside params."""
        model = SimpleModel()
        named_params = select_parameters_with_names(model, ["attn"])

        assert len(named_params) == 1
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

        # layers.0.attn.weight, layers.0.mlp.weight = 2 params
        assert len(target_params) == 2

        # embed.weight, ln.weight, ln.bias, lm_head.weight = 4 params
        assert len(other_params) == 4

    def test_partition_with_regex(self):
        """Test partitioning with regex patterns."""
        model = SimpleModel()
        # Use regex to match layer params
        target_patterns = [r"layers\..*"]

        target_params, other_params = partition_parameters(model, target_patterns)

        assert len(target_params) == 2  # attn, mlp weights
        assert len(other_params) == 4  # embed, ln (2), lm_head

    def test_partition_empty_targets(self):
        """Test that empty targets puts all params in other."""
        model = SimpleModel()
        target_patterns = []

        target_params, other_params = partition_parameters(model, target_patterns)

        assert len(target_params) == 0
        assert len(other_params) == 6  # All 6 params

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

        # Target should have attn params
        assert len(target_named) == 1
        for name, param in target_named:
            assert isinstance(name, str)
            assert isinstance(param, torch.Tensor)
            assert "attn" in name

        # Other should not have attn in the module name (but might have in path)
        assert len(other_named) == 5
        for name, param in other_named:
            assert isinstance(name, str)
            assert isinstance(param, torch.Tensor)
            # The module component should not be "attn"
            parts = name.split(".")
            assert "attn" not in parts[:-1]  # Exclude .weight/.bias from check


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
        assert len(groups[0]) == 1  # attn.weight
        assert len(groups[1]) == 1  # mlp.weight
        assert len(groups[2]) == 4  # embed, ln (2), lm_head

    def test_catch_all_only(self):
        """Test with only a catch-all group."""
        model = SimpleModel()
        pattern_groups = [[]]

        groups = partition_parameters_multi(model, pattern_groups)

        assert len(groups) == 1
        assert len(groups[0]) == 6  # All 6 params

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
        assert len(groups[0]) == 1  # attn.weight
        assert len(groups[1]) == 1  # mlp.weight
        # embed, ln, lm_head are unassigned (4 params)!

    def test_overlap_first_match_wins(self):
        """Test that first match wins with overlap_strategy='first'."""
        model = OverlapModel()
        # Use regex patterns to create overlap on attn_mlp_shared
        pattern_groups = [
            [".*attn.*"],  # Group 0: regex matches attn_proj, attn_mlp_shared
            [".*mlp.*"],  # Group 1: regex matches mlp_proj, attn_mlp_shared
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
        assert len(groups[0]) == 2  # attn_proj + attn_mlp_shared
        assert len(groups[1]) == 1  # mlp_proj only (attn_mlp_shared went to group 0)
        assert len(groups[2]) == 1  # other

    def test_overlap_no_warning_when_disabled(self):
        """Test that warning can be disabled."""
        model = OverlapModel()
        pattern_groups = [[".*attn.*"], [".*mlp.*"], []]

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
        pattern_groups = [[".*attn.*"], [".*mlp.*"], []]

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

        # Use regex to match first two layers
        summary = summarize_partition(model, [r"0\..*", r"1\..*"], max_params_to_show=5)

        assert "... and" in summary
        assert "more" in summary
