"""
Utilities for selecting and partitioning model parameters into optimizer groups.

This module provides functions to separate model parameters based on naming
patterns, following PEFT's target_modules pattern matching convention.

Pattern Matching Behavior:
    The pattern matching follows HuggingFace PEFT's convention:

    1. **Simple strings** (no regex metacharacters): Suffix matching
       - Pattern "attn" matches "layers.0.attn" but NOT "layers.0.attn_proj"
       - Pattern "q_proj" matches "layers.0.self_attn.q_proj"
       - Specifically: matches if ``name == pattern`` or ``name.endswith("." + pattern)``

    2. **Regex patterns** (contains metacharacters like ., *, ^, $, etc.): Full regex match
       - Pattern ".*attn.*" matches any name containing "attn"
       - Pattern r"layers\\.\\d+\\.attn" matches "layers.0.attn", "layers.1.attn", etc.
       - Uses ``re.fullmatch(pattern, name)``

    Regex metacharacters detected: . ^ $ * + ? { } [ ] | ( ) \\

    Migration from substring matching:
        If you previously used patterns like ["attn", "mlp"] expecting substring matching,
        update to [".*attn.*", ".*mlp.*"] for equivalent behavior. However, the PEFT-style
        suffix matching is recommended as it's more precise and ecosystem-compatible.

Design Rationale:
    Using name-based selection rather than shape-based selection because:
    1. Embeddings are 2D but should NOT use Muon
    2. Some 2D params (like lm_head) may need special treatment
    3. Name patterns are explicit and self-documenting

Architecture:
    The module follows a layered design:

    1. **Primitives** (most flexible):
       - :func:`select_parameters`: Select params matching patterns
       - :func:`select_parameters_with_names`: Same, but returns (name, param) tuples

    2. **Convenience Functions** (built on primitives):
       - :func:`partition_parameters`: Split into target/other (2-group case)
       - :func:`partition_parameters_multi`: Split into N groups (N-group case)

    Users needing custom logic (e.g., custom overlap resolution) should use
    the primitives directly rather than extending the convenience functions.

Overlap Handling:
    When a parameter matches multiple patterns (e.g., "attn_mlp_proj" matches
    both "attn" and "mlp"), the behavior depends on the function and settings:

    - :func:`select_parameters`: Returns the param (no overlap concern for single selection)
    - :func:`partition_parameters`: Target patterns are checked, no overlap issue
    - :func:`partition_parameters_multi`: Configurable via `overlap_strategy`:
        - "first" (default): First matching group wins, with optional warning
        - "error": Raise ValueError if any param matches multiple groups

    For custom overlap resolution strategies (e.g., priority weights, custom
    predicates), use :func:`select_parameters` directly and implement your
    own logic. This keeps the library simple while allowing full flexibility.

Example:
    Basic partitioning for Muon + AdamW training::

        from research_lib.training.param_utils import partition_parameters

        # Suffix matching: matches layers.*.attn, layers.*.mlp, etc.
        target_modules = ["attn", "mlp", "q_proj", "k_proj", "v_proj"]
        muon_params, adam_params = partition_parameters(
            model=model,
            target_patterns=target_modules,
        )

    Using regex for more control::

        # Match all attention-related params (substring-like behavior)
        target_modules = [".*attn.*", ".*mlp.*"]
        muon_params, adam_params = partition_parameters(model, target_modules)

        # Match specific layer indices
        target_modules = [r"layers\\.[0-5]\\.attn"]  # Only layers 0-5
        selected, _ = partition_parameters(model, target_modules)

See Also:
    - HuggingFace PEFT's target_modules pattern: https://huggingface.co/docs/peft
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch.nn as nn
from torch import Tensor

# =============================================================================
# Pattern Matching
# =============================================================================

# Characters that indicate a pattern should be treated as regex
_REGEX_METACHARACTERS = set(r"\.^$*+?{}[]|()")


def _is_regex_pattern(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters.

    Args:
        pattern: The pattern to check.

    Returns:
        True if the pattern contains regex metacharacters, False otherwise.
    """
    return any(c in pattern for c in _REGEX_METACHARACTERS)


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a parameter name matches a pattern.

    Follows PEFT's target_modules matching convention:
    - Simple strings: suffix matching (checks if pattern appears as a complete
      component in the module path, ignoring the final .weight/.bias suffix)
    - Regex patterns: full regex match via re.fullmatch

    Args:
        name: The full parameter name (e.g., "model.layers.0.self_attn.q_proj.weight").
        pattern: The pattern to match against.

    Returns:
        True if the name matches the pattern, False otherwise.

    Example:
        >>> _matches_pattern("layers.0.attn.weight", "attn")
        True
        >>> _matches_pattern("layers.0.attn_proj.weight", "attn")
        False
        >>> _matches_pattern("layers.0.attn_proj.weight", ".*attn.*")
        True
    """
    if _is_regex_pattern(pattern):
        # Regex pattern: use fullmatch on the full name
        return re.fullmatch(pattern, name) is not None
    else:
        # Simple string: suffix matching (PEFT convention)
        # For parameter names, we need to check the module path components
        # e.g., "layers.0.attn.weight" should match pattern "attn"
        #
        # Split by "." and check if pattern matches any component exactly,
        # OR if the name (minus common suffixes) ends with the pattern
        parts = name.split(".")

        # Check exact match with any component
        if pattern in parts:
            return True

        # Also check if pattern matches the module name (excluding .weight/.bias/etc)
        # This handles cases like pattern="q_proj" matching "layers.0.q_proj.weight"
        if len(parts) >= 2:
            module_name = parts[-2]  # Second to last is usually the module name
            if module_name == pattern:
                return True

        # Original PEFT check: full name ends with .pattern
        return name == pattern or name.endswith(f".{pattern}")


# =============================================================================
# Primitives: Core Selection Functions
# =============================================================================


def select_parameters(
    model: nn.Module,
    patterns: List[str],
    require_grad: bool = True,
) -> List[Tensor]:
    """Select model parameters whose names match any of the given patterns.

    This is the core primitive for parameter selection. All other partitioning
    functions are built on top of this.

    Pattern matching follows PEFT's convention:
    - Simple strings (no regex chars): suffix matching
      e.g., "attn" matches "layers.0.attn" but not "layers.0.attn_proj"
    - Regex patterns: full match via re.fullmatch
      e.g., ".*attn.*" matches anything containing "attn"

    Args:
        model: The model whose parameters to select from.
        patterns: List of patterns to match against parameter names.
            A parameter is selected if ANY pattern matches.
            Pass an empty list to select nothing.
        require_grad: If True, only consider parameters with requires_grad=True.
            Default: True.

    Returns:
        List of parameter tensors matching the patterns.

    Example:
        >>> model = GPT(...)
        >>> # Suffix matching (recommended, PEFT-compatible)
        >>> attn_params = select_parameters(model, ["attn", "q_proj", "v_proj"])
        >>>
        >>> # Regex for substring matching
        >>> attn_params = select_parameters(model, [".*attn.*"])
        >>>
        >>> print(f"Selected {len(attn_params)} attention parameters")

    Note:
        If patterns is empty, returns an empty list.
        If no parameters match, returns an empty list.
    """
    if not patterns:
        return []

    selected = []
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        if any(_matches_pattern(name, pattern) for pattern in patterns):
            selected.append(param)

    return selected


def select_parameters_with_names(
    model: nn.Module,
    patterns: List[str],
    require_grad: bool = True,
) -> List[Tuple[str, Tensor]]:
    """Select model parameters with their names, matching any of the given patterns.

    Same as :func:`select_parameters` but returns (name, param) tuples for
    debugging, logging, or custom processing.

    Pattern matching follows PEFT's convention (see :func:`select_parameters`).

    Args:
        model: The model whose parameters to select from.
        patterns: List of patterns to match against parameter names.
        require_grad: If True, only consider parameters with requires_grad=True.

    Returns:
        List of (name, param) tuples for matching parameters.

    Example:
        >>> named_params = select_parameters_with_names(model, ["attn"])
        >>> for name, param in named_params:
        ...     print(f"{name}: {param.shape}")
    """
    if not patterns:
        return []

    selected = []
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        if any(_matches_pattern(name, pattern) for pattern in patterns):
            selected.append((name, param))

    return selected


# =============================================================================
# Convenience Functions: Built on Primitives
# =============================================================================


def partition_parameters(
    model: nn.Module,
    target_patterns: List[str],
    require_grad: bool = True,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Partition model parameters into target and non-target groups.

    This is a convenience function for the common 2-group case (e.g., Muon
    for attention/MLP weights, AdamW for embeddings). It is built on
    :func:`select_parameters`.

    Parameters whose names match any of the target_patterns are placed in
    the first group (target_params). All other parameters go to the second
    group (other_params).

    Args:
        model: The model whose parameters to partition.
        target_patterns: List of patterns (suffixes or regex) to match against parameter names.
            A parameter matches if it satisfies the PEFT matching convention for ANY pattern.
            Examples: ["attn", "mlp"], ["q_proj", "v_proj", "c_fc"].
        require_grad: If True, only include parameters with requires_grad=True.
            Default: True.

    Returns:
        Tuple of (target_params, other_params):
            - target_params: List of parameters matching target_patterns
            - other_params: List of all other (non-matching) parameters

    Example:
        >>> model = GPT(...)
        >>> target_patterns = ["attn", "mlp"]
        >>> muon_params, adam_params = partition_parameters(model, target_patterns)
        >>> print(f"Muon: {len(muon_params)} params, Adam: {len(adam_params)} params")

    Note:
        If target_patterns is empty, all parameters go to other_params.
        This enables single-optimizer training by passing target_patterns=[].

    See Also:
        :func:`partition_parameters_multi` for partitioning into 3+ groups.
        :func:`select_parameters` for the underlying primitive.
    """
    # Use primitive to get target params
    target_params = select_parameters(model, target_patterns, require_grad)
    target_set = set(id(p) for p in target_params)

    # Collect non-target params
    other_params = []
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        if id(param) not in target_set:
            other_params.append(param)

    return target_params, other_params


def partition_parameters_with_names(
    model: nn.Module,
    target_patterns: List[str],
    require_grad: bool = True,
) -> Tuple[List[Tuple[str, Tensor]], List[Tuple[str, Tensor]]]:
    """Partition model parameters into target and non-target groups, with names.

    Same as :func:`partition_parameters` but returns (name, param) tuples
    for debugging and logging purposes.

    Args:
        model: The model whose parameters to partition.
        target_patterns: List of patterns (suffixes or regex) to match against parameter names.
        require_grad: If True, only include parameters with requires_grad=True.

    Returns:
        Tuple of (target_named_params, other_named_params), where each is a
        list of (name, param) tuples.

    Example:
        >>> muon_named, adam_named = partition_parameters_with_names(model, ["attn"])
        >>> for name, param in muon_named:
        ...     print(f"Muon: {name}, shape={param.shape}")
    """
    # Use primitive to get target params with names
    target_named = select_parameters_with_names(model, target_patterns, require_grad)
    target_names = set(name for name, _ in target_named)

    # Collect non-target params with names
    other_named = []
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        if name not in target_names:
            other_named.append((name, param))

    return target_named, other_named


def partition_parameters_multi(
    model: nn.Module,
    pattern_groups: List[List[str]],
    require_grad: bool = True,
    overlap_strategy: Literal["first", "error"] = "first",
    warn_on_overlap: bool = True,
) -> List[List[Tensor]]:
    """Partition model parameters into multiple groups based on pattern lists.

    This is a convenience function for N-group partitioning. Each parameter
    is assigned to exactly one group based on which pattern group it matches.
    It is built on :func:`select_parameters_with_names`.

    The last pattern group can be an empty list [] to act as a catch-all for
    any parameters not matching previous groups.

    Args:
        model: The model whose parameters to partition.
        pattern_groups: List of pattern lists. Each inner list contains patterns
            for one group. A parameter is assigned to the first group whose
            patterns it matches. An empty list [] matches all remaining params.
            Example: [["attn"], ["mlp"], []] for attn/mlp/other split.
        require_grad: If True, only include parameters with requires_grad=True.
            Default: True.
        overlap_strategy: How to handle parameters matching multiple groups:
            - "first" (default): Assign to first matching group. If warn_on_overlap
              is True, emit a warning listing overlapping parameters.
            - "error": Raise ValueError if any parameter matches multiple groups.
        warn_on_overlap: If True and overlap_strategy="first", emit a warning
            when parameters match multiple groups. Default: True.

    Returns:
        List of parameter lists, one per pattern group. The order matches
        the order of pattern_groups.

    Raises:
        ValueError: If overlap_strategy="error" and a parameter matches
            multiple non-empty pattern groups.

    Example:
        Three-way split for different optimizers::

            pattern_groups = [
                ["attn", "qkv"],    # Group 0: Attention (Muon)
                ["mlp", "c_fc"],    # Group 1: MLP (Muon with different LR)
                [],                  # Group 2: Everything else (AdamW)
            ]
            attn_params, mlp_params, other_params = partition_parameters_multi(
                model, pattern_groups
            )

    Note:
        For complex overlap resolution (e.g., priority weights, custom predicates),
        use :func:`select_parameters` directly and implement your own logic.
        This function intentionally keeps overlap handling simple.

    See Also:
        :func:`partition_parameters` for the simpler 2-group case.
        :func:`select_parameters` for building custom partitioning logic.
    """
    if not pattern_groups:
        return []

    # Initialize result groups
    result: List[List[Tensor]] = [[] for _ in pattern_groups]

    # Track assignments for overlap detection
    param_to_groups: Dict[str, List[int]] = {}  # name -> list of matching group indices

    # First pass: determine which groups each param matches
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        matching_groups = []
        for group_idx, patterns in enumerate(pattern_groups):
            if not patterns:
                # Empty pattern list = catch-all (matches everything)
                # But only counts as a match if no other group matched
                continue
            if any(_matches_pattern(name, pattern) for pattern in patterns):
                matching_groups.append(group_idx)

        param_to_groups[name] = matching_groups

    # Check for overlaps (only among non-empty pattern groups)
    overlapping_params = {
        name: groups for name, groups in param_to_groups.items() if len(groups) > 1
    }

    if overlapping_params:
        if overlap_strategy == "error":
            overlap_msg = "\n".join(
                f"  {name}: matches groups {groups}"
                for name, groups in overlapping_params.items()
            )
            raise ValueError(
                f"Parameters match multiple pattern groups:\n{overlap_msg}\n"
                f"Use overlap_strategy='first' or adjust patterns to avoid overlap."
            )
        elif overlap_strategy == "first" and warn_on_overlap:
            overlap_msg = "\n".join(
                f"  {name}: matches groups {groups}, assigned to group {groups[0]}"
                for name, groups in overlapping_params.items()
            )
            warnings.warn(
                f"Parameters match multiple pattern groups (using first match):\n{overlap_msg}",
                UserWarning,
                stacklevel=2,
            )

    # Second pass: assign params to groups
    assigned_params: set = set()

    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        matching_groups = param_to_groups.get(name, [])

        if matching_groups:
            # Assign to first matching group
            result[matching_groups[0]].append(param)
            assigned_params.add(id(param))

    # Handle catch-all groups (empty pattern lists)
    for group_idx, patterns in enumerate(pattern_groups):
        if not patterns:  # Empty = catch-all
            for name, param in model.named_parameters():
                if require_grad and not param.requires_grad:
                    continue
                if id(param) not in assigned_params:
                    result[group_idx].append(param)
                    assigned_params.add(id(param))

    return result


def partition_parameters_multi_with_names(
    model: nn.Module,
    pattern_groups: List[List[str]],
    require_grad: bool = True,
    overlap_strategy: Literal["first", "error"] = "first",
    warn_on_overlap: bool = True,
) -> List[List[Tuple[str, Tensor]]]:
    """Partition model parameters into multiple groups, with names.

    Same as :func:`partition_parameters_multi` but returns (name, param) tuples.

    Args:
        model: The model whose parameters to partition.
        pattern_groups: List of pattern lists for each group.
        require_grad: If True, only include parameters with requires_grad=True.
        overlap_strategy: "first" or "error" for overlap handling.
        warn_on_overlap: If True, warn on overlaps when strategy is "first".

    Returns:
        List of (name, param) tuple lists, one per pattern group.

    See Also:
        :func:`partition_parameters_multi` for details on arguments and behavior.
    """
    if not pattern_groups:
        return []

    # Initialize result groups
    result: List[List[Tuple[str, Tensor]]] = [[] for _ in pattern_groups]

    # Track assignments for overlap detection
    param_to_groups: Dict[str, List[int]] = {}

    # First pass: determine which groups each param matches
    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        matching_groups = []
        for group_idx, patterns in enumerate(pattern_groups):
            if not patterns:
                continue
            if any(_matches_pattern(name, pattern) for pattern in patterns):
                matching_groups.append(group_idx)

        param_to_groups[name] = matching_groups

    # Check for overlaps
    overlapping_params = {
        name: groups for name, groups in param_to_groups.items() if len(groups) > 1
    }

    if overlapping_params:
        if overlap_strategy == "error":
            overlap_msg = "\n".join(
                f"  {name}: matches groups {groups}"
                for name, groups in overlapping_params.items()
            )
            raise ValueError(
                f"Parameters match multiple pattern groups:\n{overlap_msg}\n"
                f"Use overlap_strategy='first' or adjust patterns to avoid overlap."
            )
        elif overlap_strategy == "first" and warn_on_overlap:
            overlap_msg = "\n".join(
                f"  {name}: matches groups {groups}, assigned to group {groups[0]}"
                for name, groups in overlapping_params.items()
            )
            warnings.warn(
                f"Parameters match multiple pattern groups (using first match):\n{overlap_msg}",
                UserWarning,
                stacklevel=2,
            )

    # Second pass: assign params to groups
    assigned_names: set = set()

    for name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        matching_groups = param_to_groups.get(name, [])

        if matching_groups:
            result[matching_groups[0]].append((name, param))
            assigned_names.add(name)

    # Handle catch-all groups
    for group_idx, patterns in enumerate(pattern_groups):
        if not patterns:
            for name, param in model.named_parameters():
                if require_grad and not param.requires_grad:
                    continue
                if name not in assigned_names:
                    result[group_idx].append((name, param))
                    assigned_names.add(name)

    return result


# =============================================================================
# Counting and Summary Utilities
# =============================================================================


@dataclass
class ParameterCounts:
    """Container for parameter count statistics.

    Attributes:
        total: Total number of parameters in the model.
        trainable: Number of parameters with requires_grad=True.
        frozen: Number of parameters with requires_grad=False.
    """

    total: int
    trainable: int
    frozen: int

    @property
    def trainable_ratio(self) -> float:
        """Fraction of parameters that are trainable."""
        return self.trainable / self.total if self.total > 0 else 0.0

    @property
    def frozen_ratio(self) -> float:
        """Fraction of parameters that are frozen."""
        return self.frozen / self.total if self.total > 0 else 0.0


def count_parameters(params: List[Tensor]) -> int:
    """Count total number of scalar parameters in a list.

    Args:
        params: List of parameter tensors.

    Returns:
        Total number of scalar parameters (sum of numel()).

    Example:
        >>> params = [torch.randn(10, 20), torch.randn(5)]
        >>> count_parameters(params)
        205
    """
    return sum(p.numel() for p in params)


def count_parameters_by_status(model: nn.Module) -> ParameterCounts:
    """Count parameters by trainable/frozen status.

    Args:
        model: The model to analyze.

    Returns:
        ParameterCounts dataclass with total, trainable, and frozen counts.

    Example:
        >>> model = GPT(...)
        >>> model.embed.weight.requires_grad = False  # Freeze embedding
        >>> counts = count_parameters_by_status(model)
        >>> print(f"Total: {counts.total:,}, Trainable: {counts.trainable:,}, Frozen: {counts.frozen:,}")
    """
    total = 0
    trainable = 0
    frozen = 0

    for param in model.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
        else:
            frozen += numel

    return ParameterCounts(total=total, trainable=trainable, frozen=frozen)


def summarize_partition(
    model: nn.Module,
    target_patterns: List[str],
    max_params_to_show: int = 10,
) -> str:
    """Generate a summary string of parameter partitioning.

    Provides a comprehensive overview including:
    - Total/trainable/frozen parameter counts
    - Partition of trainable parameters by target_patterns
    - Sample parameter names from each group

    Args:
        model: The model to summarize.
        target_patterns: Target module patterns for partitioning.
        max_params_to_show: Maximum number of parameter names to list
            per group. Default: 10.

    Returns:
        Multi-line string summarizing the partition.

    Example:
        >>> print(summarize_partition(model, ["attn", "mlp"]))
        Parameter Partition Summary
        ===========================
        Target patterns: ['attn', 'mlp']

        Parameter Counts:
        ├── Total: 15,345,678
        ├── Trainable: 12,000,000 (78.2%)
        └── Frozen: 3,345,678 (21.8%)

        Trainable Parameters by Optimizer:
        ├── Matrix optimizer (target): 9,500,000 (79.2% of trainable)
        └── Vector optimizer (other): 2,500,000 (20.8% of trainable)
        ...
    """
    # Get overall counts
    counts = count_parameters_by_status(model)

    # Partition trainable parameters
    target_named, other_named = partition_parameters_with_names(
        model, target_patterns, require_grad=True
    )

    target_count = count_parameters([p for _, p in target_named])
    other_count = count_parameters([p for _, p in other_named])

    # Calculate percentages
    trainable_pct = 100 * counts.trainable_ratio
    frozen_pct = 100 * counts.frozen_ratio

    target_pct = 100 * target_count / counts.trainable if counts.trainable > 0 else 0
    other_pct = 100 * other_count / counts.trainable if counts.trainable > 0 else 0

    # Get frozen parameter names for display
    frozen_named = [
        (name, param)
        for name, param in model.named_parameters()
        if not param.requires_grad
    ]

    # Build summary
    lines = [
        "Parameter Partition Summary",
        "===========================",
        f"Target patterns: {target_patterns}",
        "",
        "Parameter Counts:",
        f"├── Total: {counts.total:,}",
        f"├── Trainable: {counts.trainable:,} ({trainable_pct:.1f}%)",
        f"└── Frozen: {counts.frozen:,} ({frozen_pct:.1f}%)",
        "",
        "Trainable Parameters by Optimizer:",
        f"├── Matrix optimizer (target): {target_count:,} ({target_pct:.1f}% of trainable)",
        f"└── Vector optimizer (other): {other_count:,} ({other_pct:.1f}% of trainable)",
    ]

    # Add target parameter samples
    if target_named:
        lines.append("")
        lines.append("Target parameters (matrix optimizer):")
        for name, param in target_named[:max_params_to_show]:
            lines.append(f"  {name}: {tuple(param.shape)}")
        if len(target_named) > max_params_to_show:
            lines.append(f"  ... and {len(target_named) - max_params_to_show} more")

    # Add other parameter samples
    if other_named:
        lines.append("")
        lines.append("Other parameters (vector optimizer):")
        for name, param in other_named[:max_params_to_show]:
            lines.append(f"  {name}: {tuple(param.shape)}")
        if len(other_named) > max_params_to_show:
            lines.append(f"  ... and {len(other_named) - max_params_to_show} more")

    # Add frozen parameter samples
    if frozen_named:
        lines.append("")
        lines.append("Frozen parameters (not trained):")
        for name, param in frozen_named[:max_params_to_show]:
            lines.append(f"  {name}: {tuple(param.shape)}")
        if len(frozen_named) > max_params_to_show:
            lines.append(f"  ... and {len(frozen_named) - max_params_to_show} more")

    return "\n".join(lines)
