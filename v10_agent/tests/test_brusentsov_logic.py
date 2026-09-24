"""Unit tests for Brusentsov Ternary Logic and necessary implication."""

from __future__ import annotations

import pytest

from v10_agent.brusentsov_logic import (
    BrusentsovJudgment,
    Ternary,
    contradicts,
    implies_brusentsov,
    is_necessarily_contained,
)
from v10_agent.types import AtomicProposition, PropositionSet


def test_proposition_family_validation():
    # Valid family
    p = AtomicProposition(family="object_identity", subject_id="obj_0", predicate="preserved")
    assert p.family == "object_identity"

    # Invalid family raises ValueError
    with pytest.raises(ValueError, match="Unknown proposition family"):
        AtomicProposition(family="raw_grid_pixel", subject_id="p1", predicate="color")


def test_brusentsov_follow():
    # Expected: obj_0 moved with positive row delta, color preserved
    expected = PropositionSet.from_iterable([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=1),
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="color", value=2),
    ])

    # Observed contains exact expected propositions plus extra side-effect
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=1),
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="color", value=2),
        AtomicProposition(family="relation_existence", subject_id="obj_0", predicate="touches", secondary_id="obj_1", value=True),
    ])

    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.TRUE  # FOLLOW


def test_brusentsov_null_contradiction_metric():
    # Expected: row_delta +1 (moved down)
    expected = PropositionSet.from_iterable([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=1),
    ])

    # Observed: row_delta -1 (moved up) -> Physical contradiction
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=-1),
    ])

    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.FALSE  # NULL


def test_brusentsov_null_contradiction_attribute():
    # Expected: color changed to 2 (red)
    expected = PropositionSet.from_iterable([
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="color", value=2),
    ])

    # Observed: color changed to 3 (green) -> Contradiction
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="color", value=3),
    ])

    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.FALSE  # NULL


def test_brusentsov_null_destroyed_object_is_contradiction():
    """Carroll nullity xy'_0 → NULL: expected=preserved + observed=destroyed is a physical contradiction."""
    # Expected: obj_0 preserved
    expected = PropositionSet.from_iterable([
        AtomicProposition(family="object_identity", subject_id="obj_0", predicate="preserved"),
    ])

    # Observed: obj_0 was destroyed (Carroll nullity — direct physical contradiction)
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="object_identity", subject_id="obj_0", predicate="destroyed"),
    ])

    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.FALSE  # NULL: Carroll nullity xy'_0 severs the trajectory


def test_brusentsov_omit_passive_outcome():
    # Expected: color changed to 2
    expected = PropositionSet.from_iterable([
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="color", value=2),
    ])

    # Observed: no-op / no color delta observed, but no physical laws or invariants violated
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="object_identity", subject_id="obj_0", predicate="preserved"),
    ])

    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.IRRELEVANT  # OMIT


def test_type_coercion_normalization():
    # String "1" vs integer 1
    p_str = AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value="1")
    p_int = AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=1)
    obs_set = PropositionSet.from_iterable([p_int])
    assert is_necessarily_contained(p_str, obs_set) is True

    # Float "3.5" vs 3.5
    p_fstr = AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="scale", value="3.5")
    p_float = AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="scale", value=3.5)
    obs_fset = PropositionSet.from_iterable([p_float])
    assert is_necessarily_contained(p_fstr, obs_fset) is True

    # Bool "true" vs True
    p_bstr = AtomicProposition(family="relation_existence", subject_id="obj_0", predicate="touches", secondary_id="obj_1", value="true")
    p_bool = AtomicProposition(family="relation_existence", subject_id="obj_0", predicate="touches", secondary_id="obj_1", value=True)
    obs_bset = PropositionSet.from_iterable([p_bool])
    assert is_necessarily_contained(p_bstr, obs_bset) is True

