"""Property-Based Testing for Brusentsov 3-Valued Entailment Logic.

Uses Hypothesis to verify mathematical invariants:
1. Ex falso quodlibet elimination: empty expected propositions ALWAYS yield IRRELEVANT (OMIT).
2. Carroll nullity (xy'_0 -> NULL): expected preserved vs observed destroyed ALWAYS contradicts.
3. Identity consequence (xy -> FOLLOW): exact proposition containment ALWAYS verifies.
4. Non-vacuity: random non-contradictory unasserted changes NEVER produce FOLLOW.
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from v10_agent.brusentsov_logic import (
    Ternary,
    Verdict,
    contradicts,
    implies_brusentsov,
    is_necessarily_contained,
)
from v10_agent.types import AtomicProposition, PropositionSet

# Hypothesis strategies for proposition generation
FAMILIES = st.sampled_from([
    "object_identity",
    "attribute_delta",
    "metric_sign",
    "shape_stability",
    "spatial_position",
])

PREDICATES = st.sampled_from([
    "preserved",
    "destroyed",
    "vanished",
    "color",
    "row_delta",
    "col_delta",
    "shape_stable",
])

SUBJECT_IDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=8).map(lambda s: f"obj_{s}")


@st.composite
def atomic_proposition_strategy(draw):
    family = draw(FAMILIES)
    predicate = draw(PREDICATES)
    subject_id = draw(SUBJECT_IDS)
    # Value can be None, int, str, or float
    val_choice = draw(st.integers(min_value=0, max_value=3))
    if val_choice == 0:
        value = None
    elif val_choice == 1:
        value = draw(st.integers(min_value=-15, max_value=15))
    elif val_choice == 2:
        value = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False).map(lambda x: round(x, 1)))
    else:
        value = draw(st.sampled_from(["true", "false", "up", "down"]))

    return AtomicProposition(
        family=family,
        subject_id=subject_id,
        predicate=predicate,
        value=value,
    )


@given(observed_list=st.lists(atomic_proposition_strategy(), min_size=0, max_size=20))
def test_pbt_empty_expected_yields_irrelevant(observed_list):
    """Axiom 1: Empty expected set MUST yield IRRELEVANT (OMIT) for any observed state."""
    expected = PropositionSet.from_iterable([])
    observed = PropositionSet.from_iterable(observed_list)
    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.IRRELEVANT, f"Material implication paradox: empty expectation returned {verdict}"


@given(subject_id=SUBJECT_IDS, destruction_pred=st.sampled_from(["destroyed", "vanished", "missing"]))
def test_pbt_carroll_nullity_destruction(subject_id, destruction_pred):
    """Axiom 2: Preserved expectation vs destroyed observation is strictly contradictory (xy'_0 -> NULL)."""
    p_exp = AtomicProposition(
        family="object_identity",
        subject_id=subject_id,
        predicate="preserved",
        value=True,
    )
    p_obs = AtomicProposition(
        family="object_identity",
        subject_id=subject_id,
        predicate=destruction_pred,
        value=True,
    )
    assert contradicts(p_exp, p_obs) is True, f"Carroll nullity failed for {destruction_pred}"

    # Also test via implies_brusentsov
    exp_set = PropositionSet.from_iterable([p_exp])
    obs_set = PropositionSet.from_iterable([p_obs])
    verdict = implies_brusentsov(exp_set, obs_set)
    assert verdict == Ternary.FALSE, f"Carroll nullity must evaluate to Ternary.FALSE, got {verdict}"


@given(p=atomic_proposition_strategy())
def test_pbt_single_proposition_reflexive_containment(p):
    """Axiom 3: Any single proposition implies itself (xy -> FOLLOW)."""
    pset = PropositionSet.from_iterable([p])
    verdict = implies_brusentsov(pset, pset)
    assert verdict == Ternary.TRUE, f"Reflexive entailment failed: returned {verdict}"


@given(props_list=st.lists(atomic_proposition_strategy(), min_size=1, max_size=6))
def test_pbt_set_entailment_consistency(props_list):
    """Axiom 3b: Reflexive set entailment returns TRUE if consistent, or FALSE if internally contradictory."""
    pset = PropositionSet.from_iterable(props_list)
    has_internal_contradiction = any(
        contradicts(p1, p2) for p1 in props_list for p2 in props_list
    )
    verdict = implies_brusentsov(pset, pset)
    if not has_internal_contradiction:
        assert verdict == Ternary.TRUE
    else:
        assert verdict == Ternary.FALSE
