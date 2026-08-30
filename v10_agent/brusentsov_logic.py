"""
Brusentsov Ternary Logic Module

Ref: Engineering Specification V10.0, Section 3.1 - Brusentsov Ternary Logic
Ref: Engineering Specification V10.0, Section 5 - implies_brusentsov Engineering Realisation
Ref: Architectural Specification V10.0, Section 3 - Brusentsov Ternary Logic in Transition Judgment

This module implements the Brusentsov ternary logic system for empirical
transition judgment in the ARC-AGI-3 LCLD Agent Version 10.0.

The ternary logic distinguishes three states:
- TRUE (1, FOLLOW): Expected effect is necessarily contained in observed state
- FALSE (-1, NULL): Expected effect is physically contradicted
- IRRELEVANT (0, OMIT): Expected effect did not occur but no contradiction exists
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PropositionSet, AtomicProposition


class Ternary(Enum):
    """
    Brusentsov Ternary Logic Values.
    
    Ref: Spec 3.1 - Brusentsov Ternary (Engineering Specification)
    Ref: Spec 3.1 - Core mapping table (Architectural Specification)
    
    Maps to verifier conditions and branch effects:
    
    | Verifier Condition                                          | Value | Verdict | Branch Effect          |
    |-------------------------------------------------------------|-------|---------|------------------------|
    | Expected effect necessarily contained in observed state     | 1     | FOLLOW  | Trajectory continues   |
    | Expected effect physically contradicted (incompatibility)   | -1    | NULL    | Branch severed         |
    | Expected effect absent but no physical laws violated        | 0     | OMIT    | Branch kept alive      |
    
    The naming convention follows Brusentsov's original terminology:
    - TRUE corresponds to FOLLOW (necessary implication holds)
    - FALSE corresponds to NULL (nullity/contradiction detected)
    - IRRELEVANT corresponds to OMIT (inessential difference)
    """
    
    #: FOLLOW - Necessary implication holds; trajectory continues
    TRUE = 1
    
    #: NULL - Physical contradiction detected; branch must be severed
    FALSE = -1
    
    #: OMIT - No contradiction but expected effect absent; branch remains live
    IRRELEVANT = 0
    
    @property
    def verdict_name(self) -> str:
        """
        Get the human-readable verdict name for this ternary value.
        
        Returns:
            'FOLLOW' for TRUE, 'NULL' for FALSE, 'OMIT' for IRRELEVANT
        """
        _mapping = {
            Ternary.TRUE: "FOLLOW",
            Ternary.FALSE: "NULL",
            Ternary.IRRELEVANT: "OMIT",
        }
        return _mapping[self]
    
    def is_follow(self) -> bool:
        """Check if this is a FOLLOW judgment (trajectory continues)."""
        return self == Ternary.TRUE
    
    def is_null(self) -> bool:
        """Check if this is a NULL judgment (branch must be severed)."""
        return self == Ternary.FALSE
    
    def is_omit(self) -> bool:
        """Check if this is an OMIT judgment (branch remains viable)."""
        return self == Ternary.IRRELEVANT


def implies_brusentsov(expected: "PropositionSet", observed: "PropositionSet") -> Ternary:
    """
    Evaluate whether expected propositions are necessarily implied by observed propositions
    using Brusentsov's ternary logic semantics.
    
    Ref: Spec 3.1 - Brusentsov Ternary Logic (Engineering Specification)
    Ref: Spec 5 - implies_brusentsov Engineering Realisation
    Ref: Spec 3.2 - Necessary implication (Architectural Specification)
    
    This function implements the core judgment logic for the LayeredVerifier.
    It evaluates every real environment transition using Brusentsov's
    necessary-implication semantics rather than material implication.
    
    Necessary Implication Semantics:
    -------------------------------
    Following Brusentsov, necessary implication x ⇒ y means "the essence of y
    is entirely contained in the essence of x" (or "all x are y"). In the
    improved ternary DNF this is expressed as:
    
        xy ∨ xy'₀ ∨ x'y'
    
    where:
    - xy represents positive confirmation (TRUE/FOLLOW)
    - xy'₀ represents nullity/incompatibility with index 0 (FALSE/NULL)
    - x'y' represents omission of inessential differences (IRRELEVANT/OMIT)
    - The term x'y (material implication's extra term) is deliberately excluded
    
    Algorithm:
    ----------
    1. NULLITY CHECK: For each expected proposition e, check if any observed
       proposition o contradicts e. If contradiction found → return FALSE (NULL).
       
    2. NECESSARY CONTAINMENT: Check if every expected proposition e is necessarily
       contained in the observed set. If all are contained → return TRUE (FOLLOW).
       
    3. OMISSION: If no contradiction exists but some expected propositions are
       not contained, return IRRELEVANT (OMIT) - the missing effects are
       inessential for current judgment.
    
    Atomic Proposition Families:
    ----------------------------
    Propositions must be drawn from registered families (Spec 3.3):
    - object_identity: Object identity preservation/change
    - attribute_delta: Object attribute deltas (color, size, pattern, etc.)
    - positional_delta: Row/column/centroid signs
    - relation_existence: Relation existence/absence/error-metric sign
    - action_surface: Action-surface change (availability changes)
    - terminal_flag: Terminal/win/score metadata
    - affordance_flag: Controllability/affordance flags from Explorer
    
    IMPORTANT: Raw grid Hamming distance is NEVER used as a proposition.
    All propositions must be grounded on PlanningSet identifiers.
    
    Args:
        expected: PropositionSet containing expected atomic propositions
                  (what the DSL function declared would happen)
        observed: PropositionSet containing observed atomic propositions
                  (what actually happened after env.step)
    
    Returns:
        Ternary.TRUE (FOLLOW) if every expected atomic proposition is necessarily
                       contained in the observed set (necessary implication holds)
        Ternary.FALSE (NULL) if any expected proposition is incompatible with
                        the observed set (nullity/contradiction detected)
        Ternary.IRRELEVANT (OMIT) if expected set is not implied but no
                              incompatibility exists (missing effects inessential)
    
    Example Usage:
    --------------
    >>> from v10_agent.types import PropositionSet, AtomicProposition
    >>> expected = PropositionSet.create("hash1", [
    ...     AtomicProposition(family="metric_sign", data={"metric": "distance", "sign": -1})
    ... ])
    >>> observed = PropositionSet.create("hash2", [
    ...     AtomicProposition(family="metric_sign", data={"metric": "distance", "sign": -1})
    ... ])
    >>> result = implies_brusentsov(expected, observed)
    >>> result  # Should be Ternary.TRUE (FOLLOW)
    <Ternary.TRUE: 1>
    """
    # Extract proposition sets for iteration
    expected_props = expected.propositions if hasattr(expected, 'propositions') else set()
    observed_props = observed.propositions if hasattr(observed, 'propositions') else set()
    
    # =========================================================================
    # Step 1: Build incompatibility pairs (nullity check)
    # Ref: Spec 5, lines 329-332
    # =========================================================================
    for e in expected_props:
        if any(_contradicts(e, o) for o in observed_props):
            return Ternary.FALSE  # NULL - physical contradiction detected
    
    # =========================================================================
    # Step 2: Check necessary containment
    # Ref: Spec 5, lines 334-336
    # =========================================================================
    if all(_is_necessarily_contained(e, observed_props) for e in expected_props):
        return Ternary.TRUE  # FOLLOW - necessary implication holds
    
    # =========================================================================
    # Step 3: Otherwise the missing effects are treated as inessential
    # Ref: Spec 5, lines 338-339
    # =========================================================================
    return Ternary.IRRELEVANT  # OMIT - no contradiction but not all effects present


def _contradicts(expected: "AtomicProposition", observed: "AtomicProposition") -> bool:
    """
    Check if two atomic propositions contradict each other (nullity detection).
    
    Ref: Spec 5, line 331: "if any(contradicts(e, o) for o in observed)"
    
    Contradiction rules per family:
    - metric_sign: Same metric with opposite signs (+1 vs -1)
    - object_identity: Same object with different identity claims
    - attribute_delta: Same attribute with incompatible delta values
    - relation_existence: Same relation with existence vs absence
    - terminal_flag: Conflicting terminal state claims
    - positional_delta: Same object with incompatible row/column/centroid sign deltas
    - action_surface: Same action with incompatible availability claims
    - affordance_flag: Same affordance type with conflicting controllability claims
    
    This is an internal helper function. The actual implementation must
    be extended for each registered atomic proposition family.
    
    Args:
        expected: Expected atomic proposition
        observed: Observed atomic proposition
    
    Returns:
        True if the propositions are mutually incompatible (NULL condition)
    """
    # Different families cannot contradict each other directly
    if expected.family != observed.family:
        return False
    
    # Family-specific contradiction logic
    # NOTE: This is a stub implementation. Full implementation requires
    # explicit handling of each registered proposition family per Spec 3.3.
    
    if expected.family == "metric_sign":
        # Contradiction if same metric has opposite signs
        e_metric = expected.data.get("metric")
        o_metric = observed.data.get("metric")
        if e_metric == o_metric:
            e_sign = expected.data.get("sign")
            o_sign = observed.data.get("sign")
            if e_sign is not None and o_sign is not None and e_sign != o_sign:
                return True
    
    elif expected.family == "object_identity":
        # Contradiction if same object has conflicting identity claims
        e_obj_id = expected.data.get("object_id")
        o_obj_id = observed.data.get("object_id")
        if e_obj_id == o_obj_id:
            e_preserved = expected.data.get("preserved")
            o_preserved = observed.data.get("preserved")
            if e_preserved is not None and o_preserved is not None and e_preserved != o_preserved:
                return True
    
    elif expected.family == "attribute_delta":
        # Contradiction if same attribute has incompatible changes
        e_attr = expected.data.get("attribute")
        o_attr = observed.data.get("attribute")
        if e_attr == o_attr:
            e_obj = expected.data.get("object_id")
            o_obj = observed.data.get("object_id")
            if e_obj == o_obj:
                e_delta = expected.data.get("delta")
                o_delta = observed.data.get("delta")
                # Incompatible deltas for same attribute of same object
                if e_delta != o_delta:
                    return True
    
    elif expected.family == "relation_existence":
        # Contradiction if same relation has opposite existence claims
        e_rel = expected.data.get("relation_id")
        o_rel = observed.data.get("relation_id")
        if e_rel == o_rel:
            e_exists = expected.data.get("exists")
            o_exists = observed.data.get("exists")
            if e_exists is not None and o_exists is not None and e_exists != o_exists:
                return True
    
    elif expected.family == "terminal_flag":
        # Contradiction if terminal state claims conflict
        e_terminal = expected.data.get("is_terminal")
        o_terminal = observed.data.get("is_terminal")
        if e_terminal is not None and o_terminal is not None and e_terminal != o_terminal:
            return True
    
    elif expected.family == "positional_delta":
        # Contradiction if same object has incompatible positional deltas
        e_obj_id = expected.data.get("object_id")
        o_obj_id = observed.data.get("object_id")
        if e_obj_id == o_obj_id:
            # Check row delta sign contradiction
            e_row_sign = expected.data.get("row_sign")
            o_row_sign = observed.data.get("row_sign")
            if e_row_sign is not None and o_row_sign is not None and e_row_sign != o_row_sign:
                return True
            # Check column delta sign contradiction
            e_col_sign = expected.data.get("col_sign")
            o_col_sign = observed.data.get("col_sign")
            if e_col_sign is not None and o_col_sign is not None and e_col_sign != o_col_sign:
                return True
            # Check centroid delta sign contradiction
            e_centroid_row = expected.data.get("centroid_row_sign")
            o_centroid_row = observed.data.get("centroid_row_sign")
            if e_centroid_row is not None and o_centroid_row is not None and e_centroid_row != o_centroid_row:
                return True
            e_centroid_col = expected.data.get("centroid_col_sign")
            o_centroid_col = observed.data.get("centroid_col_sign")
            if e_centroid_col is not None and o_centroid_col is not None and e_centroid_col != o_centroid_col:
                return True
    
    elif expected.family == "action_surface":
        # Contradiction if same action has incompatible availability claims
        e_action = expected.data.get("action")
        o_action = observed.data.get("action")
        if e_action == o_action:
            e_available = expected.data.get("available")
            o_available = observed.data.get("available")
            if e_available is not None and o_available is not None and e_available != o_available:
                return True
    
    elif expected.family == "affordance_flag":
        # Contradiction if same affordance has conflicting controllability claims
        e_affordance = expected.data.get("affordance_type")
        o_affordance = observed.data.get("affordance_type")
        if e_affordance == o_affordance:
            e_controllable = expected.data.get("controllable")
            o_controllable = observed.data.get("controllable")
            if e_controllable is not None and o_controllable is not None and e_controllable != o_controllable:
                return True
    
    # No contradiction detected
    return False


def _is_necessarily_contained(prop: "AtomicProposition", observed_set: frozenset) -> bool:
    """
    Check if an atomic proposition is necessarily contained in the observed set.
    
    Ref: Spec 5, line 335: "if all(is_necessarily_contained(e, observed) for e in expected)"
    
    Necessary containment means the essence of the expected proposition is
    entirely contained in some observed proposition. This is stronger than
    simple equality - it allows for observed propositions that are more
    specific than expected ones.
    
    Containment rules per family:
    - metric_sign: Same metric with same or more specific sign information
    - object_identity: Same object with compatible identity information
    - attribute_delta: Same attribute with matching delta
    - relation_existence: Same relation with matching existence claim
    - terminal_flag: Matching terminal state claim
    - positional_delta: Same object with matching row/column/centroid signs
    - action_surface: Same action with matching availability
    - affordance_flag: Same affordance type with matching controllability
    
    Args:
        prop: The atomic proposition to check for containment
        observed_set: The set of observed atomic propositions
    
    Returns:
        True if prop is necessarily contained in observed_set
    """
    for obs in observed_set:
        if _prop_matches(prop, obs):
            return True
    return False


def _prop_matches(expected: "AtomicProposition", observed: "AtomicProposition") -> bool:
    """
    Check if an expected proposition matches an observed proposition.
    
    This is a helper for _is_necessarily_contained that checks if the
    observed proposition contains at least the information in the expected
    proposition (may be more specific).
    
    Args:
        expected: The expected atomic proposition
        observed: The observed atomic proposition
    
    Returns:
        True if observed contains the essence of expected
    """
    # Must be same family
    if expected.family != observed.family:
        return False
    
    # Check data containment based on family
    if expected.family == "metric_sign":
        e_metric = expected.data.get("metric")
        o_metric = observed.data.get("metric")
        if e_metric != o_metric:
            return False
        e_sign = expected.data.get("sign")
        o_sign = observed.data.get("sign")
        return e_sign == o_sign
    
    elif expected.family == "object_identity":
        return expected.data.get("object_id") == observed.data.get("object_id")
    
    elif expected.family == "attribute_delta":
        if expected.data.get("attribute") != observed.data.get("attribute"):
            return False
        if expected.data.get("object_id") != observed.data.get("object_id"):
            return False
        return expected.data.get("delta") == observed.data.get("delta")
    
    elif expected.family == "relation_existence":
        if expected.data.get("relation_id") != observed.data.get("relation_id"):
            return False
        return expected.data.get("exists") == observed.data.get("exists")
    
    elif expected.family == "terminal_flag":
        return expected.data.get("is_terminal") == observed.data.get("is_terminal")
    
    elif expected.family == "positional_delta":
        # Match if same object and all specified positional signs match
        if expected.data.get("object_id") != observed.data.get("object_id"):
            return False
        # Check row_sign if specified in expected
        e_row_sign = expected.data.get("row_sign")
        o_row_sign = observed.data.get("row_sign")
        if e_row_sign is not None and e_row_sign != o_row_sign:
            return False
        # Check col_sign if specified in expected
        e_col_sign = expected.data.get("col_sign")
        o_col_sign = observed.data.get("col_sign")
        if e_col_sign is not None and e_col_sign != o_col_sign:
            return False
        # Check centroid_row_sign if specified in expected
        e_centroid_row = expected.data.get("centroid_row_sign")
        o_centroid_row = observed.data.get("centroid_row_sign")
        if e_centroid_row is not None and e_centroid_row != o_centroid_row:
            return False
        # Check centroid_col_sign if specified in expected
        e_centroid_col = expected.data.get("centroid_col_sign")
        o_centroid_col = observed.data.get("centroid_col_sign")
        if e_centroid_col is not None and e_centroid_col != o_centroid_col:
            return False
        return True
    
    elif expected.family == "action_surface":
        # Match if same action and availability matches
        if expected.data.get("action") != observed.data.get("action"):
            return False
        e_available = expected.data.get("available")
        o_available = observed.data.get("available")
        if e_available is not None and e_available != o_available:
            return False
        return True
    
    elif expected.family == "affordance_flag":
        # Match if same affordance type and controllability matches
        if expected.data.get("affordance_type") != observed.data.get("affordance_type"):
            return False
        e_controllable = expected.data.get("controllable")
        o_controllable = observed.data.get("controllable")
        if e_controllable is not None and e_controllable != o_controllable:
            return False
        return True
    
    # Default: exact match required
    return expected.data == observed.data


# =============================================================================
# Registered Proposition Families Registry
# Ref: Spec 3.3 - Atomic proposition families (normative)
# =============================================================================

REGISTERED_PROPOSITION_FAMILIES = frozenset([
    "object_identity",      # Object identity preservation / change
    "attribute_delta",      # Object attribute deltas (color, size, pattern, ...)
    "positional_delta",     # Positional deltas (row/column/centroid signs)
    "relation_existence",   # Relation existence / absence / error-metric sign
    "action_surface",       # Action-surface change (availability of actions)
    "terminal_flag",        # Terminal / win / score metadata
    "affordance_flag",      # Controllability / affordance flags from Explorer
    "metric_sign",          # Metric delta signs for transitions
])


def register_proposition_family(family_name: str) -> None:
    """
    Register a new atomic proposition family.
    
    Ref: Spec 5, paragraph after line 342:
    "Families are registered in a single table; adding a new family requires
    an explicit engineering change and test."
    
    WARNING: Adding a new family requires updating _contradicts() and
    _is_necessarily_contained() functions above.
    
    Args:
        family_name: Name of the new proposition family to register
    
    Raises:
        ValueError: If family_name conflicts with existing registration
    """
    global REGISTERED_PROPOSITION_FAMILIES
    if family_name in REGISTERED_PROPOSITION_FAMILIES:
        raise ValueError(f"Proposition family '{family_name}' is already registered")
    REGISTERED_PROPOSITION_FAMILIES = REGISTERED_PROPOSITION_FAMILIES | {family_name}
