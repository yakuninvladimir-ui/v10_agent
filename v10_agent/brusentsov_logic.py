"""Brusentsov Ternary Logic Engine for empirical transition judgments.

Implements necessary implication semantics:
  - TRUE (1)       : FOLLOW - expected effect is necessarily contained in observed state.
  - FALSE (-1)     : NULL   - physical contradiction / nullity violation (hard branch sever).
  - IRRELEVANT (0) : OMIT   - expected effect did not occur, but no physical laws or invariants
                              were violated (branch paused for future growth/pivot).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from v10_agent.types import AtomicProposition, PropositionSet

if TYPE_CHECKING:
    from v10_agent.memory_contours import StructuredInvariant


class Ternary(Enum):
    """Brusentsov ternary truth values for transition verdicts."""
    TRUE = 1          # FOLLOW: Trajectory step confirmed; necessary containment held.
    FALSE = -1        # NULL: Hard contradiction; branch severed.
    IRRELEVANT = 0    # OMIT: Inessential / passive outcome; branch paused.

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Verdict):
            return other == self
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class EpistemicSignal(Enum):
    """Controller signals. Not truth values."""
    SEEK_EVIDENCE = 1


class Verdict(Enum):
    """Full judge verdict = logical value or epistemic signal."""
    FOLLOW = "FOLLOW"        # maps to Ternary.TRUE
    NULL = "NULL"            # maps to Ternary.FALSE
    OMIT = "OMIT"            # maps to Ternary.IRRELEVANT
    UNDECIDED = "UNDECIDED"  # maps to EpistemicSignal.SEEK_EVIDENCE

    @property
    def ternary(self) -> Ternary | None:
        if self is Verdict.FOLLOW:
            return Ternary.TRUE
        if self is Verdict.NULL:
            return Ternary.FALSE
        if self is Verdict.OMIT:
            return Ternary.IRRELEVANT
        return None

    @classmethod
    def from_ternary(cls, t: Ternary) -> "Verdict":
        if t is Ternary.TRUE:
            return cls.FOLLOW
        if t is Ternary.FALSE:
            return cls.NULL
        if t is Ternary.IRRELEVANT:
            return cls.OMIT
        raise ValueError(f"Cannot map {t} to Verdict")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Ternary):
            if self is Verdict.FOLLOW:
                return other is Ternary.TRUE
            if self is Verdict.NULL:
                return other is Ternary.FALSE
            if self is Verdict.OMIT:
                return other is Ternary.IRRELEVANT
            return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass(frozen=True)
class BrusentsovJudgment:
    """Auditable transition evaluation judgment grounded on Brusentsov logic."""
    trajectory_id: str
    step_id: str
    verdict: Verdict | Ternary
    expected_propositions: PropositionSet
    observed_propositions: PropositionSet
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)
    ambiguity_score: float | None = None
    evidence_hint: str | None = None
    matching_candidates: list[str] = field(default_factory=list)
    track_confidence_min: float | None = None

    def to_dict(self) -> dict[str, Any]:
        v_name = self.verdict.name if hasattr(self.verdict, "name") else str(self.verdict)
        v_val = self.verdict.value if hasattr(self.verdict, "value") else str(self.verdict)
        d: dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "step_id": self.step_id,
            "verdict": v_name,
            "verdict_value": v_val,
            "expected_count": len(self.expected_propositions),
            "observed_count": len(self.observed_propositions),
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }
        if self.ambiguity_score is not None:
            d["ambiguity_score"] = self.ambiguity_score
        if self.evidence_hint is not None:
            d["evidence_hint"] = self.evidence_hint
        if self.matching_candidates:
            d["matching_candidates"] = list(self.matching_candidates)
        if self.track_confidence_min is not None:
            d["track_confidence_min"] = self.track_confidence_min
        return d


def contradicts(expected: AtomicProposition, observed: AtomicProposition) -> bool:
    """Check if an observed proposition physically contradicts an expected proposition."""
    # 1. Object Identity preservation contradiction:
    # Expected object preserved, but observed destroyed/missing
    if expected.family == "object_identity":
        if expected.subject_id == observed.subject_id:
            if expected.predicate == "preserved" and observed.predicate in {"destroyed", "missing", "vanished"}:
                return True
            if expected.predicate in {"destroyed", "vanished"} and observed.predicate == "preserved":
                return True

    # 2. Attribute Delta contradiction:
    # Same subject, same attribute, but differing values (e.g. expected color 2, observed color 3)
    if expected.family == "attribute_delta" and observed.family == "attribute_delta":
        if expected.subject_id == observed.subject_id and expected.predicate == observed.predicate:
            if expected.value is not None and observed.value is not None:
                if expected.value != observed.value:
                    return True

    # 3. Metric Sign contradiction:
    # Expected change in a specific direction (+1 or -1), but observed opposite direction
    if expected.family == "metric_sign" and observed.family == "metric_sign":
        if expected.subject_id == observed.subject_id and expected.predicate == observed.predicate:
            if expected.secondary_id == observed.secondary_id:
                try:
                    exp_sign = int(expected.value)
                    obs_sign = int(observed.value)
                    # Contradiction if opposite sign (e.g. expected +1, observed -1)
                    if exp_sign != 0 and obs_sign != 0 and exp_sign != obs_sign:
                        return True
                    # If strictly non-zero expected delta was anticipated, but movement was strictly zero and blocked
                    if exp_sign != 0 and obs_sign == 0 and expected.predicate.endswith("_delta"):
                        # Zero displacement when non-zero was expected = physical wall/blockage = NULLITY
                        return True
                except (ValueError, TypeError):
                    pass

    # 4. Relation Existence contradiction:
    # Same pair of objects, contradictory relational state
    if expected.family == "relation_existence" and observed.family == "relation_existence":
        if (
            expected.subject_id == observed.subject_id
            and expected.secondary_id == observed.secondary_id
            and expected.predicate == observed.predicate
        ):
            if expected.value is not None and observed.value is not None:
                if bool(expected.value) != bool(observed.value):
                    return True

    # 5. Terminal Metadata contradiction:
    # e.g., Expected win, but observed game_over
    if expected.family == "terminal_metadata" and observed.family == "terminal_metadata":
        if expected.predicate == "win" and observed.predicate in {"game_over", "lost"}:
            return True

    # 6. Spatial position contradiction:
    # Object cannot be at two different positions simultaneously
    if expected.family == "spatial_position" and observed.family == "spatial_position":
        if expected.subject_id == observed.subject_id:
            if expected.value != observed.value:
                return True

    # 7. Area conservation contradiction:
    # Movement should not change object area (unless explicitly expected to scale)
    if expected.family == "area_conservation" and observed.family == "area_conservation":
        if expected.subject_id == observed.subject_id:
            if expected.value != observed.value:
                return True

    return False


def _normalize_value(v: Any) -> Any:
    """Normalize proposition value for safe type-coerced equality comparisons."""
    if isinstance(v, str):
        v_s = v.strip()
        if v_s.lower() == "true":
            return True
        if v_s.lower() == "false":
            return False
        try:
            return int(v_s)
        except ValueError:
            try:
                return float(v_s)
            except ValueError:
                return v_s
    return v


def is_necessarily_contained(expected: AtomicProposition, observed_set: PropositionSet) -> bool:
    """Check if the expected proposition is necessarily contained in the observed proposition set."""
    for obs in observed_set:
        if obs.family != expected.family:
            continue
        if obs.subject_id != expected.subject_id:
            continue
        if obs.predicate != expected.predicate:
            continue
        if expected.secondary_id is not None and obs.secondary_id != expected.secondary_id:
            continue

        # If value is specified, verify compatibility
        if expected.value is not None:
            if obs.value is None:
                continue
            if _normalize_value(obs.value) != _normalize_value(expected.value):
                continue

        return True

    return False


def implies_brusentsov(expected: PropositionSet, observed: PropositionSet) -> Ternary:
    """Evaluate necessary implication following Brusentsov ternary logic.

    Returns:
      TRUE (1)       : Every expected atomic proposition is necessarily contained in the observed set.
      FALSE (-1)     : Any expected proposition is physically contradicted (incompatibility / nullity).
      IRRELEVANT (0) : The expected set is not implied, yet no incompatibility exists (inessential missing effect).
    """
    if len(expected) == 0:
        # Trivial fulfillment
        return Ternary.TRUE

    # 1. Incompatibility check (NULL check)
    for e in expected:
        # Check against every observed proposition
        for o in observed:
            if contradicts(e, o):
                return Ternary.FALSE

        # Object preservation check
        if e.family == "object_identity" and e.predicate == "preserved":
            is_destroyed = any(
                o.family == "object_identity"
                and o.subject_id == e.subject_id
                and o.predicate in {"destroyed", "missing", "vanished"}
                for o in observed
            )
            if is_destroyed:
                return Ternary.FALSE

    # 2. Necessary containment check (FOLLOW check)
    if all(is_necessarily_contained(e, observed) for e in expected):
        return Ternary.TRUE

    # 3. Inessential missing effect without physical contradiction (OMIT check)
    return Ternary.IRRELEVANT
def evaluate_invariant_across_levels(
    invariant: "StructuredInvariant",
    current_level_observations: PropositionSet,
) -> Ternary:
    """Evaluate whether an invariant holds in a new level context.
    
    TRUE: observations directly confirm the invariant
    FALSE: observations directly contradict the invariant  
    IRRELEVANT: insufficient observations to confirm or deny
    """
    if not current_level_observations or not current_level_observations.propositions:
        return Ternary.IRRELEVANT

    inv_type = invariant.invariant_type.lower()
    desc = invariant.description.lower()
    meta = getattr(invariant, "metadata", {}) or {}

    # 1. Kinematics invariants
    if inv_type in ("kinematics", "physics"):
        act_id = meta.get("action_id") or ""
        if not act_id:
            import re
            m = re.search(r"action\s*(action\d+|\d+)", desc, re.IGNORECASE)
            if m:
                raw_act = m.group(1).upper()
                act_id = raw_act if raw_act.startswith("ACTION") else f"ACTION{raw_act}"

        relevant_metric_props = [
            p for p in current_level_observations.propositions
            if p.family in ("metric_sign", "attribute_delta")
        ]

        if not relevant_metric_props:
            return Ternary.IRRELEVANT

        # Check for blocked or zero displacement contradiction
        has_motion = any(
            p.family == "metric_sign" and p.predicate.endswith("_delta") and p.value not in (0, None)
            for p in relevant_metric_props
        )
        has_blocked_zero = any(
            p.family == "metric_sign" and p.predicate.endswith("_delta") and p.value == 0
            for p in relevant_metric_props
        )

        if "blocked" in desc or "wall" in desc:
            if has_blocked_zero:
                return Ternary.TRUE
        elif has_motion and ("moves" in desc or "step" in desc or "displace" in desc):
            return Ternary.TRUE
        elif has_blocked_zero and ("moves" in desc or "displace" in desc) and not has_motion:
            return Ternary.FALSE

    # 2. Area conservation invariants
    if inv_type in ("area_conservation", "topology") or "area" in desc:
        area_props = [
            p for p in current_level_observations.propositions
            if p.predicate == "area" or p.family == "area_conservation"
        ]
        if area_props:
            has_destroyed = any(
                p.family == "object_identity" and p.predicate in ("destroyed", "vanished")
                for p in current_level_observations.propositions
            )
            if has_destroyed and "conserv" in desc:
                return Ternary.FALSE
            return Ternary.TRUE

    # 3. Spatial position invariants
    p_pos = meta.get("target_position")
    if inv_type in ("spatial_position", "positional_pattern") or p_pos:
        pos_props = [
            p for p in current_level_observations.propositions
            if p.family == "spatial_position"
        ]
        if pos_props:
            for p in pos_props:
                if p_pos and p.value == p_pos:
                    return Ternary.TRUE
                elif p_pos and p.value != p_pos and p.subject_id == meta.get("subject_id"):
                    return Ternary.FALSE

    # 4. Symmetry and relational invariants
    if inv_type in ("symmetry", "axial_symmetry_vertical", "axial_symmetry_horizontal", "socket_coverage"):
        rel_props = [
            p for p in current_level_observations.propositions
            if p.family == "relation_existence"
        ]
        if rel_props:
            for p in rel_props:
                if inv_type in str(p.predicate).lower() or inv_type in str(p.value).lower():
                    return Ternary.TRUE if p.value else Ternary.FALSE

    return Ternary.IRRELEVANT
