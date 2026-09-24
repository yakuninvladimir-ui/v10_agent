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
    action_dict: dict[str, Any] = field(default_factory=dict)
    is_effective: bool = False

    @property
    def ternary_verdict(self) -> Verdict | Ternary:
        """Compatibility property matching Solver epistemic prompt expectations."""
        return self.verdict

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
            "is_effective": self.is_effective,
        }
        if self.action_dict:
            d["action_dict"] = dict(self.action_dict)
        if self.ambiguity_score is not None:
            d["ambiguity_score"] = self.ambiguity_score
        if self.evidence_hint is not None:
            d["evidence_hint"] = self.evidence_hint
        if self.matching_candidates:
            d["matching_candidates"] = list(self.matching_candidates)
        if self.track_confidence_min is not None:
            d["track_confidence_min"] = self.track_confidence_min
        return d


def _unpack_dy_dx(v: Any) -> tuple[int, int] | None:
    """Safely unpack a 2D motion vector (dy, dx) from tuple, list, or string."""
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        try:
            return int(v[0]), int(v[1])
        except (ValueError, TypeError):
            return None
    if isinstance(v, str):
        cleaned = v.strip().strip("()[]")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except (ValueError, TypeError):
                return None
    return None


def contradicts(expected: AtomicProposition, observed: AtomicProposition) -> bool:
    """Check if an observed proposition physically contradicts an expected proposition (Carrollian nullity xy'_0)."""
    # 0. Subject matching: propositions regarding different distinct entities cannot directly contradict each other
    if expected.subject_id and observed.subject_id and expected.subject_id != observed.subject_id:
        return False

    # 1. Object Identity preservation (Carroll nullity xy'_0 → NULL):
    # When the antecedent explicitly expects preservation and observation confirms destruction,
    # this is a direct physical contradiction — the object cannot be simultaneously preserved and destroyed.
    if expected.family == "object_identity" and observed.family == "object_identity":
        if expected.subject_id == observed.subject_id or not expected.subject_id:
            if expected.predicate == "preserved" and observed.predicate in ("destroyed", "vanished", "missing"):
                return True
            if expected.predicate in ("destroyed", "vanished", "missing") and observed.predicate == "preserved":
                return True

    # 2. Attribute Delta contradiction:
    # Same subject, same attribute, but differing values (e.g. expected color 2, observed color 3)
    if expected.family == "attribute_delta" and observed.family == "attribute_delta":
        if (expected.subject_id == observed.subject_id or not expected.subject_id) and expected.predicate == observed.predicate:
            if expected.value is not None and observed.value is not None:
                if _normalize_value(expected.value) != _normalize_value(observed.value):
                    return True

    # 3. Kinematic motion & Invariant Nullity (Physical Stagnation, Unintended Mutation, Direction Inversion)
    exp_p = expected.predicate.lower()
    obs_p = observed.predicate.lower()
    subj_matches = (expected.subject_id == observed.subject_id or not expected.subject_id)

    # 3a. Tuple-based kinematics: ("moved", "step_moved")
    if subj_matches and exp_p in ("moved", "step_moved") and obs_p in ("moved", "step_moved"):
        exp_vec = _unpack_dy_dx(expected.value)
        obs_vec = _unpack_dy_dx(observed.value)
        if exp_vec is not None and obs_vec is not None:
            exp_dy, exp_dx = exp_vec
            obs_dy, obs_dx = obs_vec
            # 1. Stagnation: expected motion on an axis, but observed stationary (0)
            if (exp_dy != 0 and obs_dy == 0) or (exp_dx != 0 and obs_dx == 0):
                return True
            # 2. Unintended Mutation: expected 0 on an axis, but observed motion
            if (exp_dy == 0 and obs_dy != 0) or (exp_dx == 0 and obs_dx != 0):
                return True
            # 3. Direction Inversion
            if (exp_dy * obs_dy < 0) or (exp_dx * obs_dx < 0):
                return True

    # 3b. Invariant violation: expected unchanged / stationary, but observed motion
    if subj_matches and exp_p in ("unchanged", "stationary"):
        if obs_p in ("moved", "step_moved"):
            obs_vec = _unpack_dy_dx(observed.value)
            if obs_vec is not None and (obs_vec[0] != 0 or obs_vec[1] != 0):
                return True
        elif obs_p in ("row_delta", "delta_r", "dy", "col_delta", "delta_c", "dx"):
            try:
                if int(observed.value) != 0:
                    return True
            except (ValueError, TypeError):
                pass

    # 3c. Scalar metric sign contradictions (dy / dx / row_delta / col_delta)
    if expected.family == "metric_sign" and observed.family == "metric_sign" and subj_matches:
        is_row = exp_p in ("row_delta", "delta_r", "dy") and obs_p in ("row_delta", "delta_r", "dy")
        is_col = exp_p in ("col_delta", "delta_c", "dx") and obs_p in ("col_delta", "delta_c", "dx")
        if is_row or is_col:
            if expected.secondary_id == observed.secondary_id:
                try:
                    exp_sign = int(expected.value)
                    obs_sign = int(observed.value)
                    # 1. Stagnation: expected motion, observed 0
                    if exp_sign != 0 and obs_sign == 0:
                        return True
                    # 2. Unintended mutation: expected 0, observed motion
                    if exp_sign == 0 and obs_sign != 0:
                        return True
                    # 3. Direction inversion
                    if exp_sign * obs_sign < 0:
                        return True
                except (ValueError, TypeError):
                    pass

    # 3d. Cross-check: tuple expected vs scalar observed
    if subj_matches and exp_p in ("moved", "step_moved") and observed.family == "metric_sign":
        exp_vec = _unpack_dy_dx(expected.value)
        if exp_vec is not None:
            exp_dy, exp_dx = exp_vec
            if obs_p in ("row_delta", "delta_r", "dy"):
                try:
                    obs_dy = int(observed.value)
                    if (exp_dy != 0 and obs_dy == 0) or (exp_dy == 0 and obs_dy != 0) or (exp_dy * obs_dy < 0):
                        return True
                except (ValueError, TypeError):
                    pass
            elif obs_p in ("col_delta", "delta_c", "dx"):
                try:
                    obs_dx = int(observed.value)
                    if (exp_dx != 0 and obs_dx == 0) or (exp_dx == 0 and obs_dx != 0) or (exp_dx * obs_dx < 0):
                        return True
                except (ValueError, TypeError):
                    pass

    # 3e. Cross-check: scalar expected vs tuple observed
    if subj_matches and expected.family == "metric_sign" and obs_p in ("moved", "step_moved"):
        obs_vec = _unpack_dy_dx(observed.value)
        if obs_vec is not None:
            obs_dy, obs_dx = obs_vec
            if exp_p in ("row_delta", "delta_r", "dy"):
                try:
                    exp_dy = int(expected.value)
                    if (exp_dy != 0 and obs_dy == 0) or (exp_dy == 0 and obs_dy != 0) or (exp_dy * obs_dy < 0):
                        return True
                except (ValueError, TypeError):
                    pass
            elif exp_p in ("col_delta", "delta_c", "dx"):
                try:
                    exp_dx = int(expected.value)
                    if (exp_dx != 0 and obs_dx == 0) or (exp_dx == 0 and obs_dx != 0) or (exp_dx * obs_dx < 0):
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
    if isinstance(v, (list, tuple)):
        return tuple(_normalize_value(x) for x in v)
    if isinstance(v, str):
        v_s = v.strip()
        if v_s.lower() == "true":
            return True
        if v_s.lower() == "false":
            return False
        if (v_s.startswith("(") and v_s.endswith(")")) or (v_s.startswith("[") and v_s.endswith("]")):
            inner = v_s[1:-1]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            return tuple(_normalize_value(p) for p in parts)
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
        # Trivial fulfillment / silence is inessential neutrality (x'y'), NOT necessary consequence (xy)
        return Ternary.IRRELEVANT

    # 1. Incompatibility check (NULL check)
    for e in expected:
        # Check against every observed proposition
        for o in observed:
            if contradicts(e, o):
                return Ternary.FALSE


    # 2. Necessary containment check (FOLLOW check)
    if all(is_necessarily_contained(e, observed) for e in expected):
        return Ternary.TRUE

    # 3. Inessential missing effect without physical contradiction (OMIT check)
    return Ternary.IRRELEVANT
def evaluate_invariant_across_levels(
    invariant: "StructuredInvariant",
    current_level_observations: Any,
) -> Ternary:
    """Evaluate whether an invariant holds in a new level context.
    
    Supports PropositionSet, PlanningSet, or raw observation dictionary.
    TRUE: observations directly confirm the invariant
    FALSE: observations directly contradict the invariant  
    IRRELEVANT: insufficient observations to confirm or deny
    """
    if current_level_observations is None:
        return Ternary.IRRELEVANT

    props: list[AtomicProposition] = []
    if isinstance(current_level_observations, PropositionSet):
        props = list(current_level_observations.propositions)
    elif hasattr(current_level_observations, "propositions"):
        props = list(current_level_observations.propositions)
    elif hasattr(current_level_observations, "objects") and hasattr(current_level_observations, "relations"):
        # Extracted directly from PlanningSet
        for obj in current_level_observations.objects:
            props.append(AtomicProposition(family="object_identity", subject_id=obj.id, predicate="preserved"))
            props.append(AtomicProposition(family="attribute_delta", subject_id=obj.id, predicate="color", value=obj.color))
            props.append(AtomicProposition(family="attribute_delta", subject_id=obj.id, predicate="area", value=obj.area))
            props.append(AtomicProposition(family="spatial_position", subject_id=obj.id, predicate="centroid", value=(obj.centroid.row, obj.centroid.col)))
        for rel in current_level_observations.relations:
            props.append(AtomicProposition(
                family="relation_existence",
                subject_id=rel.subject_id,
                predicate=rel.relation_type,
                value=True,
                secondary_id=rel.target_id,
            ))
    elif isinstance(current_level_observations, dict) and "grid" in current_level_observations:
        from v10_agent.arga_lite import extract_arga_snapshot
        snap = extract_arga_snapshot(current_level_observations["grid"])
        for obj in snap.objects:
            props.append(AtomicProposition(family="object_identity", subject_id=obj.id, predicate="preserved"))
            props.append(AtomicProposition(family="attribute_delta", subject_id=obj.id, predicate="color", value=obj.color))
            props.append(AtomicProposition(family="attribute_delta", subject_id=obj.id, predicate="area", value=obj.area))
            props.append(AtomicProposition(family="spatial_position", subject_id=obj.id, predicate="centroid", value=(obj.centroid.row, obj.centroid.col)))
        for rel in snap.relations:
            props.append(AtomicProposition(
                family="relation_existence",
                subject_id=rel.subject_id,
                predicate=rel.relation_type,
                value=True,
                secondary_id=rel.target_id,
            ))

    if not props:
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

        if not act_id:
            # Guard: On pristine frame S₀ (no actions taken), kinematic invariants
            # cannot be empirically confirmed or falsified. Return IRRELEVANT.
            return Ternary.IRRELEVANT

        relevant_metric_props = [
            p for p in props
            if p.family in ("metric_sign", "attribute_delta")
        ]

        if not relevant_metric_props:
            return Ternary.IRRELEVANT

        # Check for blocked or zero displacement contradiction
        has_motion = any(
            p.family == "metric_sign" and (p.predicate.endswith("_delta") or p.predicate in ("dy", "dx", "delta_r", "delta_c")) and p.value not in (0, None)
            for p in relevant_metric_props
        )
        has_blocked_zero = any(
            p.family == "metric_sign" and (p.predicate.endswith("_delta") or p.predicate in ("dy", "dx", "delta_r", "delta_c")) and p.value == 0
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
            p for p in props
            if p.predicate == "area" or p.family == "area_conservation"
        ]
        if area_props:
            has_destroyed = any(
                p.family == "object_identity" and p.predicate in ("destroyed", "vanished")
                for p in props
            )
            if has_destroyed and "conserv" in desc:
                return Ternary.FALSE
            return Ternary.TRUE

    # 3. Spatial position invariants
    p_pos = meta.get("target_position")
    if inv_type in ("spatial_position", "positional_pattern") or p_pos:
        pos_props = [
            p for p in props
            if p.family == "spatial_position"
        ]
        if pos_props:
            for p in pos_props:
                if p_pos and p.value == p_pos:
                    return Ternary.TRUE
                elif p_pos and p.value != p_pos and p.subject_id == meta.get("subject_id"):
                    return Ternary.FALSE

    # 4. Symmetry and relational invariants
    if inv_type in ("symmetry", "axial_symmetry_vertical", "axial_symmetry_horizontal", "socket_coverage", "alignment"):
        rel_props = [
            p for p in props
            if p.family == "relation_existence"
        ]
        if rel_props:
            matching = any(
                inv_type in str(p.predicate).lower() or str(p.predicate).lower() in inv_type
                for p in rel_props if p.value
            )
            if matching:
                return Ternary.TRUE

    # 5. Goal & Victory Invariants
    if inv_type in ("goal", "victory", "win_condition") or "win" in desc or "goal" in desc:
        term_props = [
            p for p in props
            if p.family == "terminal_outcome" or p.predicate in ("won", "lost", "in_progress")
        ]
        if term_props:
            for p in term_props:
                if p.predicate == "won" and ("win" in desc or "goal" in desc):
                    return Ternary.TRUE
                elif p.predicate == "lost" and ("win" in desc or "goal" in desc):
                    return Ternary.FALSE

    # 6. Control & Selection Invariants
    if inv_type in ("control", "selection", "modality") or "toggle" in desc or "switch" in desc:
        ctrl_props = [
            p for p in props
            if p.family in ("control_scheme", "selection_mechanics") or "toggle" in str(p.predicate)
        ]
        if ctrl_props:
            return Ternary.TRUE

    # 7. Palette & Color/Role Invariants
    if inv_type in ("palette", "color_role") or "color" in desc or "palette" in desc:
        color_props = [
            p for p in props
            if p.family == "attribute_delta" and p.predicate == "color"
        ]
        if color_props:
            return Ternary.TRUE

    # 8. Transformation & State Change Invariants
    if inv_type in ("transformation", "state_flip", "rotation") or "rotate" in desc or "flip" in desc:
        tf_props = [
            p for p in props
            if p.family in ("transformation", "attribute_delta", "object_identity")
        ]
        if tf_props:
            return Ternary.TRUE

    return Ternary.IRRELEVANT
