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

from v10_agent.action_semantics import parse_referenced_action_id
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
        # Must agree with __eq__: Ternary.TRUE equals Verdict.FOLLOW, so both
        # spellings of one truth value have to hash identically. Delegating to
        # the underlying ternary integer keeps each truth value to a single
        # bucket and lets set()/dict()/Counter arithmetic stay correct.
        return hash(self.value)


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
        # __eq__ identifies FOLLOW/NULL/OMIT with TRUE/FALSE/IRRELEVANT, so those
        # three have to hash as their ternary integer - the same integer
        # Ternary.__hash__ produces - or equal spellings would land in different
        # buckets. UNDECIDED has no ternary counterpart and compares equal to
        # nothing but itself, so it hashes on its own name.
        ternary = self.ternary
        if ternary is None:
            return hash(self.value)
        return hash(ternary.value)


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

    def __post_init__(self) -> None:
        # Two spellings of one judgement (Verdict.FOLLOW and Ternary.TRUE)
        # compare equal, so the stored value is normalised to the Verdict form.
        # Otherwise to_dict() would emit "FOLLOW" or "TRUE" for the same
        # logical result and any consumer keyed on that token would disagree
        # with == .
        if isinstance(self.verdict, Ternary):
            object.__setattr__(self, "verdict", Verdict.from_ternary(self.verdict))

    @property
    def ternary_verdict(self) -> Verdict | Ternary:
        """Compatibility property matching Solver epistemic prompt expectations."""
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        verdict = self.verdict
        v_name = verdict.name if hasattr(verdict, "name") else str(verdict)
        v_val = verdict.value if hasattr(verdict, "value") else str(verdict)
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


def _sign(x: int | float) -> int:
    """Return sign of a number (-1, 0, or 1)."""
    return (x > 0) - (x < 0)


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
            # Case 1: Both zero in expected -> zero motion expected
            if exp_dy == 0 and exp_dx == 0:
                if obs_dy != 0 or obs_dx != 0:
                    return True
            else:
                # Case 2: Active vector -> check ONLY non-zero components of expected vector!
                # Zero components in expected mean axis unconstrained by this action.
                if exp_dy != 0 and _sign(exp_dy) != _sign(obs_dy):
                    return True
                if exp_dx != 0 and _sign(exp_dx) != _sign(obs_dx):
                    return True
                return False

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
                    # Check only non-zero expected scalar; exp_sign==0 does not forbid movement
                    if exp_sign != 0 and _sign(exp_sign) != _sign(obs_sign):
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
                    if exp_dy == 0 and exp_dx == 0:
                        if obs_dy != 0:
                            return True
                    elif exp_dy != 0 and _sign(exp_dy) != _sign(obs_dy):
                        return True
                except (ValueError, TypeError):
                    pass
            elif obs_p in ("col_delta", "delta_c", "dx"):
                try:
                    obs_dx = int(observed.value)
                    if exp_dy == 0 and exp_dx == 0:
                        if obs_dx != 0:
                            return True
                    elif exp_dx != 0 and _sign(exp_dx) != _sign(obs_dx):
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
                    if exp_dy != 0 and _sign(exp_dy) != _sign(obs_dy):
                        return True
                except (ValueError, TypeError):
                    pass
            elif exp_p in ("col_delta", "delta_c", "dx"):
                try:
                    exp_dx = int(expected.value)
                    if exp_dx != 0 and _sign(exp_dx) != _sign(obs_dx):
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
            norm_obs = _normalize_value(obs.value)
            norm_exp = _normalize_value(expected.value)
            if norm_obs != norm_exp:
                # Allow cumulative multi-step displacement expectations (e.g. step k expecting k * step_delta)
                # when direction signs match on both axes and non-zero components are exact integer multiples
                if expected.family == "metric_sign":
                    pred_low = expected.predicate.lower()
                    if pred_low in ("moved", "step_moved"):
                        exp_vec = _unpack_dy_dx(norm_exp)
                        obs_vec = _unpack_dy_dx(norm_obs)
                        if exp_vec is not None and obs_vec is not None:
                            e_dy, e_dx = exp_vec
                            o_dy, o_dx = obs_vec
                            if (
                                (e_dy != 0 or e_dx != 0)
                                and _sign(e_dy) == _sign(o_dy)
                                and _sign(e_dx) == _sign(o_dx)
                                and (e_dy == 0 or (abs(o_dy) > 0 and abs(e_dy) >= abs(o_dy) and abs(e_dy) % abs(o_dy) == 0))
                                and (e_dx == 0 or (abs(o_dx) > 0 and abs(e_dx) >= abs(o_dx) and abs(e_dx) % abs(o_dx) == 0))
                            ):
                                return True
                    elif pred_low in ("dy", "dx", "row_delta", "col_delta", "delta_r", "delta_c"):
                        if isinstance(norm_exp, int) and isinstance(norm_obs, int) and norm_exp != 0 and norm_obs != 0:
                            if _sign(norm_exp) == _sign(norm_obs) and abs(norm_exp) >= abs(norm_obs) and abs(norm_exp) % abs(norm_obs) == 0:
                                return True
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


#: Every `invariant_type` the evaluator can judge by name. Types outside this
#: set are treated as unclassified and are dispatched by description keywords.
KNOWN_INVARIANT_TYPES = frozenset({
    "kinematics", "physics",
    "area_conservation", "topology",
    "spatial_position", "positional_pattern",
    "symmetry", "axial_symmetry_vertical", "axial_symmetry_horizontal",
    "socket_coverage", "alignment",
    "goal", "victory", "win_condition",
    "control", "selection", "modality",
    "palette", "color_role",
    "transformation", "state_flip", "rotation",
})


def declares_family(
    inv_type: str,
    type_names: tuple[str, ...],
    desc: str,
    desc_terms: tuple[str, ...],
) -> bool:
    """Decide whether an invariant belongs to a family.

    A recognised ``invariant_type`` is authoritative; description keywords are
    consulted only for types outside :data:`KNOWN_INVARIANT_TYPES`. Without that
    precedence an invariant whose prose happens to mention a foreign concept —
    a rotation rule that says pieces "rotate when toggled", an area rule that
    mentions "palette" — would be judged by the wrong family's rule.
    """
    if inv_type in type_names:
        return True
    if inv_type in KNOWN_INVARIANT_TYPES:
        return False
    return any(term in desc for term in desc_terms)


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
    if declares_family(inv_type, ("kinematics", "physics"), desc, ()):
        act_id = meta.get("action_id") or parse_referenced_action_id(desc)

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

        # The kinematics family applies, but this level produced no decisive
        # observation of it. That is silence, not a licence for a later family
        # to judge the same invariant by an unrelated rule.
        return Ternary.IRRELEVANT

    # 2. Area conservation invariants
    if declares_family(inv_type, ("area_conservation", "topology"), desc, ()) or (
        inv_type not in KNOWN_INVARIANT_TYPES and "area" in desc
    ):
        area_props = [
            p for p in props
            if p.predicate == "area" or p.family == "area_conservation"
        ]
        if not area_props:
            return Ternary.IRRELEVANT
        # An area prop exists on essentially every frame, so its mere presence is
        # silence about conservation. Only an invariant that actually asserts
        # conservation may be judged by the absence of destruction, and only a
        # non-zero observed mass can count as surviving matter.
        asserts_conservation = (
            inv_type in ("area_conservation", "topology")
            or "conserv" in desc
            or "preserv" in desc
        )
        if not asserts_conservation:
            return Ternary.IRRELEVANT
        has_destroyed = any(
            p.family == "object_identity" and p.predicate in ("destroyed", "vanished")
            for p in props
        )
        if has_destroyed:
            return Ternary.FALSE
        has_surviving_mass = any(
            isinstance(_normalize_value(p.value), int) and _normalize_value(p.value) > 0
            for p in area_props
            if p.value is not None
        )
        return Ternary.TRUE if has_surviving_mass else Ternary.IRRELEVANT

    # 3. Spatial position invariants
    p_pos = meta.get("target_position")
    if declares_family(inv_type, ("spatial_position", "positional_pattern"), desc, ()) or p_pos:
        if not p_pos:
            # The family applies but names no position to check against, so no
            # observation of it can be decisive.
            return Ternary.IRRELEVANT
        pos_props = [
            p for p in props
            if p.family == "spatial_position"
        ]
        for p in pos_props:
            if p.value == p_pos:
                return Ternary.TRUE
            if p.subject_id == meta.get("subject_id"):
                return Ternary.FALSE
        return Ternary.IRRELEVANT

    # 4. Symmetry and relational invariants
    if declares_family(
        inv_type,
        ("symmetry", "axial_symmetry_vertical", "axial_symmetry_horizontal", "socket_coverage", "alignment"),
        desc,
        (),
    ):
        rel_props = [
            p for p in props
            if p.family == "relation_existence"
        ]
        matching = any(
            inv_type in str(p.predicate).lower() or str(p.predicate).lower() in inv_type
            for p in rel_props if p.value
        )
        return Ternary.TRUE if matching else Ternary.IRRELEVANT

    # 5. Goal & Victory Invariants
    if declares_family(inv_type, ("goal", "victory", "win_condition"), desc, ("win", "goal")):
        term_props = [
            p for p in props
            if p.family == "terminal_metadata"
        ]
        for p in term_props:
            if p.predicate == "win" and _normalize_value(p.value) is True:
                return Ternary.TRUE
            if p.predicate == "game_over" and _normalize_value(p.value) is True:
                return Ternary.FALSE
        return Ternary.IRRELEVANT

    # 6. Control & Selection Invariants
    if declares_family(inv_type, ("control", "selection", "modality"), desc, ("toggle", "switch")):
        # Control is expressed through the action surface and affordance flags,
        # not through a dedicated proposition family.
        ctrl_props = [
            p for p in props
            if p.family in ("action_surface", "affordance_flag")
            or "toggle" in str(p.predicate)
        ]
        # Only an observation that positively asserts a control effect counts.
        # The existence of an action-surface snapshot says nothing about who is
        # controllable, so an unflagged entry is silence.
        decisive = [
            p for p in ctrl_props
            if _normalize_value(p.value) not in (None, False, 0, "")
        ]
        return Ternary.TRUE if decisive else Ternary.IRRELEVANT

    # 7. Palette & Color/Role Invariants
    if declares_family(inv_type, ("palette", "color_role"), desc, ("color", "palette")):
        declared_color = meta.get("target_color", meta.get("color"))
        if declared_color is None:
            # Every frame carries a color prop per object; without a declared
            # palette member there is nothing a color observation can confirm.
            return Ternary.IRRELEVANT
        color_props = [
            p for p in props
            if p.family == "attribute_delta" and p.predicate == "color"
        ]
        observed_colors = {
            _normalize_value(p.value) for p in color_props if p.value is not None
        }
        if not observed_colors:
            return Ternary.IRRELEVANT
        return Ternary.TRUE if _normalize_value(declared_color) in observed_colors else Ternary.IRRELEVANT

    # 8. Transformation & State Change Invariants
    if declares_family(inv_type, ("transformation", "state_flip", "rotation"), desc, ("rotate", "flip")):
        # A static identity/attribute snapshot is not evidence of a state flip.
        # Only an event-shaped proposition may confirm this family.
        event_predicates = (
            "transformed", "rotated", "flipped", "created", "destroyed",
            "vanished", "merged", "changed",
        )
        tf_props = [
            p for p in props
            if p.family == "object_identity" and p.predicate in event_predicates
        ]
        return Ternary.TRUE if tf_props else Ternary.IRRELEVANT

    return Ternary.IRRELEVANT
