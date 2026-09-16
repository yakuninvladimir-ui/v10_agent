"""LayeredVerifier integrating Brusentsov ternary logic for empirical transition evaluation."""

from __future__ import annotations

import math
from typing import Any

from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, implies_brusentsov
from v10_agent.config import V10Config
from v10_agent.observe import compute_grid_hash
from v10_agent.planning_set import PlanningSet
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep


class LayeredVerifier:
    """Absolute authority on empirical transition evaluation (Brusentsov FOLLOW / NULL / OMIT)."""

    def __init__(self, config: V10Config):
        self.config = config

    def _find_matching_object(self, obj: Any, snapshot: ARGALiteSnapshot) -> Any | None:
        """Find matching object in snapshot by color and spatial proximity with topological fallback."""
        best = None
        best_d = float("inf")
        # Pass 1: exact color match
        for o in snapshot.objects:
            if o.color == obj.color:
                d = math.hypot(o.centroid.row - obj.centroid.row, o.centroid.col - obj.centroid.col)
                if d < best_d:
                    best_d = d
                    best = o
        if best is not None and best_d <= max(3.5, obj.bbox.height + 2, obj.bbox.width + 2):
            return best

        # Pass 2: topological fallback (in-place color transition or small displacement)
        fallback_best = None
        fallback_d = float("inf")
        perc = getattr(self.config, "perception", None)
        area_tol = getattr(perc, "object_match_area_tolerance", 0.15) if perc else 0.15
        for o in snapshot.objects:
            d = math.hypot(o.centroid.row - obj.centroid.row, o.centroid.col - obj.centroid.col)
            is_close = d <= max(2.5, min(obj.bbox.height, obj.bbox.width) + 1)
            is_shape_similar = abs(o.area - obj.area) <= max(2, int(round(obj.area * area_tol))) and o.color == obj.color
            if (is_close or is_shape_similar) and d <= max(3.0, obj.bbox.height + 1, obj.bbox.width + 1):
                if d < fallback_d:
                    fallback_d = d
                    fallback_best = o
        return fallback_best

    def extract_observed_propositions(
        self,
        before_snapshot: ARGALiteSnapshot,
        after_obs: dict[str, Any],
        planning_set: PlanningSet,
    ) -> PropositionSet:
        """Derive observed atomic propositions from before-snapshot and after-observation."""
        raw_grid = after_obs.get("grid", [])
        after_snapshot = extract_arga_snapshot(raw_grid)

        props: list[AtomicProposition] = []

        # 1. Match objects between before and after snapshots (two-pass with topological fallback)
        matched_pairs: list[tuple[str, Any]] = []
        matched_after_ids: set[str] = set()

        # Pass 1: Exact color matching with proximity
        unmatched_before = []
        for b_obj in before_snapshot.objects:
            best_match = None
            best_dist = float("inf")
            for a_obj in after_snapshot.objects:
                if a_obj.id in matched_after_ids:
                    continue
                if a_obj.color == b_obj.color:
                    d = math.hypot(b_obj.centroid.row - a_obj.centroid.row, b_obj.centroid.col - a_obj.centroid.col)
                    if d < best_dist:
                        best_dist = d
                        best_match = a_obj

            if best_match is not None and best_dist <= max(2.5, b_obj.bbox.height, b_obj.bbox.width):
                matched_pairs.append((b_obj.id, best_match))
                matched_after_ids.add(best_match.id)
            else:
                unmatched_before.append(b_obj)

        perc = getattr(self.config, "perception", None)
        area_tol = getattr(perc, "object_match_area_tolerance", 0.15) if perc else 0.15
        move_thresh = getattr(perc, "movement_detection_threshold", 1.0) if perc else 1.0

        # Pass 2: Topological matching fallback for objects undergoing in-place color/area transition
        for b_obj in unmatched_before:
            best_match = None
            best_dist = float("inf")
            for a_obj in after_snapshot.objects:
                if a_obj.id in matched_after_ids:
                    continue
                d = math.hypot(b_obj.centroid.row - a_obj.centroid.row, b_obj.centroid.col - a_obj.centroid.col)
                is_close = d <= max(2.5, min(b_obj.bbox.height, b_obj.bbox.width) + 1)
                is_shape_similar = abs(a_obj.area - b_obj.area) <= max(2, int(round(b_obj.area * area_tol))) and a_obj.color == b_obj.color
                if (is_close or is_shape_similar) and d <= max(3.0, b_obj.bbox.height + 1, b_obj.bbox.width + 1):
                    if d < best_dist:
                        best_dist = d
                        best_match = a_obj

            if best_match is not None:
                matched_pairs.append((b_obj.id, best_match))
                matched_after_ids.add(best_match.id)
            else:
                # Object destroyed or disappeared
                props.append(AtomicProposition(family="object_identity", subject_id=b_obj.id, predicate="destroyed"))

        # Process matched pairs
        for b_id, best_match in matched_pairs:
            b_obj = before_snapshot.get_object(b_id)
            if not b_obj:
                continue
            # Object identity preserved
            props.append(AtomicProposition(family="object_identity", subject_id=b_obj.id, predicate="preserved"))

            # Positional metric signs
            dr = best_match.centroid.row - b_obj.centroid.row
            dc = best_match.centroid.col - b_obj.centroid.col
            r_sign = 1 if dr >= move_thresh else (-1 if dr <= -move_thresh else 0)
            c_sign = 1 if dc >= move_thresh else (-1 if dc <= -move_thresh else 0)

            props.append(AtomicProposition(family="metric_sign", subject_id=b_obj.id, predicate="row_delta", value=r_sign))
            props.append(AtomicProposition(family="metric_sign", subject_id=b_obj.id, predicate="col_delta", value=c_sign))

            # Attribute deltas
            props.append(AtomicProposition(family="attribute_delta", subject_id=b_obj.id, predicate="color", value=best_match.color))
            props.append(AtomicProposition(family="attribute_delta", subject_id=b_obj.id, predicate="area", value=best_match.area))

        # 2. Pairwise distance metric signs
        for i in range(len(matched_pairs)):
            id_a, a_after = matched_pairs[i]
            a_before = before_snapshot.get_object(id_a)
            if not a_before:
                continue
            for j in range(i + 1, len(matched_pairs)):
                id_b, b_after = matched_pairs[j]
                b_before = before_snapshot.get_object(id_b)
                if not b_before:
                    continue

                dist_before = math.hypot(a_before.centroid.row - b_before.centroid.row, a_before.centroid.col - b_before.centroid.col)
                dist_after = math.hypot(a_after.centroid.row - b_after.centroid.row, a_after.centroid.col - b_after.centroid.col)
                dd = dist_after - dist_before
                d_sign = 1 if dd >= 1.0 else (-1 if dd <= -1.0 else 0)

                props.append(
                    AtomicProposition(
                        family="metric_sign",
                        subject_id=id_a,
                        predicate="distance",
                        value=d_sign,
                        secondary_id=id_b,
                    )
                )

        # 3. Spatial relations in after-state
        for rel in after_snapshot.relations:
            props.append(
                AtomicProposition(
                    family="relation_existence",
                    subject_id=rel.subject_id,
                    predicate=rel.relation_type,
                    value=True,
                    secondary_id=rel.target_id,
                )
            )

        # 4. Terminal Metadata
        state = str(after_obs.get("state", "")).upper()
        if state in {"WIN", "WON", "DONE", "VICTORY"}:
            props.append(AtomicProposition(family="terminal_metadata", subject_id="game", predicate="win", value=True))
        elif state in {"GAME_OVER", "LOST", "FAILED"}:
            props.append(AtomicProposition(family="terminal_metadata", subject_id="game", predicate="game_over", value=True))

        levels_completed = after_obs.get("levels_completed", 0)
        props.append(AtomicProposition(family="terminal_metadata", subject_id="game", predicate="levels_completed", value=int(levels_completed or 0)))

        return PropositionSet.from_iterable(props)

    def evaluate_transition(
        self,
        step: GroundedStep,
        before_snapshot: ARGALiteSnapshot,
        after_obs: dict[str, Any],
        planning_set: PlanningSet,
        game_memory: Any = None,
        action_dict: dict[str, Any] | None = None,
    ) -> BrusentsovJudgment:
        """Evaluate empirical post-step transition using Brusentsov 3-valued logic of necessary consequence.
        
        Evaluates:
        - xy: TRUE (Follow) - consequence followed necessarily from asserted antecedent
        - xy'_0: FALSE (Nullity) - consequence denied / contradiction (wall collision, game over, divergence)
        - x'y': IRRELEVANT (Omit) - inessential / auxiliary outcome
        """
        if not isinstance(before_snapshot, ARGALiteSnapshot):
            if isinstance(before_snapshot, (list, tuple)):
                before_snapshot = extract_arga_snapshot(before_snapshot)
            else:
                before_snapshot = extract_arga_snapshot([])

        raw_grid = after_obs.get("grid", [])
        after_snapshot = extract_arga_snapshot(raw_grid)
        observed = self.extract_observed_propositions(before_snapshot, after_obs, planning_set)

        state = str(after_obs.get("state", "")).upper()
        after_levels = int(after_obs.get("levels_completed", 0) or 0)
        before_levels = int(getattr(before_snapshot, "levels_completed", 0) or 0)

        # 1. Terminal Victory or Level Completed (Follow: xy -> TRUE)
        if state in {"WIN", "WON", "DONE", "VICTORY"} or after_levels > before_levels:
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Ternary.TRUE,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function}): Level/game advance confirmed (levels {before_levels} -> {after_levels}, state={state}). Brusentsov follow xy.",
            )

        # 2. Terminal Failure Contradiction (Nullity: xy'_0 -> FALSE)
        if state in {"GAME_OVER", "LOST", "FAILED"}:
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Ternary.FALSE,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function}): Terminal failure contradiction: entered state {state}. Brusentsov nullity xy'_0.",
            )

        # 2b. Explicit Expected Propositions Contradiction Check (if explicitly asserted by step)
        if len(step.expected_propositions) > 0:
            prop_verdict = implies_brusentsov(step.expected_propositions, observed)
            if prop_verdict == Ternary.FALSE:
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=Ternary.FALSE,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function}): Step proposition contradiction: "
                        f"asserted expected propositions physically refuted by observation. Brusentsov nullity xy'_0."
                    ),
                )
            elif prop_verdict == Ternary.IRRELEVANT:
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=Ternary.IRRELEVANT,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function}): Expected consequence was inessential / omitted. "
                        f"Brusentsov omit x'y'."
                    ),
                )

        # 2c. Motion Step Zero-Delta Contradiction Check (Brusentsov nullity: xy'_0 -> FALSE)
        # If a directional motion action produces zero change in the grid (wall collision,
        # boundary barrier, or inactive entity), the kinematic consequence failed immediately.
        before_grid = getattr(before_snapshot, "grid", None)
        if before_grid is None and hasattr(planning_set, "grid"):
            before_grid = planning_set.grid
        if before_grid is not None and raw_grid and before_grid == raw_grid:
            act_id = ""
            if action_dict:
                act_id = str(action_dict.get("action_id") or action_dict.get("id") or "").upper()
            if not act_id:
                import re
                m = re.search(r"action(\d+)", step.dsl_function, re.IGNORECASE)
                if m:
                    act_id = f"ACTION{m.group(1)}"
            confirmed_eff = ""
            if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
                confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "")

            is_confirmed_directional_motion = (
                "moved" in confirmed_eff
                and any(d in confirmed_eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))
            )
            # A step where confirmed directional motion produces zero grid delta
            # is a Brusentsov nullity (wall collision, boundary barrier, or inactive/immobile actor).
            if is_confirmed_directional_motion and act_id not in ("ACTION5", "ACTION6", "RESET"):
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=Ternary.FALSE,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function} / {act_id}): "
                        f"Motion action produced zero grid delta (данное действие не валидно при текущих координатах объекта / local obstacle). Brusentsov nullity xy'_0."
                    ),
                )

        # 3. Approved Full Trajectory Step Execution (No step-by-step distance or displacement checks)
        # As long as the trajectory was validated against environment constraints and hypothesis upfront,
        # each step is executed unconditionally toward the goal.
        return BrusentsovJudgment(
            trajectory_id=step.step_id,
            step_id=step.step_id,
            verdict=Ternary.TRUE,
            expected_propositions=step.expected_propositions,
            observed_propositions=observed,
            explanation=(
                f"Step {step.step_id} ({step.dsl_function}): Executed as part of approved full trajectory "
                f"without step-by-step metric interruptions. Brusentsov follow xy."
            ),
        )
