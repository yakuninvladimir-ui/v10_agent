"""LayeredVerifier integrating Brusentsov ternary logic for empirical transition evaluation."""

from __future__ import annotations

import math
import re
from typing import Any

from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, Verdict, contradicts, implies_brusentsov
from v10_agent.config import V10Config
from v10_agent.observe import compute_grid_hash
from v10_agent.planning_set import PlanningSet
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep


class LayeredVerifier:
    """Absolute authority on empirical transition evaluation (Brusentsov FOLLOW / NULL / OMIT)."""

    def __init__(self, config: V10Config):
        self.config = config
        self.tracker = None
        if getattr(config, "enable_persistent_tracker", True):
            from v10_agent.tracker import PersistentObjectTracker
            self.tracker = PersistentObjectTracker(config)

    def reset(self) -> None:
        """Reset internal tracker state on environmental reset."""
        if self.tracker is not None:
            self.tracker.reset()

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
        tracker: Any | None = None,
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
                alias = planning_set.object_real_to_alias.get(b_obj.id) if planning_set and hasattr(planning_set, "object_real_to_alias") else None
                if alias and alias != b_obj.id:
                    props.append(AtomicProposition(family="object_identity", subject_id=alias, predicate="destroyed"))

        # Process matched pairs
        for b_id, best_match in matched_pairs:
            b_obj = before_snapshot.get_object(b_id)
            if not b_obj:
                continue

            alias = planning_set.object_real_to_alias.get(b_obj.id) if planning_set and hasattr(planning_set, "object_real_to_alias") else None
            subjects = [b_obj.id]
            if alias and alias != b_obj.id:
                subjects.append(alias)

            # Positional metric signs
            dr = best_match.centroid.row - b_obj.centroid.row
            dc = best_match.centroid.col - b_obj.centroid.col
            r_sign = 1 if dr >= move_thresh else (-1 if dr <= -move_thresh else 0)
            c_sign = 1 if dc >= move_thresh else (-1 if dc <= -move_thresh else 0)

            for s_id in subjects:
                # Object identity preserved
                props.append(AtomicProposition(family="object_identity", subject_id=s_id, predicate="preserved"))

                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="row_delta", value=r_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_r", value=r_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dy", value=r_sign))

                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="col_delta", value=c_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_c", value=c_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dx", value=c_sign))

                # Step-level and cumulative motion tuples
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="step_moved", value=(r_sign, c_sign)))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="moved", value=(r_sign, c_sign)))
                if r_sign == 0 and c_sign == 0:
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="unchanged", value=(0, 0)))

                # Attribute deltas
                props.append(AtomicProposition(family="attribute_delta", subject_id=s_id, predicate="color", value=best_match.color))
                props.append(AtomicProposition(family="attribute_delta", subject_id=s_id, predicate="area", value=best_match.area))

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

        # 5. Tracker propositions (cumulative_motion, shape_stability, occlusion)
        active_tracker = tracker if tracker is not None else self.tracker
        if active_tracker is not None:
            if not active_tracker.tracks and before_snapshot:
                active_tracker.update(before_snapshot, frame_index=0)
            if after_snapshot:
                active_tracker.update(after_snapshot, frame_index=active_tracker.frame_index + 1)
            for trk in active_tracker.get_tracked():
                # cumulative_motion
                props.append(
                    AtomicProposition(
                        family="cumulative_motion",
                        subject_id=trk.persistent_id,
                        predicate="moved_total_over_N_frames",
                        value=trk.cumulative_delta,
                    )
                )
                # shape_stability
                props.append(
                    AtomicProposition(
                        family="shape_stability",
                        subject_id=trk.persistent_id,
                        predicate="shape_stable" if trk.shape_stability_score > 0.7 else "shape_changed",
                        value=trk.shape_stability_score,
                    )
                )
                # occlusion
                if trk.occluded:
                    props.append(
                        AtomicProposition(
                            family="object_identity",
                            subject_id=trk.persistent_id,
                            predicate="occluded",
                            secondary_id=trk.occluded_by,
                            value=True,
                        )
                    )
                else:
                    props.append(
                        AtomicProposition(
                            family="object_identity",
                            subject_id=trk.persistent_id,
                            predicate="preserved",
                        )
                    )

                # Positional metric signs for tracked object
                vr_sign = 1 if trk.velocity[0] >= 0.5 else (-1 if trk.velocity[0] <= -0.5 else 0)
                vc_sign = 1 if trk.velocity[1] >= 0.5 else (-1 if trk.velocity[1] <= -0.5 else 0)
                props.append(AtomicProposition(family="metric_sign", subject_id=trk.persistent_id, predicate="row_delta", value=vr_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=trk.persistent_id, predicate="col_delta", value=vc_sign))

                # Attribute deltas for tracked object
                props.append(AtomicProposition(family="attribute_delta", subject_id=trk.persistent_id, predicate="color", value=trk.color))
                props.append(AtomicProposition(family="attribute_delta", subject_id=trk.persistent_id, predicate="area", value=trk.area))

        return PropositionSet.from_iterable(props)

    def evaluate_transition(
        self,
        step: GroundedStep,
        before_snapshot: ARGALiteSnapshot,
        after_obs: dict[str, Any],
        planning_set: PlanningSet,
        game_memory: Any = None,
        action_dict: dict[str, Any] | None = None,
        tracker: Any | None = None,
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
            elif hasattr(before_snapshot, "grid") and isinstance(before_snapshot.grid, (list, tuple)):
                before_snapshot = extract_arga_snapshot(before_snapshot.grid)
            else:
                before_snapshot = extract_arga_snapshot([])

        active_tracker = tracker if tracker is not None else self.tracker
        raw_grid = after_obs.get("grid", [])
        after_snapshot = extract_arga_snapshot(raw_grid)
        observed = self.extract_observed_propositions(before_snapshot, after_obs, planning_set, tracker=active_tracker)

        state = str(after_obs.get("state", "")).upper()
        after_levels = int(after_obs.get("levels_completed", 0) or 0)
        before_levels = int(getattr(before_snapshot, "levels_completed", 0) or 0)

        before_grid = getattr(before_snapshot, "grid", None)
        if before_grid is None and hasattr(planning_set, "grid"):
            before_grid = planning_set.grid
        zero_delta = (before_grid is not None and raw_grid and before_grid == raw_grid)
        is_effective = bool(not zero_delta or after_levels > before_levels)
        act_dict = action_dict or {}
        act_id = ""
        if action_dict:
            act_id = str(action_dict.get("action_id") or action_dict.get("id") or "").upper()
        if not act_id:
            m = re.search(r"action(\d+)", step.dsl_function, re.IGNORECASE)
            if m:
                act_id = f"ACTION{m.group(1)}"

        confirmed_eff = ""
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "")

        eff_lower = confirmed_eff.lower()
        is_modal_selection = (
            act_id in ("ACTION5", "ACTION6")
            or any(k in eff_lower for k in ("selection", "toggle", "indicator", "modal", "active entity"))
            or any(k in step.dsl_function.lower() for k in ("selection", "toggle", "switch", "modal"))
        )
        if game_memory is not None and hasattr(game_memory, "action_affordances"):
            for aff in getattr(game_memory, "action_affordances", []):
                if isinstance(aff, dict) and str(aff.get("action_id", "")).upper() == act_id:
                    cls_name = str(aff.get("effect_class", "")).upper()
                    if cls_name == "MODAL_SELECTION":
                        is_modal_selection = True

        is_confirmed_motion = (
            not is_modal_selection
            and any(k in eff_lower for k in ("dy=", "dx=", "moved", "moves", "displace"))
            and not any(k in eff_lower for k in ("blocked", "wall", "no_effect", "null", "selection", "toggle", "indicator"))
        )

        # 1. Tier 1: Terminal Victory or Level Completed (Follow)
        if state in {"WIN", "WON", "DONE", "VICTORY"} or after_levels > before_levels:
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Verdict.FOLLOW,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function}): Level/game advance confirmed (levels {before_levels} -> {after_levels}, state={state}). Brusentsov follow xy.",
                action_dict=act_dict,
                is_effective=True,
            )

        # 2. Tier 2: Terminal Failure Contradiction (Nullity)
        if state in {"GAME_OVER", "LOST", "FAILED"}:
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Verdict.NULL,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function}): Terminal failure contradiction: entered state {state}. Brusentsov nullity xy'_0.",
                action_dict=act_dict,
                is_effective=False,
            )

        # 3. Tier 3: Zero Grid Delta on a Confirmed Motion Action
        # Distinguish hard contradiction (NULL) from wall/boundary collision (OMIT) or modal locking (UNDECIDED).
        # Wall collision is a soft stop — the trajectory step failed to advance,
        # but no physical law was violated. The actor simply cannot pass through obstacles.
        if zero_delta and is_confirmed_motion and act_id not in ("ACTION5", "ACTION6", "RESET"):
            # Check if this could be a boundary/obstacle collision (soft stop)
            is_boundary_collision = False
            if game_memory is not None:
                inv_rules = getattr(game_memory, "invariant_rules", [])
                is_boundary_collision = any(
                    re.search(r"\b(?:wall|blocked|boundary|obstacle|barrier|collision|impassable)\b", r, re.IGNORECASE)
                    for r in inv_rules
                )
            if is_boundary_collision:
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=Verdict.OMIT,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function} / {act_id}): "
                        f"Motion action produced zero grid delta (wall/boundary collision). "
                        f"Soft stop — Brusentsov omit x'y' (trajectory step did not advance but no physical law broken)."
                    ),
                    action_dict=act_dict,
                    is_effective=False,
                )

            # Invariant of Orthogonal Modality: Check if a modal selector / switch is available
            allowed_acts = []
            if planning_set and hasattr(planning_set, "allowed_action_ids"):
                raw_acts = planning_set.allowed_action_ids
                if isinstance(raw_acts, (list, tuple, set)):
                    allowed_acts = [str(a).upper() for a in raw_acts]

            has_selection_mechanics = bool(game_memory and getattr(game_memory, "selection_mechanics", None))
            modal_switch_actions = [
                act for act in allowed_acts
                if act in ("ACTION5", "ACTION6")
                or (game_memory and any(kw in getattr(game_memory, "confirmed_action_effects", {}).get(act, "").lower() for kw in ("selection", "toggle", "indicator")))
            ]
            unconfirmed_actions = [
                act for act in allowed_acts
                if act not in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET", "ACTION7")
                and (not game_memory or act not in getattr(game_memory, "confirmed_action_effects", {}))
            ]
            if modal_switch_actions or has_selection_mechanics or unconfirmed_actions:
                modal_target = modal_switch_actions[0] if modal_switch_actions else (unconfirmed_actions[0] if unconfirmed_actions else "ACTION5")
                enable_undecided = getattr(self.config, "enable_undecided_verdict", True)
                u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=u_verdict,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function} / {act_id}): "
                        f"Motion action produced zero grid delta along kinematic axis, but modal selector is available ({modal_target}). "
                        f"Orthogonal modality: degree of freedom may be blocked in current mode. Epistemic signal seek evidence."
                    ),
                    evidence_hint=f"probe_{modal_target}",
                    action_dict=act_dict,
                    is_effective=False,
                )

            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Verdict.NULL,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=(
                    f"Step {step.step_id} ({step.dsl_function} / {act_id}): "
                    f"Motion action produced zero grid delta (obstacle or boundary collision). Brusentsov nullity xy'_0."
                ),
                action_dict=act_dict,
                is_effective=False,
            )

        enable_undecided = getattr(self.config, "enable_undecided_verdict", True)

        # 4. Tier 4: Epistemic Uncertainty & Multi-Frame Tracking Ambiguity (UNDECIDED)
        # When observations or track identities are ambiguous / low confidence, EXPECT matching is unreliable.

        # (a) Upstream low confidence grounded step or ambiguous matching status
        if getattr(step, "confidence", None) == "low" or getattr(step, "matching_status", None) == "ambiguous":
            u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=u_verdict,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function}): Low confidence grounded step. Epistemic signal seek evidence.",
                evidence_hint="probe_environment",
                action_dict=act_dict,
                is_effective=is_effective,
            )

        # (b) Ambiguous tracker matching (difference between top candidates < matching_ambiguity_threshold)
        # Restricted strictly to the main controllable / participating actors of the step
        # (b) Ambiguous tracker matching (difference between top candidates < matching_ambiguity_threshold)
        # Point 1: Participating filter is strictly mandatory (expected_propositions OR non-zero observed delta)
        # Point 2: Ambiguity between internal identical micro-dots of the same parent is ignored
        # Point 3: Raw last_ambiguity_score is never used as a sufficient condition
        participating_ids: set[str] = set()
        if active_tracker is not None:
            amb_thresh = getattr(self.config, "matching_ambiguity_threshold", 0.15)

            # 1. Point 1 & 3: Participating filter:
            # When expected_propositions are specified, evaluate ambiguity on tracks influencing EXPECT.
            # When expected_propositions are empty (e.g. repeat steps), evaluate on tracks with observed motion.
            expected_ids: set[str] = set()
            for p in step.expected_propositions:
                if p.subject_id:
                    expected_ids.add(p.subject_id)
                if p.secondary_id:
                    expected_ids.add(p.secondary_id)
            if getattr(step, "target_object_ids", None):
                expected_ids.update(step.target_object_ids)

            if expected_ids:
                participating_ids = expected_ids
            else:
                # Narrow participating actors strictly to objects with real physical motion
                # (rigid coordinate displacement + preserved area), explicitly excluding passive objects
                # whose visible form only changed due to occlusion/overlap (e.g. static target sockets).
                observed_motion_ids: set[str] = set()

                if before_snapshot and after_snapshot:
                    matched_after_ids: set[str] = set()
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
                            matched_after_ids.add(best_match.id)
                            # 1. Area preservation: an object that lost >25% area was occluded/overlapped
                            area_max = max(b_obj.area, best_match.area)
                            area_min = min(b_obj.area, best_match.area)
                            if area_max > 0 and (area_min / area_max) < 0.75:
                                continue
                            # 2. Rigid translation: bounding box boundaries must shift uniformly
                            dr_min = best_match.bbox.min_row - b_obj.bbox.min_row
                            dr_max = best_match.bbox.max_row - b_obj.bbox.max_row
                            dc_min = best_match.bbox.min_col - b_obj.bbox.min_col
                            dc_max = best_match.bbox.max_col - b_obj.bbox.max_col
                            if abs(dr_min - dr_max) <= 1 and abs(dc_min - dc_max) <= 1:
                                if abs(dr_min) >= 1 or abs(dc_min) >= 1:
                                    observed_motion_ids.add(b_obj.id)
                                    observed_motion_ids.add(best_match.id)

                # Fallback to observed propositions if snapshots comparison was unavailable
                if not observed_motion_ids:
                    for p in observed:
                        if p.family == "metric_sign" and p.predicate in (
                            "row_delta", "col_delta", "delta_r", "delta_c", "dy", "dx"
                        ):
                            if p.value != 0 and p.subject_id:
                                observed_motion_ids.add(p.subject_id)

                participating_ids = observed_motion_ids

            # Map participating IDs to track persistent IDs
            participating_track_ids: set[str] = set()
            for pid in participating_ids:
                if pid.startswith("trk_"):
                    participating_track_ids.add(pid)

            for snap_src in (getattr(planning_set, "objects", []), getattr(before_snapshot, "objects", []), getattr(after_snapshot, "objects", [])):
                for o in snap_src:
                    oid = getattr(o, "id", None)
                    pid = getattr(o, "persistent_id", None)
                    if oid in participating_ids and pid:
                        participating_track_ids.add(pid)

            for trk in active_tracker.get_tracked():
                if getattr(trk, "source_object_id", None) in participating_ids or trk.persistent_id in participating_ids:
                    participating_track_ids.add(trk.persistent_id)

            # 2. Point 2: Check ambiguity ONLY among participating tracks, ignoring internal sibling 1x1 micro-dots
            ambiguity_by_track = getattr(active_tracker, "ambiguity_by_track", {})
            relevant_ambiguities: list[tuple[float, str]] = []
            for trk_id in participating_track_ids:
                entry = ambiguity_by_track.get(trk_id)
                if entry is None:
                    continue
                diff = entry["diff"] if isinstance(entry, dict) else entry
                is_internal = entry.get("is_internal_sibling", False) if isinstance(entry, dict) else False

                # Sibling micro-dots of the same parent / cluster are ignored for verdict purposes
                if is_internal:
                    continue

                trk = active_tracker.get_track(trk_id)
                if trk and (trk.occluded or (hasattr(trk, "shape_stability_score") and trk.shape_stability_score < 0.7)):
                    continue

                # If object is destroyed/missing, it is an empirical event (OMIT), not a tracking ambiguity
                is_destroyed = any(
                    p.family == "object_identity"
                    and (p.subject_id == trk_id or (trk and p.subject_id == getattr(trk, "source_object_id", None)))
                    and p.predicate in ("destroyed", "missing", "vanished")
                    for p in observed
                )
                if is_destroyed:
                    continue

                # Also verify directly on track: if belongs to a composite parent
                if trk and getattr(trk, "source_object_id", None):
                    src_id = trk.source_object_id
                    has_parent = False
                    for snap_src in (getattr(planning_set, "objects", []), getattr(before_snapshot, "objects", [])):
                        for o in snap_src:
                            if getattr(o, "id", None) == src_id and getattr(o, "parent_id", None):
                                has_parent = True
                                break
                        if has_parent:
                            break
                    if has_parent:
                        continue

                relevant_ambiguities.append((diff, trk_id))

            # 3. Point 3: Do NOT use raw last_ambiguity_score as a sufficient condition.
            # Only evaluated participating tracks can trigger UNDECIDED.
            if relevant_ambiguities:
                min_amb_score, worst_track = min(relevant_ambiguities, key=lambda x: x[0])
                if min_amb_score < amb_thresh:
                    u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
                    return BrusentsovJudgment(
                        trajectory_id=step.step_id,
                        step_id=step.step_id,
                        verdict=u_verdict,
                        expected_propositions=step.expected_propositions,
                        observed_propositions=observed,
                        explanation=(
                            f"Step {step.step_id} ({step.dsl_function}): Ambiguous object matching detected for participating actor "
                            f"{worst_track} (score difference {min_amb_score:.3f} < {amb_thresh})."
                        ),
                        ambiguity_score=min_amb_score,
                        evidence_hint="probe_motion",
                        matching_candidates=[worst_track] if worst_track else [],
                        action_dict=act_dict,
                        is_effective=is_effective,
                    )

        # (c) Low tracking confidence on any participating object
        if active_tracker is not None:
            conf_thresh = getattr(self.config, "track_confidence_threshold", 0.6)
            for trk in active_tracker.get_tracked():
                if trk.occluded or trk.last_frame_id < getattr(active_tracker, "frame_index", 0) or (hasattr(trk, "shape_stability_score") and trk.shape_stability_score < 0.7):
                    continue
                is_destr = any(
                    p.family == "object_identity"
                    and (p.subject_id == trk.persistent_id or p.subject_id == getattr(trk, "source_object_id", None))
                    and p.predicate in ("destroyed", "missing", "vanished")
                    for p in observed
                )
                if is_destr:
                    continue

                is_part = (
                    trk.persistent_id in participating_ids
                    or getattr(trk, "source_object_id", None) in participating_ids
                )
                if not is_part and hasattr(planning_set, "objects"):
                    for po in planning_set.objects:
                        if getattr(po, "persistent_id", None) == trk.persistent_id and po.id in participating_ids:
                            is_part = True
                            break
                # Ignore low confidence on internal child components of composite objects
                if is_part and trk.confidence < conf_thresh:
                    if getattr(trk, "source_object_id", None):
                        src_id = trk.source_object_id
                        has_parent = any(
                            getattr(o, "parent_id", None)
                            for o in getattr(planning_set, "objects", [])
                            if getattr(o, "id", None) == src_id
                        )
                        if has_parent:
                            continue
                    u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
                    return BrusentsovJudgment(
                        trajectory_id=step.step_id,
                        step_id=step.step_id,
                        verdict=u_verdict,
                        expected_propositions=step.expected_propositions,
                        observed_propositions=observed,
                        explanation=(
                            f"Step {step.step_id} ({step.dsl_function}): Participating track {trk.persistent_id} "
                            f"confidence {trk.confidence:.2f} < threshold {conf_thresh}."
                        ),
                        evidence_hint="probe_tracking",
                        track_confidence_min=trk.confidence,
                        action_dict=act_dict,
                        is_effective=is_effective,
                    )

        # 5. Tier 5: Explicit EXPECT Contradiction Check (Physical contradiction: implies_brusentsov == FALSE)
        prop_verdict = None
        if len(step.expected_propositions) > 0:
            prop_verdict = implies_brusentsov(step.expected_propositions, observed)
            if prop_verdict == Ternary.FALSE:
                # Epistemic reservation: if action is unconfirmed with zero grid delta, or metric delta is below
                # the measurement noise threshold (< min_reliable_delta), observation cannot reliably refute expectations.
                # Such transitions are epistemically uncertain and must be deferred to Tier 7.
                is_unconfirmed_zero = (
                    zero_delta
                    and act_id
                    and act_id.startswith("ACTION")
                    and not is_confirmed_motion
                    and act_id != "RESET"
                )
                min_reliable = getattr(self.config, "min_reliable_delta", 0.8)
                is_sub_threshold = False
                if not zero_delta and state not in {"WIN", "WON", "DONE", "VICTORY", "GAME_OVER", "LOST", "FAILED"}:
                    max_disp = 0.0
                    has_displacement_data = False
                    if active_tracker is not None:
                        for trk in active_tracker.get_tracked():
                            speed = math.hypot(trk.velocity[0], trk.velocity[1])
                            if speed > max_disp:
                                max_disp = speed
                            has_displacement_data = True
                    if before_snapshot and after_snapshot:
                        for b_obj in before_snapshot.objects:
                            for a_obj in after_snapshot.objects:
                                if b_obj.color == a_obj.color:
                                    d = math.hypot(a_obj.centroid.row - b_obj.centroid.row, a_obj.centroid.col - b_obj.centroid.col)
                                    if d < 5.0:
                                        if d > max_disp:
                                            max_disp = d
                                        has_displacement_data = True
                    if has_displacement_data and 0.0 < max_disp < min_reliable:
                        is_sub_threshold = True

                # Check if there are non-kinematic contradictions (e.g. color, identity, state)
                has_non_kinematic_contradiction = False
                for p_exp in step.expected_propositions:
                    is_kinematic = (
                        p_exp.family == "metric_sign"
                        or p_exp.predicate in ("moved", "step_moved", "delta_r", "delta_c", "row_delta", "col_delta", "dy", "dx")
                    )
                    if not is_kinematic:
                        for p_obs in observed:
                            if contradicts(p_exp, p_obs):
                                has_non_kinematic_contradiction = True
                                break
                    if has_non_kinematic_contradiction:
                        break

                if has_non_kinematic_contradiction or (not is_unconfirmed_zero and not is_sub_threshold):
                    return BrusentsovJudgment(
                        trajectory_id=step.step_id,
                        step_id=step.step_id,
                        verdict=Verdict.NULL,
                        expected_propositions=step.expected_propositions,
                        observed_propositions=observed,
                        explanation=(
                            f"Step {step.step_id} ({step.dsl_function}): Step proposition contradiction: "
                            f"asserted expected propositions physically refuted by observation. Brusentsov nullity xy'_0."
                        ),
                        action_dict=act_dict,
                        is_effective=is_effective,
                    )

        # 6. Tier 6: Explicit EXPECT Necessary Containment (Follow: implies_brusentsov == TRUE)
        if prop_verdict == Ternary.TRUE:
            if is_modal_selection:
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=Verdict.FOLLOW,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function} / {act_id}): Modal selection verified with "
                        f"confirmed predicate expectations. Brusentsov follow xy."
                    ),
                    action_dict=act_dict,
                    is_effective=is_effective,
                )
            # Guard against vacuous truth for mode changes:
            # Absence of effect upon mode change (ACTION5, ACTION6, or selection/toggle)
            # must evaluate to OMIT or UNDECIDED, strictly preventing vacuous truth.
            is_modal_act = (
                act_id in ("ACTION5", "ACTION6")
                or "selection" in step.dsl_function.lower()
                or "toggle" in step.dsl_function.lower()
            )
            if zero_delta and is_modal_act:
                u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=u_verdict,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function} / {act_id}): Absence of effect upon mode change "
                        f"with zero grid delta. Brusentsov non-vacuity: {u_verdict.value} (no vacuous confirmation)."
                    ),
                    evidence_hint=f"probe_{act_id}" if act_id else "probe_motion",
                    action_dict=act_dict,
                    is_effective=False,
                )
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Verdict.FOLLOW,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=(
                    f"Step {step.step_id} ({step.dsl_function}): Expected consequence necessarily contained in observation. "
                    f"Brusentsov follow xy."
                ),
                action_dict=act_dict,
                is_effective=True,
            )

        # 7. Tier 7: Unconfirmed Zero Delta or Low Metric Delta (< min_reliable_delta)
        # 7a. Zero grid delta on an unconfirmed action carrying non-empty EXPECT
        if (
            zero_delta
            and act_id
            and act_id.startswith("ACTION")
            and not is_confirmed_motion
            and len(step.expected_propositions) > 0
            and act_id != "RESET"
        ):
            u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=u_verdict,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=f"Step {step.step_id} ({step.dsl_function} / {act_id}): Zero grid delta on non-confirmed action with expected effects. Epistemic signal seek evidence.",
                evidence_hint=f"probe_{act_id}" if act_id else "probe_motion",
                action_dict=act_dict,
                is_effective=False,
            )

        # 7b. Low metric delta: detected changes exist, but all displacements < min_reliable_delta (0.8 px)
        min_reliable = getattr(self.config, "min_reliable_delta", 0.8)
        if not zero_delta and state not in {"WIN", "WON", "DONE", "VICTORY", "GAME_OVER", "LOST", "FAILED"}:
            max_disp = 0.0
            has_displacement_data = False

            if active_tracker is not None:
                for trk in active_tracker.get_tracked():
                    speed = math.hypot(trk.velocity[0], trk.velocity[1])
                    if speed > max_disp:
                        max_disp = speed
                    has_displacement_data = True

            if before_snapshot and after_snapshot:
                for b_obj in before_snapshot.objects:
                    for a_obj in after_snapshot.objects:
                        if b_obj.color == a_obj.color:
                            d = math.hypot(a_obj.centroid.row - b_obj.centroid.row, a_obj.centroid.col - b_obj.centroid.col)
                            if d < 5.0:
                                if d > max_disp:
                                    max_disp = d
                                has_displacement_data = True

            if has_displacement_data and 0.0 < max_disp < min_reliable and len(step.expected_propositions) > 0:
                u_verdict = Verdict.UNDECIDED if enable_undecided else Verdict.OMIT
                return BrusentsovJudgment(
                    trajectory_id=step.step_id,
                    step_id=step.step_id,
                    verdict=u_verdict,
                    expected_propositions=step.expected_propositions,
                    observed_propositions=observed,
                    explanation=(
                        f"Step {step.step_id} ({step.dsl_function}): Low metric delta detected "
                        f"(max displacement {max_disp:.2f} < {min_reliable}). Epistemic signal seek evidence."
                    ),
                    evidence_hint="probe_motion",
                    action_dict=act_dict,
                    is_effective=is_effective,
                )

        # 8. Tier 8: Positive certificate from GameMemory (verified by non-zero delta)
        # Guard against material implication paradox: empty/unverified expected propositions
        # with non-zero delta must NOT produce FOLLOW (Brusentsov: x'y → OMIT, not FOLLOW).
        if act_id and confirmed_eff:
            if not zero_delta:
                if len(step.expected_propositions) > 0 and prop_verdict == Ternary.TRUE:
                    return BrusentsovJudgment(
                        trajectory_id=step.step_id,
                        step_id=step.step_id,
                        verdict=Verdict.FOLLOW,
                        expected_propositions=step.expected_propositions,
                        observed_propositions=observed,
                        explanation=f"Step {step.step_id} ({step.dsl_function} / {act_id}): Certified action effect verified against GameMemory with confirmed propositions. Brusentsov follow xy.",
                        action_dict=act_dict,
                        is_effective=True,
                    )
                else:
                    return BrusentsovJudgment(
                        trajectory_id=step.step_id,
                        step_id=step.step_id,
                        verdict=Verdict.OMIT,
                        expected_propositions=step.expected_propositions,
                        observed_propositions=observed,
                        explanation=(
                            f"Step {step.step_id} ({step.dsl_function} / {act_id}): Certified action produced non-zero delta "
                            f"but expected propositions {'empty' if len(step.expected_propositions) == 0 else 'unverified'}. "
                            f"Brusentsov omit x'y' (no vacuous confirmation)."
                        ),
                        action_dict=act_dict,
                        is_effective=is_effective,
                    )

        # Default Fallthrough (ISO-10)
        # If expected propositions were inessential or omitted without physical contradiction -> OMIT
        if len(step.expected_propositions) > 0 and prop_verdict == Ternary.IRRELEVANT:
            return BrusentsovJudgment(
                trajectory_id=step.step_id,
                step_id=step.step_id,
                verdict=Verdict.OMIT,
                expected_propositions=step.expected_propositions,
                observed_propositions=observed,
                explanation=(
                    f"Step {step.step_id} ({step.dsl_function}): Expected consequence was inessential / omitted. "
                    f"Brusentsov omit x'y'."
                ),
                action_dict=act_dict,
                is_effective=is_effective,
            )

        # Tier 8: Default Fallthrough (ISO-10) - strictly return Verdict.OMIT for benign passive transitions
        return BrusentsovJudgment(
            trajectory_id=step.step_id,
            step_id=step.step_id,
            verdict=Verdict.OMIT,
            expected_propositions=step.expected_propositions,
            observed_propositions=observed,
            explanation=(
                f"Step {step.step_id} ({step.dsl_function}): Executed without necessary containment proof. "
                f"Tier 8 default fallback / benign passive transition. Brusentsov omit x'y'."
            ),
            action_dict=act_dict,
            is_effective=is_effective,
        )
