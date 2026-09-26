"""LayeredVerifier integrating Brusentsov ternary logic for empirical transition evaluation."""

from __future__ import annotations

import math
from typing import Any

from v10_agent.action_semantics import (
    action_vector,
    is_non_vector_move,
    is_observable_move,
    is_vector_action,
    parse_referenced_action_id,
)
from v10_agent.arga_lite import ARGALiteSnapshot, detect_background_color, extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, Verdict, contradicts, implies_brusentsov
from v10_agent.config import V10Config
from v10_agent.observe import compute_grid_hash
from v10_agent.planning_set import PlanningSet
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep


def _cell_value(grid: Any, row: int, col: int) -> int | None:
    """Read a single grid cell, returning None when the coordinate is out of range.

    The guards matter: a planning set may be a test double whose ``grid`` is not a
    nested sequence at all, and a comparison against a non-numeric sentinel would
    silently decide the verdict.
    """
    if not isinstance(grid, (list, tuple)) or not (0 <= row < len(grid)):
        return None
    line = grid[row]
    if not isinstance(line, (list, tuple)) or not (0 <= col < len(line)):
        return None
    value = line[col]
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def is_bbox_pinned_to_field_edge(
    bbox: Any,
    grid_dims: Any,
    action_id: str,
    crop_offset: int = 0,
) -> bool:
    """True when ``bbox`` already occupies the outermost reachable cell on its axis.

    The outer border pixels are removed by the observation crop, so the first
    reachable cell sits ``crop_offset`` pixels in from each edge. Taking the offset
    as an argument instead of assuming one crop pixel keeps the predicate exact for
    any crop width, and the extent is read from the caller rather than assumed.
    """
    if bbox is None:
        return False
    if not isinstance(grid_dims, (tuple, list)) or len(grid_dims) < 2:
        return False
    try:
        grid_h, grid_w = int(grid_dims[0]), int(grid_dims[1])
    except (TypeError, ValueError):
        return False
    if grid_h <= 0 or grid_w <= 0:
        return False
    try:
        offset = int(crop_offset or 0)
        action = str(action_id).upper()
        if action == "ACTION1":  # UP
            return int(bbox.min_row) <= offset
        if action == "ACTION2":  # DOWN
            return int(bbox.max_row) >= grid_h - 1 - offset
        if action == "ACTION3":  # LEFT
            return int(bbox.min_col) <= offset
        if action == "ACTION4":  # RIGHT
            return int(bbox.max_col) >= grid_w - 1 - offset
    except (TypeError, ValueError):
        return False
    return False


def is_bbox_absorbed_by_contact(bbox: Any, action_id: str, grid: Any) -> bool:
    """True when a solid, non-background entity fills the cells ahead of ``bbox``.

    This is the second soft-stop mechanism: the subject is not at the field edge,
    yet it cannot move because something physical occupies the destination. Only
    the cells immediately beyond the bounding box are inspected, spanning the full
    cross-axis extent of the subject, so a subject resting on a floor - or pressed
    against any other solid body - is recognised without knowing what that body is.

    The check is deliberately conservative: every probed cell must be occupied.
    A single free cell anywhere along the contact face means the motion was
    geometrically possible, so the zero delta remains a genuine contradiction.
    """
    if bbox is None:
        return False
    vector = action_vector(action_id)
    if vector is None:
        return False
    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        return False
    if not isinstance(grid[0], (list, tuple)) or len(grid[0]) == 0:
        return False
    try:
        dy, dx = int(vector[0]), int(vector[1])
    except (TypeError, ValueError):
        return False

    background = detect_background_color(grid)

    # Shift the whole bounding box by one step; the resulting face is exactly the
    # set of cells the subject would occupy if the motion had succeeded.
    try:
        row_lo = int(bbox.min_row) + dy
        row_hi = int(bbox.max_row) + dy
        col_lo = int(bbox.min_col) + dx
        col_hi = int(bbox.max_col) + dx
    except (TypeError, ValueError):
        return False

    probed = 0
    for row in range(row_lo, row_hi + 1):
        for col in range(col_lo, col_hi + 1):
            value = _cell_value(grid, row, col)
            if value is None:
                # Off-field destinations are handled by the field-edge predicate.
                continue
            probed += 1
            if value == background:
                # A free cell exists along the contact face, so the motion was
                # physically available and the zero delta is a real contradiction.
                return False
    return probed > 0


def is_blocked_by_boundary(
    planning_set: Any,
    action_id: str,
    subject_id: str | None = None,
    before_grid: Any = None,
) -> bool:
    """Detect a soft stop: the subject cannot advance one cell along ``action_id``.

    Two independent physical causes produce an identical zero grid delta, and both
    must be reported as an omit rather than a nullity:

    * **Field edge** - the subject already touches the outermost reachable cell, so
      the motion has nowhere to go. This is resolved purely from geometry.
    * **Solid contact** - a non-background entity occupies the cell immediately
      beyond the subject, so the motion was absorbed. This needs the observation
      that preceded the step, because the extent of the subject alone cannot reveal
      what lies ahead of it.

    Both arms are expressed through the canonical action vector and the perceived
    background colour, so nothing here depends on a particular field size, on the
    position of the subject inside it, or on which colours a game happens to use.
    """
    if not subject_id or not planning_set:
        return False
    if not hasattr(planning_set, "get_object"):
        return False
    obj = planning_set.get_object(subject_id)
    bbox = getattr(obj, "bbox", None) if obj is not None else None
    if bbox is None:
        return False
    if is_bbox_pinned_to_field_edge(
        bbox,
        getattr(planning_set, "grid_dims", None),
        action_id,
        int(getattr(planning_set, "crop_offset", 0) or 0),
    ):
        return True
    grid = before_grid
    if not isinstance(grid, (list, tuple)):
        grid = getattr(planning_set, "grid", None)
    return is_bbox_absorbed_by_contact(bbox, action_id, grid)


def _safe_get_grid_dims(planning_set: Any, snapshot: Any | None = None) -> tuple[int, int]:
    grid_dims = getattr(planning_set, "grid_dims", None)
    if isinstance(grid_dims, (tuple, list)) and len(grid_dims) == 2:
        try:
            gh, gw = int(grid_dims[0]), int(grid_dims[1])
            if gh > 0 and gw > 0:
                return gh, gw
        except (ValueError, TypeError):
            pass
    if snapshot is not None:
        snap_dims = getattr(snapshot, "grid_dims", None)
        if isinstance(snap_dims, (tuple, list)) and len(snap_dims) == 2:
            try:
                gh, gw = int(snap_dims[0]), int(snap_dims[1])
                if gh > 0 and gw > 0:
                    return gh, gw
            except (ValueError, TypeError):
                pass
        grid = getattr(snapshot, "grid", None)
        if isinstance(grid, (tuple, list)) and len(grid) > 0 and isinstance(grid[0], (tuple, list)):
            return len(grid), len(grid[0])
    # Neither source published an extent. Report "unknown" rather than inventing
    # a field size: callers scale this value into a proximity radius, so zero
    # degrades to the tightest search window instead of a fabricated one.
    return (0, 0)


class SpatialIndex:
    """Simple grid-based spatial bucket index for accelerating nearest-neighbor queries."""
    def __init__(self, cell_size: int = 4):
        self.cell_size = max(1, cell_size)
        self.buckets: dict[tuple[int, int], list[Any]] = {}

    def build(self, objects: Sequence[Any]) -> None:
        self.buckets.clear()
        for obj in objects:
            c_row = getattr(obj.centroid, "row", 0) if hasattr(obj, "centroid") else 0
            c_col = getattr(obj.centroid, "col", 0) if hasattr(obj, "centroid") else 0
            cell = (int(c_row) // self.cell_size, int(c_col) // self.cell_size)
            self.buckets.setdefault(cell, []).append(obj)

    def find_nearby(self, row: float, col: float, radius_cells: int = 2) -> list[Any]:
        candidates: list[Any] = []
        center_cell = (int(row) // self.cell_size, int(col) // self.cell_size)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                cell = (center_cell[0] + dr, center_cell[1] + dc)
                if cell in self.buckets:
                    candidates.extend(self.buckets[cell])
        return candidates


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
        grid_h, grid_w = _safe_get_grid_dims(None, snapshot)
        adaptive_proximity = max(grid_h, grid_w) * 0.08
        # Pass 1: exact color match
        for o in snapshot.objects:
            if o.color == obj.color:
                d = math.hypot(o.centroid.row - obj.centroid.row, o.centroid.col - obj.centroid.col)
                if d < best_d:
                    best_d = d
                    best = o
        if best is not None and best_d <= max(adaptive_proximity, obj.bbox.height + 2, obj.bbox.width + 2):
            return best

        # Pass 2: topological fallback (in-place color transition or small displacement)
        fallback_best = None
        fallback_d = float("inf")
        perc = self.config.resolve("perception")
        area_tol = getattr(perc, "object_match_area_tolerance", 0.15) if perc else 0.15
        for o in snapshot.objects:
            d = math.hypot(o.centroid.row - obj.centroid.row, o.centroid.col - obj.centroid.col)
            is_close = d <= max(adaptive_proximity, min(obj.bbox.height, obj.bbox.width) + 1)
            is_shape_similar = abs(o.area - obj.area) <= max(2, int(round(obj.area * area_tol))) and o.color == obj.color
            if (is_close or is_shape_similar) and d <= max(adaptive_proximity + 0.5, obj.bbox.height + 1, obj.bbox.width + 1):
                if d < fallback_d:
                    fallback_d = d
                    fallback_best = o
        return fallback_best

    def match_objects_between_snapshots(
        self,
        before_snapshot: ARGALiteSnapshot,
        after_snapshot: ARGALiteSnapshot,
        planning_set: Any = None,
    ) -> tuple[list[tuple[str, Any]], list[Any]]:
        """Match objects between before and after snapshots using parent-coherent two-pass matching."""
        matched_pairs: list[tuple[str, Any]] = []
        matched_after_ids: set[str] = set()

        grid_h, grid_w = _safe_get_grid_dims(planning_set, before_snapshot)
        adaptive_proximity = max(grid_h, grid_w) * 0.08

        perc = self.config.resolve("perception")
        area_tol = getattr(perc, "object_match_area_tolerance", 0.15) if perc else 0.15

        spatial_idx = SpatialIndex(cell_size=max(2, int(adaptive_proximity)))
        spatial_idx.build(after_snapshot.objects)

        # Match parent / larger objects before enclosed child objects so child search centers can shift coherently
        sorted_before = sorted(
            before_snapshot.objects,
            key=lambda o: (getattr(o, "parent_id", None) is not None, -o.area),
        )
        parent_deltas: dict[str, tuple[float, float]] = {}

        # Pass 1: Exact color matching with parent-coherent offset and area similarity preference
        unmatched_before = []
        for b_obj in sorted_before:
            p_id = getattr(b_obj, "parent_id", None)
            if p_id and p_id in parent_deltas:
                p_dr, p_dc = parent_deltas[p_id]
                exp_r = b_obj.centroid.row + p_dr
                exp_c = b_obj.centroid.col + p_dc
            else:
                exp_r = b_obj.centroid.row
                exp_c = b_obj.centroid.col

            max_search_dist = max(adaptive_proximity * 2.5, b_obj.bbox.height * 2, b_obj.bbox.width * 2)
            radius_cells = max(2, int(math.ceil(max_search_dist / spatial_idx.cell_size)))
            candidates = spatial_idx.find_nearby(exp_r, exp_c, radius_cells=radius_cells)

            best_match = None
            best_dist = float("inf")
            fallback_match = None
            fallback_dist = float("inf")
            fallback_limit = max(adaptive_proximity, b_obj.bbox.height, b_obj.bbox.width)

            for a_obj in candidates:
                if a_obj.id in matched_after_ids:
                    continue
                if a_obj.color == b_obj.color:
                    d_exp = math.hypot(exp_r - a_obj.centroid.row, exp_c - a_obj.centroid.col)
                    area_diff = abs(a_obj.area - b_obj.area)
                    area_similar = area_diff <= max(2, int(round(b_obj.area * max(area_tol, 0.25))))
                    if area_similar and d_exp <= max_search_dist and d_exp < best_dist:
                        best_dist = d_exp
                        best_match = a_obj
                    d_orig = math.hypot(b_obj.centroid.row - a_obj.centroid.row, b_obj.centroid.col - a_obj.centroid.col)
                    if d_orig <= fallback_limit and d_orig < fallback_dist:
                        fallback_dist = d_orig
                        fallback_match = a_obj

            chosen = best_match if best_match is not None else fallback_match
            if chosen is not None:
                matched_pairs.append((b_obj.id, chosen))
                matched_after_ids.add(chosen.id)
                parent_deltas[b_obj.id] = (
                    chosen.centroid.row - b_obj.centroid.row,
                    chosen.centroid.col - b_obj.centroid.col,
                )
            else:
                unmatched_before.append(b_obj)

        # Pass 2: Topological matching fallback for objects undergoing in-place color/area transition
        still_unmatched = []
        for b_obj in unmatched_before:
            best_match = None
            best_dist = float("inf")
            max_search_dist = max(adaptive_proximity + 0.5, b_obj.bbox.height + 1, b_obj.bbox.width + 1)
            radius_cells = max(2, int(math.ceil(max_search_dist / spatial_idx.cell_size)))
            candidates = spatial_idx.find_nearby(b_obj.centroid.row, b_obj.centroid.col, radius_cells=radius_cells)
            for a_obj in candidates:
                if a_obj.id in matched_after_ids:
                    continue
                d = math.hypot(b_obj.centroid.row - a_obj.centroid.row, b_obj.centroid.col - a_obj.centroid.col)
                area_compatible = abs(a_obj.area - b_obj.area) <= max(4, int(round(b_obj.area * max(area_tol, 0.5))))
                is_close = d <= max(adaptive_proximity, min(b_obj.bbox.height, b_obj.bbox.width) + 1) and area_compatible
                is_shape_similar = abs(a_obj.area - b_obj.area) <= max(2, int(round(b_obj.area * area_tol))) and a_obj.color == b_obj.color
                if (is_close or is_shape_similar) and d <= max_search_dist:
                    if d < best_dist:
                        best_dist = d
                        best_match = a_obj

            if best_match is not None:
                matched_pairs.append((b_obj.id, best_match))
                matched_after_ids.add(best_match.id)
            else:
                still_unmatched.append(b_obj)

        return matched_pairs, still_unmatched

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
        matched_pairs, still_unmatched = self.match_objects_between_snapshots(
            before_snapshot, after_snapshot, planning_set
        )

        perc = self.config.resolve("perception")
        move_thresh = getattr(perc, "movement_detection_threshold", 1.0) if perc else 1.0

        for b_obj in still_unmatched:
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

            # Positional metric signs and exact integer displacements
            dr = best_match.centroid.row - b_obj.centroid.row
            dc = best_match.centroid.col - b_obj.centroid.col
            r_sign = 1 if dr >= move_thresh else (-1 if dr <= -move_thresh else 0)
            c_sign = 1 if dc >= move_thresh else (-1 if dc <= -move_thresh else 0)
            int_dr = int(round(dr)) if r_sign != 0 else 0
            int_dc = int(round(dc)) if c_sign != 0 else 0

            for s_id in subjects:
                # Object identity preserved
                props.append(AtomicProposition(family="object_identity", subject_id=s_id, predicate="preserved"))

                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="row_delta", value=r_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_r", value=r_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dy", value=r_sign))
                if int_dr != r_sign:
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="row_delta", value=int_dr))
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_r", value=int_dr))
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dy", value=int_dr))

                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="col_delta", value=c_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_c", value=c_sign))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dx", value=c_sign))
                if int_dc != c_sign:
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="col_delta", value=int_dc))
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="delta_c", value=int_dc))
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="dx", value=int_dc))

                # Step-level and cumulative motion tuples
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="step_moved", value=(r_sign, c_sign)))
                props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="moved", value=(r_sign, c_sign)))
                if (int_dr, int_dc) != (r_sign, c_sign):
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="step_moved", value=(int_dr, int_dc)))
                    props.append(AtomicProposition(family="metric_sign", subject_id=s_id, predicate="moved", value=(int_dr, int_dc)))
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
            act_id = parse_referenced_action_id(step.dsl_function)

        confirmed_eff = ""
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "")

        eff_lower = confirmed_eff.lower()
        is_modal_selection = (
            is_non_vector_move(act_id)
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
            # Check if this could be a boundary collision geometrically (soft stop)
            subject_id_from_step = None
            if step and step.expected_propositions:
                for p in step.expected_propositions:
                    if getattr(p, "family", None) == "metric_sign" and getattr(p, "subject_id", None):
                        subject_id_from_step = p.subject_id
                        break
                if not subject_id_from_step:
                    for p in step.expected_propositions:
                        if getattr(p, "subject_id", None):
                            subject_id_from_step = p.subject_id
                            break

            is_boundary_collision = is_blocked_by_boundary(
                planning_set, act_id, subject_id_from_step, before_grid
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
                if is_non_vector_move(act)
                or (game_memory and any(kw in getattr(game_memory, "confirmed_action_effects", {}).get(act, "").lower() for kw in ("selection", "toggle", "indicator")))
            ]
            unconfirmed_actions = [
                act for act in allowed_acts
                if is_observable_move(act) and not is_vector_action(act)
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
                    grid_h, grid_w = _safe_get_grid_dims(planning_set, before_snapshot)
                    adaptive_proximity = max(grid_h, grid_w) * 0.08
                    spatial_idx = SpatialIndex(cell_size=max(2, int(adaptive_proximity)))
                    spatial_idx.build(after_snapshot.objects)

                    for b_obj in before_snapshot.objects:
                        best_match = None
                        best_dist = float("inf")
                        max_search_dist = max(adaptive_proximity, b_obj.bbox.height, b_obj.bbox.width)
                        radius_cells = max(2, int(math.ceil(max_search_dist / spatial_idx.cell_size)))
                        candidates = spatial_idx.find_nearby(b_obj.centroid.row, b_obj.centroid.col, radius_cells=radius_cells)
                        for a_obj in candidates:
                            if a_obj.id in matched_after_ids:
                                continue
                            if a_obj.color == b_obj.color:
                                d = math.hypot(b_obj.centroid.row - a_obj.centroid.row, b_obj.centroid.col - a_obj.centroid.col)
                                if d < best_dist:
                                    best_dist = d
                                    best_match = a_obj
                        if best_match is not None and best_dist <= max_search_dist:
                            matched_after_ids.add(best_match.id)
                            # 1. Area preservation: adaptive occlusion threshold
                            area_max = max(b_obj.area, best_match.area)
                            area_min = min(b_obj.area, best_match.area)
                            area_occlusion_threshold = 0.5 + 0.2 * (1.0 - min(b_obj.area, 50) / 50.0)
                            if area_max > 0 and (area_min / area_max) < area_occlusion_threshold:
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
