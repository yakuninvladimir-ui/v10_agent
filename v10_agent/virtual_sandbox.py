"""Virtual Kinematic Sandbox for full-trajectory evaluation and autonomous repair.

Domain-general in-memory forward simulator verifying:
1. Spatial boundary constraints (no wall crashes).
2. Hit-testing of coordinate clicks (ensuring clicks hit object pixels, not background).
3. Universal algebraic invariant satisfaction (axial horizontal/vertical symmetries, sockets, contact).
4. Autonomous trajectory repair:
   - Trimming trailing steps once goal invariant is satisfied.
   - Clamping horizontal/vertical overshoots to exact alignment.
   - Patching misaligned click coordinates to valid object pixel masks.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from v10_agent.action_semantics import (
    ACTION_DIRECTION_NAMES,
    ACTION_VECTORS,
    contains_token,
    normalize_action_id,
    parse_displacement,
    parse_object_displacements,
    parse_object_ids_after_keywords,
)
from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import PlanningObject, PlanningSet
from v10_agent.universal_invariants import DiscoveredInvariant, discover_invariants, compute_invariant_distance

logger = logging.getLogger("v10_agent.virtual_sandbox")


def preprocess_grid(raw_grid: list[list[int]]) -> list[list[int]]:
    """Crop 1-pixel border from raw ARC-AGI-3 grid to remove system indicator frame.

    ARC-AGI-3 grids are up to 64x64 with 16 colors (0..15).
    The outermost 1-pixel ring contains system-level indicators.

    Returns:
        Cropped grid with 1px border removed: (H-2) x (W-2)
    """
    if not raw_grid or len(raw_grid) < 3:
        return raw_grid
    if len(raw_grid[0]) < 3:
        return raw_grid
    return [row[1:-1] for row in raw_grid[1:-1]]


def translate_coords_to_raw(x: int, y: int, offset: int = 1) -> tuple[int, int]:
    """Translate agent-space coordinates back to raw grid coordinates (add offset)."""
    return (x + offset, y + offset)


ACTION7_BLOCKED_MSG = "ACTION7 (Undo) is hardware-blocked and excluded from the search space."

#: Relational keywords that name the entity a selection mechanic acts on.
_SELECTION_KEYWORDS: tuple[str, ...] = ("moved", "controlled", "active")

#: Phrases that mark a selection mechanic as a state switch even when no
#: entity token is available.
_SELECTION_STATUS_KEYWORDS: tuple[str, ...] = ("toggle", "switch", "action5", "active entity toggled")


@dataclass
class SandboxEvaluationResult:
    verdict: str  # "APPROVED" | "REPAIRED" | "REJECTED"
    goal_reached: bool
    repaired_steps: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    terminal_step_index: int = -1
    min_distance_achieved: float = 999.0
    active_invariant: DiscoveredInvariant | None = None
    has_boundary_violation: bool = False
    has_collision: bool = False
    collision_object_id: str = ""


class VirtualKinematicSandbox:
    """In-memory forward kinematic simulator for trajectory validation and repair."""

    def __init__(self, planning_set: PlanningSet, game_memory: GameMemory | None = None):
        self.planning_set = planning_set
        self.game_memory = game_memory
        self.grid_h, self.grid_w = planning_set.grid_dims
        self._extract_kinematics()
        confirmed_actors = getattr(game_memory, "confirmed_actors", set()) if game_memory else set()
        self.invariants = discover_invariants(planning_set, confirmed_actors=confirmed_actors)

    def _extract_kinematics(self) -> None:
        """Extract displacement vectors and mechanics from confirmed game memory."""
        self.action_vectors: dict[str, tuple[int, int]] = {
            normalize_action_id(action_id): vector
            for action_id, vector in ACTION_VECTORS.items()
        }
        self.object_action_vectors: dict[str, dict[str, tuple[int, int]]] = {}
        self.axis_actions: set[str] = set()
        self.piece_actions: set[str] = set(self.action_vectors.keys())

        if self.game_memory and self.game_memory.confirmed_action_effects:
            for act_id, eff_str in self.game_memory.confirmed_action_effects.items():
                act_id = normalize_action_id(act_id)
                has_motion = False
                # Parse per-object motion: "moved obj_1 ... by dy=0, dx=-3"
                for oid, displacement in parse_object_displacements(eff_str):
                    self.object_action_vectors.setdefault(oid, {})[act_id] = displacement
                    has_motion = True

                # Fallback general vector
                displacement = parse_displacement(eff_str)
                if displacement is not None:
                    self.action_vectors[act_id] = displacement
                    has_motion = True

                # If an action had confirmed non-motion effects (e.g. palette transition, color change, toggle)
                # but NO motion was observed (no dy/dx or moved), zero-out its default kinematic vector!
                if not has_motion and any(kw in str(eff_str).lower() for kw in ("color", "palette", "transition", "swap", "cells changed", "toggle", "switch", "indicator")):
                    self.action_vectors[act_id] = (0, 0)

        # Zero-out displacement for actions confirmed as non-kinematic (palette, trigger, selection)
        if self.game_memory and hasattr(self.game_memory, "action_affordances"):
            for aff in getattr(self.game_memory, "action_affordances", []):
                if isinstance(aff, dict):
                    act_id = normalize_action_id(aff.get("action_id", ""))
                    eff_cls = str(aff.get("effect_class", "")).upper()
                    if eff_cls and eff_cls not in ("KINEMATIC", "UNKNOWN") and act_id in self.action_vectors:
                        self.action_vectors[act_id] = (0, 0)

        # Pieces can move in all 4 directions by default
        self.piece_actions = set(self.action_vectors.keys())

        # Universal action classification — no game-specific axis geometry assumptions.
        # All directional actions are available by default; the simulation layer
        # determines mobility via collision checks, not shape heuristics.
        self.v_axis_actions: set[str] = {act for act, (dy, dx) in self.action_vectors.items() if dy == 0 and dx != 0}
        self.h_axis_actions: set[str] = {act for act, (dy, dx) in self.action_vectors.items() if dy != 0 and dx == 0}
        self.axis_actions = set(self.action_vectors.keys())

    def _get_displacement(
        self,
        obj_id: str | None,
        fn_name: str,
        doc: str,
        is_target: bool = False,
        actor_type: str = "piece",
    ) -> tuple[int, int]:
        """Resolve displacement vector for a specific object and DSL function."""
        fn_lower = fn_name.lower()
        doc_lower = doc.lower()
        # 1. Target sockets/goals are stationary by default unless probes showed otherwise
        if is_target or actor_type == "target":
            return (0, 0)

        # Non-kinematic operations (palette swaps, triggers, clicks, rotations, selections) have no positional translation
        if any(kw in doc_lower or kw in fn_lower for kw in (
            "palette", "transition", "color", "toggle", "trigger", "select", "switch", "coord", "click"
        )):
            return (0, 0)

        dy, dx = (0, 0)
        # 2. Check object-specific kinematics from game memory
        if obj_id and obj_id in self.object_action_vectors:
            for act_id, vec in self.object_action_vectors[obj_id].items():
                if act_id.lower() in fn_lower or fn_lower in act_id.lower():
                    dy, dx = vec
                    break

        # 3. Check general action_vectors
        if dy == 0 and dx == 0:
            for act_id, vec in self.action_vectors.items():
                if act_id.lower() in fn_lower or fn_lower in act_id.lower():
                    dy, dx = vec
                    break

        # 4. Check keywords in docstring or function name
        if dy == 0 and dx == 0:
            if "up" in fn_lower or "up" in doc_lower:
                dy, dx = self.action_vectors.get("ACTION1", ACTION_VECTORS["ACTION1"])
            elif "down" in fn_lower or "down" in doc_lower:
                dy, dx = self.action_vectors.get("ACTION2", ACTION_VECTORS["ACTION2"])
            elif "left" in fn_lower or "left" in doc_lower:
                dy, dx = self.action_vectors.get("ACTION3", ACTION_VECTORS["ACTION3"])
            elif "right" in fn_lower or "right" in doc_lower:
                dy, dx = self.action_vectors.get("ACTION4", ACTION_VECTORS["ACTION4"])

        # 5. Actor isolation for symmetry axes:
        # A vertical axis (height > width) shifts horizontally; motion along the axis (dy != 0) is constrained.
        # A horizontal axis (width > height) shifts vertically; motion along the axis (dx != 0) is constrained.
        if actor_type == "axis" and obj_id and self.planning_set:
            obj = self.planning_set.get_object(obj_id)
            if obj:
                if obj.height > obj.width and dy != 0:
                    return (0, 0)
                elif obj.width > obj.height and dx != 0:
                    return (0, 0)

        return (dy, dx)

    def compute_initial_distance(self, inv: DiscoveredInvariant) -> float:
        """Compute initial invariant distance at step 0 to distinguish unsatisfied goals from baseline symmetries."""
        piece_obj = self.planning_set.get_object(inv.subject_id)
        target_obj = self.planning_set.get_object(inv.target_id)
        axis_obj = self.planning_set.get_object(inv.axis_id) if inv.axis_id else None

        sim_piece_r = piece_obj.centroid.row if piece_obj else 0.0
        sim_piece_c = piece_obj.centroid.col if piece_obj else 0.0
        sim_target_r = target_obj.centroid.row if target_obj else (inv.target_position[0] if inv.target_position else 0.0)
        sim_target_c = target_obj.centroid.col if target_obj else (inv.target_position[1] if inv.target_position else 0.0)

        is_h_axis = (inv.invariant_type == "axial_symmetry_horizontal")
        is_v_axis = (inv.invariant_type == "axial_symmetry_vertical")
        if is_h_axis:
            sim_axis_coord = axis_obj.centroid.row if axis_obj else (inv.axis_coordinate or 0.0)
        elif is_v_axis:
            sim_axis_coord = axis_obj.centroid.col if axis_obj else (inv.axis_coordinate or 0.0)
        else:
            sim_axis_coord = None

        # Initial distance is computed purely from object centroids and geometric invariants.

        return compute_invariant_distance(
            subject_r=sim_piece_r,
            subject_c=sim_piece_c,
            axis_val=sim_axis_coord,
            target_r=sim_target_r,
            target_c=sim_target_c,
            invariant_type=inv.invariant_type,
        )

    def _simulate_invariant(
        self,
        chosen_inv: DiscoveredInvariant,
        steps: list[dict[str, Any]],
        manifest_functions: dict[str, Any],
    ) -> SandboxEvaluationResult:
        """Simulate a candidate trajectory against a specific discovered invariant."""
        piece_obj = self.planning_set.get_object(chosen_inv.subject_id)
        target_obj = self.planning_set.get_object(chosen_inv.target_id)
        axis_obj = self.planning_set.get_object(chosen_inv.axis_id) if chosen_inv.axis_id else None

        if piece_obj is None and target_obj is None:
            return SandboxEvaluationResult(
                verdict="APPROVED",
                goal_reached=False,
                repaired_steps=steps,
                reason="Unconstrained domain: allowed without simulation",
                active_invariant=chosen_inv,
            )

        inv_type = chosen_inv.invariant_type
        sim_piece_r = piece_obj.centroid.row if piece_obj else 0.0
        sim_piece_c = piece_obj.centroid.col if piece_obj else 0.0
        sim_target_r = target_obj.centroid.row if target_obj else (chosen_inv.target_position[0] if chosen_inv.target_position else 0.0)
        sim_target_c = target_obj.centroid.col if target_obj else (chosen_inv.target_position[1] if chosen_inv.target_position else 0.0)

        is_h_axis = (inv_type == "axial_symmetry_horizontal")
        is_v_axis = (inv_type == "axial_symmetry_vertical")

        if is_h_axis:
            sim_axis_coord = axis_obj.centroid.row if axis_obj else (chosen_inv.axis_coordinate or 0.0)
        elif is_v_axis:
            sim_axis_coord = axis_obj.centroid.col if axis_obj else (chosen_inv.axis_coordinate or 0.0)
        else:
            sim_axis_coord = None

        # Invariant simulation operates purely on object centroids and geometric properties.

        # Determine selectable actors in the scene
        selectable_actors: list[dict[str, Any]] = []
        if axis_obj:
            selectable_actors.append({"type": "axis", "id": axis_obj.id, "obj": axis_obj})
        if piece_obj and piece_obj.id not in [a["id"] for a in selectable_actors]:
            selectable_actors.append({"type": "piece", "id": piece_obj.id, "obj": piece_obj})
        if self.planning_set:
            for o in self.planning_set.objects:
                if o.id not in [a["id"] for a in selectable_actors]:
                    if axis_obj and o.id == axis_obj.id:
                        selectable_actors.append({"type": "axis", "id": o.id, "obj": o})
                    elif o.area >= 1 and o.color != 0:
                        if target_obj and o.id == target_obj.id:
                            continue
                        selectable_actors.append({"type": "piece", "id": o.id, "obj": o})
        if not selectable_actors:
            selectable_actors = [{"type": "piece", "id": piece_obj.id if piece_obj else "obj_piece", "obj": piece_obj}]

        # Determine initial active actor: check confirmed selection mechanics in game_memory
        active_actor = "piece"
        if self.game_memory and self.game_memory.selection_mechanics:
            for note in self.game_memory.selection_mechanics:
                # General actor identification from selection mechanics
                src_ids = [e for _, e in parse_object_ids_after_keywords(note, _SELECTION_KEYWORDS)]
                if src_ids:
                    src_id = src_ids[0]
                    if axis_obj and src_id == axis_obj.id:
                        active_actor = "axis"
                        break
                if any(kw in note.lower() for kw in _SELECTION_STATUS_KEYWORDS):
                    if axis_obj:
                        active_actor = "axis"
                        break

        actor_idx = 0
        for i, a in enumerate(selectable_actors):
            if a["type"] == active_actor:
                actor_idx = i
                break

        repaired_steps: list[dict[str, Any]] = []
        current_dist = compute_invariant_distance(
            subject_r=sim_piece_r,
            subject_c=sim_piece_c,
            axis_val=sim_axis_coord,
            target_r=sim_target_r,
            target_c=sim_target_c,
            invariant_type=inv_type,
        )
        min_dist = current_dist
        initial_dist = current_dist
        total_dist = current_dist
        TOLERANCE = 1.0 if (is_h_axis or is_v_axis) else 0.2
        is_already_satisfied = (initial_dist <= TOLERANCE)
        has_unsatisfied_invariants = any(
            inv.invariant_type == chosen_inv.invariant_type and self.compute_initial_distance(inv) > TOLERANCE
            for inv in self.invariants
        )
        is_baseline_symmetry = has_unsatisfied_invariants and is_already_satisfied
        goal_reached_idx = -1

        has_multi_actor = sum(
            1 for s in steps if any(k in s.get("dsl_function", "").lower() for k in ("click", "coord", "action6", "toggle", "switch", "action5"))
        ) >= 1

        # Track spatial simulation states to prune non-progressive cyclic loops
        seen_states: dict[tuple[Any, ...], tuple[int, float]] = {}
        initial_state_key = (
            actor_idx,
            int(round(sim_piece_r)),
            int(round(sim_piece_c)),
            int(round(sim_axis_coord)) if sim_axis_coord is not None else None,
        )
        seen_states[initial_state_key] = (0, initial_dist)

        has_boundary_violation = False
        has_collision = False
        collision_obj_id = ""

        for idx, step in enumerate(steps):
            fn_name = step.get("dsl_function", "")
            fn_meta = manifest_functions.get(fn_name, {})
            doc = fn_meta.get("docstring", "").lower()
            args = step.get("arguments", {})

            # 1. Focus / Actor Toggle
            if any(k in fn_name.lower() or k in doc for k in ("toggle", "select_axis", "switch", "action5")):
                actor_idx = (actor_idx + 1) % len(selectable_actors)
                active_actor = selectable_actors[actor_idx]["type"]
                repaired_steps.append(step)

            # 2. Coordinate Click (ACTION6 / CLICK)
            elif any(k in fn_name.lower() or k in doc for k in ("click", "coord", "action6")):
                if "target" in args and args["target"]:
                    tgt_val = str(args["target"]).strip()
                    for i, a in enumerate(selectable_actors):
                        obj = a["obj"]
                        alias = self.planning_set.object_real_to_alias.get(obj.id, obj.id)
                        if tgt_val in (obj.id, alias):
                            actor_idx = i
                            active_actor = a["type"]
                            break
                elif "x" in args and "y" in args and args["x"] is not None and args["y"] is not None:
                    try:
                        click_x = int(args["x"])
                        click_y = int(args["y"])
                        for i, a in enumerate(selectable_actors):
                            obj = a["obj"]
                            if obj and (obj.bbox.min_col <= click_x <= obj.bbox.max_col and
                                        obj.bbox.min_row <= click_y <= obj.bbox.max_row):
                                actor_idx = i
                                active_actor = a["type"]
                                break
                    except (ValueError, TypeError):
                        pass
                repaired_steps.append(step)

            # 3. Kinematic Displacement
            else:
                if "piece" in fn_name.lower() or "piece" in doc:
                    current_actor_type = "piece"
                elif "axis" in fn_name.lower() or "axis" in doc:
                    current_actor_type = "axis"
                else:
                    current_actor_type = active_actor

                if current_actor_type == "axis" and sim_axis_coord is not None:
                    dr_ax, dc_ax = self._get_displacement(axis_obj.id if axis_obj else None, fn_name, doc, actor_type="axis")
                    d_axis = dr_ax if is_h_axis else dc_ax
                    if d_axis != 0:
                        limit = self.grid_h if is_h_axis else self.grid_w
                        new_axis_coord = sim_axis_coord + d_axis
                        if new_axis_coord < 0 or new_axis_coord >= limit:
                            new_axis_coord = max(0, min(limit - 1, new_axis_coord))
                            logger.info(f"Sandbox: step {idx} ({fn_name}) axis clamped at boundary ({new_axis_coord}).")

                        if is_v_axis:
                            old_refl = 2.0 * sim_axis_coord - sim_piece_c
                            new_refl = 2.0 * new_axis_coord - sim_piece_c
                            tgt_val = sim_target_c
                        else:  # is_h_axis
                            old_refl = 2.0 * sim_axis_coord - sim_piece_r
                            new_refl = 2.0 * new_axis_coord - sim_piece_r
                            tgt_val = sim_target_r

                        old_delta = abs(old_refl - tgt_val)
                        new_delta = abs(new_refl - tgt_val)
                        if not is_baseline_symmetry and old_delta <= TOLERANCE and new_delta > old_delta:
                            logger.info(f"Sandbox: step {idx} ({fn_name}) overshoots axis alignment ({new_delta:.2f} > {old_delta:.2f}). Auto-clamping.")
                            continue

                        sim_axis_coord = new_axis_coord
                    repaired_steps.append(step)

                elif current_actor_type == "axis" and sim_axis_coord is None:
                    repaired_steps.append(step)

                else:
                    # Check if step explicitly specifies an object target / subject
                    step_target_obj = None
                    target_arg = args.get("target") or args.get("subject")
                    if target_arg:
                        res_id = self.planning_set.resolve_object_id(str(target_arg))
                        if res_id:
                            step_target_obj = self.planning_set.get_object(res_id)
                    if not step_target_obj:
                        for p in step.get("expected_propositions", []):
                            p_dict = dict(p) if isinstance(p, dict) else p.to_dict()
                            if p_dict.get("predicate") in ("moved", "step_moved") or p_dict.get("family") == "metric_sign":
                                s_id = p_dict.get("subject_id")
                                if s_id:
                                    res_id = self.planning_set.resolve_object_id(str(s_id))
                                    if res_id:
                                        step_target_obj = self.planning_set.get_object(res_id)
                                        break

                    # Target object motion vs piece object motion
                    is_target_moving = (target_obj is not None and step_target_obj is not None and step_target_obj.id == target_obj.id)

                    if is_target_moving:
                        dr_t, dc_t = self._get_displacement(target_obj.id, fn_name, doc, is_target=True, actor_type="target")
                        if dr_t != 0 or dc_t != 0:
                            new_target_r = sim_target_r + dr_t
                            new_target_c = sim_target_c + dc_t
                            dr_total_t = int(round(new_target_r - target_obj.centroid.row))
                            dc_total_t = int(round(new_target_c - target_obj.centroid.col))
                            t_min_r = target_obj.bbox.min_row + dr_total_t
                            t_max_r = target_obj.bbox.max_row + dr_total_t
                            t_min_c = target_obj.bbox.min_col + dc_total_t
                            t_max_c = target_obj.bbox.max_col + dc_total_t
                            if t_min_r < 0 or t_max_r >= self.grid_h or t_min_c < 0 or t_max_c >= self.grid_w:
                                has_boundary_violation = True
                                logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes target out of grid.")
                            else:
                                sim_target_r = new_target_r
                                sim_target_c = new_target_c
                    else:
                        active_piece = step_target_obj or piece_obj
                        curr_actor_meta = selectable_actors[actor_idx] if actor_idx < len(selectable_actors) else None
                        curr_obj_id = curr_actor_meta["id"] if curr_actor_meta else (active_piece.id if active_piece else None)
                        is_inv_subject = (step_target_obj is not None or piece_obj is None or curr_obj_id == (active_piece.id if active_piece else None) or len([a for a in selectable_actors if a["type"] == "piece"]) <= 1)

                        dr_p, dc_p = self._get_displacement(active_piece.id if active_piece else None, fn_name, doc, is_target=False, actor_type="piece")

                        if is_inv_subject and (dr_p != 0 or dc_p != 0):
                            new_piece_r = sim_piece_r + dr_p
                            new_piece_c = sim_piece_c + dc_p
                            if active_piece:
                                dr_total = int(round(new_piece_r - active_piece.centroid.row))
                                dc_total = int(round(new_piece_c - active_piece.centroid.col))
                                p_min_r = active_piece.bbox.min_row + dr_total
                                p_max_r = active_piece.bbox.max_row + dr_total
                                p_min_c = active_piece.bbox.min_col + dc_total
                                p_max_c = active_piece.bbox.max_col + dc_total
                                if p_min_r < 0 or p_max_r >= self.grid_h or p_min_c < 0 or p_max_c >= self.grid_w:
                                    has_boundary_violation = True
                                    logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes piece out of grid.")
                                else:
                                    sim_piece_r = new_piece_r
                                    sim_piece_c = new_piece_c
                            else:
                                if new_piece_r < 0 or new_piece_r >= self.grid_h or new_piece_c < 0 or new_piece_c >= self.grid_w:
                                    has_boundary_violation = True
                                    logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes piece out of grid.")
                                else:
                                    sim_piece_r = new_piece_r
                                    sim_piece_c = new_piece_c

                            # Pixel-level collision check for moving piece
                            if not has_boundary_violation and active_piece:
                                moved_pixels = {(r + dr_total, c + dc_total) for (r, c) in active_piece.pixels}
                                for other in self.planning_set.objects:
                                    if other.id == active_piece.id or other.color == 0 or other.area <= 0:
                                        continue
                                    if inv_type == "socket_coverage" and target_obj and other.id == target_obj.id:
                                        continue
                                    if axis_obj and other.id == axis_obj.id:
                                        continue
                                    if target_obj and other.id == target_obj.id:
                                        continue
                                    # Skip parent/child containers
                                    if hasattr(active_piece, "child_ids") and other.id in getattr(active_piece, "child_ids", ()):
                                        continue
                                    if hasattr(other, "child_ids") and active_piece.id in getattr(other, "child_ids", ()):
                                        continue
                                    other_role = str(getattr(other, "role", "")).upper()
                                    if any(kw in other_role for kw in ("GOAL", "TARGET", "SOCKET", "COLLECTIBLE", "RECEPTACLE", "TRIGGER")):
                                        continue
                                    # Fast AABB prune before pixel check
                                    if not (p_max_r < other.bbox.min_row or p_min_r > other.bbox.max_row or
                                            p_max_c < other.bbox.min_col or p_min_c > other.bbox.max_col):
                                        if any(cell in other.pixels for cell in moved_pixels):
                                            has_collision = True
                                            collision_obj_id = other.id
                                            logger.warning(f"Sandbox: step {idx} ({fn_name}) collides piece with {other.id}.")
                                            break

                    repaired_steps.append(step)

            total_dist = compute_invariant_distance(
                subject_r=sim_piece_r,
                subject_c=sim_piece_c,
                axis_val=sim_axis_coord,
                target_r=sim_target_r,
                target_c=sim_target_c,
                invariant_type=inv_type,
            )
            min_dist = min(min_dist, total_dist)

            # Check logical cycle: if state was visited before without net progress or environmental mutation
            state_key = (
                actor_idx,
                int(round(sim_piece_r)),
                int(round(sim_piece_c)),
                int(round(sim_axis_coord)) if sim_axis_coord is not None else None,
            )
            if state_key in seen_states and not has_multi_actor:
                prev_idx, prev_dist = seen_states[state_key]
                if total_dist >= prev_dist and (len(repaired_steps) - prev_idx) >= 2:
                    logger.info(
                        f"Sandbox: Pruned non-progressive circular trajectory loop between step {prev_idx} and {len(repaired_steps)}."
                    )
                    repaired_steps = repaired_steps[:prev_idx]
            else:
                seen_states[state_key] = (len(repaired_steps), total_dist)

            if not is_baseline_symmetry and total_dist <= TOLERANCE:
                if goal_reached_idx < 0:
                    goal_reached_idx = len(repaired_steps)
                if not has_multi_actor and len(steps) <= 30:
                    logger.info(f"Sandbox: Invariant satisfaction achieved at step {goal_reached_idx} ({chosen_inv.description})! Trimming remaining steps.")
                    break
                else:
                    logger.info(f"Sandbox: Invariant satisfaction milestone at step {goal_reached_idx} ({chosen_inv.description}). Multi-actor trajectory continuing.")

        if has_boundary_violation:
            return SandboxEvaluationResult(
                verdict="REJECTED",
                goal_reached=False,
                repaired_steps=repaired_steps,
                reason="Trajectory contradicts environment: exceeds grid boundary",
                terminal_step_index=len(repaired_steps),
                min_distance_achieved=min_dist,
                active_invariant=chosen_inv,
                has_boundary_violation=True,
                has_collision=has_collision,
                collision_object_id=collision_obj_id,
            )

        can_claim_goal = not is_baseline_symmetry and len(repaired_steps) > 0

        if can_claim_goal and (goal_reached_idx > 0 or total_dist <= TOLERANCE) and len(steps) <= 20 and len(repaired_steps) > 0:
            end_idx = goal_reached_idx if goal_reached_idx > 0 else len(repaired_steps)
            final_steps = repaired_steps[:end_idx]
            return SandboxEvaluationResult(
                verdict="APPROVED" if len(final_steps) == len(steps) else "REPAIRED",
                goal_reached=True,
                repaired_steps=final_steps,
                reason=f"Invariant satisfaction achieved at step {end_idx} ({chosen_inv.description})",
                terminal_step_index=end_idx,
                min_distance_achieved=min_dist,
                active_invariant=chosen_inv,
                has_boundary_violation=False,
                has_collision=False,
            )

        if repaired_steps:
            goal_achieved = can_claim_goal and (total_dist <= TOLERANCE)
            return SandboxEvaluationResult(
                verdict="REPAIRED" if len(repaired_steps) != len(steps) else "APPROVED",
                goal_reached=goal_achieved,
                repaired_steps=repaired_steps,
                reason=f"Invariant satisfaction achieved ({chosen_inv.description})" if goal_achieved else f"Trajectory advancing toward invariant (min_dist={min_dist:.1f}, inv={inv_type})",
                terminal_step_index=len(repaired_steps),
                min_distance_achieved=min_dist,
                active_invariant=chosen_inv,
                has_boundary_violation=False,
                has_collision=False,
            )

        return SandboxEvaluationResult(
            verdict="REJECTED",
            goal_reached=False,
            reason="All proposed steps collided with boundaries, obstacles, or overshot invariant target",
            active_invariant=chosen_inv,
            has_boundary_violation=has_boundary_violation,
            has_collision=has_collision,
            collision_object_id=collision_obj_id,
        )

    def _simulate_kinematic_bounds_and_collisions(
        self,
        steps: list[dict[str, Any]],
        manifest_functions: dict[str, Any],
    ) -> tuple[bool, bool, str | None]:
        """Check whether trajectory steps violate grid boundaries or collide with other objects."""
        if not self.planning_set or not self.planning_set.objects:
            return False, False, None

        subject_obj = None
        for step in steps:
            args = step.get("arguments", {})
            for val in args.values():
                obj = self.planning_set.get_object(str(val))
                if obj:
                    subject_obj = obj
                    break
            if subject_obj:
                break
        if not subject_obj:
            # Check confirmed actors from game memory
            if self.game_memory and hasattr(self.game_memory, "confirmed_actors") and self.game_memory.confirmed_actors:
                for act_id in self.game_memory.confirmed_actors:
                    cand_act = self.planning_set.get_object(act_id)
                    if cand_act:
                        subject_obj = cand_act
                        break
            # Or inspect objects with distinct dynamic actor/player/piece roles
            if not subject_obj:
                for o in self.planning_set.objects:
                    r_name = str(getattr(o, "role", "")).upper()
                    if any(kw in r_name for kw in ("PLAYER", "PIECE", "DYNAMIC_ACTOR")):
                        subject_obj = o
                        break

        # If no identifiable movable subject is present, actions are non-spatial/board-wide or ungrounded
        if not subject_obj:
            return False, False, None

        sim_r = float(subject_obj.centroid.row)
        sim_c = float(subject_obj.centroid.col)

        for idx, step in enumerate(steps):
            fn_name = str(step.get("dsl_function", ""))
            doc = str(manifest_functions.get(fn_name, {}).get("docstring", ""))
            dr, dc = self._get_displacement(subject_obj.id, fn_name, doc, is_target=False)
            new_r = sim_r + dr
            new_c = sim_c + dc

            if new_r < 0 or new_r >= self.grid_h or new_c < 0 or new_c >= self.grid_w:
                return True, False, None

            dr_total = int(round(new_r - subject_obj.centroid.row))
            dc_total = int(round(new_c - subject_obj.centroid.col))
            p_min_r = subject_obj.bbox.min_row + dr_total
            p_max_r = subject_obj.bbox.max_row + dr_total
            p_min_c = subject_obj.bbox.min_col + dc_total
            p_max_c = subject_obj.bbox.max_col + dc_total

            for other in self.planning_set.objects:
                if other.id == subject_obj.id or other.color == 0 or other.area <= 0:
                    continue
                # Skip target objects and symmetry axes of active spatial invariants for subject collision check
                if any((other.id == inv.target_id or other.id == inv.axis_id) for inv in self.invariants if inv.invariant_type in ("socket_coverage", "spatial_contact", "axial_symmetry_vertical", "axial_symmetry_horizontal")):
                    continue
                # Skip targets, receptacles, and goal sockets from being treated as blocking collisions
                other_role = str(getattr(other, "role", "")).upper()
                if any(kw in other_role for kw in ("GOAL", "TARGET", "SOCKET", "COLLECTIBLE", "RECEPTACLE", "TRIGGER")):
                    continue
                # Skip parent or child objects of subject
                if hasattr(subject_obj, "child_ids") and other.id in getattr(subject_obj, "child_ids", ()):
                    continue
                if hasattr(other, "child_ids") and subject_obj.id in getattr(other, "child_ids", ()):
                    continue

                if not (p_max_r < other.bbox.min_row or p_min_r > other.bbox.max_row or
                        p_max_c < other.bbox.min_col or p_min_c > other.bbox.max_col):
                    return False, True, other.id

            sim_r = new_r
            sim_c = new_c

        return False, False, None

    def evaluate_and_repair_trajectory(
        self,
        steps: list[dict[str, Any]],
        manifest_functions: dict[str, Any],
        hypothesis: dict[str, Any] | None = None,
    ) -> SandboxEvaluationResult:
        """Simulate trajectory in memory, verifying goal reach and repairing step counts."""
        if not steps:
            return SandboxEvaluationResult(
                verdict="REJECTED",
                goal_reached=False,
                reason="Empty trajectory proposed",
            )

        # Only spatial goal invariants can be simulated kinematically
        spatial_types = {"axial_symmetry_vertical", "axial_symmetry_horizontal", "socket_coverage", "spatial_contact"}
        candidate_invariants = [inv for inv in self.invariants if inv.invariant_type in spatial_types]

        if not candidate_invariants:
            has_b, has_c, coll_id = self._simulate_kinematic_bounds_and_collisions(steps, manifest_functions)
            if has_b or has_c:
                reason = "Trajectory exceeds grid boundary" if has_b else f"Trajectory collides with object {coll_id}"
                return SandboxEvaluationResult(
                    verdict="REJECTED",
                    goal_reached=False,
                    repaired_steps=steps,
                    reason=f"Trajectory contradicts environment: {reason}",
                    has_boundary_violation=has_b,
                    has_collision=has_c,
                    collision_object_id=coll_id,
                )
            return SandboxEvaluationResult(
                verdict="APPROVED",
                goal_reached=False,
                repaired_steps=steps,
                reason="Trajectory approved without environment contradiction",
            )

        if hypothesis:
            hypo_text = str(hypothesis).lower()
            matched = []
            for inv in candidate_invariants:
                s_id = inv.subject_id.lower()
                t_id = inv.target_id.lower()
                s_alias = (self.planning_set.object_real_to_alias.get(inv.subject_id) or "").lower()
                t_alias = (self.planning_set.object_real_to_alias.get(inv.target_id) or "").lower()
                axis_alias = (self.planning_set.object_real_to_alias.get(inv.axis_id) or "").lower() if inv.axis_id else ""

                if s_id in hypo_text or contains_token(hypo_text, s_alias):
                    matched.append(inv)
                elif t_id in hypo_text or contains_token(hypo_text, t_alias):
                    matched.append(inv)
                elif inv.axis_id and (inv.axis_id.lower() in hypo_text or contains_token(hypo_text, axis_alias)):
                    matched.append(inv)
            if matched:
                candidate_invariants = matched

        candidate_invariants.sort(
            key=lambda inv: self.compute_initial_distance(inv) > 1.0,
            reverse=True,
        )

        best_result: SandboxEvaluationResult | None = None
        for inv in candidate_invariants[:8]:
            res = self._simulate_invariant(inv, steps, manifest_functions)
            if res.goal_reached:
                return res
            if best_result is None or res.min_distance_achieved < best_result.min_distance_achieved:
                best_result = res

        return best_result or SandboxEvaluationResult(
            verdict="APPROVED",
            goal_reached=False,
            repaired_steps=steps,
            reason="Trajectory approved without active invariant collision",
        )

    def synthesize_invariant_trajectory(
        self,
        inv: DiscoveredInvariant,
        manifest_functions: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Synthesize domain-general candidate trajectory to satisfy a discovered spatial invariant."""
        sub = self.planning_set.get_object(inv.subject_id)
        tgt = self.planning_set.get_object(inv.target_id)
        if not sub or not tgt:
            return None

        axis = self.planning_set.get_object(inv.axis_id) if inv.axis_id else None
        # Determine step size dynamically from confirmed action vectors or fallback to 1
        magnitudes = [
            max(abs(dy), abs(dx))
            for vec_map in ([self.action_vectors] + list(self.object_action_vectors.values()))
            for dy, dx in vec_map.values()
            if max(abs(dy), abs(dx)) > 0
        ]
        step_sz = max(1, magnitudes[0]) if magnitudes else 1

        has_directional_primitives = any(
            action_id.lower() in manifest_functions for action_id in ACTION_VECTORS
        ) or any(
            any(d in f.lower() or d in m.get("docstring", "").lower() for d in ("up", "down", "left", "right"))
            for f, m in manifest_functions.items()
        )
        if not has_directional_primitives:
            return None

        # Directional and toggle primitives
        def _find_directional(action_id: str) -> str:
            name = action_id.lower()
            keyword = ACTION_DIRECTION_NAMES[action_id].lower()
            return next(
                (f for f, m in manifest_functions.items()
                 if name in f.lower() or keyword in f.lower() or keyword in m.get("docstring", "").lower()),
                name,
            )

        up_fn = _find_directional("ACTION1")
        down_fn = _find_directional("ACTION2")
        left_fn = _find_directional("ACTION3")
        right_fn = _find_directional("ACTION4")

        has_selection_toggle = bool(
            self.game_memory and any(
                any(k in m.lower() for k in ("toggle", "switch", "action5", "active entity toggled"))
                for m in getattr(self.game_memory, "selection_mechanics", [])
            )
        )
        toggle_fn = (
            next((f for f, m in manifest_functions.items() if any(k in f.lower() or k in m.get("docstring", "").lower() for k in ("toggle", "switch", "action5"))), None)
            if has_selection_toggle else None
        )

        # Controllable entities in the search graph
        controllable_objects: list[Any] = []
        if axis is not None:
            controllable_objects.append(axis)
        if sub is not None and sub.id not in [o.id for o in controllable_objects]:
            controllable_objects.append(sub)
        if not controllable_objects:
            controllable_objects = [sub]

        # Determine initial active actor
        init_actor = "piece"
        if self.game_memory and self.game_memory.selection_mechanics:
            for note in self.game_memory.selection_mechanics:
                src_ids = [e for _, e in parse_object_ids_after_keywords(note, _SELECTION_KEYWORDS)]
                if src_ids:
                    src_id = src_ids[0]
                    if axis and src_id == axis.id:
                        init_actor = "axis"
                        break
                if any(kw in note.lower() for kw in _SELECTION_STATUS_KEYWORDS):
                    if axis:
                        init_actor = "axis"
                        break

        init_act_idx = 0
        for i, obj in enumerate(controllable_objects):
            if (init_actor == "axis" and axis and obj.id == axis.id) or (init_actor == "piece" and obj.id == sub.id):
                init_act_idx = i
                break

        def get_displacement(act_name: str, obj: Any) -> tuple[int, int]:
            act_doc = str(manifest_functions.get(act_name, {}).get("docstring", ""))
            a_type = "axis" if (axis and obj.id == axis.id) else "piece"
            return self._get_displacement(obj.id, act_name, act_doc, actor_type=a_type)

        # Initial entity positions: (row, col)
        init_pos = tuple((float(obj.centroid.row), float(obj.centroid.col)) for obj in controllable_objects)

        # Check goal satisfaction and heuristic distance
        def eval_feature_delta(positions: tuple[tuple[float, float], ...]) -> tuple[bool, float]:
            pos_map = {obj.id: pos for obj, pos in zip(controllable_objects, positions)}
            sub_pos = pos_map.get(sub.id, (float(sub.centroid.row), float(sub.centroid.col)))
            if inv.invariant_type == "axial_symmetry_vertical":
                axis_val = pos_map[axis.id][1] if axis else (inv.axis_coordinate or 0.0)
            elif inv.invariant_type == "axial_symmetry_horizontal":
                axis_val = pos_map[axis.id][0] if axis else (inv.axis_coordinate or 0.0)
            else:
                axis_val = None

            dist = compute_invariant_distance(
                subject_r=sub_pos[0],
                subject_c=sub_pos[1],
                axis_val=axis_val,
                target_r=float(tgt.centroid.row),
                target_c=float(tgt.centroid.col),
                invariant_type=inv.invariant_type,
            )
            tol = 1.0 if inv.invariant_type in ("axial_symmetry_vertical", "axial_symmetry_horizontal") else 0.2
            return dist <= tol, dist / step_sz

        # Generalized A* search over object feature delta space
        import heapq

        start_satisfied, start_h = eval_feature_delta(init_pos)
        if start_satisfied:
            return []

        dir_actions = [up_fn, down_fn, left_fn, right_fn]

        counter = 0
        open_set = [(start_h, 0, counter, init_pos, init_act_idx, [])]
        best_g = {(init_pos, init_act_idx): 0}
        max_expansions = 1500

        while open_set and max_expansions > 0:
            max_expansions -= 1
            f, g, _, curr_pos, curr_act_idx, path = heapq.heappop(open_set)

            if g > best_g.get((curr_pos, curr_act_idx), float("inf")):
                continue

            satisfied, _ = eval_feature_delta(curr_pos)
            if satisfied:
                return [{"dsl_function": fn, "arguments": {}} for fn in path]

            curr_obj = controllable_objects[curr_act_idx]

            # 1. Branch on directional movements of active entity
            for act_fn in dir_actions:
                dr, dc = get_displacement(act_fn, curr_obj)
                if dr == 0 and dc == 0:
                    continue

                new_pos_list = list(curr_pos)
                new_pos_list[curr_act_idx] = (curr_pos[curr_act_idx][0] + dr, curr_pos[curr_act_idx][1] + dc)
                new_pos_tuple = tuple(new_pos_list)

                sat, h = eval_feature_delta(new_pos_tuple)
                new_g = g + 1
                state_key = (new_pos_tuple, curr_act_idx)
                if new_g < best_g.get(state_key, float("inf")):
                    best_g[state_key] = new_g
                    counter += 1
                    heapq.heappush(open_set, (new_g + h, new_g, counter, new_pos_tuple, curr_act_idx, path + [act_fn]))

            # 2. Branch on entity selection toggle (if available)
            if toggle_fn and len(controllable_objects) > 1:
                next_act_idx = (curr_act_idx + 1) % len(controllable_objects)
                sat, h = eval_feature_delta(curr_pos)
                new_g = g + 1
                state_key = (curr_pos, next_act_idx)
                if new_g < best_g.get(state_key, float("inf")):
                    best_g[state_key] = new_g
                    counter += 1
                    heapq.heappush(open_set, (new_g + h, new_g, counter, curr_pos, next_act_idx, path + [toggle_fn]))

        return None

    def _find_path_astar(
        self,
        subject: PlanningObject,
        start_pos: tuple[float, float],
        goal_pos: tuple[float, float],
        manifest_functions: dict[str, Any],
        allowed_target_id: str | None = None,
        avoid_hazard_colors: set[int] | None = None,
        max_nodes: int = 2000,
    ) -> list[tuple[str, int, int]] | None:
        """A* pathfinding for subject object from start_pos to goal_pos.

        Returns list of (function_name, dy, dx) tuples, or None if no path found.
        """
        import heapq

        if avoid_hazard_colors is None:
            avoid_hazard_colors = set()
            if self.game_memory and hasattr(self.game_memory, "palette_role_map"):
                from v10_agent.types import EntityRole
                hazards = self.game_memory.palette_role_map.get_colors_by_role(EntityRole.HAZARD)
                avoid_hazard_colors = {h.color_id for h in hazards}

        start_r = int(round(start_pos[0]))
        start_c = int(round(start_pos[1]))
        goal_r = int(round(goal_pos[0]))
        goal_c = int(round(goal_pos[1]))

        if (start_r, start_c) == (goal_r, goal_c):
            return []

        # Available directional actions from manifest. Direction geometry comes
        # from the canonical vocabulary; only the mapping onto DSL function names
        # is discovered here.
        dir_actions = []
        for action_id, (dy, dx) in ACTION_VECTORS.items():
            name = action_id.lower()
            keyword = ACTION_DIRECTION_NAMES[action_id].lower()
            fn_name = next(
                (f for f, m in manifest_functions.items() if f.lower() == name or name in f.lower()
                 or keyword in f.lower() or keyword in m.get("docstring", "").lower()),
                name,
            )
            dir_actions.append((fn_name, dy, dx))

        # Check collision with static objects and boundaries
        def is_valid_placement(r: int, c: int) -> bool:
            dr = r - int(round(subject.centroid.row))
            dc = c - int(round(subject.centroid.col))

            p_min_r = subject.bbox.min_row + dr
            p_max_r = subject.bbox.max_row + dr
            p_min_c = subject.bbox.min_col + dc
            p_max_c = subject.bbox.max_col + dc

            if p_min_r < 0 or p_max_r >= self.grid_h or p_min_c < 0 or p_max_c >= self.grid_w:
                return False

            for other in self.planning_set.objects:
                if other.id == subject.id or other.color == 0 or other.area <= 0:
                    continue
                if allowed_target_id and other.id == allowed_target_id:
                    continue
                # Hazard check
                if other.color in avoid_hazard_colors:
                    if not (p_max_r < other.bbox.min_row - 1 or p_min_r > other.bbox.max_row + 1 or
                            p_max_c < other.bbox.min_col - 1 or p_min_c > other.bbox.max_col + 1):
                        return False
                # Solid obstacle check
                if not (p_max_r < other.bbox.min_row or p_min_r > other.bbox.max_row or
                        p_max_c < other.bbox.min_col or p_min_c > other.bbox.max_col):
                    return False

            return True

        h_start = abs(start_r - goal_r) + abs(start_c - goal_c)
        counter = 0
        open_set = [(h_start, 0, counter, (start_r, start_c), [])]
        visited: set[tuple[int, int]] = {(start_r, start_c)}

        nodes_expanded = 0
        while open_set and nodes_expanded < max_nodes:
            _, g, _, curr, path = heapq.heappop(open_set)
            nodes_expanded += 1

            if curr == (goal_r, goal_c):
                return path

            for fn_name, dy, dx in dir_actions:
                nxt = (curr[0] + dy, curr[1] + dx)
                if nxt not in visited:
                    if nxt == (goal_r, goal_c) or is_valid_placement(nxt[0], nxt[1]):
                        visited.add(nxt)
                        h = abs(nxt[0] - goal_r) + abs(nxt[1] - goal_c)
                        counter += 1
                        heapq.heappush(open_set, (g + 1 + h, g + 1, counter, nxt, path + [(fn_name, dy, dx)]))

        return None

    def expand_waypoints_to_trajectory(
        self,
        waypoints: list[dict[str, Any]],
        manifest_functions: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Expand high-level strategic waypoints into executable steps with typed EXPECT clauses."""
        if not waypoints or not self.planning_set:
            return None

        expanded_steps: list[dict[str, Any]] = []
        simulated_positions: dict[str, tuple[float, float]] = {
            obj.id: (float(obj.centroid.row), float(obj.centroid.col))
            for obj in self.planning_set.objects
        }

        # Resolve subject object (default to primary actor if not specified)
        default_subject = None
        actors = getattr(self.game_memory, "confirmed_actors", set()) if self.game_memory else set()
        for act_id in actors:
            obj = self.planning_set.get_object(act_id)
            if obj:
                default_subject = obj
                break
        if not default_subject:
            objs = [o for o in self.planning_set.objects if o.color != 0 and o.area > 0]
            if objs:
                default_subject = objs[0]

        for wp in waypoints:
            wp_type = str(wp.get("type", "")).upper()
            subj_key = wp.get("subject") or wp.get("actor")
            subject = (
                self.planning_set.get_object(self.planning_set.resolve_object_id(str(subj_key)) or "")
                if subj_key
                else default_subject
            )
            if not subject:
                continue

            curr_pos = simulated_positions.get(subject.id, (float(subject.centroid.row), float(subject.centroid.col)))

            if wp_type in ("NAVIGATE_TO", "NAVIGATE", "MOVE_TO"):
                target_key = wp.get("target") or wp.get("destination") or wp.get("arg_0")
                target_pos = None
                target_obj_id = None
                if target_key:
                    res_id = self.planning_set.resolve_object_id(str(target_key))
                    t_obj = self.planning_set.get_object(res_id or "") if res_id else None
                    if t_obj:
                        target_pos = simulated_positions.get(t_obj.id, (float(t_obj.centroid.row), float(t_obj.centroid.col)))
                        target_obj_id = t_obj.id
                if target_pos is None and "r" in wp and "c" in wp:
                    try:
                        target_pos = (float(wp["r"]), float(wp["c"]))
                    except (ValueError, TypeError):
                        pass

                if target_pos is not None:
                    path = self._find_path_astar(
                        subject, curr_pos, target_pos, manifest_functions, allowed_target_id=target_obj_id
                    )
                    if path:
                        for fn_name, dy, dx in path:
                            curr_pos = (curr_pos[0] + dy, curr_pos[1] + dx)
                            simulated_positions[subject.id] = curr_pos
                            expanded_steps.append({
                                "dsl_function": fn_name,
                                "arguments": {},
                                "expected_propositions": [
                                    {"family": "metric_sign", "subject_id": subject.id, "predicate": "moved", "value": (dy, dx)},
                                    {"family": "metric_sign", "subject_id": subject.id, "predicate": "step_moved", "value": (dy, dx)},
                                ],
                            })

            elif wp_type in ("COLLECT_TARGET", "COLLECT"):
                color_val = wp.get("target_color") or wp.get("color")
                target_color = int(color_val) if color_val is not None and str(color_val).isdigit() else None
                targets = [
                    o for o in self.planning_set.objects
                    if o.id != subject.id and (target_color is None or o.color == target_color) and o.area <= 4
                ]
                targets.sort(key=lambda t: abs(t.centroid.row - curr_pos[0]) + abs(t.centroid.col - curr_pos[1]))

                for tgt in targets:
                    t_pos = simulated_positions.get(tgt.id, (float(tgt.centroid.row), float(tgt.centroid.col)))
                    path = self._find_path_astar(subject, curr_pos, t_pos, manifest_functions, allowed_target_id=tgt.id)
                    if path:
                        for idx, (fn_name, dy, dx) in enumerate(path):
                            curr_pos = (curr_pos[0] + dy, curr_pos[1] + dx)
                            simulated_positions[subject.id] = curr_pos
                            props = [
                                {"family": "metric_sign", "subject_id": subject.id, "predicate": "moved", "value": (dy, dx)},
                                {"family": "metric_sign", "subject_id": subject.id, "predicate": "step_moved", "value": (dy, dx)},
                            ]
                            if idx == len(path) - 1:
                                props.append({"family": "object_identity", "subject_id": tgt.id, "predicate": "gone"})
                            expanded_steps.append({
                                "dsl_function": fn_name,
                                "arguments": {},
                                "expected_propositions": props,
                            })

            elif wp_type in ("ACTIVATE_TRIGGER", "TRIGGER"):
                trig_key = wp.get("trigger") or wp.get("target")
                trig_id = self.planning_set.resolve_object_id(str(trig_key)) if trig_key else None
                trig_obj = self.planning_set.get_object(trig_id or "") if trig_id else None
                if trig_obj:
                    t_pos = simulated_positions.get(trig_obj.id, (float(trig_obj.centroid.row), float(trig_obj.centroid.col)))
                    path = self._find_path_astar(subject, curr_pos, t_pos, manifest_functions, allowed_target_id=trig_obj.id)
                    if path:
                        for fn_name, dy, dx in path:
                            curr_pos = (curr_pos[0] + dy, curr_pos[1] + dx)
                            simulated_positions[subject.id] = curr_pos
                            expanded_steps.append({
                                "dsl_function": fn_name,
                                "arguments": {},
                                "expected_propositions": [
                                    {"family": "metric_sign", "subject_id": subject.id, "predicate": "moved", "value": (dy, dx)},
                                    {"family": "metric_sign", "subject_id": subject.id, "predicate": "step_moved", "value": (dy, dx)},
                                ],
                            })

        return expanded_steps if expanded_steps else None


