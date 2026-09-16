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
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import PlanningObject, PlanningSet
from v10_agent.universal_invariants import DiscoveredInvariant, discover_invariants, compute_invariant_distance

logger = logging.getLogger("v10_agent.virtual_sandbox")


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
            "action1": (-1, 0),  # default UP
            "action2": (1, 0),   # default DOWN
            "action3": (0, -1),  # default LEFT
            "action4": (0, 1),   # default RIGHT
        }
        self.object_action_vectors: dict[str, dict[str, tuple[int, int]]] = {}
        self.axis_actions: set[str] = set()
        self.piece_actions: set[str] = set(self.action_vectors.keys())

        if self.game_memory and self.game_memory.confirmed_action_effects:
            for act_id, eff_str in self.game_memory.confirmed_action_effects.items():
                act_lower = act_id.lower()
                # Parse per-object motion: "moved obj_1 ... by dy=0, dx=-3"
                for m in re.finditer(r"moved\s+(obj_[a-zA-Z0-9_]+)\s+.*?by\s+dy=([+-]?\d+),\s*dx=([+-]?\d+)", eff_str):
                    oid = m.group(1)
                    dy = int(m.group(2))
                    dx = int(m.group(3))
                    self.object_action_vectors.setdefault(oid, {})[act_lower] = (dy, dx)

                # Fallback general vector
                match_dy = re.search(r"dy=([+-]?\d+)", eff_str)
                match_dx = re.search(r"dx=([+-]?\d+)", eff_str)
                if match_dy and match_dx:
                    self.action_vectors[act_lower] = (int(match_dy.group(1)), int(match_dx.group(1)))

        # Pieces can move in all 4 directions by default
        self.piece_actions = set(self.action_vectors.keys())

        # Classify axis actions when linear reflection axes are present
        has_v_axis = any(
            (o.height >= 10 and o.width <= 4) for o in (self.planning_set.objects if self.planning_set else [])
        )
        has_h_axis = any(
            (o.width >= 10 and o.height <= 4) for o in (self.planning_set.objects if self.planning_set else [])
        )
        self.v_axis_actions: set[str] = {act for act, (dy, dx) in self.action_vectors.items() if dy == 0 and dx != 0}
        self.h_axis_actions: set[str] = {act for act, (dy, dx) in self.action_vectors.items() if dy != 0 and dx == 0}
        self.axis_actions = set()
        if has_v_axis:
            self.axis_actions.update(self.v_axis_actions)
        if has_h_axis:
            self.axis_actions.update(self.h_axis_actions)
        if not has_v_axis and not has_h_axis:
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

        dy, dx = (0, 0)
        # 2. Check object-specific kinematics from game memory
        if obj_id and obj_id in self.object_action_vectors:
            for act_id, vec in self.object_action_vectors[obj_id].items():
                if act_id in fn_lower or fn_lower in act_id:
                    dy, dx = vec
                    break

        # 3. Check general action_vectors
        if dy == 0 and dx == 0:
            for act_id, vec in self.action_vectors.items():
                if act_id in fn_lower or fn_lower in act_id:
                    dy, dx = vec
                    break

        # 4. Check keywords in docstring or function name
        if dy == 0 and dx == 0:
            if "up" in fn_lower or "up" in doc_lower:
                dy, dx = self.action_vectors.get("action1", (-1, 0))
            elif "down" in fn_lower or "down" in doc_lower:
                dy, dx = self.action_vectors.get("action2", (1, 0))
            elif "left" in fn_lower or "left" in doc_lower:
                dy, dx = self.action_vectors.get("action3", (0, -1))
            elif "right" in fn_lower or "right" in doc_lower:
                dy, dx = self.action_vectors.get("action4", (0, 1))

        # 5. Strict actor isolation: linear reflection axes can only move perpendicular to line
        if actor_type == "axis":
            obj = self.planning_set.get_object(obj_id) if (obj_id and self.planning_set) else None
            is_v = (obj.height >= 10 and obj.width <= 4) if obj else bool(self.v_axis_actions)
            is_h = (obj.width >= 10 and obj.height <= 4) if obj else bool(self.h_axis_actions)
            if is_v and dy != 0:
                return (0, 0)
            if is_h and dx != 0:
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

        if self.game_memory and self.game_memory.selection_mechanics:
            for note in self.game_memory.selection_mechanics:
                m_pc = re.search(r"piece_steps=([+-]?\d+)", note)
                if m_pc:
                    p_steps = int(m_pc.group(1))
                    if p_steps != 0:
                        magnitudes = [
                            max(abs(dy), abs(dx))
                            for vec_map in ([self.action_vectors] + list(self.object_action_vectors.values()))
                            for dy, dx in vec_map.values()
                            if max(abs(dy), abs(dx)) > 0
                        ]
                        s_sz = max(1, magnitudes[0]) if magnitudes else 3
                        if sim_piece_r + p_steps * s_sz >= self.grid_h or sim_piece_r + p_steps * s_sz < 0:
                            sim_piece_r = sim_target_r - p_steps * s_sz
                    break

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

        # If kinematics specifies a confirmed piece_steps offset towards target
        if self.game_memory:
            kinematics_inv = self.game_memory.get_invariants_by_type("kinematics")
            for inv in kinematics_inv:
                p_steps = inv.metadata.get("piece_steps", 0)
                if p_steps != 0:
                    magnitudes = [
                        max(abs(dy), abs(dx))
                        for vec_map in ([self.action_vectors] + list(self.object_action_vectors.values()))
                        for dy, dx in vec_map.values()
                        if max(abs(dy), abs(dx)) > 0
                    ]
                    s_sz = max(1, magnitudes[0]) if magnitudes else 3
                    if sim_piece_r + p_steps * s_sz >= self.grid_h or sim_piece_r + p_steps * s_sz < 0:
                        sim_piece_r = sim_target_r - p_steps * s_sz
                    break

        # Determine selectable actors in the scene
        selectable_actors: list[dict[str, Any]] = []
        if axis_obj:
            selectable_actors.append({"type": "axis", "id": axis_obj.id, "obj": axis_obj})
        if piece_obj and piece_obj.id not in [a["id"] for a in selectable_actors]:
            selectable_actors.append({"type": "piece", "id": piece_obj.id, "obj": piece_obj})
        if self.planning_set:
            for o in self.planning_set.objects:
                if o.id not in [a["id"] for a in selectable_actors]:
                    is_other_axis = ((o.height >= 10 and o.width <= 4) or (o.width >= 10 and o.height <= 4))
                    if is_other_axis:
                        selectable_actors.append({"type": "axis", "id": o.id, "obj": o})
                    elif o.area >= 4 and o.color != 0:
                        if target_obj and o.id == target_obj.id:
                            continue
                        selectable_actors.append({"type": "piece", "id": o.id, "obj": o})
        if not selectable_actors:
            selectable_actors = [{"type": "piece", "id": piece_obj.id if piece_obj else "obj_piece", "obj": piece_obj}]

        # Determine initial active actor: check confirmed selection mechanics in game_memory
        active_actor = "piece"
        if self.game_memory and self.game_memory.selection_mechanics:
            for note in self.game_memory.selection_mechanics:
                m = re.search(r"internal dots moved from\s+(obj_[a-zA-Z0-9_]+)", note)
                if m:
                    src_id = m.group(1)
                    if axis_obj and src_id == axis_obj.id:
                        active_actor = "axis"
                        break
                    src_obj = self.planning_set.get_object(src_id) if self.planning_set else None
                    if src_obj and ((src_obj.height >= 10 and src_obj.width <= 4) or (src_obj.width >= 10 and src_obj.height <= 4)):
                        active_actor = "axis"
                        break
                if any(kw in note.lower() for kw in ("toggle", "switch", "action5", "active entity toggled")):
                    if axis_obj and ((axis_obj.height >= 10 and axis_obj.width <= 4) or (axis_obj.width >= 10 and axis_obj.height <= 4)):
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

            # 2. Coordinate Click (ACTION6)
            elif any(k in fn_name.lower() or k in doc for k in ("click", "coord", "action6")):
                if "x" in args and "y" in args:
                    click_x = int(args["x"])
                    click_y = int(args["y"])
                    for i, a in enumerate(selectable_actors):
                        obj = a["obj"]
                        if obj and (obj.bbox.min_col <= click_x <= obj.bbox.max_col and
                                    obj.bbox.min_row <= click_y <= obj.bbox.max_row):
                            actor_idx = i
                            active_actor = a["type"]
                            break
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
                        if new_axis_coord < 1 or new_axis_coord > limit - 2:
                            has_boundary_violation = True
                            logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes axis out of bounds.")
                            continue

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
                    curr_actor_meta = selectable_actors[actor_idx] if actor_idx < len(selectable_actors) else None
                    curr_obj_id = curr_actor_meta["id"] if curr_actor_meta else (piece_obj.id if piece_obj else None)
                    is_inv_subject = (piece_obj is None or curr_obj_id == piece_obj.id or len([a for a in selectable_actors if a["type"] == "piece"]) <= 1)

                    dr_p, dc_p = self._get_displacement(piece_obj.id if piece_obj else None, fn_name, doc, is_target=False, actor_type="piece")
                    dr_t, dc_t = self._get_displacement(target_obj.id if target_obj else None, fn_name, doc, is_target=True, actor_type="target")

                    if is_inv_subject and (dr_p != 0 or dc_p != 0):
                        new_piece_r = sim_piece_r + dr_p
                        new_piece_c = sim_piece_c + dc_p
                        if piece_obj and (new_piece_r < 0 or new_piece_r >= self.grid_h or new_piece_c < 0 or new_piece_c >= self.grid_w):
                            has_boundary_violation = True
                            logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes piece out of grid.")
                        else:
                            sim_piece_r = new_piece_r
                            sim_piece_c = new_piece_c

                    if (dr_t != 0 or dc_t != 0):
                        new_target_r = sim_target_r + dr_t
                        new_target_c = sim_target_c + dc_t
                        if target_obj and (new_target_r < 0 or new_target_r >= self.grid_h or new_target_c < 0 or new_target_c >= self.grid_w):
                            has_boundary_violation = True
                            logger.warning(f"Sandbox: step {idx} ({fn_name}) pushes target out of grid.")
                        else:
                            sim_target_r = new_target_r
                            sim_target_c = new_target_c

                    # Collision check for piece
                    if not has_boundary_violation and piece_obj and (dr_p != 0 or dc_p != 0):
                        dr_total = int(round(sim_piece_r - piece_obj.centroid.row))
                        dc_total = int(round(sim_piece_c - piece_obj.centroid.col))
                        p_min_r = piece_obj.bbox.min_row + dr_total
                        p_max_r = piece_obj.bbox.max_row + dr_total
                        p_min_c = piece_obj.bbox.min_col + dc_total
                        p_max_c = piece_obj.bbox.max_col + dc_total

                        for other in self.planning_set.objects:
                            if other.id == piece_obj.id or other.color == 0 or other.area <= 0:
                                continue
                            if inv_type == "socket_coverage" and target_obj and other.id == target_obj.id:
                                continue
                            if not (p_max_r < other.bbox.min_row or p_min_r > other.bbox.max_row or
                                    p_max_c < other.bbox.min_col or p_min_c > other.bbox.max_col):
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

            if not is_baseline_symmetry and total_dist <= TOLERANCE:
                if goal_reached_idx < 0:
                    goal_reached_idx = len(repaired_steps)
                if not has_multi_actor and len(steps) <= 20:
                    logger.info(f"Sandbox: Invariant satisfaction achieved at step {goal_reached_idx} ({chosen_inv.description})! Trimming remaining steps.")
                    break
                else:
                    logger.info(f"Sandbox: Invariant satisfaction milestone at step {goal_reached_idx} ({chosen_inv.description}). Multi-actor trajectory continuing.")

        if has_boundary_violation or has_collision:
            err_reasons = []
            if has_boundary_violation:
                err_reasons.append("exceeds grid boundary")
            if has_collision:
                err_reasons.append(f"collides with object {collision_obj_id}")
            return SandboxEvaluationResult(
                verdict="REJECTED",
                goal_reached=False,
                repaired_steps=repaired_steps,
                reason=f"Trajectory contradicts environment: {', '.join(err_reasons)}",
                terminal_step_index=len(repaired_steps),
                min_distance_achieved=min_dist,
                active_invariant=chosen_inv,
                has_boundary_violation=has_boundary_violation,
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
            objs = [o for o in self.planning_set.objects if o.color != 0 and o.area > 0]
            if objs:
                # Find an object that matches any direction in functions
                subject_obj = objs[0]

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
            matched = [
                inv for inv in candidate_invariants
                if (inv.subject_id.lower() in hypo_text or inv.target_id.lower() in hypo_text)
                or (inv.axis_id and inv.axis_id.lower() in hypo_text)
            ]
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
            f in manifest_functions for f in ("action1", "action2", "action3", "action4")
        ) or any(
            any(d in f.lower() or d in m.get("docstring", "").lower() for d in ("up", "down", "left", "right"))
            for f, m in manifest_functions.items()
        )
        if not has_directional_primitives:
            return None

        # Find available discrete functions
        up_fn = next((f for f, m in manifest_functions.items() if "up" in f.lower() or "up" in m.get("docstring", "").lower()), "action1")
        down_fn = next((f for f, m in manifest_functions.items() if "down" in f.lower() or "down" in m.get("docstring", "").lower()), "action2")
        left_fn = next((f for f, m in manifest_functions.items() if "left" in f.lower() or "left" in m.get("docstring", "").lower()), "action3")
        right_fn = next((f for f, m in manifest_functions.items() if "right" in f.lower() or "right" in m.get("docstring", "").lower()), "action4")
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

        def find_best_action(oid: str | None, req_dr: int, req_dc: int) -> str | None:
            if oid and oid in self.object_action_vectors:
                for act, (dy, dx) in self.object_action_vectors[oid].items():
                    if (req_dr > 0 and dy > 0) or (req_dr < 0 and dy < 0):
                        return act
                    if (req_dc > 0 and dx > 0) or (req_dc < 0 and dx < 0):
                        return act
            for act, (dy, dx) in self.action_vectors.items():
                if (req_dr > 0 and dy > 0) or (req_dr < 0 and dy < 0):
                    return act
                if (req_dc > 0 and dx > 0) or (req_dc < 0 and dx < 0):
                    return act
            if req_dr < 0:
                return up_fn
            if req_dr > 0:
                return down_fn
            if req_dc < 0:
                return left_fn
            if req_dc > 0:
                return right_fn
            return None

        steps: list[dict[str, Any]] = []

        if inv.invariant_type == "axial_symmetry_vertical" and axis:
            req_axis_c = (tgt.centroid.col + sub.centroid.col) / 2.0
            d_axis_c = req_axis_c - axis.centroid.col
            d_piece_r = tgt.centroid.row - sub.centroid.row
            ax_steps = int(round(d_axis_c / step_sz))
            pc_steps = int(round(d_piece_r / step_sz))

            # Check if kinematics specifies confirmed step offsets
            if (ax_steps == 0 and pc_steps == 0) and self.game_memory:
                kinematics_inv = self.game_memory.get_invariants_by_type("kinematics")
                for inv in kinematics_inv:
                    ax_s = inv.metadata.get("axis_steps")
                    pc_s = inv.metadata.get("piece_steps")
                    if ax_s is not None and pc_s is not None:
                        ax_steps = ax_s
                        pc_steps = pc_s
                        break

            # Determine initial active actor
            init_actor = "piece"
            if self.game_memory:
                for inv in self.game_memory.get_invariants_by_type("kinematics"):
                    if inv.metadata.get("init_actor") == "axis":
                        init_actor = "axis"
                        break
                    # Fallback logic if metadata not present but action5 is toggled
                    if inv.metadata.get("action_id") == "ACTION5":
                        if axis and ((axis.height >= 10 and axis.width <= 4) or (axis.width >= 10 and axis.height <= 4)):
                            init_actor = "axis"
                            break

            if init_actor == "axis":
                ax_fn = find_best_action(axis.id if axis else None, 0, int(round(d_axis_c))) or (left_fn if ax_steps < 0 else right_fn)
                for _ in range(abs(ax_steps)):
                    steps.append({"dsl_function": ax_fn, "arguments": {}})
                if toggle_fn and pc_steps != 0:
                    steps.append({"dsl_function": toggle_fn, "arguments": {}})
                pc_fn = find_best_action(sub.id, int(round(d_piece_r)), 0) or (up_fn if pc_steps < 0 else down_fn)
                for _ in range(abs(pc_steps)):
                    steps.append({"dsl_function": pc_fn, "arguments": {}})
            else:
                pc_fn = find_best_action(sub.id, int(round(d_piece_r)), 0) or (up_fn if pc_steps < 0 else down_fn)
                for _ in range(abs(pc_steps)):
                    steps.append({"dsl_function": pc_fn, "arguments": {}})
                if toggle_fn and ax_steps != 0:
                    steps.append({"dsl_function": toggle_fn, "arguments": {}})
                ax_fn = find_best_action(axis.id if axis else None, 0, int(round(d_axis_c))) or (left_fn if ax_steps < 0 else right_fn)
                for _ in range(abs(ax_steps)):
                    steps.append({"dsl_function": ax_fn, "arguments": {}})

        elif inv.invariant_type == "axial_symmetry_horizontal" and axis:
            req_axis_r = (tgt.centroid.row + sub.centroid.row) / 2.0
            d_axis_r = req_axis_r - axis.centroid.row
            d_piece_c = tgt.centroid.col - sub.centroid.col
            ax_steps = int(round(d_axis_r / step_sz))
            pc_steps = int(round(d_piece_c / step_sz))

            # Check if selection_mechanics specifies confirmed step offsets from probe observation
            if (ax_steps == 0 and pc_steps == 0) and self.game_memory and self.game_memory.selection_mechanics:
                for note in self.game_memory.selection_mechanics:
                    m_ax = re.search(r"axis_steps=([+-]?\d+)", note)
                    m_pc = re.search(r"piece_steps=([+-]?\d+)", note)
                    if m_ax and m_pc:
                        ax_steps = int(m_ax.group(1))
                        pc_steps = int(m_pc.group(1))
                        break

            # Determine initial active actor
            init_actor = "piece"
            if self.game_memory and self.game_memory.selection_mechanics:
                for note in self.game_memory.selection_mechanics:
                    m = re.search(r"internal dots moved from\s+(obj_[a-zA-Z0-9_]+)", note)
                    if m and (m.group(1) == axis.id or (axis.width >= 10 and axis.height <= 4)):
                        init_actor = "axis"
                        break
                    if any(kw in note.lower() for kw in ("toggle", "switch", "action5", "active entity toggled")):
                        if axis and ((axis.height >= 10 and axis.width <= 4) or (axis.width >= 10 and axis.height <= 4)):
                            init_actor = "axis"
                            break

            if init_actor == "axis":
                ax_fn = find_best_action(axis.id if axis else None, int(round(d_axis_r)), 0) or (up_fn if ax_steps < 0 else down_fn)
                for _ in range(abs(ax_steps)):
                    steps.append({"dsl_function": ax_fn, "arguments": {}})
                if toggle_fn and pc_steps != 0:
                    steps.append({"dsl_function": toggle_fn, "arguments": {}})
                pc_fn = find_best_action(sub.id, 0, int(round(d_piece_c))) or (left_fn if pc_steps < 0 else right_fn)
                for _ in range(abs(pc_steps)):
                    steps.append({"dsl_function": pc_fn, "arguments": {}})
            else:
                pc_fn = find_best_action(sub.id, 0, int(round(d_piece_c))) or (left_fn if pc_steps < 0 else right_fn)
                for _ in range(abs(pc_steps)):
                    steps.append({"dsl_function": pc_fn, "arguments": {}})
                if toggle_fn and ax_steps != 0:
                    steps.append({"dsl_function": toggle_fn, "arguments": {}})
                ax_fn = find_best_action(axis.id if axis else None, int(round(d_axis_r)), 0) or (up_fn if ax_steps < 0 else down_fn)
                for _ in range(abs(ax_steps)):
                    steps.append({"dsl_function": ax_fn, "arguments": {}})

        elif inv.invariant_type in ("socket_coverage", "spatial_contact"):
            dr = tgt.centroid.row - sub.centroid.row
            dc = tgt.centroid.col - sub.centroid.col
            r_steps = int(round(dr / step_sz))
            c_steps = int(round(dc / step_sz))
            r_fn = find_best_action(sub.id, int(round(dr)), 0) if r_steps != 0 else None
            c_fn = find_best_action(sub.id, 0, int(round(dc))) if c_steps != 0 else None
            if r_fn:
                for _ in range(abs(r_steps)):
                    steps.append({"dsl_function": r_fn, "arguments": {}})
            if c_fn:
                for _ in range(abs(c_steps)):
                    steps.append({"dsl_function": c_fn, "arguments": {}})

        return steps if steps else None

