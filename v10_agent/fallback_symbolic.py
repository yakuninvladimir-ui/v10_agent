"""Pure-symbolic deterministic fallback operators when retries are exhausted."""

from __future__ import annotations

import collections
import logging
from typing import Any

from v10_agent.action_semantics import effect_vectors
from v10_agent.config import V10Config
from v10_agent.planning_set import PlanningSet
from v10_agent.types import ActionDeclaration, EffectDeclaration

logger = logging.getLogger(__name__)


class SymbolicFallbackEngine:
    """Deterministic fallback policy activated when Coder or Solver retries are exhausted."""

    def __init__(self, config: V10Config):
        self.config = config
        self.step_counter = 0
        self._clicked_candidate_ids: set[str] = set()

    def _find_bfs_path(
        self,
        planning_set: PlanningSet,
        game_memory: Any | None = None,
    ) -> list[str] | None:
        """Find 2D BFS grid path for active actor to target object avoiding static obstacles."""
        if len(planning_set.objects) < 2:
            return None

        allowed = planning_set.allowed_action_ids
        action_vectors = effect_vectors(allowed, getattr(game_memory, "confirmed_action_effects", None))

        if not action_vectors:
            return None

        grid_h, grid_w = planning_set.grid_dims
        objects = planning_set.objects

        # 1. Identify actor and target
        confirmed_actors = getattr(game_memory, "confirmed_actors", set()) if game_memory else set()
        actor = None
        if confirmed_actors:
            for obj in objects:
                if obj.id in confirmed_actors:
                    actor = obj
                    break

        if actor is None:
            # Pick the object with smallest area as default mobile entity candidate
            sorted_objs = sorted(objects, key=lambda o: (len(o.pixels), o.id))
            actor = sorted_objs[0]

        # Target: prefer object referenced in goal/invariant relations, or non-touching / matching-shape object
        target = None
        TARGET_REL_TYPES = {
            "chiral_mirror_h", "chiral_mirror_v", "identical_shape",
            "symmetric_axis_of", "mirrored_across_axis", "matches_shape",
            "socket_of", "goal_of"
        }
        if planning_set.relations:
            for rel in planning_set.relations:
                if rel.relation_type in TARGET_REL_TYPES:
                    if rel.subject_id == actor.id and rel.target_id != actor.id:
                        target = planning_set.get_object(rel.target_id)
                        if target:
                            break
                    elif rel.target_id == actor.id and rel.subject_id != actor.id:
                        target = planning_set.get_object(rel.subject_id)
                        if target:
                            break

        if target is None:
            # Check objects matching actor size/shape
            matching_size = [o for o in objects if o.id != actor.id and len(o.pixels) == len(actor.pixels)]
            if matching_size:
                target = matching_size[0]

        if target is None:
            # Prefer non-touching objects (avoid mistaking immediate obstacle walls for target destinations)
            touching_ids = {
                rel.target_id for rel in planning_set.relations
                if rel.relation_type == "touches" and rel.subject_id == actor.id
            } | {
                rel.subject_id for rel in planning_set.relations
                if rel.relation_type == "touches" and rel.target_id == actor.id
            }
            non_touching = [o for o in objects if o.id != actor.id and o.id not in touching_ids]
            if non_touching:
                target = max(
                    non_touching,
                    key=lambda o: abs(o.centroid.row - actor.centroid.row) + abs(o.centroid.col - actor.centroid.col)
                )
            else:
                other_objs = [o for o in objects if o.id != actor.id]
                if other_objs:
                    target = other_objs[0]

        if target is None:
            return None

        # 2. Obstacles: all other objects excluding actor and target
        obstacle_cells: set[tuple[int, int]] = set()
        for obj in objects:
            if obj.id != actor.id and obj.id != target.id:
                obstacle_cells.update(obj.pixels)

        # 3. Actor footprint and target coordinates
        start_r = int(round(actor.centroid.row))
        start_c = int(round(actor.centroid.col))
        footprint = [(r - start_r, c - start_c) for (r, c) in actor.pixels]

        target_r = int(round(target.centroid.row))
        target_c = int(round(target.centroid.col))

        if start_r == target_r and start_c == target_c:
            return None

        # 4. BFS search over grid
        queue: collections.deque[tuple[int, int, list[str]]] = collections.deque([(start_r, start_c, [])])
        visited: set[tuple[int, int]] = {(start_r, start_c)}
        best_path: list[str] | None = None
        best_dist = abs(start_r - target_r) + abs(start_c - target_c)

        max_nodes = 500
        nodes_explored = 0

        while queue and nodes_explored < max_nodes:
            nodes_explored += 1
            curr_r, curr_c, path = queue.popleft()

            dist = abs(curr_r - target_r) + abs(curr_c - target_c)
            if dist < best_dist and path:
                best_dist = dist
                best_path = path

            # Check if reached target
            if (curr_r, curr_c) == (target_r, target_c) or dist == 0:
                return path

            for act_id, (dr, dc) in action_vectors.items():
                nr = curr_r + dr
                nc = curr_c + dc

                if (nr, nc) in visited:
                    continue

                # Collision check for actor footprint at (nr, nc)
                collides = False
                for f_dr, f_dc in footprint:
                    cell_r = nr + f_dr
                    cell_c = nc + f_dc
                    if not (0 <= cell_r < grid_h and 0 <= cell_c < grid_w):
                        collides = True
                        break
                    if (cell_r, cell_c) in obstacle_cells:
                        collides = True
                        break

                if collides:
                    continue

                visited.add((nr, nc))
                queue.append((nr, nc, path + [act_id]))

        if best_path:
            return best_path

        return None

    def select_fallback_action(
        self,
        planning_set: PlanningSet,
        game_memory: Any | None = None,
    ) -> EffectDeclaration:
        """Deterministically select a safe, grounded fallback action."""
        self.step_counter += 1
        allowed = planning_set.allowed_action_ids

        # Heuristic 0: 2D BFS pathfinding toward target avoiding static obstacles
        bfs_path = self._find_bfs_path(planning_set, game_memory)
        if bfs_path:
            chosen_act = bfs_path[0]
            logger.info(
                f"SymbolicFallbackEngine: selected BFS step {chosen_act} (path length: {len(bfs_path)})"
            )
            return EffectDeclaration(
                declared_action=ActionDeclaration(
                    action_id=chosen_act,
                    reasoning={
                        "source": "symbolic_fallback",
                        "strategy": "bfs_pathfinding",
                        "direction": chosen_act,
                        "path_length": len(bfs_path),
                    },
                ),
                expected_metric_deltas={"delta": 1},
                target_object_ids=list(planning_set.object_ids[:1]),
            )

        # Heuristic 1: If ACTION6 (coordinate action) is available and there are unclicked candidates,
        # click unique deduplicated object centroids
        directional_allowed = [act for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if act in allowed]

        if "ACTION6" in allowed and planning_set.coordinate_candidates:
            unclicked = [c for c in planning_set.coordinate_candidates if c.candidate_id not in self._clicked_candidate_ids]
            if not unclicked:
                self._clicked_candidate_ids.clear()
                unclicked = list(planning_set.coordinate_candidates)

            # Prioritize unclicked candidates or interleave with directional movement to test unlock
            if unclicked and (not directional_allowed or self.step_counter % 2 == 1):
                target_cand = unclicked[0]
                self._clicked_candidate_ids.add(target_cand.candidate_id)
                alias = target_cand.label or (target_cand.object_id or "target")
                return EffectDeclaration(
                    declared_action=ActionDeclaration(
                        action_id="ACTION6",
                        data={"x": target_cand.x, "y": target_cand.y, "target": alias},
                        reasoning={
                            "source": "symbolic_fallback",
                            "strategy": "click_unclicked_canonical_candidate",
                            "target_id": target_cand.object_id,
                            "target_alias": alias,
                        },
                    ),
                    expected_metric_deltas={"interaction": 1},
                    target_object_ids=[target_cand.object_id] if target_cand.object_id else [],
                )

        # Heuristic 2: Try directional motions (ACTION1..ACTION4) if available
        if directional_allowed:
            # If ACTION5 (switch entity) is available, cycle entity periodically
            if "ACTION5" in allowed and (self.step_counter % 6 == 0):
                return EffectDeclaration(
                    declared_action=ActionDeclaration(
                        action_id="ACTION5",
                        reasoning={"source": "symbolic_fallback", "strategy": "switch_entity"},
                    ),
                    expected_metric_deltas={"switch": 1},
                    target_object_ids=list(planning_set.object_ids[:1]),
                )

            chosen_dir = directional_allowed[(self.step_counter - 1) % len(directional_allowed)]
            return EffectDeclaration(
                declared_action=ActionDeclaration(
                    action_id=chosen_dir,
                    reasoning={"source": "symbolic_fallback", "strategy": "directional_cycle", "direction": chosen_dir},
                ),
                expected_metric_deltas={"delta": 1},
                target_object_ids=list(planning_set.object_ids[:1]),
            )

        # Heuristic 3: Default allowed action
        fallback_action = allowed[0] if allowed else "RESET"
        return EffectDeclaration(
            declared_action=ActionDeclaration(
                action_id=fallback_action,
                reasoning={"source": "symbolic_fallback", "strategy": "default_allowed"},
            )
        )

    def clear_cycle_memory(self) -> None:
        """Clear clicked cache and offset counter phase to break loop synchronization."""
        self._clicked_candidate_ids.clear()
        self.step_counter += 3
