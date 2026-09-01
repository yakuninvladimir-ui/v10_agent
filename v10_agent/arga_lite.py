"""
ARGALite Module
Deterministic object-centric perception and SnapshotBuilder.
Ref: Engineering Specification V10.0 Section 1.1, 3.3
"""
from typing import Dict, Any, List, Set, Tuple
from collections import deque

class ARGALite:
    """
    Deterministic object-centric perception grid parser.
    """
    def __init__(self):
        pass

    def get_connected_components(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Extract objects from the grid using connected component labeling (excluding background 0).
        """
        if not grid:
            return []

        height = len(grid)
        width = len(grid[0])
        visited = set()
        objects = []
        obj_id_counter = 0

        def bfs(r, c, color):
            cells = []
            queue = deque([(r, c)])
            visited.add((r, c))

            while queue:
                curr_r, curr_c = queue.popleft()
                cells.append((curr_r, curr_c))

                # Check 4-connected neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            return cells

        for r in range(height):
            for c in range(width):
                color = grid[r][c]
                if color != 0 and (r, c) not in visited:
                    cells = bfs(r, c, color)
                    # Compute centroid
                    r_sum = sum(cell[0] for cell in cells)
                    c_sum = sum(cell[1] for cell in cells)
                    centroid_r = r_sum / len(cells)
                    centroid_c = c_sum / len(cells)

                    objects.append({
                        "id": f"obj_{obj_id_counter}",
                        "color": color,
                        "cells": cells,
                        "centroid": (centroid_r, centroid_c)
                    })
                    obj_id_counter += 1

        return objects

    def parse_grid(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        Parse raw grid into an object-centric representation.
        """
        objects = self.get_connected_components(grid)
        return {
            "objects": objects,
            "relations": [] # Simplification for ARGALite
        }


class SnapshotBuilder:
    """
    Extracts atomic propositions from observations.
    Ref: Engineering Specification V10.0 Section 3.3
    """
    def __init__(self, arga: ARGALite):
        self.arga = arga

    def extract_propositions(self, observation: Dict[str, Any], snapshot_hash: str) -> List[Any]:
        """
        Extract propositions like attribute_delta, positional_delta, etc.
        """
        from .types import AtomicProposition

        propositions = []
        grid = observation.get("grid", [])
        if not grid:
            return propositions

        parsed = self.arga.parse_grid(grid)

        for obj in parsed.get("objects", []):
            obj_id = obj["id"]

            # Object identity
            propositions.append(AtomicProposition(
                family="object_identity",
                data={"object_id": obj_id, "color": obj["color"]},
                objects=(obj_id,),
                relations=()
            ))

            # Attribute (color)
            propositions.append(AtomicProposition(
                family="attribute_delta",
                data={"object_id": obj_id, "attribute": "color", "delta": obj["color"]},
                objects=(obj_id,),
                relations=()
            ))

            # Positional (centroid)
            propositions.append(AtomicProposition(
                family="positional_delta",
                data={
                    "object_id": obj_id,
                    "centroid_row_sign": 1 if obj["centroid"][0] > 0 else 0,
                    "centroid_col_sign": 1 if obj["centroid"][1] > 0 else 0
                },
                objects=(obj_id,),
                relations=()
            ))

        return propositions
