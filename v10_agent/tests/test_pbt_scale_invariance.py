"""Property-Based Testing for Scale Invariance.

Verifies that topological segmentation, connected components, and relative spatial orders
are strictly preserved under grid scaling (Kronecker product upscaling by factor K).
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set


def upscale_grid(grid: list[list[int]], scale: int) -> list[list[int]]:
    """Upscale each pixel to a scale x scale block."""
    upscaled = []
    for row in grid:
        up_row = []
        for val in row:
            up_row.extend([val] * scale)
        for _ in range(scale):
            upscaled.append(list(up_row))
    return upscaled


@given(
    scale=st.integers(min_value=2, max_value=3),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_pbt_scale_invariance_topology(scale, seed):
    """Property: Connected component count and relative spatial ordering are invariant under scaling."""
    # Create base grid 6x6 with two distinct non-touching objects
    base_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    scaled_grid = upscale_grid(base_grid, scale)

    snap_base = extract_arga_snapshot(base_grid)
    snap_scaled = extract_arga_snapshot(scaled_grid)

    pset_base = build_planning_set(snap_base, available_actions=["ACTION1"])
    pset_scaled = build_planning_set(snap_scaled, available_actions=["ACTION1"])

    # 1. Number of foreground objects must be strictly conserved
    assert len(pset_base.objects) == len(pset_scaled.objects)

    # 2. Area must scale exactly by scale^2
    base_objs = sorted(list(pset_base.objects), key=lambda o: o.color)
    scaled_objs = sorted(list(pset_scaled.objects), key=lambda o: o.color)

    for b_obj, s_obj in zip(base_objs, scaled_objs):
        assert s_obj.area == b_obj.area * (scale ** 2)

    # 3. Relative vertical ordering of centroids must be preserved
    # In base grid, obj 1 (color 1) is above obj 2 (color 2)
    assert base_objs[0].centroid.row < base_objs[1].centroid.row
    assert scaled_objs[0].centroid.row < scaled_objs[1].centroid.row
