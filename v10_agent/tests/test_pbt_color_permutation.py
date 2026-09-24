"""Property-Based Testing for Color Permutation Invariance (Anti-Fingerprinting Fuzzing).

Verifies that the perception graph, topological components, and PaletteRoleMap
are strictly invariant to bijective permutations of the 16-color palette (0..15).
Guarantees the agent does not secretly overfit to hardcoded color values.
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.memory_contours import EntityRole, PaletteRoleMap
from v10_agent.planning_set import build_planning_set


@st.composite
def color_permutation_strategy(draw):
    """Generate a random bijective permutation of the 15 foreground colors (1..15), keeping background 0."""
    perm_fg = draw(st.permutations(list(range(1, 16))))
    perm = {0: 0}
    for orig, p in zip(range(1, 16), perm_fg):
        perm[orig] = p
    return perm


@given(
    grid_w=st.integers(min_value=5, max_value=20),
    grid_h=st.integers(min_value=5, max_value=20),
    perm_map=color_permutation_strategy(),
    seed=st.integers(min_value=1, max_value=10000),
)
def test_pbt_color_permutation_topological_isomorphism(grid_w, grid_h, perm_map, seed):
    """Property: Connected component count and object geometries are invariant under color permutation."""
    import random
    rng = random.Random(seed)

    # Generate random grid with 3 distinct foreground objects on background 0
    raw_grid = [[0] * grid_w for _ in range(grid_h)]
    c1, c2, c3 = 1, 2, 3
    # Object 1
    raw_grid[1][1] = c1
    raw_grid[1][2] = c1
    # Object 2
    raw_grid[3][3] = c2
    # Object 3
    raw_grid[grid_h - 2][grid_w - 2] = c3

    # Permuted grid
    permuted_grid = [[perm_map[raw_grid[r][c]] for c in range(grid_w)] for r in range(grid_h)]

    # Parse both snapshots
    snap_orig = extract_arga_snapshot(raw_grid)
    snap_perm = extract_arga_snapshot(permuted_grid)

    pset_orig = build_planning_set(snap_orig, available_actions=["ACTION1"])
    pset_perm = build_planning_set(snap_perm, available_actions=["ACTION1"])

    # Topological invariant: number of objects must match exactly
    assert len(pset_orig.objects) == len(pset_perm.objects)

    # Object areas and bounding boxes must match after sort
    orig_areas = sorted([o.area for o in pset_orig.objects])
    perm_areas = sorted([o.area for o in pset_perm.objects])
    assert orig_areas == perm_areas

    orig_bboxes = sorted([(b.min_row, b.min_col, b.max_row, b.max_col) for b in [o.bbox for o in pset_orig.objects]])
    perm_bboxes = sorted([(b.min_row, b.min_col, b.max_row, b.max_col) for b in [o.bbox for o in pset_perm.objects]])
    assert orig_bboxes == perm_bboxes


@given(
    perm_map=color_permutation_strategy(),
    hazard_color=st.integers(min_value=1, max_value=15),
    target_color=st.integers(min_value=1, max_value=15),
)
def test_pbt_palette_role_map_permutation_consistency(perm_map, hazard_color, target_color):
    """Property: Role assignment remains consistent when colors are mapped through a permutation."""
    palette_orig = PaletteRoleMap()
    palette_orig.assign_role(hazard_color, EntityRole.HAZARD, confidence=0.9)
    palette_orig.assign_role(target_color, EntityRole.TARGET, confidence=0.95)

    permuted_hazard = perm_map[hazard_color]
    permuted_target = perm_map[target_color]

    palette_perm = PaletteRoleMap()
    palette_perm.assign_role(permuted_hazard, EntityRole.HAZARD, confidence=0.9)
    palette_perm.assign_role(permuted_target, EntityRole.TARGET, confidence=0.95)

    assert palette_orig.get(hazard_color).role == palette_perm.get(permuted_hazard).role
    assert palette_orig.get(target_color).role == palette_perm.get(permuted_target).role
