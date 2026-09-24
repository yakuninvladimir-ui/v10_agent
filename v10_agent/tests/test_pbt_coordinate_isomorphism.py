"""Property-Based Testing for Dual Coordinate Space Isomorphism.

Verifies:
1. Reversibility of local agent coordinates to raw environment coordinates.
2. Boundary safety across arbitrary grid dimensions up to 64x64.
3. Edge cell preservation under 1px border preprocessing.
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from v10_agent.virtual_sandbox import preprocess_grid, translate_coords_to_raw


@given(
    grid_w=st.integers(min_value=4, max_value=64),
    grid_h=st.integers(min_value=4, max_value=64),
    crop_offset=st.integers(min_value=1, max_value=1),
)
def test_pbt_coordinate_roundtrip(grid_w, grid_h, crop_offset):
    """Property: Any valid coordinate in the cropped workspace transforms and restores losslessly."""
    local_w = grid_w - 2 * crop_offset
    local_h = grid_h - 2 * crop_offset

    for lx in (0, local_w // 2, local_w - 1):
        for ly in (0, local_h // 2, local_h - 1):
            raw_x, raw_y = translate_coords_to_raw(lx, ly, offset=crop_offset)
            # Must be strictly within raw grid boundaries
            assert 0 <= raw_x < grid_w
            assert 0 <= raw_y < grid_h
            # Reverse decode
            restored_lx = raw_x - crop_offset
            restored_ly = raw_y - crop_offset
            assert (restored_lx, restored_ly) == (lx, ly)


@given(
    w=st.integers(min_value=4, max_value=64),
    h=st.integers(min_value=4, max_value=64),
    border_color=st.integers(min_value=10, max_value=15),
    interior_color=st.integers(min_value=1, max_value=9),
)
def test_pbt_preprocess_grid_crops_only_border(w, h, border_color, interior_color):
    """Property: 1px border preprocessing strictly removes perimeter without distorting interior."""
    raw_grid = [[border_color] * w for _ in range(h)]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            raw_grid[r][c] = interior_color

    cropped = preprocess_grid(raw_grid)
    assert len(cropped) == h - 2
    assert len(cropped[0]) == w - 2

    # Entire cropped interior must contain only interior_color
    for row in cropped:
        for val in row:
            assert val == interior_color
