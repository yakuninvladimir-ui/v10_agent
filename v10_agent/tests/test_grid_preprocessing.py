"""Unit tests for grid preprocessing and coordinate translation.

Verifies:
1. 1-pixel border crop to remove system indicators from ARC-AGI-3 frames.
2. Coordinate re-translation from agent-space (H-2, W-2) to raw environment space (H, W).
3. Handling of grid scales up to 64x64.
4. Edge case safety for minimal dimensions.
"""
from __future__ import annotations

import pytest

from v10_agent.virtual_sandbox import preprocess_grid, translate_coords_to_raw


class TestGridPreprocessing:
    def test_crop_10x10_to_8x8(self):
        # Create 10x10 grid with boundary marker 9 and interior 1
        grid = [[9] * 10] + [[9] + [1] * 8 + [9] for _ in range(8)] + [[9] * 10]
        assert len(grid) == 10
        assert len(grid[0]) == 10

        cropped = preprocess_grid(grid)
        assert len(cropped) == 8
        assert len(cropped[0]) == 8
        # Ensure boundary 9 is completely removed
        for row in cropped:
            for val in row:
                assert val == 1

    def test_crop_64x64_to_62x62(self):
        # ARC-AGI-3 max size 64x64
        grid = [[0] * 64 for _ in range(64)]
        grid[0][0] = 5  # system indicator
        grid[63][63] = 5  # system indicator
        grid[1][1] = 2  # game object

        cropped = preprocess_grid(grid)
        assert len(cropped) == 62
        assert len(cropped[0]) == 62
        assert cropped[0][0] == 2

    def test_translate_coords_to_raw(self):
        # (x, y) in cropped space -> (+1, +1) in raw space
        raw_x, raw_y = translate_coords_to_raw(0, 0)
        assert raw_x == 1
        assert raw_y == 1

        raw_x, raw_y = translate_coords_to_raw(15, 20)
        assert raw_x == 16
        assert raw_y == 21

    def test_small_grid_edge_cases(self):
        # Grids smaller than 3x3 cannot be cropped without destroying contents
        small_2x2 = [[1, 2], [3, 4]]
        assert preprocess_grid(small_2x2) == small_2x2

        empty = []
        assert preprocess_grid(empty) == []
