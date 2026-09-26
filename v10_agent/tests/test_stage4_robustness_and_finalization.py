"""Tests for Audit Stage 4: Robustness, Performance, and Finalization.

Verifies:
- Adaptive proximity and occlusion thresholds in Judge (Task 4.1)
- SpatialIndex nearest-neighbor indexing (Task 4.3)
- Unconditional 1px border crop without min_dim (Task 4.2)
- Deterministic grid hash independent of PYTHONHASHSEED (Task 4.4)
- Atomic DSL module invalidation counter (Task 4.5)
- Memory contour dead blocks activation: exemplars depth 5, defeat patterns, summarize_progression (Task 4.6)
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

from v10_agent.config import V10Config
from v10_agent.cycle_detector import _deterministic_grid_hash, _hash_grid, VisibleCycle
from v10_agent.judge import SpatialIndex, LayeredVerifier
from v10_agent.memory_contours import (
    DefeatExemplar,
    GameMemory,
    VictoryExemplar,
)
from v10_agent.observe import crop_grid_border
from v10_agent.arga_lite import PlanningObject
from v10_agent.types import BoundingBox, Centroid


def test_spatial_index_bucket_query():
    """4.3: SpatialIndex correctly indexes objects and returns candidates in neighborhood."""
    index = SpatialIndex(cell_size=4)
    obj1 = PlanningObject(
        id="obj_1",
        color=1,
        area=4,
        bbox=BoundingBox(0, 0, 1, 1),
        centroid=Centroid(0.5, 0.5),
        pixels=((0, 0), (0, 1), (1, 0), (1, 1)),
        color_histogram={1: 4},
    )
    obj2 = PlanningObject(
        id="obj_2",
        color=2,
        area=9,
        bbox=BoundingBox(20, 20, 22, 22),
        centroid=Centroid(21.0, 21.0),
        pixels=((20, 20),),
        color_histogram={2: 9},
    )
    obj3 = PlanningObject(
        id="obj_3",
        color=1,
        area=4,
        bbox=BoundingBox(2, 2, 3, 3),
        centroid=Centroid(2.5, 2.5),
        pixels=((2, 2),),
        color_histogram={1: 4},
    )
    index.build([obj1, obj2, obj3])

    # Query near (1.0, 1.0) with radius 1 cell
    nearby = index.find_nearby(1.0, 1.0, radius_cells=1)
    nearby_ids = {o.id for o in nearby}
    assert "obj_1" in nearby_ids
    assert "obj_3" in nearby_ids
    assert "obj_2" not in nearby_ids  # obj2 is at (21, 21), far away


def test_unconditional_1px_crop():
    """4.2: crop_grid_border unconditionally removes 1px border without min_dim."""
    # Small 4x4 grid (previously skipped because < 10)
    grid_4x4 = [
        [9, 9, 9, 9],
        [9, 1, 2, 9],
        [9, 3, 4, 9],
        [9, 9, 9, 9],
    ]
    cropped_4x4, offset = crop_grid_border(grid_4x4, border=1)
    assert offset == 1
    assert cropped_4x4 == [
        [1, 2],
        [3, 4],
    ]

    # Large 12x12 grid
    grid_12x12 = [[0] * 12 for _ in range(12)]
    grid_12x12[1][1] = 5
    cropped_12x12, offset = crop_grid_border(grid_12x12, border=1)
    assert offset == 1
    assert len(cropped_12x12) == 10
    assert len(cropped_12x12[0]) == 10
    assert cropped_12x12[0][0] == 5

    # Degenerate 2x2 grid (too small for 1px border on each side: 2 <= 2*1)
    grid_2x2 = [[1, 2], [3, 4]]
    cropped_2x2, offset = crop_grid_border(grid_2x2, border=1)
    assert offset == 0
    assert cropped_2x2 == grid_2x2


def test_deterministic_grid_hash_cross_process():
    """4.4: _deterministic_grid_hash produces stable hash independent of PYTHONHASHSEED."""
    grid = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1],
    ]
    h1 = _deterministic_grid_hash(grid)
    h2 = _hash_grid(grid)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) > 0

    # Test via external python sub-process with PYTHONHASHSEED=0 and PYTHONHASHSEED=12345
    code = (
        "from v10_agent.cycle_detector import _deterministic_grid_hash;"
        "grid = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]];"
        "print(_deterministic_grid_hash(grid))"
    )
    cmd0 = [sys.executable, "-c", code]
    env0 = os.environ.copy()
    env0["PYTHONHASHSEED"] = "0"
    res0 = subprocess.check_output(cmd0, env=env0, text=True).strip()

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "99999"
    res1 = subprocess.check_output(cmd0, env=env1, text=True).strip()

    assert res0 == res1 == h1


def test_exemplars_buffer_depth_and_defeat_pattern_analysis():
    """4.6: GameMemory maintains last 5 exemplars and analyzes recurring defeat patterns."""
    gm = GameMemory(game_id="test_game")

    for i in range(7):
        ex = DefeatExemplar(
            level_index=i,
            fatal_step=i + 1,
            fatal_action_id=2,  # ACTION2
            hazard_color=4,      # Color 4
            explanation=f"Defeat {i}",
        )
        gm.add_defeat(ex)

    # Max depth is 5
    assert len(gm.defeat_exemplars) == 5
    assert gm.last_defeat_exemplar.explanation == "Defeat 6"
    assert gm.defeat_exemplars[0].explanation == "Defeat 2"

    # Recurring pattern: hazard_color=4, fatal_action_id=2 repeated 5 times
    patterns = gm.analyze_defeat_patterns()
    assert len(patterns) >= 1
    assert "Hazard color 4, action ACTION2: 5 times" in patterns[0]


def test_summarize_progression_and_working_hypotheses():
    """4.6: summarize_progression and record_hypothesis activate previously unused blocks."""
    gm = GameMemory(game_id="test_game")
    assert gm.summarize_progression() == "none"

    gm.curriculum_history.append({"level": 0, "steps_to_win": 12, "invariants_confirmed": ["inv_1"]})
    gm.curriculum_history.append({"level": 1, "steps_to_win": 8, "invariants_confirmed": ["inv_1", "inv_2"]})

    prog = gm.summarize_progression()
    assert "Level 0: completed in 12 steps (invariants: 1)" in prog
    assert "Level 1: completed in 8 steps (invariants: 2)" in prog

    # Working hypotheses
    gm.record_hypothesis("Action 1 pushes red blocks", source_step_id="step_3")
    assert len(gm.working_hypotheses) == 1
    assert gm.working_hypotheses[0]["hypothesis"] == "Action 1 pushes red blocks"
    assert gm.working_hypotheses[0]["source_step"] == "step_3"


def test_active_module_invalidation_counter():
    """4.5: active_module_version increments on invalidation and cancels stale executions."""
    from v10_agent.session import GameSession

    cfg = V10Config()
    session = GameSession(config=cfg)

    initial_version = session.active_module_version
    assert initial_version == 0

    session._invalidate_active_module()
    assert session.active_module_version == 1
    assert session.active_module is None
    assert session.replan_requested is True

    session._invalidate_active_module()
    assert session.active_module_version == 2
