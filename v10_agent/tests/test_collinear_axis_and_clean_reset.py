"""Unit tests verifying collinear mirror axis merging and clean RESET propagation across levels."""

import pytest
from v10_agent.types import BoundingBox
from v10_agent.arga_lite import extract_arga_snapshot, merge_collinear_axis_components
from v10_agent.session import GameSession
from v10_agent.config import V10Config


def test_merge_collinear_horizontal_axis_segments():
    # 20x20 grid with background 0
    # Horizontal line of color 10 on row 10, split into two segments by an obstacle of color 2 at cols 8..11
    # Segment 1: cols 0..7 (width 8 >= 6)
    # Obstacle: cols 8..11 of color 2
    # Segment 2: cols 12..19 (width 8 >= 6)
    grid = [[0] * 20 for _ in range(20)]
    for c in range(0, 8):
        grid[10][c] = 10
    for c in range(8, 12):
        grid[10][c] = 2
    for c in range(12, 20):
        grid[10][c] = 10

    snap = extract_arga_snapshot(grid)
    axis_objs = [o for o in snap.objects if o.color == 10]
    # Should be merged into exactly 1 axis object
    assert len(axis_objs) == 1
    axis = axis_objs[0]
    assert axis.bbox.min_row == 10
    assert axis.bbox.max_row == 10
    assert axis.bbox.min_col == 0
    assert axis.bbox.max_col == 19
    assert axis.shape_type == "horizontal_line"
    assert axis.width == 20


def test_merge_collinear_vertical_axis_segments():
    # 20x20 grid with background 0
    # Vertical line of color 3 on col 5, split by obstacle of color 1 at rows 8..11
    grid = [[0] * 20 for _ in range(20)]
    for r in range(0, 8):
        grid[r][5] = 3
    for r in range(8, 12):
        grid[r][5] = 1
    for r in range(12, 20):
        grid[r][5] = 3

    snap = extract_arga_snapshot(grid)
    axis_objs = [o for o in snap.objects if o.color == 3]
    assert len(axis_objs) == 1
    axis = axis_objs[0]
    assert axis.bbox.min_col == 5
    assert axis.bbox.max_col == 5
    assert axis.bbox.min_row == 0
    assert axis.bbox.max_row == 19
    assert axis.shape_type == "vertical_line"
    assert axis.height == 20


from v10_agent.llm_advisor import MockLLMAdvisor


def test_clean_reset_on_level_greater_than_zero():
    cfg = V10Config(max_actions_per_game=50, max_actions_per_level=50, llm_advisor_backend="fake")
    session = GameSession(cfg, MockLLMAdvisor())
    session.handle_game_transition("test_game")
    session.handle_level_transition("level_1")
    session.levels_completed_observed = 1  # Level > 0!

    # Simulate solver_reset_pending = True (e.g. from evaluate_transition)
    session.solver_reset_pending = True
    session.solver_reset_reason = "replan_reset_clean_state"

    obs = {
        "grid": [[0] * 10 for _ in range(10)],
        "grid_hash": "dummy_hash_1",
        "available_actions": ["RESET", "ACTION1", "ACTION2"],
        "levels_completed": 1,
        "state": "NOT_FINISHED",
    }
    action = session.act(obs)
    # Invariant: session MUST emit RESET even when levels_completed_observed > 0
    assert action["action_id"] == "RESET"
    assert action["reasoning"]["source"] == "replan_reset_clean_state"
    assert session.solver_reset_pending is False
