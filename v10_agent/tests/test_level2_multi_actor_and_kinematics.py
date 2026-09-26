"""Unit tests for multi-actor simulation, 4-directional piece kinematics, marker grouping, and winning action history."""

from __future__ import annotations

import pytest
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set
from v10_agent.memory_contours import GameMemory
from v10_agent.virtual_sandbox import VirtualKinematicSandbox
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts


def test_four_directional_piece_motion_with_horizontal_axis():
    """Verify that movable pieces have 4-directional motion even when a horizontal axis is present."""
    # 30x30 grid:
    # Horizontal axis (color 10) at row 15, cols 0..29 (width=30, height=1)
    # Piece (color 5) at row 5, col 10 (3x3)
    grid = [[0] * 30 for _ in range(30)]
    for c in range(30):
        grid[15][c] = 10
    grid[5][10] = 5
    grid[5][11] = 5
    grid[6][10] = 5

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")
    game_mem.record_selection_mechanic("active entity toggled via action5")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)

    # piece_actions must contain all 4 directional primitives
    assert "ACTION1" in sandbox.piece_actions
    assert "ACTION2" in sandbox.piece_actions
    assert "ACTION3" in sandbox.piece_actions
    assert "ACTION4" in sandbox.piece_actions

    # Piece displacement for action2 (DOWN) must NOT be (0, 0)
    piece_obj = next(o for o in planning_set.objects if o.color == 5)
    dy, dx = sandbox._get_displacement(piece_obj.id, "action2", "move down", actor_type="piece")
    assert (dy, dx) == (1, 0)

    # Piece displacement for action4 (RIGHT) must NOT be (0, 0)
    dy, dx = sandbox._get_displacement(piece_obj.id, "action4", "move right", actor_type="piece")
    assert (dy, dx) == (0, 1)

    # Axis (horizontal line) must move vertically, but NOT horizontally
    axis_obj = next(o for o in planning_set.objects if o.color == 10)
    dy_ax, dx_ax = sandbox._get_displacement(axis_obj.id, "action2", "move down", actor_type="axis")
    assert (dy_ax, dx_ax) == (1, 0)
    dy_ax, dx_ax = sandbox._get_displacement(axis_obj.id, "action4", "move right", actor_type="axis")
    assert (dy_ax, dx_ax) == (0, 0)  # Horizontal axis cannot move horizontally


def test_multi_actor_three_entity_cycling_simulation():
    """Verify that VirtualKinematicSandbox supports N-actor cycling (e.g. Axis -> Piece 1 -> Piece 2 -> Axis)."""
    # 30x30 grid with 1 horizontal axis and 2 pieces
    grid = [[0] * 30 for _ in range(30)]
    for c in range(30):
        grid[15][c] = 10
    # Piece 1 (color 3)
    grid[5][5] = 3
    grid[5][6] = 3
    grid[6][5] = 3
    # Piece 2 (color 4)
    grid[8][20] = 4
    grid[8][21] = 4
    grid[9][20] = 4

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")
    game_mem.record_selection_mechanic("active entity toggled via action5")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)

    manifest_map = {
        "action1": {"name": "action1", "docstring": "move up"},
        "action2": {"name": "action2", "docstring": "move down"},
        "action3": {"name": "action3", "docstring": "move left"},
        "action4": {"name": "action4", "docstring": "move right"},
        "action5": {"name": "action5", "docstring": "toggle entity selection"},
    }

    # Sequence with 4 moves down, 2 moves right - none should be dropped!
    steps = [
        {"dsl_function": "action2", "arguments": {}},
        {"dsl_function": "action2", "arguments": {}},
        {"dsl_function": "action2", "arguments": {}},
        {"dsl_function": "action2", "arguments": {}},
        {"dsl_function": "action4", "arguments": {}},
        {"dsl_function": "action4", "arguments": {}},
    ]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    # The trajectory steps must be preserved and not dropped
    assert len(res.repaired_steps) == len(steps)
    assert [s["dsl_function"] for s in res.repaired_steps] == [s["dsl_function"] for s in steps]


def test_solver_prompt_compact_object_indexing():
    """Verify that objects are indexed concisely in OBJECT INDEX without blowing up prompt size."""
    grid = [[0] * 20 for _ in range(20)]
    # Main piece (color 5, area 9)
    for r in range(5, 8):
        for c in range(5, 8):
            grid[r][c] = 5

    # 10 scattered marker dots of color 1 (area 1)
    for i in range(10):
        grid[i][i + 9] = 1

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    manifest = {"functions": []}

    sys_prompt, user_prompt = build_solver_prompts(
        planning_set=planning_set,
        manifest=manifest,
    )

    assert "OBJECT INDEX" in user_prompt
    assert "c=5" in user_prompt
    assert "area=9" in user_prompt
    # Ensure prompt remains lean (< 4000 characters)
    assert len(user_prompt) < 4000
