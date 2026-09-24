"""Tests for VirtualKinematicSandbox, Two-Tier Memory, and Text Trajectory Parsing."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.memory_contours import EpistemicMemory, GameMemory
from v10_agent.observe import grid_to_hex_rows
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import parse_text_trajectory
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_parse_text_trajectory():
    """Verify that structured text with [HYPOTHESIS] and [TRAJECTORY] parses correctly."""
    manifest_funcs = {"move_left", "move_down", "toggle_piece"}
    grid = [[0] * 10 for _ in range(10)]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION2"], grid_to_hex_rows(grid))

    text = """
Some reasoning before...
[HYPOTHESIS]
Actor: obj_piece
Target: obj_socket
Transform: reflection across vertical mirror axis
Strategy: Phase 1 align axis X, Phase 2 toggle, Phase 3 align piece Y

[TRAJECTORY]
1. move_left(count=2)
2. toggle_piece()
3. move_down(count=3)
"""
    pkg = parse_text_trajectory(text, manifest_funcs, planning_set)
    assert pkg is not None
    assert pkg["schema_version"] == "v10.trajectory_package.1"
    assert "reflection across vertical mirror axis" in pkg["hypothesis"]
    assert len(pkg["candidates"]) == 1

    steps = pkg["candidates"][0]["steps"]
    assert len(steps) == 6  # 2 left + 1 toggle + 3 down
    assert steps[0]["dsl_function"] == "move_left"
    assert steps[1]["dsl_function"] == "move_left"
    assert steps[2]["dsl_function"] == "toggle_piece"
    assert steps[3]["dsl_function"] == "move_down"
    assert steps[4]["dsl_function"] == "move_down"
    assert steps[5]["dsl_function"] == "move_down"


def test_two_tier_memory_curriculum_and_scratchpad():
    """Verify that GameMemory records curriculum solution patterns and EpistemicMemory records scratchpad attempts."""
    game_mem = GameMemory(game_id="ar25")
    game_mem.record_level_solution(
        level_id="level_0",
        setup_summary="1 piece, 1 socket, 1 axis",
        invariant_rule="Piece reflects into socket",
        winning_macro="Align axis -> toggle -> align piece",
    )
    curriculum = game_mem.format_curriculum_context()
    assert "CURRICULUM INVARIANTS" in curriculum
    assert "level_0" in curriculum
    assert "Piece reflects into socket" in curriculum

    # Scratchpad in EpistemicMemory
    ep_mem = EpistemicMemory(level_id="level_1")
    ep_mem.record_attempt_feedback(
        hypothesis="H1",
        trajectory_summary="move_left, move_left, move_left",
        status="rejected_in_sandbox",
        reason="Overshoots target column",
    )
    scratchpad = ep_mem.format_scratchpad_context()
    assert "CURRENT LEVEL FAILED ATTEMPTS" in scratchpad
    assert "Overshoots target column" in scratchpad

    # Clear on level transition
    ep_mem.clear()
    assert len(ep_mem.current_level_attempts) == 0
    assert ep_mem.format_scratchpad_context() == ""


def test_virtual_kinematic_sandbox_overshoot_clamping():
    """Verify that VirtualKinematicSandbox clamps horizontal overshoots and trims trailing steps."""
    # Build 24x24 grid with:
    # 1. Piece (color 1) at row 5..6, col 5..6
    # 2. Axis (color 2) at col 14, rows 0..23
    # 3. Socket (color 8) at row 10..11, col 23..24 (refl col: 2*14 - 5 = 23!)
    grid = [[0] * 25 for _ in range(25)]
    # Piece
    grid[5][5] = 1
    grid[5][6] = 1
    grid[6][5] = 1
    # Axis (vertical line spanning height)
    for r in range(25):
        grid[r][14] = 2
    # Target socket
    grid[10][23] = 8
    grid[10][24] = 8
    grid[11][23] = 8

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"], grid_to_hex_rows(grid)
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest_map = {
        "step_left": {"name": "step_left", "docstring": "shift axis left"},
        "step_down": {"name": "step_down", "docstring": "shift piece down"},
        "toggle": {"name": "toggle", "docstring": "toggle entity control"},
    }

    # Since piece is at col 5 and axis is at col 14:
    # Reflection column: 2 * 14 - 5 = 23 (matches socket col 23!).
    # If the solver proposed shifting left, it would overshoot!
    steps = [
        {"step_id": "s1", "dsl_function": "step_left", "arguments": {}},
        {"step_id": "s2", "dsl_function": "toggle", "arguments": {}},
        {"step_id": "s3", "dsl_function": "step_down", "arguments": {}},
        {"step_id": "s4", "dsl_function": "step_down", "arguments": {}},
        {"step_id": "s5", "dsl_function": "step_down", "arguments": {}},
        {"step_id": "s6", "dsl_function": "step_down", "arguments": {}},
        {"step_id": "s7", "dsl_function": "step_down", "arguments": {}},
        {"step_id": "s8", "dsl_function": "step_down", "arguments": {}},  # redundant step
    ]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict in ("APPROVED", "REPAIRED")
    assert res.goal_reached is True
    fns = [s["dsl_function"] for s in res.repaired_steps]
    assert "step_left" not in fns  # Overshoot clamped out!
    assert "toggle" in fns
    assert "step_down" in fns


def test_compound_axis_and_piece_selection_simulation():
    """Verify that sandbox models empirical initial entity selection and compound alignment."""
    # 30x30 grid:
    # Axis (color 10) at col 18, rows 0..29 (vertical line)
    # Piece (color 5) at row 5, col 24
    # Socket (color 8) at row 15, col 6
    # Target midpoint col = (24 + 6) / 2 = 15.
    # Current axis is at col 18.
    # 1. Shift axis left by 3 pixels -> col 15!
    # 2. Toggle control to piece.
    # 3. Shift piece down by 10 pixels (row 5 -> row 15).
    # Then reflection of piece at col 24 across axis col 15 lands at 2*15 - 24 = 6, row 15!
    grid = [[0] * 30 for _ in range(30)]
    for r in range(30):
        grid[r][18] = 10
    grid[5][23] = 5
    grid[5][24] = 5
    grid[6][23] = 5
    grid[15][6] = 8
    grid[15][7] = 8
    grid[16][7] = 8

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"], grid_to_hex_rows(grid)
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")
    # Axis object id
    axis_obj = next(o for o in planning_set.objects if o.height >= 10 and o.width <= 4)
    piece_obj = next(o for o in planning_set.objects if o.color == 5)
    game_mem.record_selection_mechanic(
        f"ACTION5: selection indicator transferred: internal dots moved from {axis_obj.id} (color 10) to {piece_obj.id} (color 5) (active entity toggled)"
    )

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest_map = {
        "shift_left": {"name": "shift_left", "docstring": "shift entity left"},
        "shift_down": {"name": "shift_down", "docstring": "shift entity down"},
        "toggle_control": {"name": "toggle_control", "docstring": "toggle active entity control / selection"},
    }

    # 3 steps left for axis (18 -> 15), 1 toggle, 10 steps down for piece (5 -> 15)
    steps = [
        {"step_id": "s1", "dsl_function": "shift_left", "arguments": {}},
        {"step_id": "s2", "dsl_function": "shift_left", "arguments": {}},
        {"step_id": "s3", "dsl_function": "shift_left", "arguments": {}},
        {"step_id": "s4", "dsl_function": "toggle_control", "arguments": {}},
    ] + [{"step_id": f"s{5+i}", "dsl_function": "shift_down", "arguments": {}} for i in range(10)]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict in ("APPROVED", "REPAIRED")
    assert res.goal_reached is True
    assert res.active_invariant is not None
    assert res.active_invariant.invariant_type == "axial_symmetry_vertical"


def test_synthesize_invariant_trajectory_and_fallback():
    """Verify that VirtualKinematicSandbox autonomously synthesizes a valid trajectory for unsatisfied invariants."""
    grid = [[0] * 30 for _ in range(30)]
    # Piece
    grid[5][5] = 5
    grid[5][6] = 5
    grid[6][5] = 5
    # Axis at col 18
    for r in range(30):
        grid[r][18] = 10
    # Socket at (15, 25)
    grid[15][24] = 4
    grid[15][25] = 4
    grid[16][25] = 4

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"], grid_to_hex_rows(grid)
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")
    axis_obj = next(o for o in planning_set.objects if o.height >= 10 and o.width <= 4)
    piece_obj = next(o for o in planning_set.objects if o.color == 5)
    game_mem.record_selection_mechanic(
        f"ACTION5: selection indicator transferred: internal dots moved from {axis_obj.id} (color 10) to {piece_obj.id} (color 5) (active entity toggled)"
    )

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest_map = {
        "action1": {"name": "action1", "docstring": "shift entity up"},
        "action2": {"name": "action2", "docstring": "shift entity down"},
        "action3": {"name": "action3", "docstring": "shift entity left"},
        "action4": {"name": "action4", "docstring": "shift entity right"},
        "action5": {"name": "action5", "docstring": "toggle entity control"},
    }

    inv = next(i for i in sandbox.invariants if i.invariant_type == "axial_symmetry_vertical")
    steps = sandbox.synthesize_invariant_trajectory(inv, manifest_map)
    assert steps is not None
    assert len(steps) > 0
    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict in ("APPROVED", "REPAIRED")
    assert res.goal_reached is True


def test_in_grid_divider_or_axis_does_not_reject_trajectory():
    """Verify that moving towards or across an in-grid divider or axis does not cause a false collision rejection."""
    grid = [[0] * 30 for _ in range(30)]
    # Piece at cols 10..12
    grid[5][10] = 5
    grid[5][11] = 5
    grid[6][10] = 5
    # Central divider/axis at col 18
    for r in range(30):
        grid[r][18] = 10
    # Symmetric target at cols 24..26
    grid[5][24] = 4
    grid[5][25] = 4
    grid[6][25] = 4

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"], grid_to_hex_rows(grid)
    )

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=3")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest_map = {
        "action4": {"name": "action4", "docstring": "shift entity right"},
    }

    # Propose 3 steps of action4 moving right (10 -> 13 -> 16 -> 19, crossing col 18)
    steps = [
        {"dsl_function": "action4", "arguments": {}},
        {"dsl_function": "action4", "arguments": {}},
        {"dsl_function": "action4", "arguments": {}},
    ]
    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict != "REJECTED"
    assert res.has_boundary_violation is False
