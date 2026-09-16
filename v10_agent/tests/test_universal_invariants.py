"""Unit tests for Universal Invariant Module and Horizontal/Vertical Sandbox Simulation."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.memory_contours import GameMemory
from v10_agent.observe import grid_to_hex_rows
from v10_agent.planning_set import build_planning_set
from v10_agent.universal_invariants import discover_invariants, compute_invariant_distance
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_horizontal_axis_reflection_simulation():
    """Verify that horizontal axis (like in ar25 Level 2) simulates vertical reflection correctly."""
    # 30x30 grid:
    # Piece (color 1) at row 5, col 10
    # Horizontal Axis (color 2) at row 15, cols 0..29 (width=30, height=1)
    # Target Socket (color 8) at row 25, col 10 (refl row: 2*15 - 5 = 25!)
    grid = [[0] * 30 for _ in range(30)]
    # Piece
    grid[5][10] = 1
    grid[5][11] = 1
    grid[6][10] = 1
    # Horizontal Axis
    for c in range(30):
        grid[15][c] = 2
    # Target Socket
    grid[25][10] = 8
    grid[25][11] = 8
    grid[26][10] = 8

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"], grid_to_hex_rows(grid)
    )

    invariants = discover_invariants(planning_set)
    assert len(invariants) > 0
    h_invariants = [inv for inv in invariants if inv.invariant_type == "axial_symmetry_horizontal"]
    assert len(h_invariants) > 0
    assert h_invariants[0].axis_coordinate == pytest.approx(15.0, abs=1.0)

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action2", "moved DOWN by dy=1, dx=0")
    game_mem.record_action_effect("action3", "moved LEFT by dy=0, dx=-1")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest_map = {
        "step_up": {"name": "step_up", "docstring": "shift axis up"},
        "step_right": {"name": "step_right", "docstring": "shift piece right"},
        "toggle": {"name": "toggle", "docstring": "toggle entity control"},
    }

    # Since piece is at row 5 and axis is at row 15, reflection row is 2*15 - 5 = 25 (matches socket row 25!).
    # If the solver proposed moving the axis UP (step_up), it would overshoot and increase distance!
    # Sandbox should clamp the overshoot step and achieve goal alignment!
    steps = [
        {"step_id": "s1", "dsl_function": "step_up", "arguments": {}},  # overshoot step for axis
        {"step_id": "s2", "dsl_function": "toggle", "arguments": {}},
        {"step_id": "s3", "dsl_function": "step_right", "arguments": {}}, # redundant trailing step
    ]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict in ("APPROVED", "REPAIRED")
    assert res.goal_reached is True
    assert res.active_invariant.invariant_type == "axial_symmetry_horizontal"


def test_socket_coverage_translation_simulation():
    """Verify that socket coverage (like in wa30) minimizes translation distance."""
    # 20x20 grid:
    # Piece/Box at row 2, col 2
    # Socket at row 2, col 6
    grid = [[0] * 20 for _ in range(20)]
    grid[2][2] = 3
    grid[2][3] = 3
    grid[3][2] = 3

    grid[2][6] = 5
    grid[2][7] = 5
    grid[3][6] = 5

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"], grid_to_hex_rows(grid)
    )

    invariants = discover_invariants(planning_set)
    assert len(invariants) > 0
    socket_invs = [inv for inv in invariants if inv.invariant_type == "socket_coverage"]
    assert len(socket_invs) > 0

    sandbox = VirtualKinematicSandbox(planning_set)
    manifest_map = {
        "move_right": {"name": "move_right", "docstring": "move piece right"},
    }

    # Moving right 4 times covers the socket!
    steps = [
        {"step_id": "s1", "dsl_function": "move_right", "arguments": {}},
        {"step_id": "s2", "dsl_function": "move_right", "arguments": {}},
        {"step_id": "s3", "dsl_function": "move_right", "arguments": {}},
        {"step_id": "s4", "dsl_function": "move_right", "arguments": {}},
        {"step_id": "s5", "dsl_function": "move_right", "arguments": {}},  # trailing step
        {"step_id": "s6", "dsl_function": "move_right", "arguments": {}},  # trailing step
    ]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest_map)
    assert res.verdict in ("APPROVED", "REPAIRED")
    assert res.goal_reached is True
    assert len(res.repaired_steps) == 4  # trimmed trailing steps 5 and 6!
