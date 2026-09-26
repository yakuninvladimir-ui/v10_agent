"""Tests for Stage 3: Virtual Sandbox Generalization & Prompt Hygiene.

Verifies:
1. VirtualKinematicSandbox does NOT ignore collisions with elongated obstacles (aspect >= 3.0).
2. Elimination of hardcoded aspect ratio checks across actor selection, kinematics, and collisions.
3. Solver prompt does not display unconfirmed invariants (confidence <= 0.0).
4. Execution loop facts accurately represent Brusentsov ternary verdicts without false cert claims.
"""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.memory_contours import GameMemory, StructuredInvariant, Ternary
from v10_agent.observe import grid_to_hex_rows
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.solver_prompt import (
    SOLVER_SYSTEM_PROMPT,
    build_solver_prompts,
)
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_sandbox_detects_collision_with_elongated_obstacles():
    """Verify that elongated obstacles (aspect >= 3.0) trigger collisions and are not bypassed."""
    # 20x20 grid with:
    # 1. Piece (color 1) at (5, 5)
    # 2. Elongated vertical wall (color 3, height=10, width=1, aspect=10.0) at col 7, rows 2..11
    # 3. Target (color 8) at (5, 12)
    grid = [[0] * 20 for _ in range(20)]
    grid[5][5] = 1
    for r in range(2, 12):
        grid[r][7] = 3
    grid[5][12] = 8

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"], grid_to_hex_rows(grid))

    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    manifest = {
        "step_right": {"name": "step_right", "docstring": "move right"},
    }

    piece = [o for o in planning_set.objects if o.color == 1][0]
    # Moving right 4 times from col 5 reaches col 9, crashing through wall at col 7
    steps = [
        {"step_id": "s1", "dsl_function": "step_right", "arguments": {"obj": piece.id}},
        {"step_id": "s2", "dsl_function": "step_right", "arguments": {"obj": piece.id}},
        {"step_id": "s3", "dsl_function": "step_right", "arguments": {"obj": piece.id}},
        {"step_id": "s4", "dsl_function": "step_right", "arguments": {"obj": piece.id}},
    ]

    res = sandbox.evaluate_and_repair_trajectory(steps, manifest)
    assert res.has_collision is True
    assert res.collision_object_id != ""


def test_sandbox_unconstrained_elongated_piece_moves_freely():
    """Verify that elongated pieces without symmetry constraints move freely in both axes."""
    # 15x15 grid with a 6x2 piece (aspect = 3.0)
    grid = [[0] * 15 for _ in range(15)]
    for r in range(3, 9):
        grid[r][3] = 1
        grid[r][4] = 1

    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION4"], grid_to_hex_rows(grid))

    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("action1", "moved UP by dy=-1, dx=0")
    game_mem.record_action_effect("action4", "moved RIGHT by dy=0, dx=1")

    sandbox = VirtualKinematicSandbox(planning_set, game_mem)
    piece = planning_set.objects[0]

    # Verify both vertical and horizontal displacements resolve naturally
    dy_up, dx_up = sandbox._get_displacement(piece.id, "action1", "move up", actor_type="piece")
    assert dy_up == -1
    assert dx_up == 0

    dy_rt, dx_rt = sandbox._get_displacement(piece.id, "action4", "move right", actor_type="piece")
    assert dy_rt == 0
    assert dx_rt == 1


def test_solver_prompt_zero_confidence_structured_invariants_filtered():
    """Verify that structured invariants with confidence <= 0.0 are not shown in knowledge tiers."""
    grid = [[0, 1], [0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])

    gm = GameMemory(game_id="test_game")
    # Add confirmed invariant (conf > 0)
    gm.record_stratified_invariant("Movable blocks slide horizontally", tier=1, invariant_type="kinematics", confidence=0.8)
    # Add unconfirmed hypothesis (conf = 0.0)
    gm.structured_invariants.append(StructuredInvariant(
        invariant_id="zero_conf_inv",
        tier=2,
        invariant_type="hypothesis",
        description="Unverified hypothesis that should not appear",
        ternary_status=Ternary.TRUE,
        confidence=0.0,
    ))

    _, user_prompt = build_solver_prompts(
        planning_set=pset,
        game_memory=gm,
        manifest={"functions": [{"name": "action1", "parameters": []}]},
        has_image=False,
    )

    assert "Movable blocks slide horizontally" in user_prompt
    assert "Unverified hypothesis that should not appear" not in user_prompt


def test_solver_system_prompt_ternary_loop_facts_purity():
    """Verify execution loop facts are accurate and free from deceptive certification claims."""
    assert "Empty EXPECT (len==0) -> verdict IRRELEVANT/OMIT" in SOLVER_SYSTEM_PROMPT
    assert "Brusentsov x'y'" in SOLVER_SYSTEM_PROMPT
    assert "weakly certified" not in SOLVER_SYSTEM_PROMPT
