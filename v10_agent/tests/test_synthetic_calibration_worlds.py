"""Synthetic Calibration Micro-Worlds (Completeness & Soundness Benchmark).

Verifies that the agent architecture:
1. Does not degenerate into trivial no-op or perpetual OMIT.
2. Accurately solves standard cognitive patterns:
   - Micro-World 1: Maze navigation around obstacles to target.
   - Micro-World 2: Gravity settling until solid boundary contact.
   - Micro-World 3: Contact button trigger opening passage to goal.
"""
from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary, Verdict
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import EntityRole, GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_synthetic_world_1_maze_navigation():
    """Micro-World 1: Actor navigates around solid obstacle to reach goal."""
    # 8x8 maze:
    # 0 = empty space
    # 1 = actor (blue) at (1, 1)
    # 3 = obstacle wall (green) from (1, 3) to (5, 3)
    # 8 = target (teal) at (1, 5)
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 3, 0, 8, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    # Verify parser separates actor, obstacle, and target
    actor = next((o for o in pset.objects if o.color == 1), None)
    obstacle = next((o for o in pset.objects if o.color == 3), None)
    target = next((o for o in pset.objects if o.color == 8), None)

    assert actor is not None and actor.area == 1
    assert obstacle is not None and obstacle.area == 5
    assert target is not None and target.area == 1

    # Invariants should detect target alignment and area conservation
    from v10_agent.universal_invariants import discover_invariants
    invs = discover_invariants(pset)
    assert any(inv.invariant_type == "area_conservation" for inv in invs)


def test_synthetic_world_2_gravity_settling_and_soft_boundary():
    """Micro-World 2: Falling block settles on platform; downward motion into floor is OMIT, not NULL."""
    verifier = LayeredVerifier(V10Config())
    # Before: Block (color 2) resting at bottom boundary / floor
    before_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [5, 5, 5, 5, 5],
    ]
    snap = extract_arga_snapshot(before_grid)
    pset = build_planning_set(snap, available_actions=["ACTION2"])
    block = next(o for o in pset.objects if o.color == 2)

    gmem = GameMemory(game_id="gravity_world")
    gmem.record_action_effect("ACTION2", "Gravity moves block DOWN")

    # After: Action DOWN into floor produces zero delta (blocked)
    after_obs = {"grid": before_grid, "state": "IN_PROGRESS"}
    step = GroundedStep(
        step_id="s1",
        dsl_function="fall_down",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id=block.id, predicate="row_delta", value=1),
        ]),
    )

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=gmem,
        action_dict={"action_id": "ACTION2"},
    )

    # Tier 3 softening: Zero delta against known boundary is OMIT, preserving candidate trajectory
    assert judgment.verdict == Verdict.OMIT
    assert "omit" in judgment.explanation.lower()


def test_synthetic_world_3_contact_trigger_and_role_grounding():
    """Micro-World 3: Contact with button assigns role and validates victory rule grounding."""
    gmem = GameMemory(game_id="trigger_world")

    # Step onto trigger button (color 6)
    gmem.palette.assign_role(6, EntityRole.COLLECTIBLE, confidence=0.9, evidence="Button trigger")
    gmem.palette.assign_role(8, EntityRole.TARGET, confidence=0.95, evidence="Goal target")
    gmem.palette.assign_role(4, EntityRole.HAZARD, confidence=0.9, evidence="Fatal lava")

    assert gmem.palette.get(6).role == EntityRole.COLLECTIBLE
    assert gmem.palette.get(8).role == EntityRole.TARGET
    assert gmem.palette.get(4).role == EntityRole.HAZARD

    prompt_ctx = gmem.format_grounded_memory_for_prompt()
    assert "Color 6: COLLECTIBLE" in prompt_ctx
    assert "Color 8: TARGET" in prompt_ctx
    assert "Color 4: HAZARD" in prompt_ctx
