"""Unit tests for Cyclic Combinatorial Probing, Solver Action Quarantine, and Coordinate Affordance Suppression."""

from __future__ import annotations

import pytest
from v10_agent.explorer_agent import PrimitiveProbeManager, compute_probe_effect
from v10_agent.memory_contours import GameMemory, EnvironmentSpecMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.types import ActionDeclaration


def test_cyclic_combinatorial_chaining():
    """Verify combinatorial chaining pairs effective motion prefixes with candidate inactive actions."""
    mgr = PrimitiveProbeManager(max_probes=16)
    allowed = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    
    # 1. Initial sweep
    sweep = mgr.plan_discrete_probes(allowed)
    assert [p.action_id for p in sweep] == allowed
    
    # 2. Simulate sweep execution: 1, 2 inactive; 3 (LEFT), 4 (RIGHT), 5 (STAMP) effective
    mgr.record_probe_result("ACTION1", {}, None, {"grid": None})  # None grid -> no_grid -> inactive
    mgr.record_probe_result("ACTION2", {}, None, {"grid": None})
    mgr.confirmed_effective_actions["ACTION3"] = "moved obj_10 by dy=1, dx=-9 (LEFT)"
    mgr.confirmed_effective_actions["ACTION4"] = "moved obj_10 by dy=-1, dx=9 (RIGHT)"
    mgr.confirmed_effective_actions["ACTION5"] = "color transition (stamp/draw): 50 cells changed color"
    
    assert mgr.inactive_actions == {"ACTION1", "ACTION2"}
    assert mgr.is_motion_action("ACTION3") is True
    assert mgr.is_motion_action("ACTION4") is True
    assert mgr.is_motion_action("ACTION5") is False
    
    # 3. First combinatorial chain: motion prefix ACTION3 + remaining inactive actions [1, 2]
    chain1 = mgr.get_next_combinatorial_chain()
    assert [p.action_id for p in chain1] == ["ACTION3", "ACTION1", "ACTION2"]
    
    # 4. Simulate chain1 execution: ACTION2 becomes effective (DOWN), ACTION1 still inactive
    mgr.confirmed_effective_actions["ACTION2"] = "moved obj_10 by dy=9, dx=-1 (DOWN)"
    mgr.inactive_actions.discard("ACTION2")
    assert mgr.inactive_actions == {"ACTION1"}
    
    # 5. Next combinatorial chain: next motion prefix + remaining inactive action [1]
    chain2 = mgr.get_next_combinatorial_chain()
    assert chain2[0].action_id in ("ACTION3", "ACTION4", "ACTION2")
    assert chain2[-1].action_id == "ACTION1"
    
    # 6. Simulate ACTION1 becoming effective (UP)
    mgr.confirmed_effective_actions["ACTION1"] = "moved obj_10 by dy=-9, dx=1 (UP)"
    mgr.inactive_actions.discard("ACTION1")
    
    # 7. No inactive actions remain -> chain returns empty, probing finishes
    chain_end = mgr.get_next_combinatorial_chain()
    assert chain_end == []


def test_solver_prompt_suppresses_unconfirmed_coordinate_affordances():
    """Verify salient_coordinate_affordances are strictly suppressed if ACTION6 is unconfirmed."""
    grid = [[0, 1], [2, 3]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION6", "RESET"])
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action6", "parameters": [{"name": "x"}, {"name": "y"}]},
        ]
    }
    
    # Case A: ACTION6 is unconfirmed in GameMemory
    game_mem_unconfirmed = GameMemory(game_id="test_game")
    game_mem_unconfirmed.record_action_effect("ACTION1", "moved tool (UP)")
    game_mem_unconfirmed.record_unconfirmed_action("ACTION6", "zero observable effect")
    
    sys_prompt, user_prompt = build_solver_prompts(
        manifest=manifest,
        planning_set=planning_set,
        game_memory=game_mem_unconfirmed,
    )
    
    assert "salient_coordinate_affordances" not in user_prompt
    assert "FALSIFIED - do not rely on:" in user_prompt
    assert "ACTION6: unconfirmed/inactive" in user_prompt
    
    # Case B: ACTION6 is confirmed active
    game_mem_confirmed = GameMemory(game_id="test_game")
    game_mem_confirmed.record_action_effect("ACTION1", "moved tool (UP)")
    game_mem_confirmed.record_action_effect("ACTION6", "clicked entity toggled state")
    
    sys_prompt_b, user_prompt_b = build_solver_prompts(
        manifest=manifest,
        planning_set=planning_set,
        game_memory=game_mem_confirmed,
    )
    
    assert "ACTION6: clicked entity toggled state" in user_prompt_b
    assert "ACTION6: unconfirmed/inactive" not in user_prompt_b


def test_game_memory_unconfirmed_action_lifecycle():
    """Verify unconfirmed actions are removed from unconfirmed when confirmed later."""
    gm = GameMemory(game_id="test_game")
    gm.record_unconfirmed_action("ACTION2", "conditional action (inactive at boundary)")
    assert "ACTION2" in gm.unconfirmed_actions
    
    gm.record_action_effect("ACTION2", "moved obj_10 (DOWN)")
    assert "ACTION2" in gm.confirmed_action_effects
    assert "ACTION2" not in gm.unconfirmed_actions
    
    gm.clear()
    assert len(gm.confirmed_action_effects) == 0
    assert len(gm.unconfirmed_actions) == 0


def test_ranked_multi_hypothesis_detector_compound_motion():
    """Verify Ranked Multi-Hypothesis Detector recognizes compound body translations and diagonal directions."""
    from v10_agent.explorer_agent import describe_vector, compute_probe_effect, PrimitiveProbeManager

    # 1. Verify lazy vector descriptions
    assert describe_vector(0, 0) == "stationary"
    assert describe_vector(-3, 0) == "UP"
    assert describe_vector(3, 0) == "DOWN"
    assert describe_vector(0, -3) == "LEFT"
    assert describe_vector(0, 3) == "RIGHT"
    assert describe_vector(2, 3) == "DOWN+RIGHT"
    assert describe_vector(-2, -1) == "UP+LEFT"

    # 2. Grid with outer container (color 5) and inner dot (color 1)
    grid_before = [[0 for _ in range(12)] for _ in range(12)]
    for r in range(2, 6):
        for c in range(2, 6):
            grid_before[r][c] = 5
    grid_before[3][3] = 1

    snap_before = extract_arga_snapshot(grid_before)
    parent = next(o for o in snap_before.objects if o.color == 5)
    dot = next(o for o in snap_before.objects if o.color == 1)
    assert dot.id in parent.children_ids
    assert dot.parent_id == parent.id

    # Diagonal shift by dy=+2, dx=+3
    grid_after = [[0 for _ in range(12)] for _ in range(12)]
    for r in range(4, 8):
        for c in range(5, 9):
            grid_after[r][c] = 5
    grid_after[5][6] = 1

    effect = compute_probe_effect(snap_before, {"grid": grid_after})
    assert "moved" in effect
    assert "compound" in effect
    assert "dy=+2, dx=+3" in effect
    assert "DOWN+RIGHT" in effect

    mgr = PrimitiveProbeManager()
    mgr.confirmed_effective_actions["ACTION1"] = effect
    assert mgr.is_motion_action("ACTION1") is True

