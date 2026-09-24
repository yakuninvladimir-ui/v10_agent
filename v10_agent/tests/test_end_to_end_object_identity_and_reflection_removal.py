"""Unit tests for end-to-end unified object identification and elimination of intra-level failure reflection."""

import pytest
from typing import Any
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.explorer_agent import compute_probe_effect, PrimitiveProbeManager
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.explorer_prompt import build_explorer_synthesis_prompt, build_coordinate_hypothesis_prompt
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.session import GameSession


def test_compute_probe_effect_unified_object_identity():
    """Verify compute_probe_effect uses canonical aliases and explicit child relationships."""
    # 1. Create a grid with a parent object (c=5, 4x4) and 2 child holes (c=0, 1x1) inside it
    grid0 = [[10] * 10 for _ in range(10)]
    # Parent from row 2..5, col 2..5
    for r in range(2, 6):
        for c in range(2, 6):
            grid0[r][c] = 5
    # Two holes inside parent
    grid0[3][3] = 0
    grid0[4][4] = 0

    snap0 = extract_arga_snapshot(grid0)
    pset0 = build_planning_set(snap0, available_actions=["ACTION1", "ACTION2"])

    # Verify parent and children detected
    parent_obj = next(o for o in pset0.objects if o.color == 5)
    child_objs = [o for o in pset0.objects if o.color == 0]
    assert len(child_objs) == 2
    assert parent_obj.id in pset0.object_real_to_alias
    parent_alias = pset0.object_real_to_alias[parent_obj.id]
    child_aliases = [pset0.object_real_to_alias[c.id] for c in child_objs]

    # 2. Shift the parent and its holes UP by 2 rows (dy=-2)
    grid1 = [[10] * 10 for _ in range(10)]
    for r in range(0, 4):
        for c in range(2, 6):
            grid1[r][c] = 5
    grid1[1][3] = 0
    grid1[2][4] = 0

    effect_str = compute_probe_effect(snap0, {"grid": grid1}, planning_set=pset0)

    # Verify the effect explicitly mentions the parent alias, object ID, and its children
    assert parent_alias in effect_str
    assert parent_obj.id in effect_str
    for ca in child_aliases:
        assert ca in effect_str
    assert "UP" in effect_str
    # Must NOT use the old ambiguous '(compound, 2 parts)' string without aliases
    assert "(compound," not in effect_str


def test_prompt_builders_object_hierarchy():
    """Verify Explorer and Solver prompts display explicit parent-of and child-of relationships."""
    grid = [[10] * 10 for _ in range(10)]
    for r in range(2, 6):
        for c in range(2, 6):
            grid[r][c] = 5
    grid[3][3] = 0
    grid[4][4] = 0

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2"])

    parent_obj = next(o for o in pset.objects if o.color == 5)
    parent_alias = pset.object_real_to_alias[parent_obj.id]

    # 1. Explorer synthesis prompt
    _, exp_user = build_explorer_synthesis_prompt(pset)
    assert f"parent of:" in exp_user
    assert f"child of: {parent_alias}" in exp_user

    # 2. Solver prompt
    _, sol_user = build_solver_prompts(manifest={"functions": []}, planning_set=pset)
    assert f"parent of:" in sol_user
    assert f"child of: {parent_alias}" in sol_user

    # 3. Explorer coordinate hypothesis prompt
    _, coord_user = build_coordinate_hypothesis_prompt(pset)
    assert "children_ids" in coord_user or "children_aliases" in coord_user
    assert "parent_id" in coord_user or "parent_alias" in coord_user


def test_no_failure_reflection_during_session_replanning():
    """Verify GameSession does NOT trigger reflect_and_revise_on_failure during replan after candidate failure."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    session = GameSession(config=config, advisor=advisor)

    # Simulate attempt failure
    session._last_attempt_failed = True
    session._last_failure_reason = "Step produced NULL verdict"
    session._last_failure_summary = "Candidate traj_01 failed at step 4"
    session.level_chain_attempts = 1

    gm = session.memory_manager.get_game_memory("session")
    initial_invariants_count = len(gm.structured_invariants)

    # Mock response for solver trajectory planning only (NO solver_reflection provided)
    advisor.set_response(
        "solver",
        "<trajectory_1>\n1. action1()\n</trajectory_1>",
    )

    grid = [[0] * 8 for _ in range(8)]
    grid[2][2] = 3
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])
    from v10_agent.sandbox import SandboxExecutor
    manifest = {"schema_version": "v10.dsl_manifest.1", "functions": [{"name": "action1", "parameters": []}]}
    source = "def action1(api):\n    return api.declare_environment_action('ACTION1')\n"
    session.level_initial_grid_hash = pset.grid_hash
    session.active_manifest = manifest
    session.active_module = SandboxExecutor().load_module(source, manifest)

    obs = {"state": "NOT_FINISHED", "grid": grid, "available_actions": [1]}
    session.act(obs)

    # No solver_reflection was called, no invariants added or revised
    assert session._last_attempt_failed is False
    assert len(gm.structured_invariants) == initial_invariants_count
