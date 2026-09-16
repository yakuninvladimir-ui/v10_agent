"""Unit tests for Solver two-turn reflection and invariant distillation on level win."""

from typing import Any
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import SolverAgent


def test_solver_two_turn_reflection_workflow():
    """Verify that Solver preserves conversation history from Turn 1 and distills clean invariants in Turn 2."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    solver = SolverAgent(config, advisor)

    # Mock PlanningSet & Manifest
    grid = [[0, 1, 0], [0, 0, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot,
        available_actions=["ACTION1", "ACTION2", "ACTION5", "RESET"],
    )
    manifest = {
        "schema_version": "v10.dsl_manifest.1",
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action5", "parameters": []},
        ],
    }

    # Turn 1: Solver generates trajectory package
    advisor.set_response(
        "solver",
        "<invariant_analysis>\n"
        "Hypothesis: Align horizontal mirror axis to reflect piece into upper targets\n"
        "</invariant_analysis>\n"
        "<trajectory_1>\n"
        "1. action1()\n"
        "2. action5()\n"
        "</trajectory_1>",
    )

    pkg = solver.generate_trajectory_package(
        manifest=manifest,
        planning_set=planning_set,
        budget=20,
    )
    assert pkg is not None
    assert solver.last_conversation_messages is not None
    assert len(solver.last_conversation_messages) >= 2
    assert "Align horizontal mirror axis" in solver.last_hypothesis

    # Turn 2: Simulate level win reflection
    advisor.set_response(
        "solver_reflection",
        "<distilled_invariants>\n"
        "- [PALETTE & ROLES]: Color 11 represents static target sockets; Color 5 represents movable piece entities; Color 10 represents the reflection axis.\n"
        "- [GOAL]: Level goal is satisfied when targets are covered by pieces and their mirror reflections.\n"
        "- [ENTITIES & MECHANICS]: Movable axis line acts as a reflection plane; pieces move independently into symmetric target sockets.\n"
        "- [CONTROL]: ACTION5 sequentially shifts active focus across controllable entities.\n"
        "- Bad rule that should be filtered: action1() -> action5() -> action2()\n"
        "- Another bad coordinate rule: dx=7, dy=-7 at row=16\n"
        "</distilled_invariants>",
    )

    invariants = solver.distill_level_win_invariants(
        winning_candidate=pkg["candidates"][0],
        execution_summary="action1 executed 7 times, action5 executed once",
    )

    assert len(invariants) == 4
    assert any("[PALETTE & ROLES]" in inv for inv in invariants)
    assert any("[GOAL]" in inv for inv in invariants)
    assert any("[ENTITIES & MECHANICS]" in inv for inv in invariants)
    assert any("[CONTROL]" in inv for inv in invariants)
    # Ensure poisoned action macros and raw coordinates are filtered out
    assert not any("action1() ->" in inv for inv in invariants)
    assert not any("dx=" in inv for inv in invariants)

    # Verify GameMemory integration
    game_mem = GameMemory(game_id="ar25")
    for inv in invariants:
        game_mem.record_stratified_invariant(inv, tier=3)

    game_mem.record_level_solution(
        level_id="level_1",
        setup_summary="3 entities",
        invariant_rule=" | ".join(invariants),
        winning_macro="",  # Omitted
    )

    curriculum_ctx = game_mem.format_curriculum_context()
    assert "CURRICULUM INVARIANTS" in curriculum_ctx
    assert "level_1" in curriculum_ctx
    assert "[PALETTE & ROLES]" in curriculum_ctx
    assert "[GOAL]" in curriculum_ctx
    assert "Winning Macro: action" not in curriculum_ctx

    stratified_ctx = game_mem.format_stratified_context()
    assert "TIER 3: HIGH-LEVEL DEDUCED RULES & STRATEGIES" in stratified_ctx
    assert "[ENTITIES & MECHANICS]" in stratified_ctx
    assert "action1() -> action5()" not in stratified_ctx
