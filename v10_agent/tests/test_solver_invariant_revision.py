"""Unit tests for mandatory Solver invariant revision on win and defeat."""

from typing import Any
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import GameMemory, StructuredInvariant, Ternary
from v10_agent.planning_set import build_planning_set
from v10_agent.session import GameSession, SessionPhase
from v10_agent.solver_agent import SolverAgent


def test_solver_revise_invariants_win_flow():
    """Verify Solver receives active/invalidated rules and revises invariants upon level win."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    solver = SolverAgent(config, advisor)

    advisor.set_response(
        "solver_reflection",
        "<revised_invariants>\n"
        "- [PHYSICS]: Sliding entities preserve rigid form across transitions\n"
        "- [STRUCTURE]: Symmetric sockets indicate dual placement requirement\n"
        "- [GOAL]: All socket apertures must be occupied simultaneously\n"
        "- Bad rule to filter: action1() -> action2()\n"
        "- Bad coordinate: dx=4, dy=-2\n"
        "</revised_invariants>",
    )

    active_invs = [
        {
            "rule": "[PHYSICS]: Movable pieces slide horizontally",
            "inclusion_reason": "Observed on level_0 and level_1",
            "confidence": 0.9,
        }
    ]
    invalidated_invs = [
        {
            "rule": "[CONSTRAINTS]: Upward motion is unrestricted",
            "invalidation_reason": "Zero delta on top boundary collision",
        }
    ]

    revised = solver.revise_invariants(
        outcome="WIN",
        execution_summary="6 actions executed successfully to win",
        active_invariants=active_invs,
        invalidated_invariants=invalidated_invs,
    )

    assert len(revised) == 3
    assert any("[PHYSICS]" in r for r in revised)
    assert any("[STRUCTURE]" in r for r in revised)
    assert any("[GOAL]" in r for r in revised)
    assert not any("action1" in r for r in revised)
    assert not any("dx=" in r for r in revised)

    # Verify GameMemory integration
    gm = GameMemory(game_id="test_game")
    gm.apply_invariant_revision(revised, outcome="WIN", level_id="level_1")
    assert len(gm.structured_invariants) == 3
    assert len(gm.tier1_kinematics_and_topology) >= 1
    assert len(gm.tier2_interactions) >= 1
    assert len(gm.tier3_level_rules) >= 1


def test_solver_revise_invariants_failure_flow():
    """Verify Solver receives failure data, invalidation reasons, and reformulates valid invariant."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    solver = SolverAgent(config, advisor)

    # Turn 1: Trajectory proposal to seed hypothesis and conversation history
    grid = [[0, 1, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "RESET"])
    manifest = {
        "schema_version": "v10.dsl_manifest.1",
        "functions": [{"name": "action1", "parameters": []}],
    }

    advisor.set_response(
        "solver",
        "<invariant_analysis>\nHypothesis: Direct horizontal push into wall\n</invariant_analysis>\n"
        "<trajectory_1>\n1. action1()\n</trajectory_1>",
    )
    solver.generate_trajectory_package(manifest, pset)

    # Turn 2: Reflection on failure
    advisor.set_response(
        "solver_reflection",
        "<revised_invariants>\n"
        "- [PHYSICS]: ACTION1 moves active piece rightward until obstacle or boundary collision\n"
        "- [CONSTRAINTS]: Motion into immovable boundary yields zero displacement\n"
        "</revised_invariants>",
    )

    gm = GameMemory(game_id="test_game")
    gm.invalidate_action_effect(
        action_id="ACTION1",
        level_id="level_0",
        reason="Zero delta produced: entity collided with right boundary",
    )

    inv_data = gm.get_invariants_for_revision()
    assert len(inv_data["invalidated"]) >= 1
    assert "collided with right boundary" in inv_data["invalidated"][0]["invalidation_reason"]

    revised = solver.reflect_and_revise_on_failure(
        execution_summary="action1 executed 3 times before zero delta",
        failure_reason="Step produced zero delta: entity collided with right boundary",
        active_invariants=inv_data["active"],
        invalidated_invariants=inv_data["invalidated"],
    )

    assert len(revised) == 2
    assert any("[PHYSICS]: ACTION1 moves" in r for r in revised)

    # Applying revised invariant should resolve previously invalidated invariant
    gm.apply_invariant_revision(revised, outcome="FAILURE", level_id="level_0")
    # Action ACTION1 was reformulated in rule, so it should be resolved from invalidated_invariants
    assert len(gm.invalidated_invariants) == 0
    # And recorded into tier 1 kinematics/constraints
    assert len(gm.tier1_kinematics_and_topology) >= 1


def test_session_triggers_revision_on_win_and_failure():
    """Verify GameSession triggers Solver revision on both level win and attempt failure."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    session = GameSession(config=config, advisor=advisor)

    # 1. Simulate failure state in session
    session._last_attempt_failed = True
    session._last_failure_reason = "Circuit breaker triggered: static loop"
    session._last_failure_summary = "4 actions executed: action1, action1, action1, action1"
    session.level_chain_attempts = 1

    gm = session.memory_manager.get_game_memory("session")
    gm.record_stratified_invariant("[GOAL]: Fill sockets", tier=3)
    gm.falsify_invariant("inv_0", level_id="level_0", reason="Socket count mismatch")

    advisor.set_response(
        "solver_reflection",
        "<revised_invariants>\n"
        "- [GOAL]: Sockets accept pieces of matching color only\n"
        "- [PHYSICS]: Rigid translation along unobstructed grid lanes\n"
        "</revised_invariants>",
    )
    advisor.set_response(
        "solver",
        "<trajectory_1>\n1. action1()\n</trajectory_1>",
    )

    # Act on pristine grid
    grid = [[0] * 12 for _ in range(12)]
    grid[5][5] = 1
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

    assert session._last_attempt_failed is False
    assert any("matching color only" in r for r in gm.tier3_level_rules)

    # 2. Simulate level win
    advisor.set_response(
        "solver_reflection",
        "<revised_invariants>\n"
        "- [GOAL]: All color-matched sockets filled\n"
        "- [STRUCTURE]: Sockets positioned along grid perimeter\n"
        "</revised_invariants>",
    )

    session.handle_level_transition(new_level_id="level_1")
    assert any("All color-matched sockets filled" in r for r in gm.tier3_level_rules)
