"""Unit tests for GameSession complete lifecycle and competition invariants."""

from __future__ import annotations

import json
import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.logging import StructuredAuditLogger
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def test_tufa_game_over_single_reset_invariant():
    """Tufa loop invariant: GAME_OVER emits RESET within budget (attempts < 5); exhausts at 5; breaks on repeat."""
    config = V10Config(llm_advisor_backend="fake", reset_on_game_over=True, max_chain_attempts_per_level=5)
    session = GameSession(config, MockLLMAdvisor())

    # First GAME_OVER emits RESET
    action = session.act({"state": "GAME_OVER", "grid": [[0]]})
    assert action["id"] == "RESET"
    assert session.game_over_reset_count == 1
    assert session.replan_requested is True
    assert session.active_pool is None

    # Second immediate GAME_OVER without any intervening action raises to prevent infinite loops
    with pytest.raises(RuntimeError, match="GAME_OVER persisted after single RESET"):
        session.act({"state": "GAME_OVER", "grid": [[0]]})

    # Budget exhaustion (attempts >= 5): raises LevelAttemptsExhaustedError without reset
    session_exhausted = GameSession(config, MockLLMAdvisor())
    session_exhausted.level_chain_attempts = 5
    with pytest.raises(LevelAttemptsExhaustedError, match="attempts exhausted"):
        session_exhausted.act({"state": "GAME_OVER", "grid": [[0]]})

    # When explicitly disabled: raises LevelAttemptsExhaustedError immediately without emitting RESET
    config_disabled = V10Config(llm_advisor_backend="fake", reset_on_game_over=False)
    session_disabled = GameSession(config_disabled, MockLLMAdvisor())
    with pytest.raises(LevelAttemptsExhaustedError, match="attempts exhausted|resets disabled"):
        session_disabled.act({"state": "GAME_OVER", "grid": [[0]]})


def test_session_multi_step_loop_and_telemetry():
    valid_py = """
def step_action(api, obj):
    return api.declare_environment_action("ACTION1", target_object_ids=[obj])
"""
    manifest = {
        "functions": [
            {"name": "step_action", "parameters": [{"name": "obj", "type": "planning_object_id"}]}
        ]
    }
    traj_pkg = {
        "candidates": [
            {
                "trajectory_id": "t1",
                "steps": [
                    {"step_id": "s1", "dsl_function": "step_action", "arguments": {"obj": "obj_0"}},
                    {"step_id": "s2", "dsl_function": "step_action", "arguments": {"obj": "obj_0"}},
                ],
            }
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", f"```json\n{json.dumps(traj_pkg)}\n```")

    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config, advisor)

    grid = [[0, 1, 0]]

    # Step 1
    act1 = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})
    assert act1["id"] == "ACTION1"
    committed1 = session.observe_action_result({"grid": grid})
    assert committed1 is True

    # Duplicate observe returns False
    committed_dup = session.observe_action_result({"grid": grid})
    assert committed_dup is False

    # Step 2
    act2 = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})
    assert act2["id"] == "ACTION1"
    committed2 = session.observe_action_result({"grid": grid})
    assert committed2 is True

    # Telemetry
    telemetry = session.harness_telemetry()
    assert telemetry["accepted_action_count"] == 2
    assert telemetry["observed_transition_ingestions"] == 2
    assert telemetry["observed_transition_duplicate_skips"] == 1

    # Audit logger recorded events
    records = session.audit_logger.get_records()
    assert len(records) >= 2


def test_level_transition_preserves_game_memory():
    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config, MockLLMAdvisor())

    # Level 0
    session.memory_manager.get_game_memory("session").record_action_effect("ACTION1", "confirmed moves right")
    grid = [[0, 1, 0]]
    session.act({"grid": grid, "levels_completed": 0})

    # Transition to Level 1
    session.act({"grid": grid, "levels_completed": 1})
    assert session.current_level_id == "level_1"
    assert session.levels_completed_observed == 1

    # GameMemory retained and known_actions seeded
    game_mem = session.memory_manager.get_game_memory("session")
    assert game_mem.confirmed_action_effects["ACTION1"] == "confirmed moves right"
    assert "ACTION1" in session.known_actions


def test_multi_trajectory_pool_sequential_reset_execution():
    """Verify that multiple candidates in a pool are executed sequentially with RESET in between,
    WITHOUT calling Solver until all candidates in the pool are exhausted."""
    valid_py = """
def action1(api):
    return api.declare_environment_action("ACTION1")

def action2(api):
    return api.declare_environment_action("ACTION2")
"""
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
        ]
    }
    traj_pkg = {
        "candidates": [
            {
                "trajectory_id": "cand_01",
                "steps": [
                    {"step_id": "s1", "dsl_function": "action1", "arguments": {}},
                ],
            },
            {
                "trajectory_id": "cand_02",
                "steps": [
                    {"step_id": "s2", "dsl_function": "action2", "arguments": {}},
                ],
            },
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", f"```json\n{json.dumps(traj_pkg)}\n```")

    # Primitive probing is on in production; this test targets sequential candidate execution.
    config = V10Config(llm_advisor_backend="fake", enable_primitive_probing=False)
    session = GameSession(config, advisor)

    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION1", "ACTION2", "RESET"], "levels_completed": 0}

    # Step 1: candidate 1 executes action1
    act1 = session.act(obs)
    assert act1["id"] == "ACTION1"
    assert session.active_pool is not None
    assert session.level_chain_attempts == 1

    # Ingest observation: candidate 1 finishes its 1 step without winning
    session.observe_action_result(obs)

    # Step 2: Session must emit RESET to clean the board for candidate 2!
    # Solver must NOT be called again!
    act2 = session.act(obs)
    assert act2["id"] == "RESET"
    assert session.level_chain_attempts == 1  # Still 1! Solver was NOT called!
    assert session.active_pool is not None

    # Ingest RESET observation
    session.observe_action_result(obs)

    # Step 3: candidate 2 executes action2!
    act3 = session.act(obs)
    assert act3["id"] == "ACTION2"
    assert session.level_chain_attempts == 1  # Still 1! Solver was NOT called!

    # Ingest observation: candidate 2 finishes its step without winning
    session.observe_action_result(obs)

    # Step 4: Both candidates exhausted! Session must emit RESET
    act4 = session.act(obs)
    assert act4["id"] == "RESET"

    # Ingest RESET observation
    session.observe_action_result(obs)

    # Step 5: Now that all candidates failed, Solver IS called for attempt 2!
    act5 = session.act(obs)
    assert session.level_chain_attempts == 2


def test_no_double_reset_between_solver_candidates():
    """Verify that exactly ONE reset is emitted between candidates even with primitive probing enabled."""
    valid_py = """
def action1(api):
    return api.declare_environment_action("ACTION1")

def action2(api):
    return api.declare_environment_action("ACTION2")
"""
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
        ]
    }
    traj_pkg = {
        "candidates": [
            {"trajectory_id": "c1", "steps": [{"step_id": "s1", "dsl_function": "action1", "arguments": {}}]},
            {"trajectory_id": "c2", "steps": [{"step_id": "s2", "dsl_function": "action2", "arguments": {}}]},
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", f"```json\n{json.dumps(traj_pkg)}\n```")

    config = V10Config(llm_advisor_backend="fake", enable_primitive_probing=True)
    session = GameSession(config, advisor)

    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION1", "ACTION2", "RESET"], "levels_completed": 0}

    # Complete initial probing sweep if any
    actions = []
    for _ in range(20):
        act = session.act(obs)
        actions.append(act["id"])
        session.observe_action_result(obs)
        if act["id"] == "ACTION1" and session.active_pool is not None:
            break

    assert session.active_pool is not None
    # Candidate 1 executed ACTION1. Now candidate 1 is finished.
    # The next action MUST be a single RESET.
    reset_act = session.act(obs)
    assert reset_act["id"] == "RESET"
    assert reset_act["reasoning"]["source"] == "next_candidate_reset_clean_state"
    session.observe_action_result(obs)

    # The action immediately following RESET MUST be Candidate 2's action (ACTION2), NOT another RESET!
    cand2_act = session.act(obs)
    assert cand2_act["id"] == "ACTION2", f"Expected ACTION2 but got {cand2_act['id']} (double reset bug!)"


def test_structured_audit_logger_bounded_deque():
    logger = StructuredAuditLogger(max_records=5)
    for i in range(10):
        logger.log("step", step_idx=i)
    records = logger.get_records()
    assert len(records) == 5
    assert [r["data"]["step_idx"] for r in records] == [5, 6, 7, 8, 9]



