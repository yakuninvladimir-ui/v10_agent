"""Unit tests for Solver and Coder retry exhaustion engaging persistent symbolic fallback."""

from __future__ import annotations

import json
import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession, LevelAttemptsExhaustedError, SessionPhase


def test_solver_retry_exhaustion_engages_persistent_fallback():
    """When Solver proposals fail 5 times, session must NOT abort with an exception.
    It must enter persistent fallback and emit symbolic fallback actions step-by-step."""
    valid_py = """
def step_action(api, obj):
    return api.declare_environment_action("ACTION1", target_object_ids=[obj])
"""
    manifest = {
        "functions": [
            {"name": "step_action", "parameters": [{"name": "obj", "type": "planning_object_id"}]}
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    # Solver fails to produce any valid json on all attempts
    advisor.set_response("solver", "invalid non-json solver response")

    config = V10Config(
        llm_advisor_backend="fake",
        max_chain_attempts_per_level=5,
        enable_symbolic_fallback=True,
        solver_exhaustion_forces_fallback=True,
        max_actions_per_game=250,
        max_actions_per_level=250,
        # Primitive probing is on in production; this test counts Solver chain attempts.
        enable_primitive_probing=False,
    )
    session = GameSession(config, advisor)

    grid = [[0, 1, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION1", "ACTION2", "RESET"], "state": "IN_PROGRESS"}

    # Execute 6 steps: 5 attempts fail solver proposal; on attempt 5, it transitions to persistent fallback
    action = None
    for step in range(6):
        action = session.act(obs)
        session.observe_action_result(obs)

    assert action is not None
    assert session.session_aborted is False
    assert session.in_persistent_fallback is True
    assert session.current_phase == SessionPhase.FALLBACK
    assert action["id"] in {"ACTION1", "ACTION2"}
    assert action["reasoning"].get("strategy") == "symbolic_fallback"


def test_persistent_fallback_does_not_emit_board_reset():
    """While in persistent fallback, stepping with a dirty board must NOT emit board RESETs."""
    config = V10Config(
        llm_advisor_backend="fake",
        max_chain_attempts_per_level=5,
        enable_symbolic_fallback=True,
        solver_exhaustion_forces_fallback=True,
        max_actions_per_game=250,
        max_actions_per_level=250,
        # Primitive probing is on in production; this test counts Solver chain attempts.
        enable_primitive_probing=False,
    )
    session = GameSession(config, MockLLMAdvisor())
    session.in_persistent_fallback = True

    grid = [[1, 0, 0], [0, 0, 2]]
    obs = {"grid": grid, "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"], "state": "IN_PROGRESS"}

    # Take 10 steps in fallback - none of them should ever be RESET
    actions = []
    for step in range(10):
        moved_grid = [[0, 1, 0], [0, 0, 2]] if step % 2 == 0 else [[0, 0, 1], [0, 0, 2]]
        obs["grid"] = moved_grid
        act = session.act(obs)
        session.observe_action_result(obs)
        actions.append(act["id"])

    assert "RESET" not in actions
    assert len(actions) == 10
    assert all(a in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} for a in actions)


def test_level_win_resets_persistent_fallback_for_next_level():
    """Winning a level while in persistent fallback must restore full LLM mode for Level 1."""
    config = V10Config(
        llm_advisor_backend="fake",
        max_chain_attempts_per_level=5,
        enable_symbolic_fallback=True,
        solver_exhaustion_forces_fallback=True,
        max_actions_per_game=250,
        max_actions_per_level=250,
        # Primitive probing is on in production; this test counts Solver chain attempts.
        enable_primitive_probing=False,
    )
    session = GameSession(config, MockLLMAdvisor())
    session.in_persistent_fallback = True
    session.level_chain_attempts = 5

    # Simulate Level 0 win
    grid = [[0, 1, 0]]
    win_obs = {
        "grid": grid,
        "available_actions": ["ACTION1", "RESET"],
        "state": "WIN",
        "levels_completed": 1,
    }
    _ = session.act(win_obs)

    # Invariants should be reset for new level and Solver initiates attempt 1 on Level 1
    assert session.in_persistent_fallback is False
    assert session.level_chain_attempts == 1
    assert session.current_level_id == "level_1"


def test_coder_exhaustion_default_engages_fallback():
    """Under default V10Config, Coder retry exhaustion engages symbolic fallback without raising."""
    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", "invalid syntax syntax error")

    config = V10Config(
        llm_advisor_backend="fake",
        max_coder_retries_per_level=1,
        # Primitive probing is on in production; this test targets Coder exhaustion.
        enable_primitive_probing=False,
    )
    # Default config has coder_exhaustion_forces_fallback=True, abort_on_dsl_exhaustion=False
    assert config.coder_exhaustion_forces_fallback is True
    assert config.abort_on_dsl_exhaustion is False

    session = GameSession(config, advisor)
    grid = [[0, 1, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION1", "RESET"], "state": "IN_PROGRESS"}

    action = session.act(obs)
    assert action is not None
    assert action["id"] in {"ACTION1", "RESET"}
    assert session.in_persistent_fallback is True
    assert session.session_aborted is False
