"""Unit tests for SymbolicFallbackEngine and retry exhaustion fallback."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.fallback_symbolic import SymbolicFallbackEngine
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.planning_set import build_planning_set
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def test_fallback_engine_action_selection():
    config = V10Config()
    engine = SymbolicFallbackEngine(config)

    # 1. ACTION6 available with object
    grid = [[0, 1, 0]]
    planning_set = build_planning_set(extract_arga_snapshot(grid), ["ACTION6", "RESET"])
    effect = engine.select_fallback_action(planning_set)
    assert effect.declared_action.action_id == "ACTION6"
    assert "x" in effect.declared_action.data
    assert "y" in effect.declared_action.data

    # 2. Only directional actions available
    planning_set_dir = build_planning_set(extract_arga_snapshot(grid), ["ACTION1", "ACTION2", "RESET"])
    effect_dir = engine.select_fallback_action(planning_set_dir)
    assert effect_dir.declared_action.action_id in {"ACTION1", "ACTION2"}


def test_coder_exhaustion_forces_symbolic_fallback():
    # Model returns broken code on all attempts with fallback explicitly enabled
    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", "```python\nimport sys\n```\n```json\n{}\n```")

    config = V10Config(
        llm_advisor_backend="fake",
        max_coder_retries_per_level=1,
        abort_on_dsl_exhaustion=False,
        coder_exhaustion_forces_fallback=True,
    )
    session = GameSession(config, advisor)

    grid = [[0, 1, 0]]
    action = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})

    assert action is not None
    assert action["id"] in {"ACTION1", "RESET"}
    assert action["reasoning"].get("source") == "symbolic_fallback"


def test_coder_exhaustion_clean_abort_in_strict_mode():
    # Option 4 strict mode: aborts cleanly rather than spinning blind fallback
    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", "```python\nimport sys\n```\n```json\n{}\n```")

    config = V10Config(
        llm_advisor_backend="fake",
        max_coder_retries_per_level=1,
        abort_on_dsl_exhaustion=True,
    )
    session = GameSession(config, advisor)

    grid = [[0, 1, 0]]
    with pytest.raises(LevelAttemptsExhaustedError):
        session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})

    assert session.session_aborted is True


def test_fallback_engine_2d_bfs_pathfinding_around_obstacle():
    """Verify that 2D BFS pathfinder selects an obstacle-avoiding direction (DOWN)
    when the straight horizontal path to the target is blocked by an obstacle wall."""
    config = V10Config()
    engine = SymbolicFallbackEngine(config)

    # Actor (1 at 0,0), Obstacle (3 at 0,1 and 1,1), Target (2 at 0,2)
    grid = [
        [1, 3, 2, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot,
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"],
    )

    effect = engine.select_fallback_action(planning_set)
    assert effect.declared_action.reasoning.get("strategy") == "bfs_pathfinding"
    # To bypass obstacle wall at (0,1) and (1,1), first step must be DOWN (ACTION2)
    assert effect.declared_action.action_id == "ACTION2"

