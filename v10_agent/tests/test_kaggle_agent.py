"""Unit tests for ARC_AGI_Agent Kaggle competition shim."""

from __future__ import annotations

import pytest

from kaggle_agent import ARC_AGI_Agent, arcade_step_args
from submission import _action_name, _state_name, default_config


def test_kaggle_agent_act_and_observe():
    cfg = default_config()
    cfg["llm_advisor_backend"] = "fake"
    agent = ARC_AGI_Agent(cfg)

    obs = {
        "grid": [
            [0, 1, 0],
            [0, 0, 2],
        ],
        "available_actions": ["ACTION1", "ACTION2", "RESET"],
        "state": "IN_PROGRESS",
    }

    action = agent.act(obs)
    assert action is not None
    action_id, data, reasoning = arcade_step_args(action)
    assert action_id is not None

    # Observe result
    committed = agent.observe_action_result(obs)
    assert committed is True

    # Duplicate observe skips
    committed_dup = agent.observe_action_result(obs)
    assert committed_dup is False

    telemetry = agent.harness_telemetry()
    assert telemetry["observed_transition_ingestions"] == 1
    assert telemetry["observed_transition_duplicate_skips"] == 1


def test_kaggle_agent_game_over_single_reset():
    cfg = default_config()
    cfg["llm_advisor_backend"] = "fake"
    agent = ARC_AGI_Agent(cfg)

    # Calling reset_after_game_over on IN_PROGRESS raises
    with pytest.raises(RuntimeError, match="requires GAME_OVER state"):
        agent.reset_after_game_over({"state": "IN_PROGRESS", "grid": [[0]]})

    # By default (reset_on_game_over=True, attempts < 5), calling on GAME_OVER succeeds and returns RESET
    reset_act = agent.reset_after_game_over({"state": "GAME_OVER", "grid": [[0]]})
    action_id, _, _ = arcade_step_args(reset_act)
    assert _action_name(action_id) == "RESET"

    # Second immediate reset_after_game_over raises to avoid loop
    with pytest.raises(RuntimeError, match="GAME_OVER persisted after single RESET"):
        agent.reset_after_game_over({"state": "GAME_OVER", "grid": [[0]]})

    # When attempts budget exhausted (>= 5), calling reset_after_game_over raises cleanly to abandon game without reset
    agent_exhausted = ARC_AGI_Agent(cfg)
    agent_exhausted._session.level_chain_attempts = 5
    with pytest.raises(RuntimeError, match="exhausted"):
        agent_exhausted.reset_after_game_over({"state": "GAME_OVER", "grid": [[0]]})

    # When explicitly disabled via reset_on_game_over=False, raises cleanly
    cfg_disabled = dict(cfg)
    cfg_disabled["reset_on_game_over"] = False
    agent_disabled = ARC_AGI_Agent(cfg_disabled)
    with pytest.raises(RuntimeError, match="exhausted|disabled"):
        agent_disabled.reset_after_game_over({"state": "GAME_OVER", "grid": [[0]]})


def test_kaggle_agent_cleanup():
    cfg = default_config()
    cfg["llm_advisor_backend"] = "fake"
    agent = ARC_AGI_Agent(cfg)
    assert agent._session is not None
    agent._cleanup_old_session()
    assert agent._session is None


def test_run_concurrent_arcade_games():
    from lcld_competition_child import run_concurrent_arcade_games

    class MockEnv:
        def __init__(self, game_id: str):
            self.game_id = game_id
            self.step_count = 0

        def step(self, action_id, data=None, reasoning=None):
            self.step_count += 1
            state = "WIN" if self.step_count >= 2 else "IN_PROGRESS"
            return {
                "state": state,
                "grid": [[0, 1]],
                "score": 1,
                "game_id": self.game_id,
            }

    games = [
        (MockEnv("game_1"), {"state": "IN_PROGRESS", "grid": [[0, 1]], "game_id": "game_1"}),
        (MockEnv("game_2"), {"state": "IN_PROGRESS", "grid": [[0, 1]], "game_id": "game_2"}),
    ]

    cfg = default_config()
    cfg["llm_advisor_backend"] = "fake"
    results = run_concurrent_arcade_games(games, concurrency=2, config=cfg)
    assert len(results) == 2
    for r in results:
        assert r["status"] == "completed"
        assert r["action_count"] == 2

