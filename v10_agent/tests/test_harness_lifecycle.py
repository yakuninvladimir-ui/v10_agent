"""Integration regression test: harness lifecycle covering C1, C2, C3, C4.

This single test simulates the full competition harness flow that existing
unit tests cannot cover because they test components in isolation.  One
green run of this file guarantees the four critical Sprint-0/1 fixes
(config update_runtime, candidate sever, levels_completed_observed reset,
and invariant_registry liveness) have not regressed.
"""

from __future__ import annotations

import json

import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def _build_step(step_id: str, dsl_function: str = "action1") -> dict:
    return {"step_id": step_id, "dsl_function": dsl_function, "arguments": {}}


def test_harness_lifecycle_regression_protection():
    """End-to-end harness flow: C1 config, C3 transition, C4 registry, C2 GAME_OVER."""

    valid_py = (
        "def action1(api):\
"
        "    return api.declare_environment_action(\"ACTION1\")\n"
    )
    manifest = {"functions": [{"name": "action1", "parameters": []}]}
    traj_pkg = {
        "candidates": [
            {"trajectory_id": "c1", "steps": [_build_step("s1")]},
            {"trajectory_id": "c2", "steps": [_build_step("s2")]},
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response(
        "coder",
        f"`python\
{valid_py}\
`\
`json\
{json.dumps(manifest)}\
`",
    )
    advisor.set_response("solver", f"`json\
{json.dumps(traj_pkg)}\
`")

    config = V10Config(llm_advisor_backend="fake", enable_primitive_probing=False)
    session = GameSession(config, advisor)

    grid = [[0, 1, 0], [0, 0, 2]]
    obs = {
        "grid": grid,
        "available_actions": ["ACTION1", "ACTION2", "RESET"],
        "levels_completed": 0,
    }

    # --- C1: update_runtime survives harness dicts ---
    session.config.update_runtime({
        "_deadline_time": None,
        "nonexistent_harness_key": 42,
        "max_actions_per_game": 500,
    })
    assert session.config.max_actions_per_game == 500

    d = session.config.to_dict()
    for key in d:
        assert not key.startswith("_"), f"Private field leaked: {key!r}"
    assert "_deadline_time" not in d

    # --- C3: handle_game_transition resets all counters ---
    session.levels_completed_observed = 3
    session.game_over_reset_count = 2
    session.level_chain_attempts = 4
    session.accepted_action_count = 15

    reg_before = session.invariant_registry
    assert reg_before is not None  # C4

    session.handle_game_transition("game_B")

    assert session.levels_completed_observed == 0
    assert session.game_over_reset_count == 0
    assert session.level_chain_attempts == 0
    assert session.accepted_action_count == 0
    assert session.current_game_id == "game_B"
    assert session.active_pool is None

    reg_after = session.invariant_registry
    assert reg_after is not None  # C4

    # --- C2: GAME_OVER severs active candidate, preserves pool ---
    act1 = session.act(obs)
    assert act1["id"] == "ACTION1"
    assert session.active_pool is not None
    assert session.level_chain_attempts == 1

    game_over_obs = {"state": "GAME_OVER", "grid": grid, "levels_completed": 0}
    reset_act = session.act(game_over_obs)

    assert reset_act["id"] == "RESET"
    assert reset_act["reasoning"]["source"] == "tufa_game_over_auto_reset"
    assert session.game_over_reset_count == 1

    assert session.active_pool is not None, "C2: active_pool was cleared"
    next_c = session.active_pool.peek_active_candidate()
    assert next_c is not None, "No remaining candidate after sever"
    assert next_c.trajectory_id == "c2", f"Expected c2, got {next_c.trajectory_id}"

    # Double-RESET guard
    with pytest.raises(RuntimeError, match="GAME_OVER persisted"):
        session.act({"state": "GAME_OVER", "grid": grid})

    # Exhausted-attempts guard
    exhausted = GameSession(config, MockLLMAdvisor())
    exhausted.level_chain_attempts = 5
    with pytest.raises(LevelAttemptsExhaustedError, match="attempts exhausted"):
        exhausted.act({"state": "GAME_OVER", "grid": grid})


def test_config_to_dict_excludes_all_private_fields():
    """C1: to_dict() must never leak _-prefixed fields."""
    config = V10Config()
    d = config.to_dict()
    for key in d:
        assert not key.startswith("_"), f"Private field leaked: {key!r}"
    assert "_deadline_time" not in d
    assert d.get("max_actions_per_game") == 500
    assert d.get("crop_border_pixels") == 1
