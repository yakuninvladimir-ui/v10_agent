"""Unit tests for Explorer Recursive Probing and Unified 5-Attempt Ceilings."""

from __future__ import annotations

import json
import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def test_explorer_quota_allocation_pure_and_mixed():
    """Verify Explorer allocates 5 probes for pure ACTION6 and 3 probes for mixed actions."""
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    # Case 1: Pure ACTION6
    advisor1 = MockLLMAdvisor()
    config1 = V10Config(llm_advisor_backend="fake", enable_primitive_probing=True)
    session1 = GameSession(config1, advisor1)

    obs_pure = {"grid": grid, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0}
    session1.act(obs_pure)

    explorer_calls = [c for c in advisor1.call_history if c["role"] == "explorer"]
    assert len(explorer_calls) == 1
    assert "between 2 and 5" in explorer_calls[0]["user_prompt"]

    # Case 2: Mixed (ACTION1 + ACTION6)
    advisor2 = MockLLMAdvisor()
    config2 = V10Config(llm_advisor_backend="fake", enable_primitive_probing=True)
    session2 = GameSession(config2, advisor2)

    obs_mixed = {"grid": grid, "available_actions": ["ACTION1", "ACTION6", "RESET"], "levels_completed": 0}
    session2.act(obs_mixed)

    explorer_calls2 = [c for c in advisor2.call_history if c["role"] == "explorer"]
    assert len(explorer_calls2) == 1
    assert "between 2 and 3" in explorer_calls2[0]["user_prompt"]


def test_explorer_recursive_loop_and_5_attempt_exhaustion():
    """Verify Explorer repeats probing across up to 5 attempts when 0 effects are observed, then aborts."""
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0}

    advisor = MockLLMAdvisor()
    mock_payload = {
        "coordinate_hypotheses": [
            {"x": 0, "y": 0, "target_description": "t0", "rationale": "test"},
            {"x": 1, "y": 0, "target_description": "t1", "rationale": "test"},
            {"x": 2, "y": 0, "target_description": "t2", "rationale": "test"},
        ]
    }
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_payload)}\n```")

    config = V10Config(
        llm_advisor_backend="fake",
        enable_primitive_probing=True,
        max_explorer_attempts_per_level=5,
    )
    session = GameSession(config, advisor)

    reset_reasons = []

    with pytest.raises(LevelAttemptsExhaustedError, match=r"Explorer probe budget exhausted \(5 attempts\)"):
        for step in range(50):
            act = session.act(obs)
            session.observe_action_result(obs)
            if act["id"] == "RESET":
                src = act.get("reasoning", {}).get("source", "")
                reset_reasons.append(src)

    assert len(reset_reasons) == 0  # Continuous probing without intermediate resets
    assert session.explorer_attempts_this_level == 5
    assert session.session_aborted is True


def test_explorer_stops_loop_when_action_confirmed():
    """Verify Explorer exits the loop as soon as an action with observable effect is discovered."""
    grid1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    grid2 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]

    valid_py = 'def action6(api, x=0, y=0):\n    return api.declare_environment_action("ACTION6", data={"x": x, "y": y})\n'
    manifest = {
        "functions": [{"name": "action6", "parameters": [{"name": "x", "type": "int"}, {"name": "y", "type": "int"}]}]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response(
        "explorer",
        """```json
{
  "coordinate_hypotheses": [
    {"x": 1, "y": 1, "target_description": "center", "rationale": "test"}
  ]
}
```""",
    )
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response(
        "solver",
        """<invariant_analysis>Solved</invariant_analysis>
<trajectory_1>
1. action6(x=1, y=1)
</trajectory_1>""",
    )

    config = V10Config(
        llm_advisor_backend="fake",
        enable_primitive_probing=True,
        max_explorer_attempts_per_level=5,
    )
    session = GameSession(config, advisor)

    # Step 1: Probe proposed at (1, 1)
    act1 = session.act({"grid": grid1, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0})
    assert act1["id"] == "ACTION6"

    # Step 1 result: grid changes to grid2 (color transition)
    session.observe_action_result({"grid": grid2, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0})

    # Step 2: Probing phase complete. Board reset to pristine.
    act2 = session.act({"grid": grid2, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0})
    assert act2["id"] == "RESET"
    assert act2["reasoning"]["source"] == "probe_phase_complete_reset_to_pristine"

    # Step 3: Pristine frame received -> Explorer finishes, Coder & Solver take over
    session.observe_action_result({"grid": grid1, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0})
    act3 = session.act({"grid": grid1, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0})

    assert session.probing_phase is False
    assert act3["reasoning"]["strategy"] == "solver_candidate"