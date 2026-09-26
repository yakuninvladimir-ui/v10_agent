"""Unit tests for Explorer Recursive Probing and Unified 5-Attempt Ceilings."""

from __future__ import annotations

import json
import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def test_explorer_quota_allocation_pure_and_mixed():
    """Verify Explorer allocates at most 2 coordinate probes."""
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

    # Case 1: Pure ACTION6
    advisor1 = MockLLMAdvisor()
    config1 = V10Config(llm_advisor_backend="fake", enable_primitive_probing=True)
    session1 = GameSession(config1, advisor1)

    obs_pure = {"grid": grid, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0}
    session1.act(obs_pure)

    explorer_calls = [c for c in advisor1.call_history if c["role"] == "explorer"]
    assert len(explorer_calls) == 1
    assert "between 2 and 2" in explorer_calls[0]["user_prompt"]


def test_explorer_recursive_loop_and_2_attempt_exhaustion_yields_invariants():
    """Verify Explorer executes up to 2 probe attempts when 0 effects observed, then yields invariants without aborting."""
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    obs = {"grid": grid, "available_actions": ["ACTION6", "RESET"], "levels_completed": 0}

    advisor = MockLLMAdvisor()
    mock_payload = {
        "coordinate_hypotheses": [
            {"x": 0, "y": 0, "target_description": "t0", "rationale": "test"},
            {"x": 1, "y": 0, "target_description": "t1", "rationale": "test"},
        ]
    }
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_payload)}\n```")

    config = V10Config(
        llm_advisor_backend="fake",
        enable_primitive_probing=True,
        max_explorer_attempts_per_level=2,
    )
    session = GameSession(config, advisor)

    # Execute steps until probing phase finishes
    for step in range(10):
        if not session.probing_phase:
            break
        act = session.act(obs)
        session.observe_action_result(obs)

    assert session.explorer_attempts_this_level == 2
    assert session.probing_phase is False
    assert session.session_aborted is False


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


def test_invariant_verification_module_budget_and_yield():
    """Verify invariant verification module is limited to 3 zero-effect invariant probes, then yields invariants without reprobing."""
    from v10_agent.explorer_agent import PrimitiveProbeManager

    mgr = PrimitiveProbeManager(max_probes=16, max_invariant_probes=3, max_steps_per_probe=2)
    mgr.available_discrete_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}
    mgr.inactive_actions = {"ACTION2"}

    # Effective-action reprobe (moved UP): Case 1 retest — does NOT consume invariant budget
    reprobes_eff = mgr.get_dynamic_reprobes("ACTION1", "moved UP")
    assert len(reprobes_eff) <= 2
    assert mgr.invariant_probes_count == 0, "Effective-action reprobes should not increment invariant count"

    # Zero-effect invariant probe 1: vector action with no visible effect (Case 3)
    mgr.inactive_actions = {"ACTION3"}
    reprobes1 = mgr.get_dynamic_reprobes("ACTION2", "no visible effect")
    assert mgr.invariant_probes_count == 1

    # Zero-effect invariant probe 2
    reprobes2 = mgr.get_dynamic_reprobes("ACTION3", "0 cells changed")
    assert mgr.invariant_probes_count == 2

    # Zero-effect invariant probe 3
    reprobes3 = mgr.get_dynamic_reprobes("ACTION4", "zero delta observed")
    assert mgr.invariant_probes_count == 3

    # Attempt 4: Budget exhausted -> returns empty and yields unverified invariant
    mgr.inactive_actions = {"ACTION5"}
    reprobes4 = mgr.get_dynamic_reprobes("ACTION1", "no visible effect")
    assert reprobes4 == []
    assert mgr.invariant_probes_count == 3
    assert "ACTION5" in mgr.confirmed_effective_actions
    assert mgr.confirmed_effective_actions["ACTION5"] == "unverified_invariant"