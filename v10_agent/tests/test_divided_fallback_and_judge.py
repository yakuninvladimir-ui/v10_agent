"""Unit tests for Divided Fallback, Judge in the loop, Coder 3-retries, and 3 Orchestration Pipelines."""

from __future__ import annotations

import json
import pytest

from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession


def test_coder_three_retries_with_syntax_feedback():
    """Verify Coder retries up to 3 times, passing previous errors in prompt, and succeeding on attempt 3."""
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
                "steps": [{"step_id": "s1", "dsl_function": "step_action", "arguments": {"obj": "obj_0"}}],
            }
        ]
    }

    call_count = 0

    class SequentialAdvisor(MockLLMAdvisor):
        def generate(self, system_prompt: str, user_prompt: str, config: any = None, **kwargs) -> str:
            nonlocal call_count
            role = kwargs.get("agent_role", "")
            if role == "coder":
                call_count += 1
                if call_count == 1:
                    # Attempt 1: Syntax error
                    return "```python\ndef broken(\n```\n```json\n{}\n```"
                elif call_count == 2:
                    # Attempt 2: Missing manifest
                    assert "PREVIOUS COMPILATION" in user_prompt
                    return f"```python\n{valid_py}\n```\nNot a json block"
                else:
                    # Attempt 3: Valid code and manifest
                    assert "Attempt #1 Failure" in user_prompt
                    assert "Attempt #2 Failure" in user_prompt
                    return f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```"
            elif role == "solver":
                return f"```json\n{json.dumps(traj_pkg)}\n```"
            return "{}"

    advisor = SequentialAdvisor()
    # Primitive probing is on in production; this test targets Coder retry feedback.
    config = V10Config(
        llm_advisor_backend="fake", max_coder_retries_per_level=3, enable_primitive_probing=False
    )
    session = GameSession(config, advisor)

    grid = [[0, 1, 0]]
    action = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})

    assert call_count == 3
    assert session.active_module is not None
    assert action["id"] == "ACTION1"
    assert session.session_aborted is False


def test_judge_detects_false_triggers_replan_and_reset():
    """Judge detects Ternary.FALSE, severs branch into EpistemicMemory, and queues RESET for clean replanning."""
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
                    {
                        "step_id": "s1",
                        "dsl_function": "step_action",
                        "arguments": {"obj": "obj_0"},
                        "expected_propositions": [
                            {"family": "metric_sign", "subject_id": "obj_0", "predicate": "row_delta", "value": 1}
                        ],
                    }
                ],
            }
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", f"```json\n{json.dumps(traj_pkg)}\n```")

    # Primitive probing is on in production; this test targets Judge contradiction handling.
    config = V10Config(llm_advisor_backend="fake", enable_primitive_probing=False)
    session = GameSession(config, advisor)

    grid_before = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    # Act 1: Emits ACTION1 from solver candidate
    action1 = session.act({"grid": grid_before, "available_actions": ["ACTION1", "RESET"]})
    assert action1["id"] == "ACTION1"

    # Observe result: Object moved UP instead of DOWN (row delta was -1 instead of +1) -> Direct Contradiction!
    grid_after = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    session.observe_action_result({"grid": grid_after})

    # Verify Judge classified as Ternary.FALSE and severed the signature
    ep_mem = session.memory_manager.get_epistemic_memory("session")
    assert len(ep_mem.judgments) == 1
    assert ep_mem.judgments[0].verdict == Ternary.FALSE
    assert any(s.startswith("step_action") for s in ep_mem.severed_null_signatures) or "s1" in ep_mem.severed_null_signatures
    assert session.replan_requested is True

    # Act 2: Should emit RESET to return environment to clean state for alternative hypothesis
    action2 = session.act({"grid": grid_after, "available_actions": ["ACTION1", "RESET"]})
    assert action2["id"] == "RESET"
    assert action2["reasoning"]["source"] == "replan_reset_clean_state"


def test_coordinate_pipeline_selection_and_targeted_probing():
    """Detects coordinate actions and selects coordinate pipeline with targeted coordinate probes."""
    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config, MockLLMAdvisor())

    grid = [
        [0, 1, 0],
        [0, 0, 0],
    ]
    available_actions = ["ACTION6", "click_coordinate", "RESET"]

    action = session.act({"grid": grid, "available_actions": available_actions})
    assert session.active_pipeline == "coordinate"
    assert action["id"] == "ACTION6"
    assert "x" in action["data"] and "y" in action["data"]
