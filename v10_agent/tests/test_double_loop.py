"""Unit tests for Double-Loop error routing (External vs Internal loops)."""

from __future__ import annotations

import json
import pytest

from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.session import GameSession, LevelAttemptsExhaustedError


def test_external_loop_syntax_and_execution_error_routing():
    """External Loop: execution exceptions travel ONLY to SyntaxErrorMemory."""
    # A DSL function that raises ZeroDivisionError when called
    broken_py = """
def broken_fn(api, obj):
    x = 1 / 0
    return api.declare_environment_action("ACTION1")
"""
    manifest = {
        "functions": [
            {"name": "broken_fn", "parameters": [{"name": "obj", "type": "planning_object_id"}]}
        ]
    }
    traj_pkg = {
        "candidates": [
            {
                "trajectory_id": "t1",
                "steps": [{"step_id": "s1", "dsl_function": "broken_fn", "arguments": {"obj": "obj_0"}}],
            }
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{broken_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", f"```json\n{json.dumps(traj_pkg)}\n```")

    config = V10Config(llm_advisor_backend="fake", max_coder_retries_per_level=1, abort_on_dsl_exhaustion=True)
    session = GameSession(config, advisor)

    # Note: load_module compiles it, but dry-run will catch the 1/0
    # Let's test that Coder records the dry-run failure into SyntaxErrorMemory and Solver/EpistemicMemory is clean
    grid = [[0, 1, 0]]
    with pytest.raises(LevelAttemptsExhaustedError):
        session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})

    syntax_mem = session.memory_manager.get_syntax_error_memory("session")
    ep_mem = session.memory_manager.get_epistemic_memory("session")

    # SyntaxErrorMemory recorded the failure
    assert len(syntax_mem.entries) >= 1
    # EpistemicMemory received NO syntax errors
    assert len(ep_mem.judgments) == 0


def test_internal_loop_semantic_judgment_routing():
    """Internal Loop: semantic transition outcomes travel ONLY to EpistemicMemory."""
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

    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config, advisor)

    # Initial step
    grid_before = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    action = session.act({"grid": grid_before, "available_actions": ["ACTION1", "RESET"]})
    assert action["id"] == "ACTION1"

    # Observe after state where object moved UP or stayed same (violates expected row_delta +1)
    grid_after = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    session.observe_action_result({"grid": grid_after})

    syntax_mem = session.memory_manager.get_syntax_error_memory("session")
    ep_mem = session.memory_manager.get_epistemic_memory("session")

    # EpistemicMemory received the Brusentsov judgment
    assert len(ep_mem.judgments) == 1
    # SyntaxErrorMemory was NOT touched
    assert len(syntax_mem.entries) == 0
