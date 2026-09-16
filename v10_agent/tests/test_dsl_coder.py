"""Unit tests for DSLCoder (Call 2)."""

from __future__ import annotations

import json
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.dsl_coder import DSLCoder
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import SyntaxErrorMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.sandbox import SandboxExecutor


def test_coder_generates_valid_dsl():
    valid_py = """
def test_action(api, obj):
    return api.declare_environment_action("ACTION1", target_object_ids=[obj])
"""
    valid_manifest = {
        "functions": [
            {
                "name": "test_action",
                "parameters": [{"name": "obj", "type": "planning_object_id"}],
            }
        ]
    }
    response_text = f"```python\n{valid_py}\n```\n```json\n{json.dumps(valid_manifest)}\n```"

    advisor = MockLLMAdvisor()
    advisor.set_response("coder", response_text)

    config = V10Config(llm_advisor_backend="fake", max_coder_retries_per_level=2)
    coder = DSLCoder(config, advisor, SandboxExecutor())

    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    module, manifest, errors = coder.generate_dsl({}, syntax_mem, planning_set)
    assert module is not None
    assert manifest is not None
    assert len(errors) == 0
    assert len(syntax_mem.entries) == 0


def test_coder_retry_on_syntax_error():
    # Attempt 1: bad import with valid manifest structure
    bad_resp = """```python
import os
def bad(api, obj): pass
```
```json
{
  "functions": [
    {"name": "bad", "parameters": [{"name": "obj", "type": "planning_object_id"}]}
  ]
}
```"""

    # Attempt 2: valid code
    good_py = "def ok_action(api, obj):\n    return api.declare_environment_action('ACTION1', target_object_ids=[obj])"
    good_manifest = {
        "functions": [
            {
                "name": "ok_action",
                "parameters": [{"name": "obj", "type": "planning_object_id"}],
            }
        ]
    }
    good_resp = f"```python\n{good_py}\n```\n```json\n{json.dumps(good_manifest)}\n```"

    advisor = MockLLMAdvisor()
    advisor.set_response("coder", bad_resp)
    advisor.set_response("coder", good_resp)

    config = V10Config(llm_advisor_backend="fake", max_coder_retries_per_level=2)
    coder = DSLCoder(config, advisor, SandboxExecutor())

    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    module, manifest, errors = coder.generate_dsl({}, syntax_mem, planning_set)
    assert module is not None
    assert manifest is not None
    assert len(errors) == 0
    # Attempt 1 recorded into SyntaxErrorMemory
    assert len(syntax_mem.entries) == 1
    assert "Forbidden import: 'os'" in syntax_mem.entries[0].error_message


def test_coder_exhaustion_returns_none():
    bad_resp = "```python\nimport sys\n```\n```json\n{}\n```"
    advisor = MockLLMAdvisor()
    advisor.set_response("coder", bad_resp)
    advisor.set_response("coder", bad_resp)

    config = V10Config(llm_advisor_backend="fake", max_coder_retries_per_level=2)
    coder = DSLCoder(config, advisor, SandboxExecutor())

    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    module, manifest, errors = coder.generate_dsl({}, syntax_mem, planning_set)
    assert module is None
    assert manifest is None
    assert len(errors) > 0
    assert len(syntax_mem.entries) == 2
