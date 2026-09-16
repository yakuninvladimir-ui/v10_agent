"""Unit tests ensuring complete prompt content isolation across all three agent roles."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.memory_contours import EpistemicMemory, SyntaxErrorRecord
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.prompt_builders.explorer_prompt import build_explorer_prompts
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts


def test_solver_prompt_isolation():
    """ISO-1: Solver prompt must never contain tracebacks or Python source code."""
    grid = [[0, 1, 0]]
    planning_set = build_planning_set(extract_arga_snapshot(grid), ["ACTION1"])
    manifest = {
        "functions": [
            {"name": "test_func", "parameters": [{"name": "obj", "type": "str"}], "docstring": "test doc"}
        ]
    }
    ep_mem = EpistemicMemory(level_id="l0")

    sys_p, user_p = build_solver_prompts(manifest, planning_set, ep_mem)
    combined = (sys_p + " " + user_p).lower()

    # Assert no python exception strings
    for banned in ("traceback", "syntaxerror", "typeerror", "exception:"):
        assert banned not in combined

    # Assert no Python source definitions
    assert "def " not in combined
    assert "return api." not in combined


def test_coder_prompt_isolation():
    """ISO-2: Coder prompt must never contain level goals or Epistemic judgments."""
    env_spec = {
        "schema_version": "v10.env_spec.1",
        "researched_actions": [{"action_id": "ACTION1", "effect": "moves"}],
    }
    syntax_errors = [
        SyntaxErrorRecord("h1", "def bad(): pass", "SyntaxError", "invalid syntax")
    ]

    sys_p, user_p = build_coder_prompts(env_spec, syntax_errors)
    combined = (sys_p + " " + user_p).lower()

    # Assert no level goals or winning condition references
    for banned in ("level_goal", "win_condition", "target_score", "hypothesis_family", "epistemic_verdict"):
        assert banned not in combined


def test_explorer_prompt_isolation():
    """ISO-3: Explorer prompt must never contain trajectory plans or goal speculation."""
    grid = [[0, 1, 0]]
    planning_set = build_planning_set(extract_arga_snapshot(grid), ["ACTION1"])

    sys_p, user_p = build_explorer_prompts(planning_set)
    combined = (sys_p + " " + user_p).lower()

    # Assert no trajectory proposing language
    assert "trajectory" not in combined
    assert "propose a sequence" not in combined
    assert "win condition" not in combined
