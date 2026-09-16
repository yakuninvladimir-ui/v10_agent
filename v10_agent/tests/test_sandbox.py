"""Unit tests for AST static analysis and restricted SandboxExecutor."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set
from v10_agent.sandbox import SandboxExecutor, validate_dsl_source
from v10_agent.types import EffectDeclaration


def test_ast_rejects_disallowed_imports():
    bad_code = "import os\ndef test(): pass"
    diagnostics = validate_dsl_source(bad_code)
    assert any("Forbidden import: 'os'" in d for d in diagnostics)

    bad_code2 = "from sys import exit\ndef test(): pass"
    diagnostics2 = validate_dsl_source(bad_code2)
    assert any("Forbidden import from: 'sys'" in d for d in diagnostics2)


def test_ast_rejects_banned_calls():
    bad_code = "def exploit():\n    eval('1+1')"
    diagnostics = validate_dsl_source(bad_code)
    assert any("Forbidden function call: eval()" in d for d in diagnostics)

    bad_code2 = "def file_op():\n    open('secret.txt')"
    diagnostics2 = validate_dsl_source(bad_code2)
    assert any("Forbidden function call: open()" in d for d in diagnostics2)


def test_ast_rejects_dunder_traversal():
    bad_code = "def breakout():\n    return ().__class__.__subclasses__()"
    diagnostics = validate_dsl_source(bad_code)
    assert any("Forbidden dunder attribute access: '__subclasses__'" in d for d in diagnostics)


def test_manifest_parameter_mismatch():
    code = "def move_to(obj):\n    return None"
    manifest = {
        "functions": [
            {
                "name": "move_to",
                "parameters": [{"name": "obj"}, {"name": "target"}],
            }
        ]
    }
    diagnostics = validate_dsl_source(code, expected_manifest=manifest)
    assert any("Parameter mismatch for 'move_to'" in d for d in diagnostics)


def test_valid_sandbox_execution_and_dry_run():
    valid_source = """
import math

def move_obj(api, obj, target):
    dist = api.metric_distance(obj, target)
    return api.declare_environment_action(
        action_id="ACTION1",
        expected_metric_deltas={"distance": -1},
        target_object_ids=[obj, target],
    )
"""
    manifest = {
        "functions": [
            {
                "name": "move_obj",
                "parameters": [
                    {"name": "obj", "type": "planning_object_id"},
                    {"name": "target", "type": "planning_object_id"},
                ],
            }
        ]
    }

    grid = [
        [0, 1, 0, 0],
        [0, 0, 0, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])

    executor = SandboxExecutor()
    module = executor.load_module(valid_source, manifest)

    # 1. Dry run
    ok, err = executor.dry_run_manifest(module, planning_set)
    assert ok is True, err

    # 2. Execution
    effect = executor.execute(
        module,
        "move_obj",
        {"obj": "obj_0", "target": "obj_1"},
        planning_set,
    )
    assert isinstance(effect, EffectDeclaration)
    assert effect.declared_action.action_id == "ACTION1"
    assert effect.target_object_ids == ["obj_0", "obj_1"]
