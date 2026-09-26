"""Tests for domain-general invariant deduction, color transition probes, and parameter parsing."""

from __future__ import annotations

import pytest
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.explorer_agent import compute_probe_effect
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.sandbox import SandboxExecutor
from v10_agent.solver_agent import parse_text_trajectory


def test_cell_color_mutation_probe_detection():
    grid0 = [[8 for _ in range(10)] for _ in range(10)]
    snap0 = extract_arga_snapshot(grid0)

    grid1 = [row[:] for row in grid0]
    for r in range(3, 6):
        for c in range(3, 6):
            grid1[r][c] = 9

    effect = compute_probe_effect(snap0, {"grid": grid1})
    assert "color transition" in effect
    assert "9 cells changed color" in effect
    assert "8->9 (9 cells)" in effect


def test_solver_positional_and_keyword_coordinate_arguments():
    manifest = {
        "functions": [
            {
                "name": "action6",
                "parameters": [
                    {"name": "x", "type": "int"},
                    {"name": "y", "type": "int"},
                ],
            },
            {
                "name": "action1",
                "parameters": [],
            },
        ]
    }

    text_pos = '''
[HYPOTHESIS]
Invariants & Goal: flip pattern
Strategy: click (40, 5) then click (10, 20)

[TRAJECTORY]
1. action6(40, 5)
2. action6(10, 20)
3. action1()
'''
    grid = [[1, 0], [0, 1]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION6"])
    pkg_pos = parse_text_trajectory(text_pos, manifest, planning_set=pset)
    assert pkg_pos is not None
    steps_pos = pkg_pos["candidates"][0]["steps"]
    assert len(steps_pos) == 3
    assert steps_pos[0]["dsl_function"] == "action6"
    assert steps_pos[0]["arguments"] == {"x": 40, "y": 5}
    assert steps_pos[1]["arguments"] == {"x": 10, "y": 20}
    assert steps_pos[2]["dsl_function"] == "action1"
    assert steps_pos[2]["arguments"] == {}

    text_kw = '''
[HYPOTHESIS]
Invariants & Goal: flip pattern
Strategy: click (40, 5)

[TRAJECTORY]
1. action6(x=40, y=5)
2. action1()
'''
    pkg_kw = parse_text_trajectory(text_kw, manifest, planning_set=pset)
    assert pkg_kw is not None
    steps_kw = pkg_kw["candidates"][0]["steps"]
    assert steps_kw[0]["arguments"] == {"x": 40, "y": 5}


def test_solver_prompt_dynamic_function_template():
    grid = [[1, 0], [0, 1]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6"])
    manifest = {
        "functions": [
            {
                "name": "action6",
                "parameters": [
                    {"name": "x", "type": "int"},
                    {"name": "y", "type": "int"},
                ],
            }
        ]
    }
    sys_p, user_p = build_solver_prompts(manifest, pset)
    assert "action6(x: int = 0, y: int = 0)" in user_p
    assert "discovered_invariants" not in user_p


def test_sandbox_executor_safe_missing_arguments():
    grid = [[1, 0], [0, 1]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6"])

    source = '''def action6(api, x: int = 0, y: int = 0):
    return api.declare_environment_action("ACTION6", data={"x": x, "y": y})
'''
    manifest = {
        "functions": [
            {
                "name": "action6",
                "parameters": [
                    {"name": "x", "type": "int"},
                    {"name": "y", "type": "int"},
                ],
            }
        ]
    }
    executor = SandboxExecutor()
    mod = executor.load_module(source, manifest)

    decl = executor.execute(mod, "action6", {}, pset)
    assert decl.declared_action.action_id == "ACTION6"
    assert "x" in decl.declared_action.data
    assert "y" in decl.declared_action.data
