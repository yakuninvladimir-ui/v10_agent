"""Unit tests for Action Partitioning, ACTION6 Qwen Coordinate Discovery, and Trajectory Advancing."""

from __future__ import annotations

import json
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config
from v10_agent.explorer_agent import ExplorerAgent, PrimitiveProbeManager
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.explorer_prompt import build_coordinate_hypothesis_prompt
from v10_agent.session import GameSession


def test_action7_strictly_excluded():
    grid = [[0, 1, 0], [0, 0, 0]]
    snapshot = extract_arga_snapshot(grid)
    available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"]
    
    pset = build_planning_set(snapshot, available_actions)
    
    assert "ACTION7" not in pset.allowed_action_ids
    assert pset.is_valid_action("ACTION7") is False
    assert "ACTION1" in pset.allowed_action_ids
    assert "ACTION6" in pset.allowed_action_ids


def test_discrete_probes_restricted_to_action1_to_action5():
    mgr = PrimitiveProbeManager(max_probes=10)
    all_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"]
    
    probes = mgr.plan_discrete_probes(all_actions)
    probe_ids = [p["action_id"] for p in probes]
    
    assert probe_ids == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    assert "ACTION6" not in probe_ids
    assert "ACTION7" not in probe_ids
    assert "RESET" not in probe_ids


def test_qwen_coordinate_hypothesis_probes():
    grid = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    pset = build_planning_set(snapshot, ["ACTION1", "ACTION6", "RESET"])
    
    advisor = MockLLMAdvisor()
    mock_payload = {
        "coordinate_hypotheses": [
            {"x": 1, "y": 0, "target_description": "cyan pixel", "rationale": "test pixel"},
            {"x": 99, "y": 99, "target_description": "invalid out of bounds", "rationale": "bad bounds"},
            {"x": 2, "y": 2, "target_description": "red pixel", "rationale": "test red piece"},
        ]
    }
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_payload)}\n```")
    
    config = V10Config(llm_advisor_backend="fake")
    explorer = ExplorerAgent(config, advisor)
    
    coord_probes = explorer.propose_coordinate_probes(pset, max_coords=4)
    assert len(coord_probes) == 2
    coords = [(p.data["x"], p.data["y"]) for p in coord_probes]
    assert (1, 0) in coords
    assert (2, 2) in coords
    assert (99, 99) not in coords
    for p in coord_probes:
        assert p.action_id == "ACTION6"


def test_trajectory_cursor_advances_on_irrelevant():
    valid_py = """
def move_obj(api, obj):
    return api.declare_environment_action("ACTION1", target_object_ids=[obj])
"""
    manifest = {
        "functions": [
            {"name": "move_obj", "parameters": [{"name": "obj", "type": "planning_object_id"}]}
        ]
    }
    traj_pkg = {
        "candidates": [
            {
                "trajectory_id": "traj_0",
                "steps": [
                    {
                        "step_id": "step_1",
                        "dsl_function": "move_obj",
                        "arguments": {"obj": "obj_0"},
                        "expected_propositions": [
                            {"family": "grid_delta", "subject_id": "obj_0", "predicate": "unmatched_passive", "value": 0}
                        ],
                    },
                    {
                        "step_id": "step_2",
                        "dsl_function": "move_obj",
                        "arguments": {"obj": "obj_0"},
                        "expected_propositions": [],
                    },
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

    grid = [[0, 1, 0], [0, 0, 0]]
    act1 = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})
    assert act1["id"] == "ACTION1"
    assert session.pending_step.step_id == "step_1"

    session.observe_action_result({"grid": grid})
    assert session.active_pool.active_candidate().cursor == 1

    act2 = session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})
    assert act2["id"] == "ACTION1"
    assert session.pending_step.step_id == "step_2"


def test_multi_signal_level_transition_detection():
    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config, MockLLMAdvisor())

    grid = [[0, 1, 0]]
    session.act({"grid": grid, "levels_completed": 0, "state": "IN_PROGRESS", "available_actions": ["RESET"]})
    assert session.current_level_id == "level_0"
    session.observe_action_result({"grid": grid, "levels_completed": 0, "state": "IN_PROGRESS"})

    session.act({"grid": grid, "levels_completed": 1, "state": "IN_PROGRESS", "available_actions": ["RESET"]})
    assert session.current_level_id == "level_1"
    assert session.levels_completed_observed == 1
    session.observe_action_result({"grid": grid, "levels_completed": 1, "state": "IN_PROGRESS"})

    session.act({"grid": grid, "levels_completed": 1, "state": "WIN", "available_actions": ["RESET"]})
    assert session.current_level_id == "level_2"
    assert session.levels_completed_observed == 2


def test_coordinate_hypothesis_prompt_num_hypotheses_sync():
    grid = [[0, 1, 0], [0, 0, 0]]
    snapshot = extract_arga_snapshot(grid)
    pset = build_planning_set(snapshot, ["ACTION1", "ACTION6", "RESET"])

    sys_p, user_p = build_coordinate_hypothesis_prompt(pset, num_hypotheses=6)
    assert "Propose between 2 and 6" in sys_p
    assert "Propose between 2 and 6" in user_p

