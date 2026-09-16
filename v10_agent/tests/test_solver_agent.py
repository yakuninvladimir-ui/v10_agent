"""Unit tests for SolverAgent (Call 3)."""

from __future__ import annotations

import json
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import EpistemicMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import SolverAgent


def test_solver_generates_valid_trajectory_package():
    manifest = {
        "functions": [
            {
                "name": "move_right",
                "parameters": [{"name": "obj", "type": "planning_object_id"}],
            }
        ]
    }

    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    ep_mem = EpistemicMemory(level_id="l0")

    mock_package = {
        "schema_version": "v10.trajectory_package.1",
        "proposal_id": "prop_01",
        "candidates": [
            {
                "trajectory_id": "traj_01",
                "steps": [
                    {
                        "step_id": "s1",
                        "dsl_function": "move_right",
                        "arguments": {"obj": "obj_0"},
                        "expected_propositions": [],
                    }
                ],
            }
        ],
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("solver", f"```json\n{json.dumps(mock_package)}\n```")

    config = V10Config(llm_advisor_backend="fake")
    solver = SolverAgent(config, advisor)

    pkg = solver.generate_trajectory_package(manifest, planning_set, ep_mem, budget=30)
    assert pkg is not None
    assert pkg["schema_version"] == "v10.trajectory_package.1"
    assert len(pkg["candidates"]) == 1
    assert pkg["candidates"][0]["steps"][0]["dsl_function"] == "move_right"


def test_solver_rejects_unmanifested_function():
    manifest = {
        "functions": [{"name": "move_right", "parameters": []}]
    }
    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])

    # Calls unknown function "teleport"
    bad_package = {
        "candidates": [
            {
                "trajectory_id": "traj_01",
                "steps": [{"step_id": "s1", "dsl_function": "teleport", "arguments": {}}],
            }
        ]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("solver", f"```json\n{json.dumps(bad_package)}\n```")

    config = V10Config(llm_advisor_backend="fake")
    solver = SolverAgent(config, advisor)

    pkg = solver.generate_trajectory_package(manifest, planning_set)
    assert pkg is None


def test_solver_prompt_contains_geometric_and_prioritized_relations():
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts

    grid = [
        [1, 1, 0, 3, 0, 2, 2],
        [1, 0, 0, 3, 0, 0, 2],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2"])
    manifest = {"functions": [{"name": "step_action", "parameters": []}]}

    sys_p, user_p = build_solver_prompts(manifest, planning_set)
    assert "geometric/topological invariants" in sys_p
    assert "shape_signature" in user_p
    assert "compact_ascii" in user_p
    assert "chiral_mirror_h" in user_p
    assert "symmetric_axis_of" in user_p


def test_solver_retains_grounded_candidates_even_if_sandbox_rejects():
    """Verify that if VirtualKinematicSandbox rejects candidates (e.g. non-reflection puzzle),
    the validly grounded candidate is retained and returned instead of dropped."""
    grid = [
        [1, 1, 0, 3, 0, 2, 2],
        [1, 0, 0, 3, 0, 0, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2"])
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
        ]
    }

    mock_package = {
        "schema_version": "v10.trajectory_package.1",
        "proposal_id": "prop_test",
        "candidates": [
            {
                "trajectory_id": "traj_custom",
                "steps": [
                    {"step_id": "s1", "dsl_function": "action2", "arguments": {}},
                ],
            }
        ],
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("solver", f"```json\n{json.dumps(mock_package)}\n```")
    config = V10Config(llm_advisor_backend="fake")
    solver = SolverAgent(config, advisor)

    pkg = solver.generate_trajectory_package(manifest, planning_set)
    assert pkg is not None
    assert len(pkg["candidates"]) == 1
    assert pkg["candidates"][0]["trajectory_id"] == "traj_custom"
    assert pkg["candidates"][0]["steps"][0]["dsl_function"] == "action2"
 
 
def test_solver_parses_xml_tags_format():
    """Verify that Solver output structured with XML tags parses into valid trajectory packages."""
    from v10_agent.solver_agent import parse_solver_output

    xml_text = """
<invariant_analysis>
Inferred invariant: Translate actor entity to align horizontally with socket.
Strategy: Call action1 twice then action2 once.
</invariant_analysis>

<trajectory_1>
1. action1()
2. action1()
3. action2()
</trajectory_1>

<trajectory_2>
action2()
action1()
</trajectory_2>
"""
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
        ]
    }
    parsed = parse_solver_output(xml_text, manifest)
    assert parsed is not None
    assert "Translate actor entity" in parsed["hypothesis"]
    assert len(parsed["candidates"]) == 2
    assert parsed["candidates"][0]["trajectory_id"] == "traj_text_01"
    assert len(parsed["candidates"][0]["steps"]) == 3
    assert [s["dsl_function"] for s in parsed["candidates"][0]["steps"]] == ["action1", "action1", "action2"]

    assert parsed["candidates"][1]["trajectory_id"] == "traj_text_02"
    assert len(parsed["candidates"][1]["steps"]) == 2
    assert [s["dsl_function"] for s in parsed["candidates"][1]["steps"]] == ["action2", "action1"]


def test_solver_prompt_image_note_and_grid_hash():
    """Verify solver prompt includes grid_hash and the English image notice when has_image is True."""
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts

    grid = [[0, 1, 2], [3, 4, 5]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2"])
    manifest = {"functions": [{"name": "action1", "parameters": []}]}

    # With image enabled (default)
    sys_p, user_p = build_solver_prompts(manifest, planning_set, has_image=True)
    assert "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations." in user_p
    assert f'"grid_hash": "{planning_set.grid_hash}"' in user_p
    assert '"planning_objects":' in user_p

    # With image disabled
    _, user_p_no_img = build_solver_prompts(manifest, planning_set, has_image=False)
    assert "solver_raw_frame.png is the exact same frame" not in user_p_no_img
    assert f'"grid_hash": "{planning_set.grid_hash}"' in user_p_no_img


def test_solver_agent_multimodal_png_transmission():
    """Verify SolverAgent passes image_bytes to advisor when solver_multimodal_enabled is True."""
    manifest = {"functions": [{"name": "action1", "parameters": []}]}
    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1"])
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    # 1. When multimodal enabled
    advisor = MockLLMAdvisor()
    config = V10Config(llm_advisor_backend="fake", solver_multimodal_enabled=True)
    solver = SolverAgent(config, advisor)
    solver.generate_trajectory_package(manifest, planning_set, image_png=fake_png)

    assert len(advisor.call_history) == 1
    solver_call = advisor.call_history[0]
    assert solver_call["role"] == "solver"
    assert solver_call["has_image"] is True
    assert "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations." in solver_call["user_prompt"]

    # 2. When multimodal disabled
    advisor2 = MockLLMAdvisor()
    config2 = V10Config(llm_advisor_backend="fake", solver_multimodal_enabled=False)
    solver2 = SolverAgent(config2, advisor2)
    solver2.generate_trajectory_package(manifest, planning_set, image_png=fake_png)

    assert len(advisor2.call_history) == 1
    solver_call2 = advisor2.call_history[0]
    assert solver_call2["has_image"] is False
    assert "solver_raw_frame.png is the exact same frame" not in solver_call2["user_prompt"]


def test_session_solver_receives_identical_frame_after_probe_reset():
    """Verify session ensures PNG and parsed block passed to Solver are strictly from pristine post-reset frame."""
    from v10_agent.session import GameSession

    initial_grid = [[0, 1, 0], [0, 0, 0]]
    probed_grid = [[0, 0, 1], [0, 0, 0]]

    valid_py = """
def action1(api):
    return api.declare_environment_action("ACTION1")
"""
    manifest = {
        "functions": [{"name": "action1", "parameters": []}]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", "{}")
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")

    config = V10Config(
        llm_advisor_backend="fake",
        enable_primitive_probing=True,
        max_primitive_probes_per_level=1,
        max_explorer_probe_actions_per_level=1,
        solver_multimodal_enabled=True,
    )
    session = GameSession(config, advisor)

    # 1. First act: initial pristine frame. Probing triggers.
    act1 = session.act({"grid": initial_grid, "available_actions": ["ACTION1", "RESET"]})
    initial_hash = session.level_initial_grid_hash
    assert initial_hash is not None

    # 2. Second act: probe executed, grid changed in env. Probing queue ends -> emits RESET.
    act2 = session.act({"grid": probed_grid, "available_actions": ["ACTION1", "RESET"]})
    assert act2["id"] == "RESET"

    # 3. Third act: board has been reset back to pristine initial_grid.
    # Solver should now plan on pristine frame with PNG attached.
    act3 = session.act({"grid": initial_grid, "available_actions": ["ACTION1", "RESET"]})

    # Find solver call in advisor history
    solver_calls = [c for c in advisor.call_history if c["role"] == "solver"]
    assert len(solver_calls) == 1
    call = solver_calls[0]

    assert call["has_image"] is True
    assert "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations." in call["user_prompt"]
    assert f'"grid_hash": "{initial_hash}"' in call["user_prompt"]
    assert session.level_initial_grid_hash == initial_hash



