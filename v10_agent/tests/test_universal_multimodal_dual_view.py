"""Unit tests for universal dual-view multimodal image injection across all Qwen roles."""

from __future__ import annotations

import json
import os
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.dsl_coder import DSLCoder
from v10_agent.explorer_agent import ExplorerAgent
from v10_agent.frame_media import render_annotated_frame_png, render_grid_png
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import EnvironmentSpecMemory, SyntaxErrorMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.prompt_builders.explorer_prompt import (
    build_coordinate_hypothesis_prompt,
    build_explorer_prompts,
)
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.sandbox import SandboxExecutor
from v10_agent.session import GameSession
from v10_agent.solver_agent import SolverAgent


FAKE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def test_prompt_builders_english_dual_view_notes():
    grid = [[0, 1, 0], [0, 0, 2]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "ACTION6", "RESET"])
    manifest = {"functions": [{"name": "action1", "parameters": []}]}

    # 1. Solver
    _, s_user_img = build_solver_prompts(manifest, planning_set, has_image=True)
    assert "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations." in s_user_img
    _, s_user_no_img = build_solver_prompts(manifest, planning_set, has_image=False)
    assert "solver_raw_frame.png is the exact same frame" not in s_user_no_img

    # 2. Explorer (spec)
    _, e_user_img = build_explorer_prompts(planning_set, has_image=True)
    assert "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations." in e_user_img
    _, e_user_no_img = build_explorer_prompts(planning_set, has_image=False)
    assert "explorer_raw_frame.png is the exact same frame" not in e_user_no_img

    # 3. Explorer (coords)
    _, c_user_img = build_coordinate_hypothesis_prompt(planning_set, has_image=True)
    assert "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations." in c_user_img
    _, c_user_no_img = build_coordinate_hypothesis_prompt(planning_set, has_image=False)
    assert "explorer_raw_frame.png is the exact same frame" not in c_user_no_img

    # 4. Coder
    _, cd_user_img = build_coder_prompts({}, planning_set=planning_set, has_image=True)
    assert "coder_raw_frame.png is the exact same frame as coder_annotated_frame.png, but without object annotations." in cd_user_img
    _, cd_user_no_img = build_coder_prompts({}, planning_set=planning_set, has_image=False)
    assert "coder_raw_frame.png is the exact same frame" not in cd_user_no_img


def test_explorer_multimodal_dual_view_transmission():
    grid = [[0, 1, 0], [0, 0, 2]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION6", "RESET"])
    memory = EnvironmentSpecMemory(game_id="g1", level_id="l0")

    mock_spec = {
        "schema_version": "v10.env_spec.1",
        "snapshot_hash": planning_set.grid_hash,
        "planning_set_id": planning_set.snapshot_id,
        "researched_actions": [{"action_id": "ACTION1", "effect_summary": "moves_right"}],
        "coordinate_affordances": [],
    }
    mock_coords = {
        "coordinate_hypotheses": [{"x": 1, "y": 0, "target_description": "token", "rationale": "click"}]
    }

    # Enabled
    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_spec)}\n```")
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_coords)}\n```")
    config = V10Config(llm_advisor_backend="fake", explorer_multimodal_enabled=True)
    explorer = ExplorerAgent(config, advisor)

    explorer.generate_environment_spec(planning_set, memory, image_png=[FAKE_PNG, FAKE_PNG])
    assert len(advisor.call_history) == 1
    call1 = advisor.call_history[0]
    assert call1["role"] == "explorer"
    assert call1["has_image"] is True
    assert "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations." in call1["user_prompt"]

    explorer.propose_coordinate_probes(planning_set, memory, image_png=[FAKE_PNG, FAKE_PNG])
    assert len(advisor.call_history) == 2
    call2 = advisor.call_history[1]
    assert call2["role"] == "explorer"
    assert call2["has_image"] is True
    assert "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations." in call2["user_prompt"]

    # Disabled
    advisor_dis = MockLLMAdvisor()
    advisor_dis.set_response("explorer", f"```json\n{json.dumps(mock_spec)}\n```")
    advisor_dis.set_response("explorer", f"```json\n{json.dumps(mock_coords)}\n```")
    config_dis = V10Config(llm_advisor_backend="fake", explorer_multimodal_enabled=False)
    explorer_dis = ExplorerAgent(config_dis, advisor_dis)

    explorer_dis.generate_environment_spec(planning_set, memory, image_png=[FAKE_PNG, FAKE_PNG])
    assert advisor_dis.call_history[0]["has_image"] is False
    assert "explorer_raw_frame.png is the exact same frame" not in advisor_dis.call_history[0]["user_prompt"]

    explorer_dis.propose_coordinate_probes(planning_set, memory, image_png=[FAKE_PNG, FAKE_PNG])
    assert advisor_dis.call_history[1]["has_image"] is False
    assert "explorer_raw_frame.png is the exact same frame" not in advisor_dis.call_history[1]["user_prompt"]


def test_coder_multimodal_dual_view_transmission():
    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    valid_py = "def action1(api):\n    return api.declare_environment_action('ACTION1')\n"
    valid_manifest = {"functions": [{"name": "action1", "parameters": []}]}
    resp = f"```python\n{valid_py}\n```\n```json\n{json.dumps(valid_manifest)}\n```"

    # Enabled
    advisor = MockLLMAdvisor()
    advisor.set_response("coder", resp)
    config = V10Config(llm_advisor_backend="fake", coder_multimodal_enabled=True)
    coder = DSLCoder(config, advisor, SandboxExecutor())

    coder.generate_dsl({}, syntax_mem, planning_set, image_png=[FAKE_PNG, FAKE_PNG])
    assert len(advisor.call_history) == 1
    call = advisor.call_history[0]
    assert call["role"] == "coder"
    assert call["has_image"] is True
    assert "coder_raw_frame.png is the exact same frame as coder_annotated_frame.png, but without object annotations." in call["user_prompt"]

    # Disabled
    advisor_dis = MockLLMAdvisor()
    advisor_dis.set_response("coder", resp)
    config_dis = V10Config(llm_advisor_backend="fake", coder_multimodal_enabled=False)
    coder_dis = DSLCoder(config_dis, advisor_dis, SandboxExecutor())

    coder_dis.generate_dsl({}, syntax_mem, planning_set, image_png=[FAKE_PNG, FAKE_PNG])
    assert advisor_dis.call_history[0]["has_image"] is False
    assert "coder_raw_frame.png is the exact same frame" not in advisor_dis.call_history[0]["user_prompt"]


def test_full_session_all_roles_receive_dual_view():
    grid = [[0, 1, 0], [0, 0, 0]]
    valid_py = "def action1(api):\n    return api.declare_environment_action('ACTION1')\n"
    manifest = {"functions": [{"name": "action1", "parameters": []}]}
    mock_spec = {
        "schema_version": "v10.env_spec.1",
        "researched_actions": [{"action_id": "ACTION1", "effect_summary": "ok"}],
    }
    mock_traj = {
        "proposal_id": "p1",
        "candidates": [{"trajectory_id": "c1", "steps": [{"dsl_function": "action1", "arguments": {}}]}]
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", json.dumps(mock_spec))
    advisor.set_response("coder", f"```python\n{valid_py}\n```\n```json\n{json.dumps(manifest)}\n```")
    advisor.set_response("solver", json.dumps(mock_traj))

    config = V10Config(
        llm_advisor_backend="fake",
        multimodal_enabled=True,
        explorer_multimodal_enabled=True,
        coder_multimodal_enabled=True,
        solver_multimodal_enabled=True,
    )
    session = GameSession(config, advisor)

    session.act({"grid": grid, "available_actions": ["ACTION1", "RESET"]})

    roles = [c["role"] for c in advisor.call_history]
    assert "explorer" in roles
    assert "coder" in roles
    assert "solver" in roles

    for call in advisor.call_history:
        assert call["has_image"] is True
        role = call["role"]
        expected_note = f"{role}_raw_frame.png is the exact same frame as {role}_annotated_frame.png, but without object annotations."
        assert expected_note in call["user_prompt"]
