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
    assert "GAME INVARIANTS" in sys_p
    assert "shape_" in user_p
    assert "ascii=" in user_p
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
    assert f"hash {planning_set.grid_hash}" in user_p
    assert "OBJECT INDEX" in user_p

    # With image disabled
    _, user_p_no_img = build_solver_prompts(manifest, planning_set, has_image=False)
    assert "solver_raw_frame.png is the exact same frame" not in user_p_no_img
    assert f"hash {planning_set.grid_hash}" in user_p_no_img


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
    assert f"hash {initial_hash}" in call["user_prompt"]
    assert session.level_initial_grid_hash == initial_hash


def test_solver_expect_grammar_parsing():
    """Verify parse_expect_grammar_str correctly parses all typed EXPECT forms."""
    from v10_agent.solver_agent import parse_expect_grammar_str

    grid = [[0, 1, 0], [0, 0, 2]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])

    # Test moved, color, unchanged, appears, gone, state, levels_completed
    raw_expect = "moved(A, 0, 1), color(B)=4, unchanged(A), appears(C), gone(D), state=WIN, levels_completed=2"
    parsed = parse_expect_grammar_str(raw_expect, pset)

    # Check moved
    assert any(p.get("predicate") == "dy" and p.get("value") == 0 for p in parsed)
    assert any(p.get("predicate") == "dx" and p.get("value") == 1 for p in parsed)

    # Check color
    assert any(p.get("predicate") == "color" and p.get("value") == 4 for p in parsed)

    # Check unchanged / preserved
    assert any(p.get("predicate") == "preserved" and p.get("subject_id") == pset.resolve_object_id("A") for p in parsed)

    # Check gone / destroyed
    assert any(p.get("predicate") == "destroyed" and p.get("subject_id") == "D" for p in parsed)

    # Check state and levels_completed
    assert any(p.get("predicate") == "state" and p.get("value") == "WIN" for p in parsed)
    assert any(p.get("predicate") == "levels_completed" and p.get("value") == 2 for p in parsed)


def test_solver_analysis_tag_extraction():
    """Verify parse_solver_output extracts <analysis> alongside <invariant_analysis>."""
    from v10_agent.solver_agent import parse_solver_output

    xml_text = """
<analysis>
Hypothesis A: Moving the key will unlock the door.
Discriminator: Check if key moves right on action1.
</analysis>

<invariant_analysis>
relied: Tier 1 physics
proposed: key opens door
conflicts: none
</invariant_analysis>

<trajectory_1>
1. action1() EXPECT: moved(A, 0, 1)
</trajectory_1>
"""
    manifest = {"functions": [{"name": "action1", "parameters": []}]}
    parsed = parse_solver_output(xml_text, manifest)
    assert parsed is not None
    assert "Moving the key will unlock the door" in parsed["analysis"]
    assert "Tier 1 physics" in parsed["hypothesis"]
    assert len(parsed["candidates"]) == 1
    props = parsed["candidates"][0]["steps"][0]["expected_propositions"]
    assert any(p.get("predicate") == "dy" and p.get("value") == 0 for p in props)
    assert any(p.get("predicate") == "dx" and p.get("value") == 1 for p in props)
    assert any(p.get("predicate") == "moved" and p.get("value") == (0, 1) for p in props)


def test_solver_budget_and_probe_counters_in_prompt():
    """Verify prompt builder injects explicit action, attempt, and probe counters."""
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts

    grid = [[0, 1], [0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])
    manifest = {"functions": [{"name": "action1", "parameters": []}]}

    _, user_p = build_solver_prompts(
        manifest,
        pset,
        action_budget=42,
        attempts_remaining=3,
        max_attempts=5,
        probes_remaining=2,
        max_probes=3,
    )
    assert "LIMITS & BUDGET:" in user_p
    assert "Actions remaining: 42" in user_p
    assert "Planning attempts: 3 of 5 remaining" in user_p
    assert "Diagnostic probes" not in user_p


def test_solver_core_principles_in_prompt():
    """Verify prompt builder injects the 5 core principles and has no length-biasing directives."""
    from v10_agent.prompt_builders.solver_prompt import SOLVER_SYSTEM_PROMPT

    assert "CORE PRINCIPLES:" in SOLVER_SYSTEM_PROMPT
    assert "1. Use only the provided action names and their real parameters." in SOLVER_SYSTEM_PROMPT
    assert "2. Use only object IDs that appear in the provided data. Do not invent IDs." in SOLVER_SYSTEM_PROMPT
    assert "3. Do not repeat sequences that are already listed as failed." in SOLVER_SYSTEM_PROMPT
    assert "4. You may propose short or long sequences. Length is your choice. The goal is to pass the level." in SOLVER_SYSTEM_PROMPT
    assert "5. Reason from what is visible and from confirmed action effects. Do not assume mechanics that are not supported by the data." in SOLVER_SYSTEM_PROMPT

    assert "shortest sufficient plan" not in SOLVER_SYSTEM_PROMPT.lower()
    assert "minimal discriminating experiment" not in SOLVER_SYSTEM_PROMPT.lower()
