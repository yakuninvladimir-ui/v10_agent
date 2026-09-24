"""Unit tests for ExplorerAgent (Call 1)."""

from __future__ import annotations

import json
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.explorer_agent import ExplorerAgent
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import EnvironmentSpecMemory
from v10_agent.planning_set import build_planning_set


def test_explorer_generates_and_records_env_spec():
    grid = [
        [0, 1, 0],
        [0, 0, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "RESET"])
    memory = EnvironmentSpecMemory(game_id="g1", level_id="l0")

    mock_spec = {
        "schema_version": "v10.env_spec.1",
        "snapshot_hash": planning_set.grid_hash,
        "planning_set_id": planning_set.snapshot_id,
        "researched_actions": [
            {
                "action_id": "ACTION1",
                "effect_summary": "moves_right",
                "supporting_evidence_ids": [],
                "confidence": 0.8,
                "contradicted": False,
            }
        ],
        "coordinate_affordances": [],
        "object_class_notes": [],
        "action_surface_notes": [],
        "invariants": [],
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_spec)}\n```")

    config = V10Config(llm_advisor_backend="fake")
    explorer = ExplorerAgent(config, advisor)

    result_spec = explorer.generate_environment_spec(planning_set, memory)
    assert result_spec["schema_version"] == "v10.env_spec.1"
    assert len(result_spec["researched_actions"]) == 2
    act_ids = {a["action_id"] for a in result_spec["researched_actions"]}
    assert act_ids == {"ACTION1", "ACTION2"}
    assert len(memory.specs) == 1
    assert memory.specs[0]["snapshot_hash"] == planning_set.grid_hash


def test_explorer_lean_spec_generation():
    """Verify Explorer handles lean JSON spec without technical schema boilerplate."""
    grid = [[0, 1, 0], [0, 0, 2]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "RESET"])
    memory = EnvironmentSpecMemory(game_id="g1", level_id="l0")

    lean_spec = {
        "researched_actions": [
            {
                "action_id": "ACTION1",
                "effect_summary": "moves piece UP by 1",
                "confidence": 0.85,
            }
        ],
        "invariants": ["pieces stay rigid"],
        "structural_notes": ["symmetry across vertical axis"],
    }

    advisor = MockLLMAdvisor()
    advisor.set_response("explorer", f"```json\n{json.dumps(lean_spec)}\n```")

    config = V10Config(llm_advisor_backend="fake")
    explorer = ExplorerAgent(config, advisor)

    result_spec = explorer.generate_environment_spec(planning_set, memory)
    assert result_spec["schema_version"] == "v10.env_spec.1"
    assert result_spec["snapshot_hash"] == planning_set.grid_hash
    assert result_spec["planning_set_id"] == planning_set.snapshot_id
    assert "pieces stay rigid" in result_spec["invariants"]
    assert "symmetry across vertical axis" in result_spec["invariants"]
    assert len(result_spec["researched_actions"]) == 2


def test_explorer_plan_probes():
    grid = [[0, 1, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION6", "RESET"])
    memory = EnvironmentSpecMemory(game_id="g1", level_id="l0")

    config = V10Config(llm_advisor_backend="fake")
    explorer = ExplorerAgent(config, MockLLMAdvisor())

    probes = explorer.plan_probes(planning_set, memory, max_probes=3)
    assert len(probes) >= 1
    action_ids = [p.action_id for p in probes]
    assert "ACTION1" in action_ids or "ACTION6" in action_ids
    # Probes never include RESET
    assert "RESET" not in action_ids


def test_explorer_synthesize_level_spec_factual():
    """Verify synthesize_level_spec incorporates factual visual scene analysis without coordinates."""
    from v10_agent.memory_contours import GameMemory

    grid = [
        [0, 5, 0, 10, 0, 4, 0],
        [0, 5, 5, 10, 0, 4, 4],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"])
    memory = EnvironmentSpecMemory(game_id="ar25", level_id="0")
    game_mem = GameMemory(game_id="ar25")

    # Set confirmed actions in probe manager
    config = V10Config(llm_advisor_backend="mock")
    advisor = MockLLMAdvisor()
    explorer = ExplorerAgent(config, advisor)
    explorer.probe_manager.confirmed_effective_actions["ACTION1"] = "moves obj_1 UP (dy=-1)"
    explorer.probe_manager.confirmed_effective_actions["ACTION2"] = "moves obj_1 DOWN (dy=+1)"
    explorer.probe_manager.inactive_actions.add("ACTION5")

    mock_synthesis = {
        "action_displacements": [
            {
                "action_id": "ACTION1",
                "displacements": [
                    {"alias": "B", "dy": -1, "dx": 0},
                    {"alias": "C", "dy": -1, "dx": 0},
                ],
                "coordination": "synchronous vertical movement",
            }
        ],
        "static_objects": [
            {"alias": "A", "description": "vertical line, did not move"}
        ],
        "structural_geometry": [
            "Vertical line A divides the canvas into symmetric halves.",
        ],
        "invariants": [
            "Piece B moves in discrete 1-cell increments."
        ],
        # Even if LLM erroneously outputs forbidden keys, they must be ignored
        "coordinate_hypotheses": [{"x": 10, "y": 20}],
        "goal": "Align piece C with target",
    }
    advisor.set_response("explorer", f"```json\n{json.dumps(mock_synthesis)}\n```")

    spec = explorer.synthesize_level_spec(
        planning_set=planning_set,
        memory=memory,
        game_memory=game_mem,
    )

    # 1. Action gating: strictly confirmed actions only
    assert spec["available_actions"] == ["ACTION1", "ACTION2"]
    assert "ACTION5" not in spec["available_actions"]

    # 2. Action displacements
    assert len(spec["action_displacements"]) == 1
    assert spec["action_displacements"][0]["action_id"] == "ACTION1"
    assert spec["action_displacements"][0]["coordination"] == "synchronous vertical movement"

    # 3. Static objects
    assert len(spec["static_objects"]) == 1
    assert spec["static_objects"][0]["alias"] == "A"

    # 4. Structural geometry & invariants
    assert "Vertical line A divides the canvas into symmetric halves." in spec["structural_geometry"]
    assert "Piece B moves in discrete 1-cell increments." in spec["invariants"]

    # 5. Forbidden keys strictly excluded
    assert "coordinate_hypotheses" not in spec
    assert "goal" not in spec

    # 6. GameMemory updated
    assert any("Vertical line A" in inv for inv in game_mem.tier1_kinematics_and_topology)


def test_solver_prompt_includes_explorer_phenomenology():
    """Verify build_solver_prompts includes EMPIRICAL PROBE DYNAMICS and completed failed sequences."""
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
    from v10_agent.memory_contours import EpistemicMemory

    grid = [[0, 1], [0, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "RESET"])

    manifest = {
        "functions": [
            {"name": "action1", "parameters": [], "returns": "effect_declaration", "docstring": "moves up"},
        ]
    }

    env_spec = {
        "action_displacements": [
            {
                "action_id": "ACTION1",
                "displacements": [
                    {"alias": "B", "dy": -3, "dx": 0},
                    {"alias": "C", "dy": -3, "dx": 0},
                ],
                "coordination": "synchronous movement of B and C",
            }
        ],
        "static_objects": [
            {"alias": "A", "description": "vertical divider, did not move"},
        ],
        "structural_geometry": [
            "Axis A divides grid into left and right halves",
        ],
    }

    ep_mem = EpistemicMemory(level_id="l0")
    ep_mem.record_failed_completed_trajectory(("action4", "action4", "action4"))

    sys_p, user_p = build_solver_prompts(
        manifest=manifest,
        planning_set=planning_set,
        epistemic_memory=ep_mem,
        env_spec=env_spec,
    )

    assert "EMPIRICAL PROBE DYNAMICS & OBJECT DISPLACEMENTS:" in user_p
    assert "ACTION1: B moved (dy=-3, dx=+0), C moved (dy=-3, dx=+0) [synchronous movement of B and C]" in user_p
    assert "A (vertical divider, did not move)" in user_p
    assert "Axis A divides grid into left and right halves" in user_p
    assert "- Completed sequence failed without win: action4 -> action4 -> action4" in user_p

