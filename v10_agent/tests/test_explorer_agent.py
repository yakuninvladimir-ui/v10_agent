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
