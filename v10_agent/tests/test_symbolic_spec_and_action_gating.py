"""Unit test verifying Symbolic Environment Spec generation, strict Action Gating, and no visible effect fallback."""

from __future__ import annotations

import json
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.explorer_agent import (
    PrimitiveProbeManager,
    compute_probe_effect,
    generate_symbolic_environment_spec,
)
from v10_agent.memory_contours import EnvironmentSpecMemory, GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts


def test_compute_probe_effect_zero_diff_fallback():
    """Verify compute_probe_effect returns explicit 'no visible effect' when board does not change."""
    grid = [[1, 2], [3, 4]]
    snapshot = extract_arga_snapshot(grid)
    effect = compute_probe_effect(snapshot, {"grid": grid})
    assert effect == "no visible effect (0 cells changed)"
    assert "conditional" not in effect


def test_symbolic_spec_strictly_excludes_zero_effect_actions():
    """Verify generate_symbolic_environment_spec includes ONLY confirmed effective actions."""
    grid = [[0, 1], [0, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "RESET"])
    memory = EnvironmentSpecMemory(game_id="ar25", level_id="0")

    probe_mgr = PrimitiveProbeManager(max_probes=16)
    # Simulate: ACTION1..4 confirmed effective, ACTION5 had 0 diff
    probe_mgr.confirmed_effective_actions["ACTION1"] = "moved obj_1 by dy=0, dx=-1 (LEFT)"
    probe_mgr.confirmed_effective_actions["ACTION2"] = "moved obj_1 by dy=0, dx=1 (RIGHT)"
    probe_mgr.confirmed_effective_actions["ACTION3"] = "moved obj_1 by dy=-1, dx=0 (UP)"
    probe_mgr.confirmed_effective_actions["ACTION4"] = "moved obj_1 by dy=1, dx=0 (DOWN)"
    probe_mgr.inactive_actions.add("ACTION5")
    probe_mgr.zero_effect_actions.add("ACTION5")

    spec = generate_symbolic_environment_spec(
        planning_set=planning_set,
        memory=memory,
        probe_manager=probe_mgr,
    )

    # ACTION5 must NOT be in available_actions or researched_actions
    assert spec["available_actions"] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    assert "ACTION5" not in spec["available_actions"]
    researched_ids = [a["action_id"] for a in spec["researched_actions"]]
    assert researched_ids == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    assert "ACTION5" not in researched_ids


def test_coder_prompt_only_includes_confirmed_actions():
    """Verify build_coder_prompts restricts available actions list strictly to confirmed ones."""
    grid = [[0, 1], [0, 0]]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "RESET"])

    spec = {
        "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        "researched_actions": [
            {"action_id": "ACTION1", "effect_summary": "moved LEFT"},
            {"action_id": "ACTION2", "effect_summary": "moved RIGHT"},
            {"action_id": "ACTION3", "effect_summary": "moved UP"},
            {"action_id": "ACTION4", "effect_summary": "moved DOWN"},
        ],
    }

    game_mem = GameMemory(game_id="ar25")
    game_mem.record_unconfirmed_action("ACTION5", "no visible effect (0 cells changed)")

    sys_p, user_p = build_coder_prompts(
        env_spec=spec,
        planning_set=planning_set,
        game_memory=game_mem,
    )

    assert "CONFIRMED EFFECTIVE ACTIONS" in user_p
    confirmed_section = user_p.split("CONFIRMED EFFECTIVE ACTIONS")[1].split("Sandbox API Contract")[0]
    assert "- ACTION1" in confirmed_section
    assert "- ACTION2" in confirmed_section
    assert "- ACTION3" in confirmed_section
    assert "- ACTION4" in confirmed_section
    assert "- ACTION5" not in confirmed_section
    assert "UNCONFIRMED / INACTIVE ACTIONS" in user_p
    assert "- ACTION5: no visible effect (0 cells changed)" in user_p
    assert "You MUST implement functions ONLY for the confirmed effective actions" in sys_p
