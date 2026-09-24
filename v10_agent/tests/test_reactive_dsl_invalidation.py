"""Tests for Two-Tier Action Lifecycle and Reactive DSL Invalidation.

Verifies:
1. Reactive DSL invalidation when any newly confirmed action (not in active manifest) is discovered.
2. Preservation of conditional candidate actions (unconfirmed on S0) as CONDITIONAL_TRIGGER in available_actions.
3. DSLCoder auto-augmentation ensuring newly confirmed actions are synthesized and sandboxed.
4. Reactive invalidation during solver step execution with observable delta.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from v10_agent.llm_advisor import BaseLLMAdvisor
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.dsl_coder import DSLCoder
from v10_agent.explorer_agent import ExplorerAgent, PrimitiveProbeManager
from v10_agent.memory_contours import EnvironmentSpecMemory, GameMemory, SyntaxErrorMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.sandbox import SandboxExecutor, SandboxedModule
from v10_agent.session import GameSession
from v10_agent.types import PropositionSet
from v10_agent.verification import GroundedStep


class MockCoderAdvisor(BaseLLMAdvisor):
    def __init__(self, dsl_code: str):
        self.dsl_code = dsl_code

    def generate(self, **kwargs) -> str:
        return f"```python\n{self.dsl_code}\n```"


def test_reactive_dsl_invalidation_on_new_effect():
    """Verify active DSL module and manifest are invalidated when a new confirmed action is observed."""
    config = V10Config()
    session = GameSession(config)

    # Setup active DSL module with only action1..action4
    dummy_code = (
        "def action1(api):\n    return api.declare_environment_action('ACTION1')\n"
        "def action2(api):\n    return api.declare_environment_action('ACTION2')\n"
        "def action3(api):\n    return api.declare_environment_action('ACTION3')\n"
        "def action4(api):\n    return api.declare_environment_action('ACTION4')\n"
    )
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
            {"name": "action3", "parameters": []},
            {"name": "action4", "parameters": []},
        ]
    }
    session.active_module = SandboxExecutor().load_module(dummy_code, manifest)
    session.active_manifest = manifest
    session.coder_failed_for_level = True
    session.replan_requested = False

    grid = [[0] * 10 for _ in range(10)]
    after_grid = [[0] * 10 for _ in range(10)]
    for r in range(3):
        for c in range(4):
            after_grid[r][c] = 5  # 12 cells changed color

    session.last_snapshot = extract_arga_snapshot(grid)
    session.last_planning_set = build_planning_set(
        session.last_snapshot,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
    )

    # Simulate probe of ACTION5 producing a color transition
    session.last_probe_action = {"action_id": "ACTION5", "id": "ACTION5", "data": {}}
    session.pending_action = session.last_probe_action

    res = session.observe_action_result({"grid": after_grid, "state": "IN_PROGRESS"})
    assert res is True

    # Assertions per plan
    assert session.active_module is None
    assert session.active_manifest is None
    assert session.replan_requested is True
    assert session.coder_failed_for_level is False
    assert "ACTION5" in session.known_actions


def test_conditional_candidates_preserved_in_spec():
    """Verify actions with zero delta on S0 are preserved in available_actions as CONDITIONAL_TRIGGER."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MagicMock(spec=BaseLLMAdvisor)
    explorer = ExplorerAgent(config=config, advisor=advisor)

    grid = [[0] * 10 for _ in range(10)]
    grid[2][2] = 2
    snap0 = extract_arga_snapshot(grid)
    pset = build_planning_set(
        snap0,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
    )
    env_mem = EnvironmentSpecMemory(game_id="game_test", level_id="level_0")

    # ACTION1 moves object -> confirmed
    after_grid_1 = [[0] * 10 for _ in range(10)]
    after_grid_1[2][3] = 2
    explorer.probe_manager.record_probe_result(
        action_id="ACTION1",
        action_data={},
        before_snapshot=snap0,
        after_obs={"grid": after_grid_1, "state": "IN_PROGRESS"},
        memory=env_mem,
        planning_set=pset,
    )

    # ACTION5 has zero delta on pristine frame S0 -> conditional candidate
    explorer.probe_manager.record_probe_result(
        action_id="ACTION5",
        action_data={},
        before_snapshot=snap0,
        after_obs={"grid": grid, "state": "IN_PROGRESS"},
        memory=env_mem,
        planning_set=pset,
    )

    assert "ACTION5" in explorer.probe_manager.conditional_candidate_actions
    assert "ACTION1" in explorer.probe_manager.confirmed_effective_actions
    assert "ACTION5" not in explorer.probe_manager.confirmed_effective_actions

    # Synthesize spec
    spec = explorer.synthesize_level_spec(planning_set=pset, memory=env_mem)

    # Verify ACTION5 is NOT removed from available_actions
    assert "ACTION1" in spec["available_actions"]
    assert "ACTION5" in spec["available_actions"]

    # Verify ACTION5 affordance is CONDITIONAL_TRIGGER
    aff_5 = next((a for a in spec["action_affordances"] if a.get("action_id") == "ACTION5"), None)
    assert aff_5 is not None
    assert aff_5.get("effect_class") == "CONDITIONAL_TRIGGER"
    assert aff_5.get("parameters", {}).get("status") == "unconfirmed_on_s0"


def test_dsl_auto_augmentation_for_newly_confirmed_action():
    """Verify DSLCoder auto-supplements newly confirmed actions missing from LLM Coder output."""
    config = V10Config()
    # Mock coder LLM only produces action1..action4, omitting action5
    coder_code = """\
def action1(api):
    return api.declare_environment_action('ACTION1')

def action2(api):
    return api.declare_environment_action('ACTION2')

def action3(api):
    return api.declare_environment_action('ACTION3')

def action4(api):
    return api.declare_environment_action('ACTION4')
"""
    advisor = MockCoderAdvisor(coder_code)
    coder = DSLCoder(config, advisor)

    grid = [[0] * 8 for _ in range(8)]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])

    # Spec and GameMemory both confirm ACTION5 as newly discovered
    env_spec = {
        "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        "action_affordances": [
            {"action_id": "ACTION1", "effect_class": "KINEMATIC"},
            {"action_id": "ACTION2", "effect_class": "KINEMATIC"},
            {"action_id": "ACTION3", "effect_class": "KINEMATIC"},
            {"action_id": "ACTION4", "effect_class": "KINEMATIC"},
            {"action_id": "ACTION5", "effect_class": "PALETTE_TRANSITION"},
        ],
    }
    game_mem = GameMemory(game_id="test_g")
    game_mem.record_action_effect("ACTION5", "color transition: 12 cells changed")

    syntax_mem = SyntaxErrorMemory(level_id="l0")
    module, manifest, diags = coder.generate_dsl(
        env_spec=env_spec,
        syntax_memory=syntax_mem,
        planning_set=pset,
        game_memory=game_mem,
    )

    assert module is not None
    assert manifest is not None
    assert len(diags) == 0

    # Verify manifest contains all 5 actions
    fn_names = {str(fn.get("name", "")).lower() for fn in manifest.get("functions", [])}
    for expected_fn in ("action1", "action2", "action3", "action4", "action5"):
        assert expected_fn in fn_names

    # Verify action5 is compiled and functional in sandbox
    assert "action5" in module.namespace
    assert callable(module.namespace["action5"])

    # Verify dry-run with mock API
    mock_api = MagicMock()
    mock_api.declare_environment_action.return_value = {"action_id": "ACTION5"}
    decl = module.namespace["action5"](mock_api)
    mock_api.declare_environment_action.assert_called_with(action_id="ACTION5")


def test_reactive_dsl_invalidation_on_solver_step_with_delta():
    """Verify active DSL module is invalidated when a non-manifest action is executed during solver step."""
    config = V10Config()
    session = GameSession(config)

    dummy_code = (
        "def action1(api):\n    return api.declare_environment_action('ACTION1')\n"
        "def action2(api):\n    return api.declare_environment_action('ACTION2')\n"
    )
    manifest = {
        "functions": [
            {"name": "action1", "parameters": []},
            {"name": "action2", "parameters": []},
        ]
    }
    session.active_module = SandboxExecutor().load_module(dummy_code, manifest)
    session.active_manifest = manifest
    session.replan_requested = False

    grid = [[0] * 6 for _ in range(6)]
    after_grid = [[0] * 6 for _ in range(6)]
    after_grid[1][1] = 3

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3"])

    session.last_snapshot = snap
    session.last_planning_set = pset

    # Pending solver step executing ACTION3 (missing from active_manifest)
    step = GroundedStep(
        step_id="s1",
        dsl_function="action3",
        arguments={},
        expected_propositions=PropositionSet.empty(),
    )
    session.pending_step = step
    session.pending_action = {"action_id": "ACTION3", "id": "ACTION3", "data": {}}

    res = session.observe_action_result({"grid": after_grid, "state": "IN_PROGRESS"})
    assert res is True

    # Assert DSL was reactively invalidated for ACTION3
    assert session.active_module is None
    assert session.active_manifest is None
    assert session.replan_requested is True
    assert "ACTION3" in session.known_actions
