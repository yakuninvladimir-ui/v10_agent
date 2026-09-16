"""Tests for dynamic falsification detection, modal reprobing, and DSL invalidation pipeline."""

import pytest
from unittest.mock import MagicMock
from v10_agent.config import V10Config
from v10_agent.brusentsov_logic import Ternary, PropositionSet
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.verification import GroundedStep
from v10_agent.planning_set import PlanningSet
from v10_agent.memory_contours import EpistemicMemory, GameMemory, EnvironmentSpecMemory
from v10_agent.explorer_agent import PrimitiveProbeManager
from v10_agent.session import GameSession


class MockSnapshot:
    def __init__(self, grid):
        self.grid = grid
        self.grid_hash = str(hash(str(grid)))
        self.objects = []
        self.levels_completed = 0

    def get_object(self, obj_id):
        return None


def test_symbolic_executor_detects_zero_delta_falsification_on_step_0():
    config = V10Config()
    sandbox = MagicMock()
    binder = MagicMock()
    verifier = MagicMock()
    
    verifier.evaluate_transition.return_value = MagicMock(
        verdict=Ternary.TRUE,
        observed_propositions=PropositionSet(),
        explanation="Default follow",
    )

    executor = SymbolicTrajectoryExecutor(config, sandbox, binder, verifier)

    cand = CandidateTrajectory(
        trajectory_id="cand_1",
        confidence=0.9,
        steps=[{"dsl_function": "action3", "arguments": {}}],
        cursor=0,
    )
    pool = TrajectoryPool([cand])
    ep_mem = EpistemicMemory(level_id="level_2")
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    snap = MockSnapshot(grid)
    pset = MagicMock(spec=PlanningSet)
    pset.grid = grid
    pset.grid_hash = "hash_123"

    pending_step = GroundedStep(
        step_id="step_1",
        dsl_function="action3",
        arguments={},
        expected_propositions=PropositionSet(),
    )


    # Observation after action3 has identical grid (zero delta)
    after_obs = {"grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "state": "RUNNING"}
    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("ACTION3", "moved LEFT by dy=0, dx=-3")

    eval_res = executor.evaluate_transition(
        pending_step=pending_step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION3"},
    )


    assert eval_res.verdict == Ternary.FALSE
    assert eval_res.falsification_detected is True
    assert eval_res.falsified_action == "ACTION3"
    assert eval_res.candidate_severed is True
    assert eval_res.replan_needed is True
    assert eval_res.reset_needed is True


def test_symbolic_executor_passes_when_grid_changes_on_step_0():
    config = V10Config()
    sandbox = MagicMock()
    binder = MagicMock()
    verifier = MagicMock()
    verifier.evaluate_transition.return_value = MagicMock(
        verdict=Ternary.TRUE,
        observed_propositions=PropositionSet(),
        explanation="Default follow",
    )

    executor = SymbolicTrajectoryExecutor(config, sandbox, binder, verifier)

    cand = CandidateTrajectory(
        trajectory_id="cand_1",
        confidence=0.9,
        steps=[{"dsl_function": "action1", "arguments": {}}],
        cursor=0,
    )
    pool = TrajectoryPool([cand])
    ep_mem = EpistemicMemory(level_id="level_2")
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    snap = MockSnapshot(grid)
    pset = MagicMock(spec=PlanningSet)
    pset.grid = grid
    pset.grid_hash = "hash_123"

    pending_step = GroundedStep(
        step_id="step_1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet(),
    )


    # Observation after action1 moved the cell
    after_obs = {"grid": [[0, 1, 0], [0, 0, 0], [0, 0, 0]], "state": "RUNNING"}

    eval_res = executor.evaluate_transition(
        pending_step=pending_step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
        action_dict={"action_id": "ACTION1"},
    )

    assert eval_res.verdict == Ternary.TRUE
    assert eval_res.falsification_detected is False
    assert eval_res.candidate_advanced is True
    assert eval_res.candidate_severed is False


def test_primitive_probe_manager_modal_reprobing():
    mgr = PrimitiveProbeManager(max_probes=16)
    mgr.confirmed_effective_actions["ACTION1"] = "moved UP"
    mgr.confirmed_effective_actions["ACTION2"] = "moved DOWN"

    effect_summary = "selection indicator transferred: internal dots moved from obj_1 to obj_2 (active entity toggled)"
    reprobes = mgr.get_dynamic_reprobes("ACTION5", effect_summary)

    reprobe_ids = [p.action_id for p in reprobes]
    assert "ACTION1" in reprobe_ids
    assert "ACTION2" in reprobe_ids
    assert "ACTION3" in reprobe_ids
    assert "ACTION4" in reprobe_ids


def test_primitive_probe_manager_schedule_falsification_reprobe():
    mgr = PrimitiveProbeManager(max_probes=16)
    mgr.confirmed_effective_actions["ACTION3"] = "moved LEFT"
    mgr._initial_sweep_planned = True
    mgr.total_probes_executed = 10

    probes = mgr.schedule_falsification_reprobe(["ACTION3"])

    assert "ACTION3" not in mgr.confirmed_effective_actions
    assert mgr._initial_sweep_planned is False
    assert mgr.total_probes_executed == 0
    assert len(probes) == 4
    probe_ids = [p.action_id for p in probes]
    assert probe_ids == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def test_game_memory_invalidate_action_effect():
    mem = GameMemory(game_id="game_1")
    mem.record_action_effect("ACTION3", "moved LEFT by dy=0, dx=-3")
    assert "ACTION3" in mem.confirmed_action_effects
    assert any("Action ACTION3" in r for r in mem.tier1_kinematics_and_topology)

    mem.invalidate_action_effect("ACTION3")
    assert "ACTION3" not in mem.confirmed_action_effects
    assert not any("Action ACTION3" in r for r in mem.tier1_kinematics_and_topology)
    assert mem.unconfirmed_actions.get("ACTION3") == "falsified_by_empirical_verifier"


def test_session_falsification_triggers_clean_reset_and_micro_reprobe():
    config = V10Config()
    advisor = MagicMock()
    session = GameSession(config=config, advisor=advisor)
    session.handle_game_transition("test_game")

    # Seed mock active module and candidate pool
    session.active_module = MagicMock()
    session.active_manifest = {"functions": [{"name": "action3"}]}
    cand = CandidateTrajectory(
        trajectory_id="cand_1",
        confidence=0.9,
        steps=[{"dsl_function": "action3", "arguments": {}}],
        cursor=0,
    )
    session.active_pool = TrajectoryPool([cand])
    session.probing_phase = False

    game_mem = session.memory_manager.get_game_memory("session")
    game_mem.record_action_effect("ACTION3", "moved LEFT by dy=0, dx=-3")
    session.known_actions.add("ACTION3")

    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    session.level_initial_grid = grid
    session.level_initial_grid_hash = str(hash(str(grid)))

    # Simulate pending step
    session.pending_action = {"action_id": "ACTION3", "id": "ACTION3"}
    session.pending_step = GroundedStep(
        step_id="step_1",
        dsl_function="action3",
        arguments={},
        expected_propositions=PropositionSet(),
    )

    session.last_snapshot = MockSnapshot(grid)
    session.last_planning_set = MagicMock(spec=PlanningSet)
    session.last_planning_set.grid = grid
    session.last_planning_set.grid_hash = session.level_initial_grid_hash

    # Observe transition with zero delta
    after_obs = {"grid": grid, "state": "RUNNING", "levels_completed": 0}
    session.observe_action_result(after_obs)

    # 1. DSL and active pool must be invalidated
    assert session.active_module is None
    assert session.active_manifest is None
    assert session.active_pool is None
    assert session.replan_requested is True

    # 2. GameMemory and known_actions must be cleared of falsified action
    assert "ACTION3" not in game_mem.confirmed_action_effects
    assert "ACTION3" not in session.known_actions

    # 3. Probing phase and clean reset must be scheduled
    assert session.probing_phase is True
    assert session.solver_reset_pending is True
    assert session.solver_reset_reason == "falsification_clean_reprobe_reset"
    assert len(session.probe_queue) == 4

    # 4. act() must first emit RESET
    action_1 = session.act({"grid": grid, "state": "RUNNING", "levels_completed": 0, "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"]})
    assert action_1["action_id"] == "RESET"

    # 5. Observe RESET result
    session.observe_action_result({"grid": grid, "state": "RUNNING", "levels_completed": 0})

    # 6. act() must now emit the first micro-probe on the clean board
    action_2 = session.act({"grid": grid, "state": "RUNNING", "levels_completed": 0, "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"]})
    assert action_2["action_id"] == "ACTION1"
