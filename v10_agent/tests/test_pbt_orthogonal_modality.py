"""Property-Based and Unit Testing for Level Transition Contract and Orthogonal Modality Invariant.

Verifies:
1. Level transition contract invariance: Passing a set or non-dict to handle_level_transition
   never clears unprobed actions or marks them as confirmed reusable actions.
2. Invariant of Orthogonal Modality: With a modal selector available, zero grid response
   along a kinematic axis returns Verdict.UNDECIDED requesting modal probe rather than
   falsifying the action.
3. Dynamic modal reprobing: When a vector direction yields zero response, get_dynamic_reprobes
   schedules modal switches (ACTION5, ACTION6) followed by re-testing the vector direction.
4. Brusentsov 3-valued non-vacuity: Lack of effect upon mode change or zero delta evaluates to
   Verdict.OMIT or Verdict.UNDECIDED, strictly preventing vacuous truth (material implication paradox).
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from hypothesis import given, strategies as st

from v10_agent.config import V10Config
from v10_agent.brusentsov_logic import Ternary, Verdict, PropositionSet, AtomicProposition, BrusentsovJudgment
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.verification import GroundedStep
from v10_agent.planning_set import PlanningSet
from v10_agent.memory_contours import EpistemicMemory, GameMemory
from v10_agent.explorer_agent import PrimitiveProbeManager
from v10_agent.judge import LayeredVerifier


class MockSnapshot:
    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.grid_hash = str(hash(str(grid)))
        self.objects: list[Any] = []
        self.levels_completed = 0

    def get_object(self, obj_id: str):
        return None


# Strategy generating arbitrary action names (e.g. ACTION1..ACTION6, custom verbs)
ACTION_NAMES = st.sampled_from(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "CUSTOM_VERB"])
ACTION_SETS = st.sets(ACTION_NAMES, min_size=1, max_size=7)


@given(action_set=ACTION_SETS)
def test_pbt_level_transition_ignores_set_and_preserves_exploration(action_set: set[str]):
    """Invariant: Passing a set of actions to handle_level_transition MUST NOT mark actions as

    confirmed reusable or clear inactive_actions. All probeable actions must remain unconfirmed.
    """
    mgr = PrimitiveProbeManager(max_probes=16)
    # Call handle_level_transition with a set (the bug condition)
    mgr.handle_level_transition(action_set)  # type: ignore[arg-type]

    # No actions should be marked confirmed_reusable_action
    assert len(mgr.confirmed_effective_actions) == 0

    # All allowed discrete probes must remain in inactive_actions for probing
    for allowed_act in mgr.DISCRETE_PROBE_ALLOWED:
        assert allowed_act in mgr.inactive_actions


@given(
    motion_dir=st.sampled_from(["UP", "DOWN", "LEFT", "RIGHT"]),
    modal_act=st.sampled_from(["ACTION5", "ACTION6"]),
)
def test_pbt_level_transition_preserves_only_pure_kinematics(motion_dir: str, modal_act: str):
    """Invariant: Level transition carries forward only pure directional motion verbs.

    Modal selectors (ACTION5, ACTION6, or toggle actions) MUST be kept unconfirmed.
    """
    mgr = PrimitiveProbeManager(max_probes=16)
    confirmed_map = {
        "ACTION1": f"moves entity {motion_dir} by 1 step",
        modal_act: "selection indicator appeared on target entity",
        "ACTION4": "toggle active state",
    }
    mgr.handle_level_transition(confirmed_map)

    # ACTION1 was pure motion -> confirmed reusable
    assert "ACTION1" in mgr.confirmed_effective_actions
    assert "ACTION1" not in mgr.inactive_actions

    # Modal switch and toggle MUST NOT be confirmed reusable
    assert modal_act not in mgr.confirmed_effective_actions
    if modal_act in mgr.DISCRETE_PROBE_ALLOWED:
        assert modal_act in mgr.inactive_actions
    assert "ACTION4" not in mgr.confirmed_effective_actions
    assert "ACTION4" in mgr.inactive_actions


@given(
    vector_act=st.sampled_from(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]),
    modal_act=st.sampled_from(["ACTION5", "ACTION6"]),
)
def test_pbt_orthogonal_modality_zero_response_arbitration(vector_act: str, modal_act: str):
    """Invariant of Orthogonal Modality:

    If an action space contains a modal switch, zero response on a directional movement
    indicates degree of freedom locking in current mode, NOT global falsification.
    System must return Verdict.UNDECIDED with a modal probe request.
    """
    config = V10Config()
    sandbox = MagicMock()
    binder = MagicMock()
    verifier = LayeredVerifier(config)
    executor = SymbolicTrajectoryExecutor(config, sandbox, binder, verifier)

    cand = CandidateTrajectory(
        trajectory_id="cand_modal",
        confidence=0.9,
        steps=[{"dsl_function": vector_act.lower(), "arguments": {}}],
        cursor=0,
    )
    pool = TrajectoryPool([cand])
    ep_mem = EpistemicMemory(level_id="level_0")

    # Grid with entity in the center (no boundary collision)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    snap = MockSnapshot(grid)
    pset = MagicMock(spec=PlanningSet)
    pset.grid = grid
    pset.grid_hash = "hash_modal"
    # Action space contains the vector action AND the modal switch
    pset.allowed_action_ids = [vector_act, modal_act, "RESET"]
    pset.objects = []

    game_mem = GameMemory(game_id="game_pbt")
    game_mem.record_action_effect(vector_act, "moves entity UP by 1 step")

    pending_step = GroundedStep(
        step_id="step_1",
        dsl_function=vector_act.lower(),
        arguments={},
        expected_propositions=PropositionSet(),
    )

    # Zero grid delta observed
    after_obs = {"grid": grid, "state": "RUNNING"}

    eval_res = executor.evaluate_transition(
        pending_step=pending_step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
        game_memory=game_mem,
        action_dict={"action_id": vector_act},
    )

    # Invariant: Must NOT falsify or sever candidate
    assert eval_res.falsification_detected is False
    assert eval_res.candidate_severed is False
    assert eval_res.verdict == Verdict.UNDECIDED
    assert eval_res.evidence_needed is True
    assert eval_res.evidence_hint == f"probe_{modal_act}"


@given(
    vector_act=st.sampled_from(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]),
)
def test_pbt_dynamic_reprobing_vector_zero_response(vector_act: str):
    """Invariant: When a vector action yields zero visible response, get_dynamic_reprobes

    schedules a mode switch (ACTION5/ACTION6) followed by re-testing the vector direction.
    """
    mgr = PrimitiveProbeManager(max_probes=16)
    reprobes = mgr.get_dynamic_reprobes(vector_act, "no visible effect (0 cells changed)", max_steps=2)

    assert len(reprobes) == 2
    # First action must be a modal switch
    assert reprobes[0].action_id in ("ACTION5", "ACTION6")
    # Second action must re-test the vector direction
    assert reprobes[1].action_id == vector_act


def test_brusentsov_judge_mode_change_zero_delta_non_vacuity():
    """Invariant: Lack of effect upon mode change must be evaluated as Verdict.OMIT or Verdict.UNDECIDED,

    strictly preventing vacuous truth (Verdict.FOLLOW).
    """
    config = V10Config()
    verifier = LayeredVerifier(config)

    step = GroundedStep(
        step_id="step_modal_test",
        dsl_function="action5",
        arguments={},
        # Assert object identity preserved (trivially contained in zero delta)
        expected_propositions=PropositionSet([
            AtomicProposition(family="object_identity", subject_id="obj_1", predicate="preserved")
        ]),
    )

    grid = [[0, 1], [0, 0]]
    snap_before = MockSnapshot(grid)
    after_obs = {"grid": grid, "state": "RUNNING"}

    pset = MagicMock(spec=PlanningSet)
    pset.grid = grid
    pset.grid_hash = "h"
    pset.allowed_action_ids = ["ACTION1", "ACTION2", "ACTION5", "RESET"]
    pset.objects = []

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap_before,
        after_obs=after_obs,
        planning_set=pset,
        action_dict={"action_id": "ACTION5"},
    )

    # Invariant: Must NOT be vacuous FOLLOW
    assert judgment.verdict != Verdict.FOLLOW
    assert judgment.verdict in (Verdict.OMIT, Verdict.UNDECIDED)
    assert judgment.is_effective is False
