"""
Unit tests for V10.1 Phase 4:
- SymbolicTrajectoryExecutor reaction to Verdict.UNDECIDED
- Evidence-seeking loop in GameSession
- Streak limit fallback to NULL
- Resolution of ambiguity via probe
- Strict NOOP prohibition (never emit NOOP)
- Telemetry counter verification
"""

import pytest
from v10_agent.config import V10Config
from v10_agent.brusentsov_logic import Verdict, Ternary
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.sandbox import SandboxExecutor
from v10_agent.verification import VerificationBinder
from v10_agent.session import GameSession
from v10_agent.memory_contours import EpistemicMemory, GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot


def test_symbolic_executor_undecided_does_not_advance_or_sever():
    """SymbolicTrajectoryExecutor on Verdict.UNDECIDED leaves candidate cursor unchanged and sets evidence_needed=True."""
    cfg = V10Config(enable_undecided_verdict=True)
    from v10_agent.judge import LayeredVerifier
    executor = SymbolicTrajectoryExecutor(cfg, SandboxExecutor(), VerificationBinder(), LayeredVerifier(cfg))

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        confidence="low",  # Triggers UNDECIDED in LayeredVerifier
    )
    cand = CandidateTrajectory(
        trajectory_id="traj_1",
        steps=[{"dsl_function": "action1", "arguments": {}}],
        cursor=0,
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand])
    ep_mem = EpistemicMemory(level_id="lvl_1")

    snap = extract_arga_snapshot([[1, 0], [0, 0]])
    pset = build_planning_set(snap, ["ACTION1"])

    res = executor.evaluate_transition(
        pending_step=step,
        before_snapshot=snap,
        after_obs={"grid": [[0, 1], [0, 0]], "state": "IN_PROGRESS"},
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )

    assert res.verdict == Verdict.UNDECIDED
    assert res.evidence_needed is True
    assert res.candidate_advanced is False
    assert res.candidate_severed is False
    assert cand.cursor == 0
    assert not cand.is_severed

    # ISO-9: UNDECIDED routed to epistemic_signals, not judgments
    assert len(ep_mem.epistemic_signals) == 1
    assert len(ep_mem.judgments) == 0


def test_session_evidence_seeking_dispatches_probe_and_never_emits_noop():
    """GameSession enqueues targeted probe on UNDECIDED and act() emits it without emitting NOOP."""
    cfg = V10Config(
        enable_undecided_verdict=True,
        max_evidence_probes_per_level=2,
    )
    session = GameSession(cfg)

    grid = [[1, 0], [0, 0]]
    obs = {"grid": grid, "state": "NOT_STARTED", "action_space": {"actions": ["ACTION1", "ACTION2", "RESET"]}}

    # Step into session
    session.observe_action_result(obs)

    # Simulate an active step that produces UNDECIDED
    step = GroundedStep(
        step_id="s1",
        dsl_function="action2",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="delta_r", value=1)
        ]),
    )
    session.last_snapshot = extract_arga_snapshot(grid)
    session.last_planning_set = build_planning_set(session.last_snapshot, ["ACTION1", "ACTION2", "RESET"])
    session.pending_step = step
    session.pending_action = {"action_id": "ACTION2", "id": "ACTION2", "data": {}}

    cand = CandidateTrajectory(
        trajectory_id="traj_1",
        steps=[{"dsl_function": "action2", "arguments": {}}],
        cursor=0,
    )
    session.active_pool = TrajectoryPool(proposal_id="p1", candidates=[cand])

    # Feed transition with zero grid delta on unconfirmed ACTION2 -> triggers UNDECIDED
    session.observe_action_result({"grid": grid, "state": "IN_PROGRESS", "action_space": {"actions": ["ACTION1", "ACTION2", "RESET"]}})

    assert session.evidence_seeking_active is True
    assert session.undecided_streak == 1
    assert session.pending_step_snapshot is not None
    assert len(session.probe_queue) >= 1

    # Call act(): MUST return the enqueued probe, NEVER NOOP
    next_action = session.act({"grid": grid, "state": "IN_PROGRESS", "action_space": {"actions": ["ACTION1", "ACTION2", "RESET"]}})
    assert next_action is not None
    assert next_action["action_id"] != "NOOP"
    assert next_action["action_id"] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET")
    assert session.evidence_probes_executed == 1


def test_session_streak_limit_falls_back_to_null():
    """When streak limit is reached or probe budget exhausted, GameSession treats UNDECIDED as NULL."""
    cfg = V10Config(
        enable_undecided_verdict=True,
        max_evidence_probes_per_level=1,
        max_undecided_streak=1,
    )
    session = GameSession(cfg)

    grid = [[1, 0], [0, 0]]
    session.observe_action_result({"grid": grid, "state": "NOT_STARTED", "action_space": {"actions": ["ACTION1", "RESET"]}})

    # Setup pending step that produces UNDECIDED
    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        confidence="low",
    )
    session.last_snapshot = extract_arga_snapshot(grid)
    session.last_planning_set = build_planning_set(session.last_snapshot, ["ACTION1", "RESET"])
    session.pending_step = step
    session.pending_action = {"action_id": "ACTION1", "id": "ACTION1", "data": {}}

    cand = CandidateTrajectory(
        trajectory_id="traj_1",
        steps=[{"dsl_function": "action1", "arguments": {}}],
        cursor=0,
    )
    session.active_pool = TrajectoryPool(proposal_id="p1", candidates=[cand])

    # Exhaust remaining probes to 0
    session.evidence_probes_remaining = 0

    session.observe_action_result({"grid": grid, "state": "IN_PROGRESS", "action_space": {"actions": ["ACTION1", "RESET"]}})

    # Should fall back to NULL immediately: candidate severed, solver reset pending, no replan requested
    assert session.evidence_seeking_active is False
    assert cand.is_severed is True
    assert session.solver_reset_pending is True
    assert session.replan_requested is False
    assert session.undecided_fallback_to_null >= 1


def test_telemetry_records_all_v10_1_fields():
    """Harness telemetry includes undecided counts and epistemic signal metrics."""
    cfg = V10Config()
    session = GameSession(cfg)

    telem = session.harness_telemetry()
    assert "undecided_count" in telem
    assert "undecided_resolved_by_probe" in telem
    assert "undecided_fallback_to_null" in telem
    assert "evidence_probes_executed" in telem
    assert "epistemic_signals_count" in telem
