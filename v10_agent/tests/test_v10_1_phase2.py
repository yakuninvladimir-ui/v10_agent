"""
Unit tests for V10.1 Phase 2:
- Strict 8-tier decision order in LayeredVerifier
- ISO-10 compliance (no NULL on mere EXPECT mismatch)
- Epistemic signal generation (Verdict.UNDECIDED)
- Fallback to Verdict.OMIT when enable_undecided_verdict=False
"""

import pytest
from v10_agent.config import V10Config
from v10_agent.brusentsov_logic import Verdict, Ternary
from v10_agent.judge import LayeredVerifier
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep
from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot


def _make_dummy_snapshot():
    grid = [[0, 1, 0], [0, 0, 0]]
    return extract_arga_snapshot(grid)


def test_tier1_terminal_win():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])
    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})

    # Terminal win: state is WIN or levels_completed increased
    after_obs = {"grid": snap.grid, "state": "WIN", "levels_completed": 1}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
    )
    assert judgment.verdict == Verdict.FOLLOW
    assert judgment.verdict == Ternary.TRUE


def test_tier2_terminal_failure():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])
    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})

    after_obs = {"grid": snap.grid, "state": "GAME_OVER"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
    )
    assert judgment.verdict == Verdict.NULL
    assert judgment.verdict == Ternary.FALSE


def test_tier3_explicit_expect_contradiction():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])

    step_destr = GroundedStep(
        step_id="s2",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="object_identity", subject_id="obj_0", predicate="destroyed")
        ]),
    )
    # observed still has obj_0 preserved
    j_destr = verifier.evaluate_transition(
        step=step_destr,
        before_snapshot=snap,
        after_obs={"grid": snap.grid, "state": "IN_PROGRESS"},
        planning_set=pset,
    )
    assert j_destr.verdict == Verdict.NULL


def test_tier4_explicit_expect_necessary_containment():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="object_identity", subject_id="obj_0", predicate="preserved")
        ]),
    )
    after_obs = {"grid": snap.grid, "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
    )
    assert judgment.verdict == Verdict.FOLLOW


def test_tier5_zero_delta_confirmed_motion():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])
    gmem = GameMemory(game_id="g1")
    gmem.record_action_effect("ACTION1", "Object moved UP dy=-1")

    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})
    after_obs = {"grid": snap.grid, "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=gmem,
        action_dict={"action_id": "ACTION1"},
    )
    assert judgment.verdict == Verdict.NULL
    assert "Motion action produced zero grid delta" in judgment.explanation


def test_tier6_zero_delta_unconfirmed_action_undecided():
    cfg = V10Config(enable_undecided_verdict=True)
    verifier = LayeredVerifier(cfg)
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION2"])
    gmem = GameMemory(game_id="g1")

    step = GroundedStep(
        step_id="s1",
        dsl_function="action2",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="delta_r", value=1)
        ]),
    )
    after_obs = {"grid": snap.grid, "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=gmem,
        action_dict={"action_id": "ACTION2"},
    )
    assert judgment.verdict == Verdict.UNDECIDED
    assert judgment.evidence_hint == "probe_ACTION2"


def test_tier6_undecided_disabled_maps_to_omit():
    cfg = V10Config(enable_undecided_verdict=False)
    verifier = LayeredVerifier(cfg)
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION2"])
    gmem = GameMemory(game_id="g1")

    step = GroundedStep(
        step_id="s1",
        dsl_function="action2",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="delta_r", value=1)
        ]),
    )
    after_obs = {"grid": snap.grid, "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=gmem,
        action_dict={"action_id": "ACTION2"},
    )
    assert judgment.verdict == Verdict.OMIT
    assert judgment.verdict == Ternary.IRRELEVANT


def test_tier6_upstream_low_confidence_undecided():
    cfg = V10Config(enable_undecided_verdict=True)
    verifier = LayeredVerifier(cfg)
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        confidence="low",
    )
    after_obs = {"grid": [[0, 0, 1], [0, 0, 0]], "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        action_dict={"action_id": "ACTION1"},
    )
    assert judgment.verdict == Verdict.UNDECIDED
    assert judgment.evidence_hint == "probe_environment"


def test_tier7_positive_certificate_from_game_memory():
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])
    gmem = GameMemory(game_id="g1")
    gmem.record_action_effect("ACTION1", "Object displaced")

    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})
    after_obs = {"grid": [[0, 0, 1], [0, 0, 0]], "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=gmem,
        action_dict={"action_id": "ACTION1"},
    )
    assert judgment.verdict == Verdict.FOLLOW
    assert "Certified action effect verified against GameMemory" in judgment.explanation


def test_iso_10_no_null_on_mere_expect_mismatch():
    """ISO-10: NULL requires physical contradiction, never mere mismatch of LLM-generated EXPECT."""
    verifier = LayeredVerifier(V10Config())
    snap = _make_dummy_snapshot()
    pset = build_planning_set(snap, ["ACTION1"])

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_nonexistent", predicate="delta_r", value=1)
        ]),
    )
    after_obs = {"grid": [[0, 0, 1], [0, 0, 0]], "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        action_dict={"action_id": "ACTION1"},
    )
    assert judgment.verdict != Verdict.NULL
    assert judgment.verdict in (Verdict.OMIT, Verdict.FOLLOW)
