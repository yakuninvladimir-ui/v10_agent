"""
Unit tests for V10.1 Phase 3:
- Change-centric proposition families (cumulative_motion, shape_stability, occlusion)
- Persistent object grounding in PlanningSet and resolve_object_id
- LayeredVerifier UNDECIDED emission on tracker ambiguity and low track confidence
- Clean tracker reset and pure ARGA fallback ablation
"""

import pytest
from v10_agent.config import V10Config
from v10_agent.brusentsov_logic import Verdict, Ternary
from v10_agent.judge import LayeredVerifier
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot, PlanningObject
from v10_agent.tracker import PersistentObjectTracker, TrackedObject


def test_registered_proposition_families():
    """Verify cumulative_motion and shape_stability are valid proposition families."""
    p1 = AtomicProposition(
        family="cumulative_motion",
        subject_id="trk_0001",
        predicate="moved_total_over_N_frames",
        value=(0.0, 2.0),
    )
    assert p1.family == "cumulative_motion"

    p2 = AtomicProposition(
        family="shape_stability",
        subject_id="trk_0001",
        predicate="shape_stable",
        value=0.95,
    )
    assert p2.family == "shape_stability"


def test_planning_set_persistent_id_resolution():
    """Verify PlanningSet resolves objects via persistent_id as well as canonical id."""
    grid = [[1, 0, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    tracker = PersistentObjectTracker()
    pset = build_planning_set(snap, ["ACTION1"], tracker=tracker)

    # The single object should have a persistent_id assigned
    obj = pset.objects[0]
    assert obj.persistent_id == "trk_0001"
    assert obj.track_confidence == 1.0

    # resolve_object_id should resolve trk_0001 to obj.id
    resolved = pset.resolve_object_id("trk_0001")
    assert resolved == obj.id

    # get_object should retrieve the object via trk_0001
    retrieved = pset.get_object("trk_0001")
    assert retrieved is not None
    assert retrieved.id == obj.id


def test_layered_verifier_emits_change_centric_propositions():
    """Verify LayeredVerifier extracts cumulative_motion, shape_stability, and identity propositions."""
    cfg = V10Config(enable_persistent_tracker=True)
    verifier = LayeredVerifier(cfg)

    grid1 = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    grid2 = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    snap1 = extract_arga_snapshot(grid1)
    pset1 = build_planning_set(snap1, ["ACTION1"], tracker=verifier.tracker)

    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap1,
        after_obs={"grid": grid2, "state": "IN_PROGRESS"},
        planning_set=pset1,
    )

    observed_props = judgment.observed_propositions
    families = {p.family for p in observed_props}

    assert "cumulative_motion" in families
    assert "shape_stability" in families
    assert "object_identity" in families

    # Verify persistent subject_id exists in cumulative_motion
    cm_props = [p for p in observed_props if p.family == "cumulative_motion"]
    assert len(cm_props) >= 1
    assert cm_props[0].subject_id.startswith("trk_")


def test_tracker_ambiguity_triggers_undecided():
    """Verify ambiguous matching cost (< matching_ambiguity_threshold) triggers Verdict.UNDECIDED."""
    cfg = V10Config(
        enable_persistent_tracker=True,
        enable_undecided_verdict=True,
        matching_ambiguity_threshold=0.20,
    )
    verifier = LayeredVerifier(cfg)

    # Frame 1: 1 red object at col 5
    grid1 = [[0] * 20]
    grid1[0][5] = 2
    snap1 = extract_arga_snapshot(grid1)
    pset = build_planning_set(snap1, ["ACTION1"], tracker=verifier.tracker)

    # Frame 2: 2 identical red objects equidistant from col 5 (at col 4 and col 6)
    grid2 = [[0] * 20]
    grid2[0][4] = 2
    grid2[0][6] = 2

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="col_delta", value=1)
        ]),
    )

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap1,
        after_obs={"grid": grid2, "state": "IN_PROGRESS"},
        planning_set=pset,
    )

    assert judgment.verdict == Verdict.UNDECIDED
    assert judgment.ambiguity_score is not None
    assert judgment.ambiguity_score < 0.20
    assert judgment.evidence_hint == "probe_motion"


def test_low_tracking_confidence_triggers_undecided():
    """Verify low tracking confidence on participating track triggers Verdict.UNDECIDED."""
    cfg = V10Config(
        enable_persistent_tracker=True,
        enable_undecided_verdict=True,
        track_confidence_threshold=0.75,
    )
    verifier = LayeredVerifier(cfg)

    grid1 = [[1, 0, 0], [0, 0, 0]]
    snap1 = extract_arga_snapshot(grid1)
    pset = build_planning_set(snap1, ["ACTION1"], tracker=verifier.tracker)

    # Ensure active track retains low confidence after update
    orig_update = verifier.tracker.update
    def mock_update(*args, **kwargs):
        res = orig_update(*args, **kwargs)
        for t in verifier.tracker.tracks.values():
            t.confidence = 0.40
        return res
    verifier.tracker.update = mock_update

    step = GroundedStep(
        step_id="s1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="object_identity", subject_id="trk_0001", predicate="preserved")
        ]),
    )

    after_obs = {"grid": [[0, 1, 0], [0, 0, 0]], "state": "IN_PROGRESS"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap1,
        after_obs=after_obs,
        planning_set=pset,
    )

    assert judgment.verdict == Verdict.UNDECIDED
    assert judgment.track_confidence_min == 0.40
    assert judgment.evidence_hint == "probe_tracking"


def test_pure_arga_ablation_flag():
    """When enable_persistent_tracker=False, verifier operates in pure ARGA mode without tracker."""
    cfg = V10Config(enable_persistent_tracker=False)
    verifier = LayeredVerifier(cfg)
    assert verifier.tracker is None

    grid = [[1, 0, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1"])

    step = GroundedStep(step_id="s1", dsl_function="action1", arguments={})
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs={"grid": [[0, 1, 0], [0, 0, 0]], "state": "IN_PROGRESS"},
        planning_set=pset,
    )

    assert judgment.verdict in (Verdict.FOLLOW, Verdict.OMIT)
    families = {p.family for p in judgment.observed_propositions}
    assert "cumulative_motion" not in families


def test_clean_tracker_reset_on_environmental_reset():
    """Verify verifier.reset() cleans all tracker state and resets ID numbering."""
    cfg = V10Config(enable_persistent_tracker=True)
    verifier = LayeredVerifier(cfg)

    grid = [[1, 0, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1"], tracker=verifier.tracker)
    assert len(verifier.tracker.tracks) >= 1

    # Reset
    verifier.reset()
    assert len(verifier.tracker.tracks) == 0
    assert verifier.tracker.next_track_num == 1
    assert verifier.tracker.frame_index == 0
