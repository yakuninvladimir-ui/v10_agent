"""Unit tests for V10.1-r1 Phase 1: Verdict types, EpistemicMemory ISO-9, and PersistentObjectTracker."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import (
    BrusentsovJudgment,
    EpistemicSignal,
    Ternary,
    Verdict,
)
from v10_agent.config import V10Config
from v10_agent.memory_contours import EpistemicMemory
from v10_agent.tracker import PersistentObjectTracker, TrackedObject, jaccard_similarity, mask_jaccard
from v10_agent.types import PropositionSet


def test_verdict_ternary_bidirectional_equality():
    """Verify Verdict and Ternary seamlessly evaluate equal under == while retaining types."""
    # Equality with Ternary
    assert Verdict.FOLLOW == Ternary.TRUE
    assert Ternary.TRUE == Verdict.FOLLOW
    assert Verdict.NULL == Ternary.FALSE
    assert Ternary.FALSE == Verdict.NULL
    assert Verdict.OMIT == Ternary.IRRELEVANT
    assert Ternary.IRRELEVANT == Verdict.OMIT

    # Inequalities
    assert Verdict.FOLLOW != Ternary.FALSE
    assert Verdict.NULL != Ternary.TRUE
    assert Verdict.OMIT != Ternary.TRUE

    # UNDECIDED is epistemic, not a truth value
    assert Verdict.UNDECIDED != Ternary.TRUE
    assert Verdict.UNDECIDED != Ternary.FALSE
    assert Verdict.UNDECIDED != Ternary.IRRELEVANT
    assert Ternary.TRUE != Verdict.UNDECIDED
    assert Ternary.FALSE != Verdict.UNDECIDED
    assert Ternary.IRRELEVANT != Verdict.UNDECIDED

    # Property and factory
    assert Verdict.FOLLOW.ternary == Ternary.TRUE
    assert Verdict.NULL.ternary == Ternary.FALSE
    assert Verdict.OMIT.ternary == Ternary.IRRELEVANT
    assert Verdict.UNDECIDED.ternary is None

    assert Verdict.from_ternary(Ternary.TRUE) == Verdict.FOLLOW
    assert Verdict.from_ternary(Ternary.FALSE) == Verdict.NULL
    assert Verdict.from_ternary(Ternary.IRRELEVANT) == Verdict.OMIT


def test_brusentsov_judgment_fields_and_dict():
    """Verify BrusentsovJudgment accommodates new V10.1 metadata fields."""
    j = BrusentsovJudgment(
        trajectory_id="traj_1",
        step_id="s1",
        verdict=Verdict.UNDECIDED,
        expected_propositions=PropositionSet.empty(),
        observed_propositions=PropositionSet.empty(),
        explanation="Low confidence track",
        ambiguity_score=0.12,
        evidence_hint="probe_ACTION1",
        matching_candidates=["trk_0001", "trk_0002"],
        track_confidence_min=0.45,
    )
    d = j.to_dict()
    assert d["verdict"] == "UNDECIDED"
    assert d["ambiguity_score"] == 0.12
    assert d["evidence_hint"] == "probe_ACTION1"
    assert d["matching_candidates"] == ["trk_0001", "trk_0002"]
    assert d["track_confidence_min"] == 0.45


def test_epistemic_memory_iso9_undecided_isolation():
    """Verify ISO-9: UNDECIDED judgments are routed to epistemic_signals and not committed judgments."""
    mem = EpistemicMemory(level_id="level_0")

    j_follow = BrusentsovJudgment(
        trajectory_id="t1",
        step_id="s1",
        verdict=Verdict.FOLLOW,
        expected_propositions=PropositionSet.empty(),
        observed_propositions=PropositionSet.empty(),
        explanation="Follow step",
    )
    j_undecided = BrusentsovJudgment(
        trajectory_id="t1",
        step_id="s2",
        verdict=Verdict.UNDECIDED,
        expected_propositions=PropositionSet.empty(),
        observed_propositions=PropositionSet.empty(),
        explanation="Undecided step",
    )

    mem.record_judgment(j_follow)
    mem.record_judgment(j_undecided)

    # j_follow is in committed judgments
    assert len(mem.judgments) == 1
    assert mem.judgments[0].verdict == Verdict.FOLLOW

    # j_undecided is isolated in epistemic_signals (ISO-9)
    assert len(mem.epistemic_signals) == 1
    assert mem.epistemic_signals[0].verdict == Verdict.UNDECIDED

    # Clear resets both
    mem.clear()
    assert len(mem.judgments) == 0
    assert len(mem.epistemic_signals) == 0


def test_tracker_identity_stability_across_translation():
    """Verify PersistentObjectTracker maintains stable persistent_id across a 5-step translation."""
    cfg = V10Config(track_match_threshold=0.45)
    tracker = PersistentObjectTracker(cfg)

    # 10x10 grid with background 0 and a 2x2 object moving right
    persistent_ids = []
    for frame_idx in range(5):
        grid = [[0] * 10 for _ in range(10)]
        # Object moves 1 column right each frame
        col_offset = frame_idx
        grid[2][col_offset + 1] = 3
        grid[2][col_offset + 2] = 3
        grid[3][col_offset + 1] = 3
        grid[3][col_offset + 2] = 3

        snapshot = extract_arga_snapshot(grid)
        tracked = tracker.update(snapshot, frame_index=frame_idx)
        assert len(tracked) == 1
        persistent_ids.append(tracked[0].persistent_id)

    # Exactly the same persistent_id across all 5 translation frames
    assert all(pid == "trk_0001" for pid in persistent_ids)

    # Verify velocity EMA and history
    final_track = tracker.get_track("trk_0001")
    assert final_track is not None
    assert final_track.last_frame_id == 4
    # Horizontal motion velocity should be positive in col dimension
    assert final_track.velocity[1] > 0.0
    assert len(final_track.history) == 4  # max length = cumulative_window + 1 = 4
    assert final_track.cumulative_delta[1] > 0.0  # moved right cumulatively


def test_tracker_occlusion_and_recovery():
    """Verify occlusion detection heuristic when a smaller object is hidden by a larger one."""
    cfg = V10Config(occlusion_radius=3)
    tracker = PersistentObjectTracker(cfg)

    # Frame 0: Small object (color 2, 1x2 area 2) at (4, 4)
    grid0 = [[0] * 10 for _ in range(10)]
    grid0[4][4] = 2
    grid0[4][5] = 2
    snap0 = extract_arga_snapshot(grid0)
    tracker.update(snap0, frame_index=0)
    assert len(tracker.get_tracked()) == 1
    small_id = tracker.get_tracked()[0].persistent_id

    # Frame 1: A larger object (color 4, 3x3 area 9) appears over (3..5, 3..5) and small is covered
    grid1 = [[0] * 10 for _ in range(10)]
    for r in range(3, 6):
        for c in range(3, 6):
            grid1[r][c] = 4
    snap1 = extract_arga_snapshot(grid1)
    tracker.update(snap1, frame_index=1)

    # Small object should be marked occluded rather than killed immediately
    small_track = tracker.get_track(small_id)
    assert small_track is not None
    assert small_track.occluded is True
    assert small_track.occluded_frames == 1

    # Frame 2: Small object re-emerges at (4, 7)
    grid2 = [[0] * 10 for _ in range(10)]
    for r in range(3, 6):
        for c in range(3, 6):
            grid2[r][c] = 4
    grid2[4][7] = 2
    grid2[4][8] = 2
    snap2 = extract_arga_snapshot(grid2)
    tracker.update(snap2, frame_index=2)

    small_track_recovered = tracker.get_track(small_id)
    assert small_track_recovered is not None
    assert small_track_recovered.occluded is False
    assert small_track_recovered.occluded_frames == 0


def test_tracker_shape_stability_decay():
    """Verify shape_stability_score decays when object geometry significantly morphs."""
    tracker = PersistentObjectTracker()

    # Frame 0: 2x2 square
    grid0 = [[0] * 10 for _ in range(10)]
    grid0[2][2] = 5
    grid0[2][3] = 5
    grid0[3][2] = 5
    grid0[3][3] = 5
    tracker.update(extract_arga_snapshot(grid0), frame_index=0)
    track = tracker.get_track("trk_0001")
    assert track.shape_stability_score == 1.0

    # Frame 1: In-place morph into long line (color same, centroid similar, but shape completely changed)
    grid1 = [[0] * 10 for _ in range(10)]
    grid1[2][1] = 5
    grid1[2][2] = 5
    grid1[2][3] = 5
    grid1[2][4] = 5
    tracker.update(extract_arga_snapshot(grid1), frame_index=1)

    track_morphed = tracker.get_track("trk_0001")
    assert track_morphed is not None
    # Shape stability decayed
    assert track_morphed.shape_stability_score < 1.0


def test_tracker_clean_reset():
    """Verify reset() restores tracker to a clean pristine state."""
    tracker = PersistentObjectTracker()
    grid = [[0] * 10 for _ in range(10)]
    grid[1][1] = 2
    tracker.update(extract_arga_snapshot(grid), frame_index=0)
    assert len(tracker.get_tracked()) == 1

    tracker.reset()
    assert len(tracker.get_tracked()) == 0
    assert tracker.next_track_num == 1
    assert tracker.frame_index == 0
