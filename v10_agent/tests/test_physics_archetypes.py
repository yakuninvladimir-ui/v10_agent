"""Tests for Universal Physics & Topological Archetypes in universal_invariants.

Verifies SlidingKinematics, SokobanPush, MarkerCollection, and ToggleTrigger
empirical invariant evaluation across action transitions.
"""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.memory_contours import CoreInvariantRegistry, EmpiricalInvariant
from v10_agent.planning_set import build_planning_set
from v10_agent.universal_invariants import (
    MarkerCollection,
    SlidingKinematics,
    SokobanPush,
    ToggleTrigger,
    evaluate_invariant_after_action,
    propose_invariant_candidates,
)


def test_archetype_dataclasses_serialization():
    """Verify serialization to dict for all newly introduced physics archetypes."""
    slide = SlidingKinematics(subject_id="obj_1", direction=(0, 1), obstacle_id="wall_1", confidence=0.7)
    assert slide.to_dict() == {
        "type": "sliding_kinematics",
        "subject": "obj_1",
        "direction": (0, 1),
        "obstacle": "wall_1",
        "confidence": 0.7,
    }

    push = SokobanPush(actor_id="actor_1", pushed_id="block_1", direction=(1, 0), confidence=0.8)
    assert push.to_dict() == {
        "type": "sokoban_push",
        "actor": "actor_1",
        "pushed": "block_1",
        "direction": (1, 0),
        "confidence": 0.8,
    }

    collect = MarkerCollection(collector_id="actor_1", target_role_color=3, target_id="target_1", confidence=0.9)
    assert collect.to_dict() == {
        "type": "marker_collection",
        "collector": "actor_1",
        "target_role_color": 3,
        "target_id": "target_1",
        "confidence": 0.9,
    }

    toggle = ToggleTrigger(trigger_id="switch_1", barrier_id="gate_1", toggle_state="open", confidence=0.6)
    assert toggle.to_dict() == {
        "type": "toggle_trigger",
        "trigger": "switch_1",
        "barrier": "gate_1",
        "toggle_state": "open",
        "confidence": 0.6,
    }


def test_sliding_kinematics_empirical_evaluation():
    """Verify that an object moving > 1 pixel in a single step confirms sliding_kinematics."""
    grid_before = [[0] * 10 for _ in range(10)]
    grid_before[2][2] = 4

    # Object slides from col 2 to col 6 (displacement = 4)
    grid_after = [[0] * 10 for _ in range(10)]
    grid_after[2][6] = 4

    snap_b = extract_arga_snapshot(grid_before)
    snap_a = extract_arga_snapshot(grid_after)
    pset = build_planning_set(snap_b, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    candidates = propose_invariant_candidates(pset)
    slide_cands = [c for c in candidates if c.invariant_type == "sliding_kinematics"]
    assert len(slide_cands) > 0
    slide_cand = slide_cands[0]
    assert slide_cand.confidence == 0.0

    registry = CoreInvariantRegistry()
    registry.register_candidate(slide_cand)

    evaluate_invariant_after_action(registry, snap_b, snap_a, level_index=0)
    assert slide_cand.times_confirmed == 1
    assert slide_cand.confidence > 0.0


def test_sokoban_push_empirical_evaluation():
    """Verify that actor moving into adjacent object and both shifting confirms sokoban_push."""
    grid_before = [[0] * 10 for _ in range(10)]
    grid_before[3][3] = 1  # Actor
    grid_before[3][4] = 2  # Adjacent block

    grid_after = [[0] * 10 for _ in range(10)]
    grid_after[3][4] = 1  # Actor advanced right
    grid_after[3][5] = 2  # Block displaced right

    snap_b = extract_arga_snapshot(grid_before)
    snap_a = extract_arga_snapshot(grid_after)
    pset = build_planning_set(snap_b, ["ACTION4"])

    candidates = propose_invariant_candidates(pset)
    push_cands = [c for c in candidates if c.invariant_type == "sokoban_push"]
    assert len(push_cands) > 0
    push_cand = push_cands[0]
    assert push_cand.confidence == 0.0

    registry = CoreInvariantRegistry()
    registry.register_candidate(push_cand)

    evaluate_invariant_after_action(registry, snap_b, snap_a, level_index=0)
    assert push_cand.times_confirmed == 1
    assert push_cand.confidence > 0.0


def test_marker_collection_empirical_evaluation():
    """Verify that an actor moving onto a marker causes it to be marked collected."""
    grid_before = [[0] * 10 for _ in range(10)]
    grid_before[2][2] = 1  # Actor
    grid_before[2][3] = 5  # Marker

    grid_after = [[0] * 10 for _ in range(10)]
    grid_after[2][3] = 1  # Actor occupies marker position, marker 5 disappeared

    snap_b = extract_arga_snapshot(grid_before)
    snap_a = extract_arga_snapshot(grid_after)
    pset = build_planning_set(snap_b, ["ACTION4"])

    candidates = propose_invariant_candidates(pset)
    marker_obj = next(o for o in snap_b.objects if o.color == 5)
    collect_cands = [c for c in candidates if c.invariant_type == "marker_collection"]
    assert len(collect_cands) > 0
    m_cand = next(c for c in collect_cands if c.subject_pattern == f"id=={marker_obj.id}")

    registry = CoreInvariantRegistry()
    registry.register_candidate(m_cand)

    evaluate_invariant_after_action(registry, snap_b, snap_a, level_index=0)
    assert m_cand.times_confirmed == 1
    assert m_cand.confidence > 0.0
