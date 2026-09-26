"""Stage 2: Comprehensive tests for the Empirical Invariant System.

Verifies:
1. EmpiricalInvariant lifecycle: confidence starts at 0.0, requires 3 confirmations to become active (0.3).
2. Invariant falsification penalizes confidence, increments times_falsified, and deactivates.
3. Cross-level scope demotion for falsified core laws.
4. CoreInvariantRegistry candidate management, pruning, and ASCII table formatting.
5. Invariant candidate proposition from PlanningSet with initial 0.0 confidence.
6. Transition evaluation via evaluate_invariant_after_action.
7. Legacy invariant migration into EmpiricalInvariant.
8. GameSession integration: preservation across level transitions and previous_level_registry tracking.
"""

from __future__ import annotations

import copy
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.memory_contours import (
    CoreInvariantRegistry,
    DefeatExemplar,
    EmpiricalInvariant,
    GameMemory,
    GroundedInvariant,
    StructuredInvariant,
    VictoryExemplar,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.session import GameSession
from v10_agent.universal_invariants import (
    evaluate_invariant_after_action,
    extract_observed_value_for,
    propose_invariant_candidates,
    values_match,
)


def test_empirical_invariant_confidence_and_active_state():
    """Candidates start at 0.0 confidence and need exactly 3 confirmations to become active (0.3)."""
    inv = EmpiricalInvariant(
        invariant_id="test_inv_1",
        invariant_type="area_conservation",
        abstract_description="Objects conserve area",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
    )

    assert inv.confidence == 0.0
    assert inv.is_active is False
    assert inv.times_confirmed == 0
    assert inv.times_falsified == 0
    assert inv.net_support == 0

    # Confirmation 1: 0.1
    inv.confirm(level_index=0, observed_value="conserved")
    assert inv.confidence == 0.1
    assert inv.is_active is False
    assert inv.confirmed_on_levels == [0]

    # Confirmation 2: 0.2
    inv.confirm(level_index=0, observed_value="conserved")
    assert inv.confidence == 0.2
    assert inv.is_active is False

    # Confirmation 3: 0.3 -> becomes ACTIVE
    inv.confirm(level_index=0, observed_value="conserved")
    assert inv.confidence == 0.3
    assert inv.is_active is True
    assert inv.net_support == 3

    # Falsification penalty: -0.2 confidence, times_falsified +1, is_active -> False
    inv.falsify(level_index=0, observed_value="violated")
    assert inv.confidence == 0.1
    assert inv.times_falsified == 1
    assert inv.is_active is False
    assert inv.net_support == 3 - 2 * 1  # 1


def test_scope_demotion_on_multiple_level_falsification():
    """If a CORE_GAME_LAW invariant is falsified across 2+ distinct levels, demote to LEVEL_SPECIFIC."""
    inv = EmpiricalInvariant(
        invariant_id="test_law",
        invariant_type="kinematic_law",
        abstract_description="Directional motion is unimpeded",
        subject_pattern="action==1",
        expected_value=(0, 1),
        scope="CORE_GAME_LAW",
    )

    inv.falsify(level_index=0)
    assert inv.scope == "CORE_GAME_LAW"
    assert inv.falsified_on_levels == [0]

    # Falsified again on a different level
    inv.falsify(level_index=1)
    assert inv.scope == "LEVEL_SPECIFIC"
    assert inv.falsified_on_levels == [0, 1]


def test_core_invariant_registry_prune_and_format():
    """Registry dedupes candidates, prunes severely contradicted invariants, and formats prompt table."""
    registry = CoreInvariantRegistry()

    inv1 = EmpiricalInvariant(
        invariant_id="inv_1",
        invariant_type="area_conservation",
        abstract_description="Objects maintain area",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
    )
    inv2 = EmpiricalInvariant(
        invariant_id="inv_2",
        invariant_type="symmetry",
        abstract_description="Reflect across vertical axis",
        subject_pattern="pair==1:2",
        expected_value="symmetric",
        scope="DOMAIN_PATTERN",
    )

    registry.register_candidate(inv1)
    registry.register_candidate(inv2)
    # Deduplication
    registry.register_candidate(inv1)
    assert len(registry.invariants) == 2

    # Confirm inv1 3 times
    for _ in range(3):
        registry.confirm_by_type("area_conservation", "all_objects", level_index=0, observed_value="conserved")

    assert inv1.confidence == 0.3
    assert inv1.is_active is True

    # Heavily falsify inv2 so net_support <= -3
    inv2.falsify(level_index=0)
    inv2.falsify(level_index=0)
    # times_confirmed=0, times_falsified=2 -> net_support = -4
    registry._prune()
    assert inv2 not in registry.invariants
    assert inv1 in registry.invariants

    # Prompt formatting table
    table_str = registry.format_for_prompt()
    assert "INVARIANT REGISTRY:" in table_str
    assert "INVARIANT" in table_str
    assert "TYPE" in table_str
    assert "✓" in table_str
    assert "✗" in table_str
    assert "CONF" in table_str
    assert "SCOPE" in table_str
    assert "area_conservation" in table_str


def test_propose_invariant_candidates_initial_zero_confidence():
    """Proposed invariant candidates always start with 0.0 confidence."""
    # 6x6 grid with two separated objects of color 2 and color 3
    grid = [[0] * 6 for _ in range(6)]
    grid[1][1] = 2
    grid[1][2] = 2
    grid[4][4] = 3

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2"])

    candidates = propose_invariant_candidates(pset)
    assert len(candidates) > 0

    for cand in candidates:
        assert cand.confidence == 0.0
        assert cand.times_confirmed == 0
        assert cand.times_falsified == 0
        assert cand.is_active is False

    # Check global laws are proposed
    global_ids = [c.invariant_id for c in candidates]
    assert "emp_area_conservation_global" in global_ids
    assert "emp_topology_conservation_global" in global_ids


def test_evaluate_invariant_after_action():
    """evaluate_invariant_after_action updates candidate confidence based on transitions."""
    registry = CoreInvariantRegistry()
    inv_global = EmpiricalInvariant(
        invariant_id="emp_area_conservation_global",
        invariant_type="area_conservation",
        abstract_description="Objects conserve their pixel area across transitions",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
        confidence=0.0,
    )
    registry.register_candidate(inv_global)

    # Frame 1: 2-cell object at (1, 1), (1, 2)
    grid_before = [[0] * 6 for _ in range(6)]
    grid_before[1][1] = 2
    grid_before[1][2] = 2
    snap_before = extract_arga_snapshot(grid_before)

    # Frame 2: Same 2-cell object shifted to (2, 1), (2, 2)
    grid_after = [[0] * 6 for _ in range(6)]
    grid_after[2][1] = 2
    grid_after[2][2] = 2
    snap_after = extract_arga_snapshot(grid_after)

    # Evaluate transition where area is conserved
    evaluate_invariant_after_action(
        registry=registry,
        before_snapshot=snap_before,
        after_snapshot=snap_after,
        action_id="ACTION2",
        level_index=0,
    )

    assert inv_global.times_confirmed == 1
    assert inv_global.confidence == 0.1
    assert inv_global.times_falsified == 0

    # Frame 3: Object gets overwritten/shrunk to 1 cell (violation)
    grid_shrunk = [[0] * 6 for _ in range(6)]
    grid_shrunk[2][1] = 2
    snap_shrunk = extract_arga_snapshot(grid_shrunk)

    evaluate_invariant_after_action(
        registry=registry,
        before_snapshot=snap_after,
        after_snapshot=snap_shrunk,
        action_id="ACTION2",
        level_index=0,
    )

    assert inv_global.times_falsified == 1
    assert inv_global.confidence == 0.0  # max(0.0, 0.1 - 0.2)
    assert inv_global.is_active is False


def test_legacy_invariant_migration():
    """GameMemory migrates legacy StructuredInvariant and GroundedInvariant into EmpiricalInvariant."""
    gm = GameMemory(game_id="generic_game")
    gm.structured_invariants.append(
        StructuredInvariant(
            invariant_id="legacy_s1",
            invariant_type="kinematics",
            tier=1,
            description="dy=-1, dx=0 on action1",
            confidence=0.5,
        )
    )
    gm.grounded_invariants.append(
        GroundedInvariant(
            invariant_id="legacy_g1",
            antecedent="Contact(ACTOR, Color_5)",
            consequent="LevelVictory()",
            brusentsov_type="POSITIVE_CANON",
            scope="CORE_GAME_LAW",
            confidence=0.8,
            times_confirmed=2,
        )
    )

    gm.migrate_legacy_invariants(level_index=1)

    reg_ids = [i.invariant_id for i in gm.invariant_registry.invariants]
    assert "legacy_s1" in reg_ids
    assert "legacy_g1" in reg_ids

    migrated_s = next(i for i in gm.invariant_registry.invariants if i.invariant_id == "legacy_s1")
    assert migrated_s.confidence == 0.3
    assert migrated_s.times_confirmed == 1
    assert migrated_s.confirmed_on_levels == [1]

    migrated_g = next(i for i in gm.invariant_registry.invariants if i.invariant_id == "legacy_g1")
    assert migrated_g.confidence == 0.3
    assert migrated_g.times_confirmed == 2
    assert migrated_g.scope == "CORE_GAME_LAW"


def test_session_level_transition_preserves_active_invariants():
    """GameSession initializes registry and preserves active/core invariants across level transitions."""
    config = V10Config(llm_advisor_backend="fake")
    session = GameSession(config=config)

    assert hasattr(session, "invariant_registry")
    assert isinstance(session.invariant_registry, CoreInvariantRegistry)

    # Register an active core law and a level-specific dead invariant
    core_law = EmpiricalInvariant(
        invariant_id="core_law_1",
        invariant_type="area_conservation",
        abstract_description="Objects conserve area",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
        confidence=0.5,
        times_confirmed=5,
        times_falsified=0,
    )
    level_specific = EmpiricalInvariant(
        invariant_id="level_inv_1",
        invariant_type="puzzle_rule",
        abstract_description="Level specific switch",
        subject_pattern="switch",
        expected_value=1,
        scope="LEVEL_SPECIFIC",
        confidence=0.0,
        times_confirmed=0,
        times_falsified=1,
    )

    session.invariant_registry.register_candidate(core_law)
    session.invariant_registry.register_candidate(level_specific)

    # Perform level transition
    session.handle_level_transition("level_1")

    game_mem = session.memory_manager.get_game_memory("session")
    assert game_mem.previous_level_registry is not None
    # Previous level registry had both
    prev_ids = [i.invariant_id for i in game_mem.previous_level_registry.invariants]
    assert "core_law_1" in prev_ids
    assert "level_inv_1" in prev_ids

    # Carried forward into current registry: core law preserved, inactive level-specific filtered
    curr_ids = [i.invariant_id for i in session.invariant_registry.invariants]
    assert "core_law_1" in curr_ids
    assert "level_inv_1" not in curr_ids
