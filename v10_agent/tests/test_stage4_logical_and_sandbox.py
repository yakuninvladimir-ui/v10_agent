"""Unit tests for Stage 4: Logical Invariants, State Machine, DTOs, and Perception Config."""

from __future__ import annotations

import pytest
from v10_agent.brusentsov_logic import (
    Ternary,
    contradicts,
    evaluate_invariant_across_levels,
)
from v10_agent.config import PerceptionConfig, V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import GameMemory, StructuredInvariant
from v10_agent.session import GameSession, PhaseTransition, SessionPhase
from v10_agent.types import (
    AtomicProposition,
    CoderInput,
    PropositionSet,
    SolverInput,
)
from v10_agent.universal_invariants import (
    DiscoveredInvariant,
    compare_invariants_across_levels,
)


def test_contradicts_spatial_position_and_area_conservation():
    """Verify contradicts() checks spatial_position and area_conservation contradictions."""
    # 1. Spatial position contradiction
    exp_pos = AtomicProposition(
        family="spatial_position", subject_id="obj_1", predicate="at", value=(5, 5)
    )
    obs_pos_diff = AtomicProposition(
        family="spatial_position", subject_id="obj_1", predicate="at", value=(10, 10)
    )
    obs_pos_same = AtomicProposition(
        family="spatial_position", subject_id="obj_1", predicate="at", value=(5, 5)
    )
    assert contradicts(exp_pos, obs_pos_diff) is True
    assert contradicts(exp_pos, obs_pos_same) is False

    # 2. Area conservation contradiction
    exp_area = AtomicProposition(
        family="area_conservation", subject_id="obj_2", predicate="area", value=12
    )
    obs_area_diff = AtomicProposition(
        family="area_conservation", subject_id="obj_2", predicate="area", value=8
    )
    obs_area_same = AtomicProposition(
        family="area_conservation", subject_id="obj_2", predicate="area", value=12
    )
    assert contradicts(exp_area, obs_area_diff) is True
    assert contradicts(exp_area, obs_area_same) is False


def test_evaluate_invariant_across_levels_kinematics():
    """Verify evaluate_invariant_across_levels confirms or falsifies kinematics under ternary logic."""
    inv = StructuredInvariant(
        invariant_id="inv_kin_1",
        invariant_type="kinematics",
        tier=1,
        description="Action ACTION1 moves entity dy=-1",
        metadata={"action_id": "ACTION1"},
    )

    # TRUE case: motion observed
    props_motion = PropositionSet([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=-1),
    ])
    assert evaluate_invariant_across_levels(inv, props_motion) == Ternary.TRUE

    # FALSE case: blocked zero displacement observed when motion expected
    props_blocked = PropositionSet([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=0),
    ])
    assert evaluate_invariant_across_levels(inv, props_blocked) == Ternary.FALSE

    # IRRELEVANT case: no relevant propositions
    props_empty = PropositionSet([])
    assert evaluate_invariant_across_levels(inv, props_empty) == Ternary.IRRELEVANT


def test_evaluate_invariant_across_levels_area_conservation():
    """Verify evaluate_invariant_across_levels checks area conservation against object identity."""
    inv = StructuredInvariant(
        invariant_id="inv_area_1",
        invariant_type="area_conservation",
        tier=1,
        description="Area conserved during interactions",
    )

    props_ok = PropositionSet([
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="area", value=10),
    ])
    assert evaluate_invariant_across_levels(inv, props_ok) == Ternary.TRUE

    props_destroyed = PropositionSet([
        AtomicProposition(family="object_identity", subject_id="obj_0", predicate="destroyed"),
        AtomicProposition(family="attribute_delta", subject_id="obj_0", predicate="area", value=0),
    ])
    assert evaluate_invariant_across_levels(inv, props_destroyed) == Ternary.FALSE


def test_compare_invariants_across_levels_multi_instances():
    """Verify compare_invariants_across_levels handles multiple invariants of same type without collapsing."""
    inv_a1 = DiscoveredInvariant(
        invariant_type="axial_symmetry_vertical",
        subject_id="obj_1",
        target_id="obj_2",
        axis_coordinate=5.0,
        description="Vertical symmetry across axis 5.0",
    )
    inv_a2 = DiscoveredInvariant(
        invariant_type="axial_symmetry_vertical",
        subject_id="obj_3",
        target_id="obj_4",
        axis_coordinate=15.0,
        description="Vertical symmetry across axis 15.0",
    )
    inv_sock = DiscoveredInvariant(
        invariant_type="socket_coverage",
        subject_id="obj_1",
        target_id="obj_5",
        target_position=(10.0, 10.0),
        description="Socket coverage target (10, 10)",
    )

    prev_invs = [inv_a1, inv_a2]
    curr_invs = [inv_a1, inv_sock]

    res = compare_invariants_across_levels(prev_invs, curr_invs)
    # Both inv_a1 and inv_a2 should be recognized distinctly
    assert len(res["persistent"]) == 1
    assert res["persistent"][0].axis_coordinate == 5.0

    assert len(res["new"]) == 1
    assert res["new"][0].invariant_type == "socket_coverage"

    assert len(res["disappeared"]) == 1
    assert res["disappeared"][0].axis_coordinate == 15.0


def test_session_phase_state_machine():
    """Verify SessionPhase transitions and PhaseTransition validation."""
    assert PhaseTransition.can_transition(SessionPhase.PROBING, SessionPhase.CODING) is True
    assert PhaseTransition.can_transition(SessionPhase.PROBING, SessionPhase.FALLBACK) is True
    assert PhaseTransition.can_transition(SessionPhase.CODING, SessionPhase.SOLVING) is True
    assert PhaseTransition.can_transition(SessionPhase.SOLVING, SessionPhase.EXECUTING) is True

    session = GameSession(config=V10Config(llm_advisor_backend="fake"))
    assert session.current_phase == SessionPhase.PROBING
    assert session.probing_phase is True

    session.transition_to(SessionPhase.CODING, "start coding")
    assert session.current_phase == SessionPhase.CODING
    assert session.probing_phase is False

    session.transition_to(SessionPhase.SOLVING, "start solving")
    assert session.current_phase == SessionPhase.SOLVING

    session.transition_to(SessionPhase.EXECUTING, "execute steps")
    assert session.current_phase == SessionPhase.EXECUTING

    # Setting probing_phase = True should transition back to PROBING
    session.probing_phase = True
    assert session.current_phase == SessionPhase.PROBING


def test_typed_dtos_solver_and_coder_input():
    """Verify SolverInput and CoderInput construction, validation, and immutability."""
    gm = GameMemory(game_id="g1")
    inv = gm.record_stratified_invariant("Action ACTION1 moves entity dy=1", tier=1)

    solver_in = SolverInput(
        planning_objects=[],
        relations=[],
        confirmed_invariants=[inv],
        past_failures=[{"failed_step": "s1", "verdict": "FALSE"}],
        game_model_summary="GAME MODEL",
        action_budget=150,
    )
    assert solver_in.action_budget == 150
    assert len(solver_in.confirmed_invariants) == 1

    # Immutable
    with pytest.raises(AttributeError):
        solver_in.action_budget = 200  # type: ignore

    coder_in = CoderInput(
        env_spec={"actions": ["ACTION1", "ACTION2"]},
        syntax_errors=[],
        available_actions=["ACTION1", "ACTION2"],
    )
    assert len(coder_in.available_actions) == 2
    with pytest.raises(AttributeError):
        coder_in.available_actions = []  # type: ignore


def test_perception_config_injection():
    """Verify PerceptionConfig values are read by LayeredVerifier and V10Config."""
    custom_perc = PerceptionConfig(
        object_match_area_tolerance=0.25,
        movement_detection_threshold=2.0,
    )
    config = V10Config(perception=custom_perc)
    assert config.perception.object_match_area_tolerance == 0.25
    assert config.perception.movement_detection_threshold == 2.0

    verifier = LayeredVerifier(config)
    assert verifier.config.perception.movement_detection_threshold == 2.0
    assert verifier.config.perception.object_match_area_tolerance == 0.25
