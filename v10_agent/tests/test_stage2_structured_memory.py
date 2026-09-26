"""Comprehensive unit tests validating Stage 2: Structured Memory & Invariant Lifecycle."""

from __future__ import annotations

import pytest

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary
from v10_agent.memory_contours import (
    EpistemicMemory,
    GameMemory,
    MemoryContourManager,
    StructuredInvariant,
    _classify_tier,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.types import AtomicProposition, PropositionSet


def test_structured_invariant_dataclass():
    """2.1: Verify StructuredInvariant properties and serialization."""
    inv = StructuredInvariant(
        invariant_id="inv_1_kinematics",
        invariant_type="kinematics",
        tier=1,
        description="Action ACTION1 moves entity UP by dy=-1",
        confidence=0.5,
        ternary_status=Ternary.IRRELEVANT,
        confirmed_on_levels=["level_0"],
    )
    assert inv.invariant_id == "inv_1_kinematics"
    assert inv.tier == 1
    assert inv.ternary_status == Ternary.IRRELEVANT
    d = inv.to_dict()
    assert d["invariant_type"] == "kinematics"
    assert d["confidence"] == 0.5
    assert d["ternary_status"] == "IRRELEVANT"
    assert d["confirmed_on_levels"] == ["level_0"]


def test_confirm_and_falsify_invariant_lifecycle():
    """2.2: Test invariant confirmation promotion to Ternary.TRUE and falsification to Ternary.FALSE."""
    gm = GameMemory(game_id="test_game")
    inv = gm.record_stratified_invariant(
        "Action ACTION1 moves piece UP",
        tier=1,
        invariant_type="kinematics",
        confidence=0.3,
    )
    assert inv is not None
    inv_id = inv.invariant_id
    assert inv.ternary_status == Ternary.IRRELEVANT
    assert inv.confidence == 0.3

    # Confirm on level 1: confidence increases
    gm.confirm_invariant(inv_id, level_id="level_1")
    assert inv.confidence == pytest.approx(0.5)
    assert "level_1" in inv.confirmed_on_levels

    # Confirm on level 2: promoted to Ternary.TRUE
    gm.confirm_invariant(inv_id, level_id="level_2")
    assert inv.confidence == pytest.approx(0.7)
    assert inv.ternary_status == Ternary.TRUE

    # Falsify on level 3: demoted to Ternary.FALSE
    gm.falsify_invariant(inv_id, level_id="level_3")
    assert inv.confidence == pytest.approx(0.4)
    assert inv.ternary_status == Ternary.FALSE
    assert "level_3" in inv.falsified_on_levels


def test_role_mapping_generalization():
    """2.3: Verify confirmed actors → [ACTOR], destinations → [TARGET], unknown → [ENTITY]."""
    gm = GameMemory(game_id="test_game")
    gm.record_action_effect("ACTION1", "moved obj_1 toward obj_5 by dy=-1")
    gm.record_selection_mechanic("ACTION5 toggles focus to obj_2")

    # obj_1 and obj_2 are confirmed actors
    assert "obj_1" in gm.confirmed_actors
    assert "obj_2" in gm.confirmed_actors

    # obj_5 is mentioned as destination via "toward" → should be TARGET
    role_map = gm._build_role_map()
    assert role_map.get("obj_5") == "TARGET"

    raw_text = "obj_1 moved toward obj_5 and obj_9 aligned with obj_2"
    gen_text = gm._generalize_text(raw_text)

    # obj_1 and obj_2 → [ACTOR], obj_5 → [TARGET], obj_9 → [ENTITY]
    assert "[ACTOR] moved toward [TARGET]" in gen_text
    assert "[ENTITY] aligned with [ACTOR]" in gen_text
    assert "obj_1" not in gen_text
    assert "obj_2" not in gen_text
    assert "obj_5" not in gen_text
    assert "obj_9" not in gen_text


def test_cumulative_tier3_memory_across_level_transitions():
    """2.4: Verify that domain-general Tier 3 rules survive level transition while transient ones are dropped."""
    gm = GameMemory(game_id="test_game")

    # Domain-general Tier 3 invariant
    gm.record_stratified_invariant(
        "Mirrored entities across symmetry axis must be preserved to win",
        tier=3,
        invariant_type="goal",
    )

    # Transient level-specific rule
    gm.record_stratified_invariant(
        "Level 0 specific rule: target is at row 12, col 45 (needs 5 steps)",
        tier=3,
        invariant_type="goal",
    )

    assert len(gm.tier3_level_rules) == 2

    # Transition to next level
    gm.handle_level_transition()

    # The general rule survives, the transient rule is filtered
    assert len(gm.tier3_level_rules) == 1
    assert "Mirrored entities across symmetry axis" in gm.tier3_level_rules[0]
    assert "row 12" not in gm.tier3_level_rules[0]


def test_auto_classification_tier_and_type():
    """2.5: Test _classify_tier heuristics for kinematics, symmetry, control, interaction, palette."""
    assert _classify_tier("moved piece by dy=2, dx=0") == (1, "kinematics")
    assert _classify_tier("axial_symmetry across vertical mirror axis") == (1, "symmetry")
    assert _classify_tier("ACTION5 selection toggle between tokens") == (2, "control")
    assert _classify_tier("stepping on socket triggers color change") == (2, "interaction")
    assert _classify_tier("indicator dots highlight active piece") == (2, "palette")
    assert _classify_tier("general puzzle solution principle") == (3, "general")

    gm = GameMemory(game_id="test_game")
    gm.record_invariant_rule("Object reflection across center boundary")
    invs = gm.get_invariants_by_type("symmetry")
    assert len(invs) == 1
    assert invs[0].tier == 1


def test_structured_failures_in_epistemic_memory():
    """2.6: Test format_structured_failures extracts NULL verdicts with explanations."""
    ep_mem = EpistemicMemory(level_id="level_0")
    # Follow step
    ep_mem.record_judgment(
        BrusentsovJudgment(
            trajectory_id="traj_1",
            step_id="s1",
            verdict=Ternary.TRUE,
            expected_propositions=PropositionSet([
                AtomicProposition(family="metric_sign", subject_id="obj_1", predicate="dy_sign", value=1)
            ]),
            observed_propositions=PropositionSet([
                AtomicProposition(family="metric_sign", subject_id="obj_1", predicate="dy_sign", value=1)
            ]),
            explanation="Step moved 1 pixel down as expected",
        )
    )
    # Contradicted (NULL) step
    ep_mem.record_judgment(
        BrusentsovJudgment(
            trajectory_id="traj_1",
            step_id="s2",
            verdict=Ternary.FALSE,
            expected_propositions=PropositionSet([
                AtomicProposition(family="metric_sign", subject_id="obj_1", predicate="dy_sign", value=1)
            ]),
            observed_propositions=PropositionSet(),
            explanation="Zero displacement observed: hit solid boundary wall",
        )
    )

    failures = ep_mem.format_structured_failures()
    assert len(failures) == 1
    assert failures[0]["trajectory_id"] == "traj_1"
    assert failures[0]["failed_step"] == "s2"
    assert failures[0]["verdict"] == "FALSE (NULLITY)"
    assert "solid boundary wall" in failures[0]["explanation"]


def test_game_model_summary_formatting():
    """2.8: Test format_game_model_summary formats confirmed, pending, and falsified counts."""
    gm = GameMemory(game_id="test_game")
    inv1 = gm.record_stratified_invariant("Action ACTION1 moves UP", tier=1, confidence=0.8)
    inv2 = gm.record_stratified_invariant("ACTION5 switches active actor", tier=2, confidence=0.4)
    inv3 = gm.record_stratified_invariant("Hitting wall changes color", tier=2, confidence=0.2)

    # inv1 is confirmed (confidence >= 0.7 -> Ternary.TRUE)
    assert inv1.ternary_status == Ternary.TRUE
    # inv2 is pending
    assert inv2.ternary_status == Ternary.IRRELEVANT
    # falsify inv3
    gm.falsify_invariant(inv3.invariant_id, level_id="level_0")
    assert inv3.ternary_status == Ternary.FALSE

    summary = gm.format_game_model_summary()
    assert "confirmed: 1" in summary
    assert "pending: 1" in summary
    assert "falsified: 1" in summary
    assert "✓ [kinematics]" in summary
    assert "? [control]" in summary


def test_dynamic_falsification_updates_structured_invariants():
    """2.9: Verify that invalidate_action_effect updates matching StructuredInvariants to Ternary.FALSE."""
    gm = GameMemory(game_id="test_game")
    gm.record_action_effect("ACTION2", "moves piece DOWN by dy=1")
    assert "ACTION2" in gm.confirmed_action_effects

    inv = next(i for i in gm.structured_invariants if "ACTION2" in i.description)
    assert inv.ternary_status == Ternary.TRUE

    # Invalidate ACTION2
    gm.invalidate_action_effect("ACTION2", level_id="level_1")
    assert "ACTION2" not in gm.confirmed_action_effects
    assert inv.ternary_status == Ternary.FALSE
    assert "level_1" in inv.falsified_on_levels


def test_solver_prompt_integration_with_stage2():
    """2.7 & 2.8: Test build_solver_prompts receives structured_failures and game_model_summary."""
    grid = [[0, 1], [1, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2"])

    ep_mem = EpistemicMemory(level_id="level_0")
    ep_mem.record_judgment(
        BrusentsovJudgment(
            trajectory_id="traj_fail",
            step_id="s1",
            verdict=Ternary.FALSE,
            expected_propositions=PropositionSet(),
            observed_propositions=PropositionSet(),
            explanation="Blocked by obstacle",
        )
    )

    gm = GameMemory(game_id="test_game")
    gm.record_stratified_invariant("Action ACTION1 moves UP by dy=-1", tier=1, confidence=0.9)

    manifest = {"functions": [{"name": "action1", "parameters": []}]}
    _, user_prompt = build_solver_prompts(
        manifest=manifest,
        planning_set=pset,
        epistemic_memory=ep_mem,
        game_memory=gm,
    )

    assert "FAILED CANDIDATES" in user_prompt
    assert "Blocked by obstacle" in user_prompt
    assert "ACCUMULATED GAME KNOWLEDGE" in user_prompt
    assert "Action ACTION1 moves UP by dy=-1" in user_prompt
