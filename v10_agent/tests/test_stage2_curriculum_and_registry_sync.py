"""Unit and integration tests for Stage 2: Curriculum history & Invariant Registry sync.

Verifies:
1. Synchronization of apply_invariant_revision with CoreInvariantRegistry, palette roles, and GroundedInvariants.
2. Recognition and parsing of [NEGATIVE_BARRIER], [POSITIVE_CANON], [PALETTE & ROLES], [PHYSICS], [CONTROL], [GOAL].
3. update_last_defeat and update_last_victory automatic registration into CoreInvariantRegistry.
4. record_curriculum_transition with steps_to_win and invariants_confirmed and rich summarize_progression formatting.
5. PaletteRoleMap.get_role_label returns clean 'Color X' when role is UNKNOWN.
6. Cross-level invariant preservation and curriculum continuity in GameSession.
"""

from typing import Any
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import (
    DefeatExemplar,
    EntityRole,
    GameMemory,
    GroundedInvariant,
    VictoryExemplar,
)
from v10_agent.session import GameSession


def test_apply_invariant_revision_negative_barrier_sync():
    """Verify [NEGATIVE_BARRIER] updates palette, grounded invariants, and invariant registry."""
    gm = GameMemory(game_id="test_game")
    revised = [
        "- [NEGATIVE_BARRIER] Contact(ACTOR, Color_3) => DefeatReset() : Orange hazard spikes kill player",
    ]
    gm.apply_invariant_revision(revised, outcome="WIN", level_id="level_0")

    # 1. Palette assignment
    assert gm.palette.get(3).role == EntityRole.HAZARD
    assert gm.palette.get(3).confidence >= 0.85

    # 2. Grounded invariant
    nb_invs = [g for g in gm.grounded_invariants if g.brusentsov_type == "NEGATIVE_BARRIER"]
    assert len(nb_invs) == 1
    assert "Color_3" in nb_invs[0].antecedent
    assert nb_invs[0].is_active()

    # 3. CoreInvariantRegistry
    emp_invs = [i for i in gm.invariant_registry.invariants if i.subject_pattern == "color==3"]
    assert len(emp_invs) == 1
    assert emp_invs[0].scope == "CORE_GAME_LAW"
    assert emp_invs[0].expected_value == "DefeatReset()"

    # 4. Structured invariants backward compatibility
    assert len(gm.structured_invariants) == 1
    assert len(gm.tier1_kinematics_and_topology) == 1


def test_apply_invariant_revision_positive_canon_sync():
    """Verify [POSITIVE_CANON] updates palette, grounded invariants, and invariant registry."""
    gm = GameMemory(game_id="test_game")
    revised = [
        "- [POSITIVE_CANON] Contact(ACTOR, Color_5) => LevelVictory() : Reaching magenta portal wins level",
    ]
    gm.apply_invariant_revision(revised, outcome="WIN", level_id="level_1")

    # 1. Palette assignment
    assert gm.palette.get(5).role == EntityRole.TARGET
    assert gm.palette.get(5).confidence >= 0.85

    # 2. Grounded invariant
    pc_invs = [g for g in gm.grounded_invariants if g.brusentsov_type == "POSITIVE_CANON"]
    assert len(pc_invs) == 1
    assert "Color_5" in pc_invs[0].antecedent
    assert pc_invs[0].is_active()

    # 3. CoreInvariantRegistry
    emp_invs = [i for i in gm.invariant_registry.invariants if i.subject_pattern == "color==5"]
    assert len(emp_invs) == 1
    assert emp_invs[0].scope == "CORE_GAME_LAW"
    assert emp_invs[0].expected_value == "LevelVictory()"

    # 4. Structured invariants backward compatibility
    assert len(gm.structured_invariants) == 1
    assert len(gm.tier3_level_rules) == 1


def test_apply_invariant_revision_palette_and_roles_sync():
    """Verify [PALETTE & ROLES] parses multiple roles and updates palette and registry."""
    gm = GameMemory(game_id="test_game")
    revised = [
        "- [PALETTE & ROLES]: Color 2 is HAZARD, Color 4 is TARGET, Color 1 is ACTOR, Color 0 is BACKGROUND, Color 6 is WALL",
    ]
    gm.apply_invariant_revision(revised, outcome="WIN", level_id="level_0")

    assert gm.palette.get(2).role == EntityRole.HAZARD
    assert gm.palette.get(4).role == EntityRole.TARGET
    assert gm.palette.get(1).role == EntityRole.ACTOR
    assert gm.palette.get(0).role == EntityRole.BACKGROUND
    assert gm.palette.get(6).role == EntityRole.OBSTACLE  # "wall" alias -> OBSTACLE

    # Check registry registrations
    palette_emp = [i for i in gm.invariant_registry.invariants if i.invariant_type == "palette_role"]
    assert len(palette_emp) == 5
    assigned_colors = {i.subject_pattern for i in palette_emp}
    assert assigned_colors == {"color==2", "color==4", "color==1", "color==0", "color==6"}


def test_apply_invariant_revision_physics_control_goal_sync():
    """Verify standard tags ([PHYSICS], [CONTROL], [GOAL]) are registered in CoreInvariantRegistry."""
    gm = GameMemory(game_id="test_game")
    revised = [
        "- [PHYSICS]: Sliding entities preserve rigid form across transitions",
        "- [CONTROL]: Action 1 moves controllable actor horizontally",
        "- [GOAL]: All socket apertures must be occupied simultaneously",
    ]
    gm.apply_invariant_revision(revised, outcome="WIN", level_id="level_0")

    # Invariant registry entries
    core_laws = gm.invariant_registry.get_core_game_laws()
    assert len(core_laws) == 3
    types = {i.invariant_type for i in core_laws}
    assert "kinematic_law" in types
    assert "control" in types
    assert "goal" in types

    # Backward compatibility
    assert len(gm.structured_invariants) == 3
    assert len(gm.tier1_kinematics_and_topology) >= 1
    assert len(gm.tier2_interactions) >= 1
    assert len(gm.tier3_level_rules) >= 1


def test_update_last_defeat_and_victory_registry_sync():
    """Verify update_last_defeat and update_last_victory sync into CoreInvariantRegistry."""
    gm = GameMemory(game_id="test_game")

    # Defeat exemplar
    defeat_ex = DefeatExemplar(
        level_index=0,
        fatal_step=4,
        fatal_action_id=2,
        hazard_color=7,
        explanation="Collided with red hazard",
    )
    gm.update_last_defeat(defeat_ex)

    assert gm.palette.get(7).role == EntityRole.HAZARD
    defeat_emp = next((i for i in gm.invariant_registry.invariants if i.subject_pattern == "color==7"), None)
    assert defeat_emp is not None
    assert defeat_emp.invariant_type == "hazard_barrier"
    assert defeat_emp.scope == "CORE_GAME_LAW"

    # Victory exemplar
    vic_ex = VictoryExemplar(
        level_index=0,
        total_steps=10,
        target_color=9,
        final_action_id=1,
        explanation="Touched green flag",
    )
    gm.update_last_victory(vic_ex)

    assert gm.palette.get(9).role == EntityRole.TARGET
    vic_emp = next((i for i in gm.invariant_registry.invariants if i.subject_pattern == "color==9"), None)
    assert vic_emp is not None
    assert vic_emp.invariant_type == "goal_canon"
    assert vic_emp.scope == "CORE_GAME_LAW"


def test_curriculum_transition_and_summarize_progression_sync():
    """Verify record_curriculum_transition stores all fields and summarize_progression outputs them."""
    gm = GameMemory(game_id="test_game")

    # Empty progression
    assert gm.summarize_progression() == "none"

    # Record curriculum transition with step counts and confirmed invariants
    gm.record_curriculum_transition(
        level_from=0,
        level_to=1,
        delta_summary="Introduced moving hazard entities",
        steps_to_win=15,
        invariants_confirmed=["inv_rigid_physics", "inv_hazard_avoidance"],
    )

    assert len(gm.curriculum_history) == 1
    entry = gm.curriculum_history[0]
    assert entry["level"] == 0
    assert entry["from_level"] == 0
    assert entry["to_level"] == 1
    assert entry["steps_to_win"] == 15
    assert len(entry["invariants_confirmed"]) == 2
    assert entry["summary"] == "Introduced moving hazard entities"

    # Summarize progression
    summary = gm.summarize_progression()
    assert "Level 0: completed in 15 steps (invariants: 2)" in summary
    assert "Introduced moving hazard entities" in summary

    # Legacy format backward compatibility
    gm.curriculum_history.append({"level": 1, "steps_to_win": 8, "invariants_confirmed": ["inv_1"]})
    summary2 = gm.summarize_progression()
    assert "Level 1: completed in 8 steps (invariants: 1)" in summary2


def test_palette_get_role_label_clean():
    """Verify get_role_label returns clean 'Color X' when UNKNOWN and 'Color X (ROLE)' when assigned."""
    gm = GameMemory(game_id="test_game")

    # Unknown role
    assert gm.palette.get_role_label(8) == "Color 8"
    assert "(UNKNOWN)" not in gm.palette.get_role_label(8)

    # Generalize text preserves clean format
    gen_text = gm.palette.generalize_color_reference("Avoid color 8 and step on color 9")
    assert gen_text == "Avoid Color 8 and step on Color 9"

    # Assigned role
    gm.palette.assign_role(9, EntityRole.TARGET, confidence=0.9)
    assert gm.palette.get_role_label(9) == "Color 9 (TARGET)"

    gen_text_assigned = gm.palette.generalize_color_reference("Reach color 9 to win")
    assert gen_text_assigned == "Reach Color 9 (TARGET) to win"


def test_end_to_end_session_win_curriculum_preservation():
    """Verify GameSession preserves distilled invariants across level transitions."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    session = GameSession(config=config, advisor=advisor)

    # Turn 2 reflection response from Solver
    advisor.set_response(
        "solver_reflection",
        "<invariants_update>\n"
        "- [NEGATIVE_BARRIER] Contact(ACTOR, Color_2) => DefeatReset() : Spike trap kills player\n"
        "- [POSITIVE_CANON] Contact(ACTOR, Color_4) => LevelVictory() : Goal flag completes level\n"
        "- [PHYSICS]: Dynamic blocks slide unimpeded until collision\n"
        "</invariants_update>",
    )

    # Simulate level transition with executed steps
    session.level_executed_actions = ["ACTION1", "ACTION2"]
    session.handle_level_transition("level_1")

    game_mem = session.memory_manager.get_game_memory("session")

    # Check curriculum history
    assert len(game_mem.curriculum_history) >= 1
    last_curriculum = game_mem.curriculum_history[-1]
    assert last_curriculum["steps_to_win"] == 2
    assert "Completed" in last_curriculum["summary"]

    # Check preserved core invariants across the level transition
    carried_ids = [i.invariant_id for i in session.invariant_registry.invariants]
    # Invariants marked as CORE_GAME_LAW should survive into the next level registry
    assert any("hazard" in inv_id or "target" in inv_id or "phys" in inv_id for inv_id in carried_ids)

    # Check palette roles survived in GameMemory
    assert game_mem.palette.get(2).role == EntityRole.HAZARD
    assert game_mem.palette.get(4).role == EntityRole.TARGET
