"""Unit tests for Memory Architecture: Exemplars, PaletteRoleMap, and GroundedInvariants.

Verifies:
1. Palette mapping over 16 colors (0..15) without destructive [COLOR] erasure.
2. DefeatExemplar and VictoryExemplar serialization/deserialization and prompt formatting.
3. GroundedInvariant linking to concrete exemplars and Brusentsov entailment.
4. GameMemory cross-level transfer and curriculum updates.
"""
from __future__ import annotations

import pytest

from v10_agent.memory_contours import (
    ColorAffordance,
    DefeatExemplar,
    EntityRole,
    GameMemory,
    GroundedInvariant,
    PaletteRoleMap,
    VictoryExemplar,
)


class TestPaletteRoleMap:
    def test_palette_initialization_16_colors(self):
        palette = PaletteRoleMap()
        assert len(palette.colors) == 16
        for i in range(16):
            assert palette.get(i).color_id == i
            assert palette.get(i).role == EntityRole.UNKNOWN

    def test_assign_role_and_filter(self):
        palette = PaletteRoleMap()
        palette.assign_role(2, EntityRole.HAZARD, confidence=0.9, evidence="Defeat on contact")
        palette.assign_role(3, EntityRole.TARGET, confidence=0.95, evidence="Win on contact")
        palette.assign_role(0, EntityRole.BACKGROUND, confidence=1.0)

        assert palette.get(2).role == EntityRole.HAZARD
        assert palette.get(3).role == EntityRole.TARGET
        assert len(palette.get_colors_by_role(EntityRole.HAZARD)) == 1
        assert palette.get_colors_by_role(EntityRole.HAZARD)[0].color_id == 2

    def test_role_based_generalization_preserves_color_identity(self):
        palette = PaletteRoleMap()
        palette.assign_role(2, EntityRole.HAZARD)
        palette.assign_role(3, EntityRole.TARGET)

        text = "Avoid color 2 because it is dangerous, reach color 3 to win"
        generalized = palette.generalize_color_reference(text)
        assert "Color 2 (HAZARD)" in generalized
        assert "Color 3 (TARGET)" in generalized
        assert "[COLOR]" not in generalized

    def test_palette_format_for_prompt(self):
        palette = PaletteRoleMap()
        palette.assign_role(1, EntityRole.ACTOR, confidence=0.8)
        palette.assign_role(2, EntityRole.HAZARD, confidence=0.9)
        formatted = palette.format_for_prompt()
        assert "Color 1: ACTOR" in formatted
        assert "Color 2: HAZARD" in formatted


class TestExemplars:
    def test_defeat_exemplar_serialization_and_prompt(self):
        defeat = DefeatExemplar(
            level_index=1,
            fatal_step=14,
            fatal_action_id=2,
            fatal_coords=None,
            actor_position_before=(5, 8),
            hazard_color=4,
            environment_signal="GAME_OVER",
            explanation="Moved down into red hazard cell",
        )
        d = defeat.to_dict()
        assert d["level_index"] == 1
        assert d["fatal_action_id"] == 2
        assert d["hazard_color"] == 4

        prompt_str = defeat.format_for_prompt()
        assert "LAST DEFEAT (Level 1, Step 14)" in prompt_str
        assert "ACTION2" in prompt_str
        assert "Hazard color: 4" in prompt_str

    def test_victory_exemplar_serialization_and_prompt(self):
        victory = VictoryExemplar(
            level_index=0,
            total_steps=8,
            action_sequence=[1, 1, 4, 4, 2, 2, 3, 3],
            final_action_id=3,
            target_color=8,
            explanation="Navigated maze to yellow goal",
            winning_invariants_used=["Contact(ACTOR, Color_8) => LevelVictory()"],
        )
        d = victory.to_dict()
        assert d["total_steps"] == 8
        assert d["target_color"] == 8
        assert len(d["action_sequence"]) == 8

        prompt_str = victory.format_for_prompt()
        assert "LAST VICTORY (Level 0, 8 steps)" in prompt_str
        assert "Color 8 (TARGET)" in prompt_str


class TestGroundedInvariant:
    def test_grounded_invariant_active_and_falsification(self):
        inv = GroundedInvariant(
            invariant_id="gi_hazard_4",
            antecedent="Contact(ACTOR, Color_4)",
            consequent="DefeatReset()",
            brusentsov_type="NEGATIVE_BARRIER",
            scope="CORE_GAME_LAW",
            confidence=0.9,
        )
        assert inv.is_active() is True
        inv.times_falsified += 1
        assert inv.is_active() is False


class TestGameMemoryIntegration:
    def test_game_memory_update_exemplars_and_invariants(self):
        gm = GameMemory(game_id="test_game")

        defeat = DefeatExemplar(
            level_index=0,
            fatal_step=5,
            fatal_action_id=1,
            hazard_color=2,
            explanation="Touched red hazard",
        )
        gm.update_last_defeat(defeat)

        assert gm.last_defeat_exemplar is not None
        assert gm.palette.get(2).role == EntityRole.HAZARD
        assert len(gm.grounded_invariants) == 1
        assert gm.grounded_invariants[0].brusentsov_type == "NEGATIVE_BARRIER"

        victory = VictoryExemplar(
            level_index=0,
            total_steps=12,
            action_sequence=[1, 2, 3, 4],
            final_action_id=4,
            target_color=5,
            explanation="Reached gold target",
        )
        gm.update_last_victory(victory)

        assert gm.last_victory_exemplar_v2 is not None
        assert gm.palette.get(5).role == EntityRole.TARGET
        assert len(gm.grounded_invariants) == 2
        assert gm.grounded_invariants[1].brusentsov_type == "POSITIVE_CANON"

        # Check formatted grounded memory for prompt
        prompt_block = gm.format_grounded_memory_for_prompt()
        assert "PALETTE MAPPING" in prompt_block
        assert "LAST DEFEAT" in prompt_block
        assert "LAST VICTORY" in prompt_block
        assert "CORE GAME LAWS" in prompt_block
