"""Unit tests for Stage 1: DefeatExemplar and VictoryExemplar physical grounding.

Verifies:
1. Actor identification and pre-transition centroid tracking.
2. Hazard color detection on directional collision and coordinate click.
3. Victory target color detection from object diffs (transformations, collectibles, destination overlap).
4. PaletteRoleMap role assignment and pixel count synchronization.
5. GroundedInvariant (NEGATIVE_BARRIER and POSITIVE_CANON) auto-creation on defeat and victory.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from v10_agent.config import V10Config
from v10_agent.memory_contours import EntityRole, GameMemory, DefeatExemplar, VictoryExemplar, PaletteRoleMap
from v10_agent.planning_set import PlanningSet, PlanningObject, build_planning_set
from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.session import GameSession
from v10_agent.types import BoundingBox, Centroid


def _build_test_planning_set(grid: list[list[int]], objects: list[PlanningObject] | None = None) -> PlanningSet:
    snapshot = extract_arga_snapshot(grid)
    return build_planning_set(
        snapshot=snapshot,
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"],
    )


class TestStage1ActorAndHazardGrounding:
    def test_identify_actor_object_by_role_and_saliency(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)

        # 10x10 grid with background 0, small actor (color 1) at (3, 3) and large obstacle (color 2)
        grid = [[0] * 10 for _ in range(10)]
        grid[3][3] = 1
        for r in range(5, 8):
            for c in range(5, 8):
                grid[r][c] = 2

        snapshot = extract_arga_snapshot(grid)
        pset = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        session.last_snapshot = snapshot
        session.last_planning_set = pset

        actor_pos, actor_obj = session._identify_actor_object()
        assert actor_obj is not None
        assert actor_obj.color == 1
        assert actor_pos == (3, 3)

    def test_extract_hazard_color_on_directional_collision(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)

        # Grid: background 0. Actor is at (2, 2) [color 1].
        # Directly above it at (1, 2) is a red hazard tile [color 4].
        grid = [[0] * 6 for _ in range(6)]
        grid[2][2] = 1
        grid[1][2] = 4  # Hazard

        snapshot = extract_arga_snapshot(grid)
        pset = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        session.last_snapshot = snapshot
        session.last_planning_set = pset

        actor_pos, actor_obj = session._identify_actor_object()
        assert actor_obj is not None

        # Fatal action: ACTION1 (UP, fatal_act_id=1)
        hazard = session._extract_hazard_color(
            actor_obj=actor_obj,
            fatal_act_id=1,  # UP
            fatal_coords=None,
            before_grid=grid,
            after_grid=None,
        )
        assert hazard == 4

    def test_extract_hazard_color_on_coordinate_click(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)

        grid = [[0] * 8 for _ in range(8)]
        grid[4][5] = 7  # Spike at row 4, col 5

        snapshot = extract_arga_snapshot(grid)
        session.last_snapshot = snapshot

        # Click at col 5, row 4
        hazard = session._extract_hazard_color(
            actor_obj=None,
            fatal_act_id=6,
            fatal_coords=(5, 4),  # x=col=5, y=row=4
            before_grid=grid,
            after_grid=None,
        )
        assert hazard == 7

    def test_defeat_exemplar_updates_palette_and_negative_barrier(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)
        gm = session.memory_manager.get_game_memory("session")

        grid = [[0] * 6 for _ in range(6)]
        grid[3][3] = 1  # Actor
        grid[3][4] = 9  # Hazard to the right

        snapshot = extract_arga_snapshot(grid)
        pset = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        session.last_snapshot = snapshot
        session.last_planning_set = pset
        session.level_executed_actions = ["ACTION4"]

        # Simulate GAME_OVER observation
        obs = {
            "state": "GAME_OVER",
            "grid": grid,
            "levels_completed": 0,
            "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        }
        session.pending_action = {"action_id": "ACTION4", "data": {}}

        # Process observation
        try:
            session.act(obs)
        except Exception:
            pass

        assert gm.last_defeat_exemplar is not None
        assert gm.last_defeat_exemplar.hazard_color == 9
        assert gm.last_defeat_exemplar.actor_position_before == (3, 3)

        # Palette must reflect the hazard and actor roles
        assert gm.palette.get(9).role == EntityRole.HAZARD
        assert gm.palette.get(1).role == EntityRole.ACTOR
        assert gm.palette.get(0).role == EntityRole.BACKGROUND

        # GroundedInvariant must be created with NEGATIVE_BARRIER
        barrier_invs = [g for g in gm.grounded_invariants if g.brusentsov_type == "NEGATIVE_BARRIER"]
        assert len(barrier_invs) == 1
        assert "Color_9" in barrier_invs[0].antecedent
        assert barrier_invs[0].scope == "CORE_GAME_LAW"


class TestStage1VictoryGrounding:
    def test_extract_victory_target_color_from_color_transformation(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)

        object_diffs = [
            {"alias": "obj_1", "dy": 0, "dx": 0, "color_change": "2->8"},
        ]
        target_color = session._extract_victory_target_color(
            primary_inv="Transform entity",
            object_diffs=object_diffs,
            executed_steps=["ACTION1"],
        )
        assert target_color == 8

    def test_extract_victory_target_color_from_collectible_removal(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)

        # Initial grid with collectible item (color 3)
        init_grid = [[0] * 6 for _ in range(6)]
        init_grid[2][2] = 1  # Actor
        init_grid[4][4] = 3  # Collectible coin
        session.level_initial_grid = init_grid
        init_snap = extract_arga_snapshot(init_grid)
        init_pset = build_planning_set(init_snap, available_actions=[])
        coin_obj = next(o for o in init_pset.objects if o.color == 3)
        alias = init_pset.object_real_to_alias.get(coin_obj.id, coin_obj.id)

        # Object diff showing collectible item vanished/merged
        object_diffs = [
            {"alias": alias, "id": coin_obj.id, "status": "gone/merged"},
        ]

        target_color = session._extract_victory_target_color(
            primary_inv="Collect item",
            object_diffs=object_diffs,
            executed_steps=["ACTION2", "ACTION3"],
        )
        assert target_color == 3

    def test_victory_exemplar_updates_palette_and_positive_canon(self):
        cfg = V10Config(llm_advisor_backend="fake")
        session = GameSession(config=cfg)
        gm = session.memory_manager.get_game_memory("session")

        # Initial frame
        init_grid = [[0] * 6 for _ in range(6)]
        init_grid[1][1] = 2  # Actor
        init_grid[4][4] = 5  # Goal marker
        session.level_initial_grid = init_grid

        # Final frame: actor reached goal and transformed goal to color 5
        win_grid = [[0] * 6 for _ in range(6)]
        win_grid[4][4] = 5
        win_snap = extract_arga_snapshot(win_grid)
        session.last_snapshot = win_snap
        session.last_planning_set = build_planning_set(win_snap, available_actions=["ACTION1", "ACTION2"])
        session.level_executed_actions = ["ACTION2", "ACTION3"]

        session.handle_level_transition("level_1")

        assert gm.last_victory_exemplar_v2 is not None
        assert gm.last_victory_exemplar_v2.target_color == 5

        # Palette must reflect TARGET role
        assert gm.palette.get(5).role == EntityRole.TARGET

        # GroundedInvariant must be created with POSITIVE_CANON
        canon_invs = [g for g in gm.grounded_invariants if g.brusentsov_type == "POSITIVE_CANON"]
        assert len(canon_invs) == 1
        assert "Color_5" in canon_invs[0].antecedent
        assert canon_invs[0].scope == "CORE_GAME_LAW"


class TestPaletteRoleMapPixelCounts:
    def test_update_pixel_counts_and_formatting(self):
        palette = PaletteRoleMap()
        grid = [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
        ]
        palette.update_pixel_counts(grid)

        assert palette.get(0).pixel_count == 3
        assert palette.get(1).pixel_count == 3
        assert palette.get(2).pixel_count == 2
        assert palette.get(3).pixel_count == 0

        palette.assign_role(1, EntityRole.ACTOR, confidence=0.8)
        palette.assign_role(2, EntityRole.HAZARD, confidence=0.9)

        formatted = palette.format_for_prompt()
        assert "PALETTE MAPPING (16 Colors, 0..15):" in formatted
        assert "Color 1: ACTOR" in formatted
        assert "pixels=3" in formatted
        assert "Color 2: HAZARD" in formatted
        assert "pixels=2" in formatted
