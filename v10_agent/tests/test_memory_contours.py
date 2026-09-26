"""Unit tests for Memory Contours and cross-level lifecycle."""

from __future__ import annotations

import pytest

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary
from v10_agent.memory_contours import (
    BranchSignature,
    MemoryContourManager,
    ProbeRecord,
    SyntaxErrorRecord,
)
from v10_agent.types import AtomicProposition, PropositionSet


def test_independent_memory_updates():
    manager = MemoryContourManager(game_id="game_1", level_id="level_0")

    # 1. Update EnvironmentSpecMemory
    env_mem = manager.get_env_spec_memory("session")
    env_mem.record_probe(ProbeRecord("p1", "ACTION1", {}, "moved_right", 0.9))
    assert len(env_mem.probe_history) == 1

    # 2. Update SyntaxErrorMemory
    syntax_mem = manager.get_syntax_error_memory("session")
    syntax_mem.record_error(SyntaxErrorRecord("h1", "def foo(): pass", "SyntaxError", "invalid syntax"))
    assert len(syntax_mem.entries) == 1

    # 3. Update EpistemicMemory
    ep_mem = manager.get_epistemic_memory("session")
    prop = AtomicProposition("object_identity", "obj_0", "preserved")
    ep_mem.record_judgment(BrusentsovJudgment("traj_1", "s1", Ternary.TRUE, PropositionSet.from_iterable([prop]), PropositionSet.from_iterable([prop])))
    assert len(ep_mem.judgments) == 1

    # Assert contours remained completely disjoint
    assert len(manager.env_spec_memory.probe_history) == 1
    assert len(manager.syntax_error_memory.entries) == 1
    assert len(manager.epistemic_memory.judgments) == 1


def test_epistemic_omit_and_sever_lifecycle():
    ep_mem = MemoryContourManager().get_epistemic_memory("session")

    branch = BranchSignature("sig_omit_1", "traj_1", "s1", [], [])
    ep_mem.pause_branch_as_omit(branch)
    assert len(ep_mem.live_omit_branches) == 1
    assert ep_mem.is_severed("sig_omit_1") is False

    # Sever branch
    ep_mem.sever_branch("sig_omit_1")
    assert len(ep_mem.live_omit_branches) == 0
    assert ep_mem.is_severed("sig_omit_1") is True

    # Attempting to re-add severed branch is ignored
    ep_mem.pause_branch_as_omit(branch)
    assert len(ep_mem.live_omit_branches) == 0


def test_level_vs_game_transitions():
    manager = MemoryContourManager(game_id="game_1", level_id="level_0")
    game_mem = manager.get_game_memory("session")
    game_mem.record_action_effect("ACTION1", "moves right")

    syntax_mem = manager.get_syntax_error_memory("session")
    syntax_mem.record_error(SyntaxErrorRecord("h1", "x = 1", "TypeError", "type error"))

    # Level transition: GameMemory preserved, SyntaxErrorMemory cleared
    manager.handle_level_transition("level_1")
    assert len(manager.syntax_error_memory.entries) == 0
    assert manager.game_memory.confirmed_action_effects["ACTION1"] == "moves right"
    assert manager.game_memory.completed_levels == 1

    # Game transition: all cleared
    manager.handle_game_transition("game_2", "level_0")
    assert len(manager.game_memory.confirmed_action_effects) == 0
    assert manager.game_memory.completed_levels == 0


def test_game_memory_empirical_context_and_reusable_primitives():
    manager = MemoryContourManager(game_id="game_1", level_id="level_0")
    game_mem = manager.get_game_memory("session")
    game_mem.record_action_effect("ACTION1", "moved obj_1 by dy=-1, dx=0 (UP)")
    game_mem.record_action_effect("ACTION2", "moved obj_1 by dy=1, dx=0 (DOWN)")
    game_mem.record_selection_mechanic("ACTION5: selection indicator transferred: internal dots moved from obj_0 to obj_8")
    game_mem.record_reusable_primitive({
        "name": "move_up",
        "parameters": [{"name": "obj"}],
        "docstring": "Move object UP via ACTION1",
    })

    ctx = game_mem.format_empirical_context()
    assert "CONFIRMED ACTION KINEMATICS" in ctx
    assert "ACTION1: moved obj_1 by dy=-1, dx=0 (UP)" in ctx
    assert "CONFIRMED ENTITY SELECTION MECHANICS" in ctx
    assert "selection indicator transferred" in ctx
    assert "REUSABLE CERTIFIED MOVEMENT PRIMITIVES" in ctx
    assert "move_up" in ctx


def test_indicator_dot_transfer_detection():
    from v10_agent.explorer_agent import compute_probe_effect
    from v10_agent.arga_lite import extract_arga_snapshot

    # grid0: obj_0 (color 10) at rows 0..10 has 3 dots of color 0; obj_1 (color 4) at rows 15..25 has no dots
    grid0 = [[9 for _ in range(10)] for _ in range(30)]
    for r in range(0, 11):
        for c in range(0, 6):
            grid0[r][c] = 10
    grid0[2][1] = 0
    grid0[4][1] = 0
    grid0[6][1] = 0

    for r in range(15, 26):
        for c in range(0, 6):
            grid0[r][c] = 4

    snap0 = extract_arga_snapshot(grid0)

    # grid1: dots moved to obj_1 (color 4), obj_0 is solid
    grid1 = [[9 for _ in range(10)] for _ in range(30)]
    for r in range(0, 11):
        for c in range(0, 6):
            grid1[r][c] = 10

    for r in range(15, 26):
        for c in range(0, 6):
            grid1[r][c] = 4
    grid1[16][1] = 0
    grid1[18][1] = 0
    grid1[20][1] = 0

    effect = compute_probe_effect(snap0, {"grid": grid1})
    assert "selection indicator transferred" in effect
    assert "active entity toggled" in effect


def test_cross_level_memory_sanitization_and_coordinate_filtering():
    """Verify that GameMemory sanitizes level-specific coordinates, offset notes, and Tier 3 rules on level transition."""
    from v10_agent.memory_contours import GameMemory

    game_mem = GameMemory(game_id="game_test")
    # Action effects
    game_mem.record_action_effect("ACTION1", "moved obj_1 by dy=-3, dx=0 (UP)")
    # Selection mechanics with level-specific offset note
    game_mem.record_selection_mechanic(
        "ACTION5: selection indicator transferred: internal dots moved from obj_0 to obj_8 (active entity toggled) (axis_steps=0, piece_steps=8)"
    )
    # Tier 1 with transient coordinates vs general kinematics
    game_mem.record_stratified_invariant("Action ACTION1 moves entity UP by dy=-3, dx=0", tier=1)
    game_mem.record_stratified_invariant("Transient displacement: rows 44-52, cols 50-58 need 10 steps down", tier=1)

    # Tier 2 with level-specific failure speculation vs general interaction
    game_mem.record_stratified_invariant("Stepping onto yellow button toggles barrier open", tier=2)
    game_mem.record_stratified_invariant("Failed at row 48 because action4() failed twice", tier=2)

    # Tier 3 level-specific rules
    game_mem.record_stratified_invariant("Level 0 specific rule: align at centroid (30, 30)", tier=3)

    # Perform level transition
    game_mem.handle_level_transition()

    # 1. Selection mechanics: object IDs replaced with role markers, level-specific offset stripped
    assert len(game_mem.selection_mechanics) == 1
    sm = game_mem.selection_mechanics[0]
    assert "obj_0" not in sm
    assert "obj_8" not in sm
    assert "[ACTOR]" in sm  # confirmed actors get [ACTOR] role marker
    assert "axis_steps" not in sm
    assert "piece_steps" not in sm

    # 2. Confirmed actions: object IDs replaced with role markers
    assert "obj_1" not in game_mem.confirmed_action_effects["ACTION1"]
    assert "[ACTOR]" in game_mem.confirmed_action_effects["ACTION1"]

    # 3. Tier 1: transient coordinate line removed, general kinematic line preserved
    t1_str = " ".join(game_mem.tier1_kinematics_and_topology)
    assert "moves entity UP" in t1_str
    assert "rows 44-52" not in t1_str
    assert "10 steps" not in t1_str

    # 4. Tier 2: failed/row 48 line removed, general interaction preserved
    t2_str = " ".join(game_mem.tier2_interactions)
    assert "yellow button" in t2_str
    assert "failed" not in t2_str
    assert "row 48" not in t2_str

    # 5. Tier 3: cleared
    assert len(game_mem.tier3_level_rules) == 0


def test_brusentsov_judgment_immutability_and_sanitization():
    """Verify that recording a judgment sanitizes syntax without violating dataclass immutability."""
    from v10_agent.brusentsov_logic import BrusentsovJudgment, Verdict
    from v10_agent.memory_contours import EpistemicMemory
    from v10_agent.types import PropositionSet

    ep_mem = EpistemicMemory(level_id="level_0")
    original_expl = "Traceback (most recent call last): SyntaxError: invalid syntax in candidate"
    judgment = BrusentsovJudgment(
        trajectory_id="t1",
        step_id="s1",
        verdict=Verdict.FOLLOW,
        expected_propositions=PropositionSet.from_iterable([]),
        observed_propositions=PropositionSet.from_iterable([]),
        explanation=original_expl,
    )

    ep_mem.record_judgment(judgment)
    # Stored judgment explanation is sanitized
    assert len(ep_mem.judgments) == 1
    assert "[REDACTED SYNTAX]" in ep_mem.judgments[0].explanation
    assert "Traceback" not in ep_mem.judgments[0].explanation


def test_step_id_signatures_preserved_in_failed_trajectories():
    """Verify that legitimate step IDs starting with 's' (e.g. 's1') are not discarded."""
    from v10_agent.memory_contours import EpistemicMemory

    ep_mem = EpistemicMemory(level_id="level_0")
    ep_mem.sever_branch("s1")
    assert ("s1",) in ep_mem.failed_completed_trajectories
    ep_mem.sever_branch("step_42")
    assert ("step_42",) in ep_mem.failed_completed_trajectories

