"Unit tests for Stratified Memory, Topological Judge Matching, and Invariant Evolution."

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary
from v10_agent.config import V10Config
from v10_agent.explorer_agent import clean_and_parse_json, extract_json_block
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import (
    GameMemory,
    IsolationViolationError,
    MemoryContourManager,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import parse_text_trajectory
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep


def test_topological_judge_matching_on_color_transition():
    "Verify that an object changing color retains identity ('preserved') instead of false 'destroyed'."
    # Before grid: 5x5 with a 2x2 object of color 1 at (1, 1)
    grid_before = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # After grid: same 2x2 object at (1, 1), but color changed from 1 to 4!
    grid_after = [
        [0, 0, 0, 0, 0],
        [0, 4, 4, 0, 0],
        [0, 4, 4, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    before_snap = extract_arga_snapshot(grid_before)
    pset = build_planning_set(before_snap, ["ACTION1", "ACTION6", "RESET"])
    verifier = LayeredVerifier(V10Config())

    after_obs = {"grid": grid_after, "state": "IN_PROGRESS", "levels_completed": 0}
    props = verifier.extract_observed_propositions(before_snap, after_obs, pset)

    # Must preserve identity rather than claiming destroyed!
    preserved_props = [p for p in props if p.family == "object_identity" and p.predicate == "preserved"]
    destroyed_props = [p for p in props if p.family == "object_identity" and p.predicate == "destroyed"]
    assert len(preserved_props) >= 1
    assert len(destroyed_props) == 0

    # Color delta should record the transition to color 4
    color_props = [p for p in props if p.family == "attribute_delta" and p.predicate == "color"]
    assert any(p.value == 4 for p in color_props)

    # Step expecting preservation should evaluate to TRUE (Follow), not FALSE (Null)
    obj_id = before_snap.objects[0].id
    step = GroundedStep(
        step_id="s1",
        dsl_function="click_piece",
        arguments={},
        expected_propositions=PropositionSet([
            AtomicProposition(family="object_identity", subject_id=obj_id, predicate="preserved")
        ]),
    )
    judgment = verifier.evaluate_transition(step, before_snap, after_obs, pset)
    assert judgment.verdict != Ternary.FALSE


def test_stratified_game_memory():
    """Verify that GameMemory stores and formats invariants across 3 explicit tiers."""
    game_mem = GameMemory(game_id="game_audit")

    # Tier 1: Physics and kinematics
    game_mem.record_action_effect("ACTION1", "moved active_entity UP by dy=-1, dx=0")
    game_mem.record_stratified_invariant("Entities stop upon hitting outer boundary wall", tier=1)

    # Tier 2: Interaction dynamics
    game_mem.record_selection_mechanic("ACTION5 toggles active focus between green and blue entities")
    game_mem.record_stratified_invariant("Stepping onto yellow cell toggles gate open", tier=2)

    # Tier 3: Curriculum rules
    game_mem.record_level_solution(
        level_id="level_0",
        setup_summary="Two mirrored tokens across center line",
        invariant_rule="Symmetric reflection alignment satisfies goal",
        winning_macro="action1() -> action4()",
    )

    ctx = game_mem.format_stratified_context()
    assert "TIER 1: FOUNDATIONAL PHYSICS, KINEMATICS & BOUNDARIES" in ctx
    assert "TIER 2: INTERACTION DYNAMICS & STATE TRANSITIONS" in ctx
    assert "TIER 3: HIGH-LEVEL DEDUCED RULES & STRATEGIES" in ctx
    assert "CURRICULUM INVARIANTS & WINNING PATTERNS" in ctx
    assert "outer boundary wall" in ctx
    assert "toggles active focus" in ctx
    assert "Symmetric reflection alignment" in ctx


def test_robust_json_cleaning_and_parsing():
    """Verify robust JSON parser handles trailing commas, comments, and single quotes."""
    # 1. Trailing comma in dict and list
    dirty_json = '{"schema_version": "v10.env_spec.1", "items": [1, 2, 3, ], "name": "test", }'
    res = clean_and_parse_json(dirty_json)
    assert res is not None
    assert res["schema_version"] == "v10.env_spec.1"
    assert res["items"] == [1, 2, 3]

    # 2. Markdown with comments
    commented_markdown = """```json
{
    // This is an environment spec
    "schema_version": "v10.env_spec.1",
    "confidence": 0.95
}
```"""
    res2 = extract_json_block(commented_markdown)
    assert res2 is not None
    assert res2["confidence"] == 0.95

    # 3. Single quotes
    single_quoted = "{'schema_version': 'v10.env_spec.1', 'valid': true}"
    res3 = clean_and_parse_json(single_quoted)
    assert res3 is not None
    assert res3["schema_version"] == "v10.env_spec.1"


def test_solver_invariant_evolution_parsing():
    """Verify that solver prompt parser extracts [INVARIANT_EVOLUTION] and attaches it to package."""
    text_response = """
[INVARIANT_EVOLUTION]
- Basic Laws Confirmed: ACTION1 moves active token UP by 1 pixel, stopped by walls.
- New Complex Mechanics: Level introduces a lock that requires touching the orange key first.

[HYPOTHESIS]
Invariants & Goal: Collect orange key then enter target chamber.
Strategy: Move up to grab key, then right.

[TRAJECTORY]
**Trajectory 1**
1. action1()
2. action4()
"""
    manifest = {"functions": [{"name": "action1"}, {"name": "action4"}]}
    pkg = parse_text_trajectory(text_response, manifest)

    assert pkg is not None
    assert "invariant_evolution" in pkg
    assert "ACTION1 moves active token UP" in pkg["invariant_evolution"]
    assert "orange key" in pkg["invariant_evolution"]
    assert len(pkg["candidates"]) == 1
    assert len(pkg["candidates"][0]["steps"]) == 2


def test_isolation_boundary_check_allows_natural_reasoning_words():
    """Verify that natural language containing 'without exception' does NOT trigger false ISO-1 violation."""
    manager = MemoryContourManager()
    ep_mem = manager.get_epistemic_memory("solver")

    # Explanation with 'without exception' (should NOT raise because it's not 'exception:')
    safe_judgment = BrusentsovJudgment(
        trajectory_id="traj_1",
        step_id="s1",
        verdict=Ternary.TRUE,
        expected_propositions=PropositionSet(),
        observed_propositions=PropositionSet(),
        explanation="The action executed without exception and satisfied all geometric conditions.",
    )
    ep_mem.record_judgment(safe_judgment)
    assert len(ep_mem.judgments) == 1

    # Real traceback should be sanitized by ISO-1 (softened behavior)
    bad_judgment = BrusentsovJudgment(
        trajectory_id="traj_1",
        step_id="s2",
        verdict=Ternary.FALSE,
        expected_propositions=PropositionSet(),
        observed_propositions=PropositionSet(),
        explanation="Error: Traceback (most recent call last):\nTypeError: unexpected None",
    )
    # Should NOT crash — sanitizes instead
    ep_mem.record_judgment(bad_judgment)
    assert len(ep_mem.judgments) == 2
    # Traceback and TypeError text must be redacted
    recorded = ep_mem.judgments[1].explanation
    assert "Traceback" not in recorded or "REDACTED" in recorded
    assert "TypeError" not in recorded or "REDACTED" in recorded
