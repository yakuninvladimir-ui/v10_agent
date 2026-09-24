"""Tests for Two-Tier Memory Architecture, Inter-Level Victory Example, and Event-Driven Probes."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, Verdict
from v10_agent.config import V10Config
from v10_agent.memory_contours import (
    EpistemicMemory,
    GameMemory,
    LevelVictoryExample,
)
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import PlanningObject, PlanningSet, build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.explorer_agent import PrimitiveProbeManager


def _make_dummy_pset() -> PlanningSet:
    pset = MagicMock(spec=PlanningSet)
    pset.grid_dims = (64, 64)
    pset.grid_hash = "test_hash"
    pset.objects = []
    pset.relations = []
    pset.object_real_to_alias = {}
    return pset


def test_epistemic_memory_clear_for_new_level():
    """Verify intra-level memory clears completely on level transition."""
    ep_mem = EpistemicMemory(level_id="level_0")
    j = BrusentsovJudgment(
        trajectory_id="t1",
        step_id="s1",
        verdict=Ternary.FALSE,
        expected_propositions=frozenset(),
        observed_propositions=frozenset(),
        explanation="test failure",
    )
    ep_mem.record_judgment(j)
    ep_mem.sever_branch("ACTION1 -> ACTION2")
    ep_mem.record_failed_completed_trajectory(("ACTION1", "ACTION2"))
    ep_mem.record_attempt_feedback({"goal": "test"}, "action1 -> action2", "failed", "null")

    assert len(ep_mem.judgments) == 1
    assert len(ep_mem.severed_null_signatures) == 1
    assert len(ep_mem.failed_completed_trajectories) == 1
    assert len(ep_mem.current_level_attempts) == 1

    # Clear for new level
    ep_mem.clear_for_new_level()

    assert len(ep_mem.judgments) == 0
    assert len(ep_mem.severed_null_signatures) == 0
    assert len(ep_mem.failed_completed_trajectories) == 0
    assert len(ep_mem.current_level_attempts) == 0


def test_game_memory_level_victory_example_recording_and_formatting():
    """Verify LevelVictoryExample records object diffs and formats neutrally as an example."""
    gm = GameMemory(game_id="test_game")
    assert gm.last_level_victory_example is None
    assert gm.format_previous_level_example() == ""

    gm.record_level_victory_example(
        level_id="0",
        winning_actions=["ACTION2", "ACTION2", "ACTION3"],
        object_diffs=[
            {"alias": "C", "id": "obj_1", "dy": 30, "dx": 15},
            {"alias": "D", "id": "obj_2", "color_change": "4->1"},
        ],
        static_objects=["A", "B", "I"],
        goal_rule="Matched C to I",
    )

    assert gm.last_level_victory_example is not None
    assert gm.last_level_victory_example.level_id == "0"
    assert len(gm.last_level_victory_example.winning_actions) == 3

    formatted = gm.format_previous_level_example()
    assert "PREVIOUS LEVEL EXAMPLE" in formatted
    assert "illustrative only, do not blindly copy" in formatted
    assert "C: displaced by (dy=+30, dx=+15)" in formatted
    assert "D: color 4->1" in formatted
    assert "Objects Remaining Completely Static: A, B, I" in formatted
    assert "Goal pattern observed: Matched C to I" in formatted
    assert "NOTE: This is an example of game logic from the previous level" in formatted


def test_solver_prompt_includes_previous_level_example_and_no_diagnostic_probes():
    """Verify Solver prompt contains PREVIOUS LEVEL EXAMPLE and completely excludes Diagnostic probes."""
    pset = _make_dummy_pset()
    gm = GameMemory(game_id="test_game")
    gm.record_level_victory_example(
        level_id="0",
        winning_actions=["ACTION2", "ACTION3"],
        object_diffs=[{"alias": "A", "dy": 10, "dx": 5}],
        static_objects=["B"],
        goal_rule="Alignment achieved",
    )

    manifest = {"functions": [{"name": "action1", "parameters": [], "docstring": "ACTION1"}]}
    _, user_p = build_solver_prompts(
        manifest=manifest,
        planning_set=pset,
        game_memory=gm,
        action_budget=50,
        attempts_remaining=4,
        max_attempts=5,
    )

    # Must contain illustrative example
    assert "PREVIOUS LEVEL EXAMPLE" in user_p
    assert "Level 0 victory - illustrative only" in user_p
    assert "A: displaced by (dy=+10, dx=+5)" in user_p

    # Must NOT contain Diagnostic probes
    assert "Diagnostic probes" not in user_p
    assert "Actions remaining: 50" in user_p
    assert "Planning attempts: 4 of 5 remaining" in user_p


def test_explorer_dynamic_reprobes_capped_to_two_steps():
    """Verify explorer dynamic reprobes strictly cap probe sequences to at most 2 actions."""
    manager = PrimitiveProbeManager(max_probes=10)
    # Simulate a modal toggle (e.g. selection indicator or state toggle) with max_steps=2
    reprobes = manager.get_dynamic_reprobes("ACTION6", "selection indicator appeared on obj_2", max_steps=2)
    assert len(reprobes) <= 2
    for r in reprobes:
        assert r.action_id in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def test_level_victory_example_includes_initial_objects():
    """Verify LevelVictoryExample formats initial objects cleanly in prompt context."""
    gm = GameMemory(game_id="test_game")
    gm.record_level_victory_example(
        level_id="level_0",
        winning_actions=["action1", "action3"],
        object_diffs=[{"alias": "A", "dy": 0, "dx": 3, "color_change": ""}],
        static_objects=["B"],
        initial_objects=[
            {"name": "A", "color": 1, "bbox": [0, 0, 5, 5], "role": "ACTOR", "description": "blue box"},
            {"name": "B", "color": 2, "bbox": [10, 10, 15, 15], "role": "STATIC", "description": "red barrier"},
        ],
        goal_rule="Alignment",
    )
    formatted = gm.format_previous_level_example()
    assert "- Initial State Objects:" in formatted
    assert "A (color 1, bbox [0, 0, 5, 5], ACTOR): blue box" in formatted
    assert "B (color 2, bbox [10, 10, 15, 15], STATIC): red barrier" in formatted
    assert "A: displaced by (dy=+0, dx=+3)" in formatted


def test_dsl_coder_augments_missing_confirmed_actions():
    """Verify DSLCoder automatically appends missing confirmed actions from earlier levels."""
    from unittest.mock import MagicMock
    from v10_agent.config import V10Config
    from v10_agent.dsl_coder import DSLCoder
    from v10_agent.memory_contours import SyntaxErrorMemory
    from v10_agent.sandbox import SandboxExecutor

    config = V10Config()
    advisor = MagicMock()
    # LLM only generated action5, omitting action1 and action2
    advisor.generate.return_value = (
        "```python\n"
        "def action5(api):\n"
        "    \"\"\"Toggle active actor.\"\"\"\n"
        "    return api.declare_environment_action(action_id='ACTION5')\n"
        "```"
    )

    coder = DSLCoder(config, advisor, SandboxExecutor())
    gm = GameMemory(game_id="test_game")
    gm.record_action_effect("ACTION1", "moves UP")
    gm.record_action_effect("ACTION2", "moves DOWN")

    syntax_mem = SyntaxErrorMemory(level_id="level_1")
    snap = extract_arga_snapshot([[0, 1, 0]])
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION5", "RESET"])
    module, manifest, errors = coder.generate_dsl(
        env_spec={"available_actions": ["ACTION5"]},
        syntax_memory=syntax_mem,
        planning_set=pset,
        game_memory=gm,
    )

    assert module is not None
    assert manifest is not None
    fn_names = {f["name"] for f in manifest["functions"]}
    assert "action5" in fn_names
    assert "action1" in fn_names
    assert "action2" in fn_names
    assert hasattr(module, "action1")
    assert hasattr(module, "action2")
    assert hasattr(module, "action5")


def test_symbolic_executor_boundary_zero_delta_does_not_falsify():
    """Verify boundary collision producing zero delta does not falsify confirmed motion."""
    from unittest.mock import MagicMock
    from v10_agent.arga_lite import extract_arga_snapshot
    from v10_agent.config import V10Config
    from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor, GroundedStep, CandidateTrajectory, TrajectoryPool
    from v10_agent.planning_set import PlanningSet
    from v10_agent.types import PropositionSet, BoundingBox

    config = V10Config()
    sandbox = MagicMock()
    binder = MagicMock()
    verifier = MagicMock()
    verifier.evaluate_transition.return_value = MagicMock(
        verdict=Ternary.FALSE,
        observed_propositions=PropositionSet(),
        explanation="Hit boundary",
    )

    executor = SymbolicTrajectoryExecutor(config, sandbox, binder, verifier)

    cand = CandidateTrajectory(
        trajectory_id="cand_1",
        confidence=0.9,
        steps=[{"dsl_function": "action1", "arguments": {}}],
        cursor=0,
    )
    pool = TrajectoryPool([cand])
    ep_mem = EpistemicMemory(level_id="level_1")
    gm = GameMemory(game_id="test_game")
    gm.record_action_effect("ACTION1", "moved UP")

    grid = [[1, 0], [1, 0]]
    snap = extract_arga_snapshot(grid)
    pset = MagicMock(spec=PlanningSet)
    pset.grid = grid
    pset.grid_hash = "hash_boundary"

    # Actor touches row 0 (pinned against top border)
    actor_obj = MagicMock()
    actor_obj.role = "ACTOR"
    actor_obj.bbox = BoundingBox(min_row=0, min_col=0, max_row=1, max_col=0)
    pset.objects = [actor_obj]

    pending_step = GroundedStep(
        step_id="step_1",
        dsl_function="action1",
        arguments={},
        expected_propositions=PropositionSet(),
    )

    # Action1 (UP) produced zero delta because it's pinned at row 0
    after_obs = {"grid": grid, "state": "RUNNING"}

    eval_res = executor.evaluate_transition(
        pending_step=pending_step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
        game_memory=gm,
        action_dict={"action_id": "ACTION1"},
    )

    assert eval_res.verdict == Ternary.FALSE
    # Crucial: falsification_detected must be False because the entity is pinned at the boundary!
    assert eval_res.falsification_detected is False
    assert eval_res.candidate_severed is True

