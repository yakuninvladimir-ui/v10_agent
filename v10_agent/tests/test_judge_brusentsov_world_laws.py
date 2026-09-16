"""Unit tests for LayeredVerifier (Judge) implementing Brusentsov 3-valued logic of necessary consequence."""

import pytest
from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.types import PropositionSet
from v10_agent.verification import GroundedStep


@pytest.fixture
def verifier():
    return LayeredVerifier(V10Config())


def test_judge_motion_step_zero_delta_returns_false(verifier):
    """Execution immediately severs on zero grid delta for confirmed directional motion (Brusentsov nullity xy'_0)."""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
    ]
    before_snap = extract_arga_snapshot(grid)
    pset = build_planning_set(before_snap, ["ACTION4", "RESET"])

    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("ACTION4", "Object moved RIGHT")

    step = GroundedStep(
        step_id="s1",
        dsl_function="move_right",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([]),
    )

    # After state has identical grid: motion into boundary produced zero delta
    after_obs = {"grid": grid, "state": "IN_PROGRESS"}
    action_dict = {"id": "ACTION4", "action_id": "ACTION4", "data": {}}

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=before_snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=game_mem,
        action_dict=action_dict,
    )

    # Brusentsov nullity: zero grid delta on confirmed motion returns FALSE
    assert judgment.verdict == Ternary.FALSE
    assert "Motion action produced zero grid delta" in judgment.explanation


def test_judge_valid_motion_returns_true(verifier):
    """Valid directional motion confirmed by Explorer that displaces entity returns TRUE."""
    before_grid = [
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    after_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    before_snap = extract_arga_snapshot(before_grid)
    pset = build_planning_set(before_snap, ["ACTION4", "RESET"])

    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("ACTION4", "Object moved RIGHT")

    step = GroundedStep(
        step_id="s1",
        dsl_function="move_right",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([]),
    )

    after_obs = {"grid": after_grid, "state": "IN_PROGRESS"}
    action_dict = {"id": "ACTION4", "action_id": "ACTION4", "data": {}}

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=before_snap,
        after_obs=after_obs,
        planning_set=pset,
        game_memory=game_mem,
        action_dict=action_dict,
    )

    assert judgment.verdict == Ternary.TRUE
    assert "Kinematics verified" in judgment.explanation or "follow xy" in judgment.explanation


def test_judge_terminal_failure_returns_false(verifier):
    """Transition resulting in GAME_OVER returns FALSE."""
    grid = [[0, 1, 0]]
    before_snap = extract_arga_snapshot(grid)
    pset = build_planning_set(before_snap, ["ACTION1", "RESET"])

    step = GroundedStep(
        step_id="s1",
        dsl_function="step_action",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([]),
    )

    after_obs = {"grid": grid, "state": "GAME_OVER"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=before_snap,
        after_obs=after_obs,
        planning_set=pset,
    )

    assert judgment.verdict == Ternary.FALSE
    assert "Terminal failure" in judgment.explanation


def test_judge_terminal_victory_returns_true(verifier):
    """Transition resulting in WIN or levels_completed increase returns TRUE."""
    grid = [[0, 1, 0]]
    before_snap = extract_arga_snapshot(grid)
    pset = build_planning_set(before_snap, ["ACTION1", "RESET"])

    step = GroundedStep(
        step_id="s1",
        dsl_function="step_action",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([]),
    )

    after_obs = {"grid": grid, "state": "WIN", "levels_completed": 1}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=before_snap,
        after_obs=after_obs,
        planning_set=pset,
    )

    assert judgment.verdict == Ternary.TRUE
    assert "Level/game advance" in judgment.explanation


def test_judge_execution_uninterrupted_by_distance_metrics(verifier):
    """In updated architecture, intermediate steps are not interrupted by distance metrics."""
    before_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 3, 3, 0],
    ]
    before_snap = extract_arga_snapshot(before_grid)
    pset = build_planning_set(before_snap, ["ACTION1", "ACTION2", "ACTION4"])

    game_mem = GameMemory(game_id="test_game")
    game_mem.record_action_effect("ACTION2", "Object moved DOWN")
    game_mem.record_action_effect("ACTION1", "Object moved UP")

    # Move DOWN
    after_grid_closer = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 3, 3, 0],
    ]
    step_down = GroundedStep(step_id="s1", dsl_function="move_down", arguments={}, expected_propositions=PropositionSet.from_iterable([]))
    judgment_closer = verifier.evaluate_transition(
        step=step_down,
        before_snapshot=before_snap,
        after_obs={"grid": after_grid_closer, "state": "IN_PROGRESS"},
        planning_set=pset,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION2"},
    )
    assert judgment_closer.verdict == Ternary.TRUE
    assert "follow xy" in judgment_closer.explanation

    # Move UP
    after_grid_away = [
        [0, 2, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 3, 3, 0],
    ]
    step_up = GroundedStep(step_id="s2", dsl_function="move_up", arguments={}, expected_propositions=PropositionSet.from_iterable([]))
    judgment_away = verifier.evaluate_transition(
        step=step_up,
        before_snapshot=before_snap,
        after_obs={"grid": after_grid_away, "state": "IN_PROGRESS"},
        planning_set=pset,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION1"},
    )
    assert judgment_away.verdict == Ternary.TRUE
    assert "follow xy" in judgment_away.explanation


def test_judge_auxiliary_selection_toggle_follows_without_interruption(verifier):
    """ACTION5 (selection toggle) follows without step-by-step interruption."""
    grid = [[0, 2, 0], [0, 0, 0]]
    before_snap = extract_arga_snapshot(grid)
    pset = build_planning_set(before_snap, ["ACTION5"])

    step = GroundedStep(step_id="s1", dsl_function="toggle_selection", arguments={}, expected_propositions=PropositionSet.from_iterable([]))
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=before_snap,
        after_obs={"grid": grid, "state": "IN_PROGRESS"},
        planning_set=pset,
        action_dict={"action_id": "ACTION5"},
    )
    assert judgment.verdict == Ternary.TRUE
    assert "follow xy" in judgment.explanation


def test_solver_prompt_contains_grid_bounds_and_freedom_of_motion():
    """Verify that Solver prompt user payload contains grid_bounds and per-object freedom_of_motion."""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    manifest = {"functions": [{"name": "move_obj", "parameters": [{"name": "obj", "type": "str"}]}]}
    sys_p, user_p = build_solver_prompts(manifest, pset)

    assert "grid_bounds" in user_p
    assert '"height": 4' in user_p
    assert '"width": 5' in user_p
    assert "freedom_of_motion" in user_p
    assert "up_to_border" in user_p
    assert "down_to_border" in user_p
    assert "left_to_border" in user_p
    assert "right_to_border" in user_p
