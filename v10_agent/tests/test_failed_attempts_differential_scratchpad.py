"""Tests for failed attempts scratchpad with differential before/after analysis."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, Verdict
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import EpistemicMemory, summarize_grid_diff
from v10_agent.observe import normalize_observation
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.sandbox import SandboxExecutor
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.verification import GroundedStep, VerificationBinder


def test_summarize_grid_diff_no_data():
    """Verify handling of None or empty grids."""
    assert summarize_grid_diff(None, None) == "no grid data available"
    assert summarize_grid_diff([], []) == "0 cells changed (empty grid)"


def test_summarize_grid_diff_identical():
    """Verify that identical grids return 0 cells changed."""
    grid = [[0, 1, 2], [3, 4, 5]]
    assert summarize_grid_diff(grid, grid) == "0 cells changed (no visible change on grid)"


def test_summarize_grid_diff_three_changed_tiles():
    """Verify explicit coordinates and inclusive bounding box for small changes (<= 12 cells)."""
    # 5x5 grid with 3 changed cells:
    # (col=1, row=2) changed 0 -> 2
    # (col=3, row=2) changed 0 -> 2
    # (col=2, row=4) changed 0 -> 4
    before = [[0] * 5 for _ in range(5)]
    after = [row[:] for row in before]
    after[2][1] = 2  # x=1, y=2
    after[2][3] = 2  # x=3, y=2
    after[4][2] = 4  # x=2, y=4

    diff = summarize_grid_diff(before, after)
    assert "3 cells changed" in diff
    assert "0->2 (2 cells)" in diff
    assert "0->4 (1 cells)" in diff
    assert "(x=1, y=2): 0->2" in diff
    assert "(x=3, y=2): 0->2" in diff
    assert "(x=2, y=4): 0->4" in diff
    assert "in bbox: cols 1..3 (inclusive), rows 2..4 (inclusive)" in diff


def test_summarize_grid_diff_large_changes():
    """Verify aggregate summary for large changes (> 12 cells) to prevent prompt bloat."""
    before = [[0] * 10 for _ in range(10)]
    after = [row[:] for row in before]
    # Change 15 cells
    for r in range(3, 6):
        for c in range(2, 7):
            after[r][c] = 5

    diff = summarize_grid_diff(before, after)
    assert "15 cells changed" in diff
    assert "0->5 (15 cells)" in diff
    # Must NOT have individual (x=..., y=...) listings for 15 cells
    assert "(x=" not in diff
    assert "in bbox: cols 2..6 (inclusive), rows 3..5 (inclusive)" in diff


def test_summarize_grid_diff_dimension_mismatch():
    """Verify dimension mismatch detection."""
    before = [[0, 0], [0, 0]]
    after = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    diff = summarize_grid_diff(before, after)
    assert "grid dimensions changed: 2x2 -> 3x3" in diff


@settings(max_examples=50, deadline=None)
@given(
    h=st.integers(min_value=2, max_value=20),
    w=st.integers(min_value=2, max_value=20),
    num_changes=st.integers(min_value=0, max_value=10),
)
def test_summarize_grid_diff_pbt(h: int, w: int, num_changes: int):
    """Property-based test verifying invariant behavior on arbitrary matrices."""
    before = [[0] * w for _ in range(h)]
    after = [row[:] for row in before]

    for i in range(num_changes):
        r = i % h
        c = (i * 2) % w
        color = ((i + 1) % 15) + 1
        after[r][c] = color

    actual_changes = sum(1 for r in range(h) for c in range(w) if before[r][c] != after[r][c])

    diff = summarize_grid_diff(before, after)
    assert isinstance(diff, str)
    if actual_changes == 0:
        assert "0 cells changed" in diff
    else:
        assert f"{actual_changes} cells changed" in diff
        assert "inclusive" in diff


def test_epistemic_memory_scratchpad_formatting():
    """Verify format_scratchpad_context with differential before/after and effective steps."""
    ep_mem = EpistemicMemory(level_id="level_0")
    assert ep_mem.format_scratchpad_context() == ""

    ep_mem.record_attempt_feedback(
        hypothesis="Candidate c0",
        trajectory_summary="action6(x=10, y=10) -> action6(x=20, y=20)",
        status="executed_but_level_not_won",
        reason="Trajectory executed completely but level not won.",
        diff_summary="2 cells changed [0->2 (2 cells)] at (x=10, y=10): 0->2, (x=20, y=20): 0->2; in bbox: cols 10..20 (inclusive), rows 10..20 (inclusive)",
        effective_steps=[
            "Step 1 (action6): 1 cells changed [0->2 (1 cells)] at (x=10, y=10): 0->2",
            "Step 2 (action6): 1 cells changed [0->2 (1 cells)] at (x=20, y=20): 0->2",
        ],
    )

    formatted = ep_mem.format_scratchpad_context()
    assert "CURRENT LEVEL FAILED ATTEMPTS (DIFFERENTIAL BEFORE/AFTER SCRATCHPAD):" in formatted
    assert "Attempt 1: [executed_but_level_not_won]" in formatted
    assert "Physical grid diff (before vs after attempt):" in formatted
    assert "(x=10, y=10): 0->2" in formatted
    assert "Effective steps during attempt:" in formatted
    assert "Step 1 (action6)" in formatted
    assert "Failure reason:" in formatted


def test_build_solver_prompts_injects_scratchpad():
    """Verify build_solver_prompts embeds the scratchpad into Section 7."""
    ep_mem = EpistemicMemory(level_id="level_0")
    ep_mem.record_attempt_feedback(
        hypothesis="Candidate c0",
        trajectory_summary="action1 -> action2",
        status="severed_step_failed",
        reason="Hit boundary wall.",
        diff_summary="1 cells changed [0->3 (1 cells)] at (x=5, y=5): 0->3; in bbox: cols 5..5 (inclusive), rows 5..5 (inclusive)",
        effective_steps=["Step 1 (action1): 1 cells changed [0->3 (1 cells)] at (x=5, y=5): 0->3"],
    )

    raw_grid = [[0] * 10 for _ in range(10)]
    snapshot = extract_arga_snapshot(raw_grid)
    pset = build_planning_set(snapshot, available_actions=["ACTION1", "ACTION2"])
    manifest = {"functions": [{"name": "action1", "action_id": "ACTION1"}, {"name": "action2", "action_id": "ACTION2"}]}

    sys_prompt, user_prompt = build_solver_prompts(
        manifest=manifest,
        planning_set=pset,
        epistemic_memory=ep_mem,
    )

    assert "FAILED CANDIDATES THIS LEVEL" in user_prompt
    assert "DIFFERENTIAL BEFORE/AFTER SCRATCHPAD" in user_prompt
    assert "(x=5, y=5): 0->3" in user_prompt
    assert "Hit boundary wall." in user_prompt
    assert "DIFFERENTIAL ANALYSIS OF FAILED ATTEMPTS" in sys_prompt


def test_symbolic_executor_differential_feedback_recording():
    """Verify SymbolicTrajectoryExecutor records step diffs and attempt diff summary."""
    config = V10Config()
    executor = SymbolicTrajectoryExecutor(
        config=config,
        sandbox_executor=SandboxExecutor(),
        binder=VerificationBinder(),
        verifier=LayeredVerifier(config=config),
    )
    ep_mem = EpistemicMemory(level_id="level_0")

    cand = CandidateTrajectory(
        trajectory_id="c_diff_test",
        steps=[
            {"step_id": "s0", "dsl_function": "action6", "arguments": {"x": 1, "y": 1}},
            {"step_id": "s1", "dsl_function": "action6", "arguments": {"x": 2, "y": 2}},
        ],
    )
    pool = TrajectoryPool(proposal_id="p0", candidates=[cand])

    # Initial grid 4x4
    initial_grid = [[0] * 4 for _ in range(4)]
    cand.initial_grid = [row[:] for row in initial_grid]

    # Setup planning_set
    snapshot = extract_arga_snapshot(initial_grid)
    pset = build_planning_set(snapshot, available_actions=["ACTION6"])

    # Step 1: executes and changes (x=1, y=1) to 2
    grid_after_s0 = [row[:] for row in initial_grid]
    grid_after_s0[1][1] = 2  # row 1, col 1

    pending_s0 = GroundedStep(
        step_id="s0",
        dsl_function="action6",
        arguments={"x": 1, "y": 1},
    )

    res0 = executor.evaluate_transition(
        pending_step=pending_s0,
        before_snapshot=type("Snapshot", (), {"grid": initial_grid, "levels_completed": 0})(),
        after_obs={"grid": grid_after_s0, "state": "RUNNING", "levels_completed": 0},
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )
    assert res0.candidate_advanced is True
    assert len(cand.step_effects) == 1
    assert "(x=1, y=1): 0->2" in cand.step_effects[0]

    # Step 2: executes and changes (x=2, y=2) to 2, trajectory completes without win
    grid_after_s1 = [row[:] for row in grid_after_s0]
    grid_after_s1[2][2] = 2  # row 2, col 2

    pending_s1 = GroundedStep(
        step_id="s1",
        dsl_function="action6",
        arguments={"x": 2, "y": 2},
    )

    res1 = executor.evaluate_transition(
        pending_step=pending_s1,
        before_snapshot=type("Snapshot", (), {"grid": grid_after_s0, "levels_completed": 0})(),
        after_obs={"grid": grid_after_s1, "state": "RUNNING", "levels_completed": 0},
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )
    assert res1.candidate_advanced is True
    assert cand.is_finished() is True

    # Check EpistemicMemory has recorded attempt feedback with differential summary
    assert len(ep_mem.current_level_attempts) == 1
    att = ep_mem.current_level_attempts[0]
    assert att["status"] == "executed_but_level_not_won"
    assert "2 cells changed" in att["diff_summary"]
    assert "(x=1, y=1): 0->2" in att["diff_summary"]
    assert "(x=2, y=2): 0->2" in att["diff_summary"]
    assert len(att["effective_steps"]) == 2
