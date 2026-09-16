"""Unit tests explicitly enforcing Isolation Invariants ISO-1 through ISO-5."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary
from v10_agent.memory_contours import (
    IsolationViolationError,
    MemoryContourManager,
    SyntaxErrorRecord,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.sandbox import SandboxAPI
from v10_agent.types import PropositionSet


def test_iso_1_traceback_never_reaches_solver_memory():
    """ISO-1: Traceback or Python exception text must be sanitized in EpistemicMemory."""
    manager = MemoryContourManager()
    ep_mem = manager.get_epistemic_memory("solver")

    bad_judgment = BrusentsovJudgment(
        trajectory_id="traj_1",
        step_id="s1",
        verdict=Ternary.FALSE,
        expected_propositions=PropositionSet(),
        observed_propositions=PropositionSet(),
        explanation="Execution failed: Traceback (most recent call last):\n  File 'dsl.py', line 1\nSyntaxError: invalid",
    )

    # Should NOT crash — sanitizes instead
    ep_mem.record_judgment(bad_judgment)
    assert len(ep_mem.judgments) == 1
    # Traceback and SyntaxError text must be redacted
    recorded_explanation = ep_mem.judgments[0].explanation
    assert "Traceback" not in recorded_explanation or "REDACTED" in recorded_explanation
    assert "SyntaxError" not in recorded_explanation or "REDACTED" in recorded_explanation


def test_iso_2_level_goal_never_reaches_coder_memory():
    """ISO-2: Level goal or hypothesis statements must be sanitized in SyntaxErrorMemory."""
    manager = MemoryContourManager()
    syntax_mem = manager.get_syntax_error_memory("coder")

    bad_syntax_record = SyntaxErrorRecord(
        prompt_hash="hash1",
        source_code="def move(): pass",
        error_type="SyntaxError",
        error_message="Error while attempting to satisfy level_goal and win_condition",
    )

    # Should NOT crash — sanitizes instead
    syntax_mem.record_error(bad_syntax_record)
    assert len(syntax_mem.entries) == 1
    # Goal keywords must be redacted
    recorded_msg = syntax_mem.entries[0].error_message
    assert "level_goal" not in recorded_msg or "REDACTED" in recorded_msg
    assert "win_condition" not in recorded_msg or "REDACTED" in recorded_msg


def test_iso_3_explorer_cannot_emit_trajectories():
    """ISO-3: Explorer writes exclusively factual specs; trajectory keys are stripped, not crashed."""
    manager = MemoryContourManager()
    env_mem = manager.get_env_spec_memory("explorer")

    bad_spec = {
        "schema_version": "v10.env_spec.1",
        "trajectory": [{"step": 1, "action": "ACTION1"}],
    }

    # Should NOT crash — removes forbidden keys instead
    env_mem.record_spec(bad_spec)
    assert len(env_mem.specs) == 1
    assert "trajectory" not in env_mem.specs[0]  # Key was stripped

    bad_spec2 = {
        "schema_version": "v10.env_spec.1",
        "candidate_steps": [{"step": 1}],
    }
    env_mem.record_spec(bad_spec2)
    assert len(env_mem.specs) == 2
    assert "candidate_steps" not in env_mem.specs[1]  # Key was stripped

    # Valid factual spec where a note mentions the word 'trajectory' in natural language text
    safe_spec = {
        "schema_version": "v10.env_spec.1",
        "action_surface_notes": ["The trajectory_line object moved right on ACTION4"],
    }
    # Should NOT raise IsolationViolationError
    env_mem.record_spec(safe_spec)
    assert len(env_mem.specs) == 3


def test_iso_4_role_based_contour_access():
    """ISO-4: Only authorized caller roles may read/write memory contours."""
    manager = MemoryContourManager()

    # Solver cannot access Coder's SyntaxErrorMemory
    with pytest.raises(IsolationViolationError, match="ISO-4 Violation"):
        manager.get_syntax_error_memory(caller_role="solver")

    # Coder cannot access Solver's EpistemicMemory
    with pytest.raises(IsolationViolationError, match="ISO-4 Violation"):
        manager.get_epistemic_memory(caller_role="coder")

    # Coder cannot access Explorer's EnvironmentSpecMemory
    with pytest.raises(IsolationViolationError, match="ISO-4 Violation"):
        manager.get_env_spec_memory(caller_role="coder")

    # GameSession can access all three
    assert manager.get_env_spec_memory(caller_role="session") is not None
    assert manager.get_syntax_error_memory(caller_role="session") is not None
    assert manager.get_epistemic_memory(caller_role="session") is not None
    assert manager.get_game_memory(caller_role="session") is not None


def test_iso_5_planning_set_grounds_all_ids():
    """ISO-5: Identifiers must be grounded strictly in the current PlanningSet."""
    grid = [
        [0, 1, 0],
        [0, 0, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snapshot, available_actions=["ACTION1", "RESET"])
    api = SandboxAPI(planning_set)

    # Valid target IDs pass
    effect = api.declare_environment_action("ACTION1", target_object_ids=["obj_0", "obj_1"])
    assert effect.target_object_ids == ["obj_0", "obj_1"]

    # Invented object ID fails
    with pytest.raises(KeyError, match="does not resolve in current PlanningSet"):
        api.declare_environment_action("ACTION1", target_object_ids=["invented_obj_999"])

    # Disallowed action fails
    with pytest.raises(ValueError, match="is not in allowed actions"):
        api.declare_environment_action("ACTION7")
