"""End-to-end integration test simulating an Arcade game playthrough with ARC-AGI-3 LCLD Agent V10."""

from __future__ import annotations

from typing import Any
import pytest

from lcld_competition_child import run_direct_game


class SyntheticFrame:
    """Mock frame mimicking an Arcade FrameData object."""
    def __init__(self, step_idx: int, levels_completed: int, state: str, grid: list[list[int]]):
        self.game_id = "test_synth_game"
        self.levels_completed = levels_completed
        self.win_levels = 2
        self.state = state
        self.guid = f"guid_step_{step_idx}"
        self.available_actions = ["ACTION1", "ACTION2", "ACTION6", "RESET"]
        self.frame = grid


class SyntheticArcadeEnv:
    """Mock environment stepping through level progression to victory."""
    def __init__(self):
        self.step_idx = 0
        self.levels_completed = 0
        self.state = "IN_PROGRESS"

    def initial_frame(self) -> SyntheticFrame:
        grid = [
            [0, 1, 0],
            [0, 0, 2],
        ]
        return SyntheticFrame(self.step_idx, self.levels_completed, self.state, grid)

    def step(self, action: Any, data: dict[str, Any] | None = None, reasoning: dict[str, Any] | None = None) -> SyntheticFrame:
        self.step_idx += 1

        if self.step_idx == 3:
            # Advance to Level 1
            self.levels_completed = 1
            grid = [
                [0, 0, 3],
                [0, 4, 0],
            ]
        elif self.step_idx >= 5:
            # Win game!
            self.state = "WIN"
            grid = [
                [0, 0, 0],
                [0, 0, 0],
            ]
        else:
            # Continue on current level
            grid = [
                [0, 1, 0],
                [0, 0, 2],
            ]

        return SyntheticFrame(self.step_idx, self.levels_completed, self.state, grid)


def test_v10_end_to_end_game_playthrough(monkeypatch):
    # Ensure offline test backend
    monkeypatch.setenv("ARC_LLM_ADVISOR_BACKEND", "fake")
    monkeypatch.setenv("LCLD_MAX_ACTIONS_PER_GAME", "20")

    env = SyntheticArcadeEnv()
    initial_frame = env.initial_frame()

    result = run_direct_game(
        env=env,
        game_id="synthetic_e2e_game",
        initial_frame=initial_frame,
    )

    assert result["status"] == "completed"
    assert result["action_count"] == 5
    assert result["levels_completed"] == 1
    assert result["final_state"] == "WIN"
    assert "WIN" in result["stop_reason"]
    assert result["telemetry_summary"]["accepted_action_count"] == 5
    assert result["game_over_reset_count"] == 0
