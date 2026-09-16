"""Game and environment adapter for ARC-AGI-3 Arcade / ArcEngine."""

from __future__ import annotations

from typing import Any, Mapping

from v10_agent.observe import normalize_observation, normalize_state_name


def frame_to_observation(frame: Any, frame_index: int = 0, game_id: str = "") -> dict[str, Any]:
    """Convert an Arcade FrameData or dict to canonical V10 observation."""
    if hasattr(frame, "__dict__") and not isinstance(frame, Mapping):
        data: dict[str, Any] = {}
        for attr in ("frame", "available_actions", "game_id", "guid", "score", "state", "levels_completed", "win_levels", "full_reset"):
            if hasattr(frame, attr):
                data[attr] = getattr(frame, attr)
        return normalize_observation(data, frame_index=frame_index, game_id=game_id)
    if isinstance(frame, Mapping):
        return normalize_observation(frame, frame_index=frame_index, game_id=game_id)
    return normalize_observation({"frame": frame}, frame_index=frame_index, game_id=game_id)


def is_terminal_success(state: str) -> bool:
    """Check if the state indicates successful completion of the game."""
    s = normalize_state_name(state)
    return s in {"WIN", "WON", "DONE", "TERMINAL", "VICTORY"}


def is_game_over(state: str) -> bool:
    """Check if the state indicates recoverable GAME_OVER."""
    s = normalize_state_name(state)
    return s in {"GAME_OVER", "LOST", "FAILED"}
