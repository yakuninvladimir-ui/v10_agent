"""Observation ingestion and normalization for ARC-AGI-3 environments."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping

from v10_agent.action_semantics import (
    FORBIDDEN_ACTION_IDS,
    LEGAL_ACTION_IDS,
    RESET_ACTION_ID,
    filter_legal_actions,
    normalize_action_id,
)
from v10_agent.types import Grid2D

logger = logging.getLogger(__name__)


def collapse_frame_axes(grid: Any) -> Grid2D:
    """Collapse temporal/batch axes to return the active 2D integer grid.

    ARC's FrameData.frame may be a temporal list of 2D grids (depth 3) or a
    bare 2D grid (depth 2) or a numpy array, or a list containing 2D numpy arrays.
    This function guarantees a normalized list[list[int]] representation with
    values clamped to 0..9.
    """
    if grid is None:
        return []

    # If it's a list or tuple of frames:
    if isinstance(grid, (list, tuple)) and len(grid) > 0:
        # Check if first element is a 2D numpy array or 2D list
        first = grid[0]
        if hasattr(first, "ndim") and int(getattr(first, "ndim", 0)) == 2:
            # List of 2D numpy arrays: take latest frame
            grid = grid[-1]
        elif isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            # List of 2D lists (depth 3): take latest frame
            grid = grid[-1]

    # Handle numpy ndarray
    if hasattr(grid, "ndim"):
        ndim = int(getattr(grid, "ndim", 0))
        curr = grid
        while ndim > 2:
            if len(curr) == 0:
                return []
            curr = curr[-1]
            ndim = int(getattr(curr, "ndim", 0))
        if hasattr(curr, "tolist"):
            grid = curr.tolist()
        else:
            grid = curr

    if not isinstance(grid, (list, tuple)) or not grid:
        return []

    # Convert to pure list[list[int]]
    out: Grid2D = []
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            continue
        int_row: list[int] = []
        for cell in row:
            try:
                val = int(cell)
                # ARC-AGI-3 colors are 0-15 (16 colors)
                val = max(0, min(15, val))
                int_row.append(val)
            except (ValueError, TypeError):
                int_row.append(0)
        out.append(int_row)

    return out


def grid_to_hex_rows(grid: Grid2D) -> list[str]:
    """Convert each grid row to a string of single-digit hex color values (0-f)."""
    return ["".join(format(c, "x") for c in row) for row in grid]


def compute_grid_hash(grid: Grid2D) -> str:
    """Compute deterministic SHA-256 hash of grid contents."""
    canonical_repr = "\n".join(grid_to_hex_rows(grid))
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def normalize_state_name(state: Any) -> str:
    """Normalize game state enum or string to canonical uppercase string."""
    if hasattr(state, "name"):
        return str(getattr(state, "name")).split(".")[-1].upper()
    value = getattr(state, "value", None)
    if value is not None:
        return str(value).split(".")[-1].upper()
    text = str(state or "").split(".")[-1].strip().upper()
    return text or "IN_PROGRESS"


def normalize_action_name(action: Any) -> str:
    """Normalize action enum, integer or string to the canonical label.

    Delegates to the single competition vocabulary; the forbidden Undo label is
    preserved rather than rewritten so callers can detect and drop it.
    """
    name = normalize_action_id(action)
    return name or "ACTION1"


def crop_grid_border(grid: Grid2D, border: int = 1) -> tuple[Grid2D, int]:
    """Unconditionally crop 1px border pixels to eliminate outer system frame per §4.3."""
    if border <= 0 or not grid or not grid[0]:
        return grid, 0
    h = len(grid)
    w = len(grid[0])
    if h <= 2 * border + 1 or w <= 2 * border + 1:
        logger.warning(f"Grid {h}x{w} too small for {border}px crop (would leave < 2x2 workspace), skipping")
        return grid, 0
    return [row[border:w - border] for row in grid[border:h - border]], border


def normalize_observation(
    raw_obs: Mapping[str, Any],
    frame_index: int = 0,
    game_id: str = "",
    crop_border: int = 1,
) -> dict[str, Any]:
    """Normalize arbitrary environment observation into canonical dictionary."""
    obs_dict = dict(raw_obs)
    raw_grid = obs_dict.get("frame", obs_dict.get("grid"))
    grid = collapse_frame_axes(raw_grid)
    grid, crop_offset = crop_grid_border(grid, border=crop_border)

    available_actions_raw = obs_dict.get("available_actions", obs_dict.get("allowed_action_ids", ())) or ()
    raw_allowed = [normalize_action_id(a) for a in available_actions_raw]
    if not raw_allowed:
        # Default full action space if unspecified
        raw_allowed = list(LEGAL_ACTION_IDS)
    offered_forbidden = sorted({a for a in raw_allowed if a in FORBIDDEN_ACTION_IDS})
    if offered_forbidden:
        logger.warning(
            "Environment advertised forbidden actions %s; excluded from the action surface.",
            ",".join(offered_forbidden),
        )
    available_actions = [
        action
        for action in filter_legal_actions(raw_allowed)
        if action != RESET_ACTION_ID
    ]

    state_raw = obs_dict.get("state")
    if state_raw is None:
        metadata = dict(obs_dict.get("metadata", {}) or {})
        state_raw = metadata.get("state", "IN_PROGRESS")
    state = normalize_state_name(state_raw)

    levels_completed = obs_dict.get("levels_completed")
    if levels_completed is None:
        levels_completed = obs_dict.get("score", 0)
    try:
        levels_completed = int(levels_completed or 0)
    except (ValueError, TypeError):
        levels_completed = 0

    win_levels = obs_dict.get("win_levels", obs_dict.get("win_score"))
    if win_levels is not None:
        try:
            win_levels = int(win_levels)
        except (ValueError, TypeError):
            win_levels = None

    gid = str(obs_dict.get("game_id") or game_id or "anonymous_game")
    guid = getattr(raw_obs, "guid", obs_dict.get("guid"))

    return {
        "grid": grid,
        "grid_hash": compute_grid_hash(grid),
        "available_actions": available_actions,
        "allowed_action_ids": available_actions,
        "game_id": gid,
        "guid": guid,
        "state": state,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "full_reset": bool(obs_dict.get("full_reset", False)),
        "frame_index": frame_index,
        "crop_offset": crop_offset,
    }

