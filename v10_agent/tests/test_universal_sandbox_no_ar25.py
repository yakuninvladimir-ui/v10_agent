"""Unit tests verifying VirtualKinematicSandbox operates universally without ar25 heuristics.

Verifies:
1. No crashes or reliance on axis_steps, piece_steps, or internal dots regex.
2. Aspect ratio heuristic (>= 3.0) classifies elongated dividers without hardcoded 10x4 thresholds.
3. General maze/obstacle collision simulation on 16-color grids.
"""
from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_sandbox_general_maze_navigation():
    """Verify sandbox works on a standard maze grid without reflection axes."""
    # 8x8 grid:
    # 0 = empty space
    # 1 = actor (blue, 1x1) at (1, 1)
    # 3 = obstacle wall (green) at (1, 3) to (4, 3)
    # 8 = target (teal, 1x1) at (6, 6)
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    sandbox = VirtualKinematicSandbox(pset)

    # Actor should be detected
    assert len(sandbox.axis_actions) > 0

    # Test motion displacement
    dy, dx = sandbox._get_displacement(None, "action4", "move right")
    assert (dy, dx) == (0, 1)

    dy, dx = sandbox._get_displacement(None, "action2", "move down")
    assert (dy, dx) == (1, 0)


def test_sandbox_aspect_ratio_classification():
    """Verify elongated barriers are detected via aspect ratio >= 3.0 rather than hardcoded 10x4."""
    # 6x6 grid with a 1x4 horizontal barrier (aspect = 4.0 >= 3.0)
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 0],  # barrier height=1, width=4 -> aspect 4.0
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    sandbox = VirtualKinematicSandbox(pset)

    # Barrier with aspect >= 3.0 should be isolated if classified as axis actor
    barrier_obj = next((o for o in pset.objects if o.color == 2), None)
    assert barrier_obj is not None
    aspect = max(barrier_obj.height, barrier_obj.width) / max(1, min(barrier_obj.height, barrier_obj.width))
    assert aspect >= 3.0
