"""Property-based and unit tests for coordinate disambiguation and DualView grounding.

Verifies:
1. PBT calculation of diff bounding boxes with inclusive bounds notation.
2. Corner and single-pixel coordinate accuracy.
3. DualView Grounding and parser over-segmentation compensation rules in system prompts.
4. Synchronization of local cropped workspace coordinates in probe logs.
"""

from __future__ import annotations

import re
from hypothesis import given, strategies as st, settings

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.explorer_agent import compute_probe_effect
from v10_agent.memory_contours import ProbeRecord
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import CODER_SYSTEM_PROMPT
from v10_agent.prompt_builders.explorer_prompt import (
    EXPLORER_SYSTEM_PROMPT,
    COORDINATE_HYPOTHESIS_SYSTEM_PROMPT,
    build_explorer_prompts,
    build_explorer_synthesis_prompt,
)
from v10_agent.prompt_builders.solver_prompt import SOLVER_SYSTEM_PROMPT, build_solver_prompts


@settings(max_examples=50)
@given(
    grid_w=st.integers(min_value=2, max_value=64),
    grid_h=st.integers(min_value=2, max_value=64),
    points=st.lists(
        st.tuples(st.integers(min_value=0, max_value=63), st.integers(min_value=0, max_value=63)),
        min_size=1,
        max_size=25,
    ),
)
def test_pbt_diff_bbox_inclusive_bounds(grid_w: int, grid_h: int, points: list[tuple[int, int]]):
    """Property: Any modified cell cluster produces exact, verified inclusive coordinate bounds."""
    # Filter points to fit within current grid dimensions
    valid_points = [(r, c) for r, c in set(points) if r < grid_h and c < grid_w]
    if not valid_points:
        valid_points = [(0, 0)]

    before_grid = [[0] * grid_w for _ in range(grid_h)]
    after_grid = [[0] * grid_w for _ in range(grid_h)]
    for r, c in valid_points:
        after_grid[r][c] = 2

    snap_before = extract_arga_snapshot(before_grid)
    effect = compute_probe_effect(snap_before, {"grid": after_grid})

    expected_min_r = min(r for r, _ in valid_points)
    expected_max_r = max(r for r, _ in valid_points)
    expected_min_c = min(c for _, c in valid_points)
    expected_max_c = max(c for _, c in valid_points)

    expected_pattern = f"cols {expected_min_c}..{expected_max_c} (inclusive), rows {expected_min_r}..{expected_max_r} (inclusive)"
    assert expected_pattern in effect

    # Invariant: 0 <= min_c <= max_c < width and 0 <= min_r <= max_r < height
    assert 0 <= expected_min_c <= expected_max_c < grid_w
    assert 0 <= expected_min_r <= expected_max_r < grid_h

    # Ensure dangerous/ambiguous tokens are absent
    assert "range [" not in effect
    assert re.search(r"region\s*\(\d+,\d+\)", effect) is None


def test_diff_bbox_corners():
    """Verify single-cell corner clicks at (0, 0) and (W-1, H-1) format precisely."""
    w, h = 30, 20
    # Top-left corner (0, 0)
    g_before = [[0] * w for _ in range(h)]
    g_after = [[0] * w for _ in range(h)]
    g_after[0][0] = 3
    snap = extract_arga_snapshot(g_before)
    eff_tl = compute_probe_effect(snap, {"grid": g_after})
    assert "cols 0..0 (inclusive), rows 0..0 (inclusive)" in eff_tl

    # Bottom-right corner (h-1, w-1) -> row 19, col 29
    g_after_br = [[0] * w for _ in range(h)]
    g_after_br[h - 1][w - 1] = 4
    eff_br = compute_probe_effect(snap, {"grid": g_after_br})
    assert f"cols {w - 1}..{w - 1} (inclusive), rows {h - 1}..{h - 1} (inclusive)" in eff_br


def test_dualview_grounding_in_solver_prompt():
    """Verify SOLVER_SYSTEM_PROMPT includes DualView Grounding & Cartesian coordinates."""
    assert "strictly Cartesian" in SOLVER_SYSTEM_PROMPT
    assert "x is horizontal (column, 0 <= x < width), y is vertical (row, 0 <= y < height)" in SOLVER_SYSTEM_PROMPT
    assert "DUALVIEW GROUNDING & PARSER OVER-SEGMENTATION:" in SOLVER_SYSTEM_PROMPT
    assert "Raw Frame Image is Ground Truth:" in SOLVER_SYSTEM_PROMPT
    assert "Parser Over-Segmentation:" in SOLVER_SYSTEM_PROMPT
    assert "use them for your solution hypotheses" in SOLVER_SYSTEM_PROMPT
    assert "apply these designations in your response" in SOLVER_SYSTEM_PROMPT
    assert "action7" not in SOLVER_SYSTEM_PROMPT
    assert "3x3" not in SOLVER_SYSTEM_PROMPT


def test_dualview_grounding_in_explorer_prompts():
    """Verify EXPLORER_SYSTEM_PROMPT and COORDINATE_HYPOTHESIS_SYSTEM_PROMPT have DualView & Cartesian rules."""
    # Explorer system prompt
    assert "DUALVIEW GROUNDING & PARSER OVER-SEGMENTATION:" in EXPLORER_SYSTEM_PROMPT
    assert "Raw frame image is Ground Truth" in EXPLORER_SYSTEM_PROMPT
    assert "COORDINATE CONVENTION:" in EXPLORER_SYSTEM_PROMPT
    assert "All spatial coordinates are strictly Cartesian: x = column" in EXPLORER_SYSTEM_PROMPT
    assert "action6(x=col, y=row)" in EXPLORER_SYSTEM_PROMPT

    # Coordinate hypothesis system prompt
    assert "COORDINATE CONVENTION (STRICT CARTESIAN):" in COORDINATE_HYPOTHESIS_SYSTEM_PROMPT
    assert "x is the HORIZONTAL coordinate (column index, 0 <= x < width)" in COORDINATE_HYPOTHESIS_SYSTEM_PROMPT
    assert "y is the VERTICAL coordinate (row index, 0 <= y < height)" in COORDINATE_HYPOTHESIS_SYSTEM_PROMPT
    assert "geometric center" in COORDINATE_HYPOTHESIS_SYSTEM_PROMPT
    assert "composite multi-color tiles" in COORDINATE_HYPOTHESIS_SYSTEM_PROMPT


def test_coder_prompt_coordinate_axes():
    """Verify CODER_SYSTEM_PROMPT specifies horizontal col and vertical row."""
    assert "x is the column (horizontal) and y is the row (vertical)" in CODER_SYSTEM_PROMPT


def test_probe_log_coordinate_normalization():
    """Verify probe logs format local coordinates when engine coordinates and crop_offset are stored."""
    grid = [[0, 1], [0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6", "RESET"], crop_offset=1)

    probe = ProbeRecord(
        probe_id="p1",
        action_id="ACTION6",
        # Stored with engine coords 42, 48 and crop_offset 1 -> local is 41, 47
        action_data={"x": 42, "y": 48, "crop_offset": 1},
        observed_effect="color transition: in bbox: cols 40..42 (inclusive), rows 46..48 (inclusive)",
        confidence=0.9,
    )

    _, user_p = build_explorer_prompts(pset, probe_history=[probe])
    assert "PRIOR PROBE HISTORY (format: action6(x=col, y=row)):" in user_p
    assert "#1 ACTION6(x=41, y=47) ->" in user_p
    # Ensure internal offset bookkeeping keys are not leaked into LLM text
    assert "crop_offset" not in user_p

    _, synth_user_p = build_explorer_synthesis_prompt(
        pset,
        confirmed_actions={"ACTION6": "color transition"},
        probe_history=[probe],
    )
    assert "CONFIRMED ACTION EFFECTS (from empirical probe deltas, format: action6(x=col, y=row)):" in synth_user_p
    assert "RECENT PROBE EXECUTION LOG (format: action6(x=col, y=row)):" in synth_user_p
    assert "#1 ACTION6(x=41, y=47) ->" in synth_user_p


def test_solver_prompt_evidence_log_unambiguous_convention():
    """Verify build_solver_prompts asserts Cartesian convention and excludes empirical deduction trap."""
    grid = [[0, 1], [0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6"])
    manifest = {"functions": [{"name": "action6", "parameters": [{"name": "x", "type": "int"}, {"name": "y", "type": "int"}]}]}

    _, user_p = build_solver_prompts(manifest, pset)
    assert "All coordinate actions strictly follow Cartesian convention: action6(x=col, y=row)" in user_p
    assert "Bounding boxes in the evidence log declare x cols and y rows (inclusive)." in user_p
    assert "empirical questions: read them from the evidence log" not in user_p
