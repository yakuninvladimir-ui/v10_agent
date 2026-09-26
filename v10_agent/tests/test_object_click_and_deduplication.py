"""Tests for object-targeted click, deduplication with spatial NMS, and post-click verification probing."""

import pytest
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.explorer_agent import PrimitiveProbeManager
from v10_agent.fallback_symbolic import SymbolicFallbackEngine
from v10_agent.planning_set import build_planning_set
from v10_agent.sandbox import SandboxAPI, SandboxExecutor
from v10_agent.solver_agent import _parse_fn_call_args, parse_text_trajectory


def test_planning_set_spatial_nms_and_zero_duplicate_candidates():
    """Verify PlanningSet eliminates duplicate coordinates (TL, BR) and applies spatial NMS."""
    # Create grid with a 4x4 object and an adjacent 2x2 object
    grid = [[0] * 20 for _ in range(20)]
    # Object 1: at (2..5, 2..5)
    for r in range(2, 6):
        for c in range(2, 6):
            grid[r][c] = 1
    # Object 2: right next to Object 1 at (2..3, 6..7) - within 2 cells
    for r in range(2, 4):
        for c in range(6, 8):
            grid[r][c] = 2
    # Object 3: far away at (15..17, 15..17)
    for r in range(15, 18):
        for c in range(15, 18):
            grid[r][c] = 3

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6", "RESET"])

    # Verify no TL or BR candidate IDs exist
    assert not any("coord_tl" in c.candidate_id for c in pset.coordinate_candidates)
    assert not any("coord_br" in c.candidate_id for c in pset.coordinate_candidates)

    # Verify all coordinate candidates are separated by at least 2.5 cells
    coords = pset.coordinate_candidates
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = ((coords[i].x - coords[j].x) ** 2 + (coords[i].y - coords[j].y) ** 2) ** 0.5
            assert dist >= 2.4, f"Candidates {coords[i]} and {coords[j]} violate NMS min distance!"


def test_sandbox_api_click_object():
    """Verify SandboxAPI.click_object resolves object alias to exact centroid."""
    grid = [[0] * 10 for _ in range(10)]
    # 3x3 square at row 2..4, col 4..6 -> center is (row 3, col 5)
    for r in range(2, 5):
        for c in range(4, 7):
            grid[r][c] = 1

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6"])
    api = SandboxAPI(pset)

    alias = list(pset.object_real_to_alias.values())[0]
    effect = api.click_object(target=alias)

    assert effect.declared_action.action_id == "ACTION6"
    assert effect.declared_action.data["x"] == 5  # col
    assert effect.declared_action.data["y"] == 3  # row
    assert effect.declared_action.reasoning["target_alias"] == alias


def test_solver_parsing_object_click_target():
    """Verify solver parser accepts click(target=C), click(C), and action6(target=C)."""
    # 1. AST parser for keyword args with Name node
    fn, kwargs = _parse_fn_call_args("click(target=C)")
    assert fn == "click"
    assert kwargs.get("target") == "C"

    fn, kwargs = _parse_fn_call_args("action6(target=AL)")
    assert fn == "action6"
    assert kwargs.get("target") == "AL"

    # 2. Manifest-aware full candidate extraction
    manifest = {
        "functions": [
            {"name": "action6", "parameters": [{"name": "target", "type": "str"}, {"name": "x", "type": "int"}, {"name": "y", "type": "int"}]},
            {"name": "click", "parameters": [{"name": "target", "type": "str"}, {"name": "x", "type": "int"}, {"name": "y", "type": "int"}]},
        ]
    }
    solver_output = """
    <trajectory_1>
    click(target=B) EXPECT: color(B)=8
    action6(target=C) EXPECT: moved(C, 0, 1)
    click(D)
    </trajectory_1>
    """
    pkg = parse_text_trajectory(solver_output, manifest_or_funcs=manifest)
    cands = pkg["candidates"]
    assert len(cands) == 1
    steps = cands[0]["steps"]
    assert len(steps) == 3
    assert steps[0]["dsl_function"] == "click"
    assert steps[0]["arguments"]["target"] == "B"
    assert steps[1]["dsl_function"] == "action6"
    assert steps[1]["arguments"]["target"] == "C"
    assert steps[2]["arguments"]["target"] == "D"


def test_explorer_post_click_verification_probe():
    """Verify Explorer schedules verification probe when click mutates grid without clear motion."""
    mgr = PrimitiveProbeManager(max_probes=10)
    mgr.available_discrete_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

    # Simulate an ambiguous click result (e.g. 5 cells changed color, but actor didn't translate)
    effect_summary = "color transition (stamp/draw): 5 cells changed color [9->8 (5 cells)] in bbox: cols 35..40, rows 35..40"
    reprobes = mgr.get_dynamic_reprobes("ACTION6", effect_summary, max_steps=2)

    assert len(reprobes) > 0
    assert reprobes[0].action_id in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
    assert reprobes[0].reasoning["source"] == "post_click_verification_probe"


def test_fallback_engine_unclicked_candidate_deduplication():
    """Verify SymbolicFallbackEngine clicks unclicked canonical candidates without looping."""
    grid = [[0] * 20 for _ in range(20)]
    grid[2][2] = 1
    grid[15][15] = 2

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION6"])
    cfg = V10Config()
    engine = SymbolicFallbackEngine(cfg)

    # Step 1
    engine.step_counter = 1
    eff1 = engine.select_fallback_action(pset)
    cand1 = eff1.declared_action.data["target"]

    # Step 2
    engine.step_counter = 2
    eff2 = engine.select_fallback_action(pset)
    cand2 = eff2.declared_action.data["target"]

    # Ensure different targets were selected
    assert cand1 != cand2, "Fallback engine looped on the same target!"
