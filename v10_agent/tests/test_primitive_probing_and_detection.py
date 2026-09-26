"""Unit tests for sequential primitive probing, nested object motion, multi-vector affordances, and solver primitive preservation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.explorer_agent import (
    PrimitiveProbeManager,
    compute_probe_effect,
    classify_probe_affordance,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import parse_text_trajectory


def test_sequential_initial_sweep_and_reprobe_gating():
    """Verify discrete actions are planned in 1->2->3->4->5 order and dynamic reprobes do not preempt sweep."""
    manager = PrimitiveProbeManager(max_probes=16)
    allowed = ["ACTION5", "ACTION3", "ACTION1", "ACTION4", "ACTION2"]
    snap = extract_arga_snapshot([[0, 0], [0, 0]])
    pset = build_planning_set(snap, available_actions=allowed)

    probes = manager.plan_discrete_probes(pset, max_probes=16)
    assert len(probes) == 5
    probe_ids = [p.action_id for p in probes]
    # Must be strictly ordered 1 to 5 regardless of input order
    assert probe_ids == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    assert manager._initial_sweep_planned is True
    assert manager._initial_sweep_done is False

    # Simulate dynamic reprobe request while initial sweep is incomplete
    reprobes = manager.get_dynamic_reprobes("ACTION1", "selection indicator appeared on obj_0")
    # Must return empty list during initial sweep to prevent preemption
    assert reprobes == []

    # Record first 4 probes
    for act in ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]:
        manager.record_probe_result(act, {}, snap, {"grid": [[0, 0], [0, 0]]})
        assert manager._initial_sweep_done is False

    # Record 5th probe (completing sweep)
    manager.record_probe_result("ACTION5", {}, snap, {"grid": [[0, 0], [0, 0]]})
    assert manager._initial_sweep_done is True


def test_nested_child_motion_detection_inside_static_container():
    """Verify child object moving inside a static parent enclosure is detected as kinematic displacement."""
    # Outer frame (color 1, rows 0..29, cols 0..29) with inner space (color 0)
    grid0 = [[0 for _ in range(30)] for _ in range(30)]
    for r in range(30):
        grid0[r][0] = 1
        grid0[r][29] = 1
    for c in range(30):
        grid0[0][c] = 1
        grid0[29][c] = 1

    # Inner player block (color 4, rows 4..9, cols 4..9)
    for r in range(4, 10):
        for c in range(4, 10):
            grid0[r][c] = 4

    snap0 = extract_arga_snapshot(grid0)

    # grid1: player block moved DOWN by 6 cells (rows 10..15, cols 4..9), outer frame static
    grid1 = [[0 for _ in range(30)] for _ in range(30)]
    for r in range(30):
        grid1[r][0] = 1
        grid1[r][29] = 1
    for c in range(30):
        grid1[0][c] = 1
        grid1[29][c] = 1

    for r in range(10, 16):
        for c in range(4, 10):
            grid1[r][c] = 4

    pset = build_planning_set(snap0, available_actions=["ACTION1", "ACTION2"])
    effect = compute_probe_effect(snap0, {"grid": grid1}, planning_set=pset)
    assert "moved" in effect
    assert "dy=+6" in effect
    assert "DOWN" in effect
    assert "color transition" not in effect


def test_multi_vector_and_indicator_shift_detection():
    """Verify simultaneous player rewind (dy=-6) and corner indicator shift (dx=+4) are both preserved."""
    # Base grid: static board, player at rows 10..15, cols 4..9; indicator at rows 1..2, cols 1..2
    grid0 = [[0 for _ in range(30)] for _ in range(30)]
    for r in range(10, 16):
        for c in range(4, 10):
            grid0[r][c] = 4
    for r in range(1, 3):
        for c in range(1, 3):
            grid0[r][c] = 2

    snap0 = extract_arga_snapshot(grid0)

    # grid1: player moved back UP to rows 4..9 (dy=-6); indicator shifted right to cols 5..6 (dx=+4)
    grid1 = [[0 for _ in range(30)] for _ in range(30)]
    for r in range(4, 10):
        for c in range(4, 10):
            grid1[r][c] = 4
    for r in range(1, 3):
        for c in range(5, 7):
            grid1[r][c] = 2

    pset = build_planning_set(snap0, available_actions=["ACTION1", "ACTION5"])
    effect = compute_probe_effect(snap0, {"grid": grid1}, planning_set=pset)
    assert "dy=-6" in effect
    assert "UP" in effect
    assert "dx=+4" in effect
    assert "RIGHT" in effect


def test_solver_agent_preserves_primitives_not_in_manifest():
    """Verify parse_text_trajectory does not drop canonical engine primitives even if omitted from manifest."""
    manifest = {
        "functions": [
            {"name": "custom_macro", "parameters": [], "docstring": "A custom macro"}
        ]
    }
    response_text = """
HYPOTHESIS: test
PLAN:
step_1 = action1()
step_2 = action2()
step_3 = action5()
step_4 = custom_macro()
"""
    parsed = parse_text_trajectory(response_text, manifest_or_funcs=manifest)
    assert parsed is not None
    steps = parsed["candidates"][0]["steps"]
    fn_names = [s["dsl_function"] for s in steps]
    assert "action1" in fn_names
    assert "action2" in fn_names
    assert "action5" in fn_names
    assert "custom_macro" in fn_names


def test_unconfirmed_primitive_research_threshold_and_prompt():
    """Verify Explorer research_unconfirmed_primitives builds valid prompt and parses sequences."""
    from v10_agent.config import V10Config
    from v10_agent.explorer_agent import ExplorerAgent
    from v10_agent.memory_contours import EnvironmentSpecMemory

    config = V10Config()
    advisor = MagicMock()
    advisor.generate.return_value = '{"targeted_probe_sequences": [["ACTION2", "ACTION5"], ["ACTION5", "ACTION1"]]}'

    explorer = ExplorerAgent(config=config, advisor=advisor)
    snap = extract_arga_snapshot([[0, 0], [0, 0]])
    pset = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])
    env_mem = EnvironmentSpecMemory(game_id="game_1", level_id="level_0")

    # Record ACTION2 as confirmed, ACTION1, 3, 4, 5 as unconfirmed (80% unconfirmed >= 40%)
    explorer.probe_manager.available_discrete_actions.update(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])
    explorer.probe_manager.confirmed_effective_actions["ACTION2"] = "moved DOWN"
    for a in ["ACTION1", "ACTION3", "ACTION4", "ACTION5"]:
        explorer.probe_manager.inactive_actions.add(a)

    probes = explorer.research_unconfirmed_primitives(
        planning_set=pset,
        memory=env_mem,
        unconfirmed_actions=["ACTION1", "ACTION3", "ACTION4", "ACTION5"],
    )
    assert len(probes) == 4
    assert [p.action_id for p in probes] == ["ACTION2", "ACTION5", "ACTION5", "ACTION1"]
    assert explorer.probe_manager._primitive_research_done is True


def test_outer_border_cropped_no_raw_grid_false_positives():
    """Verify 1px outer border changes (engine timer bar) are cropped out and never trigger false action effects."""
    from v10_agent.observe import normalize_observation

    frame0 = [[0 for _ in range(10)] for _ in range(10)]
    frame0[0][3] = 9  # outer border timer pixel
    frame0[5][5] = 4  # interior object

    frame1 = [list(row) for row in frame0]
    frame1[0][3] = 1  # only the outer 1px border changed (9 -> 1)

    obs0 = normalize_observation({"frame": frame0, "state": "IN_PROGRESS"}, crop_border=True)
    obs1 = normalize_observation({"frame": frame1, "state": "IN_PROGRESS"}, crop_border=True)

    assert "raw_grid" not in obs0
    assert "raw_grid" not in obs1
    assert obs0["grid"] == obs1["grid"]

    snap0 = extract_arga_snapshot(obs0["grid"])
    effect = compute_probe_effect(snap0, obs1)
    assert "no visible effect" in effect


def test_composite_action_effect_accumulation():
    """Verify multiple distinct effects for the same action are accumulated as a composite action via ' | '."""
    from v10_agent.memory_contours import GameMemory

    manager = PrimitiveProbeManager(max_probes=16)
    grid0 = [[0, 2, 0], [0, 0, 0], [0, 0, 0]]
    grid1 = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    grid2 = [[0, 0, 0], [0, 0, 2], [0, 0, 0]]

    snap0 = extract_arga_snapshot(grid0)
    snap1 = extract_arga_snapshot(grid1)

    rec1 = manager.record_probe_result("ACTION5", {}, snap0, {"grid": grid1})
    rec2 = manager.record_probe_result("ACTION5", {}, snap1, {"grid": grid2})

    assert rec1.observed_effect in manager.confirmed_effective_actions["ACTION5"]
    assert rec2.observed_effect in manager.confirmed_effective_actions["ACTION5"]
    assert " | " in manager.confirmed_effective_actions["ACTION5"]

    gm = GameMemory(game_id="test_composite")
    gm.record_action_effect("ACTION5", rec1.observed_effect)
    gm.record_action_effect("ACTION5", rec2.observed_effect)
    assert " | " in gm.confirmed_action_effects["ACTION5"]
    assert rec1.observed_effect in gm.confirmed_action_effects["ACTION5"]
    assert rec2.observed_effect in gm.confirmed_action_effects["ACTION5"]


def test_judge_exact_displacement_and_executor_object_reindexing():
    """Verify exact displacement propositions (e.g. dy=6), child enclosure tracking, and multi-step object ID remapping."""
    from v10_agent.config import V10Config
    from v10_agent.judge import LayeredVerifier
    from v10_agent.memory_contours import EpistemicMemory
    from v10_agent.sandbox import SandboxExecutor
    from v10_agent.solver_agent import parse_expect_grammar_str
    from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
    from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
    from v10_agent.verification import VerificationBinder

    # Frame 1: hollow 3x3 player at rows 2..4, cols 10..12 (color 6) with 1x1 center dot at (3, 11) (color 7),
    # and a static obstacle at rows 6..6, cols 2..5 (color 8).
    def make_grid(player_top_row: int) -> list[list[int]]:
        g = [[0 for _ in range(24)] for _ in range(24)]
        # Static object at row 6, cols 2..5
        for c in range(2, 6):
            g[6][c] = 8
        # Hollow 3x3 player (color 6) + 1x1 center child (color 7)
        for r in range(player_top_row, player_top_row + 3):
            for c in range(10, 13):
                if r == player_top_row + 1 and c == 11:
                    g[r][c] = 7
                else:
                    g[r][c] = 6
        return g

    grid_f1 = make_grid(2)   # Player at rows 2..4 (above static object at row 6) -> player is obj_0 (A), child is obj_1 (B), static is obj_2 (C)
    grid_f2 = make_grid(8)   # Player moved down by +6 to rows 8..10 (below static object at row 6) -> static is obj_0 (A), player is obj_1 (B), child is obj_2 (C)
    grid_f3 = make_grid(14)  # Player moved down by +6 again to rows 14..16

    snap_f1 = extract_arga_snapshot(grid_f1)
    pset_f1 = build_planning_set(snap_f1, available_actions=["ACTION2"])

    # In Frame 1, A is obj_0 (player), B is obj_1 (child), C is obj_2 (static)
    exp_step1 = parse_expect_grammar_str("moved(A, 6, 0), moved(B, 6, 0), unchanged(C)", planning_set=pset_f1)
    exp_step2 = parse_expect_grammar_str("moved(A, 6, 0), moved(B, 6, 0), unchanged(C)", planning_set=pset_f1)

    cand = CandidateTrajectory(
        trajectory_id="cand_0",
        steps=[
            {"step_id": "s1", "dsl_function": "action2", "arguments": {}, "expected_propositions": exp_step1},
            {"step_id": "s2", "dsl_function": "action2", "arguments": {}, "expected_propositions": exp_step2},
        ],
    )
    pool = TrajectoryPool(proposal_id="prop_0", candidates=[cand])
    cfg = V10Config()
    verifier = LayeredVerifier(cfg)
    binder = VerificationBinder()
    executor = SymbolicTrajectoryExecutor(cfg, SandboxExecutor(), binder, verifier)
    ep_mem = EpistemicMemory(level_id="level_0")

    # Ground and evaluate Step 1 (Frame 1 -> Frame 2)
    grounded_s1 = binder.ground_step(cand.current_step(), pset_f1)
    res1 = executor.evaluate_transition(
        pending_step=grounded_s1,
        before_snapshot=snap_f1,
        after_obs={"grid": grid_f2, "state": "IN_PROGRESS"},
        planning_set=pset_f1,
        active_pool=pool,
        epistemic_memory=ep_mem,
        action_dict={"action_id": "ACTION2"},
    )
    assert res1.candidate_advanced is True
    assert res1.candidate_severed is False
    assert cand.cursor == 1

    # Verify Step 2's expected propositions were automatically remapped to Frame 2's object IDs
    snap_f2 = extract_arga_snapshot(grid_f2)
    pset_f2 = build_planning_set(snap_f2, available_actions=["ACTION2"])
    grounded_s2 = binder.ground_step(cand.current_step(), pset_f2)
    res2 = executor.evaluate_transition(
        pending_step=grounded_s2,
        before_snapshot=snap_f2,
        after_obs={"grid": grid_f3, "state": "IN_PROGRESS"},
        planning_set=pset_f2,
        active_pool=pool,
        epistemic_memory=ep_mem,
        action_dict={"action_id": "ACTION2"},
    )
    assert res2.candidate_advanced is True
    assert res2.candidate_severed is False
    assert cand.is_finished() is True

