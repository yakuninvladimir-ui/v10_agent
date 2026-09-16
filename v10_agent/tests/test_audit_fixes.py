"""Regression tests for V10 audit fixes.

Tests:
1. Multi-level progression: ensure levels_completed > 0 does not lock is_won to True on level 1+.
2. Candidate pool exhaustion on level 1 triggers replan and clean reset.
3. Sandbox timeout: verify infinite loop raises SandboxTimeoutError.
4. Policy dispatch: verify live unsevered OMIT branches are dispatched.
"""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import BranchSignature, EpistemicMemory, SyntaxErrorMemory
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.policy import ActionSelectionPolicy
from v10_agent.sandbox import SandboxedModule, SandboxExecutor, SandboxTimeoutError
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.types import PropositionSet
from v10_agent.verification import GroundedStep, VerificationBinder


def _make_sym_exec() -> SymbolicTrajectoryExecutor:
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)
    return SymbolicTrajectoryExecutor(config, executor, binder, verifier)


def test_level_completion_does_not_lock_is_won_on_subsequent_steps():
    sym_exec = _make_sym_exec()
    ep_mem = EpistemicMemory(level_id="level_1")
    pool = TrajectoryPool(
        proposal_id="p1",
        candidates=[
            CandidateTrajectory(
                trajectory_id="c1",
                steps=[
                    {"step_id": "s1", "dsl_function": "act1", "arguments": {}},
                    {"step_id": "s2", "dsl_function": "act1", "arguments": {}},
                ],
            )
        ],
    )
    step = GroundedStep(
        step_id="s1",
        dsl_function="act1",
        arguments={},
        expected_propositions=PropositionSet(),
    )
    snap = extract_arga_snapshot([[0, 1, 0]])
    snap.levels_completed = 1
    pset = build_planning_set(snapshot=snap, available_actions=["ACTION1"])

    # Step on level 1: levels_completed was 1 before and is still 1 after
    after_obs = {"grid": [[0, 1, 0]], "levels_completed": 1, "state": "RUNNING"}

    eval_res = sym_exec.evaluate_transition(
        pending_step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )

    assert eval_res.candidate_advanced is True
    assert eval_res.replan_needed is False
    assert eval_res.reset_needed is False


def test_candidate_exhaustion_on_level_1_triggers_replan_and_reset():
    sym_exec = _make_sym_exec()
    ep_mem = EpistemicMemory(level_id="level_1")
    cand = CandidateTrajectory(
        trajectory_id="c1",
        steps=[{"step_id": "s1", "dsl_function": "act1", "arguments": {}}],
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand])
    step = GroundedStep(
        step_id="s1",
        dsl_function="act1",
        arguments={},
        expected_propositions=PropositionSet(),
    )
    snap = extract_arga_snapshot([[0, 1, 0]])
    snap.levels_completed = 1
    pset = build_planning_set(snapshot=snap, available_actions=["ACTION1"])

    after_obs = {"grid": [[0, 1, 0]], "levels_completed": 1, "state": "RUNNING"}

    eval_res = sym_exec.evaluate_transition(
        pending_step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )

    assert eval_res.candidate_advanced is True
    assert eval_res.replan_needed is True
    assert eval_res.reset_needed is True


def test_sandbox_timeout_prevents_infinite_loop():
    executor = SandboxExecutor(timeout_seconds=0.3)
    source = """
def infinite_step(api):
    x = 0
    while True:
        x += 1
    return api.declare_environment_action("ACTION1")
"""
    manifest = {
        "functions": [
            {"name": "infinite_step", "parameters": []}
        ]
    }
    module = executor.load_module(source, manifest)
    pset = build_planning_set(extract_arga_snapshot([[0, 1]]), ["ACTION1"])

    with pytest.raises(SandboxTimeoutError):
        executor.execute(module, "infinite_step", {}, pset)


def test_policy_dispatches_live_omit_branch():
    policy = ActionSelectionPolicy()
    ep_mem = EpistemicMemory(level_id="level_0")
    cand1 = CandidateTrajectory(
        trajectory_id="traj_omit",
        steps=[{"step_id": "s1", "dsl_function": "act_omit", "arguments": {}}],
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand1])

    # Sever cand1 from standard candidate traversal
    cand1.sever()
    assert pool.active_candidate() is None

    # Now add to live omit branches and re-enable active step
    cand1.active = True
    cand1.cursor = 0
    branch = BranchSignature(
        signature_id="sig_omit_1",
        trajectory_id="traj_omit",
        last_step_id="s0",
        expected_propositions=[],
        observed_propositions=[],
    )
    ep_mem.live_omit_branches.append(branch)
    pset = build_planning_set(extract_arga_snapshot([[0, 1]]), ["ACTION1"])

    step, strategy = policy.select_next_step(pool, ep_mem, pset)
    assert step is not None
    assert step["dsl_function"] == "act_omit"
    assert strategy in ("solver_candidate", "branch_dispatch")


def test_is_level_transient_preserves_kinematics_displacement():
    """Kinematics facts containing 'displacement' survive level transition, while level-transient notes are purged."""
    from v10_agent.memory_contours import GameMemory

    gm = GameMemory(game_id="game_test")
    gm.tier1_kinematics_and_topology.append("Object motion: discrete displacement of dy=-3, dx=0 per action")
    gm.tier1_kinematics_and_topology.append("Object motion: discrete displacement of 3 cells")
    gm.tier1_kinematics_and_topology.append("Transient note: failed at row 4, col 5 (took 5 steps)")

    gm.handle_level_transition()

    # The 2 displacement facts should survive
    assert any("displacement of dy=-3, dx=0" in s for s in gm.tier1_kinematics_and_topology)
    assert any("displacement of 3 cells" in s for s in gm.tier1_kinematics_and_topology)
    # The transient failure note should be purged
    assert not any("took 5 steps" in s for s in gm.tier1_kinematics_and_topology)


def test_format_empirical_context_iso2_quarantines_curriculum():
    """ISO-2: DSLCoder empirical context strictly excludes curriculum winning patterns and goals."""
    from v10_agent.memory_contours import GameMemory

    gm = GameMemory(game_id="game_test")
    gm.record_action_effect("ACTION1", "Object moved UP by dy=-1, dx=0")
    gm.record_level_solution(
        level_id="level_0",
        setup_summary="Two objects separated by 5 rows",
        invariant_rule="Reach spatial contact between obj_0 and obj_1",
        winning_macro="action1() x 5",
    )

    coder_ctx = gm.format_empirical_context(include_curriculum=False)
    solver_ctx = gm.format_empirical_context(include_curriculum=True)

    # Coder context has physics but NO curriculum goals/macros
    assert "ACTION1: Object moved UP" in coder_ctx
    assert "CURRICULUM INVARIANTS & WINNING PATTERNS" not in coder_ctx
    assert "Reach spatial contact" not in coder_ctx

    # Solver context includes curriculum winning patterns
    assert "CURRICULUM INVARIANTS & WINNING PATTERNS" in solver_ctx
    assert "Reach spatial contact" in solver_ctx


def test_layered_verifier_emits_irrelevant_on_inessential_transition():
    """LayeredVerifier returns Ternary.IRRELEVANT when expected consequence is inessential (Brusentsov OMIT x'y')."""
    from v10_agent.types import AtomicProposition

    verifier = LayeredVerifier(V10Config())
    grid = [[0, 1, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1"])

    # Expected proposition that is inessential / not applicable (family not matching observed)
    step = GroundedStep(
        step_id="s1",
        dsl_function="noop_step",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id="obj_nonexistent", predicate="delta_r", value=1)
        ]),
    )

    after_obs = {"grid": grid, "state": "RUNNING"}
    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs=after_obs,
        planning_set=pset,
    )

    assert judgment.verdict == Ternary.IRRELEVANT
    assert "omit" in judgment.explanation.lower() or "inessential" in judgment.explanation.lower()


def test_solver_agent_parses_expect_propositions():
    """SolverAgent extracts EXPECT: clauses into candidate step expected_propositions."""
    from v10_agent.solver_agent import parse_text_trajectory

    text = """
<trajectory_1>
move_up(obj="obj_0") EXPECT: dy=-3, dx=0
move_right(obj="obj_0")
</trajectory_1>
"""
    manifest_map = {"move_up": {}, "move_right": {}}
    pkg = parse_text_trajectory(text, manifest_or_funcs=manifest_map)

    assert pkg is not None
    candidates = pkg.get("candidates", [])
    assert len(candidates) == 1
    steps = candidates[0].get("steps", [])
    assert len(steps) == 2

    # Step 1 should have expected_propositions parsed from EXPECT:
    s1 = steps[0]
    assert s1["dsl_function"] == "move_up"
    props = s1.get("expected_propositions", [])
    assert len(props) == 2
    preds = {p["predicate"]: p["value"] for p in props}
    assert preds["dy"] == -3
    assert preds["dx"] == 0

    # Step 2 should have empty expected_propositions
    s2 = steps[1]
    assert s2["dsl_function"] == "move_right"
    assert len(s2.get("expected_propositions", [])) == 0


def test_solver_preserves_model_candidate_order_without_synth_override():
    """Model-generated candidate trajectories are preserved in original order and not hijacked by synthesis."""
    import json
    from unittest.mock import MagicMock
    from v10_agent.solver_agent import SolverAgent

    config = V10Config()
    advisor = MagicMock()
    # Advisor returns JSON package with 2 distinct candidates
    advisor.generate.return_value = json.dumps({
        "candidates": [
            {
                "trajectory_id": "model_traj_01",
                "steps": [{"dsl_function": "action1", "arguments": {}}],
                "confidence": 0.9,
            },
            {
                "trajectory_id": "model_traj_02",
                "steps": [{"dsl_function": "action2", "arguments": {}}],
                "confidence": 0.8,
            },
        ]
    })
    solver = SolverAgent(config, advisor)
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    pset = build_planning_set(extract_arga_snapshot(grid), ["ACTION1", "ACTION2"])
    ep_mem = EpistemicMemory(level_id="level_0")

    pkg = solver.generate_trajectory_package(
        manifest={"functions": [{"name": "action1"}, {"name": "action2"}]},
        planning_set=pset,
        epistemic_memory=ep_mem,
    )

    assert pkg is not None
    candidates = pkg["candidates"]
    assert len(candidates) >= 2
    # Model's first candidate must be at index 0
    assert candidates[0]["trajectory_id"] == "model_traj_01"
    assert candidates[1]["trajectory_id"] == "model_traj_02"


def test_session_hybrid_pipeline_activation():
    """When both discrete actions and ACTION6 are available, active_pipeline is set to 'hybrid'."""
    from v10_agent.session import GameSession
    from unittest.mock import MagicMock

    config = V10Config(enable_primitive_probing=False)
    session = GameSession(config, MagicMock())

    grid = [[0, 1, 0], [0, 0, 0]]
    # Hybrid environment with discrete actions AND coordinate action
    available_actions = ["ACTION1", "ACTION2", "ACTION6", "RESET"]

    _ = session.act({"grid": grid, "available_actions": available_actions, "state": "IN_PROGRESS"})
    assert session.active_pipeline == "hybrid"

