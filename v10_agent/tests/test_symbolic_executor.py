"""Unit tests for SymbolicTrajectoryExecutor and Circuit Breaker."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import EpistemicMemory, MemoryContourManager, SyntaxErrorMemory
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.sandbox import SandboxExecutor
from v10_agent.symbolic_executor import StepExecutionVerdict, SymbolicTrajectoryExecutor
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.types import EffectDeclaration
from v10_agent.verification import VerificationBinder


@pytest.fixture
def dummy_planning_set() -> PlanningSet:
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    return build_planning_set(
        snapshot=snapshot,
        available_actions=["ACTION1", "ACTION2", "RESET"],
    )


def test_sandbox_argument_filtering():
    """Verify that unexpected keyword arguments (e.g. obj) are safely filtered out."""
    executor = SandboxExecutor()
    code = """
def action1(api):
    return api.declare_environment_action("ACTION1")
"""
    manifest = {
        "functions": [{"name": "action1", "parameters": []}]
    }
    module = executor.load_module(code, manifest)

    planning_set = build_planning_set(extract_arga_snapshot([[0]]), ["ACTION1", "RESET"])
    # Pass extraneous 'obj' argument that function doesn't accept
    effect = executor.execute(
        module=module,
        function_name="action1",
        arguments={"obj": "obj_0"},
        planning_set=planning_set,
    )
    assert isinstance(effect, EffectDeclaration)
    assert effect.declared_action.action_id == "ACTION1"


def test_pre_verification_severs_severed_step(dummy_planning_set):
    """If a step was previously severed in EpistemicMemory, pre-verification rejects it immediately."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)

    ep_mem = EpistemicMemory(level_id="l0")
    syntax_mem = SyntaxErrorMemory(level_id="l0")
    ep_mem.sever_branch("action1")

    code = "def action1(api):\n    return api.declare_environment_action('ACTION1')"
    module = executor.load_module(code, {"functions": [{"name": "action1"}]})

    pool = TrajectoryPool(
        proposal_id="p1",
        candidates=[
            CandidateTrajectory(
                trajectory_id="c1",
                steps=[{"step_id": "s1", "dsl_function": "action1", "arguments": {}}],
            )
        ],
    )

    res = sym_exec.prepare_and_execute_step(pool, dummy_planning_set, module, ep_mem, syntax_mem)
    assert res.verdict == StepExecutionVerdict.PRE_VERIFICATION_FAILED
    assert res.circuit_broken is True
    # Candidate should be severed
    assert not pool.candidates[0].active
    assert pool.active_candidate() is None


def test_pre_verification_missing_dsl_function(dummy_planning_set):
    """If candidate calls a function missing from module namespace, reject and sever candidate."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)

    ep_mem = EpistemicMemory(level_id="l0")
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    code = "def action1(api):\n    return api.declare_environment_action('ACTION1')"
    module = executor.load_module(code, {"functions": [{"name": "action1"}]})

    pool = TrajectoryPool(
        proposal_id="p1",
        candidates=[
            CandidateTrajectory(
                trajectory_id="c1",
                steps=[{"step_id": "s1", "dsl_function": "nonexistent_fn", "arguments": {}}],
            )
        ],
    )

    res = sym_exec.prepare_and_execute_step(pool, dummy_planning_set, module, ep_mem, syntax_mem)
    assert res.verdict == StepExecutionVerdict.PRE_VERIFICATION_FAILED
    assert res.circuit_broken is True
    assert not pool.candidates[0].active
    assert len(syntax_mem.entries) == 1


def test_circuit_breaker_on_sandbox_runtime_exception(dummy_planning_set):
    """If sandbox execution raises a runtime exception, circuit breaker severs candidate and records error."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)

    ep_mem = EpistemicMemory(level_id="l0")
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    # Code that raises runtime exception
    code = """
def bad_action(api):
    raise ValueError("Unexpected board state crash")
"""
    module = executor.load_module(code, {"functions": [{"name": "bad_action"}]})

    pool = TrajectoryPool(
        proposal_id="p1",
        candidates=[
            CandidateTrajectory(
                trajectory_id="c1",
                steps=[{"step_id": "s1", "dsl_function": "bad_action", "arguments": {}}],
            )
        ],
    )

    res = sym_exec.prepare_and_execute_step(pool, dummy_planning_set, module, ep_mem, syntax_mem)
    assert res.verdict == StepExecutionVerdict.SANDBOX_EXECUTION_FAILED
    assert res.circuit_broken is True
    # Candidate permanently severed
    assert not pool.candidates[0].active
    assert "bad_action" in ep_mem.severed_null_signatures or "s1" in ep_mem.severed_null_signatures
    assert len(syntax_mem.entries) == 1


def test_transition_evaluation_follow_and_null(dummy_planning_set):
    """Test transition evaluation: TRUE advances cursor, FALSE severs candidate and requests reset."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)
    ep_mem = EpistemicMemory(level_id="l0")

    cand = CandidateTrajectory(
        trajectory_id="c1",
        steps=[
            {
                "step_id": "s1",
                "dsl_function": "action1",
                "arguments": {},
                "expected_propositions": [
                    {"family": "metric_sign", "subject_id": dummy_planning_set.object_ids[0], "predicate": "row_delta", "value": 1}
                ],
            },
            {
                "step_id": "s2",
                "dsl_function": "action1",
                "arguments": {},
            },
        ],
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand])
    grounded_s1 = binder.ground_step(cand.steps[0], dummy_planning_set)

    # 1. Contradiction: object actually moved UP (-1) instead of DOWN (+1)
    before_snap = extract_arga_snapshot([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    after_grid_contradict = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    res_false = sym_exec.evaluate_transition(
        pending_step=grounded_s1,
        before_snapshot=before_snap,
        after_obs={"grid": after_grid_contradict},
        planning_set=dummy_planning_set,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )
    assert res_false.verdict == Ternary.FALSE
    assert res_false.candidate_severed is True
    assert res_false.reset_needed is True
    assert res_false.replan_needed is True
    assert not cand.active


def test_candidate_exhaustion_triggers_replan(dummy_planning_set):
    """When cand1 finishes in a multi-candidate pool, replan_needed MUST be False (advance to cand2).
    Only when the pool is fully exhausted must replan_needed be True."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)
    ep_mem = EpistemicMemory(level_id="l0")

    # 1-step candidate in a multi-candidate pool
    cand1 = CandidateTrajectory(
        trajectory_id="c1",
        steps=[{"step_id": "s1", "dsl_function": "action1", "arguments": {}}],
    )
    cand2 = CandidateTrajectory(
        trajectory_id="c2",
        steps=[{"step_id": "s1", "dsl_function": "action2", "arguments": {}}],
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand1, cand2])
    grounded_s1 = binder.ground_step(cand1.steps[0], dummy_planning_set)

    before_snap = extract_arga_snapshot([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    after_grid_moved = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]
    res_cand1 = sym_exec.evaluate_transition(
        pending_step=grounded_s1,
        before_snapshot=before_snap,
        after_obs={"grid": after_grid_moved},
        planning_set=dummy_planning_set,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )
    assert res_cand1.verdict in (Ternary.TRUE, Ternary.IRRELEVANT)
    assert cand1.is_finished() is True
    # Invariant: replan_needed MUST be False because cand2 exists in pool!
    assert res_cand1.replan_needed is False
    assert res_cand1.reset_needed is True
    assert pool.active_candidate() is cand2

    # Now execute cand2's step
    grounded_s2 = binder.ground_step(cand2.steps[0], dummy_planning_set)
    res_cand2 = sym_exec.evaluate_transition(
        pending_step=grounded_s2,
        before_snapshot=before_snap,
        after_obs={"grid": after_grid_moved},
        planning_set=dummy_planning_set,
        active_pool=pool,
        epistemic_memory=ep_mem,
    )
    assert res_cand2.verdict in (Ternary.TRUE, Ternary.IRRELEVANT)
    assert cand2.is_finished() is True
    # Now all candidates exhausted: replan_needed MUST be True!
    assert res_cand2.replan_needed is True
    assert res_cand2.reset_needed is True
    assert pool.active_candidate() is None


def test_parameterized_signature_severing(dummy_planning_set):
    """Verify that severing action6(x=10, y=12) does NOT sever action6(x=5, y=5)."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)

    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)

    ep_mem = EpistemicMemory(level_id="l0")
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    # Sever only action6(x=10, y=12)
    ep_mem.sever_branch("action6(x=10, y=12)")

    code = """
def action6(api, x=0, y=0):
    return api.declare_environment_action("ACTION1")
"""
    manifest = {
        "functions": [{"name": "action6", "parameters": [{"name": "x"}, {"name": "y"}]}]
    }
    module = executor.load_module(code, manifest)

    # Candidate 1 calls action6(x=10, y=12) -> should be severed
    cand1 = CandidateTrajectory(
        trajectory_id="c1",
        steps=[{"step_id": "s1", "dsl_function": "action6", "arguments": {"x": 10, "y": 12}}],
    )
    # Candidate 2 calls action6(x=5, y=5) -> should NOT be severed
    cand2 = CandidateTrajectory(
        trajectory_id="c2",
        steps=[{"step_id": "s2", "dsl_function": "action6", "arguments": {"x": 5, "y": 5}}],
    )
    pool = TrajectoryPool(proposal_id="p1", candidates=[cand1, cand2])

    res1 = sym_exec.prepare_and_execute_step(pool, dummy_planning_set, module, ep_mem, syntax_mem)
    assert res1.verdict == StepExecutionVerdict.PRE_VERIFICATION_FAILED
    assert not cand1.active

    # Next active candidate is cand2
    assert pool.active_candidate() is cand2
    res2 = sym_exec.prepare_and_execute_step(pool, dummy_planning_set, module, ep_mem, syntax_mem)
    assert res2.verdict == StepExecutionVerdict.SUCCESS
    assert cand2.active


def test_full_trajectory_upfront_verification_bounds_and_collisions():
    """Verify upfront trajectory verification approves valid trajectories and rejects boundary/collision violations."""
    config = V10Config()
    executor = SandboxExecutor()
    binder = VerificationBinder()
    verifier = LayeredVerifier(config)
    sym_exec = SymbolicTrajectoryExecutor(config, executor, binder, verifier)

    # 3x3 grid with actor (color 1 at 1,1) and obstacle (color 3 at 0,1)
    grid = [
        [0, 3, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    pset = build_planning_set(snapshot, ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"])

    code = """
def move_up(api):
    return api.declare_environment_action("ACTION1")

def move_down(api):
    return api.declare_environment_action("ACTION2")
"""
    manifest = {
        "functions": [
            {"name": "move_up", "parameters": []},
            {"name": "move_down", "parameters": []},
        ]
    }
    module = executor.load_module(code, manifest)
    syntax_mem = SyntaxErrorMemory(level_id="l0")

    actor = [o for o in pset.objects if o.color == 1][0]

    # 1. Trajectory moving UP hits obstacle at (0,1) -> rejected upfront
    cand_coll = CandidateTrajectory(
        trajectory_id="c_coll",
        steps=[{"step_id": "s1", "dsl_function": "move_up", "arguments": {"obj": actor.id}}],
    )
    is_valid, reason, _ = sym_exec.verify_full_candidate_trajectory(
        candidate=cand_coll,
        planning_set=pset,
        active_module=module,
        syntax_memory=syntax_mem,
    )
    assert is_valid is False
    assert "collision" in reason.lower() or "collides" in reason.lower()

    # 2. Trajectory moving DOWN 5 times violates grid bounds -> rejected upfront
    cand_oob = CandidateTrajectory(
        trajectory_id="c_oob",
        steps=[
            {"step_id": "s1", "dsl_function": "move_down", "arguments": {"obj": actor.id}},
            {"step_id": "s2", "dsl_function": "move_down", "arguments": {"obj": actor.id}},
            {"step_id": "s3", "dsl_function": "move_down", "arguments": {"obj": actor.id}},
            {"step_id": "s4", "dsl_function": "move_down", "arguments": {"obj": actor.id}},
            {"step_id": "s5", "dsl_function": "move_down", "arguments": {"obj": actor.id}},
        ],
    )
    is_valid, reason, _ = sym_exec.verify_full_candidate_trajectory(
        candidate=cand_oob,
        planning_set=pset,
        active_module=module,
        syntax_memory=syntax_mem,
    )
    assert is_valid is False
    assert "boundary" in reason.lower()

    # 3. Trajectory moving DOWN 1 step within bounds and no obstacle -> approved upfront
    cand_ok = CandidateTrajectory(
        trajectory_id="c_ok",
        steps=[{"step_id": "s1", "dsl_function": "move_down", "arguments": {"obj": actor.id}}],
    )
    is_valid, reason, _ = sym_exec.verify_full_candidate_trajectory(
        candidate=cand_ok,
        planning_set=pset,
        active_module=module,
        syntax_memory=syntax_mem,
    )
    assert is_valid is True
    assert "approved" in reason.lower()


