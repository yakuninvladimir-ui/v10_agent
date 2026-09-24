"""Property-Based Testing for Carrollian Nullity, Macro-Step Decomposition, and Early Severance.

Verifies:
1. PBT for contradicts() and implies_brusentsov() completeness:
   - Physical Stagnation (expected motion, observed stationary -> NULL)
   - Unintended Mutation (expected invariant/stationary, observed motion -> NULL)
   - Direction Inversion (expected vector opposite to observed -> NULL)
   - Consistent Containment (expected equals observed -> FOLLOW)
   - Subject Isolation (different subjects never contradict each other)
2. Macro-step decomposition in solver_agent.py:
   - State invariants are replicated across all substeps
   - Incremental step kinematics are distributed to intermediate substeps
   - Full vector expectation preserved on the terminal substep
   - No break_on_null bypass
3. Early Severance on Step 1:
   - Trajectory severed immediately on step 1 upon physical mismatch
   - Remaining steps aborted
   - Diagnostic generated: EXPECTED, OBSERVED, DIAGNOSIS
   - Scratchpad context and Section 7 updated for subsequent attempt
   - Environment RESET cleanly triggered
"""

from __future__ import annotations

import json
from hypothesis import given, settings, strategies as st

from v10_agent.brusentsov_logic import Ternary, contradicts, implies_brusentsov
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import EpistemicMemory, GameMemory
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.sandbox import SandboxExecutor
from v10_agent.solver_agent import parse_text_trajectory
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor, TrajectoryPool
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep, VerificationBinder
from v10_agent.arga_lite import extract_arga_snapshot


# =============================================================================
# 1. PBT for Carrollian Nullity Completeness in contradicts()
# =============================================================================

@given(
    exp_dy=st.integers(min_value=-5, max_value=5),
    exp_dx=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=40, deadline=None)
def test_pbt_physical_stagnation(exp_dy, exp_dx):
    """Physical Stagnation: expected motion on an axis, but observed stationary (0) -> Contradiction = True."""
    if exp_dy == 0 and exp_dx == 0:
        exp_dy = 1

    p_exp = AtomicProposition(
        family="metric_sign",
        subject_id="obj_A",
        predicate="step_moved",
        value=(exp_dy, exp_dx),
    )
    p_obs = AtomicProposition(
        family="metric_sign",
        subject_id="obj_A",
        predicate="step_moved",
        value=(0, 0),
    )

    assert contradicts(p_exp, p_obs) is True
    exp_set = PropositionSet.from_iterable([p_exp])
    obs_set = PropositionSet.from_iterable([p_obs])
    assert implies_brusentsov(exp_set, obs_set) == Ternary.FALSE


@given(
    obs_dy=st.integers(min_value=-5, max_value=5),
    obs_dx=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=40, deadline=None)
def test_pbt_unintended_mutation(obs_dy, obs_dx):
    """Unintended Mutation: expected stationary/invariant, but observed motion -> Contradiction = True."""
    if obs_dy == 0 and obs_dx == 0:
        obs_dx = 1

    # Case A: expected unchanged
    p_exp_unchanged = AtomicProposition(
        family="metric_sign",
        subject_id="obj_B",
        predicate="unchanged",
        value=(0, 0),
    )
    p_obs = AtomicProposition(
        family="metric_sign",
        subject_id="obj_B",
        predicate="step_moved",
        value=(obs_dy, obs_dx),
    )
    assert contradicts(p_exp_unchanged, p_obs) is True
    assert implies_brusentsov(
        PropositionSet.from_iterable([p_exp_unchanged]),
        PropositionSet.from_iterable([p_obs]),
    ) == Ternary.FALSE

    # Case B: expected zero motion vector
    p_exp_zero = AtomicProposition(
        family="metric_sign",
        subject_id="obj_B",
        predicate="step_moved",
        value=(0, 0),
    )
    assert contradicts(p_exp_zero, p_obs) is True
    assert implies_brusentsov(
        PropositionSet.from_iterable([p_exp_zero]),
        PropositionSet.from_iterable([p_obs]),
    ) == Ternary.FALSE


@given(
    s_dy=st.sampled_from([-2, -1, 1, 2]),
    s_dx=st.sampled_from([-2, -1, 1, 2]),
)
@settings(max_examples=25, deadline=None)
def test_pbt_direction_inversion(s_dy, s_dx):
    """Direction Inversion: movement vector opposite to expected -> Contradiction = True."""
    p_exp = AtomicProposition(
        family="metric_sign",
        subject_id="obj_C",
        predicate="step_moved",
        value=(s_dy, s_dx),
    )
    p_obs = AtomicProposition(
        family="metric_sign",
        subject_id="obj_C",
        predicate="step_moved",
        value=(-s_dy, -s_dx),
    )

    assert contradicts(p_exp, p_obs) is True
    assert implies_brusentsov(
        PropositionSet.from_iterable([p_exp]),
        PropositionSet.from_iterable([p_obs]),
    ) == Ternary.FALSE


@given(
    s_dy=st.integers(min_value=-3, max_value=3),
    s_dx=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_pbt_consistent_containment(s_dy, s_dx):
    """Consistent Containment: observed motion matches expected -> Contradiction = False, Implication = TRUE."""
    p_exp = AtomicProposition(
        family="metric_sign",
        subject_id="obj_D",
        predicate="step_moved",
        value=(s_dy, s_dx),
    )
    p_obs = AtomicProposition(
        family="metric_sign",
        subject_id="obj_D",
        predicate="step_moved",
        value=(s_dy, s_dx),
    )

    assert contradicts(p_exp, p_obs) is False
    assert implies_brusentsov(
        PropositionSet.from_iterable([p_exp]),
        PropositionSet.from_iterable([p_obs]),
    ) == Ternary.TRUE


def test_subject_isolation_no_false_contradiction():
    """Propositions for distinct entities must not contradict each other."""
    p_exp_A = AtomicProposition(family="metric_sign", subject_id="obj_A", predicate="step_moved", value=(1, 0))
    p_obs_B = AtomicProposition(family="metric_sign", subject_id="obj_B", predicate="step_moved", value=(0, 0))
    assert contradicts(p_exp_A, p_obs_B) is False


# =============================================================================
# 2. Macro-step Decomposition in solver_agent.py
# =============================================================================

def test_macro_count_decomposition_preserves_invariants_and_distributes_steps():
    """Verify count=N decomposition distributes invariants and incremental kinematics without break_on_null."""
    proposal_text = """\
<trajectory_1>
action3(count=3) # EXPECT: moved(H, 0, 3), unchanged(A)
</trajectory_1>
"""
    manifest = {
        "functions": [
            {"name": "action3", "parameters": [{"name": "count", "type": "int"}]}
        ]
    }
    grid = [[0] * 12 for _ in range(12)]
    grid[2][2] = 2
    grid[6][6] = 3
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION3"])
    h_id = next(o.id for o in snap.objects if o.color == 2)
    a_id = next(o.id for o in snap.objects if o.color == 3)
    pset.object_real_to_alias.clear()
    pset.object_real_to_alias.update({h_id: "H", a_id: "A"})
    pset.object_alias_to_real.clear()
    pset.object_alias_to_real.update({"H": h_id, "A": a_id})

    res = parse_text_trajectory(proposal_text, manifest_or_funcs=manifest, planning_set=pset)
    assert res is not None
    assert len(res["candidates"]) == 1
    steps = res["candidates"][0]["steps"]

    # 1. 3 substeps produced
    assert len(steps) == 3

    # 2. None of the steps have break_on_null bypass
    for s in steps:
        assert "break_on_null" not in s or s["break_on_null"] is False

    # 3. Check predicates on intermediate substeps 0 and 1
    for s_idx in (0, 1):
        s = steps[s_idx]
        exp_preds = {(p["predicate"], str(p.get("value"))) for p in s["expected_propositions"] if p.get("subject_id") in ("A", a_id)}
        assert ("unchanged", "(0, 0)") in exp_preds
        assert ("preserved", "None") in exp_preds or ("preserved", "True") in exp_preds or ("dy", "0") in exp_preds

        # Moving object H has incremental step expectations
        h_preds = {(p["predicate"], str(p.get("value"))) for p in s["expected_propositions"] if p.get("subject_id") in ("H", h_id)}
        assert ("step_moved", "(0, 1)") in h_preds
        assert ("dx", "1") in h_preds

    # 4. Final substep 2 preserves full cumulative vector (dx=3, moved=(0, 3))
    s2 = steps[2]
    h_preds_final = {(p["predicate"], str(p.get("value"))) for p in s2["expected_propositions"] if p.get("subject_id") in ("H", h_id)}
    assert ("step_moved", "(0, 1)") in h_preds_final
    assert ("moved", "(0, 3)") in h_preds_final
    assert ("dx", "3") in h_preds_final


# =============================================================================
# 3. Early Severance on Step 1, Diagnostic Generation & Clean Reset
# =============================================================================

def test_early_severance_on_step1_and_diagnostic_generation():
    """Verify early severance at s1, strict diagnostic generation, and scratchpad/reset trigger."""
    config = V10Config()
    verifier = LayeredVerifier(config)
    executor = SymbolicTrajectoryExecutor(
        config=config,
        sandbox_executor=SandboxExecutor(),
        binder=VerificationBinder(),
        verifier=verifier,
    )

    # Initial grid with Shape H (color 2 at (2,2)) and Axis A (color 3 at (6,6))
    grid_0 = [[0] * 12 for _ in range(12)]
    grid_0[2][2] = 2  # Shape H
    grid_0[6][6] = 3  # Axis A
    snap_0 = extract_arga_snapshot(grid_0)

    # Identify object IDs
    obj_h_id = next(o.id for o in snap_0.objects if o.color == 2)
    obj_a_id = next(o.id for o in snap_0.objects if o.color == 3)

    pset = build_planning_set(snap_0, available_actions=["ACTION3"])
    pset.object_real_to_alias.clear()
    pset.object_real_to_alias.update({obj_h_id: "H", obj_a_id: "A"})
    pset.object_alias_to_real.clear()
    pset.object_alias_to_real.update({"H": obj_h_id, "A": obj_a_id})

    ep_mem = EpistemicMemory(level_id="level_0")
    game_mem = GameMemory(game_id="game_test")

    # Step 1 expected: moved(H) and unchanged(A)
    step1 = GroundedStep(
        step_id="s1",
        dsl_function="action3",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="metric_sign", subject_id=obj_h_id, predicate="step_moved", value=(0, 1)),
            AtomicProposition(family="metric_sign", subject_id=obj_a_id, predicate="unchanged", value=(0, 0)),
        ]),
    )

    # Reality after step 1: Shape H stayed stationary at (2,2), but Axis A moved to (6,7)!
    grid_after = [[0] * 12 for _ in range(12)]
    grid_after[2][2] = 2  # H stationary (stagnation)
    grid_after[6][7] = 3  # A moved (unintended mutation)

    # Create candidate pool with 3 steps
    cand_pkg = {
        "candidates": [
            {
                "trajectory_id": "traj_01",
                "steps": [
                    {"step_id": "s1", "dsl_function": "action3", "arguments": {}},
                    {"step_id": "s2", "dsl_function": "action3", "arguments": {}},
                    {"step_id": "s3", "dsl_function": "action3", "arguments": {}},
                ],
            }
        ]
    }
    pool = TrajectoryPool.from_package(cand_pkg)

    # Evaluate transition
    eval_res = executor.evaluate_transition(
        pending_step=step1,
        before_snapshot=snap_0,
        after_obs={"grid": grid_after, "levels_completed": 0},
        planning_set=pset,
        active_pool=pool,
        epistemic_memory=ep_mem,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION3"},
    )

    # 1. Verification results: NULL verdict, candidate severed, reset requested
    assert eval_res.verdict == Ternary.FALSE
    assert eval_res.candidate_severed is True
    assert eval_res.candidate_advanced is False
    assert eval_res.reset_needed is True
    assert eval_res.replan_needed is True

    # 2. Candidate pool is finished/severed immediately at s1
    active_cand = pool.candidates[0]
    assert active_cand.is_severed is True
    assert active_cand.active is False
    # Cursor should not have advanced to s2 or s3
    assert active_cand.cursor == 0

    # 3. Diagnostic record in EpistemicMemory
    assert len(ep_mem.current_level_attempts) == 1
    att = ep_mem.current_level_attempts[0]
    assert att["status"] == "severed_step_failed"
    assert "diagnostic" in att
    diag = att["diagnostic"]

    # Verify diagnostic structure
    assert "[FAILED ATTEMPT 1 DIAGNOSTIC]" in diag
    assert "Failed at step s1 (action3): Mismatch detected." in diag
    assert "EXPECTED:" in diag
    assert "moved(H)" in diag
    assert "unchanged(A)" in diag
    assert "OBSERVED:" in diag
    assert "stationary(H)" in diag
    assert "moved(A)" in diag
    assert "DIAGNOSIS:" in diag
    assert "action3 mutates alias A, not alias H" in diag
    assert "Do NOT repeat action3 for moving H" in diag

    # 4. Formatted scratchpad context
    scratchpad_text = ep_mem.format_scratchpad_context()
    assert "[FAILED ATTEMPT 1 DIAGNOSTIC]" in scratchpad_text
    assert "Failed at step s1 (action3): Mismatch detected." in scratchpad_text

    # 5. Section 7 in Solver Prompt contains the diagnostic
    sys_prompt, user_prompt = build_solver_prompts(
        manifest={"functions": [{"name": "action3", "parameters": []}]},
        planning_set=pset,
        epistemic_memory=ep_mem,
        game_memory=game_mem,
    )
    assert "[FAILED ATTEMPT 1 DIAGNOSTIC]" in user_prompt
    assert "action3 mutates alias A, not alias H" in user_prompt
