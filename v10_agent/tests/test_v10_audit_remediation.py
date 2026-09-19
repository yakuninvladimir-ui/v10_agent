"""Targeted unit tests verifying the full remediation of V10 architecture defects.

Covers:
1. Brusentsov logic: elimination of vacuous truth (empty expected -> Ternary.IRRELEVANT).
2. Judge cascade: Tier 8 default fallthrough strictly returns Verdict.OMIT.
3. VirtualKinematicSandbox: generalized Feature Delta A* planner operates without axis/piece hardcoding.
4. Session orchestration: max_actions_per_level budget parameter resolution.
5. Invariant extraction: preservation of valid kinematic displacement rules (dx/dy).
6. Config & Packaging: ARC_MAX_EXPLORER_PROBES default 30 and isGpuEnabled=True.
"""

import json
from pathlib import Path
import pytest
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.brusentsov_logic import implies_brusentsov, Ternary, Verdict
from v10_agent.judge import LayeredVerifier, BrusentsovJudgment
from v10_agent.config import V10Config, config_from_env
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set
from v10_agent.virtual_sandbox import VirtualKinematicSandbox
from v10_agent.memory_contours import GameMemory, StructuredInvariant
from v10_agent.universal_invariants import DiscoveredInvariant
from v10_agent.verification import GroundedStep
from v10_agent.solver_agent import SolverAgent


from v10_agent.llm_advisor import MockLLMAdvisor


def test_remediation_brusentsov_empty_expected_is_irrelevant():
    """Empty expected proposition set must return Ternary.IRRELEVANT, not Ternary.TRUE (no vacuous truth)."""
    expected = PropositionSet.from_iterable([])
    observed = PropositionSet.from_iterable([
        AtomicProposition(family="metric_sign", subject_id="obj_0", predicate="row_delta", value=1),
    ])
    verdict = implies_brusentsov(expected, observed)
    assert verdict == Ternary.IRRELEVANT


def test_remediation_judge_tier8_fallback_is_strictly_omit():
    """Default fallthrough in LayeredVerifier must return Verdict.OMIT (Tier 8 ISO-10)."""
    config = V10Config(llm_advisor_backend="fake")
    verifier = LayeredVerifier(config)

    grid = [[0, 2, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1"])

    step = GroundedStep(
        step_id="step_passive",
        dsl_function="passive_action",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([]),
    )

    judgment = verifier.evaluate_transition(
        step=step,
        before_snapshot=snap,
        after_obs={"grid": grid, "state": "IN_PROGRESS"},
        planning_set=pset,
        action_dict={"action_id": "ACTION1"},
    )

    assert judgment.verdict == Verdict.OMIT
    assert "omit x'y'" in judgment.explanation.lower()
    assert judgment.verdict != Verdict.FOLLOW
    assert judgment.verdict != Verdict.NULL


def test_remediation_generalized_astar_feature_delta_planner():
    """FeatureDeltaAStarPlanner finds optimal action sequence for spatial contact without hardcoded roles."""
    grid = [[0] * 10 for _ in range(10)]
    # Subject at (2, 2)
    grid[2][2] = 2
    # Target at (5, 6)
    grid[5][6] = 3

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    gm = GameMemory(game_id="generic_game")
    gm.record_action_effect("action1", "dy=-1, dx=0 (UP)")
    gm.record_action_effect("action2", "dy=1, dx=0 (DOWN)")
    gm.record_action_effect("action3", "dy=0, dx=-1 (LEFT)")
    gm.record_action_effect("action4", "dy=0, dx=1 (RIGHT)")

    sandbox = VirtualKinematicSandbox(pset, gm)
    manifest = {
        "action1": {"name": "action1", "docstring": "shift entity up"},
        "action2": {"name": "action2", "docstring": "shift entity down"},
        "action3": {"name": "action3", "docstring": "shift entity left"},
        "action4": {"name": "action4", "docstring": "shift entity right"},
    }

    sub = next(o for o in pset.objects if o.color == 2)
    tgt = next(o for o in pset.objects if o.color == 3)

    inv = DiscoveredInvariant(
        invariant_type="spatial_contact",
        subject_id=sub.id,
        target_id=tgt.id,
        description=f"Move {sub.id} to contact {tgt.id}",
    )

    steps = sandbox.synthesize_invariant_trajectory(inv, manifest)
    assert steps is not None
    # Distance: dr = 3 (DOWN), dc = 4 (RIGHT) -> 7 steps total
    assert len(steps) == 7
    down_count = sum(1 for s in steps if s["dsl_function"] == "action2")
    right_count = sum(1 for s in steps if s["dsl_function"] == "action4")
    assert down_count == 3
    assert right_count == 4


def test_remediation_kinematics_displacement_preserved_in_distillation():
    """Solver invariant parser preserves valid kinematic rules with dx/dy and displacement."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    advisor.set_response(
        "solver_reflection",
        "<distilled_invariants>\n"
        "- [KINEMATICS]: ACTION1 moves active piece UP with dy=-1, dx=0\n"
        "- [PHYSICS]: ACTION2 shifts object DOWN by dy=1, dx=0\n"
        "- [GOAL]: Reach spatial contact between pieces\n"
        "- Bad macro 1: action1 -> action2\n"
        "- Bad coordinate: dx=5, dy=2 at row=12\n"
        "</distilled_invariants>",
    )
    solver = SolverAgent(config, advisor)
    invariants = solver.distill_level_win_invariants()

    assert len(invariants) == 3
    assert any("[KINEMATICS]" in inv and "dy=-1" in inv for inv in invariants)
    assert any("[PHYSICS]" in inv and "dy=1" in inv for inv in invariants)
    assert any("[GOAL]" in inv for inv in invariants)
    assert not any("action1 -> action2" in inv for inv in invariants)
    assert not any("row=12" in inv for inv in invariants)


def test_remediation_explorer_probes_config_default():
    """config_from_env must default ARC_MAX_EXPLORER_PROBES to 30."""
    cfg = config_from_env()
    assert cfg.max_explorer_probe_actions_per_level == 30


def test_remediation_notebook_gpu_metadata():
    """build_notebook_v10.py must configure isGpuEnabled=True in notebook metadata."""
    build_script = Path("build_notebook_v10.py").read_text(encoding="utf-8")
    assert '"isGpuEnabled": True' in build_script
