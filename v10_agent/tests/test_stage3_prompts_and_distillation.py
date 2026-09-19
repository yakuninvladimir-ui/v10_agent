"""Comprehensive unit tests validating Stage 3: Prompts & Domain-General Distillation."""

from __future__ import annotations

import json
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import GameMemory
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.prompt_builders.explorer_prompt import EXPLORER_SYSTEM_PROMPT, build_explorer_prompts
from v10_agent.prompt_builders.solver_prompt import (
    SOLVER_SYSTEM_PROMPT,
    _build_phase_instruction,
    build_solver_prompts,
)
from v10_agent.solver_agent import SolverAgent


def test_progressive_phase_instructions():
    """3.1: Test _build_phase_instruction returns Exploration, Confirmation, and Exploitation phases."""
    phase_0 = _build_phase_instruction(0)
    assert "EXPLORATION PHASE" in phase_0
    assert "complete winning trajectories" in phase_0

    phase_1 = _build_phase_instruction(1)
    assert "EXPLORATION PHASE" in phase_1

    phase_2 = _build_phase_instruction(2)
    assert "CONFIRMATION PHASE" in phase_2
    assert "Apply confirmed invariants" in phase_2

    phase_3 = _build_phase_instruction(3)
    assert "CONFIRMATION PHASE" in phase_3

    phase_4 = _build_phase_instruction(4)
    assert "EXPLOITATION PHASE" in phase_4
    assert "Execute optimal trajectories" in phase_4

    # Verify build_solver_prompts incorporates the phase instruction
    grid = [[0, 1], [0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])
    sys_p, _ = build_solver_prompts({"functions": []}, pset, level_index=4)
    assert "EXPLOITATION PHASE" in sys_p


def test_game_model_block_in_solver_prompt():
    """3.2: Verify confirmed_game_model is injected into user_payload."""
    grid = [[1, 0], [0, 1]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])
    gm = GameMemory(game_id="test_game")
    gm.record_action_effect("ACTION1", "moves entity UP by dy=-1")

    _, user_p = build_solver_prompts({"functions": []}, pset, game_memory=gm)
    assert "confirmed_game_model" in user_p
    assert "GAME MODEL" in user_p


def test_domain_agnostic_distillation_prompt():
    """3.3: Verify distillation prompt uses domain-general categories."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    solver = SolverAgent(config, advisor)

    advisor.set_response(
        "solver_reflection",
        "<distilled_invariants>\n"
        "- [PHYSICS]: Objects move rigidly without deformation on ACTION1\n"
        "- [STRUCTURE]: Axis line bisects active pieces symmetrically\n"
        "- [GOAL]: Target sockets must be completely covered\n"
        "- [CONTROL]: ACTION5 transfers active indicator dot\n"
        "</distilled_invariants>",
    )

    invariants = solver.distill_level_win_invariants()
    assert len(invariants) == 4
    assert any("[PHYSICS]" in inv for inv in invariants)
    assert any("[STRUCTURE]" in inv for inv in invariants)
    assert any("[GOAL]" in inv for inv in invariants)
    assert any("[CONTROL]" in inv for inv in invariants)


def test_explorer_prompt_structural_observations():
    """3.4: Verify Explorer system prompt requests symmetry, periodicity, topology, and colors."""
    assert "SYMMETRY:" in EXPLORER_SYSTEM_PROMPT
    assert "PERIODICITY:" in EXPLORER_SYSTEM_PROMPT
    assert "TOPOLOGY:" in EXPLORER_SYSTEM_PROMPT
    assert "COLOR PATTERNS:" in EXPLORER_SYSTEM_PROMPT


def test_coder_prompt_action5_example():
    """3.5: Verify Coder prompt includes action5 example in code and manifest."""
    _, user_p = build_coder_prompts({}, syntax_errors=[])
    assert "def action5(api):" in user_p
    assert "Declare entity selection/cycling action ACTION5" in user_p
    assert '"name": "action5"' in user_p


def test_solver_prompt_ternary_semantics():
    """3.6: Verify Solver system prompt explains TRUE/FOLLOW, IRRELEVANT/OMIT, FALSE/NULL."""
    assert "TERNARY EVALUATION SEMANTICS:" in SOLVER_SYSTEM_PROMPT
    assert "TRUE (FOLLOW)" in SOLVER_SYSTEM_PROMPT
    assert "IRRELEVANT (OMIT)" in SOLVER_SYSTEM_PROMPT
    assert "FALSE (NULL)" in SOLVER_SYSTEM_PROMPT


def test_solver_prompt_freedom_of_motion_disclaimer():
    """3.7: Verify freedom_of_motion contains boundary_distance_only and WARNING disclaimer."""
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, available_actions=["ACTION1"])
    _, user_p = build_solver_prompts({"functions": []}, pset)

    assert "boundary_distance_only" in user_p
    assert "These are distances to GRID EDGES only" in user_p


def test_distillation_button_macro_filtering():
    """3.8: Verify robust filtering of action macros in distilled invariants."""
    config = V10Config(llm_advisor_backend="fake")
    advisor = MockLLMAdvisor()
    solver = SolverAgent(config, advisor)

    advisor.set_response(
        "solver_reflection",
        "<distilled_invariants>\n"
        "- [GOAL]: Cover targets to win\n"
        "- Bad macro 1: action1 -> action5 -> action2\n"
        "- Bad macro 2: action1() -> action5()\n"
        "- Bad macro 3: action1() , action2()\n"
        "- Bad macro 4: action1 > action2\n"
        "- Bad coordinates: dx=3, dy=0\n"
        "- [PHYSICS]: Entities halt at boundary walls\n"
        "</distilled_invariants>",
    )

    invariants = solver.distill_level_win_invariants()
    assert len(invariants) == 2
    assert any("[GOAL]" in inv for inv in invariants)
    assert any("[PHYSICS]" in inv for inv in invariants)
    assert not any("action1" in inv for inv in invariants)
    assert not any("dx=" in inv for inv in invariants)


def test_structured_invariant_analysis_format():
    """3.9: Verify invariant_analysis format in Solver prompt contains the 4 structured questions."""
    assert "1. What is the likely goal of this level?" in SOLVER_SYSTEM_PROMPT
    assert "2. Which confirmed invariants apply here?" in SOLVER_SYSTEM_PROMPT
    assert "3. What is NEW or DIFFERENT about this level vs previous ones?" in SOLVER_SYSTEM_PROMPT
    assert "4. Strategy for this level:" in SOLVER_SYSTEM_PROMPT
