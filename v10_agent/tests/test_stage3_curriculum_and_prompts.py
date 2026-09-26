"""Stage 3: Comprehensive tests for Prompts, ISO-2 Quarantine, and Curriculum Learning.

Verifies:
1. ISO-2 Quarantine: Coder prompt strictly strips goal invariants, win rules, and curriculum history goal markers.
2. Solver prompt EXECUTION LOOP FACTS: correctly states empty EXPECT yields IRRELEVANT/OMIT (not FOLLOW).
3. Solver prompt invariant ranking: stratified into Tier 1 (Core Game Laws), Tier 2 (Domain Patterns), Tier 3 (Level Specific), and Falsified.
4. Solver prompt <curriculum_delta>: correctly computes and formats newly confirmed, newly falsified, and core laws across level transitions.
5. Role-specific temperatures: solver=0.7, coder=0.5, explorer=0.9, generic=1.0.
6. Explorer level synthesis prompt: supports CONDITIONAL_TRIGGER without forcing false effects on zero-delta probes.
"""

from __future__ import annotations

import copy
import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import MockLLMAdvisor
from v10_agent.memory_contours import (
    CoreInvariantRegistry,
    EmpiricalInvariant,
    FORBIDDEN_GOAL_PATTERNS_IN_CODER,
    GameMemory,
)
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.prompt_builders.explorer_prompt import (
    EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT,
    build_explorer_synthesis_prompt,
)
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts


def _make_dummy_pset():
    snap = extract_arga_snapshot([[0, 1, 0], [0, 0, 0]])
    return build_planning_set(snap, available_actions=["ACTION1", "ACTION2"])


def test_iso2_quarantine_goal_invariants_stripped_from_coder():
    """Coder prompt strictly strips any goal invariants, win rules, or target markings (ISO-2)."""
    env_spec = {
        "action_affordances": [
            {"action_id": "ACTION1", "effect_class": "KINEMATIC", "parameters": {"dy": -1, "dx": 0}}
        ],
        "invariants": [
            "objects move rigidly without deformation",
            "Goal invariant: align pieces with mirror axis",
            "winning_invariant for level 2",
            "Contact with TARGET=3 triggers win",
            "POSITIVE_CANON: touch color 5",
            "DEFEAT on hazard color 2",
        ],
        "curriculum_history": [
            "Level 0 solved: Goal invariant reached via ACTION1",
            "Level 1: target_score 100",
            "Standard kinematic displacement observed",
        ],
    }

    sys_prompt, user_prompt = build_coder_prompts(env_spec=env_spec, has_image=False)
    combined = sys_prompt + "\n" + user_prompt

    # Allowed general physics invariant should be present
    assert "objects move rigidly without deformation" in combined

    # All forbidden goal/win/target/defeat markers must be quarantined
    for pattern in FORBIDDEN_GOAL_PATTERNS_IN_CODER:
        assert not pattern.search(combined), f"Violation: Coder prompt leaked forbidden pattern {pattern.pattern}"


def test_solver_prompt_execution_loop_facts_accuracy():
    """Solver prompt EXECUTION LOOP FACTS accurately specifies that empty EXPECT yields IRRELEVANT/OMIT, not FOLLOW."""
    pset = _make_dummy_pset()
    sys_prompt, user_prompt = build_solver_prompts(
        planning_set=pset,
        manifest={"functions": [{"name": "action1", "parameters": []}]},
        has_image=False,
    )

    # Empty EXPECT must lead to IRRELEVANT/OMIT, never FOLLOW
    assert "Empty EXPECT (len==0) -> verdict IRRELEVANT/OMIT" in sys_prompt
    assert "Non-empty EXPECT confirmed -> verdict FOLLOW" in sys_prompt
    assert "Physically contradicted EXPECT -> candidate terminated (verdict NULL)" in sys_prompt
    assert "weakly certified (verdict FOLLOW or OMIT)" not in sys_prompt


def test_solver_prompt_ranked_empirical_invariants():
    """Solver prompt presents invariants stratified and ranked by scope and net support."""
    pset = _make_dummy_pset()
    gm = GameMemory(game_id="generic_game")
    gm.record_action_effect("action1", "dy=-1, dx=0 (UP)")

    reg = gm.invariant_registry
    core_law = EmpiricalInvariant(
        invariant_id="core_area",
        invariant_type="area_conservation",
        abstract_description="Objects conserve their pixel area",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
        confidence=0.5,
        times_confirmed=3,
        confirmed_on_levels=[0, 1, 2],
    )
    domain_inv = EmpiricalInvariant(
        invariant_id="domain_sym",
        invariant_type="axial_symmetry_vertical",
        abstract_description="Vertical symmetry across center line",
        subject_pattern="pair==1:2",
        expected_value="symmetric",
        scope="DOMAIN_PATTERN",
        confidence=0.4,
        times_confirmed=2,
        confirmed_on_levels=[1],
    )
    level_inv = EmpiricalInvariant(
        invariant_id="level_rule",
        invariant_type="selection_focus",
        abstract_description="Controllable focus cycles between pieces",
        subject_pattern="selection",
        expected_value=1,
        scope="LEVEL_SPECIFIC",
        confidence=0.3,
        times_confirmed=1,
        confirmed_on_levels=[2],
    )
    falsified_inv = EmpiricalInvariant(
        invariant_id="falsified_rule",
        invariant_type="kinematics",
        abstract_description="Entities move through obstacles",
        subject_pattern="motion",
        expected_value=0,
        scope="LEVEL_SPECIFIC",
        confidence=0.0,
        times_confirmed=1,
        times_falsified=1,
        falsified_on_levels=[1],
    )

    reg.register_candidate(core_law)
    reg.register_candidate(domain_inv)
    reg.register_candidate(level_inv)
    reg.register_candidate(falsified_inv)

    sys_prompt, user_prompt = build_solver_prompts(
        planning_set=pset,
        game_memory=gm,
        manifest={"functions": [{"name": "action1", "parameters": []}]},
        has_image=False,
    )

    assert "[CORE_LAW] Objects conserve their pixel area" in user_prompt
    assert "[axial_symmetry_vertical] Vertical symmetry across center line" in user_prompt
    assert "[selection_focus] Controllable focus cycles between pieces" in user_prompt
    assert "[kinematics] Entities move through obstacles (falsified 1x" in user_prompt


def test_solver_prompt_curriculum_delta_section():
    """Solver prompt includes <curriculum_delta> when transitioning between levels."""
    pset = _make_dummy_pset()
    gm = GameMemory(game_id="generic_game")
    gm.completed_levels = 1

    # Simulate previous level registry snapshot
    prev_reg = CoreInvariantRegistry()
    gm.previous_level_registry = prev_reg

    # Current registry after actions on level 1
    reg = gm.invariant_registry
    new_confirmed = EmpiricalInvariant(
        invariant_id="new_law",
        invariant_type="area_conservation",
        abstract_description="Area remains constant under gravity",
        subject_pattern="all_objects",
        expected_value="conserved",
        scope="CORE_GAME_LAW",
        confidence=0.3,
        times_confirmed=3,
        confirmed_on_levels=[1],
    )
    new_falsified = EmpiricalInvariant(
        invariant_id="broken_law",
        invariant_type="symmetry",
        abstract_description="Objects retain horizontal reflection",
        subject_pattern="pair==3:4",
        expected_value=1,
        scope="DOMAIN_PATTERN",
        confidence=0.0,
        times_falsified=1,
        falsified_on_levels=[1],
    )
    reg.register_candidate(new_confirmed)
    reg.register_candidate(new_falsified)

    sys_prompt, user_prompt = build_solver_prompts(
        planning_set=pset,
        game_memory=gm,
        level_index=1,
        manifest={"functions": [{"name": "action1", "parameters": []}]},
        has_image=False,
    )

    assert "<curriculum_delta>" in user_prompt
    assert "CURRICULUM DELTA (Level 0 -> 1):" in user_prompt
    assert "Area remains constant under gravity (now 0.30)" in user_prompt
    assert "Objects retain horizontal reflection (falsified on levels [1])" in user_prompt
    assert "</curriculum_delta>" in user_prompt


def test_role_temperatures_and_config():
    """Different agent roles receive their dedicated temperatures."""
    config = V10Config(
        solver_temperature=0.7,
        coder_temperature=0.5,
        explorer_temperature=0.9,
        temperature=1.0,
    )
    advisor = MockLLMAdvisor()

    advisor.generate("sys", "user", config=config, agent_role="solver")
    advisor.generate("sys", "user", config=config, agent_role="coder")
    advisor.generate("sys", "user", config=config, agent_role="explorer")
    advisor.generate("sys", "user", config=config, agent_role="generic")

    assert len(advisor.call_history) == 4
    assert advisor.call_history[0]["temperature"] == 0.7
    assert advisor.call_history[1]["temperature"] == 0.5
    assert advisor.call_history[2]["temperature"] == 0.9
    assert advisor.call_history[3]["temperature"] == 1.0


def test_explorer_prompt_conditional_trigger_support():
    """Explorer prompt encourages CONDITIONAL_TRIGGER and removes punishment for uncertainty."""
    assert "CONDITIONAL_TRIGGER" in EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT
    assert "It is BETTER to mark an action as 'effect_class: CONDITIONAL_TRIGGER'" in EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT
    assert "Do NOT overgeneralize or invent unobserved effects" in EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT
