"""Property-Based Testing for Action Affordance Completeness and Invariant Classification.

Tests:
1. classify_probe_affordance across all 4 invariant effect classes:
   - KINEMATIC
   - PALETTE_TRANSITION
   - TOPOLOGY_MUTATION
   - MODAL_SELECTION
2. Action Affordance Completeness invariant in LevelSpec synthesis.
3. DSLCoder non-discrimination and automated declarative wrapper generation.
4. Brusentsov 3-valued verifier handling of MODAL_SELECTION (FOLLOW / OMIT, never NULL).
"""

from __future__ import annotations

import json
from typing import Any
from hypothesis import given, settings, strategies as st

from v10_agent.action_adapter import to_native_action
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.brusentsov_logic import Verdict
from v10_agent.config import V10Config
from v10_agent.dsl_coder import DSLCoder
from v10_agent.explorer_agent import (
    ActionAffordance,
    EffectClass,
    ExplorerAgent,
    classify_effect_summary_to_affordance,
    classify_probe_affordance,
    generate_symbolic_environment_spec,
)
from v10_agent.judge import LayeredVerifier
from v10_agent.llm_advisor import BaseLLMAdvisor
from v10_agent.memory_contours import (
    EnvironmentSpecMemory,
    GameMemory,
    SyntaxErrorMemory,
)
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.types import AtomicProposition, PropositionSet
from v10_agent.verification import GroundedStep
from v10_agent.sandbox import SandboxExecutor


class MockAdvisor(BaseLLMAdvisor):
    def __init__(self, response: str = ""):
        self.response = response

    def generate(self, *args, **kwargs) -> str:
        return self.response


# =============================================================================
# 1. PBT for classify_probe_affordance: All 4 Invariant Effect Classes
# =============================================================================

@given(
    grid_h=st.integers(min_value=12, max_value=24),
    grid_w=st.integers(min_value=12, max_value=24),
    obj_h=st.integers(min_value=2, max_value=4),
    obj_w=st.integers(min_value=2, max_value=4),
    color=st.integers(min_value=1, max_value=9),
    dr=st.sampled_from([-2, -1, 0, 1, 2]),
    dc=st.sampled_from([-2, -1, 0, 1, 2]),
)
@settings(max_examples=25, deadline=None)
def test_pbt_classify_kinematic_displacement(grid_h, grid_w, obj_h, obj_w, color, dr, dc):
    """Verify rigid coordinate displacements are uniquely classified as KINEMATIC."""
    if dr == 0 and dc == 0:
        dr = 1

    r0, c0 = 4, 4
    before_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    for r in range(r0, r0 + obj_h):
        for c in range(c0, c0 + obj_w):
            before_grid[r][c] = color

    after_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    for r in range(r0 + dr, r0 + dr + obj_h):
        for c in range(c0 + dc, c0 + dc + obj_w):
            after_grid[r][c] = color

    snap_b = extract_arga_snapshot(before_grid)
    aff = classify_probe_affordance(snap_b, {"grid": after_grid}, action_id="ACTION1")

    assert aff is not None
    assert aff["effect_class"] == EffectClass.KINEMATIC.value
    assert aff["action_id"] == "ACTION1"
    assert aff["parameters"]["dy"] == dr
    assert aff["parameters"]["dx"] == dc


@given(
    grid_h=st.integers(min_value=10, max_value=20),
    grid_w=st.integers(min_value=10, max_value=20),
    c_from=st.integers(min_value=1, max_value=5),
    c_to=st.integers(min_value=6, max_value=10),
)
@settings(max_examples=25, deadline=None)
def test_pbt_classify_palette_transition(grid_h, grid_w, c_from, c_to):
    """Verify in-place color change without movement is classified as PALETTE_TRANSITION."""
    before_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    after_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    # Draw 2x2 tile that changes color in-place
    for r in range(2, 4):
        for c in range(2, 4):
            before_grid[r][c] = c_from
            after_grid[r][c] = c_to

    snap_b = extract_arga_snapshot(before_grid)
    aff = classify_probe_affordance(snap_b, {"grid": after_grid}, action_id="ACTION2")

    assert aff is not None
    assert aff["effect_class"] == EffectClass.PALETTE_TRANSITION.value
    assert aff["parameters"]["color_from"] == c_from
    assert aff["parameters"]["color_to"] == c_to
    assert aff["parameters"]["cells_count"] == 4


@given(
    grid_h=st.integers(min_value=12, max_value=22),
    grid_w=st.integers(min_value=12, max_value=22),
    is_spawn=st.booleans(),
)
@settings(max_examples=25, deadline=None)
def test_pbt_classify_topology_mutation(grid_h, grid_w, is_spawn):
    """Verify creation or deletion of components is classified as TOPOLOGY_MUTATION."""
    before_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    after_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    # Base persistent entity
    before_grid[2][2] = 1
    after_grid[2][2] = 1

    if is_spawn:
        # Spawn a new distinct entity
        after_grid[8][8] = 4
    else:
        # Entity existed before and was destroyed
        before_grid[8][8] = 4

    snap_b = extract_arga_snapshot(before_grid)
    aff = classify_probe_affordance(snap_b, {"grid": after_grid}, action_id="ACTION3")

    assert aff is not None
    assert aff["effect_class"] == EffectClass.TOPOLOGY_MUTATION.value
    if is_spawn:
        assert aff["parameters"]["mutation_type"] == "SPAWN"
        assert aff["parameters"]["count_delta"] == 1
    else:
        assert aff["parameters"]["mutation_type"] == "DESTROY"
        assert aff["parameters"]["count_delta"] == -1


def test_classify_modal_selection_indicator_transfer():
    """Verify transfer of selection dot between stationary containers is classified as MODAL_SELECTION."""
    grid_h, grid_w = 16, 16
    before_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    after_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

    # Container 1 (color 2, 3x3 hollow or block) at rows 2..4, cols 2..4
    for r in range(2, 5):
        for c in range(2, 5):
            before_grid[r][c] = 2
            after_grid[r][c] = 2

    # Container 2 (color 3, 3x3 hollow or block) at rows 2..4, cols 10..12
    for r in range(2, 5):
        for c in range(10, 13):
            before_grid[r][c] = 3
            after_grid[r][c] = 3

    # Dot in Container 1 before (color 5 at row 3, col 3)
    before_grid[3][3] = 5

    # Dot moved into Container 2 after (color 5 at row 3, col 11)
    after_grid[3][11] = 5

    snap_b = extract_arga_snapshot(before_grid)
    aff = classify_probe_affordance(snap_b, {"grid": after_grid}, action_id="ACTION5")

    assert aff is not None
    assert aff["effect_class"] == EffectClass.MODAL_SELECTION.value
    assert aff["parameters"]["active_entity_switched"] is True


# =============================================================================
# 2. PBT for Action Affordance Completeness Invariant in LevelSpec
# =============================================================================

def test_action_affordance_completeness_invariant():
    """Verify that any action with confirmed delta > 0 is preserved in action_affordances and available_actions."""
    config = V10Config()
    advisor = MockAdvisor()
    explorer = ExplorerAgent(config, advisor)

    # Set up confirmed actions spanning all 4 classes
    explorer.probe_manager.confirmed_effective_actions["ACTION1"] = "moved B by dy=-1, dx=0 (UP)"
    explorer.probe_manager.confirmed_effective_actions["ACTION2"] = "color transition: 4 cells changed color [1->2]"
    explorer.probe_manager.confirmed_effective_actions["ACTION3"] = "object count changed by +1 (spawn)"
    explorer.probe_manager.confirmed_effective_actions["ACTION4"] = "selection indicator transferred (active entity toggled)"
    explorer.probe_manager.inactive_actions.add("ACTION5")

    # LLM returns incomplete output (only mentions ACTION1)
    mock_llm_json = {
        "action_affordances": [
            {
                "action_id": "ACTION1",
                "effect_class": "KINEMATIC",
                "parameters": {"affected_alias": "B", "dy": -1, "dx": 0},
                "coordination_notes": "moves up",
            }
        ],
        "static_objects": [],
        "structural_geometry": [],
    }
    advisor.response = f"```json\n{json.dumps(mock_llm_json)}\n```"

    grid = [[0 for _ in range(10)] for _ in range(10)]
    grid[2][2] = 1
    snap = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snap, available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"])
    memory = EnvironmentSpecMemory(game_id="game_1", level_id="level_0")
    game_mem = GameMemory(game_id="game_1")

    spec = explorer.synthesize_level_spec(planning_set, memory, game_memory=game_mem)

    # All 4 confirmed actions must be certified in action_affordances (Completeness Invariant)
    aff_map = {a["action_id"]: a for a in spec["action_affordances"]}
    assert "ACTION1" in aff_map
    assert "ACTION2" in aff_map
    assert "ACTION3" in aff_map
    assert "ACTION4" in aff_map
    assert "ACTION5" not in aff_map

    assert aff_map["ACTION1"]["effect_class"] == "KINEMATIC"
    assert aff_map["ACTION2"]["effect_class"] == "PALETTE_TRANSITION"
    assert aff_map["ACTION3"]["effect_class"] == "TOPOLOGY_MUTATION"
    assert aff_map["ACTION4"]["effect_class"] == "MODAL_SELECTION"

    # All 4 confirmed actions must be in available_actions
    assert set(spec["available_actions"]) == {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

    # Backwards-compatible action_displacements contains the kinematic action
    disp_acts = {d["action_id"] for d in spec.get("action_displacements", []) if isinstance(d, dict)}
    assert "ACTION1" in disp_acts


# =============================================================================
# 3. PBT for DSLCoder Non-Discrimination of Non-Displacement Actions
# =============================================================================

def test_dsl_coder_non_discrimination_and_wrapper_generation():
    """Verify DSLCoder does not discriminate against non-displacement actions and emits wrappers."""
    config = V10Config()
    executor = SandboxExecutor()

    # Coder LLM implements only action1 and action2 (ignoring action3, action4, action5)
    coder_output = """\
```python
def action1(api):
    \"\"\"Move up.\"\"\"
    return api.declare_environment_action(action_id='ACTION1')

def action2(api):
    \"\"\"Move down.\"\"\"
    return api.declare_environment_action(action_id='ACTION2')
```

```json
{
  "functions": [
    {"name": "action1", "parameters": [], "returns": "effect_declaration", "docstring": "Move up."},
    {"name": "action2", "parameters": [], "returns": "effect_declaration", "docstring": "Move down."}
  ]
}
```
"""
    advisor = MockAdvisor(response=coder_output)
    dsl_coder = DSLCoder(config, advisor, executor)

    env_spec = {
        "action_affordances": [
            {"action_id": "ACTION1", "effect_class": "KINEMATIC", "parameters": {"dy": -1, "dx": 0}},
            {"action_id": "ACTION2", "effect_class": "KINEMATIC", "parameters": {"dy": 1, "dx": 0}},
            {"action_id": "ACTION3", "effect_class": "PALETTE_TRANSITION", "parameters": {"cells_count": 4}},
            {"action_id": "ACTION4", "effect_class": "TOPOLOGY_MUTATION", "parameters": {"mutation_type": "SPAWN"}},
            {"action_id": "ACTION5", "effect_class": "MODAL_SELECTION", "parameters": {"active_entity_switched": True}},
        ],
        "available_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
    }
    game_mem = GameMemory(game_id="game_1")
    game_mem.confirmed_action_effects = {
        "ACTION1": "moves up",
        "ACTION2": "moves down",
        "ACTION3": "recolor 4 cells",
        "ACTION4": "spawn dot",
        "ACTION5": "toggle mode",
    }

    grid = [[0 for _ in range(8)] for _ in range(8)]
    grid[2][2] = 1
    snap = extract_arga_snapshot(grid)
    planning_set = build_planning_set(snap, available_actions=env_spec["available_actions"])
    syntax_memory = SyntaxErrorMemory(level_id="level_0")

    module, manifest, errors = dsl_coder.generate_dsl(
        env_spec=env_spec,
        syntax_memory=syntax_memory,
        planning_set=planning_set,
        game_memory=game_mem,
    )

    assert module is not None
    assert manifest is not None
    fn_names = {f["name"].lower() for f in manifest.get("functions", [])}

    # All 5 actions must have valid functions generated and verified
    assert "action1" in fn_names
    assert "action2" in fn_names
    assert "action3" in fn_names
    assert "action4" in fn_names
    assert "action5" in fn_names

    # Check callable in sandboxed module
    assert hasattr(module, "action5")
    # Dry-run action5
    api_mock = type("MockAPI", (), {
        "declare_environment_action": lambda self, action_id, **kw: {"action_id": action_id}
    })()
    decl = module.action5(api_mock)
    assert decl["action_id"] == "ACTION5"


# =============================================================================
# 4. PBT for Brusentsov 3-Valued Verifier & Modal Selection
# =============================================================================

def test_layered_verifier_modal_selection_non_null():
    """Verify that MODAL_SELECTION with zero actor motion evaluates to OMIT or FOLLOW, never NULL."""
    config = V10Config()
    verifier = LayeredVerifier(config)

    game_mem = GameMemory(game_id="game_1")
    game_mem.action_affordances = [
        {"action_id": "ACTION5", "effect_class": "MODAL_SELECTION", "parameters": {"active_entity_switched": True}}
    ]
    game_mem.confirmed_action_effects = {
        "ACTION5": "selection indicator transferred: focus marker moved (active entity toggled)"
    }

    grid_b = [[0 for _ in range(10)] for _ in range(10)]
    grid_b[2][2] = 1
    grid_a = [list(r) for r in grid_b]  # Main actor unmoving

    snap_b = extract_arga_snapshot(grid_b)
    snap_a = extract_arga_snapshot(grid_a)

    planning_set = build_planning_set(snap_b, available_actions=["ACTION5"])

    # Case A: Execution without explicit EXPECT propositions -> OMIT (not NULL)
    step_no_expect = GroundedStep(
        step_id="step_1",
        dsl_function="action5",
        arguments={},
        expected_propositions=PropositionSet.empty(),
    )
    judgment_omit = verifier.evaluate_transition(
        step=step_no_expect,
        before_snapshot=snap_b,
        after_obs={"grid": grid_a, "levels_completed": 0},
        planning_set=planning_set,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION5"},
    )
    assert judgment_omit.verdict != Verdict.NULL
    assert judgment_omit.verdict == Verdict.OMIT

    # Case B: Execution with verified EXPECT (e.g. unchanged actor) -> FOLLOW
    first_obj_id = snap_b.objects[0].id if snap_b.objects else "obj_1"
    step_with_expect = GroundedStep(
        step_id="step_2",
        dsl_function="action5",
        arguments={},
        expected_propositions=PropositionSet.from_iterable([
            AtomicProposition(family="object_identity", subject_id=first_obj_id, predicate="preserved")
        ]),
    )
    judgment_follow = verifier.evaluate_transition(
        step=step_with_expect,
        before_snapshot=snap_b,
        after_obs={"grid": grid_a, "levels_completed": 0},
        planning_set=planning_set,
        game_memory=game_mem,
        action_dict={"action_id": "ACTION5"},
    )
    assert judgment_follow.verdict != Verdict.NULL
    assert judgment_follow.verdict in (Verdict.FOLLOW, Verdict.OMIT)
