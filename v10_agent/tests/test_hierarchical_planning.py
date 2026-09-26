"""Unit tests for Phase 2: Hybrid Hierarchical Planning.

Verifies:
1. Parsing of <waypoints> blocks into structured sub-goal milestones.
2. Symbolic A* pathfinding with obstacle and boundary collision avoidance.
3. Waypoint expansion into collision-free actions with typed EXPECT clauses.
4. Prioritization of waypoint-synthesized trajectories in SolverAgent.
"""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.config import V10Config
from v10_agent.llm_advisor import BaseLLMAdvisor
from v10_agent.planning_set import build_planning_set
from v10_agent.solver_agent import SolverAgent, parse_waypoints_block
from v10_agent.virtual_sandbox import VirtualKinematicSandbox


def test_parse_waypoints_block():
    """Verify parsing diverse waypoint syntax variants into structured dictionaries."""
    sample_xml = """
    <analysis>Navigating towards the green exit.</analysis>
    <waypoints>
    - WAYPOINT: NAVIGATE_TO(subject=A, target=B)
    - WAYPOINT: COLLECT_TARGET(subject=A, target_color=5)
    - WAYPOINT: PUSH_OBJECT(actor=A, pushed=C, destination=D)
    - WAYPOINT: ACTIVATE_TRIGGER(subject=A, trigger=E)
    </waypoints>
    """
    wps = parse_waypoints_block(sample_xml)
    assert len(wps) == 4
    assert wps[0] == {"type": "NAVIGATE_TO", "subject": "A", "target": "B"}
    assert wps[1] == {"type": "COLLECT_TARGET", "subject": "A", "target_color": "5"}
    assert wps[2] == {"type": "PUSH_OBJECT", "actor": "A", "pushed": "C", "destination": "D"}
    assert wps[3] == {"type": "ACTIVATE_TRIGGER", "subject": "A", "trigger": "E"}


def test_astar_obstacle_avoidance_pathfinding():
    """Verify that A* pathfinding navigates around an obstacle to reach the goal."""
    # 8x8 grid
    # Actor at (1, 1) [color 1]
    # Obstacle wall at (2, 1) [color 2]
    # Target at (3, 1) [color 3]
    grid = [[0] * 8 for _ in range(8)]
    grid[1][1] = 1  # Actor
    grid[2][1] = 2  # Wall blocking direct downward move
    grid[3][1] = 3  # Target

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    manifest_map = {
        "action1": {"name": "action1", "docstring": "move up"},
        "action2": {"name": "action2", "docstring": "move down"},
        "action3": {"name": "action3", "docstring": "move left"},
        "action4": {"name": "action4", "docstring": "move right"},
    }

    sandbox = VirtualKinematicSandbox(pset)
    actor_obj = next(o for o in snap.objects if o.color == 1)
    target_obj = next(o for o in snap.objects if o.color == 3)

    path = sandbox._find_path_astar(
        subject=actor_obj,
        start_pos=(1.0, 1.0),
        goal_pos=(3.0, 1.0),
        manifest_functions=manifest_map,
        allowed_target_id=target_obj.id,
    )

    assert path is not None
    assert len(path) > 0

    # Path must detour around (2, 1), so it cannot just be two DOWN moves!
    actions = [p[0] for p in path]
    assert actions != ["action2", "action2"]

    # Execute path and verify end coordinate is (3, 1)
    curr_r, curr_c = 1, 1
    for _, dr, dc in path:
        curr_r += dr
        curr_c += dc
    assert (curr_r, curr_c) == (3, 1)


def test_expand_waypoints_to_trajectory():
    """Verify that high-level waypoints expand to concrete steps with EXPECT clauses."""
    grid = [[0] * 8 for _ in range(8)]
    grid[1][1] = 1  # Actor
    grid[1][4] = 5  # Target marker

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    manifest_map = {
        "action1": {"name": "action1", "docstring": "move up"},
        "action2": {"name": "action2", "docstring": "move down"},
        "action3": {"name": "action3", "docstring": "move left"},
        "action4": {"name": "action4", "docstring": "move right"},
    }

    sandbox = VirtualKinematicSandbox(pset)
    actor_obj = next(o for o in snap.objects if o.color == 1)
    marker_obj = next(o for o in snap.objects if o.color == 5)

    waypoints = [
        {"type": "COLLECT_TARGET", "subject": actor_obj.id, "color": "5"}
    ]

    steps = sandbox.expand_waypoints_to_trajectory(waypoints, manifest_map)
    assert steps is not None
    # Distance from col 1 to col 4 is 3 steps right (action4)
    assert len(steps) == 3
    for s in steps:
        assert s["dsl_function"] == "action4"
        assert len(s["expected_propositions"]) > 0

    # Final step should expect target marker to be gone
    last_step = steps[-1]
    gone_props = [p for p in last_step["expected_propositions"] if p.get("predicate") == "gone"]
    assert len(gone_props) > 0


class MockWaypointAdvisor(BaseLLMAdvisor):
    """Mock advisor that returns a <waypoints> plan."""

    def generate(self, system_prompt: str, user_prompt: str, config: V10Config, **kwargs) -> str:
        return """
        <analysis>I see the player object A at (1, 1) and goal target B at (1, 3).</analysis>
        <waypoints>
        - WAYPOINT: NAVIGATE_TO(subject=A, target=B)
        </waypoints>
        """


def test_solver_agent_hierarchical_waypoint_integration():
    """Verify that SolverAgent prioritizes hierarchical waypoint-synthesized trajectory."""
    grid = [[0] * 8 for _ in range(8)]
    grid[1][1] = 1  # Object A
    grid[1][3] = 4  # Object B

    snap = extract_arga_snapshot(grid)
    pset = build_planning_set(snap, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    manifest = {
        "functions": [
            {"name": "action1", "docstring": "move up"},
            {"name": "action2", "docstring": "move down"},
            {"name": "action3", "docstring": "move left"},
            {"name": "action4", "docstring": "move right"},
        ]
    }

    cfg = V10Config()
    advisor = MockWaypointAdvisor()
    solver = SolverAgent(cfg, advisor)

    package = solver.generate_trajectory_package(
        manifest=manifest,
        planning_set=pset,
    )

    assert package is not None
    candidates = package.get("candidates", [])
    assert len(candidates) > 0

    top_cand = candidates[0]
    assert top_cand.get("strategy") == "hierarchical_waypoint_planner"
    assert top_cand.get("sandbox_goal_reached") is True
    assert len(top_cand.get("steps", [])) == 2
