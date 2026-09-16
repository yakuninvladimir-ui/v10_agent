"""Unit tests for PlanningSet identity contract (Invariants I1-I8)."""

from __future__ import annotations

import pytest

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.observe import grid_to_hex_rows
from v10_agent.planning_set import build_planning_set, generate_alias_sequence


def test_alias_sequence_generation():
    aliases = generate_alias_sequence(60)
    assert aliases[0] == "A"
    assert aliases[25] == "Z"
    assert aliases[26] == "AA"
    assert aliases[27] == "AB"
    assert len(set(aliases)) == 60


def test_planning_set_invariants():
    # 5x5 grid with two objects: a 2x2 square of color 1 (blue) and a 1x1 dot of color 2 (red)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    available_actions = ["ACTION1", "ACTION2", "RESET"]
    hex_rows = grid_to_hex_rows(grid)

    planning_set = build_planning_set(
        snapshot=snapshot,
        available_actions=available_actions,
        grid_hex_rows=hex_rows,
    )

    # Invariant I1: object_ids matches snapshot objects
    assert len(planning_set.object_ids) == 2
    assert tuple(obj.id for obj in planning_set.objects) == planning_set.object_ids

    # Invariant I2: bijective alias mapping
    assert len(planning_set.object_real_to_alias) == 2
    assert len(planning_set.object_alias_to_real) == 2
    for obj_id, alias in planning_set.object_real_to_alias.items():
        assert planning_set.object_alias_to_real[alias] == obj_id

    # Invariant I3: grid hash stability
    assert len(planning_set.grid_hash) == 64
    grid_modified = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 3],  # changed color 2 to 3
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    ps_mod = build_planning_set(
        snapshot=extract_arga_snapshot(grid_modified),
        available_actions=available_actions,
        grid_hex_rows=grid_to_hex_rows(grid_modified),
    )
    assert ps_mod.grid_hash != planning_set.grid_hash

    # Invariant I4: allowed actions
    assert set(planning_set.allowed_action_ids) == {"ACTION1", "ACTION2", "RESET"}
    assert planning_set.is_valid_action("ACTION1") is True
    assert planning_set.is_valid_action("ACTION7") is False

    # Invariant I5: Resolving object IDs and aliases
    alias_0 = planning_set.object_real_to_alias["obj_0"]
    assert planning_set.resolve_object_id("obj_0") == "obj_0"
    assert planning_set.resolve_object_id(alias_0) == "obj_0"
    assert planning_set.resolve_object_id("non_existent_obj") is None

    # Get object by alias or real ID
    obj_by_real = planning_set.get_object("obj_0")
    obj_by_alias = planning_set.get_object(alias_0)
    assert obj_by_real is not None
    assert obj_by_real == obj_by_alias

    # Coordinate candidates
    assert len(planning_set.coordinate_candidates) > 0
    center = planning_set.get_coordinate_candidate("coord_center")
    assert center is not None
    assert center.x == 2
    assert center.y == 2
