"""Unit tests for Dual-View media (visual PNG & symbolic verifier packet)."""

from __future__ import annotations

import io
import pytest
from PIL import Image

from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.frame_media import render_annotated_frame_png, render_dual_frame_png, render_grid_png
from v10_agent.observe import grid_to_hex_rows
from v10_agent.planning_set import build_planning_set
from v10_agent.verifier_packet import build_verifier_packet

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def test_render_grid_png():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    png_bytes = render_grid_png(grid, scale=10)
    assert png_bytes.startswith(PNG_HEADER)

    # Verify image can be opened by PIL
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (30, 30)
    assert img.format == "PNG"


def test_dual_view_shared_planning_set():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    available_actions = ["ACTION1", "ACTION2", "ACTION6", "RESET"]
    hex_rows = grid_to_hex_rows(grid)

    planning_set = build_planning_set(
        snapshot=snapshot,
        available_actions=available_actions,
        grid_hex_rows=hex_rows,
    )

    # 1. Visual channel
    annotated_png = render_annotated_frame_png(grid, planning_set, scale=16)
    assert annotated_png.startswith(PNG_HEADER)
    img = Image.open(io.BytesIO(annotated_png))
    assert img.size == (80, 80)

    # 2. Symbolic channel
    packet = build_verifier_packet(planning_set, metadata={"frame_index": 1, "game_id": "test_game"})

    # Assert invariant: both channels share the exact same PlanningSet
    assert packet["snapshot_id"] == planning_set.snapshot_id
    assert packet["grid_hash"] == planning_set.grid_hash
    assert packet["grid_dims"]["height"] == 5
    assert packet["grid_dims"]["width"] == 5
    assert packet["action_surface"] == list(planning_set.allowed_action_ids)

    # Assert object catalog matches planning_set
    assert len(packet["object_catalog"]) == len(planning_set.objects)
    for obj_meta in packet["object_catalog"]:
        obj_id = obj_meta["id"]
        assert obj_id in planning_set.object_ids
        assert obj_meta["alias"] == planning_set.object_real_to_alias[obj_id]
        real_obj = planning_set.get_object(obj_id)
        assert real_obj is not None
        assert obj_meta["color"] == real_obj.color
        assert obj_meta["area"] == real_obj.area

    # Assert coordinate affordances match planning_set
    assert len(packet["coordinate_affordances"]) == len(planning_set.coordinate_candidates)


def test_render_dual_frame_png():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    snapshot = extract_arga_snapshot(grid)
    planning_set = build_planning_set(
        snapshot=snapshot,
        available_actions=["ACTION1"],
        grid_hex_rows=grid_to_hex_rows(grid),
    )
    png_bytes = render_dual_frame_png(grid, planning_set, scale=16)
    assert png_bytes.startswith(PNG_HEADER)

    img = Image.open(io.BytesIO(png_bytes))
    # Dual frame has width = 3*16 * 2 + 4 = 100, height = 3*16 + 24 = 72
    assert img.size == (100, 72)
    assert img.format == "PNG"

