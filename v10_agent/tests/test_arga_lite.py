"""Unit tests for ARGALite deterministic object-centric perception."""

from __future__ import annotations

import math
import pytest

from v10_agent.arga_lite import (
    detect_background_color,
    extract_arga_snapshot,
    extract_spatial_relations,
)


def test_empty_grid():
    snapshot = extract_arga_snapshot([])
    assert snapshot.grid_dims == (0, 0)
    assert len(snapshot.objects) == 0
    assert len(snapshot.relations) == 0


def test_single_component_properties():
    # 4x4 grid with a 2x2 square in top-left
    grid = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    assert snapshot.grid_dims == (4, 4)
    assert snapshot.background_color == 0
    assert len(snapshot.objects) == 1

    obj = snapshot.objects[0]
    assert obj.id == "obj_0"
    assert obj.color == 1
    assert obj.area == 4
    assert obj.is_single_color is True
    assert obj.bbox.min_row == 0
    assert obj.bbox.min_col == 0
    assert obj.bbox.max_row == 1
    assert obj.bbox.max_col == 1
    assert obj.centroid.row == 0.5
    assert obj.centroid.col == 0.5


def test_spatial_relations_touches_and_aligned():
    # Two adjacent 1x1 objects
    grid = [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    assert len(snapshot.objects) == 2

    # Verify touches relation
    touch_relations = [
        r for r in snapshot.relations if r.relation_type == "touches"
    ]
    assert len(touch_relations) == 2  # A touches B and B touches A

    # Verify horizontal alignment
    aligned_h = [
        r for r in snapshot.relations if r.relation_type == "aligned_h"
    ]
    assert len(aligned_h) == 2

    # Verify distance
    dist_rels = [
        r for r in snapshot.relations if r.relation_type == "distance"
    ]
    assert len(dist_rels) == 1
    assert math.isclose(dist_rels[0].metric_value, 1.0, abs_tol=1e-5)


def test_spatial_relations_containment():
    # 5x5 grid: outer 3x3 hollow square of color 3 enclosing a 1x1 dot of color 4
    grid = [
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 4, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    assert len(snapshot.objects) == 2

    outer_obj = [o for o in snapshot.objects if o.color == 3][0]
    inner_obj = [o for o in snapshot.objects if o.color == 4][0]

    contains_rels = [
        r for r in snapshot.relations
        if r.relation_type == "contains" and r.subject_id == outer_obj.id and r.target_id == inner_obj.id
    ]
    assert len(contains_rels) == 1


def test_non_zero_background_detection():
    # Grid where color 5 is the dominant border background, no 0s
    grid = [
        [5, 5, 5, 5],
        [5, 2, 2, 5],
        [5, 2, 2, 5],
        [5, 5, 5, 5],
    ]
    bg = detect_background_color(grid)
    assert bg == 5

    snapshot = extract_arga_snapshot(grid)
    assert snapshot.background_color == 5
    assert len(snapshot.objects) == 1
    assert snapshot.objects[0].color == 2


def test_shape_geometric_descriptors_and_ascii():
    # L-shaped 3-pixel corner
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    assert len(snapshot.objects) == 1
    obj = snapshot.objects[0]
    assert obj.width == 2
    assert obj.height == 2
    assert obj.aspect_ratio == 1.0
    assert obj.shape_type == "compound_shape"
    assert obj.compact_ascii == ["##", "#."]
    assert obj.shape_signature.startswith("shape_2x2_")


def test_shape_congruence_identical_shape():
    # Two separate identical L-shaped objects of different colors
    grid = [
        [1, 1, 0, 2, 2],
        [1, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    assert len(snapshot.objects) == 2
    obj_a = snapshot.objects[0]
    obj_b = snapshot.objects[1]
    assert obj_a.shape_signature == obj_b.shape_signature

    identical_rels = [
        r for r in snapshot.relations
        if r.relation_type == "identical_shape"
    ]
    assert len(identical_rels) == 2  # A -> B and B -> A


def test_chiral_mirror_reflection_and_axis():
    # Left L-shape (color 1), vertical axis (color 3), Right mirrored L-shape (color 2)
    grid = [
        [1, 1, 0, 3, 0, 2, 2],
        [1, 0, 0, 3, 0, 0, 2],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    axis = next(o for o in snapshot.objects if o.color == 3)
    left_obj = next(o for o in snapshot.objects if o.color == 1)
    right_obj = next(o for o in snapshot.objects if o.color == 2)

    assert axis.shape_type == "vertical_line"

    # Chiral reflection relations
    mirror_rels = [
        r for r in snapshot.relations
        if r.relation_type == "chiral_mirror_h"
    ]
    assert len(mirror_rels) == 2  # left -> right and right -> left

    # Axis symmetry relations
    axis_rels = [
        r for r in snapshot.relations
        if r.relation_type == "symmetric_axis_of" and r.subject_id == axis.id
    ]
    assert len(axis_rels) == 2  # axis -> left and axis -> right

    mirrored_across = [
        r for r in snapshot.relations
        if r.relation_type == "mirrored_across_axis" and r.target_id == axis.id
    ]
    assert len(mirrored_across) == 2


def test_directional_offsets_substantive_only():
    # Grid with a 2x2 shape (color 1), a 1x1 noise dot (color 2), and a 2x2 shape (color 3)
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 3, 3],
    ]
    snapshot = extract_arga_snapshot(grid)
    obj_1 = next(o for o in snapshot.objects if o.color == 1)
    obj_2 = next(o for o in snapshot.objects if o.color == 2)  # area 1 noise dot
    obj_3 = next(o for o in snapshot.objects if o.color == 3)

    # obj_1 and obj_3 are area 4, so directional offset relations must exist between them
    dir_13 = [
        r for r in snapshot.relations
        if "target_is_" in r.relation_type and r.subject_id == obj_1.id and r.target_id == obj_3.id
    ]
    assert len(dir_13) > 0

    # obj_2 has area 1, so no directional offset relations should exist involving obj_2
    dir_noise = [
        r for r in snapshot.relations
        if "target_is_" in r.relation_type and (r.subject_id == obj_2.id or r.target_id == obj_2.id)
    ]
    assert len(dir_noise) == 0


