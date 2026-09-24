"""Unit tests for Integration Point 1: Continuous 2D Spatial Grounding & Directional Projections."""

import math
from v10_agent.arga_lite import extract_arga_snapshot, PlanningObject, SpatialRelation
from v10_agent.planning_set import build_planning_set
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts


def test_normalized_centroid_mapping():
    # 5x5 grid with objects at corners and center
    grid = [
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [4, 0, 0, 0, 5],
    ]
    snapshot = extract_arga_snapshot(grid)
    objs_by_color = {o.color: o for o in snapshot.objects}

    # Color 1 is at (0, 0) -> normalized (-1.0, -1.0)
    assert objs_by_color[1].normalized_centroid == (-1.0, -1.0)
    # Color 2 is at (0, 4) -> normalized (-1.0, 1.0)
    assert objs_by_color[2].normalized_centroid == (-1.0, 1.0)
    # Color 3 is at (2, 2) -> normalized (0.0, 0.0)
    assert objs_by_color[3].normalized_centroid == (0.0, 0.0)
    # Color 4 is at (4, 0) -> normalized (1.0, -1.0)
    assert objs_by_color[4].normalized_centroid == (1.0, -1.0)
    # Color 5 is at (4, 4) -> normalized (1.0, 1.0)
    assert objs_by_color[5].normalized_centroid == (1.0, 1.0)


def test_multi_dir_canonical_projections():
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    obj = snapshot.objects[0]
    # At center, norm_r=0, norm_c=0 -> all projections 0
    assert obj.multi_dir_relative["proj_0deg_E"] == 0.0
    assert obj.multi_dir_relative["proj_90deg_N"] == 0.0

    # Object at top-right in 3x3 (row 0, col 2) -> norm_r = -1.0, norm_c = 1.0
    grid2 = [
        [0, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
    ]
    snapshot2 = extract_arga_snapshot(grid2)
    obj2 = snapshot2.objects[0]
    assert obj2.multi_dir_relative["proj_0deg_E"] == 1.0
    assert obj2.multi_dir_relative["proj_90deg_N"] == 1.0  # -(-1.0) = 1.0 (North)
    assert obj2.multi_dir_relative["proj_180deg_W"] == -1.0
    assert obj2.multi_dir_relative["proj_270deg_S"] == -1.0


def test_chiral_features_and_mirror_detection():
    # Left-facing L shape (3x2) and right-facing L shape (3x2)
    grid = [
        [1, 0, 0, 0, 2],
        [1, 0, 0, 0, 2],
        [1, 1, 0, 2, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    obj_left = [o for o in snapshot.objects if o.color == 1][0]
    obj_right = [o for o in snapshot.objects if o.color == 2][0]

    assert obj_left.chiral_features["chiral_type"] == "asymmetric_chiral"
    assert obj_left.chiral_features["is_h_symmetric"] is False
    assert obj_right.chiral_features["chiral_type"] == "asymmetric_chiral"

    # Cross-object mirror linkages
    assert obj_right.id in obj_left.chiral_features.get("chiral_mirror_h_of", [])
    assert obj_left.id in obj_right.chiral_features.get("chiral_mirror_h_of", [])


def test_spatial_relation_continuous_angles_and_bins():
    grid = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ]
    snapshot = extract_arga_snapshot(grid)
    dist_rels = [r for r in snapshot.relations if r.relation_type == "distance"]
    assert len(dist_rels) >= 1
    rel = dist_rels[0]
    assert rel.relative_angle_deg is not None
    # Angle from (0,0) to (2,2): dy=2, dx=2 -> 45 degrees (SE)
    assert math.isclose(rel.relative_angle_deg, 45.0, abs_tol=0.1)
    assert "south" in rel.directional_bins
    assert "east" in rel.directional_bins


def test_solver_prompt_ascii_limit_and_single_pixel_filtering():
    # Grid with several substantive objects and single pixels
    # Substantive: 2x2 squares of colors 1, 2, 3
    # Single pixels of colors 4, 5, 6, 7
    grid = [
        [1, 1, 0, 2, 2, 0, 3, 3],
        [1, 1, 0, 2, 2, 0, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 5, 0, 6, 0, 7, 0],
    ]
    snapshot = extract_arga_snapshot(grid)
    pset = build_planning_set(snapshot, ["ACTION1", "ACTION2"])
    _, user_prompt = build_solver_prompts({"functions": []}, pset)

    assert "OBJECT INDEX" in user_prompt
    found_area_1 = False
    found_area_4 = False
    for line in user_prompt.splitlines():
        if "area=1" in line:
            found_area_1 = True
            assert "ascii=" not in line, f"Single-pixel object line {line} must NOT have ascii"
        elif "area=4" in line:
            found_area_4 = True
            assert "ascii=" in line, f"Substantive object line {line} must have ascii"
    assert found_area_1
    assert found_area_4
