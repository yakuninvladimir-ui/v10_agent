"""ARGALite: Deterministic object-centric perception for ARC-AGI-3."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from v10_agent.types import BoundingBox, Centroid, Grid2D


@dataclass
class PlanningObject:
    """Deterministic representation of a detected grid object."""
    id: str
    color: int
    color_histogram: dict[int, int]
    area: int
    bbox: BoundingBox
    centroid: Centroid
    pixels: list[tuple[int, int]]
    is_single_color: bool = True
    width: int = 0
    height: int = 0
    aspect_ratio: float = 1.0
    shape_type: str = "compound_shape"
    shape_signature: str = ""
    filled_shape_signature: str = ""
    compact_ascii: list[str] = field(default_factory=list)
    mask: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    filled_mask: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    role: str = "generic_entity"
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    normalized_centroid: tuple[float, float] = (0.0, 0.0)
    multi_dir_relative: dict[str, float] = field(default_factory=dict)
    freedom_of_motion: dict[str, Any] = field(default_factory=dict)
    chiral_features: dict[str, Any] = field(default_factory=dict)
    persistent_id: str | None = None
    track_confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.width == 0 and self.bbox is not None:
            self.width = self.bbox.max_col - self.bbox.min_col + 1
        if self.height == 0 and self.bbox is not None:
            self.height = self.bbox.max_row - self.bbox.min_row + 1
        if self.height > 0:
            self.aspect_ratio = round(self.width / self.height, 2)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "color": self.color,
            "color_histogram": dict(self.color_histogram),
            "area": self.area,
            "bbox": self.bbox.to_dict(),
            "centroid": self.centroid.to_dict(),
            "pixel_count": len(self.pixels),
            "is_single_color": self.is_single_color,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "shape_type": self.shape_type,
            "shape_signature": self.shape_signature,
            "persistent_id": self.persistent_id,
            "track_confidence": self.track_confidence,
        }
        if self.normalized_centroid != (0.0, 0.0):
            d["normalized_centroid"] = {
                "row": self.normalized_centroid[0],
                "col": self.normalized_centroid[1],
            }
        if self.multi_dir_relative:
            d["multi_dir_relative"] = dict(self.multi_dir_relative)
        if self.freedom_of_motion:
            d["freedom_of_motion"] = dict(self.freedom_of_motion)
        if self.chiral_features:
            d["chiral_features"] = dict(self.chiral_features)
        if self.parent_id is not None:
            d["parent_id"] = self.parent_id
        if self.children_ids:
            d["children_ids"] = list(self.children_ids)
        if self.compact_ascii:
            d["compact_ascii"] = list(self.compact_ascii)
        return d


@dataclass(frozen=True)
class SpatialRelation:
    """Spatial or topological relation between two planning objects."""
    subject_id: str
    relation_type: str  # "touches", "aligned_h", "aligned_v", "contains", "distance"
    target_id: str
    metric_value: float | None = None
    relative_angle_deg: float | None = None
    directional_bins: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "subject": self.subject_id,
            "relation": self.relation_type,
            "target": self.target_id,
        }
        if self.metric_value is not None:
            out["value"] = round(self.metric_value, 2)
        if self.relative_angle_deg is not None:
            out["relative_angle_deg"] = round(self.relative_angle_deg, 1)
        if self.directional_bins:
            out["directional_bins"] = list(self.directional_bins)
        return out


@dataclass
class ARGALiteSnapshot:
    """Immutable perception snapshot for a single grid observation."""
    objects: list[PlanningObject] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    grid_dims: tuple[int, int] = (0, 0)
    background_color: int = 0
    grid: Grid2D = field(default_factory=list)
    levels_completed: int = 0

    @property
    def object_ids(self) -> list[str]:
        return [obj.id for obj in self.objects]

    def get_object(self, object_id: str) -> PlanningObject | None:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_dims": {"height": self.grid_dims[0], "width": self.grid_dims[1]},
            "background_color": self.background_color,
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations],
        }


def detect_background_color(grid: Grid2D) -> int:
    """Determine the background color of the grid based on border and frequency dominance."""
    if not grid or not grid[0]:
        return 0

    height = len(grid)
    width = len(grid[0])
    total_cells = height * width

    # Inspect border pixels
    border_pixels: list[int] = []
    border_pixels.extend(grid[0])
    if height > 1:
        border_pixels.extend(grid[-1])
    for r in range(1, height - 1):
        border_pixels.append(grid[r][0])
        border_pixels.append(grid[r][-1])

    border_counter = Counter(border_pixels)
    border_color, border_count = border_counter.most_common(1)[0]

    all_pixels = [cell for row in grid for cell in row]
    freq = Counter(all_pixels)
    most_common_color, most_common_count = freq.most_common(1)[0]

    # If border has a dominant color (>40% of border), that is the background
    if border_count >= len(border_pixels) * 0.4:
        return border_color

    # If overall most common color occupies >40% of the grid, that is the background
    if most_common_count >= total_cells * 0.4:
        return most_common_color

    # If 0 is present and relatively frequent (>10% of grid), default to 0
    if freq.get(0, 0) >= total_cells * 0.1:
        return 0

    return most_common_color


def segment_connected_components(grid: Grid2D, background_color: int) -> list[list[tuple[int, int]]]:
    """Segment non-background cells of identical color using 4-connectivity."""
    if not grid or not grid[0]:
        return []

    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            color = grid[r][c]
            if color == background_color or (r, c) in visited:
                continue

            # BFS flood fill for monochromatic 4-connected component
            component: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            visited.add((r, c))

            while queue:
                curr_r, curr_c = queue.popleft()
                component.append((curr_r, curr_c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

            components.append(component)

    return components


def compute_shape_properties(
    pixels: list[tuple[int, int]],
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
) -> tuple[
    int,
    int,
    float,
    str,
    str,
    str,
    list[str],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    """Compute geometric properties, canonical masks, signatures, and ASCII representation."""
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    aspect_ratio = round(width / max(height, 1), 2)
    area = len(pixels)

    # 1. Classify basic shape type
    if min(width, height) <= 3 and max(width, height) >= 6:
        shape_type = "vertical_line" if height > width else "horizontal_line"
    elif area == width * height:
        shape_type = "solid_rectangle"
    else:
        shape_type = "compound_shape"

    # 2. Binary mask
    mask_list = [[0] * width for _ in range(height)]
    for r, c in pixels:
        mask_list[r - min_row][c - min_col] = 1

    # 3. Interior hole-filling via flood-fill from bbox borders
    exterior = set()
    queue = deque()
    for r in range(height):
        for c in (0, width - 1):
            if mask_list[r][c] == 0 and (r, c) not in exterior:
                exterior.add((r, c))
                queue.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if mask_list[r][c] == 0 and (r, c) not in exterior:
                exterior.add((r, c))
                queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width:
                if mask_list[nr][nc] == 0 and (nr, nc) not in exterior:
                    exterior.add((nr, nc))
                    queue.append((nr, nc))

    filled_list = [[0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if mask_list[r][c] == 1 or (r, c) not in exterior:
                filled_list[r][c] = 1

    mask = tuple(tuple(row) for row in mask_list)
    filled_mask = tuple(tuple(row) for row in filled_list)

    # 4. Compact ASCII representation for substantive, reasonably sized objects
    compact_ascii: list[str] = []
    if 1 < area and width <= 25 and height <= 25:
        for row in mask:
            compact_ascii.append("".join("#" if cell else "." for cell in row))

    # 5. Deterministic shape signatures
    raw_str = "/".join("".join(str(c) for c in row) for row in mask)
    raw_hash = hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:8]
    shape_signature = f"shape_{width}x{height}_{raw_hash}"

    filled_str = "/".join("".join(str(c) for c in row) for row in filled_mask)
    filled_hash = hashlib.md5(filled_str.encode("utf-8")).hexdigest()[:8]
    filled_shape_signature = f"shape_{width}x{height}_{filled_hash}"

    # 6. Intrinsic chiral / symmetry features
    is_h_symmetric = (mask == tuple(row[::-1] for row in mask))
    is_v_symmetric = (mask == mask[::-1])
    is_diag_symmetric = (width == height and all(mask[r][c] == mask[c][r] for r in range(height) for c in range(width)))
    if is_h_symmetric and is_v_symmetric:
        chiral_type = "fully_symmetric"
    elif is_h_symmetric:
        chiral_type = "h_symmetric"
    elif is_v_symmetric:
        chiral_type = "v_symmetric"
    else:
        chiral_type = "asymmetric_chiral"

    chiral_features = {
        "chiral_type": chiral_type,
        "is_h_symmetric": is_h_symmetric,
        "is_v_symmetric": is_v_symmetric,
        "is_diag_symmetric": is_diag_symmetric,
    }

    return (
        height,
        width,
        aspect_ratio,
        shape_type,
        shape_signature,
        filled_shape_signature,
        compact_ascii,
        mask,
        filled_mask,
        chiral_features,
    )


def extract_spatial_relations(objects: list[PlanningObject]) -> list[SpatialRelation]:
    """Extract pairwise spatial and topological relations between planning objects."""
    relations: list[SpatialRelation] = []
    num_objs = len(objects)

    for i in range(num_objs):
        obj_a = objects[i]
        pix_set_a = set(obj_a.pixels)

        for j in range(i + 1, num_objs):
            obj_b = objects[j]
            pix_set_b = set(obj_b.pixels)

            # 1. Touches (Chebyshev adjacency <= 1)
            touches = False
            for r, c in obj_a.pixels:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in pix_set_b:
                            touches = True
                            break
                    if touches:
                        break
                if touches:
                    break

            if touches:
                relations.append(SpatialRelation(obj_a.id, "touches", obj_b.id))
                relations.append(SpatialRelation(obj_b.id, "touches", obj_a.id))

            # 2. Horizontal alignment (row interval overlap)
            if max(obj_a.bbox.min_row, obj_b.bbox.min_row) <= min(obj_a.bbox.max_row, obj_b.bbox.max_row):
                relations.append(SpatialRelation(obj_a.id, "aligned_h", obj_b.id))
                relations.append(SpatialRelation(obj_b.id, "aligned_h", obj_a.id))

            # 3. Vertical alignment (col interval overlap)
            if max(obj_a.bbox.min_col, obj_b.bbox.min_col) <= min(obj_a.bbox.max_col, obj_b.bbox.max_col):
                relations.append(SpatialRelation(obj_a.id, "aligned_v", obj_b.id))
                relations.append(SpatialRelation(obj_b.id, "aligned_v", obj_a.id))

            # 4. Containment
            if (
                obj_a.bbox.min_row <= obj_b.bbox.min_row
                and obj_a.bbox.max_row >= obj_b.bbox.max_row
                and obj_a.bbox.min_col <= obj_b.bbox.min_col
                and obj_a.bbox.max_col >= obj_b.bbox.max_col
                and obj_a.area > obj_b.area
            ):
                relations.append(SpatialRelation(obj_a.id, "contains", obj_b.id))

            elif (
                obj_b.bbox.min_row <= obj_a.bbox.min_row
                and obj_b.bbox.max_row >= obj_a.bbox.max_row
                and obj_b.bbox.min_col <= obj_a.bbox.min_col
                and obj_b.bbox.max_col >= obj_a.bbox.max_col
                and obj_b.area > obj_a.area
            ):
                relations.append(SpatialRelation(obj_b.id, "contains", obj_a.id))

            # 5. Centroid Distance & Relative Directional Offsets
            dist = math.hypot(obj_a.centroid.row - obj_b.centroid.row, obj_a.centroid.col - obj_b.centroid.col)
            angle_deg_a_to_b = round(math.degrees(math.atan2(obj_b.centroid.row - obj_a.centroid.row, obj_b.centroid.col - obj_a.centroid.col)) % 360.0, 1)

            bins_a_to_b = []
            if obj_b.centroid.row > obj_a.centroid.row:
                bins_a_to_b.append("south")
            elif obj_b.centroid.row < obj_a.centroid.row:
                bins_a_to_b.append("north")
            if obj_b.centroid.col > obj_a.centroid.col:
                bins_a_to_b.append("east")
            elif obj_b.centroid.col < obj_a.centroid.col:
                bins_a_to_b.append("west")

            relations.append(SpatialRelation(
                obj_a.id, "distance", obj_b.id,
                metric_value=dist,
                relative_angle_deg=angle_deg_a_to_b,
                directional_bins=tuple(bins_a_to_b),
            ))

            if obj_a.area > 1 and obj_b.area > 1 and obj_a.color != 0 and obj_b.color != 0:
                delta_row = round(obj_b.centroid.row - obj_a.centroid.row, 1)
                delta_col = round(obj_b.centroid.col - obj_a.centroid.col, 1)
                if abs(delta_row) >= 1.0:
                    rel_v = "target_is_below" if delta_row > 0 else "target_is_above"
                    relations.append(SpatialRelation(
                        obj_a.id, rel_v, obj_b.id,
                        metric_value=abs(delta_row),
                        relative_angle_deg=angle_deg_a_to_b,
                        directional_bins=tuple(bins_a_to_b),
                    ))
                if abs(delta_col) >= 1.0:
                    rel_h = "target_is_to_the_right" if delta_col > 0 else "target_is_to_the_left"
                    relations.append(SpatialRelation(
                        obj_a.id, rel_h, obj_b.id,
                        metric_value=abs(delta_col),
                        relative_angle_deg=angle_deg_a_to_b,
                        directional_bins=tuple(bins_a_to_b),
                    ))

            # 6. Shape Congruence (Identical relative mask or filled mask for substantive objects)
            if obj_a.area > 1 and obj_b.area > 1 and obj_a.width == obj_b.width and obj_a.height == obj_b.height:
                if (obj_a.mask == obj_b.mask) or (obj_a.filled_mask == obj_b.filled_mask):
                    relations.append(SpatialRelation(obj_a.id, "identical_shape", obj_b.id))
                    relations.append(SpatialRelation(obj_b.id, "identical_shape", obj_a.id))
                    if obj_b.id not in obj_a.chiral_features.setdefault("identical_to", []):
                        obj_a.chiral_features["identical_to"].append(obj_b.id)
                    if obj_a.id not in obj_b.chiral_features.setdefault("identical_to", []):
                        obj_b.chiral_features["identical_to"].append(obj_a.id)

            # 7. Horizontal Reflection (Chiral Mirror H)
            if obj_a.area > 1 and obj_b.area > 1 and obj_a.width == obj_b.width and obj_a.height == obj_b.height:
                flipped_b_mask = tuple(row[::-1] for row in obj_b.mask)
                flipped_b_filled = tuple(row[::-1] for row in obj_b.filled_mask)
                if (obj_a.mask == flipped_b_mask) or (obj_a.filled_mask == flipped_b_filled):
                    relations.append(SpatialRelation(obj_a.id, "chiral_mirror_h", obj_b.id))
                    relations.append(SpatialRelation(obj_b.id, "chiral_mirror_h", obj_a.id))
                    if obj_b.id not in obj_a.chiral_features.setdefault("chiral_mirror_h_of", []):
                        obj_a.chiral_features["chiral_mirror_h_of"].append(obj_b.id)
                    if obj_a.id not in obj_b.chiral_features.setdefault("chiral_mirror_h_of", []):
                        obj_b.chiral_features["chiral_mirror_h_of"].append(obj_a.id)

            # 8. Vertical Reflection (Chiral Mirror V)
            if obj_a.area > 1 and obj_b.area > 1 and obj_a.width == obj_b.width and obj_a.height == obj_b.height:
                flipped_b_mask_v = obj_b.mask[::-1]
                flipped_b_filled_v = obj_b.filled_mask[::-1]
                if (obj_a.mask == flipped_b_mask_v) or (obj_a.filled_mask == flipped_b_filled_v):
                    relations.append(SpatialRelation(obj_a.id, "chiral_mirror_v", obj_b.id))
                    relations.append(SpatialRelation(obj_b.id, "chiral_mirror_v", obj_a.id))
                    if obj_b.id not in obj_a.chiral_features.setdefault("chiral_mirror_v_of", []):
                        obj_a.chiral_features["chiral_mirror_v_of"].append(obj_b.id)
                    if obj_a.id not in obj_b.chiral_features.setdefault("chiral_mirror_v_of", []):
                        obj_b.chiral_features["chiral_mirror_v_of"].append(obj_a.id)

    # 9. Axis Symmetry (Linear axis object positioned symmetrically between mirrored/congruent pairs)
    chiral_or_identical_pairs = {
        (r.subject_id, r.target_id)
        for r in relations
        if r.relation_type in ("chiral_mirror_h", "chiral_mirror_v", "identical_shape")
    }

    for axis_obj in objects:
        if axis_obj.shape_type == "vertical_line" or (axis_obj.height >= 5 and axis_obj.width <= 3):
            for i in range(num_objs):
                obj_a = objects[i]
                if obj_a.id == axis_obj.id or obj_a.area <= 1:
                    continue
                for j in range(i + 1, num_objs):
                    obj_b = objects[j]
                    if obj_b.id == axis_obj.id or obj_b.area <= 1:
                        continue
                    if (obj_a.id, obj_b.id) not in chiral_or_identical_pairs:
                        continue
                    if (obj_a.centroid.col < axis_obj.centroid.col < obj_b.centroid.col) or (
                        obj_b.centroid.col < axis_obj.centroid.col < obj_a.centroid.col
                    ):
                        dist_a = abs(axis_obj.centroid.col - obj_a.centroid.col)
                        dist_b = abs(axis_obj.centroid.col - obj_b.centroid.col)
                        if abs(dist_a - dist_b) <= 2.0:
                            relations.append(SpatialRelation(axis_obj.id, "symmetric_axis_of", obj_a.id, metric_value=round(dist_a, 1)))
                            relations.append(SpatialRelation(axis_obj.id, "symmetric_axis_of", obj_b.id, metric_value=round(dist_b, 1)))
                            relations.append(SpatialRelation(obj_a.id, "mirrored_across_axis", axis_obj.id, metric_value=round(dist_a, 1)))
                            relations.append(SpatialRelation(obj_b.id, "mirrored_across_axis", axis_obj.id, metric_value=round(dist_b, 1)))

        elif axis_obj.shape_type == "horizontal_line" or (axis_obj.width >= 5 and axis_obj.height <= 3):
            for i in range(num_objs):
                obj_a = objects[i]
                if obj_a.id == axis_obj.id or obj_a.area <= 1:
                    continue
                for j in range(i + 1, num_objs):
                    obj_b = objects[j]
                    if obj_b.id == axis_obj.id or obj_b.area <= 1:
                        continue
                    if (obj_a.id, obj_b.id) not in chiral_or_identical_pairs:
                        continue
                    if (obj_a.centroid.row < axis_obj.centroid.row < obj_b.centroid.row) or (
                        obj_b.centroid.row < axis_obj.centroid.row < obj_a.centroid.row
                    ):
                        dist_a = abs(axis_obj.centroid.row - obj_a.centroid.row)
                        dist_b = abs(axis_obj.centroid.row - obj_b.centroid.row)
                        if abs(dist_a - dist_b) <= 2.0:
                            relations.append(SpatialRelation(axis_obj.id, "symmetric_axis_of", obj_a.id, metric_value=round(dist_a, 1)))
                            relations.append(SpatialRelation(axis_obj.id, "symmetric_axis_of", obj_b.id, metric_value=round(dist_b, 1)))
                            relations.append(SpatialRelation(obj_a.id, "mirrored_across_axis", axis_obj.id, metric_value=round(dist_a, 1)))
                            relations.append(SpatialRelation(obj_b.id, "mirrored_across_axis", axis_obj.id, metric_value=round(dist_b, 1)))

    return relations


def merge_collinear_axis_components(
    raw_components: list[list[tuple[int, int]]],
    grid: Grid2D,
    min_area: int = 6,
    min_span: int = 15,
) -> list[list[tuple[int, int]]]:
    """Merge fragmented collinear horizontal/vertical mirror axis line segments interrupted by obstacles/markers."""
    if len(raw_components) <= 1:
        return raw_components

    def comp_stats(c: list[tuple[int, int]]) -> dict[str, Any]:
        rows = [r for r, _ in c]
        cols = [col for _, col in c]
        color = grid[c[0][0]][c[0][1]]
        return {
            "color": color,
            "min_r": min(rows),
            "max_r": max(rows),
            "min_c": min(cols),
            "max_c": max(cols),
            "area": len(c),
        }

    merged = list(raw_components)
    changed = True
    while changed:
        changed = False
        n = len(merged)
        for i in range(n):
            if changed:
                break
            st_a = comp_stats(merged[i])
            for j in range(i + 1, n):
                st_b = comp_stats(merged[j])
                if st_a["color"] != st_b["color"]:
                    continue

                # Horizontal collinear axis segments sharing identical row bounds
                h_a = st_a["max_r"] - st_a["min_r"] + 1
                h_b = st_b["max_r"] - st_b["min_r"] + 1
                is_h = (
                    h_a <= 3 and h_b <= 3
                    and st_a["min_r"] == st_b["min_r"]
                    and st_a["max_r"] == st_b["max_r"]
                    and st_a["area"] >= min_area and st_b["area"] >= min_area
                    and (max(st_a["max_c"], st_b["max_c"]) - min(st_a["min_c"], st_b["min_c"]) + 1) >= min_span
                    and (abs(st_a["min_c"] - st_b["max_c"]) <= 20 or abs(st_b["min_c"] - st_a["max_c"]) <= 20)
                )

                # Vertical collinear axis segments sharing identical column bounds
                w_a = st_a["max_c"] - st_a["min_c"] + 1
                w_b = st_b["max_c"] - st_b["min_c"] + 1
                is_v = (
                    w_a <= 3 and w_b <= 3
                    and st_a["min_c"] == st_b["min_c"]
                    and st_a["max_c"] == st_b["max_c"]
                    and st_a["area"] >= min_area and st_b["area"] >= min_area
                    and (max(st_a["max_r"], st_b["max_r"]) - min(st_a["min_r"], st_b["min_r"]) + 1) >= min_span
                    and (abs(st_a["min_r"] - st_b["max_r"]) <= 20 or abs(st_b["min_r"] - st_a["max_r"]) <= 20)
                )

                if is_h or is_v:
                    new_comp = merged[i] + merged[j]
                    merged.pop(j)
                    merged.pop(i)
                    merged.append(new_comp)
                    changed = True
                    break
    return merged


def extract_arga_snapshot(grid: Grid2D, perception_config: Any | None = None) -> ARGALiteSnapshot:
    """Extract a complete ARGALite perception snapshot from a 2D grid."""
    if not grid or not grid[0]:
        return ARGALiteSnapshot(grid_dims=(0, 0))

    min_area = getattr(perception_config, "axis_merge_min_area", 6) if perception_config else 6
    min_span = getattr(perception_config, "axis_merge_max_gap", 15) if perception_config else 15

    height = len(grid)
    width = len(grid[0])
    bg_color = detect_background_color(grid)
    raw_components = segment_connected_components(grid, bg_color)
    raw_components = merge_collinear_axis_components(raw_components, grid, min_area=min_area, min_span=min_span)

    # Sort components deterministically: top-to-bottom, left-to-right, then largest area
    def sort_key(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        min_r = min(r for r, _ in comp)
        min_c = min(c for _, c in comp)
        color = grid[comp[0][0]][comp[0][1]]
        return (min_r, min_c, -len(comp), color)

    raw_components.sort(key=sort_key)

    objects: list[PlanningObject] = []
    for idx, comp in enumerate(raw_components):
        obj_id = f"obj_{idx}"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        area = len(comp)

        color_hist = Counter(grid[r][c] for r, c in comp)
        primary_color = color_hist.most_common(1)[0][0]

        centroid_r = sum(rows) / area
        centroid_c = sum(cols) / area

        (
            h,
            w,
            ar,
            stype,
            sig,
            filled_sig,
            ascii_art,
            mask,
            filled_mask,
            chiral_feat,
        ) = compute_shape_properties(comp, min_r, min_c, max_r, max_c)

        norm_r = round(((centroid_r / (height - 1)) * 2.0 - 1.0), 3) if height > 1 else 0.0
        norm_c = round(((centroid_c / (width - 1)) * 2.0 - 1.0), 3) if width > 1 else 0.0
        norm_centroid = (norm_r, norm_c)

        multi_dir = {
            "proj_0deg_E": round(norm_c, 3),
            "proj_45deg_NE": round(0.7071 * norm_c - 0.7071 * norm_r, 3),
            "proj_90deg_N": round(-norm_r, 3),
            "proj_135deg_NW": round(-0.7071 * norm_c - 0.7071 * norm_r, 3),
            "proj_180deg_W": round(-norm_c, 3),
            "proj_225deg_SW": round(-0.7071 * norm_c + 0.7071 * norm_r, 3),
            "proj_270deg_S": round(norm_r, 3),
            "proj_315deg_SE": round(0.7071 * norm_c + 0.7071 * norm_r, 3),
        }

        fom = {
            "up_max_steps": min_r,
            "down_max_steps": max(0, (height - 1) - max_r),
            "left_max_steps": min_c,
            "right_max_steps": max(0, (width - 1) - max_c),
        }

        objects.append(
            PlanningObject(
                id=obj_id,
                color=primary_color,
                color_histogram=dict(color_hist),
                area=area,
                bbox=BoundingBox(min_row=min_r, min_col=min_c, max_row=max_r, max_col=max_c),
                centroid=Centroid(row=centroid_r, col=centroid_c),
                pixels=sorted(comp),
                is_single_color=(len(color_hist) == 1),
                width=w,
                height=h,
                aspect_ratio=ar,
                shape_type=stype,
                shape_signature=sig,
                filled_shape_signature=filled_sig,
                compact_ascii=ascii_art,
                mask=mask,
                filled_mask=filled_mask,
                normalized_centroid=norm_centroid,
                multi_dir_relative=multi_dir,
                freedom_of_motion=fom,
                chiral_features=chiral_feat,
            )
        )

    relations = extract_spatial_relations(objects)
    detect_spatial_enclosure(objects, relations)
    assign_functional_roles(objects, height, width)

    return ARGALiteSnapshot(
        objects=objects,
        relations=relations,
        grid_dims=(height, width),
        background_color=bg_color,
        grid=grid,
    )


def detect_spatial_enclosure(objects: list[PlanningObject], relations: list[SpatialRelation]) -> None:
    """Detect geometric enclosure hierarchy (parent contains child components)."""
    for child in objects:
        best_parent = None
        best_parent_area = float("inf")
        for parent in objects:
            if parent.id == child.id:
                continue
            if parent.area > child.area:
                if (
                    parent.bbox.min_row <= child.bbox.min_row
                    and child.bbox.max_row <= parent.bbox.max_row
                    and parent.bbox.min_col <= child.bbox.min_col
                    and child.bbox.max_col <= parent.bbox.max_col
                ):
                    if parent.area < best_parent_area:
                        best_parent = parent
                        best_parent_area = parent.area
        if best_parent is not None:
            child.parent_id = best_parent.id
            if child.id not in best_parent.children_ids:
                best_parent.children_ids.append(child.id)
            if not any(r.subject_id == best_parent.id and r.relation_type == "contains" and r.target_id == child.id for r in relations):
                relations.append(SpatialRelation(subject_id=best_parent.id, relation_type="contains", target_id=child.id))
            if not any(r.subject_id == child.id and r.relation_type == "inside_of" and r.target_id == best_parent.id for r in relations):
                relations.append(SpatialRelation(subject_id=child.id, relation_type="inside_of", target_id=best_parent.id))

    for obj in objects:
        if obj.children_ids and obj.shape_type in ("compound_shape", "solid_rectangle"):
            obj.shape_type = "compound_container"


def assign_functional_roles(objects: list[PlanningObject], height: int, width: int) -> None:
    """Assign neutral default role to all planning objects (enforces ISO-7)."""
    for obj in objects:
        if not hasattr(obj, "role") or not obj.role:
            obj.role = "generic_entity"

