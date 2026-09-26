"""PlanningSet identity contract and vocabulary grounding for ARC-AGI-3."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from v10_agent.arga_lite import ARGALiteSnapshot, PlanningObject, SpatialRelation
from v10_agent.types import CoordinateCandidate, PlanningAlias, PlanningObjectId


def generate_alias_sequence(count: int) -> list[PlanningAlias]:
    """Generate deterministic alphanumeric aliases: A, B, ..., Z, AA, AB, ..."""
    aliases: list[PlanningAlias] = []
    for i in range(count):
        label = ""
        n = i
        while True:
            label = chr(ord("A") + (n % 26)) + label
            n = n // 26 - 1
            if n < 0:
                break
        aliases.append(label)
    return aliases


@dataclass(frozen=True)
class PlanningSet:
    """Canonical immutable vocabulary for a single planning snapshot cycle (Invariants I1-I8)."""
    snapshot_id: str
    grid_hash: str
    full_grid_hex_rows: tuple[str, ...]
    object_ids: tuple[PlanningObjectId, ...]
    relation_ids: tuple[str, ...]
    allowed_action_ids: tuple[str, ...]
    allowed_coordinate_candidate_ids: tuple[str, ...]
    object_real_to_alias: dict[PlanningObjectId, PlanningAlias]
    object_alias_to_real: dict[PlanningAlias, PlanningObjectId]
    objects: tuple[PlanningObject, ...]
    relations: tuple[SpatialRelation, ...]
    coordinate_candidates: tuple[CoordinateCandidate, ...]
    grid_dims: tuple[int, int] = (0, 0)
    crop_offset: int = 0

    @property
    def grid(self) -> list[list[int]] | None:
        """Reconstruct 2D grid matrix from full_grid_hex_rows if available."""
        if not self.full_grid_hex_rows:
            return None
        return [[int(c, 16) for c in row] for row in self.full_grid_hex_rows]

    def resolve_object_id(self, key: str) -> PlanningObjectId | None:
        """Resolve a real object ID, alias label, or persistent_id to the canonical object ID."""
        if key in self.object_real_to_alias:
            return key
        if key in self.object_alias_to_real:
            return self.object_alias_to_real[key]
        for obj in self.objects:
            if obj.persistent_id is not None and obj.persistent_id == key:
                return obj.id
        return None

    def get_object(self, key: str) -> PlanningObject | None:
        """Retrieve PlanningObject by canonical ID, alias, or persistent ID."""
        canonical_id = self.resolve_object_id(key)
        if canonical_id is None:
            return None
        for obj in self.objects:
            if obj.id == canonical_id:
                return obj
        return None

    def get_coordinate_candidate(self, candidate_id: str) -> CoordinateCandidate | None:
        for coord in self.coordinate_candidates:
            if coord.candidate_id == candidate_id:
                return coord
        return None

    def is_valid_action(self, action_id: str) -> bool:
        return action_id.upper() in self.allowed_action_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "grid_hash": self.grid_hash,
            "grid_dims": {"height": self.grid_dims[0], "width": self.grid_dims[1]},
            "object_ids": list(self.object_ids),
            "allowed_action_ids": list(self.allowed_action_ids),
            "allowed_coordinate_candidate_ids": list(self.allowed_coordinate_candidate_ids),
            "object_aliases": dict(self.object_real_to_alias),
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations],
            "coordinate_candidates": [c.to_dict() for c in self.coordinate_candidates],
        }


def build_planning_set(
    snapshot: ARGALiteSnapshot,
    available_actions: Sequence[str],
    snapshot_id: str | None = None,
    grid_hex_rows: Sequence[str] | None = None,
    tracker: Any | None = None,
    crop_offset: int = 0,
) -> PlanningSet:
    """Build a certified PlanningSet adhering to Invariants I1-I8."""
    if tracker is not None and hasattr(tracker, "update"):
        tracker.update(snapshot.objects, getattr(tracker, "frame_index", 0))

    sid = snapshot_id or str(uuid.uuid4())
    objects = tuple(snapshot.objects)
    object_ids = tuple(obj.id for obj in objects)
    relations = tuple(snapshot.relations)

    # Invariant I2: Build bijective alias mapping
    aliases = generate_alias_sequence(len(objects))
    real_to_alias: dict[str, str] = {}
    alias_to_real: dict[str, str] = {}
    for obj_id, alias in zip(object_ids, aliases):
        real_to_alias[obj_id] = alias
        alias_to_real[alias] = obj_id

    # Compute relation IDs
    rel_ids: list[str] = []
    for r in relations:
        val_str = f"_{r.metric_value:.1f}" if r.metric_value is not None else ""
        rel_ids.append(f"{r.subject_id}:{r.relation_type}:{r.target_id}{val_str}")
    relation_ids = tuple(rel_ids)

    # Allowed actions normalized (ACTION7 is strictly excluded to prevent agent confusion)
    allowed_action_ids = tuple(sorted(set(str(a).upper() for a in available_actions if str(a).upper() != "ACTION7")))

    # Compute coordinate candidates
    coords: list[CoordinateCandidate] = []
    height, width = snapshot.grid_dims

    # Grid center
    if height > 0 and width > 0:
        coords.append(
            CoordinateCandidate(
                candidate_id="coord_center",
                x=width // 2,
                y=height // 2,
                source_type="grid_center",
                label="Center",
            )
        )

    # Order objects with spatial diversity across quadrants & area salience
    mid_r, mid_c = height / 2.0, width / 2.0
    quads: dict[str, list[Any]] = {"TL": [], "TR": [], "BL": [], "BR": []}
    for obj in objects:
        qr = "T" if obj.centroid.row < mid_r else "B"
        qc = "L" if obj.centroid.col < mid_c else "R"
        quads[qr + qc].append(obj)

    ordered_objs: list[Any] = []
    # Largest objects overall first
    for obj in sorted(objects, key=lambda o: -o.area)[:6]:
        if obj.area >= 4 and obj not in ordered_objs:
            ordered_objs.append(obj)
    # Add top objects from each quadrant
    for q_list in quads.values():
        for obj in sorted(q_list, key=lambda o: -o.area)[:10]:
            if obj not in ordered_objs:
                ordered_objs.append(obj)
    # Add remaining objects
    for obj in objects:
        if obj not in ordered_objs:
            ordered_objs.append(obj)

    # Strictly ONE canonical interaction point (centroid) per object with spatial NMS deduplication
    min_separation = 2.5
    for obj in ordered_objs:
        col = int(round(obj.centroid.col))
        row = int(round(obj.centroid.row))
        if not (0 <= col < width and 0 <= row < height):
            continue

        # Spatial NMS: do not add candidate if an existing candidate is within min_separation cells
        is_duplicate = any(math.hypot(c.x - col, c.y - row) < min_separation for c in coords)
        if is_duplicate:
            continue

        alias = real_to_alias.get(obj.id, obj.id)
        coords.append(
            CoordinateCandidate(
                candidate_id=f"coord_c_{obj.id}",
                x=col,
                y=row,
                source_type="object_centroid",
                object_id=obj.id,
                label=alias,
            )
        )
        if len(coords) >= 12:
            break

    coordinate_candidates = tuple(coords)
    allowed_coord_ids = tuple(c.candidate_id for c in coordinate_candidates)

    if grid_hex_rows is not None:
        hex_rows_tuple = tuple(grid_hex_rows)
    else:
        # Reconstruct dummy representation if none provided
        hex_rows_tuple = ()

    grid_hash = hashlib.sha256("\n".join(hex_rows_tuple).encode("utf-8")).hexdigest()

    return PlanningSet(
        snapshot_id=sid,
        grid_hash=grid_hash,
        full_grid_hex_rows=hex_rows_tuple,
        object_ids=object_ids,
        relation_ids=relation_ids,
        allowed_action_ids=allowed_action_ids,
        allowed_coordinate_candidate_ids=allowed_coord_ids,
        object_real_to_alias=real_to_alias,
        object_alias_to_real=alias_to_real,
        objects=objects,
        relations=relations,
        coordinate_candidates=coordinate_candidates,
        grid_dims=snapshot.grid_dims,
        crop_offset=crop_offset,
    )
