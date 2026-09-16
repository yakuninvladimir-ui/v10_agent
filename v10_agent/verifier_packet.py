"""Symbolic Channel: Offline verifier packet generation."""

from __future__ import annotations

from typing import Any

from v10_agent.planning_set import PlanningSet


def build_verifier_packet(
    planning_set: PlanningSet,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the offline verifier packet for the symbolic channel."""
    meta = dict(metadata or {})

    object_catalog: list[dict[str, Any]] = []
    for obj in planning_set.objects:
        alias = planning_set.object_real_to_alias.get(obj.id, obj.id)
        object_catalog.append({
            "id": obj.id,
            "alias": alias,
            "color": obj.color,
            "area": obj.area,
            "bbox": obj.bbox.to_dict(),
            "centroid": obj.centroid.to_dict(),
            "is_single_color": obj.is_single_color,
        })

    spatial_relations: list[dict[str, Any]] = [rel.to_dict() for rel in planning_set.relations]
    coordinate_affordances: list[dict[str, Any]] = [c.to_dict() for c in planning_set.coordinate_candidates]

    return {
        "schema_version": "v10.verifier_packet.1",
        "snapshot_id": planning_set.snapshot_id,
        "grid_hash": planning_set.grid_hash,
        "grid_dims": {
            "height": planning_set.grid_dims[0],
            "width": planning_set.grid_dims[1],
        },
        "full_grid_hex_rows": list(planning_set.full_grid_hex_rows),
        "object_catalog": object_catalog,
        "spatial_relations": spatial_relations,
        "coordinate_affordances": coordinate_affordances,
        "action_surface": list(planning_set.allowed_action_ids),
        "metadata": meta,
    }
