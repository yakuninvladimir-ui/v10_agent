"""Universal Invariant Module for Domain-General Discovery and Evaluation.

Extracts algebraic geometric invariants (axial/point symmetries, shape sockets,
contact, pattern matching) from the perception graph without game-specific heuristics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from v10_agent.planning_set import PlanningObject, PlanningSet

logger = logging.getLogger("v10_agent.universal_invariants")


@dataclass(frozen=True)
class DiscoveredInvariant:
    """An algebraic relational invariant linking objects or states."""
    invariant_type: str  # 'axial_symmetry_vertical' | 'axial_symmetry_horizontal' | 'socket_coverage' | 'spatial_contact' | 'pattern_match'
    subject_id: str
    target_id: str
    axis_id: str | None = None
    axis_coordinate: float | None = None  # col for vertical axis, row for horizontal axis
    target_position: tuple[float, float] | None = None  # (row, col)
    confidence: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.invariant_type,
            "subject": self.subject_id,
            "target": self.target_id,
            "confidence": self.confidence,
            "description": self.description,
        }
        if self.axis_id is not None:
            d["axis_id"] = self.axis_id
        if self.axis_coordinate is not None:
            d["axis_coordinate"] = round(self.axis_coordinate, 2)
        if self.target_position is not None:
            d["target_position"] = (round(self.target_position[0], 1), round(self.target_position[1], 1))
        return d


@dataclass(frozen=True)
class ConnectedComponentConservation:
    """Topology invariant: object retains 4/8 connectivity under translations."""
    subject_id: str
    component_count: int = 1
    area: int = 1
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "connected_component_conservation",
            "subject": self.subject_id,
            "component_count": self.component_count,
            "area": self.area,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class GravitySettling:
    """Physics invariant: dynamic entity settles in a fixed directional gradient (dy, dx)."""
    subject_id: str
    direction: tuple[int, int]  # (dy, dx), e.g. (1, 0) for down
    support_id: str | None = None
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gravity_settling",
            "subject": self.subject_id,
            "direction": self.direction,
            "support": self.support_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ContactTrigger:
    """Interaction invariant: spatial adjacency between subject and trigger produces state transition."""
    subject_id: str
    trigger_id: str
    consequence: str  # e.g., 'barrier_toggle', 'portal_teleport', 'color_change'
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "contact_trigger",
            "subject": self.subject_id,
            "trigger": self.trigger_id,
            "consequence": self.consequence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class AreaConservation:
    """Geometric invariant: object pixel area is conserved across state transformations."""
    subject_id: str
    area: int
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "area_conservation",
            "subject": self.subject_id,
            "area": self.area,
            "confidence": self.confidence,
        }


def discover_invariants(
    planning_set: PlanningSet,
    confirmed_actors: set[str] | None = None,
) -> list[DiscoveredInvariant]:
    """Extract prioritized invariant hypotheses from a PlanningSet."""
    invariants: list[DiscoveredInvariant] = []
    objects = list(planning_set.objects)
    relations = list(planning_set.relations)
    grid_h, grid_w = planning_set.grid_dims

    if not objects:
        return invariants

    # Index relations by type
    chiral_h_pairs: set[tuple[str, str]] = set()
    chiral_v_pairs: set[tuple[str, str]] = set()
    identical_pairs: set[tuple[str, str]] = set()
    axis_to_targets: dict[str, list[str]] = defaultdict(list)

    for rel in relations:
        if rel.relation_type == "chiral_mirror_h":
            chiral_h_pairs.add((rel.subject_id, rel.target_id))
        elif rel.relation_type == "chiral_mirror_v":
            chiral_v_pairs.add((rel.subject_id, rel.target_id))
        elif rel.relation_type == "identical_shape":
            identical_pairs.add((rel.subject_id, rel.target_id))
        elif rel.relation_type == "symmetric_axis_of":
            axis_to_targets[rel.subject_id].append(rel.target_id)
        elif rel.relation_type == "mirrored_across_axis":
            axis_to_targets[rel.target_id].append(rel.subject_id)

    # 1. High-confidence explicit axis relations detected by ARGA-Lite
    for axis_id, target_ids in axis_to_targets.items():
        unique_targets = list(dict.fromkeys(target_ids))
        axis_obj = planning_set.get_object(axis_id)
        if not axis_obj:
            continue
        is_vertical = axis_obj.height >= axis_obj.width

        # Check pairs of targets mirrored across this axis
        for i in range(len(unique_targets)):
            for j in range(i + 1, len(unique_targets)):
                t1_id = unique_targets[i]
                t2_id = unique_targets[j]
                if confirmed_actors:
                    if t2_id in confirmed_actors and t1_id not in confirmed_actors:
                        t1_id, t2_id = t2_id, t1_id
                t1 = planning_set.get_object(t1_id)
                t2 = planning_set.get_object(t2_id)
                if not t1 or not t2:
                    continue

                if is_vertical:
                    invariants.append(
                        DiscoveredInvariant(
                            invariant_type="axial_symmetry_vertical",
                            subject_id=t1.id,
                            target_id=t2.id,
                            axis_id=axis_obj.id,
                            axis_coordinate=float(axis_obj.centroid.col),
                            target_position=(t2.centroid.row, t2.centroid.col),
                            confidence=0.98,
                            description=f"Reflect {t1.id} across vertical axis {axis_obj.id} to match {t2.id}",
                        )
                    )
                else:
                    invariants.append(
                        DiscoveredInvariant(
                            invariant_type="axial_symmetry_horizontal",
                            subject_id=t1.id,
                            target_id=t2.id,
                            axis_id=axis_obj.id,
                            axis_coordinate=float(axis_obj.centroid.row),
                            target_position=(t2.centroid.row, t2.centroid.col),
                            confidence=0.98,
                            description=f"Reflect {t1.id} across horizontal axis {axis_obj.id} to match {t2.id}",
                        )
                    )

    # 2. Discover Virtual / Emergent Symmetries from Chiral Pairs
    for sub_id, tgt_id in chiral_h_pairs:
        if confirmed_actors and (tgt_id in confirmed_actors and sub_id not in confirmed_actors):
            sub_id, tgt_id = tgt_id, sub_id
        sub = planning_set.get_object(sub_id)
        tgt = planning_set.get_object(tgt_id)
        if sub and tgt and sub.area > 1 and tgt.area > 1:
            midpoint_c = (sub.centroid.col + tgt.centroid.col) / 2.0
            axis_candidate = None
            for obj in objects:
                if obj.id not in (sub_id, tgt_id):
                    is_vert_elongated = (obj.height / max(1, obj.width) >= 2.0) and (obj.height >= max(3, int(round(grid_h * 0.2))))
                    is_near_midpoint = (abs(obj.centroid.col - midpoint_c) <= max(1.5, grid_w * 0.08)) and (obj.height >= obj.width)
                    if is_vert_elongated or is_near_midpoint:
                        axis_candidate = obj
                        break

            axis_coord = float(axis_candidate.centroid.col) if axis_candidate else midpoint_c
            desc = (
                f"Vertical reflection of {sub_id} across axis {axis_candidate.id} (align axis to col {midpoint_c:.1f}) to match {tgt_id}"
                if axis_candidate
                else f"Vertical reflection of {sub_id} to match {tgt_id} at col {midpoint_c:.1f}"
            )
            invariants.append(
                DiscoveredInvariant(
                    invariant_type="axial_symmetry_vertical",
                    subject_id=sub_id,
                    target_id=tgt_id,
                    axis_id=axis_candidate.id if axis_candidate else None,
                    axis_coordinate=axis_coord,
                    target_position=(tgt.centroid.row, tgt.centroid.col),
                    confidence=0.92 if axis_candidate else 0.8,
                    description=desc,
                )
            )

    for sub_id, tgt_id in chiral_v_pairs:
        if confirmed_actors and (tgt_id in confirmed_actors and sub_id not in confirmed_actors):
            sub_id, tgt_id = tgt_id, sub_id
        sub = planning_set.get_object(sub_id)
        tgt = planning_set.get_object(tgt_id)
        if sub and tgt and sub.area > 1 and tgt.area > 1:
            midpoint_r = (sub.centroid.row + tgt.centroid.row) / 2.0
            axis_candidate = None
            for obj in objects:
                if obj.id not in (sub_id, tgt_id):
                    is_horiz_elongated = (obj.width / max(1, obj.height) >= 2.0) and (obj.width >= max(3, int(round(grid_w * 0.2))))
                    is_near_midpoint = (abs(obj.centroid.row - midpoint_r) <= max(1.5, grid_h * 0.08)) and (obj.width >= obj.height)
                    if is_horiz_elongated or is_near_midpoint:
                        axis_candidate = obj
                        break

            axis_coord = float(axis_candidate.centroid.row) if axis_candidate else midpoint_r
            desc = (
                f"Horizontal reflection of {sub_id} across axis {axis_candidate.id} (align axis to row {midpoint_r:.1f}) to match {tgt_id}"
                if axis_candidate
                else f"Horizontal reflection of {sub_id} to match {tgt_id} at row {midpoint_r:.1f}"
            )
            invariants.append(
                DiscoveredInvariant(
                    invariant_type="axial_symmetry_horizontal",
                    subject_id=sub_id,
                    target_id=tgt_id,
                    axis_id=axis_candidate.id if axis_candidate else None,
                    axis_coordinate=axis_coord,
                    target_position=(tgt.centroid.row, tgt.centroid.col),
                    confidence=0.92 if axis_candidate else 0.8,
                    description=desc,
                )
            )

    # 3. Discover Socket Coverage (Identical / Congruent shapes to stationary sockets)
    for sub_id, tgt_id in identical_pairs:
        if confirmed_actors and (tgt_id in confirmed_actors and sub_id not in confirmed_actors):
            sub_id, tgt_id = tgt_id, sub_id
        sub = planning_set.get_object(sub_id)
        tgt = planning_set.get_object(tgt_id)
        if sub and tgt and sub.area > 1 and tgt.area > 1:
            invariants.append(
                DiscoveredInvariant(
                    invariant_type="socket_coverage",
                    subject_id=sub_id,
                    target_id=tgt_id,
                    target_position=(tgt.centroid.row, tgt.centroid.col),
                    confidence=0.85,
                    description=f"Translate {sub_id} into congruent socket {tgt_id}",
                )
            )

    # 4. Color Palette & Dimensional Ratios
    # Group objects by color and size to find clusters
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            o1, o2 = objects[i], objects[j]
            # Same color
            if o1.color == o2.color:
                invariants.append(DiscoveredInvariant(
                    invariant_type="color_palette",
                    subject_id=o1.id, target_id=o2.id,
                    confidence=0.6, description=f"Objects share color {o1.color}"
                ))
            # Identical size
            if o1.width == o2.width and o1.height == o2.height:
                invariants.append(DiscoveredInvariant(
                    invariant_type="dimensional_ratio",
                    subject_id=o1.id, target_id=o2.id,
                    confidence=0.6, description=f"Objects have identical size ({o1.width}x{o1.height})"
                ))
            # Positional pattern (same row or col)
            if o1.centroid.row == o2.centroid.row:
                invariants.append(DiscoveredInvariant(
                    invariant_type="positional_pattern",
                    subject_id=o1.id, target_id=o2.id,
                    confidence=0.5, description=f"Objects align horizontally at row {o1.centroid.row}"
                ))
            elif o1.centroid.col == o2.centroid.col:
                invariants.append(DiscoveredInvariant(
                    invariant_type="positional_pattern",
                    subject_id=o1.id, target_id=o2.id,
                    confidence=0.5, description=f"Objects align vertically at col {o1.centroid.col}"
                ))
    
    # 5. Container Hierarchy (Object inside Object)
    for rel in relations:
        if rel.relation_type == "contained_within":
            invariants.append(DiscoveredInvariant(
                invariant_type="container_hierarchy",
                subject_id=rel.subject_id, target_id=rel.target_id,
                confidence=0.9, description=f"Object {rel.subject_id} is inside {rel.target_id}"
            ))

    # 6. Topological Invariants (Component & Area Conservation, Gravity, Contact Triggers)
    for obj in objects:
        if obj.area > 0 and obj.color != 0:
            # Area conservation
            invariants.append(DiscoveredInvariant(
                invariant_type="area_conservation",
                subject_id=obj.id, target_id=obj.id,
                confidence=0.95,
                description=f"Area conservation: {obj.id} maintains area {obj.area}",
            ))
            # Connected component conservation
            invariants.append(DiscoveredInvariant(
                invariant_type="connected_component_conservation",
                subject_id=obj.id, target_id=obj.id,
                confidence=0.95,
                description=f"Connected component conservation: {obj.id} maintains topological unity",
            ))

    # Contact triggers (spatial contact between dynamic actors and potential targets/hazards)
    for rel in relations:
        if rel.relation_type in ("adjacent_to", "touches", "aligned_with"):
            invariants.append(DiscoveredInvariant(
                invariant_type="contact_trigger",
                subject_id=rel.subject_id, target_id=rel.target_id,
                confidence=0.85,
                description=f"Contact trigger: {rel.subject_id} interacting with {rel.target_id}",
            ))

    # Deduplicate invariants
    unique_invariants: list[DiscoveredInvariant] = []
    seen_sigs: set[tuple[str, str, str]] = set()
    for inv in invariants:
        sig = (inv.subject_id, inv.target_id, inv.invariant_type)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique_invariants.append(inv)

    # Sort prioritizing unachieved invariants (D0 > TOLERANCE) and confirmed actors, then by confidence
    def sort_key(inv: DiscoveredInvariant) -> tuple[int, int, float]:
        sub = planning_set.get_object(inv.subject_id)
        tgt = planning_set.get_object(inv.target_id)
        ax = planning_set.get_object(inv.axis_id) if inv.axis_id else None
        if not sub or not tgt:
            return (0, 0, inv.confidence)
        if "vertical" in inv.invariant_type:
            ax_val = ax.centroid.col if ax else inv.axis_coordinate
            tol = 1.0
        elif "horizontal" in inv.invariant_type:
            ax_val = ax.centroid.row if ax else inv.axis_coordinate
            tol = 1.0
        else:
            ax_val = None
            tol = 0.5
        tgt_r, tgt_c = inv.target_position if inv.target_position else (tgt.centroid.row, tgt.centroid.col)
        d0 = compute_invariant_distance(sub.centroid.row, sub.centroid.col, ax_val, tgt_r, tgt_c, inv.invariant_type)
        is_unachieved = 1 if d0 > tol else 0
        is_actor = 1 if (confirmed_actors and inv.subject_id in confirmed_actors) else 0
        return (is_unachieved, is_actor, inv.confidence)

    unique_invariants.sort(key=sort_key, reverse=True)
    return unique_invariants


def compute_invariant_distance(
    subject_r: float,
    subject_c: float,
    axis_val: float | None,
    target_r: float,
    target_c: float,
    invariant_type: str,
) -> float:
    """Compute algebraic distance between current subject position and invariant target."""
    if invariant_type == "axial_symmetry_vertical":
        refl_c = (2.0 * axis_val - subject_c) if axis_val is not None else subject_c
        return abs(refl_c - target_c) + abs(subject_r - target_r)
    elif invariant_type == "axial_symmetry_horizontal":
        refl_r = (2.0 * axis_val - subject_r) if axis_val is not None else subject_r
        return abs(refl_r - target_r) + abs(subject_c - target_c)
    else:  # socket_coverage, spatial_contact
        return abs(subject_r - target_r) + abs(subject_c - target_c)
def compare_invariants_across_levels(
    prev_invariants: list[DiscoveredInvariant],
    curr_invariants: list[DiscoveredInvariant],
) -> dict[str, list[DiscoveredInvariant]]:
    """Detect persistent, new, and disappeared invariants between levels."""
    import re
    from collections import defaultdict

    def _sig(inv: DiscoveredInvariant) -> str:
        parts = [inv.invariant_type]
        if inv.axis_coordinate is not None:
            parts.append(f"axis_{round(inv.axis_coordinate, 1)}")
        if inv.target_position is not None:
            parts.append(f"tgt_{round(inv.target_position[0], 1)}_{round(inv.target_position[1], 1)}")
        norm_desc = re.sub(r"obj_\w+", "[OBJ]", inv.description)
        parts.append(norm_desc)
        return ":".join(parts)

    prev_by_sig: dict[str, list[DiscoveredInvariant]] = defaultdict(list)
    for inv in prev_invariants:
        prev_by_sig[_sig(inv)].append(inv)

    curr_by_sig: dict[str, list[DiscoveredInvariant]] = defaultdict(list)
    for inv in curr_invariants:
        curr_by_sig[_sig(inv)].append(inv)

    persistent: list[DiscoveredInvariant] = []
    new_invs: list[DiscoveredInvariant] = []
    disappeared: list[DiscoveredInvariant] = []

    all_sigs = set(prev_by_sig.keys()) | set(curr_by_sig.keys())
    for sig in all_sigs:
        p_list = list(prev_by_sig.get(sig, []))
        c_list = list(curr_by_sig.get(sig, []))
        common = min(len(p_list), len(c_list))
        persistent.extend(c_list[:common])
        if len(c_list) > common:
            new_invs.extend(c_list[common:])
        if len(p_list) > common:
            disappeared.extend(p_list[common:])

    return {
        "persistent": persistent,
        "new": new_invs,
        "disappeared": disappeared,
    }
