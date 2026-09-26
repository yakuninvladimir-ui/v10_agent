"""Universal Invariant Module for Domain-General Discovery and Evaluation.

Extracts algebraic geometric invariants (axial/point symmetries, shape sockets,
contact, pattern matching) from the perception graph without game-specific heuristics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from v10_agent.action_semantics import replace_object_tokens
from v10_agent.memory_contours import CoreInvariantRegistry, EmpiricalInvariant
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
    confidence: float = 0.3

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
    confidence: float = 0.5

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
    confidence: float = 0.5

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
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "area_conservation",
            "subject": self.subject_id,
            "area": self.area,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SlidingKinematics:
    """Physics invariant: continuous translation along direction until obstacle or border collision."""
    subject_id: str
    direction: tuple[int, int]
    obstacle_id: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sliding_kinematics",
            "subject": self.subject_id,
            "direction": self.direction,
            "obstacle": self.obstacle_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SokobanPush:
    """Physics invariant: actor movement into adjacent entity displaces it in same direction."""
    actor_id: str
    pushed_id: str
    direction: tuple[int, int]
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sokoban_push",
            "actor": self.actor_id,
            "pushed": self.pushed_id,
            "direction": self.direction,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class MarkerCollection:
    """Task invariant: agent contact with target role entities removes them from state."""
    collector_id: str
    target_role_color: int
    target_id: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "marker_collection",
            "collector": self.collector_id,
            "target_role_color": self.target_role_color,
            "target_id": self.target_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ToggleTrigger:
    """Interaction invariant: agent visiting trigger alters remote barrier passability/color."""
    trigger_id: str
    barrier_id: str
    toggle_state: str = "open"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "toggle_trigger",
            "trigger": self.trigger_id,
            "barrier": self.barrier_id,
            "toggle_state": self.toggle_state,
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
            elongation_threshold = 1.5 + 0.5 * (min(grid_h, grid_w) / 64.0)
            for obj in objects:
                if obj.id not in (sub_id, tgt_id):
                    is_vert_elongated = (obj.height / max(1, obj.width) >= elongation_threshold) and (obj.height >= max(3, int(round(grid_h * 0.2))))
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
            elongation_threshold = 1.5 + 0.5 * (min(grid_h, grid_w) / 64.0)
            for obj in objects:
                if obj.id not in (sub_id, tgt_id):
                    is_horiz_elongated = (obj.width / max(1, obj.height) >= elongation_threshold) and (obj.width >= max(3, int(round(grid_w * 0.2))))
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
                confidence=0.3,
                description=f"Area conservation: {obj.id} maintains area {obj.area}",
            ))
            # Connected component conservation
            invariants.append(DiscoveredInvariant(
                invariant_type="connected_component_conservation",
                subject_id=obj.id, target_id=obj.id,
                confidence=0.3,
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

    def _sig(inv: DiscoveredInvariant) -> str:
        parts = [inv.invariant_type]
        if inv.axis_coordinate is not None:
            parts.append(f"axis_{round(inv.axis_coordinate, 1)}")
        if inv.target_position is not None:
            parts.append(f"tgt_{round(inv.target_position[0], 1)}_{round(inv.target_position[1], 1)}")
        norm_desc = replace_object_tokens(inv.description, lambda _token: "[OBJ]")
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


def propose_invariant_candidates(planning_set: PlanningSet) -> list[EmpiricalInvariant]:
    """Extract prioritized empirical invariant hypotheses starting with confidence=0.0."""
    candidates: list[EmpiricalInvariant] = []
    if not planning_set or not planning_set.objects:
        return candidates

    # 1. Global Core Game Laws (conserve area and connectivity)
    candidates.append(
        EmpiricalInvariant(
            invariant_id="emp_area_conservation_global",
            invariant_type="area_conservation",
            abstract_description="Objects conserve their pixel area across transitions",
            subject_pattern="all_objects",
            expected_value="conserved",
            scope="CORE_GAME_LAW",
            confidence=0.0,
        )
    )
    candidates.append(
        EmpiricalInvariant(
            invariant_id="emp_topology_conservation_global",
            invariant_type="connected_component_conservation",
            abstract_description="Objects retain topological connectivity across transitions",
            subject_pattern="all_objects",
            expected_value="conserved",
            scope="CORE_GAME_LAW",
            confidence=0.0,
        )
    )

    # 2. Per-object area and topology conservation
    for obj in planning_set.objects:
        if obj.area > 0 and obj.color != 0:
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_area_{obj.id}",
                    invariant_type="area_conservation",
                    abstract_description=f"Object {obj.id} maintains area {obj.area}",
                    subject_pattern=f"id=={obj.id}",
                    expected_value=obj.area,
                    scope="LEVEL_SPECIFIC",
                    confidence=0.0,
                )
            )
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_topo_{obj.id}",
                    invariant_type="connected_component_conservation",
                    abstract_description=f"Object {obj.id} maintains topological unity",
                    subject_pattern=f"id=={obj.id}",
                    expected_value=1,
                    scope="LEVEL_SPECIFIC",
                    confidence=0.0,
                )
            )

    # 3. Discovered relational symmetries and socket coverage
    discovered = discover_invariants(planning_set)
    for inv in discovered:
        if inv.invariant_type in ("axial_symmetry_vertical", "axial_symmetry_horizontal"):
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_sym_{inv.subject_id}_{inv.target_id}_{inv.invariant_type}",
                    invariant_type=inv.invariant_type,
                    abstract_description=inv.description,
                    subject_pattern=f"pair=={inv.subject_id}:{inv.target_id}",
                    expected_value={"axis_coord": inv.axis_coordinate, "target_pos": inv.target_position},
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )
        elif inv.invariant_type == "socket_coverage":
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_socket_{inv.subject_id}_{inv.target_id}",
                    invariant_type="socket_coverage",
                    abstract_description=inv.description,
                    subject_pattern=f"pair=={inv.subject_id}:{inv.target_id}",
                    expected_value=inv.target_position,
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )

    # 4. Contact triggers
    for rel in getattr(planning_set, "relations", []):
        if rel.relation_type in ("touches", "adjacent_to"):
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_contact_{rel.subject_id}_{rel.target_id}",
                    invariant_type="contact_trigger",
                    abstract_description=f"Contact trigger between {rel.subject_id} and {rel.target_id}",
                    subject_pattern=f"pair=={rel.subject_id}:{rel.target_id}",
                    expected_value="contact",
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )

    # 5. Sliding kinematics and marker collection candidates
    for obj in planning_set.objects:
        if obj.area > 0 and obj.color != 0:
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_slide_{obj.id}",
                    invariant_type="sliding_kinematics",
                    abstract_description=f"Object {obj.id} undergoes sliding translation until obstacle",
                    subject_pattern=f"id=={obj.id}",
                    expected_value="sliding",
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )
            if obj.area <= 4:
                candidates.append(
                    EmpiricalInvariant(
                        invariant_id=f"emp_collect_{obj.id}",
                        invariant_type="marker_collection",
                        abstract_description=f"Consumable target marker {obj.id}",
                        subject_pattern=f"id=={obj.id}",
                        expected_value="collected",
                        scope="DOMAIN_PATTERN",
                        confidence=0.0,
                    )
                )

    # 6. Push mechanics and toggle trigger candidates for object pairs
    for rel in getattr(planning_set, "relations", []):
        if rel.relation_type in ("touches", "adjacent_to"):
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_push_{rel.subject_id}_{rel.target_id}",
                    invariant_type="sokoban_push",
                    abstract_description=f"Pushing interaction between {rel.subject_id} and {rel.target_id}",
                    subject_pattern=f"pair=={rel.subject_id}:{rel.target_id}",
                    expected_value="push",
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )
            candidates.append(
                EmpiricalInvariant(
                    invariant_id=f"emp_toggle_{rel.subject_id}_{rel.target_id}",
                    invariant_type="toggle_trigger",
                    abstract_description=f"Switch trigger {rel.subject_id} toggles {rel.target_id}",
                    subject_pattern=f"pair=={rel.subject_id}:{rel.target_id}",
                    expected_value="toggled",
                    scope="DOMAIN_PATTERN",
                    confidence=0.0,
                )
            )

    # Deduplicate candidates by invariant_id
    unique_candidates: list[EmpiricalInvariant] = []
    seen: set[str] = set()
    for c in candidates:
        if c.invariant_id not in seen:
            seen.add(c.invariant_id)
            unique_candidates.append(c)

    return unique_candidates


def extract_observed_value_for(
    inv: EmpiricalInvariant,
    before_snapshot: Any,
    after_snapshot: Any,
) -> Any:
    """Extract observed metric or property corresponding to the invariant."""
    if not before_snapshot or not after_snapshot:
        return None

    before_objects = getattr(before_snapshot, "objects", [])
    after_objects = getattr(after_snapshot, "objects", [])

    if not before_objects or not after_objects:
        return None

    # Global invariants
    if inv.subject_pattern == "all_objects":
        if inv.invariant_type == "area_conservation":
            conserved = True
            for b_obj in before_objects:
                if b_obj.color == 0 or b_obj.area == 0:
                    continue
                matched = next((a for a in after_objects if a.id == b_obj.id), None)
                if matched is None:
                    best_d = float("inf")
                    for a in after_objects:
                        if a.color == b_obj.color:
                            d = (b_obj.centroid.row - a.centroid.row) ** 2 + (b_obj.centroid.col - a.centroid.col) ** 2
                            if d < best_d:
                                best_d = d
                                matched = a
                if matched is not None:
                    if matched.area != b_obj.area:
                        conserved = False
                        break
                else:
                    # Object disappeared
                    conserved = False
                    break
            return "conserved" if conserved else "violated"

        elif inv.invariant_type == "connected_component_conservation":
            return "conserved"

    # Specific object invariants: id==<oid>
    if inv.subject_pattern.startswith("id=="):
        target_id = inv.subject_pattern.split("==", 1)[1]
        b_obj = next((o for o in before_objects if o.id == target_id), None)
        if b_obj is None:
            return None

        a_obj = next((o for o in after_objects if o.id == target_id), None)
        if a_obj is None:
            best_d = float("inf")
            for a in after_objects:
                if a.color == b_obj.color:
                    d = (b_obj.centroid.row - a.centroid.row) ** 2 + (b_obj.centroid.col - a.centroid.col) ** 2
                    if d < best_d:
                        best_d = d
                        a_obj = a

        if inv.invariant_type == "area_conservation":
            return a_obj.area if a_obj is not None else 0
        elif inv.invariant_type == "connected_component_conservation":
            return 1 if a_obj is not None else 0
        elif inv.invariant_type == "sliding_kinematics":
            if a_obj is not None:
                dy = abs(a_obj.centroid.row - b_obj.centroid.row)
                dx = abs(a_obj.centroid.col - b_obj.centroid.col)
                disp = max(dy, dx)
                if disp > 1.0:
                    return "sliding"
                elif disp > 0.0:
                    return "step_moved"
                else:
                    return "stationary"
            return None
        elif inv.invariant_type == "marker_collection":
            if a_obj is None or a_obj.area < b_obj.area:
                return "collected"
            return "preserved"

    # Pair invariants: pair==<sub_id>:<tgt_id>
    if inv.subject_pattern.startswith("pair=="):
        pair_part = inv.subject_pattern.split("==", 1)[1]
        if ":" not in pair_part:
            return None
        sub_id, tgt_id = pair_part.split(":", 1)
        sub_after = next((o for o in after_objects if o.id == sub_id), None)
        tgt_after = next((o for o in after_objects if o.id == tgt_id), None)

        if inv.invariant_type == "contact_trigger":
            if sub_after and tgt_after:
                dist = abs(sub_after.centroid.row - tgt_after.centroid.row) + abs(sub_after.centroid.col - tgt_after.centroid.col)
                if dist <= 1.5:
                    return "contact"
            return None

        elif inv.invariant_type == "sokoban_push":
            sub_before = next((o for o in before_objects if o.id == sub_id), None)
            tgt_before = next((o for o in before_objects if o.id == tgt_id), None)
            if sub_before and sub_after and tgt_before and tgt_after:
                s_dy = sub_after.centroid.row - sub_before.centroid.row
                s_dx = sub_after.centroid.col - sub_before.centroid.col
                t_dy = tgt_after.centroid.row - tgt_before.centroid.row
                t_dx = tgt_after.centroid.col - tgt_before.centroid.col
                if (abs(s_dy) > 0 or abs(s_dx) > 0) and (abs(t_dy) > 0 or abs(t_dx) > 0):
                    if (s_dy * t_dy >= 0) and (s_dx * t_dx >= 0):
                        return "push"
            return "no_push"

        elif inv.invariant_type == "toggle_trigger":
            tgt_before = next((o for o in before_objects if o.id == tgt_id), None)
            if sub_after and tgt_before:
                if tgt_after is None or tgt_after.area != tgt_before.area or tgt_after.color != tgt_before.color:
                    return "toggled"
            return "untoggled"

        elif inv.invariant_type == "socket_coverage":
            if sub_after:
                if tgt_after:
                    dist = abs(sub_after.centroid.row - tgt_after.centroid.row) + abs(sub_after.centroid.col - tgt_after.centroid.col)
                    if dist <= 0.5:
                        return (round(sub_after.centroid.row, 1), round(sub_after.centroid.col, 1))
                elif isinstance(inv.expected_value, (list, tuple)):
                    exp_r, exp_c = inv.expected_value
                    dist = abs(sub_after.centroid.row - exp_r) + abs(sub_after.centroid.col - exp_c)
                    if dist <= 0.5:
                        return (round(sub_after.centroid.row, 1), round(sub_after.centroid.col, 1))
            return None

        elif inv.invariant_type in ("axial_symmetry_vertical", "axial_symmetry_horizontal"):
            if sub_after and tgt_after:
                if isinstance(inv.expected_value, dict):
                    axis_coord = inv.expected_value.get("axis_coord")
                    if axis_coord is not None:
                        d = compute_invariant_distance(
                            sub_after.centroid.row, sub_after.centroid.col,
                            axis_coord,
                            tgt_after.centroid.row, tgt_after.centroid.col,
                            inv.invariant_type,
                        )
                        if d <= 1.0:
                            return "symmetric"
            return None

    return None


def values_match(expected: Any, observed: Any, invariant_type: str) -> bool:
    """Check whether observed property matches the expected invariant value."""
    if observed is None:
        return False
    if invariant_type == "area_conservation":
        return expected == observed
    if invariant_type == "connected_component_conservation":
        return expected == observed
    if invariant_type in ("axial_symmetry_vertical", "axial_symmetry_horizontal"):
        return observed == "symmetric" or expected == observed
    if invariant_type == "socket_coverage":
        if isinstance(expected, (tuple, list)) and isinstance(observed, (tuple, list)):
            return abs(expected[0] - observed[0]) <= 0.5 and abs(expected[1] - observed[1]) <= 0.5
        return expected == observed
    if invariant_type == "contact_trigger":
        return expected == observed
    if invariant_type == "sliding_kinematics":
        return observed == "sliding" or expected == observed
    if invariant_type == "sokoban_push":
        return observed == "push" or expected == observed
    if invariant_type == "marker_collection":
        return observed == "collected" or expected == observed
    if invariant_type == "toggle_trigger":
        return observed == "toggled" or expected == observed
    return expected == observed


def evaluate_invariant_after_action(
    registry: CoreInvariantRegistry,
    before_snapshot: Any,
    after_snapshot: Any,
    planning_set: PlanningSet | None = None,
    action_id: str | int = "",
    level_index: int = 0,
) -> None:
    """Evaluate empirical invariants after an action transition.
    
    Observes whether candidate invariants hold or are contradicted.
    Active candidates start at confidence 0.0 and gain confidence upon confirmation.
    """
    if not registry or before_snapshot is None or after_snapshot is None:
        return

    # Normalize snapshots if grid dicts were passed
    if isinstance(before_snapshot, dict):
        b_grid = before_snapshot.get("grid")
        if b_grid is not None:
            from v10_agent.arga_lite import extract_arga_snapshot
            before_snapshot = extract_arga_snapshot(b_grid)
        else:
            return

    if isinstance(after_snapshot, dict):
        a_grid = after_snapshot.get("grid")
        if a_grid is not None:
            from v10_agent.arga_lite import extract_arga_snapshot
            after_snapshot = extract_arga_snapshot(a_grid)
        else:
            return

    for inv in list(registry.invariants):
        # Skip completely dead/falsified candidates that have 0 confidence
        if inv.times_falsified > 0 and inv.confidence == 0.0:
            continue

        observed = extract_observed_value_for(inv, before_snapshot, after_snapshot)
        if observed is None:
            continue

        if values_match(inv.expected_value, observed, inv.invariant_type):
            inv.confirm(level_index, observed)
        else:
            inv.falsify(level_index, observed)

    registry._prune()

