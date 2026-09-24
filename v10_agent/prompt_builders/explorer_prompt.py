"""Explorer Prompt Builder (Call 1 Family).

Constructs prompts strictly quarantined from game goals and trajectory planning.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from v10_agent.memory_contours import ProbeRecord
from v10_agent.planning_set import PlanningSet

EXPLORER_SYSTEM_PROMPT = """\
You are an empirical environment analyzer for a 2D grid puzzle.
Think step by step and carefully analyze all probe observations and physical differences before providing the specification.
Your ONLY task is to deduce physical rules, action effects, and object relationships from probe history and visual structure.

RULES:
1. Describe ONLY observed physical effects (e.g., "moves entity dy=3", "toggles control").
2. NEVER guess the puzzle goal, winning conditions, or final state.
3. NEVER output solution plans or action sequences.
4. If a probe failed, do not overgeneralize; just note the lack of effect.
5. In addition to action effects, describe any observed:
   - SYMMETRY: Axes of symmetry or mirror relationships between objects
   - PERIODICITY: Repeating patterns in placement, spacing, or colors
   - TOPOLOGY: Which objects are adjacent, contained, or connected
   - COLOR PATTERNS: How colors are distributed and whether they suggest roles
6. DUALVIEW GROUNDING & PARSER OVER-SEGMENTATION:
   - Raw frame image is Ground Truth for visual patterns, compound multi-color tiles, frames, and composite glyphs.
   - The annotated frame and object index fragment compound multi-color entities into monochrome parts. Reconcile adjacent pieces into unified composite macro-objects.
7. COORDINATE CONVENTION:
   - All spatial coordinates are strictly Cartesian: x = column (0 <= x < width), y = row (0 <= y < height).
   - Coordinate action format: action6(x=col, y=row).
8. Output format: enclose your findings in a ```json ... ``` block.
"""


def _format_probe_data(action_id: str, action_data: dict[str, Any] | None) -> str:
    """Format probe action data in local workspace coordinates."""
    if not action_data:
        return "()"
    if action_id == "ACTION6":
        lx = action_data.get("local_x", action_data.get("x"))
        ly = action_data.get("local_y", action_data.get("y"))
        crop = action_data.get("crop_offset", 0)
        if "local_x" not in action_data and crop > 0 and lx is not None:
            lx = int(lx) - crop
        if "local_y" not in action_data and crop > 0 and ly is not None:
            ly = int(ly) - crop
        if lx is not None and ly is not None:
            return f"(x={lx}, y={ly})"
    filtered = {k: v for k, v in action_data.items() if k not in ("crop_offset", "local_x", "local_y")}
    return f"({', '.join(f'{k}={v}' for k, v in filtered.items())})" if filtered else "()"


def build_explorer_prompts(
    planning_set: PlanningSet,
    probe_history: Sequence[ProbeRecord] | None = None,
    has_image: bool = True,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for the Explorer Agent using lean formatting."""
    grid_h, grid_w = planning_set.grid_dims
    grid_hash = planning_set.grid_hash if planning_set.grid_hash else "unknown"

    # 1. Compact Object Index (one line per object with bbox, color, area, shape)
    sorted_objs = sorted(planning_set.objects, key=lambda o: o.area, reverse=True)[:20]
    obj_lines = []
    for obj in sorted_objs:
        alias = planning_set.object_real_to_alias.get(obj.id, obj.id)
        b = obj.bbox
        shape = getattr(obj, "shape_type", "entity")
        line = f"- {alias}: c={obj.color}, bbox=[{b.min_row},{b.min_col},{b.max_row},{b.max_col}], area={obj.area}, {shape}"
        if max(b.height, b.width) <= 12 and getattr(obj, "compact_ascii", None):
            line += f'; ascii="{"/".join(obj.compact_ascii)}"'
        obj_lines.append(line)
    object_index_text = "\n".join(obj_lines) if obj_lines else "none"

    # 2. Prior Probe History (concise log)
    probe_lines = []
    for idx, p in enumerate((probe_history or [])[-10:], 1):
        data_str = _format_probe_data(p.action_id, p.action_data)
        probe_lines.append(f"#{idx} {p.action_id}{data_str} -> {p.observed_effect} (confidence={p.confidence:.1f})")
    probe_log_text = "\n".join(probe_lines) if probe_lines else "none"

    # 3. Available Actions
    actions_list = [str(a).upper() for a in planning_set.allowed_action_ids if str(a).upper() not in ("RESET", "ACTION7")]
    actions_text = ", ".join(actions_list) if actions_list else "none"

    image_note = (
        "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations. "
        "The annotated frame's labels are the aliases used in the object index below.\n\n"
        if has_image
        else ""
    )

    user_text = f"""\
{image_note}GRID: {grid_h}x{grid_w}, hash {grid_hash}
AVAILABLE ACTIONS: {actions_text}

OBJECT INDEX:
{object_index_text}

PRIOR PROBE HISTORY (format: action6(x=col, y=row)):
{probe_log_text}

Analyze the available actions, object visual structures, and prior probe observations to formulate physical rules and invariants.
Output format:
```json
{{
  "researched_actions": [
    {{
      "action_id": "ACTION1",
      "effect_summary": "observed or hypothesized physical effect",
      "confidence": 0.8
    }}
  ],
  "invariants": [
    "objects move rigidly without deformation"
  ],
  "structural_notes": [
    "symmetry across vertical axis",
    "repeating color pattern"
  ]
}}
```
"""
    return EXPLORER_SYSTEM_PROMPT, user_text


COORDINATE_HYPOTHESIS_SYSTEM_PROMPT = """\
You are an exploratory reasoning agent for ARC-AGI-3 grid puzzles with unknown rules.
Think step by step and carefully reason about the 2D grid layout and detected visual objects to propose the most informative click target coordinates [x, y] for coordinate action (ACTION6) to uncover the mechanics of this puzzle.

COORDINATE CONVENTION (STRICT CARTESIAN):
- x is the HORIZONTAL coordinate (column index, 0 <= x < width).
- y is the VERTICAL coordinate (row index, 0 <= y < height).
- Click format is strictly [x, y] = [col, row]. Never invert x and y!

DUALVIEW GROUNDING & COMPOSITE OBJECTS:
- Raw image shows true composite visual objects (multi-color tiles, patterns, frames with inner markers).
- The detected_objects list splits multi-color patterns into monochrome pieces. Consider clicking the geometric center of composite multi-color structures seen in the raw frame!

CRITICAL CONSTRAINTS:
1. Propose between 2 and {num_hypotheses} distinct, high-priority click coordinates [x, y].
2. Prioritize clicking on:
   - Centroids of salient objects (interactive tokens, pieces, obstacles, targets) or geometric centers of composite multi-color tiles (patterned structures, frames, glyphs).
   - Distinctive corners or boundaries of objects.
   - Grid center or open background if no obvious interactive objects.
3. Coordinates MUST be integers satisfying 0 <= x < width and 0 <= y < height.
4. Output MUST be a single valid JSON object enclosed in ```json ... ``` with schema:
{
  "coordinate_hypotheses": [
    {
      "x": int,
      "y": int,
      "target_description": "short description of target object or location",
      "rationale": "why clicking here is informative"
    }
  ]
}
"""


def build_coordinate_hypothesis_prompt(
    planning_set: PlanningSet,
    num_hypotheses: int = 5,
    prior_tested_coords: Sequence[tuple[int, int]] | None = None,
    has_image: bool = True,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for Qwen ACTION6 coordinate hypothesis generation."""
    height, width = planning_set.grid_dims
    # Stratified sampling of objects across quadrants & area
    mid_r = height / 2.0
    mid_c = width / 2.0
    quads: dict[str, list[Any]] = {"TL": [], "TR": [], "BL": [], "BR": []}
    for obj in planning_set.objects:
        qr = "T" if obj.centroid.row < mid_r else "B"
        qc = "L" if obj.centroid.col < mid_c else "R"
        quads[qr + qc].append(obj)

    selected_objs: list[Any] = []
    # Largest objects overall
    for obj in sorted(planning_set.objects, key=lambda o: -o.area)[:6]:
        if obj not in selected_objs:
            selected_objs.append(obj)
    # Representative objects from each quadrant
    for q_objs in quads.values():
        for obj in sorted(q_objs, key=lambda o: -o.area)[:4]:
            if obj not in selected_objs:
                selected_objs.append(obj)

    objects_summary = []
    for obj in selected_objs[:20]:
        obj_dict: dict[str, Any] = {
            "id": obj.id,
            "alias": planning_set.object_real_to_alias.get(obj.id, obj.id),
            "color": obj.color,
            "area": obj.area,
            "centroid": {"x": int(round(obj.centroid.col)), "y": int(round(obj.centroid.row))},
            "bbox": {
                "min_x": obj.bbox.min_col,
                "min_y": obj.bbox.min_row,
                "max_x": obj.bbox.max_col,
                "max_y": obj.bbox.max_row,
            },
        }
        if getattr(obj, "children_ids", None):
            obj_dict["children_ids"] = obj.children_ids
            obj_dict["children_aliases"] = [planning_set.object_real_to_alias.get(cid, cid) for cid in obj.children_ids]
        if getattr(obj, "parent_id", None):
            obj_dict["parent_id"] = obj.parent_id
            obj_dict["parent_alias"] = planning_set.object_real_to_alias.get(obj.parent_id, obj.parent_id)
        objects_summary.append(obj_dict)

    candidates_summary = [
        {"x": c.x, "y": c.y, "label": c.label, "source": c.source_type}
        for c in planning_set.coordinate_candidates[:24]
    ]

    user_payload: dict[str, Any] = {
        "grid_dimensions": {"width": width, "height": height},
        "detected_objects": objects_summary,
        "salient_reference_points": candidates_summary,
    }
    inactive_notice = ""
    if prior_tested_coords:
        user_payload["previously_tested_inactive_coordinates"] = [
            {"x": x, "y": y} for x, y in prior_tested_coords
        ]
        inactive_notice = f"""
CRITICAL AVOIDANCE CONSTRAINT:
The {len(prior_tested_coords)} coordinates listed in 'previously_tested_inactive_coordinates' have already been clicked and produced NO EFFECT (0 pixel difference / inactive).
DO NOT propose these coordinates again! Select DIFFERENT salient objects, surrounding blocks, internal components, or boundaries."""

    image_note = (
        "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations.\n\n"
        if has_image
        else ""
    )

    user_text = f"""\
{image_note}Current Puzzle Grid & Object State:
{json.dumps(user_payload, indent=2)}
{inactive_notice}

You are solving an ARC-AGI-3 puzzle with unknown mechanics.
We have a grid of size {width}x{height} with the detected objects and candidate points listed above.
What coordinates [x, y] does it make sense to click with ACTION6 to uncover the game mechanics?
Propose between 2 and {num_hypotheses} distinct candidate click coordinates [x, y] (strictly within 0 <= x < {width}, 0 <= y < {height}).
Output format:
```json
{{
  "coordinate_hypotheses": [
    {{
      "x": 0,
      "y": 0,
      "target_description": "description",
      "rationale": "reason"
    }}
  ]
}}
```"""
    sys_prompt = COORDINATE_HYPOTHESIS_SYSTEM_PROMPT.replace("{num_hypotheses}", str(num_hypotheses))
    return sys_prompt, user_text


EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT = """\
You are an empirical kinematic analyzer for a 2D grid puzzle.
ACTIVE PROBING IS COMPLETE. Your task is to describe strictly what physically mutated or moved and what remained static during the probes.

CRITICAL CONSTRAINTS:
1. NO HYPOTHESIS PROPOSALS: Probing is already finished. Do NOT propose click coordinates [x, y]. Do NOT output coordinate_hypotheses.
2. NO ROLES OR SPECULATION: Do NOT invent roles, labels, or game interpretations (NO 'player', 'actor', 'mirror', 'goal', 'target', 'axis'). Let the solver decide game semantics.
3. ACTION AFFORDANCES & OBSERVATION:
   - For each confirmed action, classify its observable effect into one of 4 invariant classes:
     * KINEMATIC: spatial displacement (dy, dx) of rigid entities.
     * PALETTE_TRANSITION: in-place color change or tile state transition.
     * TOPOLOGY_MUTATION: entity appearance, disappearance, attachment, or detachment.
     * MODAL_SELECTION: active entity switch, focus indicator change, or mode toggle.
   - List which object aliases remained completely static across all probes.
   - Describe purely geometric layout facts (e.g. dimensions, vertical or horizontal lines, placement of clusters).
   - STRICT EMPIRICAL EVIDENCE: Do NOT invent unobserved physical barriers, walls, locks, or containment rules. If a parent object moved together with its children in the probe log, the parent object physically MOVED (do NOT declare it static!). State ONLY what is directly evidenced in the probe log.
4. Output MUST be a single valid JSON object enclosed in ```json ... ``` with schema:
{
  "action_affordances": [
    {
      "action_id": "ACTION1",
      "effect_class": "KINEMATIC",
      "parameters": {"affected_alias": "B", "dy": -3, "dx": 0},
      "coordination_notes": "synchronous movement of B and C"
    },
    {
      "action_id": "ACTION5",
      "effect_class": "MODAL_SELECTION",
      "parameters": {"active_entity_switched": true, "cycle_entities": ["axis", "shape"]},
      "coordination_notes": "toggles active entity focus"
    }
  ],
  "static_objects": [
    {"alias": "A", "description": "did not move during probes"}
  ],
  "structural_geometry": [
    "object A is a vertical line spanning rows 0-61 at cols 29-31"
  ]
}
"""


def build_explorer_synthesis_prompt(
    planning_set: PlanningSet,
    confirmed_actions: Any | None = None,
    unconfirmed_actions: Any | None = None,
    probe_history: Sequence[ProbeRecord] | None = None,
    has_image: bool = True,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for Explorer Level Synthesis (Phase 2)."""
    grid_h, grid_w = planning_set.grid_dims
    grid_hash = planning_set.grid_hash if planning_set.grid_hash else "unknown"

    # 1. Compact Object Index
    sorted_objs = sorted(planning_set.objects, key=lambda o: o.area, reverse=True)[:25]
    obj_lines = []
    for obj in sorted_objs:
        alias = planning_set.object_real_to_alias.get(obj.id, obj.id)
        b = obj.bbox
        shape = getattr(obj, "shape_type", "entity")
        rel_info = ""
        if getattr(obj, "children_ids", None):
            c_aliases = [planning_set.object_real_to_alias.get(cid, cid) for cid in obj.children_ids]
            rel_info = f", parent of: [{', '.join(c_aliases)}]"
        elif getattr(obj, "parent_id", None):
            p_alias = planning_set.object_real_to_alias.get(obj.parent_id, obj.parent_id)
            rel_info = f", child of: {p_alias}"
        line = f"- {alias} ({obj.id}): c={obj.color}, bbox=[{b.min_row},{b.min_col},{b.max_row},{b.max_col}], area={obj.area}, {shape}{rel_info}"
        if max(b.height, b.width) <= 12 and getattr(obj, "compact_ascii", None):
            line += f'; ascii="{"/".join(obj.compact_ascii)}"'
        obj_lines.append(line)
    object_index_text = "\n".join(obj_lines) if obj_lines else "none"

    # 2. Confirmed Action Effects (Physical Facts from Probing)
    confirmed_lines = []
    for act_id, eff in sorted((confirmed_actions or {}).items()):
        confirmed_lines.append(f"- {act_id}: {eff}")
    confirmed_text = "\n".join(confirmed_lines) if confirmed_lines else "none"

    # 3. Unconfirmed / Inactive Actions
    unconfirmed_lines = []
    for act_id, note in sorted((unconfirmed_actions or {}).items()):
        unconfirmed_lines.append(f"- {act_id}: inactive / {note}")
    unconfirmed_text = "\n".join(unconfirmed_lines) if unconfirmed_lines else "none"

    # 4. Probe Log (last 12 actions)
    probe_lines = []
    for idx, p in enumerate((probe_history or [])[-12:], 1):
        data_str = _format_probe_data(p.action_id, p.action_data)
        probe_lines.append(f"#{idx} {p.action_id}{data_str} -> {p.observed_effect}")
    probe_log_text = "\n".join(probe_lines) if probe_lines else "none"

    image_note = (
        "explorer_raw_frame.png is the pristine initial frame without annotations. "
        "explorer_annotated_frame.png shows object bounding boxes and their alias letters.\n\n"
        if has_image
        else ""
    )

    user_text = f"""\
{image_note}GRID DIMENSIONS: {grid_h}x{grid_w}, initial frame hash: {grid_hash}

OBJECT INDEX:
{object_index_text}

CONFIRMED ACTION EFFECTS (from empirical probe deltas, format: action6(x=col, y=row)):
{confirmed_text}

INACTIVE / ZERO-EFFECT ACTIONS:
{unconfirmed_text}

RECENT PROBE EXECUTION LOG (format: action6(x=col, y=row)):
{probe_log_text}

REMINDER: Active probing is complete. Do NOT propose coordinates or click hypotheses.
Do NOT invent game roles like 'actor', 'mirror', or 'goal'.
State purely which visual objects moved by what (dy, dx), whether they moved synchronously or oppositely, and which objects stayed completely static.
Output ONLY the JSON object.
"""
    return EXPLORER_LEVEL_SYNTHESIS_SYSTEM_PROMPT, user_text

