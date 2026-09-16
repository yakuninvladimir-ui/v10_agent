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
Think step by step and carefully analyze all probe observations and physical differences before providing the JSON specification.
Your ONLY task is to deduce physical rules, action effects, and coordinate affordances from probe history.

RULES:
1. Describe ONLY observed physical effects (e.g., "moves entity dy=3", "toggles control").
2. NEVER guess the puzzle goal, winning conditions, or final state.
3. NEVER output solution plans or action sequences.
4. If a probe failed, do not overgeneralize; just note the lack of effect.
5. Output MUST be strictly valid JSON matching the requested schema. No markdown outside the JSON block.
6. In addition to action effects, describe any observed:
   - SYMMETRY: Axes of symmetry or mirror relationships between objects
   - PERIODICITY: Repeating patterns in placement, spacing, or colors
   - TOPOLOGY: Which objects are adjacent, contained, or connected
   - COLOR PATTERNS: How colors are distributed and whether they suggest roles
"""


def build_explorer_prompts(
    planning_set: PlanningSet,
    probe_history: Sequence[ProbeRecord] | None = None,
    has_image: bool = True,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for the Explorer Agent."""
    sorted_objs = sorted(planning_set.objects, key=lambda o: o.area, reverse=True)[:16]
    objects_summary = [
        {
            "id": obj.id,
            "alias": planning_set.object_real_to_alias.get(obj.id, obj.id),
            "color": obj.color,
            "area": obj.area,
            "bbox": obj.bbox.to_dict(),
            "centroid": obj.centroid.to_dict(),
        }
        for obj in sorted_objs
    ]

    coords_summary = [
        {"id": c.candidate_id, "x": c.x, "y": c.y, "label": c.label, "source": c.source_type}
        for c in planning_set.coordinate_candidates[:24]
    ]

    probes_summary = [
        {
            "action": p.action_id,
            "data": p.action_data,
            "effect": p.observed_effect,
            "confidence": p.confidence,
        }
        for p in (probe_history or [])[-8:]
    ]

    user_payload = {
        "planning_set_id": planning_set.snapshot_id,
        "grid_hash": planning_set.grid_hash,
        "grid_dims": {"height": planning_set.grid_dims[0], "width": planning_set.grid_dims[1]},
        "available_actions": list(planning_set.allowed_action_ids),
        "planning_objects": objects_summary,
        "coordinate_candidates": coords_summary,
        "prior_probe_history": probes_summary,
    }

    image_note = (
        "explorer_raw_frame.png is the exact same frame as explorer_annotated_frame.png, but without object annotations.\n\n"
        if has_image
        else ""
    )

    user_text = f"""\
{image_note}Current Environment State:
{json.dumps(user_payload, indent=2)}

Analyze the available actions, objects, and prior probes to formulate a concise EnvironmentSpecification.
Output format:
```json
{{
  "schema_version": "v10.env_spec.1",
  "snapshot_hash": "{planning_set.grid_hash}",
  "planning_set_id": "{planning_set.snapshot_id}",
  "researched_actions": [
    {{
      "action_id": "ACTION1",
      "effect_summary": "description of observed or hypothesized physical effect",
      "supporting_evidence_ids": ["probe_0"],
      "confidence": 0.8,
      "contradicted": false
    }}
  ],
  "coordinate_affordances": [
    {{
      "coordinate_candidate_id": "coord_c_obj_0",
      "x": 10,
      "y": 12,
      "source": {{"type": "object_centroid", "object_id": "obj_0"}},
      "observed_effects": ["interaction_effect"],
      "confidence": 0.7
    }}
  ],
  "object_class_notes": [],
  "action_surface_notes": [],
  "invariants": ["object_identity_stable_under_ACTION2"]
}}
```
"""
    return EXPLORER_SYSTEM_PROMPT, user_text


COORDINATE_HYPOTHESIS_SYSTEM_PROMPT = """\
You are an exploratory reasoning agent for ARC-AGI-3 grid puzzles with unknown rules.
Think step by step and carefully reason about the 2D grid layout and detected visual objects to propose the most informative click target coordinates [x, y] for coordinate action (ACTION6) to uncover the mechanics of this puzzle.

CRITICAL CONSTRAINTS:
1. Propose between 2 and {num_hypotheses} distinct, high-priority click coordinates [x, y].
2. Prioritize clicking on:
   - Centroids of salient objects (interactive tokens, pieces, obstacles, targets).
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

    objects_summary = [
        {
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
        for obj in selected_objs[:20]
    ]

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
