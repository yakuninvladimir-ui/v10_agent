"""Solver Prompt Builder (Call 3 Family).

Constructs prompts strictly quarantined from Python source code, syntax errors, and tracebacks.
"""

from __future__ import annotations

import json
from typing import Any

from v10_agent.brusentsov_logic import Ternary
from v10_agent.memory_contours import EpistemicMemory
from v10_agent.planning_set import PlanningSet

SOLVER_SYSTEM_PROMPT = """\
You are an expert puzzle solver for 2D grid environments.
Think step by step and perform thorough geometric, topological, and invariant analysis of the visual grid and object affordances before proposing trajectories.
Given the current state, available DSL functions, and past failed attempts, your task is to deduce the underlying geometric/topological invariants and propose solution trajectories.

RULES:
1. Base your reasoning ONLY on the provided empirical facts, object relations, and past trial feedback.
2. All object arguments MUST strictly use IDs from the provided `planning_objects`. Do not invent IDs.
3. Check `past_failed_sequences` and `structured_failures`. DO NOT repeat them. Formulate alternative hypotheses.
4. Respect grid boundaries and object freedom of motion limits.
5. Provide up to 4 distinct candidate trajectories.
6. TRAJECTORY HORIZON & REPETITION SYNTAX (count=N):
   - Level solutions typically require extended trajectories (often 20 to 30 sequential actions).
   - Use the repetition parameter `count=N` to repeat an action N times cleanly: e.g. `action1(count=15)` executes 15 consecutive moves (up to 30).
   - Do not artificially truncate to micro-probes; plan the full sequence needed to reach the target winning configuration.

TERNARY EVALUATION SEMANTICS:
Your trajectories will be evaluated step-by-step using Brusentsov ternary logic:
- TRUE (FOLLOW): Step achieved expected physical effect -> trajectory continues.
- IRRELEVANT (OMIT): No contradiction, but expected effect not observed -> trajectory paused.
- FALSE (NULL): Physical contradiction detected (wall, collision, boundary blockage) -> trajectory permanently terminated.

Design complete, goal-oriented trajectories aimed directly at achieving the level win condition from attempt 1. Do not artificially truncate to micro-probes; plan the full sequence needed to reach the target winning configuration.

OUTPUT FORMAT:
You must structure your response using the following XML tags:

<invariant_analysis>
1. What is the likely goal of this level? (Cover targets / reach position / sort / align / ...)
2. Which confirmed invariants apply here? (list from game model)
3. What is NEW or DIFFERENT about this level vs previous ones?
4. Strategy for this level:
</invariant_analysis>

<trajectory_1>
[Full trajectory plan for candidate 1 using DSL function calls, e.g. action1(count=12), action6(x=5, y=10), action2(count=8)]
[Optional `EXPECT: prop=val` clauses allow the Brusentsov judge to verify step consequences.]
</trajectory_1>

<trajectory_2>
[Alternative full trajectory plan for candidate 2]
</trajectory_2>

<trajectory_3>
[Alternative full trajectory plan for candidate 3]
</trajectory_3>

<trajectory_4>
[Alternative full trajectory plan for candidate 4]
</trajectory_4>
"""


def _build_phase_instruction(level_index: int, total_levels: int = 6) -> str:
    """Build progressive strategic instructions adapting to game learning phases."""
    if level_index <= 1:
        return (
            "CURRENT PHASE: INITIAL EXPLORATION PHASE (Level index <= 1)\n"
            "Aim to generate complete winning trajectories from the very first attempt.\n"
            "Do not artificially truncate your plans into short probes or incremental tests. "
            "Formulate complete multi-step candidates that transform the initial state directly into the winning configuration."
        )
    elif level_index <= 3:
        return (
            "CURRENT PHASE: PROGRESSIVE CONFIRMATION PHASE (Level index 2-3)\n"
            "Apply confirmed invariants from the game model. "
            "Formulate complete winning plans while remaining alert to level-specific variations, new colors, or obstacles. "
            "If an invariant is challenged by the environment, adapt your hypothesis."
        )
    else:
        return (
            "CURRENT PHASE: EXPLOITATION PHASE (Level index >= 4)\n"
            "Execute optimal trajectories based on synthesized domain invariants across earlier levels to achieve direct level victory. "
            "Check for any new colors, shapes, or obstacles introduced at higher difficulty, and navigate around them."
        )


def build_solver_prompts(
    manifest: dict[str, Any],
    planning_set: PlanningSet,
    epistemic_memory: EpistemicMemory | None = None,
    action_budget: int = 50,
    game_memory: Any | None = None,
    has_image: bool = True,
    level_index: int | None = None,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for the Solver Agent."""
    functions_summary = manifest.get("functions", [])
    grid_h, grid_w = planning_set.grid_dims

    # Determine step unit size from confirmed action effects (e.g. dy=3, dx=0 -> 3 pixels per step)
    step_size_pixels = 1
    if game_memory is not None and getattr(game_memory, "confirmed_action_effects", None):
        import re
        for eff in game_memory.confirmed_action_effects.values():
            nums = [abs(int(x)) for x in re.findall(r'd[yx]=([+-]?\d+)', eff)]
            max_num = max(nums) if nums else 0
            if max_num > 0:
                step_size_pixels = max_num
                break

    confirmed_actors = getattr(game_memory, "confirmed_actors", set()) if game_memory else set()

    def salience_key(o: Any) -> tuple[int, int, int]:
        is_actor = 0 if (confirmed_actors and o.id in confirmed_actors) else 1
        # Substantive objects (area >= 4) first, then small components (area > 1), then single pixels
        sub_tier = 0 if o.area >= 4 else (1 if o.area > 1 else 2)
        return (is_actor, sub_tier, -o.area)

    mid_r = grid_h / 2.0
    mid_c = grid_w / 2.0
    # Aggregate small marker dots (area <= 2) that share the same color into composite indicator groups
    small_dots_by_color: dict[int, list[Any]] = {}
    substantive_objects: list[Any] = []
    for obj in planning_set.objects:
        if obj.area <= 2 and (not confirmed_actors or obj.id not in confirmed_actors):
            small_dots_by_color.setdefault(obj.color, []).append(obj)
        else:
            substantive_objects.append(obj)

    aggregated_marker_groups: list[dict[str, Any]] = []
    for c, dots in small_dots_by_color.items():
        if len(dots) >= 4:
            min_r = min(d.bbox.min_row for d in dots)
            max_r = max(d.bbox.max_row for d in dots)
            min_c = min(d.bbox.min_col for d in dots)
            max_c = max(d.bbox.max_col for d in dots)
            aggregated_marker_groups.append({
                "id": f"marker_group_color_{c}",
                "alias": f"marker_dots_c{c}",
                "role": "indicator_marker_pattern",
                "color": c,
                "dot_count": len(dots),
                "bounding_extent": {"min_row": min_r, "max_row": max_r, "min_col": min_c, "max_col": max_c},
                "sample_positions": [(d.centroid.row, d.centroid.col) for d in dots[:8]],
                "description": f"Group of {len(dots)} marker dots (color {c}) functioning as active indicators or visual focus markers.",
            })
        else:
            substantive_objects.extend(dots)

    quads: dict[str, list[Any]] = {"TL": [], "TR": [], "BL": [], "BR": []}
    for obj in substantive_objects:
        qr = "T" if obj.centroid.row < mid_r else "B"
        qc = "L" if obj.centroid.col < mid_c else "R"
        quads[qr + qc].append(obj)

    sorted_objs: list[Any] = []
    # Confirmed actors and largest objects first (up to 16 substantive objects)
    for obj in sorted(substantive_objects, key=salience_key)[:16]:
        if obj not in sorted_objs:
            sorted_objs.append(obj)
    # Representative objects from each quadrant
    for q_objs in quads.values():
        for obj in sorted(q_objs, key=salience_key)[:10]:
            if obj not in sorted_objs:
                sorted_objs.append(obj)
    objects_summary: list[dict[str, Any]] = []
    # Include aggregated marker groups first as context notes
    for mg in aggregated_marker_groups:
        objects_summary.append(mg)

    ascii_assigned = 0
    for rank, obj in enumerate(sorted_objs):
        d: dict[str, Any] = {
            "id": obj.id,
            "alias": planning_set.object_real_to_alias.get(obj.id, obj.id),
            "role": getattr(obj, "role", "generic_entity"),
            "color": obj.color,
            "area": obj.area,
            "width": obj.width,
            "height": obj.height,
            "shape_type": getattr(obj, "shape_type", "compound_shape"),
            "shape_signature": getattr(obj, "shape_signature", ""),
            "bbox": obj.bbox.to_dict(),
            "centroid": obj.centroid.to_dict(),
        }
        if getattr(obj, "normalized_centroid", None) and obj.normalized_centroid != (0.0, 0.0):
            d["normalized_centroid"] = {
                "row": obj.normalized_centroid[0],
                "col": obj.normalized_centroid[1],
            }
        if getattr(obj, "chiral_features", None):
            d["chiral_features"] = dict(obj.chiral_features)
        if getattr(obj, "multi_dir_relative", None):
            d["directional_projections"] = dict(obj.multi_dir_relative)

        if grid_h > 0 and grid_w > 0:
            up_px = obj.bbox.min_row
            down_px = max(0, (grid_h - 1) - obj.bbox.max_row)
            left_px = obj.bbox.min_col
            right_px = max(0, (grid_w - 1) - obj.bbox.max_col)
            d["freedom_of_motion"] = {
                "boundary_distance_only": {
                    "up_to_border": up_px // step_size_pixels,
                    "down_to_border": down_px // step_size_pixels,
                    "left_to_border": left_px // step_size_pixels,
                    "right_to_border": right_px // step_size_pixels,
                },
                "step_size_pixels": step_size_pixels,
                "WARNING": (
                    "These are distances to GRID EDGES only. Internal walls and obstacles "
                    "WILL block motion earlier. Actual free steps are likely FEWER."
                ),
            }
        # Only include ASCII art for top 12 substantive objects (area > 1), strictly excluding single pixels
        if ascii_assigned < 12 and getattr(obj, "area", 0) > 1 and getattr(obj, "compact_ascii", None):
            d["compact_ascii"] = list(obj.compact_ascii)
            ascii_assigned += 1
        objects_summary.append(d)

    def relation_priority(r: Any) -> tuple[int, int, float]:
        sub = planning_set.get_object(r.subject_id)
        tgt = planning_set.get_object(r.target_id)
        sub_area = sub.area if sub else 0
        tgt_area = tgt.area if tgt else 0

        # Tier 0: Both substantive objects (area >= 4)
        # Tier 1: One substantive object (area >= 4)
        # Tier 2: Minor / noise objects
        if sub_area >= 4 and tgt_area >= 4:
            tier = 0
        elif sub_area >= 4 or tgt_area >= 4:
            tier = 1
        else:
            tier = 2

        type_weights = {
            "identical_shape": 0,
            "chiral_mirror_h": 1,
            "chiral_mirror_v": 1,
            "symmetric_axis_of": 2,
            "mirrored_across_axis": 2,
            "target_is_below": 3,
            "target_is_above": 3,
            "target_is_to_the_right": 3,
            "target_is_to_the_left": 3,
            "contains": 4,
            "touches": 5,
            "aligned_v": 6,
            "aligned_h": 6,
            "distance": 7,
        }
        type_rank = type_weights.get(r.relation_type, 10)
        dist = r.metric_value if r.metric_value is not None else 999.0
        return (tier, type_rank, dist)

    sorted_relations = sorted(planning_set.relations, key=relation_priority)
    relations_summary: list[dict[str, Any]] = []
    for rel in sorted_relations:
        sub = planning_set.get_object(rel.subject_id)
        tgt = planning_set.get_object(rel.target_id)
        sub_area = sub.area if sub else 0
        tgt_area = tgt.area if tgt else 0
        sub_color = sub.color if sub else -1
        tgt_color = tgt.color if tgt else -1

        # Suppress directional navigation relations involving noise dots (area < 4) or background/hole pixels
        if "target_is_" in rel.relation_type and (sub_area < 4 or tgt_area < 4 or sub_color == 0 or tgt_color == 0):
            continue

        r_dict = rel.to_dict()
        if "target_is_" in rel.relation_type and rel.metric_value is not None:
            r_dict["steps_required"] = max(1, round(rel.metric_value / step_size_pixels))
        relations_summary.append(r_dict)
        if len(relations_summary) >= 50:
            break

    confirmed_effective = []
    interactive_coords: set[tuple[int, int]] = set()
    non_interactive_coords: set[tuple[int, int]] = set()
    if epistemic_memory:
        for j in epistemic_memory.judgments:
            v = getattr(j, "ternary_verdict", None)
            v_name = str(getattr(v, "name", getattr(v, "value", str(v))))
            is_true = (v == Ternary.TRUE) or ("TRUE" in v_name) or ("FOLLOW" in v_name) or bool(getattr(j, "is_effective", False))
            is_irr = (v == Ternary.IRRELEVANT) or ("IRRELEVANT" in v_name) or ("OMIT" in v_name)
            act = getattr(j, "action_dict", {}) or {}
            data = act.get("data", {}) if isinstance(act, dict) else {}
            if "x" in data and "y" in data:
                try:
                    coord = (int(data["x"]), int(data["y"]))
                    if is_true:
                        interactive_coords.add(coord)
                    elif is_irr:
                        non_interactive_coords.add(coord)
                except (ValueError, TypeError):
                    pass
            if is_true:
                confirmed_effective.append(j.to_dict())

    trial_summary: dict[str, Any] = {
        "failed_action_sequences": list(epistemic_memory.severed_null_signatures) if epistemic_memory else [],
        "partially_effective_sequences": [b.signature_id for b in (epistemic_memory.live_omit_branches if epistemic_memory else [])],
        "recent_action_outcomes": [j.to_dict() for j in (epistemic_memory.judgments[-10:] if epistemic_memory else [])],
    }
    if confirmed_effective:
        trial_summary["actions_with_observed_effects"] = confirmed_effective[-10:]

    if epistemic_memory is not None and hasattr(epistemic_memory, "format_scratchpad_context"):
        scratchpad_text = epistemic_memory.format_scratchpad_context()
        if scratchpad_text:
            trial_summary["current_level_scratchpad"] = scratchpad_text

    failed_seqs = list(epistemic_memory.severed_null_signatures) if epistemic_memory else []
    if epistemic_memory and getattr(epistemic_memory, "current_level_attempts", None):
        for att in epistemic_memory.current_level_attempts:
            traj = att.get("trajectory")
            if traj and traj not in failed_seqs:
                failed_seqs.append(traj)

    user_payload: dict[str, Any] = {
        "grid_hash": planning_set.grid_hash,
        "grid_bounds": {
            "height": grid_h,
            "width": grid_w,
        },
        "available_dsl_functions": functions_summary,
        "planning_objects": objects_summary,
        "spatial_relations": relations_summary,
        "past_failed_sequences": failed_seqs,
    }

    if trial_summary.get("current_level_scratchpad"):
        user_payload["current_level_scratchpad"] = trial_summary["current_level_scratchpad"]
    if trial_summary.get("actions_with_observed_effects"):
        user_payload["actions_with_observed_effects"] = trial_summary["actions_with_observed_effects"]
    if trial_summary.get("recent_action_outcomes"):
        user_payload["recent_action_outcomes"] = trial_summary["recent_action_outcomes"]

    if epistemic_memory is not None and hasattr(epistemic_memory, "format_structured_failures"):
        struct_failures = epistemic_memory.format_structured_failures()
        if struct_failures:
            trial_summary["structured_failures"] = struct_failures[-10:]
            user_payload["structured_failures"] = struct_failures[-10:]

    confirmed_effects = getattr(game_memory, "confirmed_action_effects", {}) if game_memory else {}
    unconfirmed_actions = getattr(game_memory, "unconfirmed_actions", {}) if game_memory else {}
    if action_budget > 0:
        user_payload["remaining_action_budget"] = action_budget
    if confirmed_effects:
        user_payload["confirmed_active_actions"] = confirmed_effects
    if unconfirmed_actions:
        user_payload["inactive_or_unconfirmed_actions"] = unconfirmed_actions
    if confirmed_actors:
        user_payload["verified_controllable_actors"] = sorted(list(confirmed_actors))

    has_coord_fn = any(
        p.get("name") in ("x", "y", "row", "col", "r", "c", "column")
        for fn in functions_summary
        for p in fn.get("parameters", [])
    )
    # Suppress coordinate affordances unless spatial actions are confirmed effective
    coord_is_confirmed = "ACTION6" in confirmed_effects
    if has_coord_fn and coord_is_confirmed and getattr(planning_set, "coordinate_candidates", None):
        user_payload["salient_coordinate_affordances"] = [
            {"x": c.x, "y": c.y, "label": c.label, "source": c.source_type}
            for c in planning_set.coordinate_candidates[:36]
        ]

    if game_memory is not None:
        if hasattr(game_memory, "format_empirical_context"):
            emp_ctx = game_memory.format_empirical_context()
        else:
            emp_ctx = getattr(game_memory, "confirmed_action_effects", {})
        if emp_ctx:
            user_payload["empirical_game_rules_confirmed"] = emp_ctx
        if hasattr(game_memory, "format_curriculum_context"):
            curr_ctx = game_memory.format_curriculum_context()
            if curr_ctx:
                user_payload["curriculum_progression_from_won_levels"] = curr_ctx
        if hasattr(game_memory, "format_game_model_summary"):
            model_summary = game_memory.format_game_model_summary()
            if model_summary and "confirmed: 0, pending: 0, falsified: 0" not in model_summary:
                user_payload["confirmed_game_model"] = model_summary

    def format_fn_call_example(fn_meta: dict[str, Any]) -> str:
        name = fn_meta.get("name", "action1")
        params = [p for p in fn_meta.get("parameters", []) if p.get("name") != "api"]
        if not params:
            return f"{name}()"
        p_strs = []
        for p in params:
            p_name = p.get("name", "x")
            if p_name == "x":
                p_strs.append("x=10")
            elif p_name == "y":
                p_strs.append("y=10")
            else:
                p_strs.append(f"{p_name}=0")
        return f"{name}({', '.join(p_strs)})"

    fn_0_ex = format_fn_call_example(functions_summary[0]) if len(functions_summary) > 0 else "action1()"

    image_note = (
        "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations.\n\n"
        if has_image
        else ""
    )

    user_text = f"""\
{image_note}Current Problem State & Available DSL Functions:
{json.dumps(user_payload, indent=2)}

Formulate up to 4 distinct candidate trajectories using the available DSL functions.
Provide complete, full-length trajectories aimed at solving the level directly.
Level solutions typically require extended sequences (20 to 30 sequential actions).

SYNTAX & REPETITION (count=N):
- Call DSL functions directly (e.g. `{fn_0_ex}`).
- You can specify repeat counts to avoid repetitive lines: e.g. `action1(count=15)` repeats action1 15 times (maximum 30).
- Optional `EXPECT: prop=val` clauses allow the Brusentsov judge to verify step consequences: e.g. `action1(count=10) EXPECT: dy=-10, dx=0`.

Provide your response strictly using XML tags:

<invariant_analysis>
1. Likely level goal: ...
2. Applicable confirmed invariants: ...
3. What is new/different: ...
4. Strategy for this level: ...
</invariant_analysis>

<trajectory_1>
[Full trajectory plan for candidate 1]
</trajectory_1>

<trajectory_2>
[Full trajectory plan for candidate 2]
</trajectory_2>

<trajectory_3>
[Full trajectory plan for candidate 3]
</trajectory_3>

<trajectory_4>
[Full trajectory plan for candidate 4]
</trajectory_4>
"""

    effective_level_index = level_index if level_index is not None else getattr(game_memory, "completed_levels", 0)
    phase_instruction = _build_phase_instruction(effective_level_index)
    sys_prompt = f"{SOLVER_SYSTEM_PROMPT}\n\n{phase_instruction}"
    if unconfirmed_actions:
        sys_prompt += (
            "\n\n# STRICT PROHIBITION ON INACTIVE OR UNCONFIRMED ACTIONS\n"
            "Do not propose actions or DSL functions corresponding to inactive or unconfirmed actions."
        )

    return sys_prompt, user_text

