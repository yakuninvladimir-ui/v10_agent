"""Solver Prompt Builder (Call 3 Family).

Constructs lean, noise-free prompts strictly quarantined from Python source code, syntax errors, and tracebacks.
Implements the Source Authority hierarchy and Brusentsov EXPECT grammar.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from v10_agent.brusentsov_logic import Ternary
from v10_agent.memory_contours import EpistemicMemory
from v10_agent.planning_set import PlanningSet

logger = logging.getLogger(__name__)

SOLVER_SYSTEM_PROMPT = """\
You are the planning module of an agent that plays unknown 2D grid games (ARC-AGI-3 environment).
You never see the rules. You receive: two images of the current frame (raw pixels; and the same frame with the perception parser's boxes and alias labels), a symbolic index of parsed objects, the accumulated empirical knowledge of this game, the empirical log of actions already executed on this level, and the remaining action budget.

ENVIRONMENT & ACTION SPACE:
- Grid size: up to 64x64 cells.
- Palette: 16 colors (0 to 15). Each color has a semantic role (BACKGROUND, ACTOR, OBSTACLE, HAZARD, TARGET, COLLECTIBLE, PORTAL, UNKNOWN) documented in <grounded_game_memory>.
- Coordinate convention: strictly Cartesian. x is horizontal (column, 0 <= x < width), y is vertical (row, 0 <= y < height).
- Action space:
  * Discrete actions: action1, action2, action3, action4, action5.
  * Coordinate action: action6(x=col, y=row) — coordinates are 0-indexed in the active cropped workspace (outer 1px system frame is already removed).

CORE PRINCIPLES:
1. Use only the provided action names and their real parameters.
2. Use only object IDs that appear in the provided data. Do not invent IDs.
3. Do not repeat sequences that are already listed as failed.
4. You may propose short or long sequences. Length is your choice. The goal is to pass the level.
5. Reason from what is visible and from confirmed action effects. Do not assume mechanics that are not supported by the data.
6. Ground your hypotheses in the provided <grounded_game_memory> (Last Victory and Last Defeat exemplars).
7. DIFFERENTIAL ANALYSIS OF FAILED ATTEMPTS: In Section 7, inspect the physical grid diffs ('Physical grid diff (before vs after attempt)') and effective step breakdowns. Identify exactly which tiles/cells changed color and which actions produced changes vs had no effect. Formulate hypotheses explaining why the partial grid changes occurred and modify the sequence to complete the pattern.
8. SEVERED ATTEMPT DIAGNOSTICS: In Section 7, pay strict attention to [FAILED ATTEMPT N DIAGNOSTIC]. If an action produced a mismatch (e.g. expected motion of target alias but observed stationary state or mutation of another alias), DO NOT repeat that action for that target! Switch immediately to alternative actions confirmed to manipulate the intended target.

SOURCE AUTHORITY (strict order):
1. Raw frame pixels — ground truth about what is on screen; use them for your solution hypotheses.
2. Empirical evidence log — ground truth about what happened when actions ran.
3. Annotated frame — parser segmentation; use it ONLY to map alias labels to pixels and to apply these designations in your response.
4. Symbolic object index — convenient coordinates/colors/sizes; it is LOSSY. Where index and pixels disagree, pixels win.

DUALVIEW GROUNDING & PARSER OVER-SEGMENTATION:
- Raw Frame Image is Ground Truth: The pristine visual image reveals true macroscopic entities, compound patterned structures, and visual symmetries.
- Parser Over-Segmentation: The perception engine decomposes grids strictly into monochromatic connected components. It automatically fragments multi-color compound entities into separate single-color pieces in the Annotated Frame and Object Index.

GAME INVARIANTS (what they are, how to use them):
An invariant is a claim about what stays stable across levels of THIS game (physics, kinematics, interaction dynamics, goal structure).
- Negative Barrier invariants (Contact with HAZARD -> Defeat): Never propose trajectories intersecting known hazard colors.
- Positive Canon invariants (Contact with TARGET -> Win): Your trajectory must aim to satisfy these conditions.
- Attach EXPECT clauses to steps where you genuinely predict an observable change.

EXECUTION LOOP FACTS (interface, not hints):
- Candidates run action-by-action on the live environment.
- After each action the judge compares observed frame change with EXPECT clauses:
  * A physically contradicted EXPECT terminates the candidate (verdict NULL).
  * A step with no EXPECT, or with an EXPECT neither confirmed nor contradicted, is weakly certified (verdict FOLLOW or OMIT): execution continues.
  * Confirmed EXPECT strongly verifies step (Brusentsov follow xy -> FOLLOW).

EXPECT GRAMMAR (the judge understands exactly these forms):
  moved(ALIAS, dy, dx)         centroid displacement in cells (e.g. moved(A, 0, 1))
  color(ALIAS)=C               dominant color of the object after the action (C in 0..15)
  unchanged(ALIAS)             bbox, position and color identical
  appears(ALIAS) / gone(ALIAS) object appeared or disappeared
  region(x0,y0,x1,y1): C1->C2  every cell of the rectangle changes C1 to C2
  state=S / levels_completed=N terminal environment state

OUTPUT CONTRACT:
Structure your response strictly using these XML tags:

<analysis>
Free-form reasoning and thinking: what this level is; OBSERVED vs INFERRED vs GUESSED; visible objects, target colors, hazard colors; differential analysis of failed attempt grid diffs.
</analysis>

<subgoals>
Decompose your plan into ordered milestones:
- Subgoal 1: [immediate milestone, e.g. navigate around obstacle to position X]
- Subgoal 2: [subsequent milestone, e.g. align with target]
- Subgoal 3: [win condition milestone, e.g. reach target color]
</subgoals>

<invariant_analysis>
relied:    for each candidate, the accumulated invariants/exemplars it depends on
proposed:  new candidate invariants, each with its discriminating observation
conflicts: any accumulated invariant that the current frame or evidence log contradicts
</invariant_analysis>

Then 1 to 4 candidates in tags <trajectory_1>, <trajectory_2>, ... numbered consecutively.
Each candidate contains one DSL call per line, with optional EXPECT clause:
  actionN(arg=val, ...) EXPECT: clause1, clause2
Repetition syntax: actionN(count=K) repeats K times (K <= 30; stops early if obstacle blocked).
"""


def _build_phase_instruction(level_index: int, total_levels: int = 6) -> str:
    """Helper retained for test and curriculum compatibility."""
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
    attempts_remaining: int | None = None,
    max_attempts: int | None = None,
    probes_remaining: int | None = None,
    max_probes: int | None = None,
    env_spec: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for Solver Agent using lean architecture."""
    grid_h, grid_w = planning_set.grid_dims
    grid_hash = planning_set.grid_hash if planning_set.grid_hash else "unknown"

    # 1. SPATIAL LAYOUT (Background & Container Segmentation)
    regions_lines = []
    bg_c = getattr(planning_set, "background_color", 0)
    regions_lines.append(f"R1: rows 0-{grid_h-1} cols 0-{grid_w-1} bg={bg_c} (canvas)")
    for o in planning_set.objects:
        if o.area >= max(16, int(grid_h * grid_w * 0.15)) or getattr(o, "shape_type", "") in ("compound_container", "container"):
            alias = planning_set.object_real_to_alias.get(o.id, o.id)
            contained = [
                planning_set.object_real_to_alias.get(c.id, c.id)
                for c in planning_set.objects
                if c.id != o.id and (
                    o.bbox.min_row <= c.bbox.min_row and c.bbox.max_row <= o.bbox.max_row and
                    o.bbox.min_col <= c.bbox.min_col and c.bbox.max_col <= o.bbox.max_col
                )
            ]
            cont_str = f" contains: [{', '.join(contained[:16])}]" if contained else ""
            regions_lines.append(
                f"R{len(regions_lines)+1}: rows {o.bbox.min_row}-{o.bbox.max_row} "
                f"cols {o.bbox.min_col}-{o.bbox.max_col} bg={o.color} container {alias}{cont_str}"
            )
    spatial_layout_text = "\n".join(regions_lines)

    # 2. OBJECT INDEX (Compact one-line per object with inline ascii for small items)
    obj_lines = []
    for o in planning_set.objects[:40]:
        alias = planning_set.object_real_to_alias.get(o.id, o.id)
        b = o.bbox
        shape = getattr(o, "shape_type", "entity")
        sig = getattr(o, "shape_signature", "")
        sig_str = f" {sig}" if sig else ""
        rel_info = ""
        if getattr(o, "children_ids", None):
            c_aliases = [planning_set.object_real_to_alias.get(cid, cid) for cid in o.children_ids]
            rel_info = f", parent of: [{', '.join(c_aliases)}]"
        elif getattr(o, "parent_id", None):
            p_alias = planning_set.object_real_to_alias.get(o.parent_id, o.parent_id)
            rel_info = f", child of: {p_alias}"
        line = f"{alias} ({o.id}): c={o.color}, bbox=[{b.min_row},{b.min_col},{b.max_row},{b.max_col}], area={o.area}, {shape}{sig_str}{rel_info}"
        if max(b.height, b.width) <= 12 and getattr(o, "compact_ascii", None):
            ascii_inline = "/".join(o.compact_ascii)
            line += f'; ascii="{ascii_inline}"'
        obj_lines.append(line)
    object_index_text = "\n".join(obj_lines) if obj_lines else "none"

    # 3. NON-TRIVIAL RELATIONS (Parser-detected, filtered)
    rel_lines = []
    seen_rel = set()
    for r in getattr(planning_set, "relations", []):
        if r.relation_type == "identical_shape":
            continue
        sub_alias = planning_set.object_real_to_alias.get(r.subject_id, r.subject_id)
        tgt_alias = planning_set.object_real_to_alias.get(r.target_id, r.target_id)
        rel_key = (sub_alias, r.relation_type, tgt_alias)
        if rel_key in seen_rel:
            continue
        seen_rel.add(rel_key)
        dist_str = f" (dist={r.metric_value:.1f})" if r.metric_value is not None else ""
        rel_lines.append(f"- {sub_alias} {r.relation_type} {tgt_alias}{dist_str}")
        if len(rel_lines) >= 30:
            break
    relations_text = "\n".join(rel_lines) if rel_lines else "none"

    # 4. ACCUMULATED GAME KNOWLEDGE (Tier 1, Tier 2, Tier 3, Falsified)
    tier1_lines = []
    tier2_lines = []
    tier3_lines = []
    falsified_lines = []

    if game_memory is not None:
        for act, eff in getattr(game_memory, "confirmed_action_effects", {}).items():
            clean_eff = re.sub(r"\s*\[compound with children [^\]]+\]", "", eff).strip()
            clean_eff = re.sub(r"\b[A-Z]\s*\(\[ACTOR\]\)", "[ACTOR]", clean_eff)
            tier1_lines.append(f"- {act}: {clean_eff}")

        for inv in getattr(game_memory, "structured_invariants", []):
            status = getattr(inv, "ternary_status", None)
            desc = inv.description
            conf = getattr(inv, "confidence", 0.5)
            lvl_count = len(getattr(inv, "confirmed_on_levels", []))
            provenance = f" (conf={conf:.1f}, confirmed_levels={lvl_count})"
            # Skip kinematic invariants containing raw local object aliases (actions are defined in DSL)
            if inv.invariant_type == "kinematics" and any(k in desc.lower() for k in ("moved", "obj_", "compound with children")):
                continue
            if status == Ternary.FALSE:
                falsified_lines.append(f"- [{inv.invariant_type}] {desc}")
            elif status == Ternary.TRUE:
                if inv.tier == 1 and not any(desc in l for l in tier1_lines):
                    tier1_lines.append(f"- [{inv.invariant_type}] {desc}{provenance}")
                elif inv.tier == 2:
                    tier2_lines.append(f"- [{inv.invariant_type}] {desc}{provenance}")
                elif inv.tier >= 3:
                    tier3_lines.append(f"- [{inv.invariant_type}] {desc}{provenance}")

        unconfirmed = getattr(game_memory, "unconfirmed_actions", {})
        for act, note in unconfirmed.items():
            falsified_lines.append(f"- {act}: unconfirmed/inactive ({note})")

    tier1_text = "\n".join(tier1_lines) if tier1_lines else "none"
    tier2_text = "\n".join(tier2_lines) if tier2_lines else "none"
    tier3_text = "\n".join(tier3_lines) if tier3_lines else "none"
    falsified_text = "\n".join(falsified_lines) if falsified_lines else "none"

    # 5. EMPIRICAL EVIDENCE LOG
    ev_lines = []
    if epistemic_memory is not None:
        judgments = getattr(epistemic_memory, "judgments", [])
        for idx, j in enumerate(judgments[-15:], start=1):
            v = getattr(j, "verdict", None)
            v_name = v.name if hasattr(v, "name") else str(v)
            act_dict = getattr(j, "action_dict", {}) or {}
            act_id = act_dict.get("action_id") or act_dict.get("id") or "action"
            act_data = act_dict.get("data", {})
            data_str = f"({', '.join(f'{k}={v}' for k, v in act_data.items())})" if act_data else "()"
            eff = getattr(j, "is_effective", False)
            eff_str = "frame changed" if eff else "none (0 cells)"
            ev_lines.append(f"#{idx} {act_id.lower()}{data_str} | changed: {eff_str} | {v_name}")
    evidence_log_text = "\n".join(ev_lines) if ev_lines else "none"

    # 6. DSL MANIFEST
    manifest_lines = []
    for fn in manifest.get("functions", []):
        name = fn.get("name", "action")
        params = [p for p in fn.get("parameters", []) if p.get("name") != "api"]
        param_str = ", ".join(f"{p.get('name', 'x')}: {p.get('type', 'int')} = {p.get('default', 0)}" for p in params)
        returns = fn.get("returns", "effect_declaration")
        doc = fn.get("docstring", "").strip()
        manifest_lines.append(f"{name}({param_str}) -> {returns}:")
        if doc:
            manifest_lines.append(f'  """[HYPOTHESIS]: {doc}"""')
    dsl_manifest_text = "\n".join(manifest_lines) if manifest_lines else "none"

    # 6.5 EMPIRICAL PROBE DYNAMICS & ACTION AFFORDANCES (from Explorer Phase 2)
    explorer_lines = []
    if env_spec is not None:
        # 1. Action Affordances
        affordances = env_spec.get("action_affordances", [])
        if affordances:
            explorer_lines.append("Observed Action Affordances during Probes:")
            for aff in affordances:
                if isinstance(aff, dict):
                    act_id = aff.get("action_id", "")
                    eff_cls = aff.get("effect_class", "KINEMATIC")
                    params = aff.get("parameters", {})
                    coord = aff.get("coordination_notes", "")
                    param_parts = []
                    for k, v in sorted(params.items()):
                        param_parts.append(f"{k}={v}")
                    p_str = f" ({', '.join(param_parts)})" if param_parts else ""
                    coord_str = f" [{coord}]" if coord else ""
                    explorer_lines.append(f"  * {act_id} [{eff_cls}]:{p_str}{coord_str}")
                elif isinstance(aff, str) and aff.strip():
                    explorer_lines.append(f"  * {aff.strip()}")
        else:
            displacements = env_spec.get("action_displacements", [])
            if displacements:
                explorer_lines.append("Observed Object Displacements during Probes:")
                for d in displacements:
                    if isinstance(d, dict):
                        act_id = d.get("action_id", "")
                        disps = d.get("displacements", [])
                        coord = d.get("coordination", "")
                        disp_parts = []
                        if isinstance(disps, list):
                            for item in disps:
                                if isinstance(item, dict):
                                    a = item.get("alias", "")
                                    dy = item.get("dy", 0)
                                    dx = item.get("dx", 0)
                                    disp_parts.append(f"{a} moved (dy={dy:+d}, dx={dx:+d})")
                                elif isinstance(item, str):
                                    disp_parts.append(item)
                        disp_str = ", ".join(disp_parts) if disp_parts else ""
                        coord_str = f" [{coord}]" if coord else ""
                        explorer_lines.append(f"  * {act_id}: {disp_str}{coord_str}")
                    elif isinstance(d, str) and d.strip():
                        explorer_lines.append(f"  * {d.strip()}")
            elif env_spec.get("action_surface_notes"):
                explorer_lines.append("Observed Object Movements:")
                for an in env_spec.get("action_surface_notes", []):
                    if isinstance(an, dict):
                        act_id = an.get("action_id", "")
                        geff = an.get("grounded_effect", "")
                        explorer_lines.append(f"  * {act_id}: {geff}")
                    elif isinstance(an, str) and an.strip():
                        explorer_lines.append(f"  * {an.strip()}")

        # 2. Static objects (never moved during probes)
        static_objs = env_spec.get("static_objects", [])
        if static_objs:
            explorer_lines.append("Objects Remaining Completely Static during Probes:")
            for so in static_objs:
                if isinstance(so, dict):
                    alias = so.get("alias", "")
                    desc = so.get("description", "")
                    desc_str = f" ({desc})" if desc else ""
                    explorer_lines.append(f"  * {alias}{desc_str}")
                elif isinstance(so, str) and so.strip():
                    explorer_lines.append(f"  * {so.strip()}")

        # 3. Structural geometry
        geom = env_spec.get("structural_geometry", []) or env_spec.get("structural_notes", [])
        if geom:
            explorer_lines.append("Observed Spatial Geometry:")
            for g in geom:
                explorer_lines.append(f"  * {g}")

    explorer_synthesis_text = "\n".join(explorer_lines) if explorer_lines else "none"

    # 7. FAILED CANDIDATES
    failed_lines = []
    if epistemic_memory is not None:
        scratchpad = epistemic_memory.format_scratchpad_context() if hasattr(epistemic_memory, "format_scratchpad_context") else ""
        if scratchpad:
            failed_lines.append(scratchpad)
        for sig in getattr(epistemic_memory, "severed_null_signatures", []):
            failed_lines.append(f"- Candidate signature {sig} severed (NULL contradiction)")
        struct_fails = epistemic_memory.format_structured_failures() if hasattr(epistemic_memory, "format_structured_failures") else getattr(epistemic_memory, "structured_failures", [])
        for sf in struct_fails[-5:]:
            step = sf.get("failed_step") or sf.get("step_id", "?")
            expl = sf.get("explanation") or sf.get("reason", "contradiction")
            traj = sf.get("trajectory_id", "")
            traj_str = f"{traj} " if traj else ""
            failed_lines.append(f"- Candidate {traj_str}died at step {step}: {expl}")
        for ft in getattr(epistemic_memory, "failed_completed_trajectories", []):
            ft_str = " -> ".join(ft)
            failed_lines.append(f"- Completed sequence failed without win: {ft_str}")
    failed_candidates_text = "\n".join(failed_lines) if failed_lines else "none"

    # 7.5 PREVIOUS LEVEL EXAMPLE (Inter-Level Memory)
    prev_level_example = ""
    if game_memory is not None and hasattr(game_memory, "format_previous_level_example"):
        formatted_ex = game_memory.format_previous_level_example()
        if formatted_ex:
            prev_level_example = f"\n{formatted_ex}\n"

    # 7.6 GROUNDED GAME MEMORY (Palette, Exemplars, Core Laws)
    grounded_memory_text = ""
    if game_memory is not None and hasattr(game_memory, "format_grounded_memory_for_prompt"):
        grounded_block = game_memory.format_grounded_memory_for_prompt()
        if grounded_block and grounded_block.strip():
            grounded_memory_text = grounded_block

    # 8. LIMITS & BUDGET COUNTERS
    rem_attempts = attempts_remaining if attempts_remaining is not None else 5
    tot_attempts = max_attempts if max_attempts is not None else 5
    eff_level = level_index if level_index is not None else (getattr(game_memory, "completed_levels", 0) if game_memory else 0)
    eff_completed = getattr(game_memory, "completed_levels", 0) if game_memory else 0

    image_header = (
        "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations. "
        "The annotated frame's labels are the aliases used in the object index below.\n\n"
        "FRAME (current state; post-reset unless stated otherwise):\n"
        "[image 1: raw pixels]\n"
        "[image 2: annotated, alias boxes]\n\n"
        if has_image else ""
    )

    user_text = f"""\
{image_header}GRID: {grid_h}x{grid_w}, hash {grid_hash}

SPATIAL LAYOUT (background segmentation computed from pixels):
{spatial_layout_text}

OBJECT INDEX (alias, color, bbox [r0,c0,r1,c1], area, shape; inline-ascii if bbox <= 12x12):
{object_index_text}

NON-TRIVIAL RELATIONS (parser-detected):
{relations_text}

EMPIRICAL PROBE DYNAMICS & OBJECT DISPLACEMENTS:
{explorer_synthesis_text}
{prev_level_example}
ACCUMULATED GAME KNOWLEDGE (empirical, per THIS game; may be wrong or level-conditional):
TIER 1 - physics & kinematics:
{tier1_text}
TIER 2 - interaction dynamics:
{tier2_text}
TIER 3 - level-rule hypotheses distilled from wins (NOT universal laws):
{tier3_text}
FALSIFIED - do not rely on:
{falsified_text}
NOTE: the DSL manifest docstring is the coder's paraphrase of Tier 1; on any discrepancy Tier 1 wording and the evidence log win.

<grounded_game_memory>
{grounded_memory_text if grounded_memory_text else "No grounded memory yet (no victories/defeats recorded)."}
</grounded_game_memory>

EMPIRICAL EVIDENCE LOG (this level, chronological; hash->hash proves whether the frame changed):
{evidence_log_text}

DSL MANIFEST (signatures = interface; docstrings = HYPOTHESIS):
{dsl_manifest_text}
All coordinate actions strictly follow Cartesian convention: action6(x=col, y=row) where x is column (0 <= x < width) and y is row (0 <= y < height). Bounding boxes in the evidence log declare x cols and y rows (inclusive).

FAILED CANDIDATES THIS LEVEL (with death step, physical before/after grid diffs, and verdicts; do not repeat them):
{failed_candidates_text}

LIMITS & BUDGET:
- Actions remaining: {action_budget} (shared across all candidates on this level)
- Planning attempts: {rem_attempts} of {tot_attempts} remaining
GAME STATE: {eff_completed} levels completed so far in this game; level index {eff_level}.

TASK: propose candidates per the output contract.
"""
    return SOLVER_SYSTEM_PROMPT, user_text
