"""Explorer Agent (Call 1 Family).

Conducts empirical research into action mechanics and coordinate affordances,
writing exclusively to EnvironmentSpecMemory.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from v10_agent.config import V10Config
from v10_agent.llm_advisor import BaseLLMAdvisor, sanitize_model_response
from v10_agent.memory_contours import EnvironmentSpecMemory, ProbeRecord
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders.explorer_prompt import (
    build_coordinate_hypothesis_prompt,
    build_explorer_prompts,
)
from v10_agent.types import ActionDeclaration

logger = logging.getLogger(__name__)


import ast

def clean_and_parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response text with robust fallbacks."""
    text = text.strip()
    # 1. Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Extract ```json ... ``` block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            text = m.group(1).strip()
    # 3. Find outermost { ... } 
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # 4. Fix common LLM JSON issues: comments, trailing commas, single quotes, Python booleans
            fixed = candidate
            # Remove JS-style comments (// ... and /* ... */)
            fixed = re.sub(r'//[^\n]*', '', fixed)
            fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            # Replace single quotes with double quotes (simple cases)
            if "'" in fixed and '"' not in fixed:
                fixed = fixed.replace("'", '"')
            fixed = re.sub(r'(?<!")\bTrue\b(?!")', 'true', fixed)
            fixed = re.sub(r'(?<!")\bFalse\b(?!")', 'false', fixed)
            fixed = re.sub(r'(?<!")\bNone\b(?!")', 'null', fixed)
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass
    raise ValueError(f"Could not parse JSON from text: {text[:200]}")


def extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract and parse JSON object from markdown fenced block or raw string."""
    clean_text = sanitize_model_response(text)
    # 1. Match ```json ... ```
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL):
        parsed = clean_and_parse_json(match.group(1))
        if parsed is not None:
            return parsed

    # 2. Try raw JSON substring finding first { to last }
    first_brace = clean_text.find("{")
    last_brace = clean_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        parsed = clean_and_parse_json(clean_text[first_brace : last_brace + 1])
        if parsed is not None:
            return parsed

    # 3. Match any candidate substring starting with {
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", clean_text, re.DOTALL):
        parsed = clean_and_parse_json(m.group(0))
        if parsed is not None:
            return parsed

    return None


class ExplorerAgent:
    """Call 1: Investigates action effects and coordinate affordances."""

    def __init__(self, config: V10Config, advisor: BaseLLMAdvisor):
        self.config = config
        self.advisor = advisor
        self.probe_manager = PrimitiveProbeManager(max_probes=config.max_primitive_probes_per_level)

    def generate_environment_spec(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        game_memory: Any | None = None,
    ) -> dict[str, Any]:
        """Generate and record a validated EnvironmentSpecification."""
        explorer_mm = getattr(
            self.config, "explorer_multimodal_enabled",
            getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
        )
        has_image = (
            any(bool(x) for x in image_png)
            if isinstance(image_png, (list, tuple))
            else (any(bool(x) for x in image_png.values()) if isinstance(image_png, dict) else bool(image_png))
        ) and explorer_mm
        effective_image = image_png if explorer_mm else None

        sys_prompt, user_prompt = build_explorer_prompts(
            planning_set=planning_set,
            probe_history=memory.probe_history,
            has_image=has_image,
        )

        try:
            response_text = self.advisor.generate(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                config=self.config,
                image_bytes=effective_image,
                agent_role="explorer",
            )
        except Exception as exc:
            logger.warning(f"Explorer LLM invocation failed: {exc}")
            response_text = ""

        spec = extract_json_block(response_text)
        if not spec or not isinstance(spec, dict):
            # Fallback deterministic minimal spec
            spec = {
                "schema_version": "v10.env_spec.1",
                "snapshot_hash": planning_set.grid_hash,
                "planning_set_id": planning_set.snapshot_id,
                "researched_actions": [
                    {
                        "action_id": act,
                        "effect_summary": "unexplored_atomic_action",
                        "supporting_evidence_ids": [],
                        "confidence": 0.5,
                        "contradicted": False,
                    }
                    for act in planning_set.allowed_action_ids
                ],
                "coordinate_affordances": [
                    {
                        "coordinate_candidate_id": c.candidate_id,
                        "x": c.x,
                        "y": c.y,
                        "source": {"type": c.source_type, "object_id": c.object_id},
                        "observed_effects": [],
                        "confidence": 0.5,
                    }
                    for c in planning_set.coordinate_candidates[:5]
                ],
                "object_class_notes": [],
                "action_surface_notes": [],
                "invariants": ["object_identities_discrete"],
            }

        # Validate schema version
        spec.setdefault("schema_version", "v10.env_spec.1")
        spec.setdefault("snapshot_hash", planning_set.grid_hash)
        spec.setdefault("planning_set_id", planning_set.snapshot_id)

        # Ensure all allowed actions from planning_set are represented in researched_actions
        researched = spec.get("researched_actions", [])
        if not isinstance(researched, list):
            researched = []
        researched_map = {
            str(a.get("action_id")).upper(): a
            for a in researched
            if isinstance(a, dict) and "action_id" in a
        }
        for act in planning_set.allowed_action_ids:
            act_str = str(act).upper()
            if act_str in ("RESET", "ACTION7"):
                continue
            if act_str not in researched_map:
                effect = "available atomic action"
                if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
                    effect = game_memory.confirmed_action_effects.get(act_str, effect)
                researched.append({
                    "action_id": act_str,
                    "effect_summary": str(effect),
                    "supporting_evidence_ids": [],
                    "confidence": 0.8 if effect != "available atomic action" else 0.5,
                    "contradicted": False,
                })
        # Invariant preservation principle: game invariants and kinematics never change, only add/enrich
        if game_memory is not None:
            confirmed_map = getattr(game_memory, "confirmed_action_effects", {})
            for act_str, conf_eff in confirmed_map.items():
                if act_str in researched_map:
                    entry = researched_map[act_str]
                    entry["effect_summary"] = conf_eff
                    entry["confidence"] = max(float(entry.get("confidence", 0.0) or 0.0), 0.95)
                    entry["contradicted"] = False

        # Accumulate confirmed invariants across levels
        invariants_list = spec.get("invariants", [])
        if not isinstance(invariants_list, list):
            invariants_list = []
        if game_memory is not None:
            for inv_rule in getattr(game_memory, "invariant_rules", []):
                if inv_rule not in invariants_list:
                    invariants_list.append(inv_rule)
        spec["invariants"] = invariants_list

        # Accumulate confirmed selection mechanics in action_surface_notes
        surface_notes = spec.get("action_surface_notes", [])
        if not isinstance(surface_notes, list):
            surface_notes = []
        if game_memory is not None:
            for sm in getattr(game_memory, "selection_mechanics", []):
                if sm not in surface_notes:
                    surface_notes.append(sm)
        spec["action_surface_notes"] = surface_notes

        spec["researched_actions"] = researched
        spec["available_actions"] = [
            str(a).upper() for a in planning_set.allowed_action_ids
            if str(a).upper() not in ("RESET", "ACTION7")
        ]

        # Write to memory (enforces ISO-3)
        memory.record_spec(spec)
        return spec

    def plan_probes(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory,
        max_probes: int = 4,
    ) -> list[ActionDeclaration]:
        """Generate systematic probe actions to test affordances (delegates to manager)."""
        mgr = PrimitiveProbeManager(max_probes=max_probes)
        discrete = mgr.plan_discrete_probes(planning_set, memory)
        if discrete:
            return discrete
        return mgr.plan_targeted_coordinate_probes(planning_set, memory)

    def propose_coordinate_probes(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory | None = None,
        max_coords: int = 4,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
    ) -> list[ActionDeclaration]:
        """Generate targeted coordinate probes via Qwen hypothesis prompt with deterministic fallback."""
        width = planning_set.grid_dims[1] if len(planning_set.grid_dims) > 1 else 0
        height = planning_set.grid_dims[0] if len(planning_set.grid_dims) > 0 else 0
        probes: list[ActionDeclaration] = []
        tested_coords: set[tuple[int, int]] = set()
        if memory:
            for p in memory.probe_history:
                if p.action_id == "ACTION6" and isinstance(p.action_data, dict):
                    x = p.action_data.get("x")
                    y = p.action_data.get("y")
                    if x is not None and y is not None:
                        try:
                            tested_coords.add((int(x), int(y)))
                        except (ValueError, TypeError):
                            pass

        explorer_mm = getattr(
            self.config, "explorer_multimodal_enabled",
            getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
        )
        has_image = (
            any(bool(x) for x in image_png)
            if isinstance(image_png, (list, tuple))
            else (any(bool(x) for x in image_png.values()) if isinstance(image_png, dict) else bool(image_png))
        ) and explorer_mm
        effective_image = image_png if explorer_mm else None

        # 1. Query LLM Advisor
        if self.advisor is not None and width > 0 and height > 0:
            sys_prompt, user_prompt = build_coordinate_hypothesis_prompt(
                planning_set=planning_set,
                num_hypotheses=max_coords,
                prior_tested_coords=sorted(list(tested_coords)) if tested_coords else None,
                has_image=has_image,
            )
            try:
                response_text = self.advisor.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    config=self.config,
                    image_bytes=effective_image,
                    agent_role="explorer",
                )
                parsed = extract_json_block(response_text)
                if parsed and isinstance(parsed, dict):
                    hypotheses = parsed.get("coordinate_hypotheses", [])
                    if isinstance(hypotheses, list):
                        for item in hypotheses:
                            if not isinstance(item, dict):
                                continue
                            x_val = item.get("x")
                            y_val = item.get("y")
                            if x_val is None or y_val is None:
                                continue
                            try:
                                x = int(x_val)
                                y = int(y_val)
                            except (ValueError, TypeError):
                                continue
                            if 0 <= x < width and 0 <= y < height and (x, y) not in tested_coords:
                                tested_coords.add((x, y))
                                probes.append(
                                    ActionDeclaration(
                                        action_id="ACTION6",
                                        data={"x": x, "y": y},
                                        reasoning={
                                            "source": "qwen_coordinate_hypothesis",
                                            "target": str(item.get("target_description", "salient_point")),
                                            "rationale": str(item.get("rationale", "click_discovery")),
                                        },
                                    )
                                )
                                if len(probes) >= max_coords:
                                    break
            except Exception as exc:
                logger.warning(f"Qwen coordinate hypothesis generation failed: {exc}")

        # 2. Deterministic Fallback if LLM produced fewer than min_expected coordinates
        min_expected = 3 if max_coords >= 5 else 2
        if len(probes) < min_expected and width > 0 and height > 0:
            for coord in planning_set.coordinate_candidates:
                if (coord.x, coord.y) not in tested_coords and 0 <= coord.x < width and 0 <= coord.y < height:
                    tested_coords.add((coord.x, coord.y))
                    probes.append(
                        ActionDeclaration(
                            action_id="ACTION6",
                            data={"x": coord.x, "y": coord.y},
                            reasoning={
                                "source": "coordinate_candidate_fallback",
                                "label": coord.label,
                                "type": coord.source_type,
                            },
                        )
                    )
                    if len(probes) >= max_coords:
                        break

        return probes


def filter_displacement_jitter(dr: int, dc: int) -> tuple[int, int]:
    """Filter out minor perpendicular rotation / alignment jitter when one axis dominates heavily.

    In 2D grid environments, genuine diagonal displacement moves equally across axes (|dr| ~= |dc|).
    When one axis displacement is significant (>=4) while the perpendicular axis has a negligible
    jitter (<=1, ratio >=3), the minor axis is a centroid shift artifact caused by sprite rotation
    or asymmetric anchoring, NOT an intentional 2D diagonal movement.
    """
    if abs(dr) >= 3 * max(1, abs(dc)) and abs(dc) <= 1 and abs(dr) >= 4:
        return dr, 0
    if abs(dc) >= 3 * max(1, abs(dr)) and abs(dr) <= 1 and abs(dc) >= 4:
        return 0, dc
    return dr, dc


def describe_vector(dr: int, dc: int) -> str:
    """Lazily produce human-readable multi-axis or diagonal direction descriptions."""
    dr, dc = filter_displacement_jitter(dr, dc)
    parts = []
    if dr < 0:
        parts.append("UP")
    if dr > 0:
        parts.append("DOWN")
    if dc < 0:
        parts.append("LEFT")
    if dc > 0:
        parts.append("RIGHT")

    if not parts:
        return "stationary"
    if len(parts) == 1:
        return parts[0]
    return "+".join(parts)  # e.g., "DOWN+RIGHT", "UP+LEFT"


def compute_probe_effect(before_snapshot: Any, after_obs: dict[str, Any]) -> str:
    """Determine empirical delta using Ranked Multi-Hypothesis Effect Detection."""
    import math
    from collections import Counter
    from v10_agent.arga_lite import extract_arga_snapshot

    raw_grid = after_obs.get("grid")
    if raw_grid is None:
        return "no_grid_observation"

    after_snapshot = extract_arga_snapshot(raw_grid)

    # =========================================================================
    # Hypothesis 1: Coherent Displacement Vector (Rigid / Compound Translation)
    # =========================================================================
    b_objs = before_snapshot.objects if hasattr(before_snapshot, "objects") else []
    a_objs = after_snapshot.objects if hasattr(after_snapshot, "objects") else []

    used_after: set[str] = set()
    matched_moves: list[tuple[Any, Any, int, int, float]] = []

    parents_b = [b for b in b_objs if getattr(b, "parent_id", None) is None and b.area >= 2]
    children_b = [b for b in b_objs if getattr(b, "parent_id", None) is not None]
    standalone_b = [b for b in b_objs if getattr(b, "parent_id", None) is None and b.area < 2]

    parent_deltas: dict[str, tuple[int, int, Any]] = {}

    # 1. Match parent objects first
    for b in parents_b:
        best_a = None
        best_dist = float("inf")
        for a in a_objs:
            if a.id in used_after or getattr(a, "parent_id", None) is not None:
                continue
            if a.color == b.color and abs(a.area - b.area) <= 4:
                dist = math.hypot(a.centroid.row - b.centroid.row, a.centroid.col - b.centroid.col)
                if dist < best_dist:
                    best_dist = dist
                    best_a = a
        if best_a is not None and best_dist <= max(24.0, b.bbox.height * 2.5, b.bbox.width * 2.5):
            dr = int(round(best_a.centroid.row - b.centroid.row))
            dc = int(round(best_a.centroid.col - b.centroid.col))
            dr, dc = filter_displacement_jitter(dr, dc)
            used_after.add(best_a.id)
            if abs(dr) >= 1 or abs(dc) >= 1:
                matched_moves.append((b, best_a, dr, dc, best_dist))
                parent_deltas[b.id] = (dr, dc, best_a)

    # 2. Match children using parent displacement as spatial reference
    for c in children_b:
        p_id = c.parent_id
        if p_id in parent_deltas:
            p_dr, p_dc, _ = parent_deltas[p_id]
            expected_r = c.centroid.row + p_dr
            expected_c = c.centroid.col + p_dc
            best_a = None
            best_dist = float("inf")
            for a in a_objs:
                if a.id in used_after:
                    continue
                if a.color == c.color and abs(a.area - c.area) <= 1:
                    dist = math.hypot(a.centroid.row - expected_r, a.centroid.col - expected_c)
                    if dist < best_dist:
                        best_dist = dist
                        best_a = a
            if best_a is not None and best_dist <= 2.0:
                used_after.add(best_a.id)
                dr = int(round(best_a.centroid.row - c.centroid.row))
                dc = int(round(best_a.centroid.col - c.centroid.col))
                dr, dc = filter_displacement_jitter(dr, dc)
                matched_moves.append((c, best_a, dr, dc, best_dist))

    # 3. Match remaining standalone components
    for s in standalone_b:
        best_a = None
        best_dist = float("inf")
        for a in a_objs:
            if a.id in used_after:
                continue
            if a.color == s.color and abs(a.area - s.area) <= 1:
                dist = math.hypot(a.centroid.row - s.centroid.row, a.centroid.col - s.centroid.col)
                if dist < best_dist:
                    best_dist = dist
                    best_a = a
        if best_a is not None and best_dist <= 24.0:
            dr = int(round(best_a.centroid.row - s.centroid.row))
            dc = int(round(best_a.centroid.col - s.centroid.col))
            dr, dc = filter_displacement_jitter(dr, dc)
            if abs(dr) >= 1 or abs(dc) >= 1:
                used_after.add(best_a.id)
                matched_moves.append((s, best_a, dr, dc, best_dist))

    if matched_moves:
        vector_groups: dict[tuple[int, int], list[tuple[Any, Any]]] = {}
        for b, a, dr, dc, _ in matched_moves:
            vector_groups.setdefault((dr, dc), []).append((b, a))

        parts = []
        for (dr, dc), pairs in sorted(vector_groups.items(), key=lambda item: -sum(b.area for b, _ in item[1])):
            dir_label = describe_vector(dr, dc)
            b_ids = {b.id for b, _ in pairs}
            top_level = [
                b for b, _ in pairs
                if getattr(b, "parent_id", None) is None or b.parent_id not in b_ids
            ]
            # Prioritize substantive objects by area descending
            top_level.sort(key=lambda o: -getattr(o, "area", 0))

            if len(top_level) <= 3:
                for b in top_level:
                    coherent_children = [
                        cid for cid in getattr(b, "children_ids", [])
                        if cid in b_ids
                    ]
                    if coherent_children:
                        b_desc = f"{b.id} (compound, {len(coherent_children)} parts)"
                    else:
                        b_desc = f"{b.id}"
                    parts.append(f"moved {b_desc} by dy={dr:+d}, dx={dc:+d} ({dir_label})")
            else:
                top_ids = [b.id for b in top_level[:3]]
                parts.append(f"moved [{', '.join(top_ids)} + {len(top_level)-3} more] by dy={dr:+d}, dx={dc:+d} ({dir_label})")

        if parts:
            return "; ".join(parts[:4])

    # =========================================================================
    # Hypothesis 2: Transfer of selection indicator / dots between STATIONARY containers (State Toggle)
    # =========================================================================
    b_subs = [o for o in before_snapshot.objects if o.area >= 4]
    a_subs = [o for o in after_snapshot.objects if o.area >= 4]
    b_dots = [o for o in before_snapshot.objects if o.area <= 2]
    a_dots = [o for o in after_snapshot.objects if o.area <= 2]

    lost_from = []
    gained_in = []
    for b_sub in b_subs:
        b_cnt = sum(
            1 for d in b_dots
            if b_sub.bbox.min_row <= d.centroid.row <= b_sub.bbox.max_row
            and b_sub.bbox.min_col <= d.centroid.col <= b_sub.bbox.max_col
        )
        match = next(
            (a for a in a_subs if a.color == b_sub.color and math.hypot(a.centroid.row - b_sub.centroid.row, a.centroid.col - b_sub.centroid.col) < 1.0),
            None,
        )
        a_cnt = sum(
            1 for d in a_dots
            if match and match.bbox.min_row <= d.centroid.row <= match.bbox.max_row
            and match.bbox.min_col <= d.centroid.col <= match.bbox.max_col
        ) if match else 0

        if b_cnt >= 1 and a_cnt == 0:
            lost_from.append(f"{b_sub.id} (color {b_sub.color})")
        elif b_cnt == 0 and a_cnt >= 1:
            gained_in.append(f"{b_sub.id} (color {b_sub.color})")

    if lost_from and gained_in:
        l_str = ", ".join(lost_from)
        g_str = ", ".join(gained_in)
        return f"selection indicator transferred: internal dots moved from {l_str} to {g_str} (active entity toggled)"

    # =========================================================================
    # Hypothesis 3: In-place cell color / state change without centroid displacement
    # =========================================================================
    b_grid = getattr(before_snapshot, "grid", None)
    if b_grid and isinstance(b_grid, (list, tuple)) and raw_grid and isinstance(raw_grid, (list, tuple)):
        h = min(len(b_grid), len(raw_grid))
        w = min(len(b_grid[0]), len(raw_grid[0])) if h > 0 else 0
        changed_cells = []
        color_counts: dict[tuple[int, int], int] = {}
        for r in range(h):
            for c in range(w):
                c_before = b_grid[r][c]
                c_after = raw_grid[r][c]
                if c_before != c_after:
                    changed_cells.append((r, c))
                    pair = (c_before, c_after)
                    color_counts[pair] = color_counts.get(pair, 0) + 1

        if changed_cells:
            min_r = min(r for r, _ in changed_cells)
            max_r = max(r for r, _ in changed_cells)
            min_c = min(c for _, c in changed_cells)
            max_c = max(c for _, c in changed_cells)
            trans_parts = [f"{c_from}->{c_to} ({cnt} cells)" for (c_from, c_to), cnt in sorted(color_counts.items())]
            if len(changed_cells) >= 50 or (max_r - min_r >= 20 and max_c - min_c >= 20):
                return f"selection indicator / state toggle: global state or active entity toggled ({len(changed_cells)} cells updated across region ({min_r},{min_c})-({max_r},{max_c}))"
            return f"color transition (stamp/draw): {len(changed_cells)} cells changed color [{', '.join(trans_parts)}] in region ({min_r},{min_c})-({max_r},{max_c})"

    # =========================================================================
    # Hypothesis 4: Changed object count or destruction/creation of entities
    # =========================================================================
    if len(before_snapshot.objects) != len(after_snapshot.objects):
        diff = len(after_snapshot.objects) - len(before_snapshot.objects)
        return f"object count changed by {diff:+d}"

    # =========================================================================
    # Hypothesis 5: Selection indicator appeared or cleared without transfer
    # =========================================================================
    if gained_in:
        g_str = ", ".join(gained_in)
        return f"selection indicator appeared in {g_str} (entity activated)"
    elif lost_from:
        l_str = ", ".join(lost_from)
        return f"selection indicator cleared from {l_str} (entity deactivated)"

    # =========================================================================
    # Hypothesis 5: Fallback conditional/inactive
    # =========================================================================
    return "conditional action (inactive at boundary or current context)"


class PrimitiveProbeManager:
    """Manages systematic empirical probe generation, cyclic combinatorial exploration, and physical effect recording."""

    DISCRETE_PROBE_ALLOWED = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}

    def __init__(self, max_probes: int = 16):
        self.max_probes = max_probes
        self.probed_actions: dict[str, str] = {}
        self.confirmed_effective_actions: dict[str, str] = {}
        self.inactive_actions: set[str] = set()
        self.zero_effect_actions: set[str] = set()
        self.retested_actions: set[str] = set()
        self.total_probes_executed: int = 0
        self._initial_sweep_planned: bool = False
        self._chains_attempted: set[tuple[str, tuple[str, ...]]] = set()

    def handle_level_transition(self, confirmed_action_effects: dict[str, str] | None = None) -> None:
        """Reset per-level probe state while preserving confirmed cross-level kinematics."""
        self.total_probes_executed = 0
        self._initial_sweep_planned = False
        self._chains_attempted.clear()
        self.retested_actions.clear()
        if confirmed_action_effects:
            for act, eff in confirmed_action_effects.items():
                if ("moved" in eff and any(d in eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))) or "selection indicator" in eff or "toggle" in eff:
                    self.confirmed_effective_actions[act] = eff
                    self.inactive_actions.discard(act)
                    self.zero_effect_actions.discard(act)
        # Any discrete action not yet confirmed remains/becomes an unconfirmed candidate for re-probing
        for act in self.DISCRETE_PROBE_ALLOWED:
            if act not in self.confirmed_effective_actions:
                self.inactive_actions.add(act)

    def is_action_effective(self, effect_summary: str) -> bool:
        """Determine if observed effect indicates active physical or visual response."""
        if not effect_summary or "inactive" in effect_summary or "conditional" in effect_summary or "no_grid" in effect_summary:
            return False
        return any(kw in effect_summary for kw in ("moved", "color transition", "selection indicator", "object count changed"))

    def is_motion_action(self, action_id: str) -> bool:
        """Check if confirmed action induces directional translation."""
        eff = self.confirmed_effective_actions.get(action_id, "")
        return "moved" in eff and any(d in eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))

    def get_dynamic_reprobes(self, triggering_action_id: str, effect_summary: str) -> list[ActionDeclaration]:
        """Dynamic re-probing invariant: re-probe inactive actions or verify motion changes after an effective state transition."""
        reprobes: list[ActionDeclaration] = []
        if self.total_probes_executed >= self.max_probes:
            return reprobes
        rem = self.max_probes - self.total_probes_executed

        is_modal_toggle = (
            "selection indicator" in effect_summary
            or "active entity toggled" in effect_summary
            or "state toggle" in effect_summary
        )

        # Case 1: Inactive discrete actions to retest in new context
        if self.inactive_actions:
            for act in list(self.inactive_actions):
                if act != triggering_action_id and act not in self.retested_actions and len(reprobes) < rem:
                    self.retested_actions.add(act)
                    reprobes.append(
                        ActionDeclaration(
                            action_id=act,
                            data={},
                            reasoning={
                                "source": "dynamic_reprobe_invariant",
                                "trigger": f"state_changed_by_{triggering_action_id}",
                                "rationale": "retest_inactive_action_in_new_context",
                            },
                        )
                    )

        # Case 2: Modal toggle occurred (selection switch):
        # Re-probe primitive motion actions (ACTION1..ACTION4) to observe affordances under the new active entity
        if is_modal_toggle:
            motion_candidates = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
            for act in motion_candidates:
                if act != triggering_action_id and not any(p.action_id == act for p in reprobes) and len(reprobes) < rem:
                    reprobes.append(
                        ActionDeclaration(
                            action_id=act,
                            data={},
                            reasoning={
                                "source": "dynamic_modal_reprobe",
                                "trigger": f"modal_toggle_by_{triggering_action_id}",
                                "rationale": "discover_motion_affordances_under_new_entity",
                            },
                        )
                    )

        return reprobes

    def schedule_falsification_reprobe(self, falsified_action_ids: list[str] | None = None) -> list[ActionDeclaration]:
        """Schedule immediate micro-reprobing of primitive actions on clean board following empirical falsification."""
        if falsified_action_ids:
            for act in falsified_action_ids:
                self.confirmed_effective_actions.pop(act, None)
                self.inactive_actions.discard(act)
                self.zero_effect_actions.discard(act)
                self.retested_actions.discard(act)

        self._initial_sweep_planned = False
        self.total_probes_executed = 0

        # Queue baseline discrete motion actions to discover actual kinematics on clean board
        actions_to_probe = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        probes = [
            ActionDeclaration(
                action_id=act,
                data={},
                reasoning={
                    "source": "falsification_micro_reprobe",
                    "rationale": "retest_baseline_kinematics_after_falsification",
                },
            )
            for act in actions_to_probe
        ]
        return probes


    def is_dynamic_action_surface(self, available_actions: Any, known_actions: set[str]) -> bool:
        """Check if action surface has dynamically introduced actions."""
        if not known_actions:
            return False
        acts = list(getattr(available_actions, "allowed_action_ids", available_actions) or [])
        new_acts = [a for a in acts if a not in known_actions and a != "RESET"]
        return len(new_acts) > 0

    def plan_discrete_probes(
        self,
        planning_set_or_actions: Any,
        memory: EnvironmentSpecMemory | None = None,
        max_probes: int | None = None,
        known_actions: set[str] | None = None,
    ) -> list[ActionDeclaration]:
        """Generate queue of discrete probes focusing on unconfirmed actions and their combinations with confirmed actions."""
        limit = max_probes if max_probes is not None else self.max_probes
        tested_in_this_level = {p.action_id for p in memory.probe_history} if memory else set()

        confirmed = set(self.confirmed_effective_actions.keys())
        if known_actions:
            confirmed.update(known_actions)

        if hasattr(planning_set_or_actions, "allowed_action_ids"):
            acts = list(planning_set_or_actions.allowed_action_ids)
        elif isinstance(planning_set_or_actions, (list, tuple, set)):
            acts = list(planning_set_or_actions)
        else:
            acts = []

        probes: list[ActionDeclaration] = []

        # 1. Unconfirmed discrete actions (actions that have not yet yielded confirmed semantics)
        unconfirmed = [
            str(act).upper() for act in acts
            if str(act).upper() in self.DISCRETE_PROBE_ALLOWED
            and str(act).upper() not in confirmed
            and str(act).upper() not in tested_in_this_level
        ]

        # First, queue individual unconfirmed actions (e.g. ACTION5 on Level 1, or ACTION1..5 on Level 0)
        for act_str in unconfirmed:
            if len(probes) < limit:
                probes.append(
                    ActionDeclaration(
                        action_id=act_str,
                        data={},
                        reasoning={"source": "primitive_probe", "type": "initial_sweep"},
                    )
                )

        # If on level 0 (no actions confirmed yet), queue all discrete actions
        if not confirmed:
            for act in acts:
                act_str = str(act).upper()
                if (
                    act_str in self.DISCRETE_PROBE_ALLOWED
                    and act_str not in tested_in_this_level
                    and not any(p.action_id == act_str for p in probes)
                    and len(probes) < limit
                ):
                    probes.append(
                        ActionDeclaration(
                            action_id=act_str,
                            data={},
                            reasoning={"source": "primitive_probe", "type": "initial_sweep"},
                        )
                    )

        self._initial_sweep_planned = True
        return probes

    def plan_targeted_coordinate_probes(
        self,
        planning_set: PlanningSet,
        available_actions: Sequence[str] | None = None,
        memory: EnvironmentSpecMemory | None = None,
        affordances: list[dict[str, Any]] | None = None,
    ) -> list[ActionDeclaration]:
        """Generate targeted coordinate probes based on Explorer affordance suggestions."""
        allowed = list(available_actions) if available_actions is not None else list(planning_set.allowed_action_ids)
        coord_act = "ACTION6" if "ACTION6" in allowed else (allowed[0] if allowed else "ACTION6")

        tested_coords = set()
        if memory:
            tested_coords = {
                (int(p.action_data.get("x", -1)), int(p.action_data.get("y", -1)))
                for p in memory.probe_history
                if p.action_id == coord_act
            }
        probes: list[ActionDeclaration] = []
        if affordances:
            for aff in affordances:
                x = int(aff.get("x", -1))
                y = int(aff.get("y", -1))
                if x >= 0 and y >= 0 and (x, y) not in tested_coords and len(probes) < 4:
                    tested_coords.add((x, y))
                    probes.append(
                        ActionDeclaration(
                            action_id=coord_act,
                            data={"x": x, "y": y},
                            reasoning={
                                "source": "primitive_probe",
                                "type": "targeted_coord_probe",
                                "affordance_id": aff.get("coordinate_candidate_id"),
                            },
                        )
                    )
        if not probes:
            for coord in planning_set.coordinate_candidates:
                if coord.source_type == "object_centroid" and (coord.x, coord.y) not in tested_coords and len(probes) < 3:
                    tested_coords.add((coord.x, coord.y))
                    probes.append(
                        ActionDeclaration(
                            action_id=coord_act,
                            data={"x": coord.x, "y": coord.y},
                            reasoning={
                                "source": "primitive_probe",
                                "type": "centroid_probe",
                                "coord_id": coord.candidate_id,
                            },
                        )
                    )
        return probes

    def record_probe_result(
        self,
        action_id: str,
        action_data: dict[str, Any],
        before_snapshot: Any,
        after_obs: dict[str, Any],
        memory: EnvironmentSpecMemory | None = None,
    ) -> ProbeRecord:
        """Compute delta and record probe observation into EnvironmentSpecMemory."""
        self.total_probes_executed += 1
        effect_str = compute_probe_effect(before_snapshot, after_obs)
        self.probed_actions[action_id] = effect_str

        if self.is_action_effective(effect_str):
            self.confirmed_effective_actions[action_id] = effect_str
            self.inactive_actions.discard(action_id)
            self.zero_effect_actions.discard(action_id)
        else:
            if action_id not in self.confirmed_effective_actions:
                self.inactive_actions.add(action_id)
                self.zero_effect_actions.add(action_id)

        probe_id = f"probe_{len(memory.probe_history)}" if memory else f"probe_{self.total_probes_executed}"
        record = ProbeRecord(
            probe_id=probe_id,
            action_id=action_id,
            action_data=action_data,
            observed_effect=effect_str,
            confidence=0.95 if self.is_action_effective(effect_str) else 0.5,
        )
        if memory:
            memory.record_probe(record)
        return record

    def get_next_combinatorial_chain(self) -> list[ActionDeclaration]:
        """Generate next chain of actions to test remaining inactive discrete actions in modified states."""
        inactive_discrete = [a for a in sorted(list(self.inactive_actions)) if a in self.DISCRETE_PROBE_ALLOWED]
        if not inactive_discrete:
            return []
        if self.total_probes_executed >= self.max_probes:
            return []

        motion_actions = [a for a in self.confirmed_effective_actions if self.is_motion_action(a)]
        if not motion_actions:
            motion_actions = [a for a in self.confirmed_effective_actions if a in self.DISCRETE_PROBE_ALLOWED]

        rem = self.max_probes - self.total_probes_executed
        if rem <= 0:
            return []

        for m_act in motion_actions:
            chain_key = (m_act, tuple(inactive_discrete))
            if chain_key not in self._chains_attempted:
                self._chains_attempted.add(chain_key)
                chain_acts = [m_act] + inactive_discrete
                return [
                    ActionDeclaration(
                        action_id=act_id,
                        data={},
                        reasoning={
                            "source": "primitive_probe",
                            "type": "combinatorial_chain",
                            "prefix": m_act,
                            "target": act_id,
                        },
                    )
                    for act_id in chain_acts[:rem]
                ]

        for inact in inactive_discrete:
            for m_act in motion_actions:
                chain_key_rev = ("rev", inact, m_act)
                if chain_key_rev not in self._chains_attempted:
                    self._chains_attempted.add(chain_key_rev)
                    chain_acts = [inact, m_act]
                    return [
                        ActionDeclaration(
                            action_id=act_id,
                            data={},
                            reasoning={
                                "source": "primitive_probe",
                                "type": "combinatorial_chain",
                                "prefix": inact,
                                "target": m_act,
                            },
                        )
                        for act_id in chain_acts[:rem]
                    ]

        return []


CyclicProbeManager = PrimitiveProbeManager

