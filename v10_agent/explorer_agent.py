"""Explorer Agent (Call 1 Family).

Conducts empirical research into action mechanics and coordinate affordances,
writing exclusively to EnvironmentSpecMemory.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from dataclasses import dataclass, field
from enum import Enum

from v10_agent.action_semantics import (
    NON_VECTOR_ACTION_IDS,
    is_observable_move,
    parse_axis_components,
    token_after_keyword,
)
from v10_agent.config import V10Config
from v10_agent.llm_advisor import BaseLLMAdvisor, sanitize_model_response
from v10_agent.memory_contours import EnvironmentSpecMemory, ProbeRecord
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders.explorer_prompt import (
    build_coordinate_hypothesis_prompt,
    build_explorer_prompts,
    build_explorer_synthesis_prompt,
    build_primitive_research_prompt,
)
from v10_agent.types import ActionDeclaration

logger = logging.getLogger(__name__)


class EffectClass(str, Enum):
    KINEMATIC = "KINEMATIC"
    PALETTE_TRANSITION = "PALETTE_TRANSITION"
    TOPOLOGY_MUTATION = "TOPOLOGY_MUTATION"
    MODAL_SELECTION = "MODAL_SELECTION"
    CONDITIONAL_TRIGGER = "CONDITIONAL_TRIGGER"


@dataclass
class ActionAffordance:
    action_id: str
    effect_class: str
    parameters: dict[str, Any] = field(default_factory=dict)
    coordination_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": str(self.action_id).upper(),
            "effect_class": str(self.effect_class).upper(),
            "parameters": dict(self.parameters),
            "coordination_notes": str(self.coordination_notes),
        }


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

    # 5. Truncated JSON repair (e.g. LLM output cut off before closing braces/brackets)
    if start != -1:
        truncated_cand = text[start:]
        # Remove trailing incomplete string/key if any (e.g. cut off mid-token)
        repair = re.sub(r',\s*[^,:{}\[\]]*$', '', truncated_cand)
        # Close strings if unclosed odd number of unescaped quotes
        quotes = [m.start() for m in re.finditer(r'(?<!\\)"', repair)]
        if len(quotes) % 2 == 1:
            repair += '"'
        # Balance open brackets/braces
        rem_cur = max(0, repair.count("{") - repair.count("}"))
        rem_sq = max(0, repair.count("[") - repair.count("]"))
        repair = repair.rstrip(', ') + (']' * rem_sq) + ('}' * rem_cur)
        repair = re.sub(r',\s*([}\]])', r'\1', repair)
        try:
            res = json.loads(repair)
            if isinstance(res, dict):
                return res
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

    # 2. Try raw JSON substring finding first { to last } (or to end of string if truncated)
    first_brace = clean_text.find("{")
    last_brace = clean_text.rfind("}")
    if first_brace != -1:
        cand_str = clean_text[first_brace : last_brace + 1] if last_brace > first_brace else clean_text[first_brace:]
        parsed = clean_and_parse_json(cand_str)
        if parsed is not None:
            return parsed

    # 3. Match any candidate substring starting with {
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", clean_text, re.DOTALL):
        parsed = clean_and_parse_json(m.group(0))
        if parsed is not None:
            return parsed

    return None


def generate_symbolic_environment_spec(
    planning_set: PlanningSet,
    memory: EnvironmentSpecMemory,
    game_memory: Any | None = None,
    probe_manager: PrimitiveProbeManager | None = None,
) -> dict[str, Any]:
    """Generate and record a validated EnvironmentSpecification deterministically from empirical diffs.

    Strict Gating: Only actions with confirmed physical/visual effects (diff > 0)
    are certified into available_actions and researched_actions.
    Unconfirmed actions with zero observable effect are strictly excluded.
    """
    confirmed_actions: dict[str, str] = {}
    if probe_manager is not None:
        confirmed_actions.update(probe_manager.confirmed_effective_actions)
    if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
        confirmed_actions.update(game_memory.confirmed_action_effects)

    # Invariants from universal_invariants
    invariants: list[str] = ["object_identities_discrete"]
    try:
        from v10_agent.universal_invariants import discover_invariants
        discovered = discover_invariants(planning_set)
        for inv in discovered:
            desc = getattr(inv, "description", str(inv))
            if desc and desc not in invariants:
                invariants.append(desc)
    except Exception as exc:
        logger.debug(f"Invariant discovery skipped: {exc}")

    # Build researched_actions, action_affordances, and action_displacements ONLY for confirmed actions
    researched: list[dict[str, Any]] = []
    affordance_list: list[dict[str, Any]] = []
    displacement_list: list[dict[str, Any]] = []
    for act_id in sorted(confirmed_actions.keys()):
        if not is_observable_move(act_id):
            continue
        eff = confirmed_actions[act_id]
        evidence_ids = []
        aff_dict = None
        if memory is not None and getattr(memory, "probe_history", None):
            for p in memory.probe_history:
                if p.action_id == act_id:
                    evidence_ids.append(p.probe_id)
                    if getattr(p, "affordance", None) and not aff_dict:
                        aff_dict = dict(p.affordance)
        if not aff_dict:
            aff_dict = classify_effect_summary_to_affordance(act_id, eff)
        affordance_list.append(aff_dict)
        if aff_dict.get("effect_class") == "KINEMATIC":
            params = aff_dict.get("parameters", {})
            disps = params.get("displacements", [])
            if not disps and "affected_alias" in params:
                disps = [{"alias": params["affected_alias"], "dy": params.get("dy", 0), "dx": params.get("dx", 0)}]
            displacement_list.append({
                "action_id": act_id,
                "displacements": disps,
                "coordination": aff_dict.get("coordination_notes", ""),
            })
        researched.append({
            "action_id": act_id,
            "effect_summary": eff,
            "supporting_evidence_ids": evidence_ids,
            "confidence": 1.0,
            "contradicted": False,
        })

    # Preserve conditional candidate actions (unconfirmed on S0, but potential contact/state-dependent actions)
    cond_candidates: set[str] = set()
    if probe_manager is not None and hasattr(probe_manager, "conditional_candidate_actions"):
        cond_candidates.update(probe_manager.conditional_candidate_actions)

    for act in planning_set.allowed_action_ids:
        act_up = str(act).upper()
        if (
            act_up in cond_candidates
            and act_up not in confirmed_actions
            and is_observable_move(act_up)
        ):
            researched.append({
                "action_id": act_up,
                "effect_summary": "unconfirmed on S0 (candidate contact or state-dependent action)",
                "supporting_evidence_ids": [],
                "confidence": 0.5,
                "contradicted": False,
            })
            affordance_list.append({
                "action_id": act_up,
                "effect_class": "CONDITIONAL_TRIGGER",
                "parameters": {
                    "status": "unconfirmed_on_s0",
                    "requires_contact_or_selection": True,
                },
                "coordination_notes": "zero observable delta on pristine frame S0; candidate contact or state-dependent action",
            })

    # Fallback if no confirmed actions yet discovered (e.g. probing disabled or initial step)
    if not researched:
        unconfirmed = set()
        if probe_manager is not None:
            unconfirmed.update(probe_manager.inactive_actions)
            unconfirmed.update(probe_manager.zero_effect_actions)
        if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
            unconfirmed.update(game_memory.unconfirmed_actions.keys())

        for act in planning_set.allowed_action_ids:
            act_str = str(act).upper()
            if not is_observable_move(act_str) or act_str in unconfirmed:
                continue
            researched.append({
                "action_id": act_str,
                "effect_summary": "available atomic action",
                "supporting_evidence_ids": [],
                "confidence": 0.5,
                "contradicted": False,
            })
            aff_dict = classify_effect_summary_to_affordance(act_str, "available atomic action")
            affordance_list.append(aff_dict)

    # Coordinate affordances only if ACTION6 is confirmed effective
    affordances: list[dict[str, Any]] = []
    if "ACTION6" in confirmed_actions:
        for c in planning_set.coordinate_candidates[:5]:
            affordances.append({
                "coordinate_candidate_id": c.candidate_id,
                "x": c.x,
                "y": c.y,
                "source": {"type": c.source_type, "object_id": c.object_id},
                "observed_effects": [confirmed_actions["ACTION6"]],
                "confidence": 0.9,
            })

    spec = {
        "schema_version": "v10.env_spec.1",
        "snapshot_hash": planning_set.grid_hash,
        "planning_set_id": planning_set.snapshot_id,
        "available_actions": [a["action_id"] for a in researched],
        "researched_actions": researched,
        "action_affordances": affordance_list,
        "action_displacements": displacement_list,
        "coordinate_affordances": affordances,
        "object_class_notes": [],
        "action_surface_notes": [],
        "invariants": invariants,
    }

    if memory is not None:
        memory.record_spec(spec)
    return spec


class ExplorerAgent:
    """Call 1: Investigates action effects and coordinate affordances."""

    def __init__(self, config: V10Config, advisor: BaseLLMAdvisor):
        self.config = config
        self.advisor = advisor
        self.probe_manager = PrimitiveProbeManager(
            max_probes=config.max_primitive_probes_per_level,
            max_invariant_probes=getattr(config, "max_invariant_verification_probes", 3),
            max_steps_per_probe=getattr(config, "max_invariant_probe_steps", 2),
        )

    def generate_symbolic_environment_spec(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory,
        game_memory: Any | None = None,
    ) -> dict[str, Any]:
        """Generate and record validated EnvironmentSpecification deterministically without LLM."""
        return generate_symbolic_environment_spec(
            planning_set=planning_set,
            memory=memory,
            game_memory=game_memory,
            probe_manager=self.probe_manager,
        )

    def synthesize_level_spec(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        game_memory: Any | None = None,
    ) -> dict[str, Any]:
        """Synthesize factual environment specification after active probing is complete.

        1. Generates a strict deterministic baseline spec via generate_symbolic_environment_spec
           to guarantee that available_actions and researched_actions contain ONLY confirmed effective actions.
        2. Queries LLM Explorer with build_explorer_synthesis_prompt to describe factual visual object roles,
           action groundings, symmetries, and structural invariants without guessing goals or coordinates.
        3. Merges validated factual descriptions into the spec and updates game_memory.
        """
        # Step 1: Base symbolic spec (strict action gating)
        base_spec = self.generate_symbolic_environment_spec(
            planning_set=planning_set,
            memory=memory,
            game_memory=game_memory,
        )

        # Immediately seed game_memory with verified empirical affordances before LLM invocation
        if game_memory is not None and hasattr(game_memory, "update_action_affordances"):
            if "action_affordances" in base_spec and base_spec["action_affordances"]:
                game_memory.update_action_affordances(base_spec["action_affordances"])

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

        if self.advisor is None or self.config.llm_advisor_backend == "fake":
            return base_spec

        # Gather confirmed and unconfirmed actions from probe manager / game memory
        confirmed_actions = dict(self.probe_manager.confirmed_effective_actions)
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_actions.update(game_memory.confirmed_action_effects)

        unconfirmed_actions = {}
        for act in self.probe_manager.inactive_actions | self.probe_manager.zero_effect_actions:
            unconfirmed_actions[act] = "no visible effect (0 cells changed)"
        if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
            unconfirmed_actions.update(game_memory.unconfirmed_actions)

        sys_prompt, user_prompt = build_explorer_synthesis_prompt(
            planning_set=planning_set,
            confirmed_actions=confirmed_actions,
            unconfirmed_actions=unconfirmed_actions,
            probe_history=getattr(memory, "probe_history", []),
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
                # 1. Action Affordances (Complete Spectrum of Observed Mutations)
                act_affs = parsed.get("action_affordances", [])
                clean_affs: list[dict[str, Any]] = []
                if isinstance(act_affs, list):
                    for item in act_affs:
                        if isinstance(item, dict):
                            act_id = str(item.get("action_id", "")).strip().upper()
                            eff_cls = str(item.get("effect_class", "KINEMATIC")).strip().upper()
                            if eff_cls not in ("KINEMATIC", "PALETTE_TRANSITION", "TOPOLOGY_MUTATION", "MODAL_SELECTION", "CONDITIONAL_TRIGGER"):
                                eff_cls = "KINEMATIC"
                            params = item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {}
                            notes = str(item.get("coordination_notes", "")).strip()
                            clean_affs.append({
                                "action_id": act_id,
                                "effect_class": eff_cls,
                                "parameters": params,
                                "coordination_notes": notes,
                            })

                # Legacy fallback: action_displacements
                act_disps = parsed.get("action_displacements", [])
                clean_disps = []
                if isinstance(act_disps, list):
                    for item in act_disps:
                        if isinstance(item, dict):
                            act_id = str(item.get("action_id", "")).strip().upper()
                            clean_disps.append({
                                "action_id": act_id,
                                "displacements": item.get("displacements", []),
                                "coordination": str(item.get("coordination", "")).strip(),
                            })
                            # If not already present in clean_affs, convert to KINEMATIC affordance
                            if not any(a["action_id"] == act_id for a in clean_affs):
                                disps = item.get("displacements", [])
                                primary = disps[0] if disps and isinstance(disps[0], dict) else {}
                                clean_affs.append({
                                    "action_id": act_id,
                                    "effect_class": "KINEMATIC",
                                    "parameters": {
                                        "affected_alias": primary.get("alias", ""),
                                        "dy": primary.get("dy", 0),
                                        "dx": primary.get("dx", 0),
                                        "displacements": disps,
                                    },
                                    "coordination_notes": str(item.get("coordination", "")).strip(),
                                })
                        elif isinstance(item, str) and item.strip():
                            clean_disps.append(item.strip())

                # ACTION AFFORDANCE COMPLETENESS INVARIANT:
                # Any confirmed action that demonstrated frame delta > 0 MUST be present in action_affordances.
                # If LLM omitted a confirmed action, synthesize its affordance from probe history / summary.
                known_aff_acts = {a["action_id"] for a in clean_affs if isinstance(a, dict)}
                for act_id in sorted(confirmed_actions.keys()):
                    act_up = str(act_id).upper()
                    if not is_observable_move(act_up) or act_up in known_aff_acts:
                        continue
                    eff_summary = confirmed_actions[act_id]
                    synth_aff = None
                    if memory is not None and getattr(memory, "probe_history", None):
                        for p in memory.probe_history:
                            if p.action_id == act_up and getattr(p, "affordance", None):
                                synth_aff = dict(p.affordance)
                                break
                    if not synth_aff:
                        synth_aff = classify_effect_summary_to_affordance(act_up, eff_summary)
                    clean_affs.append(synth_aff)
                    known_aff_acts.add(act_up)

                # Also include conditional_candidate_actions in action_affordances if omitted
                cond_candidates: set[str] = set()
                if hasattr(self, "probe_manager") and hasattr(self.probe_manager, "conditional_candidate_actions"):
                    cond_candidates.update(self.probe_manager.conditional_candidate_actions)

                for act_id in sorted(cond_candidates):
                    act_up = str(act_id).upper()
                    if is_observable_move(act_up) and act_up not in known_aff_acts and act_up not in confirmed_actions:
                        clean_affs.append({
                            "action_id": act_up,
                            "effect_class": "CONDITIONAL_TRIGGER",
                            "parameters": {
                                "status": "unconfirmed_on_s0",
                                "requires_contact_or_selection": True,
                            },
                            "coordination_notes": "zero observable delta on pristine frame S0; candidate contact or state-dependent action",
                        })
                        known_aff_acts.add(act_up)

                if clean_affs:
                    base_spec["action_affordances"] = clean_affs
                    if game_memory is not None and hasattr(game_memory, "update_action_affordances"):
                        game_memory.update_action_affordances(clean_affs)

                # Ensure backwards-compatible action_displacements exists
                if not clean_disps and clean_affs:
                    for aff in clean_affs:
                        if aff.get("effect_class") == "KINEMATIC":
                            params = aff.get("parameters", {})
                            disps = params.get("displacements", [])
                            if not disps and "affected_alias" in params:
                                disps = [{"alias": params["affected_alias"], "dy": params.get("dy", 0), "dx": params.get("dx", 0)}]
                            clean_disps.append({
                                "action_id": aff["action_id"],
                                "displacements": disps,
                                "coordination": aff.get("coordination_notes", ""),
                            })
                if clean_disps:
                    base_spec["action_displacements"] = clean_disps

                # Update available_actions: MUST include ALL confirmed affordance actions
                current_available = set(base_spec.get("available_actions", []))
                for aff in clean_affs:
                    act_up = aff["action_id"]
                    if is_observable_move(act_up) and act_up not in current_available:
                        base_spec["available_actions"].append(act_up)
                        current_available.add(act_up)

                # Backwards-compatible action_surface_notes fallback
                act_notes = parsed.get("action_surface_notes", [])
                if isinstance(act_notes, list) and not base_spec.get("action_displacements") and not base_spec.get("action_affordances"):
                    clean_act_notes = []
                    for note in act_notes:
                        if isinstance(note, dict):
                            act_id = str(note.get("action_id", "")).strip().upper()
                            effect = str(note.get("grounded_effect", "")).strip()
                            clean_act_notes.append({"action_id": act_id, "grounded_effect": effect})
                        elif isinstance(note, str) and note.strip():
                            clean_act_notes.append(note.strip())
                    if clean_act_notes:
                        base_spec["action_surface_notes"] = clean_act_notes

                # 2. Static Objects (Observed completely unmoving during probes)
                static_objs = parsed.get("static_objects", [])
                if isinstance(static_objs, list):
                    clean_static = []
                    for item in static_objs:
                        if isinstance(item, dict):
                            clean_static.append({
                                "alias": str(item.get("alias", "")).strip(),
                                "description": str(item.get("description", "")).strip(),
                            })
                        elif isinstance(item, str) and item.strip():
                            clean_static.append(item.strip())
                    if clean_static:
                        base_spec["static_objects"] = clean_static

                # 3. Structural Geometry (Pure Layout & Alignment)
                geom = parsed.get("structural_geometry", []) or parsed.get("structural_notes", [])
                if isinstance(geom, list):
                    clean_geom = [
                        str(g).strip() for g in geom
                        if str(g).strip() and not any(kw in str(g).lower() for kw in ("goal", "win", "target", "solution"))
                    ]
                    if clean_geom:
                        base_spec["structural_geometry"] = clean_geom
                        base_spec["structural_notes"] = clean_geom
                        if game_memory is not None:
                            for g in clean_geom:
                                if g not in game_memory.tier1_kinematics_and_topology:
                                    game_memory.tier1_kinematics_and_topology.append(g)

                # 4. Invariants (Factual rules without win speculation)
                inv_list = parsed.get("invariants", [])
                if isinstance(inv_list, list):
                    clean_inv = [
                        str(iv).strip() for iv in inv_list
                        if str(iv).strip() and not any(kw in str(iv).lower() for kw in ("goal", "win", "target", "solution"))
                    ]
                    for iv in clean_inv:
                        if iv not in base_spec["invariants"]:
                            base_spec["invariants"].append(iv)
                        if game_memory is not None and iv not in game_memory.tier1_kinematics_and_topology:
                            game_memory.tier1_kinematics_and_topology.append(iv)

                # Update recorded spec in memory
                if memory is not None:
                    if memory.specs:
                        memory.specs[-1] = base_spec
                    else:
                        memory.record_spec(base_spec)
        except Exception as exc:
            logger.warning(f"Explorer Level Synthesis failed: {exc}; retaining baseline symbolic spec")

        return base_spec

    def research_unconfirmed_primitives(
        self,
        planning_set: PlanningSet,
        memory: EnvironmentSpecMemory,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        game_memory: Any | None = None,
        unconfirmed_actions: list[str] | dict[str, str] | None = None,
    ) -> list[ActionDeclaration]:
        """Dedicated Explorer vision call when >= 40% of available actions remain unconfirmed."""
        if getattr(self.probe_manager, "_primitive_research_done", False):
            return []
        self.probe_manager._primitive_research_done = True

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

        if self.advisor is None or self.config.llm_advisor_backend == "fake":
            return []

        confirmed_actions = dict(self.probe_manager.confirmed_effective_actions)
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_actions.update(game_memory.confirmed_action_effects)

        unconf_dict: dict[str, str] = {}
        if isinstance(unconfirmed_actions, dict):
            unconf_dict.update(unconfirmed_actions)
        elif isinstance(unconfirmed_actions, (list, tuple, set)):
            for act in unconfirmed_actions:
                unconf_dict[str(act).upper()] = "no visible effect (0 cells changed)"
        for act in self.probe_manager.inactive_actions | self.probe_manager.zero_effect_actions:
            unconf_dict.setdefault(act, "no visible effect (0 cells changed)")
        if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
            unconf_dict.update(game_memory.unconfirmed_actions)

        sys_prompt, user_prompt = build_primitive_research_prompt(
            planning_set=planning_set,
            confirmed_actions=confirmed_actions,
            unconfirmed_actions=unconf_dict,
            probe_history=getattr(memory, "probe_history", []),
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
                seqs = parsed.get("targeted_probe_sequences", [])
                probes: list[ActionDeclaration] = []
                for seq in seqs:
                    if isinstance(seq, list):
                        for item in seq:
                            act_u = str(item).upper().strip()
                            if act_u in self.probe_manager.DISCRETE_PROBE_ALLOWED:
                                probes.append(
                                    ActionDeclaration(
                                        action_id=act_u,
                                        data={},
                                        reasoning={
                                            "source": "primitive_research_sequence",
                                            "rationale": "test_unconfirmed_action_precondition",
                                        },
                                    )
                                )
                rem = max(0, self.probe_manager.max_probes - self.probe_manager.total_probes_executed)
                return probes[:rem]
        except Exception as exc:
            logger.warning(f"Explorer primitive research failed: {exc}")
        return []

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

        # Validate schema version and standard collections
        spec.setdefault("schema_version", "v10.env_spec.1")
        spec.setdefault("snapshot_hash", planning_set.grid_hash)
        spec.setdefault("planning_set_id", planning_set.snapshot_id)
        spec.setdefault("coordinate_affordances", [])
        spec.setdefault("object_class_notes", [])
        spec.setdefault("action_surface_notes", [])
        spec.setdefault("invariants", [])

        # Integrate structural_notes into invariants
        if "structural_notes" in spec and isinstance(spec["structural_notes"], list):
            for sn in spec["structural_notes"]:
                if isinstance(sn, str) and sn not in spec["invariants"]:
                    spec["invariants"].append(sn)

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
            if not is_observable_move(act_str):
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

        # Accumulate confirmed invariants across levels (Strict ISO-2: kinematics and topology ONLY, no goals)
        invariants_list = [
            inv for inv in spec.get("invariants", [])
            if isinstance(inv, str) and not any(kw in inv.lower() for kw in ("goal", "win", "target", "curriculum"))
        ]
        if game_memory is not None:
            tier1_2 = list(getattr(game_memory, "tier1_kinematics_and_topology", [])) + list(getattr(game_memory, "tier2_interactions", []))
            for inv_rule in tier1_2:
                if isinstance(inv_rule, str) and not any(kw in inv_rule.lower() for kw in ("goal", "win", "target", "curriculum")):
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
            if is_observable_move(a)
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
        max_coords: int = 2,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        crop_offset: int | None = None,
    ) -> list[ActionDeclaration]:
        """Generate targeted coordinate probes via Qwen hypothesis prompt with deterministic fallback."""
        width = planning_set.grid_dims[1] if len(planning_set.grid_dims) > 1 else 0
        height = planning_set.grid_dims[0] if len(planning_set.grid_dims) > 0 else 0
        probes: list[ActionDeclaration] = []
        eff_crop = crop_offset if crop_offset is not None else getattr(planning_set, "crop_offset", 0)
        tested_coords: set[tuple[int, int]] = set()
        if memory:
            for p in memory.probe_history:
                if p.action_id == "ACTION6" and isinstance(p.action_data, dict):
                    lx = p.action_data.get("local_x")
                    ly = p.action_data.get("local_y")
                    if lx is not None and ly is not None:
                        try:
                            tested_coords.add((int(lx), int(ly)))
                            continue
                        except (ValueError, TypeError):
                            pass
                    x = p.action_data.get("x")
                    y = p.action_data.get("y")
                    if x is not None and y is not None:
                        try:
                            xi, yi = int(x), int(y)
                            rec_crop = p.action_data.get("crop_offset", eff_crop)
                            if rec_crop > 0:
                                tested_coords.add((xi - rec_crop, yi - rec_crop))
                            else:
                                tested_coords.add((xi, yi))
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
            local_prior_coords = sorted(list({
                (c[0], c[1]) for c in tested_coords
                if 0 <= c[0] < width and 0 <= c[1] < height
            }))
            sys_prompt, user_prompt = build_coordinate_hypothesis_prompt(
                planning_set=planning_set,
                num_hypotheses=max_coords,
                prior_tested_coords=local_prior_coords if local_prior_coords else None,
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
                    target_label = str(getattr(coord, "label", "") or getattr(coord, "object_id", ""))
                    probes.append(
                        ActionDeclaration(
                            action_id="ACTION6",
                            data={"x": coord.x, "y": coord.y, "target": target_label},
                            reasoning={
                                "source": "coordinate_candidate_fallback",
                                "label": coord.label,
                                "type": coord.source_type,
                                "target": target_label,
                            },
                        )
                    )
                    if len(probes) >= max_coords:
                        break

        return probes[:min(2, max_coords)]


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


def compute_probe_effect(
    before_snapshot: Any,
    after_obs: dict[str, Any],
    planning_set: Any | None = None,
) -> str:
    """Determine empirical delta using Ranked Multi-Hypothesis Effect Detection."""
    import math
    from collections import Counter
    from v10_agent.arga_lite import extract_arga_snapshot

    raw_grid = after_obs.get("grid")
    if raw_grid is None:
        return "no_grid_observation"

    after_snapshot = extract_arga_snapshot(raw_grid)

    alias_map: dict[str, str] = getattr(planning_set, "object_real_to_alias", {}) if planning_set else {}
    if not alias_map and hasattr(before_snapshot, "object_real_to_alias"):
        alias_map = getattr(before_snapshot, "object_real_to_alias", {})

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

    # 2. Match children using parent displacement as spatial reference if parent moved,
    # or match independently if child moved inside a stationary parent container
    children_b.sort(key=lambda o: -getattr(o, "area", 0))
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
                parent_deltas[c.id] = (dr, dc, best_a)
        else:
            # Parent container remained static: child may move independently inside container
            p_obj = next((p for p in b_objs if p.id == p_id), None)
            best_a = None
            best_dist = float("inf")
            area_tol = 4 if c.area >= 2 else 1
            for a in a_objs:
                if a.id in used_after:
                    continue
                # If child moves inside static parent container, candidate must still be inside that same container
                if p_obj is not None:
                    if not (p_obj.bbox.min_row <= a.centroid.row <= p_obj.bbox.max_row and p_obj.bbox.min_col <= a.centroid.col <= p_obj.bbox.max_col):
                        continue
                if a.color == c.color and abs(a.area - c.area) <= area_tol:
                    dist = math.hypot(a.centroid.row - c.centroid.row, a.centroid.col - c.centroid.col)
                    if dist < best_dist:
                        best_dist = dist
                        best_a = a
            max_dist = max(24.0, c.bbox.height * 2.5, c.bbox.width * 2.5) if c.area >= 2 else 24.0
            if best_a is not None and best_dist <= max_dist:
                dr = int(round(best_a.centroid.row - c.centroid.row))
                dc = int(round(best_a.centroid.col - c.centroid.col))
                dr, dc = filter_displacement_jitter(dr, dc)
                used_after.add(best_a.id)
                if abs(dr) >= 1 or abs(dc) >= 1:
                    matched_moves.append((c, best_a, dr, dc, best_dist))
                    parent_deltas[c.id] = (dr, dc, best_a)

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
                    b_alias = alias_map.get(b.id, b.id)
                    p_id = getattr(b, "parent_id", None)
                    if coherent_children:
                        c_aliases = [alias_map.get(cid, cid) for cid in coherent_children]
                        b_desc = f"{b_alias} ({b.id}) [compound with children {', '.join(c_aliases)}]" if b_alias != b.id else f"{b.id} (compound with children {', '.join(c_aliases)})"
                    elif p_id:
                        p_alias = alias_map.get(p_id, p_id)
                        b_desc = f"{b_alias} ({b.id}) [child of {p_alias}]" if b_alias != b.id else f"{b.id} (child of {p_alias})"
                    else:
                        b_desc = f"{b_alias} ({b.id})" if b_alias != b.id else f"{b.id}"
                    parts.append(f"moved {b_desc} by dy={dr:+d}, dx={dc:+d} ({dir_label})")
            else:
                top_descs = [
                    f"{alias_map.get(b.id, b.id)} ({b.id})" if alias_map.get(b.id) != b.id else str(b.id)
                    for b in top_level[:3]
                ]
                parts.append(f"moved [{', '.join(top_descs)} + {len(top_level)-3} more] by dy={dr:+d}, dx={dc:+d} ({dir_label})")

        if parts:
            return "; ".join(parts[:4])

    # =========================================================================
    # Hypothesis 2: Transfer of selection indicator / dots between STATIONARY containers (State Toggle)
    # =========================================================================
    all_b = before_snapshot.objects if hasattr(before_snapshot, "objects") else []
    all_a = after_snapshot.objects if hasattr(after_snapshot, "objects") else []
    # Dynamic classification: containers are larger structures enclosing/containing smaller indicator tokens
    b_dots = [d for d in all_b if any(p.area >= d.area * 2 for p in all_b if p.id != d.id)]
    a_dots = [d for d in all_a if any(p.area >= d.area * 2 for p in all_a if p.id != d.id)]
    b_subs = [p for p in all_b if any(p.area >= d.area * 2 for d in b_dots if p.id != d.id)]
    a_subs = [p for p in all_a if any(p.area >= d.area * 2 for d in a_dots if p.id != d.id)]

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
        return f"selection indicator transferred: focus marker moved from {l_str} to {g_str} (active entity toggled)"

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
            bbox_str = f"in bbox: cols {min_c}..{max_c} (inclusive), rows {min_r}..{max_r} (inclusive)"
            trans_parts = [f"{c_from}->{c_to} ({cnt} cells)" for (c_from, c_to), cnt in sorted(color_counts.items())]
            if len(changed_cells) >= 50 or (max_r - min_r >= 20 and max_c - min_c >= 20):
                return f"selection indicator / state toggle: global state or active entity toggled ({len(changed_cells)} cells updated across {bbox_str})"
            return f"color transition (stamp/draw): {len(changed_cells)} cells changed color [{', '.join(trans_parts)}] {bbox_str}"

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
    # Hypothesis 6: Zero visible effect fallback
    # =========================================================================
    return "no visible effect (0 cells changed)"


def classify_effect_summary_to_affordance(action_id: str, effect_summary: str) -> dict[str, Any]:
    """Parse textual effect summary into structured ActionAffordance dict."""
    eff_lower = str(effect_summary).lower()
    act_up = str(action_id).upper()

    # 1. MODAL_SELECTION
    if any(k in eff_lower for k in ("selection indicator", "active entity", "focus marker", "state toggle", "mode switch", "toggled", "selection")):
        return {
            "action_id": act_up,
            "effect_class": EffectClass.MODAL_SELECTION.value,
            "parameters": {"active_entity_switched": True},
            "coordination_notes": str(effect_summary),
        }

    # 2. TOPOLOGY_MUTATION
    if any(k in eff_lower for k in ("object count", "spawn", "destroy", "attach", "detach", "created", "deleted", "merged", "split")):
        mut_type = "DESTROY" if any(k in eff_lower for k in ("-1", "destroy", "deleted")) else "SPAWN"
        if "attach" in eff_lower:
            mut_type = "ATTACH"
        elif "detach" in eff_lower:
            mut_type = "DETACH"
        return {
            "action_id": act_up,
            "effect_class": EffectClass.TOPOLOGY_MUTATION.value,
            "parameters": {"mutation_type": mut_type},
            "coordination_notes": str(effect_summary),
        }

    # 3. PALETTE_TRANSITION
    if any(k in eff_lower for k in ("color transition", "stamp", "draw", "cells changed color", "palette", "recolor")):
        m_trans = re.search(r"(\d+)->(\d+)", effect_summary)
        c_from = int(m_trans.group(1)) if m_trans else 0
        c_to = int(m_trans.group(2)) if m_trans else 0
        m_cnt = re.search(r"(\d+)\s+cells", effect_summary)
        cnt = int(m_cnt.group(1)) if m_cnt else 1
        return {
            "action_id": act_up,
            "effect_class": EffectClass.PALETTE_TRANSITION.value,
            "parameters": {"affected_alias": "grid", "color_from": c_from, "color_to": c_to, "cells_count": cnt},
            "coordination_notes": str(effect_summary),
        }

    # 4. KINEMATIC (default for movements / generic confirmed actions)
    obs_dy, obs_dx = parse_axis_components(effect_summary)
    dy = obs_dy if obs_dy is not None else 0
    dx = obs_dx if obs_dx is not None else 0
    if obs_dy is None and obs_dx is None:
        if "up" in eff_lower:
            dy = -1
        elif "down" in eff_lower:
            dy = 1
        elif "left" in eff_lower:
            dx = -1
        elif "right" in eff_lower:
            dx = 1
    alias = token_after_keyword(effect_summary, "moved") or "actor"
    return {
        "action_id": act_up,
        "effect_class": EffectClass.KINEMATIC.value,
        "parameters": {"affected_alias": alias, "dy": dy, "dx": dx},
        "coordination_notes": str(effect_summary),
    }


def classify_probe_affordance(
    before_snapshot: Any,
    after_obs: dict[str, Any] | Any,
    action_id: str = "ACTION1",
    planning_set: Any | None = None,
) -> dict[str, Any] | None:
    """Classify empirical frame delta into one of 4 invariant ActionAffordance classes."""
    import math
    from v10_agent.arga_lite import extract_arga_snapshot

    raw_grid = after_obs.get("grid") if isinstance(after_obs, dict) else getattr(after_obs, "grid", None)
    if raw_grid is None:
        return None

    b_grid = getattr(before_snapshot, "grid", None)
    if b_grid is not None and isinstance(b_grid, (list, tuple)) and isinstance(raw_grid, (list, tuple)):
        if b_grid == raw_grid:
            return None

    after_snapshot = extract_arga_snapshot(raw_grid) if not hasattr(after_obs, "objects") else after_obs

    alias_map: dict[str, str] = getattr(planning_set, "object_real_to_alias", {}) if planning_set else {}
    if not alias_map and hasattr(before_snapshot, "object_real_to_alias"):
        alias_map = getattr(before_snapshot, "object_real_to_alias", {})

    b_objs = before_snapshot.objects if hasattr(before_snapshot, "objects") else []
    a_objs = after_snapshot.objects if hasattr(after_snapshot, "objects") else []

    # 1. Kinematic: Check rigid translation of components
    used_after: set[str] = set()
    matched_moves: list[tuple[Any, Any, int, int, float]] = []

    for b in b_objs:
        best_a = None
        best_dist = float("inf")
        for a in a_objs:
            if a.id in used_after:
                continue
            if a.color == b.color and abs(a.area - b.area) <= 1:
                dist = math.hypot(a.centroid.row - b.centroid.row, a.centroid.col - b.centroid.col)
                if dist < best_dist:
                    best_dist = dist
                    best_a = a
        if best_a is not None and best_dist <= 24.0:
            dr = int(round(best_a.centroid.row - b.centroid.row))
            dc = int(round(best_a.centroid.col - b.centroid.col))
            dr, dc = filter_displacement_jitter(dr, dc)
            if abs(dr) >= 1 or abs(dc) >= 1:
                used_after.add(best_a.id)
                matched_moves.append((b, best_a, dr, dc, best_dist))

    # Check if object counts changed (Topology Mutation)
    if len(b_objs) != len(a_objs):
        diff = len(a_objs) - len(b_objs)
        mut_type = "SPAWN" if diff > 0 else "DESTROY"
        target_alias = "unknown"
        if diff > 0 and a_objs:
            new_objs = [a for a in a_objs if not any(b.color == a.color and abs(b.area - a.area) <= 1 for b in b_objs)]
            if new_objs:
                target_alias = alias_map.get(new_objs[0].id, new_objs[0].id)
        elif diff < 0 and b_objs:
            lost_objs = [b for b in b_objs if not any(a.color == b.color and abs(a.area - b.area) <= 1 for a in a_objs)]
            if lost_objs:
                target_alias = alias_map.get(lost_objs[0].id, lost_objs[0].id)
        return ActionAffordance(
            action_id=action_id,
            effect_class=EffectClass.TOPOLOGY_MUTATION.value,
            parameters={"mutation_type": mut_type, "target_alias": target_alias, "count_delta": diff},
            coordination_notes=f"object count changed by {diff:+d} ({mut_type.lower()})",
        ).to_dict()

    # 2. Modal Selection: Indicator Transfer / Focus Marker / State Toggle
    # Evaluated before general kinematic motion to ensure indicator transfer between stationary
    # containers is recognized as focus switching rather than rigid translation.
    b_dots = [d for d in b_objs if any(p.area >= d.area * 2 for p in b_objs if p.id != d.id)]
    a_dots = [d for d in a_objs if any(p.area >= d.area * 2 for p in a_objs if p.id != d.id)]
    b_subs = [p for p in b_objs if any(p.area >= d.area * 2 for d in b_dots if p.id != d.id)]
    a_subs = [p for p in a_objs if any(p.area >= d.area * 2 for d in a_dots if p.id != d.id)]

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
            lost_from.append(alias_map.get(b_sub.id, b_sub.id))
        elif b_cnt == 0 and a_cnt >= 1:
            gained_in.append(alias_map.get(b_sub.id, b_sub.id))

    if lost_from and gained_in:
        return ActionAffordance(
            action_id=action_id,
            effect_class=EffectClass.MODAL_SELECTION.value,
            parameters={
                "active_entity_switched": True,
                "cycle_entities": [lost_from[0], gained_in[0]],
                "source": lost_from[0],
                "target": gained_in[0],
            },
            coordination_notes=f"selection indicator transferred from {lost_from[0]} to {gained_in[0]} (active entity toggled)",
        ).to_dict()

    if matched_moves:
        vector_groups: dict[tuple[int, int], list[tuple[Any, Any]]] = {}
        for b, a, dr, dc, _ in matched_moves:
            vector_groups.setdefault((dr, dc), []).append((b, a))

        sorted_groups = sorted(vector_groups.items(), key=lambda item: -sum(b.area for b, _ in item[1]))
        (top_dr, top_dc), top_pairs = sorted_groups[0]
        top_level = [b for b, _ in top_pairs if getattr(b, "parent_id", None) is None or b.parent_id not in {b.id for b, _ in top_pairs}]
        top_level.sort(key=lambda o: -getattr(o, "area", 0))
        primary_b = top_level[0] if top_level else top_pairs[0][0]
        primary_alias = alias_map.get(primary_b.id, primary_b.id)

        all_disps = []
        for (v_dr, v_dc), v_pairs in sorted_groups:
            for b, _ in v_pairs:
                all_disps.append({"alias": alias_map.get(b.id, b.id), "dy": v_dr, "dx": v_dc})

        params: dict[str, Any] = {
            "affected_alias": primary_alias,
            "dy": top_dr,
            "dx": top_dc,
        }
        if len(all_disps) > 1:
            params["displacements"] = all_disps

        dir_label = describe_vector(top_dr, top_dc)
        notes = f"moved {primary_alias} by dy={top_dr:+d}, dx={top_dc:+d} ({dir_label})"
        if len(sorted_groups) > 1:
            secondary_parts = []
            for (s_dr, s_dc), s_pairs in sorted_groups[1:]:
                s_aliases = [alias_map.get(b.id, b.id) for b, _ in s_pairs[:2]]
                secondary_parts.append(f"moved [{', '.join(s_aliases)}] by dy={s_dr:+d}, dx={s_dc:+d}")
            notes += f"; secondary shifts: {'; '.join(secondary_parts)}"
        elif len(top_pairs) > 1:
            notes += f" [synchronous movement of {len(top_pairs)} objects]"

        return ActionAffordance(
            action_id=action_id,
            effect_class=EffectClass.KINEMATIC.value,
            parameters=params,
            coordination_notes=notes,
        ).to_dict()

    # 3. Palette Transition or Global Modal Toggle
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

            # Global indicator or wide service bar update -> MODAL_SELECTION
            if len(changed_cells) >= 50 or (max_r - min_r >= 20 and max_c - min_c >= 20):
                return ActionAffordance(
                    action_id=action_id,
                    effect_class=EffectClass.MODAL_SELECTION.value,
                    parameters={"active_entity_switched": True, "cells_count": len(changed_cells)},
                    coordination_notes=f"global state or active entity toggled ({len(changed_cells)} cells updated)",
                ).to_dict()

            # Otherwise PALETTE_TRANSITION
            sorted_trans = sorted(color_counts.items(), key=lambda x: -x[1])
            top_from, top_to = sorted_trans[0][0] if sorted_trans else (0, 0)
            affected_alias = "grid"
            for b in b_objs:
                if b.bbox.min_row <= min_r and max_r <= b.bbox.max_row and b.bbox.min_col <= min_c and max_c <= b.bbox.max_col:
                    affected_alias = alias_map.get(b.id, b.id)
                    break

            trans_parts = [f"{c_from}->{c_to} ({cnt} cells)" for (c_from, c_to), cnt in sorted_trans]
            return ActionAffordance(
                action_id=action_id,
                effect_class=EffectClass.PALETTE_TRANSITION.value,
                parameters={
                    "affected_alias": affected_alias,
                    "color_from": top_from,
                    "color_to": top_to,
                    "cells_count": len(changed_cells),
                },
                coordination_notes=f"color transition: {len(changed_cells)} cells changed color [{', '.join(trans_parts)}]",
            ).to_dict()

    if gained_in or lost_from:
        target = gained_in[0] if gained_in else lost_from[0]
        return ActionAffordance(
            action_id=action_id,
            effect_class=EffectClass.MODAL_SELECTION.value,
            parameters={"active_entity_switched": True, "target": target},
            coordination_notes=f"selection indicator changed at {target}",
        ).to_dict()

    return None


class PrimitiveProbeManager:
    """Manages systematic empirical probe generation, cyclic combinatorial exploration, and physical effect recording."""

    DISCRETE_PROBE_ALLOWED = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}

    def __init__(
        self,
        max_probes: int = 16,
        max_invariant_probes: int = 3,
        max_steps_per_probe: int = 2,
    ):
        self.max_probes = max_probes
        self.max_invariant_probes = max_invariant_probes
        self.max_steps_per_probe = max_steps_per_probe
        self.invariant_probes_count: int = 0
        self.probed_actions: dict[str, str] = {}
        self.confirmed_effective_actions: dict[str, str] = {}
        self.inactive_actions: set[str] = set()
        self.zero_effect_actions: set[str] = set()
        self.conditional_candidate_actions: set[str] = set()
        self.retested_actions: set[str] = set()
        self.available_discrete_actions: set[str] = set()
        self.total_probes_executed: int = 0
        self._initial_sweep_planned: bool = False
        self._initial_sweep_done: bool = False
        self._primitive_research_done: bool = False
        self._chains_attempted: set[tuple[Any, ...]] = set()

    def handle_level_transition(self, confirmed_action_effects: dict[str, str] | None = None) -> None:
        """Reset per-level probe state while preserving confirmed cross-level kinematics."""
        self.total_probes_executed = 0
        self.invariant_probes_count = 0
        self._initial_sweep_planned = False
        self._initial_sweep_done = False
        self._primitive_research_done = False
        self._chains_attempted.clear()
        self.retested_actions.clear()
        self.confirmed_effective_actions.clear()
        self.inactive_actions.clear()
        self.zero_effect_actions.clear()
        self.conditional_candidate_actions.clear()
        self.available_discrete_actions.clear()

        if isinstance(confirmed_action_effects, dict):
            for act, eff in confirmed_action_effects.items():
                act_name = str(act).upper()
                eff_str = str(eff).lower() if isinstance(eff, str) else ""
                is_pure_motion = (
                    any(k in eff_str for k in ("moved", "moves", "dy=", "dx=", "displacement", "shift"))
                    and any(d in eff_str for d in ("up", "down", "left", "right"))
                    and not any(m in eff_str for m in ("selection indicator", "toggle", "active entity toggled", "selection"))
                    and act_name not in NON_VECTOR_ACTION_IDS
                    and is_observable_move(act_name)
                )
                if is_pure_motion:
                    self.confirmed_effective_actions[act_name] = "confirmed_reusable_action"
        elif confirmed_action_effects is not None:
            logger.warning(
                f"PrimitiveProbeManager.handle_level_transition received invalid type {type(confirmed_action_effects).__name__}; "
                f"expected dict[str, str]. Preserving unprobed status for actions."
            )

        # Invariant: Modal switches, context-dependent actions, and any unconfirmed actions
        # MUST remain/become unconfirmed (inactive_actions) to guarantee primary probing on the new level grid.
        for act in self.DISCRETE_PROBE_ALLOWED:
            if act not in self.confirmed_effective_actions:
                self.inactive_actions.add(act)

    def is_action_effective(self, effect_summary: str) -> bool:
        """Determine if observed effect indicates active physical or visual response."""
        if not effect_summary or "inactive" in effect_summary or "conditional" in effect_summary or "no_grid" in effect_summary or "no visible effect" in effect_summary or "zero observable" in effect_summary:
            return False
        return any(kw in effect_summary for kw in (
            "moved", "color transition", "selection indicator", "object count changed",
            "confirmed_reusable_action", "active entity", "state toggle", "spawn", "destroy",
            "attach", "detach", "cells changed color",
        ))

    def is_motion_action(self, action_id: str) -> bool:
        """Check if confirmed action induces directional translation."""
        eff = self.confirmed_effective_actions.get(action_id, "")
        return ("moved" in eff and any(d in eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))) or "confirmed_reusable_action" in eff

    def get_dynamic_reprobes(
        self,
        triggering_action_id: str,
        effect_summary: str,
        max_steps: int | None = None,
        allowed_actions: Sequence[str] | None = None,
        planning_set: Any | None = None,
    ) -> list[ActionDeclaration]:
        if self._initial_sweep_planned and not self._initial_sweep_done:
            return []

        if allowed_actions is not None:
            allowed_set = {str(a).upper() for a in allowed_actions}
            allowed_discrete = {a for a in self.DISCRETE_PROBE_ALLOWED if a in allowed_set}
        elif self._initial_sweep_planned:
            allowed_discrete = set(self.available_discrete_actions)
            allowed_set = set(self.available_discrete_actions)
        else:
            allowed_discrete = set(self.available_discrete_actions) if self.available_discrete_actions else set(self.DISCRETE_PROBE_ALLOWED)
            allowed_set = allowed_discrete | {"ACTION6"}

        if self.total_probes_executed >= self.max_probes or self.invariant_probes_count >= self.max_invariant_probes:
            # Invariant verification budget exhausted (3 attempts): yield invariants without further probing
            for act in list(self.inactive_actions):
                if act in allowed_discrete and act not in self.confirmed_effective_actions:
                    self.confirmed_effective_actions[act] = "unverified_invariant"
            return []

        reprobes: list[ActionDeclaration] = []
        rem = self.max_probes - self.total_probes_executed
        if max_steps is not None:
            rem = min(max_steps, rem)

        is_modal_toggle = (
            "selection indicator" in effect_summary
            or "active entity toggled" in effect_summary
            or "state toggle" in effect_summary
        )

        # Case 1: Inactive discrete actions to retest in new context
        if self.inactive_actions:
            for act in list(self.inactive_actions):
                if act in allowed_discrete and act != triggering_action_id and act not in self.retested_actions and len(reprobes) < rem:
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
            motion_candidates = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if a in allowed_discrete]
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

        # Case 3: Invariant of Orthogonal Modality
        # If a vector action yielded zero effect, probe modal switches (ACTION5, ACTION6)
        # and re-test the vector direction under the newly toggled mode.
        is_zero_effect = (
            not self.is_action_effective(effect_summary)
            or "no visible effect" in effect_summary
            or "0 cells changed" in effect_summary
            or "zero" in effect_summary
        )
        is_vector_action = triggering_action_id in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") and triggering_action_id in allowed_discrete
        if is_vector_action and is_zero_effect:
            # This is an invariant-verification probe: a vector action had no effect,
            # so we try modal switches. Count it toward invariant budget.
            self.invariant_probes_count += 1
            for modal_act in ("ACTION5", "ACTION6"):
                if modal_act not in allowed_set:
                    continue
                if modal_act != triggering_action_id and len(reprobes) < rem:
                    modal_data: dict[str, Any] = {}
                    if modal_act == "ACTION6" and planning_set is not None:
                        if getattr(planning_set, "coordinate_candidates", None):
                            c0 = planning_set.coordinate_candidates[0]
                            modal_data = {"x": int(c0.x), "y": int(c0.y)}
                        elif getattr(planning_set, "objects", None):
                            o0 = planning_set.objects[0]
                            modal_data = {"x": int(round(o0.centroid.col)), "y": int(round(o0.centroid.row))}
                    reprobes.append(
                        ActionDeclaration(
                            action_id=modal_act,
                            data=modal_data,
                            reasoning={
                                "source": "dynamic_modal_switch_probe",
                                "trigger": f"zero_effect_from_{triggering_action_id}",
                                "rationale": "toggle_mode_to_unlock_orthogonal_degree_of_freedom",
                            },
                        )
                    )
                    if len(reprobes) < rem:
                        reprobes.append(
                            ActionDeclaration(
                                action_id=triggering_action_id,
                                data={},
                                reasoning={
                                    "source": "dynamic_modal_vector_retest",
                                    "trigger": f"post_modal_switch_by_{modal_act}",
                                    "rationale": f"retest_{triggering_action_id}_under_new_mode",
                                },
                            )
                        )

        # Case 4: Post-Click Verification Probe
        # When an object-targeted click or ACTION6 mutated the grid, but the semantic effect is ambiguous
        # (e.g. non-kinematic change, palette/indicator mutation, or unrecognized state change),
        # immediately probe with available primitive motion/action (ACTION1..ACTION4) to test
        # if clicking this object unlocked movement, opened a passage, or activated an entity.
        is_click_action = triggering_action_id in ("ACTION6", "CLICK")
        is_effective = self.is_action_effective(effect_summary)
        is_pure_motion = ("moved" in effect_summary and any(d in effect_summary for d in ("UP", "DOWN", "LEFT", "RIGHT")))
        if is_click_action and is_effective and not is_pure_motion and not is_modal_toggle and len(reprobes) < rem:
            motion_candidates = [act for act in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if act in allowed_discrete]
            for act in motion_candidates[:2]:
                if not any(p.action_id == act for p in reprobes) and len(reprobes) < rem:
                    reprobes.append(
                        ActionDeclaration(
                            action_id=act,
                            data={},
                            reasoning={
                                "source": "post_click_verification_probe",
                                "trigger": f"click_mutated_grid_by_{triggering_action_id}",
                                "rationale": "verify_if_click_unlocked_movement_or_mechanic",
                            },
                        )
                    )

        if reprobes:
            if max_steps is not None:
                return reprobes[:max_steps]
            return reprobes
        return []

    def schedule_falsification_reprobe(
        self,
        falsified_action_ids: list[str] | None = None,
        modal_switch_actions: Sequence[str] | None = None,
    ) -> list[ActionDeclaration]:
        """Schedule immediate micro-reprobing of primitive and modal actions on clean board following empirical falsification."""
        if falsified_action_ids:
            for act in falsified_action_ids:
                self.confirmed_effective_actions.pop(act, None)
                self.inactive_actions.discard(act)
                self.zero_effect_actions.discard(act)
                self.retested_actions.discard(act)

        self._initial_sweep_planned = False
        self.total_probes_executed = 0

        modals = list(modal_switch_actions) if modal_switch_actions is not None else ["ACTION5", "ACTION6"]
        if self.available_discrete_actions:
            modals = [m for m in modals if m in self.available_discrete_actions]

        probes: list[ActionDeclaration] = []
        # 1. Baseline discrete motion actions
        vector_candidates = [
            a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
            if not self.available_discrete_actions or a in self.available_discrete_actions
        ]
        for act in vector_candidates:
            probes.append(
                ActionDeclaration(
                    action_id=act,
                    data={},
                    reasoning={
                        "source": "falsification_micro_reprobe",
                        "rationale": "retest_baseline_kinematics_after_falsification",
                    },
                )
            )

        # 2. Modal selectors to test discrete mode switching
        for m_act in modals:
            probes.append(
                ActionDeclaration(
                    action_id=m_act,
                    data={},
                    reasoning={
                        "source": "falsification_modal_reprobe",
                        "rationale": "toggle_mode_selector_after_falsification",
                    },
                )
            )
            # Re-test vector directions under switched mode to unlock degrees of freedom
            targets_to_retest = falsified_action_ids if falsified_action_ids else vector_candidates[:2]
            for act in targets_to_retest:
                if act in vector_candidates:
                    probes.append(
                        ActionDeclaration(
                            action_id=act,
                            data={},
                            reasoning={
                                "source": "falsification_modal_vector_retest",
                                "trigger": f"post_modal_switch_by_{m_act}",
                                "rationale": f"retest_{act}_under_switched_mode",
                            },
                        )
                    )

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

        for act in acts:
            act_u = str(act).upper()
            if act_u in self.DISCRETE_PROBE_ALLOWED:
                self.available_discrete_actions.add(act_u)

        # Prune any inactive_actions not actually in the level's allowed discrete actions
        self.inactive_actions = {a for a in self.inactive_actions if a in self.available_discrete_actions}

        probes: list[ActionDeclaration] = []

        # Sort unconfirmed discrete actions deterministically: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5...
        unconfirmed = [
            str(act).upper() for act in acts
            if str(act).upper() in self.DISCRETE_PROBE_ALLOWED
            and str(act).upper() not in confirmed
            and str(act).upper() not in tested_in_this_level
        ]
        unconfirmed_sorted = sorted(
            unconfirmed,
            key=lambda a: (int(a.replace("ACTION", "")) if a.startswith("ACTION") and a.replace("ACTION", "").isdigit() else 999, a)
        )

        for act_str in unconfirmed_sorted:
            if len(probes) < limit:
                probes.append(
                    ActionDeclaration(
                        action_id=act_str,
                        data={},
                        reasoning={"source": "primitive_probe", "type": "initial_sweep"},
                    )
                )

        self._initial_sweep_planned = True
        self._initial_sweep_done = (len(probes) == 0)
        return probes

    def plan_targeted_coordinate_probes(
        self,
        planning_set: PlanningSet,
        available_actions: Sequence[str] | None = None,
        memory: EnvironmentSpecMemory | None = None,
        affordances: list[dict[str, Any]] | None = None,
        crop_offset: int | None = None,
    ) -> list[ActionDeclaration]:
        """Generate targeted coordinate probes based on Explorer affordance suggestions."""
        allowed = list(available_actions) if available_actions is not None else list(planning_set.allowed_action_ids)
        coord_act = "ACTION6" if "ACTION6" in allowed else (allowed[0] if allowed else "ACTION6")

        eff_crop = crop_offset if crop_offset is not None else getattr(planning_set, "crop_offset", 0)
        tested_coords = set()
        if memory:
            for p in memory.probe_history:
                if p.action_id == coord_act and isinstance(p.action_data, dict):
                    lx = p.action_data.get("local_x")
                    ly = p.action_data.get("local_y")
                    if lx is not None and ly is not None:
                        try:
                            tested_coords.add((int(lx), int(ly)))
                            continue
                        except (ValueError, TypeError):
                            pass
                    x = p.action_data.get("x")
                    y = p.action_data.get("y")
                    if x is not None and y is not None:
                        try:
                            xi, yi = int(x), int(y)
                            rec_crop = p.action_data.get("crop_offset", eff_crop)
                            if rec_crop > 0:
                                tested_coords.add((xi - rec_crop, yi - rec_crop))
                            else:
                                tested_coords.add((xi, yi))
                        except (ValueError, TypeError):
                            pass
        probes: list[ActionDeclaration] = []
        if affordances:
            for aff in affordances:
                x = int(aff.get("x", -1))
                y = int(aff.get("y", -1))
                if x >= 0 and y >= 0 and (x, y) not in tested_coords and len(probes) < 2:
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
                if coord.source_type == "object_centroid" and (coord.x, coord.y) not in tested_coords and len(probes) < 2:
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
        return probes[:2]

    def record_probe_result(
        self,
        action_id: str,
        action_data: dict[str, Any],
        before_snapshot: Any,
        after_obs: dict[str, Any],
        memory: EnvironmentSpecMemory | None = None,
        planning_set: Any | None = None,
    ) -> ProbeRecord:
        """Compute delta and record probe observation into EnvironmentSpecMemory."""
        self.total_probes_executed += 1
        effect_str = compute_probe_effect(before_snapshot, after_obs, planning_set=planning_set)
        self.probed_actions[action_id] = effect_str
        affordance = classify_probe_affordance(before_snapshot, after_obs, action_id=action_id, planning_set=planning_set)

        if self.is_action_effective(effect_str):
            existing_eff = self.confirmed_effective_actions.get(action_id)
            if existing_eff and existing_eff != "confirmed_reusable_action":
                existing_parts = [p.strip() for p in existing_eff.split(" | ") if p.strip()]
                if effect_str.strip() not in existing_parts:
                    existing_parts.append(effect_str.strip())
                self.confirmed_effective_actions[action_id] = " | ".join(existing_parts)
            else:
                self.confirmed_effective_actions[action_id] = effect_str
            self.inactive_actions.discard(action_id)
            self.zero_effect_actions.discard(action_id)
            self.conditional_candidate_actions.discard(action_id)
        else:
            if action_id not in self.confirmed_effective_actions:
                self.inactive_actions.add(action_id)
                self.zero_effect_actions.add(action_id)
                if action_id in self.DISCRETE_PROBE_ALLOWED:
                    self.conditional_candidate_actions.add(action_id)

        # Check if initial sweep has completed across all available discrete actions
        if self._initial_sweep_planned and not self._initial_sweep_done:
            if self.available_discrete_actions:
                all_tested = all(
                    (a in self.probed_actions or a in self.confirmed_effective_actions)
                    for a in self.available_discrete_actions
                )
                if all_tested or self.total_probes_executed >= self.max_probes:
                    self._initial_sweep_done = True
            else:
                self._initial_sweep_done = True

        probe_id = f"probe_{len(memory.probe_history)}" if memory else f"probe_{self.total_probes_executed}"
        record = ProbeRecord(
            probe_id=probe_id,
            action_id=action_id,
            action_data=action_data,
            observed_effect=effect_str,
            confidence=0.9 if self.is_action_effective(effect_str) else 0.5,
            affordance=affordance,
        )
        if memory:
            memory.record_probe(record)
        return record

    def get_next_combinatorial_chain(self) -> list[ActionDeclaration]:
        """Generate next chain of actions to test remaining conditional/inactive discrete actions in modified states."""
        inactive_discrete = [
            a for a in sorted(list(self.inactive_actions | self.conditional_candidate_actions))
            if a in self.DISCRETE_PROBE_ALLOWED and a not in self.confirmed_effective_actions
        ]
        if not inactive_discrete:
            return []
        if self.total_probes_executed >= self.max_probes:
            return []

        motion_actions = [a for a in self.confirmed_effective_actions if self.is_motion_action(a)]
        if not motion_actions:
            motion_actions = [a for a in self.confirmed_effective_actions if a in self.DISCRETE_PROBE_ALLOWED]
        if not motion_actions:
            motion_actions = [
                a for a in sorted(self.available_discrete_actions)
                if a not in inactive_discrete
            ]
        if not motion_actions and self.available_discrete_actions:
            motion_actions = sorted(list(self.available_discrete_actions))

        rem = self.max_probes - self.total_probes_executed
        if rem <= 0:
            return []

        for m_act in motion_actions:
            chain_key = (m_act, tuple(inactive_discrete))
            if chain_key not in self._chains_attempted:
                self._chains_attempted.add(chain_key)
                chain_acts = [m_act] + [a for a in inactive_discrete if a != m_act]
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
                if m_act == inact:
                    continue
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

