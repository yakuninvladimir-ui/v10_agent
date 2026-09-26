"""Coder Prompt Builder (Call 2 Family).

Constructs prompts strictly quarantined from level goals, Solver hypotheses, and EpistemicMemory.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from v10_agent.memory_contours import FORBIDDEN_GOAL_PATTERNS_IN_CODER, SyntaxErrorRecord

CODER_SYSTEM_PROMPT = """\
You are a Python DSL (Domain Specific Language) generator.
Think step by step and reason through the environment mechanics, object structures, and coordinate affordances before writing the code.
Your task is to translate environment specifications into a pure Python module defining the available game actions.

CONSTRAINTS:
- You DO NOT know the puzzle goal. Do not attempt to solve it.
- Generate pure functions that ONLY call `api.declare_environment_action(...)`.
- Allowed imports: math, typing, dataclasses, enum, collections. NO os, sys, eval, exec.
- Preserve all previously confirmed actions from earlier levels (e.g., action1..action4). The game mechanics are persistent across levels and only augment or expand.
- You MUST implement functions ONLY for the confirmed effective actions listed in the available actions / RESEARCHED ACTION MECHANICS. Do NOT generate functions for unconfirmed or zero-effect actions.
- Do NOT omit, rename, or delete existing confirmed actions. Augment the DSL with newly discovered actions (e.g. adding action5 or action6).
- Function names must be canonical: `action1`, `action2`, ..., `action6`.
- Coordinate and click actions (e.g., ACTION6) must accept target: str = "", x: int | None = None, y: int | None = None where x is the column (horizontal) and y is the row (vertical). If target is provided, call api.click_object(target, x=x, y=y). Otherwise call api.declare_environment_action(action_id='ACTION6', data={'x': int(x or 0), 'y': int(y or 0)}). Also define click(api, target='', x=None, y=None) delegating to action6.

OUTPUT FORMAT:
Provide a single ```python ... ``` block containing the DSL module.
Do NOT provide a JSON manifest: signatures, parameters, and docstrings are extracted automatically from your Python functions.
"""

SANDBOX_API_DOC = """\
The SandboxAPI injected into your functions exposes:
  api.declare_environment_action(
      action_id: str,
      data: dict | None = None,
  ) -> EffectDeclaration
  api.click_object(
      target: str,
      x: int | None = None,
      y: int | None = None,
  ) -> EffectDeclaration
"""


def build_coder_prompts(
    env_spec: dict[str, Any],
    syntax_errors: Sequence[SyntaxErrorRecord] | None = None,
    game_memory: Any | None = None,
    planning_set: Any | None = None,
    has_image: bool = True,
) -> tuple[str, str]:
    """Construct (system_prompt, user_prompt) for the DSL Coder Agent."""
    compact_spec = dict(env_spec)
    if "coordinate_affordances" in compact_spec and len(compact_spec["coordinate_affordances"]) > 20:
        compact_spec["coordinate_affordances"] = compact_spec["coordinate_affordances"][:20]

    # ISO-2 Quarantine: ensure no goal rules or win conditions ever reach Coder prompt
    if "invariants" in compact_spec and isinstance(compact_spec["invariants"], list):
        compact_spec["invariants"] = [
            inv for inv in compact_spec["invariants"]
            if isinstance(inv, str) and not any(p.search(inv) for p in FORBIDDEN_GOAL_PATTERNS_IN_CODER)
        ]

    if "curriculum_history" in compact_spec and isinstance(compact_spec["curriculum_history"], list):
        compact_spec["curriculum_history"] = [
            h for h in compact_spec["curriculum_history"]
            if not any(p.search(str(h)) for p in FORBIDDEN_GOAL_PATTERNS_IN_CODER)
        ]

    # Resolve confirmed effective available actions from env_spec, game_memory, or planning_set
    available_actions: list[str] = []
    if "action_affordances" in compact_spec and compact_spec["action_affordances"]:
        for aff in compact_spec["action_affordances"]:
            if isinstance(aff, dict):
                act_id = str(aff.get("action_id", "")).upper()
                if act_id and act_id not in ("RESET", "ACTION7") and act_id not in available_actions:
                    available_actions.append(act_id)
    if "available_actions" in compact_spec and compact_spec["available_actions"]:
        for a in compact_spec["available_actions"]:
            act_up = str(a).upper()
            if act_up not in ("RESET", "ACTION7") and act_up not in available_actions:
                available_actions.append(act_up)
    elif "researched_actions" in compact_spec and compact_spec["researched_actions"]:
        for a in compact_spec["researched_actions"]:
            if isinstance(a, dict) and "action_id" in a:
                act_up = str(a["action_id"]).upper()
                if act_up not in ("RESET", "ACTION7") and act_up not in available_actions:
                    if not any(neg in str(a.get("effect_summary", "")).lower() for neg in ("no visible effect", "unconfirmed", "inactive", "zero effect")):
                        available_actions.append(act_up)
    elif planning_set is not None and getattr(planning_set, "allowed_action_ids", None):
        unconfirmed = set()
        if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
            unconfirmed = {str(a).upper() for a in game_memory.unconfirmed_actions}
        available_actions = [
            str(a).upper() for a in planning_set.allowed_action_ids
            if str(a).upper() not in ("RESET", "ACTION7") and str(a).upper() not in unconfirmed
        ]
    # Always ensure confirmed actions from game_memory are included to preserve earlier levels
    if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
        for act in game_memory.confirmed_action_effects:
            act_up = str(act).upper()
            if act_up not in ("RESET", "ACTION7") and act_up not in available_actions:
                available_actions.append(act_up)

    # Strictly filter by planning_set.allowed_action_ids and exclude unconfirmed actions
    if planning_set is not None and getattr(planning_set, "allowed_action_ids", None):
        allowed_set = {str(a).upper() for a in planning_set.allowed_action_ids}
        available_actions = [a for a in available_actions if a in allowed_set]
    if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
        unconfirmed = {str(a).upper() for a in game_memory.unconfirmed_actions}
        available_actions = [a for a in available_actions if a not in unconfirmed]

    image_note = (
        "coder_raw_frame.png is the exact same frame as coder_annotated_frame.png, but without object annotations.\n\n"
        if has_image
        else ""
    )

    sections: list[str] = []
    if image_note:
        sections.append(image_note.strip())

    spec_lines = []
    affordances = compact_spec.get("action_affordances", [])
    if affordances:
        spec_lines.append("ACTION AFFORDANCES (Observed State Mutations):")
        for aff in affordances:
            if isinstance(aff, dict):
                act_id = aff.get("action_id", "")
                eff_cls = aff.get("effect_class", "KINEMATIC")
                params = aff.get("parameters", {})
                notes = aff.get("coordination_notes", "")
                param_parts = [f"{k}={v}" for k, v in sorted(params.items())]
                p_str = f" ({', '.join(param_parts)})" if param_parts else ""
                notes_str = f" - {notes}" if notes else ""
                spec_lines.append(f"- {act_id} [{eff_cls}]:{p_str}{notes_str}")
    researched = compact_spec.get("researched_actions", [])
    if researched and not affordances:
        spec_lines.append("RESEARCHED ACTION MECHANICS:")
        for a in researched:
            if isinstance(a, dict) and "action_id" in a:
                eff = a.get("effect_summary", "available action")
                spec_lines.append(f"- {a['action_id']}: {eff}")

    invariants = compact_spec.get("invariants", [])
    if invariants:
        spec_lines.append("\nPHYSICS INVARIANTS:")
        for inv in invariants:
            spec_lines.append(f"- {inv}")

    env_spec_text = "\n".join(spec_lines) if spec_lines else "No prior probe research available."

    sections.extend([
        "ENVIRONMENT SPECIFICATION (Observed Action Mechanics):",
        env_spec_text,
        "",
    ])

    if available_actions:
        sections.extend([
            "CONFIRMED EFFECTIVE ACTIONS (Define a function ONLY for each action in this list):",
            "\n".join(f"- {act}" for act in sorted(set(available_actions))),
            "",
        ])

    sections.extend([
        "Sandbox API Contract:",
        SANDBOX_API_DOC,
    ])

    if game_memory is not None and hasattr(game_memory, "format_empirical_context"):
        emp_ctx = game_memory.format_empirical_context(include_curriculum=False)
        if emp_ctx:
            sections.extend([
                "",
                "EMPIRICAL FACTS DISCOVERED ACROSS PREVIOUS LEVELS:",
                emp_ctx,
            ])

    if syntax_errors:
        sections.append("PREVIOUS COMPILATION / STATIC VALIDATION DIAGNOSTICS (FIX THESE ERRORS):")
        for idx, err in enumerate(syntax_errors, 1):
            sections.append(f"--- Attempt #{idx} Failure ---")
            sections.append(f"Error Type: {err.error_type}")
            sections.append(f"Message: {err.error_message}")
            if err.diagnostics:
                sections.append("Diagnostics:\n" + "\n".join(f"- {d}" for d in err.diagnostics))
            if err.source_code:
                sections.append(f"Faulty Source Snippet:\n```python\n{err.source_code[:1500]}\n```")

    user_instructions = """\
Generate the Python DSL functions for all confirmed effective actions listed above.
Example format:
```python
import math
from typing import Any

def action1(api):
    \"\"\"Declare discrete button action ACTION1.\"\"\"
    return api.declare_environment_action(action_id="ACTION1")

def action5(api):
    \"\"\"Declare entity selection/cycling action ACTION5.\"\"\"
    return api.declare_environment_action(action_id="ACTION5")

def action6(api, x: int = 0, y: int = 0):
    \"\"\"Declare spatial action ACTION6 at target coordinates (x, y).\"\"\"
    return api.declare_environment_action(action_id="ACTION6", data={"x": int(x), "y": int(y)})
```
Output ONLY the ```python ... ``` block. No JSON manifest required (signatures and parameters are derived automatically from your Python functions).
"""
    sections.append(user_instructions)
    return CODER_SYSTEM_PROMPT, "\n".join(sections)
