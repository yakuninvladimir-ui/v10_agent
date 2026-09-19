"""Coder Prompt Builder (Call 2 Family).

Constructs prompts strictly quarantined from level goals, Solver hypotheses, and EpistemicMemory.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from v10_agent.memory_contours import SyntaxErrorRecord

CODER_SYSTEM_PROMPT = """\
You are a Python DSL (Domain Specific Language) generator.
Think step by step and reason through the environment mechanics, object structures, and coordinate affordances before writing the code.
Your task is to translate environment specifications into a pure Python module and a JSON manifest.

CONSTRAINTS:
- You DO NOT know the puzzle goal. Do not attempt to solve it.
- Generate pure functions that ONLY call `api.declare_environment_action(...)`.
- Allowed imports: math, typing, dataclasses, enum, collections. NO os, sys, eval, exec.
- You MUST implement a function for EVERY action listed in the available actions.
- Function names must be canonical: `action1`, `action2`, ..., `action6`.
- Coordinate actions (e.g., ACTION6) must accept `x: int = 0, y: int = 0`.

OUTPUT FORMAT:
Provide exactly two blocks:
1. ```python ... ``` containing the DSL module.
2. ```json ... ``` containing the manifest matching schema 'v10.dsl_manifest.1'.
"""

SANDBOX_API_DOC = """\
The SandboxAPI injected into your functions exposes:
  - api.planning_set: PlanningSet (read-only access to objects, relations, allowed actions)
  - api.get_object(obj_id_or_alias: str) -> PlanningObject | None
  - api.query_objects(predicate: Callable) -> list[PlanningObject]
  - api.metric_distance(obj_a: str, obj_b: str, metric_name: str = "centroid_distance") -> float
  - api.declare_environment_action(
        action_id: str,
        data: dict | None = None,
        reasoning: dict | None = None,
        expected_metric_deltas: dict | None = None,
        target_object_ids: list[str] | None = None,
        confidence: float = 1.0,
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
            if isinstance(inv, str) and not any(kw in inv.lower() for kw in ("goal", "win", "target", "curriculum"))
        ]

    # Resolve ground-truth available actions from planning_set or env_spec
    available_actions: list[str] = []
    if planning_set is not None and getattr(planning_set, "allowed_action_ids", None):
        available_actions = [str(a).upper() for a in planning_set.allowed_action_ids if str(a).upper() not in ("RESET", "ACTION7")]
    elif "available_actions" in compact_spec:
        available_actions = [str(a).upper() for a in compact_spec["available_actions"] if str(a).upper() not in ("RESET", "ACTION7")]
    elif "researched_actions" in compact_spec:
        available_actions = [
            str(a["action_id"]).upper()
            for a in compact_spec["researched_actions"]
            if isinstance(a, dict) and "action_id" in a and str(a["action_id"]).upper() not in ("RESET", "ACTION7")
        ]

    image_note = (
        "coder_raw_frame.png is the exact same frame as coder_annotated_frame.png, but without object annotations.\n\n"
        if has_image
        else ""
    )

    sections: list[str] = []
    if image_note:
        sections.append(image_note.strip())
    sections.extend([
        "Environment Specification (Observed Facts & Affordances):",
        json.dumps(compact_spec, indent=2),
        "",
    ])

    if available_actions:
        sections.extend([
            "ALL AVAILABLE ENVIRONMENT ACTIONS (You MUST define a function for EVERY action in this list):",
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
Generate the Python DSL functions and matching function manifest.
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

```json
{
  "schema_version": "v10.dsl_manifest.1",
  "functions": [
    {
      "name": "action1",
      "parameters": [],
      "returns": "effect_declaration",
      "docstring": "Declare discrete button action ACTION1.",
      "purity": "pure_declaration",
      "expected_effect_template": {}
    },
    {
      "name": "action5",
      "parameters": [],
      "returns": "effect_declaration",
      "docstring": "Declare entity selection/cycling action ACTION5.",
      "purity": "pure_declaration",
      "expected_effect_template": {}
    },
    {
      "name": "action6",
      "parameters": [
        {"name": "x", "type": "int", "default": 0},
        {"name": "y", "type": "int", "default": 0}
      ],
      "returns": "effect_declaration",
      "docstring": "Declare spatial action ACTION6 at coordinates (x, y).",
      "purity": "pure_declaration",
      "expected_effect_template": {}
    }
  ]
}
```
"""
    sections.append(user_instructions)
    return CODER_SYSTEM_PROMPT, "\n".join(sections)
