"""DSL Coder Agent (Call 2 Family).

Generates dynamic Python DSL and typed function manifest, verified in the Sandbox,
writing exclusively to SyntaxErrorMemory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from typing import Any

from v10_agent.action_semantics import is_observable_move
from v10_agent.config import V10Config
from v10_agent.llm_advisor import BaseLLMAdvisor, sanitize_model_response
from v10_agent.memory_contours import SyntaxErrorMemory, SyntaxErrorRecord
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.sandbox import SandboxedModule, SandboxExecutor

logger = logging.getLogger(__name__)


from v10_agent.explorer_agent import clean_and_parse_json


def derive_manifest_from_python_source(source: str) -> dict[str, Any]:
    """Parse Python source and derive canonical v10.dsl_manifest.1 using AST."""
    import ast

    tree = ast.parse(source)
    functions: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn_name = node.name
            doc = ast.get_docstring(node) or f"Declare environment action {fn_name.upper()}."

            num_args = len(node.args.args)
            num_defaults = len(node.args.defaults)
            default_start = num_args - num_defaults

            params: list[dict[str, Any]] = []
            for i, arg in enumerate(node.args.args):
                if arg.arg == "api":
                    continue
                p_name = arg.arg
                p_type = "int"
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        p_type = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        p_type = str(arg.annotation.value)

                p_dict: dict[str, Any] = {"name": p_name, "type": p_type}
                if i >= default_start:
                    def_node = node.args.defaults[i - default_start]
                    if isinstance(def_node, ast.Constant):
                        if def_node.value is None:
                            p_dict["default"] = 0 if p_type == "int" else ""
                        else:
                            p_dict["default"] = def_node.value
                    elif isinstance(def_node, ast.UnaryOp) and isinstance(def_node.op, ast.USub) and isinstance(def_node.operand, ast.Constant):
                        p_dict["default"] = -def_node.operand.value
                    else:
                        p_dict["default"] = 0 if p_type == "int" else ""
                else:
                    if p_name in ("x", "y", "count"):
                        p_dict["default"] = 0

                params.append(p_dict)

            functions.append({
                "name": fn_name,
                "parameters": params,
                "returns": "effect_declaration",
                "docstring": doc.strip(),
                "purity": "pure_declaration",
                "expected_effect_template": {},
            })

    return {
        "schema_version": "v10.dsl_manifest.1",
        "functions": functions,
    }


def extract_code_and_manifest(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract Python source and JSON manifest from Coder model output."""
    clean_text = sanitize_model_response(text)
    # 1. Extract Python block
    py_match = re.search(r"```(?:python|py)\s*(.*?)\s*```", clean_text, re.DOTALL)
    source = py_match.group(1).strip() if py_match else None

    # 2. Extract JSON manifest block
    manifest = None
    for block in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL):
        cand = clean_and_parse_json(block.group(1))
        if isinstance(cand, dict) and "functions" in cand:
            manifest = cand
            break

    if manifest is None:
        # Fallback: search for {"schema_version" or {"functions" or "functions"
        for marker in ['{"schema_version"', '{"functions"', '"functions"']:
            first = clean_text.find(marker)
            if first != -1:
                start_pos = clean_text.rfind("{", 0, first + len(marker))
                if start_pos == -1:
                    start_pos = first
                last = clean_text.rfind("}")
                if last > start_pos:
                    cand = clean_and_parse_json(clean_text[start_pos : last + 1])
                    if isinstance(cand, dict) and "functions" in cand:
                        manifest = cand
                        break

    # 3. Fallback: if manifest is missing or omitted, automatically derive from Python AST
    if manifest is None and source:
        try:
            derived = derive_manifest_from_python_source(source)
            if derived.get("functions"):
                manifest = derived
        except Exception as exc:
            logger.debug(f"Could not derive manifest from Python AST: {exc}")

    return source, manifest


class DSLCoder:
    """Call 2: Emits verified level-specific Python DSL and typed manifest."""

    def __init__(
        self,
        config: V10Config,
        advisor: BaseLLMAdvisor,
        sandbox_executor: SandboxExecutor | None = None,
    ):
        self.config = config
        self.advisor = advisor
        self.sandbox_executor = sandbox_executor or SandboxExecutor()

    def generate_dsl(
        self,
        env_spec: dict[str, Any],
        syntax_memory: SyntaxErrorMemory,
        planning_set: PlanningSet,
        game_memory: Any | None = None,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
    ) -> tuple[SandboxedModule | None, dict[str, Any] | None, list[str]]:
        """Generate, validate, and dry-run a Python DSL module within retry budget."""
        coder_mm = getattr(
            self.config, "coder_multimodal_enabled",
            getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
        )
        has_image = (
            any(bool(x) for x in image_png)
            if isinstance(image_png, (list, tuple))
            else (any(bool(x) for x in image_png.values()) if isinstance(image_png, dict) else bool(image_png))
        ) and coder_mm
        effective_image = image_png if coder_mm else None

        retries = max(1, self.config.max_coder_retries_per_level)
        diagnostics_history: list[str] = []

        for attempt in range(1, retries + 1):
            sys_prompt, user_prompt = build_coder_prompts(
                env_spec=env_spec,
                syntax_errors=syntax_memory.entries,
                game_memory=game_memory,
                planning_set=planning_set,
                has_image=has_image,
            )
            prompt_hash = hashlib.sha256((sys_prompt + user_prompt).encode("utf-8")).hexdigest()[:12]

            call_config = self.config
            if attempt > 1:
                call_config = copy.copy(self.config)
                base_temp = getattr(self.config, "coder_temperature", 0.5)
                call_config.coder_temperature = max(0.1, round(base_temp - 0.15 * (attempt - 1), 2))

            try:
                response = self.advisor.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    config=call_config,
                    image_bytes=effective_image,
                    agent_role="coder",
                )
            except Exception as exc:
                err_msg = f"Coder LLM invocation failed: {exc}"
                logger.warning(err_msg)
                syntax_memory.record_error(
                    SyntaxErrorRecord(
                        prompt_hash=prompt_hash,
                        source_code="",
                        error_type="LLMInvocationError",
                        error_message=err_msg,
                    )
                )
                diagnostics_history.append(err_msg)
                continue

            source, manifest = extract_code_and_manifest(response)
            if not source:
                err_msg = "Coder output did not contain a valid ```python ... ``` block"
                logger.warning(f"DSLCoder attempt {attempt}/{retries} failed: {err_msg}")
                syntax_memory.record_error(
                    SyntaxErrorRecord(
                        prompt_hash=prompt_hash,
                        source_code="",
                        error_type="MissingPythonBlock",
                        error_message=err_msg,
                    )
                )
                diagnostics_history.append(err_msg)
                continue

            if not manifest or not isinstance(manifest, dict) or "functions" not in manifest:
                err_msg = "Coder output did not contain a valid JSON manifest with 'functions' key"
                logger.warning(f"DSLCoder attempt {attempt}/{retries} failed: {err_msg}")
                syntax_memory.record_error(
                    SyntaxErrorRecord(
                        prompt_hash=prompt_hash,
                        source_code=source,
                        error_type="MissingManifest",
                        error_message=err_msg,
                    )
                )
                diagnostics_history.append(err_msg)
                continue

            # Ensure persistent confirmed actions from game_memory and action_affordances are preserved and augmented
            allowed_set = None
            if planning_set is not None and getattr(planning_set, "allowed_action_ids", None):
                allowed_set = {str(a).upper() for a in planning_set.allowed_action_ids}

            confirmed_acts = set()
            if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
                confirmed_acts.update(str(a).upper() for a in game_memory.confirmed_action_effects if is_observable_move(a))
            if game_memory is not None and hasattr(game_memory, "action_affordances"):
                for aff in getattr(game_memory, "action_affordances", []):
                    if isinstance(aff, dict) and aff.get("action_id"):
                        act_id = str(aff["action_id"]).upper()
                        if is_observable_move(act_id):
                            confirmed_acts.add(act_id)
            if env_spec:
                if "action_affordances" in env_spec and isinstance(env_spec["action_affordances"], list):
                    for aff in env_spec["action_affordances"]:
                        if isinstance(aff, dict) and aff.get("action_id"):
                            act_id = str(aff["action_id"]).upper()
                            if is_observable_move(act_id):
                                confirmed_acts.add(act_id)
                if "available_actions" in env_spec and isinstance(env_spec["available_actions"], list):
                    for act in env_spec["available_actions"]:
                        act_id = str(act).upper()
                        if is_observable_move(act_id):
                            confirmed_acts.add(act_id)

            unconfirmed: set[str] = set()
            if game_memory is not None and hasattr(game_memory, "unconfirmed_actions"):
                unconfirmed = {str(a).upper() for a in game_memory.unconfirmed_actions}

            if allowed_set is not None:
                for act in allowed_set:
                    act_up = str(act).upper()
                    if is_observable_move(act_up) and act_up not in unconfirmed:
                        confirmed_acts.add(act_up)
                confirmed_acts = {a for a in confirmed_acts if a in allowed_set}
            confirmed_acts = {a for a in confirmed_acts if a not in unconfirmed}

            existing_fn_names = set()
            for fn in manifest.get("functions", []):
                if isinstance(fn, dict):
                    if fn.get("name"):
                        existing_fn_names.add(str(fn["name"]).lower())
                    if fn.get("action_id"):
                        existing_fn_names.add(str(fn["action_id"]).lower())
            missing_acts = [act for act in confirmed_acts if act.lower() not in existing_fn_names and (allowed_set is None or act in allowed_set)]
            if missing_acts:
                missing_code = []
                for act in sorted(missing_acts):
                    fn_name = act.lower()
                    if act == "ACTION6":
                        missing_code.append(
                            f"\ndef {fn_name}(api, target='', x=None, y=None):\n"
                            f"    \"\"\"Preserved confirmed coordinate/click action on target object or (x, y).\"\"\"\n"
                            f"    if target:\n"
                            f"        return api.click_object(target=target, x=x, y=y)\n"
                            f"    cur_x = int(x) if x is not None else 0\n"
                            f"    cur_y = int(y) if y is not None else 0\n"
                            f"    return api.declare_environment_action(action_id='ACTION6', data={{'x': cur_x, 'y': cur_y}})\n\n"
                            f"def click(api, target='', x=None, y=None):\n"
                            f"    \"\"\"Click on the designated object alias or coordinates.\"\"\"\n"
                            f"    return {fn_name}(api, target=target, x=x, y=y)\n"
                        )
                        manifest.get("functions", []).append({
                            "name": fn_name,
                            "action_id": "ACTION6",
                            "parameters": [
                                {"name": "target", "type": "str", "default": ""},
                                {"name": "x", "type": "int", "default": 0},
                                {"name": "y", "type": "int", "default": 0},
                            ],
                            "returns": "effect_declaration",
                            "docstring": "Preserved confirmed coordinate/click action on target object or (x, y).",
                            "purity": "pure_declaration",
                            "expected_effect_template": {},
                        })
                        manifest.get("functions", []).append({
                            "name": "click",
                            "action_id": "ACTION6",
                            "parameters": [
                                {"name": "target", "type": "str", "default": ""},
                                {"name": "x", "type": "int", "default": 0},
                                {"name": "y", "type": "int", "default": 0},
                            ],
                            "returns": "effect_declaration",
                            "docstring": "Click on the designated object alias or coordinates.",
                            "purity": "pure_declaration",
                            "expected_effect_template": {},
                        })
                    else:
                        missing_code.append(
                            f"\ndef {fn_name}(api):\n"
                            f"    \"\"\"Preserved confirmed action from earlier level or verified affordance.\"\"\"\n"
                            f"    return api.declare_environment_action(action_id='{act}')\n"
                        )
                        manifest.get("functions", []).append({
                            "name": fn_name,
                            "action_id": act,
                            "parameters": [],
                            "returns": "effect_declaration",
                            "docstring": "Preserved confirmed action from earlier level or verified affordance.",
                            "purity": "pure_declaration",
                            "expected_effect_template": {},
                        })
                source = source + "\n" + "\n".join(missing_code)

            # 1. Static AST Sandbox Validation & Compilation
            try:
                module = self.sandbox_executor.load_module(source, manifest)
            except SyntaxError as exc:
                err_msg = str(exc)
                logger.warning(f"DSLCoder attempt {attempt}/{retries} validation failed: {err_msg}")
                syntax_memory.record_error(
                    SyntaxErrorRecord(
                        prompt_hash=prompt_hash,
                        source_code=source,
                        error_type="SandboxSyntaxValidationError",
                        error_message=err_msg,
                        diagnostics=[err_msg],
                    )
                )
                diagnostics_history.append(err_msg)
                continue

            # 2. Dry-Run Verification
            ok, dry_run_err = self.sandbox_executor.dry_run_manifest(module, planning_set)
            if not ok:
                err_msg = dry_run_err or "Dry run execution failure"
                logger.warning(f"DSLCoder attempt {attempt}/{retries} dry-run failed: {err_msg}")
                syntax_memory.record_error(
                    SyntaxErrorRecord(
                        prompt_hash=prompt_hash,
                        source_code=source,
                        error_type="DryRunFailure",
                        error_message=err_msg,
                        diagnostics=[err_msg],
                    )
                )
                diagnostics_history.append(err_msg)
                continue

            # Certified success!
            allowed_acts = set(confirmed_acts)
            if env_spec and "available_actions" in env_spec and env_spec["available_actions"]:
                allowed_acts.update(str(a).upper() for a in env_spec["available_actions"])
            if allowed_acts:
                manifest["functions"] = [
                    fn for fn in manifest.get("functions", [])
                    if not re.match(r"^ACTION\d+$", str(fn.get("name", "")).upper())
                    or str(fn.get("name", "")).upper() in allowed_acts
                ]
            logger.info(f"DSLCoder generated valid DSL module on attempt {attempt}/{retries}")
            return module, manifest, []

        logger.error(f"DSLCoder exhausted retries ({retries}). Fallback required.")
        return None, None, diagnostics_history
