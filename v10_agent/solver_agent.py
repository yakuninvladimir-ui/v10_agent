"""Solver Agent (Call 3 Family).

Formulates candidate trajectory packages calling ONLY declared DSL functions,
writing exclusively to EpistemicMemory.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

from v10_agent.config import V10Config
from v10_agent.explorer_agent import extract_json_block
from v10_agent.llm_advisor import (
    BaseLLMAdvisor,
    format_multimodal_message,
    is_role_vision_enabled,
    _has_image_payload,
    sanitize_model_response,
)
from v10_agent.memory_contours import EpistemicMemory
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.virtual_sandbox import VirtualKinematicSandbox

logger = logging.getLogger(__name__)


def _parse_fn_call_args(call_str: str) -> tuple[str, dict]:
    """Parse 'action6(x=10, y=20)' into ('action6', {'x': 10, 'y': 20})."""
    import ast as _ast
    call_str = call_str.strip().rstrip(';')
    try:
        tree = _ast.parse(call_str, mode='eval')
        if isinstance(tree.body, _ast.Call):
            name = tree.body.func.id if isinstance(tree.body.func, _ast.Name) else call_str.split('(')[0].strip()
            kwargs = {}
            for kw in tree.body.keywords:
                try:
                    kwargs[kw.arg] = _ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    kwargs[kw.arg] = str(_ast.dump(kw.value))
            # Positional args are handled by the manifest-aware fallback below (L223-238)
            # Only return keyword args here
            return name, kwargs
    except Exception:
        pass
    # Fallback: regex
    m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$', call_str, re.DOTALL)
    if m:
        name = m.group(1)
        arg_str = m.group(2).strip()
        if not arg_str:
            return name, {}
        pairs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,]+)', arg_str)
        return name, {k: v.strip() for k, v in pairs}
    return call_str, {}


def parse_text_trajectory(
    text: str,
    manifest_or_funcs: Any = None,
    planning_set: PlanningSet | None = None,
) -> dict[str, Any] | None:
    """Extract hypothesis and steps from clean structured text response."""
    text = sanitize_model_response(text)
    hypothesis_text = ""
    trajectory_text = ""

    manifest_map: dict[str, dict[str, Any]] = {}
    valid_func_names: set[str] = set()

    if isinstance(manifest_or_funcs, dict):
        if "functions" in manifest_or_funcs:
            for f in manifest_or_funcs["functions"]:
                if isinstance(f, dict) and "name" in f:
                    manifest_map[f["name"]] = f
                    valid_func_names.add(f["name"])
        else:
            for k, v in manifest_or_funcs.items():
                valid_func_names.add(k)
                if isinstance(v, dict):
                    manifest_map[k] = v
    elif isinstance(manifest_or_funcs, (set, list, tuple)):
        valid_func_names = set(manifest_or_funcs)

    # 0. Invariant / Hypothesis extraction (supports both XML <invariant_analysis> and markdown [HYPOTHESIS])
    inv_evo_text = ""
    hypothesis_text = ""

    xml_ia = re.search(r"<invariant_analysis>(.*?)(?:</invariant_analysis>|\Z)", text, re.DOTALL | re.IGNORECASE)
    if xml_ia:
        inv_evo_text = xml_ia.group(1).strip()
        hypothesis_text = inv_evo_text
    else:
        ie_match = re.search(
            r"(?i)(?:\[\s*invariant_evolution\s*\]|#+\s*invariant_evolution|invariant_evolution\s*:)(.*?)(?=(?:\[\s*hypothesis\s*\]|#+\s*hypothesis|hypothesis\s*:|\[\s*trajectory\s*\]|#+\s*trajectory|trajectory\s*:|\[\s*plan\s*\]|\Z))",
            text,
            re.DOTALL,
        )
        if ie_match:
            inv_evo_text = ie_match.group(1).strip()

        h_match = re.search(
            r"(?i)(?:\[\s*hypothesis\s*\]|#+\s*hypothesis|hypothesis\s*:)(.*?)(?=(?:\[\s*trajectory\s*\]|#+\s*trajectory|trajectory\s*:|\[\s*plan\s*\]|\Z))",
            text,
            re.DOTALL,
        )
        if h_match:
            hypothesis_text = h_match.group(1).strip()

    # 1. Trajectory blocks extraction (supports XML tags <trajectory_1>... and markdown [TRAJECTORY] / **Trajectory 1**)
    raw_chunks: list[tuple[str, str]] = []
    xml_trajs = list(re.finditer(r"<trajectory_(\d+)>(.*?)(?:</trajectory_\1>|\Z)", text, re.DOTALL | re.IGNORECASE))
    if xml_trajs:
        for m in xml_trajs:
            c_id = f"traj_text_{m.group(1).zfill(2)}"
            raw_chunks.append((c_id, m.group(2).strip()))
    else:
        t_match = re.search(
            r"(?i)(?:\[\s*trajectory\s*\]|#+\s*trajectory|trajectory\s*:|\[\s*plan\s*\])(.*)",
            text,
            re.DOTALL,
        )
        if t_match:
            trajectory_text = t_match.group(1).strip()
        else:
            # If no explicit section header found, search whole text
            trajectory_text = text.strip()

        # Detect multiple trajectory/candidate blocks
        header_matches = list(
            re.finditer(
                r"(?i)(?:[*#_\s]*(?:trajectory|candidate|plan)\s*(\d+)[^\n]*\n)",
                trajectory_text,
            )
        )
        if header_matches:
            for idx, m in enumerate(header_matches):
                c_id = f"traj_text_{m.group(1).zfill(2)}"
                start_pos = m.end()
                end_pos = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(trajectory_text)
                raw_chunks.append((c_id, trajectory_text[start_pos:end_pos].strip()))
        else:
            raw_chunks.append(("traj_text_01", trajectory_text))

    candidates: list[dict[str, Any]] = []
    for cand_id, chunk_text in raw_chunks:
        cand_steps: list[dict[str, Any]] = []
        step_idx = 1

        # 1. Check for ATEM protocol function calls: <atem:invoke name="...">
        atem_invokes = list(re.finditer(r'<atem:invoke\s+name=["\'](?:api\.)?([a-zA-Z0-9_]+)["\']>(.*?)(?:</atem:invoke>|\Z)', chunk_text, re.DOTALL))
        if atem_invokes:
            for aim in atem_invokes:
                fn = aim.group(1).split(".")[-1]
                body = aim.group(2).strip()
                args: dict[str, Any] = {}
                param_matches = list(re.finditer(r'<atem:parameter\s+name=["\']([a-zA-Z0-9_]+)["\']>(.*?)</atem:parameter>', body, re.DOTALL))
                if param_matches:
                    for pm in param_matches:
                        pname = pm.group(1).strip()
                        pval = pm.group(2).strip()
                        try:
                            args[pname] = int(pval)
                        except ValueError:
                            args[pname] = pval
                elif body.startswith("{") and body.endswith("}"):
                    try:
                        parsed_json = json.loads(body)
                        if isinstance(parsed_json, dict):
                            args = parsed_json
                    except Exception:
                        pass
                if not valid_func_names or fn in valid_func_names:
                    cand_steps.append({
                        "step_id": f"s{step_idx}",
                        "dsl_function": fn,
                        "arguments": args,
                        "expected_propositions": [],
                    })
                    step_idx += 1

        # 2. Check for JSON array of steps: [{"action": "RIGHT"}, ...] or [{"dsl_function": "move"}, ...]
        if not cand_steps:
            json_arr_m = re.search(r'\[\s*\{.*?\}\s*\]', chunk_text, re.DOTALL)
            if json_arr_m:
                try:
                    parsed_arr = json.loads(json_arr_m.group(0))
                    if isinstance(parsed_arr, list) and parsed_arr and isinstance(parsed_arr[0], dict):
                        for item in parsed_arr:
                            fn = item.get("dsl_function") or item.get("action") or item.get("name") or "step"
                            if not valid_func_names or fn in valid_func_names:
                                cand_steps.append({
                                    "step_id": f"s{step_idx}",
                                    "dsl_function": fn,
                                    "arguments": item.get("arguments") or item.get("params") or {},
                                    "expected_propositions": item.get("expected_propositions") or [],
                                })
                                step_idx += 1
                except Exception:
                    pass

        # 3. Check for Python-style function call lines: func(arg=val, ...)
        if not cand_steps:
            lines = chunk_text.split("\n")
            for line in lines:
                line_s = line.strip()
                if not line_s:
                    continue
                exp_props: list[dict[str, Any]] = []
                exp_match = re.search(r"(?:#\s*)?EXPECT:\s*(.*)$", line_s, re.IGNORECASE)
                if exp_match:
                    exp_str = exp_match.group(1).strip()
                    if exp_str.startswith("[") and exp_str.endswith("]"):
                        try:
                            parsed_p = json.loads(exp_str)
                            if isinstance(parsed_p, list):
                                exp_props.extend(p for p in parsed_p if isinstance(p, dict))
                        except Exception:
                            pass
                    elif exp_str.startswith("{") and exp_str.endswith("}"):
                        try:
                            parsed_p = json.loads(exp_str)
                            if isinstance(parsed_p, dict):
                                exp_props.append(parsed_p)
                        except Exception:
                            pass
                    else:
                        for item in exp_str.split(","):
                            kv = item.strip().split("=")
                            if len(kv) == 2:
                                k = kv[0].strip()
                                v_str = kv[1].strip()
                                val: Any
                                try:
                                    val = int(v_str)
                                except ValueError:
                                    val = v_str
                                exp_props.append({
                                    "family": "metric_sign",
                                    "predicate": k,
                                    "value": val,
                                })

                m = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", line_s)
                if m:
                    call_match = m.group(0)
                    fn, args = _parse_fn_call_args(call_match)
                    arg_str = m.group(2)
                    if not valid_func_names or fn in valid_func_names:
                        if not args and arg_str.strip():
                            pos_vals = [x.strip().strip("\"'") for x in arg_str.split(",") if x.strip()]
                            fn_meta = manifest_map.get(fn, {})
                            fn_params = [p for p in fn_meta.get("parameters", []) if p.get("name") != "api"]
                            for p_idx, val in enumerate(pos_vals):
                                if p_idx < len(fn_params):
                                    p_name = fn_params[p_idx]["name"]
                                    try:
                                        args[p_name] = int(val)
                                    except ValueError:
                                        args[p_name] = val
                                else:
                                    if p_idx == 0:
                                        args["x"] = int(val) if val.lstrip("-").isdigit() else val
                                    elif p_idx == 1:
                                        args["y"] = int(val) if val.lstrip("-").isdigit() else val

                        count = args.pop("count", 1)
                        if isinstance(count, int) and count > 1:
                            actual_count = min(count, 30)
                            for i in range(actual_count):
                                cand_steps.append({
                                    "step_id": f"s{step_idx}",
                                    "dsl_function": fn,
                                    "arguments": copy.deepcopy(args),
                                    "expected_propositions": exp_props if i == actual_count - 1 else [],
                                    "repeat_index": i,
                                    "repeat_total": actual_count,
                                    "break_on_null": True,
                                })
                                step_idx += 1
                        else:
                            cand_steps.append({
                                "step_id": f"s{step_idx}",
                                "dsl_function": fn,
                                "arguments": args,
                                "expected_propositions": exp_props,
                            })
                            step_idx += 1

        if cand_steps:
            candidates.append({
                "trajectory_id": cand_id,
                "steps": cand_steps,
                "success_condition": {"family": "terminal_metadata", "predicate": "win"},
                "confidence": 0.85,
            })

    if not candidates:
        return None

    res = {
        "schema_version": "v10.trajectory_package.1",
        "proposal_id": "text_proposal",
        "hypothesis": hypothesis_text,
        "candidates": candidates,
    }
    if inv_evo_text:
        res["invariant_evolution"] = inv_evo_text
    return res


parse_solver_output = parse_text_trajectory


class SolverAgent:
    """Call 3: Proposes trajectory packages over certified DSL function manifests."""

    def __init__(self, config: V10Config, advisor: BaseLLMAdvisor):
        self.config = config
        self.advisor = advisor
        self.last_conversation_messages: list[dict[str, Any]] | None = None
        self.last_hypothesis: str = ""
        self.last_package: dict[str, Any] | None = None

    def generate_trajectory_package(
        self,
        manifest: dict[str, Any],
        planning_set: PlanningSet,
        epistemic_memory: EpistemicMemory | None = None,
        budget: int = 50,
        image_png: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        game_memory: Any | None = None,
    ) -> dict[str, Any] | None:
        """Propose candidate trajectories adhering strictly to schema v10.trajectory_package.1."""
        solver_mm = getattr(
            self.config, "solver_multimodal_enabled",
            getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
        )
        has_image = (
            any(bool(x) for x in image_png)
            if isinstance(image_png, (list, tuple))
            else (any(bool(x) for x in image_png.values()) if isinstance(image_png, dict) else bool(image_png))
        ) and solver_mm
        effective_image = image_png if solver_mm else None

        sys_prompt, user_prompt = build_solver_prompts(
            manifest=manifest,
            planning_set=planning_set,
            epistemic_memory=epistemic_memory,
            action_budget=budget,
            game_memory=game_memory,
            has_image=has_image,
        )

        retries = max(1, getattr(self.config, "max_solver_retries_per_level", 5))
        for attempt in range(1, retries + 1):
            try:
                response = self.advisor.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    config=self.config,
                    image_bytes=effective_image,
                    agent_role="solver",
                )
            except Exception as exc:
                logger.warning(f"Solver LLM invocation failed (attempt {attempt}/{retries}): {exc}")
                continue

            logger.info(f"Solver LLM response received:\n{response}")
            manifest_funcs = {f["name"] for f in manifest.get("functions", []) if "name" in f}
            package = extract_json_block(response)

            # Fallback to structured text parser if JSON extraction failed
            if not package or not isinstance(package, dict) or not package.get("candidates"):
                text_pkg = parse_text_trajectory(response, manifest, planning_set)
                if text_pkg:
                    package = text_pkg
                    logger.info(f"Solver parsed trajectory from clean text ({len(text_pkg['candidates'][0]['steps'])} steps).")

            if not package or not isinstance(package, dict):
                logger.warning(f"Solver failed to return a valid JSON object or text trajectory (attempt {attempt}/{retries})")
                continue

            # Preserve conversation history for Turn 2 reflection on win
            role_vision_enabled = is_role_vision_enabled(self.config, "solver")
            supports_vision = _has_image_payload(effective_image) and role_vision_enabled
            conv_msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
            conv_msgs.extend(format_multimodal_message(user_prompt, effective_image if supports_vision else None))
            conv_msgs.append({"role": "assistant", "content": response})
            self.last_conversation_messages = conv_msgs
            self.last_package = package
            self.last_hypothesis = str(package.get("hypothesis") or package.get("invariant_evolution") or "")

            # Note: Unverified solver hypotheses and candidate-specific invariant analysis
            # are ephemeral to this planning attempt and are preserved in EpistemicMemory,
            # NOT dumped into permanent GameMemory (prevents coordinate and attempt poisoning).

            # Verify schema version and candidates
            candidates = package.get("candidates", [])
            if not candidates or not isinstance(candidates, list):
                logger.warning(f"Solver trajectory package contains no candidates (attempt {attempt}/{retries})")
                continue

            # Validate function names and arguments against PlanningSet
            valid_candidates: list[dict[str, Any]] = []
            for cand in candidates:
                steps = cand.get("steps", [])
                cand_valid = True
                for step in steps:
                    fn_name = step.get("dsl_function")
                    if fn_name not in manifest_funcs:
                        logger.warning(f"Step references undefined DSL function: {fn_name!r}")
                        cand_valid = False
                        break

                    args = step.get("arguments", {})
                    for k, v in args.items():
                        if isinstance(v, str) and (v.startswith("obj_") or v in planning_set.object_alias_to_real):
                            resolved = planning_set.resolve_object_id(v)
                            if resolved is None:
                                logger.warning(f"Argument {k}={v!r} not found in PlanningSet")
                                cand_valid = False
                                break

                if cand_valid and steps:
                    valid_candidates.append(cand)

            if not valid_candidates:
                logger.warning(f"None of the proposed Solver candidates satisfied grounding checks (attempt {attempt}/{retries})")
                continue

            # Route through Virtual Kinematic Sandbox for optional optimization and auto-repair
            sandbox = VirtualKinematicSandbox(planning_set, game_memory)
            manifest_map = {f["name"]: f for f in manifest.get("functions", []) if "name" in f}

            candidates_to_use: list[dict[str, Any]] = []
            for cand in valid_candidates:
                steps = cand.get("steps", [])
                try:
                    res = sandbox.evaluate_and_repair_trajectory(
                        steps, manifest_map, hypothesis=package.get("hypothesis")
                    )
                except Exception as exc:
                    logger.warning(f"Virtual Sandbox evaluation failed for {cand.get('trajectory_id')}: {exc}")
                    res = None

                if res and res.verdict in ("APPROVED", "REPAIRED") and res.repaired_steps:
                    cand_repaired = copy.deepcopy(cand)
                    cand_repaired["steps"] = res.repaired_steps
                    cand_repaired["sandbox_verdict"] = res.verdict
                    cand_repaired["sandbox_goal_reached"] = res.goal_reached
                    logger.info(
                        f"Virtual Sandbox {res.verdict} candidate {cand.get('trajectory_id')}: "
                        f"{res.reason} ({len(res.repaired_steps)} steps)"
                    )
                    candidates_to_use.append(cand_repaired)
                else:
                    # Retain grounded candidate directly without formulaic rejection (ISO-7 compliant)
                    cand_retained = copy.deepcopy(cand)
                    cand_retained["sandbox_verdict"] = res.verdict if res else "SKIPPED"
                    cand_retained["sandbox_goal_reached"] = res.goal_reached if res else False
                    logger.info(
                        f"Retaining grounded candidate {cand.get('trajectory_id')} "
                        f"(sandbox verdict: {cand_retained['sandbox_verdict']})"
                    )
                    candidates_to_use.append(cand_retained)

            # If model generated no valid candidates, synthesize a fallback candidate from top invariants
            if not candidates_to_use and sandbox.invariants:
                unsatisfied = [
                    inv for inv in sandbox.invariants
                    if sandbox.compute_initial_distance(inv) > 1.0
                ]
                for inv in unsatisfied[:4]:
                    synth_steps = sandbox.synthesize_invariant_trajectory(inv, manifest_map)
                    if synth_steps:
                        res = sandbox.evaluate_and_repair_trajectory(synth_steps, manifest_map)
                        if res and res.goal_reached:
                            cand_synth = {
                                "trajectory_id": f"traj_synth_inv_{len(candidates_to_use) + 1:02d}",
                                "confidence": 0.80,
                                "steps": res.repaired_steps,
                                "sandbox_verdict": res.verdict,
                                "sandbox_goal_reached": True,
                                "strategy": f"synthesized_invariant_{inv.invariant_type}",
                            }
                            logger.info(f"Synthesized invariant fallback trajectory {cand_synth['trajectory_id']}: {res.reason} ({len(res.repaired_steps)} steps)")
                            candidates_to_use.append(cand_synth)
                            break

            # Prioritize candidates by goal satisfaction first, then approved/repaired by sandbox
            def cand_priority(c: dict[str, Any]) -> tuple[int, int]:
                goal_score = 0 if c.get("sandbox_goal_reached") else 1
                verdict_score = 0 if c.get("sandbox_verdict") in ("APPROVED", "REPAIRED") else 1
                return (goal_score, verdict_score)

            candidates_to_use.sort(key=cand_priority)

            package["candidates"] = candidates_to_use[: self.config.max_candidates_per_solver_package]

            package.setdefault("schema_version", "v10.trajectory_package.1")
            package.setdefault("snapshot_hash", planning_set.grid_hash)
            return package

        return None

    def revise_invariants(
        self,
        outcome: str = "WIN",
        execution_summary: str = "",
        failure_reason: str = "",
        active_invariants: list[dict[str, Any]] | list[str] | None = None,
        invalidated_invariants: list[dict[str, Any]] | None = None,
        winning_candidate: dict[str, Any] | None = None,
    ) -> list[str]:
        """Turn 2: Follow-up reflection asking the Solver to revise domain-general invariants after win or failure."""
        cand_id = (
            winning_candidate.get("trajectory_id", "winning_candidate")
            if winning_candidate
            else (
                self.last_package.get("candidates", [{}])[0].get("trajectory_id", "candidate")
                if self.last_package and self.last_package.get("candidates")
                else "candidate"
            )
        )
        hyp = self.last_hypothesis or "Target coverage via spatial coordination"
        exec_desc = execution_summary or (
            "Candidate successfully reached win condition"
            if outcome.upper() == "WIN"
            else "Candidate halted before reaching win condition"
        )

        if outcome.upper() == "WIN":
            outcome_header = (
                f"[EXECUTION OUTCOME: LEVEL WON]\n"
                f"Your proposed trajectory ({cand_id}) successfully solved this level and achieved victory!\n"
                f"Summary of executed transitions: {exec_desc}\n"
                f"Your original hypothesis: {hyp}\n"
            )
        else:
            fail_desc = failure_reason or "Candidate failed to achieve level goal or violated physical constraints."
            outcome_header = (
                f"[EXECUTION OUTCOME: ATTEMPT FAILED / CONTRADICTION DETECTED]\n"
                f"Your proposed trajectory ({cand_id}) did not succeed.\n"
                f"Failure reason: {fail_desc}\n"
                f"Summary of executed transitions: {exec_desc}\n"
                f"Your original hypothesis: {hyp}\n"
            )

        active_lines: list[str] = []
        if active_invariants:
            for item in active_invariants:
                if isinstance(item, dict):
                    rule_text = item.get("rule") or item.get("name") or str(item)
                    reason = item.get("inclusion_reason") or f"Confidence: {item.get('confidence', 1.0)}"
                    active_lines.append(f"- {rule_text} [Reason: {reason}]")
                elif isinstance(item, str) and item.strip():
                    active_lines.append(f"- {item.strip()}")

        if active_lines:
            active_section = (
                "\n--- CURRENT ACTIVE INVARIANTS (with reasons for inclusion) ---\n"
                + "\n".join(active_lines)
                + "\n"
            )
        else:
            active_section = "\n--- CURRENT ACTIVE INVARIANTS ---\n(None currently established)\n"

        invalidated_lines: list[str] = []
        if invalidated_invariants:
            for item in invalidated_invariants:
                if isinstance(item, dict):
                    rule_text = item.get("rule") or item.get("id") or str(item)
                    reason = item.get("invalidation_reason") or item.get("reason") or "Falsified by environment transition"
                    invalidated_lines.append(f"- {rule_text} [Removed/Invalidated reason: {reason}]")
                elif isinstance(item, str) and item.strip():
                    invalidated_lines.append(f"- {item.strip()}")

        invalidated_section = ""
        if invalidated_lines:
            invalidated_section = (
                "\n--- INVALIDATED / REMOVED RULES (with reasons for removal) ---\n"
                + "\n".join(invalidated_lines)
                + "\n"
            )

        followup_user_prompt = (
            f"{outcome_header}"
            f"{active_section}"
            f"{invalidated_section}\n"
            f"TASK: Conduct a comprehensive revision of the domain invariants for subsequent levels.\n"
            f"Analyze the outcome, current active invariants, and reasons behind any invalidations:\n"
            f"- Retain and reinforce invariants that remain empirically sound.\n"
            f"- If an invariant was invalidated or removed by the symbolic engine, reformulate it into a valid version that accounts for the observed constraints (e.g. boundary conditions, obstacles, specific prerequisites), or discard it if fundamentally flawed.\n"
            f"- Introduce new domain invariants based on observed physical mechanics, spatial structures, or goal semantics.\n\n"
            f"CRITICAL RULES FOR INVARIANTS:\n"
            f"1. NO coordinates, row/column numbers, bounding boxes, or grid dimensions (these change every level).\n"
            f"2. NO step counts or action repetition numbers (distances vary across levels).\n"
            f"3. NO literal button sequences or macros like 'action1 -> action5' (order of actions varies).\n"
            f"4. Use applicable categories from:\n"
            f"   - [PHYSICS]: How objects move, collide, and interact with boundaries\n"
            f"   - [STRUCTURE]: Persistent spatial relationships (symmetry, containment, alignment)\n"
            f"   - [GOAL]: What constitutes success (abstractly, not as coordinates)\n"
            f"   - [CONTROL]: How entity selection/switching operates\n"
            f"   - [PALETTE]: Color-to-role mapping IF clearly observed\n"
            f"   - [CONSTRAINTS]: What is forbidden or impossible\n\n"
            f"Format your revised invariant list strictly inside <revised_invariants>...</revised_invariants> with bullet points:\n"
            f"<revised_invariants>\n"
            f"- [PHYSICS]: ...\n"
            f"- [STRUCTURE]: ...\n"
            f"- [GOAL]: ...\n"
            f"</revised_invariants>"
        )

        messages_to_send: list[dict[str, Any]]
        if self.last_conversation_messages:
            messages_to_send = [dict(m) for m in self.last_conversation_messages]
            messages_to_send.append({"role": "user", "content": followup_user_prompt})
        else:
            messages_to_send = [
                {"role": "system", "content": "You are an expert game theory and invariant distillation engine."},
                {"role": "user", "content": followup_user_prompt},
            ]

        try:
            raw_response = self.advisor.generate_chat(
                messages_to_send,
                config=self.config,
                agent_role="solver_reflection",
            )
            response = sanitize_model_response(raw_response)
        except Exception as exc:
            logger.warning(f"Solver invariant revision invocation failed: {exc}")
            response = ""

        invariants: list[str] = []
        xml_m = re.search(r"<revised_invariants>(.*?)(?:</revised_invariants>|\Z)", response, re.DOTALL | re.IGNORECASE)
        if not xml_m:
            xml_m = re.search(r"<distilled_invariants>(.*?)(?:</distilled_invariants>|\Z)", response, re.DOTALL | re.IGNORECASE)
        inv_text = xml_m.group(1).strip() if xml_m else response.strip()

        for line in inv_text.splitlines():
            line_s = line.strip().lstrip("-*•0123456789. ")
            if not line_s:
                continue
            line_s = re.sub(r"<[^>]+>", "", line_s).strip()
            if not line_s:
                continue
            # Filter out any accidentally leaked button macros, coordinates or code blocks
            if re.search(r"action\d+\s*[-→>]+\s*action\d+", line_s, re.IGNORECASE):
                continue
            if re.search(r"action\d+\(\)\s*[-→>]+\s*action\d+\(\)", line_s, re.IGNORECASE):
                continue
            if re.search(r"action\d+\(\)\s*,\s*action\d+\(\)", line_s, re.IGNORECASE):
                continue
            if re.search(r"action\d+\(\)", line_s, re.IGNORECASE) and any(arrow in line_s for arrow in ("->", "→", ">")):
                continue
            if re.search(r"\b(?:dx|dy)\s*=\s*-?\d+", line_s, re.IGNORECASE):
                continue
            if line_s.startswith("```") or line_s.endswith("```"):
                continue
            invariants.append(line_s)

        if not invariants and outcome.upper() == "WIN":
            if hyp:
                invariants.append(f"Confirmed goal principle: {hyp}")
            else:
                invariants.append("Level victory achieved via coordinated multi-entity alignment")

        logger.info(
            f"Solver revised {len(invariants)} cross-level invariants ({outcome.upper()}):\n"
            + "\n".join(f"  * {inv}" for inv in invariants)
        )
        return invariants

    def distill_level_win_invariants(
        self,
        winning_candidate: dict[str, Any] | None = None,
        execution_summary: str = "",
        active_invariants: list[dict[str, Any]] | list[str] | None = None,
        invalidated_invariants: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Turn 2: Follow-up reflection asking the Solver to revise/distill invariants after a win."""
        return self.revise_invariants(
            outcome="WIN",
            execution_summary=execution_summary,
            failure_reason="",
            active_invariants=active_invariants,
            invalidated_invariants=invalidated_invariants,
            winning_candidate=winning_candidate,
        )

    def reflect_and_revise_on_failure(
        self,
        execution_summary: str = "",
        failure_reason: str = "",
        active_invariants: list[dict[str, Any]] | list[str] | None = None,
        invalidated_invariants: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Turn 2: Follow-up reflection asking the Solver to revise invariants after an attempt failure / contradiction."""
        return self.revise_invariants(
            outcome="FAILURE",
            execution_summary=execution_summary,
            failure_reason=failure_reason,
            active_invariants=active_invariants,
            invalidated_invariants=invalidated_invariants,
        )

