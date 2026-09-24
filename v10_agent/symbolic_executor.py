"""SymbolicTrajectoryExecutor: Independent symbolic controller for trajectory verification and execution.

Separates declarative trajectory generation (Solver/Qwen) from trajectory verification (Judge)
and deterministic step execution (Symbolic Execution Engine).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary, Verdict
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import (
    BranchSignature,
    EpistemicMemory,
    SyntaxErrorRecord,
    SyntaxErrorMemory,
    summarize_grid_diff,
)
from v10_agent.planning_set import PlanningSet
from v10_agent.sandbox import SandboxExecutor, SandboxedModule
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.types import EffectDeclaration
from v10_agent.verification import GroundedStep, GroundingError, VerificationBinder

logger = logging.getLogger("v10_agent.symbolic_executor")


def format_step_signature(step: dict[str, Any]) -> str:
    """Format step into a canonical signature with sorted arguments, e.g. 'action6(x=10, y=12)' or 'action1'."""
    fn_name = str(step.get("dsl_function", "")).strip()
    args = step.get("arguments") or {}
    if not isinstance(args, dict) or not args:
        return fn_name
    args_str = ", ".join(f"{k}={args[k]!r}" for k in sorted(args.keys()))
    return f"{fn_name}({args_str})"


def format_sequence_signature(steps: list[dict[str, Any]]) -> str:
    """Format sequence of steps into a chained signature string."""
    return " -> ".join(format_step_signature(s) for s in steps)


def get_trajectory_signature_tuple(steps: list[dict[str, Any]]) -> tuple[str, ...]:
    """Format sequence of steps into a tuple of canonical step signatures."""
    return tuple(format_step_signature(s) for s in steps)


class StepExecutionVerdict(Enum):
    SUCCESS = "success"
    PRE_VERIFICATION_FAILED = "pre_verification_failed"
    SANDBOX_EXECUTION_FAILED = "sandbox_execution_failed"
    NO_CANDIDATE = "no_candidate"


@dataclass
class StepExecutionResult:
    verdict: StepExecutionVerdict
    effect: EffectDeclaration | None = None
    grounded_step: GroundedStep | None = None
    strategy: str = "fallback"
    error_message: str | None = None
    circuit_broken: bool = False


@dataclass
class TransitionEvaluationResult:
    verdict: Any
    judgment: BrusentsovJudgment
    candidate_advanced: bool = False
    candidate_severed: bool = False
    replan_needed: bool = False
    reset_needed: bool = False
    falsification_detected: bool = False
    falsified_action: str | None = None
    evidence_needed: bool = False
    evidence_hint: str | None = None


class SymbolicTrajectoryExecutor:
    """Independent symbolic controller executing verified trajectory steps outside LLM context.

    1. Pre-execution verification: checks groundedness, manifest presence, and epistemic severance.
    2. Sandboxed execution: safely calls DSL primitives with filtered keyword arguments.
    3. Circuit breaker: on execution failure, immediately severs candidate, records failure,
       and requests replan + clean reset.
    4. Post-transition evaluation: evaluates empirical feedback via LayeredVerifier (Brusentsov ternary logic).
    """

    def __init__(
        self,
        config: V10Config,
        sandbox_executor: SandboxExecutor,
        binder: VerificationBinder,
        verifier: LayeredVerifier,
    ):
        self.config = config
        self.sandbox_executor = sandbox_executor
        self.binder = binder
        self.verifier = verifier

    def verify_full_candidate_trajectory(
        self,
        candidate: CandidateTrajectory,
        planning_set: PlanningSet,
        active_module: SandboxedModule,
        game_memory: Any = None,
        syntax_memory: SyntaxErrorMemory | None = None,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Verify full candidate trajectory against environment constraints and hypothesis.

        Returns (is_valid, reason, repaired_steps).
        Two verification criteria:
        1. Conformity to achieving the declared hypothesis.
        2. No contradictions with the environment (executable, stays within grid, no obstacle collisions).
        """
        # 1. Executability check: all steps must refer to valid callable functions in active_module
        for s_idx, step in enumerate(candidate.steps):
            f_name = str(step.get("dsl_function", ""))
            if f_name not in active_module.namespace or not callable(active_module.namespace[f_name]):
                if syntax_memory is not None:
                    syntax_memory.record_error(
                        SyntaxErrorRecord(
                            prompt_hash="",
                            source_code=active_module.source,
                            error_type="MissingDSLFunction",
                            error_message=f"Function {f_name!r} not defined in DSL module",
                        )
                    )
                return False, f"Step {s_idx + 1} references undefined DSL function {f_name!r}", []

        # 2. Forward simulation via VirtualKinematicSandbox for bounds, collisions, and hypothesis goal
        try:
            from v10_agent.virtual_sandbox import VirtualKinematicSandbox
            sandbox = VirtualKinematicSandbox(planning_set, game_memory)
            manifest_map = {
                f_name: {"name": f_name, "docstring": getattr(fn, "__doc__", "") or ""}
                for f_name, fn in active_module.namespace.items()
                if callable(fn)
            }
            res = sandbox.evaluate_and_repair_trajectory(
                candidate.steps,
                manifest_map,
                hypothesis=getattr(candidate, "hypothesis", None),
            )
            if res.verdict == "REJECTED":
                return False, res.reason, []

            repaired = res.repaired_steps if res.repaired_steps else candidate.steps
            return True, res.reason or "Trajectory approved without environment contradiction", repaired
        except Exception as exc:
            logger.warning(f"SymbolicExecutor: virtual sandbox simulation error: {exc}")
            return True, "Simulation bypassed", candidate.steps

    def prepare_and_execute_step(
        self,
        pool: TrajectoryPool | None,
        planning_set: PlanningSet,
        active_module: SandboxedModule | None,
        epistemic_memory: EpistemicMemory,
        syntax_memory: SyntaxErrorMemory,
        game_memory: Any = None,
    ) -> StepExecutionResult:
        """Select, verify, and symbolically execute the next step from the active trajectory pool."""
        if pool is None or active_module is None:
            return StepExecutionResult(verdict=StepExecutionVerdict.NO_CANDIDATE)

        candidate = pool.active_candidate()
        if candidate is None:
            return StepExecutionResult(verdict=StepExecutionVerdict.NO_CANDIDATE)

        # 0. Full Trajectory Verification at candidate start (Two parameters: hypothesis & environment constraints)
        if candidate.cursor == 0 and not getattr(candidate, "_full_trajectory_verified", False):
            is_valid, reason, repaired = self.verify_full_candidate_trajectory(
                candidate=candidate,
                planning_set=planning_set,
                active_module=active_module,
                game_memory=game_memory,
                syntax_memory=syntax_memory,
            )
            if not is_valid:
                logger.info(f"SymbolicExecutor: Full trajectory verification REJECTED candidate {candidate.trajectory_id}: {reason}")
                candidate.sever()
                return StepExecutionResult(
                    verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED,
                    error_message=reason,
                    circuit_broken=True,
                )
            if repaired:
                candidate.steps = repaired
            candidate._full_trajectory_verified = True
            logger.info(f"SymbolicExecutor: Full trajectory for candidate {candidate.trajectory_id} APPROVED ({len(candidate.steps)} steps). Executing...")

        if candidate.initial_grid is None and planning_set is not None:
            grid_src = getattr(planning_set, "grid", None)
            if grid_src:
                candidate.initial_grid = [list(r) for r in grid_src]

        step_dict = candidate.current_step()
        if step_dict is None:
            return StepExecutionResult(verdict=StepExecutionVerdict.NO_CANDIDATE)

        step_id = str(step_dict.get("step_id", "s0"))
        fn_name = str(step_dict.get("dsl_function", ""))
        step_sig = format_step_signature(step_dict)
        full_traj_sig = format_sequence_signature(candidate.steps)
        cand_tuple = get_trajectory_signature_tuple(candidate.steps)

        # 1. Pre-execution Verification
        # 1a. Check if candidate or full trajectory is already severed in EpistemicMemory,
        # or if candidate trajectory is completely contained in a previously failed trajectory
        # matching in order from the start (index 0).
        # Rule: If failed is 1-2-3-4-5-4-3-2-1, then 1-2-3-4-5 is subsumed & severed,
        # but 5-4-3-2-1 is NOT severed because it does not match from index 0.
        is_subsumed = False
        subsuming_traj = None
        if hasattr(epistemic_memory, "is_trajectory_subsumed"):
            is_subsumed, subsuming_traj = epistemic_memory.is_trajectory_subsumed(cand_tuple)

        if (
            not candidate.active
            or epistemic_memory.is_severed(full_traj_sig)
            or is_subsumed
        ):
            reason_detail = (
                f"completely contained in previously failed trajectory {subsuming_traj!r}"
                if is_subsumed
                else f"previously severed: {full_traj_sig}"
            )
            logger.info(
                f"SymbolicExecutor: Candidate {candidate.trajectory_id} ({full_traj_sig!r}) "
                f"is rejected ({reason_detail}); severing candidate."
            )
            candidate.sever()
            return StepExecutionResult(
                verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED,
                error_message=f"Candidate trajectory {reason_detail}",
                circuit_broken=True,
            )

        # 1b. Check if DSL function exists in compiled SandboxedModule
        if fn_name not in active_module.namespace or not callable(active_module.namespace[fn_name]):
            logger.warning(f"SymbolicExecutor: Function {fn_name!r} not defined in active module; severing candidate.")
            candidate.sever()
            syntax_memory.record_error(
                SyntaxErrorRecord(
                    prompt_hash="",
                    source_code=active_module.source,
                    error_type="MissingDSLFunction",
                    error_message=f"Function {fn_name!r} not defined in DSL module",
                )
            )
            return StepExecutionResult(
                verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED,
                error_message=f"Function {fn_name!r} not found",
                circuit_broken=True,
            )

        # 1c. Ground step arguments strictly against active PlanningSet
        try:
            grounded_step = self.binder.ground_step(step_dict, planning_set)
        except GroundingError as ge:
            logger.warning(f"SymbolicExecutor: Grounding error for step {step_id}: {ge}; severing candidate.")
            candidate.sever()
            epistemic_memory.sever_branch(full_traj_sig)
            return StepExecutionResult(
                verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED,
                error_message=str(ge),
                circuit_broken=True,
            )

        # 2. Symbolic Execution inside SandboxedModule
        try:
            effect = self.sandbox_executor.execute(
                module=active_module,
                function_name=grounded_step.dsl_function,
                arguments=grounded_step.arguments,
                planning_set=planning_set,
            )
            return StepExecutionResult(
                verdict=StepExecutionVerdict.SUCCESS,
                effect=effect,
                grounded_step=grounded_step,
                strategy="solver_candidate",
            )
        except Exception as exc:
            logger.warning(
                f"SymbolicExecutor: Sandbox execution failed for step {step_id} ({type(exc).__name__}: {exc}). "
                f"Tripping circuit breaker."
            )
            syntax_memory.record_error(
                SyntaxErrorRecord(
                    prompt_hash="",
                    source_code=active_module.source,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            # Circuit breaker: sever candidate immediately so it is never repeated
            candidate.sever()
            epistemic_memory.sever_branch(full_traj_sig)
            return StepExecutionResult(
                verdict=StepExecutionVerdict.SANDBOX_EXECUTION_FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                circuit_broken=True,
            )

    def evaluate_transition(
        self,
        pending_step: GroundedStep,
        before_snapshot: Any,
        after_obs: dict[str, Any],
        planning_set: PlanningSet,
        active_pool: TrajectoryPool | None,
        epistemic_memory: EpistemicMemory,
        game_memory: Any = None,
        action_dict: dict[str, Any] | None = None,
    ) -> TransitionEvaluationResult:
        """Evaluate empirical transition via LayeredVerifier and update trajectory cursor and memory."""
        judgment = self.verifier.evaluate_transition(
            step=pending_step,
            before_snapshot=before_snapshot,
            after_obs=after_obs,
            planning_set=planning_set,
            game_memory=game_memory,
            action_dict=action_dict,
        )

        active_cand = active_pool.active_candidate() if active_pool else None

        # Check for empirical falsification:
        # If at candidate initial step (cursor == 0), a motion primitive produces zero grid delta
        # or if expected propositions were explicitly refuted (verdict == Ternary.FALSE).
        act_id = ""
        if action_dict:
            act_id = str(action_dict.get("action_id") or action_dict.get("id") or "").upper()
        if not act_id:
            try:
                manifest = None
                if hasattr(self, "sandbox_executor") and hasattr(self.sandbox_executor, "current_manifest"):
                    manifest = self.sandbox_executor.current_manifest
                elif hasattr(planning_set, "manifest"):
                    manifest = planning_set.manifest
                
                if manifest and isinstance(manifest, dict) and "functions" in manifest:
                    funcs = manifest["functions"]
                    if isinstance(funcs, list):
                        for fn in funcs:
                            if fn.get("name") == pending_step.dsl_function:
                                act_id = str(fn.get("action_id") or "").upper()
                                break
                    elif isinstance(funcs, dict):
                        f_meta = funcs.get(pending_step.dsl_function)
                        if f_meta:
                            act_id = str(f_meta.get("action_id") or "").upper()
            except Exception:
                pass

            if not act_id:
                import re
                m = re.search(r"action(\d+)", pending_step.dsl_function, re.IGNORECASE)
                if m:
                    act_id = f"ACTION{m.group(1)}"

        before_grid = getattr(before_snapshot, "grid", None)
        if before_grid is None and hasattr(planning_set, "grid"):
            before_grid = planning_set.grid
        after_grid = after_obs.get("grid")

        if active_cand is not None and active_cand.initial_grid is None and before_grid is not None:
            active_cand.initial_grid = [list(r) for r in before_grid]

        zero_grid_delta = False
        if before_grid is not None and after_grid is not None:
            if before_grid == after_grid:
                zero_grid_delta = True

        is_initial_step = (active_cand is None or active_cand.cursor == 0)
        confirmed_eff = ""
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "")
        is_modal_selection = (
            act_id in ("ACTION5", "ACTION6")
            or (game_memory and any(kw in getattr(game_memory, "confirmed_action_effects", {}).get(act_id, "").lower() for kw in ("selection", "toggle", "indicator", "active entity", "modal")))
            or (game_memory and any(isinstance(aff, dict) and str(aff.get("action_id", "")).upper() == act_id and aff.get("effect_class") == "MODAL_SELECTION" for aff in getattr(game_memory, "action_affordances", [])))
        )
        is_confirmed_motion = (
            not is_modal_selection
            and any(k in confirmed_eff.lower() for k in ("moved", "moves", "dy=", "dx=", "displace"))
            and any(d in confirmed_eff.upper() for d in ("UP", "DOWN", "LEFT", "RIGHT"))
        )

        falsification_detected = False
        falsified_action = None

        if is_initial_step and zero_grid_delta and is_confirmed_motion:
            # Check if this zero delta is due to boundary collision or multi-entity modal selection
            is_blocked_by_boundary = False
            if planning_set and hasattr(planning_set, "objects"):
                h = len(before_grid) if before_grid else 64
                w = len(before_grid[0]) if before_grid and len(before_grid) > 0 else 64
                for o in planning_set.objects:
                    if getattr(o, "role", None) in ("ACTOR", "PRIMARY", "DYNAMIC"):
                        bbox = getattr(o, "bbox", None)
                        if bbox:
                            min_r = getattr(bbox, "min_row", 0)
                            max_r = getattr(bbox, "max_row", h - 1)
                            min_c = getattr(bbox, "min_col", 0)
                            max_c = getattr(bbox, "max_col", w - 1)
                            if "UP" in confirmed_eff and min_r <= 0:
                                is_blocked_by_boundary = True
                            if "DOWN" in confirmed_eff and max_r >= h - 1:
                                is_blocked_by_boundary = True
                            if "LEFT" in confirmed_eff and min_c <= 0:
                                is_blocked_by_boundary = True
                            if "RIGHT" in confirmed_eff and max_c >= w - 1:
                                is_blocked_by_boundary = True

            # Extract available action surface
            allowed_acts: list[str] = []
            if planning_set and hasattr(planning_set, "allowed_action_ids"):
                raw_acts = planning_set.allowed_action_ids
                if isinstance(raw_acts, (list, tuple, set)):
                    allowed_acts = [str(a).upper() for a in raw_acts]

            has_selection_mechanics = bool(game_memory and getattr(game_memory, "selection_mechanics", None))

            # Potential modal switch actions: ACTION5, ACTION6 or confirmed selection/toggle actions
            modal_switch_actions = [
                act for act in allowed_acts
                if act in ("ACTION5", "ACTION6")
                or (game_memory and any(kw in getattr(game_memory, "confirmed_action_effects", {}).get(act, "").lower() for kw in ("selection", "toggle", "indicator", "active entity", "modal")))
                or (game_memory and any(isinstance(aff, dict) and str(aff.get("action_id", "")).upper() == act and aff.get("effect_class") == "MODAL_SELECTION" for aff in getattr(game_memory, "action_affordances", [])))
            ]
            has_modal_switch = bool(modal_switch_actions or has_selection_mechanics)

            # Unconfirmed / unexplored discrete actions that could be modal switches
            unconfirmed_actions = [
                act for act in allowed_acts
                if act not in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET", "ACTION7")
                and (not game_memory or act not in getattr(game_memory, "confirmed_action_effects", {}))
            ]
            has_unexplored = bool(unconfirmed_actions)

            if has_modal_switch or has_unexplored:
                # Orthogonal Modality Invariant: Zero response along one axis with selector/unexplored
                # indicates degree of freedom is blocked in current mode, NOT invalidity of action.
                modal_probe_target = modal_switch_actions[0] if modal_switch_actions else (unconfirmed_actions[0] if unconfirmed_actions else "ACTION5")
                falsification_detected = False
                falsified_action = None
                judgment = BrusentsovJudgment(
                    trajectory_id=pending_step.step_id,
                    step_id=pending_step.step_id,
                    verdict=Verdict.UNDECIDED,
                    expected_propositions=pending_step.expected_propositions,
                    observed_propositions=judgment.observed_propositions,
                    explanation=(
                        f"Step {pending_step.step_id} ({pending_step.dsl_function} / {act_id}): "
                        f"Action produced zero grid delta along kinematic axis, but modal selector / switch is available "
                        f"({modal_probe_target}). Degree of freedom is blocked in current discrete mode. "
                        f"Requesting modal probe {modal_probe_target}."
                    ),
                    evidence_hint=f"probe_{modal_probe_target}",
                    action_dict=action_dict or {},
                    is_effective=False,
                )
            elif is_blocked_by_boundary:
                # Boundary collision: action hit boundary for this candidate (sever candidate, but do NOT falsify action)
                falsification_detected = False
                falsified_action = None
                judgment = BrusentsovJudgment(
                    trajectory_id=pending_step.step_id,
                    step_id=pending_step.step_id,
                    verdict=Ternary.FALSE,
                    expected_propositions=pending_step.expected_propositions,
                    observed_propositions=judgment.observed_propositions,
                    explanation=(
                        f"Step {pending_step.step_id} ({pending_step.dsl_function} / {act_id}): "
                        f"Action produced zero grid delta due to boundary collision (soft stop/wall collision). "
                        f"Candidate branch severed, but action is not globally falsified."
                    ),
                    action_dict=action_dict or {},
                    is_effective=False,
                )
            else:
                # Truly falsified: no modal switches, no boundary collision, initial step failed
                falsification_detected = True
                falsified_action = act_id or pending_step.dsl_function
                judgment = BrusentsovJudgment(
                    trajectory_id=pending_step.step_id,
                    step_id=pending_step.step_id,
                    verdict=Ternary.FALSE,
                    expected_propositions=pending_step.expected_propositions,
                    observed_propositions=judgment.observed_propositions,
                    explanation=(
                        f"Step {pending_step.step_id} ({pending_step.dsl_function} / {act_id}): "
                        f"Action produced zero grid delta (данное действие не валидно при текущих координатах объекта / boundary / obstacle)."
                    ),
                    action_dict=action_dict or {},
                    is_effective=False,
                )

        epistemic_memory.record_judgment(judgment)

        is_won = False
        level_completed = False
        if after_obs:
            st = str(after_obs.get("state", "")).upper()
            if st in ("WIN", "WON", "DONE", "VICTORY"):
                is_won = True
            after_levels = int(after_obs.get("levels_completed", 0) or 0)
            before_levels = int(getattr(before_snapshot, "levels_completed", 0) or 0) if before_snapshot else 0
            if after_levels > before_levels:
                level_completed = True

        cand_finished = False

        if judgment.verdict == Verdict.UNDECIDED:
            # UNDECIDED: Epistemic signal seek evidence.
            # Do NOT advance cursor, do NOT sever candidate, do NOT trigger RESET.
            logger.info(
                f"SymbolicExecutor: Step {pending_step.step_id} received UNDECIDED: {judgment.explanation}. "
                f"Evidence probe needed."
            )
            return TransitionEvaluationResult(
                verdict=Verdict.UNDECIDED,
                judgment=judgment,
                candidate_advanced=False,
                candidate_severed=False,
                replan_needed=False,
                reset_needed=False,
                evidence_needed=True,
                evidence_hint=judgment.evidence_hint,
            )

        if active_cand is not None and before_grid is not None and after_grid is not None:
            step_fn = pending_step.dsl_function or act_id or "step"
            curr_cursor = active_cand.cursor
            if before_grid != after_grid:
                step_diff = summarize_grid_diff(before_grid, after_grid)
                active_cand.step_effects.append(f"Step {curr_cursor + 1} ({step_fn}): {step_diff}")
            else:
                active_cand.step_effects.append(f"Step {curr_cursor + 1} ({step_fn}): no grid change")

        if judgment.verdict == Ternary.TRUE:
            # FOLLOW: Advance cursor
            if active_cand is not None:
                active_cand.advance()
            cand_advanced = True
            cand_severed = False
            cand_finished = (active_cand is None or active_cand.is_finished()) if active_pool else True
            logger.info(f"SymbolicExecutor: Step {pending_step.step_id} verified TRUE (FOLLOW).")

        elif judgment.verdict == Ternary.FALSE:
            # Standard NULL: Sever branch & trigger candidate reset immediately
            if active_cand is not None:
                active_cand.sever()
                full_sig = format_sequence_signature(active_cand.steps)
                seq_sig = format_sequence_signature(active_cand.steps[:active_cand.cursor + 1])
                full_tuple = get_trajectory_signature_tuple(active_cand.steps)
                seq_tuple = get_trajectory_signature_tuple(active_cand.steps[:active_cand.cursor + 1])
                epistemic_memory.sever_branch(full_sig)
                epistemic_memory.sever_branch(seq_sig)
                if hasattr(epistemic_memory, "record_failed_completed_trajectory"):
                    epistemic_memory.record_failed_completed_trajectory(full_tuple)
                    epistemic_memory.record_failed_completed_trajectory(seq_tuple)
            else:
                epistemic_memory.sever_branch(pending_step.step_id)
            cand_advanced = False
            cand_severed = True
            cand_finished = True
            logger.info(
                f"SymbolicExecutor: Step {pending_step.step_id} verified FALSE (NULL contradiction): {judgment.explanation}. "
                f"Candidate severed."
            )

        else:  # Ternary.IRRELEVANT
            # OMIT: Non-contradicting step; advance cursor
            if active_cand is not None:
                active_cand.advance()
            epistemic_memory.pause_branch_as_omit(
                BranchSignature(
                    signature_id=pending_step.step_id,
                    trajectory_id=pending_step.step_id,
                    last_step_id=pending_step.step_id,
                    expected_propositions=pending_step.expected_propositions.to_list(),
                    observed_propositions=judgment.observed_propositions.to_list(),
                )
            )
            cand_advanced = True
            cand_severed = False
            cand_finished = (active_cand is None or active_cand.is_finished()) if active_pool else True

        if cand_finished and not (is_won or level_completed) and active_cand is not None:
            if epistemic_memory is not None and hasattr(epistemic_memory, "record_attempt_feedback"):
                steps_repr = []
                for s in active_cand.steps:
                    fn = s.get("dsl_function", "")
                    args = s.get("arguments", {})
                    if args:
                        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                        steps_repr.append(f"{fn}({args_str})")
                    else:
                        steps_repr.append(f"{fn}()")
                diff_summary = None
                if active_cand.initial_grid is not None and after_grid is not None:
                    diff_summary = summarize_grid_diff(active_cand.initial_grid, after_grid)

                if cand_severed:
                    failed_idx = active_cand.cursor
                    failed_fn = (
                        active_cand.steps[failed_idx].get("dsl_function", "")
                        if failed_idx < len(active_cand.steps)
                        else (pending_step.dsl_function or "step")
                    )
                    diag_text = self._build_step_contradiction_diagnostic(
                        step=pending_step,
                        judgment=judgment,
                        planning_set=planning_set,
                        attempt_idx=len(getattr(epistemic_memory, "current_level_attempts", [])) + 1,
                    )
                    epistemic_memory.record_attempt_feedback(
                        hypothesis=f"Candidate {active_cand.trajectory_id}",
                        trajectory_summary=" -> ".join(steps_repr[:failed_idx + 1]) + f" [FAILED AT STEP {failed_idx + 1}]",
                        status="severed_step_failed",
                        reason=(
                            f"Step {failed_idx + 1} ({failed_fn}) failed: {judgment.explanation}. "
                            f"The object hit an obstacle/boundary or had no effect. DO NOT execute {failed_fn} again from this position!"
                        ),
                        diff_summary=diff_summary,
                        effective_steps=list(active_cand.step_effects),
                        diagnostic=diag_text,
                    )
                else:
                    traj_sig = format_sequence_signature(active_cand.steps)
                    traj_tuple = get_trajectory_signature_tuple(active_cand.steps)
                    epistemic_memory.sever_branch(traj_sig)
                    if hasattr(epistemic_memory, "record_failed_completed_trajectory"):
                        epistemic_memory.record_failed_completed_trajectory(traj_tuple)
                    epistemic_memory.record_attempt_feedback(
                        hypothesis=f"Candidate {active_cand.trajectory_id}",
                        trajectory_summary=traj_sig,
                        status="executed_but_level_not_won",
                        reason="Trajectory executed completely but game did not advance to next level. Invariant or target pattern was incorrect. DO NOT REPEAT THIS SEQUENCE!",
                        diff_summary=diff_summary,
                        effective_steps=list(active_cand.step_effects),
                    )
        elif (is_won or level_completed) and active_cand is not None:
            logger.info(f"SymbolicExecutor: Candidate {active_cand.trajectory_id} achieved goal/level completion.")

        if is_won or level_completed:
            replan_needed = False
            reset_needed = False
        elif falsification_detected:
            logger.warning(
                f"SymbolicExecutor: Действие {falsified_action} не валидно при текущих координатах объекта. "
                f"Severing pool and requesting replan + clean reset."
            )
            replan_needed = True
            reset_needed = True
        elif cand_finished:
            # Check if active_pool has another candidate to execute
            next_cand = active_pool.active_candidate() if active_pool else None
            if next_cand is not None:
                logger.info(
                    f"SymbolicExecutor: Candidate finished without win. "
                    f"Advancing to next candidate in pool: {next_cand.trajectory_id}. Queueing board RESET."
                )
                replan_needed = False
                reset_needed = True
            else:
                logger.info("SymbolicExecutor: All candidates in pool exhausted without win. Requesting Solver replan.")
                replan_needed = True
                reset_needed = True
        else:
            replan_needed = False
            reset_needed = False

        return TransitionEvaluationResult(
            verdict=judgment.verdict,
            judgment=judgment,
            candidate_advanced=cand_advanced,
            candidate_severed=cand_severed,
            replan_needed=replan_needed,
            reset_needed=reset_needed,
            falsification_detected=falsification_detected,
            falsified_action=falsified_action,
        )

    def _build_step_contradiction_diagnostic(
        self,
        step: GroundedStep,
        judgment: BrusentsovJudgment,
        planning_set: PlanningSet | None,
        attempt_idx: int,
    ) -> str:
        """Format strict differential diagnostic for a physically contradicted step."""
        fn = step.dsl_function or "action"
        step_id = step.step_id or "s1"
        alias_map = planning_set.object_real_to_alias if planning_set and hasattr(planning_set, "object_real_to_alias") else {}

        def _alias(s_id: str) -> str:
            return alias_map.get(s_id, s_id)

        expected_parts: list[str] = []
        observed_parts: list[str] = []

        expected_motion_aliases: list[str] = []
        expected_unchanged_aliases: list[str] = []
        stagnant_aliases: list[str] = []
        mutated_aliases: list[str] = []

        # Analyze expectations
        for p in step.expected_propositions:
            alias = _alias(p.subject_id)
            pred = p.predicate.lower()
            if pred in ("moved", "step_moved") or (pred in ("dy", "dx", "row_delta", "col_delta") and p.value != 0):
                if alias not in expected_motion_aliases:
                    expected_motion_aliases.append(alias)
                    expected_parts.append(f"moved({alias})")
            elif pred in ("unchanged", "preserved") or (pred in ("dy", "dx", "row_delta", "col_delta") and p.value == 0):
                if alias not in expected_unchanged_aliases:
                    expected_unchanged_aliases.append(alias)
                    expected_parts.append(f"unchanged({alias})")

        # Analyze observations for expected actors and other participating objects
        obs_motion: dict[str, bool] = {}
        for o in judgment.observed_propositions:
            alias = _alias(o.subject_id)
            pred = o.predicate.lower()
            if pred in ("step_moved", "moved"):
                v = o.value
                is_mov = False
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    is_mov = (v[0] != 0 or v[1] != 0)
                elif isinstance(v, str):
                    clean_v = v.strip("()[]")
                    parts = [p.strip() for p in clean_v.split(",") if p.strip()]
                    if len(parts) >= 2:
                        is_mov = any(int(x) != 0 for x in parts if x.lstrip("-").isdigit())
                if is_mov:
                    obs_motion[alias] = True
                elif alias not in obs_motion:
                    obs_motion[alias] = False
            elif pred in ("dy", "dx", "row_delta", "col_delta"):
                try:
                    if int(o.value) != 0:
                        obs_motion[alias] = True
                    elif alias not in obs_motion:
                        obs_motion[alias] = False
                except (ValueError, TypeError):
                    pass

        for alias in expected_motion_aliases:
            if obs_motion.get(alias, False):
                observed_parts.append(f"moved({alias})")
            else:
                observed_parts.append(f"stationary({alias})")
                stagnant_aliases.append(alias)

        for alias in expected_unchanged_aliases:
            if obs_motion.get(alias, False):
                observed_parts.append(f"moved({alias})")
                mutated_aliases.append(alias)
            else:
                observed_parts.append(f"stationary({alias})")

        # Also check if any other alias moved that wasn't expected to move
        for alias, moved in obs_motion.items():
            if moved and alias not in expected_motion_aliases and alias not in mutated_aliases:
                mutated_aliases.append(alias)
                if f"moved({alias})" not in observed_parts:
                    observed_parts.append(f"moved({alias})")

        exp_str = " & ".join(expected_parts) if expected_parts else "unspecified"
        obs_str = " & ".join(observed_parts) if observed_parts else "unspecified"

        if mutated_aliases and stagnant_aliases:
            diagnosis = (
                f"{fn} mutates alias {', '.join(mutated_aliases)}, not alias {', '.join(stagnant_aliases)}. "
                f"Do NOT repeat {fn} for moving {', '.join(stagnant_aliases)}."
            )
        elif stagnant_aliases:
            diagnosis = (
                f"{fn} had no effect on alias {', '.join(stagnant_aliases)} (stationary/blocked). "
                f"Do NOT repeat {fn} for moving {', '.join(stagnant_aliases)}."
            )
        elif mutated_aliases:
            diagnosis = (
                f"{fn} unexpectedly mutated alias {', '.join(mutated_aliases)} violating invariance. "
                f"Do NOT repeat {fn}."
            )
        else:
            diagnosis = f"{fn} produced contradiction: {judgment.explanation}. Do NOT repeat."

        return (
            f"[FAILED ATTEMPT {attempt_idx} DIAGNOSTIC]\n"
            f"  Failed at step {step_id} ({fn}): Mismatch detected.\n"
            f"  EXPECTED: {exp_str}\n"
            f"  OBSERVED: {obs_str}\n"
            f"  DIAGNOSIS: {diagnosis}"
        )

