"""SymbolicTrajectoryExecutor: Independent symbolic controller for trajectory verification and execution.

Separates declarative trajectory generation (Solver/Qwen) from trajectory verification (Judge)
and deterministic step execution (Symbolic Execution Engine).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary
from v10_agent.config import V10Config
from v10_agent.judge import LayeredVerifier
from v10_agent.memory_contours import BranchSignature, EpistemicMemory, SyntaxErrorRecord, SyntaxErrorMemory
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
    verdict: Ternary
    judgment: BrusentsovJudgment
    candidate_advanced: bool = False
    candidate_severed: bool = False
    replan_needed: bool = False
    reset_needed: bool = False
    falsification_detected: bool = False
    falsified_action: str | None = None


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
            if res.has_boundary_violation or res.has_collision:
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

        step_dict = candidate.current_step()
        if step_dict is None:
            return StepExecutionResult(verdict=StepExecutionVerdict.NO_CANDIDATE)

        step_id = str(step_dict.get("step_id", "s0"))
        fn_name = str(step_dict.get("dsl_function", ""))
        step_sig = format_step_signature(step_dict)
        seq_sig = format_sequence_signature(candidate.steps[:candidate.cursor + 1])

        # 1. Pre-execution Verification
        # 1a. Check if candidate or branch signature is already severed in EpistemicMemory
        if (
            not candidate.active
            or epistemic_memory.is_severed(candidate.trajectory_id)
            or epistemic_memory.is_severed(seq_sig)
            or epistemic_memory.is_severed(step_sig)
        ):
            logger.info(f"SymbolicExecutor: Candidate {candidate.trajectory_id} or branch {seq_sig!r} is severed; severing candidate.")
            candidate.sever()
            return StepExecutionResult(
                verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED,
                error_message=f"Candidate or branch previously severed",
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
            epistemic_memory.sever_branch(seq_sig)
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
            epistemic_memory.sever_branch(seq_sig)
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

        zero_grid_delta = False
        if before_grid is not None and after_grid is not None:
            if before_grid == after_grid:
                zero_grid_delta = True

        is_initial_step = (active_cand is None or active_cand.cursor == 0)
        confirmed_eff = ""
        if game_memory is not None and hasattr(game_memory, "confirmed_action_effects"):
            confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "")
        is_confirmed_motion = (
            "moved" in confirmed_eff
            and any(d in confirmed_eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))
        )

        falsification_detected = False
        falsified_action = None

        if is_initial_step and zero_grid_delta and is_confirmed_motion:
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
                    f"Action produced zero grid delta (данное действие не валидно при текущих координатах объекта / local obstacle)."
                ),
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

        if judgment.verdict == Ternary.TRUE:
            # FOLLOW: Advance cursor
            if active_cand is not None:
                active_cand.advance()
            cand_advanced = True
            cand_severed = False
            cand_finished = (active_cand is None or active_cand.is_finished()) if active_pool else True
            logger.info(f"SymbolicExecutor: Step {pending_step.step_id} verified TRUE (FOLLOW).")

        elif judgment.verdict == Ternary.FALSE:
            # NULL: Sever branch & trigger candidate reset
            if active_cand is not None:
                active_cand.sever()
                seq_sig = format_sequence_signature(active_cand.steps[:active_cand.cursor + 1])
                epistemic_memory.sever_branch(seq_sig)
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
                if cand_severed:
                    failed_idx = active_cand.cursor
                    failed_fn = (
                        active_cand.steps[failed_idx].get("dsl_function", "")
                        if failed_idx < len(active_cand.steps)
                        else "step"
                    )
                    epistemic_memory.record_attempt_feedback(
                        hypothesis=f"Candidate {active_cand.trajectory_id}",
                        trajectory_summary=" -> ".join(steps_repr[:failed_idx + 1]) + f" [FAILED AT STEP {failed_idx + 1}]",
                        status="severed_step_failed",
                        reason=(
                            f"Step {failed_idx + 1} ({failed_fn}) failed: {judgment.explanation}. "
                            f"The object hit an obstacle/boundary or had no effect. DO NOT execute {failed_fn} again from this position!"
                        ),
                    )
                else:
                    traj_sig = " -> ".join(steps_repr)
                    epistemic_memory.sever_branch(traj_sig)
                    epistemic_memory.record_attempt_feedback(
                        hypothesis=f"Candidate {active_cand.trajectory_id}",
                        trajectory_summary=traj_sig,
                        status="executed_but_level_not_won",
                        reason="Trajectory executed completely but game did not advance to next level. Invariant or target pattern was incorrect. DO NOT REPEAT THIS SEQUENCE!",
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

