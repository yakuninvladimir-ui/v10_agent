"""GameSession: Master orchestrator of the Tri-Agent pipeline & Double-Loop feedback routing."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Mapping

from v10_agent.action_adapter import to_native_action
from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary, Verdict
from v10_agent.config import V10Config, config_from_mapping
from v10_agent.tracker import PersistentObjectTracker
from v10_agent.dsl_coder import DSLCoder
from v10_agent.explorer_agent import ExplorerAgent, compute_probe_effect
from v10_agent.fallback_symbolic import SymbolicFallbackEngine
from v10_agent.frame_media import render_annotated_frame_png, render_dual_frame_png, render_grid_png
from v10_agent.judge import LayeredVerifier
from v10_agent.llm_advisor import BaseLLMAdvisor, build_llm_advisor
from v10_agent.logging import StructuredAuditLogger
from v10_agent.memory_contours import (
    BranchSignature,
    DefeatExemplar,
    MemoryContourManager,
    SyntaxErrorRecord,
    VictoryExemplar,
    summarize_grid_diff,
)
from v10_agent.observe import grid_to_hex_rows, normalize_observation
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.sandbox import SandboxedModule, SandboxExecutor
from v10_agent.solver_agent import SolverAgent
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import TrajectoryPool
from v10_agent.types import EffectDeclaration, Grid2D
from v10_agent.verification import GroundedStep, GroundingError, VerificationBinder
from v10_agent.cycle_detector import VisibleCycle, _hash_grid

logger = logging.getLogger(__name__)


class LevelAttemptsExhaustedError(RuntimeError):
    """Raised when level retry budget (5 attempts) is exhausted, triggering game_over without reset."""
    pass


from enum import Enum

class SessionPhase(Enum):
    PROBING = "probing"
    CODING = "coding"
    SOLVING = "solving"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    FALLBACK = "fallback"

class PhaseTransition:
    """Определяет допустимые переходы и условия."""
    TRANSITIONS = {
        SessionPhase.PROBING: [SessionPhase.CODING, SessionPhase.FALLBACK, SessionPhase.PROBING, SessionPhase.REFLECTING],
        SessionPhase.CODING: [SessionPhase.SOLVING, SessionPhase.PROBING, SessionPhase.FALLBACK, SessionPhase.REFLECTING],
        SessionPhase.SOLVING: [SessionPhase.EXECUTING, SessionPhase.PROBING, SessionPhase.FALLBACK, SessionPhase.REFLECTING],
        SessionPhase.EXECUTING: [SessionPhase.REFLECTING, SessionPhase.SOLVING, SessionPhase.PROBING, SessionPhase.FALLBACK],
        SessionPhase.REFLECTING: [SessionPhase.PROBING, SessionPhase.SOLVING, SessionPhase.EXECUTING, SessionPhase.FALLBACK],
        SessionPhase.FALLBACK: [SessionPhase.EXECUTING, SessionPhase.PROBING, SessionPhase.SOLVING, SessionPhase.REFLECTING, SessionPhase.FALLBACK],
    }

    @classmethod
    def can_transition(cls, from_phase: SessionPhase, to_phase: SessionPhase) -> bool:
        if from_phase == to_phase:
            return True
        allowed = cls.TRANSITIONS.get(from_phase, [])
        return to_phase in allowed




class GameSession:
    """Session orchestrator managing the full ARC-AGI-3 lifecycle."""

    def __init__(
        self,
        config: V10Config | None = None,
        advisor: BaseLLMAdvisor | None = None,
    ):
        self.config = config or V10Config()
        self.advisor = advisor or build_llm_advisor(self.config)

        # Contours & Services
        self.memory_manager = MemoryContourManager()
        self.sandbox_executor = SandboxExecutor(
            allowed_modules=self.config.sandbox_allowed_modules,
            timeout_seconds=self.config.sandbox_max_cpu_seconds,
        )
        self.binder = VerificationBinder()
        self.verifier = LayeredVerifier(self.config)
        self.fallback_engine = SymbolicFallbackEngine(self.config)
        self.symbolic_executor = SymbolicTrajectoryExecutor(
            config=self.config,
            sandbox_executor=self.sandbox_executor,
            binder=self.binder,
            verifier=self.verifier,
        )
        self.audit_logger = StructuredAuditLogger()

        # Agents
        self.explorer = ExplorerAgent(self.config, self.advisor)
        self.coder = DSLCoder(self.config, self.advisor, self.sandbox_executor)
        self.solver = SolverAgent(self.config, self.advisor)

        # Active Session State
        self.active_module: SandboxedModule | None = None
        self.active_manifest: dict[str, Any] | None = None
        self.active_pool: TrajectoryPool | None = None

        self.last_snapshot: ARGALiteSnapshot | None = None
        self.last_planning_set: PlanningSet | None = None
        self.pending_step: GroundedStep | None = None
        self.pending_action: dict[str, Any] | None = None

        # Divided Fallback & Probe Orchestration State
        self.probe_queue: list[dict[str, Any]] = []
        self.current_phase: SessionPhase = SessionPhase.PROBING
        self.probe_actions_executed_this_level: int = 0
        self.known_actions: set[str] = set()
        self.active_pipeline: str = "discrete"
        self.last_probe_action: dict[str, Any] | None = None
        self.session_aborted: bool = False
        self.replan_requested: bool = False

        self.coder_failed_for_level: bool = False
        self.game_over_reset_count: int = 0
        self.solver_reset_pending: bool = False
        self.solver_reset_reason: str | None = None
        self.last_engine_action: str = ""
        self.current_level_id: str = "level_0"
        self.current_game_id: str = "game_0"
        self.accepted_action_count: int = 0
        self.levels_completed_observed: int = 0
        self.observed_transition_ingestions: int = 0
        self.observed_transition_duplicate_skips: int = 0
        self._level_win_handled: bool = False
        self._last_trajectory_id: str | None = None
        self.crop_offset: int = 0
        self.level_chain_attempts: int = 0
        self.explorer_attempts_this_level: int = 0
        self.explorer_reprobe_pending: bool = False
        self.level_initial_grid: Grid2D | None = None
        self.level_initial_grid_hash: str | None = None
        self.level_executed_actions: list[str] = []
        self.in_persistent_fallback: bool = False
        self._last_attempt_failed: bool = False
        self._last_failure_reason: str = ""
        self._last_failure_summary: str = ""

        # V10.1 Evidence-seeking loop fields
        self.evidence_seeking_active: bool = False
        self.evidence_probes_remaining: int = getattr(self.config, "max_evidence_probes_per_level", 2)
        self.undecided_streak: int = 0
        self.pending_step_snapshot: GroundedStep | None = None
        self.tracker: PersistentObjectTracker | None = getattr(self.verifier, "tracker", None)
        self.pending_cross_level_re_evaluation: bool = False
        self.last_engine_action_source: str = ""

        # V10.1 Telemetry counters
        self.undecided_count: int = 0
        self.undecided_resolved_by_probe: int = 0
        self.undecided_fallback_to_null: int = 0
        self.evidence_probes_executed: int = 0

        # Flash-Next Loop Recovery (VisibleCycle detector)
        min_act = getattr(self.config, "cycle_detector_min_actions", 24)
        max_per = getattr(self.config, "cycle_detector_max_period", 8)
        min_cyc = getattr(self.config, "cycle_detector_min_cycles", 4)
        self.cycle_detector = VisibleCycle(min_actions=min_act, max_period=max_per, min_cycles=min_cyc)
        self.cycle_interventions_this_level: int = 0

    def transition_to(self, new_phase: SessionPhase, reason: str = "") -> None:
        """Explicit state machine transition enforcing allowed lifecycle progression."""
        if not PhaseTransition.can_transition(self.current_phase, new_phase):
            logger.warning(
                f"State Machine: unexpected transition {self.current_phase.value} -> {new_phase.value} ({reason})."
            )
        else:
            logger.debug(f"State Machine: {self.current_phase.value} -> {new_phase.value} ({reason})")
        self.current_phase = new_phase

    @property
    def probing_phase(self) -> bool:
        return self.current_phase == SessionPhase.PROBING

    @probing_phase.setter
    def probing_phase(self, val: bool) -> None:
        if val:
            self.transition_to(SessionPhase.PROBING, "probing_phase set True")
        else:
            if self.current_phase == SessionPhase.PROBING:
                target = SessionPhase.CODING if self.active_module is None else SessionPhase.SOLVING
                self.transition_to(target, "probing_phase set False")

    def update_runtime_config(self, updates: Mapping[str, Any]) -> None:
        """Dynamically update runtime configuration without destroying session memory."""
        self.config.update_runtime(updates)

    def handle_level_transition(self, new_level_id: str) -> None:
        """Clean level-local state, preserve GameMemory cross-level invariants."""
        # Record distilled solution pattern into GameMemory for curriculum progression
        if self.current_level_id:
            game_mem = self.memory_manager.get_game_memory("session")
            setup_str = f"{len(self.last_planning_set.objects) if self.last_planning_set else 'several'} entities"

            winning_cand: dict[str, Any] | None = None
            if self.active_pool and self.active_pool.candidates:
                active_c = self.active_pool.active_candidate() or self.active_pool.candidates[0]
                winning_cand = {
                    "trajectory_id": active_c.trajectory_id,
                    "steps": active_c.steps,
                }

            executed_steps = [
                a for a in self.level_executed_actions
                if a and a.upper() not in ("RESET", "ACTION7")
            ]
            exec_summary = (
                f"Level solved successfully with {len(executed_steps)} coordinated actions."
                if executed_steps
                else "Direct candidate completion"
            )

            # Turn 2: Ask Solver to reflect on the win and revise/distill domain-general invariants
            self.transition_to(SessionPhase.REFLECTING, "revising invariants after level win")
            inv_data = game_mem.get_invariants_for_revision()
            distilled_invariants = self.solver.distill_level_win_invariants(
                winning_candidate=winning_cand,
                execution_summary=exec_summary,
                active_invariants=inv_data.get("active"),
                invalidated_invariants=inv_data.get("invalidated"),
            )

            game_mem.apply_invariant_revision(
                revised_invariants=distilled_invariants,
                outcome="WIN",
                level_id=self.current_level_id,
            )

            primary_inv = " | ".join(distilled_invariants) if distilled_invariants else "Satisfied level goal via coordinated alignment"
            game_mem.record_level_solution(
                level_id=self.current_level_id,
                setup_summary=setup_str,
                invariant_rule=primary_inv,
                winning_macro="",  # Omitted to prevent button-sequence pollution in cross-level memory
            )

            # Compute object diffs between initial frame and winning frame for inter-level victory memory
            object_diffs = []
            static_objects = []
            initial_objects = []
            if self.level_initial_grid and self.last_planning_set:
                try:
                    from v10_agent.arga_lite import extract_arga_snapshot
                    from v10_agent.planning_set import build_planning_set
                    init_snap = extract_arga_snapshot(self.level_initial_grid)
                    init_pset = build_planning_set(init_snap, available_actions=[])
                    for o in init_pset.objects:
                        alias = init_pset.object_real_to_alias.get(o.id, o.id)
                        bbox_dict = o.bbox.to_dict() if hasattr(o.bbox, "to_dict") else [o.bbox.min_row, o.bbox.min_col, o.bbox.max_row, o.bbox.max_col]
                        initial_objects.append({
                            "name": alias,
                            "color": o.color,
                            "bbox": bbox_dict,
                            "role": getattr(o, "role", "UNKNOWN"),
                            "description": getattr(o, "description", ""),
                        })
                    init_by_alias = {init_pset.object_real_to_alias.get(o.id, o.id): o for o in init_pset.objects}
                    win_by_alias = {self.last_planning_set.object_real_to_alias.get(o.id, o.id): o for o in self.last_planning_set.objects}
                    for alias, init_o in init_by_alias.items():
                        win_o = win_by_alias.get(alias)
                        if win_o is not None:
                            dy = win_o.centroid.row - init_o.centroid.row
                            dx = win_o.centroid.col - init_o.centroid.col
                            color_change = f"{init_o.color}->{win_o.color}" if init_o.color != win_o.color else ""
                            if abs(dy) >= 0.5 or abs(dx) >= 0.5 or color_change:
                                object_diffs.append({
                                    "alias": alias,
                                    "id": init_o.id,
                                    "dy": int(round(dy)),
                                    "dx": int(round(dx)),
                                    "color_change": color_change,
                                })
                            else:
                                static_objects.append(alias)
                        else:
                            object_diffs.append({
                                "alias": alias,
                                "id": init_o.id,
                                "status": "gone/merged",
                            })
                except Exception as exc:
                    logger.warning(f"Error computing level victory diffs: {exc}")

            game_mem.record_level_victory_example(
                level_id=self.current_level_id,
                winning_actions=executed_steps,
                object_diffs=object_diffs,
                static_objects=static_objects,
                initial_objects=initial_objects,
                goal_rule=primary_inv,
                start_grid_hash=self.level_initial_grid_hash or "",
                end_grid_hash=self.last_planning_set.grid_hash if self.last_planning_set else "",
            )

            # Grounded victory exemplar for cross-level transfer (xy -> FOLLOW)
            win_action_ints: list[int] = []
            for a in executed_steps:
                m_act = re.search(r"ACTION(\d+)", str(a).upper())
                if m_act:
                    win_action_ints.append(int(m_act.group(1)))
                elif str(a).isdigit():
                    win_action_ints.append(int(a))
            final_act = win_action_ints[-1] if win_action_ints else 0
            target_col = -1
            if primary_inv:
                m_col = re.search(r"color\s+(\d+)", primary_inv, re.IGNORECASE)
                if m_col:
                    target_col = int(m_col.group(1))

            vic_exemplar = VictoryExemplar(
                level_index=max(0, self.levels_completed_observed - 1),
                total_steps=len(executed_steps),
                action_sequence=win_action_ints,
                key_transitions=object_diffs[:5],
                final_action_id=final_act,
                target_color=target_col,
                final_subgrid=self.last_snapshot.grid if self.last_snapshot else [],
                explanation=f"Solved level with {len(executed_steps)} actions. Primary invariant: {primary_inv}",
                winning_invariants_used=[primary_inv] if primary_inv else [],
            )
            game_mem.update_last_victory(vic_exemplar)
            game_mem.record_curriculum_transition(
                level_from=max(0, self.levels_completed_observed - 1),
                level_to=self.levels_completed_observed,
                delta_summary=f"Completed {self.current_level_id} -> {new_level_id}",
            )

            # Mark cross-level invariant discovery & re-evaluation to execute on the pristine initial frame of the new level
            self.pending_cross_level_re_evaluation = True

        self.level_executed_actions.clear()
        ep_mem = self.memory_manager.get_epistemic_memory("session")
        if ep_mem is not None and hasattr(ep_mem, "clear_for_new_level"):
            ep_mem.clear_for_new_level()
        self.level_initial_grid = None
        self.level_initial_grid_hash = None

        self.current_level_id = new_level_id
        self.active_module = None
        self.active_manifest = None
        self.active_pool = None
        self.pending_step = None
        self.pending_action = None
        self.last_probe_action = None
        self.probe_queue = []
        self.transition_to(SessionPhase.PROBING, "new level transition")
        self.probing_phase = True
        self.probe_actions_executed_this_level = 0
        self.game_over_reset_count = 0
        self.level_chain_attempts = 0
        self.explorer_attempts_this_level = 0
        self.explorer_reprobe_pending = False
        # Seed known_actions from GameMemory confirmed kinematics to avoid blind re-probing
        game_mem = self.memory_manager.get_game_memory("session")
        confirmed_kinematics: dict[str, str] = {}
        if hasattr(game_mem, "confirmed_action_effects") and isinstance(game_mem.confirmed_action_effects, dict):
            for act, eff in game_mem.confirmed_action_effects.items():
                act_name = str(act).upper()
                eff_str = str(eff).lower() if isinstance(eff, str) else ""
                is_pure_motion = (
                    any(k in eff_str for k in ("moved", "moves", "dy=", "dx=", "displacement", "shift"))
                    and any(d in eff_str for d in ("up", "down", "left", "right"))
                    and not any(m in eff_str for m in ("selection", "toggle", "indicator", "active entity"))
                    and act_name not in ("ACTION5", "ACTION6", "RESET", "ACTION7")
                )
                if is_pure_motion:
                    confirmed_kinematics[act_name] = str(eff)

        self.known_actions = set(confirmed_kinematics.keys())
        self.active_pipeline = "discrete"
        self.session_aborted = False
        self.replan_requested = False
        self.coder_failed_for_level = False
        self.in_persistent_fallback = False
        self.solver_reset_pending = False
        self.solver_reset_reason = None
        self._level_win_handled = False
        self._last_trajectory_id = None
        self.evidence_seeking_active = False
        self.evidence_probes_remaining = getattr(self.config, "max_evidence_probes_per_level", 2)
        self.undecided_streak = 0
        self.pending_step_snapshot = None
        if self.tracker is not None:
            self.tracker.reset()
        self._last_attempt_failed = False
        self._last_failure_reason = ""
        self._last_failure_summary = ""
        self.level_initial_grid = None
        self.level_initial_grid_hash = None
        if hasattr(self, "cycle_detector"):
            self.cycle_detector.clear()
        self.cycle_interventions_this_level = 0
        self.memory_manager.handle_level_transition(new_level_id)
        if hasattr(self.explorer, "probe_manager") and hasattr(self.explorer.probe_manager, "handle_level_transition"):
            self.explorer.probe_manager.handle_level_transition(confirmed_kinematics)
        logger.info(f"Transitioned to new level {new_level_id}; GameMemory preserved ({len(confirmed_kinematics)} confirmed kinematics carried forward).")

    def handle_game_transition(self, new_game_id: str) -> None:
        """Reset all contours when switching games."""
        self.current_game_id = new_game_id
        self.game_over_reset_count = 0
        self.level_chain_attempts = 0
        self.explorer_attempts_this_level = 0
        self.explorer_reprobe_pending = False
        self.level_executed_actions.clear()
        self.current_level_id = "level_0"
        self.active_module = None
        self.active_manifest = None
        self.active_pool = None
        self.pending_step = None
        self.pending_action = None
        self.last_probe_action = None
        self.probe_queue = []
        self.probing_phase = True
        self.probe_actions_executed_this_level = 0
        self.solver_reset_pending = False
        self.solver_reset_reason = None
        self.level_initial_grid = None
        self.level_initial_grid_hash = None
        self.pending_cross_level_re_evaluation = False
        self.last_engine_action = ""
        self.last_engine_action_source = ""
        if hasattr(self, "cycle_detector") and self.cycle_detector:
            self.cycle_detector.clear()
        self.cycle_interventions_this_level = 0
        if self.tracker is not None:
            self.tracker.reset()
        self.known_actions = set()
        from v10_agent.explorer_agent import PrimitiveProbeManager
        self.explorer.probe_manager = PrimitiveProbeManager(max_probes=self.config.max_primitive_probes_per_level)
        self.active_pipeline = "discrete"
        self.session_aborted = False
        self.replan_requested = False
        self.coder_failed_for_level = False
        self.in_persistent_fallback = False
        self._level_win_handled = False
        self._last_trajectory_id = None
        self._last_attempt_failed = False
        self._last_failure_reason = ""
        self._last_failure_summary = ""
        self.memory_manager.handle_game_transition(new_game_id, self.current_level_id)
        logger.info(f"Reset session for new game {new_game_id}.")

    def act(self, raw_observation: Mapping[str, Any]) -> dict[str, Any]:
        """Propose the single next environment action."""
        raw_game_id = raw_observation.get("game_id")
        if raw_game_id and raw_game_id != self.current_game_id:
            logger.info(f"Game transition detected: {self.current_game_id} -> {raw_game_id}")
            self.handle_game_transition(raw_game_id)

        obs = normalize_observation(
            raw_observation,
            frame_index=self.accepted_action_count,
            game_id=self.current_game_id,
            crop_border=self.config.crop_border_pixels,
        )
        self.crop_offset = int(obs.get("crop_offset", 0))
        state_name = obs["state"]

        # 1. Exact Tufa GAME_OVER single RESET Invariant / Resets disabled / 5-attempt budget
        if state_name == "GAME_OVER":
            if self.last_engine_action == "RESET" and getattr(self, "last_engine_action_source", "") == "tufa_auto_reset":
                logger.error("GAME_OVER persisted after single RESET. Forcing loop break.")
                raise RuntimeError("GAME_OVER persisted after single RESET")

            max_resets = getattr(self.config, "max_game_over_resets_per_level", 5)
            max_attempts = getattr(self.config, "max_chain_attempts_per_level", 5)
            resets_exhausted = self.game_over_reset_count >= max_resets
            attempts_exhausted = (not self.in_persistent_fallback) and (self.level_chain_attempts >= max_attempts)

            if not self.config.reset_on_game_over or resets_exhausted or attempts_exhausted:
                logger.warning(
                    f"GAME_OVER encountered and resets/attempts exhausted (resets: {self.game_over_reset_count}/{max_resets}, "
                    f"solver attempts: {self.level_chain_attempts}/{max_attempts}, fallback: {self.in_persistent_fallback}); abandoning game without reset."
                )
                self.session_aborted = True
                raise LevelAttemptsExhaustedError(
                    f"GAME_OVER encountered and level attempts exhausted ({max_attempts} attempts); "
                    f"transitioning to next game without reset"
                )

            self.game_over_reset_count += 1
            self._last_attempt_failed = True
            self._last_failure_reason = "GAME_OVER encountered during candidate execution."
            executed_steps = [a for a in self.level_executed_actions if a and a.upper() not in ("RESET", "ACTION7")]
            self._last_failure_summary = (
                f"{len(executed_steps)} actions executed before GAME_OVER: {', '.join(executed_steps[:12])}"
                if executed_steps
                else "Immediate GAME_OVER on action"
            )

            # Grounded defeat exemplar (xy'_0 -> NULL)
            try:
                game_mem = self.memory_manager.get_game_memory("session")
                if game_mem is not None:
                    last_act_str = str(executed_steps[-1]) if executed_steps else str(self.last_engine_action or "")
                    m_act = re.search(r"ACTION(\d+)", last_act_str.upper())
                    fatal_act_id = int(m_act.group(1)) if m_act else (int(last_act_str) if last_act_str.isdigit() else 1)
                    fatal_coords = None
                    if isinstance(self.pending_action, dict) and "data" in self.pending_action:
                        data = self.pending_action["data"]
                        if isinstance(data, dict) and "x" in data and "y" in data:
                            fatal_coords = (int(data["x"]), int(data["y"]))

                    actor_pos = (0, 0)
                    if self.last_planning_set and self.last_planning_set.objects:
                        first_obj = self.last_planning_set.objects[0]
                        actor_pos = (int(first_obj.centroid.row), int(first_obj.centroid.col))

                    defeat_ex = DefeatExemplar(
                        level_index=self.levels_completed_observed,
                        fatal_step=len(executed_steps),
                        fatal_action_id=fatal_act_id,
                        fatal_coords=fatal_coords,
                        actor_position_before=actor_pos,
                        hazard_color=-1,
                        pre_defeat_subgrid=self.last_snapshot.grid if self.last_snapshot else [],
                        environment_signal="GAME_OVER",
                        explanation=self._last_failure_summary,
                    )
                    game_mem.update_last_defeat(defeat_ex)
            except Exception as exc:
                logger.warning(f"Error recording defeat exemplar: {exc}")

            self.level_executed_actions.clear()

            # Sever current active candidate
            if self.active_pool and self.active_pool.has_active_candidate():
                active_c = self.active_pool.active_candidate()
                if active_c:
                    self.active_pool.sever(active_c.trajectory_id, "GAME_OVER during execution")

            if self.active_pool and self.active_pool.has_active_candidate():
                logger.info(
                    f"GAME_OVER severed candidate. Retaining active_pool with remaining candidate: "
                    f"{self.active_pool.active_candidate().trajectory_id}"
                )
                self.replan_requested = False
            else:
                self.replan_requested = True
                self.active_pool = None

            reset_action = {
                "id": "RESET",
                "action_id": "RESET",
                "data": {},
                "reasoning": {
                    "source": "tufa_game_over_auto_reset",
                    "reset_count": self.game_over_reset_count,
                    "attempt": self.level_chain_attempts,
                },
            }
            self.pending_action = reset_action
            self.last_engine_action = "RESET"
            self.last_engine_action_source = "tufa_auto_reset"
            self.pending_step = None
            if self.tracker is not None:
                self.tracker.reset()
            return reset_action

        # Check for level change (multi-signal detection)
        observed_levels = obs.get("levels_completed", 0)
        level_transition_detected = False
        if observed_levels > self.levels_completed_observed:
            level_transition_detected = True
            self.levels_completed_observed = observed_levels
        elif state_name == "WIN" and not self._level_win_handled:
            level_transition_detected = True
            self.levels_completed_observed += 1
            self._level_win_handled = True

        if state_name != "WIN":
            self._level_win_handled = False

        if level_transition_detected:
            self.handle_level_transition(f"level_{self.levels_completed_observed}")

        available_actions = obs.get("available_actions", [])

        # 2. Hard Abort Check (if session aborted due to attempt exhaustion)
        if self.session_aborted:
            raise LevelAttemptsExhaustedError("Session aborted due to level attempt exhaustion (5 attempts). Transitioning to next game without reset.")

        # 3. Perception & PlanningSet Construction
        grid = obs["grid"]
        snapshot = extract_arga_snapshot(grid)
        snapshot.levels_completed = int(obs.get("levels_completed", 0) or 0)
        hex_rows = grid_to_hex_rows(grid)
        planning_set = build_planning_set(
            snapshot=snapshot,
            available_actions=available_actions,
            grid_hex_rows=hex_rows,
            crop_offset=self.crop_offset,
        )

        # Record pristine initial frame of level on very first step before any actions
        if self.level_initial_grid is None:
            self.level_initial_grid = [list(row) for row in grid]
            self.level_initial_grid_hash = planning_set.grid_hash
            logger.info(f"Recorded pristine level initial frame for {self.current_level_id} (hash={planning_set.grid_hash[:8]}).")
            if getattr(self, "pending_cross_level_re_evaluation", False):
                self.pending_cross_level_re_evaluation = False
                game_mem = self.memory_manager.get_game_memory("session")
                try:
                    from v10_agent.universal_invariants import discover_invariants
                    new_invs = discover_invariants(planning_set, confirmed_actors=game_mem.confirmed_actors)
                    inv_diff = game_mem.compare_and_record_invariants(new_invs)
                    re_eval_stats = game_mem.re_evaluate_invariants(planning_set, level_id=self.current_level_id)
                    logger.info(
                        f"Cross-level invariant re-evaluation on pristine frame: {re_eval_stats['confirmed']} confirmed, "
                        f"{re_eval_stats['falsified']} falsified, {re_eval_stats['unchanged']} unchanged."
                    )
                except Exception as exc:
                    logger.warning(f"Cross-level invariant re-evaluation on pristine frame skipped: {exc}")
        elif self.last_engine_action == "RESET" and self.level_initial_grid_hash is not None:
            if self.tracker is not None:
                self.tracker.reset()
            if planning_set.grid_hash == self.level_initial_grid_hash:
                logger.info(f"Verified post-reset frame strictly matches initial level frame (hash={planning_set.grid_hash[:8]}).")
            else:
                logger.warning(
                    f"Post-reset frame divergence: initial={self.level_initial_grid_hash[:8]}, current={planning_set.grid_hash[:8]}. "
                    f"Synchronizing reference to fresh reset state."
                )
                self.level_initial_grid = [list(row) for row in grid]
                self.level_initial_grid_hash = planning_set.grid_hash

        env_mem = self.memory_manager.get_env_spec_memory("session")
        syntax_mem = self.memory_manager.get_syntax_error_memory("session")
        ep_mem = self.memory_manager.get_epistemic_memory("session")

        # 3.2. Fast path for persistent symbolic fallback (bypasses LLM, replans, and clean resets)
        if self.in_persistent_fallback:
            self.transition_to(SessionPhase.FALLBACK, "persistent fallback execution")
            effect = self.fallback_engine.select_fallback_action(
                planning_set, game_memory=self.memory_manager.get_game_memory("session")
            )
            action_decl = effect.declared_action
            action_dict = {
                "id": action_decl.action_id,
                "action_id": action_decl.action_id,
                "data": action_decl.data,
                "reasoning": {
                    **action_decl.reasoning,
                    "source": action_decl.reasoning.get("source") or "symbolic_fallback",
                    "strategy": "symbolic_fallback",
                },
            }
            if self.crop_offset > 0 and isinstance(action_dict.get("data"), dict):
                if "x" in action_dict["data"] and "y" in action_dict["data"]:
                    a_data = dict(action_dict["data"])
                    a_data["local_x"] = int(a_data["x"])
                    a_data["local_y"] = int(a_data["y"])
                    a_data["crop_offset"] = self.crop_offset
                    a_data["x"] = int(a_data["x"]) + self.crop_offset
                    a_data["y"] = int(a_data["y"]) + self.crop_offset
                    action_dict["data"] = a_data

            self.last_snapshot = snapshot
            self.last_planning_set = planning_set
            self.pending_step = None
            self.pending_action = action_dict
            self.last_engine_action = action_decl.action_id
            self.level_executed_actions.append(action_decl.action_id)
            self.audit_logger.log(
                "action_emitted",
                action=action_decl.action_id,
                data=action_dict.get("data", action_decl.data),
                strategy="symbolic_fallback",
                grid_hash=planning_set.grid_hash,
                level_id=self.current_level_id,
            )
            return action_dict

        # 3.5. Clean State Reset (Takes precedence to ensure clean board before probing/replan)
        if self.solver_reset_pending:
            self.solver_reset_pending = False
            reset_action = {
                "id": "RESET",
                "action_id": "RESET",
                "data": {},
                "reasoning": {"source": self.solver_reset_reason or "solver_clean_state_reset"},
            }
            self.last_snapshot = snapshot
            self.last_planning_set = planning_set
            self.pending_action = reset_action
            self.last_engine_action = "RESET"
            self.last_engine_action_source = "solver_clean_reset"
            if self.tracker is not None:
                self.tracker.reset()
            self.pending_step = None
            self.level_executed_actions.clear()
            self.audit_logger.log(
                "action_emitted",
                action="RESET",
                data={},
                strategy="solver_reset",
                grid_hash=planning_set.grid_hash,
                level_id=self.current_level_id,
            )
            return reset_action

        # 3.8. V10.1 Evidence-Seeking Blocking Dispatch
        if self.evidence_seeking_active:
            if self.probe_queue:
                probe_action_item = self.probe_queue.pop(0)
                probe_action = (
                    probe_action_item.to_dict()
                    if hasattr(probe_action_item, "to_dict")
                    else dict(probe_action_item)
                )
                self.last_probe_action = probe_action
                self.last_snapshot = snapshot
                self.last_planning_set = planning_set
                self.pending_action = probe_action
                self.pending_step = None
                self.last_engine_action = str(probe_action.get("action_id") or probe_action.get("id") or "ACTION1").upper()
                self.evidence_probes_executed += 1
                self.audit_logger.log(
                    "action_emitted",
                    action=self.last_engine_action,
                    data=probe_action.get("data", {}),
                    strategy="evidence_probe",
                    grid_hash=planning_set.grid_hash,
                    level_id=self.current_level_id,
                )
                return probe_action
            else:
                # Probe queue exhausted: fall through to NULL (sever candidate + clean reset)
                logger.info("Session: Evidence seeking probe queue exhausted. Falling through to NULL.")
                self.undecided_fallback_to_null += 1
                self.evidence_seeking_active = False
                self.undecided_streak = 0
                self.pending_step_snapshot = None
                if self.active_pool and self.active_pool.active_candidate():
                    self.active_pool.active_candidate().sever()
                self.solver_reset_pending = True
                self.solver_reset_reason = "evidence_probe_exhausted_null"

        # 4. Pipeline Determination & Probing Phase
        if self.probing_phase:
            non_meta_actions = [a for a in available_actions if str(a).upper() not in ("RESET", "ACTION7")]
            only_coords = all(str(a).upper() == "ACTION6" for a in non_meta_actions) and len(non_meta_actions) > 0
            coord_quota = 5 if only_coords else 3

            has_coords = "ACTION6" in planning_set.allowed_action_ids or any(
                isinstance(a, str) and any(kw in a.lower() for kw in ("x", "y", "coord", "click"))
                for a in available_actions
            )

            if not self.probe_queue:
                is_dyn = self.explorer.probe_manager.is_dynamic_action_surface(available_actions, self.known_actions)

                has_discrete = any(
                    str(a).upper() in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
                    for a in available_actions
                )
                if has_coords and has_discrete:
                    self.active_pipeline = "hybrid"
                elif has_coords:
                    self.active_pipeline = "coordinate"
                elif is_dyn:
                    self.active_pipeline = "dynamic"
                else:
                    self.active_pipeline = "discrete"

                if not getattr(self.explorer.probe_manager, "_initial_sweep_planned", False):
                    if self.config.enable_primitive_probing:
                        # 1. Plan discrete probes for initial sweep of ACTION1..ACTION5
                        discrete_probes = self.explorer.probe_manager.plan_discrete_probes(
                            planning_set, memory=env_mem, max_probes=self.config.max_primitive_probes_per_level,
                            known_actions=self.known_actions,
                        )
                        self.probe_queue.extend(discrete_probes)

                        # 2. Plan coordinate probes via Qwen if ACTION6 available, not yet confirmed, and not yet probed in env_mem
                        has_tested_coords = any(
                            p.action_id == "ACTION6" for p in env_mem.probe_history
                        )
                        if has_coords and not has_tested_coords and "ACTION6" not in self.known_actions:
                            explorer_mm = getattr(
                                self.config, "explorer_multimodal_enabled",
                                getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
                            )
                            raw_png = render_grid_png(grid) if explorer_mm else None
                            annotated_png = render_annotated_frame_png(grid, planning_set) if explorer_mm else None
                            if raw_png:
                                try:
                                    with open("explorer_raw_frame.png", "wb") as f:
                                        f.write(raw_png)
                                except Exception as e:
                                    logger.debug(f"Frame save skipped: {e}")
                            if annotated_png:
                                try:
                                    with open("explorer_annotated_frame.png", "wb") as f:
                                        f.write(annotated_png)
                                except Exception as e:
                                    logger.debug(f"Frame save skipped: {e}")
                            explorer_images = [raw_png, annotated_png] if explorer_mm else None
                            coord_probes = self.explorer.propose_coordinate_probes(
                                planning_set, memory=env_mem, max_coords=coord_quota, image_png=explorer_images,
                                crop_offset=self.crop_offset,
                            )
                            self.explorer_attempts_this_level += 1
                            logger.info(
                                f"Explorer initial coordinate probes: attempt {self.explorer_attempts_this_level}/"
                                f"{getattr(self.config, 'max_explorer_attempts_per_level', 5)} (quota={coord_quota})"
                            )
                            self.probe_queue.extend(coord_probes)
                    elif has_coords and "ACTION6" not in self.known_actions:
                        coord_probes = self.explorer.probe_manager.plan_targeted_coordinate_probes(
                            planning_set, available_actions, memory=env_mem, crop_offset=self.crop_offset,
                        )
                        self.probe_queue.extend(coord_probes)
                    else:
                        self.probing_phase = False
                else:
                    # Initial sweep already executed: query combinatorial chaining for inactive actions
                    if self.config.enable_primitive_probing:
                        next_chain = self.explorer.probe_manager.get_next_combinatorial_chain()
                        if next_chain:
                            self.probe_queue.extend(next_chain)

                    # Continuous coordinate probing without reset if ACTION6 is unconfirmed
                    if not self.probe_queue and has_coords and "ACTION6" not in self.explorer.probe_manager.confirmed_effective_actions:
                        max_exp_attempts = getattr(self.config, "max_explorer_attempts_per_level", 5)
                        while not self.probe_queue and self.explorer_attempts_this_level < max_exp_attempts:
                            explorer_mm = getattr(
                                self.config, "explorer_multimodal_enabled",
                                getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
                            )
                            raw_png = render_grid_png(grid) if explorer_mm else None
                            annotated_png = render_annotated_frame_png(grid, planning_set) if explorer_mm else None
                            if raw_png:
                                try:
                                    with open("explorer_raw_frame.png", "wb") as f:
                                        f.write(raw_png)
                                except Exception as e:
                                    logger.debug(f"Frame save skipped: {e}")
                            if annotated_png:
                                try:
                                    with open("explorer_annotated_frame.png", "wb") as f:
                                        f.write(annotated_png)
                                except Exception as e:
                                    logger.debug(f"Frame save skipped: {e}")
                            explorer_images = [raw_png, annotated_png] if explorer_mm else None
                            coord_probes = self.explorer.propose_coordinate_probes(
                                planning_set, memory=env_mem, max_coords=coord_quota, image_png=explorer_images,
                                crop_offset=self.crop_offset,
                            )
                            self.explorer_attempts_this_level += 1
                            logger.info(
                                f"Explorer coordinate probes: continuous attempt {self.explorer_attempts_this_level}/"
                                f"{max_exp_attempts} (quota={coord_quota})"
                            )
                            if coord_probes:
                                self.probe_queue.extend(coord_probes)
                                break

            if self.active_pipeline == "dynamic" and not self.probe_queue and self.probing_phase:
                new_actions = [a for a in available_actions if a not in self.known_actions and a not in ("RESET", "ACTION7")]
                if new_actions:
                    new_probes = self.explorer.probe_manager.plan_discrete_probes(
                        new_actions, memory=env_mem, max_probes=self.config.max_primitive_probes_per_level,
                        known_actions=self.known_actions,
                    )
                    self.probe_queue.extend(new_probes)

            if self.probe_queue:
                probe_action_item = self.probe_queue.pop(0)
                probe_action = (
                    probe_action_item.to_dict()
                    if hasattr(probe_action_item, "to_dict")
                    else dict(probe_action_item)
                )
                if self.crop_offset > 0 and isinstance(probe_action.get("data"), dict):
                    if "x" in probe_action["data"] and "y" in probe_action["data"]:
                        p_data = dict(probe_action["data"])
                        p_data["local_x"] = int(p_data["x"])
                        p_data["local_y"] = int(p_data["y"])
                        p_data["crop_offset"] = self.crop_offset
                        p_data["x"] = int(p_data["x"]) + self.crop_offset
                        p_data["y"] = int(p_data["y"]) + self.crop_offset
                        probe_action["data"] = p_data

                self.last_probe_action = probe_action
                self.last_snapshot = snapshot
                self.last_planning_set = planning_set
                self.pending_action = probe_action
                self.pending_step = None
                self.last_engine_action = probe_action["action_id"]
                if probe_action["action_id"] != "RESET":
                    self.known_actions.add(probe_action["action_id"])
                    self.probe_actions_executed_this_level += 1

                self.audit_logger.log(
                    "action_emitted",
                    action=probe_action["action_id"],
                    data=probe_action.get("data", {}),
                    strategy="probe",
                    grid_hash=planning_set.grid_hash,
                    level_id=self.current_level_id,
                )
                return probe_action

            if self.probing_phase:
                game_mem = self.memory_manager.get_game_memory("session")
                has_effective = bool(self.explorer.probe_manager.confirmed_effective_actions) or bool(game_mem.confirmed_action_effects)

                max_exp_attempts = getattr(self.config, "max_explorer_attempts_per_level", 5)
                if not has_effective and has_coords and self.explorer_attempts_this_level >= max_exp_attempts:
                    self.session_aborted = True
                    logger.error(
                        f"Explorer probe budget exhausted ({max_exp_attempts} attempts) on {self.current_level_id} "
                        f"without discovering effective actions. Transitioning to next game without reset."
                    )
                    raise LevelAttemptsExhaustedError(
                        f"Explorer probe budget exhausted ({max_exp_attempts} attempts) on {self.current_level_id} "
                        f"without discovering effective actions. Transitioning to next game without reset."
                    )

                self.probing_phase = False
                self.explorer_reprobe_pending = False
                if self.probe_actions_executed_this_level > 0:
                    self.probe_actions_executed_this_level = 0
                    reset_action = {
                        "id": "RESET",
                        "action_id": "RESET",
                        "data": {},
                        "reasoning": {"source": "probe_phase_complete_reset_to_pristine"},
                    }
                    self.last_probe_action = None
                    self.last_snapshot = snapshot
                    self.last_planning_set = planning_set
                    self.pending_action = reset_action
                    self.pending_step = None
                    self.last_engine_action = "RESET"
                    self.last_engine_action_source = "probe_phase_reset"
                    if self.tracker is not None:
                        self.tracker.reset()
                    self.level_executed_actions.clear()
                    self.audit_logger.log(
                        "action_emitted",
                        action="RESET",
                        data={},
                        strategy="probe_phase_reset",
                        grid_hash=planning_set.grid_hash,
                        level_id=self.current_level_id,
                    )
                    return reset_action

        # 4.5. Solver Clean State Reset (Single clean reset between candidates or after replan)
        if self.solver_reset_pending:
            self.solver_reset_pending = False
            reset_action = {
                "id": "RESET",
                "action_id": "RESET",
                "data": {},
                "reasoning": {"source": self.solver_reset_reason or "solver_clean_state_reset"},
            }
            self.last_snapshot = snapshot
            self.last_planning_set = planning_set
            self.pending_action = reset_action
            self.last_engine_action = "RESET"
            self.last_engine_action_source = "solver_clean_state_reset"
            if self.tracker is not None:
                self.tracker.reset()
            self.pending_step = None
            self.level_executed_actions.clear()
            self.audit_logger.log(
                "action_emitted",
                action="RESET",
                data={},
                strategy="solver_reset",
                grid_hash=planning_set.grid_hash,
                level_id=self.current_level_id,
            )
            return reset_action

        # 5. Explorer Phase (Spec Generation & Scene Synthesis)
        game_mem = self.memory_manager.get_game_memory("session")
        if not env_mem.specs:
            explorer_mm = getattr(
                self.config, "explorer_multimodal_enabled",
                getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
            )
            raw_png = render_grid_png(grid) if explorer_mm else None
            annotated_png = render_annotated_frame_png(grid, planning_set) if explorer_mm else None
            if raw_png:
                try:
                    with open("explorer_raw_frame.png", "wb") as f:
                        f.write(raw_png)
                except Exception as e:
                    logger.debug(f"Frame save skipped: {e}")
            if annotated_png:
                try:
                    with open("explorer_annotated_frame.png", "wb") as f:
                        f.write(annotated_png)
                except Exception as e:
                    logger.debug(f"Frame save skipped: {e}")
            explorer_images = [raw_png, annotated_png] if explorer_mm else None
            self.explorer.synthesize_level_spec(
                planning_set=planning_set,
                memory=env_mem,
                image_png=explorer_images,
                game_memory=game_mem,
            )

        # 6. Coder Phase (DSL Generation with up to 3 retries)
        if self.active_module is None and not self.coder_failed_for_level:
            spec = env_mem.specs[0] if env_mem.specs else {}
            coder_mm = getattr(
                self.config, "coder_multimodal_enabled",
                getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
            )
            raw_png = render_grid_png(grid) if coder_mm else None
            annotated_png = render_annotated_frame_png(grid, planning_set) if coder_mm else None
            if raw_png:
                try:
                    with open("coder_raw_frame.png", "wb") as f:
                        f.write(raw_png)
                except Exception as e:
                    logger.debug(f"Frame save skipped: {e}")
            if annotated_png:
                try:
                    with open("coder_annotated_frame.png", "wb") as f:
                        f.write(annotated_png)
                except Exception as e:
                    logger.debug(f"Frame save skipped: {e}")
            coder_images = [raw_png, annotated_png] if coder_mm else None
            self.transition_to(SessionPhase.CODING, "generating dsl with coder")
            from v10_agent.types import CoderInput
            _ = CoderInput.from_session(spec, syntax_mem, list(self.known_actions))
            module, manifest, errors = self.coder.generate_dsl(spec, syntax_mem, planning_set, game_memory=game_mem, image_png=coder_images)
            if module is not None and manifest is not None:
                # Gating: prune unconfirmed actions from manifest functions before certifying
                conf = set(self.known_actions)
                if game_mem and hasattr(game_mem, "confirmed_action_effects") and game_mem.confirmed_action_effects:
                    conf.update(str(a).upper() for a in game_mem.confirmed_action_effects)
                if hasattr(self.explorer, "probe_manager"):
                    conf.update(str(a).upper() for a in self.explorer.probe_manager.confirmed_effective_actions)

                if conf:
                    manifest["functions"] = [
                        fn for fn in manifest.get("functions", [])
                        if not re.match(r"^ACTION\d+$", str(fn.get("name", "")).upper())
                        or str(fn.get("name", "")).upper() in conf
                    ]
                self.active_module = module
                self.active_manifest = manifest
                for fn in manifest.get("functions", []):
                    fn_name = str(fn.get("name", "")).upper()
                    # Only certify primitives that correspond to confirmed effective actions
                    act_match = next((act for act in conf if act in fn_name), None)
                    if act_match or "ACTION" not in fn_name:
                        game_mem.record_reusable_primitive(fn)
                logger.info(f"DSL Certified with {len(manifest.get('functions', []))} primitives.")
            else:
                self.coder_failed_for_level = True
                retries = self.config.max_coder_retries_per_level
                logger.warning(f"DSLCoder retries exhausted ({retries} attempts).")
                if self.config.coder_exhaustion_forces_fallback and not self.config.abort_on_dsl_exhaustion:
                    self.in_persistent_fallback = True
                    logger.info("Engaging persistent symbolic fallback after Coder exhaustion.")
                else:
                    self.session_aborted = True
                    logger.error(f"DSLCoder retries exhausted ({retries} attempts). Transitioning to next game without reset.")
                    raise LevelAttemptsExhaustedError(f"Coder retries exhausted ({retries} attempts). Transitioning to next game without reset.")

        # 7. Solver Phase (Trajectory Generation & Replanning)
        if self.replan_requested:
            self.active_pool = None
            self.replan_requested = False

        if not self.in_persistent_fallback and self.active_module is not None and (self.active_pool is None or self.active_pool.active_candidate() is None):
            # Ironclad guarantee: before calling Solver, if board is dirty from prior execution, emit RESET first
            if self.level_initial_grid_hash is not None and planning_set.grid_hash != self.level_initial_grid_hash:
                logger.info(
                    f"Solver replan requires clean board (current hash {planning_set.grid_hash[:8]} != initial {self.level_initial_grid_hash[:8]}). "
                    f"Emitting clean RESET before invoking Solver."
                )
                reset_action = {
                    "id": "RESET",
                    "action_id": "RESET",
                    "data": {},
                    "reasoning": {"source": "solver_replan_clean_state_reset"},
                }
                self.last_snapshot = snapshot
                self.last_planning_set = planning_set
                self.pending_action = reset_action
                self.last_engine_action = "RESET"
                self.pending_step = None
                self.level_executed_actions.clear()
                self.active_pool = None
                self.replan_requested = False
                return reset_action

            max_attempts = getattr(self.config, "max_chain_attempts_per_level", 5)
            if self.level_chain_attempts >= max_attempts:
                self._last_attempt_failed = False
                logger.warning(f"Level chain attempt budget exhausted ({max_attempts}) for {self.current_level_id}.")
                if self.config.enable_symbolic_fallback and getattr(self.config, "solver_exhaustion_forces_fallback", True):
                    self.in_persistent_fallback = True
                    logger.info("Engaging persistent symbolic fallback after Solver attempt budget exhausted.")
                else:
                    self.session_aborted = True
                    logger.warning(f"Level chain attempt budget exhausted ({max_attempts}) for {self.current_level_id}. Transitioning to next game without reset.")
                    raise LevelAttemptsExhaustedError(f"Level chain attempt budget exhausted ({max_attempts} attempts). Transitioning to next game without reset.")
            else:
                self._last_attempt_failed = False
                self.level_chain_attempts += 1
                logger.info(f"Initiating Solver planning attempt {self.level_chain_attempts}/{max_attempts} for {self.current_level_id}...")
                solver_mm = getattr(
                    self.config, "solver_multimodal_enabled",
                    getattr(self.config, "multimodal_enabled", getattr(self.config, "qwen_multimodal_enabled", True))
                )
                raw_png = render_grid_png(grid) if solver_mm else None
                annotated_png = render_annotated_frame_png(grid, planning_set) if solver_mm else None
                if raw_png:
                    try:
                        with open("solver_raw_frame.png", "wb") as f:
                            f.write(raw_png)
                    except Exception as e:
                        logger.debug(f"Frame save skipped: {e}")
                if annotated_png:
                    try:
                        with open("solver_annotated_frame.png", "wb") as f:
                            f.write(annotated_png)
                    except Exception as e:
                        logger.debug(f"Frame save skipped: {e}")
                solver_images = [raw_png, annotated_png] if solver_mm else None
                matches_initial = (self.level_initial_grid_hash is not None and planning_set.grid_hash == self.level_initial_grid_hash)
                logger.info(
                    f"Solver invocation prepared: frame hash={planning_set.grid_hash[:8]} "
                    f"(verified matches initial level frame: {matches_initial}, png_attached={solver_images is not None})."
                )
                self.transition_to(SessionPhase.SOLVING, "generating solver trajectory package")
                from v10_agent.types import SolverInput
                _ = SolverInput.from_session(
                    planning_set=planning_set,
                    game_memory=game_mem,
                    epistemic_memory=ep_mem,
                    action_budget=max(1, self.config.max_actions_per_level - len(self.level_executed_actions)),
                )
                max_att = max_attempts
                rem_att = max(1, max_att - self.level_chain_attempts + 1)
                rem_probes = getattr(self, "evidence_probes_remaining", 3)
                max_probes = getattr(self.config, "max_evidence_probes_per_level", 3)
                pkg = self.solver.generate_trajectory_package(
                    manifest=self.active_manifest or {},
                    planning_set=planning_set,
                    epistemic_memory=ep_mem,
                    budget=max(1, self.config.max_actions_per_level - len(self.level_executed_actions)),
                    image_png=solver_images,
                    game_memory=game_mem,
                    attempts_remaining=rem_att,
                    max_attempts=max_att,
                    probes_remaining=rem_probes,
                    max_probes=max_probes,
                    env_spec=env_mem.specs[0] if env_mem.specs else None,
                )
                if pkg is not None:
                    self.active_pool = TrajectoryPool.from_package(pkg)
                    c_summaries = [f"{c.trajectory_id}({len(c.steps)} steps: {[s.get('dsl_function') for s in c.steps[:6]]}...)" for c in self.active_pool.candidates]
                    logger.info(f"Solver generated {len(self.active_pool.candidates)} candidates: {'; '.join(c_summaries)}")
                else:
                    logger.warning(f"Solver trajectory proposal failed on attempt {self.level_chain_attempts}/{max_attempts}.")
                    self._last_attempt_failed = True
                    self._last_failure_reason = f"Solver trajectory proposal failed on attempt {self.level_chain_attempts}/{max_attempts}."
                    self._last_failure_summary = "No valid candidates generated."
                    if self.level_chain_attempts >= max_attempts:
                        if self.config.enable_symbolic_fallback and getattr(self.config, "solver_exhaustion_forces_fallback", True):
                            self.in_persistent_fallback = True
                            logger.info("Solver proposal retries exhausted: engaging persistent symbolic fallback.")
                        else:
                            self.session_aborted = True
                            raise LevelAttemptsExhaustedError(f"Solver trajectory proposal retries exhausted ({max_attempts} attempts). Transitioning to next game without reset.")

        # 8. Symbolic Step Verification & Execution (Independent from Qwen)
        effect: EffectDeclaration | None = None
        grounded_step: GroundedStep | None = None
        strategy: str = "symbolic_fallback"

        if not self.in_persistent_fallback and self.active_module is not None and self.active_pool is not None and self.active_pool.active_candidate() is not None:
            self.transition_to(SessionPhase.EXECUTING, "executing symbolic step")
            exec_res = self.symbolic_executor.prepare_and_execute_step(
                pool=self.active_pool,
                planning_set=planning_set,
                active_module=self.active_module,
                epistemic_memory=ep_mem,
                syntax_memory=syntax_mem,
                game_memory=self.memory_manager.get_game_memory("session"),
            )

            effect = exec_res.effect
            grounded_step = exec_res.grounded_step
            strategy = exec_res.strategy

            if exec_res.circuit_broken:
                next_cand = self.active_pool.active_candidate() if self.active_pool else None
                if next_cand is None:
                    self._last_attempt_failed = True
                    self._last_failure_reason = f"Candidate pool exhausted; circuit breaker triggered: {exec_res.error_message}"
                    executed_steps = [a for a in self.level_executed_actions if a and a.upper() not in ("RESET", "ACTION7")]
                    self._last_failure_summary = (
                        f"{len(executed_steps)} actions executed before circuit break: {', '.join(executed_steps[:12])}"
                        if executed_steps
                        else "Candidate failed on initial step"
                    )
                    max_attempts = getattr(self.config, "max_chain_attempts_per_level", 5)
                    if self.level_chain_attempts >= max_attempts:
                        logger.warning(f"All candidates failed and attempt budget ({max_attempts}) reached. Switching to persistent fallback.")
                        if self.config.enable_symbolic_fallback and getattr(self.config, "solver_exhaustion_forces_fallback", True):
                            self.in_persistent_fallback = True
                            self.active_pool = None
                            self.replan_requested = False
                        else:
                            self.session_aborted = True
                            raise LevelAttemptsExhaustedError(f"All candidates failed and attempt budget exhausted ({max_attempts} attempts). Transitioning to next game without reset.")
                    else:
                        self.replan_requested = True
                        self.active_pool = None
                else:
                    logger.info(f"Session: Circuit broken on candidate; advancing to next pool candidate: {next_cand.trajectory_id}")

                if not self.in_persistent_fallback:
                    reset_action = {
                        "id": "RESET",
                        "action_id": "RESET",
                        "data": {},
                        "reasoning": {"source": "circuit_breaker_immediate_reset", "error": exec_res.error_message},
                    }
                    self.pending_action = reset_action
                    self.last_engine_action = "RESET"
                    self.last_engine_action_source = "circuit_breaker_reset"
                    if self.tracker is not None:
                        self.tracker.reset()
                    self.pending_step = None
                    self.level_executed_actions.clear()
                    return reset_action

        # 9. Fallback Path
        if effect is None or self.in_persistent_fallback:
            if not self.config.enable_symbolic_fallback or (self.session_aborted and not self.in_persistent_fallback):
                self.session_aborted = True
                raise LevelAttemptsExhaustedError("DSL missing or session aborted due to retry exhaustion. Transitioning to next game without reset.")

            self.in_persistent_fallback = True
            self.transition_to(SessionPhase.FALLBACK, "selecting fallback action")
            effect = self.fallback_engine.select_fallback_action(
                planning_set, game_memory=self.memory_manager.get_game_memory("session")
            )
            grounded_step = None
            strategy = "symbolic_fallback"

        # 10. ActionBoundary: emit strictly one step
        action_decl = effect.declared_action
        action_dict = {
            "id": action_decl.action_id,
            "action_id": action_decl.action_id,
            "data": action_decl.data,
            "reasoning": {
                **action_decl.reasoning,
                "source": action_decl.reasoning.get("source") or ("dsl_execution" if grounded_step else "fallback"),
                "strategy": strategy,
            },
        }

        if self.crop_offset > 0 and isinstance(action_dict.get("data"), dict):
            if "x" in action_dict["data"] and "y" in action_dict["data"]:
                a_data = dict(action_dict["data"])
                a_data["local_x"] = int(a_data["x"])
                a_data["local_y"] = int(a_data["y"])
                a_data["crop_offset"] = self.crop_offset
                a_data["x"] = int(a_data["x"]) + self.crop_offset
                a_data["y"] = int(a_data["y"]) + self.crop_offset
                action_dict["data"] = a_data

        self.last_snapshot = snapshot
        self.last_planning_set = planning_set
        self.pending_step = grounded_step
        self.pending_action = action_dict
        self.last_engine_action = action_decl.action_id
        self.last_engine_action_source = "candidate" if grounded_step else "fallback"
        self.level_executed_actions.append(action_decl.action_id)

        self.audit_logger.log(
            "action_emitted",
            action=action_decl.action_id,
            data=action_dict.get("data", action_decl.data),
            strategy=strategy,
            grid_hash=planning_set.grid_hash,
            level_id=self.current_level_id,
        )
        return action_dict

    def _check_and_invalidate_dsl_for_new_action(self, action_id: str, effect_desc: str = "") -> bool:
        """Check if action_id is a confirmed action missing from active_manifest, and invalidate DSL if so."""
        if not action_id or str(action_id).upper() in ("RESET", "ACTION7"):
            return False
        act_up = str(action_id).upper()
        active_fns: set[str] = set()
        if self.active_manifest and isinstance(self.active_manifest, dict):
            for fn in self.active_manifest.get("functions", []):
                if isinstance(fn, dict):
                    if fn.get("name"):
                        active_fns.add(str(fn["name"]).upper())
                    if fn.get("action_id"):
                        active_fns.add(str(fn["action_id"]).upper())
        is_new_confirmed_action = act_up not in active_fns
        if self.active_module is not None and is_new_confirmed_action:
            logger.info(
                f"Session: New confirmed action {act_up} discovered with effect '{effect_desc}'; "
                f"invalidating active DSL module for re-synthesis."
            )
            self.active_module = None
            self.active_manifest = None
            self.coder_failed_for_level = False
            self.replan_requested = True

            # Also ensure env_spec reflects the new action if specs exist
            env_mem = self.memory_manager.get_env_spec_memory("session")
            if env_mem and env_mem.specs:
                spec = env_mem.specs[0]
                if "available_actions" in spec and isinstance(spec["available_actions"], list):
                    if act_up not in [str(a).upper() for a in spec["available_actions"]]:
                        spec["available_actions"].append(act_up)
            return True
        return False

    def observe_action_result(self, after_observation: Mapping[str, Any] | None = None) -> bool:
        """Commit the transition result and evaluate Brusentsov ternary judgment."""
        if self.pending_action is None:
            self.observed_transition_duplicate_skips += 1
            return False

        if after_observation is None:
            self.observed_transition_duplicate_skips += 1
            return False

        norm_after = normalize_observation(
            after_observation,
            game_id=self.current_game_id,
            crop_border=self.config.crop_border_pixels,
        )
        self.accepted_action_count += 1
        self.observed_transition_ingestions += 1

        # A. If the action was a probe, record the effect in PrimitiveProbeManager
        if self.last_probe_action is not None:
            action_id = self.last_probe_action.get("action_id", "")
            action_data = self.last_probe_action.get("data", {})
            if self.last_snapshot is not None and action_id and action_id != "RESET":
                env_mem = self.memory_manager.get_env_spec_memory("session")
                rec = self.explorer.probe_manager.record_probe_result(
                    action_id=action_id,
                    action_data=action_data,
                    before_snapshot=self.last_snapshot,
                    after_obs=norm_after,
                    memory=env_mem,
                    planning_set=self.last_planning_set,
                )
                game_mem = self.memory_manager.get_game_memory("session")
                if "moved" in rec.observed_effect and any(d in rec.observed_effect for d in ("UP", "DOWN", "LEFT", "RIGHT")):
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                elif "color transition" in rec.observed_effect:
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                elif any(k in rec.observed_effect for k in ("selection indicator", "active entity toggled", "state toggle")):
                    act_desc = f"{action_id}({action_data})" if action_data else action_id
                    offset_note = ""
                    try:
                        after_grid = norm_after.get("grid")
                        if after_grid:
                            after_snap = extract_arga_snapshot(after_grid)
                            after_pset = build_planning_set(after_snap, available_actions=list(self.known_actions))
                            from v10_agent.universal_invariants import discover_invariants
                            invs = discover_invariants(after_pset)
                    except Exception as exc:
                        logger.warning(f"Failed to extract post-toggle invariants: {exc}")
                    game_mem.record_selection_mechanic(f"{act_desc}: {rec.observed_effect}")
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                elif "object count changed" in rec.observed_effect:
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                else:
                    if action_id.upper() != "ACTION6":
                        game_mem.record_unconfirmed_action(action_id, rec.observed_effect)
                    else:
                        logger.info(f"ACTION6 coordinate probe at {action_data} produced no effect; not quarantining ACTION6.")

                # Reactive DSL invalidation upon discovering any new confirmed action
                is_effective_effect = (
                    "moved" in rec.observed_effect
                    or "color transition" in rec.observed_effect
                    or "object count changed" in rec.observed_effect
                    or any(k in rec.observed_effect for k in ("selection indicator", "active entity toggled", "state toggle"))
                    or self.explorer.probe_manager.is_action_effective(rec.observed_effect)
                )
                if is_effective_effect:
                    self._check_and_invalidate_dsl_for_new_action(action_id, rec.observed_effect)

                # Trigger dynamic reprobes (max 2 steps) if this action altered state or toggled selection
                reprobes = self.explorer.probe_manager.get_dynamic_reprobes(action_id, rec.observed_effect, max_steps=2)
                if reprobes:
                    for r in reversed(reprobes):
                        self.probe_queue.insert(0, r)

            if self.evidence_seeking_active and self.pending_step_snapshot is not None:
                re_eval = self.symbolic_executor.evaluate_transition(
                    pending_step=self.pending_step_snapshot,
                    before_snapshot=self.last_snapshot,
                    after_obs=norm_after,
                    planning_set=self.last_planning_set,
                    active_pool=None,  # Do not advance candidate during evidence probe
                    epistemic_memory=self.memory_manager.get_epistemic_memory("session"),
                    game_memory=self.memory_manager.get_game_memory("session"),
                    action_dict=self.last_probe_action or {"action_id": rec.action_id, "data": rec.action_data},
                )
                if re_eval.verdict == Verdict.UNDECIDED or re_eval.evidence_needed:
                    self.undecided_streak += 1
                    max_streak = getattr(self.config, "max_undecided_streak", 2)
                    budget_rem = getattr(self.config, "max_actions_per_level", 250) - len(self.level_executed_actions)
                    if self.evidence_probes_remaining > 0 and self.undecided_streak < max_streak and budget_rem >= 25:
                        probe_act = {"id": "ACTION1", "action_id": "ACTION1", "data": {}, "reasoning": {"source": "evidence_seeking"}}
                        if re_eval.evidence_hint and re_eval.evidence_hint.startswith("probe_ACTION"):
                            act_str = re_eval.evidence_hint.replace("probe_", "").upper()
                            if act_str in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                                probe_act = {"id": act_str, "action_id": act_str, "data": {}, "reasoning": {"source": "evidence_seeking"}}
                        self.probe_queue.insert(0, probe_act)
                        self.evidence_probes_remaining -= 1
                    else:
                        logger.info(f"Session: UNDECIDED streak limit ({self.undecided_streak}) or budget exhausted. Treating as NULL.")
                        self.undecided_fallback_to_null += 1
                        self.evidence_seeking_active = False
                        self.undecided_streak = 0
                        self.pending_step_snapshot = None
                        if self.active_pool and self.active_pool.active_candidate():
                            self.active_pool.active_candidate().sever()
                        self.solver_reset_pending = True
                        self.solver_reset_reason = "undecided_streak_exhausted_null"
                else:
                    logger.info("Session: Epistemic ambiguity resolved by evidence probe.")
                    self.undecided_resolved_by_probe += 1
                    self.evidence_seeking_active = False
                    self.undecided_streak = 0
                    self.pending_step_snapshot = None

            self.last_probe_action = None

        # B. Evaluate transition if a grounded solver step was pending
        if self.pending_step is not None and self.last_snapshot is not None and self.last_planning_set is not None:
            ep_mem = self.memory_manager.get_epistemic_memory("session")
            game_mem = self.memory_manager.get_game_memory("session")
            eval_res = self.symbolic_executor.evaluate_transition(
                pending_step=self.pending_step,
                before_snapshot=self.last_snapshot,
                after_obs=norm_after,
                planning_set=self.last_planning_set,
                active_pool=self.active_pool,
                epistemic_memory=ep_mem,
                game_memory=game_mem,
                action_dict=self.pending_action,
            )

            # Reactive DSL Invalidation: Check if step action had observable delta and is not in active manifest
            step_act_id = str(
                (self.pending_action.get("action_id") or self.pending_action.get("id") or "")
                if isinstance(self.pending_action, dict) else self.pending_action
            ).upper()
            before_grid = getattr(self.last_snapshot, "grid", None)
            after_grid = norm_after.get("grid")
            has_observable_delta = (
                (before_grid is not None and after_grid is not None and before_grid != after_grid)
                or getattr(eval_res.judgment, "is_effective", False)
            )
            if has_observable_delta and step_act_id and step_act_id not in ("RESET", "ACTION7"):
                self.known_actions.add(step_act_id)
                if game_mem and step_act_id not in getattr(game_mem, "confirmed_action_effects", {}):
                    effect_desc = getattr(eval_res.judgment, "explanation", "") or "observable transition delta"
                    game_mem.record_action_effect(step_act_id, effect_desc)
                self._check_and_invalidate_dsl_for_new_action(step_act_id, "observable solver step delta")

            if eval_res.verdict == Verdict.UNDECIDED or eval_res.evidence_needed:
                self.undecided_count += 1
                self.undecided_streak = 1
                max_streak = getattr(self.config, "max_undecided_streak", 2)
                budget_rem = getattr(self.config, "max_actions_per_level", 250) - len(self.level_executed_actions)
                if self.evidence_probes_remaining > 0 and self.undecided_streak <= max_streak and budget_rem >= 25:
                    self.evidence_seeking_active = True
                    self.pending_step_snapshot = self.pending_step
                    probe_act = {"id": "ACTION1", "action_id": "ACTION1", "data": {}, "reasoning": {"source": "evidence_seeking"}}
                    hint = eval_res.evidence_hint or ""
                    if hint.startswith("probe_ACTION"):
                        act_str = hint.replace("probe_", "").upper()
                        if act_str in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                            probe_act = {"id": act_str, "action_id": act_str, "data": {}, "reasoning": {"source": "evidence_seeking"}}
                    self.probe_queue.insert(0, probe_act)
                    self.evidence_probes_remaining -= 1
                    logger.info(f"Session: Evidence seeking active on step {self.pending_step.step_id}. Enqueued probe {probe_act.get('id')}.")
                else:
                    logger.info(f"Session: Step {self.pending_step.step_id} UNDECIDED but probe budget exhausted. Treating as NULL.")
                    self.undecided_fallback_to_null += 1
                    self.evidence_seeking_active = False
                    self.undecided_streak = 0
                    self.pending_step_snapshot = None
                    if self.active_pool and self.active_pool.active_candidate():
                        self.active_pool.active_candidate().sever()
                    self.solver_reset_pending = True
                    self.solver_reset_reason = "undecided_budget_exhausted_null"

            elif eval_res.falsification_detected:
                falsified = eval_res.falsified_action or "initial_motion"
                logger.warning(
                    f"Session: Действие {falsified} не валидно при текущих координатах объекта. "
                    f"Invalidating DSL and scheduling clean micro-reprobe cycle."
                )
                self.active_module = None
                self.active_manifest = None
                self.active_pool = None
                self.replan_requested = True
                self._last_attempt_failed = True
                self._last_failure_reason = f"Transition falsification detected: action {falsified} invalid or produced unexpected transition."
                executed_steps = [a for a in self.level_executed_actions if a and a.upper() not in ("RESET", "ACTION7")]
                self._last_failure_summary = (
                    f"{len(executed_steps)} actions executed before falsification: {', '.join(executed_steps[:12])}"
                    if executed_steps
                    else "Falsification detected on initial step"
                )

                # Invalidate stale kinematics in GameMemory and known_actions
                if eval_res.falsified_action:
                    game_mem.invalidate_action_effect(eval_res.falsified_action, level_id=self.current_level_id)
                    self.known_actions.discard(eval_res.falsified_action)

                # Clear old environment spec to force fresh generation from new probes
                env_mem = self.memory_manager.get_env_spec_memory("session")
                env_mem.specs.clear()

                # Reset board to pristine state and trigger micro-reprobe
                self.solver_reset_pending = True
                self.solver_reset_reason = "falsification_clean_reprobe_reset"
                self.probing_phase = True
                if hasattr(self.explorer, "probe_manager") and hasattr(self.explorer.probe_manager, "schedule_falsification_reprobe"):
                    reprobes = self.explorer.probe_manager.schedule_falsification_reprobe(
                        [eval_res.falsified_action] if eval_res.falsified_action else None
                    )
                    self.probe_queue.clear()
                    self.probe_queue.extend(reprobes)

            elif eval_res.replan_needed or (self.active_pool is not None and self.active_pool.active_candidate() is None):
                self.replan_requested = True
                self.active_pool = None
                self._last_attempt_failed = True
                self._last_failure_reason = "Candidate trajectory exhausted or diverged from expected intermediate states."
                executed_steps = [a for a in self.level_executed_actions if a and a.upper() not in ("RESET", "ACTION7")]
                self._last_failure_summary = (
                    f"{len(executed_steps)} actions executed before divergence: {', '.join(executed_steps[:12])}"
                    if executed_steps
                    else "Trajectory completed without reaching goal"
                )

            if eval_res.reset_needed:
                self.solver_reset_pending = True
                self.solver_reset_reason = (
                    "falsification_clean_reprobe_reset"
                    if eval_res.falsification_detected
                    else (
                        "replan_reset_clean_state"
                        if eval_res.replan_needed
                        else "next_candidate_reset_clean_state"
                    )
                )

            if self.active_pool:
                next_cand = self.active_pool.active_candidate()
                if next_cand is not None:
                    self._last_trajectory_id = next_cand.trajectory_id

        # Check if fallback or direct engine action produced an observable delta
        if self.last_probe_action is None and self.pending_step is None and self.pending_action is not None:
            direct_act_id = str(
                (self.pending_action.get("action_id") or self.pending_action.get("id") or "")
                if isinstance(self.pending_action, dict) else self.pending_action
            ).upper()
            before_grid = getattr(self.last_snapshot, "grid", None)
            after_grid = norm_after.get("grid")
            if before_grid is not None and after_grid is not None and before_grid != after_grid:
                if direct_act_id and direct_act_id not in ("RESET", "ACTION7"):
                    game_mem = self.memory_manager.get_game_memory("session")
                    self.known_actions.add(direct_act_id)
                    if game_mem and direct_act_id not in getattr(game_mem, "confirmed_action_effects", {}):
                        game_mem.record_action_effect(direct_act_id, "observable transition delta")
                    self._check_and_invalidate_dsl_for_new_action(direct_act_id, "observable direct action delta")

        # C. Visible Cycle Detection (Flash Loop Recovery port)
        if (
            getattr(self.config, "enable_cycle_detector", True)
            and hasattr(self, "cycle_detector")
            and self.cycle_detector is not None
            and self.last_snapshot is not None
            and self.pending_action is not None
        ):
            act_name = (
                (self.pending_action.get("id") or self.pending_action.get("action_id") or "")
                if isinstance(self.pending_action, dict)
                else str(self.pending_action)
            )
            before_grid = getattr(self.last_snapshot, "grid", None)
            after_grid = norm_after.get("grid")
            if before_grid is not None and after_grid is not None and act_name:
                cycle_info = self.cycle_detector.observe(before_grid, act_name, after_grid)
                per_level_limit = getattr(self.config, "cycle_detector_per_level_limit", 2)
                if cycle_info is not None and self.cycle_interventions_this_level < per_level_limit:
                    self.cycle_interventions_this_level += 1
                    logger.warning(
                        f"Session: Visible action cycle detected (period={cycle_info['period']}, "
                        f"repetitions={cycle_info['repetitions']}, pattern={cycle_info['action_pattern']}). "
                        f"Triggering loop recovery intervention ({self.cycle_interventions_this_level}/{per_level_limit})."
                    )
                    # Sever active candidate if one exists
                    diff_summary = None
                    eff_steps = None
                    if self.active_pool and self.active_pool.active_candidate():
                        cand = self.active_pool.active_candidate()
                        if cand.initial_grid is not None and after_grid is not None:
                            diff_summary = summarize_grid_diff(cand.initial_grid, after_grid)
                        eff_steps = list(cand.step_effects)
                        cand.sever()
                    self.replan_requested = True
                    self._last_attempt_failed = True
                    self._last_failure_reason = f"Visible cycle detected in state transitions with period {cycle_info['period']}."
                    ep_mem = self.memory_manager.get_epistemic_memory("session")
                    if hasattr(ep_mem, "record_attempt_feedback"):
                        ep_mem.record_attempt_feedback(
                            hypothesis="Visible action cycle loop",
                            trajectory_summary=f"Cycle period {cycle_info['period']} with pattern {cycle_info['action_pattern']}",
                            status="CYCLE_REJECTED",
                            reason=self._last_failure_reason,
                            diff_summary=diff_summary,
                            effective_steps=eff_steps,
                        )
                    self.solver_reset_pending = True
                    self.solver_reset_reason = "loop_recovery_cycle_reset"

        # Clear pending action references for next cycle
        self.pending_action = None
        self.pending_step = None
        return True

    def harness_telemetry(self) -> dict[str, Any]:
        """Return structured telemetry for competition harness."""
        ep_mem = self.memory_manager.get_epistemic_memory("session")
        return {
            "accepted_action_count": self.accepted_action_count,
            "levels_completed": self.levels_completed_observed,
            "game_over_reset_count": self.game_over_reset_count,
            "observed_transition_ingestions": self.observed_transition_ingestions,
            "observed_transition_duplicate_skips": self.observed_transition_duplicate_skips,
            "epistemic_judgments_count": len(ep_mem.judgments),
            "live_omit_branches_count": len(ep_mem.live_omit_branches),
            "severed_null_signatures_count": len(ep_mem.severed_null_signatures),
            "current_level_id": self.current_level_id,
            "current_game_id": self.current_game_id,
            "session_aborted": self.session_aborted,
            "active_pipeline": self.active_pipeline,
            "probing_phase": self.probing_phase,
            "probe_queue_length": len(self.probe_queue),
            "undecided_count": self.undecided_count,
            "undecided_resolved_by_probe": self.undecided_resolved_by_probe,
            "undecided_fallback_to_null": self.undecided_fallback_to_null,
            "evidence_probes_executed": self.evidence_probes_executed,
            "epistemic_signals_count": len(getattr(ep_mem, "epistemic_signals", [])),
            "cycle_interventions_count": getattr(self, "cycle_interventions_this_level", 0),
        }
