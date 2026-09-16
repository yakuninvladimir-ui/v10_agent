"""GameSession: Master orchestrator of the Tri-Agent pipeline & Double-Loop feedback routing."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from v10_agent.action_adapter import to_native_action
from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot
from v10_agent.brusentsov_logic import Ternary
from v10_agent.config import V10Config, config_from_mapping
from v10_agent.dsl_coder import DSLCoder
from v10_agent.explorer_agent import ExplorerAgent, compute_probe_effect
from v10_agent.fallback_symbolic import SymbolicFallbackEngine
from v10_agent.frame_media import render_annotated_frame_png, render_dual_frame_png, render_grid_png
from v10_agent.judge import LayeredVerifier
from v10_agent.llm_advisor import BaseLLMAdvisor, build_llm_advisor
from v10_agent.logging import StructuredAuditLogger
from v10_agent.memory_contours import BranchSignature, MemoryContourManager, SyntaxErrorRecord
from v10_agent.observe import grid_to_hex_rows, normalize_observation
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.sandbox import SandboxedModule, SandboxExecutor
from v10_agent.solver_agent import SolverAgent
from v10_agent.symbolic_executor import SymbolicTrajectoryExecutor
from v10_agent.trajectory import TrajectoryPool
from v10_agent.types import EffectDeclaration, Grid2D
from v10_agent.verification import GroundedStep, GroundingError, VerificationBinder

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
        SessionPhase.PROBING: [SessionPhase.CODING, SessionPhase.FALLBACK, SessionPhase.PROBING],
        SessionPhase.CODING: [SessionPhase.SOLVING, SessionPhase.PROBING, SessionPhase.FALLBACK],
        SessionPhase.SOLVING: [SessionPhase.EXECUTING, SessionPhase.PROBING, SessionPhase.FALLBACK],
        SessionPhase.EXECUTING: [SessionPhase.REFLECTING, SessionPhase.SOLVING, SessionPhase.PROBING, SessionPhase.FALLBACK],
        SessionPhase.REFLECTING: [SessionPhase.PROBING, SessionPhase.SOLVING, SessionPhase.EXECUTING],
        SessionPhase.FALLBACK: [SessionPhase.EXECUTING, SessionPhase.PROBING, SessionPhase.SOLVING],
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
                f"{len(executed_steps)} actions executed: {', '.join(executed_steps[:12])}"
                if executed_steps
                else "Direct candidate completion"
            )

            # Turn 2: Ask Solver to reflect on the win and distill domain-general invariants
            self.transition_to(SessionPhase.REFLECTING, "distilling level win invariants")
            distilled_invariants = self.solver.distill_level_win_invariants(
                winning_candidate=winning_cand,
                execution_summary=exec_summary,
            )

            for inv in distilled_invariants:
                game_mem.record_stratified_invariant(inv, tier=3)

            primary_inv = " | ".join(distilled_invariants) if distilled_invariants else "Satisfied level goal via coordinated alignment"
            game_mem.record_level_solution(
                level_id=self.current_level_id,
                setup_summary=setup_str,
                invariant_rule=primary_inv,
                winning_macro="",  # Omitted to prevent button-sequence pollution in cross-level memory
            )

            # Track discovered invariants across level transitions
            if self.last_planning_set:
                try:
                    from v10_agent.universal_invariants import discover_invariants
                    new_invs = discover_invariants(self.last_planning_set, confirmed_actors=game_mem.confirmed_actors)
                    inv_diff = game_mem.compare_and_record_invariants(new_invs)
                    re_eval_stats = game_mem.re_evaluate_invariants(self.last_planning_set, level_id=new_level_id)
                    logger.info(
                        f"Cross-level invariant re-evaluation: {re_eval_stats['confirmed']} confirmed, "
                        f"{re_eval_stats['falsified']} falsified, {re_eval_stats['unchanged']} unchanged."
                    )
                except Exception as exc:
                    logger.debug(f"Cross-level invariant comparison skipped: {exc}")

        self.level_executed_actions.clear()

        self.current_level_id = new_level_id
        self.active_module = None
        self.active_manifest = None
        self.active_pool = None
        self.pending_step = None
        self.pending_action = None
        self.last_probe_action = None
        self.probe_queue = []
        self.transition_to(SessionPhase.PROBING, "new level transition")
        self.probe_actions_executed_this_level = 0
        self.level_chain_attempts = 0
        self.explorer_attempts_this_level = 0
        self.explorer_reprobe_pending = False
        # Seed known_actions from GameMemory confirmed kinematics to avoid blind re-probing
        game_mem = self.memory_manager.get_game_memory("session")
        self.known_actions = set(game_mem.confirmed_action_effects.keys())
        self.active_pipeline = "discrete"
        self.session_aborted = False
        self.replan_requested = False
        self.coder_failed_for_level = False
        self.solver_reset_pending = False
        self.solver_reset_reason = None
        self._level_win_handled = False
        self._last_trajectory_id = None
        self.level_initial_grid = None
        self.level_initial_grid_hash = None
        self.memory_manager.handle_level_transition(new_level_id)
        if hasattr(self.explorer, "probe_manager") and hasattr(self.explorer.probe_manager, "handle_level_transition"):
            self.explorer.probe_manager.handle_level_transition(game_mem.confirmed_action_effects)
        logger.info(f"Transitioned to new level {new_level_id}; GameMemory preserved ({len(self.known_actions)} confirmed actions carried forward).")

    def handle_game_transition(self, new_game_id: str) -> None:
        """Reset all contours when switching games."""
        self.current_game_id = new_game_id
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
        self.known_actions = set()
        from v10_agent.explorer_agent import PrimitiveProbeManager
        self.explorer.probe_manager = PrimitiveProbeManager(max_probes=self.config.max_primitive_probes_per_level)
        self.active_pipeline = "discrete"
        self.session_aborted = False
        self.replan_requested = False
        self.coder_failed_for_level = False
        self._level_win_handled = False
        self._last_trajectory_id = None
        self.memory_manager.handle_game_transition(new_game_id, self.current_level_id)
        logger.info(f"Reset session for new game {new_game_id}.")

    def act(self, raw_observation: Mapping[str, Any]) -> dict[str, Any]:
        """Propose the single next environment action."""
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
            if self.last_engine_action == "RESET":
                logger.error("GAME_OVER persisted after single RESET. Forcing loop break.")
                raise RuntimeError("GAME_OVER persisted after single RESET")

            max_attempts = getattr(self.config, "max_chain_attempts_per_level", 5)
            if not self.config.reset_on_game_over or self.level_chain_attempts >= max_attempts:
                logger.warning(
                    f"GAME_OVER encountered and level attempts exhausted ({self.level_chain_attempts}/{max_attempts}) "
                    f"or resets disabled; abandoning game without reset."
                )
                self.session_aborted = True
                raise LevelAttemptsExhaustedError(
                    f"GAME_OVER encountered and level attempts exhausted ({max_attempts} attempts); "
                    f"transitioning to next game without reset"
                )

            self.game_over_reset_count += 1
            self.replan_requested = True
            self.active_pool = None
            self.level_executed_actions.clear()
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
            self.pending_step = None
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
        )

        # Record pristine initial frame of level on very first step before any actions
        if self.level_initial_grid is None:
            self.level_initial_grid = [list(row) for row in grid]
            self.level_initial_grid_hash = planning_set.grid_hash
            logger.info(f"Recorded pristine level initial frame for {self.current_level_id} (hash={planning_set.grid_hash[:8]}).")
        elif self.last_engine_action == "RESET" and self.level_initial_grid_hash is not None:
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

        # 4. Pipeline Determination & Probing Phase
        if self.probing_phase:
            non_meta_actions = [a for a in available_actions if str(a).upper() not in ("RESET", "ACTION7")]
            only_coords = all(str(a).upper() == "ACTION6" for a in non_meta_actions) and len(non_meta_actions) > 0
            coord_quota = 5 if only_coords else 3

            has_coords = "ACTION6" in planning_set.allowed_action_ids or any(
                isinstance(a, str) and any(kw in a.lower() for kw in ("x", "y", "coord", "click"))
                for a in available_actions
            )

            # Check if an explorer reprobe was scheduled after a clean state reset
            if self.explorer_reprobe_pending and not self.probe_queue:
                self.explorer_reprobe_pending = False
                if has_coords and "ACTION6" not in self.explorer.probe_manager.confirmed_effective_actions:
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
                        planning_set, memory=env_mem, max_coords=coord_quota, image_png=explorer_images
                    )
                    self.explorer_attempts_this_level += 1
                    logger.info(
                        f"Explorer reprobe initiated: attempt {self.explorer_attempts_this_level}/"
                        f"{getattr(self.config, 'max_explorer_attempts_per_level', 5)} (quota={coord_quota})"
                    )
                    self.probe_queue.extend(coord_probes)

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
                                planning_set, memory=env_mem, max_coords=coord_quota, image_png=explorer_images
                            )
                            self.explorer_attempts_this_level += 1
                            logger.info(
                                f"Explorer initial coordinate probes: attempt {self.explorer_attempts_this_level}/"
                                f"{getattr(self.config, 'max_explorer_attempts_per_level', 5)} (quota={coord_quota})"
                            )
                            self.probe_queue.extend(coord_probes)
                    elif has_coords and "ACTION6" not in self.known_actions:
                        coord_probes = self.explorer.probe_manager.plan_targeted_coordinate_probes(
                            planning_set, available_actions
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

                if has_effective or not self.config.enable_primitive_probing or not has_coords:
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

                else:
                    # No confirmed actions discovered yet: check recursive attempt budget
                    max_exp_attempts = getattr(self.config, "max_explorer_attempts_per_level", 5)
                    if self.explorer_attempts_this_level >= max_exp_attempts:
                        self.session_aborted = True
                        logger.error(
                            f"Explorer probe budget exhausted ({max_exp_attempts} attempts) on {self.current_level_id} "
                            f"without discovering effective actions. Transitioning to next game without reset."
                        )
                        raise LevelAttemptsExhaustedError(
                            f"Explorer probe budget exhausted ({max_exp_attempts} attempts) on {self.current_level_id} "
                            f"without discovering effective actions. Transitioning to next game without reset."
                        )
                    else:
                        # Queue clean board reset before launching next explorer attempt
                        self.explorer_reprobe_pending = True
                        self.probe_actions_executed_this_level = 0
                        reset_action = {
                            "id": "RESET",
                            "action_id": "RESET",
                            "data": {},
                            "reasoning": {
                                "source": "explorer_retry_clean_state",
                                "attempt": self.explorer_attempts_this_level,
                            },
                        }
                        self.last_probe_action = None
                        self.last_snapshot = snapshot
                        self.last_planning_set = planning_set
                        self.pending_action = reset_action
                        self.pending_step = None
                        self.last_engine_action = "RESET"
                        self.level_executed_actions.clear()
                        self.audit_logger.log(
                            "action_emitted",
                            action="RESET",
                            data={},
                            strategy="explorer_retry_clean_state",
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

        # 5. Explorer Phase (Spec Generation)
        game_mem = self.memory_manager.get_game_memory("session")
        if not env_mem.specs and self.config.max_explorer_probe_actions_per_level > 0:
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
            self.explorer.generate_environment_spec(planning_set, env_mem, image_png=explorer_images, game_memory=game_mem)

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
                self.active_module = module
                self.active_manifest = manifest
                for fn in manifest.get("functions", []):
                    fn_name = str(fn.get("name", "")).upper()
                    # Only certify primitives that correspond to confirmed effective actions
                    act_match = next((act for act in game_mem.confirmed_action_effects if act in fn_name), None)
                    if act_match or "ACTION" not in fn_name:
                        game_mem.record_reusable_primitive(fn)
                logger.info(f"DSL Certified with {len(manifest.get('functions', []))} primitives.")
            else:
                self.coder_failed_for_level = True
                retries = self.config.max_coder_retries_per_level
                logger.warning(f"DSLCoder retries exhausted ({retries} attempts).")
                if not (self.config.coder_exhaustion_forces_fallback and not self.config.abort_on_dsl_exhaustion):
                    self.session_aborted = True
                    logger.error(f"DSLCoder retries exhausted ({retries} attempts). Transitioning to next game without reset.")
                    raise LevelAttemptsExhaustedError(f"Coder retries exhausted ({retries} attempts). Transitioning to next game without reset.")

        # 7. Solver Phase (Trajectory Generation & Replanning)
        if self.replan_requested:
            self.active_pool = None
            self.replan_requested = False

        if self.active_module is not None and (self.active_pool is None or self.active_pool.active_candidate() is None):
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
                logger.warning(f"Level chain attempt budget exhausted ({max_attempts}) for {self.current_level_id}. Transitioning to next game without reset.")
                self.session_aborted = True
                raise LevelAttemptsExhaustedError(f"Level chain attempt budget exhausted ({max_attempts} attempts). Transitioning to next game without reset.")
            else:
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
                    action_budget=max(1, self.config.max_actions_per_level - self.accepted_action_count),
                )
                pkg = self.solver.generate_trajectory_package(
                    manifest=self.active_manifest or {},
                    planning_set=planning_set,
                    epistemic_memory=ep_mem,
                    budget=max(1, self.config.max_actions_per_level - self.accepted_action_count),
                    image_png=solver_images,
                    game_memory=game_mem,
                )
                if pkg is not None:
                    self.active_pool = TrajectoryPool.from_package(pkg)
                    c_summaries = [f"{c.trajectory_id}({len(c.steps)} steps: {[s.get('dsl_function') for s in c.steps[:6]]}...)" for c in self.active_pool.candidates]
                    logger.info(f"Solver generated {len(self.active_pool.candidates)} candidates: {'; '.join(c_summaries)}")
                else:
                    logger.warning(f"Solver trajectory proposal failed on attempt {self.level_chain_attempts}/{max_attempts}.")
                    if self.level_chain_attempts >= max_attempts:
                        self.session_aborted = True
                        raise LevelAttemptsExhaustedError(f"Solver trajectory proposal retries exhausted ({max_attempts} attempts). Transitioning to next game without reset.")

        # 8. Symbolic Step Verification & Execution (Independent from Qwen)
        self.transition_to(SessionPhase.EXECUTING, "executing symbolic step")
        exec_res = self.symbolic_executor.prepare_and_execute_step(
            pool=self.active_pool,
            planning_set=planning_set,
            active_module=self.active_module,
            epistemic_memory=ep_mem,
            syntax_memory=syntax_mem,
            game_memory=self.memory_manager.get_game_memory("session"),
        )

        effect: EffectDeclaration | None = exec_res.effect
        grounded_step: GroundedStep | None = exec_res.grounded_step
        strategy: str = exec_res.strategy

        if exec_res.circuit_broken:
            next_cand = self.active_pool.active_candidate() if self.active_pool else None
            if next_cand is None:
                self.replan_requested = True
                self.active_pool = None
            else:
                logger.info(f"Session: Circuit broken on candidate; advancing to next pool candidate: {next_cand.trajectory_id}")

            reset_action = {
                "id": "RESET",
                "action_id": "RESET",
                "data": {},
                "reasoning": {"source": "circuit_breaker_immediate_reset", "error": exec_res.error_message},
            }
            self.pending_action = reset_action
            self.last_engine_action = "RESET"
            self.pending_step = None
            self.level_executed_actions.clear()
            return reset_action

        # 9. Fallback Path
        if effect is None:
            if self.session_aborted or (self.config.abort_on_dsl_exhaustion and self.active_module is None):
                self.session_aborted = True
                raise LevelAttemptsExhaustedError("DSL missing or session aborted due to retry exhaustion. Transitioning to next game without reset.")

            self.transition_to(SessionPhase.FALLBACK, "selecting fallback action")
            effect = self.fallback_engine.select_fallback_action(
                planning_set, game_memory=self.memory_manager.game_memory
            )
            grounded_step = None

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
                a_data["x"] = int(a_data["x"]) + self.crop_offset
                a_data["y"] = int(a_data["y"]) + self.crop_offset
                action_dict["data"] = a_data

        self.last_snapshot = snapshot
        self.last_planning_set = planning_set
        self.pending_step = grounded_step
        self.pending_action = action_dict
        self.last_engine_action = action_decl.action_id
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
                )
                game_mem = self.memory_manager.get_game_memory("session")
                if "moved" in rec.observed_effect and any(d in rec.observed_effect for d in ("UP", "DOWN", "LEFT", "RIGHT")):
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                elif "color transition" in rec.observed_effect:
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                elif "selection indicator" in rec.observed_effect:
                    act_desc = f"{action_id}({action_data})" if action_data else action_id
                    offset_note = ""
                    try:
                        after_grid = norm_after.get("grid")
                        if after_grid:
                            after_snap = extract_arga_snapshot(after_grid)
                            after_pset = build_planning_set(after_snap, available_actions=list(self.known_actions))
                            from v10_agent.universal_invariants import discover_invariants
                            invs = discover_invariants(after_pset)
                            # Dynamically determine step size from confirmed action effects
                            step_sz = 1.0
                            if game_mem and getattr(game_mem, "confirmed_action_effects", None):
                                import re
                                for eff in game_mem.confirmed_action_effects.values():
                                    nums = [abs(int(x)) for x in re.findall(r'd[yx]=([+-]?\d+)', eff)]
                                    if nums and max(nums) > 0:
                                        step_sz = float(max(nums))
                                        break

                            for inv in invs:
                                if inv.invariant_type == "axial_symmetry_vertical" and inv.axis_id:
                                    sub = after_pset.get_object(inv.subject_id)
                                    tgt = after_pset.get_object(inv.target_id)
                                    ax = after_pset.get_object(inv.axis_id)
                                    if sub and tgt and ax:
                                        req_c = (tgt.centroid.col + sub.centroid.col) / 2.0
                                        d_ax = req_c - ax.centroid.col
                                        d_pc = tgt.centroid.row - sub.centroid.row
                                        ax_s = int(round(d_ax / step_sz))
                                        pc_s = int(round(d_pc / step_sz))
                                        if ax_s != 0 or pc_s != 0:
                                            offset_note = f" (axis_steps={ax_s}, piece_steps={pc_s})"
                                            break
                                elif inv.invariant_type == "axial_symmetry_horizontal" and inv.axis_id:
                                    sub = after_pset.get_object(inv.subject_id)
                                    tgt = after_pset.get_object(inv.target_id)
                                    ax = after_pset.get_object(inv.axis_id)
                                    if sub and tgt and ax:
                                        req_r = (tgt.centroid.row + sub.centroid.row) / 2.0
                                        d_ax = req_r - ax.centroid.row
                                        d_pc = tgt.centroid.col - sub.centroid.col
                                        ax_s = int(round(d_ax / step_sz))
                                        pc_s = int(round(d_pc / step_sz))
                                        if ax_s != 0 or pc_s != 0:
                                            offset_note = f" (axis_steps={ax_s}, piece_steps={pc_s})"
                                            break
                    except Exception as exc:
                        logger.warning(f"Failed to extract post-toggle invariants: {exc}")
                    game_mem.record_selection_mechanic(f"{act_desc}: {rec.observed_effect}{offset_note}")
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                    if self.active_module is not None:
                        logger.info("Session: Modal toggle discovered; invalidating active DSL module.")
                        self.active_module = None
                        self.active_manifest = None
                elif "object count changed" in rec.observed_effect:
                    game_mem.record_action_effect(action_id, rec.observed_effect)
                    self.known_actions.add(action_id)
                else:
                    game_mem.record_unconfirmed_action(action_id, rec.observed_effect)

                # Trigger dynamic reprobes if this action altered state or toggled selection
                reprobes = self.explorer.probe_manager.get_dynamic_reprobes(action_id, rec.observed_effect)
                if reprobes:
                    self.probe_queue.extend(reprobes)

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

            if eval_res.falsification_detected:
                falsified = eval_res.falsified_action or "initial_motion"
                logger.warning(
                    f"Session: Действие {falsified} не валидно при текущих координатах объекта. "
                    f"Invalidating DSL and scheduling clean micro-reprobe cycle."
                )
                self.active_module = None
                self.active_manifest = None
                self.active_pool = None
                self.replan_requested = True

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
        }
