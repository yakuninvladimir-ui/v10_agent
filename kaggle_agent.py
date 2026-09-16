"""Kaggle Arcade Competition Shim for ARC-AGI-3 LCLD Agent V10.0."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from submission import _action_name, _state_name, default_config
from v10_agent.action_adapter import arcade_step_args, to_native_action
from v10_agent.config import config_from_mapping
from v10_agent.session import GameSession, LevelAttemptsExhaustedError

logger = logging.getLogger(__name__)

try:
    from arcengine import ActionInput, GameAction
except Exception:
    ActionInput = None
    GameAction = None


class ARC_AGI_Agent:
    """Competition agent adapter exposing the standard Arcade competition interface."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = default_config()
        if config:
            cfg.update(dict(config))
        self.config = cfg
        self._session = GameSession(config_from_mapping(cfg))
        self._observed_transition_ingestions = 0
        self._observed_transition_duplicate_skips = 0
        self._termination_records: list[dict[str, Any]] = []

    def _update_config(self, config: Mapping[str, Any] | None = None) -> None:
        if not config:
            return
        merged = dict(self.config)
        merged.update(dict(config))
        self.config = merged
        self._session.update_runtime_config(merged)

    def _ingest_pending_transition(self, observation: Mapping[str, Any]) -> bool:
        if getattr(self._session, "pending_action", None) is None:
            return False
        committed = self._session.observe_action_result(dict(observation))
        if committed:
            self._observed_transition_ingestions += 1
        else:
            self._observed_transition_duplicate_skips += 1
        return committed

    def act(self, observation: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
        """Propose next action and convert to native ActionInput/dict."""
        try:
            self._update_config(config)
            obs_dict = dict(observation)
            self._ingest_pending_transition(obs_dict)
            action = self._session.act(obs_dict)
            return to_native_action(action)
        except LevelAttemptsExhaustedError:
            # Propagate level/game termination directly to competition child runner without reset
            raise
        except Exception as exc:
            logger.error(f"ARC_AGI_Agent.act caught unexpected error, engaging fallback: {exc}", exc_info=True)
            available = list(dict(observation).get("available_actions", []) or [])
            fallback_act = available[0] if available else "ACTION1"
            return to_native_action({
                "id": fallback_act,
                "action_id": fallback_act,
                "data": {},
                "reasoning": {"source": "emergency_top_level_fallback", "error": str(exc)},
            })

    def reset_after_game_over(
        self,
        observation: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return the single RESET required by the Tufa-style GAME_OVER loop.

        Zero LLM calls are consumed, failed-attempt epistemic memory is retained.
        """
        self._update_config(config)
        obs_dict = dict(observation)
        self._ingest_pending_transition(obs_dict)
        state = _state_name(dict(obs_dict.get("metadata", {}) or {}).get("state", obs_dict.get("state", "")))
        if state != "GAME_OVER":
            raise RuntimeError(f"reset_after_game_over requires GAME_OVER state, got {state!r}")

        max_attempts = int(self.config.get("max_chain_attempts_per_level", 5))
        if not self.config.get("reset_on_game_over", True) or (
            self._session and self._session.level_chain_attempts >= max_attempts
        ):
            raise RuntimeError(f"GAME_OVER reset budget exhausted ({max_attempts} attempts): abandoning game without reset")

        try:
            action = self._session.act(obs_dict)
        except LevelAttemptsExhaustedError as exc:
            raise RuntimeError(str(exc)) from exc
        act_id = action.get("id", action.get("action_id", ""))
        if _action_name(act_id) != "RESET":
            raise RuntimeError(f"GAME_OVER reset path emitted non-RESET action: {act_id!r}")
        return to_native_action(action)

    def observe_action_result(self, after_observation: Mapping[str, Any] | None = None) -> bool:
        """Commit the accepted gateway transition to LayeredVerifier and EpistemicMemory."""
        committed = self._session.observe_action_result(after_observation)
        if committed:
            self._observed_transition_ingestions += 1
        else:
            self._observed_transition_duplicate_skips += 1
        return committed

    def record_orchestration_termination(self, reason: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Record game termination telemetry."""
        self._termination_records.append({"reason": str(reason), "metadata": dict(metadata or {})})

    def harness_telemetry(self) -> dict[str, Any]:
        """Export comprehensive telemetry dictionary."""
        telemetry = dict(self._session.harness_telemetry())
        telemetry.update({
            "observed_transition_ingestions": self._observed_transition_ingestions,
            "observed_transition_duplicate_skips": self._observed_transition_duplicate_skips,
            "termination_records": list(self._termination_records),
        })
        return telemetry

    def _cleanup_old_session(self) -> None:
        """Release session and memory references when worker finishes."""
        session = getattr(self, "_session", None)
        self._session = None
        if session is None:
            return
        for attr in ("pending_action", "last_snapshot", "last_planning_set", "memory_manager", "active_module"):
            try:
                setattr(session, attr, None)
            except Exception:
                pass
