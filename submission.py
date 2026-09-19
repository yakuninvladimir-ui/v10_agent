"""Submission configuration defaults and environment variable resolution for Kaggle."""

from __future__ import annotations

import os
from typing import Any

from v10_agent.config import config_from_env


def default_config() -> dict[str, Any]:
    """Build default configuration dictionary reading environment variables."""
    cfg = config_from_env()
    cfg_dict = cfg.to_dict()

    # Add Kaggle harness compatibility keys
    cfg_dict.update({
        "model_path": cfg.model_path,
        "reasoning_strength": cfg.reasoning_strength,
        "vllm_base_url": cfg.vllm_base_url,
        "vllm_api_key": cfg.vllm_api_key,
        "qwen_backend": cfg.llm_advisor_backend,
        "qwen_model_path": cfg.qwen_model_path,
        "qwen_vllm_base_url": cfg.qwen_vllm_base_url,
        "qwen_vllm_api_key": cfg.qwen_vllm_api_key,
        "qwen_vllm_model": cfg.qwen_model_path,
        "max_actions_per_game": cfg.max_actions_per_game,
        "max_actions_per_level": cfg.max_actions_per_level,
        "game_wall_clock_limit_seconds": cfg.game_wall_clock_limit_seconds,
        "competition_wall_clock_limit_seconds": cfg.competition_wall_clock_limit_seconds,
        "concurrency": cfg.concurrency,
        "game_concurrency": cfg.concurrency,
        "vllm_max_num_seqs": cfg.vllm_max_num_seqs,
        "max_chain_attempts_per_level": cfg.max_chain_attempts_per_level,
        "max_coder_retries": cfg.max_coder_retries_per_level,
        "max_solver_retries": cfg.max_solver_retries_per_level,
        "max_explorer_probes": cfg.max_explorer_probe_actions_per_level,
        "max_game_over_resets_per_game": cfg.max_game_over_resets_per_game,
        "reset_on_game_over": cfg.reset_on_game_over,
        "enable_symbolic_fallback": cfg.enable_symbolic_fallback,
        "coder_exhaustion_forces_fallback": cfg.coder_exhaustion_forces_fallback,
        "solver_exhaustion_forces_fallback": cfg.solver_exhaustion_forces_fallback,
        "abort_on_dsl_exhaustion": cfg.abort_on_dsl_exhaustion,
        "vllm_mtp_enabled": cfg.vllm_mtp_enabled,
        "vllm_mtp_tokens": cfg.vllm_mtp_tokens,
        "vllm_speculative_method": cfg.vllm_speculative_method,
        "vllm_speculative_model": cfg.vllm_speculative_model,
        "vllm_speculative_config": cfg.vllm_speculative_config,
    })
    return cfg_dict


def _state_name(state: Any) -> str:
    """Extract canonical uppercase string from state enum or value."""
    if hasattr(state, "name"):
        return str(getattr(state, "name")).split(".")[-1].upper()
    value = getattr(state, "value", None)
    if value is not None:
        return str(value).split(".")[-1].upper()
    return str(state or "").split(".")[-1].strip().upper()


def _action_name(action: Any) -> str:
    """Extract canonical uppercase string from action enum or value."""
    if hasattr(action, "name"):
        return str(getattr(action, "name")).split(".")[-1].upper()
    value = getattr(action, "value", action)
    if isinstance(value, int):
        if value == 0:
            return "RESET"
        if 1 <= value <= 7:
            return f"ACTION{value}"
    text = str(value).split(".")[-1].strip().upper()
    if text.isdigit():
        return _action_name(int(text))
    return text or "ACTION1"
