"""Configuration engine for ARC-AGI-3 LCLD Agent V10.0.

Provides V10Config with environment variable resolution and competition ceilings.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class PerceptionConfig:
    min_substantive_area: int = 4
    axis_merge_min_area: int = 6
    axis_merge_max_gap: int = 15
    object_match_area_tolerance: float = 0.15
    movement_detection_threshold: float = 1.0
    max_probe_distance: float = 24.0

@dataclass
class V10Config:
    """Normative configuration for V10 Tri-Agent Neuro-Symbolic Agent."""

    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    # LLM Advisor & vLLM Backend
    # LLM Advisor & vLLM Backend (Qwen3.8-27B normative settings)
    llm_advisor_backend: str = "vllm"  # "vllm" | "fake" | "ollama" | "llama_cli"
    model_path: str = "Qwen/Qwen3.8-27B"
    qwen_model_path: str = "Qwen/Qwen3.8-27B"
    vllm_base_url: str = "http://127.0.0.1:1234/v1"
    qwen_vllm_base_url: str = "http://127.0.0.1:1234/v1"
    vllm_api_key: str = "EMPTY"
    qwen_vllm_api_key: str = "EMPTY"
    context_tokens: int = 131072
    qwen_context_tokens: int = 131072
    max_input_tokens: int = 65536
    qwen_max_input_tokens: int = 65536
    max_output_tokens: int = 32000
    qwen_max_output_tokens: int = 32000
    temperature: float = 1.0
    qwen_temperature: float = 1.0
    top_p: float = 0.95
    qwen_top_p: float = 0.95
    top_k: int = 20
    qwen_top_k: int = 20
    min_p: float = 0.0
    qwen_min_p: float = 0.0
    presence_penalty: float = 0.0
    qwen_presence_penalty: float = 0.0
    repeat_penalty: float = 1.0
    qwen_repeat_penalty: float = 1.0
    seed: int = 42
    qwen_seed: int = 42
    timeout_seconds: int = 700
    qwen_timeout_seconds: int = 700
    multimodal_enabled: bool = True
    qwen_multimodal_enabled: bool = True
    solver_multimodal_enabled: bool = True
    explorer_multimodal_enabled: bool = True
    coder_multimodal_enabled: bool = True
    enable_thinking: bool = True
    qwen_enable_thinking: bool = True
    reasoning_strength: str = "xhigh"  # "low" | "medium" | "high" | "xhigh"
    reasoning_budget_tokens: int = 32000
    qwen_reasoning_budget_tokens: int = 32000

    # Tri-Agent Retries & Budgets (Normative Ceilings)
    max_chain_attempts_per_level: int = 5  # Unified 5-attempt budget per level for entire chain
    max_coder_retries_per_level: int = 5
    max_solver_retries_per_level: int = 5
    max_explorer_attempts_per_level: int = 5
    max_explorer_probe_actions_per_level: int = 30
    max_total_llm_calls_per_level: int = 20
    level_wall_clock_limit_seconds: float = 1200.0  # 20-minute soft level budget

    # Trajectory & Solver Package Limits
    max_candidates_per_solver_package: int = 4
    max_steps_per_candidate: int = 20
    execute_one_step_at_a_time: bool = True

    # Deterministic Sandbox
    sandbox_enabled: bool = True
    sandbox_allowed_modules: list[str] = field(
        default_factory=lambda: ["math", "typing", "dataclasses", "enum", "collections"]
    )
    sandbox_max_cpu_seconds: float = 10.0
    sandbox_max_memory_mb: int = 512

    # Memory Contours
    game_memory_reset_on_game_change: bool = True
    game_memory_reset_on_level_change: bool = False
    epistemic_memory_max_entries: int = 50
    syntax_error_memory_max_entries: int = 5
    crop_border_pixels: int = 1

    # Fallback System
    enable_symbolic_fallback: bool = True
    coder_exhaustion_forces_fallback: bool = True
    solver_exhaustion_forces_fallback: bool = True
    abort_on_dsl_exhaustion: bool = False
    max_primitive_probes_per_level: int = 30
    enable_primitive_probing: bool = False
    probe_reset_after_discrete: bool = False

    # Competition Execution Limits
    max_actions_per_game: int = 250
    max_actions_per_level: int = 250
    max_game_over_resets_per_game: int = 5
    max_game_over_resets_per_level: int = 5
    reset_on_game_over: bool = True
    game_wall_clock_limit_seconds: float = 5000.0
    competition_wall_clock_limit_seconds: float = 30600.0
    concurrency: int = 4
    vllm_max_num_seqs: int = 4
    vllm_startup_timeout_seconds: int = 900

    # vLLM Speculative Decoding & Multi-Token Prediction (MTP=3 normative settings)
    vllm_mtp_enabled: bool = True
    vllm_mtp_tokens: int = 3
    vllm_speculative_method: str = "mtp"
    vllm_speculative_model: str | None = None
    vllm_speculative_config: str | None = None
    vllm_speculative_cli_format: str = "auto"  # "auto" | "config_json" | "spec_tokens" | "speculative_model"

    # V10.1 Persistent Object Tracker & Four-valued Verdict Knobs
    enable_persistent_tracker: bool = True
    track_match_threshold: float = 0.45
    track_max_age: int = 5
    matching_ambiguity_threshold: float = 0.15
    min_reliable_delta: float = 0.8
    track_confidence_threshold: float = 0.6
    occlusion_radius: int = 3
    cumulative_window: int = 3
    enable_undecided_verdict: bool = True
    max_undecided_streak: int = 2
    max_evidence_probes_per_level: int = 2

    @property
    def model_name(self) -> str:
        return self.model_path

    @property
    def qwen_model_name(self) -> str:
        return self.qwen_model_path

    def build_speculative_args(self, model_path: str | None = None) -> list[str]:
        return build_vllm_speculative_args(self, model_path=model_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def update_runtime(self, updates: Mapping[str, Any]) -> None:
        """Update mutable configuration settings at runtime."""
        for key, value in updates.items():
            if hasattr(self, key):
                field_type = type(getattr(self, key))
                if value is not None and field_type is not type(None):
                    try:
                        if field_type is bool and isinstance(value, str):
                            setattr(self, key, value.strip().lower() in {"1", "true", "yes", "on"})
                        else:
                            setattr(self, key, field_type(value))
                    except (ValueError, TypeError):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)


def _bool_from_env(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _float_from_env(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def config_from_env(overrides: Mapping[str, Any] | None = None) -> V10Config:
    """Build V10Config reading from environment variables with safe defaults."""
    resolved_model = (
        os.environ.get("ARC_QWEN_MODEL_PATH")
        or os.environ.get("ARC_MODEL_PATH")
        or os.environ.get("ARC_LLM_MODEL_PATH")
        or os.environ.get("ARC_MUSE_MODEL_PATH")
        or "Qwen/Qwen3.8-27B"
    )
    resolved_base_url = (
        os.environ.get("ARC_VLLM_BASE_URL")
        or os.environ.get("ARC_QWEN_VLLM_BASE_URL")
        or "http://127.0.0.1:1234/v1"
    )
    resolved_api_key = (
        os.environ.get("ARC_VLLM_API_KEY")
        or os.environ.get("ARC_QWEN_VLLM_API_KEY")
        or "EMPTY"
    )
    resolved_temp = _float_from_env("ARC_TEMPERATURE", _float_from_env("ARC_QWEN_TEMPERATURE", 1.0))
    resolved_top_p = _float_from_env("ARC_TOP_P", _float_from_env("ARC_QWEN_TOP_P", 0.95))
    resolved_top_k = _int_from_env("ARC_TOP_K", _int_from_env("ARC_QWEN_TOP_K", 20))
    resolved_min_p = _float_from_env("ARC_MIN_P", _float_from_env("ARC_QWEN_MIN_P", 0.0))
    resolved_presence = _float_from_env("ARC_PRESENCE_PENALTY", _float_from_env("ARC_QWEN_PRESENCE_PENALTY", 0.0))
    resolved_repeat = _float_from_env("ARC_REPEAT_PENALTY", _float_from_env("ARC_QWEN_REPEAT_PENALTY", 1.0))
    resolved_seed = _int_from_env("ARC_SEED", _int_from_env("ARC_QWEN_SEED", 42))
    resolved_timeout = _int_from_env("ARC_TIMEOUT_SECONDS", _int_from_env("ARC_QWEN_TIMEOUT_SECONDS", 700))
    resolved_mm = _bool_from_env("ARC_MULTIMODAL_ENABLED", _bool_from_env("ARC_QWEN_MULTIMODAL_ENABLED", True))
    resolved_solver_mm = _bool_from_env("ARC_SOLVER_MULTIMODAL_ENABLED", resolved_mm)
    resolved_explorer_mm = _bool_from_env("ARC_EXPLORER_MULTIMODAL_ENABLED", resolved_mm)
    resolved_coder_mm = _bool_from_env("ARC_CODER_MULTIMODAL_ENABLED", resolved_mm)
    resolved_thinking = _bool_from_env("ARC_ENABLE_THINKING", _bool_from_env("ARC_QWEN_ENABLE_THINKING", True))
    resolved_strength = os.environ.get("ARC_REASONING_STRENGTH", "xhigh")
    resolved_budget = _int_from_env("ARC_REASONING_BUDGET_TOKENS", _int_from_env("ARC_QWEN_REASONING_BUDGET_TOKENS", 32000))
    resolved_ctx = _int_from_env("ARC_CONTEXT_TOKENS", _int_from_env("ARC_QWEN_CONTEXT_TOKENS", 131072))
    resolved_input = _int_from_env("ARC_MAX_INPUT_TOKENS", _int_from_env("ARC_QWEN_MAX_INPUT_TOKENS", 65536))
    resolved_output = _int_from_env("ARC_MAX_OUTPUT_TOKENS", _int_from_env("ARC_QWEN_MAX_OUTPUT_TOKENS", 32000))

    # vLLM Speculative Decoding / MTP=3 resolution
    resolved_mtp_enabled = _bool_from_env(
        "ARC_VLLM_MTP_ENABLED",
        _bool_from_env("VLLM_MTP_ENABLED", True),
    )
    resolved_mtp_tokens = _int_from_env(
        "ARC_VLLM_MTP_TOKENS",
        _int_from_env("VLLM_MTP_TOKENS", _int_from_env("VLLM_SPECULATIVE_TOKENS", 3)),
    )
    resolved_spec_method = (
        os.environ.get("ARC_VLLM_SPECULATIVE_METHOD")
        or os.environ.get("VLLM_SPEC_METHOD")
        or os.environ.get("VLLM_SPECULATIVE_METHOD")
        or "mtp"
    )
    resolved_spec_model = (
        os.environ.get("ARC_VLLM_SPECULATIVE_MODEL")
        or os.environ.get("VLLM_SPEC_MODEL")
        or os.environ.get("VLLM_SPECULATIVE_MODEL")
    )
    resolved_spec_config = (
        os.environ.get("ARC_VLLM_SPECULATIVE_CONFIG")
        or os.environ.get("VLLM_SPECULATIVE_CONFIG")
    )
    resolved_spec_format = os.environ.get("ARC_VLLM_SPECULATIVE_CLI_FORMAT", "auto")

    cfg = V10Config(
        llm_advisor_backend=os.environ.get("ARC_LLM_ADVISOR_BACKEND", os.environ.get("ARC_V8_QWEN_BACKEND", "vllm")),
        model_path=resolved_model,
        qwen_model_path=resolved_model,
        vllm_base_url=resolved_base_url,
        qwen_vllm_base_url=resolved_base_url,
        vllm_api_key=resolved_api_key,
        qwen_vllm_api_key=resolved_api_key,
        context_tokens=resolved_ctx,
        qwen_context_tokens=resolved_ctx,
        max_input_tokens=resolved_input,
        qwen_max_input_tokens=resolved_input,
        max_output_tokens=resolved_output,
        qwen_max_output_tokens=resolved_output,
        temperature=resolved_temp,
        qwen_temperature=resolved_temp,
        top_p=resolved_top_p,
        qwen_top_p=resolved_top_p,
        top_k=resolved_top_k,
        qwen_top_k=resolved_top_k,
        min_p=resolved_min_p,
        qwen_min_p=resolved_min_p,
        presence_penalty=resolved_presence,
        qwen_presence_penalty=resolved_presence,
        repeat_penalty=resolved_repeat,
        qwen_repeat_penalty=resolved_repeat,
        seed=resolved_seed,
        qwen_seed=resolved_seed,
        timeout_seconds=resolved_timeout,
        qwen_timeout_seconds=resolved_timeout,
        multimodal_enabled=resolved_mm,
        qwen_multimodal_enabled=resolved_mm,
        solver_multimodal_enabled=resolved_solver_mm,
        explorer_multimodal_enabled=resolved_explorer_mm,
        coder_multimodal_enabled=resolved_coder_mm,
        enable_thinking=resolved_thinking,
        qwen_enable_thinking=resolved_thinking,
        reasoning_strength=resolved_strength,
        reasoning_budget_tokens=resolved_budget,
        qwen_reasoning_budget_tokens=resolved_budget,
        max_chain_attempts_per_level=_int_from_env("ARC_MAX_CHAIN_ATTEMPTS", 5),
        max_coder_retries_per_level=_int_from_env("ARC_MAX_CODER_RETRIES", 5),
        max_solver_retries_per_level=_int_from_env("ARC_MAX_SOLVER_RETRIES", 5),
        max_explorer_probe_actions_per_level=_int_from_env("ARC_MAX_EXPLORER_PROBES", 8),
        max_total_llm_calls_per_level=_int_from_env("ARC_MAX_TOTAL_LLM_CALLS_PER_LEVEL", 15),
        max_candidates_per_solver_package=_int_from_env("ARC_MAX_CANDIDATES_PER_PACKAGE", 4),
        max_steps_per_candidate=_int_from_env("ARC_MAX_STEPS_PER_CANDIDATE", 20),
        execute_one_step_at_a_time=_bool_from_env("ARC_EXECUTE_ONE_STEP_AT_A_TIME", True),
        sandbox_enabled=_bool_from_env("ARC_SANDBOX_ENABLED", True),
        sandbox_max_cpu_seconds=_float_from_env("ARC_SANDBOX_MAX_CPU_SECONDS", 10.0),
        sandbox_max_memory_mb=_int_from_env("ARC_SANDBOX_MAX_MEMORY_MB", 512),
        game_memory_reset_on_game_change=_bool_from_env("ARC_GAME_MEMORY_RESET_ON_GAME_CHANGE", True),
        game_memory_reset_on_level_change=_bool_from_env("ARC_GAME_MEMORY_RESET_ON_LEVEL_CHANGE", False),
        epistemic_memory_max_entries=_int_from_env("ARC_EPISTEMIC_MEMORY_MAX_ENTRIES", 50),
        syntax_error_memory_max_entries=_int_from_env("ARC_SYNTAX_ERROR_MEMORY_MAX_ENTRIES", 5),
        enable_symbolic_fallback=_bool_from_env("ARC_ENABLE_SYMBOLIC_FALLBACK", True),
        coder_exhaustion_forces_fallback=_bool_from_env("ARC_CODER_EXHAUSTION_FORCES_FALLBACK", True),
        solver_exhaustion_forces_fallback=_bool_from_env("ARC_SOLVER_EXHAUSTION_FORCES_FALLBACK", True),
        abort_on_dsl_exhaustion=_bool_from_env("ARC_ABORT_ON_DSL_EXHAUSTION", False),
        max_primitive_probes_per_level=_int_from_env("ARC_MAX_PRIMITIVE_PROBES", 30),
        enable_primitive_probing=_bool_from_env("ARC_ENABLE_PRIMITIVE_PROBING", True),
        probe_reset_after_discrete=_bool_from_env("ARC_PROBE_RESET_AFTER_DISCRETE", False),
        max_actions_per_game=_int_from_env("LCLD_MAX_ACTIONS_PER_GAME", 250),
        max_actions_per_level=_int_from_env("LCLD_MAX_ACTIONS_PER_LEVEL", 250),
        max_game_over_resets_per_game=_int_from_env("ARC_MAX_GAME_OVER_RESETS_PER_GAME", 5),
        max_game_over_resets_per_level=_int_from_env("ARC_MAX_GAME_OVER_RESETS_PER_LEVEL", 5),
        reset_on_game_over=_bool_from_env("ARC_RESET_ON_GAME_OVER", True),
        game_wall_clock_limit_seconds=_float_from_env("LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS", 5000.0),
        competition_wall_clock_limit_seconds=_float_from_env("LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS", 30600.0),
        concurrency=_int_from_env("LCLD_GAME_CONCURRENCY", 4),
        vllm_max_num_seqs=_int_from_env("LCLD_VLLM_MAX_NUM_SEQS", 4),
        vllm_startup_timeout_seconds=_int_from_env("VLLM_STARTUP_TIMEOUT_SECONDS", 900),
        vllm_mtp_enabled=resolved_mtp_enabled,
        vllm_mtp_tokens=resolved_mtp_tokens,
        vllm_speculative_method=resolved_spec_method,
        vllm_speculative_model=resolved_spec_model,
        vllm_speculative_config=resolved_spec_config,
        vllm_speculative_cli_format=resolved_spec_format,
        enable_persistent_tracker=_bool_from_env("ARC_ENABLE_PERSISTENT_TRACKER", True),
        track_match_threshold=_float_from_env("ARC_TRACK_MATCH_THRESHOLD", 0.45),
        track_max_age=_int_from_env("ARC_TRACK_MAX_AGE", 5),
        matching_ambiguity_threshold=_float_from_env("ARC_MATCHING_AMBIGUITY_THRESHOLD", 0.15),
        min_reliable_delta=_float_from_env("ARC_MIN_RELIABLE_DELTA", 0.8),
        track_confidence_threshold=_float_from_env("ARC_TRACK_CONFIDENCE_THRESHOLD", 0.6),
        occlusion_radius=_int_from_env("ARC_OCCLUSION_RADIUS", 3),
        cumulative_window=_int_from_env("ARC_CUMULATIVE_WINDOW", 3),
        enable_undecided_verdict=_bool_from_env("ARC_ENABLE_UNDECIDED_VERDICT", True),
        max_undecided_streak=_int_from_env("ARC_MAX_UNDECIDED_STREAK", 2),
        max_evidence_probes_per_level=_int_from_env("ARC_MAX_EVIDENCE_PROBES", 2),
    )
    if overrides:
        cfg.update_runtime(overrides)
    return cfg


def build_vllm_speculative_args(
    cfg: V10Config | None = None,
    model_path: str | Any | None = None,
) -> list[str]:
    """Build CLI arguments for vLLM speculative decoding / MTP.

    Supports:
    - "auto" / "config_json": --speculative-config '{"method": "...", "num_speculative_tokens": N}'
    - "spec_tokens": --spec-method <method> --spec-tokens <N> [--spec-model <model>]
    - "speculative_model": --speculative-model <model> --num-speculative-tokens <N>
    """
    import json

    if cfg is None:
        cfg = config_from_env()

    if not cfg.vllm_mtp_enabled or cfg.vllm_mtp_tokens <= 0:
        return []

    # 1. Explicit raw JSON override if provided
    if cfg.vllm_speculative_config:
        return ["--speculative-config", str(cfg.vllm_speculative_config)]

    cli_format = (cfg.vllm_speculative_cli_format or "auto").strip().lower()
    spec_model = cfg.vllm_speculative_model or (str(model_path) if model_path else None)

    if cli_format == "spec_tokens":
        args = ["--spec-method", str(cfg.vllm_speculative_method), "--spec-tokens", str(cfg.vllm_mtp_tokens)]
        if spec_model:
            args.extend(["--spec-model", str(spec_model)])
        return args

    if cli_format == "speculative_model":
        target_model = spec_model or (str(model_path) if model_path else str(cfg.model_path))
        return ["--speculative-model", target_model, "--num-speculative-tokens", str(cfg.vllm_mtp_tokens)]

    # "auto" or "config_json" (universal modern vLLM format)
    spec_dict: dict[str, Any] = {
        "method": str(cfg.vllm_speculative_method),
        "num_speculative_tokens": int(cfg.vllm_mtp_tokens),
    }
    if cfg.vllm_speculative_model:
        spec_dict["model"] = str(cfg.vllm_speculative_model)

    return ["--speculative-config", json.dumps(spec_dict)]


def config_from_mapping(mapping: Mapping[str, Any]) -> V10Config:
    """Build V10Config from a mapping/dict."""
    return config_from_env(mapping)


AgentConfig = V10Config

