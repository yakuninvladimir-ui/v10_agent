"""
V10 Agent Configuration Module

Ref: Engineering Specification V10.0, Section 2.1 - Configuration Contract
Ref: Architectural Specification V10.0, Section 6 - Budgets & Hard Limits

This module defines the V10Config class for reading configuration from
environment variables with strict defaults as per specification.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class V10Config:
    """
    Configuration container for ARC-AGI-3 LCLD Agent Version 10.0.
    
    All numeric limits that affect competition scoring or safety must be
    readable from environment variables (ARC_MAX_CODER_RETRIES, etc.).
    Hard-coded competing overrides are forbidden.
    
    Ref: Spec 2.1 - Required keys (V10Config)
    """
    
    # =========================================================================
    # LLM Backend Configuration
    # Ref: Spec 2.1, lines 98-103
    # =========================================================================
    
    #: LLM advisor backend type: 'vllm', 'ollama', 'llama_cli', or 'fake'
    llm_advisor_backend: str = "vllm"
    
    #: Path to Qwen model checkpoint (env: ARC_QWEN_VLLM_MODEL)
    qwen_model_path: str = ""
    
    #: Context window size in tokens for Qwen model
    qwen_context_tokens: int = 32768
    
    #: Maximum input tokens per request
    qwen_max_input_tokens: int = 16384
    
    #: Maximum output tokens per request
    qwen_max_output_tokens: int = 4096
    
    #: Sampling temperature for generation
    qwen_temperature: float = 0.7
    
    #: Random seed for reproducibility
    qwen_seed: int = 42
    
    #: Timeout in seconds for LLM requests
    qwen_timeout_seconds: float = 120.0
    
    #: Whether multimodal input (annotated frames) is enabled
    qwen_multimodal_enabled: bool = True
    
    # =========================================================================
    # Tri-Agent Budgets (normative ceilings)
    # Ref: Spec 2.1, lines 105-109
    # =========================================================================
    
    #: Maximum retry attempts for Coder agent per level (env: ARC_MAX_CODER_RETRIES)
    max_coder_retries_per_level: int = 3
    
    #: Maximum retry attempts for Solver agent per level (env: ARC_MAX_SOLVER_RETRIES)
    max_solver_retries_per_level: int = 4
    
    #: Maximum probe actions Explorer can take per level
    #: (env: ARC_MAX_EXPLORER_PROBE_ACTIONS)
    max_explorer_probe_actions_per_level: int = 8
    
    #: Soft monitoring limit for total LLM calls per level
    #: (env: ARC_MAX_TOTAL_LLM_CALLS)
    max_total_llm_calls_per_level: Optional[int] = None
    
    # =========================================================================
    # Trajectory / Package Limits
    # Ref: Spec 2.1, lines 111-114
    # =========================================================================
    
    #: Maximum candidate trajectories per solver package
    max_candidates_per_solver_package: int = 4
    
    #: Maximum steps allowed per candidate trajectory
    max_steps_per_candidate: int = 6
    
    #: Enforce single-step execution at a time (competition invariant)
    execute_one_step_at_a_time: bool = True
    
    # =========================================================================
    # Sandbox Configuration
    # Ref: Spec 2.1, lines 116-120
    # =========================================================================
    
    #: Whether sandbox execution is enabled
    sandbox_enabled: bool = True
    
    #: Whitelist of modules allowed in sandbox
    sandbox_allowed_modules: List[str] = field(default_factory=lambda: [
        "typing",
        "dataclasses",
        "enum",
        "json",
        "math",
    ])
    
    #: Maximum CPU seconds allowed for sandbox execution
    sandbox_max_cpu_seconds: float = 5.0
    
    #: Maximum memory (MB) allowed for sandbox execution
    sandbox_max_memory_mb: int = 256
    
    # =========================================================================
    # Memory Configuration
    # Ref: Spec 2.1, lines 122-126
    # =========================================================================
    
    #: Reset all memory contours on game change
    game_memory_reset_on_game_change: bool = True
    
    #: Reset level-specific memories on level change (default: False per spec)
    game_memory_reset_on_level_change: bool = False
    
    #: Maximum entries in EpistemicMemory
    epistemic_memory_max_entries: int = 100
    
    #: Maximum entries in SyntaxErrorMemory (spec default: 5)
    syntax_error_memory_max_entries: int = 5
    
    # =========================================================================
    # Fallback Configuration
    # Ref: Spec 2.1, lines 128-130
    # =========================================================================
    
    #: Enable symbolic fallback path when agents fail
    enable_symbolic_fallback: bool = True
    
    #: Force fallback when Coder retries exhausted
    coder_exhaustion_forces_fallback: bool = True
    
    @classmethod
    def from_env(cls) -> "V10Config":
        """
        Construct V10Config from environment variables.
        
        Environment variable mappings:
        - ARC_QWEN_VLLM_MODEL -> qwen_model_path
        - ARC_MAX_CODER_RETRIES -> max_coder_retries_per_level
        - ARC_MAX_SOLVER_RETRIES -> max_solver_retries_per_level
        - ARC_MAX_EXPLORER_PROBE_ACTIONS -> max_explorer_probe_actions_per_level
        - ARC_MAX_TOTAL_LLM_CALLS -> max_total_llm_calls_per_level
        
        Ref: Spec 2.1, line 133: "All numeric limits...must be readable from env vars"
        """
        kwargs = {}
        
        # LLM Backend
        if model_path := os.environ.get("ARC_QWEN_VLLM_MODEL"):
            kwargs["qwen_model_path"] = model_path
        
        if backend := os.environ.get("ARC_LLM_BACKEND"):
            kwargs["llm_advisor_backend"] = backend
        
        if ctx := os.environ.get("ARC_QWEN_CONTEXT_TOKENS"):
            kwargs["qwen_context_tokens"] = int(ctx)
        
        if max_in := os.environ.get("ARC_QWEN_MAX_INPUT_TOKENS"):
            kwargs["qwen_max_input_tokens"] = int(max_in)
        
        if max_out := os.environ.get("ARC_QWEN_MAX_OUTPUT_TOKENS"):
            kwargs["qwen_max_output_tokens"] = int(max_out)
        
        if temp := os.environ.get("ARC_QWEN_TEMPERATURE"):
            kwargs["qwen_temperature"] = float(temp)
        
        if seed := os.environ.get("ARC_QWEN_SEED"):
            kwargs["qwen_seed"] = int(seed)
        
        if timeout := os.environ.get("ARC_QWEN_TIMEOUT_SECONDS"):
            kwargs["qwen_timeout_seconds"] = float(timeout)
        
        if multimodal := os.environ.get("ARC_QWEN_MULTIMODAL_ENABLED"):
            kwargs["qwen_multimodal_enabled"] = multimodal.lower() in ("true", "1", "yes")
        
        # Tri-agent budgets
        if coder_retries := os.environ.get("ARC_MAX_CODER_RETRIES"):
            kwargs["max_coder_retries_per_level"] = int(coder_retries)
        
        if solver_retries := os.environ.get("ARC_MAX_SOLVER_RETRIES"):
            kwargs["max_solver_retries_per_level"] = int(solver_retries)
        
        if explorer_probes := os.environ.get("ARC_MAX_EXPLORER_PROBE_ACTIONS"):
            kwargs["max_explorer_probe_actions_per_level"] = int(explorer_probes)
        
        if total_calls := os.environ.get("ARC_MAX_TOTAL_LLM_CALLS"):
            kwargs["max_total_llm_calls_per_level"] = int(total_calls)
        
        # Trajectory limits
        if candidates := os.environ.get("ARC_MAX_CANDIDATES_PER_PACKAGE"):
            kwargs["max_candidates_per_solver_package"] = int(candidates)
        
        if steps := os.environ.get("ARC_MAX_STEPS_PER_CANDIDATE"):
            kwargs["max_steps_per_candidate"] = int(steps)
        
        if one_step := os.environ.get("ARC_EXECUTE_ONE_STEP_AT_A_TIME"):
            kwargs["execute_one_step_at_a_time"] = one_step.lower() in ("true", "1", "yes")
        
        # Sandbox
        if sandbox_enabled := os.environ.get("ARC_SANDBOX_ENABLED"):
            kwargs["sandbox_enabled"] = sandbox_enabled.lower() in ("true", "1", "yes")
        
        if allowed_mods := os.environ.get("ARC_SANDBOX_ALLOWED_MODULES"):
            kwargs["sandbox_allowed_modules"] = allowed_mods.split(",")
        
        if cpu_secs := os.environ.get("ARC_SANDBOX_MAX_CPU_SECONDS"):
            kwargs["sandbox_max_cpu_seconds"] = float(cpu_secs)
        
        if mem_mb := os.environ.get("ARC_SANDBOX_MAX_MEMORY_MB"):
            kwargs["sandbox_max_memory_mb"] = int(mem_mb)
        
        # Memory
        if reset_game := os.environ.get("ARC_GAME_MEMORY_RESET_ON_GAME_CHANGE"):
            kwargs["game_memory_reset_on_game_change"] = reset_game.lower() in ("true", "1", "yes")
        
        if reset_level := os.environ.get("ARC_GAME_MEMORY_RESET_ON_LEVEL_CHANGE"):
            kwargs["game_memory_reset_on_level_change"] = reset_level.lower() in ("true", "1", "yes")
        
        if epi_max := os.environ.get("ARC_EPISTEMIC_MEMORY_MAX_ENTRIES"):
            kwargs["epistemic_memory_max_entries"] = int(epi_max)
        
        if syn_max := os.environ.get("ARC_SYNTAX_ERROR_MEMORY_MAX_ENTRIES"):
            kwargs["syntax_error_memory_max_entries"] = int(syn_max)
        
        # Fallback
        if fallback := os.environ.get("ARC_ENABLE_SYMBOLIC_FALLBACK"):
            kwargs["enable_symbolic_fallback"] = fallback.lower() in ("true", "1", "yes")
        
        if coder_force := os.environ.get("ARC_CODER_EXHAUSTION_FORCES_FALLBACK"):
            kwargs["coder_exhaustion_forces_fallback"] = coder_force.lower() in ("true", "1", "yes")
        
        return cls(**kwargs)
