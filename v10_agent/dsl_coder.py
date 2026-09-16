"""DSL Coder Agent (Call 2 Family).

Generates dynamic Python DSL and typed function manifest, verified in the Sandbox,
writing exclusively to SyntaxErrorMemory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from v10_agent.config import V10Config
from v10_agent.llm_advisor import BaseLLMAdvisor, sanitize_model_response
from v10_agent.memory_contours import SyntaxErrorMemory, SyntaxErrorRecord
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
from v10_agent.sandbox import SandboxedModule, SandboxExecutor

logger = logging.getLogger(__name__)


from v10_agent.explorer_agent import clean_and_parse_json

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

    return source, manifest


class DSLCoder:
    """Call 2: Emits verified level-specific Python DSL and typed manifest."""

    def __init__(
        self,
        config: V10Config,
        advisor: BaseLLMAdvisor,
        sandbox_executor: SandboxExecutor,
    ):
        self.config = config
        self.advisor = advisor
        self.sandbox_executor = sandbox_executor

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

            try:
                response = self.advisor.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    config=self.config,
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
            logger.info(f"DSLCoder generated valid DSL module on attempt {attempt}/{retries}")
            return module, manifest, []

        logger.error(f"DSLCoder exhausted retries ({retries}). Fallback required.")
        return None, None, diagnostics_history
