"""
VLLM Client Module for ARC-AGI-3 LCLD Agent Version 10.0

Ref: Engineering Specification V10.0, Section 2.1 - LLM Backend Configuration
Ref: Architectural Specification V10.0, Section 4.1 - vLLM Integration

This module provides a stateless client for interacting with vLLM backend
(Qwen 3.8B FP8 model) via OpenAI-compatible API.

CRITICAL INVARIANT: Thinking traces (reasoning_content, <think> tags) are
NEVER included in the payload returned to agents. They are isolated in
reasoning_trace field for logging/audit purposes ONLY.
"""

import re
import json
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


logger = logging.getLogger(__name__)


# =============================================================================
# Role Configuration
# Ref: Spec 6 - Budgets & Hard Limits
# =============================================================================

class AgentRole(Enum):
    """
    Enumeration of agent roles with distinct LLM configurations.
    
    Each role has specific parameters for creativity vs determinism:
    - EXPLORER: High creativity for spatial search (temperature=0.6)
    - CODER: Maximum determinism (temperature=0.1, thinking DISABLED)
    - SOLVER: Deep logical reasoning (temperature=0.4, extended budget)
    """
    EXPLORER = "explorer"
    CODER = "coder"
    SOLVER = "solver"


@dataclass(frozen=True)
class RoleConfig:
    """
    Configuration for a specific agent role's LLM interaction.
    
    Attributes:
        enable_thinking: Whether to enable model's internal reasoning
        reasoning_budget_tokens: Max tokens for reasoning (if enabled)
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling parameter
        max_output_tokens: Maximum completion tokens
    """
    enable_thinking: bool
    reasoning_budget_tokens: int
    temperature: float
    top_p: float
    max_output_tokens: int


# Role-specific configurations per Spec 6
ROLE_CONFIGS: Dict[AgentRole, RoleConfig] = {
    AgentRole.EXPLORER: RoleConfig(
        enable_thinking=True,
        reasoning_budget_tokens=16000,
        temperature=0.6,
        top_p=0.95,
        max_output_tokens=4096,
    ),
    AgentRole.CODER: RoleConfig(
        enable_thinking=False,  # DISABLED for maximum determinism
        reasoning_budget_tokens=0,
        temperature=0.1,  # Near-deterministic
        top_p=0.95,
        max_output_tokens=4096,
    ),
    AgentRole.SOLVER: RoleConfig(
        enable_thinking=True,
        reasoning_budget_tokens=32000,  # Extended budget for deep reasoning
        temperature=0.4,
        top_p=0.95,
        max_output_tokens=4096,
    ),
}


# =============================================================================
# Response Types
# =============================================================================

@dataclass
class ParsedResponse:
    """
    Parsed response from vLLM with isolated thinking traces.
    
    CRITICAL: The payload field contains ONLY clean JSON with NO thinking traces.
    The reasoning_trace field is for logging/audit ONLY and must NEVER be passed
    to agents or included in memory contours.
    
    Ref: ISO-1...ISO-5 Isolation Invariants
    
    Attributes:
        status: Response status ("OK", "PARSE_ERROR", "TIMEOUT")
        payload: Clean JSON dict (NO thinking traces) - safe for agents
        reasoning_trace: Raw thinking content (LOGGING ONLY) - NEVER expose to agents
        usage: Token usage statistics
        error_message: Error description if status != "OK"
    """
    status: Literal["OK", "PARSE_ERROR", "TIMEOUT", "CONNECTION_ERROR"]
    payload: Optional[Dict[str, Any]]
    reasoning_trace: Optional[str]
    usage: Dict[str, int]
    error_message: Optional[str] = None
    
    @classmethod
    def ok(cls, payload: Dict[str, Any], reasoning_trace: Optional[str], 
           usage: Dict[str, int]) -> "ParsedResponse":
        """Factory for successful responses."""
        return cls(
            status="OK",
            payload=payload,
            reasoning_trace=reasoning_trace,
            usage=usage,
            error_message=None
        )
    
    @classmethod
    def parse_error(cls, raw_content: str, error: str) -> "ParsedResponse":
        """Factory for parse errors."""
        return cls(
            status="PARSE_ERROR",
            payload=None,
            reasoning_trace=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
            error_message=f"JSON parse failed: {error}",
        )
    
    @classmethod
    def timeout(cls, error: str) -> "ParsedResponse":
        """Factory for timeout errors."""
        return cls(
            status="TIMEOUT",
            payload=None,
            reasoning_trace=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
            error_message=f"Request timeout: {error}",
        )
    
    @classmethod
    def connection_error(cls, error: str) -> "ParsedResponse":
        """Factory for connection errors."""
        return cls(
            status="CONNECTION_ERROR",
            payload=None,
            reasoning_trace=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
            error_message=f"Connection failed: {error}",
        )


# =============================================================================
# VLLM Client
# =============================================================================

class VLLMClient:
    """
    Stateless client for vLLM backend (Qwen 3.8B FP8).
    
    This client handles:
    1. Role-specific configuration (temperature, thinking, budgets)
    2. OpenAI-compatible API communication
    3. Critical isolation of thinking traces from agent payloads
    
    CRITICAL SECURITY INVARIANT:
    The _isolate_and_parse_thinking method ensures that reasoning traces
    (whether in <think> tags or reasoning_content field) are NEVER included
    in the payload returned to agents. This prevents:
    - Goal leakage to Explorer (ISO-1)
    - Traceback exposure to Coder (ISO-2)
    - Python source exposure to Solver (ISO-3)
    
    Usage:
        client = VLLMClient(base_url="http://127.0.0.1:1234")
        response = client.generate(
            role=AgentRole.SOLVER,
            messages=[{"role": "user", "content": "..."}],
            json_schema={"type": "object", ...}
        )
        if response.status == "OK":
            clean_payload = response.payload  # Safe to use
            # response.reasoning_trace goes to logs ONLY
    """
    
    DEFAULT_BASE_URL = "http://127.0.0.1:1234"
    DEFAULT_TIMEOUT = 120.0  # seconds
    MODEL_NAME = "Qwen/Qwen3-8B-FP8"  # Default model identifier
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT,
                 model_name: Optional[str] = None):
        """
        Initialize VLLMClient.
        
        Args:
            base_url: vLLM API endpoint (default: http://127.0.0.1:1234)
            timeout: Request timeout in seconds (default: 120.0)
            model_name: Model identifier for vLLM
        """
        self.base_url = base_url.rstrip("/") if base_url else self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.model_name = model_name or self.MODEL_NAME
        
        # No global state - client is stateless
    
    def generate(self, role: AgentRole, messages: List[Dict[str, Any]], 
                 json_schema: Dict[str, Any]) -> ParsedResponse:
        """
        Generate response from vLLM with role-specific configuration.
        
        Args:
            role: Agent role (determines temperature, thinking, budgets)
            messages: Chat messages in OpenAI format
            json_schema: JSON Schema for response validation
            
        Returns:
            ParsedResponse with isolated thinking traces
            
        Ref: ISO-1...ISO-5 - Thinking traces isolated in reasoning_trace field
        """
        # Get role-specific configuration
        config = ROLE_CONFIGS[role]
        
        # Build request body
        request_body = self._build_request_body(
            messages=messages,
            json_schema=json_schema,
            config=config
        )
        
        # Execute HTTP request
        try:
            response = requests.post(
                url=f"{self.base_url}/v1/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()
        except Timeout as e:
            logger.warning(f"vLLM request timeout for role={role.value}: {e}")
            return ParsedResponse.timeout(str(e))
        except ConnectionError as e:
            logger.warning(f"vLLM connection error for role={role.value}: {e}")
            return ParsedResponse.connection_error(str(e))
        except RequestException as e:
            logger.error(f"vLLM request failed for role={role.value}: {e}")
            return ParsedResponse.parse_error(
                raw_content="",
                error=f"HTTP error: {e}"
            )
        
        # Parse response and isolate thinking traces
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vLLM response JSON: {e}")
            return ParsedResponse.parse_error(
                raw_content=response.text[:500],
                error=f"Invalid vLLM response JSON: {e}"
            )
        
        return self._isolate_and_parse_thinking(response_data)
    
    def _build_request_body(self, messages: List[Dict[str, Any]], 
                            json_schema: Dict[str, Any],
                            config: RoleConfig) -> Dict[str, Any]:
        """
        Build request body for vLLM API with role-specific configuration.
        
        Args:
            messages: Chat messages
            json_schema: JSON Schema for response_format
            config: Role-specific configuration
            
        Returns:
            Request body dict for vLLM API
        """
        # Base request structure
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_output_tokens,
            "stream": False,
        }
        
        # Add JSON Schema for structured output
        # Ref: OpenAI Structured Outputs / vLLM JSON mode
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": json_schema
            }
        }
        
        # Configure thinking based on role
        # For Qwen3 models via vLLM, thinking is controlled via chat_template_kwargs
        if config.enable_thinking:
            # Enable thinking with budget constraint
            body["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
            # Note: reasoning_budget_tokens may not be directly supported by all vLLM endpoints
            # We rely on max_tokens as a soft limit; the model's internal reasoning is bounded
            # by its architecture and the prompt engineering
            logger.debug(f"Thinking ENABLED for role with budget={config.reasoning_budget_tokens}")
        else:
            # Explicitly disable thinking for CODER role (ISO-2 compliance)
            body["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
            logger.debug(f"Thinking DISABLED for role (deterministic mode)")
        
        return body
    
    def _isolate_and_parse_thinking(self, response_data: Dict[str, Any]) -> ParsedResponse:
        """
        CRITICAL METHOD: Isolate thinking traces and parse clean JSON payload.
        
        This method implements the core security invariant: thinking traces are
        NEVER included in the payload returned to agents. It handles two cases:
        
        Case A: vLLM returns reasoning in separate 'reasoning_content' field
        Case B: Thinking leaks into 'content' field wrapped in <think>...</think> tags
        
        Steps:
        A. Extract message.content
        B. Remove <think>...</think> tags using regex (preserves content for reasoning_trace)
        C. Extract reasoning_content field separately
        D. Clean markdown wrappers (```json ... ```)
        E. Parse JSON - return PARSE_ERROR on failure
        
        Ref: ISO-1...ISO-5 Isolation Invariants
        - reasoning_trace goes to logs ONLY
        - payload is clean JSON safe for agents
        
        Args:
            response_data: Raw vLLM response dict
            
        Returns:
            ParsedResponse with isolated thinking traces
        """
        # Extract message object
        choices = response_data.get("choices", [])
        if not choices:
            return ParsedResponse.parse_error(
                raw_content="",
                error="No choices in vLLM response"
            )
        
        message = choices[0].get("message", {})
        raw_content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", None)
        
        # Track extracted thinking for reasoning_trace
        thinking_parts: List[str] = []
        
        # Step B: Extract and remove <think>...</think> tags from content
        # This handles cases where thinking template fails and leaks into content
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, raw_content, flags=re.DOTALL | re.IGNORECASE)
        
        if think_matches:
            # Collect all thinking content
            thinking_parts.extend(think_matches)
            # Remove ALL thinking tags from content
            clean_content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL | re.IGNORECASE)
            logger.debug(f"Extracted {len(think_matches)} <think> blocks from content")
        else:
            clean_content = raw_content
        
        # Step C: Handle reasoning_content field (separate from content)
        if reasoning_content:
            thinking_parts.append(reasoning_content)
            logger.debug(f"Extracted reasoning_content field ({len(reasoning_content)} chars)")
        
        # Combine all thinking traces for audit logging
        reasoning_trace = "\n---\n".join(thinking_parts) if thinking_parts else None
        
        # Step D: Clean markdown wrappers
        clean_content = clean_content.strip()
        
        # Remove ```json ... ``` wrapper
        json_markdown_pattern = r'^```json\s*(.*?)\s*```$'
        match = re.match(json_markdown_pattern, clean_content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            clean_content = match.group(1).strip()
            logger.debug("Removed ```json markdown wrapper")
        
        # Remove generic ``` ... ``` wrapper
        generic_markdown_pattern = r'^```\s*(.*?)\s*```$'
        match = re.match(generic_markdown_pattern, clean_content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            clean_content = match.group(1).strip()
            logger.debug("Removed generic ``` markdown wrapper")
        
        # Step E: Parse JSON
        try:
            payload = json.loads(clean_content)
            if not isinstance(payload, dict):
                return ParsedResponse.parse_error(
                    raw_content=clean_content[:200],
                    error=f"Expected dict, got {type(payload).__name__}"
                )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed after cleaning: {e}")
            logger.debug(f"Clean content preview: {clean_content[:200]}")
            return ParsedResponse.parse_error(
                raw_content=clean_content[:200],
                error=str(e)
            )
        
        # Extract usage statistics
        usage_data = response_data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "reasoning_tokens": usage_data.get("reasoning_tokens", 0),
        }
        
        logger.info(
            f"vLLM generation successful: prompt={usage['prompt_tokens']}, "
            f"completion={usage['completion_tokens']}, reasoning={usage['reasoning_tokens']}"
        )
        
        return ParsedResponse.ok(
            payload=payload,
            reasoning_trace=reasoning_trace,
            usage=usage
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(base_url: Optional[str] = None, timeout: float = VLLMClient.DEFAULT_TIMEOUT,
                  model_name: Optional[str] = None) -> VLLMClient:
    """
    Factory function to create VLLMClient instance.
    
    Args:
        base_url: vLLM API endpoint
        timeout: Request timeout in seconds
        model_name: Model identifier
        
    Returns:
        Configured VLLMClient instance
    """
    return VLLMClient(base_url=base_url, timeout=timeout, model_name=model_name)


if __name__ == "__main__":
    # Simple smoke test (requires running vLLM server)
    print("VLLMClient module loaded successfully")
    print(f"Default base URL: {VLLMClient.DEFAULT_BASE_URL}")
    print(f"Model: {VLLMClient.MODEL_NAME}")
    print("\nRole configurations:")
    for role, config in ROLE_CONFIGS.items():
        thinking = "ENABLED" if config.enable_thinking else "DISABLED"
        print(f"  {role.value}: temp={config.temperature}, thinking={thinking}, budget={config.reasoning_budget_tokens}")
