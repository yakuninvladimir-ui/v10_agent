"""
V10 Agent LLM Client Module

Ref: Engineering Specification V10.0, Section 8 - LLM Integration
Ref: Architectural Specification V10.0, Section 1.4 - Isolation Invariants

This module provides the VLLMClient for interacting with the Qwen 3.8B FP8 model
via vLLM's OpenAI-compatible API. Critical invariant: thinking traces are NEVER
included in the payload that goes to LayeredVerifier or Sandbox.

Accelerator Configuration (ISO-compliant):
    ACCELERATOR = "rtx6000"
    _ACCELERATORS = {
        "rtx6000": {
            "name": "nvidiaRtx6000",
            "machine_shape": "NvidiaRtxPro6000",
            "gpu": True,
        },
    }
"""

import re
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import urllib.request
import urllib.error


# =============================================================================
# Accelerator Configuration (Strict per specification)
# =============================================================================

ACCELERATOR = "rtx6000"

_ACCELERATORS = {
    "rtx6000": {
        "name": "nvidiaRtx6000",
        "machine_shape": "NvidiaRtxPro6000",
        "gpu": True,
    },
}


def get_accelerator_config(accelerator_name: str = ACCELERATOR) -> Dict[str, Any]:
    """
    Retrieve accelerator configuration by name.
    
    Args:
        accelerator_name: Name of the accelerator config
        
    Returns:
        Dictionary with accelerator configuration
        
    Raises:
        ValueError: If accelerator_name not found in _ACCELERATORS
    """
    if accelerator_name not in _ACCELERATORS:
        raise ValueError(f"Unknown accelerator: {accelerator_name}. Available: {list(_ACCELERATORS.keys())}")
    return _ACCELERATORS[accelerator_name]


# =============================================================================
# Role Configuration
# =============================================================================

class AgentRole(Enum):
    """
    Enumeration of agent roles with strict isolation boundaries.
    
    Ref: Spec 1.4 - Isolation Invariants (ISO-1 through ISO-5)
    """
    EXPLORER = "explorer"
    CODER = "coder"
    SOLVER = "solver"


@dataclass(frozen=True)
class RoleConfig:
    """
    Configuration for a specific agent role.
    
    Parameters are transmitted to vLLM via extra_body and standard OpenAI API fields.
    
    Attributes:
        role: The agent role this config applies to
        enable_thinking: Whether to enable reasoning/thinking mode
        reasoning_budget_tokens: Maximum tokens for reasoning (if supported)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum completion tokens
    """
    role: AgentRole
    enable_thinking: bool
    reasoning_budget_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    
    def to_vllm_params(self) -> Dict[str, Any]:
        """
        Convert RoleConfig to vLLM API parameters.
        
        Returns:
            Dictionary suitable for vLLM chat completions API
        """
        params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        
        # Configure thinking via chat_template_kwargs
        if self.enable_thinking:
            params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }
            # Include reasoning budget if specified
            if self.reasoning_budget_tokens is not None:
                params["extra_body"]["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        else:
            # Explicitly disable thinking
            params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        return params


# Predefined role configurations per specification
ROLE_CONFIGS: Dict[AgentRole, RoleConfig] = {
    AgentRole.EXPLORER: RoleConfig(
        role=AgentRole.EXPLORER,
        enable_thinking=True,
        reasoning_budget_tokens=16000,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    ),
    AgentRole.CODER: RoleConfig(
        role=AgentRole.CODER,
        enable_thinking=False,
        temperature=0.1,
        top_p=0.95,
        max_tokens=4096,
    ),
    AgentRole.SOLVER: RoleConfig(
        role=AgentRole.SOLVER,
        enable_thinking=True,
        reasoning_budget_tokens=32000,
        temperature=0.4,
        top_p=0.95,
        max_tokens=4096,
    ),
}


# =============================================================================
# Parsed Response Dataclass
# =============================================================================

@dataclass
class ParsedResponse:
    """
    Container for parsed LLM response with isolated thinking trace.
    
    Critical: payload contains ONLY the clean JSON output, with ALL thinking
    traces removed. reasoning_trace is stored separately for logging only.
    
    Attributes:
        status: Response status ("OK", "PARSE_ERROR", "TIMEOUT")
        payload: Clean JSON dict (no thinking traces)
        reasoning_trace: Raw reasoning content (isolated, for logs only)
        usage: Token usage statistics
    """
    status: str  # "OK" | "PARSE_ERROR" | "TIMEOUT"
    payload: Optional[Dict[str, Any]] = None
    reasoning_trace: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def create_timeout(cls) -> "ParsedResponse":
        """Create a timeout response."""
        return cls(status="TIMEOUT")
    
    @classmethod
    def create_parse_error(cls, raw_content: str) -> "ParsedResponse":
        """Create a parse error response."""
        return cls(status="PARSE_ERROR", payload={"_raw_content": raw_content})
    
    @classmethod
    def create_ok(cls, payload: Dict[str, Any], reasoning_trace: Optional[str] = None, 
                  usage: Optional[Dict[str, int]] = None) -> "ParsedResponse":
        """Create a successful response."""
        return cls(
            status="OK",
            payload=payload,
            reasoning_trace=reasoning_trace,
            usage=usage or {}
        )


# =============================================================================
# VLLM Client
# =============================================================================

class VLLMClient:
    """
    Client for interacting with vLLM OpenAI-compatible API.
    
    This client is stateless and thread-safe. All configuration is passed
    through the constructor or method parameters.
    
    Features:
    - Role-based routing with strict isolation
    - Thinking trace extraction and isolation
    - JSON schema enforcement
    - Timeout and network error handling
    
    Args:
        base_url: Base URL of the vLLM API server
        model_name: Model identifier for the API
        timeout_seconds: Request timeout in seconds
        default_role_config: Default role configuration override
    """
    
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model_name: str = "qwen-3.8b-fp8",
        timeout_seconds: float = 120.0,
        default_role_config: Optional[Dict[AgentRole, RoleConfig]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.role_configs = default_role_config or ROLE_CONFIGS.copy()
    
    def generate(
        self,
        role: AgentRole,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> ParsedResponse:
        """
        Generate a response from the LLM for a specific role.
        
        Args:
            role: The agent role (EXPLORER, CODER, SOLVER)
            messages: List of message dicts for the chat API
            json_schema: Optional JSON schema for response_format
            
        Returns:
            ParsedResponse with isolated thinking trace
            
        Raises:
            No exceptions raised; errors returned as ParsedResponse with appropriate status
        """
        # Get role configuration
        role_config = self.role_configs.get(role)
        if role_config is None:
            return ParsedResponse.create_parse_error(f"Unknown role: {role}")
        
        # Build request payload
        request_payload = self._build_request_payload(role_config, messages, json_schema)
        
        # Make HTTP request
        try:
            response_dict = self._make_http_request(request_payload)
        except TimeoutError:
            return ParsedResponse.create_timeout()
        except Exception as e:
            return ParsedResponse.create_parse_error(f"Network error: {str(e)}")
        
        # Extract and isolate thinking
        return self._isolate_and_parse_thinking(response_dict)
    
    def _build_request_payload(
        self,
        role_config: RoleConfig,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the vLLM API request payload.
        
        Args:
            role_config: Role-specific configuration
            messages: Chat messages
            json_schema: Optional JSON schema for structured output
            
        Returns:
            Complete request payload dict
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        
        # Add role-specific parameters
        vllm_params = role_config.to_vllm_params()
        payload.update(vllm_params)
        
        # Add JSON schema enforcement if provided
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "schema": json_schema
                }
            }
        
        return payload
    
    def _make_http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP POST request to vLLM API.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            Parsed JSON response
            
        Raises:
            TimeoutError: On request timeout
            urllib.error.URLError: On network errors
            json.JSONDecodeError: On invalid JSON response
        """
        api_url = f"{self.base_url}/v1/chat/completions"
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        start_time = time.time()
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data)
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or (time.time() - start_time >= self.timeout_seconds):
                raise TimeoutError(f"Request timed out after {self.timeout_seconds}s")
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
    
    def _isolate_and_parse_thinking(self, response: Dict[str, Any]) -> ParsedResponse:
        """
        Extract and isolate thinking traces from vLLM response.
        
        This is the CRITICAL method for maintaining ISO isolation invariants.
        Thinking traces MUST NEVER appear in the payload.
        
        Steps:
          A: Extract message.content
          B: Strip <think>...</think> tags from content using regex
          C: Extract message.reasoning_content to separate field
          D: Clean content from markdown wrappers (```json ... ```)
          E: Parse JSON; on failure return PARSE_ERROR
        
        Args:
            response: Raw vLLM API response dict
            
        Returns:
            ParsedResponse with isolated reasoning_trace
        """
        # Extract usage info if available
        usage = {}
        if "usage" in response:
            usage_info = response["usage"]
            usage = {
                "prompt_tokens": usage_info.get("prompt_tokens", 0),
                "completion_tokens": usage_info.get("completion_tokens", 0),
                "reasoning_tokens": usage_info.get("reasoning_tokens", 0),
            }
        
        # Extract message content
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", None)
        
        # Step A: Store raw content for processing
        raw_content = content if content else ""
        
        # Step B: Extract and remove <think>...</think> tags from content
        thinking_from_tags = None
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, raw_content, flags=re.DOTALL | re.IGNORECASE)
        
        if think_matches:
            # Join all thinking segments
            thinking_from_tags = "\n".join(think_matches).strip()
            # Remove thinking tags from content
            raw_content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Step C: Combine reasoning sources (priority: reasoning_content > tags)
        final_reasoning_trace = reasoning_content if reasoning_content else thinking_from_tags
        
        # Step D: Clean content from markdown wrappers
        cleaned_content = self._clean_markdown_wrappers(raw_content)
        
        # Step E: Parse JSON
        try:
            payload = json.loads(cleaned_content)
            if not isinstance(payload, dict):
                # If JSON is not a dict, wrap it
                payload = {"_result": payload}
            return ParsedResponse.create_ok(
                payload=payload,
                reasoning_trace=final_reasoning_trace,
                usage=usage
            )
        except json.JSONDecodeError as e:
            # Return parse error with raw content for debugging
            return ParsedResponse.create_parse_error(
                raw_content=f"JSON parse error: {str(e)} | Content: {cleaned_content[:500]}"
            )
    
    def _clean_markdown_wrappers(self, content: str) -> str:
        """
        Remove markdown code block wrappers from content.
        
        Handles patterns like:
        - ```json { ... } ```
        - ``` { ... } ```
        - ```{ ... }```
        
        Args:
            content: Raw content string
            
        Returns:
            Cleaned content without markdown wrappers
        """
        if not content:
            return content
        
        # Pattern to match markdown code blocks with optional language specifier
        pattern = r'^\s*```(?:json)?\s*(.*?)\s*```\s*$'
        match = re.match(pattern, content.strip(), flags=re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return content.strip()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    base_url: str = "http://127.0.0.1:1234",
    model_name: str = "qwen-3.8b-fp8",
    timeout_seconds: float = 120.0,
) -> VLLMClient:
    """
    Factory function to create a VLLMClient instance.
    
    Args:
        base_url: vLLM API server URL
        model_name: Model identifier
        timeout_seconds: Request timeout
        
    Returns:
        Configured VLLMClient instance
    """
    return VLLMClient(
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds
    )


def get_role_config(role: AgentRole) -> RoleConfig:
    """
    Retrieve the configuration for a specific role.
    
    Args:
        role: Agent role enum value
        
    Returns:
        RoleConfig for the specified role
    """
    return ROLE_CONFIGS[role]
