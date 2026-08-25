"""
DSL Coder Agent - ISO-2 Compliant Wrapper
Ref: Engineering Specification V10.0 Section 1.2
"""

import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError

from ..types import EnvironmentSpecification, SyntaxErrorRecord
from ..prompt_builders.coder_prompt import build_coder_prompt, validate_no_goal_in_api_manifest


class CoderResponse(BaseModel):
    """
    Pydantic model for DSL Coder Agent JSON response validation.
    Ref: Spec 3.3 (Coder Output Schema)
    """
    source_code: str = Field(..., description="Python source code implementing DSL functions")
    function_names: List[str] = Field(..., description="List of implemented function names")
    
    class Config:
        extra = "forbid"  # Strict JSON schema validation


class DSLCoder:
    """
    DSL Coder Agent wrapper that builds prompts and parses vLLM responses.
    
    ISO-2 INVARIANT: Coder never receives goal information, only environment spec.
    ISO-3 INVARIANT: Coder sees error summaries, NOT full tracebacks.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize DSL Coder Agent.
        
        Args:
            llm_client: vLLM client interface (stub compatible with Qwen 3.8B)
        """
        self.llm_client = llm_client
        self.response_history: List[Dict[str, Any]] = []
        self.syntax_error_count = 0
    
    def act(
        self,
        environment_spec: EnvironmentSpecification,
        api_manifest: Dict[str, Any],
        recent_errors: Optional[List[SyntaxErrorRecord]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Coder agent to generate DSL function implementations.
        
        Args:
            environment_spec: Environment specification from Explorer
            api_manifest: JSON Function Manifest with available DSL functions
            recent_errors: Optional list of recent syntax error records (summaries only)
        
        Returns:
            Validated JSON response with source code
        
        Raises:
            ValueError: If JSON parsing fails or schema validation fails
            ISO2ViolationError: If goal leakage is detected in manifest
        """
        # ISO-2 Check: Validate no goal leakage in manifest
        validate_no_goal_in_api_manifest(api_manifest)
        
        # Prepare error summaries (NOT full tracebacks - ISO-3 compliance)
        error_summaries = None
        if recent_errors:
            error_summaries = [
                {"summary": err.summary, "level_id": err.level_id}
                for err in recent_errors[-5:]  # Last 5 errors
            ]
            self.syntax_error_count = len(recent_errors)
        
        # Build ISO-2 compliant prompt
        prompt = build_coder_prompt(
            environment_spec=environment_spec,
            api_manifest=api_manifest,
            syntax_error_count=self.syntax_error_count,
            recent_errors=error_summaries,
        )
        
        # Call vLLM (stub interface)
        if self.llm_client:
            raw_response = self.llm_client.generate(prompt)
        else:
            # Stub for offline testing
            raw_response = self._stub_generate(prompt)
        
        # Parse and validate JSON response
        return self._parse_response(raw_response)
    
    def _stub_generate(self, prompt: str) -> str:
        """
        Stub LLM generation for offline testing.
        In production, this calls vLLM with Qwen 3.8B model.
        """
        # Return minimal valid JSON for testing
        return json.dumps({
            "source_code": "def stub_function(): pass",
            "function_names": ["stub_function"]
        })
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response with error handling.
        
        Ref: Spec 3.3 (Response Parsing Requirements)
        """
        try:
            # Try to extract JSON from response (may have markdown wrapping)
            json_str = self._extract_json(raw_response)
            
            # Validate with Pydantic
            validated = CoderResponse.model_validate_json(json_str)
            
            response_dict = validated.model_dump()
            self.response_history.append(response_dict)
            
            return response_dict
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Coder JSON response: {e}")
        except ValidationError as e:
            raise ValueError(f"Coder response schema validation failed: {e}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from potentially wrapped response text.
        Handles markdown code blocks and extra whitespace.
        """
        text = text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```python"):
            text = text[9:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()
    
    def record_syntax_error(self, error_record: SyntaxErrorRecord) -> None:
        """
        Record a syntax error for future context.
        
        Ref: Spec 3.5.2 (SyntaxErrorMemory Contract)
        Note: Only stores summary, not full traceback (ISO-3)
        """
        # This method would typically interact with SyntaxErrorMemory
        # For now, just track count
        self.syntax_error_count += 1
