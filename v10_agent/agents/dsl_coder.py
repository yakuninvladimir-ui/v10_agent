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
    function_manifest: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full function manifest with signatures, docstrings, parameters, and return types"
    )
    
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
        
        # Build ISO-2 compliant prompt using prompt builder
        from ..prompt_builders.coder_prompt import build_coder_prompt
        
        prompt_text = build_coder_prompt(
            environment_spec=environment_spec,
            api_manifest=api_manifest,
            syntax_error_count=self.syntax_error_count,
            recent_errors=error_summaries,
        )
        
        # Build JSON schema for Coder response (includes function_manifest)
        json_schema = {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "function_names": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "function_manifest": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "signature": {"type": "string"},
                            "docstring": {"type": "string"},
                            "parameters": {"type": "object"},
                            "return_type": {"type": "string"}
                        },
                        "required": ["signature", "docstring", "parameters", "return_type"]
                    }
                }
            },
            "required": ["source_code", "function_names", "function_manifest"],
            "additionalProperties": False
        }
        
        # Call vLLM with proper interface
        if self.llm_client:
            from ..llm_client import AgentRole
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.generate(
                role=AgentRole.CODER,
                messages=messages,
                json_schema=json_schema
            )
            if response.status == "OK" and response.payload:
                raw_response = json.dumps(response.payload)
            else:
                raise ValueError(f"Coder LLM call failed: {response.status} - {response.error_message}")
        else:
            # Stub for offline testing
            raw_response = self._stub_generate(prompt_text)
        
        # Parse and validate JSON response
        return self._parse_response(raw_response)
    
    def _stub_generate(self, prompt: str) -> str:
        """
        Stub LLM generation for offline testing.
        In production, this calls vLLM with Qwen 3.8B model.
        
        Generates function_manifest based on api_manifest from prompt context.
        """
        # Extract function names from prompt (they appear in "Function: <name>" lines)
        import re
        func_matches = re.findall(r"Function: (\w+)", prompt)
        
        if func_matches:
            # Build manifest from functions mentioned in prompt
            function_manifest = {}
            for func_name in func_matches:
                function_manifest[func_name] = {
                    "signature": f"{func_name}(x, y)",
                    "docstring": f"DSL function {func_name} for grid operations",
                    "parameters": {"x": "int", "y": "int"},
                    "return_type": "EffectDeclaration"
                }
            
            # Generate source code stubs for all functions
            source_lines = []
            for func_name in func_matches:
                source_lines.append(f"def {func_name}(x, y):")
                source_lines.append(f"    return {{'effect': '{func_name}', 'x': x, 'y': y}}")
                source_lines.append("")
            
            return json.dumps({
                "source_code": "\n".join(source_lines),
                "function_names": func_matches,
                "function_manifest": function_manifest
            })
        else:
            # Fallback to default stub
            return json.dumps({
                "source_code": "def stub_function(x, y): return {'effect': 'probe', 'x': x, 'y': y}",
                "function_names": ["stub_function"],
                "function_manifest": {
                    "stub_function": {
                        "signature": "stub_function(x, y)",
                        "docstring": "Stub probe function for testing",
                        "parameters": {"x": "int", "y": "int"},
                        "return_type": "EffectDeclaration"
                    }
                }
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
