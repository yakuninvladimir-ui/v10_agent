"""
Explorer Agent - ISO-1 Compliant Wrapper
Ref: Engineering Specification V10.0 Section 1.1
"""

import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError

from ..planning_set import PlanningSet
from ..prompt_builders.explorer_prompt import build_explorer_prompt


class ExplorerResponse(BaseModel):
    """
    Pydantic model for Explorer Agent JSON response validation.
    Ref: Spec 3.2 (Explorer Output Schema)
    """
    probes: List[Dict[str, Any]] = Field(..., description="List of probe recommendations")
    reasoning: str = Field(..., description="Explanation for probe selections")
    
    class Config:
        extra = "forbid"  # Strict JSON schema validation


class ExplorerAgent:
    """
    Explorer Agent wrapper that builds prompts and parses vLLM responses.
    
    ISO-1 INVARIANT: Explorer never receives goal information.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize Explorer Agent.
        
        Args:
            llm_client: vLLM client interface (stub compatible with Qwen 3.8B)
        """
        self.llm_client = llm_client
        self.response_history: List[Dict[str, Any]] = []
    
    def act(
        self,
        planning_set: PlanningSet,
        annotated_frame: str,
        action_history: List[Dict[str, Any]],
        probe_history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Execute Explorer agent to generate probe recommendations.
        
        Args:
            planning_set: Current PlanningSet snapshot
            annotated_frame: Annotated frame PNG (base64 or path)
            action_history: Previous actions taken
            probe_history: Optional probe results history
        
        Returns:
            Validated JSON response with probe recommendations
        
        Raises:
            ValueError: If JSON parsing fails or schema validation fails
            ISO1ViolationError: If goal leakage is detected
        """
        # Build ISO-1 compliant prompt using prompt builder
        from ..prompt_builders.explorer_prompt import build_explorer_prompt
        
        prompt_text = build_explorer_prompt(
            planning_set=planning_set,
            annotated_frame=annotated_frame,
            action_history=action_history,
            probe_history=probe_history,
        )
        
        # Build JSON schema for Explorer response
        json_schema = {
            "type": "object",
            "properties": {
                "probes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["x", "y", "confidence"]
                    }
                },
                "reasoning": {"type": "string"}
            },
            "required": ["probes", "reasoning"],
            "additionalProperties": False
        }
        
        # Call vLLM with proper interface
        if self.llm_client:
            from ..llm_client import AgentRole
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.generate(
                role=AgentRole.EXPLORER,
                messages=messages,
                json_schema=json_schema
            )
            if response.status == "OK" and response.payload:
                raw_response = json.dumps(response.payload)
            else:
                raise ValueError(f"Explorer LLM call failed: {response.status} - {response.error_message}")
        else:
            # Stub for offline testing
            raw_response = self._stub_generate(prompt_text)
        
        # Parse and validate JSON response
        return self._parse_response(raw_response)
    
    def _stub_generate(self, prompt: str) -> str:
        """
        Stub LLM generation for offline testing.
        In production, this calls vLLM with Qwen 3.8B model.
        """
        # Return minimal valid JSON for testing
        return json.dumps({
            "probes": [{"x": 0, "y": 0, "confidence": 0.5}],
            "reasoning": "Stub response for testing"
        })
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response with error handling.
        
        Ref: Spec 3.2 (Response Parsing Requirements)
        """
        try:
            # Try to extract JSON from response (may have markdown wrapping)
            json_str = self._extract_json(raw_response)
            
            # Validate with Pydantic
            validated = ExplorerResponse.model_validate_json(json_str)
            
            response_dict = validated.model_dump()
            self.response_history.append(response_dict)
            
            return response_dict
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Explorer JSON response: {e}")
        except ValidationError as e:
            raise ValueError(f"Explorer response schema validation failed: {e}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from potentially wrapped response text.
        Handles markdown code blocks and extra whitespace.
        """
        text = text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()
