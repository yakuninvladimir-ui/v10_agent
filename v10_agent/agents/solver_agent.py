"""
Solver Agent - ISO-3 Compliant Wrapper
Ref: Engineering Specification V10.0 Section 1.3
"""

import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError

from ..types import BrusentsovJudgment, BranchSignature
from ..prompt_builders.solver_prompt import build_solver_prompt, validate_no_python_source_in_manifest


class SolverStep(BaseModel):
    """Single step in a trajectory candidate."""
    function: str = Field(..., description="Function name from manifest")
    args: Dict[str, Any] = Field(default_factory=dict, description="Function arguments")


class SolverCandidate(BaseModel):
    """A single trajectory candidate."""
    steps: List[SolverStep] = Field(..., description="Sequence of function calls")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class SolverResponse(BaseModel):
    """
    Pydantic model for Solver Agent JSON response validation.
    Ref: Spec 3.4 (Solver Output Schema)
    """
    candidates: List[SolverCandidate] = Field(..., description="List of trajectory candidates")
    
    class Config:
        extra = "forbid"  # Strict JSON schema validation


class SolverAgent:
    """
    Solver Agent wrapper that builds prompts and parses vLLM responses.
    
    ISO-3 INVARIANT: Solver never sees Python source code, only function manifests.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize Solver Agent.
        
        Args:
            llm_client: vLLM client interface (stub compatible with Qwen 3.8B)
        """
        self.llm_client = llm_client
        self.response_history: List[Dict[str, Any]] = []
    
    def act(
        self,
        function_manifest: Dict[str, Any],
        epistemic_summary: Optional[Dict[str, Any]] = None,
        live_omit_branches: Optional[List[BranchSignature]] = None,
        severed_null_signatures: Optional[List[BranchSignature]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Solver agent to generate trajectory candidates.
        
        Args:
            function_manifest: JSON Function Manifest with DSL functions
            epistemic_summary: Summary of previous Brusentsov judgments
            live_omit_branches: Branches marked as OMIT (can be continued)
            severed_null_signatures: Branches marked as NULL (contradicted)
        
        Returns:
            Validated JSON response with trajectory candidates
        
        Raises:
            ValueError: If JSON parsing fails or schema validation fails
            ISO3ViolationError: If Python source is detected in manifest
        """
        # ISO-3 Check: Validate no Python source in manifest
        validate_no_python_source_in_manifest(function_manifest)
        
        # Build ISO-3 compliant prompt
        prompt = build_solver_prompt(
            function_manifest=function_manifest,
            epistemic_summary=epistemic_summary,
            live_omit_branches=live_omit_branches,
            severed_null_signatures=severed_null_signatures,
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
            "candidates": [
                {
                    "steps": [{"function": "probe", "args": {"x": 0, "y": 0}}],
                    "confidence": 0.7
                }
            ]
        })
    
    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response with error handling.
        
        Ref: Spec 3.4 (Response Parsing Requirements)
        """
        try:
            # Try to extract JSON from response (may have markdown wrapping)
            json_str = self._extract_json(raw_response)
            
            # Validate with Pydantic
            validated = SolverResponse.model_validate_json(json_str)
            
            # Convert to dict format for internal use
            response_dict = {
                "candidates": [
                    {
                        "steps": [step.model_dump() for step in candidate.steps],
                        "confidence": candidate.confidence
                    }
                    for candidate in validated.candidates
                ]
            }
            
            self.response_history.append(response_dict)
            
            return response_dict
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Solver JSON response: {e}")
        except ValidationError as e:
            raise ValueError(f"Solver response schema validation failed: {e}")
    
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
