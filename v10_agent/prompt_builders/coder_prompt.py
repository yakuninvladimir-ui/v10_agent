"""
Coder Prompt Builder - ISO-2 Compliant (Coder has no goal information)
Ref: Engineering Specification V10.0 Section 8.2
"""

from typing import Dict, Any, List, Optional
from ..types import EnvironmentSpecification


def build_coder_prompt(
    environment_spec: EnvironmentSpecification,
    api_manifest: Dict[str, Any],
    syntax_error_count: int = 0,
    recent_errors: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build prompt for DSL Coder Agent.
    
    ISO-2 INVARIANT: This prompt MUST explicitly state that Coder has no goal information.
    Coder only sees: EnvironmentSpecification and API contract.
    
    Args:
        environment_spec: Environment specification from Explorer (Ref: Spec 3.2)
        api_manifest: JSON Function Manifest with available DSL functions (Ref: Spec 3.3)
        syntax_error_count: Number of previous syntax errors (for context)
        recent_errors: Optional list of recent error summaries (NO traceback details)
    
    Returns:
        Formatted prompt string for vLLM/Qwen model
    """
    prompt_parts = [
        "=" * 60,
        "ARC-AGI-3 DSL CODER AGENT - FUNCTION IMPLEMENTATION",
        "=" * 60,
        "",
        "CRITICAL CONSTRAINT: You have no information about the level goal.",
        "Your task is to implement DSL functions that match the environment specification.",
        "",
        "-" * 60,
        "ENVIRONMENT SPECIFICATION",
        "-" * 60,
        f"Grid Size: {environment_spec.grid_width}x{environment_spec.grid_height}",
        f"Object Count: {len(environment_spec.object_specs)}",
        f"Relation Count: {len(environment_spec.relation_specs)}",
        f"Action Surface: {environment_spec.action_surface_type}",
        "",
        "Object Types:",
    ]
    
    for obj_spec in environment_spec.object_specs[:5]:  # Limit to first 5
        prompt_parts.append(f"  - {obj_spec.type_id}: {obj_spec.description}")
    
    if len(environment_spec.object_specs) > 5:
        prompt_parts.append(f"  ... and {len(environment_spec.object_specs) - 5} more")
    
    prompt_parts.extend([
        "",
        "Relation Types:",
    ])
    
    for rel_spec in environment_spec.relation_specs[:5]:
        prompt_parts.append(f"  - {rel_spec.type_id}: {rel_spec.description}")
    
    if len(environment_spec.relation_specs) > 5:
        prompt_parts.append(f"  ... and {len(environment_spec.relation_specs) - 5} more")
    
    prompt_parts.extend([
        "",
        "-" * 60,
        "API MANIFEST (Available DSL Functions)",
        "-" * 60,
        "You must implement functions using ONLY these signatures:",
        "",
    ])
    
    for func_name, func_info in api_manifest.get("functions", {}).items():
        prompt_parts.append(f"Function: {func_name}")
        prompt_parts.append(f"  Signature: {func_info.get('signature', 'unknown')}")
        prompt_parts.append(f"  Description: {func_info.get('docstring', 'No description')}")
        prompt_parts.append(f"  Parameters: {func_info.get('parameters', {})}")
        prompt_parts.append(f"  Returns: {func_info.get('return_type', 'unknown')}")
        prompt_parts.append("")
    
    if syntax_error_count > 0 and recent_errors:
        prompt_parts.extend([
            "-" * 60,
            f"PREVIOUS ERRORS ({syntax_error_count} total)",
            "-" * 60,
            "Note: You see error summaries, NOT full tracebacks (ISO-3 compliance).",
            "",
        ])
        for err in recent_errors[-3:]:  # Last 3 errors
            prompt_parts.append(f"- {err.get('summary', 'Unknown error')}")
    
    prompt_parts.extend([
        "",
        "-" * 60,
        "INSTRUCTIONS",
        "-" * 60,
        "1. Implement DSL functions that operate on the given object/relation IDs.",
        "2. Use only the provided API manifest - no external imports.",
        "3. Ensure all function signatures match the manifest exactly.",
        "4. Return Python code in a JSON format:",
        '   {"source_code": "python code string", "function_names": ["list", "of", "names"]}',
        "",
        "REMINDER: You do not know the goal. Implement functions based on environment spec only.",
        "=" * 60,
    ])
    
    return "\n".join(prompt_parts)


def validate_no_goal_in_api_manifest(api_manifest: Dict[str, Any]) -> None:
    """
    Validate that API manifest contains no goal-related information.
    
    Ref: Spec 1.4 ISO-2 Invariant
    """
    goal_keywords = ["goal", "target", "objective", "win", "complete", "finish"]
    
    manifest_str = str(api_manifest).lower()
    for keyword in goal_keywords:
        if keyword in manifest_str:
            raise ValueError(
                f"ISO-2 VIOLATION: Goal keyword '{keyword}' detected in API manifest. "
                "Coder must not know about goals."
            )
