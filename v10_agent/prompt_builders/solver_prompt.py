"""
Solver Prompt Builder - ISO-3 Compliant (Solver never sees Python source)
Ref: Engineering Specification V10.0 Section 8.3
"""

from typing import Dict, Any, List, Optional
from ..types import BrusentsovJudgment, BranchSignature


def build_solver_prompt(
    function_manifest: Dict[str, Any],
    epistemic_summary: Optional[Dict[str, Any]] = None,
    live_omit_branches: Optional[List[BranchSignature]] = None,
    severed_null_signatures: Optional[List[BranchSignature]] = None,
) -> str:
    """
    Build prompt for Solver Agent.
    
    ISO-3 INVARIANT: This prompt MUST explicitly state that Solver never sees Python source.
    Solver only sees: JSON Function Manifest, docstrings, and EpistemicMemory summary.
    
    Args:
        function_manifest: JSON Function Manifest with DSL functions (Ref: Spec 3.4)
        epistemic_summary: Summary of previous Brusentsov judgments (Ref: Spec 3.5.3)
        live_omit_branches: Branches marked as OMIT (can be continued) (Ref: Spec 5.2)
        severed_null_signatures: Branches marked as NULL (contradicted, severed) (Ref: Spec 5.2)
    
    Returns:
        Formatted prompt string for vLLM/Qwen model
    """
    prompt_parts = [
        "=" * 60,
        "ARC-AGI-3 SOLVER AGENT - TRAJECTORY PLANNING",
        "=" * 60,
        "",
        "CRITICAL CONSTRAINT: You never see Python source code.",
        "You only see function manifests (JSON) and their docstrings.",
        "Your task is to compose function calls into trajectory candidates.",
        "",
        "-" * 60,
        "FUNCTION MANIFEST (Available DSL Functions)",
        "-" * 60,
        "Use these functions to build action sequences:",
        "",
    ]
    
    for func_name, func_info in function_manifest.get("functions", {}).items():
        prompt_parts.append(f"Function: {func_name}")
        prompt_parts.append(f"  Signature: {func_info.get('signature', 'unknown')}")
        prompt_parts.append(f"  Description: {func_info.get('docstring', 'No description')}")
        prompt_parts.append(f"  Parameters: {func_info.get('parameters', {})}")
        prompt_parts.append(f"  Returns: {func_info.get('return_type', 'unknown')}")
        prompt_parts.append("")
    
    if epistemic_summary:
        prompt_parts.extend([
            "-" * 60,
            "EPISTEMIC MEMORY SUMMARY",
            "-" * 60,
            f"Total Judgments: {epistemic_summary.get('total_judgments', 0)}",
            f"FOLLOW Count: {epistemic_summary.get('follow_count', 0)}",
            f"NULL Count: {epistemic_summary.get('null_count', 0)}",
            f"OMIT Count: {epistemic_summary.get('omit_count', 0)}",
            "",
        ])
    
    if live_omit_branches:
        prompt_parts.extend([
            "-" * 60,
            f"LIVE OMIT BRANCHES ({len(live_omit_branches)} active)",
            "-" * 60,
            "These branches had missing effects but no contradictions.",
            "Consider continuing or completing these trajectories:",
            "",
        ])
        for branch in live_omit_branches[:5]:  # Limit to first 5
            prompt_parts.append(f"- Branch {branch.branch_id}: signature={branch.signature_hash[:16]}...")
        
        if len(live_omit_branches) > 5:
            prompt_parts.append(f"... and {len(live_omit_branches) - 5} more")
    
    if severed_null_signatures:
        prompt_parts.extend([
            "",
            "-" * 60,
            f"SEVERED NULL BRANCHES ({len(severed_null_signatures)} contradicted)",
            "-" * 60,
            "DO NOT use these branch signatures - they led to contradictions:",
            "",
        ])
        for sig in severed_null_signatures[:5]:
            prompt_parts.append(f"- {sig.signature_hash[:32]}...")
        
        if len(severed_null_signatures) > 5:
            prompt_parts.append(f"... and {len(severed_null_signatures) - 5} more")
    
    prompt_parts.extend([
        "",
        "-" * 60,
        "INSTRUCTIONS",
        "-" * 60,
        "1. Compose function calls from the manifest into trajectory candidates.",
        "2. Each candidate should be a sequence of 1-6 function calls (Ref: Spec 2.1).",
        "3. Prioritize branches that extend live OMIT trajectories.",
        "4. Avoid any function call sequences that match severed NULL signatures.",
        "5. Return JSON with format:",
        '   {"candidates": [{"steps": [{"function": "name", "args": {...}}], "confidence": float}]}',
        "",
        "REMINDER: You never see Python source. Work only with function manifests.",
        "=" * 60,
    ])
    
    return "\n".join(prompt_parts)


def validate_no_python_source_in_manifest(function_manifest: Dict[str, Any]) -> None:
    """
    Validate that function manifest contains no Python source code.
    
    Ref: Spec 1.4 ISO-3 Invariant
    """
    python_indicators = ["def ", "import ", "class ", "return ", "lambda ", ":"]
    
    manifest_str = str(function_manifest)
    for indicator in python_indicators:
        if indicator in manifest_str:
            raise ValueError(
                f"ISO-3 VIOLATION: Python source indicator '{indicator}' detected in manifest. "
                "Solver must never see Python source code."
            )
