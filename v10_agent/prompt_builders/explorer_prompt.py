"""
Explorer Prompt Builder - ISO-1 Compliant (Explorer does not know goals)
Ref: Engineering Specification V10.0 Section 8.1
"""

from typing import List, Dict, Any, Optional
from ..types import ProbeRecord
from ..planning_set import PlanningSet


def build_explorer_prompt(
    planning_set: PlanningSet,
    annotated_frame: str,
    action_history: List[Dict[str, Any]],
    probe_history: Optional[List[ProbeRecord]] = None,
) -> str:
    """
    Build prompt for Explorer Agent.
    
    ISO-1 INVARIANT: This prompt MUST NOT contain any information about level goals.
    Explorer only sees: PlanningSet, annotated frame, and action history.
    
    Args:
        planning_set: Current PlanningSet snapshot (Ref: Spec 4)
        annotated_frame: PNG frame with object/relation annotations (Ref: Spec 7.1)
        action_history: List of previous actions taken
        probe_history: Optional history of probe results
    
    Returns:
        Formatted prompt string for vLLM/Qwen model
    
    Raises:
        ValueError: If goal-related content is detected in inputs
    """
    # ISO-1 Check: Ensure no goal leakage
    _validate_no_goal_leakage(planning_set, action_history)
    
    prompt_parts = [
        "=" * 60,
        "ARC-AGI-3 EXPLORER AGENT - ENVIRONMENT DISCOVERY",
        "=" * 60,
        "",
        "TASK: Explore the environment and identify objects, relations, and affordances.",
        "CONSTRAINT: You have NO information about the level goal.",
        "OUTPUT: Return a JSON object with probe recommendations.",
        "",
        "-" * 60,
        "CURRENT STATE (PlanningSet Snapshot)",
        "-" * 60,
        f"Snapshot ID: {planning_set.snapshot_id}",
        f"Grid Hash: {planning_set.grid_hash[:16]}...",
        f"Object Count: {len(planning_set.object_ids)}",
        f"Relation Count: {len(planning_set.relation_ids)}",
        f"Allowed Actions: {len(planning_set.allowed_action_ids)}",
        "",
        "-" * 60,
        "ANNOTATED FRAME",
        "-" * 60,
        annotated_frame,
        "",
        "-" * 60,
        "ACTION HISTORY",
        "-" * 60,
    ]
    
    if action_history:
        for i, action in enumerate(action_history[-10:], 1):  # Last 10 actions
            prompt_parts.append(f"{i}. {action.get('action_id', 'unknown')}")
    else:
        prompt_parts.append("No previous actions.")
    
    if probe_history:
        prompt_parts.extend([
            "",
            "-" * 60,
            "PROBE HISTORY",
            "-" * 60,
        ])
        for probe in probe_history[-5:]:  # Last 5 probes
            prompt_parts.append(f"- Cell ({probe.cell_x}, {probe.cell_y}): confidence={probe.confidence:.2f}")
    
    prompt_parts.extend([
        "",
        "-" * 60,
        "INSTRUCTIONS",
        "-" * 60,
        "1. Analyze the annotated frame for unrecognized objects or relations.",
        "2. Recommend probe actions to discover new affordances.",
        "3. Return JSON with format:",
        '   {"probes": [{"x": int, "y": int, "confidence": float}], "reasoning": "string"}',
        "",
        "REMINDER: You do not know the goal. Focus on environment discovery only.",
        "=" * 60,
    ])
    
    return "\n".join(prompt_parts)


def _validate_no_goal_leakage(
    planning_set: PlanningSet,
    action_history: List[Dict[str, Any]],
) -> None:
    """
    Validate that no goal-related content has leaked into Explorer context.
    
    Ref: Spec 1.4 ISO-1 Invariant
    """
    goal_keywords = ["goal", "target", "objective", "win", "complete", "finish"]
    
    # Check action history for goal references
    for action in action_history:
        action_str = str(action).lower()
        for keyword in goal_keywords:
            if keyword in action_str:
                raise ValueError(
                    f"ISO-1 VIOLATION: Goal keyword '{keyword}' detected in action history. "
                    "Explorer must not know about goals."
                )
    
    # Note: PlanningSet by design does not contain goal information (Ref: Spec 4)
