"""
Action Adapter Module
Transforms agent actions into environment-specific action schemas.
"""
from typing import Dict, Any

def adapt_action(agent_action: Dict[str, Any]) -> Dict[str, Any]:
    return agent_action
