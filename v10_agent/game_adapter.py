"""
Game Adapter Module
Provides the interface to the ARC Arcade or competition environment.
"""

from typing import Dict, Any, Optional

class GameAdapter:
    """
    Adapter for the environment.
    """
    def __init__(self, env: Any = None):
        self._env = env

    def step(self, action_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute step in environment.
        """
        if self._env:
            return self._env.step(action_id, payload)
        # Mock behavior
        return {"grid": [[0]], "action_taken": action_id}

    def reset(self) -> Dict[str, Any]:
        """Reset environment."""
        if self._env:
            return self._env.reset()
        return {"grid": [[0]]}
