"""
Action Boundary Module
Ref: Engineering Specification V10.0 Section 1.1, 5, 7.2

ActionBoundary is the ONLY component allowed to call the real environment.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ActionBoundary:
    """
    Final gateway to the environment.
    Executes actions only after Verifier approval.
    """

    def __init__(self, environment_adapter: Any = None):
        """
        Args:
            environment_adapter: The actual game/environment client.
        """
        self.env = environment_adapter

    def execute_action(self, action_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute an action in the real environment.

        Args:
            action_id: The ID of the action to execute.
            payload: Arguments for the action.

        Returns:
            The raw observation from the environment after the action.
        """
        logger.info(f"ActionBoundary executing real environment action: {action_id} with args {payload}")

        if self.env is None:
            # Stub mode
            logger.warning("ActionBoundary has no environment adapter, returning mock observation.")
            return {"grid": [[0]], "mock": True, "action_taken": action_id}

        try:
            # Assuming env has a step method (to be formalized in game_adapter)
            return self.env.step(action_id, payload)
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            raise
