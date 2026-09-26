"""Action adapter translating between V10 ActionDeclaration and native Arcade/ArcEngine actions."""

from __future__ import annotations

from typing import Any, Mapping

from v10_agent.action_semantics import (
    COORDINATE_ACTION_ID,
    is_forbidden_action,
    normalize_action_id,
)
from v10_agent.types import ActionDeclaration

try:
    from arcengine import ActionInput, GameAction
except Exception:  # Local execution or testing environments where arcengine is omitted
    ActionInput = None
    GameAction = None


class ForbiddenActionError(ValueError):
    """Raised when an action the competition forbids would reach the engine."""


def game_action_by_name(value: Any) -> Any:
    """Resolve an action name or int to arcengine.GameAction if available, or return canonical name."""
    name = normalize_action_id(value)
    if GameAction is None:
        return name

    members = getattr(GameAction, "__members__", {}) or {}
    if name in members:
        return members[name]
    if hasattr(GameAction, name):
        return getattr(GameAction, name)
    if hasattr(GameAction, "from_name"):
        try:
            return GameAction.from_name(name)
        except Exception:
            pass
    if name.startswith("ACTION") and name.removeprefix("ACTION").isdigit() and hasattr(GameAction, "from_id"):
        try:
            return GameAction.from_id(int(name.removeprefix("ACTION")))
        except Exception:
            pass
    if name == "RESET" and hasattr(GameAction, "from_id"):
        try:
            return GameAction.from_id(0)
        except Exception:
            pass
    return name


def _sanitize_spatial_data(action_name: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Ensure spatial actions (ACTION6) always provide integer x and y coordinates.

    Environments implementing spatial actions access data['x'] and data['y']
    directly and crash with KeyError if omitted.
    """
    clean_name = normalize_action_id(action_name)
    if clean_name == COORDINATE_ACTION_ID:
        x_val = data.get("x")
        y_val = data.get("y")
        try:
            data["x"] = int(0 if x_val is None else x_val)
        except (TypeError, ValueError):
            data["x"] = 0
        try:
            data["y"] = int(0 if y_val is None else y_val)
        except (TypeError, ValueError):
            data["y"] = 0
    return data


def to_native_action(action: ActionDeclaration | Mapping[str, Any]) -> Any:
    """Convert ActionDeclaration to native ActionInput (if arcengine installed) or dict.

    The competition forbids the Undo channel; an action reaching this boundary is
    rejected here so it can never be handed to the engine.
    """
    if isinstance(action, ActionDeclaration):
        action_id = action.action_id
        data = dict(action.data)
        reasoning = dict(action.reasoning)
    else:
        action_id = action.get("id", action.get("action_id", "ACTION1"))
        data = dict(action.get("data", {}) or {})
        reasoning = dict(action.get("reasoning", {}) or {})

    if is_forbidden_action(action_id):
        raise ForbiddenActionError(
            f"Refusing to dispatch {normalize_action_id(action_id)}: the Undo channel is "
            f"hardware-blocked by the competition contract."
        )

    data = _sanitize_spatial_data(action_id, data)
    game_action = game_action_by_name(action_id)
    if ActionInput is not None:
        return ActionInput(id=game_action, data=data, reasoning=reasoning)
    return {
        "id": normalize_action_id(game_action),
        "data": data,
        "reasoning": reasoning,
    }


def from_native_action(native: Any) -> ActionDeclaration:
    """Extract ActionDeclaration from native ActionInput, GameAction, or dict."""
    if isinstance(native, ActionDeclaration):
        return native
    action_id, data, reasoning = arcade_step_args(native)
    if is_forbidden_action(action_id):
        raise ForbiddenActionError(f"Environment produced forbidden action {action_id!r}.")
    return ActionDeclaration(
        action_id=normalize_action_id(action_id),
        data=data,
        reasoning=reasoning,
    )


def arcade_step_args(native_action: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Extract (action_id, data, reasoning) tuple suitable for env.step().

    Raises ForbiddenActionError when the resolved action is the blocked Undo
    channel, keeping the engine boundary free of illegal actions.
    """
    if ActionInput is not None and isinstance(native_action, ActionInput):
        data = dict(getattr(native_action, "data", {}) or {})
        action_name = getattr(native_action.id, "name", str(native_action.id))
        if is_forbidden_action(action_name):
            raise ForbiddenActionError(f"Refusing to dispatch {action_name} from ActionInput.")
        data = _sanitize_spatial_data(action_name, data)
        return (
            native_action.id,
            data,
            dict(getattr(native_action, "reasoning", {}) or {}),
        )
    if isinstance(native_action, Mapping):
        action_id = native_action.get("id", native_action.get("action_id", "ACTION1"))
        if is_forbidden_action(action_id):
            raise ForbiddenActionError(f"Refusing to dispatch {action_id} from mapping.")
        data = dict(native_action.get("data", {}) or {})
        data = _sanitize_spatial_data(action_id, data)
        reasoning = dict(native_action.get("reasoning", {}) or {})
        return game_action_by_name(action_id), data, reasoning

    action_id = getattr(native_action, "id", native_action)
    if is_forbidden_action(action_id):
        raise ForbiddenActionError(f"Refusing to dispatch {action_id}.")
    data = dict(getattr(native_action, "data", {}) or {})
    data = _sanitize_spatial_data(action_id, data)
    reasoning = dict(getattr(native_action, "reasoning", {}) or {})
    return game_action_by_name(action_id), data, reasoning
