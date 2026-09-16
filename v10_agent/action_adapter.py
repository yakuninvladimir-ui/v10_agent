"""Action adapter translating between V10 ActionDeclaration and native Arcade/ArcEngine actions."""

from __future__ import annotations

from typing import Any, Mapping

from v10_agent.observe import normalize_action_name
from v10_agent.types import ActionDeclaration

try:
    from arcengine import ActionInput, GameAction
except Exception:  # Local execution or testing environments where arcengine is omitted
    ActionInput = None
    GameAction = None


def game_action_by_name(value: Any) -> Any:
    """Resolve an action name or int to arcengine.GameAction if available, or return canonical name."""
    name = normalize_action_name(value)
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


def to_native_action(action: ActionDeclaration | Mapping[str, Any]) -> Any:
    """Convert ActionDeclaration to native ActionInput (if arcengine installed) or dict."""
    if isinstance(action, ActionDeclaration):
        action_id = action.action_id
        data = dict(action.data)
        reasoning = dict(action.reasoning)
    else:
        action_id = action.get("id", action.get("action_id", "ACTION1"))
        data = dict(action.get("data", {}) or {})
        reasoning = dict(action.get("reasoning", {}) or {})

    game_action = game_action_by_name(action_id)
    if ActionInput is not None:
        return ActionInput(id=game_action, data=data, reasoning=reasoning)
    return {
        "id": normalize_action_name(game_action),
        "data": data,
        "reasoning": reasoning,
    }


def from_native_action(native: Any) -> ActionDeclaration:
    """Extract ActionDeclaration from native ActionInput, GameAction, or dict."""
    if isinstance(native, ActionDeclaration):
        return native
    action_id, data, reasoning = arcade_step_args(native)
    return ActionDeclaration(
        action_id=normalize_action_name(action_id),
        data=data,
        reasoning=reasoning,
    )


def arcade_step_args(native_action: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Extract (action_id, data, reasoning) tuple suitable for env.step()."""
    if ActionInput is not None and isinstance(native_action, ActionInput):
        return (
            native_action.id,
            dict(getattr(native_action, "data", {}) or {}),
            dict(getattr(native_action, "reasoning", {}) or {}),
        )
    if isinstance(native_action, Mapping):
        action_id = native_action.get("id", native_action.get("action_id", "ACTION1"))
        data = dict(native_action.get("data", {}) or {})
        reasoning = dict(native_action.get("reasoning", {}) or {})
        return game_action_by_name(action_id), data, reasoning

    action_id = getattr(native_action, "id", native_action)
    data = dict(getattr(native_action, "data", {}) or {})
    reasoning = dict(getattr(native_action, "reasoning", {}) or {})
    return game_action_by_name(action_id), data, reasoning
