"""Unit tests for ActionAdapter spatial sanitization and conversion invariants."""

from __future__ import annotations

import pytest
from v10_agent.action_adapter import to_native_action, arcade_step_args, _sanitize_spatial_data
from v10_agent.types import ActionDeclaration

try:
    from arcengine import ActionInput, GameAction
except Exception:
    ActionInput = None
    GameAction = None


def test_sanitize_spatial_data_defaults_missing_coordinates():
    """Verify missing coordinates in ACTION6 default to integer 0, preventing KeyError in engine step."""
    clean = _sanitize_spatial_data("ACTION6", {})
    assert clean == {"x": 0, "y": 0}
    assert isinstance(clean["x"], int)
    assert isinstance(clean["y"], int)


def test_sanitize_spatial_data_converts_none_and_strings():
    """Verify None and string coordinate values are safely cast to integers."""
    clean = _sanitize_spatial_data("ACTION6", {"x": None, "y": "15"})
    assert clean == {"x": 0, "y": 15}
    assert isinstance(clean["x"], int)
    assert isinstance(clean["y"], int)


def test_sanitize_spatial_data_ignores_discrete_actions():
    """Verify discrete actions do not have synthetic coordinates injected."""
    clean = _sanitize_spatial_data("ACTION1", {})
    assert clean == {}


def test_to_native_action_sanitizes_action6():
    """Verify to_native_action guarantees x and y for ACTION6 across declarations."""
    decl = ActionDeclaration(action_id="ACTION6", data={}, reasoning={})
    native = to_native_action(decl)
    if ActionInput is not None:
        assert isinstance(native, ActionInput)
        assert native.data.get("x") == 0
        assert native.data.get("y") == 0
    else:
        assert isinstance(native, dict)
        assert native["data"].get("x") == 0
        assert native["data"].get("y") == 0


def test_arcade_step_args_sanitizes_action6_tuple():
    """Verify arcade_step_args guarantees integer x and y in the extracted tuple."""
    mapping_act = {"id": "ACTION6", "data": {}}
    act_id, data, reasoning = arcade_step_args(mapping_act)
    assert data.get("x") == 0
    assert data.get("y") == 0
    assert isinstance(data["x"], int)
    assert isinstance(data["y"], int)

    if ActionInput is not None and GameAction is not None:
        input_act = ActionInput(id=GameAction.ACTION6, data={})
        act_id2, data2, _ = arcade_step_args(input_act)
        assert data2.get("x") == 0
        assert data2.get("y") == 0
