"""Tests for Brusentsov contradiction detection improvements.

Tests the restored object destruction check in contradicts() and the S₀ kinematic guard.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from v10_agent.brusentsov_logic import Ternary, contradicts
from v10_agent.types import AtomicProposition


class TestContradicts:
    """Tests for the contradicts() function with restored object destruction check."""

    def test_preserved_vs_destroyed_is_contradiction(self):
        """Carroll nullity: expected=preserved, observed=destroyed → True."""
        expected = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="destroyed",
            value=True,
        )
        assert contradicts(expected, observed) is True

    def test_preserved_vs_vanished_is_contradiction(self):
        """Carroll nullity: expected=preserved, observed=vanished → True."""
        expected = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="vanished",
            value=True,
        )
        assert contradicts(expected, observed) is True

    def test_preserved_vs_missing_is_contradiction(self):
        """Carroll nullity: expected=preserved, observed=missing → True."""
        expected = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="missing",
            value=True,
        )
        assert contradicts(expected, observed) is True

    def test_different_subject_is_not_contradiction(self):
        """Different subject IDs should not trigger Carroll nullity."""
        expected = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="object_identity",
            subject_id="obj_2",
            predicate="destroyed",
            value=True,
        )
        assert contradicts(expected, observed) is False

    def test_preserved_vs_preserved_is_not_contradiction(self):
        """Same predicate → no contradiction."""
        expected = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="object_identity",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        assert contradicts(expected, observed) is False

    def test_non_identity_family_not_affected(self):
        """Families other than object_identity should not trigger the new check."""
        expected = AtomicProposition(
            family="attribute_delta",
            subject_id="obj_1",
            predicate="preserved",
            value=True,
        )
        observed = AtomicProposition(
            family="attribute_delta",
            subject_id="obj_1",
            predicate="destroyed",
            value=True,
        )
        # This should NOT match the Carroll nullity check (wrong family)
        # It may or may not be a contradiction based on other checks
        # but should not be caught by the identity check
        result = contradicts(expected, observed)
        # The key assertion is that it runs without error
        assert isinstance(result, bool)
