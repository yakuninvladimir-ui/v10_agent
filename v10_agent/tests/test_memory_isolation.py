"""
Unit tests for Memory Contours Isolation.

Ref: Spec 1.4 (Isolation Invariants), ISO-3 Compliance

This test suite verifies that the three memory contours are strictly isolated
and do not contain references to each other's data.
"""

import pytest
from typing import Any, get_type_hints

from v10_agent.memory_contours import (
    EnvironmentSpecMemory,
    SyntaxErrorMemory,
    EpistemicMemory,
    GameMemory,
    EnvironmentSpecification,
)
from v10_agent.types import (
    PropositionSet,
    ProbeRecord,
    SyntaxErrorRecord,
    BrusentsovJudgment,
    BranchSignature,
)


class TestIsolationMemoryContoursAreDisjoint:
    """
    Test suite to verify ISO-3 compliance: memory contours must be disjoint.
    
    This test ensures that:
    1. No contour class has attributes referencing another contour's data types
    2. No contour class has methods that could access another contour's contents
    3. Each contour is completely self-contained
    """
    
    def test_isolation_memory_contours_are_disjoint(self) -> None:
        """
        Verify that memory contour classes do not contain references to each other's data.
        
        This test checks:
        - Type hints of all attributes
        - Method signatures
        - Internal field names
        
        Ref: Spec 1.4 - ISO-3 Invariant (Strict Separation)
        """
        # Define the expected data types for each contour
        env_spec_types = {
            'EnvironmentSpecification',
            'ProbeRecord',
            'deque',  # For internal probe history
            'Sequence',  # Return type
        }
        
        syntax_error_types = {
            'SyntaxErrorRecord',
            'list',  # For internal error storage
            'Sequence',  # Return type
        }
        
        epistemic_types = {
            'BrusentsovJudgment',
            'BranchSignature',
            'set',  # For internal branch storage
            'frozenset',  # Return type
            'Sequence',  # Return type
        }
        
        game_memory_types = {
            'dict',
            'list',
            'float',
            'Any',
        }
        
        # Check EnvironmentSpecMemory
        self._verify_contour_isolation(
            EnvironmentSpecMemory,
            allowed_types=env_spec_types,
            forbidden_types={
                'SyntaxErrorRecord',
                'BrusentsovJudgment',
                'BranchSignature',
            },
            contour_name="EnvironmentSpecMemory",
        )
        
        # Check SyntaxErrorMemory
        self._verify_contour_isolation(
            SyntaxErrorMemory,
            allowed_types=syntax_error_types,
            forbidden_types={
                'ProbeRecord',
                'EnvironmentSpecification',
                'BrusentsovJudgment',
                'BranchSignature',
            },
            contour_name="SyntaxErrorMemory",
        )
        
        # Check EpistemicMemory
        self._verify_contour_isolation(
            EpistemicMemory,
            allowed_types=epistemic_types,
            forbidden_types={
                'ProbeRecord',
                'EnvironmentSpecification',
                'SyntaxErrorRecord',
            },
            contour_name="EpistemicMemory",
        )
        
        # Check GameMemory (separate from isolated contours)
        self._verify_contour_isolation(
            GameMemory,
            allowed_types=game_memory_types,
            forbidden_types={
                'ProbeRecord',
                'SyntaxErrorRecord',
                'BrusentsovJudgment',
            },
            contour_name="GameMemory",
        )
    
    def _verify_contour_isolation(
        self,
        contour_class: type,
        allowed_types: set[str],
        forbidden_types: set[str],
        contour_name: str,
    ) -> None:
        """
        Helper method to verify a contour class's isolation.
        
        Args:
            contour_class: The memory contour class to check
            allowed_types: Set of type names that are allowed in this contour
            forbidden_types: Set of type names that must NOT appear in this contour
            contour_name: Human-readable name for error messages
        """
        # Get all type hints from the class
        try:
            type_hints = get_type_hints(contour_class)
        except Exception:
            # If we can't get type hints, check __annotations__ directly
            type_hints = getattr(contour_class, '__annotations__', {})
        
        # Also check __init__ signature
        import inspect
        init_sig = inspect.signature(contour_class.__init__)
        
        # Collect all type references from annotations
        all_type_refs: set[str] = set()
        
        for param_name, param in init_sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                all_type_refs.add(self._extract_type_name(param.annotation))
        
        # Check field annotations
        for field_name, field_type in type_hints.items():
            all_type_refs.add(self._extract_type_name(field_type))
        
        # Check that no forbidden types are present
        violations = all_type_refs.intersection(forbidden_types)
        
        assert len(violations) == 0, (
            f"{contour_name} violates ISO-3 isolation: "
            f"contains references to forbidden types: {violations}. "
            f"All type references found: {all_type_refs}"
        )
    
    def _extract_type_name(self, type_annotation: Any) -> str:
        """
        Extract the base type name from a type annotation.
        
        Args:
            type_annotation: A type annotation (may be generic)
            
        Returns:
            The base type name as a string
        """
        import typing
        
        # Handle None
        if type_annotation is type(None):
            return 'None'
        
        # Handle basic types
        if isinstance(type_annotation, type):
            return type_annotation.__name__
        
        # Handle string annotations
        if isinstance(type_annotation, str):
            return type_annotation.split('[')[0]
        
        # Handle generics (e.g., list[int], dict[str, int])
        origin = typing.get_origin(type_annotation)
        if origin is not None:
            return origin.__name__ if hasattr(origin, '__name__') else str(origin)
        
        # Handle Union types
        if hasattr(type_annotation, '__origin__'):
            return str(type_annotation.__origin__)
        
        # Fallback: convert to string and extract base name
        type_str = str(type_annotation)
        if '[' in type_str:
            return type_str.split('[')[0]
        
        return type_str
    
    def test_no_cross_contour_methods(self) -> None:
        """
        Verify that no contour class has methods that reference other contours.
        
        This test checks method names and docstrings for any indication of
        cross-contour access.
        """
        # Methods that would indicate cross-contour access
        suspicious_patterns = [
            'get_syntax_errors',  # Should not be in EnvironmentSpecMemory
            'get_probes',  # Should not be in SyntaxErrorMemory
            'get_judgments',  # Should not be in EnvironmentSpecMemory or SyntaxErrorMemory
            'add_syntax_error',  # Should only be in SyntaxErrorMemory
            'add_probe',  # Should only be in EnvironmentSpecMemory
            'add_judgment',  # Should only be in EpistemicMemory
        ]
        
        # Define which methods are valid for each contour
        # Note: 'clear' is a common pattern but operates on different data in each contour
        valid_env_methods = {'add_probe', 'get_probe_history', 'get_recent_probes', 
                            'set_specification', 'clear_probes'}
        valid_syntax_methods = {'add_error', 'get_errors', 'get_recent_errors', 'clear'}
        valid_epistemic_methods = {'add_judgment', 'get_judgments', 'get_live_omit_branches',
                                   'get_severed_null_signatures', 'is_branch_omittable',
                                   'is_branch_severed', 'prune_omit_branch', 'clear'}
        valid_game_methods = {'add_level_summary', 'get_level_summary', 'update_strategy_metric',
                             'get_strategy_metric', 'add_global_constraint', 'get_all_summaries', 'clear'}
        
        # Check EnvironmentSpecMemory - exclude generic methods like 'clear' that are common patterns
        env_methods = {m for m in dir(EnvironmentSpecMemory) if not m.startswith('_')}
        # 'clear' is allowed as it's a common Python pattern (like dict.clear())
        invalid_env = (env_methods - {'clear'}).intersection(valid_syntax_methods | valid_epistemic_methods)
        assert len(invalid_env) == 0, (
            f"EnvironmentSpecMemory has methods from other contours: {invalid_env}"
        )
        
        # Check SyntaxErrorMemory
        syntax_methods = {m for m in dir(SyntaxErrorMemory) if not m.startswith('_')}
        # 'clear' is allowed as it's a common Python pattern
        invalid_syntax = (syntax_methods - {'clear'}).intersection(valid_env_methods | valid_epistemic_methods)
        assert len(invalid_syntax) == 0, (
            f"SyntaxErrorMemory has methods from other contours: {invalid_syntax}"
        )
        
        # Check EpistemicMemory
        epistemic_methods = {m for m in dir(EpistemicMemory) if not m.startswith('_')}
        # 'clear' is allowed as it's a common Python pattern
        invalid_epistemic = (epistemic_methods - {'clear'}).intersection(valid_env_methods | valid_syntax_methods)
        assert len(invalid_epistemic) == 0, (
            f"EpistemicMemory has methods from other contours: {invalid_epistemic}"
        )
    
    def test_contours_can_be_instantiated_independently(self) -> None:
        """
        Verify that each contour can be instantiated without dependencies on others.
        
        This ensures there are no hidden coupling through constructors.
        """
        # Each should instantiate without requiring other contours
        env_memory = EnvironmentSpecMemory()
        syntax_memory = SyntaxErrorMemory()
        epistemic_memory = EpistemicMemory()
        game_memory = GameMemory()
        
        # Verify they are distinct objects
        assert env_memory is not syntax_memory
        assert env_memory is not epistemic_memory
        assert env_memory is not game_memory
        assert syntax_memory is not epistemic_memory
        assert syntax_memory is not game_memory
        assert epistemic_memory is not game_memory
        
        # Verify they have no shared mutable state
        assert id(env_memory._probe_history) != id(syntax_memory._errors)
        assert id(env_memory._probe_history) != id(epistemic_memory._judgments)
        assert id(syntax_memory._errors) != id(epistemic_memory._judgments)


class TestEnvironmentSpecMemory:
    """Tests for EnvironmentSpecMemory functionality."""
    
    def test_add_probe_requires_spec(self) -> None:
        """Test that adding a probe requires a specification to be set."""
        memory = EnvironmentSpecMemory()
        
        probe = ProbeRecord(
            action_id="test_action",
            pre_state=PropositionSet(propositions=frozenset()),
            post_state=PropositionSet(propositions=frozenset()),
            effect=EffectDeclaration(
                function_name="test_func",
                arguments={},
                result_type="test_result",
            ),
            confidence=0.9,
        )
        
        with pytest.raises(ValueError, match="Cannot add probe without EnvironmentSpecification"):
            memory.add_probe(probe)
    
    def test_probe_history_fifo_limit(self) -> None:
        """Test that probe history respects the FIFO limit."""
        # Create a spec first
        spec = EnvironmentSpecification(
            spec_id="test_spec",
            initial_propositions=PropositionSet(propositions=frozenset()),
        )
        
        memory = EnvironmentSpecMemory()
        memory.set_specification(spec)
        
        # Add more than 1000 probes (the limit)
        for i in range(1005):
            probe = ProbeRecord(
                action_id=f"action_{i}",
                pre_state=PropositionSet(propositions=frozenset()),
                post_state=PropositionSet(propositions=frozenset()),
                effect=EffectDeclaration(
                    function_name="test_func",
                    arguments={},
                    result_type="test_result",
                ),
                confidence=0.9,
            )
            memory.add_probe(probe)
        
        # Should have exactly 1000 probes (maxlen)
        assert memory.probe_count == 1000
        
        # First probe should be the 6th one added (indices 5-1004 remain)
        history = memory.get_probe_history()
        assert history[0].action_id == "action_5"


class TestSyntaxErrorMemory:
    """Tests for SyntaxErrorMemory functionality."""
    
    def test_max_capacity_enforced(self) -> None:
        """Test that maximum capacity of 5 errors is enforced."""
        memory = SyntaxErrorMemory()
        
        # Add 7 errors
        for i in range(7):
            memory.add_error(
                prompt_hash=f"prompt_{i}",
                source_hash=f"source_{i}",
                traceback=f"traceback_{i}",
            )
        
        # Should have exactly 5 errors (max capacity)
        assert memory.error_count == 5
        
        # First two errors should be removed (FIFO)
        errors = memory.get_errors()
        assert errors[0].prompt_hash == "prompt_2"
        assert errors[-1].prompt_hash == "prompt_6"
    
    def test_has_capacity(self) -> None:
        """Test has_capacity method."""
        memory = SyntaxErrorMemory()
        
        assert memory.has_capacity() is True
        
        # Fill to capacity
        for i in range(5):
            memory.add_error(
                prompt_hash=f"prompt_{i}",
                source_hash=f"source_{i}",
                traceback=f"traceback_{i}",
            )
        
        assert memory.has_capacity() is False


class TestEpistemicMemory:
    """Tests for EpistemicMemory functionality."""
    
    def test_judgment_updates_branches(self) -> None:
        """Test that adding judgments updates live/severed branches correctly."""
        memory = EpistemicMemory()
        
        # Add an OMIT judgment
        omit_judgment = BrusentsovJudgment(
            judgment_type="OMIT",
            branch_signature=BranchSignature(
                action_sequence=("action1", "action2"),
                outcome_hash="omit_hash_123",
            ),
            reasoning="Effect is inessential",
            timestamp=1234567890.0,
        )
        
        memory.add_judgment(omit_judgment)
        
        assert memory.is_branch_omittable(omit_judgment.branch_signature) is True
        assert memory.is_branch_severed(omit_judgment.branch_signature) is False
        assert memory.live_omit_count == 1
        
        # Add a NULL judgment
        null_judgment = BrusentsovJudgment(
            judgment_type="NULL",
            branch_signature=BranchSignature(
                action_sequence=("action3", "action4"),
                outcome_hash="null_hash_456",
            ),
            reasoning="Physical contradiction detected",
            timestamp=1234567891.0,
        )
        
        memory.add_judgment(null_judgment)
        
        assert memory.is_branch_severed(null_judgment.branch_signature) is True
        assert memory.is_branch_omittable(null_judgment.branch_signature) is False
        assert memory.severed_null_count == 1
        assert memory.live_omit_count == 1  # Still 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
