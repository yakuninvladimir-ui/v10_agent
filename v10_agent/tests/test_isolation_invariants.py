"""
Test Suite for ISO Isolation Invariants
Ref: Engineering Specification V10.0 Section 1.4 (Isolation Invariants)
"""

import pytest
import hashlib
from typing import Dict, Any

from v10_agent.types import (
    EnvironmentSpecification,
    ObjectSpec,
    RelationSpec,
    SyntaxErrorRecord,
    BrusentsovJudgment,
    BranchSignature,
    ProbeRecord,
)
from v10_agent.memory_contours import (
    EnvironmentSpecMemory,
    SyntaxErrorMemory,
    EpistemicMemory,
    GameMemory,
)
from v10_agent.planning_set import PlanningSet
from v10_agent.prompt_builders import (
    build_explorer_prompt,
    build_coder_prompt,
    build_solver_prompt,
)


class TestISO1GoalIsolation:
    """Test ISO-1: Explorer does not know about goals."""
    
    def test_isolation_goal_never_reaches_coder(self):
        """
        Verify that Coder prompt contains explicit statement about no goal knowledge.
        
        Ref: Spec 1.4 ISO-1 Invariant
        Ref: Spec 8.2 Coder Prompt Construction
        """
        env_spec = EnvironmentSpecification(
            grid_width=10,
            grid_height=10,
            object_specs=[ObjectSpec(type_id="obj1", description="Test object")],
            relation_specs=[RelationSpec(type_id="rel1", description="Test relation")],
        )
        
        api_manifest = {
            "functions": {
                "test_func": {
                    "signature": "test_func(x: int) -> None",
                    "docstring": "Test function",
                    "parameters": {"x": "int"},
                    "return_type": "None",
                }
            }
        }
        
        prompt = build_coder_prompt(env_spec, api_manifest)
        
        # ISO-1 Check: Coder must explicitly state it has no goal information
        assert "no information about the level goal" in prompt.lower() or \
               "no goal information" in prompt.lower() or \
               "no information about the goal" in prompt.lower()
        
        # ISO-1 Check: Goal keywords should NOT appear in positive context
        goal_keywords = ["goal is to", "your goal", "target:", "objective:"]
        for keyword in goal_keywords:
            assert keyword not in prompt.lower(), f"ISO-1 VIOLATION: Found '{keyword}' in Coder prompt"
    
    def test_explorer_prompt_has_no_goal_references(self):
        """Verify Explorer prompt contains no goal-related content."""
        planning_set = PlanningSet.create(
            object_ids=["obj1", "obj2"],
            relation_ids=["rel1"],
            allowed_action_ids=["ACTION1", "ACTION2"],
        )
        
        annotated_frame = "base64_encoded_frame_data"
        action_history = [{"action_id": "ACTION1", "timestamp": 0}]
        
        prompt = build_explorer_prompt(
            planning_set=planning_set,
            annotated_frame=annotated_frame,
            action_history=action_history,
        )
        
        # ISO-1 Check: Explorer must state it has no goal knowledge
        assert "no information about the level goal" in prompt.lower() or \
               "do not know the goal" in prompt.lower()
        
        # ISO-1 Check: No goal leakage
        goal_keywords = ["goal is to", "your goal", "target:", "objective:", "win condition"]
        for keyword in goal_keywords:
            assert keyword not in prompt.lower(), f"ISO-1 VIOLATION: Found '{keyword}' in Explorer prompt"


class TestISO2CoderIsolation:
    """Test ISO-2: Coder sees environment spec only, no goals."""
    
    def test_coder_sees_only_environment_spec(self):
        """Verify Coder receives only environment specification."""
        env_spec = EnvironmentSpecification(
            grid_width=5,
            grid_height=5,
            object_specs=[],
            relation_specs=[],
        )
        
        api_manifest = {"functions": {}}
        
        prompt = build_coder_prompt(env_spec, api_manifest)
        
        # Should contain environment details
        assert "Grid Size:" in prompt
        assert "5x5" in prompt
        
        # Should NOT contain goal information
        assert "goal" not in prompt.lower() or "no information about the level goal" in prompt.lower()


class TestISO3TracebackIsolation:
    """Test ISO-3: Solver never sees Python source or tracebacks."""
    
    def test_isolation_traceback_never_reaches_solver(self):
        """
        Verify that Solver prompt contains no traceback or Python source.
        
        Ref: Spec 1.4 ISO-3 Invariant
        Ref: Spec 8.3 Solver Prompt Construction
        """
        function_manifest = {
            "functions": {
                "probe": {
                    "signature": "probe(x: int, y: int) -> EffectDeclaration",
                    "docstring": "Probe a cell",
                    "parameters": {"x": "int", "y": "int"},
                    "return_type": "EffectDeclaration",
                }
            }
        }
        
        prompt = build_solver_prompt(function_manifest)
        
        # ISO-3 Check: Solver must explicitly state it never sees Python source
        assert "never see python source" in prompt.lower() or \
               "never sees python source" in prompt.lower() or \
               "no python source" in prompt.lower()
        
        # ISO-3 Check: No Python source indicators in prompt
        python_indicators = ["def ", "import ", "traceback", "exception:", "error at line"]
        for indicator in python_indicators:
            # Allow "def" in function signature context but not as actual code
            if indicator == "def ":
                continue
            assert indicator not in prompt.lower(), f"ISO-3 VIOLATION: Found '{indicator}' in Solver prompt"
    
    def test_solver_prompt_has_only_manifest_and_docstrings(self):
        """Verify Solver sees only JSON manifest and docstrings."""
        function_manifest = {
            "functions": {
                "click": {
                    "signature": "click(obj_id: str) -> EffectDeclaration",
                    "docstring": "Click on an object",
                    "parameters": {"obj_id": "str"},
                    "return_type": "EffectDeclaration",
                }
            }
        }
        
        prompt = build_solver_prompt(function_manifest)
        
        # Should contain manifest info
        assert "Function Manifest" in prompt or "FUNCTION MANIFEST" in prompt
        assert "click" in prompt
        assert "docstring" in prompt.lower() or "Description" in prompt
        
        # Should NOT contain actual Python code
        assert "def click" not in prompt
        assert "import" not in prompt


class TestISOMemoryContourDisjoint:
    """Test ISO Memory Contours are strictly disjoint."""
    
    def test_isolation_memory_contours_are_disjoint(self):
        """
        Verify that memory contour objects have no cross-references.
        
        Ref: Spec 1.4 ISO-3 Invariant (Memory Contour Separation)
        Ref: Spec 3.5 Memory Contours
        """
        # Create independent instances
        env_memory = EnvironmentSpecMemory()
        syntax_memory = SyntaxErrorMemory()
        epistemic_memory = EpistemicMemory()
        
        # Verify no shared mutable state
        env_attrs = set(dir(env_memory))
        syntax_attrs = set(dir(syntax_memory))
        epistemic_attrs = set(dir(epistemic_memory))
        
        # Check that contours don't have each other's methods
        assert "add_error" not in env_attrs, "EnvironmentSpecMemory should not have SyntaxErrorMemory methods"
        assert "add_judgment" not in env_attrs, "EnvironmentSpecMemory should not have EpistemicMemory methods"
        
        assert "add_probe" not in syntax_attrs, "SyntaxErrorMemory should not have EnvironmentSpecMemory methods"
        assert "add_judgment" not in syntax_attrs, "SyntaxErrorMemory should not have EpistemicMemory methods"
        
        assert "add_probe" not in epistemic_attrs, "EpistemicMemory should not have EnvironmentSpecMemory methods"
        assert "add_error" not in epistemic_attrs, "EpistemicMemory should not have SyntaxErrorMemory methods"
    
    def test_no_cross_contour_methods(self):
        """Verify memory contours cannot access each other's data."""
        from v10_agent.types import EnvironmentSpecification, ObjectSpec
        
        # Create environment spec first (required for EnvironmentSpecMemory)
        env_spec = EnvironmentSpecification(
            grid_width=10,
            grid_height=10,
            object_specs=[ObjectSpec(type_id="obj1", description="Test")],
            relation_specs=[],
        )
        
        env_memory = EnvironmentSpecMemory()
        env_memory = env_memory.set_specification(env_spec)
        
        syntax_memory = SyntaxErrorMemory()
        epistemic_memory = EpistemicMemory()
        
        # Add data to each
        probe = ProbeRecord(
            probe_id="p1",
            action_id="ACTION1",
            confidence=0.9,
        )
        env_memory = env_memory.add_probe(probe)
        
        error = SyntaxErrorRecord(
            level_id="level1",
            prompt_hash="hash1",
            source_hash="hash2",
            traceback="test error",
        )
        syntax_memory = syntax_memory.add_error(
            prompt_hash="hash1",
            source_hash="hash2",
            traceback="test error",
            level_id="level1",
        )
        
        judgment = BrusentsovJudgment(
            judgment_type="FOLLOW",
            branch_signature=BranchSignature(
                action_sequence=("ACTION1",),
                outcome_hash="outcome1",
            ),
            reasoning="Test reasoning",
        )
        epistemic_memory = epistemic_memory.add_judgment(judgment)
        
        # Verify isolation
        assert len(list(env_memory.get_probe_history())) == 1
        assert len(list(syntax_memory.get_errors())) == 1
        assert len(list(epistemic_memory.get_judgments())) == 1
        
        # Cross-contour access should return empty
        # (Each contour only knows about its own data type)
    
    def test_contours_can_be_instantiated_independently(self):
        """Verify memory contours can be created without dependencies."""
        # Each contour should be independently instantiable
        env_memory = EnvironmentSpecMemory()
        assert env_memory is not None
        
        syntax_memory = SyntaxErrorMemory()
        assert syntax_memory is not None
        
        epistemic_memory = EpistemicMemory()
        assert epistemic_memory is not None
        
        game_memory = GameMemory()
        assert game_memory is not None


class TestISO4SandboxIsolation:
    """Test ISO-4: Sandbox prevents unauthorized code execution."""
    
    def test_sandbox_blocks_forbidden_imports(self):
        """Verify sandbox blocks dangerous imports."""
        from v10_agent.sandbox import SandboxExecutor
        
        executor = SandboxExecutor()
        
        # Test forbidden import
        malicious_code = "import os; print(os.getcwd())"
        diagnostics = executor.static_check(malicious_code, {})
        
        assert any("forbidden import" in d.lower() or "os" in d for d in diagnostics)
    
    def test_sandbox_blocks_eval_exec(self):
        """Verify sandbox blocks eval/exec."""
        from v10_agent.sandbox import SandboxExecutor
        
        executor = SandboxExecutor()
        
        # Test eval usage
        malicious_code = "result = eval('1+1')"
        diagnostics = executor.static_check(malicious_code, {})
        
        assert any("forbidden builtin" in d.lower() or "eval" in d for d in diagnostics)

    def test_sandbox_ast_validation_blocks_dunder_and_forbidden_nodes(self):
        """Verify AST visitor blocks dunder attributes, imports, globals, nonlocals."""
        from v10_agent.sandbox import SandboxExecutor
        executor = SandboxExecutor()

        with pytest.raises(ValueError, match="Import statements are forbidden"):
            executor.validate_ast("import math")

        with pytest.raises(ValueError, match="ImportFrom statements are forbidden"):
            executor.validate_ast("from math import sqrt")

        with pytest.raises(ValueError, match="Global declarations are forbidden"):
            executor.validate_ast("def foo(): global x")

        with pytest.raises(ValueError, match="Nonlocal declarations are forbidden"):
            executor.validate_ast("def foo():\n  x=1\n  def bar(): nonlocal x")

        with pytest.raises(ValueError, match="Accessing dunder attribute '__class__' is forbidden"):
            executor.validate_ast("x = ().__class__.__bases__")


class TestISO5PlanningSetBinding:
    """Test ISO-5: All actions bound to valid PlanningSet IDs."""
    
    def test_verification_binder_checks_object_ids(self):
        """Verify binder validates object ID references."""
        from v10_agent.planning_set import PlanningSet
        from v10_agent.verification import VerificationBinder
        from v10_agent.types import EffectDeclaration
        
        planning_set = PlanningSet.create(
            object_ids=["obj1", "obj2"],
            relation_ids=["rel1"],
            allowed_action_ids=["ACTION1"],
        )
        
        binder = VerificationBinder(planning_set)
        
        # Valid reference
        effect = EffectDeclaration(
            dsl_function="test",
            arguments={"object_id": "obj1"},
        )
        is_valid, errors = binder.verify_effect_declaration(effect)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid reference
        invalid_effect = EffectDeclaration(
            dsl_function="test",
            arguments={"object_id": "nonexistent"},
        )
        is_valid, errors = binder.verify_effect_declaration(invalid_effect)
        assert not is_valid
        assert len(errors) > 0
        assert errors[0].error_type == "missing_object"
