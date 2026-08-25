"""
Sandbox Executor - ISO-4 Compliant (Safe Code Execution)
Ref: Engineering Specification V10.0 Section 4 & 7
"""

import ast
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from .types import SyntaxErrorRecord, EffectDeclaration, AtomicProposition
from .planning_set import PlanningSet


# Forbidden modules and functions for security
FORBIDDEN_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib',
    'socket', 'http', 'urllib', 'requests', 'ftplib',
    'pickle', 'marshal', 'ctypes', 'importlib',
}

FORBIDDEN_BUILTINS = {
    'exec', 'eval', 'compile', 'open', 'input',
    '__import__', 'getattr', 'setattr', 'delattr',
}


@dataclass
class SandboxAPI:
    """
    Restricted API available to DSL functions in sandbox.
    
    Ref: Spec 4.2 - Restricted executor (SandboxAPI)
    
    Provides read-only access to PlanningSet and declare_environment_action.
    No direct environment modification allowed.
    """
    planning_set: PlanningSet
    _declare_action: Callable[[Dict[str, Any]], EffectDeclaration]
    
    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get object info from PlanningSet (read-only)."""
        # In full implementation, this would query the actual PlanningSet
        return None
    
    def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        """Get relation info from PlanningSet (read-only)."""
        return None
    
    def declare_environment_action(
        self,
        dsl_function: str,
        arguments: Dict[str, Any],
        expected_propositions: Optional[List[AtomicProposition]] = None,
    ) -> EffectDeclaration:
        """
        Declare intended environment action effect.
        
        This does NOT execute the action - it only declares intended effects.
        Actual execution happens later via ActionBoundary after verification.
        
        Ref: Spec 4.2 - ActionBoundary separation
        """
        return self._declare_action({
            'dsl_function': dsl_function,
            'arguments': arguments,
            'expected_propositions': expected_propositions or [],
        })


class SandboxExecutor:
    """
    Secure sandbox for executing Coder-generated DSL functions.
    
    ISO-4 INVARIANT: Only validated code runs in sandbox.
    All exceptions are captured as SyntaxErrorRecord.
    """
    
    def __init__(self):
        self.allowed_functions: Dict[str, Callable] = {}
        self.error_count = 0
    
    def static_check(self, source: str, manifest: Dict[str, Any]) -> List[str]:
        """
        Perform static analysis on source code before execution.
        
        Args:
            source: Python source code string
            manifest: Function manifest with allowed function names
        
        Returns:
            List of diagnostic error messages (empty if valid)
        
        Ref: Spec 4.1 - Static Check Pipeline
        """
        diagnostics = []
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Check for forbidden imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in FORBIDDEN_IMPORTS:
                        diagnostics.append(f"Forbidden import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in FORBIDDEN_IMPORTS:
                    diagnostics.append(f"Forbidden import: {node.module}")
            
            # Check for forbidden builtins usage
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_BUILTINS:
                        diagnostics.append(f"Forbidden builtin: {node.func.id}")
        
        # Check that all manifest functions are defined
        required_functions = set(manifest.get('functions', {}).keys())
        defined_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
        
        missing = required_functions - defined_functions
        if missing:
            diagnostics.append(f"Missing functions from manifest: {missing}")
        
        return diagnostics
    
    def call(
        self,
        source: str,
        function_name: str,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        planning_set: Optional[PlanningSet] = None,
    ) -> Any:
        """
        Execute a function from source code in restricted sandbox.
        
        Args:
            source: Python source code
            function_name: Name of function to call
            args: Positional arguments
            kwargs: Keyword arguments
            planning_set: Current PlanningSet for context
        
        Returns:
            Function return value (typically EffectDeclaration)
        
        Raises:
            Exception: Captured and converted to SyntaxErrorRecord
        
        Ref: Spec 4.2 - Restricted executor
        """
        kwargs = kwargs or {}
        
        # First perform static check
        manifest = {'functions': {function_name: {}}}
        diagnostics = self.static_check(source, manifest)
        
        if diagnostics:
            raise ValueError(f"Sandbox static check failed: {'; '.join(diagnostics)}")
        
        # Create restricted globals
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name)
                for name in dir(__builtins__)
                if name not in FORBIDDEN_BUILTINS and not name.startswith('_')
            },
        }
        
        # Create sandbox API if planning_set provided
        if planning_set:
            def declare_action(data):
                return EffectDeclaration(
                    dsl_function=data['dsl_function'],
                    arguments=data['arguments'],
                    expected_propositions=data.get('expected_propositions', []),
                )
            
            safe_globals['sandbox_api'] = SandboxAPI(
                planning_set=planning_set,
                _declare_action=declare_action,
            )
        
        # Execute source in sandbox
        try:
            exec(source, safe_globals)
        except Exception as e:
            self.error_count += 1
            raise
        
        # Get function from executed namespace
        if function_name not in safe_globals:
            raise ValueError(f"Function '{function_name}' not found in source")
        
        func = safe_globals[function_name]
        
        # Call function
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_count += 1
            raise
    
    def create_error_record(
        self,
        source: str,
        exception: Exception,
        level_id: str,
        prompt: Optional[str] = None,
    ) -> SyntaxErrorRecord:
        """
        Create SyntaxErrorRecord from sandbox exception.
        
        Ref: Spec 3.5.2 - SyntaxErrorMemory Contract
        """
        prompt_hash = hashlib.sha256((prompt or '').encode()).hexdigest()[:16]
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
        
        return SyntaxErrorRecord(
            level_id=level_id,
            prompt_hash=prompt_hash,
            source_hash=source_hash,
            traceback=str(exception),
            static_diagnostics=self.static_check(source, {}),
        )
