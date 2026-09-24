"""Restricted AST Sandbox Executor and Validator for dynamic Coder DSL modules."""

from __future__ import annotations

import ast
import math
import sys
import concurrent.futures
import signal
import threading
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Callable, Sequence

from v10_agent.planning_set import PlanningSet
from v10_agent.types import ActionDeclaration, EffectDeclaration


class SandboxTimeoutError(TimeoutError):
    """Raised when sandboxed DSL code execution exceeds the allocated timeout."""
    pass

DEFAULT_ALLOWED_MODULES = frozenset({"math", "typing", "dataclasses", "enum", "collections"})

FORBIDDEN_CALL_NAMES = frozenset({
    "exec",
    "eval",
    "open",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "__import__",
})

FORBIDDEN_ATTRIBUTES = frozenset({
    "__subclasses__",
    "__globals__",
    "__code__",
    "__bases__",
    "__mro__",
    "__class__",
    "__closure__",
})

RESTRICTED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "isinstance": isinstance,
    "sum": sum,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
}


def validate_dsl_source(
    source: str,
    expected_manifest: dict[str, Any] | None = None,
    allowed_modules: Sequence[str] | None = None,
) -> list[str]:
    """Statically analyze dynamic Python source code.

    Returns a list of error diagnostic strings (empty if valid).
    """
    diagnostics: list[str] = []
    allowed = frozenset(allowed_modules) if allowed_modules is not None else DEFAULT_ALLOWED_MODULES

    try:
        tree = ast.parse(source, filename="<dynamic_dsl>")
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} at line {exc.lineno}, col {exc.offset}"]

    # 1. Inspect all nodes for forbidden operations
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name.split(".")[0]
                if mod_name not in allowed:
                    diagnostics.append(f"Forbidden import: {alias.name!r} is not in whitelist {sorted(allowed)}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = (node.module or "").split(".")[0]
            if mod_name not in allowed:
                diagnostics.append(f"Forbidden import from: {node.module!r} is not in whitelist {sorted(allowed)}")

        # Check function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                diagnostics.append(f"Forbidden function call: {node.func.id}() is banned in sandbox")

        # Check dangerous attribute lookups
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                diagnostics.append(f"Forbidden dunder attribute access: {node.attr!r}")

    # 2. Check manifest conformance if provided
    if expected_manifest and "functions" in expected_manifest:
        defined_functions: dict[str, list[str]] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                params = [arg.arg for arg in node.args.args]
                defined_functions[node.name] = params

        for func_spec in expected_manifest["functions"]:
            name = func_spec.get("name")
            if not name:
                continue
            if name not in defined_functions:
                diagnostics.append(f"Manifest declared function {name!r} but it was not defined in source")
                continue

            # Check declared parameters match AST
            declared_params = [p.get("name") for p in func_spec.get("parameters", []) if "name" in p]
            ast_params = defined_functions[name]
            # API instance may be an explicit first argument (e.g. 'api') or implicit
            if "api" in ast_params:
                filtered_ast = [p for p in ast_params if p != "api"]
            else:
                filtered_ast = ast_params

            if declared_params != filtered_ast:
                diagnostics.append(
                    f"Parameter mismatch for {name!r}: manifest declared {declared_params} but function has {filtered_ast}"
                )

    return diagnostics


class SandboxAPI:
    """Safe, pure API exposed to the sandboxed Coder DSL functions."""

    def __init__(self, planning_set: PlanningSet):
        self._planning_set = planning_set

    @property
    def planning_set(self) -> PlanningSet:
        return self._planning_set

    def query_objects(self, predicate: Callable[[Any], bool]) -> list[Any]:
        """Query objects from the current PlanningSet."""
        return [obj for obj in self._planning_set.objects if predicate(obj)]

    def get_object(self, key: str) -> Any:
        """Get planning object by real ID or alias."""
        return self._planning_set.get_object(key)

    def metric_distance(self, obj_a: str, obj_b: str, metric_name: str = "centroid_distance") -> float:
        """Calculate metric distance between two objects in PlanningSet."""
        o_a = self.get_object(obj_a)
        o_b = self.get_object(obj_b)
        if not o_a or not o_b:
            raise KeyError(f"Objects {obj_a!r} or {obj_b!r} not found in PlanningSet")
        if metric_name == "centroid_distance":
            return math.hypot(o_a.centroid.row - o_b.centroid.row, o_a.centroid.col - o_b.centroid.col)
        if metric_name == "manhattan_distance":
            return abs(o_a.centroid.row - o_b.centroid.row) + abs(o_a.centroid.col - o_b.centroid.col)
        raise ValueError(f"Unknown metric {metric_name!r}")

    def declare_environment_action(
        self,
        action_id: str,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        expected_metric_deltas: dict[str, int | float] | None = None,
        target_object_ids: list[str] | None = None,
        confidence: float = 1.0,
    ) -> EffectDeclaration:
        """Pure declaration of intended action. Does not call real environment."""
        clean_action = str(action_id).upper()
        if not self._planning_set.is_valid_action(clean_action):
            raise ValueError(
                f"Action {clean_action!r} is not in allowed actions: {self._planning_set.allowed_action_ids}"
            )

        # Validate target_object_ids resolve in planning_set (Rule ISO-5)
        clean_targets: list[str] = []
        if target_object_ids:
            for tid in target_object_ids:
                resolved = self._planning_set.resolve_object_id(tid)
                if resolved is None:
                    raise KeyError(f"Target object ID {tid!r} does not resolve in current PlanningSet")
                clean_targets.append(resolved)

        decl = ActionDeclaration(
            action_id=clean_action,
            data=dict(data or {}),
            reasoning=dict(reasoning or {}),
        )
        return EffectDeclaration(
            declared_action=decl,
            expected_metric_deltas=dict(expected_metric_deltas or {}),
            target_object_ids=clean_targets,
            confidence=confidence,
        )


@dataclass
class SandboxedModule:
    """A validated, compiled DSL module ready for safe execution."""
    source: str
    manifest: dict[str, Any]
    compiled_code: CodeType
    namespace: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        ns = self.__dict__.get("namespace", {})
        if name in ns:
            return ns[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class SandboxExecutor:
    """Restricted executor for dynamic Coder-generated Python modules."""

    def __init__(
        self,
        allowed_modules: Sequence[str] | None = None,
        timeout_seconds: float = 2.0,
    ):
        self.allowed_modules = list(allowed_modules or DEFAULT_ALLOWED_MODULES)
        self.timeout_seconds = timeout_seconds

    def _execute_with_timeout(self, func: Callable[..., Any], call_args: dict[str, Any]) -> Any:
        timeout = max(0.1, float(self.timeout_seconds))
        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _alarm_handler(signum: int, frame: Any) -> None:
                raise SandboxTimeoutError(f"Execution exceeded timeout of {timeout}s")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return func(**call_args)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            target_thread = None
            result_holder: list[Any] = []
            exc_holder: list[Exception] = []

            def _worker() -> None:
                nonlocal target_thread
                target_thread = threading.get_ident()
                try:
                    res = func(**call_args)
                    result_holder.append(res)
                except Exception as e:
                    exc_holder.append(e)

            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            th.join(timeout=timeout)
            if th.is_alive():
                if target_thread is not None:
                    try:
                        import ctypes
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(target_thread),
                            ctypes.py_object(SandboxTimeoutError),
                        )
                    except Exception:
                        pass
                raise SandboxTimeoutError(f"Execution exceeded timeout of {timeout}s")
            if exc_holder:
                raise exc_holder[0]
            if result_holder:
                return result_holder[0]
            raise SandboxTimeoutError(f"Execution exceeded timeout of {timeout}s")

    def load_module(self, source: str, manifest: dict[str, Any]) -> SandboxedModule:
        """Validate and compile source code inside restricted sandbox namespace."""
        diagnostics = validate_dsl_source(source, manifest, self.allowed_modules)
        if diagnostics:
            raise SyntaxError("Sandbox validation failed:\n" + "\n".join(diagnostics))

        compiled = compile(source, "<level_dsl>", "exec")

        # Prepare restricted execution namespace with a whitelist-enforced __import__
        allowed_set = set(self.allowed_modules)
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else getattr(__builtins__, "__import__")

        def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple = (), level: int = 0) -> Any:
            root_name = name.split(".")[0]
            if root_name not in allowed_set:
                raise ImportError(f"Sandbox security: import of {name!r} is forbidden")
            return real_import(name, globals, locals, fromlist, level)

        builtins_dict = dict(RESTRICTED_BUILTINS)
        builtins_dict["__import__"] = _safe_import

        namespace: dict[str, Any] = {
            "__builtins__": builtins_dict,
        }

        # Execute top-level definitions
        exec(compiled, namespace)  # Safe: static AST validation already proved whitelist compliance

        return SandboxedModule(
            source=source,
            manifest=manifest,
            compiled_code=compiled,
            namespace=namespace,
        )

    def dry_run_manifest(
        self,
        module: SandboxedModule,
        planning_set: PlanningSet,
    ) -> tuple[bool, str | None]:
        """Perform a dry-run test of all declared manifest functions."""
        api = SandboxAPI(planning_set)
        functions = module.manifest.get("functions", [])
        if not functions:
            return False, "Manifest contains no function declarations"

        for func_meta in functions:
            name = func_meta["name"]
            func = module.namespace.get(name)
            if not callable(func):
                return False, f"Function {name!r} is not callable in module namespace"

            # Prepare dummy arguments based on parameter declarations
            kwargs: dict[str, Any] = {}
            for param in func_meta.get("parameters", []):
                p_name = param["name"]
                p_type = param.get("type", "str")

                if p_name == "api":
                    continue
                if p_type == "planning_object_id":
                    kwargs[p_name] = planning_set.object_ids[0] if planning_set.object_ids else "obj_0"
                elif p_type == "metric_id":
                    kwargs[p_name] = param.get("default", "centroid_distance")
                elif p_type == "int":
                    kwargs[p_name] = int(param.get("default", 0))
                else:
                    kwargs[p_name] = param.get("default", "val")

            try:
                # Try calling with api as kwarg or first positional arg if accepted
                import inspect
                sig = inspect.signature(func)
                call_args: dict[str, Any] = dict(kwargs)
                if "api" in sig.parameters:
                    call_args["api"] = api

                has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if not has_var_keyword:
                    call_args = {k: v for k, v in call_args.items() if k in sig.parameters}

                res = self._execute_with_timeout(func, call_args)
                if not isinstance(res, (EffectDeclaration, dict)):
                    return False, f"Function {name!r} returned {type(res).__name__}, expected EffectDeclaration"
            except Exception as exc:
                return False, f"Dry-run execution failed for {name!r}: {type(exc).__name__}: {exc}"

        return True, None

    def execute(
        self,
        module: SandboxedModule,
        function_name: str,
        arguments: dict[str, Any],
        planning_set: PlanningSet,
    ) -> EffectDeclaration:
        """Execute a declared function inside the sandbox and return an EffectDeclaration."""
        func = module.namespace.get(function_name)
        if not callable(func):
            raise KeyError(f"Function {function_name!r} not found in module namespace")

        api = SandboxAPI(planning_set)
        import inspect
        sig = inspect.signature(func)
        call_args: dict[str, Any] = dict(arguments)
        if "api" in sig.parameters and "api" not in call_args:
            call_args["api"] = api

        # Filter arguments to prevent unexpected keyword argument errors
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_keyword:
            call_args = {k: v for k, v in call_args.items() if k in sig.parameters}

        # Provide safe default values for missing required parameters to prevent TypeError crashes
        for p_name, param in sig.parameters.items():
            if p_name != "api" and p_name not in call_args and param.default == inspect.Parameter.empty:
                if p_name == "x":
                    call_args["x"] = int(planning_set.coordinate_candidates[0].x) if planning_set.coordinate_candidates else 0
                elif p_name == "y":
                    call_args["y"] = int(planning_set.coordinate_candidates[0].y) if planning_set.coordinate_candidates else 0
                elif param.annotation == int or p_name in ("count", "step", "idx"):
                    call_args[p_name] = 0
                elif p_name in ("obj_id", "target_id", "subject_id"):
                    call_args[p_name] = planning_set.object_ids[0] if planning_set.object_ids else "obj_0"
                else:
                    call_args[p_name] = 0

        result = self._execute_with_timeout(func, call_args)

        if isinstance(result, EffectDeclaration):
            return result
        if isinstance(result, dict):
            # Parse into EffectDeclaration
            act_dict = result.get("declared_action", result)
            act_id = act_dict.get("action_id", "ACTION1")
            return api.declare_environment_action(
                action_id=act_id,
                data=act_dict.get("data"),
                reasoning=act_dict.get("reasoning"),
                expected_metric_deltas=result.get("expected_metric_deltas"),
                target_object_ids=result.get("target_object_ids"),
            )

        raise TypeError(
            f"DSL function {function_name!r} must return EffectDeclaration, got {type(result).__name__}"
        )
