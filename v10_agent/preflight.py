"""
Preflight Diagnostics Module for V10 Agent.
Performs structural smoke tests before game session initialization.
Ref: Spec 8.3 (Preflight Checks)
"""

import os
import sys
import json
import importlib
from typing import Dict, Any, List


def check_file_structure() -> Dict[str, bool]:
    """Verifies all required V10 agent files exist."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "__init__.py",
        "config.py",
        "types.py",
        "brusentsov_logic.py",
        "planning_set.py",
        "memory_contours.py",
        "sandbox.py",
        "verification.py",
        "judge.py",
        "env_setup.py",
        "vllm_lifecycle.py",
    ]
    
    results = {}
    for f in required_files:
        full_path = os.path.join(base_path, f)
        results[f] = os.path.isfile(full_path)
    
    return results


def check_module_imports() -> Dict[str, Any]:
    """
    Verifies key modules can be imported and checks their source paths.
    Ensures they load from /kaggle/working/v10_agent, not system paths.
    """
    modules_to_check = [
        "v10_agent.config",
        "v10_agent.session" if os.path.exists(os.path.join(os.path.dirname(__file__), "session.py")) else None,
        "v10_agent.sandbox",
        "v10_agent.judge",
        "v10_agent.vllm_lifecycle",
    ]
    
    # Filter out None (optional modules not yet created)
    modules_to_check = [m for m in modules_to_check if m is not None]
    
    results = {}
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            source_file = getattr(module, "__file__", "Unknown")
            is_correct_path = "/kaggle/working" in source_file or "workspace" in source_file
            results[module_name] = {
                "imported": True,
                "source": source_file,
                "valid_path": is_correct_path
            }
        except ImportError as e:
            results[module_name] = {
                "imported": False,
                "error": str(e),
                "valid_path": False
            }
    
    return results


def check_config_defaults() -> Dict[str, Any]:
    """Validates default configuration values match V10 spec."""
    try:
        from v10_agent.config import V10Config
        config = V10Config()
        
        # Check critical limits from Spec 2.1
        checks = {
            "max_coder_retries": config.max_coder_retries_per_level == 3,
            "max_solver_retries": config.max_solver_retries_per_level == 4,
            "max_explorer_probes": config.max_explorer_probe_actions_per_level == 8,
            "syntax_error_memory_max": config.syntax_error_memory_max_entries == 5,
            "epistemic_memory_max": config.epistemic_memory_max_entries == 100,
        }
        
        return {
            "valid": all(checks.values()),
            "details": checks
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def run_smoke_test_vllm() -> Dict[str, Any]:
    """
    Lightweight smoke test for vLLM connectivity.
    Does NOT start the server - just validates the manager class exists.
    Real server testing happens in the notebook.
    """
    try:
        from v10_agent.vllm_lifecycle import VLLMManager, VLLMConfig
        
        # Just validate instantiation, don't actually start
        cfg = VLLMConfig(model_path="/dummy/path")
        manager = VLLMManager(model_path="/dummy/path", config=cfg)
        
        return {
            "success": True,
            "manager_class": "VLLMManager",
            "methods_available": [
                hasattr(manager, "start"),
                hasattr(manager, "stop"),
                hasattr(manager, "wait_for_health"),
                hasattr(manager, "get_bounded_log_tail"),
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_structural_preflight() -> Dict[str, Any]:
    """
    Runs the complete preflight diagnostic suite.
    
    Returns:
        Dict with overall status and detailed results for each check.
    """
    results = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "version": "10.0.0",
        "checks": {}
    }
    
    # 1. File Structure
    results["checks"]["file_structure"] = check_file_structure()
    files_ok = all(results["checks"]["file_structure"].values())
    
    # 2. Module Imports
    results["checks"]["module_imports"] = check_module_imports()
    imports_ok = all(m.get("imported", False) for m in results["checks"]["module_imports"].values())
    
    # 3. Config Defaults
    results["checks"]["config_defaults"] = check_config_defaults()
    config_ok = results["checks"]["config_defaults"].get("valid", False)
    
    # 4. vLLM Smoke Test (class validation only)
    results["checks"]["vllm_smoke"] = run_smoke_test_vllm()
    vllm_ok = results["checks"]["vllm_smoke"].get("success", False)
    
    # Overall Status
    overall_success = files_ok and imports_ok and config_ok and vllm_ok
    results["overall_status"] = "PASS" if overall_success else "FAIL"
    results["summary"] = {
        "files_ok": files_ok,
        "imports_ok": imports_ok,
        "config_ok": config_ok,
        "vllm_ok": vllm_ok
    }
    
    return results


if __name__ == "__main__":
    result = run_structural_preflight()
    print(json.dumps(result, indent=2))
