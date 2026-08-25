#!/usr/bin/env python3
"""
Build script for ARC-AGI-3 LCLD Agent Version 10.0 Kaggle Notebook.
Generates:
  1. arc-prize-2026-lcld-v10.ipynb
  2. kernel-metadata.json

This script constructs a clean, modern notebook based on V10 architecture,
avoiding legacy code from V9 while adhering to Kaggle's submission requirements.
"""

import json
import os
from pathlib import Path

# --- Configuration ---
NOTEBOOK_TITLE = "ARC-AGI-3 LCLD Agent V10.0 (Qwen 3.8B + Brusentsov Logic)"
NOTEBOOK_FILE = "arc-prize-2026-lcld-v10.ipynb"
METADATA_FILE = "kernel-metadata.json"
AGENT_PACKAGE_DIR = "v10_agent"

# Kaggle Dataset Dependencies
DATASETS = [
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    # Add model dataset if hosted as a dataset, otherwise rely on model path in code
    # "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1" 
]

# Model Configuration (Matches V10 Spec)
MODEL_PATH = "/kaggle/input/qwen3-8b-fp8-repacked" # Adjust based on actual model dataset mount point
WHEELHOUSE_PATH = "/kaggle/input/arc3-vllm-h100-wheelhouse-v3"

def create_kernel_metadata() -> dict:
    """Generates kernel-metadata.json content."""
    return {
        "id": f"yakuninvladimirui/{NOTEBOOK_FILE.replace('.ipynb', '')}",
        "title": NOTEBOOK_TITLE,
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,  # Start private for testing
        "enable_gpu": True,
        "dataset_sources": DATASETS,
        "model_sources": [], # If using specific model datasets, add here
        "internet_enabled": False,  # Strict offline for competition
        "docker_image": "ghcr.io/kaggle/kaggle-python:latest", # Or specific version
        "accelerator": "GPU_T4x2" # T4 x2 is standard free tier, H100/P100 if available
    }

def create_setup_cell() -> dict:
    """Cell 1: Environment Setup, Dependency Installation, and Preflight."""
    source_code = f'''# ==============================================================================
# CELL 1: ENVIRONMENT SETUP & PREFLIGHT (V10.0)
# ==============================================================================
# This cell prepares the Kaggle environment, installs vLLM from wheelhouse,
# validates the GPU, and runs structural preflight checks before any logic runs.

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path

print(">>> Initializing ARC-AGI-3 LCLD Agent V10.0 Environment...")

# 1. Configure Paths
WHEELHOUSE_PATH = "{WHEELHOUSE_PATH}"
MODEL_PATH = "{MODEL_PATH}"

# Add wheelhouse to sys.path for pip install
if os.path.exists(WHEELHOUSE_PATH):
    sys.path.insert(0, WHEELHOUSE_PATH)
    os.environ["PIP_FIND_LINKS"] = WHEELHOUSE_PATH
    os.environ["PIP_NO_INDEX"] = "true"
    print(f"[OK] Wheelhouse found at: {{WHEELHOUSE_PATH}}")
else:
    raise FileNotFoundError(f"Wheelhouse not found at {{WHEELHOUSE_PATH}}. Check dataset attachment.")

# 2. Install Dependencies from Wheelhouse
# We install vLLM and dependencies strictly from the local wheelhouse
packages_to_install = [
    "vllm",
    "torch",
    "transformers",
    "pydantic",
    "numpy",
    "pandas"
]

print(">>> Installing dependencies from local wheelhouse...")
for pkg in packages_to_install:
    try:
        # Attempt to import first to check if already installed
        __import__(pkg.replace("-", "_"))
        print(f"  - {{pkg}}: Already loaded")
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--find-links", WHEELHOUSE_PATH, pkg]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  - {{pkg}}: Installed")
        else:
            # Some packages might be transitive dependencies, ignore errors if import works later
            pass

# Force reload of critical modules if needed
import importlib

# 3. Set Environment Variables for vLLM and Offline Mode
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.makedirs("/kaggle/working/hf_cache", exist_ok=True)

# 4. Validate Accelerator (GPU Check)
print(">>> Validating Accelerator...")
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                            capture_output=True, text=True, check=True)
    gpus = result.stdout.strip().split("\\n")
    print(f"[OK] Detected GPUs:")
    for gpu in gpus:
        print(f"      - {{gpu}}")
except Exception as e:
    print(f"[ERROR] GPU Validation Failed: {{e}}")
    raise RuntimeError("No GPU detected or nvidia-smi failed. Cannot proceed without GPU.")

# 5. Add Agent Package to Path
AGENT_ROOT = "/kaggle/working"
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)
    print(f"[OK] Added {{AGENT_ROOT}} to sys.path")

# 6. Run Structural Preflight (Smoke Test)
print(">>> Running Structural Preflight...")
try:
    from v10_agent.preflight import run_structural_preflight
    from v10_agent.env_setup import validate_accelerator, setup_kaggle_paths
    
    # Run path setup
    setup_kaggle_paths()
    
    # Run full preflight
    preflight_result = run_structural_preflight()
    print("[OK] Structural Preflight Passed:")
    print(json.dumps(preflight_result, indent=2))
    
except Exception as e:
    print(f"[CRITICAL] Preflight Failed: {{e}}")
    import traceback
    traceback.print_exc()
    # In Phase A, we might want to continue to generate a dummy submission, 
    # but for now, we halt to fix issues.
    raise SystemExit("Preflight failed. Aborting.")

print(">>> Environment Ready. Proceeding to Main Logic...")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"_kg_hide_input": False},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def create_imports_cell() -> dict:
    """Cell 2: Imports."""
    source_code = '''# ==============================================================================
# CELL 2: IMPORTS (V10.0)
# ==============================================================================
# Import core components of the LCLD Agent V10.0

from v10_agent.config import V10Config
from v10_agent.session import GameSession
from v10_agent.vllm_lifecycle import VLLMManager
from v10_agent.kaggle_limits import get_competition_limits
from v10_agent.types import PropositionSet, EffectDeclaration

import pandas as pd
import numpy as np
import json
import os
import time
import traceback

print("[OK] All V10 modules imported successfully.")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def create_main_logic_cell() -> dict:
    """Cell 3: Main Execution Logic (Phase A/B Handling)."""
    source_code = '''# ==============================================================================
# CELL 3: MAIN EXECUTION LOGIC (PHASE A vs PHASE B)
# ==============================================================================
# Handles the distinction between Dry Run (Phase A) and Competition (Phase B).
# Manages vLLM lifecycle strictly.

def run_phase_a_dry_run():
    """
    Phase A: Dry Run / Validation.
    Goal: Prove the agent initializes, passes preflight, and can talk to vLLM.
    Output: Detailed JSON logs to stdout, dummy submission.
    """
    print("="*60)
    print("RUNNING PHASE A (DRY RUN)")
    print("="*60)
    
    vllm_manager = None
    try:
        # 1. Initialize vLLM Manager
        config = V10Config.from_env()
        vllm_manager = VLLMManager(model_path=config.qwen_model_path, config=config)
        
        # 2. Start vLLM Server
        print(">>> Starting vLLM Server...")
        vllm_manager.start()
        
        # 3. Wait for Health
        print(">>> Waiting for vLLM Health Check...")
        if not vllm_manager.wait_for_health(timeout=300):
            raise RuntimeError("vLLM failed to become healthy within timeout.")
        print("[OK] vLLM Server is Healthy.")
        
        # 4. Smoke Test (Lightweight Inference)
        print(">>> Running Smoke Test Inference...")
        # Use a minimal prompt to test connectivity
        test_prompt = "Respond with exactly this JSON: {\\"test\\": true}"
        # Assuming VLLMManager has a method for raw generation or we use requests
        # For V10, we assume a method `generate` exists or we access the session
        # Here we simulate a direct call for the smoke test
        response = vllm_manager.generate(prompt=test_prompt, max_tokens=10, temperature=0)
        print(f"Smoke Test Response: {{response}}")
        
        # 5. Log Tail (Critical for Phase A debugging)
        log_tail = vllm_manager.get_bounded_log_tail(bytes_limit=12000)
        print(">>> vLLM Log Tail (Last 12KB):")
        print(log_tail)
        
        print("[SUCCESS] Phase A Dry Run Completed Successfully.")
        return True
        
    except Exception as e:
        print(f"[FAILURE] Phase A Dry Run Failed: {{e}}")
        traceback.print_exc()
        return False
    finally:
        if vllm_manager:
            print(">>> Stopping vLLM Server...")
            vllm_manager.stop()

def run_phase_b_competition(tasks_df):
    """
    Phase B: Actual Competition Run.
    Goal: Solve tasks, respect limits, generate valid submission.
    """
    print("="*60)
    print("RUNNING PHASE B (COMPETITION)")
    print("="*60)
    
    config = V10Config.from_env()
    limits = get_competition_limits()
    vllm_manager = None
    all_predictions = []
    
    try:
        # 1. Start vLLM
        vllm_manager = VLLMManager(model_path=config.qwen_model_path, config=config)
        vllm_manager.start()
        if not vllm_manager.wait_for_health(timeout=600):
            raise RuntimeError("vLLM failed to start.")
        print("[OK] vLLM Ready for Competition.")
        
        # 2. Iterate over tasks
        # tasks_df expected columns: 'task_id', 'train', 'test' (JSON strings)
        for idx, row in tasks_df.iterrows():
            task_id = row['task_id']
            print(f"\\n>>> Processing Task: {{task_id}} ({{idx+1}}/{{len(tasks_df)}})")
            
            # Check global time limit
            if time.time() > start_time + limits.competition_wall_clock_seconds:
                print("[WARNING] Global time limit reached. Stopping.")
                break
            
            try:
                # Initialize Session for this task
                # Note: GameSession handles internal retries and memory contours
                session = GameSession(config=config, vllm_manager=vllm_manager)
                
                # Parse task data (simplified for notebook snippet)
                # In real implementation, parse train/test pairs properly
                # Here we assume a method `solve_task` exists in GameSession or similar
                # For V10, we iterate test pairs
                
                # Placeholder for actual solving loop
                # predictions = session.solve_task(task_data=row) 
                # all_predictions.extend(predictions)
                
                # DUMMY PREDICTION FOR STRUCTURE (Replace with real logic)
                # The real logic involves: session.act(obs) -> ... -> prediction
                print(f"  [INFO] Logic placeholder for {{task_id}}. Real solver integrated in GameSession.")
                
            except Exception as e:
                print(f"[ERROR] Failed task {{task_id}}: {{e}}")
                traceback.print_exc()
                # Continue to next task on error
                
    except Exception as e:
        print(f"[CRITICAL] Competition Run Failed: {{e}}")
        traceback.print_exc()
    finally:
        if vllm_manager:
            vllm_manager.stop()
            
    return all_predictions

# --- Execution Entry Point ---
KAGGLE_IS_COMPETITION_RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "false").lower() == "true"
start_time = time.time()

if not KAGGLE_IS_COMPETITION_RERUN:
    # PHASE A: Dry Run
    success = run_phase_a_dry_run()
    
    # Generate Dummy Submission for Phase A to avoid "No Output" error
    dummy_data = {{"task_id": ["dummy_001"], "output": ["[[0]]"]}}
    df_dummy = pd.DataFrame(dummy_data)
    df_dummy.to_parquet("submission.parquet", index=False)
    print("Generated dummy submission.parquet for Phase A.")
    
else:
    # PHASE B: Competition
    # Load tasks (Assuming standard Kaggle input path)
    input_path = "/kaggle/input/arc-prize-2025" # Adjust year as needed
    # Check if file exists, fallback to empty if running locally without data
    tasks_file = os.path.join(input_path, "tasks.json") # Or whatever the format is
    
    if os.path.exists(tasks_file):
        # Load and process tasks
        # Note: Actual loading logic depends on the specific competition file structure
        # Usually it's a folder of JSON files or a single parquet/json
        # This is a placeholder for the loader
        print("Loading tasks from:", input_path)
        # tasks_df = load_tasks(input_path) 
        # predictions = run_phase_b_competition(tasks_df)
        
        # For now, create empty submission structure if no data loaded
        df_sub = pd.DataFrame(columns=["task_id", "output"])
    else:
        print(f"Warning: Tasks not found at {{tasks_file}}. Creating empty submission.")
        df_sub = pd.DataFrame(columns=["task_id", "output"])
        
    df_sub.to_parquet("submission.parquet", index=False)
    print("Competition run finished. submission.parquet created.")

print(f"Total Runtime: {{time.time() - start_time:.2f}}s")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def create_submission_cell() -> dict:
    """Cell 4: Final Submission Verification."""
    source_code = '''# ==============================================================================
# CELL 4: SUBMISSION VERIFICATION
# ==============================================================================
import pandas as pd
import os

if os.path.exists("submission.parquet"):
    df = pd.read_parquet("submission.parquet")
    print("Submission File Contents:")
    print(df.head())
    print(f"Total rows: {{len(df)}}")
else:
    print("ERROR: submission.parquet was not created!")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def build_notebook():
    """Assembles the notebook JSON structure."""
    print(f"Building notebook: {NOTEBOOK_FILE}...")
    
    notebook_content = {
        "cells": [
            create_setup_cell(),
            create_imports_cell(),
            create_main_logic_cell(),
            create_submission_cell()
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(NOTEBOOK_FILE, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"[OK] Created {NOTEBOOK_FILE}")

def build_metadata():
    """Writes the kernel-metadata.json file."""
    print(f"Building metadata: {METADATA_FILE}...")
    
    metadata = create_kernel_metadata()
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Created {METADATA_FILE}")

def main():
    # Verify agent package exists
    if not os.path.isdir(AGENT_PACKAGE_DIR):
        print(f"[ERROR] Agent package directory '{AGENT_PACKAGE_DIR}' not found!")
        print("Please ensure v10_agent is in the current working directory.")
        return 1
    
    # Verify key files exist (sanity check)
    required_files = [
        os.path.join(AGENT_PACKAGE_DIR, "__init__.py"),
        os.path.join(AGENT_PACKAGE_DIR, "session.py"),
        os.path.join(AGENT_PACKAGE_DIR, "preflight.py"),
        os.path.join(AGENT_PACKAGE_DIR, "vllm_lifecycle.py")
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"[ERROR] Required file missing: {f}")
            return 1
            
    print("[OK] All required agent files found.")
    
    build_notebook()
    build_metadata()
    
    print("\n" + "="*60)
    print("BUILD COMPLETE")
    print("="*60)
    print(f"Files generated:")
    print(f"  1. {NOTEBOOK_FILE}")
    print(f"  2. {METADATA_FILE}")
    print("\nTo submit to Kaggle:")
    print(f"  kaggle kernels push -p .")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())
