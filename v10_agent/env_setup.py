"""
Environment Setup Module for V10 Agent.
Handles GPU validation and Kaggle path configuration.
Ref: Spec 8 (Technical Environment)
"""

import os
import sys
import subprocess
from typing import List, Tuple


def validate_accelerator() -> Tuple[bool, str]:
    """
    Validates the presence and type of GPU accelerator.
    
    Returns:
        Tuple[bool, str]: (is_valid, gpu_info_string)
        
    Raises:
        RuntimeError: If no GPU is detected or nvidia-smi fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split("\n")
        gpu_info = "; ".join(gpus)
        return True, gpu_info
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. No GPU driver installed?")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GPU validation failed: {e.stderr}")


def setup_kaggle_paths() -> None:
    """
    Configures sys.path and environment variables for Kaggle environment.
    Checks for required datasets (wheelhouse, model).
    
    Ref: Spec 8.1 (Kaggle Path Setup)
    """
    wheelhouse_path = os.environ.get(
        "ARC_VLLM_WHEELHOUSE_PATH",
        "/kaggle/input/arc3-vllm-h100-wheelhouse-v3"
    )
    
    if os.path.exists(wheelhouse_path):
        if wheelhouse_path not in sys.path:
            sys.path.insert(0, wheelhouse_path)
        os.environ["PIP_FIND_LINKS"] = wheelhouse_path
        os.environ["PIP_NO_INDEX"] = "true"
        print(f"[OK] Wheelhouse configured at: {wheelhouse_path}")
    else:
        # Non-Kaggle environment or missing dataset - warn but continue
        print(f"[WARN] Wheelhouse not found at {wheelhouse_path}. Running in local mode.")
    
    # Set vLLM and HuggingFace offline modes
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    
    # Create cache directory
    hf_cache = os.environ.get("HF_HOME", "/kaggle/working/hf_cache")
    os.makedirs(hf_cache, exist_ok=True)


def get_gpu_count() -> int:
    """Returns the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except Exception:
        return 0
