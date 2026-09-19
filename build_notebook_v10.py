"""Autonomous notebook builder packaging ARC-AGI-3 LCLD Agent V10.0 for Kaggle submission."""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import py_compile
import sys
import zipfile
from typing import Any

# =============================================================================
# BUILDER TOGGLE: Phase A Heavy Combat Smoke
# Set to True for deep diagnostic logging in Phase A (Save Version log).
# Set to False for lightning-fast submission without model warm-up in Phase A.
# =============================================================================
ENABLE_PHASE_A_HEAVY_SMOKE: bool = False

ROOT_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUT_NOTEBOOK = NOTEBOOKS_DIR / "arc-prize-2026-lcld-qwen-v10.ipynb"
KERNEL_METADATA_PATH = NOTEBOOKS_DIR / "kernel-metadata.json"
MAX_NOTEBOOK_BYTES = 985_000


def collect_payload_files() -> dict[str, bytes]:
    """Collect, compile-check, and read all runtime payload files."""
    files: dict[str, bytes] = {}

    root_files = [
        "kaggle_agent.py",
        "submission.py",
        "lcld_competition_child.py",
        "lcld_preflight.py",
        "phase_a_heavy_smoke.py",
    ]
    for rf in root_files:
        p = ROOT_DIR / rf
        if not p.is_file():
            raise FileNotFoundError(f"Missing required root payload file: {rf}")
        py_compile.compile(str(p), doraise=True)
        files[rf] = p.read_bytes()

    v10_dir = ROOT_DIR / "v10_agent"
    for py_file in v10_dir.rglob("*.py"):
        # Exclude tests and pycache
        rel = py_file.relative_to(ROOT_DIR)
        parts = rel.parts
        if "tests" in parts or "__pycache__" in parts:
            continue
        py_compile.compile(str(py_file), doraise=True)
        files[str(rel).replace("\\", "/")] = py_file.read_bytes()

    return files


def build_lzma_payload(files: dict[str, bytes]) -> str:
    """Pack files into LZMA-compressed zip archive and return base64 string."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_LZMA) as zf:
        for archive_name, content in sorted(files.items()):
            zf.writestr(archive_name, content)
    compressed_bytes = buf.getvalue()
    b64_str = base64.b64encode(compressed_bytes).decode("ascii")
    print(f"Compressed {len(files)} files: {len(compressed_bytes)} bytes LZMA -> {len(b64_str)} chars b64")
    return b64_str


def make_code_cell(source: list[str], cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source,
    }


def make_markdown_cell(source: list[str], cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def generate_notebook() -> pathlib.Path:
    """Generate the complete Kaggle submission notebook."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    payload_files = collect_payload_files()
    payload_b64 = build_lzma_payload(payload_files)

    cells = [
        make_markdown_cell([
            "# ARC Prize 2026: ARC-AGI-3 LCLD Agent Version 10.0\n",
            "\n",
            "**Architecture**: Neuro-Symbolic Tri-Agent (Explorer, DSL Coder, Solver) with Brusentsov Ternary Logic,\n",
            "Isolated Memory Contours (ISO-1..ISO-5), Deterministic ARGALite Perception, and Tufa Single-RESET Protection.\n",
            "**Model**: Qwen 3.8 27B (`Qwen/Qwen3.8-27B`, `rahim3/qwen3-8-27b-bf16`) via vLLM with FlashAttention & MTP=3 Speculative Acceleration.\n",
            f"**Configuration**: Concurrency=4, Reasoning=xhigh, Context=64K..131K, MTP=3 (Graceful Fallback), HeavySmoke={ENABLE_PHASE_A_HEAVY_SMOKE}.\n",
        ], "a1b2c3d0"),
        make_code_cell([
            "# =============================================================================\n",
            "# CELL 1: OFFLINE COMPETITION RUNTIME INSTALLATION\n",
            "# =============================================================================\n",
            "import subprocess, sys, os, pathlib\n",
            "\n",
            "os.environ.setdefault('MPLBACKEND', 'Agg')\n",
            "os.environ['HF_HUB_OFFLINE'] = '1'\n",
            "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n",
            "cuda_lib = '/usr/local/nvidia/lib64'\n",
            "existing_lib = [e for e in os.environ.get('LIBRARY_PATH', '').split(os.pathsep) if e]\n",
            "if os.path.isdir(cuda_lib) and cuda_lib not in existing_lib:\n",
            "    os.environ['LIBRARY_PATH'] = os.pathsep.join([cuda_lib, *existing_lib])\n",
            "\n",
            "print('=== Installing ARC-AGI competition wheels ===', flush=True)\n",
            "# Install competition runtime wheels (arc-agi, arcengine)\n",
            "# Must be installed strictly from the competition directory with --no-deps\n",
            "# to prevent overriding pre-installed Kaggle packages (such as Pillow).\n",
            "candidate_comp_dirs = [\n",
            "    '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',\n",
            "    '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',\n",
            "]\n",
            "comp_dir = next((p for p in candidate_comp_dirs if os.path.isdir(p)), None)\n",
            "if comp_dir:\n",
            "    print(f'Found competition wheels directory: {comp_dir}', flush=True)\n",
            "    for pkg in ['arcengine', 'arc-agi']:\n",
            "        cmd = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', f'--find-links={comp_dir}', pkg]\n",
            "        res = subprocess.run(cmd, capture_output=True, text=True)\n",
            "        if res.returncode == 0:\n",
            "            print(f'[OK] Installed {pkg}', flush=True)\n",
            "        else:\n",
            "            print(f'Notice during {pkg} install: {res.stderr[-500:] if res.stderr else res.stdout[-500:]}', flush=True)\n",
            "else:\n",
            "    print('Notice: Competition wheels directory not found, assuming pre-installed.', flush=True)\n",
            "\n",
            "print('Environment initialization complete.', flush=True)\n",
        ], "e5f6a7b1"),
        make_code_cell([
            "# =============================================================================\n",
            "# CELL 2: UNPACK LCLD V10 AGENT PAYLOAD\n",
            "# =============================================================================\n",
            "import base64, io, zipfile, pathlib, sys\n",
            "\n",
            f"PAYLOAD_B64 = '{payload_b64}'\n",
            "\n",
            "DEPLOY_DIR = pathlib.Path('/tmp/arc_lcld_agent/Code')\n",
            "DEPLOY_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "zip_data = base64.b64decode(PAYLOAD_B64)\n",
            "with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:\n",
            "    zf.extractall(DEPLOY_DIR)\n",
            "\n",
            "if str(DEPLOY_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(DEPLOY_DIR))\n",
            "\n",
            "print(f'Successfully deployed {len(zip_data)} bytes to {DEPLOY_DIR}', flush=True)\n",
        ], "c9d0e1f2"),
        make_code_cell([
            "# =============================================================================\n",
            "# CELL 3: PHASE-A STRUCTURAL PREFLIGHT & SUBMISSION ARTIFACT ASSURANCE\n",
            "# =============================================================================\n",
            "import os, pathlib, json\n",
            "import pandas as pd\n",
            "import lcld_preflight\n",
            "\n",
            "# Run structural preflight test (deterministic offline verification)\n",
            "lcld_preflight.run_preflight()\n",
            "\n",
            "is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in ('1', 'true')\n",
            "print(f'KAGGLE_IS_COMPETITION_RERUN = {is_rerun}', flush=True)\n",
            "\n",
            "working_root = pathlib.Path('/kaggle/working')\n",
            "working_root.mkdir(parents=True, exist_ok=True)\n",
            "submission_path = working_root / 'submission.parquet'\n",
            "\n",
            "if not is_rerun:\n",
            "    # Phase A: Ensure submission.parquet exists for Kaggle evaluator commit validation\n",
            "    if not submission_path.exists():\n",
            "        dummy_submission = pd.DataFrame(\n",
            "            data=[['1_0', '1', True, 1]],\n",
            "            columns=['row_id', 'game_id', 'end_of_game', 'score'],\n",
            "        )\n",
            "        dummy_submission.to_parquet(submission_path, index=False)\n",
            "        print(f'Created required competition submission artifact at {submission_path}', flush=True)\n",
            "else:\n",
            "    # Phase B: Rerun mode. Remove any existing submission.parquet so failures are never masked!\n",
            "    if submission_path.exists():\n",
            "        try:\n",
            "            submission_path.unlink()\n",
            "            print('[Phase B] Removed preexisting submission.parquet to prevent masking runtime failures.', flush=True)\n",
            "        except Exception as rm_exc:\n",
            "            print(f'[Phase B] Notice removing existing submission.parquet: {rm_exc}', flush=True)\n",
        ], "a3b4c5d3"),
        make_code_cell([
            "# =============================================================================\n",
            "# CELL 4: PHASE-A HEAVY COMBAT SMOKE TEST (vLLM & QWEN-3.8-27B DIAGNOSTICS)\n",
            "# =============================================================================\n",
            f"ENABLE_PHASE_A_HEAVY_SMOKE = {ENABLE_PHASE_A_HEAVY_SMOKE}\n",
            "import os, pathlib, json\n",
            "import pandas as pd\n",
            "\n",
            "is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in ('1', 'true')\n",
            "working_root = pathlib.Path('/kaggle/working')\n",
            "submission_path = working_root / 'submission.parquet'\n",
            "\n",
            "if not is_rerun:\n",
            "    if ENABLE_PHASE_A_HEAVY_SMOKE:\n",
            "        print('=== Launching Phase-A Heavy Combat Smoke Diagnostics ===', flush=True)\n",
            "        try:\n",
            "            import phase_a_heavy_smoke\n",
            "            smoke_summary = phase_a_heavy_smoke.run_phase_a_smoke_pipeline()\n",
            "            print(f'Heavy smoke execution status: {smoke_summary.get(\"status\")}', flush=True)\n",
            "        except Exception as exc:\n",
            "            print(f'[HEAVY-SMOKE WARNING] Caught smoke exception: {exc}', flush=True)\n",
            "    else:\n",
            "        print('=== Phase-A Heavy Combat Smoke Disabled (ENABLE_PHASE_A_HEAVY_SMOKE=False) ===', flush=True)\n",
            "\n",
            "    # Always guarantee submission.parquet is written and verified\n",
            "    if not submission_path.exists():\n",
            "        dummy = pd.DataFrame(data=[['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score'])\n",
            "        dummy.to_parquet(submission_path, index=False)\n",
            "    print('=== LCLD PHASE A VALIDATION COMPLETE; SUBMISSION ARTIFACT READY ===', flush=True)\n",
            "else:\n",
            "    print('Phase A heavy smoke skipped: this execution is a Phase B competition rerun.', flush=True)\n",
        ], "e7f8a9b4"),
        make_code_cell([
            "# =============================================================================\n",
            "# CELL 5: PHASE-B GATEWAY & CONCURRENT GAMEPLAY EXECUTION (RERUN ONLY)\n",
            "# =============================================================================\n",
            "import os, sys, signal, time, pathlib, subprocess, json, urllib.request, urllib.error, traceback\n",
            "import pandas as pd\n",
            "\n",
            "is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in ('1', 'true')\n",
            "working_root = pathlib.Path('/kaggle/working')\n",
            "submission_path = working_root / 'submission.parquet'\n",
            "\n",
            "if not is_rerun:\n",
            "    print('=== Phase B competition execution skipped (Phase A commit/dry-run mode). ===', flush=True)\n",
            "else:\n",
            "    # --- Stdout/Stderr tee to file for log preservation (Fix #9) ---\n",
            "    class _Tee:\n",
            "        def __init__(self, *streams):\n",
            "            self._streams = streams\n",
            "        def write(self, data):\n",
            "            n = 0\n",
            "            for s in self._streams:\n",
            "                try: n = s.write(data)\n",
            "                except Exception: pass\n",
            "            return n\n",
            "        def flush(self):\n",
            "            for s in self._streams:\n",
            "                try: s.flush()\n",
            "                except Exception: pass\n",
            "        def isatty(self): return False\n",
            "\n",
            "    _phase_b_log = open(working_root / 'phase_b.log', 'w', buffering=1, encoding='utf-8')\n",
            "    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr\n",
            "    sys.stdout = _Tee(_orig_stdout, _phase_b_log)\n",
            "    sys.stderr = _Tee(_orig_stderr, _phase_b_log)\n",
            "\n",
            "    print('=================================================================', flush=True)\n",
            "    print('=== STARTING PHASE B ISOLATED COMPETITION RUNTIME ===', flush=True)\n",
            "    print('=================================================================', flush=True)\n",
            "\n",
            "    phase_b_success = False\n",
            "    phase_b_error = None\n",
            "\n",
            "    try:\n",
            "        # 1. Setup arcade client environment and write .env\n",
            "        base_url = 'http://gateway:8001'\n",
            "        env_path = working_root / '.env'\n",
            "        arcade_settings = {\n",
            "            'SCHEME': 'http',\n",
            "            'HOST': 'gateway',\n",
            "            'PORT': '8001',\n",
            "            'ARC_API_KEY': 'test-key-123',\n",
            "            'ARC_API_BASE': base_url,\n",
            "            'ARC_BASE_URL': base_url,\n",
            "            'OPERATION_MODE': 'competition',\n",
            "            'ENVIRONMENTS_DIR': '',\n",
            "            'RECORDINGS_DIR': str(working_root / 'server_recording'),\n",
            "            'LCLD_MAX_ACTIONS_PER_GAME': '250',\n",
            "            'LCLD_MAX_ACTIONS_PER_LEVEL': '250',\n",
            "            'LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS': '5000',\n",
            "            'LCLD_GAME_CONCURRENCY': '4',\n",
            "            'LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS': '30600',\n",
            "            'ARC_ENABLE_PRIMITIVE_PROBING': '1',\n",
            "            'ARC_MAX_PRIMITIVE_PROBES': '30',\n",
            "            'ARC_ENABLE_SYMBOLIC_FALLBACK': '1',\n",
            "            'ARC_CODER_EXHAUSTION_FORCES_FALLBACK': '1',\n",
            "            'ARC_SOLVER_EXHAUSTION_FORCES_FALLBACK': '1',\n",
            "            'ARC_ABORT_ON_DSL_EXHAUSTION': '0',\n",
            "            'ARC_MAX_CHAIN_ATTEMPTS': '5',\n",
            "            'ARC_VLLM_MTP_ENABLED': '1',\n",
            "            'ARC_VLLM_MTP_TOKENS': '3',\n",
            "            'ARC_VLLM_SPECULATIVE_METHOD': 'mtp',\n",
            "            'VLLM_MTP_TOKENS': '3',\n",
            "            'VLLM_SPECULATIVE_TOKENS': '3',\n",
            "            'VLLM_SPEC_METHOD': 'mtp',\n",
            "        }\n",
            "        os.environ.update(arcade_settings)\n",
            "        env_path.write_text('\\n'.join(f'{k}={v}' for k, v in arcade_settings.items()) + '\\n', encoding='utf-8')\n",
            "        print(f'[Phase B] Written gateway configuration to {env_path}', flush=True)\n",
            "\n",
            "        # 2. Kill zombie vLLM processes from Phase A (Fix #5)\n",
            "        import phase_a_heavy_smoke\n",
            "        phase_a_heavy_smoke.stop_vllm_server()\n",
            "        try:\n",
            "            subprocess.run(['pkill', '-f', 'vllm.entrypoints'], capture_output=True, timeout=10)\n",
            "        except Exception:\n",
            "            pass\n",
            "\n",
            "        # 3. Install vLLM wheelhouse into /kaggle/working/vllm-site-packages\n",
            "        wheelhouse = phase_a_heavy_smoke.find_wheelhouse_path()\n",
            "        if not wheelhouse:\n",
            "            raise FileNotFoundError('vLLM wheelhouse dataset not found in /kaggle/input')\n",
            "        site_packages = phase_a_heavy_smoke.install_vllm_wheelhouse(wheelhouse)\n",
            "\n",
            "        # 4. Locate model weights\n",
            "        model_path = phase_a_heavy_smoke.find_model_path()\n",
            "        if not model_path:\n",
            "            raise FileNotFoundError('Qwen-27B model weights not found in /kaggle/input')\n",
            "\n",
            "        # 5. Start vLLM server with logging\n",
            "        ready = phase_a_heavy_smoke.start_vllm_server(model_path, site_packages)\n",
            "        if not ready:\n",
            "            print('=== vLLM SERVER LOG TAIL (Startup Failure) ===', flush=True)\n",
            "            print(phase_a_heavy_smoke._vllm_log_tail(30000), flush=True)\n",
            "            raise RuntimeError('vLLM server failed to start within timeout')\n",
            "        # 6. Fast model contract smoke probe before competition scorecard\n",
            "        phase_a_heavy_smoke.phase_b_model_smoke_or_die()\n",
            "\n",
            "        # 7. SIGINT/SIGTERM handler for graceful shutdown (Fix #7)\n",
            "        def _phase_b_sigint_handler(signum, frame):\n",
            "            print(f'[Phase B] Received signal {signum} — initiating graceful shutdown...', flush=True)\n",
            "            try:\n",
            "                phase_a_heavy_smoke.stop_vllm_server()\n",
            "            except Exception:\n",
            "                pass\n",
            "            sys.exit(1)\n",
            "        for _s in (getattr(signal, 'SIGINT', None), getattr(signal, 'SIGTERM', None)):\n",
            "            if _s is not None:\n",
            "                try: signal.signal(_s, _phase_b_sigint_handler)\n",
            "                except Exception: pass\n",
            "\n",
            "        # 8. Gateway handshake check\n",
            "        print('[Phase B] Checking gateway connectivity at http://gateway:8001/api/games...', flush=True)\n",
            "        deadline = time.monotonic() + 700.0\n",
            "        gateway_ready = False\n",
            "        while time.monotonic() < deadline:\n",
            "            try:\n",
            "                req = urllib.request.Request(\n",
            "                    'http://gateway:8001/api/games',\n",
            "                    headers={'Accept': 'application/json', 'X-API-Key': os.environ.get('ARC_API_KEY', '')},\n",
            "                )\n",
            "                with urllib.request.urlopen(req, timeout=10) as r:\n",
            "                    if 200 <= r.status < 300:\n",
            "                        print(f'[Phase B] Gateway handshake OK (status={r.status})', flush=True)\n",
            "                        gateway_ready = True\n",
            "                        break\n",
            "            except urllib.error.HTTPError as he:\n",
            "                if 200 <= he.code < 500:\n",
            "                    print(f'[Phase B] Gateway handshake OK via HTTPError (code={he.code})', flush=True)\n",
            "                    gateway_ready = True\n",
            "                    break\n",
            "            except Exception:\n",
            "                time.sleep(4.0)\n",
            "        if not gateway_ready:\n",
            "            raise RuntimeError('[Phase B] FATAL: Kaggle gateway did not become ready within 700s!')\n",
            "\n",
            "        # 9. Execute games concurrently\n",
            "        from arc_agi import Arcade, OperationMode\n",
            "        from lcld_competition_child import run_concurrent_arcade_games\n",
            "        arcade = Arcade(\n",
            "            operation_mode=OperationMode.COMPETITION,\n",
            "            arc_base_url=base_url,\n",
            "            arc_api_key=os.environ.get('ARC_API_KEY', 'test-key-123'),\n",
            "            environments_dir='',\n",
            "        )\n",
            "        print(f'[Phase B] Arcade initialized: mode={arcade.operation_mode}, url={arcade.arc_base_url}', flush=True)\n",
            "        results = run_concurrent_arcade_games(arcade, concurrency=4)\n",
            "        print(f'[Phase B] Completed gameplay across {len(results)} environments.', flush=True)\n",
            "        total_actions = sum(int(r.get('action_count', 0) or 0) for r in results)\n",
            "        print(f'[Phase B] Total accepted actions across all games: {total_actions}', flush=True)\n",
            "        if total_actions <= 0:\n",
            "            raise RuntimeError('[Phase B] FATAL: Zero actions were accepted by the competition gateway!')\n",
            "        phase_b_success = True\n",
            "    except Exception as run_exc:\n",
            "        phase_b_error = run_exc\n",
            "        print(f'[Phase B ERROR] Fatal exception during Phase B runtime: {run_exc}', flush=True)\n",
            "        traceback.print_exc()\n",
            "        try:\n",
            "            print('=== vLLM SERVER LOG TAIL (Post-Error) ===', flush=True)\n",
            "            print(phase_a_heavy_smoke._vllm_log_tail(30000), flush=True)\n",
            "        except Exception:\n",
            "            pass\n",
            "    finally:\n",
            "        try:\n",
            "            phase_a_heavy_smoke.stop_vllm_server()\n",
            "        except Exception:\n",
            "            pass\n",
            "        print('=== PHASE B WORKFLOW TEARDOWN ===', flush=True)\n",
            "        # Restore stdout/stderr (Fix #9)\n",
            "        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr\n",
            "        try:\n",
            "            _phase_b_log.flush()\n",
            "            _phase_b_log.close()\n",
            "        except Exception:\n",
            "            pass\n",
            "        print('=== Phase B log saved to /kaggle/working/phase_b.log ===', flush=True)\n",
            "        sys.stdout.flush()\n",
            "        sys.stderr.flush()\n",
            "        time.sleep(1.0)\n",
            "        if not phase_b_success:\n",
            "            print(f'[Phase B FATAL] Terminating with exit code 1 due to error: {phase_b_error}', file=sys.stderr, flush=True)\n",
            "            os._exit(1)\n",
            "        else:\n",
            "            print('=== Phase B completed successfully. Terminating with exit code 0 ===', flush=True)\n",
            "            os._exit(0)\n",
            "    # End of Phase B\n",
            "\n",
            "print('Notebook execution complete.', flush=True)\n",
        ], "c1d2e3f5"),
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaRtxPro6000",
                "dataSources": [
                    {
                        "sourceType": "competition",
                        "sourceId": 133468,
                    },
                    {
                        "sourceType": "datasetVersion",
                        "sourceId": 19036547,
                    },
                    {
                        "sourceType": "datasetVersion",
                        "sourceId": 11671619,
                    },
                    {
                        "sourceType": "modelInstanceVersion",
                        "sourceId": 960909,
                    },
                ],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": False,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    raw_json = json.dumps(notebook_data, indent=1)
    raw_bytes = len(raw_json.encode("utf-8"))
    print(f"Generated notebook total size: {raw_bytes} bytes (cap: {MAX_NOTEBOOK_BYTES})")
    if raw_bytes >= MAX_NOTEBOOK_BYTES:
        raise ValueError(f"Notebook size {raw_bytes} exceeded cap of {MAX_NOTEBOOK_BYTES}")

    OUTPUT_NOTEBOOK.write_text(raw_json, encoding="utf-8")
    print(f"Wrote notebook to {OUTPUT_NOTEBOOK}")

    # Remove obsolete legacy muse notebook if present
    legacy_muse = NOTEBOOKS_DIR / "arc-prize-2026-lcld-muse-v10.ipynb"
    if legacy_muse.exists():
        legacy_muse.unlink()
        print(f"Removed legacy notebook {legacy_muse}")

    # Update kernel-metadata.json to guarantee Kaggle CLI push works out of the box
    metadata = {
        "id": "vladimiryakunin/arc-prize-2026-lcld-qwen-v10",
        "title": "ARC Prize 2026 - LCLD Qwen V10",
        "code_file": OUTPUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "keywords": [],
        "dataset_sources": [
            "vladimiryakunin/vllm-028-cuda",
            "rahim3/qwen3-8-27b-bf16",
        ],
        "kernel_sources": [],
        "competition_sources": [
            "arc-prize-2026-arc-agi-3",
        ],
        "model_sources": [],
    }
    KERNEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Updated Kaggle kernel metadata at {KERNEL_METADATA_PATH}")

    return OUTPUT_NOTEBOOK


if __name__ == "__main__":
    generate_notebook()
