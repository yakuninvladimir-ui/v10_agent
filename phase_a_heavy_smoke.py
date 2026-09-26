"""Phase A Heavy Combat Smoke test for vLLM & Muse-Glimmer-30B on Kaggle GPU.

Reproduces deep combat diagnostics in Phase A Save Version log:
1. Verifies GPU via nvidia-smi (VRAM usage before and after).
2. Installs offline vLLM wheelhouse into isolated /kaggle/working/vllm-site-packages.
3. Launches local vLLM OpenAI-compatible server on 127.0.0.1:1234.
4. Executes diagnostic probes:
   - Probe 1: Text decoding with reasoning channel validation.
   - Probe 2: Multimodal vision (ARC grid PNG sent as base64 image).
   - Probe 3: 4 concurrent workers matching VLLM_MAX_NUM_SEQS = 4.
5. Emits vLLM server log tail (last 30KB) to stdout for public debuggability.
6. Cleanly stops vLLM server so Phase A exits 0 and writes submission.parquet.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.request import Request, urlopen

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_HEALTH_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
VLLM_MAX_NUM_SEQS = int(os.environ.get("LCLD_VLLM_MAX_NUM_SEQS", "6"))
VLLM_STARTUP_TIMEOUT_SECONDS = 900
HEAVY_SMOKE_REQ_TIMEOUT = 600

_vllm_proc: subprocess.Popen | None = None
_vllm_log_file: Any = None


def get_working_root() -> pathlib.Path:
    p = pathlib.Path(os.environ.get("LCLD_WORKING_ROOT", "/kaggle/working")).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_vllm_site_packages() -> pathlib.Path:
    p = get_working_root() / "vllm-site-packages"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_vllm_log_path() -> pathlib.Path:
    return get_working_root() / "vllm.log"


def _smoke_gpu_info() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi unavailable (CPU/non-CUDA runtime)"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        return (r.stdout or "").strip() or "nvidia-smi returned empty"
    except Exception as exc:
        return f"nvidia-smi probe failed: {exc}"


def find_wheelhouse_path() -> pathlib.Path | None:
    candidates = [
        pathlib.Path("/kaggle/input/vllm-028-cuda"),
        pathlib.Path("/kaggle/input/datasets/vladimiryakunin/vllm-028-cuda"),
        pathlib.Path("/kaggle/input/vllm-028-cuda-wheels"),
        pathlib.Path("/kaggle/input/datasets/vladimiryakunin/vllm-028-cuda-wheels"),
        pathlib.Path("/kaggle/input/vllm-027-cuda-wheels"),
        pathlib.Path("/kaggle/input/datasets/vladimiryakunin/vllm-027-cuda-wheels"),
        pathlib.Path("/kaggle/input/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3"),
        pathlib.Path("/kaggle/input/arc3-vllm-h100-wheelhouse-v3"),
        pathlib.Path("/kaggle/input/arc3-vllm-wheelhouse"),
    ]
    for c in candidates:
        if c.is_dir():
            # Check if wheels are directly inside c
            try:
                if any(f.name.endswith(".whl") for f in c.iterdir() if f.is_file()):
                    return c.resolve()
            except Exception:
                pass
            # Check subdirectories of c
            for root, _, files in os.walk(c):
                if any("vllm" in f.lower() and f.endswith(".whl") for f in files):
                    return pathlib.Path(root).resolve()
            return c.resolve()
    # Scan /kaggle/input for any dir containing vllm wheel
    input_dir = pathlib.Path("/kaggle/input")
    if input_dir.is_dir():
        for root, _, files in os.walk(input_dir):
            if any("vllm" in f.lower() and f.endswith(".whl") for f in files):
                return pathlib.Path(root).resolve()
    return None


def patch_vllm_muse_reasoning_parser(site_packages: pathlib.Path) -> bool:
    """Patch vLLM's MuseGlimmerReasoningParser to implement reasoning_start_str and reasoning_end_str.

    Prevents vLLM Auto-initialization warning and enables token ID initialization.
    """
    candidates = list(site_packages.glob("**/muse_glimmer_reasoning_parser.py"))
    if not candidates:
        return False
    patched_any = False
    for p in candidates:
        try:
            code = p.read_text(encoding="utf-8")
            if "def reasoning_start_str" in code:
                continue
            target = "class MuseGlimmerReasoningParser(ReasoningParser):"
            if target not in code:
                continue
            patch = (
                "class MuseGlimmerReasoningParser(ReasoningParser):\n"
                "    @property\n"
                "    def reasoning_start_str(self) -> str | None:\n"
                "        return _REASONING_OPEN\n\n"
                "    @property\n"
                "    def reasoning_end_str(self) -> str | None:\n"
                "        return _EOM\n"
            )
            code = code.replace(target, patch, 1)
            p.write_text(code, encoding="utf-8")
            print(f"[HEAVY-SMOKE] Successfully patched {p.name} with reasoning token strings.", flush=True)
            patched_any = True
        except Exception as exc:
            print(f"[HEAVY-SMOKE] Warning: failed to patch {p}: {exc}", flush=True)
    return patched_any


def install_vllm_wheelhouse(wheelhouse_dir: pathlib.Path) -> pathlib.Path:
    site_packages = get_vllm_site_packages()
    stamp_file = site_packages / ".vllm_installed_stamp"
    if stamp_file.is_file():
        print(f"[HEAVY-SMOKE] Reusing existing vLLM target at {site_packages}", flush=True)
        patch_vllm_muse_reasoning_parser(site_packages)
        return site_packages

    req_file = wheelhouse_dir / "requirements.lock"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-index", f"--find-links={wheelhouse_dir}",
        "--target", str(site_packages),
        "--upgrade", "--ignore-installed", "--only-binary", ":all:",
        "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts",
    ]
    if req_file.is_file():
        cmd.extend(["--requirement", str(req_file)])
    else:
        cmd.append("vllm")

    print(f"[HEAVY-SMOKE] Installing vLLM wheelhouse from {wheelhouse_dir} into {site_packages}...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(f"[HEAVY-SMOKE] Pip install warning ({res.returncode}): {res.stderr[-1000:]}", flush=True)
    patch_vllm_muse_reasoning_parser(site_packages)
    stamp_file.write_text(f"installed_from={wheelhouse_dir}\n", encoding="utf-8")
    return site_packages


def find_model_path() -> pathlib.Path | None:
    # 1. Explicit env var if set and non-empty
    env_var = (
        os.environ.get("ARC_QWEN_MODEL_PATH", "")
        or os.environ.get("ARC_MODEL_PATH", "")
        or os.environ.get("ARC_MUSE_MODEL_PATH", "")
        or os.environ.get("ARC_LLM_MODEL_PATH", "")
    ).strip()
    if env_var:
        p = pathlib.Path(env_var).resolve()
        if p.is_dir() and (p / "config.json").is_file():
            print(f"[HEAVY-SMOKE] Using model path from environment variable: {p}", flush=True)
            return p

    # 2. Check candidate paths with config.json validation (prioritize Qwen3.8-27B)
    candidates = [
        pathlib.Path("/kaggle/input/qwen3-8-27b-bf16"),
        pathlib.Path("/kaggle/input/datasets/rahim3/qwen3-8-27b-bf16"),
        pathlib.Path("/kaggle/input/models/rahim3/qwen3-8-27b-bf16"),
        pathlib.Path("/kaggle/input/qwen3-8-27b-bf16/transformers/bf16/1"),
        pathlib.Path("/kaggle/input/qwen3-8-27b-bf16/1"),
        pathlib.Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pyTorch/hf-fp8/1"),
        pathlib.Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"),
        pathlib.Path("/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/hf-fp8/1"),
        pathlib.Path("/kaggle/input/qwen3-8-27b-fp8-repacked-v1"),
        pathlib.Path("/kaggle/input/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1"),
        pathlib.Path("/kaggle/input/datasets/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1"),
        pathlib.Path("/kaggle/input/vrfai-qwen3-6-27b-fp8-hf-snapshot"),
        pathlib.Path("/kaggle/input/models/noillum123/muse-glimmer-30b/transformers/bf16/1"),
        pathlib.Path("/kaggle/input/muse-glimmer-30b"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").is_file():
            print(f"[HEAVY-SMOKE] Found model at candidate path: {c}", flush=True)
            return c.resolve()

    # 3. Dynamic search across /kaggle/input for directory containing config.json
    input_dir = pathlib.Path("/kaggle/input")
    if input_dir.is_dir():
        for root, dirs, files in os.walk(input_dir):
            if "config.json" in files:
                r_path = pathlib.Path(root).resolve()
                path_str = str(r_path).lower()
                has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
                if has_weights and ("qwen3" in path_str or "qwen" in path_str):
                    print(f"[HEAVY-SMOKE] Dynamically discovered Qwen model weights at: {r_path}", flush=True)
                    return r_path
                if has_weights:
                    print(f"[HEAVY-SMOKE] Dynamically discovered model weights at: {r_path}", flush=True)
                    return r_path

    print("[HEAVY-SMOKE] Notice: No directory containing config.json was found under /kaggle/input", flush=True)
    return None


def get_vllm_env(site_packages: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(site_packages) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    cuda_lib = "/usr/local/nvidia/lib64"
    existing_lib = [e for e in env.get("LIBRARY_PATH", "").split(os.pathsep) if e]
    if os.path.isdir(cuda_lib) and cuda_lib not in existing_lib:
        env["LIBRARY_PATH"] = os.pathsep.join([cuda_lib, *existing_lib])

    env.update({
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_XGRAMMAR_CACHE_MB": "64",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "700",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Critical Blackwell / RTX 6000 Ada workaround (from v9):
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })

    # MTP & Speculative decoding environment variables
    vllm_mtp_enabled = os.getenv("ARC_VLLM_MTP_ENABLED", os.getenv("VLLM_MTP_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
    if vllm_mtp_enabled:
        mtp_tokens = os.getenv("ARC_VLLM_MTP_TOKENS") or os.getenv("VLLM_MTP_TOKENS") or os.getenv("VLLM_SPECULATIVE_TOKENS") or "3"
        spec_method = os.getenv("ARC_VLLM_SPECULATIVE_METHOD") or os.getenv("VLLM_SPEC_METHOD") or os.getenv("VLLM_SPECULATIVE_METHOD") or "mtp"
        env.update({
            "VLLM_MTP_TOKENS": str(mtp_tokens),
            "VLLM_SPECULATIVE_TOKENS": str(mtp_tokens),
            "VLLM_SPEC_METHOD": str(spec_method),
            "VLLM_SPECULATIVE_METHOD": str(spec_method),
        })
        spec_model = os.getenv("ARC_VLLM_SPECULATIVE_MODEL") or os.getenv("VLLM_SPEC_MODEL") or os.getenv("VLLM_SPECULATIVE_MODEL")
        if spec_model:
            env["VLLM_SPECULATIVE_MODEL"] = str(spec_model)
            env["VLLM_SPEC_MODEL"] = str(spec_model)
        spec_cfg = os.getenv("ARC_VLLM_SPECULATIVE_CONFIG") or os.getenv("VLLM_SPECULATIVE_CONFIG")
        if spec_cfg:
            env["VLLM_SPECULATIVE_CONFIG"] = str(spec_cfg)

    return env


def start_vllm_server(
    model_path: pathlib.Path,
    site_packages: pathlib.Path,
    enable_mtp: bool | None = None,
) -> bool:
    global _vllm_proc, _vllm_log_file
    if enable_mtp is None:
        enable_mtp = os.getenv("ARC_VLLM_MTP_ENABLED", os.getenv("VLLM_MTP_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")

    log_path = get_vllm_log_path()
    _vllm_log_file = log_path.open("w", encoding="utf-8")

    try:
        from v10_agent.config import build_vllm_server_flags, config_from_env
        _cfg = config_from_env()
        server_flags = build_vllm_server_flags(_cfg)
    except Exception:
        server_flags = [
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--no-enable-log-requests",
            "--disable-uvicorn-access-log",
        ]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", "Qwen/Qwen3.8-27B",
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--max-num-seqs", str(VLLM_MAX_NUM_SEQS),
        "--max-num-batched-tokens", "8192",
        "--limit-mm-per-prompt", json.dumps({"image": 4}),
        "--generation-config", "auto",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        *server_flags,
        "--mm-processor-cache-gb", "0",
        "--gpu-memory-utilization", "0.95",
        "--attention-backend", "FLASH_ATTN",
        "--default-chat-template-kwargs", json.dumps({"reasoning_effort": "xhigh", "enable_thinking": True}),
        "--max-model-len", "131072",
        "--trust-remote-code",
    ]

    if enable_mtp:
        try:
            from v10_agent.config import build_vllm_speculative_args, config_from_env
            cfg = config_from_env()
            cfg.vllm_mtp_enabled = True
            spec_args = build_vllm_speculative_args(cfg, model_path=model_path)
            cmd.extend(spec_args)
            print(f"[HEAVY-SMOKE] MTP=3 speculative decoding requested: {' '.join(spec_args)}", flush=True)
        except Exception as spec_exc:
            print(f"[HEAVY-SMOKE] Notice: could not load speculative args from config: {spec_exc}", flush=True)

    print(f"[HEAVY-SMOKE] Starting vLLM server: {' '.join(cmd)}", flush=True)
    _vllm_proc = subprocess.Popen(
        cmd,
        stdout=_vllm_log_file,
        stderr=subprocess.STDOUT,
        env=get_vllm_env(site_packages),
        text=True,
    )

    # Wait for ready up to 900 seconds
    started = time.monotonic()
    deadline = started + VLLM_STARTUP_TIMEOUT_SECONDS
    print(f"[HEAVY-SMOKE] Waiting for vLLM ready signal on {VLLM_HOST}:{VLLM_PORT} (timeout={VLLM_STARTUP_TIMEOUT_SECONDS}s)...", flush=True)

    while time.monotonic() < deadline:
        if _vllm_proc.poll() is not None:
            retcode = _vllm_proc.returncode
            print(f"[HEAVY-SMOKE] vLLM process exited prematurely with code {retcode}!", flush=True)
            tail = _vllm_log_tail(10000)
            print(f"[HEAVY-SMOKE] vLLM startup failure log tail:\n{tail}", flush=True)
            stop_vllm_server()

            if enable_mtp:
                print(
                    "[HEAVY-SMOKE] NOTICE: vLLM startup failed with MTP speculative decoding enabled. "
                    "Initiating graceful fallback: RESTARTING vLLM WITHOUT MTP speculative decoding...",
                    flush=True,
                )
                return start_vllm_server(model_path, site_packages, enable_mtp=False)
            return False
        try:
            with urlopen(f"{VLLM_BASE_URL}/models", timeout=3) as resp:
                if resp.status == 200:
                    elapsed = round(time.monotonic() - started, 1)
                    print(f"[HEAVY-SMOKE] vLLM server ready in {elapsed}s!", flush=True)
                    return True
        except Exception:
            time.sleep(4.0)

    print("[HEAVY-SMOKE] vLLM server startup timed out!", flush=True)
    stop_vllm_server()
    if enable_mtp:
        print(
            "[HEAVY-SMOKE] NOTICE: vLLM startup timed out with MTP enabled. "
            "Initiating graceful fallback: RESTARTING vLLM WITHOUT MTP speculative decoding...",
            flush=True,
        )
        return start_vllm_server(model_path, site_packages, enable_mtp=False)
    return False


def stop_vllm_server() -> None:
    global _vllm_proc, _vllm_log_file
    if _vllm_proc is not None and _vllm_proc.poll() is None:
        print("[HEAVY-SMOKE] Stopping vLLM server process...", flush=True)
        try:
            _vllm_proc.terminate()
            _vllm_proc.wait(timeout=15)
        except Exception:
            try:
                _vllm_proc.kill()
                _vllm_proc.wait(timeout=5)
            except Exception:
                pass
    _vllm_proc = None
    if _vllm_log_file is not None:
        try:
            _vllm_log_file.close()
        except Exception:
            pass
        _vllm_log_file = None


def _vllm_log_tail(limit: int = 30000) -> str:
    log_path = get_vllm_log_path()
    if not log_path.is_file():
        return "<no vllm.log found>"
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit), os.SEEK_SET)
            return f.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"<error reading vllm.log: {exc}>"


def _smoke_send(label: str, payload: dict[str, Any], timeout: int = HEAVY_SMOKE_REQ_TIMEOUT) -> dict[str, Any]:
    rec: dict[str, Any] = {"label": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    clean_payload = {k: v for k, v in payload.items() if not k.startswith("_")}
    started = time.monotonic()
    req = Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=json.dumps(clean_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        rec["elapsed"] = round(time.monotonic() - started, 2)
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)[:1000]
        return rec

    rec["elapsed"] = round(time.monotonic() - started, 2)
    try:
        data = json.loads(body)
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        msg = choice.get("message") or {}
        content = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning") or msg.get("reasoning_content") or "")
        usage = data.get("usage") or {}

        try:
            json.loads(content)
            is_json = True
        except Exception:
            is_json = False

        rec.update({
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_chars": len(reasoning),
            "content_chars": len(content),
            "content_is_json": is_json,
            "content_preview": content[:200] if len(content) > 200 else content,
            "content": content,
        })
    except Exception as exc:
        rec["response_parse_error"] = str(exc)
        rec["body_tail"] = body[-500:]

    return rec


def _build_multimodal_smoke_payload(model_id: str, config: Any | None = None) -> dict[str, Any]:
    """Create sample ARC Solver multimodal payload with dual-view PNG, planning objects, and grid_hash."""
    from v10_agent.config import V10Config
    from v10_agent.arga_lite import extract_arga_snapshot
    from v10_agent.planning_set import build_planning_set
    from v10_agent.frame_media import render_dual_frame_png
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts

    cfg = config or V10Config(
        model_path=model_id,
        solver_multimodal_enabled=True,
        multimodal_enabled=True,
        reasoning_strength="xhigh",
        enable_thinking=True,
    )

    sample_grid = [
        [0, 1, 1, 0, 0, 2],
        [0, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4, 4],
    ]
    snapshot = extract_arga_snapshot(sample_grid)
    planning_set = build_planning_set(
        snapshot=snapshot,
        available_actions=["ACTION1", "ACTION2", "ACTION6", "RESET"],
    )

    dual_png = render_dual_frame_png(sample_grid, planning_set, scale=16)
    raw_b64 = base64.b64encode(dual_png).decode("ascii")

    manifest = {
        "schema_version": "v10.dsl_manifest.1",
        "functions": [
            {
                "name": "action1",
                "parameters": [],
                "returns": "effect_declaration",
                "docstring": "Declare discrete button action ACTION1 (UP).",
            },
            {
                "name": "action2",
                "parameters": [],
                "returns": "effect_declaration",
                "docstring": "Declare discrete button action ACTION2 (DOWN).",
            },
            {
                "name": "action6",
                "parameters": [
                    {"name": "x", "type": "int", "default": 0},
                    {"name": "y", "type": "int", "default": 0},
                ],
                "returns": "effect_declaration",
                "docstring": "Declare spatial action ACTION6 at target coordinates (x, y).",
            },
        ],
    }

    has_image = getattr(cfg, "solver_multimodal_enabled", True) and bool(dual_png)
    sys_prompt, user_prompt = build_solver_prompts(
        manifest=manifest,
        planning_set=planning_set,
        has_image=has_image,
    )

    user_content: list[dict[str, Any]] = []
    if has_image:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{raw_b64}"},
        })
    user_content.append({
        "type": "text",
        "text": user_prompt,
    })

    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": getattr(cfg, "temperature", 1.0),
        "top_p": getattr(cfg, "top_p", 0.95),
        "top_k": getattr(cfg, "top_k", 20),
        "max_tokens": getattr(cfg, "max_output_tokens", 8192) or 8192,
        "chat_template_kwargs": {"reasoning_effort": "xhigh", "enable_thinking": True},
        "_meta": {
            "planning_set": planning_set,
            "manifest": manifest,
            "png_bytes": dual_png,
            "has_image": has_image,
            "grid_hash": planning_set.grid_hash,
        },
    }


def run_phase_a_smoke_pipeline() -> dict[str, Any]:
    """Execute complete Phase-A Heavy Combat Smoke pipeline."""
    print("=================================================================", flush=True)
    print("=== LCLD V10 PHASE-A HEAVY COMBAT SMOKE TEST START ===", flush=True)
    print("=================================================================", flush=True)

    summary: dict[str, Any] = {"status": "started", "gpu_before": _smoke_gpu_info()}
    print(f"[HEAVY-SMOKE] GPU status before: {summary['gpu_before']}", flush=True)

    wheelhouse = find_wheelhouse_path()
    if not wheelhouse:
        print("[HEAVY-SMOKE] vLLM wheelhouse dataset not found; skipping heavy smoke.", flush=True)
        summary["status"] = "skipped_no_wheelhouse"
        return summary

    model_path = find_model_path()
    if not model_path:
        print("[HEAVY-SMOKE] Qwen-27B model weights not found; skipping heavy smoke.", flush=True)
        summary["status"] = "skipped_no_model"
        return summary

    try:
        site_packages = install_vllm_wheelhouse(wheelhouse)
        ready = start_vllm_server(model_path, site_packages)
        if not ready:
            summary["status"] = "vllm_startup_failed"
            print("=== vLLM SERVER LOG TAIL (Startup Failure) ===", flush=True)
            print(_vllm_log_tail(30000), flush=True)
            return summary

        model_id = "Qwen/Qwen3.8-27B"

        # --- Probe 1: Text baseline with reasoning ---
        print("\n[HEAVY-SMOKE] Executing Probe 1: Text Baseline with Reasoning...", flush=True)
        probe1_payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Think step by step before answering."},
                {"role": "user", "content": "Count from 1 to 20, one number per line."},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 2048,
            "chat_template_kwargs": {"reasoning_effort": "xhigh", "enable_thinking": True},
        }
        p1_res = _smoke_send("probe1_text_baseline", probe1_payload, timeout=HEAVY_SMOKE_REQ_TIMEOUT)
        print(f"[HEAVY-SMOKE] Probe 1 result: {json.dumps(p1_res, indent=2)}", flush=True)
        summary["probe1"] = p1_res

        # --- Probe 2: Multimodal vision (Solver Pipeline with Dual-View PNG) ---
        print("\n[HEAVY-SMOKE] Executing Probe 2: Solver Multimodal Vision (Dual-View PNG & Invariants)...", flush=True)
        probe2_payload = _build_multimodal_smoke_payload(model_id)
        p2_meta = probe2_payload.get("_meta", {})
        print(
            f"[HEAVY-SMOKE] Probe 2 prepared Solver payload: dual_png={len(p2_meta.get('png_bytes', b''))} bytes, "
            f"grid_hash={p2_meta.get('grid_hash', '')[:8]}, objects={len(p2_meta.get('planning_set').objects if p2_meta.get('planning_set') else [])}",
            flush=True,
        )
        p2_res = _smoke_send("probe2_solver_multimodal_vision", probe2_payload, timeout=HEAVY_SMOKE_REQ_TIMEOUT)
        content_text = p2_res.get("content", "")
        if content_text:
            from v10_agent.solver_agent import extract_json_block, parse_text_trajectory
            pkg = extract_json_block(content_text)
            if not pkg or not isinstance(pkg, dict) or not pkg.get("candidates"):
                pkg = parse_text_trajectory(content_text, p2_meta.get("manifest", {}), p2_meta.get("planning_set"))
            if pkg and pkg.get("candidates"):
                cand_count = len(pkg["candidates"])
                steps_preview = [s.get("dsl_function") for s in pkg["candidates"][0].get("steps", [])[:6]]
                p2_res["parsed_candidates_count"] = cand_count
                p2_res["first_candidate_steps_preview"] = steps_preview
                print(f"[HEAVY-SMOKE] Probe 2 Solver parsed successfully: {cand_count} candidates; Candidate 1 steps: {steps_preview}", flush=True)
            else:
                print("[HEAVY-SMOKE] Probe 2 Solver response did not yield valid candidate trajectories upon parsing.", flush=True)
        print(f"[HEAVY-SMOKE] Probe 2 result: {json.dumps({k: v for k, v in p2_res.items() if k != 'content'}, indent=2)}", flush=True)
        summary["probe2"] = p2_res

        # --- Probe 3: Concurrency test (4 workers in parallel) ---
        print(f"\n[HEAVY-SMOKE] Executing Probe 3: Parallelism ({VLLM_MAX_NUM_SEQS} concurrent requests)...", flush=True)
        concurrent_payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Think step by step before answering."},
                {"role": "user", "content": "Return a 3-step arithmetic progression starting at 7."},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 4096,
            "chat_template_kwargs": {"reasoning_effort": "xhigh", "enable_thinking": True},
        }
        with ThreadPoolExecutor(max_workers=VLLM_MAX_NUM_SEQS, thread_name_prefix="smoke_worker") as executor:
            futures = [
                executor.submit(_smoke_send, f"worker_{i}", concurrent_payload, timeout=HEAVY_SMOKE_REQ_TIMEOUT)
                for i in range(VLLM_MAX_NUM_SEQS)
            ]
            concurrent_results = [f.result() for f in futures]
        print(f"[HEAVY-SMOKE] Probe 3 concurrent results: {json.dumps(concurrent_results, indent=2)}", flush=True)
        summary["probe3_concurrency"] = concurrent_results

        summary["status"] = "success"
        summary["gpu_after"] = _smoke_gpu_info()
        print(f"\n[HEAVY-SMOKE] GPU status after: {summary['gpu_after']}", flush=True)

        print("\n=== vLLM SERVER LOG TAIL (Active Combat Session) ===", flush=True)
        print(_vllm_log_tail(30000), flush=True)

    except Exception as exc:
        print(f"[HEAVY-SMOKE] Pipeline encountered exception: {exc}", flush=True)
        summary["status"] = "error"
        summary["error"] = str(exc)
        print("=== vLLM SERVER LOG TAIL (Post-Exception) ===", flush=True)
        print(_vllm_log_tail(30000), flush=True)
    finally:
        stop_vllm_server()

    print("=================================================================", flush=True)
    print("=== LCLD V10 PHASE-A HEAVY COMBAT SMOKE TEST COMPLETE ===", flush=True)
    print("=================================================================", flush=True)
    return summary


def phase_b_model_smoke_or_die() -> dict[str, Any]:
    """Execute a rapid, lightweight model contract smoke probe before creating scorecard in Phase B.

    Validates:
      1. Server is responding on http://127.0.0.1:1234/v1/models
      2. Fast chat completion query with thinking tokens returns valid content
    If this fails, dumps vLLM log tail and raises RuntimeError immediately.
    """
    print("=== LCLD Phase B Model Smoke Probe START ===", flush=True)
    started = time.monotonic()

    # 1. Check models endpoint
    try:
        with urlopen(f"{VLLM_BASE_URL}/models", timeout=5) as resp:
            models_data = json.loads(resp.read().decode("utf-8"))
            data_list = models_data.get("data", [])
            model_id = data_list[0]["id"] if data_list else "Qwen/Qwen3.8-27B"
            print(f"[Phase B Smoke] Discovered active model: {model_id}", flush=True)
    except Exception as exc:
        print(f"[Phase B Smoke FATAL] Could not reach vLLM /models: {exc}", flush=True)
        print(_vllm_log_tail(30000), flush=True)
        raise RuntimeError(f"Phase B model smoke failed at /models: {exc}") from exc

    # 2. Fast model contract probe (direct reply without thinking loop)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Reply with exactly OK."},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        req = Request(
            f"{VLLM_BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            if not choices:
                raise RuntimeError("Empty choices returned by vLLM")
            msg = choices[0].get("message", {})
            content = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            elapsed = round(time.monotonic() - started, 2)
            print(f"[Phase B Smoke OK] vLLM responded in {elapsed}s: content preview: {str(content)[:100]!r}", flush=True)
            return {"status": "ok", "elapsed": elapsed, "model": model_id}
    except Exception as exc:
        print(f"[Phase B Smoke FATAL] Production chat probe failed: {exc}", flush=True)
        print(_vllm_log_tail(30000), flush=True)
        raise RuntimeError(f"Phase B model smoke failed at /chat/completions: {exc}") from exc


if __name__ == "__main__":
    run_phase_a_smoke_pipeline()
