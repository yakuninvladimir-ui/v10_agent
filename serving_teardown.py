"""Guaranteed release of ports and sockets on exit with process-identity verification.

Ported and enhanced from Flash-Next serving teardown protocol for ARC-AGI-3 V10.0 runtime.
Ensures zero zombie vLLM/Python processes hold serving ports, prevents signaling unverified
PIDs (protects unrelated or system processes), and writes standard teardown JSON report.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any

# Allowed substrings in process command line to authorize termination.
# Any process on the serving port whose cmdline does NOT match these is protected.
AUTHORIZED_IDENTITY_KEYWORDS = (
    "vllm",
    "python",
    "uvicorn",
    "multiprocessing",
    "ray",
    "torch",
    "api_server",
)


def get_default_serving_port() -> int:
    """Resolve default serving port from environment or fallback to 8000 (or 1234)."""
    env_port = os.getenv("VLLM_PORT")
    if env_port and env_port.isdigit():
        return int(env_port)
    base_url = os.getenv("ARC_VLLM_BASE_URL", os.getenv("VLLM_BASE_URL", ""))
    if base_url:
        m = re.search(r":(\d+)", base_url)
        if m:
            return int(m.group(1))
    return 8000


def get_process_cmdline(pid: int) -> str:
    """Retrieve process command line for identity verification."""
    if pid <= 0:
        return ""

    system = platform.system().lower()
    if system == "linux":
        cmdline_path = pathlib.Path(f"/proc/{pid}/cmdline")
        if cmdline_path.is_file():
            try:
                raw = cmdline_path.read_bytes()
                # cmdline arguments are null-byte separated
                return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            except (OSError, PermissionError):
                pass
        comm_path = pathlib.Path(f"/proc/{pid}/comm")
        if comm_path.is_file():
            try:
                return comm_path.read_text(encoding="utf-8", errors="replace").strip()
            except (OSError, PermissionError):
                pass
        # Fallback to ps
        try:
            res = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass

    elif system == "windows":
        try:
            res = subprocess.run(
                ["wmic", "process", "where", f"processid={pid}", "get", "commandline"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                if len(lines) > 1:
                    return lines[1]
        except Exception:
            pass

    return ""


def verify_process_identity(pid: int) -> bool:
    """Verify that PID belongs to an authorized serving process before signaling.

    Returns:
        True if process is verified as vLLM/Python server or does not exist.
        False if process exists but its commandline belongs to an unauthorized identity.
    """
    if pid <= 0:
        return False
    # Never target the current process or init
    if pid in (os.getpid(), 1):
        return False

    cmdline = get_process_cmdline(pid).lower()
    if not cmdline:
        # Process might already have terminated
        return True

    return any(keyword in cmdline for keyword in AUTHORIZED_IDENTITY_KEYWORDS)


def get_pids_on_port(port: int) -> list[int]:
    """Find all listening process PIDs on the specified TCP port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.05)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                # Port is not listening; no processes to find
                return []
    except Exception:
        pass

    pids: set[int] = set()
    system = platform.system().lower()

    if system == "linux":
        # 1. ss tool
        try:
            res = subprocess.run(
                ["ss", "-lptn", f"sport = :{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for match in re.finditer(r"pid=(\d+)", res.stdout):
                    pids.add(int(match.group(1)))
        except Exception:
            pass

        # 2. lsof tool
        if not pids:
            try:
                res = subprocess.run(
                    ["lsof", f"-ti:{port}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        line = line.strip()
                        if line.isdigit():
                            pids.add(int(line))
            except Exception:
                pass

        # 3. fuser tool
        if not pids:
            try:
                res = subprocess.run(
                    ["fuser", f"{port}/tcp"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for part in res.stdout.split() + res.stderr.split():
                    part = part.strip()
                    if part.isdigit():
                        pids.add(int(part))
            except Exception:
                pass

    elif system == "windows":
        try:
            res = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                        pid_str = parts[4]
                        if pid_str.isdigit():
                            pids.add(int(pid_str))
        except Exception:
            pass

    return sorted(pids)


def terminate_process_tree(pid: int, timeout_seconds: float = 10.0) -> bool:
    """Gracefully terminate a process tree (SIGTERM -> timeout -> SIGKILL)."""
    if pid <= 0 or pid in (os.getpid(), 1):
        return False

    system = platform.system().lower()
    if system == "windows":
        try:
            res = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return res.returncode == 0
        except Exception:
            return False

    # POSIX / Linux: Find child processes first
    child_pids: list[int] = []
    try:
        res = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    child_pids.append(int(line))
    except Exception:
        pass

    target_pids = child_pids + [pid]

    # Step 1: SIGTERM
    for p in target_pids:
        try:
            os.kill(p, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    # Wait up to half timeout
    deadline = time.monotonic() + (timeout_seconds / 2.0)
    while time.monotonic() < deadline:
        still_running = False
        for p in target_pids:
            try:
                os.kill(p, 0)
                still_running = True
                break
            except (ProcessLookupError, OSError):
                pass
        if not still_running:
            return True
        time.sleep(0.2)

    # Step 2: SIGKILL remaining
    for p in target_pids:
        try:
            os.kill(p, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    return True


def wait_for_port_closed(
    port: int,
    host: str = "127.0.0.1",
    timeout_seconds: float = 10.0,
) -> bool:
    """Verify that the socket/port is completely closed and can be re-bound."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        # 1. Connection probe: should fail
        s_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_conn.settimeout(0.05)
        conn_res = s_conn.connect_ex((host, port))
        s_conn.close()

        # If connection was refused (nonzero), verify bind capability
        if conn_res != 0:
            try:
                s_bind = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s_bind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s_bind.bind((host, port))
                s_bind.close()
                return True
            except OSError:
                pass

        time.sleep(0.05)

    return False


def teardown_serving(
    port: int | None = None,
    working_dir: pathlib.Path | str | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    """Execute process-identity verified teardown of vLLM serving stack.

    Writes /kaggle/working/vllm-server-teardown.json with status fields:
      - shutdown_ok: bool
      - port_closed: bool
      - identity_valid: bool
      - signal_blocked_by_identity_conflict: bool

    Returns:
        dict containing the teardown report status.
    """
    resolved_port = port if port is not None else get_default_serving_port()

    if working_dir is None:
        target_dir_str = os.getenv(
            "TAAF_KAGGLE_WORKING_DIR",
            os.getenv("LCLD_WORKING_ROOT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "."),
        )
        resolved_working_dir = pathlib.Path(target_dir_str).resolve()
    else:
        resolved_working_dir = pathlib.Path(working_dir).resolve()

    resolved_working_dir.mkdir(parents=True, exist_ok=True)
    report_file = resolved_working_dir / "vllm-server-teardown.json"

    pids = get_pids_on_port(resolved_port)
    if pid is not None and pid > 0 and pid not in pids:
        pids.append(pid)

    identity_valid = True
    signal_blocked_by_identity_conflict = False
    targeted_pids: list[int] = []

    for p in pids:
        if not verify_process_identity(p):
            print(
                f"[SERVING-TEARDOWN WARNING] Process identity conflict on port {resolved_port}: "
                f"PID {p} ({get_process_cmdline(p)}) does not match authorized serving profiles. "
                "Blocking signal transmission to prevent terminating unauthorized process.",
                flush=True,
            )
            identity_valid = False
            signal_blocked_by_identity_conflict = True
        else:
            targeted_pids.append(p)
            terminate_process_tree(p, timeout_seconds=8.0)

    # Optional hook: trigger phase_a_heavy_smoke stop if present
    try:
        import phase_a_heavy_smoke
        phase_a_heavy_smoke.stop_vllm_server()
    except Exception:
        pass

    # Verify port is closed
    port_closed = wait_for_port_closed(resolved_port, timeout_seconds=8.0)
    shutdown_ok = bool(port_closed and not signal_blocked_by_identity_conflict)

    report: dict[str, Any] = {
        "shutdown_ok": shutdown_ok,
        "port_closed": port_closed,
        "identity_valid": identity_valid,
        "signal_blocked_by_identity_conflict": signal_blocked_by_identity_conflict,
        "port": resolved_port,
        "pids_targeted": targeted_pids,
        "timestamp": time.time(),
    }

    try:
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[SERVING-TEARDOWN] Wrote teardown report to {report_file}: {report}", flush=True)
    except Exception as io_err:
        print(f"[SERVING-TEARDOWN] Notice: Could not write teardown report to {report_file}: {io_err}", flush=True)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flash-Next Process-Identity Serving Teardown")
    parser.add_argument("--port", type=int, default=None, help="Serving TCP port to release")
    parser.add_argument("--working-dir", type=str, default=None, help="Working directory for report output")
    parser.add_argument("--pid", type=int, default=None, help="Explicit server PID if known")
    args = parser.parse_args()

    res = teardown_serving(port=args.port, working_dir=args.working_dir, pid=args.pid)
    sys.exit(0 if res.get("shutdown_ok", False) else 1)
