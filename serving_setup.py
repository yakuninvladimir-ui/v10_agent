"""Serving setup and process lifecycle configuration for Flash-Next vLLM runtime.

Provides the ServingSetup configuration abstraction consumed by vllm_server_watchdog
and competition execution harnesses. Supports automated vLLM server launch, health probes,
process management, and parameter resolution.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Callable


class ServingSetup:
    """Flash-Next compatible server execution specification and lifecycle manager."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        command: list[str] | str | None = None,
        env: dict[str, str] | None = None,
        working_dir: pathlib.Path | str = ".",
        log_file: pathlib.Path | str | None = None,
        pid_file: pathlib.Path | str | None = None,
        start_fn: Callable[[], Any] | None = None,
        stop_fn: Callable[[], Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else self._resolve_port()
        self.command = command
        self.env = env
        self.working_dir = pathlib.Path(working_dir).resolve()
        self.log_file = pathlib.Path(log_file).resolve() if log_file else None
        self.pid_file = pathlib.Path(pid_file).resolve() if pid_file else None
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def _resolve_port() -> int:
        """Resolve serving port from environment or fallback to 8000 (or 1234)."""
        env_port = os.getenv("VLLM_PORT")
        if env_port and env_port.isdigit():
            return int(env_port)
        base_url = os.getenv("ARC_VLLM_BASE_URL", os.getenv("VLLM_BASE_URL", ""))
        if base_url:
            m = re.search(r":(\d+)", base_url)
            if m:
                return int(m.group(1))
        return 8000

    def get_health_url(self) -> str:
        """Return the /health endpoint URL."""
        return f"http://{self.host}:{self.port}/health"

    def get_models_url(self) -> str:
        """Return the /v1/models endpoint URL."""
        return f"http://{self.host}:{self.port}/v1/models"

    def start_server(self) -> Any:
        """Start or restart the serving process using configured command or callable."""
        print(f"[SERVING-SETUP] Starting server on {self.host}:{self.port}...", flush=True)

        # 1. Custom start function (e.g. phase_a_heavy_smoke.start_vllm_server)
        if self.start_fn is not None:
            try:
                res = self.start_fn()
                if isinstance(res, subprocess.Popen):
                    self._proc = res
                return res
            except Exception as fn_err:
                print(f"[SERVING-SETUP ERROR] start_fn raised exception: {fn_err}", flush=True)
                raise

        # 2. Command list or string
        if self.command is not None:
            cmd = self.command
            log_handle = None
            if self.log_file is not None:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = self.log_file.open("a", encoding="utf-8")

            run_env = os.environ.copy()
            if self.env:
                run_env.update(self.env)

            proc = subprocess.Popen(
                cmd,
                shell=isinstance(cmd, str),
                cwd=str(self.working_dir),
                env=run_env,
                stdout=log_handle if log_handle else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
                text=True,
            )
            self._proc = proc

            if self.pid_file is not None:
                try:
                    self.pid_file.parent.mkdir(parents=True, exist_ok=True)
                    self.pid_file.write_text(str(proc.pid), encoding="utf-8")
                except OSError:
                    pass

            return proc

        # 3. Fallback to phase_a_heavy_smoke start_vllm_server if importable
        try:
            import phase_a_heavy_smoke
            model_path = phase_a_heavy_smoke.find_model_path()
            site_packages = phase_a_heavy_smoke.get_vllm_site_packages()
            if model_path and site_packages:
                print("[SERVING-SETUP] Delegating to phase_a_heavy_smoke.start_vllm_server...", flush=True)
                ok = phase_a_heavy_smoke.start_vllm_server(model_path, site_packages)
                self._proc = getattr(phase_a_heavy_smoke, "_vllm_proc", None)
                return ok
        except Exception as smoke_err:
            print(f"[SERVING-SETUP Notice] phase_a_heavy_smoke integration unavailable: {smoke_err}", flush=True)

        raise RuntimeError("No server command, start_fn, or phase_a_heavy_smoke runner available in ServingSetup.")

    def stop_server(self) -> None:
        """Stop the serving process using configured stop hook or teardown."""
        if self.stop_fn is not None:
            try:
                self.stop_fn()
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def is_alive(self) -> bool:
        """Check whether the underlying process is currently running."""
        if self._proc is not None:
            return self._proc.poll() is None
        return True


def create_setup(
    host: str = "127.0.0.1",
    port: int | None = None,
    **kwargs: Any,
) -> ServingSetup:
    """Factory helper to construct a ServingSetup instance."""
    return ServingSetup(host=host, port=port, **kwargs)


# Module-level default setup instance (consumed by watchdog.load_setup)
setup = create_setup()
