"""Background health watchdog and automated recovery daemon for vLLM serving stack.

Ported and enhanced from Flash-Next vllm_server_watchdog for ARC-AGI-3 V10.0 runtime.
Features:
- Periodic health probe every 15.0 seconds (customizable).
- HTTP /health and /v1/models contract checks with configurable request timeout (5.0s).
- Automatic restart on failure threshold (4 consecutive failures) up to 2 times (max_restart_attempts = 2).
- Non-blocking sliced sleep allowing immediate shutdown via stop_background() without waiting 15s.
- Automatic integration with serving_teardown for guaranteed socket and process-identity cleanup.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

# Local imports with fallback
try:
    from v10_agent import serving_setup
except ImportError:
    import serving_setup  # type: ignore

try:
    from v10_agent import serving_teardown
except ImportError:
    import serving_teardown  # type: ignore


@dataclasses.dataclass
class WatchdogConfig:
    """Configuration parameters for vLLM Server Watchdog."""

    interval_seconds: float = 15.0
    request_timeout_seconds: float = 5.0
    failure_threshold: int = 4
    max_restart_attempts: int = 2
    startup_grace_seconds: float = 120.0
    health_endpoint: str | None = None
    models_endpoint: str | None = None
    host: str = "127.0.0.1"
    port: int | None = None


class ServerWatchdog:
    """Daemon thread actively monitoring vLLM server health with auto-restart recovery."""

    def __init__(
        self,
        setup: Any,
        config: WatchdogConfig | None = None,
    ) -> None:
        self.setup = setup
        self.config = config or WatchdogConfig()

        # Resolve host and port
        if self.config.port is None:
            self.config.port = getattr(self.setup, "port", None) or serving_teardown.get_default_serving_port()
        if not self.config.host:
            self.config.host = getattr(self.setup, "host", "127.0.0.1")

        # Resolve health endpoints
        if not self.config.health_endpoint:
            self.config.health_endpoint = f"http://{self.config.host}:{self.config.port}/health"
        if not self.config.models_endpoint:
            self.config.models_endpoint = f"http://{self.config.host}:{self.config.port}/v1/models"

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._restart_attempts: int = 0
        self._consecutive_failures: int = 0
        self._is_healthy: bool = True
        self._is_running: bool = False
        self._last_probe_timestamp: float = 0.0
        self._last_status_code: int | None = None
        self._lock = threading.RLock()

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            return self._is_healthy

    @property
    def restart_attempts(self) -> int:
        with self._lock:
            return self._restart_attempts

    def probe_health(self) -> bool:
        """Perform a single HTTP GET health check against the server."""
        now = time.monotonic()
        self._last_probe_timestamp = now

        # Probe 1: /health endpoint
        endpoints = [self.config.health_endpoint, self.config.models_endpoint]
        for url in endpoints:
            if not url:
                continue
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Accept": "application/json", "User-Agent": "VllmServerWatchdog/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
                    status = getattr(resp, "status", 200)
                    self._last_status_code = status
                    if 200 <= status < 300:
                        return True
            except urllib.error.HTTPError as http_err:
                self._last_status_code = http_err.code
                # /health might 404 on some vLLM builds, in which case we fall back to /models
                if http_err.code == 404 and url == self.config.health_endpoint:
                    continue
                # If /v1/models returns 401 or 403, server is running but requires auth: consider running
                if http_err.code in (401, 403):
                    return True
                return False
            except Exception:
                # Connection refused, timeout, network error
                return False

        return False

    def _run_loop(self) -> None:
        """Main monitoring loop executing every interval_seconds."""
        print(
            f"[WATCHDOG] Started background monitor on {self.config.health_endpoint} "
            f"(interval={self.config.interval_seconds}s, threshold={self.config.failure_threshold}, "
            f"max_restarts={self.config.max_restart_attempts})",
            flush=True,
        )

        while not self._stop_event.is_set():
            # Non-blocking sliced sleep: check stop_event every 0.25-0.5s so stop_background()
            # can terminate cleanly without waiting the entire 15s.
            deadline = time.monotonic() + self.config.interval_seconds
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    return
                time.sleep(min(0.25, max(0.05, deadline - time.monotonic())))

            # Perform probe
            healthy = self.probe_health()

            # Also check if underlying process has crashed
            if healthy and hasattr(self.setup, "is_alive") and not self.setup.is_alive():
                print("[WATCHDOG] Process alive check failed despite probe success.", flush=True)
                healthy = False

            should_restart = False
            with self._lock:
                if healthy:
                    if self._consecutive_failures > 0:
                        print(
                            f"[WATCHDOG] Server health restored to OK (was failing {self._consecutive_failures}x).",
                            flush=True,
                        )
                    self._consecutive_failures = 0
                    self._is_healthy = True
                else:
                    self._consecutive_failures += 1
                    self._is_healthy = False
                    print(
                        f"[WATCHDOG WARNING] Health probe failed ({self._consecutive_failures}/{self.config.failure_threshold}) "
                        f"on port {self.config.port} (status={self._last_status_code})",
                        flush=True,
                    )

                    if self._consecutive_failures >= self.config.failure_threshold:
                        if self._restart_attempts < self.config.max_restart_attempts:
                            self._restart_attempts += 1
                            print(
                                f"[WATCHDOG RESTART] Failure threshold exceeded ({self._consecutive_failures}). "
                                f"Triggering automatic recovery #{self._restart_attempts}/{self.config.max_restart_attempts}...",
                                flush=True,
                            )
                            should_restart = True
                        else:
                            print(
                                f"[WATCHDOG FATAL] Maximum restart attempts ({self.config.max_restart_attempts}) "
                                "exhausted. No further automatic restarts will be attempted.",
                                flush=True,
                            )

            if should_restart:
                self._execute_restart()

    def _execute_restart(self) -> None:
        """Execute safe teardown followed by server restart and health stabilization."""
        working_dir = getattr(self.setup, "working_dir", pathlib.Path("."))
        try:
            print(f"[WATCHDOG RESTART] Tearing down serving stack on port {self.config.port}...", flush=True)
            serving_teardown.teardown_serving(port=self.config.port, working_dir=working_dir)
        except Exception as td_err:
            print(f"[WATCHDOG RESTART WARNING] Teardown encountered error: {td_err}", flush=True)

        # Short cooldown before re-launching, respecting stop_event and config
        pause_seconds = min(1.0, max(0.05, self.config.startup_grace_seconds / 2.0))
        deadline_pause = time.monotonic() + pause_seconds
        while time.monotonic() < deadline_pause:
            if self._stop_event.is_set():
                return
            time.sleep(min(0.05, deadline_pause - time.monotonic()))

        # Launch server
        try:
            print(f"[WATCHDOG RESTART] Re-launching server on port {self.config.port}...", flush=True)
            if hasattr(self.setup, "start_server"):
                self.setup.start_server()
            else:
                print("[WATCHDOG RESTART ERROR] Setup object has no start_server method!", flush=True)
                return
        except Exception as start_err:
            print(f"[WATCHDOG RESTART ERROR] Failed to start server: {start_err}", flush=True)
            return

        # Wait for recovery up to startup_grace_seconds
        recovery_deadline = time.monotonic() + self.config.startup_grace_seconds
        recovered = False
        print(f"[WATCHDOG RESTART] Waiting up to {self.config.startup_grace_seconds}s for server recovery...", flush=True)

        while time.monotonic() < recovery_deadline:
            if self._stop_event.is_set():
                return
            if self.probe_health():
                recovered = True
                break
            time.sleep(min(0.1, max(0.02, recovery_deadline - time.monotonic())))

        with self._lock:
            if recovered:
                print(
                    f"[WATCHDOG RESTART] Server successfully recovered and healthy on port {self.config.port}!",
                    flush=True,
                )
                self._consecutive_failures = 0
                self._is_healthy = True
            else:
                print(
                    f"[WATCHDOG RESTART ERROR] Server did not become healthy within {self.config.startup_grace_seconds}s.",
                    flush=True,
                )

    def start(self) -> ServerWatchdog:
        """Start the watchdog background thread."""
        with self._lock:
            if self._is_running:
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="VllmServerWatchdogDaemon", daemon=True)
            self._is_running = True
            self._thread.start()
        return self

    def stop(self, timeout_seconds: float = 15.0) -> None:
        """Signal stop event and join the background thread."""
        with self._lock:
            if not self._is_running:
                return
            self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)

        with self._lock:
            self._is_running = False
        print("[WATCHDOG] Background monitor stopped.", flush=True)

    def get_status(self) -> dict[str, Any]:
        """Return snapshot dictionary of current watchdog status."""
        with self._lock:
            return {
                "running": self._is_running,
                "healthy": self._is_healthy,
                "restarts": self._restart_attempts,
                "max_restarts": self.config.max_restart_attempts,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.config.failure_threshold,
                "interval_seconds": self.config.interval_seconds,
                "port": self.config.port,
                "last_status_code": self._last_status_code,
            }


# Module singleton
_CURRENT_WATCHDOG: ServerWatchdog | None = None
_MODULE_LOCK = threading.Lock()


def load_setup(source: Any) -> Any:
    """Load or construct a ServingSetup instance from file path, dict, or module.

    Compatible with:
        watchdog_setup = watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
    """
    if source is None:
        return serving_setup.create_setup()

    if isinstance(source, serving_setup.ServingSetup):
        return source

    if isinstance(source, (str, pathlib.Path)):
        p = pathlib.Path(source).resolve()
        if p.is_file() and p.suffix == ".py":
            module_name = f"dynamic_serving_setup_{int(time.time())}"
            spec = importlib.util.spec_from_file_location(module_name, str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "setup"):
                    return getattr(mod, "setup")
                if hasattr(mod, "ServingSetup"):
                    cls = getattr(mod, "ServingSetup")
                    return cls()
                if hasattr(mod, "create_setup"):
                    fn = getattr(mod, "create_setup")
                    return fn()
        elif p.is_file() and p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return serving_setup.create_setup(**data)
            except Exception:
                pass

    if isinstance(source, dict):
        return serving_setup.create_setup(**source)

    if hasattr(source, "start_server"):
        return source

    return serving_setup.create_setup()


def start_background(
    setup: Any,
    config: WatchdogConfig | None = None,
) -> ServerWatchdog:
    """Start the global background watchdog daemon monitoring the given setup."""
    global _CURRENT_WATCHDOG
    with _MODULE_LOCK:
        if _CURRENT_WATCHDOG is not None:
            _CURRENT_WATCHDOG.stop(timeout_seconds=5.0)

        resolved_setup = load_setup(setup)
        watchdog = ServerWatchdog(resolved_setup, config)
        watchdog.start()
        _CURRENT_WATCHDOG = watchdog
        return watchdog


def stop_background(timeout_seconds: float = 15.0) -> None:
    """Stop the global background watchdog daemon."""
    global _CURRENT_WATCHDOG
    with _MODULE_LOCK:
        if _CURRENT_WATCHDOG is not None:
            _CURRENT_WATCHDOG.stop(timeout_seconds=timeout_seconds)
            _CURRENT_WATCHDOG = None


def is_healthy() -> bool:
    """Check if the monitored vLLM server is currently healthy."""
    if _CURRENT_WATCHDOG is not None:
        return _CURRENT_WATCHDOG.is_healthy
    return False


def get_status() -> dict[str, Any]:
    """Retrieve status report from current watchdog instance."""
    if _CURRENT_WATCHDOG is not None:
        return _CURRENT_WATCHDOG.get_status()
    return {"running": False, "healthy": False, "restarts": 0}
