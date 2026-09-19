"""Comprehensive unit tests for vLLM server watchdog and serving teardown protocol."""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any
import pytest

from v10_agent import serving_setup, serving_teardown, vllm_server_watchdog
from v10_agent.serving_setup import ServingSetup
from v10_agent.serving_teardown import (
    teardown_serving,
    verify_process_identity,
    wait_for_port_closed,
    AUTHORIZED_IDENTITY_KEYWORDS,
)
from v10_agent.vllm_server_watchdog import (
    ServerWatchdog,
    WatchdogConfig,
    load_setup,
    start_background,
    stop_background,
)


def find_free_port() -> int:
    """Find an available ephemeral TCP port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class MockHealthHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler simulating vLLM /health and /v1/models endpoints."""

    status_to_return = 200

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/models"):
            self.send_response(self.status_to_return)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress logging in test output
        pass


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


@pytest.fixture
def mock_vllm_server():
    """Spin up a mock HTTP server responding to /health and /v1/models."""
    port = find_free_port()
    handler = MockHealthHandler
    handler.status_to_return = 200
    server = ThreadedTCPServer(("127.0.0.1", port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield port, handler
    finally:
        server.shutdown()
        server.server_close()


def test_watchdog_config_defaults():
    """Verify default parameters match competition requirements (15s interval, 2 restarts)."""
    cfg = WatchdogConfig()
    assert cfg.interval_seconds == 15.0
    assert cfg.request_timeout_seconds == 5.0
    assert cfg.failure_threshold == 4
    assert cfg.max_restart_attempts == 2
    assert cfg.host == "127.0.0.1"


def test_serving_setup_abstraction(tmp_path: pathlib.Path):
    """Test ServingSetup creation, URL builders, and custom start/stop hooks."""
    called = {"start": 0, "stop": 0}

    def _start():
        called["start"] += 1
        return True

    def _stop():
        called["stop"] += 1

    setup = ServingSetup(
        host="127.0.0.1",
        port=8000,
        working_dir=tmp_path,
        start_fn=_start,
        stop_fn=_stop,
    )

    assert setup.get_health_url() == "http://127.0.0.1:8000/health"
    assert setup.get_models_url() == "http://127.0.0.1:8000/v1/models"
    assert setup.start_server() is True
    assert called["start"] == 1

    setup.stop_server()
    assert called["stop"] == 1


def test_watchdog_load_setup(tmp_path: pathlib.Path):
    """Test load_setup with dict, ServingSetup instance, and python file."""
    # 1. From dict
    s1 = load_setup({"port": 8888, "host": "127.0.0.1"})
    assert isinstance(s1, ServingSetup)
    assert s1.port == 8888

    # 2. From existing instance
    s2 = load_setup(s1)
    assert s2 is s1

    # 3. From python file
    py_file = tmp_path / "custom_setup.py"
    py_file.write_text(
        "from v10_agent.serving_setup import ServingSetup\n"
        "setup = ServingSetup(port=9999)\n",
        encoding="utf-8",
    )
    s3 = load_setup(py_file)
    assert isinstance(s3, ServingSetup)
    assert s3.port == 9999


def test_watchdog_healthy_probe(mock_vllm_server):
    """Test watchdog accurately probes healthy endpoint and maintains healthy status."""
    port, _ = mock_vllm_server
    setup = ServingSetup(port=port, host="127.0.0.1")
    cfg = WatchdogConfig(
        interval_seconds=0.1,
        request_timeout_seconds=1.0,
        failure_threshold=2,
        max_restart_attempts=2,
        port=port,
    )

    wd = ServerWatchdog(setup, cfg)
    assert wd.probe_health() is True

    wd.start()
    try:
        time.sleep(0.3)
        assert wd.is_healthy is True
        status = wd.get_status()
        assert status["running"] is True
        assert status["healthy"] is True
        assert status["restarts"] == 0
    finally:
        wd.stop(timeout_seconds=2.0)
        assert wd.get_status()["running"] is False


def test_watchdog_auto_restart_on_failure(tmp_path: pathlib.Path):
    """Test watchdog detects failures, triggers restart up to max_restart_attempts, and enforces limit."""
    port = find_free_port()
    restarts_attempted = 0

    def _mock_start():
        nonlocal restarts_attempted
        restarts_attempted += 1

    setup = ServingSetup(
        port=port,
        host="127.0.0.1",
        working_dir=tmp_path,
        start_fn=_mock_start,
    )
    cfg = WatchdogConfig(
        interval_seconds=0.1,
        request_timeout_seconds=0.5,
        failure_threshold=2,
        max_restart_attempts=2,
        startup_grace_seconds=0.2,
        port=port,
    )

    wd = ServerWatchdog(setup, cfg)
    wd.start()
    try:
        # Wait for 2 failures -> 1st restart -> 2 failures -> 2nd restart -> threshold exhausted
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and restarts_attempted < 2:
            time.sleep(0.05)
        assert wd.is_healthy is False
        assert wd.restart_attempts == 2
        assert restarts_attempted == 2
    finally:
        wd.stop(timeout_seconds=2.0)


def test_serving_teardown_process_identity():
    """Verify process identity safety checks prevent signaling unknown processes."""
    # Current python process should be recognized
    my_pid = os.getpid()
    # verify_process_identity returns False for current PID or init to prevent accidental suicide
    assert verify_process_identity(my_pid) is False
    assert verify_process_identity(1) is False

    # Invalid PIDs
    assert verify_process_identity(0) is False
    assert verify_process_identity(-10) is False


def test_serving_teardown_report_generation(tmp_path: pathlib.Path):
    """Verify teardown generates vllm-server-teardown.json with all 4 required keys."""
    free_port = find_free_port()
    report = teardown_serving(port=free_port, working_dir=tmp_path)

    # Required contract keys
    assert "shutdown_ok" in report
    assert "port_closed" in report
    assert "identity_valid" in report
    assert "signal_blocked_by_identity_conflict" in report

    assert report["shutdown_ok"] is True
    assert report["port_closed"] is True
    assert report["identity_valid"] is True
    assert report["signal_blocked_by_identity_conflict"] is False

    # File check
    report_file = tmp_path / "vllm-server-teardown.json"
    assert report_file.is_file()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["shutdown_ok"] is True
    assert data["port_closed"] is True
    assert data["identity_valid"] is True
    assert data["signal_blocked_by_identity_conflict"] is False


def test_wait_for_port_closed():
    """Verify wait_for_port_closed returns True immediately for an unallocated port."""
    free_port = find_free_port()
    assert wait_for_port_closed(free_port, timeout_seconds=1.0) is True


def test_global_start_and_stop_background(mock_vllm_server):
    """Test module-level start_background and stop_background functions."""
    port, _ = mock_vllm_server
    setup = ServingSetup(port=port, host="127.0.0.1")
    cfg = WatchdogConfig(
        interval_seconds=0.1,
        request_timeout_seconds=0.5,
        port=port,
    )

    wd = start_background(setup, cfg)
    try:
        time.sleep(0.2)
        assert vllm_server_watchdog.is_healthy() is True
        status = vllm_server_watchdog.get_status()
        assert status["running"] is True
    finally:
        stop_background(timeout_seconds=2.0)
        status = vllm_server_watchdog.get_status()
        assert status["running"] is False
