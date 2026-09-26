"""Unit tests for VisibleCycle loop recovery, deadline reserve, and vLLM server configuration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from v10_agent.config import V10Config
from v10_agent.cycle_detector import VisibleCycle, _hash_grid
from v10_agent.llm_advisor import VLLMAdvisor
from v10_agent.session import GameSession


# =============================================================================
# 1. VisibleCycle Unit Tests
# =============================================================================

def test_hash_grid_consistency():
    """Verify deterministic hash computation for 2D grids and edge cases."""
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2], [3, 5]]
    assert _hash_grid(g1) == _hash_grid(g2)
    assert _hash_grid(g1) != _hash_grid(g3)
    assert _hash_grid(None) == "none"
    assert _hash_grid("already_hash") == "already_hash"


def test_cycle_detector_validation():
    """Verify constructor threshold bounds checking."""
    with pytest.raises(ValueError):
        VisibleCycle(min_actions=5)  # < 8
    with pytest.raises(ValueError):
        VisibleCycle(max_period=0)  # < 1
    with pytest.raises(ValueError):
        VisibleCycle(min_cycles=1)  # < 2


def test_cycle_detector_period_1():
    """Verify period-1 cycle (stationary or self-loop: state A -> action -> state A)."""
    # min_actions=8, max_period=4, min_cycles=4
    detector = VisibleCycle(min_actions=8, max_period=4, min_cycles=4)
    g = [[1, 1], [1, 1]]

    cycle = None
    for step in range(8):
        cycle = detector.observe(g, "ACTION1", g)

    assert cycle is not None
    assert cycle["period"] == 1
    assert cycle["action_pattern"] == ["ACTION1"]
    assert cycle["repetitions"] >= 8


def test_cycle_detector_period_2():
    """Verify period-2 oscillation (state A -> A1 -> state B -> A2 -> state A)."""
    detector = VisibleCycle(min_actions=8, max_period=4, min_cycles=4)
    gA = [[1, 0], [0, 0]]
    gB = [[0, 1], [0, 0]]

    cycle = None
    # 4 repetitions of 2 actions = 8 actions
    for _ in range(4):
        detector.observe(gA, "ACTION1", gB)
        cycle = detector.observe(gB, "ACTION2", gA)

    assert cycle is not None
    assert cycle["period"] == 2
    assert cycle["action_pattern"] == ["ACTION1", "ACTION2"]
    assert cycle["repetitions"] >= 4


def test_cycle_detector_period_4():
    """Verify period-4 cycle (e.g. 4-step directional box loop)."""
    detector = VisibleCycle(min_actions=16, max_period=4, min_cycles=4)
    g0 = [[1, 0], [0, 0]]
    g1 = [[0, 1], [0, 0]]
    g2 = [[0, 0], [0, 1]]
    g3 = [[0, 0], [1, 0]]

    cycle = None
    for _ in range(4):
        detector.observe(g0, "ACTION1", g1)
        detector.observe(g1, "ACTION2", g2)
        detector.observe(g2, "ACTION3", g3)
        cycle = detector.observe(g3, "ACTION4", g0)

    assert cycle is not None
    assert cycle["period"] == 4
    assert cycle["action_pattern"] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def test_cycle_detector_non_contiguous_break():
    """Verify trace clears when transition is non-contiguous (e.g. board reset)."""
    detector = VisibleCycle(min_actions=8, max_period=4, min_cycles=4)
    gA = [[1, 0]]
    gB = [[0, 1]]
    gDisjoint = [[9, 9]]

    # Feed 3 repetitions (6 actions)
    for _ in range(3):
        detector.observe(gA, "ACTION1", gB)
        detector.observe(gB, "ACTION2", gA)

    # Now feed non-contiguous transition: before state is gDisjoint instead of gA
    detector.observe(gDisjoint, "ACTION1", gB)
    assert len(detector.trace) == 1  # Trace reset to single entry


# =============================================================================
# 2. V10Config Deadline Management Tests
# =============================================================================

def test_config_deadline_management():
    """Verify deadline setting, remaining time, and reserve threshold check."""
    cfg = V10Config()
    assert cfg.remaining_time_seconds() is None
    assert not cfg.is_deadline_exceeded()

    # Set 100 seconds from now
    cfg.set_deadline(100.0)
    rem = cfg.remaining_time_seconds()
    assert rem is not None and 95.0 <= rem <= 100.0
    assert not cfg.is_deadline_exceeded(reserve_seconds=15.0)

    # Set 10 seconds from now (below default 15s reserve)
    cfg.set_deadline(10.0)
    assert cfg.is_deadline_exceeded(reserve_seconds=15.0)
    assert not cfg.is_deadline_exceeded(reserve_seconds=5.0)


# =============================================================================
# 3. VLLMAdvisor Deadline Reserve Abort Tests
# =============================================================================

def test_vllm_advisor_deadline_exceeded_abort():
    """Verify VLLMAdvisor immediately aborts with '{}' if deadline reserve reached."""
    cfg = V10Config()
    cfg.set_deadline(10.0)
    cfg.deadline_reserve_seconds = 15.0

    advisor = VLLMAdvisor(base_url="http://127.0.0.1:9999/v1")
    # Should not attempt HTTP connection, immediately returning "{}"
    result = advisor.generate_chat([{"role": "user", "content": "hello"}], config=cfg)
    assert result == "{}"


# =============================================================================
# 4. GameSession Loop Recovery Integration Tests
# =============================================================================

def test_session_loop_recovery_intervention():
    """Verify GameSession detects visible cycle, severs candidate, and triggers reset."""
    cfg = V10Config()
    cfg.enable_cycle_detector = True
    cfg.cycle_detector_min_actions = 8
    cfg.cycle_detector_max_period = 2
    cfg.cycle_detector_min_cycles = 4
    cfg.cycle_detector_per_level_limit = 2

    session = GameSession(cfg)
    assert session.cycle_interventions_this_level == 0

    gA = [[1, 0], [0, 0]]
    gB = [[0, 1], [0, 0]]

    # Simulate 4 cycles of oscillating actions
    for i in range(4):
        # Step A -> B
        session.last_snapshot = MagicMock(grid=gA)
        session.pending_action = {"id": "ACTION1", "action_id": "ACTION1"}
        session.observe_action_result({"grid": gB, "state": "IN_PROGRESS"})

        # Step B -> A
        session.last_snapshot = MagicMock(grid=gB)
        session.pending_action = {"id": "ACTION2", "action_id": "ACTION2"}
        session.observe_action_result({"grid": gA, "state": "IN_PROGRESS"})

    # Cycle should have triggered on the 4th cycle
    assert session.cycle_interventions_this_level == 1
    assert session.replan_requested is True
    assert session.solver_reset_pending is True
    assert session.solver_reset_reason == "loop_recovery_cycle_reset"
    assert "Visible cycle detected" in session._last_failure_reason

    # Telemetry should reflect the intervention
    telem = session.harness_telemetry()
    assert telem["cycle_interventions_count"] == 1


# =============================================================================
# 5. vLLM Server Launch Arguments in phase_a_heavy_smoke.py
# =============================================================================

def test_vllm_server_launch_flags():
    """Verify build_vllm_server_flags and start_vllm_server have exactly the approved competition keys."""
    from v10_agent.config import V10Config, build_vllm_server_flags
    import phase_a_heavy_smoke
    import inspect

    cfg = V10Config()
    flags = build_vllm_server_flags(cfg)
    assert "--enable-prefix-caching" in flags
    assert "--no-enable-prefix-caching" not in flags
    assert "--enable-chunked-prefill" in flags
    assert "--async-scheduling" in flags
    assert "--no-enable-log-requests" in flags
    assert "--disable-uvicorn-access-log" in flags

    # Also verify toggle when disabled
    cfg_disabled = V10Config(vllm_enable_prefix_caching=False)
    flags_disabled = build_vllm_server_flags(cfg_disabled)
    assert "--no-enable-prefix-caching" in flags_disabled
    assert "--enable-prefix-caching" not in flags_disabled

    src = inspect.getsource(phase_a_heavy_smoke.start_vllm_server)
    assert "build_vllm_server_flags" in src

