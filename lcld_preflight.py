"""Standalone Phase-A structural preflight verification for ARC-AGI-3 LCLD Agent V10.0."""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

# Ensure local directory is on sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_preflight() -> None:
    phase = os.environ.get("LCLD_PREFLIGHT_PHASE", "local_validation")
    print(f"=== LCLD V10 STRUCTURAL PREFLIGHT START === phase={phase}", flush=True)

    # 1. Verify required files exist
    required_files = [
        "kaggle_agent.py",
        "submission.py",
        "lcld_competition_child.py",
        "v10_agent/__init__.py",
        "v10_agent/config.py",
        "v10_agent/types.py",
        "v10_agent/observe.py",
        "v10_agent/game_adapter.py",
        "v10_agent/action_adapter.py",
        "v10_agent/arga_lite.py",
        "v10_agent/planning_set.py",
        "v10_agent/frame_media.py",
        "v10_agent/verifier_packet.py",
        "v10_agent/brusentsov_logic.py",
        "v10_agent/memory_contours.py",
        "v10_agent/sandbox.py",
        "v10_agent/llm_advisor.py",
        "v10_agent/explorer_agent.py",
        "v10_agent/dsl_coder.py",
        "v10_agent/solver_agent.py",
        "v10_agent/verification.py",
        "v10_agent/judge.py",
        "v10_agent/trajectory.py",
        "v10_agent/policy.py",
        "v10_agent/fallback_symbolic.py",
        "v10_agent/logging.py",
        "v10_agent/session.py",
        "v10_agent/symbolic_executor.py",
        "v10_agent/universal_invariants.py",
        "v10_agent/virtual_sandbox.py",
        "v10_agent/prompt_builders/__init__.py",
        "v10_agent/prompt_builders/coder_prompt.py",
        "v10_agent/prompt_builders/explorer_prompt.py",
        "v10_agent/prompt_builders/solver_prompt.py",
    ]

    for rel in required_files:
        p = ROOT_DIR / rel
        assert p.is_file(), f"Required preflight file missing: {rel}"
        print(f"[OK] Payload file verified: {rel}", flush=True)

    # 2. Verify imports
    modules_to_test = [
        "submission",
        "kaggle_agent",
        "lcld_competition_child",
        "v10_agent",
        "v10_agent.session",
        "v10_agent.brusentsov_logic",
        "v10_agent.sandbox",
        "v10_agent.policy",
    ]
    for mod_name in modules_to_test:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Module {mod_name} import returned None"
        print(f"[OK] Import verified: {mod_name}", flush=True)

    # 3. Instantiate ARC_AGI_Agent and verify interface contracts
    from kaggle_agent import ARC_AGI_Agent, arcade_step_args
    from submission import default_config

    cfg = default_config()
    cfg["llm_advisor_backend"] = "fake"  # safe offline testing
    agent = ARC_AGI_Agent(cfg)

    assert hasattr(agent, "act") and callable(agent.act), "ARC_AGI_Agent.act missing"
    assert hasattr(agent, "observe_action_result") and callable(agent.observe_action_result), "observe_action_result missing"
    assert hasattr(agent, "reset_after_game_over") and callable(agent.reset_after_game_over), "reset_after_game_over missing"
    assert hasattr(agent, "harness_telemetry") and callable(agent.harness_telemetry), "harness_telemetry missing"
    assert callable(arcade_step_args), "arcade_step_args helper missing"
    print("[OK] ARC_AGI_Agent interface verified", flush=True)

    # 4. Dry-run act() and observe_action_result() on synthetic observation
    sample_obs = {
        "grid": [
            [0, 1, 0, 0],
            [0, 0, 0, 2],
        ],
        "available_actions": ["ACTION1", "ACTION2", "ACTION6", "RESET"],
        "state": "IN_PROGRESS",
        "levels_completed": 0,
    }
    action = agent.act(sample_obs)
    assert action is not None, "act() returned None"
    action_id, action_data, reasoning = arcade_step_args(action)
    print(f"[OK] Synthetic act() succeeded -> action_id={action_id}", flush=True)

    committed = agent.observe_action_result(sample_obs)
    assert committed is True, "observe_action_result returned False for pending transition"
    print("[OK] Synthetic observe_action_result() succeeded", flush=True)

    # 5. Verify GAME_OVER contract (clean abandonment when resets disabled or exhausted; RESET when enabled)
    game_over_obs = {"state": "GAME_OVER", "grid": [[0]]}
    if not cfg.get("reset_on_game_over", True) or int(cfg.get("max_game_over_resets_per_game", 5)) <= 0:
        try:
            agent.reset_after_game_over(game_over_obs)
            raise AssertionError("Expected RuntimeError when GAME_OVER reset is disabled")
        except RuntimeError as exc:
            assert "GAME_OVER reset" in str(exc)
            print("[OK] GAME_OVER reset disabled contract verified (clean game abandonment)", flush=True)
    else:
        reset_action = agent.reset_after_game_over(game_over_obs)
        r_id, _, _ = arcade_step_args(reset_action)
        assert str(r_id).endswith("RESET"), f"GAME_OVER path returned non-RESET: {r_id}"
        print("[OK] GAME_OVER single RESET path verified", flush=True)

    # Verify 5-attempt budget exhaustion cleanly abandons without reset
    agent_exhausted = ARC_AGI_Agent(cfg)
    agent_exhausted._session.level_chain_attempts = 5
    try:
        agent_exhausted.reset_after_game_over(game_over_obs)
        raise AssertionError("Expected RuntimeError when 5-attempt budget is exhausted")
    except RuntimeError as exc:
        assert "exhausted" in str(exc).lower()
        print("[OK] GAME_OVER 5-attempt exhaustion contract verified (clean abandonment)", flush=True)

    # Also verify explicit disabled branch dynamically with isolated test config
    cfg_disabled = dict(cfg)
    cfg_disabled["reset_on_game_over"] = False
    agent_disabled = ARC_AGI_Agent(cfg_disabled)
    try:
        agent_disabled.reset_after_game_over(game_over_obs)
        raise AssertionError("Expected RuntimeError when reset_on_game_over is False")
    except RuntimeError as exc:
        assert "abandoning game without reset" in str(exc) or "exhausted" in str(exc)
        print("[OK] Explicit reset_on_game_over=False verified -> clean abandonment", flush=True)

    telemetry = agent.harness_telemetry()
    assert telemetry.get("accepted_action_count", 0) >= 1
    print(f"[OK] Telemetry verified: {telemetry}", flush=True)

    print(f"=== LCLD V10 STRUCTURAL PREFLIGHT OK === phase={phase}", flush=True)


if __name__ == "__main__":
    run_preflight()
