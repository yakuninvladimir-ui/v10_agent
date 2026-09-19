"""Isolated child process runner executing gameplay under Tufa GAME_OVER and Arcade loop rules."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Mapping

from kaggle_agent import ARC_AGI_Agent, arcade_step_args
from submission import _action_name, _state_name, default_config

logger = logging.getLogger(__name__)


class DirectGameFailure(RuntimeError):
    """Raised when an unrecoverable exception occurs during direct game execution."""
    def __init__(self, message: str, metrics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metrics = dict(metrics or {})


def _frame_data(raw: Any) -> Any:
    if raw is None:
        raise ValueError("Environment returned None frame data")
    try:
        from arcengine import FrameData
        if isinstance(raw, FrameData):
            return raw
        frame = getattr(raw, "frame", None)
        if frame is not None:
            converted_frame = []
            for row in frame:
                converted_frame.append(row.tolist() if hasattr(row, "tolist") else list(row))
            return FrameData(
                game_id=str(getattr(raw, "game_id", "") or ""),
                frame=converted_frame,
                state=getattr(raw, "state", None),
                levels_completed=int(getattr(raw, "levels_completed", 0) or 0),
                win_levels=getattr(raw, "win_levels", None),
                guid=getattr(raw, "guid", None),
                full_reset=bool(getattr(raw, "full_reset", False)),
                available_actions=getattr(raw, "available_actions", None),
            )
    except Exception:
        pass
    return raw


def _current_frame(env: Any) -> Any:
    """Extract current FrameData / FrameDataRaw from environment wrapper."""
    raw = getattr(env, "observation_space", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        raw = getattr(env, "current_frame", None)
    if raw is None:
        observe = getattr(env, "observe", None)
        if callable(observe):
            raw = observe()
    if raw is None:
        reset = getattr(env, "reset", None)
        if callable(reset):
            raw = reset()
    return _frame_data(raw)


def _state(frame: Any) -> str:
    if hasattr(frame, "state"):
        return _state_name(getattr(frame, "state"))
    if isinstance(frame, Mapping):
        return _state_name(frame.get("state", "IN_PROGRESS"))
    return "IN_PROGRESS"


def _observation(frame: Any, frame_index: int, game_id: str) -> dict[str, Any]:
    if isinstance(frame, Mapping):
        obs = dict(frame)
        obs.setdefault("frame_index", frame_index)
        obs.setdefault("game_id", game_id)
        return obs
    data: dict[str, Any] = {
        "frame_index": frame_index,
        "game_id": game_id,
        "state": _state(frame),
        "guid": getattr(frame, "guid", None),
        "levels_completed": getattr(frame, "levels_completed", getattr(frame, "score", 0)),
        "win_levels": getattr(frame, "win_levels", None),
        "available_actions": [_action_name(a) for a in getattr(frame, "available_actions", ()) or ()],
    }
    raw_f = getattr(frame, "frame", None)
    data["frame"] = raw_f
    return data


def _terminal_reason(frame: Any) -> str:
    st = _state(frame)
    if st in {"WIN", "WON", "DONE", "TERMINAL", "VICTORY"}:
        return f"state:{st}"
    completed = getattr(frame, "levels_completed", None)
    win_levels = getattr(frame, "win_levels", None)
    try:
        if completed is not None and int(win_levels or 0) > 0 and int(completed) >= int(win_levels):
            return "all_levels_completed"
    except (TypeError, ValueError):
        pass
    return ""


def run_direct_game(
    env: Any,
    game_id: str,
    initial_frame: Any,
    abort_event: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single game to completion under strict competition harness rules."""
    if config is None:
        config = default_config()
    delegate = ARC_AGI_Agent(config)

    game_wall_limit = max(0.0, float(os.getenv("LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS", "5000")))
    if game_wall_limit > 0 and hasattr(delegate, "_session") and hasattr(delegate._session, "config"):
        delegate._session.config.set_deadline(game_wall_limit)

    started = time.monotonic()
    accepted_actions = 0
    proposed_actions = 0
    rejected_actions = 0
    game_over_resets = 0
    frame_index = 0
    latest = _frame_data(initial_frame)
    stop_reason = ""
    last_engine_action = ""

    try:
        max_actions = int(config.get("max_actions_per_game", 250))
        while True:
            if abort_event is not None and abort_event.is_set():
                stop_reason = "parallel_abort"
                break

            if max_actions > 0 and accepted_actions >= max_actions:
                stop_reason = "max_actions_per_game_limit"
                break

            stop_reason = _terminal_reason(latest)
            if stop_reason:
                break

            elapsed = time.monotonic() - started
            if game_wall_limit > 0 and elapsed >= game_wall_limit:
                stop_reason = "game_wall_clock_limit"
                delegate.record_orchestration_termination(stop_reason, {
                    "elapsed_seconds": elapsed,
                    "limit_seconds": game_wall_limit,
                    "accepted_actions": accepted_actions,
                })
                break

            if hasattr(delegate, "_session") and hasattr(delegate._session, "config"):
                reserve_sec = getattr(delegate._session.config, "deadline_reserve_seconds", 15.0)
                if delegate._session.config.is_deadline_exceeded(reserve_seconds=reserve_sec):
                    stop_reason = "deadline_reserve"
                    delegate.record_orchestration_termination(stop_reason, {
                        "elapsed_seconds": elapsed,
                        "limit_seconds": game_wall_limit,
                        "remaining_seconds": delegate._session.config.remaining_time_seconds(),
                        "accepted_actions": accepted_actions,
                    })
                    break

            state_name = _state(latest)
            observation = _observation(latest, frame_index, game_id)

            if state_name == "GAME_OVER":
                # Exact Tufa loop invariant: after GAME_OVER, execute exactly one RESET
                if last_engine_action == "RESET":
                    stop_reason = "game_over_persisted_after_single_reset"
                    break

                try:
                    native_action = delegate.reset_after_game_over(observation, config)
                except Exception as exc:
                    stop_reason = f"agent_level_limit:{exc}"
                    break

                action_id, action_data, reasoning = arcade_step_args(native_action)
                act_name = _action_name(action_id)
                if act_name != "RESET":
                    raise RuntimeError(f"GAME_OVER path must emit RESET, got {act_name}")

                game_over_resets += 1
            else:
                try:
                    native_action = delegate.act(observation, config)
                except Exception as exc:
                    stop_reason = f"agent_error:{exc}"
                    break

                action_id, action_data, reasoning = arcade_step_args(native_action)
                act_name = _action_name(action_id)

            proposed_actions += 1

            try:
                try:
                    raw_next = env.step(action_id, data=action_data, reasoning=reasoning)
                except TypeError:
                    # Fallback: environment does not accept reasoning kwarg
                    raw_next = env.step(action_id, data=action_data)
                if raw_next is None:
                    raise RuntimeError(f"gateway step returned None for game {game_id}")
                next_frame = _frame_data(raw_next)
            except Exception as exc:
                rejected_actions += 1
                err_str = str(exc).lower()
                # If recoverable error (action rejected / invalid), continue
                if "unavailable" in err_str or "invalid" in err_str or "rejected" in err_str:
                    continue
                raise

            accepted_actions += 1
            frame_index += 1
            latest = next_frame
            last_engine_action = act_name

            delegate.observe_action_result(_observation(latest, frame_index, game_id))

            # Yield GIL to reduce contention in multi-threaded game execution
            time.sleep(0)

        if not stop_reason:
            stop_reason = "loop_exit"

        telemetry = delegate.harness_telemetry()
        return {
            "game_id": game_id,
            "status": "completed" if accepted_actions > 0 else "failed_zero_actions",
            "action_count": int(accepted_actions),
            "proposed_action_count": int(proposed_actions),
            "rejected_action_count": int(rejected_actions),
            "game_over_reset_count": int(game_over_resets),
            "levels_completed": int(getattr(latest, "levels_completed", getattr(latest, "score", 0)) or 0),
            "final_state": _state(latest),
            "final_guid": str(getattr(latest, "guid", "") or ""),
            "stop_reason": stop_reason,
            "telemetry_summary": telemetry,
        }
    except Exception as exc:
        failure_metrics = {
            "action_count": int(accepted_actions),
            "proposed_action_count": int(proposed_actions),
            "rejected_action_count": int(rejected_actions),
            "game_over_reset_count": int(game_over_resets),
            "stop_reason": f"exception:{type(exc).__name__}",
        }
        raise DirectGameFailure(str(exc), metrics=failure_metrics) from exc
    finally:
        delegate._cleanup_old_session()


def run_concurrent_arcade_games(
    arcade: Any,
    concurrency: int = 4,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute games yielded by Arcade gateway concurrently up to concurrency limit."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    concurrency = max(1, int(concurrency))
    print(f"[Phase B] Starting concurrent gameplay coordinator (concurrency={concurrency})", flush=True)

    scorecard_id = None
    lazy_game_tasks: list[tuple[int, str]] = []
    direct_env_tasks: list[tuple[int, str, Any, Any]] = []

    if config is None:
        config = default_config()

    # 1. Lazy environment task collection & scorecard opening
    if hasattr(arcade, "create_scorecard") and hasattr(arcade, "available_environments"):
        try:
            scorecard_id = arcade.create_scorecard()
            print(f"[Phase B] Opened competition scorecard: {scorecard_id}", flush=True)
            env_infos = list(arcade.available_environments)
            print(f"[Phase B] Available competition environments count: {len(env_infos)}", flush=True)
            for idx, env_info in enumerate(env_infos):
                game_id = str(getattr(env_info, "game_id", getattr(env_info, "id", f"game_{idx}")))
                lazy_game_tasks.append((idx, game_id))
            print(f"[Phase B] Registered {len(lazy_game_tasks)} games for lazy worker creation (avoiding 15m idle timeout).", flush=True)
        except Exception as sc_exc:
            print(f"[Phase B] Notice during scorecard setup: {sc_exc}", flush=True)

    try:
        if not lazy_game_tasks:
            # Fallback to direct iterator
            print("[Phase B] Consuming environments from direct iterator...", flush=True)
            for idx, item in enumerate(arcade):
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    env, initial_frame = item
                    initial_frame = _frame_data(initial_frame)
                else:
                    env = item
                    initial_frame = _current_frame(env)
                game_id = str(getattr(initial_frame, "game_id", getattr(env, "game_id", f"game_{idx}")))
                direct_env_tasks.append((idx, game_id, env, initial_frame))

        all_tasks = lazy_game_tasks if lazy_game_tasks else direct_env_tasks
        if not all_tasks:
            raise RuntimeError("[Phase B] FATAL: Zero games could be prepared from Arcade!")

        print(f"[Phase B] Prepared {len(all_tasks)} game tasks. Submitting to thread pool (concurrency={concurrency})...", flush=True)
        results: list[dict[str, Any]] = []

        def _execute_worker_task(task: tuple) -> dict[str, Any]:
            if len(task) == 2:
                idx, g_id = task
                print(f"[Phase B] Lazy worker opening environment for game #{idx + 1}: {g_id}", flush=True)
                try:
                    if scorecard_id:
                        env = arcade.make(g_id, scorecard_id=scorecard_id)
                    else:
                        env = arcade.make(g_id)
                    if env is None:
                        raise RuntimeError(f"arcade.make returned None for {g_id}")
                    initial_frame = _current_frame(env)
                    return run_direct_game(env, g_id, initial_frame, config=config)
                except Exception as exc:
                    print(f"[Phase B] Game #{idx + 1} ({g_id}) worker error: {exc}", flush=True)
                    return {
                        "game_id": g_id,
                        "status": "failed",
                        "error": str(exc),
                        "action_count": getattr(exc, "metrics", {}).get("action_count", 0),
                        "levels_completed": 0,
                        "stop_reason": f"lazy_worker_exception:{exc}",
                    }
            else:
                idx, g_id, env, initial_frame = task
                print(f"[Phase B] Direct worker starting game #{idx + 1}: {g_id}", flush=True)
                return run_direct_game(env, g_id, initial_frame, config=config)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: dict[Any, tuple[int, str]] = {}
            for t in all_tasks:
                task_idx = t[0]
                task_gid = t[1]
                fut = executor.submit(_execute_worker_task, t)
                futures[fut] = (task_idx, task_gid)

            for fut in as_completed(futures):
                game_idx, g_id = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    print(
                        f"[Phase B] Game #{game_idx + 1} ({g_id}) complete: "
                        f"status={res.get('status')} actions={res.get('action_count')} levels={res.get('levels_completed')} reason={res.get('stop_reason')}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[Phase B] Game #{game_idx + 1} ({g_id}) failed with unhandled exception: {exc}", flush=True)
                    results.append({
                        "game_id": g_id,
                        "status": "failed",
                        "error": str(exc),
                        "action_count": getattr(exc, "metrics", {}).get("action_count", 0),
                        "levels_completed": 0,
                        "stop_reason": f"unhandled_future_exception:{exc}",
                    })

        print(f"[Phase B] All {len(results)} games finalized.", flush=True)
    finally:
        # Close shared scorecard if opened
        if scorecard_id and hasattr(arcade, "close_scorecard"):
            try:
                arcade.close_scorecard(scorecard_id)
                print(f"[Phase B] Closed competition scorecard: {scorecard_id}", flush=True)
            except Exception as close_exc:
                print(f"[Phase B ERROR] Failed to close scorecard {scorecard_id}: {close_exc}", flush=True)

    return results
