"""Unit tests for vLLM Multi-Token Prediction (MTP=3) and speculative decoding."""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from phase_a_heavy_smoke import get_vllm_env, start_vllm_server
from submission import default_config
from v10_agent.config import (
    V10Config,
    build_vllm_speculative_args,
    config_from_env,
)


def test_default_mtp_config():
    """Verify default normative settings for Qwen 3.8 27B MTP=3."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = config_from_env({})
        assert cfg.vllm_mtp_enabled is True
        assert cfg.vllm_mtp_tokens == 3
        assert cfg.vllm_speculative_method == "mtp"
        assert cfg.vllm_speculative_cli_format == "auto"

        args = build_vllm_speculative_args(cfg)
        assert args[0] == "--speculative-config"
        parsed = json.loads(args[1])
        assert parsed["method"] == "mtp"
        assert parsed["num_speculative_tokens"] == 3


def test_mtp_env_var_resolution():
    """Verify standard vLLM and ARC environment variables configure MTP=3."""
    env = {
        "ARC_VLLM_MTP_TOKENS": "3",
        "ARC_VLLM_SPECULATIVE_METHOD": "mtp",
        "VLLM_SPECULATIVE_MODEL": "/kaggle/input/qwen-draft",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = config_from_env()
        assert cfg.vllm_mtp_enabled is True
        assert cfg.vllm_mtp_tokens == 3
        assert cfg.vllm_speculative_method == "mtp"
        assert cfg.vllm_speculative_model == "/kaggle/input/qwen-draft"

        args = build_vllm_speculative_args(cfg)
        assert args[0] == "--speculative-config"
        parsed = json.loads(args[1])
        assert parsed["method"] == "mtp"
        assert parsed["num_speculative_tokens"] == 3
        assert parsed["model"] == "/kaggle/input/qwen-draft"


def test_vllm_env_propagation():
    """Verify get_vllm_env sets vLLM speculative environment variables."""
    with patch.dict(os.environ, {"ARC_VLLM_MTP_ENABLED": "1", "VLLM_MTP_TOKENS": "3"}, clear=True):
        dummy_pkg = pathlib.Path("/tmp/dummy_packages")
        env = get_vllm_env(dummy_pkg)
        assert env.get("VLLM_MTP_TOKENS") == "3"
        assert env.get("VLLM_SPECULATIVE_TOKENS") == "3"
        assert env.get("VLLM_SPEC_METHOD") == "mtp"


def test_cli_argument_formats():
    """Verify all supported CLI formats: auto/config_json, spec_tokens, speculative_model."""
    cfg = V10Config(
        vllm_mtp_enabled=True,
        vllm_mtp_tokens=3,
        vllm_speculative_method="mtp",
        vllm_speculative_model="/models/qwen_mtp",
    )

    # 1. spec_tokens format
    cfg.vllm_speculative_cli_format = "spec_tokens"
    args_shorthand = build_vllm_speculative_args(cfg)
    assert args_shorthand == [
        "--spec-method", "mtp",
        "--spec-tokens", "3",
        "--spec-model", "/models/qwen_mtp",
    ]

    # 2. speculative_model format (legacy vLLM)
    cfg.vllm_speculative_cli_format = "speculative_model"
    args_legacy = build_vllm_speculative_args(cfg)
    assert args_legacy == [
        "--speculative-model", "/models/qwen_mtp",
        "--num-speculative-tokens", "3",
    ]

    # 3. Explicit JSON override
    cfg.vllm_speculative_config = '{"method": "mtp", "custom": true}'
    args_raw = build_vllm_speculative_args(cfg)
    assert args_raw == ["--speculative-config", '{"method": "mtp", "custom": true}']

    # 4. Disabled MTP
    cfg.vllm_mtp_enabled = False
    assert build_vllm_speculative_args(cfg) == []


def test_submission_compatibility():
    """Verify submission.py default_config exposes MTP configuration."""
    cfg_dict = default_config()
    assert "vllm_mtp_enabled" in cfg_dict
    assert "vllm_mtp_tokens" in cfg_dict
    assert cfg_dict["vllm_mtp_enabled"] is True
    assert cfg_dict["vllm_mtp_tokens"] == 3


def test_start_vllm_server_graceful_fallback(tmp_path):
    """Verify that if vLLM fails to boot with MTP, it gracefully falls back to non-MTP boot."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    site_packages = tmp_path / "site_packages"
    site_packages.mkdir()

    call_count = 0
    launched_cmds = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        launched_cmds.append(list(cmd))
        mock_proc = MagicMock()
        if call_count == 1:
            # First launch (with MTP) fails immediately
            mock_proc.poll.return_value = 1
            mock_proc.returncode = 1
        else:
            # Second launch (fallback without MTP) succeeds
            mock_proc.poll.return_value = None
            mock_proc.returncode = None
        return mock_proc

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("phase_a_heavy_smoke.subprocess.Popen", side_effect=mock_popen), \
         patch("phase_a_heavy_smoke.urlopen", return_value=mock_resp), \
         patch("phase_a_heavy_smoke.get_vllm_log_path", return_value=tmp_path / "vllm.log"), \
         patch("phase_a_heavy_smoke.stop_vllm_server"):

        success = start_vllm_server(model_dir, site_packages, enable_mtp=True)

        assert success is True
        assert call_count == 2
        # First attempt included --speculative-config
        assert any("--speculative-config" in arg for arg in launched_cmds[0])
        # Second attempt (fallback) did NOT include --speculative-config
        assert not any("--speculative-config" in arg for arg in launched_cmds[1])
