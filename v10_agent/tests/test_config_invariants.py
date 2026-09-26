"""Config invariant tests (audit item 17).

Guarantees:
1. `config_from_env()` and `V10Config()` agree on every non-private
   field when the environment is clean -- a field silently missed by the
   explicit argument list of `config_from_env` would surface here.
2. `default_config()` never leaks private fields and contains no unknown
   keys beyond the explicit harness-compatibility aliases.
3. The harness aliases in `default_config()` stay synchronized with the
   fields they mirror.
"""

from __future__ import annotations

import dataclasses
import os

from submission import default_config
from v10_agent.config import V10Config, config_from_env

# The only keys in default_config() that are not V10Config field names are
# the explicit harness-compatibility aliases added in submission.py.
HARNESS_ALIAS_KEYS = frozenset(
    {
        "qwen_backend",
        "qwen_vllm_model",
        "game_concurrency",
        "max_coder_retries",
        "max_solver_retries",
        "max_explorer_probes",
    }
)

# Every environment variable consumed by config resolution.
_ENV_PREFIXES = ("ARC_", "LCLD_", "VLLM_")


def _scrub_competition_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)


def test_config_from_env_matches_v10config_defaults(monkeypatch):
    """config_from_env() must agree with V10Config() field by field."""
    _scrub_competition_env(monkeypatch)

    from_env = config_from_env()
    defaults = V10Config()

    differing = []
    for field in dataclasses.fields(V10Config):
        if field.name.startswith("_"):
            # Private runtime state (e.g. _deadline_time), not part of env
            # resolution; both sides carry the same default.
            continue
        a = getattr(from_env, field.name)
        b = getattr(defaults, field.name)
        if a != b:
            differing.append(f"{field.name}: env={a!r} default={b!r}")

    assert not differing, "config_from_env() drifted from V10Config():\n" + "\n".join(
        differing
    )


def test_default_config_has_no_private_or_unknown_keys(monkeypatch):
    """default_config() must expose only field names plus known aliases."""
    _scrub_competition_env(monkeypatch)

    cfg_dict = default_config()
    field_names = {f.name for f in dataclasses.fields(V10Config)}

    private = [k for k in cfg_dict if k.startswith("_")]
    assert not private, f"Private keys leaked into default_config(): {private}"

    unknown = set(cfg_dict) - field_names
    assert unknown == HARNESS_ALIAS_KEYS, (
        f"default_config() drifted: unexpected keys {sorted(unknown - HARNESS_ALIAS_KEYS)}, "
        f"missing aliases {sorted(HARNESS_ALIAS_KEYS - unknown)}"
    )


def test_default_config_alias_values_match_fields(monkeypatch):
    """Every harness alias must mirror the value of its source field."""
    _scrub_competition_env(monkeypatch)

    cfg_dict = default_config()
    alias_map = {
        "qwen_backend": "llm_advisor_backend",
        "qwen_vllm_model": "qwen_model_path",
        "game_concurrency": "concurrency",
        "max_coder_retries": "max_coder_retries_per_level",
        "max_solver_retries": "max_solver_retries_per_level",
        "max_explorer_probes": "max_explorer_probe_actions_per_level",
    }
    for alias, field_name in alias_map.items():
        assert cfg_dict[alias] == cfg_dict[field_name], (
            f"alias {alias!r} ({cfg_dict[alias]!r}) drifted from field "
            f"{field_name!r} ({cfg_dict[field_name]!r})"
        )
