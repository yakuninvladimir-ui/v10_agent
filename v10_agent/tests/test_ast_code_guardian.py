"""Тесты усиления guardian (пункт аудита 18, часть 1).

Четыре новых регрессионных класса guardian должны срабатывать на
синтетических сниппетах и молчать на легитимных whitelist-паттернах:

1. Координатно-семантические regex (`re.search(r"dy=|dx=")`, `axis_steps`,
   `piece_steps`, `internal dots moved from`) вне явных whitelist-санитайзеров
   (solver_agent/memory_contours).
2. Любое использование `re.*` внутри модулей, владеющих координатной/эффектной
   семантикой (структурный запрет: tracker, action_semantics, planning_set, ...).
3. `getattr(config, "<field>", <literal>)`-фолбэки, расходящиеся с объявленным
   умолчанием V10Config (или ссылающиеся на необъявленное поле).
4. Литерал `"ACTION7"` вне whitelist-модулей-исключателей.
Whitelist-сценарии (sanitizer solver_agent dx=/dy=, таксономия memory_contours,
совпадающий getattr-фолбэк = 0 нарушений) обязаны проходить молча.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    from tools.ast_code_guardian import (
        ArchitecturePurityVisitor,
        REGEX_FREE_MODULE_STEMS,
        _extract_config_field_defaults,
        run_purity_audit,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.ast_code_guardian import (
        ArchitecturePurityVisitor,
        REGEX_FREE_MODULE_STEMS,
        _extract_config_field_defaults,
        run_purity_audit,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDIAN = REPO_ROOT / "tools" / "ast_code_guardian.py"

# Синтетический V10Config, повторяющий целевую структуру извлечения: три поля
# с разными типами литералов, чтобы дрейф-проверка сравнивала тип И значение.
_FAKE_CONFIG = """\
from dataclasses import dataclass


@dataclass(frozen=True)
class V10Config:
    max_steps: int = 64
    retry_budget: float = 1.25
    tag_limit: int = 7
"""

# Канонический конфиг репозитория: дрейф на нём означает прод-дрейф.
_REAL_CONFIG = REPO_ROOT / "v10_agent" / "config.py"


def _check_snippet(code: str, filename: str, defaults: dict) -> list[str]:
    tree = ast.parse(code, filename=filename)
    visitor = ArchitecturePurityVisitor(Path(filename), filename, defaults)
    visitor.visit(tree)
    return visitor.violations


def test_coordinate_dx_regex_fires_outside_whitelist():
    violations = _check_snippet(
        "import re\ndef k():\n    return re.search(r'dy=3', text)\n",
        "sample_agent.py",
        {},
    )
    assert violations, "re.search(r'dy=...') должен срабатывать вне solver_agent/memory_contours"
    assert "Координатная семантика" in violations[0]


def test_coordinate_dx_regex_silent_in_solver_agent_whitelist():
    violations = _check_snippet(
        "import re\ndef k():\n    return re.sub(r'dy=\\d+ at row', '', text)\n",
        "solver_agent.py",
        {},
    )
    assert violations == [], "solver_agent является санкционированным санитайзером"


def test_re_import_banned_in_tracker_like_module():
    assert "tracker" in REGEX_FREE_MODULE_STEMS
    violations = _check_snippet(
        "import re\nx = re.search('a', b)\n",
        "tracker.py",
        {},
    )
    assert violations and "запрещено структурно" in violations[0]
    assert "Координатная семантика" not in violations[0]


def _write_fake_config(tmp_path) -> Path:
    path = tmp_path / "config.py"
    path.write_text(_FAKE_CONFIG, encoding="utf-8")
    return path


def test_getattr_drift_detected(tmp_path):
    code = (
        "def f(config):\n"
        "    a = getattr(config, 'max_steps', 32)\n"
        "    b = getattr(config, 'missing_field', True)\n"
        "    c = getattr(config, 'tag_limit', 7)\n"
    )
    defaults = _extract_config_field_defaults(_write_fake_config(tmp_path))
    assert defaults["max_steps"] == 64 and defaults["retry_budget"] == 1.25
    assert defaults["tag_limit"] == 7
    violations = _check_snippet(code, "sample_agent.py", defaults)
    assert len(violations) == 2, violations
    assert "расходится" in violations[0]
    assert "необъявленному" in violations[1]


def test_action7_fires_outside_allowed_stems():
    violations = _check_snippet("def f():\n    action = 'ACTION7'\n", "rogue_agent.py", {})
    assert violations and "ACTION7" in violations[0]


def test_action7_silent_in_allowed_stems():
    for stem in ("action_semantics", "planning_set", "types", "virtual_sandbox",
                 "coder_prompt", "explorer_prompt"):
        violations = _check_snippet("def f():\n    exclude('ACTION7')\n", f"{stem}.py", {})
        assert violations == [], f"{stem} должен быть исключением для ACTION7"


def test_getattr_drift_on_real_config_is_empty():
    """Канонический конфиг: ни один getattr(config, X, literal) не дрейфует."""
    defaults = _extract_config_field_defaults(_REAL_CONFIG)
    assert defaults, "реальный V10Config должен извлекать умолчания"
    assert run_purity_audit(str(REPO_ROOT / "v10_agent")) is True


def test_forbidden_regex_literal_in_memory_contours_whitelist():
    code = (
        "import re\n"
        "# TIER_RULES kinematics taxonomy entry\n"
        "_K = re.compile(r'\\\\bdx=\\\\d+.*(?:at\\\\s+row|row=)')\n"
    )
    violations = _check_snippet(code, "memory_contours.py", {})
    assert violations == [], "memory_contours dx= в таксономии разрешен"


def test_full_guardian_audit_passes_on_repo():
    assert run_purity_audit(str(REPO_ROOT / "v10_agent")) is True


def test_guardian_runs_via_argument_default():
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(GUARDIAN), str(REPO_ROOT / "v10_agent")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "getattr-дрейф: проверено" in proc.stdout, proc.stdout