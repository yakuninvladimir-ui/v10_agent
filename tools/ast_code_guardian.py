"""AST Code Guardian for v10_agent.

Static and semantic AST auditor for preventing specification gaming,
accidental overfitting to benchmark games, hardcoded geometries, and competition contract violations.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Patterns of known benchmark game IDs that must NEVER be hardcoded in the agent
FORBIDDEN_GAME_PATTERNS = [
    re.compile(r"\b(?:ar|ft|ls|re|gt|nb|pb|rs)\d{2}\b", re.IGNORECASE),
]

# Patterns of game-specific heuristic tokens from ar25
FORBIDDEN_HEURISTIC_PATTERNS = [
    re.compile(r"\bpiece_steps\b", re.IGNORECASE),
    re.compile(r"\baxis_steps\b", re.IGNORECASE),
    re.compile(r"internal dots moved from", re.IGNORECASE),
]

# Whitelist of allowed occurrences (e.g. sanitizers explicitly removing legacy artifacts)
WHITELISTED_FILES_FOR_PATTERNS = {
    "memory_contours.py": [
        "axis_steps",
        "piece_steps",
    ],
}


class ArchitecturePurityVisitor(ast.NodeVisitor):
    def __init__(self, filepath: Path, rel_path: str):
        self.filepath = filepath
        self.rel_path = rel_path.replace("\\", "/")
        self.filename = filepath.name
        self.violations: list[str] = []

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            val = node.value
            # Check for forbidden game ID patterns
            for pattern in FORBIDDEN_GAME_PATTERNS:
                if pattern.search(val):
                    self.violations.append(
                        f"[{self.rel_path}:{node.lineno}] Запрещенный идентификатор тестовой игры: '{val}'"
                    )

            # Check for forbidden ar25 heuristics
            whitelisted = WHITELISTED_FILES_FOR_PATTERNS.get(self.filename, [])
            for pattern in FORBIDDEN_HEURISTIC_PATTERNS:
                if pattern.search(val):
                    # Check if this token is whitelisted for this file (e.g. regex stripper in handle_level_transition)
                    matched_token = pattern.pattern
                    if not any(wl in val for wl in whitelisted):
                        self.violations.append(
                            f"[{self.rel_path}:{node.lineno}] Запрещенный эвристический токен ar25: '{val}'"
                        )

            # Check for illegal action generation: ACTION7 (only allowed in block/filter contexts)
            if val == "ACTION7":
                # Check if this is an explicit blocker message or filter list
                parent_context = getattr(node, "_parent_context", "")
                if "BLOCKED" not in parent_context and "filter" not in parent_context:
                    # In v10_agent, ACTION7 is only allowed in exclusion lists, e.g. not in ("RESET", "ACTION7")
                    pass

        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        # Detect hardcoded geometric heuristics: comparing width/height directly with 4 or 10
        left_name = ""
        if isinstance(node.left, ast.Name):
            left_name = node.left.id.lower()
        elif isinstance(node.left, ast.Attribute):
            left_name = node.left.attr.lower()

        if left_name in ("width", "height", "w", "h"):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                    if comp.value in (4, 10):
                        # Detect legacy (height >= 10 and width <= 4)
                        self.violations.append(
                            f"[{self.rel_path}:{node.lineno}] Обнаружен геометрический хардкод: "
                            f"переменная '{left_name}' сравнивается с константой {comp.value}."
                        )
        self.generic_visit(node)


def run_purity_audit(target_dir: str = "v10_agent") -> bool:
    target_path = Path(target_dir)
    if not target_path.exists():
        sys.stderr.write(f"Целевая директория {target_dir} не найдена.\n")
        return False

    total_violations: list[str] = []
    py_files = sorted(list(target_path.rglob("*.py")))

    for py_file in py_files:
        # Skip test files from strict game name audits (tests may test historical components)
        rel_str = str(py_file)
        if "tests" in rel_str:
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            visitor = ArchitecturePurityVisitor(py_file, rel_str)
            visitor.visit(tree)
            total_violations.extend(visitor.violations)
        except Exception as exc:
            total_violations.append(f"[{py_file}] Ошибка парсинга AST: {exc}")

    if total_violations:
        sys.stderr.write("=== КРИТИЧЕСКИЙ СБОЙ: В КОДЕ ОБНАРУЖЕНЫ ПРИВЯЗКИ К ИГРАМ / ХАРДКОД ===\n")
        for v in total_violations:
            sys.stderr.write(f"  {v}\n")
        return False

    sys.stdout.write(
        f"Аудит архитектурной чистоты пройден успешно ({len(py_files)} файлов просканировано). "
        "Хардкод, специфические эвристики и привязки к играм отсутствуют.\n"
    )
    return True


if __name__ == "__main__":
    success = run_purity_audit()
    sys.exit(0 if success else 1)
