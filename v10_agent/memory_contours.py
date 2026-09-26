"""Isolated Memory Contours enforcing invariants ISO-1 through ISO-5.

Guarantees zero cross-contamination between:
  - Factual discovery (EnvironmentSpecMemory)
  - Syntax/compilation diagnostics (SyntaxErrorMemory)
  - Epistemic/semantic judgments (EpistemicMemory)
  - Game-wide cross-level summaries (GameMemory)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from v10_agent.action_semantics import (
    mentions_action_call,
    parse_action_ids,
    parse_displacement,
    parse_object_ids,
    parse_object_ids_after,
    parse_object_ids_after_keywords,
    replace_object_tokens,
    strip_annotated_segments,
    tokenize_note,
)
from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary


import re
import logging

logger = logging.getLogger(__name__)


from enum import Enum


class EntityRole(str, Enum):
    """Semantic functional role of a color in the game grid (palette 0..15)."""
    BACKGROUND = "background"
    ACTOR = "actor"
    OBSTACLE = "obstacle"
    HAZARD = "hazard"
    TARGET = "target"
    COLLECTIBLE = "collectible"
    PORTAL = "portal"
    UNKNOWN = "unknown"


ROLE_ALIASES: dict[str, EntityRole] = {
    "background": EntityRole.BACKGROUND,
    "actor": EntityRole.ACTOR,
    "player": EntityRole.ACTOR,
    "agent": EntityRole.ACTOR,
    "obstacle": EntityRole.OBSTACLE,
    "wall": EntityRole.OBSTACLE,
    "barrier": EntityRole.OBSTACLE,
    "hazard": EntityRole.HAZARD,
    "danger": EntityRole.HAZARD,
    "target": EntityRole.TARGET,
    "goal": EntityRole.TARGET,
    "collectible": EntityRole.COLLECTIBLE,
    "key": EntityRole.COLLECTIBLE,
    "coin": EntityRole.COLLECTIBLE,
    "portal": EntityRole.PORTAL,
    "door": EntityRole.PORTAL,
}


@dataclass
class ColorAffordance:
    """Affordance record for a single color index (0..15) in the ARC-AGI-3 palette."""
    color_id: int
    role: EntityRole = EntityRole.UNKNOWN
    is_dynamic: bool = False
    pixel_count: int = 0
    interaction_count: int = 0
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_id": self.color_id,
            "role": self.role.value,
            "is_dynamic": self.is_dynamic,
            "pixel_count": self.pixel_count,
            "interaction_count": self.interaction_count,
            "confidence": round(self.confidence, 2),
            "evidence": list(self.evidence),
        }


@dataclass
class PaletteRoleMap:
    """Semantic mapping of all 16 ARC-AGI-3 colors (0..15) to functional roles.

    Replaces destructive color erasure ([COLOR]) with role-aware generalization.
    """
    colors: list[ColorAffordance] = field(default_factory=lambda: [
        ColorAffordance(color_id=i) for i in range(16)
    ])

    def get(self, color_id: int) -> ColorAffordance:
        if 0 <= color_id < len(self.colors):
            return self.colors[color_id]
        return ColorAffordance(color_id=color_id)

    def assign_role(self, color_id: int, role: EntityRole, confidence: float = 0.5, evidence: str = "") -> None:
        if 0 <= color_id < len(self.colors):
            aff = self.colors[color_id]
            if confidence >= aff.confidence or aff.role == EntityRole.UNKNOWN:
                aff.role = role
                aff.confidence = max(aff.confidence, confidence)
                if evidence:
                    aff.evidence.append(evidence)

    def get_colors_by_role(self, role: EntityRole) -> list[ColorAffordance]:
        return [c for c in self.colors if c.role == role]

    def get_role_label(self, color_id: int) -> str:
        if 0 <= color_id < len(self.colors):
            aff = self.colors[color_id]
            if aff.role != EntityRole.UNKNOWN:
                return f"Color {color_id} ({aff.role.value.upper()})"
        return f"Color {color_id}"

    def generalize_color_reference(self, text: str) -> str:
        """Replace color IDs with role-aware labels instead of destructive [COLOR] erasure."""
        def _replace_color(m: re.Match) -> str:
            try:
                cid = int(m.group(1))
                return self.get_role_label(cid)
            except (ValueError, IndexError):
                return m.group(0)

        text = re.sub(r"\bcolor\s+(\d+)\b", _replace_color, text, flags=re.IGNORECASE)
        text = re.sub(r"\bcolor_(\d+)\b", _replace_color, text, flags=re.IGNORECASE)
        text = re.sub(r"\(color\s+(\d+)\)", lambda m: f"({_replace_color(m)})", text, flags=re.IGNORECASE)
        return text

    def update_pixel_counts(self, grid: list[list[int]]) -> None:
        """Update pixel counts and dynamic status for all colors in the grid."""
        if not grid or not grid[0]:
            return
        from collections import Counter
        counts = Counter(c for row in grid for c in row)
        for aff in self.colors:
            cnt = counts.get(aff.color_id, 0)
            if aff.pixel_count > 0 and cnt != aff.pixel_count:
                aff.is_dynamic = True
            aff.pixel_count = cnt

    def format_for_prompt(self) -> str:
        lines = ["PALETTE MAPPING (16 Colors, 0..15):"]
        for aff in self.colors:
            if aff.role != EntityRole.UNKNOWN or aff.pixel_count > 0:
                status = "dynamic" if aff.is_dynamic else "static"
                lines.append(
                    f"  Color {aff.color_id}: {aff.role.value.upper()} ({status}, "
                    f"pixels={aff.pixel_count}, confidence={aff.confidence:.1f})"
                )
        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"colors": [c.to_dict() for c in self.colors if c.role != EntityRole.UNKNOWN or c.pixel_count > 0]}


class IsolationViolationError(RuntimeError):
    """Raised when an architectural memory contour invariant (ISO-1..ISO-5) is violated."""


FORBIDDEN_SYNTAX_KEYWORDS_IN_SOLVER = {
    "traceback",
    "syntaxerror",
    "typeerror",
    "nameerror",
    "attributeerror",
    "zerodivisionerror",
    "indexerror",
    "keyerror",
    "filenotfounderror",
    "runtimeerror",
    "exception:",
    "stack trace",
}

FORBIDDEN_SYNTAX_PATTERNS_IN_SOLVER = [
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\b(?:syntax|type|name|attribute|zerodivision|index|key|filenotfound|runtime)error\b", re.IGNORECASE),
    re.compile(r"\bexception:\s*", re.IGNORECASE),
    re.compile(r"\bstack\s+trace\b", re.IGNORECASE),
]

FORBIDDEN_GOAL_KEYWORDS_IN_CODER = {
    "level_goal",
    "win_condition",
    "target_score",
    "hypothesis_family",
    "epistemic_verdict",
    "brusentsov",
    "live_omit",
    "severed_null",
    "goal invariant",
    "goal_rule",
    "winning_invariant",
    "positive_canon",
    "victory",
    "defeat",
}

FORBIDDEN_GOAL_PATTERNS_IN_CODER = [
    re.compile(r"\b(?:level_goal|win_condition|target_score)\b", re.IGNORECASE),
    re.compile(r"\bgoal\s+invariant\b", re.IGNORECASE),
    re.compile(r"\bgoal_rule\b", re.IGNORECASE),
    re.compile(r"\bwinning_invariant\b", re.IGNORECASE),
    re.compile(r"\bPOSITIVE_CANON\b", re.IGNORECASE),
    re.compile(r"\bTARGET\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"\bHAZARD\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"\bVICTORY\b", re.IGNORECASE),
    re.compile(r"\bDEFEAT\b", re.IGNORECASE),
    re.compile(r"\bhypothesis_family\b", re.IGNORECASE),
    re.compile(r"\bepistemic_verdict\b", re.IGNORECASE),
    re.compile(r"\bbrusentsov\b", re.IGNORECASE),
    re.compile(r"\blive_omit\b", re.IGNORECASE),
    re.compile(r"\bsevered_null\b", re.IGNORECASE),
]


def summarize_grid_diff(
    before_grid: Sequence[Sequence[int]] | None,
    after_grid: Sequence[Sequence[int]] | None,
) -> str:
    """Summarize the physical cell-by-cell differential between two grids.

    Follows Cartesian coordinate system: x=col, y=row.
    Reports:
    - If 0 cells changed: '0 cells changed (no visible change on grid)'
    - If <= 12 cells changed: explicit listing '(x=col, y=row): c_before -> c_after'
    - Aggregate transition counts: '[c_from->c_to (cnt cells)]'
    - Inclusive bounding box of changes: 'in bbox: cols {min_c}..{max_c} (inclusive), rows {min_r}..{max_r} (inclusive)'
    """
    if before_grid is None or after_grid is None:
        return "no grid data available"
    h = len(before_grid)
    w = len(before_grid[0]) if h > 0 else 0
    h_after = len(after_grid)
    w_after = len(after_grid[0]) if h_after > 0 else 0
    if h != h_after or w != w_after:
        return f"grid dimensions changed: {h}x{w} -> {h_after}x{w_after}"
    if h == 0 or w == 0:
        return "0 cells changed (empty grid)"

    changed_cells: list[tuple[int, int, int, int]] = []
    transitions: dict[tuple[int, int], int] = {}
    min_r, max_r = h, -1
    min_c, max_c = w, -1

    for r in range(h):
        row_b = before_grid[r]
        row_a = after_grid[r]
        for c in range(min(len(row_b), len(row_a))):
            cb = int(row_b[c])
            ca = int(row_a[c])
            if cb != ca:
                changed_cells.append((c, r, cb, ca))
                transitions[(cb, ca)] = transitions.get((cb, ca), 0) + 1
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

    if not changed_cells:
        return "0 cells changed (no visible change on grid)"

    trans_strs = [f"{cb}->{ca} ({cnt} cells)" for (cb, ca), cnt in sorted(transitions.items())]
    trans_part = ", ".join(trans_strs)
    bbox_part = f"in bbox: cols {min_c}..{max_c} (inclusive), rows {min_r}..{max_r} (inclusive)"

    if len(changed_cells) <= 12:
        cells_str = ", ".join(f"(x={c}, y={r}): {cb}->{ca}" for c, r, cb, ca in changed_cells)
        return f"{len(changed_cells)} cells changed [{trans_part}] at {cells_str}; {bbox_part}"
    else:
        return f"{len(changed_cells)} cells changed [{trans_part}]; {bbox_part}"


@dataclass
class ProbeRecord:
    """Record of an exploratory probe action executed in the environment."""
    probe_id: str
    action_id: str
    action_data: dict[str, Any]
    observed_effect: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    affordance: dict[str, Any] | None = None


FORBIDDEN_SPEC_KEYS = frozenset({
    "trajectory",
    "candidate_steps",
    "trajectory_id",
    "goal_statement",
    "candidates",
})


@dataclass
class EnvironmentSpecMemory:
    """Memory contour for Explorer Agent.

    Stores ONLY verified factual environment discoveries and probe histories.
    Guaranteed free of planning goals or trajectory proposals (ISO-3).
    """
    game_id: str
    level_id: str
    specs: list[dict[str, Any]] = field(default_factory=list)
    probe_history: list[ProbeRecord] = field(default_factory=list)

    def record_spec(self, spec: dict[str, Any]) -> None:
        """Record an EnvironmentSpecification JSON, asserting no goals or trajectory steps exist."""
        forbidden_present = FORBIDDEN_SPEC_KEYS & {str(k).lower() for k in spec.keys()}
        spec_copy = dict(spec)
        if forbidden_present:
            logger.warning(
                f"ISO-3 Violation (softened): Explorer attempted to write planning/trajectory data into EnvironmentSpecMemory (found keys: {forbidden_present}). Removing them."
            )
            for k in list(spec_copy.keys()):
                if str(k).lower() in FORBIDDEN_SPEC_KEYS:
                    del spec_copy[k]
        self.specs.append(spec_copy)

    def record_probe(self, probe: ProbeRecord) -> None:
        self.probe_history.append(probe)

    def clear_level(self, new_level_id: str) -> None:
        self.level_id = new_level_id
        # Keep high-confidence specs from previous level if general, but reset level-local specs
        self.specs.clear()
        self.probe_history.clear()


@dataclass
class SyntaxErrorRecord:
    """Diagnostic record of a failed DSL compilation or sandbox static validation."""
    prompt_hash: str
    source_code: str
    error_type: str
    error_message: str
    diagnostics: list[str] = field(default_factory=list)
    traceback_str: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "diagnostics": list(self.diagnostics),
            "traceback": self.traceback_str,
            "timestamp": self.timestamp,
        }


@dataclass
class SyntaxErrorMemory:
    """Memory contour for Coder Agent.

    Stores ONLY syntactic and ontological diagnostic records.
    Guaranteed free of level goals or epistemic judgments (ISO-2).
    """
    level_id: str
    entries: list[SyntaxErrorRecord] = field(default_factory=list)
    max_entries: int = 5

    def record_error(self, record: SyntaxErrorRecord) -> None:
        """Record a syntax or validation failure, asserting no goals leak in."""
        for pat in FORBIDDEN_GOAL_PATTERNS_IN_CODER:
            rec_str = f"{record.error_message} {record.error_type} {record.source_code}"
            if pat.search(rec_str):
                logger.warning(
                    f"ISO-2 Violation (softened): Level goal keyword pattern {pat.pattern!r} leaked into SyntaxErrorMemory. Sanitizing."
                )
                record.error_message = pat.sub("[REDACTED GOAL]", record.error_message)
                record.error_type = pat.sub("[REDACTED GOAL]", record.error_type)
                record.source_code = pat.sub("[REDACTED GOAL]", record.source_code)
        self.entries.append(record)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def clear(self) -> None:
        self.entries.clear()


@dataclass
class BranchSignature:
    """A paused or active branch signature in EpistemicMemory."""
    signature_id: str
    trajectory_id: str
    last_step_id: str
    expected_propositions: list[dict[str, Any]]
    observed_propositions: list[dict[str, Any]]
    created_at: float = field(default_factory=time.time)


@dataclass
class EpistemicMemory:
    """Memory contour for Solver Agent.

    Stores ONLY Brusentsov judgments, live OMIT branches, and severed NULL branches.
    Guaranteed free of raw Python tracebacks, syntax errors, or Python source (ISO-1).
    """
    level_id: str
    judgments: list[BrusentsovJudgment] = field(default_factory=list)
    live_omit_branches: list[BranchSignature] = field(default_factory=list)
    severed_null_signatures: set[str] = field(default_factory=set)
    failed_completed_trajectories: list[tuple[str, ...]] = field(default_factory=list)
    current_level_attempts: list[dict[str, Any]] = field(default_factory=list)
    epistemic_signals: list[BrusentsovJudgment] = field(default_factory=list)
    max_entries: int = 50

    def record_judgment(self, judgment: BrusentsovJudgment) -> None:
        """Record a Brusentsov transition judgment, strictly asserting ISO-1 and ISO-9."""
        expl = judgment.explanation or ""
        sanitized_expl = expl
        for pat in FORBIDDEN_SYNTAX_PATTERNS_IN_SOLVER:
            if pat.search(sanitized_expl):
                logger.warning(
                    f"ISO-1 Violation (softened): Python syntax error / traceback pattern {pat.pattern!r} appeared in EpistemicMemory. Sanitizing."
                )
                sanitized_expl = pat.sub("[REDACTED SYNTAX]", sanitized_expl)
        if sanitized_expl != judgment.explanation:
            judgment = replace(judgment, explanation=sanitized_expl)

        # ISO-9: UNDECIDED epistemic signals are kept in epistemic_signals, not committed judgments
        from v10_agent.brusentsov_logic import Verdict
        if judgment.verdict == Verdict.UNDECIDED:
            self.epistemic_signals.append(judgment)
            if len(self.epistemic_signals) > self.max_entries:
                self.epistemic_signals.pop(0)
            return

        self.judgments.append(judgment)
        if len(self.judgments) > self.max_entries:
            self.judgments.pop(0)

    def record_attempt_feedback(
        self,
        hypothesis: dict[str, Any] | str,
        trajectory_summary: str,
        status: str,
        reason: str,
        diff_summary: str | None = None,
        effective_steps: list[str] | None = None,
        diagnostic: str | None = None,
    ) -> None:
        """Record attempt failure/rejection in current level scratchpad with differential before/after tracking."""
        entry: dict[str, Any] = {
            "attempt_index": len(self.current_level_attempts) + 1,
            "hypothesis": hypothesis,
            "trajectory": trajectory_summary,
            "status": status,
            "reason": reason,
        }
        if diff_summary:
            entry["diff_summary"] = diff_summary
        if effective_steps:
            entry["effective_steps"] = list(effective_steps)
        if diagnostic:
            entry["diagnostic"] = diagnostic
        self.current_level_attempts.append(entry)
        if len(self.current_level_attempts) > 5:
            self.current_level_attempts.pop(0)

    def format_scratchpad_context(self) -> str:
        """Format current-level failed attempts with differential before/after analysis."""
        if not self.current_level_attempts:
            return ""
        lines = ["CURRENT LEVEL FAILED ATTEMPTS (DIFFERENTIAL BEFORE/AFTER SCRATCHPAD):"]
        for att in self.current_level_attempts:
            lines.append(
                f"- Attempt {att['attempt_index']}: [{att.get('status')}] Trajectory: {att.get('trajectory')}"
            )
            if att.get("diagnostic"):
                for d_line in att["diagnostic"].splitlines():
                    lines.append(f"  {d_line}")
            if att.get("diff_summary"):
                lines.append(f"  * Physical grid diff (before vs after attempt): {att['diff_summary']}")
            if att.get("effective_steps"):
                lines.append(f"  * Effective steps during attempt: {'; '.join(att['effective_steps'])}")
            if att.get("reason"):
                lines.append(f"  * Failure reason: {att['reason']}")
        return "\n".join(lines)

    def pause_branch_as_omit(self, branch: BranchSignature) -> None:
        """Record an inessential missing effect as a live OMIT growth point."""
        # Ensure not already severed
        if branch.signature_id not in self.severed_null_signatures:
            self.live_omit_branches.append(branch)

    def sever_branch(self, signature_id: str) -> None:
        """Permanently sever a contradicted branch (NULL verdict) or failed trajectory."""
        self.severed_null_signatures.add(signature_id)
        self.live_omit_branches = [b for b in self.live_omit_branches if b.signature_id != signature_id]
        if " -> " in signature_id:
            parts = tuple(s.strip() for s in signature_id.split(" -> ") if s.strip())
            if parts and parts not in self.failed_completed_trajectories:
                self.failed_completed_trajectories.append(parts)
        elif signature_id.strip() and not signature_id.startswith("sig_") and signature_id != "":
            parts = (signature_id.strip(),)
            if parts not in self.failed_completed_trajectories:
                self.failed_completed_trajectories.append(parts)

    def record_failed_completed_trajectory(self, trajectory_tuple: tuple[str, ...]) -> None:
        """Record a completed trajectory that failed to win the level."""
        sig_tuple = tuple(trajectory_tuple)
        if sig_tuple and sig_tuple not in self.failed_completed_trajectories:
            self.failed_completed_trajectories.append(sig_tuple)

    def is_trajectory_subsumed(self, candidate_tuple: tuple[str, ...]) -> tuple[bool, str | None]:
        """Check if candidate trajectory is completely contained in a failed trajectory from index 0 in order.

        Rule:
        - If failed is 1-2-3-4-5-4-3-2-1:
          * 1-2-3-4-5 is subsumed (True, prefix of length 5 matching from the start).
          * 5-4-3-2-1 is NOT subsumed (False, does not start with action 1 from the start).
          * 1-2-3-4-5-4-3-2-1 is subsumed (True, exact full match).
          * 1-2-3-4-5-4-3-2-1-6 is NOT subsumed (False, extends beyond failed trajectory).
        """
        if not candidate_tuple:
            return False, None
        for failed in self.failed_completed_trajectories:
            if len(candidate_tuple) <= len(failed) and failed[:len(candidate_tuple)] == candidate_tuple:
                return True, " -> ".join(failed)
        return False, None

    def is_severed(self, signature_id: str) -> bool:
        return signature_id in self.severed_null_signatures

    def format_structured_failures(self) -> list[dict[str, Any]]:
        """Extract structured failure diagnostics from NULL verdicts for Solver reasoning."""
        failures = []
        for j in self.judgments:
            if j.verdict == Ternary.FALSE:
                exp_list: list[Any] = []
                if hasattr(j.expected_propositions, "__iter__"):
                    for p in j.expected_propositions:
                        if hasattr(p, "to_dict"):
                            exp_list.append(p.to_dict())
                        else:
                            exp_list.append(str(p))
                failures.append({
                    "trajectory_id": j.trajectory_id,
                    "failed_step": j.step_id,
                    "expected": exp_list,
                    "verdict": "FALSE (NULLITY)",
                    "explanation": j.explanation,
                })
        return failures

    def clear(self) -> None:
        self.judgments.clear()
        self.epistemic_signals.clear()
        self.live_omit_branches.clear()
        self.severed_null_signatures.clear()
        self.current_level_attempts.clear()
        self.failed_completed_trajectories.clear()

    def clear_for_new_level(self) -> None:
        """Clear intra-level history when transitioning to a new level."""
        self.clear()


TIER_RULES = [
    # (tier, pattern, description/type)
    (3, re.compile(r"\b(?:\[goal\]|win_condition|level victory)\b", re.IGNORECASE), "goal"),
    (1, re.compile(r"\b(?:symmetr\w*|mirror\w*|axis|axes|reflect\w*|axial_symmetry)", re.IGNORECASE), "symmetry"),
    (1, re.compile(r"\b(?:mov\w*|displace\w*|dy=|dx=|step_size|boundary|collision|wall|blocked|physics|gravity|momentum)", re.IGNORECASE), "kinematics"),
    (2, re.compile(r"\b(?:select\w*|toggl\w*|switch\w*|cycl\w*|ACTION5)", re.IGNORECASE), "control"),
    (2, re.compile(r"\b(?:click\w*|trigger\w*|interact\w*|socket|touch\w*|cover\w*)", re.IGNORECASE), "interaction"),
    (2, re.compile(r"\b(?:color\w*|palette|indicator\w*)", re.IGNORECASE), "palette"),
]


def _classify_tier(rule: str) -> tuple[int, str]:
    for tier, pattern, inv_type in TIER_RULES:
        if pattern.search(rule):
            return tier, inv_type
    return 3, "general"


@dataclass
class StructuredInvariant:
    """[DEPRECATED: Use EmpiricalInvariant] A typed domain-general invariant evaluated using Brusentsov ternary logic."""
    invariant_id: str                      # Unique ID
    invariant_type: str                    # 'kinematics' | 'symmetry' | 'palette' | 'goal' | 'topology' | 'control' | 'interaction' | 'general'
    tier: int                              # 1, 2, 3
    description: str                       # Textual description
    confidence: float = 0.3                # Initial confidence
    ternary_status: Ternary = Ternary.IRRELEVANT  # Status under Brusentsov logic
    confirmed_on_levels: list[str] = field(default_factory=list)
    falsified_on_levels: list[str] = field(default_factory=list)
    first_discovered_level: int = 0
    source: str = ""                       # 'explorer' | 'solver_distillation' | 'arga_lite'
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "invariant_type": self.invariant_type,
            "tier": self.tier,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "ternary_status": self.ternary_status.name,
            "confirmed_on_levels": list(self.confirmed_on_levels),
            "falsified_on_levels": list(self.falsified_on_levels),
            "first_discovered_level": self.first_discovered_level,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass
class LevelVictoryExample:
    """Neutral illustrative example of how a past level was solved (not a prescriptive template)."""
    level_id: str
    winning_actions: list[str] = field(default_factory=list)
    object_diffs: list[dict[str, Any]] = field(default_factory=list)
    static_objects: list[str] = field(default_factory=list)
    initial_objects: list[dict[str, Any]] = field(default_factory=list)
    goal_rule: str = ""
    start_grid_hash: str = ""
    end_grid_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "winning_actions": list(self.winning_actions),
            "object_diffs": list(self.object_diffs),
            "static_objects": list(self.static_objects),
            "initial_objects": list(self.initial_objects),
            "goal_rule": self.goal_rule,
            "start_grid_hash": self.start_grid_hash,
            "end_grid_hash": self.end_grid_hash,
        }


@dataclass
class DefeatExemplar:
    """Concrete example of a fatal error — grounding for NEGATIVE_BARRIER invariants (xy'_0 → NULL)."""
    level_index: int
    fatal_step: int
    fatal_action_id: int                         # 1..6
    fatal_coords: tuple[int, int] | None = None  # (x, y) for ACTION6
    actor_position_before: tuple[int, int] = (0, 0)
    hazard_color: int = -1                       # Color 0..15
    pre_defeat_subgrid: list[list[int]] = field(default_factory=list)
    environment_signal: str = ""
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_index": self.level_index, "fatal_step": self.fatal_step,
            "fatal_action_id": self.fatal_action_id, "fatal_coords": self.fatal_coords,
            "actor_position_before": self.actor_position_before,
            "hazard_color": self.hazard_color, "environment_signal": self.environment_signal,
            "explanation": self.explanation,
        }

    def format_for_prompt(self) -> str:
        coord_str = f" at coords ({self.fatal_coords[0]}, {self.fatal_coords[1]})" if self.fatal_coords else ""
        return (
            f"LAST DEFEAT (Level {self.level_index}, Step {self.fatal_step}):\n"
            f"  Action: ACTION{self.fatal_action_id}{coord_str}\n"
            f"  Actor was at: row={self.actor_position_before[0]}, col={self.actor_position_before[1]}\n"
            f"  Hazard color: {self.hazard_color}\n"
            f"  Signal: {self.environment_signal}\n"
            f"  Lesson: {self.explanation}"
        )


@dataclass
class VictoryExemplar:
    """Concrete example of level completion — grounding for POSITIVE_CANON invariants (xy → FOLLOW)."""
    level_index: int
    total_steps: int
    action_sequence: list[int] = field(default_factory=list)
    key_transitions: list[dict[str, Any]] = field(default_factory=list)
    final_action_id: int = 0
    target_color: int = -1                      # Color 0..15
    final_subgrid: list[list[int]] = field(default_factory=list)
    explanation: str = ""
    winning_invariants_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_index": self.level_index, "total_steps": self.total_steps,
            "action_sequence": list(self.action_sequence),
            "key_transitions": list(self.key_transitions),
            "final_action_id": self.final_action_id, "target_color": self.target_color,
            "explanation": self.explanation,
            "winning_invariants_used": list(self.winning_invariants_used),
        }

    def format_for_prompt(self) -> str:
        seq_str = " → ".join(str(a) for a in self.action_sequence[:20])
        if len(self.action_sequence) > 20:
            seq_str += "..."
        lines = [
            f"LAST VICTORY (Level {self.level_index}, {self.total_steps} steps):",
            f"  Winning sequence: [{seq_str}]",
            f"  Final action: ACTION{self.final_action_id} → reached Color {self.target_color} (TARGET)",
        ]
        if self.key_transitions:
            lines.append("  Key breakthroughs:")
            for kt in self.key_transitions[:5]:
                lines.append(f"    - Step {kt.get('step', '?')}: {kt.get('description', 'transition')}")
        lines.append(f"  Explanation: {self.explanation}")
        return "\n".join(lines)


@dataclass
class GroundedInvariant:
    """[DEPRECATED: Use EmpiricalInvariant] Symbolic invariant grounded in concrete exemplars via Brusentsov logic of entailment."""
    invariant_id: str
    antecedent: str               # e.g., 'Contact(ACTOR, Color_2)'
    consequent: str               # e.g., 'DefeatReset()'
    brusentsov_type: str          # 'NEGATIVE_BARRIER' or 'POSITIVE_CANON'
    scope: str                    # 'CORE_GAME_LAW' or 'LEVEL_SPECIFIC'
    grounded_in_defeat: DefeatExemplar | None = None
    grounded_in_victory: VictoryExemplar | None = None
    times_confirmed: int = 0
    times_falsified: int = 0
    confidence: float = 0.5
    created_on_level: int = 0

    def is_active(self) -> bool:
        return self.times_falsified == 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "invariant_id": self.invariant_id, "antecedent": self.antecedent,
            "consequent": self.consequent, "brusentsov_type": self.brusentsov_type,
            "scope": self.scope, "times_confirmed": self.times_confirmed,
            "times_falsified": self.times_falsified, "confidence": round(self.confidence, 2),
        }
        if self.grounded_in_defeat:
            result["defeat_exemplar"] = self.grounded_in_defeat.to_dict()
        if self.grounded_in_victory:
            result["victory_exemplar"] = self.grounded_in_victory.to_dict()
        return result

    def format_for_prompt(self) -> str:
        status = "ACTIVE" if self.is_active() else f"FALSIFIED ({self.times_falsified}x)"
        return (
            f"[{self.brusentsov_type}] {self.antecedent} => {self.consequent} "
            f"(scope={self.scope}, confirmed={self.times_confirmed}x, status={status})"
        )


@dataclass
class EmpiricalInvariant:
    invariant_id: str
    invariant_type: str  # "area_conservation", "contact_trigger", "kinematic_law", etc.
    abstract_description: str  # Обобщенная формулировка ("Objects conserve area")
    subject_pattern: str  # Паттерн субъекта ("color==ACTOR", "area>4")
    expected_value: Any  # Ожидаемое значение/паттерн
    scope: str  # "CORE_GAME_LAW" | "DOMAIN_PATTERN" | "LEVEL_SPECIFIC"
    confirmed_on_levels: list[int] = field(default_factory=list)
    falsified_on_levels: list[int] = field(default_factory=list)
    times_confirmed: int = 0
    times_falsified: int = 0
    confidence: float = 0.0  # НАЧИНАЕТСЯ С 0!
    last_observed_value: Any = None

    @property
    def is_active(self) -> bool:
        return self.times_falsified == 0 and self.confidence >= 0.3

    @property
    def net_support(self) -> int:
        return self.times_confirmed - 2 * self.times_falsified  # Штраф за опровержение

    def confirm(self, level_index: int, observed_value: Any = None) -> None:
        if level_index not in self.confirmed_on_levels:
            self.confirmed_on_levels.append(level_index)
        self.times_confirmed += 1
        self.confidence = min(1.0, round(self.confidence + 0.1, 2))
        self.last_observed_value = observed_value

    def falsify(self, level_index: int, observed_value: Any = None) -> None:
        if level_index not in self.falsified_on_levels:
            self.falsified_on_levels.append(level_index)
        self.times_falsified += 1
        self.confidence = max(0.0, round(self.confidence - 0.2, 2))
        self.last_observed_value = observed_value
        # Если опровергнуто 2+ раза на разных уровнях — понижаем scope
        if len(self.falsified_on_levels) >= 2 and self.scope == "CORE_GAME_LAW":
            self.scope = "LEVEL_SPECIFIC"

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "invariant_type": self.invariant_type,
            "abstract_description": self.abstract_description,
            "subject_pattern": self.subject_pattern,
            "expected_value": self.expected_value,
            "scope": self.scope,
            "confirmed_on_levels": list(self.confirmed_on_levels),
            "falsified_on_levels": list(self.falsified_on_levels),
            "times_confirmed": self.times_confirmed,
            "times_falsified": self.times_falsified,
            "confidence": round(self.confidence, 2),
            "last_observed_value": self.last_observed_value,
            "is_active": self.is_active,
            "net_support": self.net_support,
        }


@dataclass
class CoreInvariantRegistry:
    invariants: list[EmpiricalInvariant] = field(default_factory=list)
    max_active: int = 20

    def register_candidate(self, inv: EmpiricalInvariant) -> None:
        # Дедупликация по invariant_id
        existing = next((i for i in self.invariants if i.invariant_id == inv.invariant_id), None)
        if existing is None:
            self.invariants.append(inv)
            self._prune()

    def confirm_by_type(self, invariant_type: str, subject_pattern: str, 
                        level_index: int, observed_value: Any) -> None:
        for inv in self.invariants:
            if inv.invariant_type == invariant_type and inv.subject_pattern == subject_pattern:
                inv.confirm(level_index, observed_value)

    def falsify_by_type(self, invariant_type: str, subject_pattern: str,
                        level_index: int, observed_value: Any) -> None:
        for inv in self.invariants:
            if inv.invariant_type == invariant_type and inv.subject_pattern == subject_pattern:
                inv.falsify(level_index, observed_value)

    def get_core_game_laws(self) -> list[EmpiricalInvariant]:
        return sorted(
            [i for i in self.invariants if i.scope == "CORE_GAME_LAW" and i.is_active],
            key=lambda x: x.net_support, reverse=True
        )

    def format_for_prompt(self) -> str:
        # Таблица: Инвариант | Тип | Уровни ✓ | Уровни ✗ | Confidence | Scope
        lines = ["INVARIANT REGISTRY:"]
        lines.append(f"{'INVARIANT':<40} {'TYPE':<20} {'✓':<5} {'✗':<5} {'CONF':<6} {'SCOPE':<15}")
        lines.append("-" * 95)
        for inv in sorted(self.invariants, key=lambda x: x.net_support, reverse=True)[:15]:
            lines.append(
                f"{inv.abstract_description[:40]:<40} "
                f"{inv.invariant_type[:20]:<20} "
                f"{len(inv.confirmed_on_levels):<5} "
                f"{len(inv.falsified_on_levels):<5} "
                f"{inv.confidence:<6.2f} "
                f"{inv.scope:<15}"
            )
        return "\n".join(lines)

    def _prune(self) -> None:
        # Удаляем инварианты с сильным отрицательным net_support
        self.invariants = [i for i in self.invariants if i.net_support > -3]


@dataclass
class GameMemory:
    """Cross-level memory summary surviving level transitions within the same game.
    
    Stratifies invariants into 3 tiers:
      - Tier 1: Foundational Physics, Kinematics & Topology (movement deltas, collisions, boundaries)
      - Tier 2: Interaction Dynamics & State Transitions (selection toggles, color changes, triggers)
      - Tier 3: Level Progression Rules & Curriculum Patterns (distilled level solution invariants)
    """
    game_id: str
    confirmed_action_effects: dict[str, str] = field(default_factory=dict)
    unconfirmed_actions: dict[str, str] = field(default_factory=dict)
    selection_mechanics: list[str] = field(default_factory=list)
    reusable_primitives: list[dict[str, Any]] = field(default_factory=list)
    invariant_rules: list[str] = field(default_factory=list)
    tier1_kinematics_and_topology: list[str] = field(default_factory=list)
    tier2_interactions: list[str] = field(default_factory=list)
    tier3_level_rules: list[str] = field(default_factory=list)
    level_solution_patterns: list[dict[str, Any]] = field(default_factory=list)
    completed_levels: int = 0
    structured_invariants: list[StructuredInvariant] = field(default_factory=list)
    last_discovered_invariants: list[Any] = field(default_factory=list)
    invalidated_invariants: list[dict[str, Any]] = field(default_factory=list)
    last_level_victory_example: LevelVictoryExample | None = None
    confirmed_actor_ids: set[str] = field(default_factory=set)
    # --- New memory blocks ---
    palette: PaletteRoleMap = field(default_factory=PaletteRoleMap)
    defeat_exemplars: list[DefeatExemplar] = field(default_factory=list)
    victory_exemplars: list[VictoryExemplar] = field(default_factory=list)
    last_defeat_exemplar: DefeatExemplar | None = None
    last_victory_exemplar_v2: VictoryExemplar | None = None
    grounded_invariants: list[GroundedInvariant] = field(default_factory=list)
    curriculum_history: list[dict[str, Any]] = field(default_factory=list)
    working_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    action_affordances: list[dict[str, Any]] = field(default_factory=list)
    invariant_registry: CoreInvariantRegistry = field(default_factory=CoreInvariantRegistry)
    previous_level_registry: CoreInvariantRegistry | None = None

    def add_defeat(self, ex: DefeatExemplar) -> None:
        """Store defeat exemplar maintaining last 5 exemplars buffer."""
        self.defeat_exemplars.append(ex)
        if len(self.defeat_exemplars) > 5:
            self.defeat_exemplars.pop(0)
        self.last_defeat_exemplar = ex

    def add_victory(self, ex: VictoryExemplar) -> None:
        """Store victory exemplar maintaining last 5 exemplars buffer."""
        self.victory_exemplars.append(ex)
        if len(self.victory_exemplars) > 5:
            self.victory_exemplars.pop(0)
        self.last_victory_exemplar_v2 = ex

    def analyze_defeat_patterns(self) -> list[str]:
        """Detect recurring failure patterns from recent defeat exemplars."""
        if len(self.defeat_exemplars) < 2:
            return []
        from collections import Counter
        patterns = Counter(
            (e.hazard_color, e.fatal_action_id) for e in self.defeat_exemplars
        )
        return [f"Hazard color {c}, action ACTION{a}: {n} times" 
                for (c, a), n in patterns.most_common(3) if n >= 2]

    def summarize_progression(self) -> str:
        """Summarize progression history across completed levels for curriculum grounding."""
        if not self.curriculum_history:
            return "none"
        lines = []
        for entry in self.curriculum_history[-5:]:
            lvl = entry.get("level", entry.get("from_level", entry.get("to_level", "?")))
            steps = entry.get("steps_to_win", entry.get("steps", "?"))
            invs = entry.get("invariants_confirmed", [])
            carried = entry.get("core_invariants_carried", len(invs) if isinstance(invs, list) else 0)
            inv_count = len(invs) if isinstance(invs, list) and invs else carried
            summary = entry.get("summary", "")
            line = f"Level {lvl}: completed in {steps} steps (invariants: {inv_count})"
            if summary:
                line += f" | {summary}"
            lines.append(line)
        return "\n".join(lines)

    def record_hypothesis(self, hypothesis: str, source_step_id: str = "") -> None:
        """Record an active working hypothesis from evidence probes or planner."""
        if not hypothesis:
            return
        self.working_hypotheses.append({
            "hypothesis": hypothesis,
            "source_step": source_step_id,
            "status": "pending",
            "created_at": time.time(),
        })
        if len(self.working_hypotheses) > 10:
            self.working_hypotheses.pop(0)

    def migrate_legacy_invariants(self, level_index: int = 0) -> None:
        """Migrate legacy StructuredInvariant and GroundedInvariant into EmpiricalInvariant with confidence=0.3."""
        for s_inv in self.structured_invariants:
            emp_inv = EmpiricalInvariant(
                invariant_id=s_inv.invariant_id,
                invariant_type=s_inv.invariant_type,
                abstract_description=s_inv.description,
                subject_pattern="legacy",
                expected_value=None,
                scope="DOMAIN_PATTERN" if s_inv.tier < 3 else "LEVEL_SPECIFIC",
                confidence=0.3,
                times_confirmed=1,
                confirmed_on_levels=[level_index],
            )
            self.invariant_registry.register_candidate(emp_inv)

        for g_inv in self.grounded_invariants:
            emp_inv = EmpiricalInvariant(
                invariant_id=g_inv.invariant_id,
                invariant_type="grounded_exemplar",
                abstract_description=f"{g_inv.antecedent} => {g_inv.consequent}",
                subject_pattern="exemplar",
                expected_value=g_inv.consequent,
                scope=g_inv.scope,
                confidence=0.3,
                times_confirmed=g_inv.times_confirmed or 1,
                confirmed_on_levels=[level_index],
            )
            self.invariant_registry.register_candidate(emp_inv)

    def update_action_affordances(self, affordances: list[dict[str, Any]]) -> None:
        """Record or accumulate confirmed action affordances (preserving composite effects)."""
        for aff in affordances:
            if isinstance(aff, dict) and aff.get("action_id"):
                act_id = str(aff["action_id"]).upper()
                existing_aff = next(
                    (a for a in self.action_affordances if isinstance(a, dict) and str(a.get("action_id", "")).upper() == act_id),
                    None,
                )
                if existing_aff is not None:
                    merged = dict(existing_aff)
                    old_notes = str(existing_aff.get("coordination_notes", "")).strip()
                    new_notes = str(aff.get("coordination_notes", "")).strip()
                    if new_notes and new_notes not in [p.strip() for p in old_notes.split(" | ") if p.strip()]:
                        merged["coordination_notes"] = f"{old_notes} | {new_notes}" if old_notes else new_notes
                    # Preserve kinematic class if either is kinematic, and record composite effects in parameters
                    old_cls = str(existing_aff.get("effect_class", ""))
                    new_cls = str(aff.get("effect_class", ""))
                    if new_cls == "KINEMATIC" and old_cls != "KINEMATIC":
                        merged["effect_class"] = "KINEMATIC"
                        merged_params = dict(aff.get("parameters", {}) or {})
                        merged_params.setdefault("secondary_effects", []).append(existing_aff.get("parameters", {}))
                        merged["parameters"] = merged_params
                    elif new_notes and new_notes != old_notes:
                        merged_params = dict(existing_aff.get("parameters", {}) or {})
                        sec_list = list(merged_params.get("secondary_effects", []))
                        new_p = aff.get("parameters", {})
                        if new_p and new_p not in sec_list and new_p != existing_aff.get("parameters"):
                            sec_list.append(new_p)
                            merged_params["secondary_effects"] = sec_list
                        merged["parameters"] = merged_params
                    self.action_affordances = [
                        a for a in self.action_affordances
                        if not (isinstance(a, dict) and str(a.get("action_id", "")).upper() == act_id)
                    ]
                    self.action_affordances.append(merged)
                else:
                    self.action_affordances.append(dict(aff))

    @property
    def confirmed_actors(self) -> set[str]:
        """Return set of object IDs empirically observed to move or receive selection indicators."""
        actors: set[str] = set(self.confirmed_actor_ids)
        for note in self.selection_mechanics:
            actors.update(parse_object_ids(note))
        for eff in self.confirmed_action_effects.values():
            actors.update(parse_object_ids(eff))
        return actors

    def _build_role_map(self) -> dict[str, str]:
        """Build role mapping for generalizing object references across levels.

        Assigns semantic roles based on empirical evidence:
          - ACTOR: objects empirically observed to move or receive selection indicators
          - TARGET: objects mentioned as destinations in confirmed action effects
          - OBSTACLE: objects associated with collisions/blocks in invariant rules
          - (remaining unknown obj_IDs become [ENTITY] via _generalize_text)
        """
        role_map: dict[str, str] = {}
        # Step 1: Confirmed actors
        for actor_id in self.confirmed_actors:
            role_map[actor_id] = "ACTOR"

        # Step 2: Identify TARGETs — objects named as destinations in action effects
        for eff in self.confirmed_action_effects.values():
            for tid in parse_object_ids_after(eff, "toward"):
                role_map[tid] = "TARGET"

        # Step 3: Identify OBSTACLEs — objects associated with collision/blocked events
        obstacle_lexicon = {"blocked", "wall", "collision", "obstacle", "barrier"}
        for rule in self.invariant_rules:
            if obstacle_lexicon.intersection(tokenize_note(rule)):
                for oid in parse_object_ids(rule):
                    role_map[oid] = "OBSTACLE"

        return role_map

    def _generalize_text(self, text: str) -> str:
        r"""Replace specific object IDs with functional role markers and generalize colors via palette roles (ISO-8).

        Confirmed actors → [ACTOR], destinations → [TARGET],
        collision objects → [OBSTACLE], unknown → [ENTITY].
        Colors are generalized using PaletteRoleMap semantic roles instead of destructive [COLOR] erasure.
        """
        role_map = self._build_role_map()
        lower_role_map = {key.lower(): role for key, role in role_map.items()}
        # Every obj_... token is rewritten in a single scan: known entities become
        # their functional role, unknown entities collapse to [ENTITY].
        text = replace_object_tokens(text, lambda tok: f"[{lower_role_map.get(tok.lower(), 'ENTITY')}]")
        # Use PaletteRoleMap for semantic color generalization instead of [COLOR] erasure
        if hasattr(self, 'palette') and self.palette is not None:
            text = self.palette.generalize_color_reference(text)
        else:
            # Fallback: preserve color identity with uppercase marker
            text = re.sub(r"\b(?:color\s+\d+|color_\d+)\b", lambda m: f"[{m.group(0).upper()}]", text, flags=re.IGNORECASE)
            text = re.sub(r"\(color\s+\d+\)", lambda m: f"[{m.group(0).upper()}]", text, flags=re.IGNORECASE)
        return text

    def _is_level_specific(self, text: str) -> bool:
        """Check if an invariant rule is explicitly tagged as specific to a single level."""
        return bool(re.search(
            r"\b(?:only\s+on\s+level\s+\d+|level_\d+\s+transient|level\s+\d+\s+specific\s+obstacle|temporary\s+level\s+rule)\b",
            text,
            re.IGNORECASE,
        ))

    def record_action_effect(self, action_id: str, summary: str) -> None:
        existing = self.confirmed_action_effects.get(action_id)
        if existing and existing != "confirmed_reusable_action":
            existing_parts = [p.strip() for p in existing.split(" | ") if p.strip()]
            for np in [p.strip() for p in summary.split(" | ") if p.strip()]:
                if np not in existing_parts:
                    existing_parts.append(np)
            self.confirmed_action_effects[action_id] = " | ".join(existing_parts)
        else:
            self.confirmed_action_effects[action_id] = summary
        self.unconfirmed_actions.pop(action_id, None)
        rule = f"Action {action_id} kinematic effect: {summary}"
        self.record_stratified_invariant(
            rule,
            tier=1,
            invariant_type="kinematics",
            confidence=0.9,
            source="action_probe",
            metadata={"action_id": action_id},
        )

    def record_unconfirmed_action(self, action_id: str, reason: str = "zero observable effect") -> None:
        if action_id not in self.confirmed_action_effects:
            self.unconfirmed_actions[action_id] = reason

    def invalidate_action_effect(self, action_id: str, level_id: str = "", reason: str = "") -> None:
        """Remove invalidated action effect from confirmed kinematics upon empirical falsification."""
        self.confirmed_action_effects.pop(action_id, None)
        reason_str = reason or "falsified_by_empirical_verifier"
        self.unconfirmed_actions[action_id] = reason_str
        inv_reason_str = reason or "falsified_by_empirical_verifier (action produced zero delta or collided with obstacle)"
        prefix = f"Action {action_id} kinematic effect:"
        found_rule = False
        for r in list(self.invariant_rules):
            if r.startswith(prefix) or f"action {action_id.lower()}" in r.lower():
                found_rule = True
                if not any(entry.get("description") == r for entry in self.invalidated_invariants):
                    self.invalidated_invariants.append({
                        "description": r,
                        "invariant_type": "kinematics",
                        "reason": inv_reason_str,
                        "level_id": level_id,
                        "action_id": action_id,
                    })
        if not found_rule and not any(entry.get("action_id") == action_id for entry in self.invalidated_invariants):
            self.invalidated_invariants.append({
                "description": f"Kinematic effect of {action_id}",
                "invariant_type": "kinematics",
                "reason": inv_reason_str,
                "level_id": level_id,
                "action_id": action_id,
            })
        self.tier1_kinematics_and_topology = [
            r for r in self.tier1_kinematics_and_topology if not r.startswith(prefix)
        ]
        self.invariant_rules = [
            r for r in self.invariant_rules if not r.startswith(prefix)
        ]
        for inv in self.structured_invariants:
            if action_id.lower() in inv.description.lower() and inv.invariant_type == "kinematics":
                self.falsify_invariant(inv.invariant_id, level_id, reason=reason_str)

    def record_selection_mechanic(self, note: str) -> None:
        if note not in self.selection_mechanics:
            self.selection_mechanics.append(note)
        
        metadata = {}
        # Extract general actor/source entity from selection mechanic note.
        # The earliest relational keyword in the note wins, as an alternation scan did.
        source_pairs = parse_object_ids_after_keywords(note, ("moved", "controlled", "active", "source"))
        if source_pairs:
            metadata["init_source_entity"] = source_pairs[0][1]

        for action_id in parse_action_ids(note):
            metadata["action_id"] = action_id.upper()
            break

        # Extract displacement if present (both axes required)
        displacement = parse_displacement(note)
        if displacement is not None:
            metadata["dy"], metadata["dx"] = displacement

        inv_type = "kinematics"
        self.record_stratified_invariant(note, tier=2, invariant_type=inv_type, confidence=0.8, source="selection_probe", metadata=metadata)

    def record_reusable_primitive(self, fn_meta: dict[str, Any]) -> None:
        name = fn_meta.get("name")
        if name and not any(p.get("name") == name for p in self.reusable_primitives):
            self.reusable_primitives.append(fn_meta)

    def record_stratified_invariant(
        self,
        rule: str,
        tier: int = 1,
        invariant_type: str | None = None,
        confidence: float = 0.5,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StructuredInvariant | None:
        """Record an invariant into an explicit hierarchy (Tier 1, 2, or 3) and StructuredInvariant store."""
        rule = str(rule).strip()
        if not rule:
            return None
        if tier == 1:
            if rule not in self.tier1_kinematics_and_topology:
                self.tier1_kinematics_and_topology.append(rule)
        elif tier == 2:
            if rule not in self.tier2_interactions:
                self.tier2_interactions.append(rule)
        else:
            if rule not in self.tier3_level_rules:
                self.tier3_level_rules.append(rule)
        if rule not in self.invariant_rules:
            self.invariant_rules.append(rule)

        if invariant_type is None:
            _, invariant_type = _classify_tier(rule)

        existing = next((inv for inv in self.structured_invariants if inv.description == rule), None)
        if existing:
            return existing

        inv_id = f"inv_{len(self.structured_invariants) + 1}_{invariant_type}"
        new_inv = StructuredInvariant(
            invariant_id=inv_id,
            invariant_type=invariant_type,
            tier=tier,
            description=rule,
            confidence=confidence,
            ternary_status=Ternary.TRUE if confidence >= 0.7 else Ternary.IRRELEVANT,
            confirmed_on_levels=[f"level_{self.completed_levels}"],
            first_discovered_level=self.completed_levels,
            source=source,
            metadata=metadata or {},
        )
        self.structured_invariants.append(new_inv)
        return new_inv

    def record_invariant_rule(
        self,
        rule: str,
        tier: int | None = None,
        invariant_type: str | None = None,
    ) -> None:
        """Record an invariant rule, auto-classifying into appropriate tier if omitted."""
        if tier is None or invariant_type is None:
            auto_tier, auto_type = _classify_tier(rule)
            tier = tier if tier is not None else auto_tier
            invariant_type = invariant_type if invariant_type is not None else auto_type

        self.record_stratified_invariant(rule, tier=tier, invariant_type=invariant_type)

    def record_level_solution(
        self,
        level_id: str,
        setup_summary: str,
        invariant_rule: str,
        winning_macro: str = "",
    ) -> None:
        """Record a distilled solution pattern from a completed level."""
        if not any(p.get("level_id") == level_id for p in self.level_solution_patterns):
            self.level_solution_patterns.append({
                "level_id": level_id,
                "setup": setup_summary,
                "invariant": invariant_rule,
                "winning_macro": winning_macro,
            })
            if winning_macro and not mentions_action_call(winning_macro):
                self.record_stratified_invariant(
                    f"Goal invariant ({level_id}): {invariant_rule} (macro: {winning_macro})",
                    tier=3,
                    invariant_type="goal",
                    confidence=0.8,
                    source="level_solution",
                    metadata={"level_id": level_id, "winning_macro": winning_macro},
                )
            else:
                self.record_stratified_invariant(
                    f"Goal invariant ({level_id}): {invariant_rule}",
                    tier=3,
                    invariant_type="goal",
                    confidence=0.8,
                    source="level_solution",
                    metadata={"level_id": level_id},
                )

    def record_level_victory_example(
        self,
        level_id: str,
        winning_actions: list[str],
        object_diffs: list[dict[str, Any]],
        static_objects: list[str] | None = None,
        initial_objects: list[dict[str, Any]] | None = None,
        goal_rule: str = "",
        start_grid_hash: str = "",
        end_grid_hash: str = "",
    ) -> None:
        """Record an illustrative factual victory example from a completed level."""
        self.last_level_victory_example = LevelVictoryExample(
            level_id=str(level_id),
            winning_actions=list(winning_actions),
            object_diffs=list(object_diffs),
            static_objects=list(static_objects or []),
            initial_objects=list(initial_objects or []),
            goal_rule=str(goal_rule),
            start_grid_hash=start_grid_hash,
            end_grid_hash=end_grid_hash,
        )

    def format_previous_level_example(self) -> str:
        """Format last completed level victory as an illustrative example, not a prescriptive template."""
        if not self.last_level_victory_example:
            return ""
        ex = self.last_level_victory_example
        lines = [
            f"PREVIOUS LEVEL EXAMPLE (Level {ex.level_id} victory - illustrative only, do not blindly copy):",
        ]
        if ex.initial_objects:
            lines.append("- Initial State Objects:")
            for io in ex.initial_objects:
                name = io.get("name", io.get("alias", io.get("id", "object")))
                color = io.get("color", "?")
                bbox = io.get("bbox", [])
                role = io.get("role", "object")
                desc = io.get("description", "")
                extra = f": {desc}" if desc else ""
                lines.append(f"  * {name} (color {color}, bbox {bbox}, {role}){extra}")

        if ex.object_diffs:
            lines.append("- Observed Object Changes for Win:")
            for od in ex.object_diffs:
                alias = od.get("alias", od.get("id", "object"))
                dy = od.get("dy")
                dx = od.get("dx")
                color_change = od.get("color_change")
                change_parts = []
                if dy is not None and dx is not None and (dy != 0 or dx != 0):
                    change_parts.append(f"displaced by (dy={dy:+d}, dx={dx:+d})")
                if color_change:
                    change_parts.append(f"color {color_change}")
                if od.get("status"):
                    change_parts.append(str(od["status"]))
                details = ", ".join(change_parts) if change_parts else "transformed"
                lines.append(f"  * {alias}: {details}")
        if ex.static_objects:
            lines.append(f"- Objects Remaining Completely Static: {', '.join(ex.static_objects)}")
        if ex.winning_actions:
            lines.append(f"- Winning action sequence ({len(ex.winning_actions)} steps): {' -> '.join(ex.winning_actions[:15])}{'...' if len(ex.winning_actions) > 15 else ''}")
        if ex.goal_rule:
            lines.append(f"- Goal pattern observed: {ex.goal_rule}")
        lines.append("NOTE: This is an example of game logic from the previous level. Use it to understand how actions interact with shapes, not to assume the exact same goal on this level.")
        return "\n".join(lines)

    def _get_invariant(self, inv_id: str) -> StructuredInvariant | None:
        for inv in self.structured_invariants:
            if inv.invariant_id == inv_id:
                return inv
        return None

    def confirm_invariant(self, inv_id: str, level_id: str = "") -> None:
        """Increase confidence and promote invariant to Ternary.TRUE upon empirical confirmation."""
        inv = self._get_invariant(inv_id)
        if inv:
            if level_id and level_id not in inv.confirmed_on_levels:
                inv.confirmed_on_levels.append(level_id)
            inv.confidence = min(1.0, inv.confidence + 0.2)
            if len(inv.confirmed_on_levels) >= 2 or inv.confidence >= 0.7:
                inv.ternary_status = Ternary.TRUE

    def falsify_invariant(self, inv_id: str, level_id: str = "", reason: str = "") -> None:
        """Decrease confidence and set status to Ternary.FALSE upon empirical contradiction."""
        inv = self._get_invariant(inv_id)
        if inv:
            if level_id and level_id not in inv.falsified_on_levels:
                inv.falsified_on_levels.append(level_id)
            inv.confidence = max(0.0, inv.confidence - 0.3)
            inv.ternary_status = Ternary.FALSE
            fail_reason = reason or "Contradicted by empirical transition / zero grid delta / boundary collision"
            inv.metadata["falsification_reason"] = fail_reason
            if not any(entry.get("description") == inv.description for entry in self.invalidated_invariants):
                self.invalidated_invariants.append({
                    "description": inv.description,
                    "invariant_id": inv.invariant_id,
                    "invariant_type": inv.invariant_type,
                    "reason": fail_reason,
                    "level_id": level_id,
                })

    def get_invariants_for_revision(self) -> dict[str, list[dict[str, Any]]]:
        """Return active and invalidated invariants with full explanations and evidence for Solver revision."""
        active_list = []
        for inv in self.structured_invariants:
            if inv.ternary_status != Ternary.FALSE:
                evidence = []
                if inv.confirmed_on_levels:
                    evidence.append(f"Confirmed on {', '.join(inv.confirmed_on_levels)}")
                if inv.source:
                    evidence.append(f"Source: {inv.source}")
                if inv.confidence:
                    evidence.append(f"Confidence: {inv.confidence:.2f}")
                if inv.metadata and "init_actor" in inv.metadata:
                    evidence.append(f"Initial actor: {inv.metadata['init_actor']}")
                evidence_str = "; ".join(evidence) if evidence else "Observed structural property"

                active_list.append({
                    "id": inv.invariant_id,
                    "type": inv.invariant_type.upper(),
                    "description": inv.description,
                    "rule": inv.description,
                    "evidence": evidence_str,
                    "inclusion_reason": evidence_str,
                    "confidence": inv.confidence,
                })

        invalidated_list = []
        for item in self.invalidated_invariants:
            desc = item.get("description", "")
            r_str = item.get("reason", "Contradicted by empirical observation")
            invalidated_list.append({
                "description": desc,
                "rule": desc,
                "type": item.get("invariant_type", "GENERAL").upper(),
                "reason": r_str,
                "invalidation_reason": r_str,
                "level_id": item.get("level_id", ""),
                "action_id": item.get("action_id", ""),
            })

        return {"active": active_list, "invalidated": invalidated_list}

    def apply_invariant_revision(
        self,
        revised_invariants: list[str],
        outcome: str = "WIN",
        level_id: str = "",
    ) -> None:
        """Apply revised invariants from Solver, updating active pool and resolving invalidated ones."""
        if not revised_invariants:
            return

        level_idx = 0
        if level_id:
            num_m = re.search(r"\d+", level_id)
            if num_m:
                try:
                    level_idx = int(num_m.group(0))
                except ValueError:
                    level_idx = self.completed_levels
            else:
                level_idx = self.completed_levels
        else:
            level_idx = self.completed_levels

        for line in revised_invariants:
            rule_clean = line.strip().lstrip("-*•0123456789. ")
            if not rule_clean:
                continue

            # Check if this rule resolves or reformulates any previously invalidated invariant
            for inv_rec in list(self.invalidated_invariants):
                inv_desc = inv_rec.get("description", "")
                action_match = re.search(r"ACTION\d+", inv_desc, re.IGNORECASE)
                if action_match and action_match.group(0).lower() in rule_clean.lower():
                    self.invalidated_invariants.remove(inv_rec)

            # Route based on explicit category tag or classifier
            r_lower = rule_clean.lower()
            if r_lower.startswith("[negative_barrier]") or "negative_barrier" in r_lower:
                tier = 1
                inv_type = "hazard_barrier"
                cid_match = re.search(r"Color[_\s]+(\d+)", rule_clean, re.IGNORECASE)
                if cid_match:
                    cid = int(cid_match.group(1))
                    self.palette.assign_role(cid, EntityRole.HAZARD, confidence=0.85, evidence=f"LLM invariant revision: {rule_clean}")
                    inv_id = f"gi_hazard_color_{cid}"
                    existing_g = next((g for g in self.grounded_invariants if g.invariant_id == inv_id), None)
                    if existing_g:
                        existing_g.times_confirmed += 1
                        existing_g.confidence = min(1.0, existing_g.confidence + 0.15)
                    else:
                        self.grounded_invariants.append(GroundedInvariant(
                            invariant_id=inv_id,
                            antecedent=f"Contact(ACTOR, Color_{cid})",
                            consequent="DefeatReset()",
                            brusentsov_type="NEGATIVE_BARRIER",
                            scope="CORE_GAME_LAW",
                            times_confirmed=1,
                            confidence=0.85,
                            created_on_level=level_idx,
                        ))
                    emp_id = f"emp_hazard_color_{cid}"
                    emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                    if emp_inv:
                        emp_inv.confirm(level_idx, rule_clean)
                    else:
                        self.invariant_registry.register_candidate(EmpiricalInvariant(
                            invariant_id=emp_id,
                            invariant_type="hazard_barrier",
                            abstract_description=rule_clean,
                            subject_pattern=f"color=={cid}",
                            expected_value="DefeatReset()",
                            scope="CORE_GAME_LAW",
                            confirmed_on_levels=[level_idx],
                            times_confirmed=1,
                            confidence=0.85,
                        ))
                else:
                    emp_id = f"emp_nb_{abs(hash(rule_clean)) % 100000}"
                    emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                    if emp_inv:
                        emp_inv.confirm(level_idx, rule_clean)
                    else:
                        self.invariant_registry.register_candidate(EmpiricalInvariant(
                            invariant_id=emp_id,
                            invariant_type="hazard_barrier",
                            abstract_description=rule_clean,
                            subject_pattern="hazard",
                            expected_value="DefeatReset()",
                            scope="CORE_GAME_LAW",
                            confirmed_on_levels=[level_idx],
                            times_confirmed=1,
                            confidence=0.85,
                        ))

            elif r_lower.startswith("[positive_canon]") or "positive_canon" in r_lower:
                tier = 3
                inv_type = "goal_canon"
                cid_match = re.search(r"Color[_\s]+(\d+)", rule_clean, re.IGNORECASE)
                if cid_match:
                    cid = int(cid_match.group(1))
                    self.palette.assign_role(cid, EntityRole.TARGET, confidence=0.85, evidence=f"LLM invariant revision: {rule_clean}")
                    inv_id = f"gi_target_color_{cid}"
                    existing_g = next((g for g in self.grounded_invariants if g.invariant_id == inv_id), None)
                    if existing_g:
                        existing_g.times_confirmed += 1
                        existing_g.confidence = min(1.0, existing_g.confidence + 0.15)
                    else:
                        self.grounded_invariants.append(GroundedInvariant(
                            invariant_id=inv_id,
                            antecedent=f"Contact(ACTOR, Color_{cid})",
                            consequent="LevelVictory()",
                            brusentsov_type="POSITIVE_CANON",
                            scope="CORE_GAME_LAW",
                            times_confirmed=1,
                            confidence=0.85,
                            created_on_level=level_idx,
                        ))
                    emp_id = f"emp_target_color_{cid}"
                    emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                    if emp_inv:
                        emp_inv.confirm(level_idx, rule_clean)
                    else:
                        self.invariant_registry.register_candidate(EmpiricalInvariant(
                            invariant_id=emp_id,
                            invariant_type="goal_canon",
                            abstract_description=rule_clean,
                            subject_pattern=f"color=={cid}",
                            expected_value="LevelVictory()",
                            scope="CORE_GAME_LAW",
                            confirmed_on_levels=[level_idx],
                            times_confirmed=1,
                            confidence=0.85,
                        ))
                else:
                    emp_id = f"emp_pc_{abs(hash(rule_clean)) % 100000}"
                    emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                    if emp_inv:
                        emp_inv.confirm(level_idx, rule_clean)
                    else:
                        self.invariant_registry.register_candidate(EmpiricalInvariant(
                            invariant_id=emp_id,
                            invariant_type="goal_canon",
                            abstract_description=rule_clean,
                            subject_pattern="goal",
                            expected_value="LevelVictory()",
                            scope="CORE_GAME_LAW",
                            confirmed_on_levels=[level_idx],
                            times_confirmed=1,
                            confidence=0.85,
                        ))

            elif r_lower.startswith("[palette & roles]") or r_lower.startswith("[palette]") or r_lower.startswith("[roles]"):
                tier = 2
                inv_type = "palette_role"
                role_matches = list(re.finditer(r"Color[_\s]+(\d+)\s+(?:is|as|=|:|indicates|represents)\s+([A-Za-z_]+)", rule_clean, re.IGNORECASE))
                matched_any = False
                for rm in role_matches:
                    cid = int(rm.group(1))
                    role_str = rm.group(2).lower()
                    role_enum = ROLE_ALIASES.get(role_str)
                    if role_enum:
                        matched_any = True
                        self.palette.assign_role(cid, role_enum, confidence=0.85, evidence=f"LLM palette revision: {rule_clean}")
                        emp_id = f"emp_palette_{cid}_{role_enum.value}"
                        emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                        if emp_inv:
                            emp_inv.confirm(level_idx, role_enum.value)
                        else:
                            self.invariant_registry.register_candidate(EmpiricalInvariant(
                                invariant_id=emp_id,
                                invariant_type="palette_role",
                                abstract_description=f"Color {cid} is {role_enum.value.upper()}",
                                subject_pattern=f"color=={cid}",
                                expected_value=role_enum.value.upper(),
                                scope="CORE_GAME_LAW",
                                confirmed_on_levels=[level_idx],
                                times_confirmed=1,
                                confidence=0.85,
                            ))
                if not matched_any:
                    emp_id = f"emp_palette_{abs(hash(rule_clean)) % 100000}"
                    self.invariant_registry.register_candidate(EmpiricalInvariant(
                        invariant_id=emp_id,
                        invariant_type="palette_role",
                        abstract_description=rule_clean,
                        subject_pattern="palette",
                        expected_value="ROLES",
                        scope="CORE_GAME_LAW",
                        confirmed_on_levels=[level_idx],
                        times_confirmed=1,
                        confidence=0.85,
                    ))

            elif r_lower.startswith("[physics]") or any(kw in r_lower for kw in ("displacement", "velocity", "kinematic effect", "collision", "wall")):
                tier = 1
                inv_type = "kinematics"
                emp_id = f"emp_phys_{abs(hash(rule_clean)) % 100000}"
                emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                if emp_inv:
                    emp_inv.confirm(level_idx, rule_clean)
                else:
                    self.invariant_registry.register_candidate(EmpiricalInvariant(
                        invariant_id=emp_id,
                        invariant_type="kinematic_law",
                        abstract_description=rule_clean,
                        subject_pattern="physics",
                        expected_value="VALID",
                        scope="CORE_GAME_LAW" if outcome == "WIN" else "DOMAIN_PATTERN",
                        confirmed_on_levels=[level_idx],
                        times_confirmed=1,
                        confidence=0.85 if outcome == "WIN" else 0.5,
                    ))

            elif r_lower.startswith("[control]") or r_lower.startswith("[entities]") or r_lower.startswith("[structure]"):
                tier = 2
                inv_type = "control" if "control" in r_lower else "topology"
                emp_id = f"emp_{inv_type}_{abs(hash(rule_clean)) % 100000}"
                emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                if emp_inv:
                    emp_inv.confirm(level_idx, rule_clean)
                else:
                    self.invariant_registry.register_candidate(EmpiricalInvariant(
                        invariant_id=emp_id,
                        invariant_type=inv_type,
                        abstract_description=rule_clean,
                        subject_pattern=inv_type,
                        expected_value="VALID",
                        scope="CORE_GAME_LAW" if outcome == "WIN" else "DOMAIN_PATTERN",
                        confirmed_on_levels=[level_idx],
                        times_confirmed=1,
                        confidence=0.85 if outcome == "WIN" else 0.5,
                    ))

            elif r_lower.startswith("[goal]"):
                tier = 3
                inv_type = "goal"
                emp_id = f"emp_goal_{abs(hash(rule_clean)) % 100000}"
                emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                if emp_inv:
                    emp_inv.confirm(level_idx, rule_clean)
                else:
                    self.invariant_registry.register_candidate(EmpiricalInvariant(
                        invariant_id=emp_id,
                        invariant_type="goal",
                        abstract_description=rule_clean,
                        subject_pattern="goal",
                        expected_value="VALID",
                        scope="CORE_GAME_LAW" if outcome == "WIN" else "DOMAIN_PATTERN",
                        confirmed_on_levels=[level_idx],
                        times_confirmed=1,
                        confidence=0.85 if outcome == "WIN" else 0.5,
                    ))

            else:
                tier, inv_type = _classify_tier(rule_clean)
                emp_id = f"emp_{inv_type}_{abs(hash(rule_clean)) % 100000}"
                emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
                if emp_inv:
                    emp_inv.confirm(level_idx, rule_clean)
                else:
                    self.invariant_registry.register_candidate(EmpiricalInvariant(
                        invariant_id=emp_id,
                        invariant_type=inv_type,
                        abstract_description=rule_clean,
                        subject_pattern=inv_type,
                        expected_value="VALID",
                        scope="CORE_GAME_LAW" if outcome == "WIN" else "DOMAIN_PATTERN",
                        confirmed_on_levels=[level_idx],
                        times_confirmed=1,
                        confidence=0.8 if outcome == "WIN" else 0.4,
                    ))

            self.record_stratified_invariant(
                rule=rule_clean,
                tier=tier,
                invariant_type=inv_type,
                confidence=0.8 if outcome == "WIN" else 0.4,
                source=f"solver_revision_{outcome.lower()}",
            )

    def get_invariants_by_type(self, inv_type: str) -> list[StructuredInvariant]:
        return [inv for inv in self.structured_invariants if inv.invariant_type == inv_type]

    def get_invariants_by_tier(self, tier: int) -> list[StructuredInvariant]:
        return [inv for inv in self.structured_invariants if inv.tier == tier]

    def format_game_model_summary(self, include_goals: bool = True) -> str:
        """Format consolidated GameModel for Solver guidance based on all confirmed invariants."""
        confirmed = [
            inv for inv in self.structured_invariants
            if inv.ternary_status == Ternary.TRUE
            and (include_goals or inv.invariant_type != "goal")
        ]
        unconfirmed = [
            inv for inv in self.structured_invariants
            if inv.ternary_status == Ternary.IRRELEVANT
            and (include_goals or inv.invariant_type != "goal")
        ]
        falsified_count = sum(1 for inv in self.structured_invariants if inv.ternary_status == Ternary.FALSE)

        lines = [f"GAME MODEL (confirmed: {len(confirmed)}, pending: {len(unconfirmed)}, falsified: {falsified_count}):"]
        for inv in sorted(confirmed, key=lambda i: -i.confidence):
            lines.append(f"  ✓ [{inv.invariant_type}] {inv.description} (conf={inv.confidence:.1f}, levels={len(inv.confirmed_on_levels)})")
        for inv in unconfirmed[:5]:
            lines.append(f"  ? [{inv.invariant_type}] {inv.description} (conf={inv.confidence:.1f})")
        return "\n".join(lines)

    def re_evaluate_invariants(self, observations: Any, level_id: str = "") -> dict[str, int]:
        """Re-evaluate existing structured invariants against observations using Brusentsov ternary logic."""
        from v10_agent.brusentsov_logic import evaluate_invariant_across_levels
        stats = {"confirmed": 0, "falsified": 0, "unchanged": 0}
        for inv in self.structured_invariants:
            res = evaluate_invariant_across_levels(inv, observations)
            if res == Ternary.TRUE:
                self.confirm_invariant(inv.invariant_id, level_id)
                stats["confirmed"] += 1
            elif res == Ternary.FALSE:
                self.falsify_invariant(inv.invariant_id, level_id)
                stats["falsified"] += 1
            else:
                stats["unchanged"] += 1
        return stats

    def compare_and_record_invariants(self, current_invariants: list[Any]) -> dict[str, list[Any]]:
        """Compare current level invariants with previous level invariants."""
        from v10_agent.universal_invariants import compare_invariants_across_levels
        diff = compare_invariants_across_levels(self.last_discovered_invariants, current_invariants)
        self.last_discovered_invariants = list(current_invariants)
        return diff

    def format_curriculum_context(self) -> str:
        """Format distilled progression patterns across won levels for Solver guidance."""
        if not self.level_solution_patterns:
            return ""
        lines = ["CURRICULUM INVARIANTS & WINNING PATTERNS (CONFIRMED ACROSS PREVIOUS LEVELS):"]
        for p in self.level_solution_patterns:
            macro = p.get("winning_macro", "")
            if macro and not mentions_action_call(macro):
                lines.append(
                    f"- {p.get('level_id', 'Level')}: Setup: {p.get('setup')} | Invariant: {p.get('invariant')} | Strategy: {macro}"
                )
            else:
                lines.append(
                    f"- {p.get('level_id', 'Level')}: Setup: {p.get('setup')} | Invariant: {p.get('invariant')}"
                )
        return "\n".join(lines)

    def format_stratified_context(self, include_curriculum: bool = True) -> str:
        """Format invariants hierarchically by tier for structured LLM reasoning."""
        sections = []

        # Tier 1: Foundational Physics & Kinematics
        if self.confirmed_action_effects or self.tier1_kinematics_and_topology:
            sections.append("TIER 1: FOUNDATIONAL PHYSICS, KINEMATICS & BOUNDARIES (CONFIRMED ACTION KINEMATICS):")
            for act, eff in sorted(self.confirmed_action_effects.items()):
                sections.append(f"- {act}: {eff}")
            for item in self.tier1_kinematics_and_topology:
                if not any(item.startswith(f"Action {act}") for act in self.confirmed_action_effects):
                    sections.append(f"- {item}")

        # Tier 2: Interaction Dynamics & State Transitions
        if self.selection_mechanics or self.tier2_interactions:
            sections.append("TIER 2: INTERACTION DYNAMICS & STATE TRANSITIONS (CONFIRMED ENTITY SELECTION MECHANICS):")
            for m in self.selection_mechanics:
                sections.append(f"- {m}")
            for item in self.tier2_interactions:
                if item not in self.selection_mechanics:
                    sections.append(f"- {item}")

        # Tier 3: Level-Specific Rules & Curriculum Invariants
        if include_curriculum:
            tier3_items = list(self.tier3_level_rules)
            if tier3_items:
                sections.append("TIER 3: HIGH-LEVEL DEDUCED RULES & STRATEGIES:")
                for item in tier3_items:
                    sections.append(f"- {item}")

            if self.level_solution_patterns:
                sections.append(self.format_curriculum_context())

        if self.unconfirmed_actions:
            sections.append("UNCONFIRMED / INACTIVE ACTIONS (Do NOT call in solution plans):")
            for act, reason in sorted(self.unconfirmed_actions.items()):
                sections.append(f"- {act}: {reason}")

        return "\n".join(sections)

    def update_last_defeat(self, exemplar: DefeatExemplar) -> None:
        """Record the most recent defeat exemplar and auto-create a grounded hazard invariant."""
        self.add_defeat(exemplar)
        if exemplar.hazard_color >= 0:
            self.palette.assign_role(
                exemplar.hazard_color, EntityRole.HAZARD, confidence=0.9,
                evidence=f"Contact caused defeat on level {exemplar.level_index} step {exemplar.fatal_step}",
            )
            inv_id = f"gi_hazard_color_{exemplar.hazard_color}"
            existing = next((g for g in self.grounded_invariants if g.invariant_id == inv_id), None)
            if existing:
                existing.grounded_in_defeat = exemplar
                existing.times_confirmed += 1
                existing.confidence = min(1.0, existing.confidence + 0.2)
            else:
                self.grounded_invariants.append(GroundedInvariant(
                    invariant_id=inv_id,
                    antecedent=f"Contact(ACTOR, Color_{exemplar.hazard_color})",
                    consequent="DefeatReset()",
                    brusentsov_type="NEGATIVE_BARRIER",
                    scope="CORE_GAME_LAW",
                    grounded_in_defeat=exemplar,
                    times_confirmed=1, confidence=0.9,
                    created_on_level=exemplar.level_index,
                ))
            emp_id = f"emp_hazard_color_{exemplar.hazard_color}"
            emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
            if emp_inv:
                emp_inv.confirm(exemplar.level_index, f"Defeat on step {exemplar.fatal_step}")
            else:
                self.invariant_registry.register_candidate(EmpiricalInvariant(
                    invariant_id=emp_id,
                    invariant_type="hazard_barrier",
                    abstract_description=f"Contact with Color {exemplar.hazard_color} causes defeat",
                    subject_pattern=f"color=={exemplar.hazard_color}",
                    expected_value="DefeatReset()",
                    scope="CORE_GAME_LAW",
                    confirmed_on_levels=[exemplar.level_index],
                    times_confirmed=1,
                    confidence=0.9,
                ))

    def update_last_victory(self, exemplar: VictoryExemplar) -> None:
        """Record the most recent victory exemplar and auto-create a grounded target invariant."""
        self.add_victory(exemplar)
        if exemplar.target_color >= 0:
            self.palette.assign_role(
                exemplar.target_color, EntityRole.TARGET, confidence=0.9,
                evidence=f"Contact triggered victory on level {exemplar.level_index}",
            )
            inv_id = f"gi_target_color_{exemplar.target_color}"
            existing = next((g for g in self.grounded_invariants if g.invariant_id == inv_id), None)
            if existing:
                existing.grounded_in_victory = exemplar
                existing.times_confirmed += 1
                existing.confidence = min(1.0, existing.confidence + 0.2)
            else:
                self.grounded_invariants.append(GroundedInvariant(
                    invariant_id=inv_id,
                    antecedent=f"Contact(ACTOR, Color_{exemplar.target_color})",
                    consequent="LevelVictory()",
                    brusentsov_type="POSITIVE_CANON",
                    scope="CORE_GAME_LAW",
                    grounded_in_victory=exemplar,
                    times_confirmed=1, confidence=0.9,
                    created_on_level=exemplar.level_index,
                ))
            emp_id = f"emp_target_color_{exemplar.target_color}"
            emp_inv = next((i for i in self.invariant_registry.invariants if i.invariant_id == emp_id), None)
            if emp_inv:
                emp_inv.confirm(exemplar.level_index, f"Victory in {exemplar.total_steps} steps")
            else:
                self.invariant_registry.register_candidate(EmpiricalInvariant(
                    invariant_id=emp_id,
                    invariant_type="goal_canon",
                    abstract_description=f"Reaching Color {exemplar.target_color} completes level",
                    subject_pattern=f"color=={exemplar.target_color}",
                    expected_value="LevelVictory()",
                    scope="CORE_GAME_LAW",
                    confirmed_on_levels=[exemplar.level_index],
                    times_confirmed=1,
                    confidence=0.9,
                ))

    def record_curriculum_transition(
        self,
        level_from: int,
        level_to: int,
        delta_summary: str,
        steps_to_win: int | None = None,
        invariants_confirmed: list[str] | None = None,
    ) -> None:
        """Record what changed between levels for curriculum learning."""
        inv_list = invariants_confirmed if invariants_confirmed is not None else []
        active_core_count = len([g for g in self.grounded_invariants if g.is_active() and g.scope == "CORE_GAME_LAW"])
        self.curriculum_history.append({
            "level": level_from,
            "from_level": level_from,
            "to_level": level_to,
            "summary": delta_summary,
            "steps_to_win": steps_to_win,
            "steps": steps_to_win,
            "invariants_confirmed": inv_list,
            "core_invariants_carried": active_core_count,
            "timestamp": time.time(),
        })

    def format_grounded_memory_for_prompt(self) -> str:
        """Format the complete grounded memory block for Solver Turn-1 prompt."""
        sections = []
        palette_str = self.palette.format_for_prompt()
        if palette_str:
            sections.append(palette_str)
        if self.last_defeat_exemplar:
            sections.append(self.last_defeat_exemplar.format_for_prompt())
        patterns = self.analyze_defeat_patterns()
        if patterns:
            sections.append("RECURRING DEFEAT PATTERNS (avoid these combinations):\n" + "\n".join(f"  - {p}" for p in patterns))
        if self.last_victory_exemplar_v2:
            sections.append(self.last_victory_exemplar_v2.format_for_prompt())
        active_core = [g for g in self.grounded_invariants if g.is_active() and g.scope == "CORE_GAME_LAW"]
        if active_core:
            sections.append("CORE GAME LAWS (verified across levels):")
            for g in active_core:
                sections.append(f"  {g.format_for_prompt()}")
        if self.invariant_registry and self.invariant_registry.invariants:
            sections.append(self.invariant_registry.format_for_prompt())
        return "\n\n".join(sections)

    def clear(self) -> None:
        self.confirmed_action_effects.clear()
        self.unconfirmed_actions.clear()
        self.selection_mechanics.clear()
        self.reusable_primitives.clear()
        self.invariant_rules.clear()
        self.tier1_kinematics_and_topology.clear()
        self.tier2_interactions.clear()
        self.tier3_level_rules.clear()
        self.level_solution_patterns.clear()
        self.structured_invariants.clear()
        self.completed_levels = 0

    def handle_level_transition(self) -> None:
        """Sanitize level-transient object IDs and coordinates when moving to a new level (ISO-7)."""
        clean_sm = []
        for s in self.selection_mechanics:
            s_clean = self._generalize_text(s)
            s_clean = strip_annotated_segments(s_clean, ("axis_steps", "piece_steps"))
            if s_clean not in clean_sm:
                clean_sm.append(s_clean)
        self.selection_mechanics = clean_sm

        sanitized = {}
        for act, eff in self.confirmed_action_effects.items():
            clean_eff = self._generalize_text(eff)
            sanitized[act] = clean_eff
        self.confirmed_action_effects = sanitized

        def is_level_transient(text: str) -> bool:
            t_low = text.lower()
            if "displacement of dy=" in t_low or "displacement of" in t_low:
                return False
            if "moves entity" in t_low or "kinematic effect" in t_low:
                return False
            transient_patterns = [
                r"\brows?\s*\d+",
                r"\bcols?\s*\d+",
                r"\bbbox\b",
                r"\bcentroid\b",
                r"\bfailed\b",
                r"\bat\s+row\b",
                r"\bat\s+col\b",
                r"\(\d+\s*,\s*\d+\)",
                r"\btook\s+\d+\s*steps\b",
                r"\bin\s+\d+\s*steps\b",
                r"\bwithin\s+\d+\s*steps\b",
                r"\bneed\s+\d+\s*steps\b",
            ]
            return any(re.search(p, t_low) for p in transient_patterns)

        self.tier1_kinematics_and_topology = [
            self._generalize_text(s)
            for s in self.tier1_kinematics_and_topology
            if not is_level_transient(s)
        ]
        self.tier2_interactions = [
            self._generalize_text(s)
            for s in self.tier2_interactions
            if not is_level_transient(s)
        ]
        # Tier 3: preserve domain-general invariants, filter out level-specific transient rules
        self.tier3_level_rules = [
            self._generalize_text(r)
            for r in self.tier3_level_rules
            if not self._is_level_specific(r) and not is_level_transient(r)
        ]

        # Sanitize descriptions of existing structured invariants and filter level-specific ones from Tier 3
        surviving_invariants: list[StructuredInvariant] = []
        for inv in self.structured_invariants:
            inv.description = self._generalize_text(inv.description)
            if inv.tier == 3:
                if not self._is_level_specific(inv.description) and not is_level_transient(inv.description):
                    surviving_invariants.append(inv)
            else:
                if not is_level_transient(inv.description):
                    surviving_invariants.append(inv)
        self.structured_invariants = surviving_invariants

    def format_empirical_context(self, include_curriculum: bool = True) -> str:
        """Format purely empirical facts observed across levels for LLM guidance."""
        lines = []
        stratified = self.format_stratified_context(include_curriculum=include_curriculum)
        if stratified:
            lines.append(stratified)

        if self.structured_invariants:
            model_summary = self.format_game_model_summary(include_goals=include_curriculum)
            if model_summary and "confirmed: 0, pending: 0, falsified: 0" not in model_summary:
                lines.append(f"\n{model_summary}")

        if self.reusable_primitives:
            lines.append("\nREUSABLE CERTIFIED MOVEMENT PRIMITIVES FROM PREVIOUS LEVELS:")
            for p in self.reusable_primitives:
                params = ", ".join(param.get("name", "") for param in p.get("parameters", []))
                lines.append(f"- {p.get('name')}(api, {params}): {p.get('docstring', '')}")
        return "\n".join(lines)


class MemoryContourManager:
    """Sole owner and mediator of the four memory contours (ISO-4)."""

    def __init__(self, game_id: str = "init_game", level_id: str = "init_level"):
        self.env_spec_memory = EnvironmentSpecMemory(game_id=game_id, level_id=level_id)
        self.syntax_error_memory = SyntaxErrorMemory(level_id=level_id)
        self.epistemic_memory = EpistemicMemory(level_id=level_id)
        self.game_memory = GameMemory(game_id=game_id)

    def get_env_spec_memory(self, caller_role: str) -> EnvironmentSpecMemory:
        """Authorized for Explorer and GameSession only."""
        if caller_role not in {"explorer", "session"}:
            raise IsolationViolationError(
                f"ISO-4 Violation: Role {caller_role!r} is not permitted to access EnvironmentSpecMemory"
            )
        return self.env_spec_memory

    def get_syntax_error_memory(self, caller_role: str) -> SyntaxErrorMemory:
        """Authorized for Coder and GameSession only."""
        if caller_role not in {"coder", "session"}:
            raise IsolationViolationError(
                f"ISO-4 Violation: Role {caller_role!r} is not permitted to access SyntaxErrorMemory"
            )
        return self.syntax_error_memory

    def get_epistemic_memory(self, caller_role: str) -> EpistemicMemory:
        """Authorized for Solver and GameSession only."""
        if caller_role not in {"solver", "session"}:
            raise IsolationViolationError(
                f"ISO-4 Violation: Role {caller_role!r} is not permitted to access EpistemicMemory"
            )
        return self.epistemic_memory

    def get_game_memory(self, caller_role: str) -> GameMemory:
        """Authorized for GameSession only."""
        if caller_role != "session":
            raise IsolationViolationError(
                f"ISO-4 Violation: Role {caller_role!r} is not permitted to access GameMemory"
            )
        return self.game_memory

    def handle_level_transition(self, new_level_id: str) -> None:
        """Preserve GameMemory, summarize EpistemicMemory, clear SyntaxErrorMemory."""
        self.syntax_error_memory.clear()
        self.epistemic_memory.clear()
        self.env_spec_memory.clear_level(new_level_id)
        self.game_memory.handle_level_transition()
        self.game_memory.completed_levels += 1

    def handle_game_transition(self, new_game_id: str, new_level_id: str) -> None:
        """Completely reset all contours when switching games."""
        self.env_spec_memory = EnvironmentSpecMemory(game_id=new_game_id, level_id=new_level_id)
        self.syntax_error_memory = SyntaxErrorMemory(level_id=new_level_id)
        self.epistemic_memory = EpistemicMemory(level_id=new_level_id)
        self.game_memory = GameMemory(game_id=new_game_id)
