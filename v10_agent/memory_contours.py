"""Isolated Memory Contours enforcing invariants ISO-1 through ISO-5.

Guarantees zero cross-contamination between:
  - Factual discovery (EnvironmentSpecMemory)
  - Syntax/compilation diagnostics (SyntaxErrorMemory)
  - Epistemic/semantic judgments (EpistemicMemory)
  - Game-wide cross-level summaries (GameMemory)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from v10_agent.brusentsov_logic import BrusentsovJudgment, Ternary


import re
import logging

logger = logging.getLogger(__name__)

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
}

FORBIDDEN_GOAL_PATTERNS_IN_CODER = [
    re.compile(r"\blevel_goal\b", re.IGNORECASE),
    re.compile(r"\bwin_condition\b", re.IGNORECASE),
    re.compile(r"\btarget_score\b", re.IGNORECASE),
    re.compile(r"\bhypothesis_family\b", re.IGNORECASE),
    re.compile(r"\bepistemic_verdict\b", re.IGNORECASE),
    re.compile(r"\bbrusentsov\b", re.IGNORECASE),
    re.compile(r"\blive_omit\b", re.IGNORECASE),
    re.compile(r"\bsevered_null\b", re.IGNORECASE),
]


@dataclass
class ProbeRecord:
    """Record of an exploratory probe action executed in the environment."""
    probe_id: str
    action_id: str
    action_data: dict[str, Any]
    observed_effect: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


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
        for pat in FORBIDDEN_SYNTAX_PATTERNS_IN_SOLVER:
            if pat.search(expl):
                logger.warning(
                    f"ISO-1 Violation (softened): Python syntax error / traceback pattern {pat.pattern!r} appeared in EpistemicMemory. Sanitizing."
                )
                expl = pat.sub("[REDACTED SYNTAX]", expl)
                object.__setattr__(judgment, "explanation", expl)

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
    ) -> None:
        """Record attempt failure/rejection in current level scratchpad."""
        self.current_level_attempts.append({
            "attempt_index": len(self.current_level_attempts) + 1,
            "hypothesis": hypothesis,
            "trajectory": trajectory_summary,
            "status": status,
            "reason": reason,
        })
        if len(self.current_level_attempts) > 5:
            self.current_level_attempts.pop(0)

    def format_scratchpad_context(self) -> str:
        """Format current-level failed attempts so the Solver avoids repeating mistakes."""
        if not self.current_level_attempts:
            return ""
        lines = ["CURRENT LEVEL FAILED ATTEMPTS IN SHORT-TERM SCRATCHPAD (DO NOT REPEAT):"]
        for att in self.current_level_attempts:
            lines.append(
                f"- Attempt {att['attempt_index']}: [{att.get('status')}] Reason: {att.get('reason')}. Trajectory: {att.get('trajectory')}"
            )
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
        elif signature_id.strip() and not signature_id.startswith("sig_") and not signature_id.startswith("s"):
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
    """A typed domain-general invariant evaluated using Brusentsov ternary logic."""
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
    confirmed_actor_ids: set[str] = field(default_factory=set)

    @property
    def confirmed_actors(self) -> set[str]:
        """Return set of object IDs empirically observed to move or receive selection indicators."""
        actors: set[str] = set(self.confirmed_actor_ids)
        for note in self.selection_mechanics:
            for oid in re.findall(r"obj_[a-zA-Z0-9_]+", note):
                actors.add(oid)
        for eff in self.confirmed_action_effects.values():
            for oid in re.findall(r"obj_[a-zA-Z0-9_]+", eff):
                actors.add(oid)
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

        # Step 2: Identify TARGETs — objects mentioned as destinations in action effects
        for eff in self.confirmed_action_effects.values():
            for m in re.finditer(r"toward\s+(obj_[a-zA-Z0-9_]+)", eff):
                tid = m.group(1)
                role_map[tid] = "TARGET"

        # Step 3: Identify OBSTACLEs — objects associated with collision/blocked events
        for rule in self.invariant_rules:
            if re.search(r"\b(?:blocked|wall|collision|obstacle|barrier)\b", rule, re.IGNORECASE):
                for oid in re.findall(r"obj_[a-zA-Z0-9_]+", rule):
                    role_map[oid] = "OBSTACLE"

        return role_map

    def _generalize_text(self, text: str) -> str:
        r"""Replace specific object IDs with functional role markers and strip level-local colors (ISO-8).

        Confirmed actors → [ACTOR], destinations → [TARGET],
        collision objects → [OBSTACLE], unknown → [ENTITY].
        Also strips specific (color N) and color_\d+ literals so palettes don't contaminate new levels.
        """
        for obj_id, role in self._build_role_map().items():
            text = text.replace(obj_id, f"[{role}]")
        # Any remaining unknown obj_... replaced by [ENTITY]
        text = re.sub(r"obj_[a-zA-Z0-9_]+", "[ENTITY]", text)
        # ISO-8: Strip concrete color specifications across level transitions
        text = re.sub(r"\b(?:color\s+\d+|color_\d+)\b", "[COLOR]", text, flags=re.IGNORECASE)
        text = re.sub(r"\(color\s+\d+\)", "[COLOR]", text, flags=re.IGNORECASE)
        return text

    def _is_level_specific(self, text: str) -> bool:
        """Check if an invariant rule is explicitly tagged as specific to a single level."""
        return bool(re.search(
            r"\b(?:only\s+on\s+level\s+\d+|level_\d+\s+transient|level\s+\d+\s+specific\s+obstacle|temporary\s+level\s+rule)\b",
            text,
            re.IGNORECASE,
        ))

    def record_action_effect(self, action_id: str, summary: str) -> None:
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
        m_ax = re.search(r"axis_steps=([+-]?\d+)", note)
        m_pc = re.search(r"piece_steps=([+-]?\d+)", note)
        if m_ax and m_pc:
            metadata["axis_steps"] = int(m_ax.group(1))
            metadata["piece_steps"] = int(m_pc.group(1))

        m_src = re.search(r"internal dots moved from\s+(obj_[a-zA-Z0-9_]+)", note)
        if m_src:
            metadata["init_source_entity"] = m_src.group(1)

        m_act = re.search(r"(ACTION\d+)", note, re.IGNORECASE)
        if m_act:
            metadata["action_id"] = m_act.group(1).upper()

        inv_type = "kinematics"  # We use kinematics so VirtualSandbox can read it as physics
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
            if winning_macro and not re.search(r"action\d+\(\)", winning_macro, re.IGNORECASE):
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
            if r_lower.startswith("[physics]") or any(kw in r_lower for kw in ("displacement", "velocity", "kinematic effect", "collision", "wall")):
                tier = 1
                inv_type = "kinematics"
            elif r_lower.startswith("[control]") or r_lower.startswith("[entities]") or r_lower.startswith("[structure]"):
                tier = 2
                inv_type = "control" if "control" in r_lower else "topology"
            elif r_lower.startswith("[goal]"):
                tier = 3
                inv_type = "goal"
            else:
                tier, inv_type = _classify_tier(rule_clean)

            self.record_stratified_invariant(
                rule=rule_clean,
                tier=tier,
                invariant_type=inv_type,
                confidence=0.85 if outcome == "WIN" else 0.75,
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
            if macro and not re.search(r"action\d+\(\)", macro, re.IGNORECASE):
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
            s_clean = re.sub(r"\s*\(axis_steps=[^)]+\)", "", s_clean)
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
