"""
Memory Contours Implementation with Strict Isolation.

Ref: Spec 1.4 (Isolation Invariants), Spec 3.5 (Memory Contours)

This module implements three strictly separated memory contours:
1. EnvironmentSpecMemory - For environment specifications and probe history
2. SyntaxErrorMemory - For syntax error tracking (max 5 entries)
3. EpistemicMemory - For Brusentsov judgments and branch management

ISO-3 Invariant: These classes MUST NOT have methods to read each other's contents.
Each contour is completely isolated and can only be accessed through explicit interfaces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence, Optional, List, Dict

from .types import (
    PropositionSet,
    EffectDeclaration,
    ProbeRecord,
    SyntaxErrorRecord,
    BrusentsovJudgment,
    BranchSignature,
    EnvironmentSpecification,
)


@dataclass(frozen=True)
class EnvironmentSpecMemory:
    """
    Memory contour for environment specifications and probe history.
    
    Ref: Spec 3.5.1 - EnvironmentSpecMemory
    
    This memory contour stores:
    - Current EnvironmentSpecification
    - History of probe actions and their results (max 1000 entries)
    
    ISO-3 Compliance: This class has NO methods to access SyntaxErrorMemory or
    EpistemicMemory contents. It is completely isolated.
    """
    
    # Current environment specification
    current_spec: EnvironmentSpecification | None = None
    
    # History of probe records (tuple for immutability, maxlen 1000)
    _probe_history: tuple[ProbeRecord, ...] = field(default_factory=tuple)
    MAX_PROBES: int = 1000

    def add_probe(self, record: ProbeRecord) -> EnvironmentSpecMemory:
        """
        Add a probe record to the history and return a new EnvironmentSpecMemory instance.
        
        Args:
            record: The probe record to add
            
        Returns:
            New EnvironmentSpecMemory instance with updated probe history

        Raises:
            ValueError: If no EnvironmentSpecification is set
        """
        if self.current_spec is None:
            raise ValueError("Cannot add probe without EnvironmentSpecification")
        new_history = self._probe_history + (record,)
        if len(new_history) > self.MAX_PROBES:
            new_history = new_history[-self.MAX_PROBES:]
        return EnvironmentSpecMemory(current_spec=self.current_spec, _probe_history=new_history)
    
    def get_probe_history(self) -> Sequence[ProbeRecord]:
        """
        Get the probe history as an immutable sequence.
        
        Returns:
            Sequence of probe records in chronological order
        """
        return self._probe_history
    
    def get_recent_probes(self, count: int) -> Sequence[ProbeRecord]:
        """
        Get the most recent probe records.
        
        Args:
            count: Number of recent probes to retrieve
            
        Returns:
            Sequence of up to `count` most recent probe records
        """
        return self._probe_history[-count:] if len(self._probe_history) > count else self._probe_history
    
    def set_specification(self, spec: EnvironmentSpecification) -> EnvironmentSpecMemory:
        """
        Set the current environment specification and return a new EnvironmentSpecMemory instance.
        
        Args:
            spec: The environment specification to set

        Returns:
            New EnvironmentSpecMemory instance with updated specification
        """
        return EnvironmentSpecMemory(current_spec=spec, _probe_history=self._probe_history)
    
    def clear_probes(self) -> EnvironmentSpecMemory:
        """Clear all probe history and return a new EnvironmentSpecMemory instance."""
        return EnvironmentSpecMemory(current_spec=self.current_spec, _probe_history=())
    
    @property
    def probe_count(self) -> int:
        """Get the number of probe records in history."""
        return len(self._probe_history)


@dataclass(frozen=True)
class SyntaxErrorMemory:
    """
    Memory contour for syntax error tracking.
    
    Ref: Spec 3.5.2 - SyntaxErrorMemory
    
    This memory contour stores syntax errors from Coder agent failures.
    Maximum capacity: 5 entries (as per Spec 2.1).
    
    ISO-3 Compliance: This class has NO methods to access EnvironmentSpecMemory or
    EpistemicMemory contents. It is completely isolated.
    """
    
    # Maximum capacity as per Spec 2.1
    MAX_ENTRIES: int = 5
    
    # Internal storage (tuple for immutability)
    _errors: tuple[SyntaxErrorRecord, ...] = field(default_factory=tuple)
    
    def add_error(
        self,
        prompt_hash: str,
        source_hash: str,
        traceback: str,
        level_id: str = "unknown",
        static_diagnostics: list[str] | None = None,
        timestamp: int = 0,
    ) -> SyntaxErrorMemory:
        """
        Add a syntax error record and return a new SyntaxErrorMemory instance.
        
        Args:
            prompt_hash: Hash of the prompt that caused the error
            source_hash: Hash of the source code that failed
            traceback: The traceback string
            level_id: Level where the error occurred (default: "unknown")
            static_diagnostics: Optional static analysis diagnostics
            timestamp: Error occurrence timestamp
            
        Returns:
            New SyntaxErrorMemory instance with updated errors

        Note:
            If capacity is exceeded, the oldest error is removed (FIFO).
        """
        record = SyntaxErrorRecord(
            level_id=level_id,
            prompt_hash=prompt_hash,
            source_hash=source_hash,
            traceback=traceback,
            static_diagnostics=static_diagnostics or [],
            timestamp=timestamp,
        )
        
        new_errors = self._errors + (record,)
        if len(new_errors) > self.MAX_ENTRIES:
            new_errors = new_errors[-self.MAX_ENTRIES:]

        return SyntaxErrorMemory(_errors=new_errors)
    
    def get_errors(self) -> Sequence[SyntaxErrorRecord]:
        """
        Get all syntax error records.
        
        Returns:
            Sequence of syntax error records in chronological order
        """
        return self._errors
    
    def get_recent_errors(self, count: int) -> Sequence[SyntaxErrorRecord]:
        """
        Get the most recent syntax error records.
        
        Args:
            count: Number of recent errors to retrieve
            
        Returns:
            Sequence of up to `count` most recent error records
        """
        return self._errors[-count:] if len(self._errors) > count else self._errors

    def get_recent_summaries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent error summaries for Coder feedback.

        Args:
            limit: Maximum number of error summaries to return

        Returns:
            List of dicts with summary and level_id keys
        """
        recent = self.get_recent_errors(limit)
        result = []
        for record in recent:
            summary_text = getattr(record, 'summary', None) or record.traceback or ""
            result.append({
                "summary": summary_text[:200],
                "level_id": record.level_id
            })
        return result
    
    def clear(self) -> SyntaxErrorMemory:
        """Clear all syntax error records and return a new SyntaxErrorMemory instance."""
        return SyntaxErrorMemory(_errors=())
    
    @property
    def error_count(self) -> int:
        """Get the number of error records stored."""
        return len(self._errors)
    
    def has_capacity(self) -> bool:
        """Check if there is capacity for more errors."""
        return len(self._errors) < self.MAX_ENTRIES


@dataclass(frozen=True)
class EpistemicMemory:
    """
    Memory contour for epistemic reasoning and branch management.
    
    Ref: Spec 3.5.3 - EpistemicMemory
    
    This memory contour stores:
    - Brusentsov judgments from Solver evaluations
    - Live OMIT branches (IRRELEVANT outcomes)
    - Severed NULL branch signatures
    
    ISO-3 Compliance: This class has NO methods to access EnvironmentSpecMemory or
    SyntaxErrorMemory contents. It is completely isolated.
    """
    
    # Internal storage (immutable collections)
    _judgments: tuple[BrusentsovJudgment, ...] = field(default_factory=tuple)
    _live_omit_branches: frozenset[BranchSignature] = field(default_factory=frozenset)
    _severed_null_signatures: frozenset[BranchSignature] = field(default_factory=frozenset)
    
    def add_judgment(self, judgment: BrusentsovJudgment) -> EpistemicMemory:
        """
        Add a Brusentsov judgment and return a new EpistemicMemory instance.
        
        Args:
            judgment: The judgment to add

        Returns:
            New EpistemicMemory instance with updated state
        """
        new_judgments = self._judgments + (judgment,)
        new_live_omit = set(self._live_omit_branches)
        new_severed_null = set(self._severed_null_signatures)
        
        if judgment.judgment_type == "OMIT":
            new_live_omit.add(judgment.branch_signature)
        elif judgment.judgment_type == "NULL":
            new_severed_null.add(judgment.branch_signature)

        return EpistemicMemory(
            _judgments=new_judgments,
            _live_omit_branches=frozenset(new_live_omit),
            _severed_null_signatures=frozenset(new_severed_null)
        )
    
    def get_judgments(self) -> Sequence[BrusentsovJudgment]:
        """
        Get all Brusentsov judgments.
        
        Returns:
            Sequence of judgments in chronological order
        """
        return self._judgments

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregated epistemic summary for Solver.

        Returns:
            Dict containing counts of judgments, live OMIT, and severed NULL branches
        """
        return {
            "judgment_count": self.judgment_count,
            "live_omit_count": self.live_omit_count,
            "severed_null_count": self.severed_null_count,
        }
    
    def get_live_omit_branches(self, limit: Optional[int] = None) -> frozenset[BranchSignature]:
        """
        Get live OMIT branches.
        
        Args:
            limit: Optional maximum number of branches to return

        Returns:
            FrozenSet of branch signatures that can be omitted
        """
        if limit is None:
            return self._live_omit_branches
        return frozenset(list(self._live_omit_branches)[:limit])
    
    def get_severed_null_signatures(self, limit: Optional[int] = None) -> frozenset[BranchSignature]:
        """
        Get severed NULL branch signatures.
        
        Args:
            limit: Optional maximum number of signatures to return

        Returns:
            FrozenSet of branch signatures that caused contradictions
        """
        if limit is None:
            return self._severed_null_signatures
        return frozenset(list(self._severed_null_signatures)[:limit])
    
    def is_branch_omittable(self, signature: BranchSignature) -> bool:
        """
        Check if a branch can be safely omitted.
        
        Args:
            signature: The branch signature to check
            
        Returns:
            True if the branch is in live_omit_branches
        """
        return signature in self._live_omit_branches
    
    def is_branch_severed(self, signature: BranchSignature) -> bool:
        """
        Check if a branch has been severed due to contradiction.
        
        Args:
            signature: The branch signature to check
            
        Returns:
            True if the branch is in severed_null_signatures
        """
        return signature in self._severed_null_signatures
    
    def prune_omit_branch(self, signature: BranchSignature) -> EpistemicMemory:
        """
        Remove a branch from live OMIT branches and return a new EpistemicMemory instance.
        
        Args:
            signature: The branch signature to remove

        Returns:
            New EpistemicMemory instance without the pruned branch
        """
        new_live_omit = set(self._live_omit_branches)
        new_live_omit.discard(signature)
        return EpistemicMemory(
            _judgments=self._judgments,
            _live_omit_branches=frozenset(new_live_omit),
            _severed_null_signatures=self._severed_null_signatures
        )
    
    def clear(self) -> EpistemicMemory:
        """Clear all epistemic memory and return a new EpistemicMemory instance."""
        return EpistemicMemory()
    
    @property
    def judgment_count(self) -> int:
        """Get the number of judgments stored."""
        return len(self._judgments)
    
    @property
    def live_omit_count(self) -> int:
        """Get the number of live OMIT branches."""
        return len(self._live_omit_branches)
    
    @property
    def severed_null_count(self) -> int:
        """Get the number of severed NULL branches."""
        return len(self._severed_null_signatures)


@dataclass(frozen=True)
class GameMemory:
    """
    Cross-level memory for game summaries and meta-information.
    
    Ref: Spec 3.5.4 - GameMemory (Inter-Level Summaries)
    
    This memory contour stores information that persists across levels:
    - Level completion summaries
    - Strategy effectiveness metrics
    - Global constraints discovered
    
    Note: This is separate from the three isolated contours and serves as
    a higher-level aggregation layer.
    """
    
    # Level summaries (level_id -> summary)
    _level_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Global strategy metrics
    _strategy_metrics: dict[str, float] = field(default_factory=dict)
    
    # Discovered global constraints
    _global_constraints: list[str] = field(default_factory=list)
    
    def add_level_summary(self, level_id: str, summary: dict[str, Any]) -> None:
        """
        Add a summary for a completed level.
        
        Args:
            level_id: The level identifier
            summary: Dictionary containing level completion details
        """
        self._level_summaries[level_id] = summary
    
    def get_level_summary(self, level_id: str) -> dict[str, Any] | None:
        """
        Get the summary for a specific level.
        
        Args:
            level_id: The level identifier
            
        Returns:
            The level summary dictionary, or None if not found
        """
        return self._level_summaries.get(level_id)
    
    def update_strategy_metric(self, metric_name: str, value: float) -> None:
        """
        Update a strategy metric.
        
        Args:
            metric_name: Name of the metric
            value: New value for the metric
        """
        self._strategy_metrics[metric_name] = value
    
    def get_strategy_metric(self, metric_name: str) -> float | None:
        """
        Get a strategy metric value.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            The metric value, or None if not found
        """
        return self._strategy_metrics.get(metric_name)
    
    def add_global_constraint(self, constraint: str) -> None:
        """
        Add a discovered global constraint.
        
        Args:
            constraint: The constraint description
        """
        if constraint not in self._global_constraints:
            self._global_constraints.append(constraint)
    
    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """
        Get all level summaries.
        
        Returns:
            Dictionary mapping level IDs to summaries
        """
        return dict(self._level_summaries)
    
    def clear(self) -> None:
        """Clear all game memory."""
        self._level_summaries.clear()
        self._strategy_metrics.clear()
        self._global_constraints.clear()
