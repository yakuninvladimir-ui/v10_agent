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
from typing import Any, Sequence
from collections import deque

from .types import (
    PropositionSet,
    EffectDeclaration,
    ProbeRecord,
    SyntaxErrorRecord,
    BrusentsovJudgment,
    BranchSignature,
)


@dataclass
class EnvironmentSpecification:
    """
    Environment specification for a task level.
    
    Ref: Spec 3.5.1 - EnvironmentSpecMemory
    
    Contains the canonical description of the environment including:
    - Initial state propositions
    - Goal conditions (if any)
    - Available DSL actions
    - Constraints and invariants
    """
    
    # Unique identifier for this environment spec
    spec_id: str
    
    # Initial state propositions
    initial_propositions: PropositionSet
    
    # Goal propositions (may be empty for exploration tasks)
    goal_propositions: PropositionSet | None = None
    
    # Available DSL action IDs
    available_actions: frozenset[str] = field(default_factory=frozenset)
    
    # Additional constraints as free-form metadata
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSpecMemory:
    """
    Memory contour for environment specifications and probe history.
    
    Ref: Spec 3.5.1 - EnvironmentSpecMemory
    
    This memory contour stores:
    - Current EnvironmentSpecification
    - History of probe actions and their results
    
    ISO-3 Compliance: This class has NO methods to access SyntaxErrorMemory or
    EpistemicMemory contents. It is completely isolated.
    """
    
    # Current environment specification
    current_spec: EnvironmentSpecification | None = None
    
    # History of probe records (FIFO queue)
    _probe_history: deque[ProbeRecord] = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_probe(self, record: ProbeRecord) -> None:
        """
        Add a probe record to the history.
        
        Args:
            record: The probe record to add
            
        Raises:
            ValueError: If no EnvironmentSpecification is set
        """
        if self.current_spec is None:
            raise ValueError("Cannot add probe without EnvironmentSpecification")
        self._probe_history.append(record)
    
    def get_probe_history(self) -> Sequence[ProbeRecord]:
        """
        Get the probe history as an immutable sequence.
        
        Returns:
            Sequence of probe records in chronological order
        """
        return tuple(self._probe_history)
    
    def get_recent_probes(self, count: int) -> Sequence[ProbeRecord]:
        """
        Get the most recent probe records.
        
        Args:
            count: Number of recent probes to retrieve
            
        Returns:
            Sequence of up to `count` most recent probe records
        """
        history_list = list(self._probe_history)
        return tuple(history_list[-count:] if len(history_list) > count else history_list)
    
    def set_specification(self, spec: EnvironmentSpecification) -> None:
        """
        Set the current environment specification.
        
        Args:
            spec: The environment specification to set
        """
        object.__setattr__(self, 'current_spec', spec)
    
    def clear_probes(self) -> None:
        """Clear all probe history."""
        self._probe_history.clear()
    
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
    
    # Internal storage (mutable for implementation, but interface is read-only)
    _errors: list[SyntaxErrorRecord] = field(default_factory=list)
    
    def add_error(
        self,
        prompt_hash: str,
        source_hash: str,
        traceback: str,
        level_id: str = "unknown",
        static_diagnostics: list[str] | None = None,
        timestamp: int = 0,
    ) -> None:
        """
        Add a syntax error record.
        
        Args:
            prompt_hash: Hash of the prompt that caused the error
            source_hash: Hash of the source code that failed
            traceback: The traceback string
            level_id: Level where the error occurred (default: "unknown")
            static_diagnostics: Optional static analysis diagnostics
            timestamp: Error occurrence timestamp
            
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
        
        self._errors.append(record)
        
        # Enforce maximum capacity (FIFO removal)
        if len(self._errors) > self.MAX_ENTRIES:
            self._errors.pop(0)  # Remove oldest
    
    def get_errors(self) -> Sequence[SyntaxErrorRecord]:
        """
        Get all syntax error records.
        
        Returns:
            Sequence of syntax error records in chronological order
        """
        return tuple(self._errors)
    
    def get_recent_errors(self, count: int) -> Sequence[SyntaxErrorRecord]:
        """
        Get the most recent syntax error records.
        
        Args:
            count: Number of recent errors to retrieve
            
        Returns:
            Sequence of up to `count` most recent error records
        """
        return tuple(self._errors[-count:] if len(self._errors) > count else self._errors)
    
    def clear(self) -> None:
        """Clear all syntax error records."""
        self._errors.clear()
    
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
    
    # Internal storage for judgments (mutable for implementation)
    _judgments: list[BrusentsovJudgment] = field(default_factory=list)
    
    # Live OMIT branches (branches that can be safely omitted)
    _live_omit_branches: set[BranchSignature] = field(default_factory=set)
    
    # Severed NULL branch signatures (branches that caused contradictions)
    _severed_null_signatures: set[BranchSignature] = field(default_factory=set)
    
    def add_judgment(self, judgment: BrusentsovJudgment) -> None:
        """
        Add a Brusentsov judgment.
        
        Args:
            judgment: The judgment to add
        """
        self._judgments.append(judgment)
        
        # Update live/severed branches based on judgment type
        if judgment.judgment_type == "OMIT":
            self._live_omit_branches.add(judgment.branch_signature)
        elif judgment.judgment_type == "NULL":
            self._severed_null_signatures.add(judgment.branch_signature)
    
    def get_judgments(self) -> Sequence[BrusentsovJudgment]:
        """
        Get all Brusentsov judgments.
        
        Returns:
            Sequence of judgments in chronological order
        """
        return tuple(self._judgments)
    
    def get_live_omit_branches(self) -> frozenset[BranchSignature]:
        """
        Get all live OMIT branches.
        
        Returns:
            FrozenSet of branch signatures that can be omitted
        """
        return frozenset(self._live_omit_branches)
    
    def get_severed_null_signatures(self) -> frozenset[BranchSignature]:
        """
        Get all severed NULL branch signatures.
        
        Returns:
            FrozenSet of branch signatures that caused contradictions
        """
        return frozenset(self._severed_null_signatures)
    
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
    
    def prune_omit_branch(self, signature: BranchSignature) -> None:
        """
        Remove a branch from live OMIT branches.
        
        Args:
            signature: The branch signature to remove
        """
        self._live_omit_branches.discard(signature)
    
    def clear(self) -> None:
        """Clear all epistemic memory."""
        self._judgments.clear()
        self._live_omit_branches.clear()
        self._severed_null_signatures.clear()
    
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
