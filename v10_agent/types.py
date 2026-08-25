"""
V10 Agent Base Types Module

Ref: Engineering Specification V10.0, Section 3 - Core Data Types
Ref: Architectural Specification V10.0, Section 3.3 - Atomic Proposition Families

This module defines the foundational dataclasses for the agent's type system:
- PropositionSet: Collection of atomic propositions for transition judgment
- EffectDeclaration: DSL function output describing intended effects
- ProbeRecord: Explorer's probe action history entry
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


def _freeze_dict(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Convert a dict to a hashable tuple of sorted key-value pairs."""
    return tuple(sorted((k, v) for k, v in d.items()))


@dataclass(frozen=True)
class AtomicProposition:
    """
    Represents a single atomic proposition for Brusentsov logic evaluation.
    
    Ref: Spec 3.3 - Atomic proposition families (normative)
    
    Propositions are drawn only from registered families and are always
    grounded on PlanningSet identifiers.
    
    Attributes:
        family: The proposition family type (e.g., 'object_identity', 'attribute_delta')
        data: Family-specific proposition data
        objects: List of PlanningSet object IDs this proposition references
        relations: List of PlanningSet relation IDs this proposition references
    """
    #: Proposition family from registered families (Spec 3.3)
    family: str
    
    #: Family-specific proposition data (structure varies by family)
    data: Dict[str, Any] = field(default_factory=dict, hash=False, compare=True)
    
    #: PlanningSet object IDs referenced by this proposition
    objects: Tuple[str, ...] = field(default_factory=tuple)
    
    #: PlanningSet relation IDs referenced by this proposition
    relations: Tuple[str, ...] = field(default_factory=tuple)
    
    def __hash__(self) -> int:
        """Make AtomicProposition hashable for use in frozensets."""
        return hash((self.family, _freeze_dict(self.data), self.objects, self.relations))
    
    def __eq__(self, other: object) -> bool:
        """Equality check based on all fields."""
        if not isinstance(other, AtomicProposition):
            return NotImplemented
        return (
            self.family == other.family and
            self.data == other.data and
            self.objects == other.objects and
            self.relations == other.relations
        )


@dataclass(frozen=True)
class PropositionSet:
    """
    A collection of atomic propositions representing expected or observed state.
    
    Ref: Spec 3.1 - Brusentsov Ternary Logic
    Ref: Spec 5 - implies_brusentsov Engineering Realisation
    
    Used by LayeredVerifier to compare expected vs observed transitions
    using Brusentsov ternary logic (FOLLOW/NULL/OMIT).
    
    Attributes:
        snapshot_hash: Hash of the planning snapshot this set belongs to
        propositions: Immutable set of atomic propositions
        timestamp: Logical timestamp for ordering judgments
    """
    #: Hash of the associated PlanningSet snapshot
    snapshot_hash: str
    
    #: Immutable collection of atomic propositions
    propositions: frozenset = field(default_factory=frozenset)
    
    #: Logical timestamp for ordering (monotonic within level)
    timestamp: int = 0
    
    @classmethod
    def create(cls, snapshot_hash: str, propositions: Optional[List[AtomicProposition]] = None, timestamp: int = 0) -> "PropositionSet":
        """
        Factory method to create a PropositionSet from a list of propositions.
        
        Args:
            snapshot_hash: The PlanningSet snapshot identifier
            propositions: List of AtomicProposition instances (optional)
            timestamp: Logical timestamp
            
        Returns:
            New PropositionSet instance with frozen proposition set
        """
        prop_set = frozenset(propositions) if propositions else frozenset()
        return cls(
            snapshot_hash=snapshot_hash,
            propositions=prop_set,
            timestamp=timestamp
        )
    
    def __contains__(self, proposition: AtomicProposition) -> bool:
        """Check if a proposition is in this set."""
        return proposition in self.propositions
    
    def __len__(self) -> int:
        """Return the number of propositions in this set."""
        return len(self.propositions)
    
    def is_empty(self) -> bool:
        """Check if this proposition set is empty."""
        return len(self.propositions) == 0


@dataclass(frozen=True)
class EffectDeclaration:
    """
    Pure declaration of intended effects from a DSL function call.
    
    Ref: Spec 3.3 - DSL Function Manifest (Coder output)
    Ref: Spec 4.2 - Restricted executor (SandboxAPI)
    
    This is the return type of sandboxed DSL functions. It declares
    intended effects without executing them. Actual environment actions
    are performed later by ActionBoundary after Verifier approval.
    
    Attributes:
        dsl_function: Name of the DSL function that produced this declaration
        arguments: Arguments passed to the DSL function
        expected_propositions: Propositions expected to hold after execution
        metric_delta_sign: Expected sign of metric change (-1, 0, +1) if applicable
        object_ids: PlanningSet object IDs affected by this effect
    """
    #: Name of the DSL function producing this effect
    dsl_function: str
    
    #: Arguments passed to the DSL function (must use PlanningSet IDs)
    arguments: Dict[str, Any]
    
    #: Expected propositions after effect materialization
    expected_propositions: List[AtomicProposition] = field(default_factory=list)
    
    #: Expected metric delta sign (-1=decrease, 0=no-change, +1=increase)
    metric_delta_sign: Optional[int] = None
    
    #: PlanningSet object IDs affected by this effect
    object_ids: tuple = field(default_factory=tuple)
    
    #: Whether this declaration has been validated against PlanningSet
    validated: bool = False


@dataclass(frozen=True)
class ProbeRecord:
    """
    Record of an Explorer probe action for EnvironmentSpecMemory.
    
    Ref: Spec 3.2 - EnvironmentSpecification (Explorer output)
    Ref: Spec 3.5 - Memory Contours (EnvironmentSpecMemory)
    
    Stores factual evidence from Explorer's probe actions. No trajectory
    steps, no goal statements, no Python code.
    
    Attributes:
        probe_id: Unique identifier for this probe record
        action_id: The action ID tested (from PlanningSet.allowed_action_ids)
        coordinate_candidate_id: Optional coordinate candidate ID if ACTION6-style
        pre_snapshot_hash: Snapshot hash before probe execution
        post_snapshot_hash: Snapshot hash after probe execution
        observed_effects: List of observed effect summaries
        supporting_evidence_ids: IDs of evidence supporting this probe's findings
        confidence: Confidence score [0.0, 1.0] in the probe's conclusions
        contradicted: Whether this probe's findings were later contradicted
    """
    #: Unique probe identifier (e.g., "probe_017")
    probe_id: str
    
    #: Action ID from PlanningSet (e.g., "ACTION1", "ACTION6")
    action_id: str
    
    #: Optional coordinate candidate ID for coordinate-based actions
    coordinate_candidate_id: Optional[str] = None
    
    #: Snapshot hash before probe execution
    pre_snapshot_hash: str = ""
    
    #: Snapshot hash after probe execution
    post_snapshot_hash: str = ""
    
    #: Observed effect summaries (factual, no interpretation)
    observed_effects: List[str] = field(default_factory=list)
    
    #: Evidence IDs supporting this probe's findings
    supporting_evidence_ids: List[str] = field(default_factory=list)
    
    #: Confidence in probe conclusions [0.0, 1.0]
    confidence: float = 0.0
    
    #: Whether findings were contradicted by later evidence
    contradicted: bool = False
    
    #: Timestamp of probe execution
    timestamp: int = 0
    
    def __post_init__(self) -> None:
        """Validate confidence range."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ProbeRecord.confidence must be in [0.0, 1.0], got {self.confidence}")


# =============================================================================
# Additional Supporting Types for Memory Contours
# Ref: Spec 3.5 - Memory Contours
# =============================================================================

@dataclass(frozen=True)
class SyntaxErrorRecord:
    """
    Record of a DSL syntax/execution error for SyntaxErrorMemory.
    
    Ref: Spec 3.5 - SyntaxErrorMemory contour
    Ref: Spec 4 - Sandbox & Validation Pipeline
    
    Contains prompt/source hashes and traceback for Coder feedback.
    Never visible to Solver (ISO-1 invariant).
    
    Attributes:
        level_id: Level where the error occurred
        prompt_hash: Hash of the prompt sent to Coder
        source_hash: Hash of the generated Python source
        traceback: Captured exception text (if any)
        static_diagnostics: Static analysis diagnostics
        timestamp: Error occurrence timestamp
    """
    level_id: str
    prompt_hash: str
    source_hash: str
    traceback: Optional[str] = None
    static_diagnostics: List[str] = field(default_factory=list)
    timestamp: int = 0


@dataclass(frozen=True)
class BrusentsovJudgment:
    """
    Judgment record for EpistemicMemory.
    
    Ref: Spec 3.5 - EpistemicMemory contour
    Ref: Spec 3.1 - Brusentsov Ternary Logic
    
    Stores the result of implies_brusentsov evaluation for a transition.
    Only written to EpistemicMemory, never visible to Coder (ISO-2 invariant).
    
    Attributes:
        judgment_type: Type of judgment ("FOLLOW", "NULL", or "OMIT")
        branch_signature: Signature of the branch being judged
        reasoning: Human-readable explanation of the judgment
        timestamp: Judgment timestamp
    """
    judgment_type: str  # "FOLLOW", "NULL", or "OMIT"
    branch_signature: "BranchSignature"
    reasoning: str
    timestamp: float = 0.0


@dataclass(frozen=True)
class BranchSignature:
    """
    Signature for tracking live OMIT branches in EpistemicMemory.
    
    Ref: Spec 3.5 - EpistemicMemory (live_omit_branches)
    
    Identifies a branch that remains alive despite OMIT judgments,
    allowing Solver to pivot on it later.
    
    Attributes:
        action_sequence: Sequence of action IDs in the branch
        outcome_hash: Hash of the branch outcome
    """
    action_sequence: tuple
    outcome_hash: str


@dataclass(frozen=True)
class ObjectSpec:
    """
    Object specification for EnvironmentSpecification.
    
    Ref: Spec 3.2 - EnvironmentSpecification
    
    Attributes:
        type_id: Unique identifier for object type
        description: Human-readable description
        attributes: Dictionary of attribute names to values
    """
    type_id: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationSpec:
    """
    Relation specification for EnvironmentSpecification.
    
    Ref: Spec 3.2 - EnvironmentSpecification
    
    Attributes:
        type_id: Unique identifier for relation type
        description: Human-readable description
        arity: Number of objects involved in the relation
    """
    type_id: str
    description: str
    arity: int = 2


@dataclass(frozen=True)
class EnvironmentSpecification:
    """
    Complete environment specification output by Explorer.
    
    Ref: Spec 3.2 - EnvironmentSpecification (Explorer output)
    Ref: Spec 8.2 - Coder Prompt Construction
    
    Contains factual information about objects, relations, and action surface.
    NO goal information (ISO-1 invariant).
    
    Attributes:
        grid_width: Width of the game grid
        grid_height: Height of the game grid
        object_specs: List of object type specifications
        relation_specs: List of relation type specifications
        action_surface_type: Type of action surface (e.g., \"grid\", \"graph\")
        allowed_actions: List of allowed action IDs
    """
    grid_width: int
    grid_height: int
    object_specs: List[ObjectSpec] = field(default_factory=list)
    relation_specs: List[RelationSpec] = field(default_factory=list)
    action_surface_type: str = "grid"
    allowed_actions: List[str] = field(default_factory=list)
