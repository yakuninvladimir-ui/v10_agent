"""
V10 Agent Package - ARC-AGI-3 LCLD Agent Version 10.0
Ref: Engineering Specification V10.0 Section 1 (Repository Layout)
"""

from .config import V10Config
from .types import (
    AtomicProposition,
    PropositionSet,
    EffectDeclaration,
    ProbeRecord,
    SyntaxErrorRecord,
    BrusentsovJudgment,
    BranchSignature,
    ObjectSpec,
    RelationSpec,
    EnvironmentSpecification,
)
from .brusentsov_logic import Ternary, implies_brusentsov, register_proposition_family
from .planning_set import PlanningSet
from .memory_contours import (
    EnvironmentSpecMemory,
    SyntaxErrorMemory,
    EpistemicMemory,
    GameMemory,
)

__version__ = "10.0.0"
__all__ = [
    "V10Config",
    "AtomicProposition",
    "PropositionSet",
    "EffectDeclaration",
    "ProbeRecord",
    "SyntaxErrorRecord",
    "BrusentsovJudgment",
    "BranchSignature",
    "ObjectSpec",
    "RelationSpec",
    "EnvironmentSpecification",
    "Ternary",
    "implies_brusentsov",
    "register_proposition_family",
    "PlanningSet",
    "EnvironmentSpecMemory",
    "SyntaxErrorMemory",
    "EpistemicMemory",
    "GameMemory",
]
