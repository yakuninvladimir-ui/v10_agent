"""ARC-AGI-3 LCLD Agent Version 10.0.

Core package implementing Tri-Agent hierarchy, isolated memory contours,
Brusentsov ternary logic, and deterministic ARGALite perception.
"""

from __future__ import annotations

from v10_agent.config import V10Config, config_from_env, config_from_mapping
from v10_agent.types import (
    ActionDeclaration,
    AtomicProposition,
    BoundingBox,
    Centroid,
    EffectDeclaration,
    Grid2D,
    PlanningAlias,
    PlanningObjectId,
    PropositionFamily,
    PropositionSet,
)
from v10_agent.observe import collapse_frame_axes, normalize_observation
from v10_agent.action_adapter import to_native_action, from_native_action, arcade_step_args
from v10_agent.arga_lite import ARGALiteSnapshot, extract_arga_snapshot, PlanningObject, SpatialRelation
from v10_agent.planning_set import PlanningSet, build_planning_set
from v10_agent.frame_media import render_grid_png, render_annotated_frame_png, render_dual_frame_png
from v10_agent.verifier_packet import build_verifier_packet

from v10_agent.brusentsov_logic import (
    BrusentsovJudgment,
    EpistemicSignal,
    Ternary,
    Verdict,
    contradicts,
    implies_brusentsov,
    is_necessarily_contained,
)
from v10_agent.memory_contours import (
    BranchSignature,
    EnvironmentSpecMemory,
    EpistemicMemory,
    GameMemory,
    IsolationViolationError,
    MemoryContourManager,
    ProbeRecord,
    StructuredInvariant,
    SyntaxErrorMemory,
    SyntaxErrorRecord,
)
from v10_agent.tracker import (
    PersistentObjectTracker,
    TrackedObject,
)
from v10_agent.sandbox import (
    SandboxAPI,
    SandboxExecutor,
    SandboxedModule,
    validate_dsl_source,
)

from v10_agent.llm_advisor import (
    BaseLLMAdvisor,
    MockLLMAdvisor,
    VLLMAdvisor,
    build_llm_advisor,
)
from v10_agent.prompt_builders import (
    build_coder_prompts,
    build_explorer_prompts,
    build_solver_prompts,
)
from v10_agent.explorer_agent import ExplorerAgent
from v10_agent.dsl_coder import DSLCoder
from v10_agent.solver_agent import SolverAgent

from v10_agent.verification import GroundedStep, GroundingError, VerificationBinder
from v10_agent.judge import LayeredVerifier
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool
from v10_agent.fallback_symbolic import SymbolicFallbackEngine
from v10_agent.logging import AuditRecord, StructuredAuditLogger
from v10_agent.session import GameSession

__version__ = "10.0.0"

__all__ = [
    "__version__",
    "V10Config",
    "config_from_env",
    "config_from_mapping",
    "Grid2D",
    "PlanningObjectId",
    "PlanningAlias",
    "BoundingBox",
    "Centroid",
    "ActionDeclaration",
    "EffectDeclaration",
    "PropositionFamily",
    "AtomicProposition",
    "PropositionSet",
    "collapse_frame_axes",
    "normalize_observation",
    "to_native_action",
    "from_native_action",
    "arcade_step_args",
    "ARGALiteSnapshot",
    "extract_arga_snapshot",
    "PlanningObject",
    "SpatialRelation",
    "PlanningSet",
    "build_planning_set",
    "render_grid_png",
    "render_annotated_frame_png",
    "render_dual_frame_png",
    "build_verifier_packet",
    "Ternary",
    "Verdict",
    "BrusentsovJudgment",
    "implies_brusentsov",
    "contradicts",
    "is_necessarily_contained",
    "EnvironmentSpecMemory",
    "SyntaxErrorMemory",
    "EpistemicMemory",
    "EpistemicSignal",
    "GameMemory",
    "MemoryContourManager",
    "IsolationViolationError",
    "ProbeRecord",
    "StructuredInvariant",
    "SyntaxErrorRecord",
    "BranchSignature",
    "PersistentObjectTracker",
    "TrackedObject",
    "SandboxAPI",
    "SandboxExecutor",
    "SandboxedModule",
    "validate_dsl_source",
    "BaseLLMAdvisor",
    "VLLMAdvisor",
    "MockLLMAdvisor",
    "build_llm_advisor",
    "build_explorer_prompts",
    "build_coder_prompts",
    "build_solver_prompts",
    "ExplorerAgent",
    "DSLCoder",
    "SolverAgent",
    "VerificationBinder",
    "GroundedStep",
    "GroundingError",
    "LayeredVerifier",
    "CandidateTrajectory",
    "TrajectoryPool",
    "SymbolicFallbackEngine",
    "AuditRecord",
    "StructuredAuditLogger",
    "GameSession",
]
