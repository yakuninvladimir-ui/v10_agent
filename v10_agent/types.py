"""Domain data types and data contracts for ARC-AGI-3 LCLD Agent V10.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

Grid2D = list[list[int]]
PlanningObjectId = str
PlanningAlias = str


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box defined by inclusive row and column coordinates."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def to_dict(self) -> dict[str, int]:
        return {
            "min_row": self.min_row,
            "min_col": self.min_col,
            "max_row": self.max_row,
            "max_col": self.max_col,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class Centroid:
    """Centroid of an object or region in row/column coordinates."""
    row: float
    col: float

    def to_dict(self) -> dict[str, float]:
        return {"row": round(self.row, 2), "col": round(self.col, 2)}


@dataclass(frozen=True)
class CoordinateCandidate:
    """A spatial point candidate grounded on the PlanningSet."""
    candidate_id: str
    x: int  # column index (0-indexed)
    y: int  # row index (0-indexed)
    source_type: str  # e.g., "centroid", "corner_tl", "corner_br", "grid_center"
    object_id: str | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "x": self.x,
            "y": self.y,
            "source_type": self.source_type,
            "object_id": self.object_id,
            "label": self.label,
        }


@dataclass
class ActionDeclaration:
    """Declaration of an intended environment action."""
    action_id: str  # "ACTION1" .. "ACTION7", "RESET"
    data: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.action_id

    def __getitem__(self, item: str) -> Any:
        if item in ("id", "action_id"):
            return self.action_id
        if item == "data":
            return self.data
        if item == "reasoning":
            return self.reasoning
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.action_id,
            "action_id": self.action_id,
            "data": dict(self.data),
            "reasoning": dict(self.reasoning),
        }


@dataclass
class EffectDeclaration:
    """Declaration of an expected transition effect produced by a DSL function."""
    declared_action: ActionDeclaration
    expected_metric_deltas: dict[str, int | float] = field(default_factory=dict)
    target_object_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_action": self.declared_action.to_dict(),
            "expected_metric_deltas": dict(self.expected_metric_deltas),
            "target_object_ids": list(self.target_object_ids),
            "confidence": self.confidence,
        }

    @classmethod
    def circuit_break(cls, reason: str) -> "EffectDeclaration":
        """Build an inert declaration that aborts a step without side effects.

        Used when an in-flight DSL module is invalidated: no action may be
        emitted on the strength of a module that no longer exists, so the
        declaration carries a reset instead of a step effect.
        """
        return cls(
            declared_action=ActionDeclaration(
                action_id="RESET",
                reasoning={"source": "circuit_break", "reason": reason},
            ),
            expected_metric_deltas={},
            target_object_ids=[],
            confidence=0.0,
        )


REGISTERED_PROPOSITION_FAMILIES = frozenset({
    "object_identity",
    "attribute_delta",
    "metric_sign",
    "relation_existence",
    "action_surface",
    "terminal_metadata",
    "affordance_flag",
    "spatial_position",
    "area_conservation",
    "cumulative_motion",
    "shape_stability",
})

PropositionFamily = str


@dataclass(frozen=True)
class AtomicProposition:
    """An atomic proposition grounded on PlanningSet identifiers.
    
    Raw grid pixels are never propositions. Only registered families are valid.
    """
    family: PropositionFamily
    subject_id: str
    predicate: str
    value: Any = None
    secondary_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, (dict, list, set)):
            object.__setattr__(self, 'value', str(self.value))
        if self.family not in REGISTERED_PROPOSITION_FAMILIES:
            raise ValueError(
                f"Unknown proposition family {self.family!r}. Must be one of {REGISTERED_PROPOSITION_FAMILIES}"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.secondary_id is not None:
            out["secondary_id"] = self.secondary_id
        return out


@dataclass(frozen=True)
class PropositionSet:
    """An immutable set of AtomicPropositions."""
    propositions: frozenset[AtomicProposition] = field(default_factory=frozenset)

    @classmethod
    def empty(cls) -> PropositionSet:
        return cls(frozenset())

    @classmethod
    def from_iterable(cls, items: Iterable[AtomicProposition]) -> PropositionSet:
        return cls(frozenset(items))

    def union(self, other: PropositionSet) -> PropositionSet:
        return PropositionSet(self.propositions | other.propositions)

    def intersection(self, other: PropositionSet) -> PropositionSet:
        return PropositionSet(self.propositions & other.propositions)

    def difference(self, other: PropositionSet) -> PropositionSet:
        return PropositionSet(self.propositions - other.propositions)

    def filter_family(self, family: str) -> PropositionSet:
        return PropositionSet(frozenset(p for p in self.propositions if p.family == family))

    def __contains__(self, item: AtomicProposition) -> bool:
        return item in self.propositions

    def __len__(self) -> int:
        return len(self.propositions)

    def __iter__(self) -> Iterable[AtomicProposition]:
        return iter(self.propositions)

    def to_list(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in sorted(self.propositions, key=lambda p: (p.family, p.subject_id, p.predicate))]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v10_agent.planning_set import PlanningObject, SpatialRelation
    from v10_agent.memory_contours import StructuredInvariant, SyntaxErrorRecord

@dataclass(frozen=True)
class SolverInput:
    """DTO for Solver - decoupled from syntax errors or code generation mechanics."""
    planning_objects: list['PlanningObject']
    relations: list['SpatialRelation']
    confirmed_invariants: list['StructuredInvariant']
    past_failures: list[dict[str, Any]] # Epistemic structured failures
    game_model_summary: str
    action_budget: int

    @classmethod
    def from_session(
        cls,
        planning_set: Any,
        game_memory: Any = None,
        epistemic_memory: Any = None,
        action_budget: int = 100,
    ) -> 'SolverInput':
        objs = list(planning_set.objects) if hasattr(planning_set, "objects") else []
        rels = list(planning_set.relations) if hasattr(planning_set, "relations") else []
        invs = list(game_memory.structured_invariants) if hasattr(game_memory, "structured_invariants") else []
        failures = epistemic_memory.format_structured_failures() if hasattr(epistemic_memory, "format_structured_failures") else []
        summary = game_memory.format_game_model_summary() if hasattr(game_memory, "format_game_model_summary") else ""
        return cls(
            planning_objects=objs,
            relations=rels,
            confirmed_invariants=invs,
            past_failures=failures,
            game_model_summary=summary,
            action_budget=action_budget,
        )

@dataclass(frozen=True)
class CoderInput:
    """DTO for Coder - decoupled from epistemic goals and focused on mechanics/syntax."""
    env_spec: dict[str, Any]
    syntax_errors: list['SyntaxErrorRecord']
    available_actions: list[str]

    @classmethod
    def from_session(
        cls,
        env_spec: dict[str, Any],
        syntax_memory: Any = None,
        available_actions: list[str] | None = None,
    ) -> 'CoderInput':
        errors = list(syntax_memory.entries) if hasattr(syntax_memory, "entries") else []
        return cls(
            env_spec=dict(env_spec),
            syntax_errors=errors,
            available_actions=list(available_actions or []),
        )
