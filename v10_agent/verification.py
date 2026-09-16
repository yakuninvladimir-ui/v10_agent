"""VerificationBinder: Grounds DSL arguments strictly to the active PlanningSet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v10_agent.planning_set import PlanningSet
from v10_agent.types import AtomicProposition, PropositionSet


class GroundingError(ValueError):
    """Raised when a DSL function argument cannot be grounded in the active PlanningSet."""


@dataclass(frozen=True)
class GroundedStep:
    """A verified, grounded trajectory step ready for sandboxed execution."""
    step_id: str
    dsl_function: str
    arguments: dict[str, Any]
    expected_propositions: PropositionSet
    abort_if: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "dsl_function": self.dsl_function,
            "arguments": dict(self.arguments),
            "expected_propositions": self.expected_propositions.to_list(),
            "abort_if": list(self.abort_if),
        }


class VerificationBinder:
    """Enforces Invariant I5 / ISO-5: all trajectory arguments ground in PlanningSet."""

    def ground_step(self, step_dict: dict[str, Any], planning_set: PlanningSet) -> GroundedStep:
        """Ground step dictionary against the active PlanningSet."""
        step_id = str(step_dict.get("step_id", "s0"))
        fn_name = str(step_dict.get("dsl_function", ""))
        raw_args = dict(step_dict.get("arguments", {}) or {})
        abort_if = tuple(step_dict.get("abort_if", ()))

        grounded_args: dict[str, Any] = {}
        for k, v in raw_args.items():
            if isinstance(v, str):
                # Check if it refers to an object or alias
                if v in planning_set.object_ids:
                    grounded_args[k] = v
                elif v in planning_set.object_alias_to_real:
                    grounded_args[k] = planning_set.object_alias_to_real[v]
                elif v in planning_set.allowed_coordinate_candidate_ids:
                    grounded_args[k] = v
                else:
                    # Check for disallowed component-graph IDs (Invariant I8)
                    if v.startswith("comp_"):
                        raise GroundingError(
                            f"Invariant I8 Violation: Component-graph id {v!r} is not a valid trajectory target"
                        )
                    grounded_args[k] = v
            else:
                grounded_args[k] = v

        # Parse expected atomic propositions
        raw_props = step_dict.get("expected_propositions", [])
        parsed_props: list[AtomicProposition] = []
        for p in raw_props:
            if isinstance(p, dict):
                fam = p.get("family", "metric_sign")
                subj = p.get("subject_id", "")
                resolved_subj = planning_set.resolve_object_id(subj) or subj
                pred = p.get("predicate", p.get("metric", "delta"))
                val = p.get("value", p.get("sign", None))
                sec = p.get("secondary_id")
                if sec:
                    sec = planning_set.resolve_object_id(sec) or sec
                try:
                    parsed_props.append(
                        AtomicProposition(
                            family=fam,
                            subject_id=resolved_subj,
                            predicate=pred,
                            value=val,
                            secondary_id=sec,
                        )
                    )
                except ValueError:
                    pass

        return GroundedStep(
            step_id=step_id,
            dsl_function=fn_name,
            arguments=grounded_args,
            expected_propositions=PropositionSet.from_iterable(parsed_props),
            abort_if=abort_if,
        )
