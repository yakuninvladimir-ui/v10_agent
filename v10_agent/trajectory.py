"""Trajectory state management and single-step execution enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateTrajectory:
    """A multi-step candidate trajectory proposed by the Solver."""
    trajectory_id: str
    steps: list[dict[str, Any]]
    success_condition: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    cursor: int = 0
    active: bool = True

    def current_step(self) -> dict[str, Any] | None:
        """Return the current step at the cursor, or None if trajectory is exhausted."""
        if not self.active or self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def advance(self) -> None:
        """Advance cursor to next step upon successful FOLLOW verdict."""
        self.cursor += 1
        if self.cursor >= len(self.steps):
            self.active = False

    def sever(self) -> None:
        """Permanently sever this trajectory upon NULL verdict."""
        self.active = False

    def is_finished(self) -> bool:
        return self.cursor >= len(self.steps) or not self.active

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "cursor": self.cursor,
            "total_steps": len(self.steps),
            "active": self.active,
            "confidence": self.confidence,
        }


@dataclass
class TrajectoryPool:
    """Pool of candidate trajectories for the current level."""
    proposal_id: str
    candidates: list[CandidateTrajectory] = field(default_factory=list)
    active_candidate_index: int = 0

    @classmethod
    def from_package(cls, package: dict[str, Any]) -> TrajectoryPool:
        prop_id = package.get("proposal_id", "prop_0")
        cands: list[CandidateTrajectory] = []
        for c in package.get("candidates", []):
            t_id = c.get("trajectory_id", f"traj_{len(cands)}")
            steps = list(c.get("steps", []))
            succ = dict(c.get("success_condition", {}))
            conf = float(c.get("confidence", 1.0))
            cands.append(
                CandidateTrajectory(
                    trajectory_id=t_id,
                    steps=steps,
                    success_condition=succ,
                    confidence=conf,
                )
            )
        return cls(proposal_id=prop_id, candidates=cands)

    def active_candidate(self) -> CandidateTrajectory | None:
        while self.active_candidate_index < len(self.candidates):
            cand = self.candidates[self.active_candidate_index]
            if cand.active and not cand.is_finished():
                return cand
            self.active_candidate_index += 1
        return None
