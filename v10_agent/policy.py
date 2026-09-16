"""Action and candidate selection policy for GameSession."""

from __future__ import annotations

from typing import Any

from v10_agent.memory_contours import EpistemicMemory
from v10_agent.planning_set import PlanningSet
from v10_agent.trajectory import CandidateTrajectory, TrajectoryPool


class ActionSelectionPolicy:
    """Selects the next trajectory step prioritizing live OMIT branches, then fresh candidates, then fallback."""

    def select_next_step(
        self,
        pool: TrajectoryPool | None,
        epistemic_memory: EpistemicMemory,
        planning_set: PlanningSet,
    ) -> tuple[dict[str, Any] | None, str]:
        """Select next step dict and strategy label.

        Returns (step_dict, strategy) or (None, 'fallback').
        """
        if pool is not None:
            active_cand = pool.active_candidate()
            if active_cand is not None:
                step = active_cand.current_step()
                if step is not None:
                    return (step, "solver_candidate")

        if epistemic_memory.live_omit_branches:
            for branch in epistemic_memory.live_omit_branches:
                if epistemic_memory.is_severed(branch.signature_id):
                    continue
                if pool is not None:
                    for cand in pool.candidates:
                        if cand.trajectory_id == branch.trajectory_id and cand.active:
                            step = cand.current_step()
                            if step is not None:
                                return (step, "branch_dispatch")

        return (None, "fallback")
