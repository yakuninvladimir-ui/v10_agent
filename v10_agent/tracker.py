"""PersistentObjectTracker: Deterministic frame-to-frame object tracking for ARC-AGI-3.

Maintains identity persistence across frames using greedy nearest-neighbor matching
with normalized geometric/color costs, velocity EMA, occlusion detection, and shape stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from v10_agent.arga_lite import ARGALiteSnapshot, PlanningObject
from v10_agent.config import V10Config
from v10_agent.types import BoundingBox, Centroid


def mask_jaccard(
    mask_a: tuple[tuple[int, ...], ...] | list[list[int]],
    mask_b: tuple[tuple[int, ...], ...] | list[list[int]],
) -> float:
    """Compute exact pixel Intersection-over-Union (Jaccard) between two binary masks aligned at top-left."""
    if not mask_a and not mask_b:
        return 1.0
    if not mask_a or not mask_b:
        return 0.0

    ha, wa = len(mask_a), len(mask_a[0])
    hb, wb = len(mask_b), len(mask_b[0])

    max_h = max(ha, hb)
    max_w = max(wa, wb)

    intersection = 0
    union = 0

    for r in range(max_h):
        for c in range(max_w):
            v_a = mask_a[r][c] if r < ha and c < wa else 0
            v_b = mask_b[r][c] if r < hb and c < wb else 0
            if v_a and v_b:
                intersection += 1
            if v_a or v_b:
                union += 1

    if union == 0:
        return 1.0
    return intersection / union


def jaccard_similarity(
    sig_a: str,
    sig_b: str,
    mask_a: tuple[tuple[int, ...], ...] | None = None,
    mask_b: tuple[tuple[int, ...], ...] | None = None,
) -> float:
    """Compute shape Jaccard similarity using exact signatures or binary masks."""
    if sig_a and sig_b and sig_a == sig_b:
        return 1.0
    if mask_a is not None and mask_b is not None and len(mask_a) > 0 and len(mask_b) > 0:
        return mask_jaccard(mask_a, mask_b)
    # Fallback to signature dimension parsing if masks unavailable
    try:
        if sig_a.startswith("shape_") and sig_b.startswith("shape_"):
            dim_a = sig_a.split("_")[1].split("x")
            dim_b = sig_b.split("_")[1].split("x")
            wa, ha = int(dim_a[0]), int(dim_a[1])
            wb, hb = int(dim_b[0]), int(dim_b[1])
            inter_area = max(0, min(wa, wb)) * max(0, min(ha, hb))
            union_area = (wa * ha) + (wb * hb) - inter_area
            if union_area > 0:
                return inter_area / union_area
    except Exception:
        pass
    return 0.0


@dataclass
class TrackedObject:
    """Deterministic persistent object track representation."""
    persistent_id: str
    last_frame_id: int
    color: int
    shape_signature: str
    filled_shape_signature: str
    centroid: Centroid
    bbox: BoundingBox
    area: int
    velocity: tuple[float, float]  # EMA, alpha = 0.3
    confidence: float  # 1.0 - cost (1.0 on initial spawn)
    history: list[tuple[int, Centroid]]  # max length = cumulative_window + 1
    shape_signature_stable: str
    shape_stability_score: float = 1.0
    occluded: bool = False
    occluded_frames: int = 0
    occluded_by: str | None = None
    cumulative_delta: tuple[float, float] = (0.0, 0.0)
    mask: tuple[tuple[int, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "last_frame_id": self.last_frame_id,
            "color": self.color,
            "shape_signature": self.shape_signature,
            "centroid": self.centroid.to_dict(),
            "bbox": self.bbox.to_dict(),
            "area": self.area,
            "velocity": self.velocity,
            "confidence": self.confidence,
            "shape_stability_score": self.shape_stability_score,
            "occluded": self.occluded,
            "occluded_frames": self.occluded_frames,
            "occluded_by": self.occluded_by,
            "cumulative_delta": self.cumulative_delta,
        }


class PersistentObjectTracker:
    """Tracks detected objects across frames using cost-optimal bipartite matching."""

    def __init__(self, config: V10Config | None = None) -> None:
        self.config = config or V10Config()
        self.tracks: dict[str, TrackedObject] = {}
        self.next_track_num: int = 1
        self.frame_index: int = 0
        self.last_ambiguity_score: float | None = None
        self.last_ambiguous_track: str | None = None

    def reset(self) -> None:
        """Full reset called on every environmental RESET."""
        self.tracks.clear()
        self.next_track_num = 1
        self.frame_index = 0
        self.last_ambiguity_score = None
        self.last_ambiguous_track = None

    def get_tracked(self) -> list[TrackedObject]:
        """Return all currently active or occluded tracked objects sorted by persistent_id."""
        return sorted(self.tracks.values(), key=lambda t: t.persistent_id)

    def get_track(self, persistent_id: str) -> TrackedObject | None:
        """Retrieve a tracked object by persistent ID."""
        return self.tracks.get(persistent_id)

    def _compute_cost(
        self,
        track: TrackedObject,
        comp: PlanningObject,
        grid_dim: int,
    ) -> float:
        """Compute normalized matching cost in [0, 1]."""
        color_diff = 0.0 if track.color == comp.color else 1.0
        manhattan = abs(track.centroid.row - comp.centroid.row) + abs(track.centroid.col - comp.centroid.col)
        centroid_dist = manhattan / max(grid_dim, 1)

        shape_jaccard = jaccard_similarity(
            track.shape_signature_stable,
            comp.shape_signature,
            track.mask,
            comp.mask,
        )
        area_rel_diff = abs(track.area - comp.area) / max(track.area, comp.area, 1)

        cost = (
            0.30 * color_diff
            + 0.40 * centroid_dist
            + 0.20 * (1.0 - shape_jaccard)
            + 0.10 * area_rel_diff
        )
        return min(1.0, max(0.0, cost))

    def update(
        self,
        snapshot: ARGALiteSnapshot | Sequence[PlanningObject],
        frame_index: int,
    ) -> list[TrackedObject]:
        """Update tracker state with a new ARGALite perception snapshot or list of objects."""
        self.frame_index = frame_index
        min_area = getattr(self.config, "track_min_area", 1)
        if hasattr(snapshot, "objects"):
            grid_h, grid_w = snapshot.grid_dims if snapshot.grid_dims else (30, 30)
            bg_color = getattr(snapshot, "background_color", 0)
            components: list[PlanningObject] = [
                obj for obj in snapshot.objects if obj.area >= min_area and obj.color != bg_color
            ]
        else:
            grid_h, grid_w = (30, 30)
            components = [obj for obj in snapshot if getattr(obj, "area", 1) >= min_area]

        grid_dim = max(grid_h, grid_w, 1)

        match_threshold = getattr(self.config, "track_match_threshold", 0.45)
        track_max_age = getattr(self.config, "track_max_age", 5)
        occlusion_radius = getattr(self.config, "occlusion_radius", 3)
        cum_window = getattr(self.config, "cumulative_window", 3)

        active_track_ids = list(self.tracks.keys())

        # 1. Build full cost matrix
        cost_entries: list[tuple[float, str, int]] = []
        for trk_id in active_track_ids:
            track = self.tracks[trk_id]
            for c_idx, comp in enumerate(components):
                cost = self._compute_cost(track, comp, grid_dim)
                cost_entries.append((cost, trk_id, c_idx))

        # 2. Sort pairs by ascending cost; deterministic tie-breaking on persistent_id then c_idx
        cost_entries.sort(key=lambda x: (x[0], x[1], x[2]))

        # Check matching ambiguity (difference between best and second-best candidate < matching_ambiguity_threshold)
        ambiguity_scores: list[tuple[float, str]] = []
        for trk_id in active_track_ids:
            trk_costs = sorted([cost for cost, t_id, _ in cost_entries if t_id == trk_id and cost < match_threshold])
            if len(trk_costs) >= 2:
                diff = trk_costs[1] - trk_costs[0]
                ambiguity_scores.append((round(diff, 4), trk_id))

        if ambiguity_scores:
            ambiguity_scores.sort(key=lambda x: x[0])
            self.last_ambiguity_score = ambiguity_scores[0][0]
            self.last_ambiguous_track = ambiguity_scores[0][1]
        else:
            self.last_ambiguity_score = None
            self.last_ambiguous_track = None

        matched_tracks: set[str] = set()
        matched_comps: set[int] = set()
        accepted_matches: list[tuple[str, int, float]] = []

        # 3. Greedy assignment
        for cost, trk_id, c_idx in cost_entries:
            if trk_id in matched_tracks or c_idx in matched_comps:
                continue
            if cost < match_threshold:
                matched_tracks.add(trk_id)
                matched_comps.add(c_idx)
                accepted_matches.append((trk_id, c_idx, cost))

        # 4. Update matched tracks
        for trk_id, c_idx, cost in accepted_matches:
            track = self.tracks[trk_id]
            comp = components[c_idx]

            inst_v = (
                comp.centroid.row - track.centroid.row,
                comp.centroid.col - track.centroid.col,
            )
            # Velocity EMA with fixed alpha = 0.3
            new_vx = 0.3 * inst_v[0] + 0.7 * track.velocity[0]
            new_vy = 0.3 * inst_v[1] + 0.7 * track.velocity[1]

            track.last_frame_id = frame_index
            track.velocity = (round(new_vx, 3), round(new_vy, 3))
            track.confidence = round(1.0 - cost, 4)

            # Shape stability update: decay score if Jaccard < 0.85
            jaccard = jaccard_similarity(comp.shape_signature, track.shape_signature_stable, comp.mask, track.mask)
            if jaccard < 0.85:
                track.shape_stability_score = round(track.shape_stability_score * 0.9, 4)

            track.color = comp.color
            track.centroid = comp.centroid
            track.bbox = comp.bbox
            track.area = comp.area
            track.shape_signature = comp.shape_signature
            track.filled_shape_signature = comp.filled_shape_signature
            track.mask = comp.mask
            comp.persistent_id = trk_id
            comp.track_confidence = track.confidence

            # History ring buffer
            track.history.append((frame_index, comp.centroid))
            max_hist_len = cum_window + 1
            if len(track.history) > max_hist_len:
                track.history.pop(0)

            # Cumulative motion
            if len(track.history) >= cum_window:
                oldest_centroid = track.history[-cum_window][1]
                track.cumulative_delta = (
                    round(comp.centroid.row - oldest_centroid.row, 2),
                    round(comp.centroid.col - oldest_centroid.col, 2),
                )

            # Occlusion cleared
            if track.occluded:
                track.occluded = False
                track.occluded_frames = 0
                track.occluded_by = None

        # 5. Handle unmatched tracks (occlusion heuristic or aging out)
        unmatched_track_ids = [t for t in active_track_ids if t not in matched_tracks]
        for trk_id in unmatched_track_ids:
            track = self.tracks[trk_id]

            # Check occlusion heuristic
            occluding_comp: PlanningObject | None = None
            min_occl_dist = float("inf")
            for comp in components:
                if comp.area > track.area:
                    manhattan = abs(track.centroid.row - comp.centroid.row) + abs(track.centroid.col - comp.centroid.col)
                    if manhattan <= occlusion_radius and manhattan < min_occl_dist:
                        min_occl_dist = manhattan
                        occluding_comp = comp

            if occluding_comp is not None:
                track.occluded = True
                track.occluded_frames += 1
                # Find persistent_id of occluder if already assigned
                occluder_id = occluding_comp.id
                for m_id, m_c_idx, _ in accepted_matches:
                    if components[m_c_idx] is occluding_comp:
                        occluder_id = m_id
                        break
                track.occluded_by = occluder_id
            else:
                track.occluded = False
                track.occluded_by = None

            # Lifecycle check: kill if unseen for track_max_age frames
            frames_unseen = frame_index - track.last_frame_id
            if frames_unseen >= track_max_age and (not track.occluded or track.occluded_frames >= track_max_age * 2):
                del self.tracks[trk_id]

        # 6. Handle unmatched components (spawn new tracks)
        for c_idx, comp in enumerate(components):
            if c_idx not in matched_comps:
                trk_id = f"trk_{self.next_track_num:04d}"
                self.next_track_num += 1

                new_track = TrackedObject(
                    persistent_id=trk_id,
                    last_frame_id=frame_index,
                    color=comp.color,
                    shape_signature=comp.shape_signature,
                    filled_shape_signature=comp.filled_shape_signature,
                    centroid=comp.centroid,
                    bbox=comp.bbox,
                    area=comp.area,
                    velocity=(0.0, 0.0),
                    confidence=1.0,
                    history=[(frame_index, comp.centroid)],
                    shape_signature_stable=comp.shape_signature,
                    shape_stability_score=1.0,
                    occluded=False,
                    occluded_frames=0,
                    occluded_by=None,
                    cumulative_delta=(0.0, 0.0),
                    mask=comp.mask,
                )
                self.tracks[trk_id] = new_track
                comp.persistent_id = trk_id
                comp.track_confidence = 1.0

        return self.get_tracked()
