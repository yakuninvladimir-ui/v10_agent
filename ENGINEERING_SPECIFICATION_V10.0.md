# ARC-AGI-3 LCLD Agent
# Engineering Specification — Version 10.1-r1
# (Tri-Agent Implementation Contracts, Independent Symbolic Execution, Multi-Candidate Reset Traversal, 4-Valued Brusentsov Logic, Persistent Object Grounding & Safe Evidence Probing)

---

## 0. Engineering Objective

This document provides the complete, authoritative implementation specification for the **ARC-AGI-3 LCLD Agent (Version 10.1-r1)**. An engineer reading this document should be able to implement or maintain the complete codebase from scratch without ambiguity, external documentation, or prior version references.

The system implements:
1. **Deterministic Perception**: ARGA-Lite object extraction, topological cavity detection, substantive spatial relation filtering, and freedom of motion bounds.
2. **Tri-Agent Role Hierarchy**: Independent LLM call families for Explorer, Coder, and Solver with complete, non-overlapping authority, enforced by strict ISO-1 through ISO-10 isolation invariants (including ISO-2 curriculum quarantine shielding Coder from puzzle goals).
3. **Strict Stratified Memory Contours**: `EnvironmentSpecMemory`, `SyntaxErrorMemory`, `EpistemicMemory`, and cross-level `GameMemory` with Tier 1 kinematics protection and Brusentsov cross-level invariant re-evaluation.
4. **Independent Symbolic Trajectory Executor**: Pre-execution signature verification, sandbox isolation, dynamic argument filtering, and circuit-breaking.
5. **Multi-Trajectory Candidate Pool Traversal & 5-Attempt Ceiling**: The Solver is invoked for full multi-step trajectory packages (up to 4 candidates per package) with a hard limit of 5 planning attempts per level (`max_chain_attempts_per_level = 5`, ~10–20 candidate trajectories total). Under **no circumstances** is the Solver called per step; execution is handled deterministically step-by-step by the `SymbolicTrajectoryExecutor` from clean initial states ($S_0$) via environmental `RESET`.
6. **Unified Hybrid Environment Support**: Seamless classification and execution of hybrid environments (`active_pipeline = "hybrid"`) where discrete buttons (`ACTION1..5`) and coordinate clicks (`ACTION6`) coexist, scheduling probing without mutual lockout.
7. **Domain-General Memory-Based Invariant Deduction**: Zero game-specific prompt bindings; invariants are deduced by the LLM from empirical memory, object affordances, and the short-term trial scratchpad.
8. **Brusentsov 4-Valued Logic & Strict 8-Tier Verification**: Mathematical realization of 4-valued judgments (`FOLLOW`, `OMIT`, `NULL`, `UNDECIDED`) in `brusentsov_logic.py`, `EpistemicSignal` segregation (ISO-9), and strict 8-tier cascade in `LayeredVerifier` with ISO-10 expectation non-severance.
9. **Robust State Orchestration & Safe Fallback**: No dirty-board cascading, clean-state resets on contradiction, single-reset `GAME_OVER` invariant, and Virtual Sandbox candidate repair without index 0 hijacking.
10. **Offline Packaging**: Self-extracting LZMA base64 packaging complying with Kaggle limits (< 985 KB).
11. **Persistent Object Tracking & Identity Grounding**: Multi-frame object permanence in `tracker.py` using centroid distance, IoU matching, track confidence decay, and ambiguity scoring wired into `PlanningSet` and `LayeredVerifier`.
12. **Safe Evidence-Seeking Loop & Strict NOOP Elimination**: Epistemic probe queue in `session.py` (Section 3.8 of `act()`) emitting real actions (`ACTION1..7`), bounded by `max_evidence_probes_per_level = 2` and streak limit fallback to `NULL`. Strictly zero `NOOP` emission.
13. **MTP=3 Speculative Decoding & Safe Boot Fallback**: Speculative execution ($k=3$) for Qwen 3.8 27B via vLLM with automated non-speculative restart on server boot crashes.

---

## 1. Module Architecture & File Layout

Active package root: `v10_agent/` (28 core source files + 4 prompt builder files)

```
v10_agent/
├── __init__.py                                 # Package exports and version metadata
├── action_adapter.py                           # Action adapter and normalization
├── arga_lite.py                                # Deterministic ARGA-Lite perception & relation extraction
├── brusentsov_logic.py                         # 4-valued Verdict, EpistemicSignal, and implies_brusentsov operator
├── config.py                                   # V10Config dataclass with environment overrides & MTP CLI builder
├── dsl_coder.py                                # Coder Agent (Call 2) & SandboxedModule compiler
├── explorer_agent.py                           # Explorer Agent (Call 1) & PrimitiveProbeManager
├── fallback_symbolic.py                        # Deterministic symbolic fallback engine
├── frame_media.py                              # Visual rendering: dual-frame raw and annotated PNG
├── game_adapter.py                             # Competition environment and game interface adapter
├── judge.py                                    # LayeredVerifier with strict 8-tier decision cascade & ISO-10
├── llm_advisor.py                              # LLM client supporting vLLM, OpenAI, DashScope, and Ollama
├── logging.py                                  # Structured JSON audit and session logger
├── memory_contours.py                          # Stratified 3-tier memory stores, EpistemicMemory & MemoryContourManager
├── observe.py                                  # Observation normalization and border cropping
├── planning_set.py                             # PlanningSet, PlanningObject, SpatialRelation, CoordinateCandidate
├── policy.py                                   # Action selection policy and fallback arbitration
├── sandbox.py                                  # Restricted AST checker and SandboxExecutor
├── session.py                                  # GameSession top-level orchestrator & safe evidence-seeking loop
├── solver_agent.py                             # Solver Agent (Call 3), XML parser & reflection engine
├── symbolic_executor.py                        # Independent SymbolicTrajectoryExecutor controller & UNDECIDED handling
├── tracker.py                                  # PersistentObjectTracker & TrackedObject multi-frame identity
├── trajectory.py                               # CandidateTrajectory (with is_severed) and TrajectoryPool
├── types.py                                    # Core types: Grid2D, ActionId, BoundingBox, Centroid, Propositions
├── universal_invariants.py                     # Universal algebraic & collinear axial invariant discovery
├── verification.py                             # VerificationBinder and PropositionSet grounding
├── verifier_packet.py                          # Structured symbolic verifier packet
├── virtual_sandbox.py                          # Polymorphic forward kinematic simulation & A* fallback
│
├── prompt_builders/
│   ├── __init__.py                             # Prompt builder exports
│   ├── coder_prompt.py                         # DSL implementation prompt construction
│   ├── explorer_prompt.py                      # Probing and hypothesis prompt construction
│   └── solver_prompt.py                        # Memory-based invariant trajectory planning prompt
```

Competition Entrypoints:
- `kaggle_agent.py`: Gateway adapter implementing the official `ARC_AGI_Agent` interface.
- `local_combat_harness.py`: Local combat harness supporting DashScope and vLLM backends.
- `build_notebook_v10.py`: Standalone LZMA packager for Kaggle submission.

---

## 2. Configuration Contract (`v10_agent/config.py`)

All operational parameters are defined in `V10Config`:

```python
@dataclass
class V10Config:
    # LLM Backend & vLLM Parameters (Qwen 2.5 / 3.8 normative settings)
    llm_advisor_backend: str = "vllm"                      # "vllm" | "dashscope" | "fake" | "ollama"
    model_path: str = "Qwen/Qwen3.8-27B"
    qwen_model_path: str = "Qwen/Qwen3.8-27B"
    vllm_base_url: str = "http://127.0.0.1:1234/v1"
    qwen_vllm_base_url: str = "http://127.0.0.1:1234/v1"
    vllm_api_key: str = "EMPTY"
    qwen_vllm_api_key: str = "EMPTY"
    context_tokens: int = 131072
    qwen_context_tokens: int = 131072
    max_input_tokens: int = 65536
    qwen_max_input_tokens: int = 65536
    max_output_tokens: int = 32000
    qwen_max_output_tokens: int = 32000
    temperature: float = 1.0
    qwen_temperature: float = 1.0
    top_p: float = 0.95
    qwen_top_p: float = 0.95
    top_k: int = 20
    qwen_top_k: int = 20
    seed: int = 42
    qwen_seed: int = 42
    timeout_seconds: int = 700
    qwen_timeout_seconds: int = 700
    multimodal_enabled: bool = True
    qwen_multimodal_enabled: bool = True
    solver_multimodal_enabled: bool = True
    explorer_multimodal_enabled: bool = True
    coder_multimodal_enabled: bool = True
    enable_thinking: bool = True
    qwen_enable_thinking: bool = True
    reasoning_strength: str = "xhigh"
    reasoning_budget_tokens: int = 32000
    qwen_reasoning_budget_tokens: int = 32000

    # Speculative Decoding & Multi-Token Prediction (MTP=3 for Qwen 3.8 27B)
    vllm_mtp_enabled: bool = True                          # Enable MTP speculative decoding
    vllm_mtp_tokens: int = 3                               # k=3 prediction depth
    vllm_speculative_method: str = "mtp"                   # Speculative method
    vllm_speculative_model: str | None = None              # Optional draft model path
    vllm_speculative_config: str | None = None             # Optional raw JSON override
    vllm_speculative_cli_format: str = "auto"              # "auto" | "config_json" | "spec_tokens" | "speculative_model"

    # V10.1-r1 4-Valued Logic & Persistent Tracking Parameters
    enable_undecided_verdict: bool = True                  # Enable Verdict.UNDECIDED
    enable_persistent_tracker: bool = True                 # Multi-frame object identity tracking
    max_evidence_probes_per_level: int = 2                 # Strict ceiling on evidence-seeking probes per level
    max_undecided_streak: int = 2                          # Maximum consecutive UNDECIDED verdicts before NULL fallback
    tracker_iou_threshold: float = 0.3                     # Minimum bounding-box IoU for track match
    tracker_max_distance: float = 5.0                      # Maximum centroid pixel distance for track match
    tracker_max_missed_frames: int = 2                     # Frame threshold before retiring lost tracks
    tracker_confidence_decay: float = 0.85                 # Multiplicative decay per missed frame
    tracker_ambiguity_threshold: float = 0.5               # Ambiguity score threshold triggering UNDECIDED
    undecided_min_remaining_actions: int = 25              # Threshold below which evidence seeking is auto-disabled
    strict_8tier_order: bool = True                        # Enforce strict 8-tier verification cascade

    # Perception
    crop_border_pixels: int = 1                            # 1 px crop to eliminate frame borders

    # Tri-Agent Hard Budgets
    max_chain_attempts_per_level: int = 5                  # Unified 5-attempt budget per level for entire chain
    max_coder_retries_per_level: int = 5
    max_solver_retries_per_level: int = 5
    max_explorer_attempts_per_level: int = 5
    max_explorer_probe_actions_per_level: int = 30
    max_total_llm_calls_per_level: int = 20
    level_wall_clock_limit_seconds: float = 1200.0         # 20-minute soft level budget
    max_primitive_probes_per_level: int = 30
    enable_primitive_probing: bool = True
    probe_reset_after_discrete: bool = False

    # Trajectory Generation
    max_candidates_per_solver_package: int = 4
    max_steps_per_candidate: int = 20
    execute_one_step_at_a_time: bool = True

    # Execution & Safety
    abort_on_dsl_exhaustion: bool = True
    max_actions_per_level: int = 500
    max_actions_per_game: int = 500
    max_game_over_resets_per_game: int = 5
```

---

## 3. Core Data Structures & Models

### 3.1 Brusentsov 4-Valued Logic & Epistemic Signals (`v10_agent/brusentsov_logic.py`)

```python
class Ternary(Enum):
    TRUE = 1          # FOLLOW: Necessary implication held; physical progress made
    FALSE = -1        # NULL: Contradiction; wall collision; branch severed
    IRRELEVANT = 0    # OMIT: Non-contradicting step; entity toggle; branch kept live

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Verdict):
            return other == self
        return super().__eq__(other)


class EpistemicSignal(Enum):
    """Controller signals. Not truth values (ISO-9)."""
    SEEK_EVIDENCE = 1


class Verdict(Enum):
    """Full judge verdict = logical value or epistemic signal."""
    FOLLOW = "FOLLOW"        # maps to Ternary.TRUE
    NULL = "NULL"            # maps to Ternary.FALSE
    OMIT = "OMIT"            # maps to Ternary.IRRELEVANT
    UNDECIDED = "UNDECIDED"  # maps to EpistemicSignal.SEEK_EVIDENCE

    @property
    def ternary(self) -> Ternary | None:
        if self is Verdict.FOLLOW:
            return Ternary.TRUE
        if self is Verdict.NULL:
            return Ternary.FALSE
        if self is Verdict.OMIT:
            return Ternary.IRRELEVANT
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ternary):
            return self.ternary == other
        return super().__eq__(other)
```

#### The `implies_brusentsov` Operator & ISO-10:
$$x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$$

```python
def implies_brusentsov(expected: PropositionSet, observed: PropositionSet) -> Ternary:
    """Evaluate necessary implication following Brusentsov ternary logic.

    Returns:
      TRUE (1)       : Every expected atomic proposition is necessarily contained in the observed set.
      FALSE (-1)     : Any expected proposition is physically contradicted (incompatibility / nullity).
      IRRELEVANT (0) : The expected set is not implied, yet no incompatibility exists (inessential missing effect).
    """
    if len(expected) == 0:
        return Ternary.TRUE

    # 1. Incompatibility check (NULL check: xy'_0 -> FALSE)
    for e in expected:
        for o in observed:
            if contradicts(e, o):
                return Ternary.FALSE

        # Object preservation check
        if e.family == "object_identity" and e.predicate == "preserved":
            is_destroyed = any(
                o.family == "object_identity"
                and o.subject_id == e.subject_id
                and o.predicate in {"destroyed", "missing", "vanished"}
                for o in observed
            )
            if is_destroyed:
                return Ternary.FALSE

    # 2. Necessary containment check (FOLLOW check: xy -> TRUE)
    if all(is_necessarily_contained(e, observed) for e in expected):
        return Ternary.TRUE

    # 3. Inessential missing effect without physical contradiction (OMIT check: x'y' -> IRRELEVANT)
    return Ternary.IRRELEVANT
```

def evaluate_invariant_across_levels(
    invariant: StructuredInvariant,
    current_level_observations: PropositionSet,
) -> Ternary:
    """Evaluate whether a structured invariant holds in a new level context.
    
    TRUE       : observations directly confirm the invariant (kinematics motion, area conservation).
    FALSE      : observations directly contradict the invariant (zero delta on motion, destruction).
    IRRELEVANT : insufficient observations to confirm or deny in the current level state.
    """
    ...
```

---

### 3.2 Perception Data Structures (`v10_agent/planning_set.py` & `v10_agent/arga_lite.py`)

```python
@dataclass(frozen=True)
class BoundingBox:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_row": self.min_row, "min_col": self.min_col,
            "max_row": self.max_row, "max_col": self.max_col,
            "height": self.height, "width": self.width,
        }

@dataclass(frozen=True)
class Centroid:
    row: float
    col: float

    def to_dict(self) -> dict[str, Any]:
        return {"row": round(self.row, 1), "col": round(self.col, 1)}

@dataclass
class PlanningObject:
    id: str
    color: int
    color_histogram: dict[int, int]
    area: int
    bbox: BoundingBox
    centroid: Centroid
    pixels: list[tuple[int, int]]
    is_single_color: bool
    width: int
    height: int
    aspect_ratio: float
    shape_type: str                  # "square", "rectangle", "vertical_line", "horizontal_line", "compound_shape"
    shape_signature: str             # MD5 of raw binary mask
    filled_shape_signature: str      # MD5 of filled binary mask (cavity detection)
    compact_ascii: list[str] = field(default_factory=list)
    mask: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    filled_mask: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    persistent_id: str | None = None # Temporal identity assigned by PersistentObjectTracker
    track_confidence: float = 1.0    # Confidence score from multi-frame tracker


@dataclass
class TrackedObject:
    """Persistent object identity tracked across successive frames."""
    track_id: str                    # Stable identity (e.g. "track_0")
    color: int
    area: int
    bbox: BoundingBox
    centroid: Centroid
    shape_signature: str
    filled_shape_signature: str
    age: int = 1                     # Total frames this track has existed
    hit_streak: int = 1              # Consecutive frames matched
    missed_frames: int = 0           # Consecutive frames unobserved
    confidence: float = 1.0          # Decays exponentially on missed frames
    history: list[Centroid] = field(default_factory=list)


@dataclass(frozen=True)
class SpatialRelation:
    subject_id: str
    relation_type: str               # "touches", "aligned_h", "aligned_v", "contains", "distance",
                                     # "identical_shape", "chiral_mirror_h", "chiral_mirror_v",
                                     # "symmetric_axis_of", "mirrored_across_axis",
                                     # "target_is_below", "target_is_above",
                                     # "target_is_to_the_right", "target_is_to_the_left"
    target_id: str
    metric_value: float | None = None
```

#### 3.2.1 Registered Change-Centric Proposition Families
`REGISTERED_PROPOSITION_FAMILIES` registers propositions evaluating state changes over time:
- `"cumulative_motion"`: Delta accumulation over multiple steps ($\Delta r, \Delta c$).
- `"shape_stability"`: Shape invariance check across transitions ($1 = \text{preserved}, 0 = \text{deformed}$).
- `"occlusion"`: Visual layering / occlusion detection.

#### 3.2.2 Action Space Invariant & Exclusion of ACTION7 (Undo)
Under competition conditions, `ACTION7` is fixed as an `Undo` operation in the ARC-AGI-3 environment. The V10 architecture fundamentally does not use `ACTION7`:
1. **Epistemic State Consistency**: Rather than attempting reverse-step backtracking via `Undo` (which creates ambiguous intermediate states, complicates invariant tracking, and risks infinite oscillatory loops), the agent relies on clean-slate restarts ($S_0$) via environment `RESET`.
2. **Deterministic Forward Traversal**: All candidate execution is strictly forward-directed. `ACTION7` is deliberately excluded from `PlanningSet.allowed_action_ids`, primitive probing, and code generation prompts to enforce unambiguous trajectory evaluation.

---

### 3.3 Trajectory & Candidate Pool Structures (`v10_agent/trajectory.py`)

```python
@dataclass
class GroundedStep:
    step_id: str                     # "s1", "s2", ...
    dsl_function: str                # "action1", "action6", ...
    arguments: dict[str, Any]        # Grounded planning IDs or coordinates
    expected_propositions: PropositionSet
    confidence: str = "confirmed"    # "confirmed" | "low" | "unconfirmed" (ISO-10)
    matching_status: str = "exact"   # "exact" | "ambiguous" | "none"

@dataclass
class CandidateTrajectory:
    trajectory_id: str
    steps: list[dict[str, Any]]
    success_condition: dict[str, Any]
    confidence: float = 1.0
    cursor: int = 0
    active: bool = True

    def current_step(self) -> dict[str, Any] | None:
        if self.active and self.cursor < len(self.steps):
            return self.steps[self.cursor]
        return None

    def advance(self) -> None:
        self.cursor += 1

    def sever(self) -> None:
        self.active = False

    @property
    def is_severed(self) -> bool:
        return not self.active

    def is_finished(self) -> bool:
        return self.cursor >= len(self.steps) or not self.active

@dataclass
class TrajectoryPool:
    proposal_id: str
    candidates: list[CandidateTrajectory] = field(default_factory=list)
    active_candidate_index: int = 0

    def active_candidate(self) -> CandidateTrajectory | None:
        """Advance past finished or inactive candidates to return the next active candidate."""
        while self.active_candidate_index < len(self.candidates):
            cand = self.candidates[self.active_candidate_index]
            if cand.active and not cand.is_finished():
                return cand
            self.active_candidate_index += 1
        return None
```

---

### 3.4 Stratified Memory & Invariant Structures (`v10_agent/memory_contours.py` & `universal_invariants.py`)

```python
@dataclass
class GameMemory:
    """Stratified 3-tier memory store surviving across level transitions within a game."""
    confirmed_action_effects: dict[str, str] = field(default_factory=dict)
    unconfirmed_actions: dict[str, str] = field(default_factory=dict)
    selection_mechanics: list[str] = field(default_factory=list)
    reusable_primitives: list[dict[str, Any]] = field(default_factory=list)
    invariant_rules: list[str] = field(default_factory=list)
    structured_invariants: list[StructuredInvariant] = field(default_factory=list)
    tier1_kinematics_and_topology: list[str] = field(default_factory=list)
    tier2_interactions: list[str] = field(default_factory=list)
    tier3_level_rules: list[str] = field(default_factory=list)
    level_solution_patterns: list[dict[str, Any]] = field(default_factory=list)
    completed_levels: int = 0

    @property
    def confirmed_actors(self) -> set[str]:
        """Return set of object IDs empirically observed to move or receive selection indicators."""
        actors: set[str] = set()
        for note in self.selection_mechanics:
            for oid in re.findall(r"obj_[a-zA-Z0-9_]+", note):
                actors.add(oid)
        for eff in self.confirmed_action_effects.values():
            for oid in re.findall(r"obj_[a-zA-Z0-9_]+", eff):
                actors.add(oid)
        return actors

    def invalidate_action_effect(self, action_id: str) -> None:
        """Remove invalidated action effect from confirmed kinematics upon empirical falsification."""
        self.confirmed_action_effects.pop(action_id, None)
        self.unconfirmed_actions[action_id] = "falsified_by_empirical_verifier"
        prefix = f"Action {action_id} kinematic effect:"
        self.tier1_kinematics_and_topology = [
            r for r in self.tier1_kinematics_and_topology if not r.startswith(prefix)
        ]
        self.invariant_rules = [
            r for r in self.invariant_rules if not r.startswith(prefix)
        ]

    def re_evaluate_invariants(self, observations: Any, level_id: str = "") -> dict[str, int]:
        """Re-evaluate structured invariants against new level observations using Brusentsov logic.
        
        Transitions:
          - TRUE       -> Invariant confirmed on new level (confirmed_on_levels.add(level_id)).
          - FALSE      -> Invariant falsified on new level (falsified_on_levels.add(level_id)).
          - IRRELEVANT -> Invariant remains unchanged / pending evidence.
        """
        from v10_agent.brusentsov_logic import evaluate_invariant_across_levels
        stats = {"confirmed": 0, "falsified": 0, "unchanged": 0}
        for inv in self.structured_invariants:
            verdict = evaluate_invariant_across_levels(inv, observations)
            if verdict == Ternary.TRUE:
                inv.confirmed_on_levels.add(level_id)
                stats["confirmed"] += 1
            elif verdict == Ternary.FALSE:
                inv.falsified_on_levels.add(level_id)
                inv.is_falsified = True
                stats["falsified"] += 1
            else:
                stats["unchanged"] += 1
        return stats

    def handle_level_transition(self) -> None:
        """Sanitize level-transient object IDs, coordinates, and local color bindings across levels.
        
        Tier 1 kinematics (displacement dy, dx) are preserved as foundational world laws.
        Transient coordinates, row/col numbers, bounding boxes, and step counts are purged.
        """
        def is_level_transient(text: str) -> bool:
            t_low = text.lower()
            transient_patterns = [
                r"\brows?\s*\d+", r"\bcols?\s*\d+", r"\bbbox\b", r"\bcentroid\b",
                r"\bfailed\b", r"\bat\s+row\b", r"\bat\s+col\b", r"\(\d+\s*,\s*\d+\)",
                r"\btook\s+\d+\s*steps\b", r"\bin\s+\d+\s*steps\b", r"\bwithin\s+\d+\s*steps\b",
            ]
            return any(re.search(p, t_low) for p in transient_patterns)

        clean_sm = []
        for s in self.selection_mechanics:
            s_clean = re.sub(r"obj_[a-zA-Z0-9_]+", "entity", s)
            s_clean = re.sub(r"\s*\(axis_steps=[^)]+\)", "", s_clean)
            s_clean = re.sub(r"\(color \d+\)", "", s_clean)
            if s_clean not in clean_sm:
                clean_sm.append(s_clean)
        self.selection_mechanics = clean_sm

        sanitized = {}
        for act, eff in self.confirmed_action_effects.items():
            clean_eff = re.sub(r"obj_[a-zA-Z0-9_]+", "entity", eff)
            sanitized[act] = clean_eff
        self.confirmed_action_effects = sanitized

        self.tier1_kinematics_and_topology = [
            self._generalize_text(s) for s in self.tier1_kinematics_and_topology
            if not is_level_transient(s)
        ]
        self.tier2_interactions = [
            self._generalize_text(s) for s in self.tier2_interactions
            if not is_level_transient(s)
        ]
        self.tier3_level_rules = [
            self._generalize_text(r) for r in self.tier3_level_rules
            if not self._is_level_specific(r) and not is_level_transient(r)
        ]

    def format_empirical_context(self, include_curriculum: bool = True) -> str:
        """Format purely empirical facts observed across levels for LLM guidance.
        
        Strict ISO-2 Quarantine Contract:
        When called with include_curriculum=False (for Coder), Tier 3 rules, goal invariants,
        and winning macros are strictly excluded, shielding DSL synthesis from goal leakage.
        """
        ...


@dataclass
class EpistemicMemory:
    """Stores declarative judgments and epistemic signals for the active level."""
    level_id: str
    judgments: list[BrusentsovJudgment] = field(default_factory=list)
    epistemic_signals: list[EpistemicSignal] = field(default_factory=list) # ISO-9 segregated controller signals
    severed_null_signatures: set[str] = field(default_factory=set)
    live_omit_branches: list[dict[str, Any]] = field(default_factory=list)
    current_level_scratchpad: list[dict[str, Any]] = field(default_factory=list)

    def record_judgment(self, judgment: BrusentsovJudgment) -> None:
        """Record truth-valued judgment (FOLLOW, NULL, OMIT)."""
        self.judgments.append(judgment)

    def record_signal(self, signal: EpistemicSignal) -> None:
        """Record controller signal (ISO-9). Segregated from truth-value judgments."""
        self.epistemic_signals.append(signal)

@dataclass
class UniversalInvariant:
    """Discovered geometric or algebraic invariant rule."""
    invariant_type: str              # "axial_symmetry_horizontal", "axial_symmetry_vertical", "chiral_pair"
    subject_id: str
    target_id: str
    axis_id: str | None = None
    metric_value: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
```

---

## 4. Complete System Prompts & Schemas

### 4.1 Explorer Agent Prompts (`v10_agent/prompt_builders/explorer_prompt.py`)

#### `EXPLORER_SYSTEM_PROMPT`:
```text
You are the Explorer Agent for an ARC-AGI-3 environment.
Your task is purely empirical discovery: discover the physical rules, action effects, and coordinate affordances of the current level.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. You describe ONLY factual action effects and coordinate affordances.
2. You NEVER speculate on puzzle goals, winning conditions, or strategies.
3. You NEVER emit action sequences, solution plans, or step-by-step paths.
4. Output MUST be a valid JSON object matching schema 'v10.env_spec.1' enclosed in ```json ... ```.
5. Do NOT overgeneralize limited probe failures (e.g. if 2-3 probed coordinates showed no effect, do NOT conclude that clicking objects is impossible; describe the action's coordinate parameter format and note that active clickable entities across the board remain to be mapped).
```

#### `COORDINATE_HYPOTHESIS_SYSTEM_PROMPT`:
```text
You are an exploratory reasoning agent for ARC-AGI-3 grid puzzles with unknown rules.
Your task is to analyze the 2D grid and detected visual objects, and propose the most informative click target coordinates [x, y] for coordinate action (ACTION6) to uncover the mechanics of this puzzle.

CRITICAL CONSTRAINTS:
1. Propose between 2 and 4 distinct, high-priority click coordinates [x, y].
2. Prioritize clicking on:
   - Centroids of salient objects (interactive tokens, pieces, obstacles, targets).
   - Distinctive corners or boundaries of objects.
   - Grid center or open background if no obvious interactive objects.
3. Coordinates MUST be integers satisfying 0 <= x < width and 0 <= y < height.
4. Output MUST be a single valid JSON object enclosed in ```json ... ``` with schema:
{
  "coordinate_hypotheses": [
    {
      "x": int,
      "y": int,
      "target_description": "short description of target object or location",
      "rationale": "why clicking here is informative"
    }
  ]
}
```

#### Schema `v10.env_spec.1`:
```json
{
  "schema_version": "v10.env_spec.1",
  "snapshot_hash": "e3b0c44298fc1c14",
  "planning_set_id": "ps_01",
  "researched_actions": [
    {
      "action_id": "ACTION2",
      "effect_summary": "active_entity moves dy=3, dx=0",
      "supporting_evidence_ids": ["probe_0"],
      "confidence": 0.8,
      "contradicted": false
    }
  ],
  "coordinate_affordances": [
    {
      "coordinate_candidate_id": "coord_c_obj_0",
      "x": 10,
      "y": 12,
      "source": {"type": "object_centroid", "object_id": "obj_0"},
      "observed_effects": ["interaction_effect"],
      "confidence": 0.7
    }
  ],
  "object_class_notes": [],
  "action_surface_notes": [],
  "invariants": ["object_identity_stable_under_ACTION2"]
}
```

---

### 4.2 Coder Agent Prompts (`v10_agent/prompt_builders/coder_prompt.py`)

#### `CODER_SYSTEM_PROMPT`:
```text
You are the DSL Coder Agent for an ARC-AGI-3 environment.
Your ONLY role is to implement a deterministic, side-effect-free Python 3.12 module and a typed JSON function manifest based strictly on the provided EnvironmentSpecification and SandboxAPI.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. You have NO INFORMATION about the puzzle goal or winning conditions. Do NOT attempt to solve the puzzle.
2. Generate pure functions that declare actions using `api.declare_environment_action(action_id, ...)`.
3. Allowed imports ONLY: math, typing, dataclasses, enum, collections.
4. Strictly FORBIDDEN: os, sys, subprocess, socket, open, eval, exec, compile, dunder traversal (__subclasses__).
5. Function naming & action channels:
   - Inspect ALL AVAILABLE ENVIRONMENT ACTIONS. You MUST implement a function for EVERY available action in the list. Do NOT omit any action.
   - Channel types:
     * Discrete button channel (e.g. ACTION1..ACTION5): Parameterless discrete actions (e.g. `action1(api)`, `action2(api)`). In docstrings, describe their empirical effects from observed facts or confirmed facts.
     * Coordinate channel (e.g. ACTION6): Spatial actions targeting coordinates. Define as `action6(api, x: int = 0, y: int = 0)` with safe default parameter values. In docstrings, describe coordinate targeting affordances.
   - Function names MUST be clean and canonical (e.g. 'action1', 'action2', ..., 'action6'). Do NOT embed transient object IDs into function names.
6. Output MUST contain exactly two blocks:
   - A ```python ... ``` code block containing the DSL module.
   - A ```json ... ``` block containing the JSON manifest matching schema 'v10.dsl_manifest.1'.
```

#### Schema `v10.dsl_manifest.1`:
```json
{
  "schema_version": "v10.dsl_manifest.1",
  "functions": [
    {
      "name": "action1",
      "parameters": [],
      "returns": "effect_declaration",
      "docstring": "Declare discrete button action ACTION1.",
      "purity": "pure_declaration",
      "expected_effect_template": {}
    },
    {
      "name": "action6",
      "parameters": [
        {"name": "x", "type": "int", "default": 0},
        {"name": "y", "type": "int", "default": 0}
      ],
      "returns": "effect_declaration",
      "docstring": "Declare spatial action ACTION6 at coordinates (x, y).",
      "purity": "pure_declaration",
      "expected_effect_template": {}
    }
  ]
}
```

---

### 4.3 Solver Agent Prompts (`v10_agent/prompt_builders/solver_prompt.py`)

#### `SOLVER_SYSTEM_PROMPT`:
```text
You are an expert puzzle solver for 2D grid environments.
Think step by step and perform thorough geometric, topological, and invariant analysis of the visual grid and object affordances before proposing trajectories.
Given the current state, available DSL functions, and past failed attempts, your task is to deduce the underlying geometric/topological invariants and propose solution trajectories.

RULES:
1. Base your reasoning ONLY on the provided empirical facts, object relations, and past trial feedback.
2. All object arguments MUST strictly use IDs from the provided `planning_objects`. Do not invent IDs.
3. Check `past_failed_sequences`. DO NOT repeat them. Formulate alternative hypotheses.
4. Respect grid boundaries and object freedom of motion limits.
5. Provide up to 3 distinct candidate trajectories.

TERNARY EVALUATION SEMANTICS:
Your trajectories will be evaluated step-by-step using Brusentsov ternary logic:
- TRUE (FOLLOW): Step achieved expected physical effect -> trajectory continues.
- IRRELEVANT (OMIT): No contradiction, but expected effect not observed -> trajectory paused / kept live.
- FALSE (NULL): Physical contradiction detected (wall, collision, boundary blockage) -> trajectory permanently terminated.

OUTPUT FORMAT:
You must structure your response using the following XML tags:

<invariant_analysis>
[Briefly explain the inferred geometric/state invariant and your overall strategy]
</invariant_analysis>

<trajectory_1>
[Sequence of DSL function calls, optionally annotated with physical expectations, e.g.:
 action1() EXPECT: dy=-3, dx=0
 action6(x=5, y=10)
 action2()]
</trajectory_1>

<trajectory_2>
[Alternative sequence of DSL function calls]
</trajectory_2>

<trajectory_3>
[Optional third sequence]
</trajectory_3>
```

#### Canonical Solver Output Response:
```xml
<invariant_analysis>
The puzzle requires aligning the mobile entity with the target socket across the vertical symmetry axis.
We first move the entity to the line of symmetry, then advance it into the target cavity.
</invariant_analysis>

<trajectory_1>
1. action1() EXPECT: dy=-3, dx=0
2. action1() EXPECT: dy=-3, dx=0
3. action4() EXPECT: dy=0, dx=3
</trajectory_1>

<trajectory_2>
1. action5()
2. action2() EXPECT: dy=3, dx=0
3. action2() EXPECT: dy=3, dx=0
</trajectory_2>
```

#### 4.4 Solver Turn 2 Win Reflection Prompt (`distill_level_win_invariants`):
```text
[EXECUTION OUTCOME: LEVEL WON]
Your proposed trajectory ({cand_id}) successfully solved this level and achieved victory!
Summary of executed transitions: {exec_desc}
Your original hypothesis: {hyp}

Now, reflect on this victory, your hypothesis, and the physical mechanisms observed.
Formulate 2-4 domain-general physical, palette, and goal invariants for subsequent levels.

CRITICAL RULES FOR INVARIANTS:
1. NO coordinates, row/column numbers, bounding boxes, or grid dimensions (these change every level).
2. NO step counts or action repetition numbers (distances vary across levels).
3. NO literal button sequences or macros like 'action1 -> action5' (order of actions varies).
4. Focus on describing:
   - [PALETTE & ROLES]: Explicitly map observed object colors to their functional roles across levels (which color is the static target/socket, which color is the movable actor piece, which color is the symmetry axis/tool, and what color represents active selection indicators or background holes).
   - [GOAL]: How the win condition is satisfied (e.g. covering targets of target color with primary pieces and their reflections).
   - [ENTITIES & MECHANICS]: The distinct roles and physical mechanics of entities (e.g. mirror axis reflecting pieces across itself).
   - [CONTROL]: How entity cycling or toggling operates across active elements.

Format your response strictly inside <distilled_invariants>...</distilled_invariants> with bullet points:
<distilled_invariants>
- [PALETTE & ROLES]: Color X is ..., Color Y is ...
- [GOAL]: ...
- [ENTITIES & MECHANICS]: ...
- [CONTROL]: ...
</distilled_invariants>
```

---

## 5. Perception Engine Implementation (`v10_agent/arga_lite.py`)

### 5.1 Substantive Spatial Relation Filtering (The Anti-Cavity Filter)
To prevent internal background cavities (holes) from polluting directional navigation relations:

```python
for i in range(num_objs):
    obj_a = objects[i]
    for j in range(i + 1, num_objs):
        obj_b = objects[j]

        # Standard topological relations
        # (touches, aligned_h, aligned_v, contains, distance, identical_shape, chiral_mirror_h/v)

        # SUBSTANTIVE FILTER: Directional relations are computed strictly between
        # non-background, non-noise objects (area >= 4, color != 0)
        if obj_a.area >= 4 and obj_b.area >= 4 and obj_a.color != 0 and obj_b.color != 0:
            delta_row = round(obj_b.centroid.row - obj_a.centroid.row, 1)
            delta_col = round(obj_b.centroid.col - obj_a.centroid.col, 1)
            if abs(delta_row) >= 1.0:
                rel_v = "target_is_below" if delta_row > 0 else "target_is_above"
                relations.append(SpatialRelation(obj_a.id, rel_v, obj_b.id, metric_value=abs(delta_row)))
            if abs(delta_col) >= 1.0:
                rel_h = "target_is_to_the_right" if delta_col > 0 else "target_is_to_the_left"
                relations.append(SpatialRelation(obj_a.id, rel_h, obj_b.id, metric_value=abs(delta_col)))
```

### 5.2 Universal Invariant Discovery (`v10_agent/universal_invariants.py`)
Extracts collinear mirror axes and computes symmetric step offsets:

```python
def discover_axial_symmetries(planning_set: PlanningSet) -> list[UniversalInvariant]:
    """Identify horizontal and vertical mirror axes and corresponding symmetric sprite pairs."""
    invariants = []
    # Identify horizontal axes (width >= 10, height <= 2) and vertical axes (height >= 10, width <= 2)
    for obj in planning_set.objects:
        if obj.width >= 10 and obj.height <= 2:
            invariants.append(UniversalInvariant(
                invariant_type="axial_symmetry_horizontal",
                subject_id="piece", target_id="target", axis_id=obj.id
            ))
        elif obj.height >= 10 and obj.width <= 2:
            invariants.append(UniversalInvariant(
                invariant_type="axial_symmetry_vertical",
                subject_id="piece", target_id="target", axis_id=obj.id
            ))
    return invariants
```

### 5.3 Visual Dual-View Rendering & Indicator Marker Aggregation
Implemented in `v10_agent/frame_media.py` and `v10_agent/prompt_builders/solver_prompt.py`:
- Generates `solver_raw_frame.png` (unannotated pristine pixels) and `solver_annotated_frame.png` (labeled bounding boxes with object IDs, colors, roles).
- Aggregates small dots ($area \le 2$) of identical color into composite marker groups (`marker_group_color_{c}` / `marker_dots_c{c}`).
- Quadrant-balanced salience sorting (TL, TR, BL, BR + confirmed actors) preventing visual/spatial bias.

### 5.4 Persistent Object Tracking & Strict 8-Tier Cascade Implementation

#### 5.4.1 Tracker Matching Algorithm (`v10_agent/tracker.py`)
`PersistentObjectTracker.update(snapshot_or_objects)` executes a greedy bipartite match between existing `TrackedObject` tracks and detected `PlanningObject` instances:
1. Matches candidates using bounding-box IoU ($\ge 0.3$) and centroid Euclidean distance ($\le 5.0$ px).
2. Computes ambiguity margin between top-1 and top-2 match scores; if margin $< 0.1$, flags `last_ambiguity_score > 0.5`.
3. Updates matched tracks: `confidence = 1.0`, `hit_streak += 1`, `missed_frames = 0`, appends centroid to `history`.
4. Updates unmatched tracks: `missed_frames += 1`, `confidence *= 0.85`. Tracks with `missed_frames > 2` are retired.

#### 5.4.2 Strict 8-Tier Decision Cascade (`v10_agent/judge.py`)
`LayeredVerifier.evaluate_transition` executes the immutable 8-tier decision cascade enforcing ISO-10:

```python
# Tier 1: Syntactic & Invariant Malformation
if malformed_arguments or invariant_malformation:
    return BrusentsovJudgment(verdict=Verdict.NULL, explanation="Tier 1: Malformed step syntax or invariant")

# Tier 2: Fatal Domain Violations (Zero displacement on confirmed directional motion)
if is_confirmed_motion and zero_grid_delta:
    return BrusentsovJudgment(verdict=Verdict.NULL, explanation="Tier 2: Confirmed motion yielded zero delta (collision)")

# Epistemic Uncertainty Short-Circuit (ISO-10)
if (getattr(step, "confidence", "confirmed") in ("low", "unconfirmed")
    or getattr(step, "matching_status", "exact") == "ambiguous"
    or (tracker and tracker.last_ambiguity_score > tracker.ambiguity_threshold)):
    return BrusentsovJudgment(verdict=Verdict.UNDECIDED, explanation="Epistemic uncertainty short-circuit")

# Tier 3: Physical Contradiction (Direct contradiction between expected and observed)
if any(contradicts(e, o) for e in expected for o in observed):
    return BrusentsovJudgment(verdict=Verdict.NULL, explanation="Tier 3: Physical contradiction in propositions")

# Tier 4: Necessary Containment (Affirmation / FOLLOW)
if all(is_necessarily_contained(e, observed) for e in expected):
    return BrusentsovJudgment(verdict=Verdict.FOLLOW, explanation="Tier 4: Necessary containment satisfied")

# Tier 5: Downstream Disconfirmation (High confidence step violated confirmed physics)
if confirmed_physics_violated:
    return BrusentsovJudgment(verdict=Verdict.NULL, explanation="Tier 5: Downstream disconfirmation")

# Tier 6: Downstream Ambiguity / Unconfirmed Delta
if zero_grid_delta and not is_confirmed_motion:
    return BrusentsovJudgment(verdict=Verdict.UNDECIDED, explanation="Tier 6: Zero delta on unconfirmed action")

# Tier 7 & 8: Irrelevant Frame Noise & Default Fallback
return BrusentsovJudgment(verdict=Verdict.OMIT, explanation="Tier 7/8: Benign passive transition")
```

---

## 6. Independent Symbolic Trajectory Executor (`v10_agent/symbolic_executor.py`)

### 6.1 `prepare_and_execute_step`
Pre-verifies signature grounding, handles kwargs filtering, catches sandbox faults, and trips circuit breakers:

```python
def prepare_and_execute_step(
    self,
    pool: TrajectoryPool | None,
    planning_set: PlanningSet,
    active_module: SandboxedModule | None,
    epistemic_memory: EpistemicMemory,
    syntax_memory: SyntaxErrorMemory,
) -> StepExecutionResult:
    candidate = pool.active_candidate() if pool else None
    if candidate is None:
        return StepExecutionResult(verdict=StepExecutionVerdict.NO_CANDIDATE)

    step_dict = candidate.current_step()
    step_id = str(step_dict.get("step_id", "s0"))
    fn_name = str(step_dict.get("dsl_function", ""))
    seq_sig = " -> ".join(str(s.get("dsl_function", "")) for s in candidate.steps[:candidate.cursor + 1])

    # 1. Pre-verification against severed signatures
    if not candidate.active or epistemic_memory.is_severed(candidate.trajectory_id) or epistemic_memory.is_severed(seq_sig):
        candidate.sever()
        return StepExecutionResult(verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED, circuit_broken=True)

    # 2. Check function existence in SandboxedModule
    if fn_name not in active_module.namespace or not callable(active_module.namespace[fn_name]):
        candidate.sever()
        syntax_memory.record_error(SyntaxErrorRecord(error_type="MissingDSLFunction", error_message=f"Function {fn_name!r} not found"))
        return StepExecutionResult(verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED, circuit_broken=True)

    # 3. Ground arguments against PlanningSet
    try:
        grounded_step = self.binder.ground_step(step_dict, planning_set)
    except GroundingError as ge:
        candidate.sever()
        epistemic_memory.sever_branch(seq_sig)
        return StepExecutionResult(verdict=StepExecutionVerdict.PRE_VERIFICATION_FAILED, circuit_broken=True)

    # 4. Sandboxed execution
    try:
        effect = self.sandbox_executor.execute(
            module=active_module, function_name=grounded_step.dsl_function,
            arguments=grounded_step.arguments, planning_set=planning_set,
        )
        return StepExecutionResult(verdict=StepExecutionVerdict.SUCCESS, effect=effect, grounded_step=grounded_step)
    except Exception as exc:
        candidate.sever()
        epistemic_memory.sever_branch(seq_sig)
        syntax_memory.record_error(SyntaxErrorRecord(error_type=type(exc).__name__, error_message=str(exc)))
        return StepExecutionResult(verdict=StepExecutionVerdict.SANDBOX_EXECUTION_FAILED, circuit_broken=True)
```

### 6.2 `evaluate_transition` with Empirical Falsification & Multi-Candidate Reset Traversal
Evaluates empirical transitions with Brusentsov logic, checks for kinematic falsification, and controls pool candidate traversal:

```python
def evaluate_transition(
    self,
    pending_step: GroundedStep,
    before_snapshot: Any,
    after_obs: dict[str, Any],
    planning_set: PlanningSet,
    active_pool: TrajectoryPool | None,
    epistemic_memory: EpistemicMemory,
    game_memory: Any = None,
    action_dict: dict[str, Any] | None = None,
) -> TransitionEvaluationResult:
    judgment = self.verifier.evaluate_transition(
        step=pending_step, before_snapshot=before_snapshot, after_obs=after_obs,
        planning_set=planning_set, game_memory=game_memory, action_dict=action_dict,
    )

    active_cand = active_pool.active_candidate() if active_pool else None

    # Check for empirical falsification at cursor == 0
    before_grid = getattr(before_snapshot, "grid", None)
    if before_grid is None and hasattr(planning_set, "grid"):
        before_grid = planning_set.grid
    after_grid = after_obs.get("grid")
    zero_grid_delta = (before_grid is not None and after_grid is not None and before_grid == after_grid)

    is_initial_step = (active_cand is None or active_cand.cursor == 0)
    confirmed_eff = game_memory.confirmed_action_effects.get(act_id, "") if game_memory else ""
    is_confirmed_motion = "moved" in confirmed_eff and any(d in confirmed_eff for d in ("UP", "DOWN", "LEFT", "RIGHT"))

    falsification_detected = False
    falsified_action = None

    if is_initial_step and zero_grid_delta and is_confirmed_motion:
        falsification_detected = True
        falsified_action = act_id or pending_step.dsl_function
        judgment = BrusentsovJudgment(
            trajectory_id=pending_step.step_id,
            step_id=pending_step.step_id,
            verdict=Ternary.FALSE,
            expected_propositions=pending_step.expected_propositions,
            observed_propositions=judgment.observed_propositions,
            explanation=f"Confirmed motion action {act_id} produced zero grid delta from pristine board. Kinematic assumption falsified!",
        )

    epistemic_memory.record_judgment(judgment)

    is_won = False
    if after_obs:
        st = str(after_obs.get("state", "")).upper()
        if st in ("WIN", "WON", "DONE", "VICTORY", "TERMINAL"):
            is_won = True
        if (after_obs.get("levels_completed", 0) or 0) > 0:
            is_won = True

    cand_finished = False

    if judgment.verdict == Ternary.TRUE:
        if active_cand is not None:
            active_cand.advance()
        cand_advanced = True
        cand_severed = False
        cand_finished = (active_cand is None or active_cand.is_finished()) if active_pool else True

    elif judgment.verdict == Verdict.NULL:
        if active_cand is not None:
            active_cand.sever()
            seq_sig = " -> ".join(str(s.get("dsl_function", "")) for s in active_cand.steps[:active_cand.cursor + 1])
            epistemic_memory.sever_branch(seq_sig)
        cand_advanced = False
        cand_severed = True
        cand_finished = True

    elif judgment.verdict == Verdict.UNDECIDED:
        # Candidate not severed, cursor held, evidence probe needed (ISO-9 / ISO-10)
        cand_advanced = False
        cand_severed = False
        cand_finished = False
        evidence_needed = True
        evidence_hint = getattr(judgment, "explanation", "Epistemic uncertainty requires evidence probe")
        replan_needed = False
        reset_needed = False

    else:  # Verdict.OMIT
        if active_cand is not None:
            active_cand.advance()
        cand_advanced = True
        cand_severed = False
        cand_finished = (active_cand is None or active_cand.is_finished()) if active_pool else True

    # Record failed sequence feedback to scratchpad
    if cand_finished and not is_won and active_cand is not None:
        if hasattr(epistemic_memory, "record_attempt_feedback"):
            epistemic_memory.record_attempt_feedback(
                hypothesis=f"Candidate {active_cand.trajectory_id}",
                trajectory_summary=" -> ".join(f"{s.get('dsl_function')}()" for s in active_cand.steps),
                status="executed_but_level_not_won",
                reason="Trajectory executed completely but game did not advance to next level. DO NOT REPEAT THIS SEQUENCE!",
            )

    # Multi-Candidate Pool Traversal via RESET
    if is_won:
        replan_needed = False
        reset_needed = False
    elif judgment.verdict == Verdict.UNDECIDED:
        replan_needed = False
        reset_needed = False
    elif cand_finished:
        next_cand = active_pool.active_candidate() if active_pool else None
        if next_cand is not None and not falsification_detected:
            # Another candidate exists: RESET board and continue without calling Solver!
            replan_needed = False
            reset_needed = True
        else:
            # All candidates exhausted or falsification detected: RESET board and call Solver/Reprobe!
            replan_needed = True
            reset_needed = True
    else:
        replan_needed = False
        reset_needed = False

    return TransitionEvaluationResult(
        verdict=judgment.verdict, judgment=judgment,
        candidate_advanced=cand_advanced, candidate_severed=cand_severed,
        replan_needed=replan_needed, reset_needed=reset_needed,
        falsification_detected=falsification_detected, falsified_action=falsified_action,
        evidence_needed=evidence_needed, evidence_hint=evidence_hint,
    )
```

### 6.3 Virtual Sandbox Role & Non-Override Contract (`v10_agent/virtual_sandbox.py`)
1. **Offline Kinematic Safety & Step Repair**:
   The `VirtualSandbox` verifies proposed LLM candidate trajectories by simulating object displacements against confirmed Tier 1 kinematics. If a proposed step results in an off-grid displacement or wall collision, the candidate is cleanly truncated before dispatch, eliminating unviable action bursts.
2. **Strict Non-Override Rule (No Index 0 Hijacking)**:
   The Virtual Sandbox must **never** synthesize fallback scripts and insert them at index 0 ahead of LLM candidates (`candidates.insert(0, cand_synth)` is strictly prohibited). The LLM's geometric and topological reasoning must always be tested first. Synthetic generation acts **strictly as an offline fallback** when the LLM produces zero valid candidates (`if not candidates_to_use:`).
3. **Trajectory Package Planning Ceiling**:
   The Solver plans full multi-step trajectory packages (up to 4 candidates per package, 10–30 steps each) within a hard ceiling of 5 attempts per level (`max_chain_attempts_per_level = 5`). Under no circumstances is the Solver called per step; all per-step execution is handled deterministically by `SymbolicTrajectoryExecutor`.

---

## 7. GameSession Orchestration Loop (`v10_agent/session.py`)

### 7.1 Circuit Breaker Immediate Reset & Fallback Shielding
When `prepare_and_execute_step` trips the circuit breaker:

```python
if exec_res.circuit_broken:
    next_cand = self.active_pool.active_candidate() if self.active_pool else None
    if next_cand is None:
        self.replan_requested = True
        self.active_pool = None
    else:
        logger.info(f"Session: Circuit broken on candidate; advancing to next candidate: {next_cand.trajectory_id}")

    # Immediately emit RESET to clean board instead of executing random fallbacks
    reset_action = {
        "id": "RESET",
        "action_id": "RESET",
        "data": {},
        "reasoning": {"source": "circuit_breaker_immediate_reset", "error": exec_res.error_message},
    }
    self.pending_action = reset_action
    self.last_engine_action = "RESET"
    self.pending_step = None
    return reset_action
```

### 7.2 Post-Step Observation Ingestion & Falsification Handling
When `observe_action_result` receives the post-step observation:

```python
if eval_res.falsification_detected:
    logger.warning(
        f"Session: Kinematic falsification detected for action {eval_res.falsified_action!r}. "
        f"Invalidating confirmed kinematic rules, scheduling clean board reset and micro-reprobes."
    )
    # Invalidate stale kinematics in GameMemory and known_actions
    if eval_res.falsified_action:
        game_mem.invalidate_action_effect(eval_res.falsified_action)
        self.known_actions.discard(eval_res.falsified_action)

    # Clear old environment spec to force fresh generation from new probes
    env_mem = self.memory_manager.get_env_spec_memory("session")
    env_mem.specs.clear()

    # Reset board to pristine state and trigger micro-reprobe
    self.solver_reset_pending = True
    self.solver_reset_reason = "falsification_clean_reprobe_reset"
    self.probing_phase = True
    if hasattr(self.explorer, "probe_manager") and hasattr(self.explorer.probe_manager, "schedule_falsification_reprobe"):
        reprobes = self.explorer.probe_manager.schedule_falsification_reprobe(
            [eval_res.falsified_action] if eval_res.falsified_action else None
        )
        self.probe_queue.clear()
        self.probe_queue.extend(reprobes)

elif eval_res.replan_needed or (self.active_pool is not None and self.active_pool.active_candidate() is None):
    self.replan_requested = True
    self.active_pool = None

if eval_res.reset_needed:
    self.solver_reset_pending = True
    self.solver_reset_reason = (
        "falsification_clean_reprobe_reset" if eval_res.falsification_detected
        else ("replan_reset_clean_state" if eval_res.replan_needed else "next_candidate_reset_clean_state")
    )
```

### 7.3 Level Transition & Solver Turn 2 Reflection (`handle_level_transition`)
Upon solving a level, `GameSession` invokes Turn 2 reflection to distill invariants and sanitizes cross-level memory:

```python
def handle_level_transition(self, new_level_id: str) -> None:
    """Clean level-local state, distill win invariants, and preserve stratified GameMemory."""
    if self.current_level_id:
        game_mem = self.memory_manager.get_game_memory("session")
        setup_str = f"{len(self.last_planning_set.objects) if self.last_planning_set else 'several'} entities"

        # Ask Solver to reflect on the win and distill domain-general invariants
        distilled_invariants = self.solver.distill_level_win_invariants(
            winning_candidate=winning_cand,
            execution_summary=exec_summary,
        )
        for inv in distilled_invariants:
            game_mem.record_stratified_invariant(inv, tier=3)

        primary_inv = distilled_invariants[0] if distilled_invariants else "Satisfied level goal via coordinated alignment"
        game_mem.record_level_solution(
            level_id=self.current_level_id,
            setup_summary=setup_str,
            invariant_rule=primary_inv,
            winning_macro="",  # Omitted to prevent button-sequence pollution in future levels
        )

    self.current_level_id = new_level_id
    self.active_module = None
    self.active_manifest = None
    self.active_pool = None
    self.pending_step = None
    self.pending_action = None
    self.probe_queue = []
    self.probing_phase = True
    self.known_actions = set(game_mem.confirmed_action_effects.keys())
    self.memory_manager.handle_level_transition(new_level_id)
```

### 7.4 Action Surface Classification & Unified Hybrid Probing Queue
At the onset of each level or upon action surface shifts, `GameSession` classifies the environment into one of four operational modes:

```python
has_coords = "ACTION6" in available_actions
has_discrete = any(str(a).upper() in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5") for a in available_actions)
is_dyn = self.explorer.probe_manager.is_dynamic_action_surface(available_actions, self.known_actions)

if has_coords and has_discrete:
    self.active_pipeline = "hybrid"
elif has_coords:
    self.active_pipeline = "coordinate"
elif is_dyn:
    self.active_pipeline = "dynamic"
else:
    self.active_pipeline = "discrete"
```

In `hybrid` environments, spatial coordinate clicks and discrete button presses coexist. The agent avoids artificial mode separation: both discrete sweep probes and coordinate hypothesis probes are enqueued directly into `self.probe_queue`. After probing completes, a clean `RESET` is executed before any Solver candidate trajectory executes from pristine state $S_0$.

### 7.5 Safe Evidence-Seeking Loop & Strict NOOP Elimination (`v10_agent/session.py`)
When `SymbolicTrajectoryExecutor` issues a `Verdict.UNDECIDED`:

```python
# 3.8. V10.1 Evidence-Seeking Blocking Dispatch (Strict NOOP Prohibition)
if self.evidence_seeking_active:
    if self.probe_queue:
        probe_action_item = self.probe_queue.pop(0)
        probe_action = probe_action_item.to_dict() if hasattr(probe_action_item, "to_dict") else dict(probe_action_item)
        self.last_probe_action = probe_action
        self.last_snapshot = snapshot
        self.last_planning_set = planning_set
        self.pending_action = probe_action
        self.pending_step = None
        self.last_engine_action = str(probe_action.get("action_id") or probe_action.get("id") or "ACTION1").upper()
        self.evidence_probes_executed += 1
        return probe_action
    else:
        # Probe queue exhausted: fall through cleanly to NULL (sever candidate + clean reset)
        self.undecided_fallback_to_null += 1
        self.evidence_seeking_active = False
        self.undecided_streak = 0
        self.pending_step_snapshot = None
        if self.active_pool and self.active_pool.active_candidate():
            self.active_pool.active_candidate().sever()
        self.solver_reset_pending = True
        self.solver_reset_reason = "evidence_probe_exhausted_null"
```

---

## 8. Verification & Validation Commands

All **235 unit tests** across 45 test files validate every module, invariant, and mathematical formulation:

```powershell
# Run complete test suite (235 tests across 45 test files)
py -3.12 -m pytest v10_agent/tests

# Compile and build self-extracting Kaggle notebook
py -3.12 build_notebook_v10.py

# Run standalone Phase-A structural preflight check
py -3.12 lcld_preflight.py

# Verify Phase 1 (4-valued logic, Verdict, Tracker dataclasses)
py -3.12 -m pytest v10_agent/tests/test_v10_1_phase1.py

# Verify Phase 2 (Strict 8-tier cascade & ISO-10 compliance)
py -3.12 -m pytest v10_agent/tests/test_v10_1_phase2.py

# Verify Phase 3 (Change-centric propositions & persistent grounding)
py -3.12 -m pytest v10_agent/tests/test_v10_1_phase3.py

# Verify Phase 4 (Safe evidence-seeking loop & NOOP elimination)
py -3.12 -m pytest v10_agent/tests/test_v10_1_phase4.py

# Verify vLLM MTP=3 speculative decoding and graceful fallback
py -3.12 -m pytest v10_agent/tests/test_vllm_mtp_config.py
```

---

## 9. Architectural Axioms & Final Invariants

```text
1. Three isolated agents: Explorer (Facts), Coder (Syntax), Solver (Epistemic Reasoning).
2. Four isolated stores: EnvironmentSpecMemory, SyntaxErrorMemory, EpistemicMemory, GameMemory.
3. Declarative planning is separated from execution: Solver proposes XML trajectories; Symbolic Executor runs step-by-step.
4. Trajectory budget ceiling: Solver plans full multi-step trajectory packages within a hard limit of 5 attempts per level (max_chain_attempts_per_level = 5, ~10-20 candidates total). The Solver is NEVER called per step.
5. Unified hybrid pipeline: Seamless execution of hybrid environments (active_pipeline = "hybrid") combining discrete actions and coordinate clicks without artificial mode lockout.
6. Transitions are judged by Brusentsov necessary implication: xy (FOLLOW), xy'_0 (OMIT), x'y' (NULL), and epistemic uncertainty (UNDECIDED). Active consequence checking via implies_brusentsov verifies asserted EXPECT: propositions.
7. Perception is filtered: Directional navigation relations strictly exclude 1-pixel cavities and noise dots.
8. The board is protected: Multi-candidate pools execute sequentially via RESET without premature replanning; contradictions trigger clean resets.
9. Zero game-specific bias: No hardcoded game keywords, quadrant templates, or movement assumptions.
10. Dynamic kinematics & empirical falsification: Kinematic assumptions falsified on pristine boards trigger automatic invalidation, clean reset, and focused micro-reprobing.
11. Stratified 3-tier memory with cross-level sanitization: Knowledge is partitioned into Physics (Tier 1), Interaction Dynamics (Tier 2), and Deduced Rules (Tier 3), preserving kinematics world laws while sanitizing transient local colors and coordinates.
12. Strict ISO-2 curriculum quarantine: Coder empirical context strictly excludes Tier 3 rules, goal invariants, and winning macros, preventing goal leakage into DSL synthesis.
13. Virtual Sandbox non-override: Forward simulation repairs candidates by truncating collisions, but never hijacks position 0 ahead of LLM candidates.
14. 4-Valued Brusentsov Logic (ISO-9): Logical truth values (FOLLOW, OMIT, NULL) are strictly segregated from controller signals (UNDECIDED). Epistemic signals must never be treated as truth values.
15. Strict Containment (ISO-10): Raw LLM expectation mismatches without physical invariant breaches never emit NULL, preventing catastrophic trajectory severance.
16. Strict NOOP Elimination: The agent strictly never emits NOOP to the Arcade runtime under any condition. Probes strictly use real available actions (ACTION1..7), falling through to NULL if probe queue is exhausted.
17. Speculative MTP=3 with Graceful Boot Fallback: Speculative model execution failure automatically re-launches standard non-speculative serving without crashing the competition run.
```
