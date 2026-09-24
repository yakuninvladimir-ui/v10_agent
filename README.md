# ARC-AGI-3 LCLD Agent — Version 10.5

[![Tests](https://img.shields.io/badge/tests-360%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Authoritative repository for the **ARC-AGI-3 LCLD (Locally Constrained Learning and Discovery) Agent (Version 10.5)** for the ARC Prize 2026 competition.

The agent implements an autonomous, neuro-symbolic, double-loop learning architecture with strict Tri-Agent separation of powers, Brusentsov 4-valued entailment logic, Carrollian nullity completeness, atomic macro-step decomposition, early trajectory severance with structured epistemic diagnostics, two-tier action affordances, reactive DSL invalidation, universal multimodal DualView perception, 16-color semantic palette mapping, persistent object tracking, and production vLLM runtime serving.

---

## Architectural Highlights

1. **Tri-Agent Separation of Powers (Explorer, Coder, Solver)**
   - **Explorer Agent (Facts)**: Maps empirical physics, action kinematics, and coordinate affordances on pristine frames ($S_0$) without speculating on puzzle goals.
   - **Coder Agent (DSL)**: Synthesizes a deterministic, sandboxed Python 3.12 DSL module and typed manifest. Governed by strict ISO-2 quarantine (receives zero puzzle goals or curriculum winning macros). Features automatic augmentation for missing confirmed actions.
   - **Solver Agent (Epistemic Reasoning)**: Deduces geometric invariants and emits declarative multi-step trajectory packages (`<trajectory_1>` through `<trajectory_4>`) annotated with physical expectations (`EXPECT:`). Receives structured differential scratchpads with exact tile deltas and failure diagnostics.
   - **Double-Loop Routing**: Syntax/sandbox errors route exclusively to `SyntaxErrorMemory` for Coder repair; physical/logical transition outcomes route exclusively to `EpistemicMemory` for Solver adaptation.

2. **Brusentsov 4-Valued Logic & Carrollian Nullity Completeness**
   - Grounded in Nikolai P. Brusentsov's logic of necessary entailment ($x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$):
     - **$+1$ (`Verdict.FOLLOW` / `Ternary.TRUE`)**: Necessary physical progress observed ($xy$). Candidate cursor advances.
     - **$0$ (`Verdict.OMIT` / `Ternary.IRRELEVANT`)**: Neutral, non-contradicting transition ($x'y$, e.g. entity selection, passive step, empty expectation). Step is accepted; branch remains live in memory (ISO-10).
     - **$-1$ (`Verdict.NULL` / `Ternary.FALSE`)**: Strict physical contradiction ($xy'_0$). Candidate is severed immediately; clean `RESET` is queued.
     - **`Verdict.UNDECIDED` (Epistemic Signal)**: Ambiguous outcome ($|\text{cost}_1 - \text{cost}_2| < 0.15$ or track confidence $< 0.60$). Schedules safe evidence-seeking probe without premature branch severing.
   - **Exhaustive Contradiction Detection (`contradicts`)**:
     - *Object Identity*: Expected `preserved`, observed `destroyed` / `vanished` / `missing`.
     - *Physical Stagnation*: Expected motion ($\Delta_{\text{exp}} \neq 0$), observed stationary ($\Delta_{\text{obs}} == 0$).
     - *Unintended Mutation*: Expected stationary/invariant ($\Delta_{\text{exp}} == 0$ or `unchanged`), observed motion ($\Delta_{\text{obs}} \neq 0$).
     - *Direction Inversion*: Expected vector sign opposite to observed vector sign ($\text{sign}(\Delta_{\text{exp}}) \times \text{sign}(\Delta_{\text{obs}}) < 0$).
     - Multi-format kinematics: seamless support for tuple vectors `(dy, dx)`, `moved`, `step_moved`, scalar signs `row_delta`, `col_delta`, and cross-comparisons.

3. **Macro-Step Decomposition & Early Null Severance (Elimination of `break_on_null`)**
   - Macro-commands like `action1(count=N)` are unrolled into $N$ atomic steps by `_decompose_expected_propositions_for_substep`.
   - State invariants (`unchanged`, `preserved`) are enforced on every intermediate step. Incremental motion signs (`step_moved: (sgn(dy), sgn(dx))`) are verified at each transition, and cumulative displacement (`moved: (dy, dx)`) is evaluated on the final step.
   - Contradictions immediately halt the candidate trajectory (`active_cand.sever()`), dropping remaining repetitions and invoking `_build_step_contradiction_diagnostic()` to log a structured diagnostic (`[FAILED ATTEMPT N DIAGNOSTIC]`).
   - Clean environmental `RESET` restores $S_0$ before proceeding to the next candidate.

4. **Two-Tier Action Lifecycle & Action Affordance Completeness**
   - Actions probed on pristine $S_0$ yielding zero delta ($\Delta = 0$) are **not discarded**. They are classified as `conditional_candidate_actions` and preserved in `available_actions` with affordance `EffectClass.CONDITIONAL_TRIGGER` (`status: unconfirmed_on_s0`).
   - Ensures actions requiring conditional preconditions (e.g. holding an item or standing on a switch) remain visible to the Coder.
   - Transitions immediately into `confirmed_effective_actions` as soon as an observable delta is detected in any state.

5. **Reactive DSL Invalidation & Auto-Augmentation**
   - If an action missing from the certified manifest produces an observable delta during execution, `GameSession` triggers reactive invalidation (`active_module = None`, `active_manifest = None`, `coder_failed_for_level = False`, `replan_requested = True`).
   - `DSLCoder.generate_dsl` automatically synthesizes canonical sandboxed wrappers (`def actionX(api): ...`) and manifest descriptors for any confirmed actions omitted by the LLM Coder.

6. **Universal Multimodal Dual-View Perception**
   - Generates synchronized pairs of PNG frames: `raw_frame.png` (unannotated pristine pixels) and `annotated_frame.png` (colored bounding boxes, alias labels, centroids).
   - Passed to Explorer, Coder, and Solver agents to eliminate perceptual blindness to multi-color compound objects caused by monochromatic connected-component parsing.
   - Strict Cartesian $(Row, Col) \leftrightarrow (Y, X)$ coordinate notation ($X=\text{column}, Y=\text{row}$).

7. **16-Color PaletteRoleMap & Role-Aware Semantic Labeling**
   - Manages the full ARC-AGI-3 16-color palette ($0..15$).
   - Replaces destructive `[COLOR]` tokens with non-destructive, role-aware semantic labeling: `Color N (ROLE)` (e.g. `Color 1 (ACTOR)`, `Color 2 (HAZARD)`, `Color 8 (TARGET)`).

8. **Empirically Grounded Exemplars**
   - `DefeatExemplar`: Captured upon `GAME_OVER` / lethal failure; grounds negative barrier invariants ($xy'_0 \to \text{NULL}$).
   - `VictoryExemplar`: Captured upon level victory; grounds positive canonical exemplars ($xy \to \text{FOLLOW}$) for cross-level transfer.

9. **Persistent Multi-Frame Object Tracking (`PersistentObjectTracker`)**
   - 4-component normalized association cost:
     $$\text{Cost} = 0.30 \cdot \text{Cost}_{\text{color}} + 0.40 \cdot \text{Cost}_{\text{dist}} + 0.20 \cdot (1 - \text{Jaccard}) + 0.10 \cdot \text{Cost}_{\text{area}}$$
   - Exponential moving average velocity smoothing ($\alpha = 0.3$), shape stability decay, cumulative motion tracking, and occlusion heuristics.

10. **VisibleCycle Orbit Loop Recovery**
    - High-speed row-level MD5 state hashing detects repeating state-action cycles (periods 1..8, $\ge 4$ repetitions, $\ge 24$ actions).
    - Upon detection, the active candidate is severed, replanning is requested, and a clean reset restores board state, preventing level budget burn (limit 2 per level).

11. **Production Time Budgeting & Resilient vLLM Serving**
    - Dynamic deadline tracking with a 15-second reserve (`is_deadline_exceeded`). Automatic abort of LLM requests upon reserve breach; socket timeout clamping.
    - Strictly validated vLLM production flags: `--no-enable-prefix-caching`, `--enable-chunked-prefill`, `--async-scheduling`, `--no-enable-log-requests`, `--disable-uvicorn-access-log`.
    - Sub-millisecond socket probe teardown (`serving_teardown.py`) and non-blocking `threading.RLock` watchdog daemon (`vllm_server_watchdog.py`).

---

## Authoritative Specifications

The repository includes complete, self-contained specifications reflecting the entire architecture and implementation:

- **[Architectural Specification (Version 10.5)](ARCHITECTURAL_SPECIFICATION_V10.0.md)**: Exhaustive mathematical grounding on Brusentsov's logic of entailment, Aristotelian syllogistics, Lewis Carroll's nullity indices, two-tier action affordances, reactive invalidation, 5-tier memory stratification, isolation axioms ISO-1 through ISO-12, and runtime orchestrator state machines.
- **[Engineering Specification (Version 10.5)](ENGINEERING_SPECIFICATION_V10.0.md)**: Exhaustive implementation blueprint detailing all core source modules, method signatures, data structures, algorithms, configuration parameters, and test suites.

---

## Repository Layout

```
.
├── ARCHITECTURAL_SPECIFICATION_V10.0.md   # Authoritative system architecture specification (v10.5)
├── ENGINEERING_SPECIFICATION_V10.0.md     # Authoritative engineering implementation specification (v10.5)
├── README.md                              # This documentation
├── .gitignore                             # Clean repository exclusion rules
│
├── build_notebook_v10.py                  # Self-extracting LZMA Kaggle notebook builder
├── lcld_competition_child.py              # Isolated child process runner for competition execution
├── kaggle_agent.py                        # ARC_AGI_Agent competition gateway adapter
├── submission.py                          # Competition entrypoint and environment interface
├── lcld_preflight.py                      # Preflight environment and model validation
├── phase_a_heavy_smoke.py                 # Deep diagnostic verification runner for Phase A
├── serving_setup.py                       # vLLM serving configuration abstraction
├── serving_teardown.py                    # Fast socket-based server probe and cleanup
├── vllm_server_watchdog.py                # Non-blocking health watchdog daemon
│
├── tools/
│   └── ast_code_guardian.py               # Static AST code purity & anti-gaming auditor
│
└── v10_agent/                             # Core Agent Package
    ├── action_adapter.py                  # Action normalization and boundary validation
    ├── arga_lite.py                       # Deterministic ARGA-Lite perception & relation extraction
    ├── brusentsov_logic.py                # Brusentsov 4-valued logic engine & Carroll nullity
    ├── config.py                          # Normative V10Config with deadline tracking & CLI builder
    ├── cycle_detector.py                  # VisibleCycle orbit loop recovery engine
    ├── dsl_coder.py                       # Coder Agent (Call 2) & auto-augmentation compiler
    ├── explorer_agent.py                  # Explorer Agent (Call 1) & Two-tier PrimitiveProbeManager
    ├── fallback_symbolic.py               # Deterministic symbolic fallback engine
    ├── frame_media.py                     # Dual-frame raw and annotated PNG visual rendering
    ├── game_adapter.py                    # Competition environment and game interface adapter
    ├── judge.py                           # LayeredVerifier with 8-tier decision cascade
    ├── llm_advisor.py                     # LLM client supporting vLLM, OpenAI, DashScope, Ollama
    ├── logging.py                         # Structured JSON audit and session logger
    ├── memory_contours.py                 # 5 memory stores (Palette, Exemplars, Invariants, etc.)
    ├── observe.py                         # Observation normalization and 1px border cropping
    ├── planning_set.py                    # PlanningSet, PlanningObject, SpatialRelation, CoordinateCandidate
    ├── policy.py                          # Action selection policy and fallback arbitration
    ├── sandbox.py                         # Restricted AST checker and SandboxExecutor
    ├── session.py                         # GameSession top-level state machine & reactive invalidation
    ├── solver_agent.py                    # Solver Agent (Call 3), XML parser & macro-step unrolling
    ├── symbolic_executor.py               # SymbolicTrajectoryExecutor with early null severance
    ├── tracker.py                         # PersistentObjectTracker multi-frame identity engine
    ├── trajectory.py                      # CandidateTrajectory and TrajectoryPool
    ├── types.py                           # Core types: Grid2D, ActionId, BoundingBox, Centroid
    ├── universal_invariants.py            # Universal algebraic & collinear axial invariant discovery
    ├── verification.py                    # VerificationBinder and PropositionSet grounding
    ├── verifier_packet.py                 # Structured symbolic verifier packet
    ├── virtual_sandbox.py                 # Forward kinematic simulation & generalized A* fallback
    │
    ├── prompt_builders/                   # Isolated prompt generation modules
    │   ├── coder_prompt.py
    │   ├── explorer_prompt.py
    │   └── solver_prompt.py
    │
    └── tests/                             # Complete test suite (360 tests across 52 files)
```

---

## Verification & Test Suite

The codebase is thoroughly covered by **360 tests** (unit tests, property-based tests with Hypothesis, and synthetic calibration micro-worlds):

```bash
# 1. Run the AST Code Guardian audit (103 files)
python tools/ast_code_guardian.py v10_agent

# 2. Run all unit and property-based tests
pytest v10_agent/tests -q

# 3. Run structural preflight verification
python lcld_preflight.py

# 4. Run Brusentsov logic & severance PBT tests
pytest v10_agent/tests/test_pbt_brusentsov_axioms.py
pytest v10_agent/tests/test_pbt_brusentsov_severance.py
pytest v10_agent/tests/test_judge_brusentsov_world_laws.py

# 5. Run Reactive DSL Invalidation & Two-Tier Action tests
pytest v10_agent/tests/test_reactive_dsl_invalidation.py

# 6. Run DualView & Multimodal tests
pytest v10_agent/tests/test_universal_multimodal_dual_view.py
```

**Test Execution Status**:
```
360 passed in 13.23s (100% SUCCESS)
```

---

## Kaggle Submission Packaging

The agent is packaged into a self-extracting, base64-encoded LZMA archive embedded inside a single Kaggle notebook cell, strictly complying with Kaggle's < 985 KB submission ceiling:

```bash
python build_notebook_v10.py
```

- **Output**: `notebooks/arc-prize-2026-lcld-qwen-v10.ipynb` (~251 KB, using only ~25% of the allowed limit).
- **Metadata**: `notebooks/kernel-metadata.json` ready for Kaggle CLI push (`kaggle kernels push -p notebooks/`).

---

## Formal Isolation Axioms

1. **Tri-Agent Isolation**: Explorer writes only facts; Coder synthesizes pure DSL functions without goal visibility (ISO-2); Solver plans declaratively using only DSL functions.
2. **Four Memory Contours**: `EnvironmentSpecMemory`, `SyntaxErrorMemory`, `EpistemicMemory`, and `GameMemory` are strictly disjoint.
3. **No Per-Step Solver Invocations**: Trajectories are batched (up to 4 candidates per package, max 5 attempts per level); per-step execution is deterministic.
4. **Board Protection via Clean RESET**: Candidates execute sequentially from clean pristine board states ($S_0$); contradictions trigger immediate early severance and clean resets.
5. **Brusentsov Implication & Carroll Nullity**: Transitions are verified via necessary entailment ($xy \lor xy'_0 \lor x'y'$), rejecting material implication paradoxes.
6. **Two-Tier Action Preservation**: Actions unconfirmed on $S_0$ are preserved as `CONDITIONAL_TRIGGER` affordances and reactively certified upon observing deltas.
