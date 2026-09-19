# ARC-AGI-3 LCLD Agent — Version 10.2

[![Tests](https://img.shields.io/badge/tests-262%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Authoritative repository for the **ARC-AGI-3 LCLD Agent (Version 10.2)** for the ARC Prize 2026 competition.

The agent implements an autonomous, neuro-symbolic, double-loop learning architecture with strict Tri-Agent separation of powers, Brusentsov 4-valued entailment logic, multi-frame persistent object tracking, safe evidence-seeking loop, multi-candidate sequential reset traversal, extended trajectory horizons (20–30 steps), stratified cross-level memory contours, VisibleCycle orbit loop recovery, and production vLLM runtime serving.

---

## Architectural Highlights

1. **Tri-Agent Separation of Powers (Explorer, Coder, Solver)**
   - **Explorer Agent (Facts)**: Maps empirical physics, action displacements, and coordinate affordances without speculating on puzzle goals.
   - **Coder Agent (DSL)**: Synthesizes a deterministic, sandboxed Python 3.12 DSL module and typed manifest. Governed by strict ISO-2 quarantine (receives zero puzzle goals or curriculum winning macros).
   - **Solver Agent (Epistemic Reasoning)**: Deduces geometric invariants and emits declarative multi-step trajectory packages (`<trajectory_1>` through `<trajectory_4>`) annotated with physical expectations (`EXPECT:`).
   - **Double-Loop Routing**: Syntax/sandbox errors route exclusively to `SyntaxErrorMemory` for Coder repair; physical/logical transition outcomes route exclusively to `EpistemicMemory` for Solver adaptation.

2. **Brusentsov 4-Valued Logic of Necessary Entailment**
   - Grounded in Nikolai P. Brusentsov's logic of necessary entailment ($x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$):
     - **$+1$ (`Verdict.FOLLOW` / `Ternary.TRUE`)**: Necessary physical progress observed ($xy$). Candidate cursor advances.
     - **$0$ (`Verdict.OMIT` / `Ternary.IRRELEVANT`)**: Neutral, non-contradicting transition ($x'y$, e.g. entity selection, passive step). Step is accepted; branch remains live in memory.
     - **$-1$ (`Verdict.NULL` / `Ternary.FALSE`)**: Strict physical contradiction ($xy'_0$, wall collision, zero displacement on motion primitive, or refutation of asserted `EXPECT:` proposition). Candidate is severed immediately; clean `RESET` is queued.
     - **`Verdict.UNDECIDED` (Epistemic Signal)**: Ambiguous outcome ($|\text{cost}_1 - \text{cost}_2| < 0.15$ or track confidence $< 0.60$). Schedules safe evidence-seeking probe without premature branch severing.
   - Evaluated via `implies_brusentsov(expected_propositions, observed_propositions)` and `contradicts()` across 7 semantic families.

3. **Persistent Multi-Frame Object Tracking (`PersistentObjectTracker`)**
   - 4-component normalized association cost:
     $$\text{Cost} = 0.30 \cdot \text{Cost}_{\text{color}} + 0.40 \cdot \text{Cost}_{\text{dist}} + 0.20 \cdot (1 - \text{Jaccard}) + 0.10 \cdot \text{Cost}_{\text{area}}$$
   - Exponential moving average velocity smoothing ($\alpha = 0.3$), shape stability decay, ring buffer history, cumulative motion tracking, and occlusion heuristics.

4. **Multi-Trajectory Candidate Pool Traversal & Extended Horizons (20–30 Steps)**
   - Solver plans full multi-step trajectory packages (up to 4 candidates per package, 20–30 steps each) using concise `action(count=N)` repetition syntax, under a hard ceiling of **5 planning attempts per level** (`max_chain_attempts_per_level = 5`).
   - **Solver is NEVER called per step**. Per-step execution is handled deterministically by the independent `SymbolicTrajectoryExecutor`.
   - When a candidate finishes or is severed without a win, an environmental `RESET` restores the pristine board ($S_0$), and the executor advances to the next candidate without re-invoking the LLM.

5. **VisibleCycle Orbit Loop Recovery**
   - High-speed row-level MD5 state hashing detects repeating state-action cycles (periods 1..8, $\ge 4$ repetitions, $\ge 24$ actions).
   - Upon detection, the active candidate is severed, replanning is requested, and a clean reset restores board state, preventing level budget burn (limit 2 per level).

6. **Production Time Budgeting & Resilient vLLM Serving**
   - Dynamic deadline tracking with a 15-second reserve (`is_deadline_exceeded`). Automatic abort of LLM requests upon reserve breach; socket timeout clamping.
   - Strictly validated vLLM production flags: `--no-enable-prefix-caching`, `--enable-chunked-prefill`, `--async-scheduling`, `--no-enable-log-requests`, `--disable-uvicorn-access-log`.
   - Sub-millisecond socket probe teardown (`serving_teardown.py`) and non-blocking `threading.RLock` watchdog daemon (`vllm_server_watchdog.py`).

7. **Stratified 3-Tier Cross-Level Memory (`GameMemory`)**
   - **Tier 1 (Kinematics & Physics)**: Confirmed action displacements ($dy, dx$) are preserved across levels as foundational world laws (`is_level_transient` kinematics protection).
   - **Tier 2 (Interactions)**: Entity selection mechanics and coordinate affordances.
   - **Tier 3 (Deduced Rules & Curriculum)**: High-level invariant symmetries and abstract strategies distilled from won levels via Solver Turn 2 Win Reflection (`distill_level_win_invariants`).
   - Pristine frame invariant discovery: Cross-level invariant re-evaluation is deferred to the pristine initial frame ($S_0$, `level_initial_grid is None`).

8. **Deterministic Perception (ARGA-Lite)**
   - High-speed 4-connectivity connected components segmentation (< 2 ms/frame).
   - Dual shape hashing (`shape_signature` and `filled_shape_signature` for cavity detection).
   - Substantive directional relation filtering (excluding 1-pixel cavities and background noise dots).
   - Collinear mirror axis discovery and freedom-of-motion bounds computation.

---

## Authoritative Specifications

The repository includes complete, self-contained specifications reflecting the entire architecture and implementation:

- **[Architectural Specification (Version 10.2)](ARCHITECTURAL_SPECIFICATION_V10.0.md)**: Exhaustive mathematical grounding on Brusentsov's logic of entailment, Aristotelian syllogistics, Lewis Carroll's nullity indices, 3-tier memory stratification, isolation axioms ISO-1 through ISO-10, and runtime orchestrator state machines.
- **[Engineering Specification (Version 10.2)](ENGINEERING_SPECIFICATION_V10.0.md)**: Exhaustive implementation blueprint detailing all 35 source modules, method signatures, data structures, algorithms, configuration parameters, and test suites.

---

## Repository Layout

```
.
├── ARCHITECTURAL_SPECIFICATION_V10.0.md   # Authoritative system architecture specification (v10.2)
├── ENGINEERING_SPECIFICATION_V10.0.md     # Authoritative engineering implementation specification (v10.2)
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
└── v10_agent/                             # Core Agent Package
    ├── action_adapter.py                  # Action normalization and boundary validation
    ├── arga_lite.py                       # Deterministic ARGA-Lite perception & relation extraction
    ├── brusentsov_logic.py                # Brusentsov 4-valued logic engine & implies_brusentsov
    ├── config.py                          # Normative V10Config with deadline tracking & CLI builder
    ├── cycle_detector.py                  # VisibleCycle orbit loop recovery engine
    ├── dsl_coder.py                       # Coder Agent (Call 2) & SandboxedModule compiler
    ├── explorer_agent.py                  # Explorer Agent (Call 1) & PrimitiveProbeManager
    ├── fallback_symbolic.py               # Deterministic symbolic fallback engine
    ├── frame_media.py                     # Dual-frame raw and annotated PNG visual rendering
    ├── game_adapter.py                    # Competition environment and game interface adapter
    ├── judge.py                           # LayeredVerifier with 8-tier decision cascade
    ├── llm_advisor.py                     # LLM client supporting vLLM, OpenAI, DashScope, Ollama
    ├── logging.py                         # Structured JSON audit and session logger
    ├── memory_contours.py                 # Stratified 3-tier memory stores & MemoryContourManager
    ├── observe.py                         # Observation normalization and border cropping
    ├── planning_set.py                    # PlanningSet, PlanningObject, SpatialRelation, CoordinateCandidate
    ├── policy.py                          # Action selection policy and fallback arbitration
    ├── sandbox.py                         # Restricted AST checker and SandboxExecutor
    ├── serving_setup.py                   # Serving setup export for package consumers
    ├── serving_teardown.py                # Serving teardown export for package consumers
    ├── session.py                         # GameSession top-level state machine orchestrator
    ├── solver_agent.py                    # Solver Agent (Call 3), XML parser & reflection engine
    ├── symbolic_executor.py               # Independent SymbolicTrajectoryExecutor controller
    ├── tracker.py                         # PersistentObjectTracker multi-frame identity engine
    ├── trajectory.py                      # CandidateTrajectory and TrajectoryPool
    ├── types.py                           # Core types: Grid2D, ActionId, BoundingBox, Centroid
    ├── universal_invariants.py            # Universal algebraic & collinear axial invariant discovery
    ├── verification.py                    # VerificationBinder and PropositionSet grounding
    ├── verifier_packet.py                 # Structured symbolic verifier packet
    ├── virtual_sandbox.py                 # Forward kinematic simulation & generalized A* fallback
    ├── vllm_server_watchdog.py            # Server watchdog export for package consumers
    │
    ├── prompt_builders/                   # Isolated prompt generation modules
    │   ├── coder_prompt.py
    │   ├── explorer_prompt.py
    │   └── solver_prompt.py
    │
    └── tests/                             # Complete test suite (260 unit tests across 48 files)
```

---

## Verification & Test Suite

The codebase is thoroughly covered by **260 unit tests** across 48 test files:

```bash
# Run all unit tests
python -m pytest v10_agent/tests -q

# Run structural preflight verification
python lcld_preflight.py

# Run Brusentsov logic & verification tests
pytest v10_agent/tests/test_brusentsov_logic.py
pytest v10_agent/tests/test_judge_brusentsov_world_laws.py

# Run PersistentObjectTracker and cycle detector tests
pytest v10_agent/tests/test_v10_1_phase1.py
pytest v10_agent/tests/test_cycle_detector_and_deadline.py

# Run memory stratification and curriculum quarantine tests
pytest v10_agent/tests/test_memory_contours.py
pytest v10_agent/tests/test_prompt_isolation.py
```

**Test Execution Status**:
```
262 passed in 8.30s (100% SUCCESS)
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
4. **Board Protection via RESET**: Candidates execute sequentially from clean pristine board states ($S_0$); contradictions trigger immediate clean resets.
5. **Brusentsov Implication**: Transitions are verified via necessary entailment ($xy \lor xy'_0 \lor x'y'$), rejecting material implication paradoxes.
