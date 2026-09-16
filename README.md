# ARC-AGI-3 LCLD Agent — Version 10.0

[![Tests](https://img.shields.io/badge/tests-191%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Authoritative repository for the **ARC-AGI-3 LCLD Agent (Version 10.0)** for the ARC Prize 2026 competition.

The agent implements an autonomous, neuro-symbolic, double-loop learning architecture with strict Tri-Agent separation of powers, Brusentsov 3-valued necessary implication logic, multi-trajectory sequential reset traversal, stratified cross-level memory contours, and unified hybrid environment support.

---

## Architectural Highlights

1. **Tri-Agent Separation of Powers (Explorer, Coder, Solver)**
   - **Explorer Agent (Facts)**: Maps empirical physics, action displacements, and coordinate affordances without speculating on puzzle goals.
   - **Coder Agent (DSL)**: Synthesizes a deterministic, sandboxed Python 3.12 DSL module and typed manifest. Governed by strict ISO-2 quarantine (receives zero puzzle goals or curriculum winning macros).
   - **Solver Agent (Epistemic Reasoning)**: Deduces geometric invariants and emits declarative multi-step trajectory packages (`<trajectory_1>`, `<trajectory_2>`, `<trajectory_3>`) annotated with physical expectations (`EXPECT:`).
   - **Double-Loop Routing**: Syntax/sandbox errors route exclusively to `SyntaxErrorMemory` for Coder repair; physical/logical transition outcomes route exclusively to `EpistemicMemory` for Solver adaptation.

2. **Brusentsov 3-Valued Necessary Implication Logic**
   - Grounded in Nikolai P. Brusentsov's ternary logic ($x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$):
     - **$+1$ (`Ternary.TRUE`, FOLLOW)**: Necessary physical progress observed ($xy$). Candidate cursor advances.
     - **$0$ (`Ternary.IRRELEVANT`, OMIT)**: Neutral, non-contradicting transition ($xy'_0$, e.g. toggling entity control). Step is accepted; branch remains live in memory.
     - **$-1$ (`Ternary.FALSE`, NULL)**: Strict physical contradiction ($x'y'$, wall collision, zero displacement on motion primitive, or refutation of asserted `EXPECT:` proposition). Candidate is severed immediately; clean `RESET` is queued.
   - Evaluated via `implies_brusentsov(expected_propositions, observed_propositions)`.

3. **Multi-Trajectory Candidate Pool Traversal & 5-Attempt Ceiling**
   - Solver plans full multi-step trajectory packages (up to 4 candidates per package, 10–30 steps each) within a hard ceiling of **5 planning attempts per level** (`max_chain_attempts_per_level = 5`), yielding ~10–20 total candidate trajectories.
   - **Solver is NEVER called per step**. Per-step execution is handled deterministically by the independent `SymbolicTrajectoryExecutor`.
   - When a candidate finishes or is severed without a win, an environmental `RESET` restores the pristine board ($S_0$), and the executor advances to the next candidate without re-invoking the LLM.

4. **Unified Hybrid Environment Support (`active_pipeline = "hybrid"`)**
   - Dynamic classification into `hybrid`, `discrete`, `coordinate`, or `dynamic`.
   - Seamlessly handles hybrid action surfaces where discrete buttons (`ACTION1..5`) and spatial clicking (`ACTION6`) coexist. Both discrete sweep probes and coordinate hypothesis probes are enqueued directly into `probe_queue` without mutual lockout.

5. **Stratified 3-Tier Cross-Level Memory (`GameMemory`)**
   - **Tier 1 (Kinematics & Physics)**: Confirmed action displacements ($dy, dx$) are preserved across levels as foundational world laws (`is_level_transient` kinematics protection).
   - **Tier 2 (Interactions)**: Entity selection mechanics and coordinate affordances.
   - **Tier 3 (Deduced Rules & Curriculum)**: High-level invariant symmetries and abstract strategies distilled from won levels via Solver Turn 2 Win Reflection (`distill_level_win_invariants`).
   - Cross-level invariant re-evaluation (`re_evaluate_invariants`) validates persisting invariants against new level observations.

6. **Virtual Sandbox Safety & Non-Override Invariant**
   - Offline forward kinematic simulation validates steps and repairs candidates by cleanly truncating paths before wall collisions.
   - **Strict Non-Override Rule**: The sandbox never synthesizes fallback scripts at index 0 ahead of LLM candidates. Heuristic synthesis acts strictly as an offline fallback when zero valid model candidates exist.

7. **Deterministic Perception (ARGA-Lite)**
   - High-speed 4-connectivity connected components segmentation (< 2 ms/frame).
   - Dual shape hashing (`shape_signature` and `filled_shape_signature` for cavity detection).
   - Substantive directional relation filtering (excluding 1-pixel cavities and background noise dots).
   - Collinear mirror axis discovery and freedom-of-motion bounds computation.

---

## Repository Layout

```
.
├── ARCHITECTURAL_SPECIFICATION_V10.0.md   # Authoritative system architecture specification
├── ENGINEERING_SPECIFICATION_V10.0.md     # Authoritative engineering implementation specification
├── README.md                              # This documentation
├── .gitignore                             # Clean repository exclusion rules
│
├── build_notebook_v10.py                  # Self-extracting LZMA Kaggle notebook builder
├── lcld_competition_child.py              # Isolated child process runner for competition execution
├── kaggle_agent.py                        # ARC_AGI_Agent competition gateway adapter
├── submission.py                          # Competition entrypoint and environment interface
├── lcld_preflight.py                      # Preflight environment and model validation
├── phase_a_heavy_smoke.py                 # Deep diagnostic verification runner for Phase A
│
└── v10_agent/                             # Core Agent Implementation
    ├── action_adapter.py                  # Action normalization and boundary validation
    ├── arga_lite.py                       # Deterministic ARGA-Lite perception & relation extraction
    ├── brusentsov_logic.py                # Brusentsov ternary logic & implies_brusentsov operator
    ├── config.py                          # Normative V10Config with environment overrides
    ├── dsl_coder.py                       # Coder Agent (Call 2) & SandboxedModule compiler
    ├── explorer_agent.py                  # Explorer Agent (Call 1) & PrimitiveProbeManager
    ├── fallback_symbolic.py               # Deterministic symbolic fallback engine
    ├── frame_media.py                     # Dual-frame raw and annotated PNG visual rendering
    ├── game_adapter.py                    # Competition environment and game interface adapter
    ├── judge.py                           # LayeredVerifier with multi-signal ground-truth verification
    ├── llm_advisor.py                     # LLM client supporting vLLM, OpenAI, DashScope, Ollama
    ├── logging.py                         # Structured JSON audit and session logger
    ├── memory_contours.py                 # Stratified 3-tier memory stores & MemoryContourManager
    ├── observe.py                         # Observation normalization and border cropping
    ├── planning_set.py                    # PlanningSet, PlanningObject, SpatialRelation, CoordinateCandidate
    ├── policy.py                          # Action selection policy and fallback arbitration
    ├── sandbox.py                         # Restricted AST checker and SandboxExecutor
    ├── session.py                         # GameSession top-level state machine orchestrator
    ├── solver_agent.py                    # Solver Agent (Call 3), XML parser & reflection engine
    ├── symbolic_executor.py               # Independent SymbolicTrajectoryExecutor controller
    ├── trajectory.py                      # CandidateTrajectory and TrajectoryPool
    ├── types.py                           # Core types: Grid2D, ActionId, BoundingBox, Centroid
    ├── universal_invariants.py            # Universal algebraic & collinear axial invariant discovery
    ├── verification.py                    # VerificationBinder and PropositionSet grounding
    ├── verifier_packet.py                 # Structured symbolic verifier packet
    ├── virtual_sandbox.py                 # Forward kinematic simulation & A* fallback
    ├── prompt_builders/                   # Isolated prompt generation modules
    │   ├── coder_prompt.py
    │   ├── explorer_prompt.py
    │   └── solver_prompt.py
    └── tests/                             # Complete test suite (191 unit tests across 38 files)
```

---

## Verification & Test Suite

The codebase is thoroughly covered by **191 unit tests** across 38 test files, validating every perception descriptor, memory tier, isolation invariant, and Brusentsov transition rule.

```bash
# Run all unit tests
pytest v10_agent/tests

# Run dynamic falsification and reprobing tests
pytest v10_agent/tests/test_dynamic_falsification_and_reprobe.py

# Run Brusentsov ternary logic and active checking tests
pytest v10_agent/tests/test_brusentsov_logic.py
pytest v10_agent/tests/test_judge_brusentsov_world_laws.py

# Run memory stratification and curriculum quarantine tests
pytest v10_agent/tests/test_memory_contours.py
pytest v10_agent/tests/test_prompt_isolation.py

# Run collinear axis and clean reset tests
pytest v10_agent/tests/test_collinear_axis_and_clean_reset.py
```

---

## Kaggle Submission Packaging

The agent is packaged into a self-extracting, base64-encoded LZMA archive embedded inside a single Kaggle notebook cell, strictly complying with Kaggle's < 985 KB submission ceiling:

```bash
python build_notebook_v10.py
```

- **Output**: `notebooks/arc-prize-2026-lcld-qwen-v10.ipynb` (~195 KB, using only ~20% of the allowed limit).
- **Metadata**: `notebooks/kernel-metadata.json` ready for Kaggle CLI push (`kaggle kernels push -p notebooks/`).

---

## Formal Isolation Axioms

1. **Tri-Agent Isolation**: Explorer writes only facts; Coder synthesizes pure DSL functions without goal visibility; Solver plans declaratively using only DSL functions.
2. **Four Memory Contours**: `EnvironmentSpecMemory`, `SyntaxErrorMemory`, `EpistemicMemory`, and `GameMemory` are strictly disjoint.
3. **No Per-Step Solver Invocations**: Trajectories are batched (up to 4 candidates per package, max 5 attempts per level); per-step execution is deterministic.
4. **Board Protection via RESET**: Candidates execute sequentially from clean pristine board states ($S_0$); contradictions trigger immediate clean resets.
5. **Brusentsov Implication**: Transitions are verified via necessary implication ($xy \lor xy'_0 \lor x'y'$), rejecting vacuous Boolean truths.
