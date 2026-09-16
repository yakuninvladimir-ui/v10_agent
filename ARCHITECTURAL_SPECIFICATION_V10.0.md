# ARC-AGI-3 LCLD Agent
# Architectural Specification — Version 10.0
# (Tri-Agent Hierarchy, Independent Symbolic Execution, Multi-Trajectory Reset Traversal, Hybrid Environment Support & Brusentsov Ternary Logic)

---

## 0. Purpose and Architectural Vision

This document defines the Version 10.0 architecture of the **ARC-AGI-3 LCLD (Locally Constrained Learning and Discovery) Agent**. The agent is engineered to operate autonomously under:

1. **Hidden environment mechanics, hybrid action spaces (A+B), and dynamic action surfaces** across diverse, unseen ARC-AGI-3 grid environments;
2. **Deterministic, object-centric perception** grounded by ARGA-Lite and an immutable `PlanningSet`;
3. **Strict offline competition execution constraints** (Kaggle runtime, no external internet, strict token/time limits, total artifact size under 985 KB);
4. **Multimodal local LLM reasoning (Qwen FP8 via vLLM / OpenAI-compatible API)** deployed exclusively through three strictly segregated roles;
5. **Tri-Agent Authority Separation** (Explorer, Coder, Solver) with non-overlapping authority contours;
6. **Strictly isolated memory stores** (`EnvironmentSpecMemory`, `SyntaxErrorMemory`, `EpistemicMemory`, and cross-level `GameMemory`) eliminating cross-role context contamination;
7. **An Independent Symbolic Trajectory Executor** that physically decouples declarative trajectory generation (Qwen Solver) from verification (Judge) and environment interaction;
8. **Multi-Trajectory Candidate Pool Traversal via Clean RESET**: Sequential execution of multiple proposed candidate trajectories through environmental `RESET` without premature Solver replanning;
9. **Full Multi-Step Trajectory Packages with Strict Budget Ceilings**: Solver plans complete multi-step trajectory packages (2–4 candidates of 10–30 steps) under a strict ceiling of 5 attempts per level (~10–20 trajectories total per level), completely eliminating per-step LLM invocations;
10. **Brusentsov 3-Valued Logic ($xy \lor xy'_0 \lor x'y'$) & Active Consequence Verification**: Mathematical and operational evaluation of physical progress (FOLLOW: $+1$), benign non-contradictory neutrality (OMIT: $0$), and physical contradictions (NULL: $-1$) powered by explicit `EXPECT:` proposition checking;
11. **Hybrid Environment Support (A+B)**: Seamless unified handling of discrete actions (`ACTION1..5`), spatial coordinate clicks (`ACTION6`), and hybrid mixtures without mutual exclusion or artificial mode barriers;
12. **Double-Loop Learning** with domain-specific error routing (syntax/execution faults to Coder; physical/logical outcomes to Solver);
13. **Substantive Perception Filtering & Motion Freedom Grounding** preventing distractors, background cavities, and wall collisions from derailing the planning loop;
14. **Strict state orchestration** preventing dirty-board cascades and guaranteeing clean-state resets on branch contradictions;
15. **Dual-Signal Dynamic Kinematics & Empirical Falsification**: Detection of action nullity on pristine board ($S_0$) triggering immediate invalidation of falsified kinematics in `GameMemory`, pruning of known actions, clean board reset, and focused micro-reprobing;
16. **Stratified 3-Tier Episodic and Cross-Level Memory Hierarchy**: Disjoint stratification of Foundational Physics & Kinematics (Tier 1), Interaction Dynamics & Selection (Tier 2), and High-Level Invariant Rules & Curriculum (Tier 3), featuring confirmed actor tracking, Tier 1 displacement preservation, and ISO-2 curriculum quarantine;
17. **Dynamic Cross-Level Invariant Discovery & Re-evaluation**: Progressive accumulation, cross-level diffing, and active ternary re-evaluation of structural invariants as level complexity increases;
18. **Solver Two-Turn Reflection & Win Invariant Distillation**: Epistemic reflection upon level win where Solver distills domain-general invariants and strategies into curriculum memory without button-sequence pollution;
19. **Universal Multimodal Dual-View Perception**: Dual-frame visual projection (unannotated raw frame + bounding-box labeled annotated frame with indicator marker grouping and quadrant-balanced salience).

### 0.1 What the Architecture Is NOT
- It is **not** a monolithic prompt-loop that asks a language model to "play the game" by generating raw action keys.
- It is **not** a step-by-step solver caller: the Solver is **NEVER** called on every step. The Solver plans complete multi-step trajectories upfront, which the deterministic symbolic executor carries out step-by-step with clean resets between attempts.
- It is **not** a system that allows an LLM to execute arbitrary Python or directly emit environment actions without formal verification.
- It is **not** a classical reinforcement learning agent or raw-pixel policy network requiring millions of training frames.
- It is **not** an unconstrained classical planner relying on pre-specified, human-authored domain models (PDDL).
- It is **not** an overfitted heuristic script with hardcoded assumptions tailored to specific games (such as `ar25` or `ft09`).
- It does **not** use `ACTION7`: by competition environment conditions, `ACTION7` is a hardcoded `Undo` operation. The agent fundamentally does not rely on step-reversal, preserving epistemic state consistency instead through clean resets ($S_0$) and forward deterministic planning.

### 0.2 What the Architecture IS
- A **high-assurance, neuro-symbolic multi-agent system** with rigorous separation of powers.
- An environment where high-level declarative planning (Solver) is translated into typed symbolic actions, validated against sandboxed domain-specific languages (Coder), verified empirically against physical world laws (Judge), and executed deterministically (Symbolic Executor).
- A hybrid-capable engine supporting discrete actions (`ACTION1..5`), coordinate clicks (`ACTION6`), or both simultaneously within a unified execution pipeline.
- An epistemic engine grounded in Brusentsov's logic of necessary implication, preserving plausible hypotheses while aggressively severing contradictory branches.
- A multi-hypothesis executor that tests candidate trajectories sequentially from clean starting states ($S_0$), maintaining short-term scratchpad memory to avoid repeating failed sequences.

### 0.3 Normative Authority Hierarchy
Authority is strictly ordered from highest to lowest:

```
1. Gateway & Environment Contracts (Arcade/env.step, official RESET, GAME_OVER semantics)
                                 │
2. Brusentsov LayeredVerifier (Absolute authority on transition validity: FOLLOW / NULL / OMIT)
                                 │
3. GameSession Orchestrator (State machine owner, budget manager, lifecycle coordinator)
                                 │
4. SymbolicTrajectoryExecutor (Independent execution controller, sandbox caller, circuit-breaker)
                                 │
5. VerificationBinder Contracts (Grounds typed DSL calls strictly against the PlanningSet)
                                 │
6. Solver Agent (Declarative planning; proposes XML trajectory packages over DSL manifest)
                                 │
7. Coder Agent (Synthesizes sandboxed Python DSL implementations from factual specs)
                                 │
8. Explorer Agent (Probes primitive actions and coordinate affordances; extracts factual rules)
                                 │
9. ARGA-Lite & PlanningSet (Deterministic perceptual vocabulary for the active cycle)
                                 │
10. Isolated Memory Stores (EnvironmentSpecMemory, SyntaxErrorMemory, EpistemicMemory, GameMemory)
```

No output generated by an LLM ever touches the environment directly. All actions pass through the `SymbolicTrajectoryExecutor`, are vetted by the `LayeredVerifier`, and are emitted strictly one step at a time through the `ActionBoundary`.

### 0.4 Dual-View Perception Contract
Under environment uncertainty, the agent projects each grid observation into two synchronized views sharing a single, immutable identity contract:

| View Channel | Data Representation | Primary Consumer | Operational Function |
| :--- | :--- | :--- | :--- |
| **Visual Channel** | Dual High-resolution PNGs (`raw_frame.png` + `annotated_frame.png`) | Explorer & Solver (Multimodal) | Spatial-geometric intuition, chiral symmetry detection, bounding box labels, and indicator dot focus markers. |
| **Symbolic Channel** | `PlanningSet` graph + hex-coded grid rows + `verifier_packet` | Coder, Verifier, Symbolic Executor | Deterministic topological relations, pixel deltas, shape signatures, sandbox execution. |

**Identity Invariant**: Both views share the exact same object IDs (`obj_0`, `obj_1`, ...), spatial relation IDs, and bounding coordinates for each observation frame.

---

## 1. Tri-Agent Separation of Powers & Complete System Prompts

To prevent context degradation, hallucination loops, and role confusion, all cognitive tasks are partitioned among three independent agent roles. Each agent possesses a dedicated, non-overlapping memory contour.

```
                   ┌─────────────────────────────────────────────────────────┐
                   │                       GameSession                       │
                   │              (State Machine & Orchestrator)             │
                   └──────┬────────────────────┬────────────────────┬────────┘
                          │                    │                    │
              Factual     │         Syntax     │        Declarative │
              Probing     ▼         Synthesis  ▼        Planning    ▼
                   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
                   │  Explorer   │      │    Coder    │      │   Solver    │
                   │    Agent    │      │    Agent    │      │    Agent    │
                   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
                          │ Writes             │ Writes             │ Writes
                          ▼ Only               ▼ Only               ▼ Only
                   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
                   │ EnvSpec     │      │ SyntaxError │      │ Epistemic   │
                   │ Memory      │      │ Memory      │      │ Memory      │
                   └─────────────┘      └─────────────┘      └─────────────┘
                          ▲                    ▲                    ▲
                          └────────────────────┴────────────────────┘
                                   Isolated: Zero Cross-Contamination
```

### 1.1 Explorer Agent (Call Family 1: Empirical Fact Discovery)
- **Primary Mission**: Systematically test available actions and coordinate affordances to determine the physical laws of the environment. Formulate empirical hypotheses regarding movement deltas, entity selection mechanics, and coordinate targeting.
- **Memory Store**: `EnvironmentSpecMemory`. Contains verified empirical facts:
  * Directional kinematics (e.g. `ACTION2: active_entity moves dy=3, dx=0`).
  * Entity selection behavior (e.g. `ACTION5: toggles active entity control`).
  * Coordinate response patterns (e.g. clicking centroid of `obj_9` triggers state change).
  * History of executed probes.
- **Strict Prohibition**: Contains **no** goal statements, **no** planning hypotheses, and **no** trajectory sequences.
- **Output**: JSON payload matching schema `v10.env_spec.1`.

#### Verbatim System Prompt (`EXPLORER_SYSTEM_PROMPT`):
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

#### Verbatim Coordinate Hypothesis System Prompt (`COORDINATE_HYPOTHESIS_SYSTEM_PROMPT`):
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

---

### 1.2 Coder Agent (Call Family 2: Sandboxed DSL Synthesis)
- **Primary Mission**: Translate the factual `EnvironmentSpecification` produced by the Explorer into a deterministic, typed Python domain-specific language (`level_dsl.py`). Expose a machine-readable function manifest with typed parameters, docstrings, and expected effect templates.
- **Memory Store**: `SyntaxErrorMemory`. Contains:
  * Prompt hash and generated Python source code.
  * Static analysis diagnostics (AST parsing errors, disallowed imports).
  * Runtime tracebacks encountered during sandbox compilation or dry-run validation.
- **Strict Prohibition & ISO-2 Quarantine**: Receives **no** level goals, **no** Solver trajectory hypotheses, **no** EpistemicMemory records, and **no** curriculum solution patterns. Through `format_empirical_context(include_curriculum=False)`, Tier 3 level-specific rules, goal invariants, and winning macros are strictly quarantined and filtered out.
- **Output**: Validated Python module source code + JSON manifest matching schema `v10.dsl_manifest.1`.

#### Verbatim System Prompt (`CODER_SYSTEM_PROMPT`):
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

#### Canonical DSL Wrapper Specification:
```python
import math
from typing import Any

def action1(api):
    """Declare discrete button action ACTION1."""
    return api.declare_environment_action(action_id="ACTION1")

def action6(api, x: int = 0, y: int = 0):
    """Declare spatial action ACTION6 at target coordinates (x, y)."""
    return api.declare_environment_action(action_id="ACTION6", data={"x": int(x), "y": int(y)})
```

---

### 1.3 Solver Agent (Call Family 3: Declarative Trajectory Planning)
- **Primary Mission**: Formulate candidate solution trajectories advancing toward the puzzle goal by calling **only** the typed functions exposed in the DSL manifest. Deduces invariants directly from empirical memory, past level curriculum, and the short-term trial scratchpad.
- **Trajectory Package Ceiling**: Generates complete multi-step trajectory packages (up to 4 candidates with 10–30 steps each) within a hard limit of 5 planning attempts per level. Solver is **never invoked per step**.
- **Memory Store**: `EpistemicMemory`. Contains:
  * Brusentsov transition judgments (FOLLOW / NULL / OMIT).
  * Permanently severed branch signatures (`severed_null_signatures`).
  * Live omit branches available for future adaptation (`live_omit_branches`).
  * Short-term scratchpad with failed action sequence summaries (`DO NOT REPEAT THIS SEQUENCE!`).
  * Actions with confirmed observed physical effects (`actions_with_observed_effects`).
- **Strict Prohibition**: Sees **no** Python source code, **no** tracebacks, **no** SyntaxErrorMemory records, and **no** internal sandbox APIs.
- **Output**: Structured XML block containing `<invariant_analysis>` and candidate trajectories `<trajectory_1>`, `<trajectory_2>`, `<trajectory_3>` of pure DSL function calls, optionally annotated with physical expectations (`EXPECT:`).

#### Verbatim System Prompt (`SOLVER_SYSTEM_PROMPT`):
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
1. What is the likely goal of this level? (Cover targets / reach position / sort / align / ...)
2. Which confirmed invariants apply here? (list from game model)
3. What is NEW or DIFFERENT about this level vs previous ones?
4. Strategy for this level:
</invariant_analysis>

<trajectory_1>
[Sequence of DSL function calls, e.g.:
 action1() EXPECT: dy=-3, dx=0
 action6(x=5, y=10)
 action2()
Optional `EXPECT: prop=val` clauses allow the Brusentsov judge to verify step consequences.]
</trajectory_1>

<trajectory_2>
[Alternative sequence of DSL function calls]
</trajectory_2>

<trajectory_3>
[Optional third sequence]
</trajectory_3>
```

#### 1.3.1 Turn 2: Epistemic Win Reflection (`distill_level_win_invariants`)
Upon solving a level, Solver is invoked in a dedicated reflection turn to distill domain-general invariants and color palette semantics:
```text
[EXECUTION OUTCOME: LEVEL WON]
Your proposed trajectory (winning_candidate) successfully solved this level and achieved victory!
Summary of executed transitions: ...
Your original hypothesis: ...

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

### 1.4 Stratified 3-Tier Cross-Level Store: GameMemory
`GameMemory` persists across level transitions within the same game. It organizes knowledge hierarchically into three explicit tiers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GameMemory Hierarchy                          │
├─────────────────────────────────────────────────────────────────────────┤
│ TIER 1: Foundational Physics, Kinematics & Boundaries                   │
│ - Confirmed action displacements: ACTION1 -> dy=-1, dx=0 (dynamic step) │
│ - Unconfirmed / Inactive action tracking & falsified status             │
├─────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Interaction Dynamics & State Transitions                        │
│ - Selection mechanics: ACTION5 round-robin actor toggle                 │
│ - Coordinate affordances: Click response patterns & button mappings     │
├─────────────────────────────────────────────────────────────────────────┤
│ TIER 3: High-Level Deduced Rules & Curriculum Invariants                │
│ - Level solution patterns: Invariant symmetries & abstract strategies   │
│ - Distilled rules from Solver Turn 2 Reflection                         │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Dynamic Empirical Falsification**:
  When empirical execution falsifies a previously confirmed rule on a pristine board ($S_0$), `invalidate_action_effect(action_id)` removes the rule from Tier 1 and triggers micro-reprobing on a clean board.
- **Confirmed Actors Property (`confirmed_actors`)**:
  Dynamically extracts all entity IDs empirically observed to undergo motion displacement or receive selection indicator focus.
- **Tier 1 Kinematics Preservation Across Levels**:
  Foundational movement physics (e.g. `displacement dy=-1, dx=0`) represent universal world laws. In `is_level_transient()`, kinematics facts containing `displacement` are explicitly protected from erasure, while transient level execution counts (e.g. `took 5 steps`) are purged.
- **Cross-Level Invariant Re-evaluation (`re_evaluate_invariants`)**:
  At each level boundary, structured invariants are re-evaluated against the new level state via Brusentsov ternary logic (`evaluate_invariant_across_levels`). Invariants that remain consistent are confirmed on the new level (`confirmed_on_levels`), while contradictory invariants are immediately falsified.
- **Strict ISO-2 Curriculum Quarantine**:
  When empirical facts are formatted for the Coder (`format_empirical_context(include_curriculum=False)`), Tier 3 level-specific rules, goal invariants (`[goal]`), and winning macros are strictly quarantined and stripped out, preventing goal leakage into the DSL generator.
- **Cross-Level Sanitization Contract (`handle_level_transition`)**:
  When transitioning between levels, `handle_level_transition()` strips transient level-local object IDs (`obj_XX` $\to$ `entity`), step offsets, and local color bindings (`(color \d+)` $\to$ generalized focus), preventing false color overfitting while preserving genuine mechanics across the entire game.

### 1.5 Formal Isolation Invariants (Normative)
| Invariant ID | Formulation |
| :--- | :--- |
| **ISO-1** | Python syntax errors, AST failures, TypeErrors, or traceback strings must **never** be injected into prompts or memory visible to the Solver Agent. |
| **ISO-2** | Level goals, success conditions, heuristic scores, or EpistemicMemory records must **never** be injected into prompts or memory visible to the Coder Agent. |
| **ISO-3** | Memory stores are strictly disjoint: Explorer writes only to `EnvironmentSpecMemory`; Coder writes only to `SyntaxErrorMemory`; Solver writes only to `EpistemicMemory`. |
| **ISO-4** | `GameSession` is the sole orchestrator authorized to read from all three memory contours and route feedback according to Double-Loop rules. |
| **ISO-5** | `PlanningSet` object, relation, and action IDs are the sole permissible identifiers; agents are strictly forbidden from inventing identifiers outside the active `PlanningSet`. |
| **ISO-6** | The Solver Agent proposes trajectories declaratively; it never executes them. All execution is handled by the independent `SymbolicTrajectoryExecutor`. |
| **ISO-7** | Zero game-specific bias: Prompts must not contain hardcoded references to specific game mechanics (e.g. quadrant cloning, avatar names, key/door assumptions). All invariants must be inferred dynamically from the empirical memory block. |
| **ISO-8** | Zero color-binding leakage across levels: Selection mechanics must not bind actor-toggle operations to local color artifacts from prior levels. |

---

## 2. Independent Symbolic Trajectory Executor & Multi-Trajectory Traversal

The **`SymbolicTrajectoryExecutor`** operates as an autonomous, deterministic symbolic controller outside the LLM context. It decouples high-level declarative trajectory packages from physical execution.

```
  Solver Agent (Qwen)
  Emits: TrajectoryPackage with Candidates [C1, C2, C3]
         │
         ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                      SymbolicTrajectoryExecutor                        │
  │                                                                        │
  │  1. Pre-Verification:                                                  │
  │     - Check if candidate / step signature is severed in EpistemicMemory│
  │     - Verify DSL function exists in SandboxedModule namespace          │
  │     - Dynamic Signature Binding (inspect parameters, filter kwargs)    │
  │     - Ground arguments against active PlanningSet                      │
  │                                                                        │
  │  2. Sandboxed Execution:                                               │
  │     - Execute function inside SandboxExecutor                          │
  │     - If exception: trip Circuit Breaker, sever candidate, emit RESET  │
  │     - Obtain EffectDeclaration (pure action intent)                    │
  │                                                                        │
  │  3. Transition Evaluation (Post-Step via LayeredVerifier):             │
  │     - TRUE  (FOLLOW)  --> Advance candidate cursor                     │
  │     - IRRELEVANT (OMIT) --> Advance cursor, record omit branch         │
  │     - FALSE (NULL)    --> Sever candidate, queue clean RESET           │
  │                                                                        │
  │  4. Multi-Trajectory Candidate Pool Traversal:                         │
  │     - If Candidate finishes without WIN:                               │
  │         * Record sequence in Epistemic Memory Scratchpad               │
  │         * Advance to Next Candidate in active_pool                     │
  │         * If Next Candidate exists:                                    │
  │             replan_needed = False (DO NOT call Solver!)                │
  │             reset_needed  = True  (Emit RESET to restore clean S0)     │
  │         * If All Candidates exhausted:                                 │
  │             replan_needed = True  (Call Solver for fresh attempt)      │
  │             reset_needed  = True  (Emit RESET to clean S0)             │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │ Emits strictly verified step
                                     ▼
                                ActionBoundary ───► env.step(action)
```

### 2.1 Dynamic Argument Binding & Kwargs Filtering
Different Coder implementations emit functions with differing parameter signatures:
- Discrete button channel: `def action1(api): ...`
- Coordinate channel: `def action6(api, x=0, y=0): ...`

The `SymbolicTrajectoryExecutor` inspects function signatures via `inspect.signature(func).parameters`. If extraneous arguments (such as hallucinated object names) are present in the Solver's candidate step, they are filtered out dynamically. If a required parameter cannot be grounded against the `PlanningSet`, the step is rejected *before* execution.

### 2.2 Circuit Breaker & Immediate Reset
If sandbox execution encounters an unhandled runtime exception or pre-verification failure:
1. The active candidate trajectory is permanently marked inactive (`candidate.sever()`).
2. The sequence signature is recorded in `EpistemicMemory.severed_null_signatures`.
3. The exception details are written exclusively to `SyntaxErrorMemory`.
4. The executor checks if another candidate exists in `active_pool`:
   - If another candidate exists: `replan_requested = False` (the next candidate will execute after reset).
   - If no candidates remain: `replan_requested = True` (Solver will replan).
5. An immediate `RESET` action is emitted through `ActionBoundary` to restore the board to clean initial state $S_0$ without executing arbitrary fallback actions.

### 2.3 Dynamic Kinematics & Dual-Signal Falsification Architecture
In ARC-AGI-3, action semantics can change between levels or when switching active actors. The architecture implements a dual-signal verification and reprobing loop:

1. **Pristine Board Falsification (Cursor == 0)**:
   - When a candidate's initial step (`cursor == 0`) executes a confirmed motion action (`is_confirmed_motion`) but produces `before_grid == after_grid` (`zero_grid_delta`), the baseline kinematic law is empirically falsified.
   - `SymbolicTrajectoryExecutor` sets `falsification_detected = True`.
   - `GameSession` invalidates the action in `GameMemory` via `invalidate_action_effect(act_id)`, removes it from `known_actions`, resets the board to $S_0$, and invokes `schedule_falsification_reprobe()` to retest baseline motion primitives.
2. **Mid-Trajectory Motion Contradiction (Cursor > 0)**:
   - If a confirmed motion action produces `zero_grid_delta` mid-trajectory, `LayeredVerifier` returns `Ternary.FALSE` (Brusentsov nullity $xy'_0$: wall collision, boundary barrier, or inactive actor).
   - The candidate is severed immediately (`candidate.sever()`), preventing the execution of dozens of wasted phantom steps. A clean `RESET` is queued to restore state for the next candidate.
3. **Dynamic Modal Toggle Reprobing**:
   - When an action triggers an entity selection toggle (e.g. `ACTION5`), `get_dynamic_reprobes()` queues immediate re-probing of primitive motion actions (`ACTION1..ACTION4`) to discover the affordances of the newly selected entity.
4. **Universal Clean RESET ($N \ge 0$)**:
   - Clean state resets via `RESET` are enforced across all levels (Levels 0, 1, 2, ...), ensuring that every candidate trajectory begins from an uncontaminated, pristine board.

### 2.4 Virtual Sandbox Role & Non-Override Contract
The `VirtualSandbox` in `v10_agent/virtual_sandbox.py` operates strictly as an offline forward kinematic simulator and candidate evaluator/repairer. Its operational boundaries are formally defined:
1. **Kinematic Safety Verification and Step Repair**:
   - Evaluates trajectory packages proposed by the declarative LLM Solver.
   - Simulates physical step displacements ($dy, dx$) using confirmed Tier 1 physics.
   - Truncates candidates cleanly upon detecting wall collisions, boundary violations, or impossible steps, filtering out fatal steps before physical execution.
2. **Strict Non-Override Invariant (No Index 0 Hijacking)**:
   - The Virtual Sandbox **must never** synthesize a programmatic fallback script and inject it ahead of the LLM Solver's candidates (e.g. inserting at index 0 via `candidates.insert(0, cand_synth)`). Doing so silences the LLM's topological and geometric reasoning.
   - Programmatic heuristic synthesis by the Virtual Sandbox acts **strictly as an offline fallback** when the LLM Solver returns zero valid candidate trajectories (`if not candidates_to_use:`).
3. **Planning Attempt Ceiling and No Per-Step Invocations**:
   - Trajectory planning is batched into full multi-step trajectory packages (up to 4 candidates per package, 10–30 steps each) within a hard ceiling of 5 planning attempts per level (`max_chain_attempts_per_level = 5`), yielding ~10–20 total candidate trajectories per level.
   - The Solver is **never invoked per step**. Per-step execution is handled deterministically by the `SymbolicTrajectoryExecutor` and validated step-by-step by the `LayeredVerifier`.

---

## 3. Brusentsov 3-Valued Logic in Transition Verification

The core evaluation engine of the agent is the `LayeredVerifier`, governed by **Brusentsov's 3-valued logic**.

### 3.1 Foundations: Why Classical Boolean Logic Fails
Classical Boolean logic operates with two truth values: $\{0, 1\}$ (or $\{F, T\}$). Its standard conditional is **material implication**:
$$x \to y \equiv \neg x \lor y$$

In verification of autonomous agent trajectories, material implication suffers from the **paradox of vacuous truth**: if the precondition $x$ is false (e.g. an action fails to execute or an entity was not selected), the implication $\neg x \lor y$ evaluates to **TRUE**. In an agentic loop, this causes the verifier to approve non-functional, broken, or hallucinated steps as "valid"!

### 3.2 Nikolai P. Brusentsov and Necessary Implication
In the development of the ternary computer *Setun* (Moscow State University, 1958) and his formalization of Aristotelian syllogistics, Nikolai Petrovich Brusentsov established that genuine empirical reasoning requires **three truth values**:
- $+1$ (True / Affirmation)
- $0$ (Neutral / Omission / Irrelevance)
- $-1$ (False / Incompatibility / Contradiction)

Brusentsov defined **necessary implication** ($x \Rightarrow y$), which asserts that the consequence $y$ is *necessarily contained* in the condition $x$:
$$x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$$

In this ternary disjunctive normal form:
1. **$xy$ (Affirmation / FOLLOW, $+1$)**: The action occurred, and its expected physical effect is necessarily observed in the transition ($x=1, y=1$).
2. **$xy'_0$ (Neutrality / OMIT, $0$)**: The action occurred, but the specific expected effect has not materialized yet ($y'_0$ denotes non-presence without contradiction). No world laws were broken, no walls were hit, and object identities remain valid. The step is benign / passive.
3. **$x'y'$ (Contraposition / NULL, $-1$)**: A strict contradiction. The expected progress was violated by an impossible state ($x'$) and a contradicting outcome ($y'$), such as a wall collision, 0 displacement on a movement command, an increase in distance to the target socket, or an illegal shape deformation.
4. **$x'y$ (Excluded Term)**: Classical material implication includes the term $x'y$ (the condition did not occur, yet the consequence did). Brusentsov's logic **strictly omits** this term as inessential and invalid for empirical causality.

### 3.3 The Three Truth Values in Agent Operation

```
                             Transition Evaluation
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     Brusentsov TRUE           Brusentsov IRRELEVANT       Brusentsov FALSE
       (Value: +1)                  (Value: 0)               (Value: -1)
      Verdict: FOLLOW              Verdict: OMIT            Verdict: NULL
            │                          │                          │
   ┌────────┴────────┐        ┌────────┴────────┐        ┌────────┴────────┐
   │ Physical        │        │ Non-contradict- │        │ Collision,      │
   │ progress toward │        │ ing transition  │        │ zero delta,     │
   │ target achieved.│        │ (e.g. toggle    │        │ divergence from │
   │ Advance cursor. │        │ entity control).│        │ target socket.  │
   │ Extend branch.  │        │ Advance cursor. │        │ Sever branch.   │
   └─────────────────┘        │ Keep branch live│        │ Queue RESET.    │
                              │ in memory.      │        │ Next candidate. │
                              └─────────────────┘        └─────────────────┘
```

| Brusentsov State | Truth Value | Operational Verdict | Ground-Truth Verification Criteria | Action / Memory Effect |
| :--- | :---: | :---: | :--- | :--- |
| **TRUE** | $+1$ | **FOLLOW** | - The expected entity was displaced in the specified direction.<br>- State changed in-place (color, toggle, orientation).<br>- Distance to target decreased or invariant maintained. | Candidate cursor advances ($c \leftarrow c + 1$). If candidate finishes without win, queues clean `RESET` and advances to next candidate. |
| **IRRELEVANT** | $0$ | **OMIT** | - Action did not displace primary target, but produced a legitimate state change (e.g. toggling active entity control).<br>- No boundary collision, no goal divergence, no invariant violation. | Candidate cursor advances ($c \leftarrow c + 1$). Signature recorded as live omit branch in `EpistemicMemory`. |
| **FALSE** | $-1$ | **NULL** | - Movement action produced 0 displacement (boundary collision / wall hit).<br>- Contradiction with confirmed physics.<br>- Target entity was destroyed or mutated into an invalid shape. | **Branch is severed immediately**. Sequence recorded in `severed_null_signatures`. Queues clean `RESET` to $S_0$. Advances to next pool candidate. |

### 3.4 Active Consequence Checking via `implies_brusentsov`
When Solver candidate steps declare explicit expectations (e.g. `action1() EXPECT: dy=-1, dx=0`), `LayeredVerifier` in `v10_agent/judge.py` evaluates the step transition using `implies_brusentsov(step.expected_propositions, observed_propositions)` from `v10_agent/brusentsov_logic.py`:

1. **Incompatibility / Nullity Check ($\to \text{FALSE}$)**:
   - For every expected atomic proposition $e \in \text{expected}$, the verifier checks if any observed proposition $o \in \text{observed}$ contradicts $e$ (`contradicts(e, o)`).
   - If an object identity was expected to be preserved (`family="object_identity"`, `predicate="preserved"`) but is observed as `destroyed`, `missing`, or `vanished`, a contradiction occurs.
   - Upon contradiction, `implies_brusentsov` returns `Ternary.FALSE` (Brusentsov nullity $xy'_0$). The candidate trajectory is severed immediately and a clean `RESET` is queued.
2. **Necessary Containment Check ($\to \text{TRUE}$)**:
   - If every expected atomic proposition is necessarily contained in the observed transition set (`is_necessarily_contained(e, observed)`), `implies_brusentsov` returns `Ternary.TRUE` (Brusentsov affirmation $xy$).
   - The candidate step is verified as successful; the candidate cursor advances to the next step.
3. **Inessential Missing Consequence Check ($\to \text{IRRELEVANT}$)**:
   - If the expected consequence is not observed, but no physical laws or boundary constraints were violated, `implies_brusentsov` returns `Ternary.IRRELEVANT` (Brusentsov neutrality $xy'_0$ / omit branch).
   - The step is treated as non-fatal: the cursor advances, and the state transition signature is preserved in `EpistemicMemory.live_omit_branches` for potential structural adaptation.

---

## 4. Perception Engine: ARGA-Lite & Substantive Filtering Contract

Perception in Version 10.0 is deterministic, fast ($< 2$ ms per frame), and completely independent of neural networks. Implemented in `v10_agent/arga_lite.py`.

### 4.1 Segmentation and Topological Analysis
1. **Background Detection (`detect_background_color`)**:
   - Inspects the four borders of the grid. If a single color occupies $\ge 40\%$ of the border, it is identified as the background.
   - If border counts are inconclusive, checks global color frequency.
2. **Connected Components Segmentation**:
   - Applies 4-connectivity flood-fill to group all non-background pixels of identical color into discrete `PlanningObject` entities.
3. **Geometric & Topological Descriptors**:
   - Bounding box (`min_row`, `min_col`, `max_row`, `max_col`, `height`, `width`).
   - Centroid coordinates $(\bar{r}, \bar{c})$.
   - Compact ASCII representation for substantive objects ($height, width \le 25$).
   - Dual shape signatures:
     * `shape_signature`: MD5 hash of raw binary mask.
     * `filled_shape_signature`: MD5 hash of topologically filled mask (using exterior flood-fill). Identifies whether an object contains internal cavities.
4. **Spatial and Topological Relations**:
   - Adjacency (`touches`), Alignment (`aligned_h`, `aligned_v`), Containment (`contains`), Euclidean centroid distance (`distance`).
   - Symmetries: `identical_shape`, `chiral_mirror_h` (horizontal flip congruence), `chiral_mirror_v` (vertical flip congruence).
   - Axis symmetries: `symmetric_axis_of`, `mirrored_across_axis`.

### 4.2 The Substantive Relations Contract (The Noise/Cavity Filter)
Directional navigation relations (`target_is_below`, `target_is_above`, `target_is_to_the_left`, `target_is_to_the_right`) are **strictly prohibited** from including any object that has $area < 4$ or $color == 0$. They are computed and presented to the Solver **only** between substantive entities ($area \ge 4$, $color \ne 0$), eliminating the "1-pixel cavity trap".

### 4.3 Spatial Bounds and Freedom of Motion Grounding
The game space is strictly bounded by `grid_bounds` ($height \times width$). Using confirmed step sizes in pixels ($step\_size\_pixels$), the perception engine computes explicit **freedom of motion budgets** for every object:
$$\text{up\_max\_steps} = \lfloor \text{min\_row} / \text{step\_size\_pixels} \rfloor$$
$$\text{down\_max\_steps} = \lfloor \max(0, (\text{height} - 1) - \text{max\_row}) / \text{step\_size\_pixels} \rfloor$$
$$\text{left\_max\_steps} = \lfloor \text{min\_col} / \text{step\_size\_pixels} \rfloor$$
$$\text{right\_max\_steps} = \lfloor \max(0, (\text{width} - 1) - \text{max\_col}) / \text{step\_size\_pixels} \rfloor$$

### 4.4 Universal Invariant Engine (`v10_agent/universal_invariants.py`)
To discover higher-order mathematical structures without game-specific heuristics, the agent analyzes the `PlanningSet` for universal symmetries:
1. **Collinear Axis Discovery**: Detects horizontal and vertical mirror axes (`axial_symmetry_horizontal`, `axial_symmetry_vertical`) spanning the grid.
2. **Symmetric Entity Mapping**: Pairs symmetric sprites across the discovered axis and calculates exact step offsets:
   - `axis_steps`: Number of discrete steps required to translate the mirror axis to the centroid of the target sockets.
   - `piece_steps`: Number of discrete steps required to translate the active piece into its target socket.
3. **Integration with Perception & Virtual Sandbox**: Invariants are fed directly to `GameMemory` and `VirtualSandbox` to guide forward simulation and A* fallback trajectories.

### 4.5 Visual Salience & Indicator Marker Aggregation
To prevent perceptual context fragmentation from large clusters of indicator dots:
1. **Marker Aggregation**: Small dots ($area \le 2$) sharing the same color are aggregated into composite marker groups (`marker_group_color_{c}` / `marker_dots_c{c}`).
2. **Quadrant-Balanced Salience**: Objects are partitioned across the 4 grid quadrants (TL, TR, BL, BR). Up to 10 substantive objects per quadrant and all confirmed actors are prioritized, guaranteeing spatial diversity and preventing attention collapse on local clusters.

---

## 5. PlanningSet Identity Contract & Dynamic Probing Pipelines

### 5.1 The PlanningSet Contract
For every observation cycle, `PlanningSet` forms an immutable, canonical vocabulary:
```
PlanningSet
├── snapshot_id: str
├── grid_hash: str
├── grid_dims: (height, width)
├── allowed_action_ids: tuple[str, ...]
├── allowed_coordinate_candidate_ids: tuple[str, ...]
├── objects: tuple[PlanningObject, ...]
├── relations: tuple[SpatialRelation, ...]
└── coordinate_candidates: tuple[CoordinateCandidate, ...]
```
Every DSL function call, argument, proposition, and verifier check must ground strictly within these identifiers.

### 5.2 Dynamic Probing Pipelines & Unified Pipeline Classification
At the beginning of each level, `GameSession` inspects the available action surface and classifies the environment into one of four operational pipelines (`self.active_pipeline`):
- **`hybrid`**: Coexistence of spatial coordinate actions (`has_coords`, e.g. `ACTION6`) and discrete motion/operation buttons (`has_discrete`, e.g. `ACTION1..ACTION5`). Rather than imposing artificial mode separation, both discrete probes and coordinate candidate probes are scheduled directly into `probe_queue`.
- **`discrete`**: Environments governed exclusively by discrete buttons (`ACTION1..ACTION5`). Probing tests kinematic displacements and toggle operations.
- **`coordinate`**: Environments where interaction occurs strictly through spatial clicking (`ACTION6`). Probing targets salient object centroids and boundaries.
- **`dynamic`**: Environments featuring conditionally surfaced or dynamic action names.

The structured probing sequence proceeds as follows:
1. **Discrete Probing Sweep**:
   - Tests unconfirmed discrete actions (`ACTION1` through `ACTION5`).
   - If `probe_reset_after_discrete` is enabled, executes `RESET` after each probe to return the board to clean state $S_0$.
   - Deduces kinematic deltas ($dy, dx$), entity selection mechanisms, and toggle invariants, storing confirmed rules in `GameMemory`.
2. **Targeted Coordinate Probing**:
   - If `ACTION6` (coordinate click) is available, uses `COORDINATE_HYPOTHESIS_SYSTEM_PROMPT` to propose 2-4 candidate click targets based on visual object centroids and boundaries.
3. **Dynamic Action Surface Detection**:
   - Detects dynamic action surfaces (e.g. actions whose names change or appear conditionally) and probes newly surfaced capabilities immediately.
4. **Universal Post-Probe Clean Reset ($N \ge 0$)**:
   - After probing completes on any level, a clean `RESET` is executed before any Solver candidate begins execution.

---

## 6. Orchestration Invariants & State Machine

The top-level orchestrator is `GameSession` in [`v10_agent/session.py`](file:///c:/arcprize/v10_agent/session.py). It enforces seven critical orchestration invariants:

### 6.1 Multi-Trajectory Candidate Pool Sequential Traversal via RESET & 5-Attempt Ceiling
The Solver is invoked to plan full multi-step trajectories with a strict ceiling of **5 planning attempts per level** (`max_chain_attempts_per_level = 5`). Under no circumstances is the Solver invoked per step.
Each Solver invocation produces a package of 2–4 candidate trajectories (10–30 steps each), yielding a pool of ~10–20 candidate trajectories across the entire level budget.

When the Solver proposes multiple candidate trajectories (e.g. Candidates 1, 2, 3 in `TrajectoryPool`):
1. Candidate 1 executes step by step from clean initial state $S_0$.
2. If Candidate 1 completes its steps without winning (or is severed by contradiction):
   - Epistemic Memory records the exact failed sequence in `current_level_scratchpad` with `DO NOT REPEAT THIS SEQUENCE!`.
   - `SymbolicTrajectoryExecutor` advances to Candidate 2.
   - **`replan_needed` is set to `False`** (the Solver is **NOT** invoked!).
   - **`reset_needed` is set to `True`** (a `RESET` action is queued).
3. The environment executes `RESET`, restoring the board to pristine starting state $S_0$.
4. Candidate 2 executes from its step 0 on the clean board.
5. **Only when all candidates in the pool are exhausted** without winning is `replan_needed` set to `True`, triggering a new Solver planning attempt (if within the 5-attempt ceiling).

### 6.2 No Dirty-Board Cascading Invariant
Candidate trajectories are planned assuming pristine initial state $S_0$. Executing a candidate on a partially modified or corrupted board is strictly forbidden. Any candidate transition to a subsequent candidate or replan requires a preceding `RESET`.

### 6.3 Clean Reset on Contradiction Invariant
When the `LayeredVerifier` issues a `Ternary.FALSE` (NULL) verdict:
1. The active candidate is immediately severed.
2. The sequence signature is recorded in `EpistemicMemory.severed_null_signatures`.
3. A `RESET` action is emitted through `ActionBoundary`.
4. If another candidate exists in `active_pool`, it is executed after reset; otherwise, Solver replans.

### 6.4 Tufa Single-Reset Invariant on GAME_OVER
Under the official competition harness, when state becomes `GAME_OVER`:
- Exactly **one** `RESET` action must be emitted.
- If `GAME_OVER` persists after that single `RESET`, the session immediately breaks the loop to prevent infinite retry loops and disqualification.

### 6.5 Level Win Invariant
When observation reports `levels_completed > levels_completed_observed` or state becomes `WIN`:
- `GameSession` invokes `handle_level_transition`.
- Epistemic judgments and environment specs are summarized into `GameMemory`.
- Level-specific pools and caches are reset.
- Action counters for the level are cleared.

### 6.6 Solver Two-Turn Reflection Invariant
When a level is successfully completed:
1. `GameSession` does not blindly carry forward raw button sequences into future level prompts (which would cause button-sequence pollution).
2. It triggers Turn 2 reflection via `solver.distill_level_win_invariants()`, prompting the Solver to extract high-level domain invariants, geometric symmetries, and abstract strategies.
3. These distilled insights are recorded into `GameMemory.record_level_solution()` and rendered into `curriculum_progression_from_won_levels` across subsequent levels.

### 6.7 Dynamic Action Surface & Clean State Invariant
When actions appear dynamically (e.g. conditional affordances or modal triggers):
1. Newly surfaced actions are identified and probed immediately.
2. If an action toggles active control (e.g. `ACTION5`), dynamic reprobing tests directional movement for the newly selected entity.
3. Universal post-probe clean `RESET` ensures that all Solver candidate executions begin from a clean, pristine board state ($S_0$).

---

## 7. Double-Loop Learning & Error Routing

Error handling is partitioned strictly by domain:

```
                           Error Detected
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   Sandbox Execution Error                  Physical / Logical Outcome
 (Syntax, TypeError, AST, Limits)               (NULL or OMIT Verdict)
            │                                         │
            ▼                                         ▼
      EXTERNAL LOOP                             INTERNAL LOOP
            │                                         │
  Writes to SyntaxErrorMemory               Writes to EpistemicMemory
  Invokes Coder for DSL repair              Pivots Solver trajectory
  Solver context untouched                  Coder context untouched
```

- **External Loop (Syntax & Sandbox Routing)**:
  When generated DSL code fails static checks, raises a `TypeError`, or exceeds sandbox limits, diagnostics are routed **exclusively** to `SyntaxErrorMemory`. The Coder is re-invoked (up to `max_coder_retries`). The Solver is completely shielded from syntax tracebacks.
- **Internal Loop (Logical & Empirical Routing)**:
  When DSL code executes cleanly but the step fails to advance the puzzle (NULL or OMIT), the judgment is routed **exclusively** to `EpistemicMemory`. The Coder is not penalized or re-invoked. The Solver receives the judgment in its prompt and adapts its next trajectory package.

---

## 8. Budgets, Safety Caps & Fallback Architecture

### 8.1 Hard Ceilings & Level Budgets
- `max_chain_attempts_per_level`: 5 attempts (unified chain budget per level).
- `max_coder_retries_per_level`: 5 attempts.
- `max_solver_retries_per_level`: 5 attempts.
- `max_explorer_attempts_per_level`: 5 attempts.
- `max_explorer_probe_actions_per_level`: 30 actions.
- `max_candidates_per_solver_package`: 4 candidates.
- `max_steps_per_candidate`: 20 steps.
- `max_actions_per_level`: 500 actions.
- `max_actions_per_game`: 500 actions.
- `level_wall_clock_limit_seconds`: 1200.0s (20 minutes soft level budget).
- `execute_one_step_at_a_time`: True.

### 8.2 Mandatory Symbolic Fallback
If Coder retries are exhausted without a valid DSL, or if Solver packages fail to make progress under the action budget:
1. `GameSession` activates `fallback_symbolic.py`.
2. Employs deterministic heuristic operators:
   - Centroid-directed delta steps (`move_toward`).
   - Coordinate targeting of unvisited socket centroids.
   - Primitive cycle stepping with collision detection.
   - Forward kinematic simulation via `VirtualSandbox` and A* shortest path search.
3. All fallback steps remain strictly governed by the `LayeredVerifier` and Brusentsov logic.

### 8.3 Action Space Invariants & Exclusion of ACTION7 (Undo)
Under ARC-AGI-3 competition environment conditions, `ACTION7` is fixed as an `Undo` operation. The V10 architecture fundamentally does not use `ACTION7`:
1. **Epistemic State Consistency**: Rather than attempting reverse-step backtracking via `Undo` (which creates ambiguous intermediate states, complicates invariant tracking, and risks infinite oscillatory loops), the agent relies on clean-slate restarts ($S_0$) via environment `RESET`.
2. **Deterministic Forward Traversal**: All candidate trajectory execution is strictly forward-directed. `ACTION7` is deliberately filtered out from `PlanningSet.allowed_action_ids`, primitive probing, and code generation prompts to enforce unambiguous trajectory evaluation.

---

## 9. Packaging, Kaggle Deployment & Auditability

### 9.1 Self-Extracting LZMA Packaging (`build_notebook_v10.py`)
To comply with the Kaggle submission limit (< 985 KB):
- The entire `v10_agent` package (27 core modules + 4 prompt builder modules) is compressed into an in-memory LZMA archive.
- The compressed payload is encoded as base64 text and embedded in a single competition notebook cell.
- A lightweight bootstrap unpacks the code into the Kaggle runtime environment upon notebook execution.
- **Current Total Size**: ~142 KB (uses only **14.4%** of the 985 KB ceiling).

### 9.2 Audit Log Contract
Every action emitted to the competition gateway is auditable through a complete structured trace:
$$\text{Grid Hash} \longrightarrow \text{PlanningSet} \longrightarrow \text{Solver Candidate} \longrightarrow \text{Symbolic Executor} \longrightarrow \text{Judge Verdict} \longrightarrow \text{ActionBoundary}$$
All traces are written to `local_harness_run.json` during evaluation runs.

### 9.3 Test Suite & Quality Assurance
The V10 codebase is verified by a suite of **191 unit tests** across 38 test modules, covering:
- ARGA-Lite perception, 2D descriptors, cavity detection, and dual-view image generation.
- Tri-agent separation of powers, prompt isolation, and syntax routing.
- Brusentsov ternary truth evaluation and active consequence checking via `implies_brusentsov`.
- Dynamic kinematics, falsification detection, micro-reprobing, and universal clean reset.
- Stratified 3-tier memory contours, Tier 1 kinematics protection, confirmed actor tracking, and cross-level invariant re-evaluation.
- Virtual sandbox polymorphic simulation, step repair without index 0 hijacking, and A* fallback planning.
- Unified hybrid environment execution without per-step Solver invocations.
