# ARC-AGI-3 LCLD Agent
# Architectural Specification — Version 10.5
# (Neuro-Symbolic Tri-Agent Architecture, Brusentsov Entailment Logic, Null Severance & Epistemic Recovery, Two-Tier Action Lifecycle, Reactive DSL Invalidation, Universal Multimodal Dual-View, 5-Block Stratified Memory & Production vLLM Runtime)

---

## 0. Purpose and Architectural Vision

This document defines the authoritative architecture of the **ARC-AGI-3 LCLD (Locally Constrained Learning and Discovery) Agent (Version 10.5)**. The agent is engineered to autonomously solve diverse, completely unseen ARC-AGI-3 interactive grid tasks under official Kaggle competition runtime constraints:

1. **Unknown Mechanics & Hidden Rules**: No pre-programmed rules, object identities, goal definitions, or physics models are provided. Every environment must be empirically identified, modeled, and solved entirely online.
2. **Standard 16-Color Palette ($0..15$) & Semantic Role Mapping**: ARC-AGI-3 operates over a 16-color palette (integers $0..15$). The agent maintains a persistent `PaletteRoleMap` tracking functional roles (`EntityRole`) for each color index, completely replacing destructive `[COLOR]` erasure with role-aware semantic labeling (`Color N (ROLE)`).
3. **Partitioned Action Space & Hardware Constraints**: Environments feature discrete directional/toggle actions (`ACTION1..5`) and spatial coordinate clicks (`ACTION6(x, y)`). Step-reversal (`ACTION7` / `Undo`) is **hardware-blocked** by architectural contract. Epistemic consistency is maintained strictly through forward deterministic planning and clean state resets ($S_0$).
4. **Dual-Space Coordinate Synchronization & 1px Boundary Crop**: Raw environment frames are cropped by 1 border pixel on all 4 sides ($[1:-1, 1:-1]$), yielding a local perception grid $(H-2) \times (W-2)$. Coordinates are strictly maintained in a dual-space representation (local ARGA-Lite coordinates vs. engine coordinates via `crop_offset = 1`), guaranteeing boundary precision and tested-coordinate deduplication.
5. **Brusentsov 4-Valued Logic of Entailment & Carrollian Nullity Completeness**: Foundational epistemic engine based on N.P. Brusentsov’s non-paradoxical logic of entailment ($xy \lor xy'_0 \lor x'y'$), distinguishing necessary incompatibility (Carroll's nullity $xy'_0 \to \text{NULL}$) from benign inessentiality (omission/silence $\to \text{OMIT}$), with strict elimination of material implication paradoxes ($len(expected)=0 \to \text{IRRELEVANT}$, vacuous `FOLLOW` prevention). Incompatibility ($xy'_0$) covers physical stagnation (expected motion, observed stationary), unintended mutation (expected invariant, observed motion), direction inversion (opposite kinematic signs), and object identity violation (expected preserved, observed destroyed/missing).
6. **5 Stratified Memory Blocks**: Memory is partitioned into five specialized, non-interfering stores: `PaletteRoleMap`, `CoreInvariantRegistry` (`GroundedInvariant`), `EpisodicExemplarBuffer` (`DefeatExemplar` & `VictoryExemplar`), `CurriculumProgressionBuffer` (`curriculum_history`), and `WorkingRolloutMemory`.
7. **Empirically Grounded Exemplars**:
   - `DefeatExemplar`: Generated upon `GAME_OVER` / lethal failure; grounds negative barrier invariants ($xy'_0 \to \text{NULL}$) to prevent recurring mistakes.
   - `VictoryExemplar`: Generated upon level victory; grounds positive canonical exemplars ($xy \to \text{FOLLOW}$) for cross-level transfer.
8. **Symbolic Trajectory Execution with Early Null Severance & Epistemic Recovery**: Macro-step commands (`count=N`) are unrolled into $N$ atomic steps by `_decompose_expected_propositions_for_substep`. State invariants (`unchanged`, `preserved`) are distributed across all intermediate steps, single-step motion signs (`step_moved`) are verified at each transition, and cumulative vectors are checked on the terminal step. Any physical contradiction ($xy'_0 \to \text{NULL}$) causes immediate candidate branch severance (`sever()`), early execution termination, structured diagnostic synthesis (`[FAILED ATTEMPT N DIAGNOSTIC]`), and a clean environmental reset ($S_0$) without dirty-state cascading.
9. **Two-Tier Action Lifecycle & Action Affordance Completeness**: Primitive probe management distinguishes immediately effective actions from conditional trigger actions. Actions yielding $\Delta = 0$ on the pristine initial frame $S_0$ are strictly preserved in `available_actions` and typed as `CONDITIONAL_TRIGGER` affordances (`status: unconfirmed_on_s0`) rather than being discarded, ensuring the Coder synthesizes primitives for deferred or modal game mechanics.
10. **Reactive DSL Invalidation & Auto-Augmentation**: When any newly confirmed action (not in the active manifest) is observed during execution, `GameSession` reactively invalidates `active_module` and `active_manifest`, unblocks Coder retry ceilings, and requests a clean replan. The Coder auto-augments output with canonical sandboxed wrappers for any omitted confirmed actions.
11. **Universal Multimodal Dual-View & Coordinate Disambiguation**: Perception provides synchronized DualView imagery (raw pixel frame + annotated frame with object bounding boxes and aliases) across all three agents, mitigating parser over-segmentation of multi-color composite structures. Grid coordinates follow strict Cartesian $(Row, Col) \leftrightarrow (Y, X)$ disambiguation.
12. **Tri-Agent Authority Segregation**: Cognition is strictly divided among three distinct agent roles (**Explorer**, **Coder**, and **Solver**) operating through dedicated, non-overlapping memory contours guarded by ISO-1 through ISO-12 invariants.
13. **Multi-Candidate Package Synthesis & Clean RESET Traversal**: The Solver plans complete multi-step trajectory packages (up to 4 candidate paths per package) upfront, under a hard ceiling of 5 chain attempts per level (~10–20 trajectories total). Step-by-step LLM querying is **fundamentally prohibited**. Unsuccessful trajectories are severed, and subsequent candidates are tested from clean starting states ($S_0$) via environmental `RESET`.
14. **Clean Physics Simulator & Topological Invariants**: Aspect ratio heuristics ($\ge 3.0$) replace puzzle-specific constants. Domain-general topological invariants (`ConnectedComponentConservation`, `GravitySettling`, `ContactTrigger`, `AreaConservation`) model world physics without geometric hardcoding.
15. **Persistent Multi-Frame Object Tracking**: Object permanence across frames is maintained by `PersistentObjectTracker` via 4-part bipartite matching (color, centroid distance, shape Jaccard, relative area), velocity EMA, and confidence tracking.
16. **Safe Evidence-Seeking Loop (E-FSM)**: Epistemic uncertainty (`Verdict.UNDECIDED`) schedules targeted empirical probes without premature candidate branch severing.
17. **VisibleCycle Loop Recovery**: Orbit detection for periods 1..8 detects repeating state-action cycles before burning the level action budget, severs the loop, and triggers clean replanning.
18. **Production Time Budgeting & Resilient vLLM Serving**: Absolute deadline tracking with 15-second reserves prevents timeouts; background watchdog and sub-second socket-based teardown ensure serving stability.
19. **Strict Offline Competition Constraints**: Kaggle evaluation environment with zero internet access, maximum artifact notebook size under 985 KB, fixed GPU resources (NVIDIA H100 / RTX 6000 Ada), 12-hour total competition wall-clock limit, and a maximum of 5000 seconds per game. Local LLM (Qwen 3.8 27B) served via vLLM with FlashAttention-2 and MTP=3 speculative decoding.
20. **Automated Quality & Purity Assurance Stack**: AST Code Guardian (`tools/ast_code_guardian.py`), Property-Based Testing with Hypothesis (`test_pbt_*.py`), and Synthetic Calibration Micro-Worlds (`test_synthetic_calibration_worlds.py`).

---

### 0.1 What the Architecture Is NOT

- **NOT a monolithic LLM game player**: The LLM is never prompted with "What action do you want to take next?" on every frame. Such approaches suffer catastrophic hallucination, high latency, context explosion, and rapid budget depletion.
- **NOT a step-by-step solver**: The Solver plans complete multi-step trajectories upfront. The deterministic symbolic executor carries out execution step-by-step, verifying world laws at every transition.
- **NOT an unconstrained code executor**: The Coder synthesizes Python DSL implementations that are verified by an AST whitelist parser and executed inside a restricted memory/CPU sandbox. Arbitrary Python execution is blocked.
- **NOT a reinforcement learning policy**: The agent requires zero training frames, weight updates, or offline learning loops.
- **NOT hardcoded for specific games**: No heuristics tailored to specific benchmark puzzles exist. All invariant rules are deduced online from empirical observations.
- **NOT dependent on `ACTION7`**: By competition rules, `ACTION7` is a step-reversal (`Undo`). The agent hardware-blocks `ACTION7`, maintaining epistemic consistency strictly through forward deterministic planning and clean state resets ($S_0$).

---

### 0.2 Anti-Specification Gaming Contract

The codebase is governed by a non-negotiable contract of abstract purity enforced by continuous AST validation:

1. **Categorically Forbidden**:
   - Creating conditions, variables, comments, or branches containing benchmark puzzle IDs (`ar25`, `ft09`, `ls20`, `re86`, `rs01`, etc.).
   - Hardcoding grid dimensions or object dimensions (e.g. `width <= 4`, `height >= 10`, `grid_size == 62`, `10x4`), except for reading dynamic parameters from the perception grid or `PlanningSet`.
   - Using regular expressions (`re`, `regex`) to parse coordinate trajectories or action semantics (`axis_steps`, `piece_steps`, `internal dots`).
   - Bypassing, reordering, or truncating the 8-tier verification cascade of `LayeredVerifier`.
   - Replacing color indices with destructive `[COLOR]` tokens that erase semantic roles.
2. **Invariance Principles**:
   - All algorithms must operate invariantly across grid dimensions ($2 \times 2$ up to $64 \times 64$) and the full 16-color palette ($0..15$).
   - Color role classification (`EntityRole`) must be empirically grounded in victory exemplars (`VictoryExemplar`) and defeat exemplars (`DefeatExemplar`).

---

### 0.3 Normative Authority Hierarchy

Operational authority in the agent is strictly ordered from highest to lowest:

```
1. Gateway & Competition Contracts (Arcade step/reset interface, GAME_OVER protocol)
                                 │
2. Brusentsov LayeredVerifier (Strict 8-Tier Decision Cascade: FOLLOW / NULL / OMIT / UNDECIDED)
                                 │
3. GameSession Orchestrator (State machine, action budgets, cycle recovery, time budgeting, reactive DSL invalidation)
                                 │
4. SymbolicTrajectoryExecutor (Deterministic execution controller, early null severance, atomic substep unrolling)
                                 │
5. VerificationBinder Contracts (Grounds declarative steps strictly against the PlanningSet)
                                 │
6. Solver Agent (Declarative planning: packages multi-step trajectories and distills invariants)
                                 │
7. Coder Agent (Synthesizes sandboxed Python DSL implementations with auto-augmentation from factual specs)
                                 │
8. Explorer Agent (Two-tier probing of primitive actions and coordinate affordances; extracts factual rules)
                                 │
9. ARGA-Lite, PersistentObjectTracker & PlanningSet (Deterministic perception, tracking & DualView)
                                 │
10. Isolated Memory Stores (PaletteRoleMap, CoreInvariantRegistry, EpisodicExemplarBuffer, etc.)
```

No raw text or candidate action generated by an LLM is ever sent directly to the environment. Every action must be verified against world laws, bound through formal contracts, and emitted one transition at a time.

---

## 1. Theoretical Foundation: Brusentsov's Logic of Entailment

### 1.1 Historical Context & Epistemic Flaw of Classical Material Implication

Standard two-valued mathematical logic, founded on Boolean algebra and the truth-functional definition of material implication:
$$(x \to y) \equiv \neg x \lor y \equiv x' \lor y$$
introduces fatal paradoxes when applied to autonomous empirical reasoning and invariant deduction:

1. **Ex Falso Quodlibet (Truth from Falsehood)**: Under material implication, if premise $x$ is false ($x = 0$), the formula $(x \to y)$ evaluates to `TRUE` regardless of whether conclusion $y$ is true, false, or physically impossible.
2. **Vacuous Truth in Verification**: If an agent formulates an invariant rule "If object $A$ moves right, its color changes to red", and the agent performs an action where object $A$ does not move, classical logic treats the rule as completely verified and confirmed ($1$). In reality, the observation provided **zero evidence** regarding the validity of the rule.
3. **Loss of Necessary Consequence**: Material implication denotes mere non-exclusion (possibility) rather than necessary consequence.

In his seminal paper *"Усовершенствование логики умозаключений"* (2012, МГУ им. М.В. Ломоносова), **Nikolay Petrovich Brusentsov** demonstrated that Aristotle's original syllogistic entailment $Axy$ ("All $x$ are $y$", "y belongs to every x", $x = xy$ — *из $x$ необходимо следует $y$*) was distorted by modern logicians who forced a two-valued reduction upon an inherently three-valued relation.

---

### 1.2 Aristotle's Syllogistic Entailment ($Axy$)

Aristotle established that entailment is the containment of essences: to state that $y$ follows from $x$ means that the characteristic $y$ is necessarily inherent in every instance of $x$:
$$x = xy$$
Taking the contrapositive:
$$y' = x'y'$$
which expresses that if characteristic $y$ is absent ($y'$), characteristic $x$ must also be absent ($x'$), i.e. $A(y', x')$.

Aristotle's universe of discourse ($\text{УА}$) is populated by non-empty terms:
$$x \neq 0, \quad x' \neq 0, \quad y \neq 0, \quad y' \neq 0$$
Neither a characteristic nor its opposite is universally empty or universally exhaustive.

---

### 1.3 Lewis Carroll's Biliteral Diagrams & The Nullity Index ($xy'_0$)

Brusentsov recovered the genuine relation of entailment using the biliteral and indexical methods of Lewis Carroll (Charles Lutwidge Dodgson):

1. **Elementary Characteristics and Coexistence of Opposites**:
   Every primary characteristic $x$ necessarily coexists with its opposite $x'$. Their coexistence is subject to mutual incompatibility:
   $$xx'_0, \quad yy'_0, \quad zz'_0$$
   where the prime symbol ($'$) denotes opposition (negation), and the subscript index **«0»** denotes the **nullity (incompatibility / non-existence)** of the conjunction.

2. **The Entailment Relation as Incompatibility**:
   To state that $y$ necessarily follows from $x$ ($x \Rightarrow y$) means that it is physically and logically impossible for $x$ to exist without $y$ (i.e. for $x$ to coexist with $y'$):
   $$(x \Rightarrow y) \equiv xy'_0$$
   Taking into account contraposition ($(x \Rightarrow y) \equiv (y' \Rightarrow x')$), Carroll's complete formula for entailment is:
   $$(x \Rightarrow y)(y' \Rightarrow x') \equiv x_1 y'_0 \land y'_1 x_0 \equiv xy'_0$$
   In Aristotelian discourse, affirming entailment is mathematically identical to asserting the nullity of $xy'$.

3. **Material Implication vs. Entailment**:
   Material implication $(x \to y)$ can be expanded algebraically:
   $$(x \to y) \equiv (x \Rightarrow y) \lor x'y \equiv xy \lor xy'_0 \lor x'y \lor x'y'$$
   Material implication erroneously includes the term $x'y$ (where $x$ is absent and $y$ is present) as an explicitly affirmed truth condition. In necessary entailment, $x'y$ does not constitute part of the entailment relationship.

---

### 1.4 Overcoming DNF Incompleteness: Inessentiality (Omission) vs Exclusion

A central insight of Brusentsov’s theory is the **fundamental distinction between inessentiality (omission) and exclusion (incompatibility)**:

* **In Classical Boolean Algebra**: Omission of a conjunction term from a DNF formula implicitly signifies its **exclusion (falsity / non-existence)**:
  $$\text{omitted} \implies 0 \ (\text{false})$$
* **In Brusentsov’s Three-Valued Algebra**:
  * **Exclusion (Incompatibility)** must be explicitly designated by Carroll's nullity index **«0»** ($xy'_0$).
  * **Affirmation (Coexistence)** is designated by unindexed terms ($xy$).
  * **Omission (Silence / Умалчивание)** signifies **inessentiality (несущественность / irrelevance)**: the relationship holds regardless of whether the omitted term occurs or does not occur.

---

### 1.5 The 3-Valued DNF of Necessary Entailment

Thus, the relation of necessary entailment ($x \Rightarrow y$) is represented by the **three-valued DNF**:
$$x \Rightarrow y \equiv xy \lor xy'_0 \lor x'y'$$
where the term $x'y$ is **omitted as inessential**.

| Term | Algebraic Status | Semantics in Logic of Entailment | Semantics in LCLD Agent Verifier |
| :--- | :--- | :--- | :--- |
| **$xy$** | Unindexed | Necessary coexistence: premise and conclusion both occur. | **`FOLLOW (+1)`**: Positive confirmation of physical expectation. |
| **$xy'_0$** | Indexed with «0» | Strict incompatibility: premise occurs, conclusion fails. | **`NULL (-1)`**: Invariant contradiction. Candidate branch severed. |
| **$x'y$** | Omitted | Inessential: premise does not occur; conclusion occurs. | **`OMIT (0)`**: Neutral transition. Candidate branch preserved. |
| **$x'y'$** | Unindexed | Coexistence of opposites: neither premise nor conclusion occurs. | **`OMIT (0)`**: Neutral transition. World state unaffected. |

---

### 1.6 The 4-Valued Decision Semantics of `LayeredVerifier`

In real ARC-AGI-3 environments, observations are subject to **partial observability** (hidden layers, occluded entities, ambiguous affordances). To handle partial observability without violating Brusentsov’s axioms, the agent extends ternary logic into a **4-valued epistemic decision system**:

```
                                  Transition Result Evaluated
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     │                                                   │
              Contradiction                                        No Contradiction
                     │                                                   │
              ┌──────┴──────┐                                     ┌──────┴──────┐
              │  VERDICT:   │                                     │             │
              │    NULL     │                           Expectation Met?        Ambiguity
              │    (-1)     │                                     │             Detected?
              └─────────────┘                        ┌────────────┴───────────┐         │
                Incompatibility                      │                        │         │
                  $xy'_0$                           YES                       NO        ▼
                                                     │                        │   ┌───────────┐
                                               ┌─────┴─────┐            ┌─────┴───┤ VERDICT:  │
                                               │ VERDICT:  │            │VERDICT: │ UNDECIDED │
                                               │  FOLLOW   │            │  OMIT   │   (SEEK)  │
                                               │   (+1)    │            │  (0)    └───────────┘
                                               └───────────┘            └─────────┘ Epistemic
                                                Entailment              Omission    Ambivalence
                                                   $xy$                   $x'y$
```

1. **`Verdict.FOLLOW (+1)` (Entailment Confirmed, $xy$)**:
   - The executed action caused the explicit state change predicted by the grounded step's `EXPECT:` proposition (all expected propositions necessarily contained in observed transitions).
   - Physical progression is affirmed. The candidate trajectory advances its execution cursor (`cursor += 1`).

2. **`Verdict.OMIT (0)` (Inessentiality / Neutral Omission, $x'y$)**:
   - The action produced no contradiction with known physical world laws, but did not trigger the specific expected delta (e.g. passive interaction, non-essential background shift, or empty expectation set).
   - **Crucial Invariant (ISO-10)**: In accordance with Brusentsov's principle, inessentiality does **not** signify falsehood. The active candidate trajectory is **NEVER severed** on an `OMIT` verdict, and execution advances safely.
   - **Elimination of Vacuous Truth**: If a step declares zero expected propositions (`len(expected) == 0`), Brusentsov evaluation strictly returns `Ternary.IRRELEVANT` (neutral $x'y'$), mapping to `Verdict.OMIT`. It is fundamentally forbidden to treat absence of expectation as logical confirmation (`TRUE`).

3. **`Verdict.NULL (-1)` (Incompatibility / Falsification, $xy'_0$)**:
   - The action directly contradicted an established physical invariant or expected proposition.
   - Forms of contradiction detected by `contradicts()` include:
     - **Object identity destruction**: Expected `preserved`, observed `destroyed`, `vanished`, or `missing`.
     - **Physical stagnation**: Expected motion along an axis ($\text{exp} \neq 0$), but observed stationary ($\text{obs} == 0$).
     - **Unintended mutation**: Expected stationary/invariant state ($\text{exp} == 0$ or `unchanged`), but observed movement ($\text{obs} \neq 0$).
     - **Direction inversion**: Expected motion vector opposite to observed motion vector ($\text{exp} \times \text{obs} < 0$).
     - **Attribute/Area mismatch**: Value differences in color, bounding box, or pixel count.
   - The active candidate trajectory is **immediately severed (`sever()`)**. Execution of the current candidate terminates cleanly without executing trailing steps. The orchestrator schedules an environmental reset (`RESET` $\to S_0$) to restore a clean state before executing the next candidate or replanning.

4. **`Verdict.UNDECIDED` (Epistemic Ambivalence / Partial Observability)**:
   - Emitted when empirical evidence is insufficient to distinguish between entailment and contradiction:
     a) Bipartite track matching cost difference between best and second-best candidate is $< \text{matching\_ambiguity\_threshold}$ (0.15).
     b) Any participating `TrackedObject` exhibits confidence $< \text{track\_confidence\_threshold}$ (0.60).
     c) Zero grid delta observed on an unconfirmed action carrying a non-empty `EXPECT` set away from boundaries.
     d) Detected change metrics have magnitudes $< \text{min\_reliable\_delta}$ (0.8 px) with non-empty `EXPECT` and without terminal completion.
     e) Upstream `GroundedStep` or binder sets `confidence == "low"` or `matching_status == "ambiguous"`.
   - **Ablation Invariant**: When `enable_undecided_verdict = False`, all conditions above map conservatively to `Verdict.OMIT`, maintaining classic 3-valued operation.

---

### 1.7 Elimination of Material Implication Paradox in Verification

In earlier neuro-symbolic systems, when an action produced a non-zero grid delta and matched a confirmed action effect from memory, the verifier would grant a positive confirmation (`FOLLOW`) even if the step's expected proposition set was empty or unverified.

This was a direct manifestation of the **material implication paradox** ($x'y \to \text{FOLLOW}$). Under Brusentsov's logic:
- If expected propositions are empty (`len(expected) == 0`), the premise $x$ was not asserted. Any observed change $y$ is an unasserted side effect ($x'y$).
- The transition must evaluate to **`Verdict.OMIT`**, never `FOLLOW`.
- `Verdict.FOLLOW` is granted **if and only if** `len(step.expected_propositions) > 0` and `prop_verdict == Ternary.TRUE` ($xy$).

---

### 1.8 Carroll Nullity Completeness in Invariant Verification

In `brusentsov_logic.py`, the `contradicts()` operator exhaustively implements Carroll nullity ($xy'_0 \to \text{NULL}$) across all kinematic and topological dimensions:

1. **Object Identity Preservation**: When antecedent expectation asserts `predicate == "preserved"` and observation confirms `predicate in ("destroyed", "vanished", "missing")`, $x$ (preservation expected) and $y'$ (destruction observed) co-occur. Under Carroll's nullity rule, $xy'_0 \to \text{NULL}$.
2. **Physical Stagnation**: When expected motion is non-zero ($\Delta_{\text{exp}} \neq 0$) along row, column, or tuple vector, and observed displacement is zero ($\Delta_{\text{obs}} == 0$), the physical claim is refutable. The transition evaluates to incompatibility ($xy'_0$).
3. **Unintended Mutation**: When a step asserts that an entity remains stationary or invariant (`unchanged`, `stationary`, or $\Delta_{\text{exp}} == 0$), but observed displacement is non-zero ($\Delta_{\text{obs}} \neq 0$), an unexpected mutation occurred. This evaluates to incompatibility ($xy'_0$).
4. **Direction Inversion**: When expected and observed displacements move in opposite directions along an axis ($\text{sign}(\Delta_{\text{exp}}) \times \text{sign}(\Delta_{\text{obs}}) < 0$), the action's intended kinematic effect failed. This evaluates to incompatibility ($xy'_0$).
5. **Multi-Format Kinematics Support**: The operator handles tuple representations `(dy, dx)`, `moved`, `step_moved`, scalar signs `row_delta`, `col_delta`, and cross-comparisons with symmetric normalization.

---

### 1.9 Kinematic Boundary Protection & Obstacle Collision Soft Stop

To prevent spurious disqualification of confirmed directional actions when an actor collides with grid boundaries or impassable obstacles:

1. **Boundary Collision Detection (`is_blocked_by_boundary`)**:
   - Before evaluating a zero delta as action falsification, the executor checks whether the active component is in physical contact with the perimeter corresponding to the action direction ($y_{\min} == 0$ for UP, $y_{\max} == H-1$ for DOWN, $x_{\min} == 0$ for LEFT, $x_{\max} == W-1$ for RIGHT).
   - If in contact, the zero delta is recognized as physical obstruction rather than action failure. The candidate branch is severed without globally disqualifying the action from `GameMemory`.

2. **Obstacle Collision Soft Stop (`Tier 3`)**:
   - When motion produces zero grid delta against an established obstacle or wall rule in memory, the transition is classified as a **soft stop (`Verdict.OMIT`)** rather than a hard contradiction (`Verdict.NULL`).
   - The actor simply cannot pass through solid matter; no physical world law was broken. Confirmed kinematics in `GameMemory` remain protected from spurious deletion.

3. **Macro-Step Decomposition (Excising `break_on_null`)**:
   - Legacy implementations relied on `break_on_null` flags to silently skip remaining repeated steps when contacting obstacles, which masked contradictions and leaked invalid execution states.
   - In Version 10.5, `break_on_null` has been completely eliminated. Macro-commands (`count=N`) are unrolled into atomic steps where state invariants are enforced at every step. If an unexpected contradiction occurs, the candidate is cleanly severed, a diagnostic is recorded, and execution resets to $S_0$.

---

### 1.10 Evidence-Seeking Finite State Machine (E-FSM)

When `Verdict.UNDECIDED` is emitted, candidate trajectory execution is suspended, entering an active evidence-seeking loop:

1. **State Snapshotting & Execution Freezing**:
   - Immutable snapshot captured: `pending_step_snapshot = active_step`.
   - `evidence_seeking_active = True` engaged; execution cursor frozen (`candidate_advanced = False`, `candidate_severed = False`).

2. **Blocking `act()` Action Arbitration**:
   - Trajectory advancement and Solver calls are blocked.
   - If `evidence_probes_remaining > 0`, a targeted diagnostic probe action is emitted.
   - If `probe_queue` is exhausted, falls back to `Verdict.NULL`, severs the candidate, and resets cleanly to $S_0$.

3. **Post-Probe Re-evaluation**:
   - Upon observing the post-probe state, the verifier checks whether the epistemic ambiguity condition on `pending_step_snapshot` has resolved.
   - Resolved: `evidence_seeking_active` cleared; trajectory execution resumes cleanly.
   - Persisting: `undecided_streak` increments. If streak reaches `max_undecided_streak = 2`, falls back to `Verdict.NULL` and resets.

---

### 1.11 The 8-Tier Verification Cascade (`LayeredVerifier`)

Every transition $(S_{t-1}, a_t, S_t)$ is evaluated through a strict, non-invertible 8-tier hierarchy:

```
[Incoming Step Transition]
           │
  Tier 1:  ▼ Terminal Win Condition (WIN / levels_completed increase) ────────► FOLLOW
  Tier 2:  ▼ Terminal Loss Condition (GAME_OVER / LOST) ──────────────────────► NULL
  Tier 3:  ▼ Zero Grid Delta on Confirmed Motion:
           │   - Known obstacle / wall collision? ────────────────────────────► OMIT (Soft Stop)
           │   - Unexpected motion failure? ──────────────────────────────────► NULL (Contradiction)
  Tier 4:  ▼ Epistemic Uncertainty & Multi-Frame Tracking Ambiguity:
           │   - GroundedStep confidence == 'low' / ambiguous status ─────────► UNDECIDED
           │   - Ambiguity diff < matching_ambiguity_threshold (0.15) ────────► UNDECIDED
           │   - Participating TrackedObject confidence < 0.60 ───────────────► UNDECIDED
  Tier 5:  ▼ Explicit EXPECT Contradiction (implies_brusentsov == FALSE) ─────► NULL
  Tier 6:  ▼ Explicit EXPECT Necessary Containment (implies == TRUE) ─────────► FOLLOW
  Tier 7:  ▼ Unconfirmed Zero Delta or Low Metric Delta:
           │   - Zero Delta on Unconfirmed Action with non-empty EXPECT ──────► UNDECIDED
           │   - Sub-pixel displacement (0 < Delta < min_reliable_delta 0.8) ─► UNDECIDED
  Tier 8:  ▼ Certified Action Effect with Verified Propositions:
           │   - Non-zero delta + verified non-empty EXPECT? ─────────────────► FOLLOW
           │   - Non-zero delta + empty / unverified EXPECT? ─────────────────► OMIT (No Vacuous Follow)
           │   - Default Fallback (Fail-Safe Preservation, ISO-10) ───────────► OMIT
```

- **Fail-Safe Default Guarantee (ISO-10)**: Tiers 7 and 8 strictly return `Verdict.OMIT`. Unless affirmative physical incompatibility ($xy'_0$) is proved, the candidate trajectory is never severed.
- **Topological Invariant Grounding**: Judgments verify 2D spatial descriptors (centroids, bounding boxes, contact topology, and color palettes) without assuming benchmark-specific symmetries.

---

### 1.12 Structured Contradiction Diagnostics & Epistemic Recovery

When a candidate trajectory is severed at step $k$, `SymbolicTrajectoryExecutor` invokes `_build_step_contradiction_diagnostic()` to synthesize an informative diagnostic block:

```text
[FAILED ATTEMPT N DIAGNOSTIC]
Failed at step sK (function_name): Mismatch detected.
EXPECTED: moved(A, 0, 1) & unchanged(B)
OBSERVED: stationary(A, 0, 0) & moved(B, 0, 1)
DIAGNOSIS: function_name mutates alias B, not alias A. Do NOT repeat function_name for moving A!
```

This diagnostic, along with the physical grid diff (`summarize_grid_diff`), is recorded into `EpistemicMemory.record_attempt_feedback` and rendered directly into the Solver's `failed_attempts_scratchpad`. This differential feedback prevents the Solver from repeating contradictory hypotheses and enables rapid epistemic recovery.

---

## 2. Tri-Agent Separation of Powers

To guarantee zero cross-contamination, cognitive responsibilities are strictly partitioned among three specialized LLM call families:

```
                      ┌─────────────────────────────────────────┐
                      │               GameSession               │
                      │      (State Machine & Orchestrator)     │
                      └───────┬────────────┬────────────┬───────┘
                              │            │            │
                  Empirical   │     Syntax │ Declarative│ Multi-Candidate
                  Probing     │  Synthesis │   Planning │ Packages
                              ▼            ▼            ▼
                       ┌────────────┐┌────────────┐┌────────────┐
                       │  Explorer  ││   Coder    ││   Solver   │
                       │   Agent    ││   Agent    ││   Agent    │
                       └──────┬─────┘└─────┬──────┘└─────┬──────┘
                              │ Writes     │ Writes      │ Writes
                              ▼ Only       ▼ Only        ▼ Only
                       ┌────────────┐┌────────────┐┌────────────┐
                       │  EnvSpec   ││SyntaxError ││ Epistemic  │
                       │   Memory   ││   Memory   ││   Memory   │
                       └────────────┘└────────────┘└────────────┘
                              ▲            ▲             ▲
                              └────────────┴─────────────┘
                                  Strictly Quarantined
```

### 2.1 Explorer Agent (Call Family 1: Empirical Fact Discovery)
* **Mission**: Probe available actions (`ACTION1..5`) and spatial coordinates (`ACTION6`) on the pristine board ($S_0$) to extract deterministic action kinematics, coordinate affordances, and entity switching mechanics.
* **Memory Ownership**: Writes exclusively to `EnvironmentSpecMemory`.
* **Prohibitions**: **Must not** formulate puzzle goals, hypothesize winning conditions, or plan multi-step paths.
* **Output Contract**: Valid JSON payload adhering to schema `v10.env_spec.1`.
* **Two-Tier Action Lifecycle & Affordance Completeness**:
  - Discrete actions probed on the pristine initial frame $S_0$ are partitioned into `confirmed_effective_actions` (producing observable grid delta $\Delta > 0$) and `conditional_candidate_actions` (zero grid delta $\Delta = 0$ on $S_0$).
  - Zero-delta actions are **not discarded**. They are preserved in `available_actions` and typed as `EffectClass.CONDITIONAL_TRIGGER` affordances (`status: unconfirmed_on_s0`) in `EnvironmentSpecMemory`.
  - This guarantees that actions which require specific preconditions (such as having picked up an item or standing on a trigger) remain available to the Coder.
* **Action Space Partitioning & Coordinate Dual-Space Synchronization**:
  - Probes are strictly partitioned: `ACTION1..5` are evaluated as discrete actions by `PrimitiveProbeManager.plan_discrete_probes()`.
  - `ACTION6` spatial coordinate probes are handled via `propose_coordinate_probes()` using Qwen coordinate hypotheses and deterministic affordance extraction.
  - All coordinate hypotheses and affordances operate strictly in local cropped grid coordinates `(0 <= x < width, 0 <= y < height)` matching the ARGA-Lite `PlanningSet`.
  - When emitted to the game engine, coordinates are shifted by `crop_offset` (`engine_x = local_x + crop_offset`, `engine_y = local_y + crop_offset`).
  - Recorded probe actions preserve dual-space metadata (`local_x`, `local_y`, `crop_offset`). Tested coordinate sets decode engine coordinates back to local coordinates (`xi - crop_offset, yi - crop_offset`), guaranteeing that previously probed coordinates are **never re-probed**.
* **Universal Multimodal Dual-View**:
  - Explorer receives both `raw_frame.png` and `annotated_frame.png`, providing perceptual grounding for compound structures that the single-color connected-component parser fragments.

---

### 2.2 Coder Agent (Call Family 2: Sandboxed DSL Synthesis)
* **Mission**: Translate verified empirical facts from `EnvironmentSpecMemory` and confirmed physics from `GameMemory` into a clean, typed Python Domain-Specific Language (DSL) module implementing high-level primitive operations.
* **Memory Ownership**: Writes exclusively to `SyntaxErrorMemory` upon compilation or runtime failure.
* **Prohibitions (ISO-2)**: **Strictly quarantined from goal definitions**. Does not know how to solve the puzzle, only how to manipulate objects according to the physics of the environment.
* **Execution Environment**: Compiled code is validated by `SafeASTVisitor` and executed inside `SandboxExecutor` under strict resource ceilings (10s CPU, 512MB RAM).
* **Incremental DSL Synthesis & Automatic Code Augmentation**:
  - The Coder develops the DSL incrementally across levels: previously confirmed actions (`action1..action4`) must be preserved and augmented with newly discovered level primitives (e.g. `action5` entity cycling or `action6` coordinate clicking).
  - **Automatic Augmentation Guard**: In `dsl_coder.py`, `generate_dsl` inspects both function names and `action_id` tags. If the LLM omitted any confirmed action from `GameMemory.confirmed_action_effects` or `env_spec["available_actions"]`, canonical implementation wrappers and manifest descriptors are automatically synthesized and injected:
    ```python
    def action5(api):
        """Canonical auto-augmented wrapper for confirmed action ACTION5."""
        return api.declare_environment_action(action_id='ACTION5')
    ```
  - **Sandbox Verification**: All augmented code undergoes restricted AST validation and dry-run execution against the current `PlanningSet`.
* **Universal Multimodal Dual-View**:
  - Coder receives both `raw_frame.png` and `annotated_frame.png` to cross-reference symbolic action semantics with macroscopic visual objects.

---

### 2.3 Solver Agent (Call Family 3: Declarative Planning & Reflection)
* **Mission (Turn-1 Planning)**: Propose a structured package of up to 4 distinct multi-step candidate trajectories expressed in terms of the Coder's DSL manifest, with explicit `EXPECT:` propositions for every step.
* **Mission (Turn-2 Reflection)**: Following level completion or attempt exhaustion, analyze outcomes and distill domain-general invariants categorized into `[PALETTE & ROLES]`, `[GOAL]`, `[ENTITIES]`, `[CONTROL]`, and `[PHYSICS]`.
* **Memory Ownership**: Reads from and writes to `EpistemicMemory` and `GameMemory`.
* **Prohibitions**: Never executes code, never emits raw actions directly to the environment, and never receives Python syntax error tracebacks. `ACTION7` does not exist for the Solver and is never proposed.
* **Source Authority Hierarchy (Prompt Grounding)**:
  1. **Raw frame pixels**: Absolute ground truth of screen content.
  2. **Empirical evidence log**: Factual records of action execution.
  3. **Annotated frame**: Parser segmentation used strictly to map alias labels to pixel regions.
  4. **Symbolic object index**: Lossy metadata representation. Where index and pixels conflict, pixels win.
* **DualView Grounding & Parser Over-Segmentation Compensation**:
  - Informs the model that the perception engine decomposes multi-color objects into monochromatic components. The model must consult raw pixels to identify compound patterns and multi-color composite shapes.
* **Differential Analysis & Severed Attempt Diagnostics**:
  - Section 7 of the Solver prompt provides `failed_attempts_scratchpad` containing exact physical grid diffs ('which tiles changed color before vs after the attempt') and `[FAILED ATTEMPT N DIAGNOSTIC]` analysis, enabling differential reasoning and preventing repetition of contradicted steps.

---

## 3. Stratified 5-Block Memory Architecture

To facilitate lifelong cross-level learning without context pollution, memory is organized into five specialized, non-interfering stores:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PaletteRoleMap: 16-color palette (0..15) affordances, roles & confidence │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. CoreInvariantRegistry: Invariants grounded in exemplars via Brusentsov   │
│    logic (NEGATIVE_BARRIER from Defeat, POSITIVE_CANON from Victory)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. EpisodicExemplarBuffer: Concrete physical exemplars of terminal states   │
│    (DefeatExemplar on GAME_OVER, VictoryExemplar on Level Win)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. CurriculumProgressionBuffer: Cross-level trajectory history & patterns    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. WorkingRolloutMemory: Active candidate steps, propositions & deltas       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 16-Color Palette & Role-Aware Semantic Labeling

ARC-AGI-3 environments operate across a fixed 16-color palette ($0..15$). The agent models this palette through `PaletteRoleMap` and `ColorAffordance`:

```python
class EntityRole(str, Enum):
    BACKGROUND = "background"     # Empty / filler cells (typically most frequent color)
    ACTOR = "actor"               # Controllable object (moves with ACTION1..5 or clicks)
    OBSTACLE = "obstacle"         # Impassable walls (zero delta on collision)
    HAZARD = "hazard"             # Lethal color (contact triggers RESET / defeat)
    TARGET = "target"             # Goal zone / exit (contact triggers VICTORY)
    COLLECTIBLE = "collectible"   # Items that vanish on contact (keys, coins)
    PORTAL = "portal"             # Teleports or state-change triggers
    UNKNOWN = "unknown"           # Not yet classified
```

- **Non-Destructive Generalization**: Instead of erasing colors with destructive `[COLOR]` tokens, `PaletteRoleMap.generalize_color_reference` transforms color references into semantic labels: `Color N (ROLE)` (e.g., `Color 1 (ACTOR)`, `Color 2 (HAZARD)`, `Color 8 (TARGET)`).
- This preserves the exact color identifier while conveying its discovered functional role to the LLM.

---

### 3.2 Negative Barrier Exemplars (`DefeatExemplar`)

Upon encountering `GAME_OVER` or a fatal state transition, the orchestrator constructs an immutable `DefeatExemplar`:

- **Fields**: `level_index`, `fatal_step`, `fatal_action_id`, `fatal_coords`, `actor_position_before`, `hazard_color`, `environment_signal`, `explanation`.
- **Brusentsov Grounding**: Grounds `NEGATIVE_BARRIER` invariants ($xy'_0 \to \text{NULL}$).
- **Prompt Injection**: Injected into the Solver's context:
  ```text
  LAST DEFEAT (Level 0, Step 8):
    Action: ACTION3
    Actor was at: row=12, col=15
    Hazard color: 2
    Signal: GAME_OVER
    Lesson: Moving right into Color 2 causes immediate destruction.
  ```
  This prevents the Solver from repeating fatal actions in future trajectory candidates.

---

### 3.3 Positive Canon Exemplars (`VictoryExemplar`)

Upon level completion, the orchestrator captures a `VictoryExemplar` (and `LevelVictoryExample`):

- **Fields**: `level_index`, `total_steps`, `action_sequence`, `key_transitions`, `final_action_id`, `target_color`, `winning_invariants_used`, `explanation`.
- **Brusentsov Grounding**: Grounds `POSITIVE_CANON` invariants ($xy \to \text{FOLLOW}$).
- **Prompt Injection**: Injected as an objective physical demonstration of successful problem-solving, guiding subsequent levels with verified behavioral canons.

---

### 3.4 Grounded Invariant Lifecycle & Falsification

Invariants in `CoreInvariantRegistry` are structured as `GroundedInvariant`:
- `antecedent`: Condition (e.g. `Contact(ACTOR, Color_2)`).
- `consequent`: Outcome (e.g. `DefeatReset()`).
- `brusentsov_type`: `NEGATIVE_BARRIER` or `POSITIVE_CANON`.
- `scope`: `CORE_GAME_LAW` (persists across all levels) or `LEVEL_SPECIFIC`.
- Grounded directly in `DefeatExemplar` or `VictoryExemplar`.
- **Falsification Rule**: An invariant is active iff `times_falsified == 0`. The first empirical contradiction on pristine $S_0$ permanently inactivates the invariant.

---

### 3.5 Architectural Isolation Invariants (ISO-1..ISO-12)

The integrity of the architecture is guarded by 12 non-negotiable invariants:

* **ISO-1 (Zero Syntax Tracebacks in Solver)**: Raw Python tracebacks, compiler errors, and AST violations are strictly confined to `SyntaxErrorMemory` and never leak into `EpistemicMemory` or Solver prompts.
* **ISO-2 (Curriculum Goal Quarantine for Coder)**: The Coder prompt and `EnvironmentSpecMemory` are strictly purged of all `[GOAL]` tags, target scores, and win conditions. Coder synthesizes mechanical manipulation primitives only.
* **ISO-3 (Sandbox Containment)**: All DSL code executes in an isolated subprocess with strict CPU, memory, and syscall restrictions.
* **ISO-4 (Immutable Perception Grounding)**: The `PlanningSet` extracted at frame $t$ is immutable. LLMs can refer to objects (`obj_0`, `obj_1`) but cannot mutate the underlying perceptual graph.
* **ISO-5 (Declarative Planning Purity)**: The Solver produces pure declarative text trajectories; it possesses no ability to invoke functions, mutate environment state, or inspect Python runtime state.
* **ISO-6 (Substantive Relation Filtering)**: Spatial relations in the `PlanningSet` are restricted to substantive entities ($area \ge 4$), eliminating combinatorial explosions from background cavities.
* **ISO-7 (Forward-Only Clean Reset Execution)**: Step reversal via `ACTION7` is prohibited. Erroneous paths are abandoned via environmental `RESET`, returning the system to pristine initial state $S_0$.
* **ISO-8 (Non-Destructive Palette Mapping)**: Concrete palette indices are converted into role-aware labels (`Color N (ROLE)`), preserving both color identity and semantic function across levels.
* **ISO-9 (Epistemic Signal Segregation)**: `Verdict.UNDECIDED` signals are stored in transient `epistemic_signals`, completely separated from confirmed historical `judgments`.
* **ISO-10 (Non-Severance on Expectation Mismatch)**: An unfulfilled step expectation that does not violate any established physical invariant yields `Verdict.OMIT`, strictly preserving the surviving candidate trajectory.
* **ISO-11 (Kinematic Boundary Protection)**: Zero grid delta on a motion action caused by contact with grid boundaries or selection-state requirements is classified as obstruction rather than physical falsification, preserving confirmed directional rules in `GameMemory`.
* **ISO-12 (Dual-Space Coordinate Consistency)**: Spatial coordinate reasoning and tested-coordinate deduction operate strictly in the local perception frame. Engine coordinate conversions ($+ crop\_offset$) must be symmetrically inverted when recording and evaluating tested coordinate history.

---

### 3.6 Persistent Object Tracking (`PersistentObjectTracker`)

To overcome object ID flicker across consecutive frames, perception is stabilized by `PersistentObjectTracker` (`v10_agent/tracker.py`):

1. **Normalized 4-Part Bipartite Matching Cost**:
   $$\text{Cost}(T, C) = 0.30 \cdot \text{color\_diff} + 0.40 \cdot \frac{\text{manhattan}(T, C)}{\max(H, W)} + 0.20 \cdot (1.0 - \text{Jaccard}(T, C)) + 0.10 \cdot \frac{|T_{\text{area}} - C_{\text{area}}|}{\max(T_{\text{area}}, C_{\text{area}}, 1)}$$
   Matches are accepted if $\text{Cost}(T, C) < 0.45$. Ties are resolved deterministically by lexicographic priority on `persistent_id`.

2. **Track Kinematics & Identity Permanence**:
   - Confidence: $\text{confidence} = 1.0 - \text{cost}$ (initialized to 1.0 upon spawn).
   - Velocity EMA: $\mathbf{v}_t = 0.3 \cdot (\mathbf{c}_t - \mathbf{c}_{t-1}) + 0.7 \cdot \mathbf{v}_{t-1}$.
   - Shape Stability: Decays by $\times 0.9$ if shape Jaccard $< 0.85$.
   - Cumulative Motion: Evaluated across a 3-frame window ($\Delta_{\text{cum}} = \mathbf{c}_{\text{newest}} - \mathbf{c}_{\text{oldest}}$).
   - Occlusion Detection: Unmatched track flagged `occluded = True` if overlapping an adjacent larger component within 3 px.

---

## 4. Action Space, Coordinates & Hardware Constraints

### 4.1 Partitioned Action Space

The action space partitions strictly into:
1. **Discrete Actions (`ACTION1..5`)**: Directional controls (`ACTION1` UP, `ACTION2` DOWN, `ACTION3` LEFT, `ACTION4` RIGHT) and entity toggle/mode switches (`ACTION5`).
2. **Spatial Coordinate Actions (`ACTION6(x, y)`)**: Spatial clicks targeting specific grid cells. Probed via deterministic component centroid heuristics and Qwen coordinate proposals.

---

### 4.2 Hardware Constraint: Prohibition of `ACTION7`

`ACTION7` represents step reversal (`Undo`). The agent fundamentally excludes `ACTION7`:
- It is never generated by the Explorer, Coder, or Solver.
- If present in environment action manifests, it is filtered out by `allowed_action_ids`.
- Epistemic recovery is achieved exclusively through forward planning from clean initial states ($S_0$) via `RESET`.

---

### 4.3 1px Boundary Crop & Dual-Space Coordinate Mapping

ARC-AGI-3 environments frequently include a 1-pixel border perimeter around the active grid.
- **Normalization**: `normalize_observation` crops 1 border pixel on all 4 sides ($[1:-1, 1:-1]$), producing local dimensions $H_{\text{local}} = H - 2$, $W_{\text{local}} = W - 2$ with `crop_offset = 1`.
- **Local Reasoning**: All perception, object extraction, spatial relations, and planning operate exclusively in local coordinates $[0, W_{\text{local}}) \times [0, H_{\text{local}})$.
- **Dispatch Translation**: When dispatching `ACTION6(x, y)` to the competition arcade gateway, coordinates are mapped to engine space:
  $$\text{engine\_x} = \text{local\_x} + \text{crop\_offset}, \quad \text{engine\_y} = \text{local\_y} + \text{crop\_offset}$$
  Action metadata records `{"x": engine_x, "y": engine_y, "local_x": local_x, "local_y": local_y, "crop_offset": crop_offset}`.

---

### 4.4 Tested Coordinate Deduplication Across Spaces

When reconstructing `tested_coords` from probe history, engine coordinates are decoded back into local space:
$$\text{local\_x} = \text{engine\_x} - \text{crop\_offset}, \quad \text{local\_y} = \text{engine\_y} - \text{crop\_offset}$$
This symmetric decoding prevents duplicate probing of identical physical locations.

---

## 5. Symbolic Execution & Domain-General Invariants

### 5.1 Symbolic Trajectory Execution with Macro-Step Unrolling & Early Null Severance

Candidate trajectories are executed step-by-step by `SymbolicTrajectoryExecutor`:
- **Atomic Substep Decomposition**: Macro-step instructions like `action1(count=N)` are decomposed into $N$ atomic steps via `_decompose_expected_propositions_for_substep`.
- **Invariant Distribution**: State invariants (`unchanged`, `preserved`) are distributed to every intermediate substep. Incremental motion expectations (`step_moved: (sgn(dy), sgn(dx))`) are assigned to each step, while the full cumulative vector (`moved: (dy, dx)`) is evaluated on the final step.
- **Early Severance on Contradiction**: If any step receives `Verdict.NULL` (falsification, boundary collision, stagnation, or unintended mutation), execution of the candidate halts immediately:
  - `active_cand.sever()` is executed.
  - Remaining unexecuted steps are cleanly dropped.
  - `SymbolicTrajectoryExecutor` compiles a structured contradiction diagnostic block.
  - `GameSession` triggers a clean `RESET` to $S_0$, restoring the environment before testing the next candidate.

---

### 5.2 Clean Physics Simulator & Generalized Aspect Ratio

Legacy heuristic constants (e.g. $10 \times 4$ bounding box checks) are completely eliminated. In `virtual_sandbox.py` and `universal_invariants.py`:
- Geometric aspect ratio $\ge 3.0$ is used generically to identify elongated or axis-like components regardless of absolute scale.
- Operates invariantly on boards from $2 \times 2$ to $64 \times 64$.

---

### 5.3 Domain-General Topological Invariants

World physics is modeled using four domain-general invariants:
1. `ConnectedComponentConservation`: Verifies that an object retains its 4/8-connectivity and pixel area under spatial transformations.
2. `GravitySettling`: Models dynamic entities that settle along a directional gradient $(\Delta y, \Delta x)$ until contacting a support surface.
3. `ContactTrigger`: Models spatial adjacency between subject and trigger entities that induces a state change (barrier opening, teleportation, color transformation).
4. `AreaConservation`: Verifies that object pixel count is conserved across translations and rotations.

---

### 5.4 Multi-Candidate Package Traversal via Clean RESET

When the Solver produces a trajectory package containing candidates $\langle T_1, T_2, T_3, T_4 \rangle$:
1. $T_1$ executes step-by-step.
2. If $T_1$ encounters a hard `Verdict.NULL`, it is severed (`T1.sever()`).
3. If unsevered candidates remain ($T_2, T_3, \dots$), the orchestrator dispatches `RESET` to restore clean state $S_0$, and execution immediately resumes with $T_2$.
4. If all candidates are severed, replanning is requested. The agent allows up to 5 chain attempts per level before falling back to symbolic A* search.

---

### 5.5 Tufa-Loop Invariant and Single-RESET `GAME_OVER` Protocol

When the environment emits `state == "GAME_OVER"`:
1. Exactly **one** `RESET` action is emitted to restart the level.
2. Receiving `GAME_OVER` severs **only** the active candidate trajectory that caused the defeat. It captures a `DefeatExemplar` and resumes from $S_0$ with the next candidate in the pool.
3. The orchestrator tracks `last_engine_action_source` to prevent double-reset traps.

---

### 5.6 Cross-Level Invariant Discovery & Context-Driven Probing Protocol

* **Pristine Frame Invariant Evaluation ($S_0$)**: Cross-level invariant re-evaluation occurs strictly during the first `act()` call on the pristine observation of the new level ($S_0$, `level_initial_grid is None`), never on the victory frame of the prior level.
* **Reactivation of Probing Phase**: `handle_level_transition` sets `probing_phase = True` and clears level-local probe records, discovering actions that become functional only on subsequent levels.
* **Selection Indicator Trigger**: When a probe reveals a selection indicator or entity toggle, targeted directional probes (`ACTION1..4`) are immediately enqueued to probe the kinematics of the newly activated entity before resetting to $S_0$.

---

### 5.7 Two-Tier Action Lifecycle & Reactive DSL Invalidation Pipeline

* **Two-Tier Preservation**: Actions evaluated during primitive probing that exhibit zero delta ($\Delta = 0$) on $S_0$ are tracked in `conditional_candidate_actions` and preserved in `available_actions` with `effect_class = CONDITIONAL_TRIGGER`. They are not discarded as ineffective.
* **Reactive Invalidation**: In `observe_action_result`, if any confirmed action is executed that is absent from `active_manifest`:
  1. `active_module` and `active_manifest` are invalidated (`None`).
  2. `coder_failed_for_level` is reset to `False`, allowing fresh Coder synthesis attempts.
  3. `replan_requested` is flagged `True` to trigger complete trajectory package replanning.
  4. `EnvironmentSpecMemory` dynamically synchronizes its action surface with the new confirmed effect.

---

## 6. Competition Reliability Infrastructure

### 6.1 VisibleCycle Loop Recovery (`cycle_detector.py`)

To prevent burning the 250-action budget in closed loops:
- Records transitions as tuples: $(\text{hash}(S_{t-1}), \text{ACTION}, \text{hash}(S_t))$ using fast row-level MD5 hashing.
- Detects closed orbits of period $P \in [1..8]$ repeating $\ge 4$ times and spanning $\ge 24$ actions.
- Automatically severs the active trajectory, captures a failure record, resets the environment to $S_0$, and triggers replanning with cycle-avoidance constraints.

---

### 6.2 Deadline Management & Dynamic Time Budgeting

To safeguard the 5000-second per-game and 30600-second competition budgets:
- Monotonic deadline tracking with a 15-second reserve (`deadline_reserve_seconds = 15.0`).
- If remaining time is $\le 15.0$ seconds, LLM generation requests abort immediately, returning `"{}"`.
- Socket timeouts are dynamically clamped: $\min(\text{base\_timeout}, \max(2.0, \text{remaining\_time} - 15.0))$.
- Competition child process exits gracefully with `stop_reason = "deadline_reserve"`.

---

### 6.3 vLLM Serving Architecture, Watchdog & Sub-Second Teardown

- Production server flags: `--enable-prefix-caching`, `--enable-chunked-prefill`, `--async-scheduling`, `--no-enable-log-requests`, `--disable-uvicorn-access-log`, `--max-model-len 131072`, `--gpu-memory-utilization 0.95`.
- MTP=3 speculative decoding: `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`.
- Background Watchdog: Non-blocking daemon probing `/health` every 15s using `threading.RLock`. Automatically restarts stalled servers (up to 2 attempts).
- Sub-millisecond Teardown: Probes socket in $< 0.05$s; if port is closed, exits in $< 1$ ms without invoking expensive shell processes.

---

### 6.4 Generalized A* Search over Feature Differentials

Deterministic fallback engine operating over a 4-feature differential vector:
$$h(S) = 0.4 \cdot \Delta_{\text{centroid}} + 0.2 \cdot \Delta_{\text{bbox}} + 0.2 \cdot (1.0 - \text{color\_match}) + 0.2 \cdot \Delta_{\text{count}}$$
Perimeter coordinate clamping prevents spurious boundary exceptions during simulation:
$$y \leftarrow \max(0, \min(H-1, y)), \quad x \leftarrow \max(0, \min(W-1, x))$$

---

## 7. Verification & Code Quality Assurance Stack

### 7.1 AST Code Guardian (`tools/ast_code_guardian.py`)

Static and semantic AST auditor enforcing the Anti-Specification Gaming Contract:
- Scans all Python source files in the codebase (103 files).
- Checks for forbidden game ID patterns (`ar\d{2}`, `ft\d{2}`, etc.).
- Checks for forbidden heuristic tokens (`piece_steps`, `axis_steps`, `internal dots`).
- Checks for hardcoded dimension comparisons (`width <= 4`, `height >= 10`).
- Verifies that `ACTION7` is never emitted as an executable action.

---

### 7.2 Property-Based Testing with Hypothesis (`v10_agent/tests/test_pbt_*.py`)

Exhaustive property-based verification of mathematical and geometric invariants:
- `test_pbt_brusentsov_axioms.py`: Verifies ex falso quodlibet elimination, Carroll nullity, identity consequence, and non-vacuous truth across randomized proposition sets.
- `test_pbt_brusentsov_severance.py`: Verifies Carrollian nullity completeness, physical stagnation, unintended mutation, direction inversion, consistent containment, macro-step decomposition, and early candidate severance.
- `test_pbt_scale_invariance.py`: Verifies grid parsing and topological invariants across arbitrary grid dimensions ($2 \times 2$ to $64 \times 64$).
- `test_pbt_color_permutation.py`: Verifies semantic role stability and prompt generalization under arbitrary permutations of the 16-color palette.
- `test_pbt_coordinate_isomorphism.py`: Verifies dual-space coordinate translations and tested-coordinate deduplication under arbitrary cropping offsets.
- `test_reactive_dsl_invalidation.py`: Verifies two-tier action lifecycle, preservation of unconfirmed candidates as `CONDITIONAL_TRIGGER`, reactive DSL module invalidation on newly confirmed effects, and Coder wrapper auto-augmentation.
- `test_universal_multimodal_dual_view.py`: Verifies DualView frame generation, perception grounding, and coordinate format consistency.

---

### 7.3 Synthetic Calibration Micro-Worlds (`test_synthetic_calibration_worlds.py`)

Deterministic synthetic micro-environments for calibrating kinematics, boundary guards, color affordances, and solver pipelines:
- **Micro-World 1 (Maze Navigation)**: Verifies actor movement around solid obstacles to reach a target zone without branch severance.
- **Micro-World 2 (Gravity Settling & Soft Boundary)**: Verifies downward motion settling onto solid surfaces with soft-stop boundary protection.
- **Micro-World 3 (Contact Trigger & Door Opening)**: Verifies spatial contact triggers opening passages to goal targets.
