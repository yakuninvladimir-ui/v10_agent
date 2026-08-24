# ARC-AGI-3 LCLD Agent
# Architectural Specification — Version 10.0
# (Tri-Agent Hierarchy, Isolated Memory Contours & Brusentsov Ternary Logic)

## 0. Purpose

This document defines the Version 10.0 architecture of an ARC-AGI-3 interactive reasoning agent operating under:

- hidden environment rules and dynamic action spaces;
- deterministic object-centric perception (ARGALite + PlanningSet);
- strict offline / competition execution constraints;
- multimodal local LLM (Qwen FP8 / vLLM) used only through three isolated roles;
- **tri-agent authority separation** (Explorer, Coder, Solver);
- **strictly isolated memory contours** that prevent cross-contamination between factual discovery, syntax generation, and epistemic reasoning;
- **Brusentsov ternary logic** for empiric transition judgment (FOLLOW / NULL / OMIT);
- Double-Loop Learning with domain-specific error routing;
- dual-view observation (annotated visual frame for multimodal input, structured verifier packet for offline verification);
- offline, replayable, auditable competition agent.

The architecture is **not**:

- a single monolithic LLM prompt loop;
- a system that allows an LLM to emit or authorize final environment actions;
- a system that hallucinates unrestricted DSL execution;
- a raw-pixel policy network or generic RL learner;
- an unconstrained transformer planner;
- a hidden-state LLM controller.

The architecture **is**:

- a neuro-symbolic multi-agent system with clear separation of powers;
- an environment in which a level-specific DSL is *dynamically generated* by a dedicated Coder but *deterministically validated, sandboxed and executed*;
- a verifier-centered system that distinguishes strict physical contradictions (NULL) from insignificant / passive outcomes (OMIT) using Brusentsov’s necessary-implication semantics;
- a system that retains the PlanningSet identity contract and dual-view media of later V9 work while restoring the high-assurance isolation philosophy of V6.

### 0.1 Authority Hierarchy

Normative authority order (highest first):

1. **Environment transition and competition gateway contracts** (accepted `env.step` frames, official RESET semantics).
2. **LayeredVerifier using Brusentsov ternary logic** — absolute authority on step relevance and branch viability (FOLLOW / NULL / OMIT).
3. **GameSession** — sole mutable-state owner and orchestrator of the three agents, budgets, and fallbacks.
4. **VerificationBinder contracts** (LOCAL_TARGET_CHANGE, OBJECT_DISPLACEMENT, RELATION_ERROR_DECREASE, …) grounded on the current PlanningSet.
5. **Solver Agent** — proposes trajectories expressed solely as sequences of typed DSL function calls (JSON / AST).
6. **Coder Agent** — generates the Python DSL implementation from an Environment Specification; has zero knowledge of level goals.
7. **Explorer Agent** — researches simple actions and coordinate affordances; produces only factual Environment Specifications.
8. **Capability surface and PlanningSet** of the current snapshot.
9. **Isolated memory contours** (EnvironmentSpecMemory, SyntaxErrorMemory, EpistemicMemory).

No LLM output may ever reach the environment directly. Only GameSession, through ActionBoundary after LayeredVerifier approval, may emit a real action.

### 0.2 Dual-view design intent (retained)

Under environment uncertainty a full fixed symbolic DSL is not feasible. Version 10 retains two projections of one planning identity:

| Channel   | Media                                      | Consumer          | Role                                      |
|-----------|--------------------------------------------|-------------------|-------------------------------------------|
| Visual    | `current_frame_png` / preferred `annotated_frame_png` | Explorer / Solver (multimodal) | geometric intuition + labeled planning objects |
| Symbolic  | `full_grid_hex_rows` + probe summaries in `verifier_packet` | Coder / Verifier / offline contour | deterministic simulation and empiric deltas |

**Invariant:** both channels share the same **PlanningSet** (object ids, relation ids, allowed actions, coordinate candidates, aliases) for one snapshot cycle. Component-graph ids remain geometry evidence only and are never trajectory targets.

---

## 1. Tri-Agent Separation of Powers & Memory Contours

To prevent LLM context degradation and role contamination the agent is split into three isolated LLM roles, each with a dedicated, non-overlapping memory contour.

### 1.1 Explorer Agent (Call 1 family)

- **Role**: Discover the mechanics of the current level / game. Test simple actions and hypothesize coordinate targets (ACTION6 and other coordinate carriers). Produce a strict factual Environment Specification.
- **Memory Contour**: `EnvironmentSpecMemory`. Stores only verified or high-confidence facts (e.g. “ACTION1 produces positive row delta on objects of class X”, “clicking centroid of obj_9 toggles color”, “ACTION3 is currently a no-op”). No planning language, no goal hypotheses, no trajectory fragments.
- **Input**: Current PlanningSet snapshot, annotated frame (optional), previously recorded action effects from the same game, list of already-tested coordinates.
- **Output**: Strict JSON `EnvironmentSpecification` (see Engineering Spec for schema). Never contains trajectory steps or goal statements.
- **Authority**: None over goals or final actions. May only expand the factual capability surface.

### 1.2 Coder Agent (Call 2 family)

- **Role**: Given an `EnvironmentSpecification`, emit a self-contained Python module that implements a typed, deterministic DSL for the current level (or game). The module exposes a fixed set of pure functions whose signatures are declared in a machine-readable manifest.
- **Memory Contour**: `SyntaxErrorMemory` (ontological / syntactic only). Contains the last prompt, the generated source, and any traceback / static-analysis failures. **The Coder never receives level goals, Solver hypotheses, EpistemicMemory contents, or environment transition judgments.**
- **Input**: `EnvironmentSpecification` + allowed primitive API contract + previous SyntaxErrorMemory entries for the same level.
- **Output**: Python source of `level_dsl.py` together with a JSON/AST function manifest (name, typed parameters, docstring, purity flags).
- **Authority**: None over game goals or trajectory selection. Its only success metric is that the generated module passes sandbox validation and exposes the declared manifest.

### 1.3 Solver Agent (Call 3 family)

- **Role**: Using only the *typed function manifest + docstrings* of the current DSL and the current ARGALite / PlanningSet graph, formulate one or more candidate trajectories expressed as sequences of DSL function calls.
- **Memory Contour**: `EpistemicMemory`. Stores Brusentsov judgments (FOLLOW / NULL / OMIT), trajectory signatures, expected-vs-observed propositions, and pivot points. **The Solver never sees Python source code, SyntaxErrorMemory, or raw tracebacks.**
- **Input**: Function manifest + docstrings, current PlanningSet / ARGALite snapshot, EpistemicMemory summary, remaining action budget.
- **Output**: JSON package of candidate trajectories (each a sequence of typed DSL calls with expected-effect propositions).
- **Authority**: Proposes only. Final authorization belongs exclusively to LayeredVerifier + GameSession.

### 1.4 Isolation Invariants (normative)

| ID  | Rule |
|-----|------|
| ISO-1 | Traceback, SyntaxError, TypeError or any Python exception text never appears in any prompt or memory visible to the Solver. |
| ISO-2 | Level goal, hypothesis family, success condition or EpistemicMemory content never appears in any prompt or memory visible to the Coder. |
| ISO-3 | Explorer writes exclusively to EnvironmentSpecMemory; Coder writes exclusively to SyntaxErrorMemory; Solver writes exclusively to EpistemicMemory. |
| ISO-4 | GameSession is the only component allowed to read all three contours and to route feedback according to the Double-Loop rules. |
| ISO-5 | PlanningSet object / relation / action ids are the sole shared vocabulary; no agent may invent ids outside the current PlanningSet. |

Violations of ISO-* are hard architectural defects and must be caught by isolation unit tests.

---

## 2. Double-Loop Learning & Feedback Routing

Errors are routed strictly to the agent responsible for the failure domain (Double-Loop Learning).

- **External Loop (Syntax / Execution Routing)**  
  If the Solver requests a DSL function and the sandboxed Python interpreter raises any exception, or if static validation of the generated module fails, the traceback and source are written *only* into SyntaxErrorMemory and the Coder is re-invoked (subject to `ARC_MAX_CODER_RETRIES`). The Solver is paused; its EpistemicMemory is left untouched by code-level diagnostics.

- **Internal Loop (Logical / Semantic Routing)**  
  If the DSL executes cleanly but the LayeredVerifier returns NULL or OMIT, the judgment (together with expected vs observed propositions) is written *only* into EpistemicMemory. The Coder is not penalized and does not receive the logical feedback. The Solver may later pivot on OMIT branches or abandon NULL branches.

This separation prevents the classic failure modes observed in earlier versions: syntax noise poisoning planning context, and logical dead-ends causing unnecessary DSL rewrites.

---

## 3. Brusentsov Ternary Logic in Transition Judgment

The `TransitionJudge` (part of LayeredVerifier) evaluates every real environment transition with Brusentsov’s necessary-implication semantics.

### 3.1 Core mapping

| Verifier Condition | Brusentsov State | Verdict | Queue / Branch Effect |
|--------------------|------------------|---------|-----------------------|
| Expected effect is *necessarily contained* in the observed state (exact match on the relevant propositions) | TRUE (1) | **FOLLOW** | Trajectory continues; step may be promoted toward confirmed rules. |
| Expected effect is *physically contradicted* (incompatibility / nullity violation) | FALSE (−1) | **NULL** | Hard rejection. Branch is severed. Signature recorded as contradiction in EpistemicMemory. |
| Expected effect did not occur, yet no physical laws, object-identity invariants or PlanningSet constraints were violated | IRRELEVANT (0) | **OMIT** | Branch is paused but kept alive in EpistemicMemory as a possible future growth point. Solver may later pivot. |

### 3.2 Necessary implication (architectural definition)

Following Brusentsov:

- Necessary implication \(x \Rightarrow y\) means “the essence of \(y\) is entirely contained in the essence of \(x\)” (or “all \(x\) are \(y\)”).
- In the improved ternary DNF this is expressed as \(xy \lor xy'_0 \lor x'y'\) where the term \(x'y\) is *omitted as inessential* and the index 0 denotes nullity / incompatibility.
- Material implication includes the extra term and is deliberately weaker; the agent uses the stronger necessary form for FOLLOW.

Operationally the TransitionJudge constructs a set of atomic propositions over the current PlanningSet (object attributes, relations, metric signs, action-surface flags, terminal flags). It then asks whether the expected proposition set is necessarily implied by the observed proposition set under the above semantics.

### 3.3 Atomic proposition families (normative)

The judge may only use propositions drawn from the following registered families (all grounded on PlanningSet ids):

- Object identity preservation / change
- Object attribute deltas (color, size, pattern, \ldots)
- Positional deltas (row / column / centroid signs)
- Relation existence / absence / error-metric sign
- Action-surface change (availability of actions or coordinate candidates)
- Terminal / win / score metadata
- Controllability / affordance flags previously recorded by Explorer

Raw pixel or unconstrained grid differences are never treated as atomic propositions.

### 3.4 Branch lifetime

- FOLLOW steps may extend an active trajectory.
- NULL steps permanently suppress the semantic signature for the current level (subject to new evidence that changes the PlanningSet surface).
- OMIT steps leave the signature alive; the Solver is permitted to re-use or adapt the branch when later evidence appears.

---

## 4. PlanningSet Identity Contract (retained & strengthened)

For snapshot \(S\), `PlanningSet` is an immutable vocabulary:

```text
snapshot_id, grid_hash, full_grid_hex_rows
object_ids, relation_ids
allowed_action_ids, allowed_coordinate_candidate_ids
object_real_to_alias / object_alias_to_real
objects, relations, coordinate_candidates
component_graph?   # evidence only
```

**Invariants (per planning cycle)**

| ID | Rule |
|----|------|
| I1 | `object_layer.objects[].id` ≡ PlanningSet.object_ids ≡ ids used by Explorer / Coder / Solver |
| I2 | Annotated frame labels ⊆ alias set; each label maps to one planning object bbox |
| I3 | packet / verifier / snapshot `grid_hash` agree |
| I4 | allowed actions and coordinate candidates agree with the current capability surface |
| I5 | Every DSL function argument that names an object, relation or coordinate must resolve inside the current PlanningSet |
| I6 | Goal specifications used by the Solver are expressed only over PlanningSet metrics or terminal flags; never invent a current-frame hash as goal |
| I7 | After any accepted environment step a new snapshot yields a new PlanningSet; active trajectories are re-bound or invalidated |
| I8 | Component-graph component ids are never legal targets of DSL functions or trajectories |

Violations of I3/I4 produce STALE identity logging; they do not by themselves generate a NULL judgment.

---

## 5. Lifecycle of the Tri-Agent Pipeline

GameSession owns the following ordered lifecycle (architectural view):

1. **Snapshot preparation**  
   Build ARGALite snapshot → PlanningSet → dual-view media (annotated PNG + verifier_packet).

2. **Explorer phase** (optional but recommended at level start and on significant surface expansion)  
   - May execute a bounded number of deterministic probes (simple actions + coordinate candidates).  
   - Produces / updates EnvironmentSpecification.  
   - Writes only to EnvironmentSpecMemory.  
   - Consumes environment-action budget.

3. **Coder phase**  
   - Invoked when no valid DSL manifest exists for the current EnvironmentSpecification, or after a SyntaxErrorMemory entry.  
   - Generates `level_dsl.py` + function manifest.  
   - Must pass sandbox validation before the manifest is published to the Solver.  
   - Limited by `ARC_MAX_CODER_RETRIES`.

4. **Solver phase**  
   - Receives the published function manifest (docstrings + typed signatures only) + current PlanningSet + EpistemicMemory summary.  
   - Emits a package of candidate trajectories (JSON sequences of DSL calls).  
   - Limited by `ARC_MAX_SOLVER_RETRIES` and package size limits.

5. **Bind → Sandbox-execute → Observe → Judge**  
   - VerificationBinder grounds every DSL call against the current PlanningSet.  
   - ActionBoundary is the only component that may call the real environment.  
   - After each real step the LayeredVerifier applies Brusentsov logic and writes the result exclusively into EpistemicMemory.

6. **Routing**  
   - Exception → External Loop → Coder.  
   - NULL / OMIT / FOLLOW → Internal Loop → Solver (via EpistemicMemory).

7. **Fallback**  
   When Coder retries are exhausted or Solver makes no progress under budget, GameSession switches to a pure-symbolic or fixed-primitive trajectory path (the exact fallback operators are an engineering concern but must exist).

8. **Level / Game transitions**  
   - Level change: EnvironmentSpecMemory and EpistemicMemory may be summarized into GameMemory; SyntaxErrorMemory is cleared or heavily compacted.  
   - Game change: all three contours are reset.

---

## 6. Budgets & Hard Limits (architectural)

The following limits are normative and may not be exceeded by configuration in the reference implementation:

- Maximum environment actions per level / game (competition-defined).
- `ARC_MAX_CODER_RETRIES` — hard ceiling on Coder invocations per level after which fallback is mandatory.
- `ARC_MAX_SOLVER_RETRIES` — hard ceiling on Solver package generations per level.
- Bounded number of Explorer probe actions per level.
- One-step execution by default: even if a trajectory contains multiple DSL calls, only the next verified call is emitted; the remainder stays provisional.

LLM call counts are secondary to environment-action counts for RHAE scoring, yet wall-clock and Kaggle quota remain real constraints; therefore the architecture forbids unbounded retry loops.

---

## 7. Sandbox & Deterministic Execution (architectural requirements)

Any Python generated by the Coder:

- must be executed only inside a restricted sandbox that exposes a minimal, pure API (PlanningSet queries, typed object/relation accessors, metric evaluators, and the ability to *declare* an intended environment action);
- may not perform I/O, network, arbitrary `exec`/`eval`, or import of unsafe modules;
- must be statically checked for the declared function manifest before any trajectory may reference it;
- must be deterministic given the same PlanningSet snapshot.

The sandbox is an architectural necessity, not an optional engineering detail. Failure to enforce it is a critical security and reproducibility defect.

---

## 8. Acceptance Criteria (Architecture)

1. The three agents (Explorer, Coder, Solver) are strictly separated by role and by memory contour; isolation invariants ISO-1 \ldots ISO-5 hold.
2. Tracebacks and syntax diagnostics never appear in Solver prompts or EpistemicMemory.
3. Level goals and epistemic judgments never appear in Coder prompts or SyntaxErrorMemory.
4. Empiric evaluation uses Brusentsov ternary logic and correctly classifies passive / insignificant outcomes as OMIT while severing only true contradictions as NULL.
5. All DSL functions and trajectory steps are grounded on the current PlanningSet; no invented identifiers are accepted.
6. Dual-view media share one PlanningSet per snapshot cycle.
7. GameSession is the sole orchestrator of agent calls, memory mutations, budgets and fallbacks.
8. Dynamic Python is always sandboxed and validated before use.
9. Exhaustion of Coder or Solver retries forces a documented pure-symbolic / fixed-primitive fallback.
10. Every emitted environment action is auditable through the chain: PlanningSet → agent outputs → Binding → Verifier judgment → ActionBoundary.
11. The design remains offline-compatible with a fixed local Qwen / vLLM stack and competition gateway contracts.

---

## 9. Relation to Prior Versions

- **From V6**: high-assurance isolation, three-valued judgment, verifier as final authority, strict budgets, GameMemory across levels.
- **From V9**: PlanningSet identity, dual-view media, empiric post-step authority, repair-friendly treatment of incomplete knowledge (now expressed via OMIT).
- **New in V10**: explicit Tri-Agent hierarchy, Double-Loop routing, Brusentsov formalization of ternary logic, dynamic yet sandboxed DSL generation, and hard memory-contour isolation that prevents the contamination failures observed in intermediate versions.

Version 10.0 is therefore a synthesis that restores the safety posture of V6 while solving the “DSL for unknown environment” problem that originally forced the move toward V9-style trajectory proposal.
