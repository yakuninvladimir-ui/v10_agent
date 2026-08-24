# ARC-AGI-3 LCLD Agent
# Engineering Specification — Version 10.0
# (Tri-Agent Hierarchy, Isolated Memory Contours & Brusentsov Ternary Logic)

## 0. Engineering Objective

Implement a competition-ready ARC-AGI-3 agent that:

- ingests gateway / Arcade observations;
- builds deterministic ARGALite snapshots and a shared PlanningSet;
- produces dual-view media (annotated PNG + offline verifier_packet);
- orchestrates three isolated LLM agents (Explorer, Coder, Solver) under hard budgets;
- dynamically generates a level-specific Python DSL that is statically validated and executed only inside a restricted sandbox;
- routes syntax / execution failures exclusively to the Coder and logical / semantic outcomes exclusively to the Solver (Double-Loop);
- judges real transitions with Brusentsov ternary logic (FOLLOW / NULL / OMIT);
- maintains three strictly isolated memory contours;
- falls back to a pure-symbolic / fixed-primitive path when Coder or Solver retries are exhausted;
- runs under Kaggle offline constraints with a fixed Qwen FP8 / vLLM stack;
- remains fully replayable and auditable.

### 0.1 Ownership

```text
GameSession              = sole orchestration + mutable state owner
ExplorerAgent            = Call-1 family, writes EnvironmentSpecMemory
DSLCoder                 = Call-2 family, writes SyntaxErrorMemory
SolverAgent              = Call-3 family, writes EpistemicMemory
PlanningSet              = canonical id vocabulary for one snapshot cycle
LayeredVerifier          = Brusentsov ternary authority
VerificationBinder       = grounds DSL calls against PlanningSet
ActionBoundary           = only component allowed to call the real environment
SandboxExecutor          = restricted interpreter for generated DSL
MemoryContours           = EnvironmentSpecMemory | SyntaxErrorMemory | EpistemicMemory
```

All other modules are pure or close-to-pure functions over explicit arguments.

---

## 1. Repository Layout

Active package: `v10_agent/`

```text
v10_agent/
  __init__.py
  config.py
  types.py
  observe.py
  game_adapter.py
  action_adapter.py
  arga_lite.py
  planning_set.py
  frame_media.py              # PNG + annotated_frame_png
  verifier_packet.py

  # Tri-Agent
  explorer_agent.py           # Call 1 – factual EnvironmentSpecification
  dsl_coder.py                # Call 2 – dynamic Python DSL + manifest
  solver_agent.py             # Call 3 – trajectory packages over manifest
  prompt_builders/
    explorer_prompt.py
    coder_prompt.py
    solver_prompt.py

  # Logic & Memory
  brusentsov_logic.py         # Ternary enum + implies_brusentsov
  memory_contours.py          # three isolated stores + GameMemory summary
  verification.py             # VerificationBinder
  judge.py                    # LayeredVerifier integrating Brusentsov
  sandbox.py                  # restricted executor + static checks

  # Core execution
  trajectory.py
  session.py                  # GameSession
  policy.py
  logging.py
  fallback_symbolic.py

  IDENTITY_CONTRACT.md
  README.md

Competition embedding:
  kaggle_agent.py
  submission.py
  lcld_competition_child.py
```

Legacy paths from V6–V9 may remain only under explicit `legacy/` or `fallback/` markers and must not silently compete with the V10 path.

---

## 2. Configuration Contract

### 2.1 Required keys (V10Config)

```text
# LLM backend
llm_advisor_backend: vllm | ollama | llama_cli | fake
qwen_model_path / ARC_QWEN_VLLM_MODEL
qwen_context_tokens, qwen_max_input_tokens, qwen_max_output_tokens
qwen_temperature, qwen_seed, qwen_timeout_seconds
qwen_multimodal_enabled: bool

# Tri-agent budgets (normative ceilings)
max_coder_retries_per_level: int = 3
max_solver_retries_per_level: int = 4
max_explorer_probe_actions_per_level: int = 8
max_total_llm_calls_per_level: int          # soft monitoring only

# Trajectory / package limits
max_candidates_per_solver_package: int = 4
max_steps_per_candidate: int = 6
execute_one_step_at_a_time: bool = True

# Sandbox
sandbox_enabled: bool = True
sandbox_allowed_modules: list[str]          # minimal whitelist
sandbox_max_cpu_seconds: float
sandbox_max_memory_mb: int

# Memory
game_memory_reset_on_game_change: bool = True
game_memory_reset_on_level_change: bool = False
epistemic_memory_max_entries: int
syntax_error_memory_max_entries: int = 5

# Fallback
enable_symbolic_fallback: bool = True
coder_exhaustion_forces_fallback: bool = True
```

All numeric limits that affect competition scoring or safety must be readable from environment variables (`ARC_MAX_CODER_RETRIES`, \ldots). Hard-coded competing overrides are forbidden.

### 2.2 Competition defaults

- Backend: vLLM with fixed FP8 Qwen checkpoint.
- Multimodal: annotated frame preferred, at most one image payload.
- `execute_one_step_at_a_time = True`.
- Zero accepted gateway actions must not finalize an empty scorecard.

---

## 3. Core Data Types

### 3.1 Brusentsov Ternary (`brusentsov_logic.py`)

```python
class Ternary(Enum):
    TRUE = 1          # FOLLOW
    FALSE = -1        # NULL
    IRRELEVANT = 0    # OMIT

def implies_brusentsov(expected: PropositionSet, observed: PropositionSet) -> Ternary:
    """
    Returns:
      TRUE  if every expected atomic proposition is necessarily contained
            in the observed set (necessary implication).
      FALSE if any expected proposition is incompatible with the observed
            set (nullity / contradiction).
      IRRELEVANT if the expected set is not implied yet no incompatibility
            exists (the missing effects are inessential for the current
            judgment).
    """
```

Atomic propositions are drawn only from the registered families listed in the Architectural Spec §3.3 and are always grounded on PlanningSet identifiers.

### 3.2 EnvironmentSpecification (Explorer output)

```json
{
  "schema_version": "v10.env_spec.1",
  "snapshot_hash": "...",
  "planning_set_id": "...",
  "researched_actions": [
    {
      "action_id": "ACTION1",
      "effect_summary": "positive_row_delta on objects matching class X",
      "supporting_evidence_ids": ["probe_017"],
      "confidence": 0.82,
      "contradicted": false
    }
  ],
  "coordinate_affordances": [
    {
      "coordinate_candidate_id": "coord_003",
      "x": 18, "y": 11,
      "source": {"type": "object_centroid", "object_id": "obj_9"},
      "observed_effects": ["color_toggle"],
      "confidence": 0.71
    }
  ],
  "object_class_notes": [...],
  "action_surface_notes": [...],
  "invariants": ["object_identity_stable_under_ACTION2", ...]
}
```

No trajectory steps, no goal statements, no Python.

### 3.3 DSL Function Manifest (Coder output, visible to Solver)

```json
{
  "schema_version": "v10.dsl_manifest.1",
  "module_hash": "...",
  "functions": [
    {
      "name": "move_toward",
      "parameters": [
        {"name": "obj", "type": "planning_object_id"},
        {"name": "target", "type": "planning_object_id"},
        {"name": "metric", "type": "metric_id", "default": "centroid_distance"}
      ],
      "returns": "effect_declaration",
      "docstring": "Declare an intent to reduce the chosen metric between obj and target. Pure. Side-effect free until ActionBoundary materialises the underlying environment action.",
      "purity": "pure_declaration",
      "expected_effect_template": {
        "metric_delta_sign": -1,
        "object_ids": ["obj"]
      }
    }
  ]
}
```

The Solver sees only this manifest + docstrings. It never receives the Python source.

### 3.4 Trajectory Package (Solver output)

```json
{
  "schema_version": "v10.trajectory_package.1",
  "proposal_id": "...",
  "snapshot_hash": "...",
  "candidates": [
    {
      "trajectory_id": "traj_001",
      "steps": [
        {
          "step_id": "s1",
          "dsl_function": "move_toward",
          "arguments": {"obj": "obj_3", "target": "obj_7", "metric": "row_centroid_distance"},
          "expected_propositions": [
            {"family": "metric_sign", "metric": "row_centroid_distance", "sign": -1, "objects": ["obj_3", "obj_7"]}
          ],
          "abort_if": ["object_identity_changed", "metric_increased"]
        }
      ],
      "success_condition": {...},
      "confidence": 0.76
    }
  ]
}
```

### 3.5 Memory Contours

```python
@dataclass
class EnvironmentSpecMemory:
    game_id: str
    level_id: str
    specs: list[EnvironmentSpecification]
    probe_history: list[ProbeRecord]

@dataclass
class SyntaxErrorMemory:
    level_id: str
    entries: list[SyntaxErrorRecord]   # prompt_hash, source_hash, traceback, static_diagnostics

@dataclass
class EpistemicMemory:
    level_id: str
    judgments: list[BrusentsovJudgment]  # trajectory_sig, ternary, expected, observed, timestamp
    live_omit_branches: list[BranchSignature]
    severed_null_signatures: set[str]
```

GameMemory holds cross-level summaries (action effects, successful strategy families) and is the only store that survives level changes inside the same game.

---

## 4. Sandbox & Validation Pipeline (Coder output)

### 4.1 Static checks (must pass before manifest publication)

1. Source parses as valid Python 3.12+.
2. Only modules from the whitelist may be imported.
3. No use of `exec`, `eval`, `open`, `os`, `sys`, `subprocess`, `socket`, `__import__` of non-whitelist names, or file/network I/O.
4. Every function listed in the manifest exists, has the declared parameter names/types, and is free of free variables that escape the sandbox API.
5. All string literals that look like PlanningSet identifiers are checked against a supplied snapshot (optional but recommended).
6. Determinism flags and purity annotations are present.

### 4.2 Restricted executor

```python
class SandboxExecutor:
    def __init__(self, allowed_api: SandboxAPI, limits: SandboxLimits): ...
    def load_module(self, source: str, manifest: dict) -> SandboxedModule: ...
    def call(self, function_name: str, arguments: dict, planning_set: PlanningSet) -> EffectDeclaration: ...
```

`SandboxAPI` exposes only:

- read-only PlanningSet queries,
- typed object / relation accessors,
- metric evaluators,
- a pure `declare_environment_action(action_id, payload)` that returns an EffectDeclaration (the real `env.step` is performed later by ActionBoundary after Verifier approval).

Any exception inside the sandbox is captured, turned into a SyntaxErrorRecord, and routed solely to SyntaxErrorMemory.

### 4.3 Publication rule

A DSL module becomes visible to the Solver only after:

- static checks pass,
- a dry-run of every manifest function with dummy PlanningSet ids succeeds,
- the module hash is recorded,
- SyntaxErrorMemory for the current attempt is cleared or marked resolved.

---

## 5. implies_brusentsov — Engineering Realisation

```python
def implies_brusentsov(expected: PropositionSet, observed: PropositionSet) -> Ternary:
    # 1. Build incompatibility pairs (nullity)
    for e in expected:
        if any(contradicts(e, o) for o in observed):
            return Ternary.FALSE          # NULL

    # 2. Check necessary containment
    if all(is_necessarily_contained(e, observed) for e in expected):
        return Ternary.TRUE               # FOLLOW

    # 3. Otherwise the missing effects are treated as inessential
    return Ternary.IRRELEVANT             # OMIT
```

`contradicts` and `is_necessarily_contained` are defined per atomic-proposition family (object identity, attribute delta, metric sign, relation existence, action-surface flag, terminal flag). Families are registered in a single table; adding a new family requires an explicit engineering change and test.

Raw grid Hamming distance is never used as a proposition.

---

## 6. GameSession Lifecycle (detailed)

```text
act(raw_observation):
  1. prepare snapshot + PlanningSet + dual-view media
  2. if explorer budget remains and (no EnvSpec or surface expanded):
       run Explorer (may consume probe actions)
       write EnvironmentSpecMemory
  3. if no valid published DSL manifest:
       run Coder (subject to max_coder_retries)
       sandbox validate → publish manifest or record SyntaxError
  4. if published manifest exists and solver budget remains:
       run Solver → obtain trajectory package
  5. select next Candidate via policy (prefer live OMIT continuations, then new Solver candidates, then symbolic fallback)
  6. VerificationBinder.grounds(step, PlanningSet)
  7. emit pending action through ActionBoundary (one step only)

observe_action_result(after_observation):
  1. commit the transition exactly once
  2. build expected vs observed PropositionSets
  3. ternary = implies_brusentsov(...)
  4. write BrusentsovJudgment exclusively into EpistemicMemory
  5. if ternary == NULL: sever branch
     if ternary == OMIT: keep live_omit_branches
     if ternary == FOLLOW: advance cursor / promote
  6. invalidate PlanningSet cache for next cycle
  7. if exception occurred during sandbox call: route to SyntaxErrorMemory only
```

GAME_OVER path issues exactly one RESET before the next model call (competition invariant).

---

## 7. Isolation Enforcement & Tests

Required unit tests (must be present and green):

- `test_isolation_traceback_never_reaches_solver`
- `test_isolation_goal_never_reaches_coder`
- `test_isolation_memory_contours_are_disjoint`
- `test_coder_output_rejected_without_sandbox_validation`
- `test_solver_cannot_see_python_source`
- `test_explorer_cannot_emit_trajectories`
- `test_implies_brusentsov_follow_null_omit_cases` (exhaustive on registered families)
- `test_planning_set_ids_ground_all_dsl_arguments`
- `test_max_coder_retries_triggers_fallback`
- `test_max_solver_retries_triggers_fallback`
- `test_one_step_execution_even_for_multi_step_candidate`
- `test_dual_view_shares_planning_set`
- `test_game_change_clears_all_contours`
- `test_level_change_preserves_game_memory_summaries`

Integration smoke (synthetic, no GPU required):

```text
new level
→ Explorer produces EnvSpec
→ Coder produces validated DSL
→ Solver produces package
→ one step executed
→ Brusentsov judgment written only to EpistemicMemory
→ intentional sandbox exception → only SyntaxErrorMemory updated
→ intentional NULL → branch severed
→ intentional OMIT → branch remains live
→ coder retries exhausted → symbolic fallback activated
```

---

## 8. Prompt Construction Rules

- Explorer prompt: PlanningSet summary + annotated frame (optional) + prior game action effects + coordinate schema. No goal language.
- Coder prompt: EnvironmentSpecification + allowed primitive API + previous SyntaxErrorMemory entries for this level. Explicit instruction: “You have no information about the level goal; generate only a correct DSL.”
- Solver prompt: function manifest + docstrings + current ARGALite / PlanningSet + EpistemicMemory summary (FOLLOW / NULL / OMIT) + remaining budget. Explicit instruction: “You never see Python source; call only the listed functions.”

Under token pressure the same compaction priority as V9 is used (never drop valid ids, action surface, current judgments, or output schema). Prompt-tail priority places the schema and the most recent EpistemicMemory entries last.

---

## 9. Fallback Path

When `max_coder_retries` or `max_solver_retries` is reached, or when no valid DSL can be published:

1. GameSession switches to `fallback_symbolic.py`.
2. A fixed set of primitive operators (probe, move_toward_metric, click_centroid, reset, undo, \ldots) is used.
3. The same VerificationBinder + LayeredVerifier + Brusentsov path remains active.
4. EpistemicMemory continues to accumulate judgments so that a later recovery of a working DSL can still benefit from prior experience.

Fallback is mandatory, not optional, once the retry ceilings are hit.

---

## 10. Logging & Audit

Every LLM call records:

```text
game_id, level_id, agent_role, call_index, budget_before/after,
snapshot_hash, prompt_hash, raw_output_hash, parsed_status,
sandbox_validation_status (for Coder), ternary_result (for post-step)
```

Every emitted environment action records the full chain:

```text
PlanningSet hash → agent role → proposal id → binder result →
verifier ternary → ActionBoundary emission → resulting ObjectActionDiff
```

Traces are bounded and serialisable.

---

## 11. Acceptance Criteria (Engineering)

1. Complete round-trip of EnvironmentSpecification, DSL manifest and trajectory package schemas.
2. Isolation tests (ISO-1 \ldots ISO-5) pass.
3. Sandbox rejects any use of disallowed modules or I/O.
4. `implies_brusentsov` returns FOLLOW / NULL / OMIT correctly on the registered proposition families.
5. Coder retries are hard-capped; exhaustion forces symbolic fallback.
6. Solver retries are hard-capped; exhaustion forces symbolic fallback.
7. Tracebacks never appear in Solver prompts or EpistemicMemory.
8. Level goals never appear in Coder prompts or SyntaxErrorMemory.
9. All DSL arguments are grounded on the current PlanningSet.
10. Dual-view media share one PlanningSet per cycle.
11. One-step execution is the default.
12. Game change clears all three memory contours; level change preserves GameMemory summaries.
13. Zero accepted actions refuse successful finalization of an empty scorecard.
14. Structural Phase-A preflight passes without starting vLLM.
15. The whole pipeline remains offline-compatible with the fixed Qwen FP8 / vLLM stack.

---

## 12. Validation Commands

```bash
python -m compileall -q v10_agent
python -m pytest -q v10_agent/tests/test_isolation*.py
python -m pytest -q v10_agent/tests/test_brusentsov*.py
python -m pytest -q v10_agent/tests/test_sandbox*.py
python -m pytest -q v10_agent/tests/test_lifecycle*.py
python -m pytest -q v10_agent/tests/test_v10_end_to_end.py
```

After packaging, assert that the notebook payload contains the three agent modules, sandbox, brusentsov_logic, memory_contours and that no competing hard-coded sampling limits remain in the child process.

---

## 13. Final Engineering Invariant

```text
Three agents, three memory contours, zero contamination.

Explorer writes facts.
Coder writes syntax.
Solver writes epistemic judgments.

Syntax failures travel only to the Coder.
Logical outcomes travel only to the Solver.

Dynamic Python exists only inside the sandbox.
All identifiers come from the PlanningSet.
Brusentsov decides FOLLOW / NULL / OMIT.

When retries are exhausted the symbolic fallback is mandatory.
GameSession is the only owner of mutable state and of the routing decisions.
```
