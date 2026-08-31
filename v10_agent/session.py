"""
Game Session Module for V10 Agent.
Orchestrates the main execution loop with strict isolation invariants.
Ref: Spec 5 (Execution Loop), Spec 6 (Error Handling)

ISO Invariants enforced:
- ISO-1: Explorer has no goal information
- ISO-2: Coder cannot see traceback from Solver
- ISO-3: Memory contours are strictly disjoint
- ISO-4: Solver never sees Python source code
- ISO-5: PlanningSet bijection maintained

CRITICAL: reasoning_trace from LLM responses is NEVER passed to agents.
It goes ONLY to logging for audit purposes. This prevents:
- Goal leakage to Explorer (ISO-1)
- Traceback exposure to Coder (ISO-2)  
- Python source exposure to Solver (ISO-3)
"""

import logging
import hashlib
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

from .config import V10Config
from .llm_client import VLLMClient, AgentRole, ParsedResponse
from .memory_contours import EnvironmentSpecMemory, SyntaxErrorMemory, EpistemicMemory
from .agents.explorer_agent import ExplorerAgent
from .agents.dsl_coder import DSLCoder
from .agents.solver_agent import SolverAgent


logger = logging.getLogger(__name__)


@dataclass
class GameSessionConfig:
    """Configuration for GameSession."""
    max_steps_per_level: int = 50
    enable_logging: bool = True
    max_coder_retries: int = 3
    max_solver_retries: int = 4


class GameSession:
    """
    Main orchestration class for ARC-AGI-3 game sessions.
    
    Implements the double-loop learning architecture:
    - Outer loop: Level progression
    - Inner loop: Hypothesis testing with Explorer/Coder/Solver
    
    ISO Invariants:
    - ISO-1: Explorer has no goal information
    - ISO-2: Coder cannot see traceback from Solver
    - ISO-3: Memory contours are strictly disjoint
    - ISO-4: Solver never sees Python source code
    - ISO-5: PlanningSet bijection maintained
    
    CRITICAL SECURITY INVARIANT:
    The VLLMClient returns ParsedResponse with:
    - payload: Clean JSON for agents (NO thinking traces)
    - reasoning_trace: Raw model reasoning (LOGGING ONLY)
    
    GameSession MUST only pass payload to agents, never reasoning_trace.
    This is enforced at the type level - agents accept Dict, not ParsedResponse.
    """
    
    def __init__(self, config: V10Config, llm_client: Optional[VLLMClient] = None):
        """
        Initialize GameSession.
        
        Args:
            config: V10Config instance with budget limits
            llm_client: VLLMClient instance for LLM calls (optional, creates default if None)
        """
        self.config = config
        self.llm_client = llm_client or VLLMClient(
            base_url=None,  # Uses default
            timeout=config.qwen_timeout_seconds,
            model_name=config.qwen_model_path or None
        )
        
        self.session_config = GameSessionConfig(
            max_coder_retries=config.max_coder_retries_per_level,
            max_solver_retries=config.max_solver_retries_per_level,
        )
        
        # Initialize agents with llm_client reference
        # NOTE: Agents receive only clean Dict payloads, never ParsedResponse
        self.explorer = ExplorerAgent(llm_client=self.llm_client)
        self.coder = DSLCoder(llm_client=self.llm_client)
        self.solver = SolverAgent(llm_client=self.llm_client)
        
        # Memory contours (strictly isolated per ISO-3)
        self.env_spec_memory: Optional[EnvironmentSpecMemory] = None
        self.syntax_error_memory: Optional[SyntaxErrorMemory] = None
        self.epistemic_memory: Optional[EpistemicMemory] = None
        
        # State tracking
        self.current_level_id: Optional[str] = None
        self.step_count: int = 0
        self.action_history: List[Dict[str, Any]] = []
        
        # Retry counters (reset per level)
        self.coder_retry_count: int = 0
        self.solver_retry_count: int = 0
        
        # LLM call counter for monitoring
        self.llm_call_count: int = 0
    
    def _log_reasoning_trace(self, role: AgentRole, reasoning_trace: Optional[str]) -> None:
        """
        Log reasoning trace for audit purposes ONLY.
        
        CRITICAL: This method ensures reasoning traces NEVER reach agents.
        They go ONLY to structured logging for debugging/audit.
        
        Ref: ISO-1...ISO-5 Isolation Invariants
        
        Args:
            role: Agent role that generated the reasoning
            reasoning_trace: Raw model reasoning (NEVER passed to other agents)
        """
        if reasoning_trace and self.session_config.enable_logging:
            logger.info(
                f"[AUDIT] {role.value.upper()} reasoning trace ({len(reasoning_trace)} chars)",
                extra={
                    "role": role.value,
                    "reasoning_length": len(reasoning_trace),
                    "level_id": self.current_level_id,
                    "step": self.step_count
                }
            )
            # Debug log contains full trace (only in debug mode)
            logger.debug(f"[AUDIT] {role.value.upper()} full trace: {reasoning_trace}")
    
    def _call_llm_with_isolation(self, role: AgentRole, messages: List[Dict], 
                                  json_schema: Dict) -> Optional[Dict]:
        """
        Call LLM and return ONLY the clean payload, logging reasoning_trace.
        
        This is the PRIMARY enforcement point for thinking trace isolation.
        
        Flow:
        1. Call VLLMClient.generate() -> ParsedResponse
        2. Log reasoning_trace via _log_reasoning_trace() (audit only)
        3. Return payload if status=="OK", else handle error
        
        Ref: ISO-1...ISO-5 - reasoning_trace NEVER reaches agents
        
        Args:
            role: Agent role (determines temperature, thinking config)
            messages: Chat messages for LLM
            json_schema: JSON Schema for response validation
            
        Returns:
            Clean Dict payload if successful, None on error
        """
        self.llm_call_count += 1
        
        response = self.llm_client.generate(
            role=role,
            messages=messages,
            json_schema=json_schema
        )
        
        # CRITICAL: Log reasoning_trace BEFORE any agent interaction
        # This ensures it goes to logs ONLY, never to agents
        self._log_reasoning_trace(role, response.reasoning_trace)
        
        if response.status == "OK":
            return response.payload
        elif response.status == "PARSE_ERROR":
            logger.warning(
                f"LLM parse error for {role.value}: {response.error_message}",
                extra={"role": role.value, "level_id": self.current_level_id}
            )
            return None
        elif response.status == "TIMEOUT":
            logger.warning(
                f"LLM timeout for {role.value}: {response.error_message}",
                extra={"role": role.value, "level_id": self.current_level_id}
            )
            return None
        else:  # CONNECTION_ERROR
            logger.error(
                f"LLM connection error for {role.value}: {response.error_message}",
                extra={"role": role.value, "level_id": self.current_level_id}
            )
            return None
    
    def act(self, raw_observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute one step of the agent loop.
        
        Flow:
        1. Prepare PlanningSet snapshot
        2. Explorer (if environment unknown) -> receives ONLY payload
        3. Coder (if manifest needed) -> receives ONLY payload
        4. Solver (generate candidates) -> receives ONLY payload
        5. Select candidate via PolicyEngine
        6. Verify arguments via VerificationBinder
        7. Execute ONE step via ActionBoundary
        
        ISO Invariant Enforcement:
        - Explorer receives NO goal info (ISO-1)
        - Coder receives NO traceback (ISO-2)
        - Solver receives NO Python source (ISO-3)
        
        Args:
            raw_observation: Current grid state from environment
            
        Returns:
            Action to execute, or None if no valid action found
        """
        try:
            # Step 1: Explorer phase (if needed)
            explorer_payload = self._run_explorer(raw_observation)
            
            if explorer_payload is None:
                # Explorer failed - use fallback or previous env spec
                logger.warning("Explorer failed, using fallback")
                explorer_payload = self._explorer_fallback(raw_observation)
            
            # Step 2: Coder phase (generate/update DSL functions)
            coder_payload = self._run_coder(explorer_payload)
            
            if coder_payload is None:
                # Coder failed - trigger double-loop retry
                logger.warning("Coder failed, triggering retry loop")
                coder_payload = self._run_coder_with_retry(explorer_payload)
                
                if coder_payload is None:
                    # Exhausted retries - force symbolic fallback
                    logger.error("Coder exhausted retries, forcing fallback")
                    return self._symbolic_fallback(raw_observation)
            
            # Step 3: Solver phase (generate trajectory candidates)
            solver_payload = self._run_solver(coder_payload)
            
            if solver_payload is None:
                # Solver failed - retry or fallback
                logger.warning("Solver failed, attempting retry")
                solver_payload = self._run_solver_with_retry(coder_payload)
                
                if solver_payload is None:
                    # Exhausted retries - symbolic fallback
                    logger.error("Solver exhausted retries, forcing fallback")
                    return self._symbolic_fallback(raw_observation)
            
            # Step 4: Select best candidate and extract first action
            # (Full implementation would use PolicyEngine here)
            selected_action = self._select_action_from_candidates(solver_payload)
            
            return selected_action
            
        except Exception as e:
            logger.exception(f"Unexpected error in act(): {e}")
            return self._symbolic_fallback(raw_observation)
    
    def _run_explorer(self, observation: Dict) -> Optional[Dict]:
        """
        Run Explorer agent with isolated LLM call.
        
        Ref: ISO-1 - Explorer receives NO goal information
        
        Args:
            observation: Current environment observation
            
        Returns:
            Explorer payload (probes, env_spec) or None on error
        """
        try:
            # Build PlanningSet snapshot from raw observation
            planning_set = self._build_planning_set_from_observation(observation)
            
            # Create annotated frame placeholder (full impl would render PNG)
            annotated_frame = self._create_annotated_frame(observation)
            
            # Call Explorer agent with proper interface
            explorer_response = self.explorer.act(
                planning_set=planning_set,
                annotated_frame=annotated_frame,
                action_history=self.action_history[-10:],  # Last 10 actions
                probe_history=None,  # Full impl would pass probe history
            )
            
            return explorer_response
            
        except Exception as e:
            logger.exception(f"Explorer agent failed: {e}")
            return None
    
    def _build_planning_set_from_observation(self, observation: Dict) -> "PlanningSet":
        """
        Build PlanningSet snapshot from raw observation.
        
        Args:
            observation: Raw grid observation dict
            
        Returns:
            PlanningSet instance for Explorer agent
        """
        from .planning_set import PlanningSet
        
        # Extract grid and compute hash
        grid = observation.get("grid", [])
        grid_str = str(grid)
        grid_hash = hashlib.sha256(grid_str.encode()).hexdigest()
        
        # Generate snapshot ID
        snapshot_id = f"snapshot_{self.current_level_id}_{self.step_count}" if self.current_level_id else f"snapshot_{self.step_count}"
        
        # Stub object/relation/action IDs as frozensets (required by PlanningSet)
        object_ids = frozenset(["obj_0", "obj_1"])  # Default stub objects
        relation_ids = frozenset(["rel_0"])  # Default stub relations
        allowed_action_ids = frozenset(["ACTION1", "ACTION6"])  # Default actions
        
        # Identity mapping for object aliases (I6 bijection requirement)
        from frozendict import frozendict
        object_real_to_alias = frozendict({oid: oid for oid in object_ids})
        
        return PlanningSet(
            snapshot_id=snapshot_id,
            grid_hash=grid_hash,
            object_ids=object_ids,
            relation_ids=relation_ids,
            allowed_action_ids=allowed_action_ids,
            object_real_to_alias=object_real_to_alias,
        )
    
    def _create_annotated_frame(self, observation: Dict) -> str:
        """
        Create annotated frame representation for Explorer.
        
        Args:
            observation: Raw grid observation dict
            
        Returns:
            Annotated frame string (base64 PNG or text representation)
        """
        # Stub implementation - returns text representation
        # Full impl would render PNG with object/relation overlays
        return f"Grid observation at step {self.step_count}: {str(observation)[:200]}..."
    
    def _run_coder(self, explorer_payload: Dict) -> Optional[Dict]:
        """
        Run Coder agent with isolated LLM call.
        
        Ref: ISO-2 - Coder receives NO traceback, only error summaries
        Ref: ISO-4 - Coder sees syntax errors but NOT Solver's Python traceback
        
        Args:
            explorer_payload: Clean Explorer output (probes, reasoning)
            
        Returns:
            Coder payload (source_code, function_names) or None on error
        """
        try:
            # Build EnvironmentSpecification from explorer payload
            env_spec = self._build_environment_spec(explorer_payload)
            
            # Build API manifest (proper dict structure, not list)
            # Extract function names from probes or use default DSL functions
            probe_list = explorer_payload.get("probes", [])
            api_manifest = {
                "functions": {
                    f"probe_{i}": {
                        "signature": f"probe_{i}(x, y)",
                        "docstring": f"Probe action at coordinates",
                        "parameters": {"x": "int", "y": "int"},
                        "return_type": "EffectDeclaration"
                    }
                    for i in range(len(probe_list))
                } if probe_list else {
                    "default_probe": {
                        "signature": "default_probe(x, y)",
                        "docstring": "Default probe action",
                        "parameters": {"x": "int", "y": "int"},
                        "return_type": "EffectDeclaration"
                    }
                }
            }
            
            # Get error summaries from SyntaxErrorMemory (NOT full tracebacks)
            error_summaries = []
            if self.syntax_error_memory:
                error_summaries = self.syntax_error_memory.get_recent_summaries(limit=5)
            
            # Convert to SyntaxErrorRecord format for Coder
            from .types import SyntaxErrorRecord
            recent_errors: Optional[List[SyntaxErrorRecord]] = None
            if error_summaries:
                recent_errors = [
                    SyntaxErrorRecord(
                        level_id=self.current_level_id or "unknown",
                        prompt_hash=f"hash_{i}",
                        source_hash=f"src_{i}",
                        summary=err.get("summary", str(err)),
                        timestamp=i
                    )
                    for i, err in enumerate(error_summaries)
                ]
            
            # Call Coder agent with proper interface
            coder_response = self.coder.act(
                environment_spec=env_spec,
                api_manifest=api_manifest,
                recent_errors=recent_errors,
            )
            
            return coder_response
            
        except Exception as e:
            logger.exception(f"Coder agent failed: {e}")
            return None
    
    def _build_environment_spec(self, explorer_payload: Dict) -> "EnvironmentSpecification":
        """
        Build EnvironmentSpecification from Explorer payload.
        
        Args:
            explorer_payload: Raw Explorer output dict
            
        Returns:
            EnvironmentSpecification instance for Coder agent
        """
        from .types import EnvironmentSpecification, ObjectSpec, RelationSpec
        
        # Extract from explorer payload (stub implementation)
        grid_width = explorer_payload.get("grid_width", 10)
        grid_height = explorer_payload.get("grid_height", 10)
        
        # Stub object/relation specs (full impl extracts from probes)
        object_specs: List[ObjectSpec] = []
        relation_specs: List[RelationSpec] = []
        
        return EnvironmentSpecification(
            grid_width=grid_width,
            grid_height=grid_height,
            object_specs=object_specs,
            relation_specs=relation_specs,
            action_surface_type="grid",
            allowed_actions=["ACTION1", "ACTION6"],
        )
    
    def _run_coder_with_retry(self, explorer_payload: Dict) -> Optional[Dict]:
        """
        Retry Coder with updated error context (double-loop).
        
        Ref: Spec 6 - Error Handling with max_coder_retries
        
        Flow:
        1. Check if retries exhausted
        2. Update SyntaxErrorMemory with latest error
        3. Retry Coder with error context
        
        Args:
            explorer_payload: Environment spec from Explorer
            
        Returns:
            Coder payload or None if retries exhausted
        """
        while self.coder_retry_count < self.session_config.max_coder_retries:
            self.coder_retry_count += 1
            logger.info(
                f"Coder retry {self.coder_retry_count}/{self.session_config.max_coder_retries}",
                extra={"level_id": self.current_level_id}
            )
            
            result = self._run_coder(explorer_payload)
            if result is not None:
                # Success - reset counter
                self.coder_retry_count = 0
                return result
        
        # Retries exhausted
        return None
    
    def _run_solver(self, coder_payload: Dict) -> Optional[Dict]:
        """
        Run Solver agent with isolated LLM call.
        
        Ref: ISO-3 - Solver NEVER sees Python source code
        Ref: ISO-4 - Solver sees function manifest, NOT implementation
        
        Args:
            coder_payload: Clean Coder output (source_code, function_names, function_manifest)
            
        Returns:
            Solver payload (candidates) or None on error
        """
        try:
            # Build function manifest from coder payload
            # ISO-3: Extract only manifest with signatures/docstrings, never pass source_code to Solver
            # Priority: Use function_manifest from CoderResponse if available, otherwise build from function_names
            if "function_manifest" in coder_payload and coder_payload["function_manifest"]:
                function_manifest = {"functions": coder_payload["function_manifest"]}
            else:
                # Fallback: build minimal manifest from function_names list
                function_manifest = {
                    "functions": {
                        name: {
                            "signature": f"{name}()",
                            "docstring": f"DSL function {name}",
                            "parameters": {},
                            "return_type": "EffectDeclaration"
                        }
                        for name in coder_payload.get("function_names", [])
                    }
                }
            
            # Get epistemic summary (Brusentsov judgments from previous steps)
            epistemic_summary = None
            if self.epistemic_memory:
                epistemic_summary = self.epistemic_memory.get_summary()
            
            # Get live/omit branches from EpistemicMemory
            live_omit_branches = None
            severed_null_signatures = None
            if self.epistemic_memory:
                live_omit_branches = self.epistemic_memory.get_live_omit_branches(limit=5)
                severed_null_signatures = self.epistemic_memory.get_severed_null_signatures(limit=5)
            
            # Call Solver agent with proper interface
            solver_response = self.solver.act(
                function_manifest=function_manifest,
                epistemic_summary=epistemic_summary,
                live_omit_branches=live_omit_branches,
                severed_null_signatures=severed_null_signatures,
            )
            
            return solver_response
            
        except Exception as e:
            logger.exception(f"Solver agent failed: {e}")
            return None
    
    def _run_solver_with_retry(self, coder_payload: Dict) -> Optional[Dict]:
        """
        Retry Solver with updated epistemic context.
        
        Ref: Spec 6 - Error Handling with max_solver_retries
        
        Args:
            coder_payload: Function manifest from Coder
            
        Returns:
            Solver payload or None if retries exhausted
        """
        while self.solver_retry_count < self.session_config.max_solver_retries:
            self.solver_retry_count += 1
            logger.info(
                f"Solver retry {self.solver_retry_count}/{self.session_config.max_solver_retries}",
                extra={"level_id": self.current_level_id}
            )
            
            result = self._run_solver(coder_payload)
            if result is not None:
                # Success - reset counter
                self.solver_retry_count = 0
                return result
        
        # Retries exhausted
        return None
    
    def _explorer_fallback(self, observation: Dict) -> Dict:
        """Fallback Explorer when LLM fails."""
        logger.warning("Using Explorer fallback")
        return {
            "probes": [{"x": 0, "y": 0, "confidence": 0.5}],
            "reasoning": "Fallback: minimal probe"
        }
    
    def _symbolic_fallback(self, observation: Dict) -> Optional[Dict[str, Any]]:
        """
        Symbolic fallback when all agents fail.
        
        Ref: Spec 6 - Fallback Configuration
        
        Returns:
            Minimal probe action
        """
        logger.warning("Using symbolic fallback")
        return {"action": "PROBE", "args": {"x": 0, "y": 0}}
    
    def _select_action_from_candidates(self, solver_payload: Dict) -> Optional[Dict[str, Any]]:
        """
        Select best action from Solver candidates.
        
        Full implementation would use PolicyEngine with confidence scoring.
        
        Args:
            solver_payload: Solver output with candidates
            
        Returns:
            First action of best candidate, or None
        """
        candidates = solver_payload.get("candidates", [])
        if not candidates:
            return None
        
        # Simple selection: highest confidence
        best = max(candidates, key=lambda c: c.get("confidence", 0.0))
        steps = best.get("steps", [])
        
        if not steps:
            return None
        
        # Return first step only (execute_one_step_at_a_time invariant)
        first_step = steps[0]
        return {
            "action": first_step.get("function", "PROBE"),
            "args": first_step.get("args", {})
        }
    
    def _extract_propositions_from_observation(
        self,
        observation: Dict[str, Any],
        snapshot_hash: str,
    ) -> List["AtomicProposition"]:
        """
        Extract AtomicPropositions from raw grid observation.
        
        This is a stub implementation that extracts basic propositions.
        Full implementation would use SnapshotBuilder to extract all
        registered proposition families per Spec 3.3.
        
        Args:
            observation: Raw grid observation dict with 'grid' key
            snapshot_hash: Hash of the snapshot for PropositionSet context
            
        Returns:
            List of AtomicProposition instances representing observed state
        
        Ref: Spec 3.3 - Atomic proposition families (normative)
        """
        from .types import AtomicProposition
        
        propositions: List[AtomicProposition] = []
        grid = observation.get("grid", [])
        
        if not grid:
            return propositions
        
        # Extract grid dimensions as positional context
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        
        # Stub proposition: grid dimensions (attribute_delta family)
        propositions.append(
            AtomicProposition(
                family="attribute_delta",
                data={
                    "attribute": "grid_dimensions",
                    "object_id": "grid_main",
                    "delta_width": width,
                    "delta_height": height,
                },
                objects=("grid_main",),
                relations=(),
            )
        )
        
        # Count non-zero cells as a simple metric_sign proposition
        non_zero_count = sum(1 for row in grid for cell in row if cell != 0)
        propositions.append(
            AtomicProposition(
                family="metric_sign",
                data={
                    "metric": "non_zero_cells",
                    "sign": 1 if non_zero_count > 0 else 0,
                    "count": non_zero_count,
                },
                objects=("grid_main",),
                relations=(),
            )
        )
        
        # Stub terminal flag (assume non-terminal unless grid is empty)
        propositions.append(
            AtomicProposition(
                family="terminal_flag",
                data={"is_terminal": False, "reason": "active_level"},
                objects=("grid_main",),
                relations=(),
            )
        )
        
        logger.debug(
            f"Extracted {len(propositions)} atomic propositions from observation",
            extra={"snapshot_hash": snapshot_hash, "grid_size": f"{width}x{height}"}
        )
        
        return propositions
    
    def observe_action_result(
        self,
        after_observation: Dict[str, Any],
        expected_propositions: Optional[List["AtomicProposition"]] = None,
        action_sequence: Optional[tuple] = None,
        sandbox_exception: Optional[Exception] = None,
    ) -> None:
        """
        Process the result of an executed action with double-loop learning.
        
        This method implements the core of the double-loop learning architecture:
        - Inner loop: Hypothesis testing via LayeredVerifier
        - Outer loop: Memory contour updates for cross-level learning
        
        Flow:
        1. IF sandbox_exception occurred:
           - Create SyntaxErrorRecord and add to SyntaxErrorMemory
           - Trigger Coder retry path (handled by caller)
        
        2. IF no exception:
           a. Build PropositionSet from observed state (after_observation)
           b. Build PropositionSet from expected effects (expected_propositions)
           c. Call LayeredVerifier.verify_transition()
           d. Route judgment to EpistemicMemory:
              - FOLLOW (TRUE): Continue current trajectory branch
              - NULL (FALSE): Record severed branch signature
              - OMIT (IRRELEVANT): Record live omit branch for potential pivot
        
        ISO Invariants Enforced:
        - ISO-3: Memory contours remain strictly disjoint
        - SyntaxErrorMemory receives only error summaries (no Python tracebacks visible to Solver)
        - EpistemicMemory receives only Brusentsov judgments (no source code)
        
        Args:
            after_observation: Grid state after action execution
            expected_propositions: Expected AtomicPropositions from DSL function's EffectDeclaration
            action_sequence: Sequence of action IDs for branch signature tracking
            sandbox_exception: Exception raised during sandbox execution (if any)
        
        Returns:
            None (side effects: updates memory contours)
        
        Ref: Spec 5 - LayeredVerifier Contract
        Ref: Spec 3.5 - Memory Contours (EpistemicMemory, SyntaxErrorMemory)
        Ref: Spec 3.1 - Brusentsov Ternary Logic
        """
        import time
        
        # =========================================================================
        # Path A: Sandbox Exception -> SyntaxErrorMemory
        # =========================================================================
        if sandbox_exception is not None:
            logger.info(
                f"Sandbox exception detected, recording to SyntaxErrorMemory",
                extra={
                    "level_id": self.current_level_id,
                    "step": self.step_count,
                    "exception_type": type(sandbox_exception).__name__,
                }
            )
            
            if self.syntax_error_memory is None:
                # Lazy initialization on first error
                from .memory_contours import SyntaxErrorMemory
                object.__setattr__(self, 'syntax_error_memory', SyntaxErrorMemory())
            
            # Create prompt/source hashes for error record
            # (In full impl, these would come from Coder's last generation)
            prompt_hash = hashlib.sha256(b"stub_prompt").hexdigest()
            source_hash = hashlib.sha256(b"stub_source").hexdigest()
            
            # Capture traceback string (NEVER passed to Solver - ISO-2)
            import traceback
            traceback_str = traceback.format_exception(type(sandbox_exception), sandbox_exception, sandbox_exception.__traceback__)
            traceback_text = "".join(traceback_str)
            
            # Add to SyntaxErrorMemory (max 5 entries, FIFO eviction)
            self.syntax_error_memory = self.syntax_error_memory.add_error(
                level_id=self.current_level_id or "unknown",
                prompt_hash=prompt_hash,
                source_hash=source_hash,
                traceback=traceback_text,
                static_diagnostics=[f"SandboxException: {type(sandbox_exception).__name__}"],
                timestamp=int(time.time()),
            )
            
            logger.info(
                f"SyntaxErrorMemory now contains {self.syntax_error_memory.error_count} errors",
                extra={"level_id": self.current_level_id}
            )
            return  # Early return - no verification possible with exception
        
        # =========================================================================
        # Path B: Normal Execution -> LayeredVerifier -> EpistemicMemory
        # =========================================================================
        
        # Initialize memory contours if not already created
        if self.env_spec_memory is None:
            from .memory_contours import EnvironmentSpecMemory, EnvironmentSpecification
            from .types import PropositionSet
            initial_spec = EnvironmentSpecification(
                spec_id=f"spec_{self.current_level_id or 'init'}",
                initial_propositions=PropositionSet.create(snapshot_hash="init"),
            )
            object.__setattr__(self, 'env_spec_memory', EnvironmentSpecMemory(current_spec=initial_spec))
        
        if self.epistemic_memory is None:
            from .memory_contours import EpistemicMemory
            object.__setattr__(self, 'epistemic_memory', EpistemicMemory())
        
        # -------------------------------------------------------------------------
        # Step 1: Build Observed PropositionSet from after_observation
        # -------------------------------------------------------------------------
        grid = after_observation.get("grid", [])
        grid_str = str(grid)
        observed_hash = hashlib.sha256(grid_str.encode()).hexdigest()
        
        # Extract atomic propositions from observed grid state
        # (Full impl would use SnapshotBuilder to extract all proposition families)
        observed_propositions = self._extract_propositions_from_observation(
            after_observation, observed_hash
        )
        
        from .types import PropositionSet
        observed_set = PropositionSet.create(
            snapshot_hash=observed_hash,
            propositions=observed_propositions,
            timestamp=self.step_count,
        )
        
        # -------------------------------------------------------------------------
        # Step 2: Build Expected PropositionSet from DSL function's EffectDeclaration
        # -------------------------------------------------------------------------
        if expected_propositions is None:
            # No expected propositions provided - create empty set
            # This represents an exploratory action with no specific prediction
            expected_set = PropositionSet.create(
                snapshot_hash=observed_hash,  # Same snapshot context
                propositions=[],
                timestamp=self.step_count,
            )
            logger.debug(
                "No expected propositions provided; using empty expected set",
                extra={"step": self.step_count}
            )
        else:
            expected_set = PropositionSet.create(
                snapshot_hash=observed_hash,
                propositions=expected_propositions,
                timestamp=self.step_count,
            )
        
        # -------------------------------------------------------------------------
        # Step 3: Call LayeredVerifier.verify_transition()
        # -------------------------------------------------------------------------
        from .judge import LayeredVerifier
        verifier = LayeredVerifier()
        
        verification_result = verifier.verify_transition(
            expected=expected_set,
            observed=observed_set,
            action_sequence=action_sequence,
        )
        
        # Log verification result for audit
        logger.info(
            f"LayeredVerifier judgment: {verification_result.judgment.verdict_name}",
            extra={
                "judgment": verification_result.judgment.value,
                "verdict": verification_result.judgment.verdict_name,
                "reasoning_length": len(verification_result.reasoning),
                "level_id": self.current_level_id,
                "step": self.step_count,
            }
        )
        logger.debug(
            f"Verification reasoning: {verification_result.reasoning}",
            extra={"level_id": self.current_level_id}
        )
        
        # -------------------------------------------------------------------------
        # Step 4: Route judgment to EpistemicMemory
        # -------------------------------------------------------------------------
        from .types import BrusentsovJudgment
        
        # Create BrusentsovJudgment record for EpistemicMemory
        judgment_record = verifier.create_judgment_record(
            verification_result,
            timestamp=float(time.time()),
        )
        
        # Add judgment to EpistemicMemory (automatically tracks live/severed branches)
        self.epistemic_memory = self.epistemic_memory.add_judgment(judgment_record)
        
        # Log memory contour state
        logger.info(
            f"EpistemicMemory updated: {self.epistemic_memory.judgment_count} judgments, "
            f"{self.epistemic_memory.live_omit_count} live OMIT branches, "
            f"{self.epistemic_memory.severed_null_count} severed NULL branches",
            extra={
                "level_id": self.current_level_id,
                "judgment_type": judgment_record.judgment_type,
            }
        )
        
        # -------------------------------------------------------------------------
        # Step 5: Branch effect handling (for caller's decision making)
        # -------------------------------------------------------------------------
        # The caller should check EpistemicMemory to determine next action:
        # - FOLLOW: Continue current trajectory
        # - NULL: Sever branch, trigger Solver retry with new candidate
        # - OMIT: Keep branch alive, may pivot later
        
        # Log actionable insight based on judgment
        if verification_result.judgment.is_null():
            logger.warning(
                "NULL judgment: Physical contradiction detected. "
                "Current trajectory branch must be severed.",
                extra={"level_id": self.current_level_id, "step": self.step_count}
            )
        elif verification_result.judgment.is_omit():
            logger.info(
                "OMIT judgment: Expected effects absent but no contradiction. "
                "Branch remains live for potential pivot.",
                extra={"level_id": self.current_level_id, "step": self.step_count}
            )
        else:  # FOLLOW
            logger.info(
                "FOLLOW judgment: Trajectory validated. Continue current branch.",
                extra={"level_id": self.current_level_id, "step": self.step_count}
            )
    
    def reset_level(self, level_id: str) -> None:
        """Reset session state for a new level."""
        self.current_level_id = level_id
        self.step_count = 0
        self.action_history = []
        self.coder_retry_count = 0
        self.solver_retry_count = 0
        # Note: Memory contours persist across levels within same game
        logger.info(f"Reset level: {level_id}")
    
    def reset_game(self) -> None:
        """Clear all memory contours for new game."""
        self.env_spec_memory = None
        self.syntax_error_memory = None
        self.epistemic_memory = None
        self.current_level_id = None
        self.step_count = 0
        self.action_history = []
        self.coder_retry_count = 0
        self.solver_retry_count = 0
        self.llm_call_count = 0
        logger.info("Game reset - all memory contours cleared")


if __name__ == "__main__":
    print("GameSession module loaded successfully")
    # Smoke test with stub config
    from .config import V10Config
    config = V10Config.from_env()
    session = GameSession(config)
    print(f"Session created with LLM client: {session.llm_client is not None}")
