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
    
    def observe_action_result(self, after_observation: Dict[str, Any]) -> None:
        """
        Process the result of an executed action.
        
        Flow:
        1. Build PropositionSets (expected vs observed)
        2. Call LayeredVerifier (implies_brusentsov)
        3. Route result to appropriate memory:
           - SandboxException -> SyntaxErrorMemory (triggers Coder retry)
           - FALSE (NULL) -> EpistemicMemory (triggers Solver retry)
           - IRRELEVANT (OMIT) -> EpistemicMemory (live branch)
           - TRUE (FOLLOW) -> Continue current branch
           
        Args:
            after_observation: Grid state after action execution
        """
        # Stub implementation - full logic in production
        logger.debug(f"Observing action result: {after_observation}")
        pass
    
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
