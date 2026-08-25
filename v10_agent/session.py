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

NOTE: reasoning_trace from ParsedResponse MUST ONLY go to logging.py (audit).
It physically cannot reach agent constructors or EpistemicMemory.
"""

import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass

from .llm_client import VLLMClient, AgentRole, ParsedResponse, ROLE_CONFIGS
from .agents.explorer_agent import ExplorerAgent
from .agents.dsl_coder import DSLCoder
from .agents.solver_agent import SolverAgent
from .types import EnvironmentSpecification, SyntaxErrorRecord, BrusentsovJudgment, BranchSignature
from .planning_set import PlanningSet


logger = logging.getLogger(__name__)


@dataclass
class GameSessionConfig:
    """Configuration for GameSession."""
    max_steps_per_level: int = 50
    enable_logging: bool = True
    max_coder_retries: int = 3
    max_solver_retries: int = 2
    max_explorer_retries: int = 2


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
    
    CRITICAL: This class enforces that reasoning_trace from LLM responses
    NEVER reaches agents or memory contours. It only goes to logging.audit().
    """
    
    def __init__(
        self, 
        config: Any, 
        vllm_client: Optional[VLLMClient] = None,
        session_config: Optional[GameSessionConfig] = None,
    ):
        """
        Initialize GameSession with VLLMClient integration.
        
        Args:
            config: V10Config instance
            vllm_client: VLLMClient instance for LLM calls (injected dependency)
            session_config: GameSessionConfig instance
        
        ISO Compliance Notes:
        - VLLMClient is stateless; all routing happens here
        - Agents receive ONLY clean payload dicts, never ParsedResponse
        - reasoning_trace is extracted and sent to logger ONLY
        """
        self.config = config
        self.vllm_client = vllm_client or VLLMClient()
        self.session_config = session_config or GameSessionConfig()
        
        # Initialize agents with shared VLLMClient
        # NOTE: Agents only receive role info, not full client
        self.explorer = ExplorerAgent(llm_client=self._create_role_client(AgentRole.EXPLORER))
        self.coder = DSLCoder(llm_client=self._create_role_client(AgentRole.CODER))
        self.solver = SolverAgent(llm_client=self._create_role_client(AgentRole.SOLVER))
        
        # Memory contours (strictly isolated per ISO-3)
        self.env_spec_memory: Optional[List] = None  # EnvironmentSpecMemory
        self.syntax_error_memory: Optional[List[SyntaxErrorRecord]] = None  # SyntaxErrorMemory
        self.epistemic_memory: Optional[List[BrusentsovJudgment]] = None  # EpistemicMemory
        
        # State tracking
        self.current_level_id: Optional[str] = None
        self.step_count: int = 0
        self.action_history: list = []
        
        # Retry counters for Double-Loop error handling
        self.coder_retry_count: int = 0
        self.solver_retry_count: int = 0
        self.explorer_retry_count: int = 0
    
    def _create_role_client(self, role: AgentRole) -> "_RoleScopedClient":
        """
        Create a role-scoped client wrapper that enforces isolation.
        
        This wrapper ensures that when an agent calls generate(), the response
        is automatically parsed and only the payload is returned. The 
        reasoning_trace is logged immediately and discarded.
        
        Ref: ISO-1, ISO-2, ISO-3, ISO-4, ISO-5
        
        Args:
            role: The agent role
            
        Returns:
            _RoleScopedClient instance for that role
        """
        return _RoleScopedClient(self.vllm_client, role, self.session_config.enable_logging)
    
    def act(self, raw_observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute one step of the agent loop.
        
        Flow:
        1. Prepare PlanningSet snapshot
        2. Explorer (if environment unknown) → receives ONLY payload
        3. Coder (if manifest needed) → receives ONLY payload
        4. Solver (generate candidates) → receives ONLY payload
        5. Select candidate via PolicyEngine
        6. Verify arguments via VerificationBinder
        7. Execute ONE step via ActionBoundary
        
        ISO Compliance:
        - Each agent receives ONLY clean dict (payload), never reasoning_trace
        - If status != "OK", retry logic triggers without exposing trace to other agents
        
        Args:
            raw_observation: Current grid state from environment
            
        Returns:
            Action to execute, or None if no valid action found
        """
        # Placeholder for full implementation
        # In production, this orchestrates the full double-loop
        logger.debug(f"GameSession.act() called with observation keys: {list(raw_observation.keys())}")
        
        # Example flow (simplified):
        # 1. Check if we need exploration
        # if self._needs_exploration():
        #     explorer_payload = self._call_explorer(...)
        #     if explorer_payload is None:
        #         return self._fallback_action()
        #
        # 2. Generate/update manifest via Coder
        # coder_payload = self._call_coder(...)
        # if coder_payload is None:
        #     # Double-Loop: record error in SyntaxErrorMemory, retry Coder
        #     return self._handle_coder_failure()
        #
        # 3. Get trajectory candidates from Solver
        # solver_payload = self._call_solver(...)
        # if solver_payload is None:
        #     # Timeout/error: fallback to symbolic
        #     return self._symbolic_fallback()
        
        return {"action": "PROBE", "args": {}}
    
    def _call_explorer(
        self,
        planning_set: PlanningSet,
        annotated_frame: str,
        probe_history: Optional[List] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Explorer agent with strict isolation.
        
        ISO-1: Explorer receives NO goal information.
        
        Args:
            planning_set: Current PlanningSet
            annotated_frame: Annotated frame (base64 or path)
            probe_history: Optional probe history
            
        Returns:
            Clean payload dict from Explorer, or None on failure
        """
        try:
            result = self.explorer.act(
                planning_set=planning_set,
                annotated_frame=annotated_frame,
                action_history=self.action_history,
                probe_history=probe_history,
            )
            self.explorer_retry_count = 0
            return result
        except Exception as e:
            self.explorer_retry_count += 1
            if self.explorer_retry_count >= self.session_config.max_explorer_retries:
                logger.error(f"Explorer failed after {self.explorer_retry_count} retries: {e}")
                return None
            logger.warning(f"Explorer retry {self.explorer_retry_count}/{self.session_config.max_explorer_retries}: {e}")
            return None
    
    def _call_coder(
        self,
        environment_spec: EnvironmentSpecification,
        api_manifest: Dict[str, Any],
        recent_errors: Optional[List[SyntaxErrorRecord]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Coder agent with strict isolation.
        
        ISO-2: Coder sees error summaries, NOT full tracebacks.
        ISO-3: Coder errors go to SyntaxErrorMemory only, NOT to Solver.
        
        Double-Loop Handling:
        - On PARSE_ERROR/TIMEOUT: record in SyntaxErrorMemory, trigger retry
        - Solver never sees that Coder failed; it only sees updated SyntaxErrorMemory
        
        Args:
            environment_spec: Environment specification
            api_manifest: DSL function manifest
            recent_errors: Recent syntax error records (summaries only)
            
        Returns:
            Clean payload dict from Coder, or None on failure
        """
        try:
            result = self.coder.act(
                environment_spec=environment_spec,
                api_manifest=api_manifest,
                recent_errors=recent_errors,
            )
            self.coder_retry_count = 0
            return result
        except Exception as e:
            self.coder_retry_count += 1
            if self.coder_retry_count >= self.session_config.max_coder_retries:
                logger.error(f"Coder failed after {self.coder_retry_count} retries: {e}")
                # Record failure fact in SyntaxErrorMemory (not raw error!)
                self._record_coder_failure_fact()
                return None
            logger.warning(f"Coder retry {self.coder_retry_count}/{self.session_config.max_coder_retries}: {e}")
            # Record error summary for next retry context
            self._record_syntax_error_summary(str(e))
            return None
    
    def _call_solver(
        self,
        function_manifest: Dict[str, Any],
        epistemic_summary: Optional[Dict[str, Any]] = None,
        live_omit_branches: Optional[List[BranchSignature]] = None,
        severed_null_signatures: Optional[List[BranchSignature]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Solver agent with strict isolation.
        
        ISO-4: Solver never sees Python source code.
        
        Error Handling:
        - On TIMEOUT/PARSE_ERROR: graceful fallback to symbolic, no crash
        - Solver does NOT know if Coder had errors (only sees manifest)
        
        Args:
            function_manifest: DSL function manifest
            epistemic_summary: Summary of previous Brusentsov judgments
            live_omit_branches: Live OMIT branches
            severed_null_signatures: Severed NULL branches
            
        Returns:
            Clean payload dict from Solver, or None on failure
        """
        try:
            result = self.solver.act(
                function_manifest=function_manifest,
                epistemic_summary=epistemic_summary,
                live_omit_branches=live_omit_branches,
                severed_null_signatures=severed_null_signatures,
            )
            self.solver_retry_count = 0
            return result
        except Exception as e:
            self.solver_retry_count += 1
            if self.solver_retry_count >= self.session_config.max_solver_retries:
                logger.error(f"Solver failed after {self.solver_retry_count} retries: {e}")
                return None
            logger.warning(f"Solver retry {self.solver_retry_count}/{self.session_config.max_solver_retries}: {e}")
            return None
    
    def _record_coder_failure_fact(self) -> None:
        """
        Record a coder failure fact in SyntaxErrorMemory.
        
        ISO-3 Compliance: Only stores hash and fact, NOT raw error text.
        This ensures Solver never sees "model broke and output garbage".
        """
        if self.syntax_error_memory is None:
            self.syntax_error_memory = []
        
        # Create minimal error record (no raw traceback!)
        error_record = SyntaxErrorRecord(
            level_id=self.current_level_id or "unknown",
            prompt_hash=f"coder_failure_{self.coder_retry_count}",
            source_hash="invalid",
            traceback=None,  # Explicitly no traceback
            static_diagnostics=[f"Coder failed after {self.coder_retry_count} attempts"],
            timestamp=self.step_count,
        )
        self.syntax_error_memory.append(error_record)
        logger.info(f"Recorded coder failure fact in SyntaxErrorMemory: {error_record.prompt_hash}")
    
    def _record_syntax_error_summary(self, error_summary: str) -> None:
        """
        Record a syntax error summary for Coder context.
        
        ISO-3: Only summary, not full traceback.
        """
        if self.syntax_error_memory is None:
            self.syntax_error_memory = []
        
        error_record = SyntaxErrorRecord(
            level_id=self.current_level_id or "unknown",
            prompt_hash=f"syntax_error_{len(self.syntax_error_memory)}",
            source_hash="pending",
            traceback=error_summary[:200],  # Truncate summary
            static_diagnostics=[],
            timestamp=self.step_count,
        )
        self.syntax_error_memory.append(error_record)
    
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
        
        ISO Compliance:
        - EpistemicMemory updates are NOT visible to Coder
        - SyntaxErrorMemory updates ARE visible to Coder (for retry context)
        
        Args:
            after_observation: Grid state after action execution
        """
        logger.debug(f"observe_action_result called with keys: {list(after_observation.keys())}")
        # Full implementation in production code
    
    def reset_level(self, level_id: str) -> None:
        """Reset session state for a new level."""
        self.current_level_id = level_id
        self.step_count = 0
        self.action_history = []
        self.coder_retry_count = 0
        self.solver_retry_count = 0
        self.explorer_retry_count = 0
        # Note: Memory contours persist across levels within same game
    
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
        self.explorer_retry_count = 0


class _RoleScopedClient:
    """
    Internal wrapper that scopes VLLMClient to a specific role.
    
    This wrapper enforces ISO isolation by:
    1. Automatically extracting payload from ParsedResponse
    2. Logging reasoning_trace immediately (never passing to agent)
    3. Handling retry logic based on status
    
    Ref: ISO-1, ISO-2, ISO-3, ISO-4, ISO-5
    
    Agents call this wrapper's generate() method and receive ONLY clean dict.
    They have NO access to reasoning_trace field.
    """
    
    def __init__(self, client: VLLMClient, role: AgentRole, enable_logging: bool = True):
        self.client = client
        self.role = role
        self.enable_logging = enable_logging
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate response for this role, returning ONLY payload.
        
        ISO Compliance:
        - reasoning_trace is logged and DISCARDED
        - Agent receives ONLY the clean payload dict
        - Type signature guarantees no reasoning_trace leakage
        
        Args:
            messages: Chat messages for the LLM
            json_schema: JSON schema for response validation
            
        Returns:
            Clean payload dict, or None on error
        """
        parsed = self.client.generate(role=self.role, messages=messages, json_schema=json_schema)
        
        # Log reasoning_trace IMMEDIATELY (audit only)
        if self.enable_logging and parsed.reasoning_trace:
            logger.info(
                f"[{self.role.value}] reasoning_trace (AUDIT ONLY): {parsed.reasoning_trace[:500]}..."
            )
        
        # Return ONLY payload based on status
        if parsed.status == "OK":
            return parsed.payload
        elif parsed.status == "TIMEOUT":
            logger.warning(f"[{self.role.value}] LLM timeout")
            return None
        elif parsed.status == "PARSE_ERROR":
            logger.error(f"[{self.role.value}] LLM parse error: {parsed.payload}")
            return None
        
        return None
