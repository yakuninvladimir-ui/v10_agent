"""
Layered Verifier - Brusentsov Ternary Logic Implementation
Ref: Engineering Specification V10.0 Section 3 & 5
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .types import PropositionSet, AtomicProposition, BrusentsovJudgment, BranchSignature
from .brusentsov_logic import Ternary, implies_brusentsov


@dataclass
class VerificationResult:
    """Result of LayeredVerifier judgment."""
    judgment: Ternary
    expected_set: PropositionSet
    observed_set: PropositionSet
    reasoning: str
    branch_signature: Optional[BranchSignature] = None


class LayeredVerifier:
    """
    Implements layered verification using Brusentsov ternary logic.
    
    Ref: Spec 5 - LayeredVerifier Contract
    Ref: Spec 3.1 - Brusentsov Ternary Logic
    
    Compares expected PropositionSet (from Trajectory Package) against
    observed PropositionSet (from actual state after step execution).
    """
    
    def __init__(self):
        self.judgment_history: List[VerificationResult] = []
    
    def verify_transition(
        self,
        expected: PropositionSet,
        observed: PropositionSet,
        action_sequence: Optional[tuple] = None,
    ) -> VerificationResult:
        """
        Verify a state transition using Brusentsov logic.
        
        Args:
            expected: Expected PropositionSet from trajectory candidate
            observed: Observed PropositionSet from actual state
            action_sequence: Optional action sequence for branch signature
        
        Returns:
            VerificationResult with judgment and reasoning
        
        Ref: Spec 5.1 - Transition Verification Pipeline
        """
        # Run Brusentsov implication check
        result = implies_brusentsov(expected, observed)
        
        # Create branch signature if action sequence provided
        branch_sig = None
        if action_sequence:
            import hashlib
            sig_data = str(action_sequence).encode()
            outcome_hash = hashlib.sha256(sig_data).hexdigest()[:32]
            branch_sig = BranchSignature(
                action_sequence=action_sequence,
                outcome_hash=outcome_hash,
            )
        
        # Generate reasoning based on judgment type
        reasoning = self._generate_reasoning(result, expected, observed)
        
        verification_result = VerificationResult(
            judgment=result,
            expected_set=expected,
            observed_set=observed,
            reasoning=reasoning,
            branch_signature=branch_sig,
        )
        
        self.judgment_history.append(verification_result)
        
        return verification_result
    
    def _generate_reasoning(
        self,
        judgment: Ternary,
        expected: PropositionSet,
        observed: PropositionSet,
    ) -> str:
        """Generate human-readable reasoning for judgment."""
        
        if judgment == Ternary.TRUE:
            return (
                f"FOLLOW: All {len(expected.propositions)} expected propositions "
                f"are contained in {len(observed.propositions)} observed propositions."
            )
        
        elif judgment == Ternary.FALSE:
            # Find contradicting propositions
            contradictions = self._find_contradictions(expected, observed)
            return (
                f"NULL: Physical contradiction detected. "
                f"{len(contradictions)} proposition(s) incompatible with observation."
            )
        
        else:  # IRRELEVANT
            missing = self._find_missing_effects(expected, observed)
            return (
                f"OMIT: No contradictions found, but {len(missing)} effect(s) "
                "missing from observation (inessential for continuation)."
            )
    
    def _find_contradictions(
        self,
        expected: PropositionSet,
        observed: PropositionSet,
    ) -> List[AtomicProposition]:
        """Find propositions that contradict between expected and observed."""
        contradictions = []
        
        for exp_prop in expected.propositions:
            for obs_prop in observed.propositions:
                if self._props_contradict(exp_prop, obs_prop):
                    contradictions.append(exp_prop)
                    break
        
        return contradictions
    
    def _find_missing_effects(
        self,
        expected: PropositionSet,
        observed: PropositionSet,
    ) -> List[AtomicProposition]:
        """Find expected propositions not present in observed set."""
        missing = []
        
        for exp_prop in expected.propositions:
            if not self._prop_in_set(exp_prop, observed.propositions):
                missing.append(exp_prop)
        
        return missing
    
    def _props_contradict(
        self,
        prop1: AtomicProposition,
        prop2: AtomicProposition,
    ) -> bool:
        """Check if two propositions contradict each other."""
        # Same family and same objects/relations but different data
        if prop1.family != prop2.family:
            return False
        
        if prop1.objects != prop2.objects or prop1.relations != prop2.relations:
            return False
        
        # Check for contradictory data values
        for key in set(prop1.data.keys()) | set(prop2.data.keys()):
            v1 = prop1.data.get(key)
            v2 = prop2.data.get(key)
            
            if v1 is not None and v2 is not None and v1 != v2:
                # Special handling for boolean flags
                if isinstance(v1, bool) and isinstance(v2, bool):
                    return True  # Direct contradiction
                # Special handling for numeric deltas
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    # Only contradict if signs are opposite
                    if (v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0):
                        return True
        
        return False
    
    def _prop_in_set(
        self,
        prop: AtomicProposition,
        prop_set: frozenset,
    ) -> bool:
        """Check if a proposition exists in a set."""
        return prop in prop_set
    
    def create_judgment_record(
        self,
        result: VerificationResult,
        timestamp: float = 0.0,
    ) -> BrusentsovJudgment:
        """
        Create a BrusentsovJudgment record for EpistemicMemory.
        
        Args:
            result: VerificationResult from verify_transition
            timestamp: Optional timestamp
        
        Returns:
            BrusentsovJudgment ready for EpistemicMemory storage
        """
        judgment_type_map = {
            Ternary.TRUE: "FOLLOW",
            Ternary.FALSE: "NULL",
            Ternary.IRRELEVANT: "OMIT",
        }
        
        return BrusentsovJudgment(
            judgment_type=judgment_type_map[result.judgment],
            branch_signature=result.branch_signature,
            reasoning=result.reasoning,
            timestamp=timestamp,
        )
