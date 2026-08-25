"""
Verification Binder - ISO-5 Compliant (PlanningSet Binding Validation)
Ref: Engineering Specification V10.0 Section 4 & 7
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .types import EffectDeclaration, AtomicProposition
from .planning_set import PlanningSet


@dataclass
class BindingError:
    """Represents a binding validation error."""
    error_type: str  # "missing_object", "missing_relation", "invalid_action"
    identifier: str
    message: str


class VerificationBinder:
    """
    Validates that DSL function arguments reference valid PlanningSet entities.
    
    ISO-5 INVARIANT: All object_id, relation_id, and action_id arguments
    must exist in the current PlanningSet before execution.
    
    Ref: Spec 4.3 - Verification Binder
    Ref: Spec 7.2 - ActionBoundary Contract
    """
    
    def __init__(self, planning_set: PlanningSet):
        """
        Initialize binder with current PlanningSet.
        
        Args:
            planning_set: The PlanningSet to validate against
        """
        self.planning_set = planning_set
        self._object_ids = set(planning_set.object_ids)
        self._relation_ids = set(planning_set.relation_ids)
        self._action_ids = set(planning_set.allowed_action_ids)
    
    def verify_effect_declaration(
        self,
        effect: EffectDeclaration,
    ) -> Tuple[bool, List[BindingError]]:
        """
        Verify that an EffectDeclaration's arguments are bound to valid PlanningSet IDs.
        
        Args:
            effect: The EffectDeclaration to verify
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        
        Ref: Spec 4.3 - Verification Binder (I5 Invariant)
        """
        errors = []
        args = effect.arguments
        
        # Check object_id references
        if 'object_id' in args:
            obj_id = args['object_id']
            if obj_id not in self._object_ids:
                errors.append(BindingError(
                    error_type="missing_object",
                    identifier=obj_id,
                    message=f"Object ID '{obj_id}' not found in PlanningSet",
                ))
        
        # Check object_ids list references
        if 'object_ids' in args:
            for obj_id in args['object_ids']:
                if obj_id not in self._object_ids:
                    errors.append(BindingError(
                        error_type="missing_object",
                        identifier=obj_id,
                        message=f"Object ID '{obj_id}' not found in PlanningSet",
                    ))
        
        # Check relation_id references
        if 'relation_id' in args:
            rel_id = args['relation_id']
            if rel_id not in self._relation_ids:
                errors.append(BindingError(
                    error_type="missing_relation",
                    identifier=rel_id,
                    message=f"Relation ID '{rel_id}' not found in PlanningSet",
                ))
        
        # Check action_id references
        if 'action_id' in args:
            action_id = args['action_id']
            if action_id not in self._action_ids:
                errors.append(BindingError(
                    error_type="invalid_action",
                    identifier=action_id,
                    message=f"Action ID '{action_id}' not in allowed actions",
                ))
        
        # Check source/target relation references
        if 'source_id' in args:
            src_id = args['source_id']
            if src_id not in self._object_ids:
                errors.append(BindingError(
                    error_type="missing_object",
                    identifier=src_id,
                    message=f"Source ID '{src_id}' not found in PlanningSet",
                ))
        
        if 'target_id' in args:
            tgt_id = args['target_id']
            if tgt_id not in self._object_ids:
                errors.append(BindingError(
                    error_type="missing_object",
                    identifier=tgt_id,
                    message=f"Target ID '{tgt_id}' not found in PlanningSet",
                ))
        
        return (len(errors) == 0, errors)
    
    def verify_proposition_bindings(
        self,
        propositions: List[AtomicProposition],
    ) -> Tuple[bool, List[BindingError]]:
        """
        Verify that AtomicPropositions reference valid PlanningSet IDs.
        
        Args:
            propositions: List of propositions to verify
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for prop in propositions:
            # Check object references
            for obj_id in prop.objects:
                if obj_id not in self._object_ids:
                    errors.append(BindingError(
                        error_type="missing_object",
                        identifier=obj_id,
                        message=f"Proposition object ID '{obj_id}' not in PlanningSet",
                    ))
            
            # Check relation references
            for rel_id in prop.relations:
                if rel_id not in self._relation_ids:
                    errors.append(BindingError(
                        error_type="missing_relation",
                        identifier=rel_id,
                        message=f"Proposition relation ID '{rel_id}' not in PlanningSet",
                    ))
        
        return (len(errors) == 0, errors)
    
    def get_validation_summary(self) -> Dict[str, int]:
        """
        Get summary of available bindings in current PlanningSet.
        
        Returns:
            Dictionary with counts of available objects, relations, actions
        """
        return {
            'available_objects': len(self._object_ids),
            'available_relations': len(self._relation_ids),
            'allowed_actions': len(self._action_ids),
        }
