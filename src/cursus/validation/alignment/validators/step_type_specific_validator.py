"""
Step Type Specific Validator Base Class

This module provides the base class for all step-type-specific validators,
implementing the priority-based validation system with universal and 
step-specific rule integration.
"""

from typing import Dict, Any, List, Optional, Type
import logging
from abc import ABC, abstractmethod

from ..config import (
    get_universal_validation_rules,
    get_step_type_validation_rules,
    get_validation_rules_for_step_type
)
from ....registry.step_names import get_sagemaker_step_type

logger = logging.getLogger(__name__)


class StepTypeSpecificValidator(ABC):
    """
    Base class for step-type-specific validators following priority hierarchy.
    
    This class implements the priority-based validation system where:
    1. Universal Builder Rules (HIGHEST PRIORITY) - Always applied
    2. Step-Type-Specific Rules (SECONDARY PRIORITY) - Applied as supplements
    """
    
    def __init__(self, workspace_dirs: Optional[List[str]] = None):
        """
        Initialize step-type-specific validator.
        
        Args:
            workspace_dirs: List of workspace directories (optional)
        """
        self.workspace_dirs = workspace_dirs
        
        # Load both rulesets for priority-based validation
        self.universal_rules = get_universal_validation_rules()
        self.step_type_rules = get_step_type_validation_rules()
        
        logger.info(f"Initialized {self.__class__.__name__} with priority-based validation")
    
    def validate_builder_config_alignment(self, step_name: str) -> Dict[str, Any]:
        """
        Validate step builder following priority hierarchy.
        
        Implementation Order:
        1. Apply universal validation rules (HIGHEST PRIORITY)
        2. Apply step-type-specific validation rules (SECONDARY PRIORITY)
        3. Combine results with proper priority handling
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results with priority information
        """
        logger.info(f"Running priority-based validation for step: {step_name}")
        
        results = {
            "step_name": step_name,
            "validator_class": self.__class__.__name__,
            "validation_results": {},
            "priority_applied": "universal_first_then_step_specific"
        }
        
        try:
            # 1. HIGHEST PRIORITY: Universal validation
            universal_validation = self._apply_universal_validation(step_name)
            results["validation_results"]["universal"] = universal_validation
            
            # 2. SECONDARY PRIORITY: Step-type-specific validation
            step_specific_validation = self._apply_step_specific_validation(step_name)
            results["validation_results"]["step_specific"] = step_specific_validation
            
            # 3. Combine with priority resolution
            combined_result = self._resolve_validation_priorities(
                universal_validation, 
                step_specific_validation
            )
            results["final_result"] = combined_result
            
            # Add compatibility fields at top level
            results["status"] = combined_result.get("status", "UNKNOWN")
            results["total_issues"] = combined_result.get("total_issues", 0)
            results["error_count"] = combined_result.get("error_count", 0)
            results["warning_count"] = combined_result.get("warning_count", 0)
            results["priority_resolution"] = combined_result.get("priority_resolution", "universal_rules_first_then_step_specific")
            
            logger.info(f"Priority-based validation completed for {step_name}")
            return results
            
        except Exception as e:
            logger.error(f"Priority-based validation failed for {step_name}: {str(e)}")
            return {
                "step_name": step_name,
                "validator_class": self.__class__.__name__,
                "status": "ERROR",
                "error": str(e),
                "priority_applied": "validation_failed"
            }
    
    def _apply_universal_validation(self, step_name: str) -> Dict[str, Any]:
        """
        Apply universal builder validation rules.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing universal validation results
        """
        logger.debug(f"Applying universal validation rules for {step_name}")
        
        try:
            # Get builder class
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}",
                    "rule_type": "universal",
                    "priority": "HIGHEST"
                }
            
            # Validate universal requirements
            issues = []
            
            # Check required abstract methods
            required_methods = self.universal_rules.get("required_methods", {})
            for method_name, method_spec in required_methods.items():
                if not hasattr(builder_class, method_name):
                    issues.append({
                        "level": "ERROR",
                        "message": f"Missing universal required method: {method_name}",
                        "method_name": method_name,
                        "rule_type": "universal",
                        "details": {
                            "expected_signature": method_spec.get("signature"),
                            "purpose": method_spec.get("purpose"),
                            "category": method_spec.get("category")
                        }
                    })
            
            # Check inherited method compliance
            inherited_methods = self.universal_rules.get("inherited_methods", {})
            for method_name, method_spec in inherited_methods.items():
                if method_spec.get("category") == "INHERITED_FINAL":
                    # Check if method is overridden when it shouldn't be
                    if self._is_method_overridden(builder_class, method_name):
                        issues.append({
                            "level": "WARNING",
                            "message": f"Method {method_name} should not be overridden (INHERITED_FINAL)",
                            "method_name": method_name,
                            "rule_type": "universal",
                            "details": {
                                "category": method_spec.get("category"),
                                "inherited_from": method_spec.get("inherited_from")
                            }
                        })
            
            return {
                "status": "COMPLETED" if not issues else "ISSUES_FOUND",
                "issues": issues,
                "rule_type": "universal",
                "priority": "HIGHEST",
                "builder_class": builder_class.__name__
            }
            
        except Exception as e:
            logger.error(f"Universal validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "rule_type": "universal",
                "priority": "HIGHEST"
            }
    
    def _apply_step_specific_validation(self, step_name: str) -> Dict[str, Any]:
        """
        Apply step-type-specific validation rules.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing step-specific validation results
        """
        logger.debug(f"Applying step-specific validation rules for {step_name}")
        
        try:
            # Get step type from registry
            step_type = get_sagemaker_step_type(step_name)
            
            # Get builder class
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}",
                    "rule_type": "step_specific",
                    "priority": "SECONDARY"
                }
            
            # Get step-type-specific rules
            step_rules = get_validation_rules_for_step_type(step_type)
            if not step_rules:
                return {
                    "status": "NO_RULES",
                    "message": f"No step-type-specific rules for {step_type}",
                    "rule_type": "step_specific",
                    "priority": "SECONDARY",
                    "step_type": step_type
                }
            
            issues = []
            
            # Check required step-specific methods
            required_methods = step_rules.get("required_methods", {})
            for method_name, method_spec in required_methods.items():
                if not hasattr(builder_class, method_name):
                    issues.append({
                        "level": "ERROR",
                        "message": f"Missing {step_type} required method: {method_name}",
                        "method_name": method_name,
                        "step_type": step_type,
                        "rule_type": "step_specific",
                        "details": {
                            "expected_signature": method_spec.get("signature"),
                            "purpose": method_spec.get("purpose"),
                            "return_type": method_spec.get("return_type")
                        }
                    })
            
            # Apply step-type-specific validation logic
            step_specific_issues = self._validate_step_type_specifics(step_name, builder_class, step_type)
            issues.extend(step_specific_issues)
            
            return {
                "status": "COMPLETED" if not issues else "ISSUES_FOUND",
                "issues": issues,
                "step_type": step_type,
                "rule_type": "step_specific",
                "priority": "SECONDARY",
                "builder_class": builder_class.__name__
            }
            
        except Exception as e:
            logger.error(f"Step-specific validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "rule_type": "step_specific",
                "priority": "SECONDARY"
            }
    
    @abstractmethod
    def _validate_step_type_specifics(self, step_name: str, builder_class: Type, step_type: str) -> List[Dict[str, Any]]:
        """
        Validate step-type-specific requirements.
        
        This method must be implemented by each step-type-specific validator
        to provide specialized validation logic for that step type.
        
        Args:
            step_name: Name of the step to validate
            builder_class: The builder class to validate
            step_type: The SageMaker step type
            
        Returns:
            List of validation issues specific to the step type
        """
        pass
    
    def _resolve_validation_priorities(self, universal_result: Dict[str, Any], step_specific_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve validation results following priority hierarchy.
        
        Args:
            universal_result: Results from universal validation
            step_specific_result: Results from step-specific validation
            
        Returns:
            Combined validation results with proper priority handling
        """
        combined_issues = []
        
        # 1. HIGHEST PRIORITY: Universal issues (always included)
        if universal_result.get("issues"):
            combined_issues.extend(universal_result["issues"])
        
        # 2. SECONDARY PRIORITY: Step-specific issues (supplementary)
        if step_specific_result.get("issues"):
            combined_issues.extend(step_specific_result["issues"])
        
        # Determine overall status
        has_errors = any(issue["level"] == "ERROR" for issue in combined_issues)
        has_warnings = any(issue["level"] == "WARNING" for issue in combined_issues)
        
        if has_errors:
            status = "FAILED"
        elif has_warnings:
            status = "PASSED_WITH_WARNINGS"
        else:
            status = "PASSED"
        
        return {
            "status": status,
            "total_issues": len(combined_issues),
            "error_count": sum(1 for issue in combined_issues if issue["level"] == "ERROR"),
            "warning_count": sum(1 for issue in combined_issues if issue["level"] == "WARNING"),
            "issues": combined_issues,
            "priority_resolution": "universal_rules_first_then_step_specific",
            "universal_status": universal_result.get("status"),
            "step_specific_status": step_specific_result.get("status")
        }
    
    def _get_builder_class(self, step_name: str) -> Optional[Type]:
        """
        Get builder class for a step name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Builder class or None if not found
        """
        try:
            # Use step catalog to find builder
            from ....step_catalog import StepCatalog
            step_catalog = StepCatalog(workspace_dirs=self.workspace_dirs)
            
            # Try to get builder class using the correct method
            builder_class = step_catalog.load_builder_class(step_name)
            if builder_class:
                return builder_class
            
            # Fallback: try to import builder directly
            builder_module_name = f"cursus.steps.builders.builder_{step_name.lower()}_step"
            builder_class_name = f"{step_name}StepBuilder"
            
            try:
                import importlib
                module = importlib.import_module(builder_module_name)
                return getattr(module, builder_class_name, None)
            except (ImportError, AttributeError):
                logger.warning(f"Could not import builder class for {step_name}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to get builder class for {step_name}: {str(e)}")
            return None
    
    def _is_method_overridden(self, builder_class: Type, method_name: str) -> bool:
        """
        Check if a method is overridden in the builder class.
        
        Args:
            builder_class: The builder class
            method_name: Name of the method to check
            
        Returns:
            True if method is overridden, False otherwise
        """
        try:
            # Get method from the class
            if not hasattr(builder_class, method_name):
                return False
            
            method = getattr(builder_class, method_name)
            
            # Check if method is defined in this class vs inherited
            for cls in builder_class.__mro__:
                if method_name in cls.__dict__:
                    # Method is defined in this class
                    return cls == builder_class
            
            return False
        
        except Exception as e:
            logger.warning(f"Could not check if method {method_name} is overridden: {str(e)}")
            return False
    
    def _combine_validation_results(self, base_results: Dict[str, Any], specific_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine base validation results with step-specific results.
        
        Args:
            base_results: Base validation results from priority system
            specific_results: Additional step-specific validation results
            
        Returns:
            Combined validation results
        """
        # Start with base results
        combined = base_results.copy()
        
        # Add specific results
        if "additional_validation" not in combined:
            combined["additional_validation"] = {}
        
        combined["additional_validation"].update(specific_results)
        
        # Update issue counts if needed
        if specific_results.get("issues"):
            final_result = combined.get("final_result", {})
            existing_issues = final_result.get("issues", [])
            existing_issues.extend(specific_results["issues"])
            
            # Update counts
            final_result["total_issues"] = len(existing_issues)
            final_result["error_count"] = sum(1 for issue in existing_issues if issue.get("level") == "ERROR")
            final_result["warning_count"] = sum(1 for issue in existing_issues if issue.get("level") == "WARNING")
            
            combined["final_result"] = final_result
        
        return combined
