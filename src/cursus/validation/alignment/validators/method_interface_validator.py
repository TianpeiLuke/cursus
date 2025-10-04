"""
Method Interface Validator

This module provides method interface compliance validation, focusing on
whether step builders implement the required methods according to universal
and step-type-specific validation rules.
"""

from typing import Dict, Any, List, Optional, Type
import logging
import inspect

from ..config import (
    get_universal_validation_rules,
    get_step_type_validation_rules,
    get_validation_rules_for_step_type
)
from ....registry.step_names import get_sagemaker_step_type

logger = logging.getLogger(__name__)


class ValidationIssue:
    """Represents a validation issue found during method interface validation."""
    
    def __init__(self, level: str, message: str, method_name: str, step_type: str = None, 
                 rule_type: str = None, details: Dict[str, Any] = None):
        self.level = level
        self.message = message
        self.method_name = method_name
        self.step_type = step_type
        self.rule_type = rule_type
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation issue to dictionary format."""
        return {
            "level": self.level,
            "message": self.message,
            "method_name": self.method_name,
            "step_type": self.step_type,
            "rule_type": self.rule_type,
            "details": self.details
        }


class MethodInterfaceValidator:
    """Validator focusing on method interface compliance."""
    
    def __init__(self, workspace_dirs: List[str]):
        """
        Initialize method interface validator.
        
        Args:
            workspace_dirs: List of workspace directories
        """
        self.workspace_dirs = workspace_dirs
        self.universal_rules = get_universal_validation_rules()
        self.step_type_rules = get_step_type_validation_rules()
        logger.info("Initialized MethodInterfaceValidator")
    
    def validate_builder_interface(self, builder_class: Type, step_type: str) -> List[ValidationIssue]:
        """
        Validate builder implements required methods.
        
        Args:
            builder_class: The builder class to validate
            step_type: The SageMaker step type
            
        Returns:
            List of validation issues
        """
        logger.info(f"Validating builder interface for {builder_class.__name__} (step type: {step_type})")
        
        issues = []
        
        # Universal method validation (HIGHEST PRIORITY)
        universal_issues = self._validate_universal_methods(builder_class, step_type)
        issues.extend(universal_issues)
        
        # Step-type-specific method validation (SECONDARY PRIORITY)
        step_specific_issues = self._validate_step_type_methods(builder_class, step_type)
        issues.extend(step_specific_issues)
        
        logger.info(f"Method interface validation completed for {builder_class.__name__}: {len(issues)} issues found")
        return issues
    
    def _validate_universal_methods(self, builder_class: Type, step_type: str) -> List[ValidationIssue]:
        """
        Validate universal methods that all builders must implement.
        
        Args:
            builder_class: The builder class to validate
            step_type: The SageMaker step type
            
        Returns:
            List of validation issues for universal methods
        """
        issues = []
        
        # Get universal required methods
        required_methods = self.universal_rules.get("required_methods", {})
        
        for method_name, method_spec in required_methods.items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    level="ERROR",
                    message=f"Missing universal required method: {method_name}",
                    method_name=method_name,
                    step_type=step_type,
                    rule_type="universal",
                    details={
                        "expected_signature": method_spec.get("signature"),
                        "purpose": method_spec.get("purpose"),
                        "category": method_spec.get("category")
                    }
                ))
            else:
                # Method exists, validate signature if possible
                signature_issues = self._validate_method_signature(
                    builder_class, method_name, method_spec, step_type, "universal"
                )
                issues.extend(signature_issues)
        
        # Check inherited methods compliance
        inherited_methods = self.universal_rules.get("inherited_methods", {})
        for method_name, method_spec in inherited_methods.items():
            if method_spec.get("category") == "INHERITED_FINAL":
                # Check if method is overridden when it shouldn't be
                if self._is_method_overridden(builder_class, method_name):
                    issues.append(ValidationIssue(
                        level="WARNING",
                        message=f"Method {method_name} should not be overridden (INHERITED_FINAL)",
                        method_name=method_name,
                        step_type=step_type,
                        rule_type="universal",
                        details={
                            "category": method_spec.get("category"),
                            "inherited_from": method_spec.get("inherited_from")
                        }
                    ))
        
        return issues
    
    def _validate_step_type_methods(self, builder_class: Type, step_type: str) -> List[ValidationIssue]:
        """
        Validate step-type-specific methods.
        
        Args:
            builder_class: The builder class to validate
            step_type: The SageMaker step type
            
        Returns:
            List of validation issues for step-type-specific methods
        """
        issues = []
        
        # Get step-type-specific rules
        step_rules = get_validation_rules_for_step_type(step_type)
        if not step_rules:
            logger.info(f"No step-type-specific rules found for {step_type}")
            return issues
        
        # Check required step-specific methods
        required_methods = step_rules.get("required_methods", {})
        for method_name, method_spec in required_methods.items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    level="ERROR",
                    message=f"Missing {step_type} required method: {method_name}",
                    method_name=method_name,
                    step_type=step_type,
                    rule_type="step_specific",
                    details={
                        "expected_signature": method_spec.get("signature"),
                        "purpose": method_spec.get("purpose"),
                        "return_type": method_spec.get("return_type")
                    }
                ))
            else:
                # Method exists, validate signature if possible
                signature_issues = self._validate_method_signature(
                    builder_class, method_name, method_spec, step_type, "step_specific"
                )
                issues.extend(signature_issues)
        
        # Check optional step-specific methods
        optional_methods = step_rules.get("optional_methods", {})
        for method_name, method_spec in optional_methods.items():
            if hasattr(builder_class, method_name):
                # Optional method is implemented, validate signature
                signature_issues = self._validate_method_signature(
                    builder_class, method_name, method_spec, step_type, "step_specific_optional"
                )
                issues.extend(signature_issues)
        
        return issues
    
    def _validate_method_signature(self, builder_class: Type, method_name: str, 
                                 method_spec: Dict[str, Any], step_type: str, 
                                 rule_type: str) -> List[ValidationIssue]:
        """
        Validate method signature against specification.
        
        Args:
            builder_class: The builder class
            method_name: Name of the method to validate
            method_spec: Method specification from rules
            step_type: The SageMaker step type
            rule_type: Type of rule (universal, step_specific, etc.)
            
        Returns:
            List of validation issues for method signature
        """
        issues = []
        
        try:
            method = getattr(builder_class, method_name)
            signature = inspect.signature(method)
            
            # Basic signature validation
            expected_signature = method_spec.get("signature", "")
            if expected_signature:
                # Simple parameter count validation
                param_count = len(signature.parameters)
                expected_params = expected_signature.count(',') + 1 if ',' in expected_signature else 1
                
                # Allow for 'self' parameter
                if 'self' in expected_signature:
                    expected_params -= 1
                    param_count -= 1
                
                # This is a basic check - could be enhanced with more sophisticated signature parsing
                if abs(param_count - expected_params) > 1:  # Allow some flexibility
                    issues.append(ValidationIssue(
                        level="WARNING",
                        message=f"Method {method_name} signature may not match expected signature",
                        method_name=method_name,
                        step_type=step_type,
                        rule_type=rule_type,
                        details={
                            "actual_signature": str(signature),
                            "expected_signature": expected_signature,
                            "actual_param_count": param_count,
                            "expected_param_count": expected_params
                        }
                    ))
        
        except Exception as e:
            logger.warning(f"Could not validate signature for {method_name}: {str(e)}")
        
        return issues
    
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
    
    def validate_builder_by_name(self, step_name: str) -> Dict[str, Any]:
        """
        Validate builder by step name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating builder by name: {step_name}")
        
        try:
            # Get step type from registry
            step_type = get_sagemaker_step_type(step_name)
            
            # Get builder class
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}"
                }
            
            # Validate builder interface
            issues = self.validate_builder_interface(builder_class, step_type)
            
            # Categorize issues
            errors = [issue for issue in issues if issue.level == "ERROR"]
            warnings = [issue for issue in issues if issue.level == "WARNING"]
            
            status = "PASSED"
            if errors:
                status = "FAILED"
            elif warnings:
                status = "PASSED_WITH_WARNINGS"
            
            return {
                "step_name": step_name,
                "step_type": step_type,
                "builder_class": builder_class.__name__,
                "status": status,
                "total_issues": len(issues),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "issues": [issue.to_dict() for issue in issues]
            }
        
        except Exception as e:
            logger.error(f"Failed to validate builder for {step_name}: {str(e)}")
            return {
                "step_name": step_name,
                "status": "ERROR",
                "error": str(e)
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
    
    def get_validation_summary(self, step_names: List[str]) -> Dict[str, Any]:
        """
        Get validation summary for multiple steps.
        
        Args:
            step_names: List of step names to validate
            
        Returns:
            Dictionary containing validation summary
        """
        logger.info(f"Getting validation summary for {len(step_names)} steps")
        
        results = {}
        total_issues = 0
        total_errors = 0
        total_warnings = 0
        passed_count = 0
        failed_count = 0
        
        for step_name in step_names:
            result = self.validate_builder_by_name(step_name)
            results[step_name] = result
            
            if result.get("status") == "PASSED":
                passed_count += 1
            elif result.get("status") in ["FAILED", "ERROR"]:
                failed_count += 1
            
            total_issues += result.get("total_issues", 0)
            total_errors += result.get("error_count", 0)
            total_warnings += result.get("warning_count", 0)
        
        return {
            "total_steps": len(step_names),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_issues": total_issues,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "results": results
        }
