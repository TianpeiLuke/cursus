"""
Validator Factory

This module provides a factory for creating step-type-specific validators
with priority system integration and validation ruleset coordination.
"""

from typing import Dict, Any, List, Optional, Type
import logging

from .step_type_specific_validator import StepTypeSpecificValidator
from .processing_step_validator import ProcessingStepBuilderValidator
from .training_step_validator import TrainingStepBuilderValidator
from .createmodel_step_validator import CreateModelStepBuilderValidator
from .transform_step_validator import TransformStepBuilderValidator

from ..config import (
    get_validation_ruleset,
    get_universal_validation_rules,
    get_step_type_validation_rules,
    is_step_type_excluded
)
from ....registry.step_names import get_sagemaker_step_type

logger = logging.getLogger(__name__)


class ValidatorFactory:
    """
    Factory for creating step-type-specific validators with priority system.
    
    This factory integrates with the validation ruleset configuration system
    to provide seamless validator creation and coordination.
    """
    
    def __init__(self, workspace_dirs: List[str]):
        """
        Initialize validator factory.
        
        Args:
            workspace_dirs: List of workspace directories
        """
        self.workspace_dirs = workspace_dirs
        
        # Load validation rulesets for factory coordination
        self.universal_rules = get_universal_validation_rules()
        self.step_type_rules = get_step_type_validation_rules()
        
        # Define available validators
        self._validator_registry = {
            "ProcessingStepBuilderValidator": ProcessingStepBuilderValidator,
            "TrainingStepBuilderValidator": TrainingStepBuilderValidator,
            "CreateModelStepBuilderValidator": CreateModelStepBuilderValidator,
            "TransformStepBuilderValidator": TransformStepBuilderValidator,
            # Additional validators can be added here
            "RegisterModelStepBuilderValidator": None,  # Placeholder for future implementation
            "LambdaStepBuilderValidator": None,  # Placeholder for future implementation
        }
        
        logger.info(f"Initialized ValidatorFactory with {len(self._validator_registry)} validator types")
    
    def get_validator(self, validator_class: str) -> Optional[StepTypeSpecificValidator]:
        """
        Get validator instance by class name with priority system.
        
        Args:
            validator_class: Name of the validator class
            
        Returns:
            Validator instance or None if not available
        """
        logger.debug(f"Creating validator instance: {validator_class}")
        
        if validator_class not in self._validator_registry:
            logger.error(f"Unknown validator class: {validator_class}")
            return None
        
        validator_cls = self._validator_registry[validator_class]
        
        if validator_cls is None:
            logger.warning(f"Validator {validator_class} is not yet implemented")
            return None
        
        try:
            # Initialize validator with workspace directories and priority system
            validator_instance = validator_cls(self.workspace_dirs)
            logger.debug(f"Successfully created {validator_class} instance")
            return validator_instance
            
        except Exception as e:
            logger.error(f"Failed to create validator {validator_class}: {str(e)}")
            return None
    
    def get_validator_for_step_type(self, step_type: str) -> Optional[StepTypeSpecificValidator]:
        """
        Get validator for a specific step type using validation ruleset.
        
        Args:
            step_type: SageMaker step type
            
        Returns:
            Validator instance or None if not available
        """
        logger.debug(f"Getting validator for step type: {step_type}")
        
        # Check if step type is excluded
        if is_step_type_excluded(step_type):
            logger.info(f"Step type {step_type} is excluded from validation")
            return None
        
        # Get validation ruleset
        ruleset = get_validation_ruleset(step_type)
        
        if not ruleset:
            logger.warning(f"No validation ruleset found for step type: {step_type}")
            return None
        
        if not ruleset.level_4_validator_class:
            logger.info(f"No Level 4 validator specified for step type: {step_type}")
            return None
        
        # Create validator using ruleset configuration
        return self.get_validator(ruleset.level_4_validator_class)
    
    def get_validator_for_step_name(self, step_name: str) -> Optional[StepTypeSpecificValidator]:
        """
        Get validator for a specific step name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Validator instance or None if not available
        """
        logger.debug(f"Getting validator for step name: {step_name}")
        
        try:
            # Get step type from registry
            step_type = get_sagemaker_step_type(step_name)
            
            # Get validator for step type
            return self.get_validator_for_step_type(step_type)
            
        except Exception as e:
            logger.error(f"Failed to get validator for step {step_name}: {str(e)}")
            return None
    
    def validate_step_with_priority_system(self, step_name: str) -> Dict[str, Any]:
        """
        Validate step using priority-based validation system.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running priority-based validation for step: {step_name}")
        
        try:
            # Get step type from registry
            step_type = get_sagemaker_step_type(step_name)
            
            # Get validation ruleset
            ruleset = get_validation_ruleset(step_type)
            
            # Handle excluded step types
            if is_step_type_excluded(step_type):
                return {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "EXCLUDED",
                    "reason": ruleset.skip_reason if ruleset else f"Step type {step_type} is excluded",
                    "category": ruleset.category.value if ruleset else "excluded"
                }
            
            # Get appropriate validator
            validator = self.get_validator_for_step_type(step_type)
            
            if not validator:
                return {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "NO_VALIDATOR",
                    "message": f"No validator available for step type: {step_type}",
                    "available_validators": list(self._validator_registry.keys())
                }
            
            # Run validation with priority system
            validation_result = validator.validate_builder_config_alignment(step_name)
            
            # Add factory metadata
            validation_result["factory_metadata"] = {
                "validator_class": validator.__class__.__name__,
                "step_type": step_type,
                "ruleset_category": ruleset.category.value if ruleset else "unknown",
                "priority_system": "universal_first_then_step_specific"
            }
            
            logger.info(f"Priority-based validation completed for {step_name}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Priority-based validation failed for {step_name}: {str(e)}")
            return {
                "step_name": step_name,
                "status": "ERROR",
                "error": str(e),
                "factory_metadata": {
                    "validation_failed": True,
                    "error_source": "validator_factory"
                }
            }
    
    def is_validator_available(self, validator_name: str) -> bool:
        """
        Check if a validator is available (either implemented or placeholder).
        
        Args:
            validator_name: Name of the validator to check
            
        Returns:
            True if validator is available (implemented or placeholder), False otherwise
        """
        return validator_name in self._validator_registry
    
    def get_validator_registry_status(self) -> Dict[str, Any]:
        """
        Get the status of the validator registry.
        
        Returns:
            Dictionary containing registry status information
        """
        implemented = sum(1 for cls in self._validator_registry.values() if cls is not None)
        total = len(self._validator_registry)
        
        return {
            "total_validators": total,
            "implemented_validators": implemented,
            "placeholder_validators": total - implemented,
            "implementation_rate": implemented / total if total > 0 else 0,
            "registry_health": "healthy" if implemented > 0 else "no_implementations"
        }
    
    def get_factory_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the factory.
        
        Returns:
            Dictionary containing factory health information
        """
        config_result = self.validate_factory_configuration()
        registry_status = self.get_validator_registry_status()
        
        return {
            "healthy": config_result["valid"],
            "configuration_issues": config_result["issues"],
            "registry_status": registry_status,
            "workspace_dirs": self.workspace_dirs,
            "workspace_configured": len(self.workspace_dirs) > 0,
            "total_issues": len(config_result["issues"])
        }
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive factory statistics.
        
        Returns:
            Dictionary containing factory statistics
        """
        implemented = sum(1 for cls in self._validator_registry.values() if cls is not None)
        total = len(self._validator_registry)
        
        return {
            "validator_counts": {
                "total": total,
                "implemented": implemented,
                "placeholder": total - implemented
            },
            "implementation_status": {
                "rate": implemented / total if total > 0 else 0,
                "health": "healthy" if implemented > 0 else "no_implementations"
            },
            "workspace_info": {
                "directories": len(self.workspace_dirs),
                "configured": len(self.workspace_dirs) > 0
            }
        }
    
    def get_available_validators(self) -> List[str]:
        """
        Get list of available validator names.
        
        Returns:
            List of validator names
        """
        return list(self._validator_registry.keys())
    
    def get_available_validators_detailed(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about available validators.
        
        Returns:
            Dictionary containing validator information
        """
        available_validators = {}
        
        for validator_name, validator_cls in self._validator_registry.items():
            if validator_cls is not None:
                # Handle Mock objects in tests
                try:
                    class_name = validator_cls.__name__
                    module_name = validator_cls.__module__
                    doc_string = validator_cls.__doc__
                except AttributeError:
                    # Handle Mock objects that don't have these attributes
                    class_name = validator_name
                    module_name = "test_mock"
                    doc_string = "Mock validator for testing"
                
                available_validators[validator_name] = {
                    "class": class_name,
                    "module": module_name,
                    "implemented": True,
                    "description": doc_string.split('\n')[0] if doc_string else "No description"
                }
            else:
                available_validators[validator_name] = {
                    "class": validator_name,
                    "module": None,
                    "implemented": False,
                    "description": "Placeholder for future implementation"
                }
        
        return available_validators
    
    def get_step_type_validator_mapping(self) -> Dict[str, str]:
        """
        Get mapping of step types to their validator classes.
        
        Returns:
            Dictionary mapping step types to validator class names
        """
        mapping = {}
        
        # Get all step types from validation rulesets
        from ..config import VALIDATION_RULESETS
        
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if not is_step_type_excluded(step_type) and ruleset.level_4_validator_class:
                mapping[step_type] = ruleset.level_4_validator_class
        
        return mapping
    
    def validate_factory_configuration(self) -> Dict[str, Any]:
        """
        Validate factory configuration for consistency issues.
        
        Returns:
            Dictionary containing validation results
        """
        issues = []
        
        # Check that all referenced validators are available
        step_type_mapping = self.get_step_type_validator_mapping()
        
        for step_type, validator_class in step_type_mapping.items():
            if validator_class not in self._validator_registry:
                issues.append(f"Step type '{step_type}' references unknown validator: {validator_class}")
            elif self._validator_registry[validator_class] is None:
                issues.append(f"Step type '{step_type}' references unimplemented validator: {validator_class}")
        
        # Check for unused validators
        used_validators = set(step_type_mapping.values())
        available_validators = {name for name, cls in self._validator_registry.items() if cls is not None}
        unused_validators = available_validators - used_validators
        
        if unused_validators:
            issues.append(f"Unused validators found: {list(unused_validators)}")
        
        # Check workspace directories
        if not self.workspace_dirs:
            issues.append("No workspace directories configured")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def validate_factory_configuration_list(self) -> List[str]:
        """
        Validate factory configuration for consistency issues.
        
        Returns:
            List of configuration issues (empty if valid)
        """
        config_result = self.validate_factory_configuration()
        return config_result["issues"]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics and factory information.
        
        Returns:
            Dictionary containing validation statistics
        """
        step_type_mapping = self.get_step_type_validator_mapping()
        available_validators = self.get_available_validators()
        
        implemented_validators = sum(1 for info in available_validators.values() if info["implemented"])
        total_validators = len(available_validators)
        
        return {
            "total_step_types": len(step_type_mapping),
            "total_validators": total_validators,
            "implemented_validators": implemented_validators,
            "implementation_rate": implemented_validators / total_validators if total_validators > 0 else 0,
            "workspace_directories": len(self.workspace_dirs),
            "priority_system_enabled": True,
            "ruleset_integration": True,
            "step_type_coverage": list(step_type_mapping.keys()),
            "available_validator_classes": list(available_validators.keys())
        }


class StepTypeValidatorIntegration:
    """
    Integration layer between validation rulesets and step-type validators.
    
    This class provides seamless coordination between the configuration system
    and the validator factory for comprehensive validation orchestration.
    """
    
    def __init__(self, workspace_dirs: List[str]):
        """
        Initialize integration layer.
        
        Args:
            workspace_dirs: List of workspace directories
        """
        self.workspace_dirs = workspace_dirs
        self.validator_factory = ValidatorFactory(workspace_dirs)
        
        logger.info("Initialized StepTypeValidatorIntegration with priority-based validation")
    
    def validate_step_with_full_integration(self, step_name: str) -> Dict[str, Any]:
        """
        Validate step using full integration with rulesets and validators.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing comprehensive validation results
        """
        logger.info(f"Running full integration validation for step: {step_name}")
        
        # Use validator factory for priority-based validation
        validation_result = self.validator_factory.validate_step_with_priority_system(step_name)
        
        # Add integration metadata
        validation_result["integration_metadata"] = {
            "integration_layer": "StepTypeValidatorIntegration",
            "factory_integration": True,
            "ruleset_integration": True,
            "workspace_aware": True,
            "priority_hierarchy": "universal_rules_first_then_step_specific"
        }
        
        return validation_result
    
    def validate_multiple_steps(self, step_names: List[str]) -> Dict[str, Any]:
        """
        Validate multiple steps with full integration.
        
        Args:
            step_names: List of step names to validate
            
        Returns:
            Dictionary containing validation results for all steps
        """
        logger.info(f"Running full integration validation for {len(step_names)} steps")
        
        results = {}
        summary = {
            "total_steps": len(step_names),
            "passed_steps": 0,
            "failed_steps": 0,
            "excluded_steps": 0,
            "error_steps": 0
        }
        
        for step_name in step_names:
            result = self.validate_step_with_full_integration(step_name)
            results[step_name] = result
            
            # Update summary
            status = result.get("status", "UNKNOWN")
            if status == "PASSED":
                summary["passed_steps"] += 1
            elif status in ["FAILED", "ISSUES_FOUND"]:
                summary["failed_steps"] += 1
            elif status == "EXCLUDED":
                summary["excluded_steps"] += 1
            elif status == "ERROR":
                summary["error_steps"] += 1
        
        return {
            "summary": summary,
            "results": results,
            "integration_metadata": {
                "validation_approach": "full_integration",
                "priority_system": True,
                "ruleset_driven": True
            }
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get integration status and health information.
        
        Returns:
            Dictionary containing integration status
        """
        # Validate factory configuration
        factory_issues = self.validator_factory.validate_factory_configuration()
        
        # Get factory statistics
        factory_stats = self.validator_factory.get_validation_statistics()
        
        return {
            "integration_healthy": len(factory_issues) == 0,
            "factory_issues": factory_issues,
            "factory_statistics": factory_stats,
            "workspace_directories": len(self.workspace_dirs),
            "priority_system_active": True,
            "ruleset_integration_active": True
        }
