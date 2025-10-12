"""
Level Validators

This module provides consolidated validation logic for each validation level,
replacing the separate level tester classes with a unified approach.
"""

from typing import Dict, Any, List, Optional
import logging

from ....step_catalog import StepCatalog
from ..config import ValidationLevel

logger = logging.getLogger(__name__)


class LevelValidators:
    """Consolidated validation logic for each level."""
    
    def __init__(self, workspace_dirs: Optional[List[str]] = None):
        """
        Initialize level validators.
        
        Args:
            workspace_dirs: Optional list of workspace directories to search
        """
        self.workspace_dirs = workspace_dirs
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
        workspace_count = len(workspace_dirs) if workspace_dirs else 0
        logger.info(f"Initialized LevelValidators with {workspace_count} workspace directories")
    
    def run_level_1_validation(self, step_name: str) -> Dict[str, Any]:
        """
        Level 1: Script ↔ Contract validation.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running Level 1 validation for step: {step_name}")
        
        try:
            # Use existing script_contract_alignment logic
            from .script_contract_alignment import ScriptContractAlignmentTester
            alignment = ScriptContractAlignmentTester(workspace_dirs=self.workspace_dirs)
            result = alignment.validate_script(step_name)
            
            logger.info(f"Level 1 validation completed for {step_name}")
            return {
                "level": 1,
                "step_name": step_name,
                "validation_type": "script_contract",
                "status": "COMPLETED",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Level 1 validation failed for {step_name}: {str(e)}")
            return {
                "level": 1,
                "step_name": step_name,
                "validation_type": "script_contract",
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_level_2_validation(self, step_name: str) -> Dict[str, Any]:
        """
        Level 2: Contract ↔ Specification validation.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running Level 2 validation for step: {step_name}")
        
        try:
            # Use existing contract_spec_alignment logic
            from .contract_spec_alignment import ContractSpecificationAlignmentTester
            alignment = ContractSpecificationAlignmentTester(workspace_dirs=self.workspace_dirs)
            result = alignment.validate_contract(step_name)
            
            logger.info(f"Level 2 validation completed for {step_name}")
            return {
                "level": 2,
                "step_name": step_name,
                "validation_type": "contract_spec",
                "status": "COMPLETED",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Level 2 validation failed for {step_name}: {str(e)}")
            return {
                "level": 2,
                "step_name": step_name,
                "validation_type": "contract_spec",
                "status": "ERROR",
                "error": str(e)
            }
    
    def run_level_3_validation(self, step_name: str) -> Dict[str, Any]:
        """
        Level 3: Specification ↔ Dependencies validation (Universal).
        
        This level is universal and applies to all non-excluded step types.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running Level 3 validation for step: {step_name}")
        
        try:
            # Use existing spec_dependency_alignment logic
            from .spec_dependency_alignment import SpecificationDependencyAlignmentTester
            alignment = SpecificationDependencyAlignmentTester(workspace_dirs=self.workspace_dirs)
            result = alignment.validate_specification(step_name)
            
            logger.info(f"Level 3 validation completed for {step_name}")
            return {
                "level": 3,
                "step_name": step_name,
                "validation_type": "spec_dependency",
                "status": "COMPLETED",
                "result": result,
                "universal": True  # Mark as universal validation
            }
            
        except Exception as e:
            logger.error(f"Level 3 validation failed for {step_name}: {str(e)}")
            return {
                "level": 3,
                "step_name": step_name,
                "validation_type": "spec_dependency",
                "status": "ERROR",
                "error": str(e),
                "universal": True
            }
    
    def run_level_4_validation(self, step_name: str, validator_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Level 4: Builder ↔ Configuration validation (Step-type-specific).
        
        Args:
            step_name: Name of the step to validate
            validator_class: Name of the step-type-specific validator class to use
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Running Level 4 validation for step: {step_name} with validator: {validator_class}")
        
        if not validator_class:
            logger.warning(f"No validator class specified for Level 4 validation of {step_name}")
            return {
                "level": 4,
                "step_name": step_name,
                "validation_type": "builder_config",
                "status": "SKIPPED",
                "reason": "No validator class specified"
            }
        
        try:
            # Use step-type-specific validator
            validator = self._get_step_type_validator(validator_class)
            if not validator:
                logger.warning(f"Validator {validator_class} is not implemented for step {step_name}")
                return {
                    "level": 4,
                    "step_name": step_name,
                    "validation_type": "builder_config",
                    "status": "SKIPPED",
                    "reason": f"Validator {validator_class} is not yet implemented",
                    "validator_class": validator_class,
                    "implementation_needed": True
                }
            
            result = validator.validate_builder_config_alignment(step_name)
            
            logger.info(f"Level 4 validation completed for {step_name}")
            return {
                "level": 4,
                "step_name": step_name,
                "validation_type": "builder_config",
                "status": "COMPLETED",
                "validator_class": validator_class,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Level 4 validation failed for {step_name}: {str(e)}")
            return {
                "level": 4,
                "step_name": step_name,
                "validation_type": "builder_config",
                "status": "ERROR",
                "validator_class": validator_class,
                "error": str(e)
            }
    
    def _get_step_type_validator(self, validator_class: str):
        """
        Get step-type-specific validator instance.
        
        Args:
            validator_class: Name of the validator class
            
        Returns:
            Validator instance or None if not found
        """
        try:
            # Import validator factory
            from ..validators.validator_factory import ValidatorFactory
            
            # Create validator factory
            factory = ValidatorFactory(self.workspace_dirs)
            
            # Get validator instance
            return factory.get_validator(validator_class)
            
        except Exception as e:
            logger.error(f"Failed to create validator {validator_class}: {str(e)}")
            return None
    
    def validate_level_configuration(self, level: ValidationLevel) -> List[str]:
        """
        Validate that a validation level can be executed.
        
        Args:
            level: Validation level to check
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if level == ValidationLevel.SCRIPT_CONTRACT:
            # Check if script_contract_alignment module is available
            try:
                from .script_contract_alignment import ScriptContractAlignment
            except ImportError:
                issues.append("script_contract_alignment module not available for Level 1 validation")
        
        elif level == ValidationLevel.CONTRACT_SPEC:
            # Check if contract_spec_alignment module is available
            try:
                from .contract_spec_alignment import ContractSpecAlignment
            except ImportError:
                issues.append("contract_spec_alignment module not available for Level 2 validation")
        
        elif level == ValidationLevel.SPEC_DEPENDENCY:
            # Check if spec_dependency_alignment module is available
            try:
                from .spec_dependency_alignment import SpecDependencyAlignment
            except ImportError:
                issues.append("spec_dependency_alignment module not available for Level 3 validation")
        
        elif level == ValidationLevel.BUILDER_CONFIG:
            # Check if validator factory is available
            try:
                from ..validators.validator_factory import ValidatorFactory
            except ImportError:
                issues.append("validator_factory module not available for Level 4 validation")
        
        return issues
    
    def get_available_levels(self) -> List[ValidationLevel]:
        """
        Get list of validation levels that can be executed.
        
        Returns:
            List of available validation levels
        """
        available_levels = []
        
        for level in ValidationLevel:
            issues = self.validate_level_configuration(level)
            if not issues:
                available_levels.append(level)
        
        return available_levels
