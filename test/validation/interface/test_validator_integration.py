"""
Integration tests for Interface Standard Validator.

This module contains integration tests that test the complete validation
workflow and integration with real step builders from the codebase.
"""

import pytest
from unittest.mock import Mock
from typing import List, Dict, Any

from cursus.validation.interface.interface_standard_validator import (
    InterfaceStandardValidator,
    InterfaceViolation
)
from cursus.core.base.builder_base import StepBuilderBase

class MockGoodStepBuilder(StepBuilderBase):
    """Mock step builder that follows all interface standards."""
    
    def __init__(self, config=None, spec=None, sagemaker_session=None, role=None, notebook_root=None):
        """
        Initialize the mock step builder.
        
        Args:
            config: Configuration for the step
            spec: Step specification
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Notebook root directory
        """
        super().__init__(config, spec, sagemaker_session, role, notebook_root)
    
    def validate_configuration(self) -> None:
        """
        Validate the configuration.
        
        This method validates that all required configuration parameters
        are present and valid.
        """
        pass
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Get inputs for the step.
        
        Args:
            inputs: Input parameters dictionary
            
        Returns:
            List of processing inputs
        """
        return []
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[Any]:
        """
        Get outputs for the step.
        
        Args:
            outputs: Output parameters dictionary
            
        Returns:
            List of processing outputs
        """
        return []
    
    def create_step(self, **kwargs) -> Any:
        """
        Create the SageMaker step.
        
        Args:
            **kwargs: Additional keyword arguments
            
        Returns:
            SageMaker step instance
        """
        return Mock()

class MockBadStepBuilder:
    """Mock step builder that violates interface standards."""
    
    def __init__(self):
        pass
    
    # Missing required methods
    # Wrong method signatures
    def validate_configuration(self, extra_param):
        pass
    
    def _get_inputs(self):  # Missing inputs parameter
        return []
    
    # _get_outputs is completely missing
    
    def create_step(self):  # Missing **kwargs
        return Mock()

class TestInterfaceValidatorIntegration:
    """Integration tests for Interface Standard Validator."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InterfaceStandardValidator()
    
    def test_complete_validation_good_builder(self):
        """Test complete validation workflow with good builder."""
        violations = self.validator.validate_step_builder_interface(MockGoodStepBuilder)
        
        # Should have minimal violations (only potential documentation quality issues)
        critical_violations = [
            v for v in violations 
            if v.violation_type in [
                "inheritance_missing", "inheritance_mro", 
                "method_missing", "method_not_callable",
                "signature_missing_param", "signature_missing_kwargs"
            ]
        ]
        
        assert len(critical_violations) == 0
    
    def test_complete_validation_bad_builder(self):
        """Test complete validation workflow with bad builder."""
        violations = self.validator.validate_step_builder_interface(MockBadStepBuilder)
        
        # Should have multiple types of violations
        violation_types = set(v.violation_type for v in violations)
        
        # Should include inheritance violations
        assert "inheritance_missing" in violation_types
        
        # Should include method violations
        assert "method_missing" in violation_types
        
        # Should include documentation violations
        # Note: The current implementation may not generate class_documentation violations
        # but should have other documentation violations
        doc_violations = [vtype for vtype in violation_types if "documentation" in vtype]
        assert len(doc_violations) > 0
    
    def test_validation_with_real_step_builder(self):
        """Test validation with a real step builder from the codebase."""
        try:
            from cursus.steps.builders.builder_dummy_training_step import DummyTrainingStepBuilder
            
            violations = self.validator.validate_step_builder_interface(DummyTrainingStepBuilder)
            
            # Real builders should have minimal critical violations
            critical_violations = [
                v for v in violations 
                if v.violation_type in [
                    "inheritance_missing", "method_missing", "method_not_callable"
                ]
            ]
            
            # Print violations for debugging if any critical ones found
            if critical_violations:
                print(f"\nCritical violations found in DummyTrainingStepBuilder:")
                for violation in critical_violations:
                    print(f"  - {violation}")
            
            # Real builders should not have critical interface violations
            assert len(critical_violations) == 0, f"Real builder has critical violations: {critical_violations}"
            
        except ImportError:
            pytest.skip("DummyTrainingStepBuilder not available for testing")

if __name__ == '__main__':
    pytest.main([__file__])
