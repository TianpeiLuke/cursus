"""
Level 1 Interface Tests for step builders.

These tests focus on the most basic requirements:
- Class inheritance
- Required method implementation
- Basic error handling
"""

from .base_test import UniversalStepBuilderTestBase


class InterfaceTests(UniversalStepBuilderTestBase):
    """
    Level 1 tests focusing on interface compliance.
    
    These tests validate that a step builder implements the correct 
    interface and basic functionality without requiring deep knowledge
    of the specification system or contracts.
    """
    
    def test_inheritance(self) -> None:
        """Test that the builder inherits from StepBuilderBase."""
        from src.pipeline_steps.builder_step_base import StepBuilderBase
        
        self._assert(
            issubclass(self.builder_class, StepBuilderBase),
            f"{self.builder_class.__name__} must inherit from StepBuilderBase"
        )
    
    def test_required_methods(self) -> None:
        """Test that the builder implements all required methods."""
        required_methods = [
            'validate_configuration',
            '_get_inputs',
            '_get_outputs',
            'create_step'
        ]
        
        for method_name in required_methods:
            method = getattr(self.builder_class, method_name, None)
            self._assert(
                method is not None,
                f"Builder must implement {method_name}()"
            )
            self._assert(
                callable(method),
                f"{method_name} must be callable"
            )
            
            # Check if method is abstract or implemented
            if method_name not in ['create_step']:  # create_step is often overridden
                self._assert(
                    not getattr(method, '__isabstractmethod__', False),
                    f"{method_name}() must be implemented, not abstract"
                )
    
    def test_error_handling(self) -> None:
        """Test that the builder handles errors appropriately."""
        # Test with invalid configuration
        try:
            # Create config without required attributes
            invalid_config = self._create_invalid_config()
            
            # Create builder with invalid config
            with self._assert_raises(ValueError):
                builder = self.builder_class(
                    config=invalid_config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role,
                    registry_manager=self.mock_registry_manager,
                    dependency_resolver=self.mock_dependency_resolver
                )
                
                # Should raise ValueError
                builder.validate_configuration()
        except Exception as e:
            self._assert(
                False,
                f"Error handling test failed: {str(e)}"
            )
