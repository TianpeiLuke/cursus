"""
Level 1 Interface Tests for step builders.

These tests focus on the most basic requirements:
- Class inheritance and naming conventions
- Required method implementation and signatures
- Registry integration and decorator usage
- Basic error handling and validation
- Documentation standards compliance
"""

import inspect
from typing import get_type_hints
from .base_test import UniversalStepBuilderTestBase


class InterfaceTests(UniversalStepBuilderTestBase):
    """
    Level 1 tests focusing on interface compliance.
    
    These tests validate that a step builder implements the correct 
    interface and basic functionality without requiring deep knowledge
    of the specification system or contracts. Enhanced to enforce
    standardization rules and alignment requirements.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return step type-specific test methods for interface tests."""
        return []  # Interface tests are generic, no step-type specific tests
    
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects for interface tests."""
        # Interface tests use generic mocks, no step-type specific configuration needed
        pass
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate step type-specific requirements for interface tests."""
        # Interface tests don't have step-type specific requirements
        return {
            "interface_tests_completed": True,
            "step_type_agnostic": True
        }
    
    def test_inheritance(self) -> None:
        """Test that the builder inherits from StepBuilderBase."""
        from ...core.base.builder_base import StepBuilderBase
        
        self._assert(
            issubclass(self.builder_class, StepBuilderBase),
            f"{self.builder_class.__name__} must inherit from StepBuilderBase"
        )
    
    def test_naming_conventions(self) -> None:
        """Test that the builder follows naming conventions."""
        class_name = self.builder_class.__name__
        
        # Check class name follows pattern: XXXStepBuilder
        self._assert(
            class_name.endswith("StepBuilder"),
            f"Class name '{class_name}' must end with 'StepBuilder'"
        )
        
        # Check step type is in PascalCase
        if class_name.endswith("StepBuilder"):
            step_type = class_name[:-11]  # Remove "StepBuilder"
            self._assert(
                step_type and step_type[0].isupper(),
                f"Step type '{step_type}' must be in PascalCase"
            )
        
        # Check method naming conventions
        for method_name in dir(self.builder_class):
            if not method_name.startswith('_') and callable(getattr(self.builder_class, method_name)):
                self._assert(
                    method_name.islower() or '_' in method_name,
                    f"Public method '{method_name}' should be in snake_case"
                )
    
    def test_required_methods(self) -> None:
        """Test that the builder implements all required methods with correct signatures."""
        required_methods = {
            'validate_configuration': [],
            '_get_inputs': ['inputs'],
            '_get_outputs': ['outputs'],
            'create_step': [],  # create_step can use **kwargs, so we don't enforce specific params
            '_get_step_name': [],
            '_get_environment_variables': [],
            '_get_job_arguments': []
        }
        
        for method_name, expected_params in required_methods.items():
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
            self._assert(
                not getattr(method, '__isabstractmethod__', False),
                f"{method_name}() must be implemented, not abstract"
            )
            
            # Check method signature
            try:
                sig = inspect.signature(method)
                param_names = [p for p in sig.parameters.keys() if p != 'self']
                
                # Special handling for create_step - be flexible with parameters
                if method_name == 'create_step':
                    # create_step can have various signatures, just ensure it's callable
                    # Common patterns: create_step(**kwargs), create_step(dependencies=None, enable_caching=True, **kwargs)
                    self._log(f"Info: {method_name}() signature: {sig}")
                else:
                    # Check that expected parameters are present for other methods
                    for expected_param in expected_params:
                        self._assert(
                            expected_param in param_names,
                            f"{method_name}() must have parameter '{expected_param}'"
                        )
            except Exception as e:
                self._log(f"Could not inspect signature of {method_name}: {str(e)}")
    
    def test_registry_integration(self) -> None:
        """Test that the builder is properly registered."""
        # Check if builder has registry key (indicates @register_builder decorator)
        has_registry_key = hasattr(self.builder_class, '_registry_key')
        
        if has_registry_key:
            self._log(f"Info: Builder has registry key - properly decorated with @register_builder")
        else:
            # Try to check if class is in a registry
            try:
                from ...registry.builder_registry import StepBuilderRegistry
                registry = StepBuilderRegistry()
                
                # Check if builder is registered using available methods
                class_name = self.builder_class.__name__
                step_type = class_name[:-11] if class_name.endswith("StepBuilder") else class_name
                
                # Try different registry methods to check registration
                try:
                    # Try to get the builder from registry
                    registered_builder = registry.get_builder(step_type)
                    if registered_builder == self.builder_class:
                        self._log(f"Info: {class_name} is properly registered in registry")
                    else:
                        self._log(f"Warning: {class_name} may not be registered with @register_builder() decorator")
                except (AttributeError, KeyError):
                    # Registry method not available or builder not found
                    self._log(f"Info: Could not verify registration for {class_name} - registry API may have changed")
                    
            except ImportError:
                self._log("StepBuilderRegistry not available, skipping registry integration test")
            except Exception as e:
                self._log(f"Registry integration test encountered error: {str(e)}")
    
    def test_documentation_standards(self) -> None:
        """Test that the builder meets documentation standards."""
        # Check class docstring
        self._assert(
            self.builder_class.__doc__ is not None,
            f"{self.builder_class.__name__} must have a class docstring"
        )
        
        if self.builder_class.__doc__:
            docstring = self.builder_class.__doc__.strip()
            self._assert(
                len(docstring) >= 30,
                f"Class docstring should be at least 30 characters, got {len(docstring)}"
            )
        
        # Check key method docstrings
        key_methods = ['validate_configuration', '_get_inputs', '_get_outputs', 'create_step']
        for method_name in key_methods:
            if hasattr(self.builder_class, method_name):
                method = getattr(self.builder_class, method_name)
                if not method.__doc__:
                    self._log(f"Warning: Method '{method_name}' is missing docstring")
    
    def test_type_hints(self) -> None:
        """Test that the builder uses proper type hints."""
        # Check if class has type hints for key methods
        key_methods = ['_get_inputs', '_get_outputs', 'create_step']
        
        for method_name in key_methods:
            if hasattr(self.builder_class, method_name):
                method = getattr(self.builder_class, method_name)
                try:
                    type_hints = get_type_hints(method)
                    if not type_hints:
                        self._log(f"Warning: Method '{method_name}' has no type hints")
                except Exception as e:
                    self._log(f"Could not get type hints for {method_name}: {str(e)}")
    
    def test_error_handling(self) -> None:
        """Test that the builder handles errors appropriately with proper exception types."""
        # Test with invalid configuration
        try:
            # Create config without required attributes
            invalid_config = self._create_invalid_config()
            
            # Try to create builder with invalid config
            try:
                builder = self.builder_class(
                    config=invalid_config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role,
                    registry_manager=self.mock_registry_manager,
                    dependency_resolver=self.mock_dependency_resolver
                )
                
                # Test validate_configuration raises ValueError
                try:
                    builder.validate_configuration()
                    self._log("Info: validate_configuration() did not raise exception - may have valid defaults")
                except ValueError:
                    # Good - it raised a ValueError as expected
                    self._log("Info: validate_configuration() correctly raised ValueError for invalid config")
                except Exception as e:
                    self._log(f"Info: validate_configuration() raised {type(e).__name__} instead of ValueError")
                    
            except ValueError as e:
                # Builder constructor itself validates config - this is also valid behavior
                self._log(f"Info: Builder constructor validates config and raised ValueError: {str(e)}")
            except Exception as e:
                # Other exceptions during construction may indicate real issues
                self._log(f"Warning: Builder constructor raised {type(e).__name__}: {str(e)}")
                
        except Exception as e:
            self._log(f"Error handling test encountered unexpected error: {str(e)}")
    
    def test_method_return_types(self) -> None:
        """Test that methods return appropriate types."""
        try:
            # Create valid builder instance
            builder = self._create_builder_instance()
            
            # Test _get_step_name returns string
            step_name = builder._get_step_name()
            self._assert(
                isinstance(step_name, str) and len(step_name) > 0,
                "_get_step_name() must return a non-empty string"
            )
            
            # Test _get_environment_variables returns dict
            env_vars = builder._get_environment_variables()
            self._assert(
                isinstance(env_vars, dict),
                "_get_environment_variables() must return a dictionary"
            )
            
            # Check that all env var keys and values are strings
            for key, value in env_vars.items():
                self._assert(
                    isinstance(key, str) and isinstance(value, str),
                    f"Environment variable {key} must have string key and value"
                )
            
            # Test _get_job_arguments returns list or None
            job_args = builder._get_job_arguments()
            if job_args is not None:
                self._assert(
                    isinstance(job_args, list),
                    "_get_job_arguments() must return a list or None"
                )
                
                # Check that all job arguments are strings
                for arg in job_args:
                    self._assert(
                        isinstance(arg, str),
                        f"Job argument must be string, got {type(arg)}"
                    )
                    
        except Exception as e:
            self._log(f"Method return type test failed: {str(e)}")
    
    def test_configuration_validation(self) -> None:
        """Test that the builder properly validates configuration parameters."""
        try:
            # Create valid builder instance
            builder = self._create_builder_instance()
            
            # Test that validate_configuration passes with valid config
            try:
                builder.validate_configuration()
                # Should not raise exception
            except Exception as e:
                self._assert(
                    False,
                    f"validate_configuration() should not raise exception with valid config: {str(e)}"
                )
            
            # Test configuration attribute access
            self._assert(
                hasattr(builder, 'config'),
                "Builder must have 'config' attribute"
            )
            
            # Check that config has essential attributes
            essential_attrs = ['region', 'pipeline_name']
            for attr in essential_attrs:
                if not hasattr(builder.config, attr):
                    self._log(f"Warning: Config missing essential attribute '{attr}'")
                    
        except Exception as e:
            self._log(f"Configuration validation test failed: {str(e)}")
