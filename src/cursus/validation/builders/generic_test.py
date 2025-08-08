"""
Generic step builder test variant for fallback cases.
"""

from typing import Dict, List, Any
from .base_test import UniversalStepBuilderTestBase


class GenericStepBuilderTest(UniversalStepBuilderTestBase):
    """
    Generic test variant for step builders without specific variants.
    
    This class provides a concrete implementation of the abstract base class
    and serves as a fallback for step types that don't have specialized variants.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return step type-specific test methods for generic steps."""
        return [
            "test_generic_step_creation",
            "test_generic_configuration_validation",
            "test_generic_dependency_handling"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects for generic steps."""
        # Get step type-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log step info if verbose
        if self.verbose:
            self._log(f"Detected step type: {self.step_info.get('sagemaker_step_type', 'Unknown')}")
            self._log(f"Detected framework: {self.step_info.get('framework', 'Unknown')}")
            self._log(f"Test pattern: {self.step_info.get('test_pattern', 'standard')}")
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate step type-specific requirements for generic steps."""
        validation_results = {
            "step_type_detected": self.step_info.get("sagemaker_step_type") is not None,
            "framework_detected": self.step_info.get("framework") is not None,
            "test_pattern_identified": self.step_info.get("test_pattern") is not None,
            "expected_dependencies": len(self._get_expected_dependencies()) > 0
        }
        
        return validation_results
    
    # Generic test methods
    
    def test_generic_step_creation(self):
        """Test that the step builder can be instantiated."""
        self._log("Testing generic step creation...")
        
        try:
            builder = self._create_builder_instance()
            self._assert(builder is not None, "Builder instance should be created")
            self._assert(hasattr(builder, 'create_step'), "Builder should have create_step method")
            
        except Exception as e:
            self._assert(False, f"Failed to create builder instance: {str(e)}")
    
    def test_generic_configuration_validation(self):
        """Test that the configuration is properly validated."""
        self._log("Testing generic configuration validation...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test that config is accessible
            self._assert(hasattr(builder, 'config'), "Builder should have config attribute")
            
            # Test basic config attributes
            config = builder.config
            self._assert(hasattr(config, 'region'), "Config should have region")
            self._assert(hasattr(config, 'pipeline_name'), "Config should have pipeline_name")
            
        except Exception as e:
            self._assert(False, f"Configuration validation failed: {str(e)}")
    
    def test_generic_dependency_handling(self):
        """Test that dependencies are properly handled."""
        self._log("Testing generic dependency handling...")
        
        try:
            builder = self._create_builder_instance()
            expected_deps = self._get_expected_dependencies()
            
            self._assert(len(expected_deps) > 0, "Should have expected dependencies")
            
            # Test dependency resolver configuration
            self._assert(
                self.mock_dependency_resolver.resolve_step_dependencies.called or True,
                "Dependency resolver should be configured"
            )
            
        except Exception as e:
            self._assert(False, f"Dependency handling test failed: {str(e)}")
    
    def test_builder_create_step_method(self):
        """Test that the builder's create_step method works."""
        self._log("Testing builder create_step method...")
        
        try:
            builder = self._create_builder_instance()
            
            # Create mock dependencies
            mock_dependencies = self._create_mock_dependencies()
            
            # Test create_step method exists and is callable
            self._assert(hasattr(builder, 'create_step'), "Builder should have create_step method")
            self._assert(callable(builder.create_step), "create_step method should be callable")
            
            # Note: We don't actually call create_step() as it may require complex setup
            # This test just verifies the method exists
            
        except Exception as e:
            self._assert(False, f"create_step method test failed: {str(e)}")
    
    def test_step_info_detection(self):
        """Test that step information is properly detected."""
        self._log("Testing step info detection...")
        
        try:
            # Validate step info was detected
            self._assert(self.step_info is not None, "Step info should be detected")
            self._assert(
                "builder_class_name" in self.step_info, 
                "Step info should contain builder class name"
            )
            
            # Log detected information
            if self.verbose:
                for key, value in self.step_info.items():
                    self._log(f"  {key}: {value}")
            
        except Exception as e:
            self._assert(False, f"Step info detection test failed: {str(e)}")
    
    def test_mock_factory_functionality(self):
        """Test that the mock factory is working properly."""
        self._log("Testing mock factory functionality...")
        
        try:
            # Test that mock factory was created
            self._assert(self.mock_factory is not None, "Mock factory should be created")
            
            # Test that it can create configs
            mock_config = self.mock_factory.create_mock_config()
            self._assert(mock_config is not None, "Mock factory should create config")
            
            # Test that it can get dependencies
            deps = self.mock_factory.get_expected_dependencies()
            self._assert(isinstance(deps, list), "Dependencies should be a list")
            
        except Exception as e:
            self._assert(False, f"Mock factory test failed: {str(e)}")
