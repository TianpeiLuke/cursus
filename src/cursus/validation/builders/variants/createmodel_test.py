"""
CreateModel Step Builder Test Variant

This module provides specialized validation for CreateModel step builders,
implementing essential validation patterns without complex integration logic.

Based on the Simplified Universal Step Builder Test Enhancement Plan Phase 1.
"""

from typing import List, Dict, Any
from unittest.mock import MagicMock

from ..base_test import UniversalStepBuilderTestBase


class CreateModelStepBuilderTest(UniversalStepBuilderTestBase):
    """
    Simple CreateModel step validation focused on essentials.
    
    This variant provides specialized validation for CreateModel step builders,
    focusing on core requirements without complex integration overhead.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return CreateModel step-specific test methods."""
        return [
            "test_model_creation",
            "test_container_definitions",
            "test_model_data_handling",
            "test_inference_configuration"
        ]
    
    def test_model_creation(self):
        """Test that builder creates appropriate model."""
        builder = self._create_builder_instance()
        
        # Check that builder has model creation method
        self._assert(
            hasattr(builder, '_create_model'),
            "CreateModel builder must implement _create_model method"
        )
        
        try:
            model = builder._create_model()
            
            self._assert(
                model is not None,
                "CreateModel builder should create a valid model"
            )
            
            # Check model type
            model_class_name = model.__class__.__name__
            self._assert(
                'Model' in model_class_name,
                f"CreateModel builder should create Model, got {model_class_name}"
            )
            
            # Check basic model attributes
            self._assert(
                hasattr(model, 'name'),
                "Model should have name attribute"
            )
            
            self._assert(
                hasattr(model, 'role'),
                "Model should have role attribute"
            )
            
        except Exception as e:
            self._assert(False, f"Model creation failed: {str(e)}")
    
    def test_container_definitions(self):
        """Test that builder configures container definitions correctly."""
        builder = self._create_builder_instance()
        
        try:
            model = builder._create_model()
            
            # Check for container definitions
            if hasattr(model, 'containers'):
                self._assert(
                    isinstance(model.containers, list),
                    "Model containers should be a list"
                )
                
                if model.containers:
                    container = model.containers[0]
                    
                    # Check container has image
                    self._assert(
                        hasattr(container, 'image'),
                        "Container should have image attribute"
                    )
                    
                    # Check container has model data URL
                    self._assert(
                        hasattr(container, 'model_data_url'),
                        "Container should have model_data_url attribute"
                    )
            
            # Check for primary container
            if hasattr(model, 'primary_container'):
                container = model.primary_container
                
                self._assert(
                    hasattr(container, 'image'),
                    "Primary container should have image attribute"
                )
                
                self._assert(
                    hasattr(container, 'model_data_url'),
                    "Primary container should have model_data_url attribute"
                )
                
        except Exception as e:
            self._assert(False, f"Container definition validation failed: {str(e)}")
    
    def test_model_data_handling(self):
        """Test that builder handles model data correctly."""
        builder = self._create_builder_instance()
        
        # Create mock model dependencies
        mock_dependencies = self._create_model_dependencies()
        
        try:
            # Test input extraction for model artifacts
            inputs = builder._get_inputs(mock_dependencies)
            
            self._assert(
                isinstance(inputs, list),
                "CreateModel builder _get_inputs must return a list"
            )
            
            # Check for model artifact input
            model_input_found = False
            for input_obj in inputs:
                if hasattr(input_obj, 'input_name'):
                    if 'model' in input_obj.input_name.lower():
                        model_input_found = True
                        break
            
            # Model input might not be required for all CreateModel steps
            # so this is a soft check
            if not model_input_found:
                self._log("No explicit model input found - may be handled differently")
                
        except Exception as e:
            self._assert(False, f"Model data handling failed: {str(e)}")
    
    def test_inference_configuration(self):
        """Test that builder configures inference settings correctly."""
        builder = self._create_builder_instance()
        
        try:
            model = builder._create_model()
            
            # Check for inference configuration
            if hasattr(model, 'enable_network_isolation'):
                self._assert(
                    isinstance(model.enable_network_isolation, bool),
                    "enable_network_isolation should be a boolean"
                )
            
            # Check for VPC configuration
            if hasattr(model, 'vpc_config'):
                if model.vpc_config is not None:
                    self._assert(
                        isinstance(model.vpc_config, dict),
                        "vpc_config should be a dictionary"
                    )
            
            # Check for execution role
            if hasattr(model, 'role'):
                self._assert(
                    model.role is not None,
                    "Model should have execution role"
                )
                
        except Exception as e:
            self._assert(False, f"Inference configuration validation failed: {str(e)}")
    
    def _create_model_dependencies(self) -> Dict[str, Any]:
        """Create mock model dependencies for testing."""
        return {
            'model_artifacts': 's3://bucket/model/',
            'inference_code': 's3://bucket/inference-code/'
        }
    
    def _configure_step_type_mocks(self) -> None:
        """Configure CreateModel step-specific mock objects."""
        super()._configure_step_type_mocks()
        
        # Add model-specific mock configuration
        if hasattr(self.mock_config, 'model_name'):
            self.mock_config.model_name = 'test-model'
        if hasattr(self.mock_config, 'image_uri'):
            self.mock_config.image_uri = '123456789012.dkr.ecr.us-east-1.amazonaws.com/test:latest'
        if hasattr(self.mock_config, 'model_data_url'):
            self.mock_config.model_data_url = 's3://bucket/model/model.tar.gz'
        
        # Add inference configuration
        if hasattr(self.mock_config, 'enable_network_isolation'):
            self.mock_config.enable_network_isolation = False
        if hasattr(self.mock_config, 'vpc_config'):
            self.mock_config.vpc_config = None
