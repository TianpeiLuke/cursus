"""
Transform Step Builder Test Variant

This module provides specialized validation for Transform step builders,
implementing essential validation patterns without complex integration logic.

Based on the Simplified Universal Step Builder Test Enhancement Plan Phase 1.
"""

from typing import List, Dict, Any
from unittest.mock import MagicMock

from ..base_test import UniversalStepBuilderTestBase


class TransformStepBuilderTest(UniversalStepBuilderTestBase):
    """
    Simple Transform step validation focused on essentials.
    
    This variant provides specialized validation for Transform step builders,
    focusing on core requirements without complex integration overhead.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Transform step-specific test methods."""
        return [
            "test_transformer_creation",
            "test_transform_inputs",
            "test_batch_strategy_configuration",
            "test_transform_outputs"
        ]
    
    def test_transformer_creation(self):
        """Test that builder creates appropriate transformer."""
        builder = self._create_builder_instance()
        
        # Check that builder has transformer creation method
        self._assert(
            hasattr(builder, '_create_transformer'),
            "Transform builder must implement _create_transformer method"
        )
        
        try:
            transformer = builder._create_transformer()
            
            self._assert(
                transformer is not None,
                "Transform builder should create a valid transformer"
            )
            
            # Check transformer type
            transformer_class_name = transformer.__class__.__name__
            self._assert(
                'Transform' in transformer_class_name,
                f"Transform builder should create Transformer, got {transformer_class_name}"
            )
            
            # Check basic transformer attributes
            self._assert(
                hasattr(transformer, 'model_name'),
                "Transformer should have model_name attribute"
            )
            
        except Exception as e:
            self._assert(False, f"Transformer creation failed: {str(e)}")
    
    def test_transform_inputs(self):
        """Test that builder handles transform inputs correctly."""
        builder = self._create_builder_instance()
        
        # Create mock transform dependencies
        mock_dependencies = self._create_transform_dependencies()
        
        try:
            # Test input extraction
            inputs = builder._get_inputs(mock_dependencies)
            
            self._assert(
                isinstance(inputs, list),
                "Transform builder _get_inputs must return a list"
            )
            
            # Check for TransformInput objects
            for input_obj in inputs:
                self._assert(
                    hasattr(input_obj, 'data_source') or hasattr(input_obj, 'source'),
                    "Transform inputs must have data_source or source"
                )
                
                # Check for content type if available
                if hasattr(input_obj, 'content_type'):
                    self._assert(
                        input_obj.content_type is not None,
                        "Transform input content_type should be specified"
                    )
                
        except Exception as e:
            self._assert(False, f"Transform input handling failed: {str(e)}")
    
    def test_batch_strategy_configuration(self):
        """Test that builder configures batch strategy correctly."""
        builder = self._create_builder_instance()
        
        try:
            transformer = builder._create_transformer()
            
            # Check for batch strategy configuration
            if hasattr(transformer, 'strategy'):
                self._assert(
                    transformer.strategy in ['SingleRecord', 'MultiRecord'],
                    f"Transform strategy should be SingleRecord or MultiRecord, got {transformer.strategy}"
                )
            
            # Check for max concurrent transforms
            if hasattr(transformer, 'max_concurrent_transforms'):
                self._assert(
                    isinstance(transformer.max_concurrent_transforms, int),
                    "max_concurrent_transforms should be an integer"
                )
                self._assert(
                    transformer.max_concurrent_transforms > 0,
                    "max_concurrent_transforms should be positive"
                )
            
            # Check for max payload
            if hasattr(transformer, 'max_payload'):
                self._assert(
                    isinstance(transformer.max_payload, int),
                    "max_payload should be an integer"
                )
                
        except Exception as e:
            self._assert(False, f"Batch strategy configuration failed: {str(e)}")
    
    def test_transform_outputs(self):
        """Test that builder handles transform outputs correctly."""
        builder = self._create_builder_instance()
        
        # Create mock output dependencies
        mock_outputs = {
            'transform_results': 's3://bucket/transform-results/',
            'transform_metadata': 's3://bucket/metadata/'
        }
        
        try:
            # Test output handling
            outputs = builder._get_outputs(mock_outputs)
            
            self._assert(
                isinstance(outputs, list),
                "Transform builder _get_outputs must return a list"
            )
            
            # Check for transform output
            transform_output_found = False
            for output_obj in outputs:
                if hasattr(output_obj, 'output_name'):
                    if 'transform' in output_obj.output_name.lower():
                        transform_output_found = True
                        break
            
            self._assert(
                transform_output_found,
                "Transform builder should have transform results output"
            )
            
        except Exception as e:
            self._assert(False, f"Transform output handling failed: {str(e)}")
    
    def _create_transform_dependencies(self) -> Dict[str, Any]:
        """Create mock transform dependencies for testing."""
        return {
            'input_data': 's3://bucket/input-data/',
            'model_artifacts': 's3://bucket/model/'
        }
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Transform step-specific mock objects."""
        super()._configure_step_type_mocks()
        
        # Add transform-specific mock configuration
        if hasattr(self.mock_config, 'transform_instance_type'):
            self.mock_config.transform_instance_type = 'ml.m5.large'
        if hasattr(self.mock_config, 'transform_instance_count'):
            self.mock_config.transform_instance_count = 1
        if hasattr(self.mock_config, 'transform_max_concurrent_transforms'):
            self.mock_config.transform_max_concurrent_transforms = 1
        if hasattr(self.mock_config, 'transform_max_payload'):
            self.mock_config.transform_max_payload = 6
        
        # Add transform strategy configuration
        if hasattr(self.mock_config, 'transform_strategy'):
            self.mock_config.transform_strategy = 'SingleRecord'
        if hasattr(self.mock_config, 'transform_output_path'):
            self.mock_config.transform_output_path = 's3://bucket/transform-output/'
