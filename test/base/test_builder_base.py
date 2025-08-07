import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any
import logging
from abc import ABC

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cursus.core.base.builder_base import StepBuilderBase
from src.cursus.core.base.config_base import BasePipelineConfig


class MockConfig(BasePipelineConfig):
    """Mock configuration for testing."""
    
    def __init__(self):
        super().__init__(
            author='test_author',
            bucket='test-bucket',
            role='test-role',
            region='NA',
            service_name='test_service',
            pipeline_version='1.0.0'
        )


class ConcreteStepBuilder(StepBuilderBase):
    """Concrete implementation of StepBuilderBase for testing."""
    
    def validate_configuration(self) -> None:
        """Mock validation."""
        pass
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        """Mock implementation."""
        return inputs
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Mock implementation."""
        return outputs
    
    def create_step(self, **kwargs):
        """Mock implementation."""
        mock_step = Mock()
        mock_step.name = "test_step"
        return mock_step


class TestStepBuilderBase(unittest.TestCase):
    """Test cases for StepBuilderBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MockConfig()
        self.mock_session = Mock()
        self.role = "arn:aws:iam::123456789012:role/TestRole"
        self.notebook_root = Path("/test/notebook")
        self.mock_registry_manager = Mock()
        self.mock_dependency_resolver = Mock()
        self.mock_spec = Mock()
    
    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        builder = ConcreteStepBuilder(
            config=self.config,
            sagemaker_session=self.mock_session,
            role=self.role,
            notebook_root=self.notebook_root
        )
        
        self.assertEqual(builder.config, self.config)
        self.assertEqual(builder.session, self.mock_session)
        self.assertEqual(builder.role, self.role)
        self.assertEqual(builder.notebook_root, self.notebook_root)
        self.assertEqual(builder.aws_region, 'us-east-1')  # NA maps to us-east-1
    
    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec,
            sagemaker_session=self.mock_session,
            role=self.role,
            notebook_root=self.notebook_root,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        self.assertEqual(builder.spec, self.mock_spec)
        self.assertEqual(builder._registry_manager, self.mock_registry_manager)
        self.assertEqual(builder._dependency_resolver, self.mock_dependency_resolver)
    
    def test_invalid_region_raises_error(self):
        """Test that invalid region raises ValueError."""
        invalid_config = BasePipelineConfig(
            author='test_author',
            bucket='test-bucket',
            role='test-role',
            region='INVALID',
            service_name='test_service',
            pipeline_version='1.0.0'
        )
        
        with self.assertRaises(ValueError) as context:
            ConcreteStepBuilder(config=invalid_config)
        
        self.assertIn("Invalid region code", str(context.exception))
    
    def test_sanitize_name_for_sagemaker(self):
        """Test name sanitization for SageMaker."""
        builder = ConcreteStepBuilder(config=self.config)
        
        # Test normal name
        sanitized = builder._sanitize_name_for_sagemaker("test-name")
        self.assertEqual(sanitized, "test-name")
        
        # Test name with special characters
        sanitized = builder._sanitize_name_for_sagemaker("test@name#with$special%chars")
        self.assertEqual(sanitized, "test-name-with-special-chars")
        
        # Test empty name
        sanitized = builder._sanitize_name_for_sagemaker("")
        self.assertEqual(sanitized, "default-name")
        
        # Test name too long
        long_name = "a" * 100
        sanitized = builder._sanitize_name_for_sagemaker(long_name, max_length=10)
        self.assertEqual(len(sanitized), 10)
    
    def test_get_step_name(self):
        """Test step name generation."""
        builder = ConcreteStepBuilder(config=self.config)
        
        # Test without job_type
        step_name = builder._get_step_name(include_job_type=False)
        self.assertEqual(step_name, "ConcreteStep")  # Removes "Builder" suffix
        
        # Test with job_type
        self.config.job_type = "training"
        step_name = builder._get_step_name(include_job_type=True)
        self.assertEqual(step_name, "ConcreteStep-Training")
    
    def test_generate_job_name(self):
        """Test job name generation."""
        builder = ConcreteStepBuilder(config=self.config)
        
        with patch('time.time', return_value=1234567890):
            job_name = builder._generate_job_name()
            self.assertIn("ConcreteStep", job_name)
            self.assertIn("1234567890", job_name)
    
    def test_get_property_path(self):
        """Test property path retrieval."""
        # Mock specification with outputs
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "test_output"
        mock_output_spec.property_path = "Steps.TestStep.Properties.{output_descriptor}"
        
        self.mock_spec.outputs = {"output1": mock_output_spec}
        
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec
        )
        
        # Test without format args
        path = builder.get_property_path("test_output")
        self.assertEqual(path, "Steps.TestStep.Properties.{output_descriptor}")
        
        # Test with format args
        path = builder.get_property_path("test_output", {"output_descriptor": "data"})
        self.assertEqual(path, "Steps.TestStep.Properties.data")
        
        # Test non-existent output
        path = builder.get_property_path("nonexistent")
        self.assertIsNone(path)
    
    def test_get_all_property_paths(self):
        """Test getting all property paths."""
        # Mock specification with multiple outputs
        mock_output1 = Mock()
        mock_output1.logical_name = "output1"
        mock_output1.property_path = "Steps.TestStep.Output1"
        
        mock_output2 = Mock()
        mock_output2.logical_name = "output2"
        mock_output2.property_path = "Steps.TestStep.Output2"
        
        self.mock_spec.outputs = {
            "out1": mock_output1,
            "out2": mock_output2
        }
        
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec
        )
        
        paths = builder.get_all_property_paths()
        expected = {
            "output1": "Steps.TestStep.Output1",
            "output2": "Steps.TestStep.Output2"
        }
        self.assertEqual(paths, expected)
    
    def test_safe_logging_methods(self):
        """Test safe logging methods."""
        builder = ConcreteStepBuilder(config=self.config)
        
        # Test that logging methods don't raise exceptions
        try:
            builder.log_info("Test info message: %s", "test_value")
            builder.log_debug("Test debug message: %s", "test_value")
            builder.log_warning("Test warning message: %s", "test_value")
            builder.log_error("Test error message: %s", "test_value")
        except Exception as e:
            self.fail(f"Safe logging methods raised an exception: {e}")
    
    def test_get_cache_config(self):
        """Test cache configuration."""
        builder = ConcreteStepBuilder(config=self.config)
        
        cache_config = builder._get_cache_config(enable_caching=True)
        self.assertIsNotNone(cache_config)
        
        cache_config = builder._get_cache_config(enable_caching=False)
        self.assertIsNotNone(cache_config)
    
    def test_get_environment_variables_no_contract(self):
        """Test environment variables without contract."""
        builder = ConcreteStepBuilder(config=self.config)
        
        env_vars = builder._get_environment_variables()
        self.assertEqual(env_vars, {})
    
    def test_get_environment_variables_with_contract(self):
        """Test environment variables with contract."""
        # Mock contract
        mock_contract = Mock()
        mock_contract.required_env_vars = ["TEST_VAR"]
        mock_contract.optional_env_vars = {"OPTIONAL_VAR": "default_value"}
        
        builder = ConcreteStepBuilder(config=self.config)
        builder.contract = mock_contract
        
        # Add test_var to config
        self.config.test_var = "test_value"
        
        env_vars = builder._get_environment_variables()
        
        self.assertIn("TEST_VAR", env_vars)
        self.assertEqual(env_vars["TEST_VAR"], "test_value")
        self.assertIn("OPTIONAL_VAR", env_vars)
        self.assertEqual(env_vars["OPTIONAL_VAR"], "default_value")
    
    def test_get_job_arguments_no_contract(self):
        """Test job arguments without contract."""
        builder = ConcreteStepBuilder(config=self.config)
        
        args = builder._get_job_arguments()
        self.assertIsNone(args)
    
    def test_get_job_arguments_with_contract(self):
        """Test job arguments with contract."""
        # Mock contract
        mock_contract = Mock()
        mock_contract.expected_arguments = {
            "input-path": "/test/input",
            "output-path": "/test/output"
        }
        
        builder = ConcreteStepBuilder(config=self.config)
        builder.contract = mock_contract
        
        args = builder._get_job_arguments()
        expected = ["--input-path", "/test/input", "--output-path", "/test/output"]
        self.assertEqual(args, expected)
    
    def test_get_required_dependencies(self):
        """Test getting required dependencies."""
        # Mock specification with dependencies
        mock_dep1 = Mock()
        mock_dep1.logical_name = "dep1"
        mock_dep1.required = True
        
        mock_dep2 = Mock()
        mock_dep2.logical_name = "dep2"
        mock_dep2.required = False
        
        self.mock_spec.dependencies = {
            "d1": mock_dep1,
            "d2": mock_dep2
        }
        
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec
        )
        
        required_deps = builder.get_required_dependencies()
        self.assertEqual(required_deps, ["dep1"])
    
    def test_get_optional_dependencies(self):
        """Test getting optional dependencies."""
        # Mock specification with dependencies
        mock_dep1 = Mock()
        mock_dep1.logical_name = "dep1"
        mock_dep1.required = True
        
        mock_dep2 = Mock()
        mock_dep2.logical_name = "dep2"
        mock_dep2.required = False
        
        self.mock_spec.dependencies = {
            "d1": mock_dep1,
            "d2": mock_dep2
        }
        
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec
        )
        
        optional_deps = builder.get_optional_dependencies()
        self.assertEqual(optional_deps, ["dep2"])
    
    def test_get_outputs(self):
        """Test getting outputs."""
        # Mock specification with outputs
        mock_output1 = Mock()
        mock_output1.logical_name = "output1"
        
        mock_output2 = Mock()
        mock_output2.logical_name = "output2"
        
        self.mock_spec.outputs = {
            "out1": mock_output1,
            "out2": mock_output2
        }
        
        builder = ConcreteStepBuilder(
            config=self.config,
            spec=self.mock_spec
        )
        
        outputs = builder.get_outputs()
        expected = {
            "output1": mock_output1,
            "output2": mock_output2
        }
        self.assertEqual(outputs, expected)
    
    def test_get_context_name(self):
        """Test getting context name."""
        builder = ConcreteStepBuilder(config=self.config)
        
        context_name = builder._get_context_name()
        # Should use pipeline_name from config
        expected = self.config.pipeline_name
        self.assertEqual(context_name, expected)
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        # This should fail because abstract methods are not implemented
        with self.assertRaises(TypeError):
            class IncompleteBuilder(StepBuilderBase):
                pass
            
            IncompleteBuilder(config=self.config)
    
    def test_region_mapping(self):
        """Test region mapping."""
        # Test all supported regions
        region_tests = [
            ('NA', 'us-east-1'),
            ('EU', 'eu-west-1'),
            ('FE', 'us-west-2')
        ]
        
        for region_code, expected_aws_region in region_tests:
            config = BasePipelineConfig(
                author='test_author',
                bucket='test-bucket',
                role='test-role',
                region=region_code,
                service_name='test_service',
                pipeline_version='1.0.0'
            )
            
            builder = ConcreteStepBuilder(config=config)
            self.assertEqual(builder.aws_region, expected_aws_region)
    
    def test_step_names_class_variable(self):
        """Test STEP_NAMES class variable."""
        builder = ConcreteStepBuilder(config=self.config)
        
        # Should have STEP_NAMES attribute
        self.assertTrue(hasattr(builder, 'STEP_NAMES'))
        self.assertIsInstance(builder.STEP_NAMES, dict)
    
    def test_common_properties_class_variable(self):
        """Test COMMON_PROPERTIES class variable."""
        builder = ConcreteStepBuilder(config=self.config)
        
        # Should have COMMON_PROPERTIES attribute
        self.assertTrue(hasattr(builder, 'COMMON_PROPERTIES'))
        self.assertIsInstance(builder.COMMON_PROPERTIES, dict)
        
        # Should contain expected common properties
        self.assertIn('dependencies', builder.COMMON_PROPERTIES)
        self.assertIn('enable_caching', builder.COMMON_PROPERTIES)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
