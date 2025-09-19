import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any
import logging
from abc import ABC

from cursus.core.base.builder_base import StepBuilderBase
from cursus.core.base.config_base import BasePipelineConfig


class MockConfig(BasePipelineConfig):
    """Mock configuration for testing."""

    def __init__(self):
        super().__init__(
            author="test_author",
            bucket="test-bucket",
            role="test-role",
            region="NA",
            service_name="test_service",
            pipeline_version="1.0.0",
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


class TestStepBuilderBase:
    """Test cases for StepBuilderBase class."""

    @pytest.fixture
    def config(self):
        """Set up test fixtures."""
        return MockConfig()

    @pytest.fixture
    def mock_session(self):
        return Mock()

    @pytest.fixture
    def role(self):
        return "arn:aws:iam::123456789012:role/TestRole"

    @pytest.fixture
    def notebook_root(self):
        return Path("/test/notebook")

    @pytest.fixture
    def mock_registry_manager(self):
        return Mock()

    @pytest.fixture
    def mock_dependency_resolver(self):
        return Mock()

    @pytest.fixture
    def mock_spec(self):
        return Mock()

    def test_init_with_required_params(self, config, mock_session, role, notebook_root):
        """Test initialization with required parameters."""
        builder = ConcreteStepBuilder(
            config=config,
            sagemaker_session=mock_session,
            role=role,
            notebook_root=notebook_root,
        )

        assert builder.config == config
        assert builder.session == mock_session
        assert builder.role == role
        assert builder.notebook_root == notebook_root
        assert builder.aws_region == "us-east-1"  # NA maps to us-east-1

    def test_init_with_optional_params(
        self,
        config,
        mock_session,
        role,
        notebook_root,
        mock_registry_manager,
        mock_dependency_resolver,
        mock_spec,
    ):
        """Test initialization with optional parameters."""
        builder = ConcreteStepBuilder(
            config=config,
            spec=mock_spec,
            sagemaker_session=mock_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        assert builder.spec == mock_spec
        assert builder._registry_manager == mock_registry_manager
        assert builder._dependency_resolver == mock_dependency_resolver

    def test_invalid_region_raises_error(self):
        """Test that invalid region raises ValidationError during config creation."""
        # Pydantic now validates the region during model construction
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            invalid_config = BasePipelineConfig(
                author="test_author",
                bucket="test-bucket",
                role="test-role",
                region="INVALID",
                service_name="test_service",
                pipeline_version="1.0.0",
            )

        # Check that the error message mentions the invalid region
        error_str = str(exc_info.value)
        assert "INVALID" in error_str
        assert "region" in error_str.lower()

    def test_sanitize_name_for_sagemaker(self, config):
        """Test name sanitization for SageMaker."""
        builder = ConcreteStepBuilder(config=config)

        # Test normal name
        sanitized = builder._sanitize_name_for_sagemaker("test-name")
        assert sanitized == "test-name"

        # Test name with special characters
        sanitized = builder._sanitize_name_for_sagemaker("test@name#with$special%chars")
        assert sanitized == "test-name-with-special-chars"

        # Test empty name
        sanitized = builder._sanitize_name_for_sagemaker("")
        assert sanitized == "default-name"

        # Test name too long
        long_name = "a" * 100
        sanitized = builder._sanitize_name_for_sagemaker(long_name, max_length=10)
        assert len(sanitized) == 10

    def test_get_step_name(self, config):
        """Test step name generation."""
        builder = ConcreteStepBuilder(config=config)

        # Test without job_type
        step_name = builder._get_step_name(include_job_type=False)
        assert step_name == "Concrete"  # Current implementation returns "Concrete"

        # Test with job_type
        config.job_type = "training"
        step_name = builder._get_step_name(include_job_type=True)
        assert step_name == "Concrete-Training"  # Current format uses dash separator

    def test_generate_job_name(self, config):
        """Test job name generation."""
        builder = ConcreteStepBuilder(config=config)

        with patch("time.time", return_value=1234567890):
            job_name = builder._generate_job_name()
            assert "Concrete" in job_name  # Current implementation uses "Concrete"
            assert "1234567890" in job_name
            # Verify the expected format: "Concrete-1234567890"
            assert job_name == "Concrete-1234567890"

    def test_get_property_path(self, config, mock_spec):
        """Test property path retrieval."""
        # Mock specification with outputs
        mock_output_spec = Mock()
        mock_output_spec.logical_name = "test_output"
        mock_output_spec.property_path = "Steps.TestStep.Properties.{output_descriptor}"

        mock_spec.outputs = {"output1": mock_output_spec}

        builder = ConcreteStepBuilder(config=config, spec=mock_spec)

        # Test without format args
        path = builder.get_property_path("test_output")
        assert path == "Steps.TestStep.Properties.{output_descriptor}"

        # Test with format args
        path = builder.get_property_path("test_output", {"output_descriptor": "data"})
        assert path == "Steps.TestStep.Properties.data"

        # Test non-existent output
        path = builder.get_property_path("nonexistent")
        assert path is None

    def test_get_all_property_paths(self, config, mock_spec):
        """Test getting all property paths."""
        # Mock specification with multiple outputs
        mock_output1 = Mock()
        mock_output1.logical_name = "output1"
        mock_output1.property_path = "Steps.TestStep.Output1"

        mock_output2 = Mock()
        mock_output2.logical_name = "output2"
        mock_output2.property_path = "Steps.TestStep.Output2"

        mock_spec.outputs = {"out1": mock_output1, "out2": mock_output2}

        builder = ConcreteStepBuilder(config=config, spec=mock_spec)

        paths = builder.get_all_property_paths()
        expected = {
            "output1": "Steps.TestStep.Output1",
            "output2": "Steps.TestStep.Output2",
        }
        assert paths == expected

    def test_safe_logging_methods(self, config):
        """Test safe logging methods."""
        builder = ConcreteStepBuilder(config=config)

        # Test that logging methods don't raise exceptions
        builder.log_info("Test info message: %s", "test_value")
        builder.log_debug("Test debug message: %s", "test_value")
        builder.log_warning("Test warning message: %s", "test_value")
        builder.log_error("Test error message: %s", "test_value")

    def test_get_cache_config(self, config):
        """Test cache configuration."""
        builder = ConcreteStepBuilder(config=config)

        cache_config = builder._get_cache_config(enable_caching=True)
        assert cache_config is not None

        cache_config = builder._get_cache_config(enable_caching=False)
        assert cache_config is not None

    def test_get_environment_variables_no_contract(self, config):
        """Test environment variables without contract."""
        builder = ConcreteStepBuilder(config=config)

        env_vars = builder._get_environment_variables()
        assert env_vars == {}

    def test_get_environment_variables_with_contract(self, config):
        """Test environment variables with contract."""
        # Mock contract
        mock_contract = Mock()
        mock_contract.required_env_vars = ["TEST_VAR"]
        mock_contract.optional_env_vars = {"OPTIONAL_VAR": "default_value"}

        builder = ConcreteStepBuilder(config=config)
        builder.contract = mock_contract

        # Add test_var to config
        config.test_var = "test_value"

        env_vars = builder._get_environment_variables()

        assert "TEST_VAR" in env_vars
        assert env_vars["TEST_VAR"] == "test_value"
        assert "OPTIONAL_VAR" in env_vars
        assert env_vars["OPTIONAL_VAR"] == "default_value"

    def test_get_job_arguments_no_contract(self, config):
        """Test job arguments without contract."""
        builder = ConcreteStepBuilder(config=config)

        args = builder._get_job_arguments()
        assert args is None

    def test_get_job_arguments_with_contract(self, config):
        """Test job arguments with contract."""
        # Mock contract
        mock_contract = Mock()
        mock_contract.expected_arguments = {
            "input-path": "/test/input",
            "output-path": "/test/output",
        }

        builder = ConcreteStepBuilder(config=config)
        builder.contract = mock_contract

        args = builder._get_job_arguments()
        expected = ["--input-path", "/test/input", "--output-path", "/test/output"]
        assert args == expected

    def test_get_required_dependencies(self, config, mock_spec):
        """Test getting required dependencies."""
        # Mock specification with dependencies
        mock_dep1 = Mock()
        mock_dep1.logical_name = "dep1"
        mock_dep1.required = True

        mock_dep2 = Mock()
        mock_dep2.logical_name = "dep2"
        mock_dep2.required = False

        mock_spec.dependencies = {"d1": mock_dep1, "d2": mock_dep2}

        builder = ConcreteStepBuilder(config=config, spec=mock_spec)

        required_deps = builder.get_required_dependencies()
        assert required_deps == ["dep1"]

    def test_get_optional_dependencies(self, config, mock_spec):
        """Test getting optional dependencies."""
        # Mock specification with dependencies
        mock_dep1 = Mock()
        mock_dep1.logical_name = "dep1"
        mock_dep1.required = True

        mock_dep2 = Mock()
        mock_dep2.logical_name = "dep2"
        mock_dep2.required = False

        mock_spec.dependencies = {"d1": mock_dep1, "d2": mock_dep2}

        builder = ConcreteStepBuilder(config=config, spec=mock_spec)

        optional_deps = builder.get_optional_dependencies()
        assert optional_deps == ["dep2"]

    def test_get_outputs(self, config, mock_spec):
        """Test getting outputs."""
        # Mock specification with outputs
        mock_output1 = Mock()
        mock_output1.logical_name = "output1"

        mock_output2 = Mock()
        mock_output2.logical_name = "output2"

        mock_spec.outputs = {"out1": mock_output1, "out2": mock_output2}

        builder = ConcreteStepBuilder(config=config, spec=mock_spec)

        outputs = builder.get_outputs()
        expected = {"output1": mock_output1, "output2": mock_output2}
        assert outputs == expected

    def test_get_context_name(self, config):
        """Test getting context name."""
        builder = ConcreteStepBuilder(config=config)

        context_name = builder._get_context_name()
        # Should use pipeline_name from config
        expected = config.pipeline_name
        assert context_name == expected

    def test_abstract_methods_must_be_implemented(self, config):
        """Test that abstract methods must be implemented."""
        # This should fail because abstract methods are not implemented
        with pytest.raises(TypeError):

            class IncompleteBuilder(StepBuilderBase):
                pass

            IncompleteBuilder(config=config)

    def test_region_mapping(self):
        """Test region mapping."""
        # Test all supported regions
        region_tests = [("NA", "us-east-1"), ("EU", "eu-west-1"), ("FE", "us-west-2")]

        for region_code, expected_aws_region in region_tests:
            config = BasePipelineConfig(
                author="test_author",
                bucket="test-bucket",
                role="test-role",
                region=region_code,
                service_name="test_service",
                pipeline_version="1.0.0",
            )

            builder = ConcreteStepBuilder(config=config)
            assert builder.aws_region == expected_aws_region

    def test_step_names_class_variable(self, config):
        """Test STEP_NAMES class variable."""
        builder = ConcreteStepBuilder(config=config)

        # Should have STEP_NAMES attribute
        assert hasattr(builder, "STEP_NAMES")
        assert isinstance(builder.STEP_NAMES, dict)

    def test_common_properties_class_variable(self, config):
        """Test COMMON_PROPERTIES class variable."""
        builder = ConcreteStepBuilder(config=config)

        # Should have COMMON_PROPERTIES attribute
        assert hasattr(builder, "COMMON_PROPERTIES")
        assert isinstance(builder.COMMON_PROPERTIES, dict)

        # Should contain expected common properties
        assert "dependencies" in builder.COMMON_PROPERTIES
        assert "enable_caching" in builder.COMMON_PROPERTIES

    def test_set_execution_prefix(self, config):
        """Test setting execution prefix for dynamic output path resolution."""
        builder = ConcreteStepBuilder(config=config)
        
        # Test with ParameterString
        from sagemaker.workflow.parameters import ParameterString
        param = ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://test-bucket/execution")
        builder.set_execution_prefix(param)
        assert builder.execution_prefix == param
        
        # Test with string
        builder.set_execution_prefix("s3://test-bucket/string-path")
        assert builder.execution_prefix == "s3://test-bucket/string-path"
        
        # Test with None
        builder.set_execution_prefix(None)
        assert builder.execution_prefix is None

    def test_get_base_output_path_with_execution_prefix(self, config):
        """Test _get_base_output_path with execution prefix set."""
        builder = ConcreteStepBuilder(config=config)
        
        # Test with execution_prefix set
        from sagemaker.workflow.parameters import ParameterString
        param = ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://test-bucket/execution")
        builder.set_execution_prefix(param)
        
        result = builder._get_base_output_path()
        assert result == param

    def test_get_base_output_path_fallback_to_config(self, config):
        """Test _get_base_output_path falls back to config.pipeline_s3_loc."""
        builder = ConcreteStepBuilder(config=config)
        
        # No execution_prefix set, should fall back to config
        result = builder._get_base_output_path()
        assert result == config.pipeline_s3_loc

    def test_execution_prefix_initialization(self, config):
        """Test that execution_prefix is properly initialized."""
        builder = ConcreteStepBuilder(config=config)
        
        # Should be initialized to None
        assert hasattr(builder, 'execution_prefix')
        assert builder.execution_prefix is None
