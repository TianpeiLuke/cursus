"""
Unit tests for the dag_compiler module.

This module tests the pipeline compilation process, particularly focusing on the 
conversion of PipelineDAG structures to SageMaker pipelines.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.compiler.dag_compiler import (
    PipelineDAGCompiler,
    compile_dag_to_pipeline,
)
from cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from cursus.core.compiler.name_generator import validate_pipeline_name
from cursus.core.compiler.exceptions import PipelineAPIError, ConfigurationError
from cursus.core.compiler.validation import (
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
)


class TestDagCompiler:
    """Tests for the dag_compiler module."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple DAG for testing
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("preprocessing")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "preprocessing")
        self.dag.add_edge("preprocessing", "training")

        # Mock config path (doesn't need to exist for these tests)
        self.config_path = "mock_config.json"

        # Mock pipeline session and role
        self.mock_session = MagicMock()
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerRole"

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dynamic_template.DynamicPipelineTemplate")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    def test_compile_with_custom_pipeline_name(
        self, mock_registry_class, mock_template_class, mock_path
    ):
        """Test that custom pipeline names are used directly."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        mock_template = MagicMock()
        mock_pipeline = MagicMock()
        mock_template.generate_pipeline.return_value = mock_pipeline
        mock_template_class.return_value = mock_template

        # Create a compiler with the mocked template
        compiler = PipelineDAGCompiler(
            config_path=self.config_path, builder_registry=mock_registry
        )

        # Test with custom pipeline name
        result = compiler.compile(self.dag, pipeline_name="custom-pipeline-name")

        # Verify custom name is used directly
        assert result.name == "custom-pipeline-name"

        yield


class TestCompileDagToPipeline:
    """Tests for the compile_dag_to_pipeline function."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "training")

        self.config_path = "test_config.json"
        self.mock_session = MagicMock()
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerRole"

        yield

    def test_compile_dag_to_pipeline_invalid_dag(self):
        """Test compile_dag_to_pipeline with invalid DAG."""
        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag="not_a_dag",
                config_path=self.config_path,
                sagemaker_session=self.mock_session,
                role=self.mock_role,
            )
        assert "dag must be a PipelineDAG instance" in str(context.value)

    def test_compile_dag_to_pipeline_empty_dag(self):
        """Test compile_dag_to_pipeline with empty DAG."""
        empty_dag = PipelineDAG()

        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag=empty_dag,
                config_path=self.config_path,
                sagemaker_session=self.mock_session,
                role=self.mock_role,
            )
        assert "DAG must contain at least one node" in str(context.value)

    def test_compile_dag_to_pipeline_missing_config_file(self):
        """Test compile_dag_to_pipeline with missing config file."""
        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag=self.dag,
                config_path="nonexistent_config.json",
                sagemaker_session=self.mock_session,
                role=self.mock_role,
            )
        assert "Configuration file not found" in str(context.value)

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.PipelineDAGCompiler")
    def test_compile_dag_to_pipeline_success(self, mock_compiler_class, mock_path):
        """Test successful compile_dag_to_pipeline execution."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_compiler = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.name = "test-pipeline"
        mock_compiler.compile.return_value = mock_pipeline
        mock_compiler_class.return_value = mock_compiler

        # Call function
        result = compile_dag_to_pipeline(
            dag=self.dag,
            config_path=self.config_path,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
            pipeline_name="custom-pipeline",
        )

        # Verify compiler was created with correct arguments
        mock_compiler_class.assert_called_once_with(
            config_path=self.config_path,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
        )

        # Verify compile was called with correct arguments
        mock_compiler.compile.assert_called_once_with(
            self.dag, pipeline_name="custom-pipeline"
        )

        # Verify result
        assert result == mock_pipeline

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.PipelineDAGCompiler")
    def test_compile_dag_to_pipeline_exception_handling(
        self, mock_compiler_class, mock_path
    ):
        """Test exception handling in compile_dag_to_pipeline."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_compiler = MagicMock()
        mock_compiler.compile.side_effect = Exception("Compilation failed")
        mock_compiler_class.return_value = mock_compiler

        # Call function and expect PipelineAPIError
        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag=self.dag,
                config_path=self.config_path,
                sagemaker_session=self.mock_session,
                role=self.mock_role,
            )

        assert "DAG compilation failed" in str(context.value)


class TestPipelineDAGCompilerInit:
    """Tests for PipelineDAGCompiler initialization."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.config_path = "test_config.json"
        self.mock_session = MagicMock()
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerRole"

        yield

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    @patch("cursus.core.compiler.dag_compiler.ValidationEngine")
    def test_compiler_init_success(
        self, mock_validation_engine, mock_resolver, mock_registry, mock_path
    ):
        """Test successful PipelineDAGCompiler initialization."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
        )

        # Verify initialization
        assert compiler.config_path == self.config_path
        assert compiler.sagemaker_session == self.mock_session
        assert compiler.role == self.mock_role
        assert compiler.config_resolver is not None
        assert compiler.builder_registry is not None
        assert compiler.validation_engine is not None
        assert compiler._last_template is None

    def test_compiler_init_missing_config_file(self):
        """Test PipelineDAGCompiler initialization with missing config file."""
        with pytest.raises(FileNotFoundError) as context:
            PipelineDAGCompiler(
                config_path="nonexistent_config.json",
                sagemaker_session=self.mock_session,
                role=self.mock_role,
            )
        assert "Configuration file not found" in str(context.value)

    @patch("cursus.core.compiler.dag_compiler.Path")
    def test_compiler_init_with_custom_components(self, mock_path):
        """Test PipelineDAGCompiler initialization with custom components."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        custom_resolver = MagicMock()
        custom_registry = MagicMock()

        # Create compiler with custom components
        compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            config_resolver=custom_resolver,
            builder_registry=custom_registry,
        )

        # Verify custom components are used
        assert compiler.config_resolver == custom_resolver
        assert compiler.builder_registry == custom_registry


class TestPipelineDAGCompilerValidation:
    """Tests for PipelineDAGCompiler validation methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "training")

        self.config_path = "test_config.json"

        yield

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    @patch("cursus.core.compiler.dag_compiler.ValidationEngine")
    def test_validate_dag_compatibility_success(
        self, mock_validation_engine, mock_resolver, mock_registry, mock_path
    ):
        """Test successful DAG compatibility validation."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_validation_engine_instance = MagicMock()
        mock_validation_result = ValidationResult(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors={},
            dependency_issues=[],
            warnings=[],
        )
        mock_validation_engine_instance.validate_dag_compatibility.return_value = (
            mock_validation_result
        )
        mock_validation_engine.return_value = mock_validation_engine_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template method
        mock_template = MagicMock()
        mock_template.configs = {"data_loading": MagicMock(), "training": MagicMock()}
        mock_template._create_config_map.return_value = {
            "data_loading": MagicMock(),
            "training": MagicMock(),
        }
        mock_template._create_step_builder_map.return_value = {
            "DataLoading": MagicMock(),
            "Training": MagicMock(),
        }
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test validation
        result = compiler.validate_dag_compatibility(self.dag)

        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_validate_dag_compatibility_config_resolution_failure(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test validation with config resolution failure."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template to raise exception during config resolution
        mock_template = MagicMock()
        mock_template.configs = {}
        mock_template._create_config_map.side_effect = Exception(
            "Config resolution failed"
        )
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test validation
        result = compiler.validate_dag_compatibility(self.dag)

        # Verify result indicates failure
        assert not result.is_valid
        assert "Config resolution failed" in str(result.config_errors)

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_preview_resolution_success(self, mock_resolver, mock_registry, mock_path):
        """Test successful resolution preview."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_resolver_instance = MagicMock()
        mock_preview_data = {
            "data_loading": [
                {
                    "config_type": "CradleDataLoadConfig",
                    "confidence": 1.0,
                    "method": "direct_name",
                }
            ],
            "training": [
                {
                    "config_type": "XGBoostTrainingConfig",
                    "confidence": 0.8,
                    "method": "semantic",
                }
            ],
        }
        mock_resolver_instance.preview_resolution.return_value = mock_preview_data
        mock_resolver.return_value = mock_resolver_instance

        mock_registry_instance = MagicMock()
        mock_registry_instance._config_class_to_step_type.side_effect = (
            lambda x: x.replace("Config", "")
        )
        mock_registry_instance.get_builder_for_step_type.return_value = MagicMock(
            __name__="MockBuilder"
        )
        mock_registry.return_value = mock_registry_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template method
        mock_template = MagicMock()
        mock_template.configs = {"data_loading": MagicMock(), "training": MagicMock()}
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test preview
        result = compiler.preview_resolution(self.dag)

        # Verify result
        assert isinstance(result, ResolutionPreview)
        assert "data_loading" in result.node_config_map
        assert "training" in result.node_config_map
        assert result.node_config_map["data_loading"] == "CradleDataLoadConfig"
        assert result.resolution_confidence["data_loading"] == 1.0

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_preview_resolution_exception_handling(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test preview resolution exception handling."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template to raise exception
        compiler.create_template = MagicMock(
            side_effect=Exception("Template creation failed")
        )

        # Test preview
        result = compiler.preview_resolution(self.dag)

        # Verify result indicates failure
        assert isinstance(result, ResolutionPreview)
        assert result.node_config_map == {}
        assert "Preview failed" in result.recommendations[0]


class TestPipelineDAGCompilerCompilation:
    """Tests for PipelineDAGCompiler compilation methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.dag = PipelineDAG()
        self.dag.add_node("data_loading")
        self.dag.add_node("training")
        self.dag.add_edge("data_loading", "training")

        self.config_path = "test_config.json"

        yield

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_success(self, mock_resolver, mock_registry, mock_path):
        """Test successful compilation."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template and pipeline generation
        mock_template = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.name = "original-name"
        mock_template.generate_pipeline.return_value = mock_pipeline
        mock_template.base_config = MagicMock()
        mock_template.base_config.pipeline_name = "test-pipeline"
        mock_template.base_config.pipeline_version = "1.0"
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test compilation
        with patch(
            "cursus.core.compiler.name_generator.generate_pipeline_name"
        ) as mock_gen_name:
            mock_gen_name.return_value = "generated-pipeline-name"
            result = compiler.compile(self.dag)

        # Verify result
        assert result == mock_pipeline
        assert result.name == "generated-pipeline-name"
        assert compiler._last_template == mock_template

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_with_custom_pipeline_name(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test compilation with custom pipeline name."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template and pipeline generation
        mock_template = MagicMock()
        mock_pipeline = MagicMock()
        mock_template.generate_pipeline.return_value = mock_pipeline
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test compilation with custom name
        result = compiler.compile(self.dag, pipeline_name="custom-pipeline-name")

        # Verify custom name is used
        assert result.name == "custom-pipeline-name"

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_exception_handling(self, mock_resolver, mock_registry, mock_path):
        """Test compilation exception handling."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template to raise exception
        compiler.create_template = MagicMock(
            side_effect=Exception("Template creation failed")
        )

        # Test compilation
        with pytest.raises(PipelineAPIError) as context:
            compiler.compile(self.dag)

        assert "DAG compilation failed" in str(context.value)

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_with_report(self, mock_resolver, mock_registry, mock_path):
        """Test compilation with detailed report."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock compile and preview_resolution methods
        mock_pipeline = MagicMock()
        mock_pipeline.name = "test-pipeline"
        compiler.compile = MagicMock(return_value=mock_pipeline)

        mock_preview = ResolutionPreview(
            node_config_map={
                "data_loading": "CradleDataLoadConfig",
                "training": "XGBoostTrainingConfig",
            },
            config_builder_map={"CradleDataLoadConfig": "CradleDataLoadingStepBuilder"},
            resolution_confidence={"data_loading": 1.0, "training": 0.8},
            ambiguous_resolutions=[],
            recommendations=[],
        )
        compiler.preview_resolution = MagicMock(return_value=mock_preview)

        mock_registry_instance = MagicMock()
        mock_registry_instance.get_registry_stats.return_value = {"total_builders": 10}
        compiler.builder_registry = mock_registry_instance

        # Test compilation with report
        pipeline, report = compiler.compile_with_report(self.dag)

        # Verify result
        assert pipeline == mock_pipeline
        assert isinstance(report, ConversionReport)
        assert report.pipeline_name == "test-pipeline"
        assert len(report.steps) == 2
        assert "data_loading" in report.resolution_details
        assert "training" in report.resolution_details


class TestPipelineDAGCompilerUtilityMethods:
    """Tests for PipelineDAGCompiler utility methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.config_path = "test_config.json"

        yield

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_get_supported_step_types(self, mock_resolver, mock_registry, mock_path):
        """Test get_supported_step_types method."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_registry_instance = MagicMock()
        mock_registry_instance.list_supported_step_types.return_value = [
            "DataLoading",
            "Training",
            "Evaluation",
        ]
        mock_registry.return_value = mock_registry_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Test method
        result = compiler.get_supported_step_types()

        # Verify result
        assert result == ["DataLoading", "Training", "Evaluation"]

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_validate_config_file_success(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test successful config file validation."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template
        mock_template = MagicMock()
        mock_template.configs = {
            "config1": MagicMock(__class__=MagicMock(__name__="Config1")),
            "config2": MagicMock(__class__=MagicMock(__name__="Config2")),
        }
        compiler.create_template = MagicMock(return_value=mock_template)

        # Test validation
        result = compiler.validate_config_file()

        # Verify result
        assert result["valid"]
        assert result["config_count"] == 2
        assert result["config_names"] == ["config1", "config2"]

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_validate_config_file_failure(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test config file validation failure."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock create_template to raise exception
        compiler.create_template = MagicMock(
            side_effect=Exception("Config loading failed")
        )

        # Test validation
        result = compiler.validate_config_file()

        # Verify result
        assert not result["valid"]
        assert "Config loading failed" in result["error"]
        assert result["config_count"] == 0

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_get_last_template(self, mock_resolver, mock_registry, mock_path):
        """Test get_last_template method."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Initially should return None
        assert compiler.get_last_template() is None

        # Set a template
        mock_template = MagicMock()
        compiler._last_template = mock_template

        # Should return the template
        assert compiler.get_last_template() == mock_template

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepBuilderRegistry")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_and_fill_execution_doc(
        self, mock_resolver, mock_registry, mock_path
    ):
        """Test compile_and_fill_execution_doc method."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Mock compile method
        mock_pipeline = MagicMock()
        compiler.compile = MagicMock(return_value=mock_pipeline)

        # Mock template with fill_execution_document method
        mock_template = MagicMock()
        filled_doc = {"filled": True}
        mock_template.fill_execution_document.return_value = filled_doc
        compiler._last_template = mock_template

        # Test method
        dag = PipelineDAG()
        dag.add_node("test")
        execution_doc = {"template": True}

        pipeline, result_doc = compiler.compile_and_fill_execution_doc(
            dag, execution_doc
        )

        # Verify results
        assert pipeline == mock_pipeline
        assert result_doc == filled_doc
        mock_template.fill_execution_document.assert_called_once_with(execution_doc)
