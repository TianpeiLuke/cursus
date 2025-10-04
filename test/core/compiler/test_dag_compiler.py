"""
Unit tests for the dag_compiler module.

This module tests the pipeline compilation process, particularly focusing on the 
conversion of PipelineDAG structures to SageMaker pipelines.

IMPROVED: Following pytest best practices and troubleshooting guide:
1. Implementation-driven test design (read source code first)
2. Precise mock path configuration based on actual imports
3. Comprehensive edge case coverage
4. Proper fixture isolation and cleanup
5. Systematic error prevention patterns
"""

import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Dict, Any

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
from sagemaker.workflow.parameters import ParameterString


class TestDagCompiler:
    """
    Tests for the dag_compiler module.
    
    IMPROVED: Following pytest best practices:
    1. Proper fixture isolation (no shared state)
    2. Realistic test data structures
    3. Implementation-driven test design
    """

    @pytest.fixture
    def sample_dag(self):
        """Create a sample DAG for testing - fresh instance per test."""
        dag = PipelineDAG()
        dag.add_node("data_loading")
        dag.add_node("preprocessing") 
        dag.add_node("training")
        dag.add_edge("data_loading", "preprocessing")
        dag.add_edge("preprocessing", "training")
        return dag

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file with realistic structure."""
        config_content = {
            "Base": {
                "config_type": "BasePipelineConfig",
                "author": "test_author",
                "bucket": "test-bucket",
                "role": "test-role",
                "region": "NA",
                "service_name": "test_service",
                "pipeline_version": "1.0.0",
                "project_root_folder": "cursus",
                "pipeline_name": "test_pipeline"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_content, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_session(self):
        """Mock SageMaker session with proper spec."""
        return Mock(spec=['default_bucket', 'region_name'])

    @pytest.fixture
    def mock_role(self):
        """Mock IAM role."""
        return "arn:aws:iam::123456789012:role/SageMakerRole"

    def test_compile_with_custom_pipeline_name(self, sample_dag, temp_config_file, mock_session, mock_role):
        """
        Test that custom pipeline names are used directly.
        
        IMPROVED: 
        - Read source code first: compile() method sets pipeline.name = pipeline_name when provided
        - Use proper mock paths based on actual imports in dag_compiler.py
        - Use realistic fixtures instead of hardcoded paths
        """
        # IMPROVED: Mock at correct import paths based on source analysis
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.StepCatalog") as mock_catalog_class, \
             patch("cursus.core.compiler.dag_compiler.StepConfigResolver") as mock_resolver_class, \
             patch("cursus.core.compiler.dag_compiler.ValidationEngine") as mock_validation_class:
            
            # Setup Path mock
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Setup component mocks
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver
            mock_validation = Mock()
            mock_validation_class.return_value = mock_validation

            # Create compiler
            compiler = PipelineDAGCompiler(
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
                step_catalog=mock_catalog
            )

            # IMPROVED: Mock create_template method with realistic behavior
            mock_template = Mock()
            mock_pipeline = Mock()
            mock_pipeline.name = "original-name"  # Initial name
            mock_template.generate_pipeline.return_value = mock_pipeline
            mock_template.base_config = Mock()
            mock_template.base_config.pipeline_name = "test_pipeline"
            mock_template.base_config.pipeline_version = "1.0.0"
            
            compiler.create_template = Mock(return_value=mock_template)

            # Test with custom pipeline name
            result = compiler.compile(sample_dag, pipeline_name="custom-pipeline-name")

            # IMPROVED: Verify actual behavior from source code
            assert result.name == "custom-pipeline-name"
            assert compiler._last_template == mock_template
            
            # Verify template creation was called with correct parameters
            compiler.create_template.assert_called_once_with(
                sample_dag, skip_validation=True
            )


class TestCompileDagToPipeline:
    """
    Tests for the compile_dag_to_pipeline function.
    
    IMPROVED: Following pytest best practices:
    1. Use proper fixtures instead of setup_method
    2. Avoid shared state between tests
    3. Test actual implementation behavior
    """

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG for testing."""
        dag = PipelineDAG()
        dag.add_node("data_loading")
        dag.add_node("training")
        dag.add_edge("data_loading", "training")
        return dag

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        config_content = {
            "Base": {
                "config_type": "BasePipelineConfig",
                "author": "test_author",
                "bucket": "test-bucket",
                "role": "test-role",
                "region": "NA",
                "service_name": "test_service",
                "pipeline_version": "1.0.0",
                "project_root_folder": "cursus",
                "pipeline_name": "test_pipeline"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_content, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_session(self):
        """Mock SageMaker session."""
        return Mock(spec=['default_bucket', 'region_name'])

    @pytest.fixture
    def mock_role(self):
        """Mock IAM role."""
        return "arn:aws:iam::123456789012:role/SageMakerRole"

    def test_compile_dag_to_pipeline_invalid_dag(self, temp_config_file, mock_session, mock_role):
        """
        Test compile_dag_to_pipeline with invalid DAG.
        
        IMPROVED: 
        - Read source code first: compile_dag_to_pipeline checks isinstance(dag, PipelineDAG)
        - Use fixtures instead of self attributes
        - Test exact error message from implementation
        """
        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag="not_a_dag",
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
            )
        # IMPROVED: Test exact error message from source code
        assert "dag must be a PipelineDAG instance" in str(context.value)

    def test_compile_dag_to_pipeline_empty_dag(self, temp_config_file, mock_session, mock_role):
        """
        Test compile_dag_to_pipeline with empty DAG.
        
        IMPROVED:
        - Read source code first: compile_dag_to_pipeline checks if dag.nodes is empty
        - Use fixtures for consistent test data
        """
        empty_dag = PipelineDAG()

        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag=empty_dag,
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
            )
        # IMPROVED: Test exact error message from source code
        assert "DAG must contain at least one node" in str(context.value)

    def test_compile_dag_to_pipeline_missing_config_file(self, simple_dag, mock_session, mock_role):
        """
        Test compile_dag_to_pipeline with missing config file.
        
        IMPROVED:
        - Read source code first: compile_dag_to_pipeline checks Path(config_path).exists()
        - Use fixtures for consistent test data
        """
        with pytest.raises(PipelineAPIError) as context:
            compile_dag_to_pipeline(
                dag=simple_dag,
                config_path="nonexistent_config.json",
                sagemaker_session=mock_session,
                role=mock_role,
            )
        # IMPROVED: Test exact error message from source code
        assert "Configuration file not found" in str(context.value)

    def test_compile_dag_to_pipeline_success(self, simple_dag, temp_config_file, mock_session, mock_role):
        """
        Test successful compile_dag_to_pipeline execution.
        
        IMPROVED:
        - Read source code first: compile_dag_to_pipeline creates PipelineDAGCompiler and calls compile()
        - Use proper mock paths based on actual imports
        - Use fixtures for consistent test data
        - Test actual parameter passing behavior
        """
        # IMPROVED: Mock at correct import paths based on source analysis
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.PipelineDAGCompiler") as mock_compiler_class:
            
            # Setup Path mock
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Setup compiler mock
            mock_compiler = Mock()
            mock_pipeline = Mock()
            mock_pipeline.name = "test-pipeline"
            mock_compiler.compile.return_value = mock_pipeline
            mock_compiler_class.return_value = mock_compiler

            # Call function
            result = compile_dag_to_pipeline(
                dag=simple_dag,
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
                pipeline_name="custom-pipeline",
            )

            # IMPROVED: Verify actual behavior from source code
            # Source shows: compiler = PipelineDAGCompiler(config_path=config_path, sagemaker_session=sagemaker_session, role=role, **kwargs)
            mock_compiler_class.assert_called_once_with(
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
            )

            # Source shows: pipeline = compiler.compile(dag, pipeline_name=pipeline_name)
            mock_compiler.compile.assert_called_once_with(
                simple_dag, pipeline_name="custom-pipeline"
            )

            # Verify result
            assert result == mock_pipeline

    def test_compile_dag_to_pipeline_exception_handling(self, simple_dag, temp_config_file, mock_session, mock_role):
        """
        Test exception handling in compile_dag_to_pipeline.
        
        IMPROVED:
        - Read source code first: compile_dag_to_pipeline wraps exceptions in PipelineAPIError
        - Use fixtures for consistent test data
        - Test exact exception handling behavior
        """
        # IMPROVED: Mock at correct import paths
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.PipelineDAGCompiler") as mock_compiler_class:
            
            # Setup Path mock
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Setup compiler mock to raise exception
            mock_compiler = Mock()
            mock_compiler.compile.side_effect = Exception("Compilation failed")
            mock_compiler_class.return_value = mock_compiler

            # Call function and expect PipelineAPIError
            with pytest.raises(PipelineAPIError) as context:
                compile_dag_to_pipeline(
                    dag=simple_dag,
                    config_path=temp_config_file,
                    sagemaker_session=mock_session,
                    role=mock_role,
                )

            # IMPROVED: Test exact error message from source code
            assert "DAG compilation failed" in str(context.value)
            # Verify original exception is preserved
            assert "Compilation failed" in str(context.value)


class TestPipelineDAGCompilerInit:
    """
    Tests for PipelineDAGCompiler initialization.
    
    IMPROVED: Following pytest best practices:
    1. Use proper fixtures instead of setup_method
    2. Avoid shared state between tests
    3. Test actual implementation behavior from source code
    """

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        config_content = {
            "Base": {
                "config_type": "BasePipelineConfig",
                "author": "test_author",
                "bucket": "test-bucket",
                "role": "test-role",
                "region": "NA",
                "service_name": "test_service",
                "pipeline_version": "1.0.0",
                "project_root_folder": "cursus",
                "pipeline_name": "test_pipeline"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_content, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_session(self):
        """Mock SageMaker session."""
        return Mock(spec=['default_bucket', 'region_name'])

    @pytest.fixture
    def mock_role(self):
        """Mock IAM role."""
        return "arn:aws:iam::123456789012:role/SageMakerRole"

    def test_compiler_init_success(self, temp_config_file, mock_session, mock_role):
        """
        Test successful PipelineDAGCompiler initialization.
        
        IMPROVED:
        - Read source code first: __init__ method initializes components and validates config file
        - Use proper mock paths based on actual imports
        - Use fixtures for consistent test data
        - Test actual initialization behavior
        """
        # IMPROVED: Mock at correct import paths based on source analysis
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.StepCatalog") as mock_catalog_class, \
             patch("cursus.core.compiler.dag_compiler.StepConfigResolver") as mock_resolver_class, \
             patch("cursus.core.compiler.dag_compiler.ValidationEngine") as mock_validation_class:
            
            # Setup Path mock
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Setup component mocks
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            mock_resolver = Mock()
            mock_resolver_class.return_value = mock_resolver
            mock_validation = Mock()
            mock_validation_class.return_value = mock_validation

            # Create compiler
            compiler = PipelineDAGCompiler(
                config_path=temp_config_file,
                sagemaker_session=mock_session,
                role=mock_role,
            )

            # IMPROVED: Verify actual initialization from source code
            assert compiler.config_path == temp_config_file
            assert compiler.sagemaker_session == mock_session
            assert compiler.role == mock_role
            assert compiler.config_resolver is not None
            assert compiler.step_catalog is not None
            assert compiler.validation_engine is not None
            assert compiler._last_template is None
            # Source shows: pipeline_parameters initialized with defaults if None provided
            assert compiler.pipeline_parameters is not None
            assert len(compiler.pipeline_parameters) > 0

    def test_compiler_init_missing_config_file(self, mock_session, mock_role):
        """
        Test PipelineDAGCompiler initialization with missing config file.
        
        IMPROVED:
        - Read source code first: __init__ checks Path(config_path).exists() and raises FileNotFoundError
        - Use fixtures for consistent test data
        - Test exact error message from implementation
        """
        with pytest.raises(FileNotFoundError) as context:
            PipelineDAGCompiler(
                config_path="nonexistent_config.json",
                sagemaker_session=mock_session,
                role=mock_role,
            )
        # IMPROVED: Test exact error message from source code
        assert "Configuration file not found" in str(context.value)

    def test_compiler_init_with_custom_components(self, temp_config_file):
        """
        Test PipelineDAGCompiler initialization with custom components.
        
        IMPROVED:
        - Read source code first: __init__ uses provided components instead of creating new ones
        - Use fixtures for consistent test data
        - Test actual component assignment behavior
        """
        # IMPROVED: Mock Path to avoid file system dependency
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            # Create custom components with proper specs
            custom_resolver = Mock(spec=['resolve_config_map', 'preview_resolution'])
            custom_catalog = Mock(spec=['list_supported_step_types', 'get_builder_map'])

            # Create compiler with custom components
            compiler = PipelineDAGCompiler(
                config_path=temp_config_file,
                config_resolver=custom_resolver,
                step_catalog=custom_catalog,
            )

            # IMPROVED: Verify actual behavior from source code
            assert compiler.config_resolver == custom_resolver
            assert compiler.step_catalog == custom_catalog

    def test_compiler_init_with_pipeline_parameters(self, temp_config_file):
        """
        Test PipelineDAGCompiler initialization with custom pipeline parameters.
        
        IMPROVED: Test new functionality for pipeline parameter handling
        """
        custom_params = [
            ParameterString(name="CUSTOM_PARAM", default_value="custom_value")
        ]
        
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            compiler = PipelineDAGCompiler(
                config_path=temp_config_file,
                pipeline_parameters=custom_params,
            )
            
            # IMPROVED: Verify actual parameter storage from source code
            assert compiler.pipeline_parameters == custom_params

    def test_compiler_init_default_parameters(self, temp_config_file):
        """
        Test PipelineDAGCompiler initialization with default parameters.
        
        IMPROVED: Test default parameter behavior from source code
        """
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            compiler = PipelineDAGCompiler(config_path=temp_config_file)
            
            # IMPROVED: Verify default parameters are set from source code
            assert compiler.pipeline_parameters is not None
            assert len(compiler.pipeline_parameters) > 0
            
            # Should include standard parameters from source
            param_names = [p.name for p in compiler.pipeline_parameters if hasattr(p, 'name')]
            assert "EXECUTION_S3_PREFIX" in param_names


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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    @patch("cursus.core.compiler.dag_compiler.ValidationEngine")
    def test_validate_dag_compatibility_success(
        self, mock_validation_engine, mock_resolver, mock_catalog, mock_path
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_preview_resolution_success(self, mock_resolver, mock_catalog, mock_path):
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

        mock_catalog_instance = MagicMock()
        mock_catalog_instance._config_class_to_step_type.side_effect = (
            lambda x: x.replace("Config", "")
        )
        mock_catalog_instance.get_builder_for_step_type.return_value = MagicMock(
            __name__="MockBuilder"
        )
        mock_catalog.return_value = mock_catalog_instance

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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_preview_resolution_exception_handling(
        self, mock_resolver, mock_catalog, mock_path
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_success(self, mock_resolver, mock_catalog, mock_path):
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_with_custom_pipeline_name(
        self, mock_resolver, mock_catalog, mock_path
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_exception_handling(self, mock_resolver, mock_catalog, mock_path):
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_compile_with_report(self, mock_resolver, mock_catalog, mock_path):
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

        mock_catalog_instance = MagicMock()
        mock_catalog_instance.get_catalog_stats.return_value = {"total_builders": 10}
        compiler.step_catalog = mock_catalog_instance

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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_get_supported_step_types(self, mock_resolver, mock_catalog, mock_path):
        """Test get_supported_step_types method."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_catalog_instance = MagicMock()
        mock_catalog_instance.list_supported_step_types.return_value = [
            "DataLoading",
            "Training",
            "Evaluation",
        ]
        mock_catalog.return_value = mock_catalog_instance

        # Create compiler
        compiler = PipelineDAGCompiler(config_path=self.config_path)

        # Test method
        result = compiler.get_supported_step_types()

        # Verify result - should call the catalog's method
        mock_catalog_instance.list_supported_step_types.assert_called_once()
        assert result == ["DataLoading", "Training", "Evaluation"]

    @patch("cursus.core.compiler.dag_compiler.Path")
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_validate_config_file_success(
        self, mock_resolver, mock_catalog, mock_path
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_validate_config_file_failure(
        self, mock_resolver, mock_catalog, mock_path
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
    @patch("cursus.core.compiler.dag_compiler.StepCatalog")
    @patch("cursus.core.compiler.dag_compiler.StepConfigResolver")
    def test_get_last_template(self, mock_resolver, mock_catalog, mock_path):
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

    # Note: compile_and_fill_execution_doc method removed as part of execution document refactoring
    # The method was removed to achieve clean separation between pipeline compilation and 
    # execution document generation. Users should now:
    # 1. Use compiler.compile(dag) to generate pipeline
    # 2. Use standalone execution document generator for execution documents
    #
    # For execution document generation, use:
    # from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
    # generator = ExecutionDocumentGenerator(config_path=config_path)
    # filled_doc = generator.fill_execution_document(dag, execution_doc)

    def test_compiler_init_with_pipeline_parameters(self):
        """Test PipelineDAGCompiler initialization with pipeline parameters."""
        from sagemaker.workflow.parameters import ParameterString
        
        custom_params = [
            ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://custom-bucket/execution"),
            ParameterString(name="KMS_ENCRYPTION_KEY_PARAM", default_value="custom-key"),
        ]
        
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            compiler = PipelineDAGCompiler(
                config_path=self.config_path,
                pipeline_parameters=custom_params,
            )
            
            assert compiler.pipeline_parameters == custom_params

    def test_compiler_init_with_default_parameters(self):
        """Test PipelineDAGCompiler initialization with default parameters when none provided."""
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            compiler = PipelineDAGCompiler(config_path=self.config_path)
            
            # Should have default parameters
            assert compiler.pipeline_parameters is not None
            assert len(compiler.pipeline_parameters) > 0
            
            # Should include standard parameters
            param_names = [p.name for p in compiler.pipeline_parameters if hasattr(p, 'name')]
            assert "EXECUTION_S3_PREFIX" in param_names

    def test_create_template_passes_parameters(self):
        """Test that create_template passes pipeline parameters to DynamicPipelineTemplate."""
        from sagemaker.workflow.parameters import ParameterString
        
        # Create test DAG
        dag = PipelineDAG()
        dag.add_node("test_node")
        
        custom_params = [
            ParameterString(name="EXECUTION_S3_PREFIX", default_value="s3://custom-bucket/execution")
        ]
        
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dynamic_template.DynamicPipelineTemplate") as mock_template_class:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            mock_template = MagicMock()
            mock_template_class.return_value = mock_template
            
            compiler = PipelineDAGCompiler(
                config_path=self.config_path,
                pipeline_parameters=custom_params,
            )
            
            # Create template
            result = compiler.create_template(dag)
            
            # Verify template was created with parameters
            mock_template_class.assert_called_once()
            call_kwargs = mock_template_class.call_args[1]
            assert 'pipeline_parameters' in call_kwargs
            assert call_kwargs['pipeline_parameters'] == custom_params

    def test_parameter_fallback_import_handling(self):
        """Test that parameter fallback works when mods_workflow_core is not available."""
        with patch("cursus.core.compiler.dag_compiler.Path") as mock_path, \
             patch("cursus.core.compiler.dag_compiler.logger") as mock_logger:
            
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            # This should work even if imports fail (fallback parameters are defined)
            compiler = PipelineDAGCompiler(config_path=self.config_path)
            
            # Should have fallback parameters
            assert compiler.pipeline_parameters is not None
            assert len(compiler.pipeline_parameters) > 0
