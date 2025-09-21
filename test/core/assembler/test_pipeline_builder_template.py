import pytest
from unittest.mock import patch, MagicMock, ANY, call
from pathlib import Path
from collections import defaultdict
import json

# Note: Removed global module mocks to prevent interference with other tests
# These were causing mock interference in test/core/base/test_specification_base.py

from cursus.core.assembler.pipeline_template_base import PipelineTemplateBase
from cursus.core.assembler.pipeline_assembler import PipelineAssembler
from cursus.api.dag.base_dag import PipelineDAG


# Create mock classes for testing
class MockBasePipelineConfig:
    """Mock BasePipelineConfig for testing."""

    @staticmethod
    def get_step_name(config_class_name):
        return config_class_name.replace("Config", "Step")


class MockStepBuilderBase:
    """Mock StepBuilderBase for testing."""

    pass


class MockRegistryManager:
    """Mock RegistryManager for testing."""

    pass


class MockUnifiedDependencyResolver:
    """Mock UnifiedDependencyResolver for testing."""

    pass


class ConcretePipelineTemplate(PipelineTemplateBase):
    """Concrete implementation of PipelineTemplateBase for testing."""

    CONFIG_CLASSES = {
        "Base": MockBasePipelineConfig,
        "TestConfig1": MockBasePipelineConfig,
        "TestConfig2": MockBasePipelineConfig,
    }

    def _validate_configuration(self) -> None:
        """Simple validation for testing."""
        if "Base" not in self.configs:
            raise ValueError("Base configuration required")

    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create a simple test DAG."""
        dag = PipelineDAG()
        dag.add_node("step1")
        dag.add_node("step2")
        dag.add_node("step3")
        dag.add_edge("step1", "step2")
        dag.add_edge("step2", "step3")
        return dag

    def _create_config_map(self):
        """Create config map for testing."""
        return {
            "step1": self.configs.get(
                "TestConfig1", MagicMock(spec=MockBasePipelineConfig)
            ),
            "step2": self.configs.get(
                "TestConfig2", MagicMock(spec=MockBasePipelineConfig)
            ),
            "step3": self.configs.get("Base", MagicMock(spec=MockBasePipelineConfig)),
        }

    def _create_step_builder_map(self):
        """Create step builder map for testing."""
        return {
            "TestStep1": MagicMock(spec=type),
            "TestStep2": MagicMock(spec=type),
            "TestStep3": MagicMock(spec=type),
        }


class TestPipelineTemplateBase:

    @pytest.fixture
    def config_data(self):
        """Set up test fixtures."""
        return {
            "Base": {
                "pipeline_name": "test-pipeline",
                "pipeline_version": "1.0",
                "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            },
            "TestConfig1": {"some_param": "value1"},
            "TestConfig2": {"some_param": "value2"},
        }

    @pytest.fixture
    def mock_base_config(self):
        mock_base_config = MagicMock(spec=MockBasePipelineConfig)
        mock_base_config.pipeline_name = "test-pipeline"
        mock_base_config.pipeline_version = "1.0"
        mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        return mock_base_config

    @pytest.fixture
    def mock_configs(self, mock_base_config):
        return {
            "Base": mock_base_config,
            "TestConfig1": MagicMock(spec=MockBasePipelineConfig),
            "TestConfig2": MagicMock(spec=MockBasePipelineConfig),
        }

    @pytest.fixture
    def setup_mocks(self, config_data, mock_configs):
        """Set up all necessary mocks."""
        with patch("builtins.open", create=True) as mock_open, patch(
            "cursus.steps.configs.utils.load_configs"
        ) as mock_load_configs, patch(
            "cursus.steps.configs.utils.build_complete_config_classes"
        ) as mock_build_complete, patch(
            "cursus.core.assembler.pipeline_template_base.create_pipeline_components"
        ) as mock_create_components, patch(
            "cursus.core.assembler.pipeline_template_base.PipelineAssembler"
        ) as mock_assembler_cls, patch(
            "cursus.core.assembler.pipeline_template_base.generate_pipeline_name"
        ) as mock_generate_name:

            # Setup file operations
            mock_file = mock_open.return_value.__enter__.return_value
            mock_file.read.return_value = json.dumps(config_data)

            # Setup load_configs
            mock_load_configs.return_value = mock_configs

            # Setup build_complete_config_classes
            mock_build_complete.return_value = {}

            # Setup create_pipeline_components
            mock_registry_manager = MagicMock(spec=MockRegistryManager)
            mock_dependency_resolver = MagicMock(spec=MockUnifiedDependencyResolver)
            mock_create_components.return_value = {
                "registry_manager": mock_registry_manager,
                "resolver": mock_dependency_resolver,
            }

            # Setup PipelineAssembler
            mock_assembler = MagicMock()
            mock_pipeline = MagicMock()
            mock_assembler.generate_pipeline.return_value = mock_pipeline
            mock_assembler_cls.return_value = mock_assembler

            # Setup generate_pipeline_name
            mock_generate_name.return_value = "test-pipeline-v1-0"

            yield {
                "mock_open": mock_open,
                "mock_load_configs": mock_load_configs,
                "mock_build_complete": mock_build_complete,
                "mock_create_components": mock_create_components,
                "mock_assembler_cls": mock_assembler_cls,
                "mock_assembler": mock_assembler,
                "mock_pipeline": mock_pipeline,
                "mock_generate_name": mock_generate_name,
                "mock_registry_manager": mock_registry_manager,
                "mock_dependency_resolver": mock_dependency_resolver,
            }

    def test_initialization(self, setup_mocks, mock_configs, mock_base_config):
        """Test that the template initializes correctly."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(
            config_path="test_config.json",
            sagemaker_session=MagicMock(),
            role="test-role",
        )

        # Verify attributes were set correctly
        assert template.config_path == "test_config.json"
        assert template.configs == mock_configs
        assert template.base_config == mock_base_config
        assert template._registry_manager is not None
        assert template._dependency_resolver is not None

        # Verify load_configs was called with correct parameters
        mocks["mock_load_configs"].assert_called_once()

        # Verify components were created
        mocks["mock_create_components"].assert_called_once_with("test-pipeline")

        # Verify loaded_config_data was set
        assert template.loaded_config_data is not None

        # Verify pipeline_metadata was initialized
        assert template.pipeline_metadata == {}

    def test_initialization_with_provided_components(self, setup_mocks, mock_configs):
        """Test initialization with provided dependency components."""
        mocks = setup_mocks
        custom_registry = MagicMock(spec=MockRegistryManager)
        custom_resolver = MagicMock(spec=MockUnifiedDependencyResolver)

        template = ConcretePipelineTemplate(
            config_path="test_config.json",
            registry_manager=custom_registry,
            dependency_resolver=custom_resolver,
        )

        # Verify provided components were used
        assert template._registry_manager == custom_registry
        assert template._dependency_resolver == custom_resolver

        # Verify create_pipeline_components was not called
        mocks["mock_create_components"].assert_not_called()

    def test_config_loading(self, setup_mocks):
        """Test that configurations are loaded correctly."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(config_path="test_config.json")

        # Verify build_complete_config_classes was called
        mocks["mock_build_complete"].assert_called_once()

        # Verify load_configs was called with merged config classes
        call_args = mocks["mock_load_configs"].call_args
        assert call_args[0][0] == "test_config.json"
        # The second argument should be the merged config classes
        config_classes = call_args[0][1]
        assert "Base" in config_classes
        assert "TestConfig1" in config_classes
        assert "TestConfig2" in config_classes

    def test_base_config_validation(self, setup_mocks):
        """Test that missing base config raises error."""
        mocks = setup_mocks
        # Mock load_configs to return configs without Base
        mocks["mock_load_configs"].return_value = {"TestConfig1": MagicMock()}

        with pytest.raises(ValueError) as exc_info:
            ConcretePipelineTemplate(config_path="test_config.json")

        assert "Base configuration not found" in str(exc_info.value)

    def test_generate_pipeline(self, setup_mocks):
        """Test that generate_pipeline creates a complete pipeline."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(
            config_path="test_config.json",
            sagemaker_session=MagicMock(),
            role="test-role",
        )

        # Call generate_pipeline
        pipeline = template.generate_pipeline()

        # Verify PipelineAssembler was created with correct parameters
        mocks["mock_assembler_cls"].assert_called_once()
        call_kwargs = mocks["mock_assembler_cls"].call_args[1]

        # Verify DAG was created
        assert call_kwargs["dag"] is not None

        # Verify config_map was created
        assert call_kwargs["config_map"] is not None

        # Verify step_builder_map was created
        assert call_kwargs["step_builder_map"] is not None

        # Verify dependency components were passed
        assert call_kwargs["registry_manager"] == mocks["mock_registry_manager"]
        assert call_kwargs["dependency_resolver"] == mocks["mock_dependency_resolver"]

        # Verify generate_pipeline was called on assembler
        mocks["mock_assembler"].generate_pipeline.assert_called_once_with(
            "test-pipeline-v1-0"
        )

        # Verify pipeline was returned
        assert pipeline == mocks["mock_pipeline"]

    def test_pipeline_name_generation(self, setup_mocks):
        """Test pipeline name generation."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(config_path="test_config.json")

        # Test default name generation
        name = template._get_pipeline_name()
        mocks["mock_generate_name"].assert_called_with("test-pipeline", "1.0")
        assert name == "test-pipeline-v1-0"

    def test_pipeline_name_explicit_override(self, setup_mocks, mock_base_config):
        """Test explicit pipeline name override."""
        mocks = setup_mocks
        mock_base_config.explicit_pipeline_name = "custom-pipeline-name"

        template = ConcretePipelineTemplate(config_path="test_config.json")
        name = template._get_pipeline_name()

        # Should return explicit name without calling generator
        assert name == "custom-pipeline-name"
        mocks["mock_generate_name"].assert_not_called()

    def test_store_pipeline_metadata(self, setup_mocks):
        """Test that pipeline metadata is stored correctly (after cleanup)."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(config_path="test_config.json")

        # Mock assembler with metadata (no cradle_loading_requests after cleanup)
        mock_assembler = MagicMock()
        mock_assembler.step_instances = {"step1": MagicMock(), "step2": MagicMock()}

        # Call _store_pipeline_metadata
        template._store_pipeline_metadata(mock_assembler)

        # Verify only step_instances metadata was stored (cradle_loading_requests removed)
        assert template.pipeline_metadata["step_instances"] == mock_assembler.step_instances
        
        # Verify cradle_loading_requests is NOT stored (removed in Phase 2 cleanup)
        assert "cradle_loading_requests" not in template.pipeline_metadata

    def test_create_with_components_class_method(self, setup_mocks):
        """Test create_with_components class method."""
        mocks = setup_mocks

        # Call class method
        template = ConcretePipelineTemplate.create_with_components(
            config_path="test_config.json",
            context_name="custom-context",
            sagemaker_session=MagicMock(),
        )

        # Verify create_pipeline_components was called with context
        mocks["mock_create_components"].assert_called_with("custom-context")

        # Verify template was created with components
        assert isinstance(template, ConcretePipelineTemplate)

    def test_build_with_context_class_method(self, setup_mocks):
        """Test build_with_context class method."""
        mocks = setup_mocks

        # Mock dependency_resolution_context
        with patch(
            "cursus.core.assembler.pipeline_template_base.dependency_resolution_context"
        ) as mock_context:
            mock_context.return_value.__enter__.return_value = {
                "registry_manager": mocks["mock_registry_manager"],
                "resolver": mocks["mock_dependency_resolver"],
            }

            # Call class method
            pipeline = ConcretePipelineTemplate.build_with_context(
                config_path="test_config.json"
            )

            # Verify context manager was used
            mock_context.assert_called_once_with(clear_on_exit=True)

            # Verify pipeline was returned
            assert pipeline == mocks["mock_pipeline"]

    def test_build_in_thread_class_method(self, setup_mocks):
        """Test build_in_thread class method."""
        mocks = setup_mocks

        # Mock get_thread_components
        with patch(
            "cursus.core.assembler.pipeline_template_base.get_thread_components"
        ) as mock_thread:
            mock_thread.return_value = {
                "registry_manager": mocks["mock_registry_manager"],
                "resolver": mocks["mock_dependency_resolver"],
            }

            # Call class method
            pipeline = ConcretePipelineTemplate.build_in_thread(
                config_path="test_config.json"
            )

            # Verify thread components were used
            mock_thread.assert_called_once()

            # Verify pipeline was returned
            assert pipeline == mocks["mock_pipeline"]

    # Note: fill_execution_document method removed as part of execution document refactoring
    # The method was removed from PipelineTemplateBase to achieve clean separation between
    # pipeline generation and execution document generation.
    #
    # For execution document generation, use the standalone module:
    # from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
    # generator = ExecutionDocumentGenerator(config_path=config_path)
    # filled_doc = generator.fill_execution_document(dag, execution_doc)


    def test_config_loading_error_handling(self, setup_mocks):
        """Test error handling when config file loading fails."""
        mocks = setup_mocks

        # Mock file operations to raise an exception
        mocks["mock_open"].return_value.__enter__.return_value.read.side_effect = (
            Exception("File read error")
        )

        template = ConcretePipelineTemplate(config_path="test_config.json")

        # Verify loaded_config_data is None when file loading fails
        assert template.loaded_config_data is None

    def test_pipeline_name_fallback_values(self, setup_mocks, mock_base_config):
        """Test pipeline name generation with fallback values."""
        mocks = setup_mocks

        # Remove pipeline_name and pipeline_version from base config
        del mock_base_config.pipeline_name
        del mock_base_config.pipeline_version

        template = ConcretePipelineTemplate(config_path="test_config.json")
        name = template._get_pipeline_name()

        # Should use fallback values
        mocks["mock_generate_name"].assert_called_with("mods", "1.0")

    def test_get_pipeline_parameters_default(self, setup_mocks):
        """Test default pipeline parameters."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(config_path="test_config.json")
        params = template._get_pipeline_parameters()

        # Default implementation should return empty list
        assert params == []

    def test_store_pipeline_metadata_without_attributes(self, setup_mocks):
        """Test storing pipeline metadata when assembler doesn't have expected attributes."""
        mocks = setup_mocks

        template = ConcretePipelineTemplate(config_path="test_config.json")

        # Mock assembler without cradle_loading_requests or step_instances
        mock_assembler = MagicMock()
        # Remove the attributes
        if hasattr(mock_assembler, "cradle_loading_requests"):
            del mock_assembler.cradle_loading_requests
        if hasattr(mock_assembler, "step_instances"):
            del mock_assembler.step_instances

        # Call _store_pipeline_metadata
        template._store_pipeline_metadata(mock_assembler)

        # Verify metadata dict is still empty since attributes don't exist
        assert template.pipeline_metadata == {}

    def test_config_classes_validation(self, setup_mocks):
        """Test that CONFIG_CLASSES must be defined."""
        mocks = setup_mocks

        # Create a template class without CONFIG_CLASSES
        class InvalidTemplate(PipelineTemplateBase):
            def _validate_configuration(self):
                pass

            def _create_pipeline_dag(self):
                return MagicMock()

            def _create_config_map(self):
                return {}

            def _create_step_builder_map(self):
                return {}

        with pytest.raises(ValueError) as exc_info:
            InvalidTemplate(config_path="test_config.json")

        assert "CONFIG_CLASSES must be defined" in str(exc_info.value)

    def test_initialization_partial_components(self, setup_mocks):
        """Test initialization with only one component provided."""
        mocks = setup_mocks

        custom_registry = MagicMock(spec=MockRegistryManager)

        template = ConcretePipelineTemplate(
            config_path="test_config.json",
            registry_manager=custom_registry,
            # dependency_resolver not provided
        )

        # Verify provided component was used
        assert template._registry_manager == custom_registry

        # Verify missing component was created
        assert template._dependency_resolver is not None

        # Verify create_pipeline_components was called to get missing component
        mocks["mock_create_components"].assert_called_once_with("test-pipeline")


class TestPipelineAssembler:

    @pytest.fixture
    def mock_dag(self):
        """Set up test fixtures for PipelineAssembler tests."""
        mock_dag = MagicMock(spec=PipelineDAG)
        mock_dag.nodes = ["step1", "step2", "step3"]
        mock_dag.edges = [("step1", "step2"), ("step2", "step3")]
        mock_dag.topological_sort.return_value = ["step1", "step2", "step3"]
        mock_dag.get_dependencies.side_effect = lambda node: {
            "step1": [],
            "step2": ["step1"],
            "step3": ["step2"],
        }[node]
        return mock_dag

    @pytest.fixture
    def mock_configs(self):
        mock_config1 = MagicMock(spec=MockBasePipelineConfig)
        mock_config2 = MagicMock(spec=MockBasePipelineConfig)
        mock_config3 = MagicMock(spec=MockBasePipelineConfig)

        return {"step1": mock_config1, "step2": mock_config2, "step3": mock_config3}

    @pytest.fixture
    def mock_step_builder_map(self):
        mock_builder_cls = MagicMock()
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder

        return {
            "TestStep1": mock_builder_cls,
            "TestStep2": mock_builder_cls,
            "TestStep3": mock_builder_cls,
        }

    @pytest.fixture
    def mock_dependency_components(self):
        mock_registry_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry_manager.get_registry.return_value = mock_registry

        mock_dependency_resolver = MagicMock(spec=MockUnifiedDependencyResolver)

        return {
            "registry_manager": mock_registry_manager,
            "dependency_resolver": mock_dependency_resolver,
        }

    @pytest.fixture
    def setup_assembler_mocks(self):
        with patch(
            "cursus.core.assembler.pipeline_assembler.CONFIG_STEP_REGISTRY"
        ) as mock_step_registry, patch(
            "cursus.core.assembler.pipeline_assembler.create_dependency_resolver"
        ) as mock_create_resolver, patch(
            "cursus.core.assembler.pipeline_assembler.Pipeline"
        ) as mock_pipeline_cls:

            mock_step_registry.get.side_effect = lambda x: {
                "MockConfig": "TestStep1",
                "BasePipelineConfig": "TestStep2",
            }.get(x, "TestStep3")

            mock_dependency_resolver = MagicMock(spec=MockUnifiedDependencyResolver)
            mock_create_resolver.return_value = mock_dependency_resolver

            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline

            yield {
                "mock_step_registry": mock_step_registry,
                "mock_create_resolver": mock_create_resolver,
                "mock_pipeline_cls": mock_pipeline_cls,
                "mock_pipeline": mock_pipeline,
                "mock_dependency_resolver": mock_dependency_resolver,
            }

    def test_assembler_initialization(
        self,
        mock_dag,
        mock_configs,
        mock_step_builder_map,
        mock_dependency_components,
        setup_assembler_mocks,
    ):
        """Test PipelineAssembler initialization."""
        assembler = PipelineAssembler(
            dag=mock_dag,
            config_map=mock_configs,
            step_builder_map=mock_step_builder_map,
            registry_manager=mock_dependency_components["registry_manager"],
            dependency_resolver=mock_dependency_components["dependency_resolver"],
        )

        # Verify attributes
        assert assembler.dag == mock_dag
        assert assembler.config_map == mock_configs
        assert assembler.step_builder_map == mock_step_builder_map
        assert (
            assembler._registry_manager
            == mock_dependency_components["registry_manager"]
        )
        assert (
            assembler._dependency_resolver
            == mock_dependency_components["dependency_resolver"]
        )

        # Verify step builders were initialized
        assert len(assembler.step_builders) == 3

    def test_assembler_validation_missing_configs(
        self, mock_dag, mock_step_builder_map, setup_assembler_mocks
    ):
        """Test assembler validation with missing configs."""
        # Remove a config
        incomplete_config_map = {"step1": MagicMock()}

        with pytest.raises(ValueError) as exc_info:
            PipelineAssembler(
                dag=mock_dag,
                config_map=incomplete_config_map,
                step_builder_map=mock_step_builder_map,
            )

        assert "Missing configs for nodes" in str(exc_info.value)

    def test_assembler_validation_missing_builders(
        self, mock_dag, mock_configs, setup_assembler_mocks
    ):
        """Test assembler validation with missing step builders."""
        # Remove a step builder
        incomplete_builder_map = {"TestStep1": MagicMock()}

        with pytest.raises(ValueError) as exc_info:
            PipelineAssembler(
                dag=mock_dag,
                config_map=mock_configs,
                step_builder_map=incomplete_builder_map,
            )

        assert "Missing step builder for step type" in str(exc_info.value)

    def test_generate_pipeline(
        self,
        mock_dag,
        mock_configs,
        mock_step_builder_map,
        mock_dependency_components,
        setup_assembler_mocks,
    ):
        """Test pipeline generation."""
        mocks = setup_assembler_mocks

        assembler = PipelineAssembler(
            dag=mock_dag,
            config_map=mock_configs,
            step_builder_map=mock_step_builder_map,
            registry_manager=mock_dependency_components["registry_manager"],
            dependency_resolver=mock_dependency_components["dependency_resolver"],
        )

        # Mock step creation
        mock_step1 = MagicMock()
        mock_step2 = MagicMock()
        mock_step3 = MagicMock()

        # Get the mock builder from the step_builder_map
        mock_builder_cls = list(mock_step_builder_map.values())[0]
        mock_builder = mock_builder_cls.return_value
        mock_builder.create_step.side_effect = [mock_step1, mock_step2, mock_step3]

        # Generate pipeline
        pipeline = assembler.generate_pipeline("test-pipeline")

        # Verify pipeline was created
        mocks["mock_pipeline_cls"].assert_called_once_with(
            name="test-pipeline",
            parameters=[],
            steps=[mock_step1, mock_step2, mock_step3],
            sagemaker_session=None,
        )

        # Verify step instances were stored
        assert len(assembler.step_instances) == 3
        assert assembler.step_instances["step1"] == mock_step1
        assert assembler.step_instances["step2"] == mock_step2
        assert assembler.step_instances["step3"] == mock_step3

    def test_create_with_components_class_method(
        self,
        mock_dag,
        mock_configs,
        mock_step_builder_map,
        mock_dependency_components,
        setup_assembler_mocks,
    ):
        """Test create_with_components class method."""
        # Mock create_pipeline_components
        with patch(
            "cursus.core.assembler.pipeline_assembler.create_pipeline_components"
        ) as mock_create:
            mock_create.return_value = {
                "registry_manager": mock_dependency_components["registry_manager"],
                "resolver": mock_dependency_components["dependency_resolver"],
            }

            assembler = PipelineAssembler.create_with_components(
                dag=mock_dag,
                config_map=mock_configs,
                step_builder_map=mock_step_builder_map,
                context_name="test-context",
            )

            # Verify create_pipeline_components was called
            mock_create.assert_called_once_with("test-context")

            # Verify assembler was created with components
            assert (
                assembler._registry_manager
                == mock_dependency_components["registry_manager"]
            )
            assert (
                assembler._dependency_resolver
                == mock_dependency_components["dependency_resolver"]
            )
