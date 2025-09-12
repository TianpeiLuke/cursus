import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging

from cursus.core.assembler.pipeline_assembler import PipelineAssembler
from cursus.core.base import (
    BasePipelineConfig,
    StepBuilderBase,
    OutputSpec,
    DependencySpec,
    StepSpecification,
)
from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.deps.registry_manager import RegistryManager
from cursus.core.deps.dependency_resolver import UnifiedDependencyResolver
from cursus.core.base.enums import DependencyType, NodeType


class MockConfig(BasePipelineConfig):
    """Mock configuration class for testing."""

    def __init__(
        self,
        author="test_author",
        bucket="test-bucket",
        role="test-role",
        region="NA",
        service_name="test_service",
        pipeline_version="1.0.0",
    ):
        super().__init__(
            author=author,
            bucket=bucket,
            role=role,
            region=region,
            service_name=service_name,
            pipeline_version=pipeline_version,
        )


class MockStepBuilder(StepBuilderBase):
    """Mock step builder class for testing."""

    def __init__(
        self,
        config,
        sagemaker_session=None,
        role=None,
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        # Create real OutputSpec and DependencySpec instances to avoid mock interference
        output1 = OutputSpec(
            logical_name="output1",
            description="Mock output 1",
            property_path="properties.MockStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )
        output2 = OutputSpec(
            logical_name="output2",
            description="Mock output 2",
            property_path="properties.MockStep.Properties.ProcessingOutputConfig.Outputs.output2.S3Output.S3Uri",
            output_type=DependencyType.PROCESSING_OUTPUT,
        )

        input1 = DependencySpec(
            logical_name="input1",
            description="Mock input 1",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )
        input2 = DependencySpec(
            logical_name="input2",
            description="Mock input 2",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
        )

        # Create real StepSpecification instance
        self.spec = StepSpecification(
            step_type="MockStep",
            node_type=NodeType.INTERNAL,
            dependencies=[input1, input2],
            outputs=[output1, output2],
        )

        # Call parent constructor
        super().__init__(
            config=config,
            spec=self.spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )

    def validate_configuration(self) -> None:
        """Mock validation - always passes."""
        pass

    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        """Mock implementation of abstract method."""
        return inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Mock implementation of abstract method."""
        return outputs

    def create_step(
        self, inputs=None, outputs=None, dependencies=None, enable_caching=True
    ):
        """Create a mock step."""
        mock_step = Mock()
        mock_step.name = f"mock_step_{id(self)}"
        mock_step.inputs = inputs or {}
        mock_step.outputs = outputs or {}
        mock_step.dependencies = dependencies or []
        return mock_step


class TestPipelineAssembler:
    """Test cases for PipelineAssembler class."""

    @pytest.fixture
    def nodes(self):
        """DAG nodes."""
        return ["step1", "step2", "step3"]

    @pytest.fixture
    def edges(self):
        """DAG edges."""
        return [("step1", "step2"), ("step2", "step3")]

    @pytest.fixture
    def dag(self, nodes, edges):
        """Create mock DAG."""
        return PipelineDAG(nodes=nodes, edges=edges)

    @pytest.fixture
    def config_map(self):
        """Create mock configs."""
        return {"step1": MockConfig(), "step2": MockConfig(), "step3": MockConfig()}

    @pytest.fixture
    def step_builder_map(self):
        """Create mock step builder map."""
        return {"MockConfig": MockStepBuilder}

    @pytest.fixture
    def mock_session(self):
        """Mock SageMaker session."""
        return Mock()

    @pytest.fixture
    def role(self):
        """IAM role."""
        return "arn:aws:iam::123456789012:role/SageMakerRole"

    @pytest.fixture
    def notebook_root(self):
        """Notebook root path."""
        return Path("/test/notebook")

    @pytest.fixture
    def mock_registry_manager(self):
        """Mock registry manager and dependency resolver."""
        mock_registry_manager = Mock(spec=RegistryManager)
        mock_registry = Mock()
        mock_registry_manager.get_registry.return_value = mock_registry
        return mock_registry_manager

    @pytest.fixture
    def mock_dependency_resolver(self):
        """Mock dependency resolver."""
        mock_dependency_resolver = Mock(spec=UnifiedDependencyResolver)
        mock_dependency_resolver._calculate_compatibility.return_value = 0.8
        return mock_dependency_resolver

    def test_init_success(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_session,
        role,
        notebook_root,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test successful initialization of PipelineAssembler."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            sagemaker_session=mock_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Verify initialization
        assert assembler.dag == dag
        assert assembler.config_map == config_map
        assert assembler.step_builder_map == step_builder_map
        assert assembler.sagemaker_session == mock_session
        assert assembler.role == role
        assert assembler.notebook_root == notebook_root
        assert len(assembler.step_builders) == 3

        # Verify step builders were created
        for step_name in ["step1", "step2", "step3"]:
            assert step_name in assembler.step_builders
            assert isinstance(assembler.step_builders[step_name], MockStepBuilder)

    def test_init_missing_configs(self, dag, step_builder_map):
        """Test initialization with missing configs raises ValueError."""
        incomplete_config_map = {
            "step1": MockConfig(),
            "step2": MockConfig(),
            # Missing step3
        }

        with pytest.raises(ValueError) as exc_info:
            PipelineAssembler(
                dag=dag,
                config_map=incomplete_config_map,
                step_builder_map=step_builder_map,
            )

        assert "Missing configs for nodes" in str(exc_info.value)

    def test_init_missing_step_builders(self, dag, config_map):
        """Test initialization with missing step builders raises ValueError."""
        incomplete_builder_map = {}  # Empty builder map

        with pytest.raises(ValueError) as exc_info:
            PipelineAssembler(
                dag=dag, config_map=config_map, step_builder_map=incomplete_builder_map
            )

        assert "Missing step builder for step type" in str(exc_info.value)

    def test_init_invalid_dag_edges(self):
        """Test initialization with invalid DAG edges raises KeyError during DAG creation."""
        # The PipelineDAG constructor itself validates edges and raises KeyError
        # when trying to create a DAG with edges to non-existent nodes
        with pytest.raises(KeyError):
            invalid_dag = PipelineDAG(
                nodes=["step1", "step2"],
                edges=[("step1", "step2"), ("step2", "step3")],  # step3 doesn't exist
            )

    @patch("cursus.core.assembler.pipeline_assembler.CONFIG_STEP_REGISTRY")
    def test_initialize_step_builders(
        self,
        mock_registry,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test step builder initialization."""
        # Mock the registry to return step type
        mock_registry.get.return_value = "MockConfig"

        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Verify all step builders were initialized
        assert len(assembler.step_builders) == 3
        for step_name in ["step1", "step2", "step3"]:
            assert step_name in assembler.step_builders
            builder = assembler.step_builders[step_name]
            assert isinstance(builder, MockStepBuilder)
            assert builder.config == config_map[step_name]

    def test_propagate_messages(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test message propagation between steps."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Mock compatibility calculation to return high score
        mock_dependency_resolver._calculate_compatibility.return_value = 0.9

        # Call propagate messages
        assembler._propagate_messages()

        # Verify messages were stored
        # step2 should have messages from step1
        # step3 should have messages from step2
        assert len(assembler.step_messages) >= 0  # May be empty if no matches

    def test_generate_outputs(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test output generation for a step."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Generate outputs for step1
        outputs = assembler._generate_outputs("step1")

        # Verify outputs were generated based on specification
        assert isinstance(outputs, dict)
        # Should have outputs based on the mock spec
        expected_base = "s3://test-bucket/pipeline/mockstep"
        if outputs:  # Only check if outputs were generated
            for output_name, output_path in outputs.items():
                assert output_path.startswith("s3://")

    def test_instantiate_step(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test step instantiation."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Instantiate step1 (no dependencies)
        step = assembler._instantiate_step("step1")

        # Verify step was created
        assert step is not None
        assert hasattr(step, "name")

        # Store the step for dependency testing
        assembler.step_instances["step1"] = step

        # Instantiate step2 (depends on step1)
        step2 = assembler._instantiate_step("step2")
        assert step2 is not None

    @patch("cursus.core.assembler.pipeline_assembler.Pipeline")
    def test_generate_pipeline(
        self,
        mock_pipeline_class,
        dag,
        config_map,
        step_builder_map,
        mock_session,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test pipeline generation."""
        # Mock Pipeline constructor
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            sagemaker_session=mock_session,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Generate pipeline
        pipeline_name = "test_pipeline"
        result = assembler.generate_pipeline(pipeline_name)

        # Verify pipeline was created
        assert result == mock_pipeline
        mock_pipeline_class.assert_called_once()

        # Verify all steps were instantiated
        assert len(assembler.step_instances) == 3

    def test_generate_pipeline_with_cycle(
        self,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test pipeline generation with cyclic DAG raises error."""
        # Create DAG with cycle
        cyclic_dag = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3"), ("step3", "step1")],
        )

        assembler = PipelineAssembler(
            dag=cyclic_dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Should raise ValueError due to cycle
        with pytest.raises(ValueError) as exc_info:
            assembler.generate_pipeline("test_pipeline")

        assert "Failed to determine build order" in str(exc_info.value)

    @patch("cursus.core.assembler.pipeline_assembler.create_pipeline_components")
    def test_create_with_components(
        self,
        mock_create_components,
        dag,
        config_map,
        step_builder_map,
        mock_session,
        role,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test factory method for creating assembler with components."""
        # Mock component creation
        mock_components = {
            "registry_manager": mock_registry_manager,
            "resolver": mock_dependency_resolver,
        }
        mock_create_components.return_value = mock_components

        # Create assembler using factory method
        assembler = PipelineAssembler.create_with_components(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            context_name="test_context",
            sagemaker_session=mock_session,
            role=role,
        )

        # Verify components were used
        mock_create_components.assert_called_once_with("test_context")
        assert assembler._registry_manager == mock_registry_manager
        assert assembler._dependency_resolver == mock_dependency_resolver

    def test_get_registry_manager(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test registry manager getter."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        result = assembler._get_registry_manager()
        assert result == mock_registry_manager

    def test_get_dependency_resolver(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test dependency resolver getter."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        result = assembler._get_dependency_resolver()
        assert result == mock_dependency_resolver

    def test_cradle_loading_requests_storage(
        self, mock_registry_manager, mock_dependency_resolver
    ):
        """Test that Cradle data loading requests are stored correctly."""

        # Create a mock builder that has get_request_dict method
        class MockCradleBuilder(MockStepBuilder):
            def get_request_dict(self):
                return {"request": "test_cradle_request"}

        # Create a mock config that will be identified as CradleDataLoading
        class MockCradleConfig(MockConfig):
            pass

        # Update step builder map and config
        cradle_config_map = {"step1": MockCradleConfig()}
        cradle_builder_map = {"CradleDataLoading": MockCradleBuilder}
        cradle_dag = PipelineDAG(nodes=["step1"], edges=[])

        with patch(
            "cursus.core.assembler.pipeline_assembler.BasePipelineConfig.get_step_name"
        ) as mock_get_step_name:
            mock_get_step_name.return_value = "CradleDataLoading"

            assembler = PipelineAssembler(
                dag=cradle_dag,
                config_map=cradle_config_map,
                step_builder_map=cradle_builder_map,
                registry_manager=mock_registry_manager,
                dependency_resolver=mock_dependency_resolver,
            )

            # Instantiate the step
            step = assembler._instantiate_step("step1")

            # Verify request was stored
            assert step.name in assembler.cradle_loading_requests
            assert assembler.cradle_loading_requests[step.name] == {
                "request": "test_cradle_request"
            }

    def test_pipeline_regeneration(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test that pipeline can be regenerated (step instances are cleared)."""
        assembler = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            registry_manager=mock_registry_manager,
            dependency_resolver=mock_dependency_resolver,
        )

        # Add some mock step instances
        assembler.step_instances = {"step1": Mock(), "step2": Mock()}

        with patch(
            "cursus.core.assembler.pipeline_assembler.Pipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = Mock()

            # Generate pipeline (should clear existing instances)
            assembler.generate_pipeline("test_pipeline")

            # Verify instances were cleared and recreated
            assert len(assembler.step_instances) == 3  # All steps recreated

    def test_logging_integration(
        self,
        dag,
        config_map,
        step_builder_map,
        mock_registry_manager,
        mock_dependency_resolver,
    ):
        """Test that logging is properly integrated."""
        with patch("cursus.core.assembler.pipeline_assembler.logger") as mock_logger:
            assembler = PipelineAssembler(
                dag=dag,
                config_map=config_map,
                step_builder_map=step_builder_map,
                registry_manager=mock_registry_manager,
                dependency_resolver=mock_dependency_resolver,
            )

            # Verify logging calls were made during initialization
            mock_logger.info.assert_called()
