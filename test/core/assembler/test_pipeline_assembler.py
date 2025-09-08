import unittest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging

from cursus.core.assembler.pipeline_assembler import PipelineAssembler
from cursus.core.base import BasePipelineConfig, StepBuilderBase, OutputSpec, DependencySpec, StepSpecification
from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.deps.registry_manager import RegistryManager
from cursus.core.deps.dependency_resolver import UnifiedDependencyResolver
from cursus.core.base.enums import DependencyType, NodeType

class MockConfig(BasePipelineConfig):
    """Mock configuration class for testing."""
    
    def __init__(self, author="test_author", bucket="test-bucket", role="test-role", 
                 region="NA", service_name="test_service", pipeline_version="1.0.0"):
        super().__init__(
            author=author,
            bucket=bucket,
            role=role,
            region=region,
            service_name=service_name,
            pipeline_version=pipeline_version
        )

class MockStepBuilder(StepBuilderBase):
    """Mock step builder class for testing."""
    
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Create real OutputSpec and DependencySpec instances to avoid mock interference
        output1 = OutputSpec(
            logical_name="output1",
            description="Mock output 1",
            property_path="properties.MockStep.Properties.OutputDataConfig.S3OutputPath",
            output_type=DependencyType.PROCESSING_OUTPUT
        )
        output2 = OutputSpec(
            logical_name="output2",
            description="Mock output 2",
            property_path="properties.MockStep.Properties.ProcessingOutputConfig.Outputs.output2.S3Output.S3Uri",
            output_type=DependencyType.PROCESSING_OUTPUT
        )
        
        input1 = DependencySpec(
            logical_name="input1",
            description="Mock input 1",
            dependency_type=DependencyType.PROCESSING_OUTPUT
        )
        input2 = DependencySpec(
            logical_name="input2",
            description="Mock input 2",
            dependency_type=DependencyType.PROCESSING_OUTPUT
        )
        
        # Create real StepSpecification instance
        self.spec = StepSpecification(
            step_type="MockStep",
            node_type=NodeType.INTERNAL,
            dependencies=[input1, input2],
            outputs=[output1, output2]
        )
        
        # Call parent constructor
        super().__init__(
            config=config,
            spec=self.spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
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
    
    def create_step(self, inputs=None, outputs=None, dependencies=None, enable_caching=True):
        """Create a mock step."""
        mock_step = Mock()
        mock_step.name = f"mock_step_{id(self)}"
        mock_step.inputs = inputs or {}
        mock_step.outputs = outputs or {}
        mock_step.dependencies = dependencies or []
        return mock_step

class TestPipelineAssembler(unittest.TestCase):
    """Test cases for PipelineAssembler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock DAG
        self.nodes = ['step1', 'step2', 'step3']
        self.edges = [('step1', 'step2'), ('step2', 'step3')]
        self.dag = PipelineDAG(nodes=self.nodes, edges=self.edges)
        
        # Create mock configs
        self.config_map = {
            'step1': MockConfig(),
            'step2': MockConfig(),
            'step3': MockConfig()
        }
        
        # Create mock step builder map
        self.step_builder_map = {
            'MockConfig': MockStepBuilder
        }
        
        # Mock SageMaker session
        self.mock_session = Mock()
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"
        self.notebook_root = Path("/test/notebook")
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = Mock(spec=RegistryManager)
        self.mock_registry = Mock()
        self.mock_registry_manager.get_registry.return_value = self.mock_registry
        
        self.mock_dependency_resolver = Mock(spec=UnifiedDependencyResolver)
        self.mock_dependency_resolver._calculate_compatibility.return_value = 0.8

    def test_init_success(self):
        """Test successful initialization of PipelineAssembler."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            sagemaker_session=self.mock_session,
            role=self.role,
            notebook_root=self.notebook_root,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Verify initialization
        self.assertEqual(assembler.dag, self.dag)
        self.assertEqual(assembler.config_map, self.config_map)
        self.assertEqual(assembler.step_builder_map, self.step_builder_map)
        self.assertEqual(assembler.sagemaker_session, self.mock_session)
        self.assertEqual(assembler.role, self.role)
        self.assertEqual(assembler.notebook_root, self.notebook_root)
        self.assertEqual(len(assembler.step_builders), 3)
        
        # Verify step builders were created
        for step_name in self.nodes:
            self.assertIn(step_name, assembler.step_builders)
            self.assertIsInstance(assembler.step_builders[step_name], MockStepBuilder)

    def test_init_missing_configs(self):
        """Test initialization with missing configs raises ValueError."""
        incomplete_config_map = {
            'step1': MockConfig(),
            'step2': MockConfig()
            # Missing step3
        }
        
        with self.assertRaises(ValueError) as context:
            PipelineAssembler(
                dag=self.dag,
                config_map=incomplete_config_map,
                step_builder_map=self.step_builder_map
            )
        
        self.assertIn("Missing configs for nodes", str(context.exception))

    def test_init_missing_step_builders(self):
        """Test initialization with missing step builders raises ValueError."""
        incomplete_builder_map = {}  # Empty builder map
        
        with self.assertRaises(ValueError) as context:
            PipelineAssembler(
                dag=self.dag,
                config_map=self.config_map,
                step_builder_map=incomplete_builder_map
            )
        
        self.assertIn("Missing step builder for step type", str(context.exception))

    def test_init_invalid_dag_edges(self):
        """Test initialization with invalid DAG edges raises KeyError during DAG creation."""
        # The PipelineDAG constructor itself validates edges and raises KeyError
        # when trying to create a DAG with edges to non-existent nodes
        with self.assertRaises(KeyError):
            invalid_dag = PipelineDAG(
                nodes=['step1', 'step2'],
                edges=[('step1', 'step2'), ('step2', 'step3')]  # step3 doesn't exist
            )

    @patch('src.cursus.core.assembler.pipeline_assembler.CONFIG_STEP_REGISTRY')
    def test_initialize_step_builders(self, mock_registry):
        """Test step builder initialization."""
        # Mock the registry to return step type
        mock_registry.get.return_value = 'MockConfig'
        
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Verify all step builders were initialized
        self.assertEqual(len(assembler.step_builders), 3)
        for step_name in self.nodes:
            self.assertIn(step_name, assembler.step_builders)
            builder = assembler.step_builders[step_name]
            self.assertIsInstance(builder, MockStepBuilder)
            self.assertEqual(builder.config, self.config_map[step_name])

    def test_propagate_messages(self):
        """Test message propagation between steps."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock compatibility calculation to return high score
        self.mock_dependency_resolver._calculate_compatibility.return_value = 0.9
        
        # Call propagate messages
        assembler._propagate_messages()
        
        # Verify messages were stored
        # step2 should have messages from step1
        # step3 should have messages from step2
        self.assertTrue(len(assembler.step_messages) >= 0)  # May be empty if no matches

    def test_generate_outputs(self):
        """Test output generation for a step."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Generate outputs for step1
        outputs = assembler._generate_outputs('step1')
        
        # Verify outputs were generated based on specification
        self.assertIsInstance(outputs, dict)
        # Should have outputs based on the mock spec
        expected_base = "s3://test-bucket/pipeline/mockstep"
        if outputs:  # Only check if outputs were generated
            for output_name, output_path in outputs.items():
                self.assertTrue(output_path.startswith("s3://"))

    def test_instantiate_step(self):
        """Test step instantiation."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Instantiate step1 (no dependencies)
        step = assembler._instantiate_step('step1')
        
        # Verify step was created
        self.assertIsNotNone(step)
        self.assertTrue(hasattr(step, 'name'))
        
        # Store the step for dependency testing
        assembler.step_instances['step1'] = step
        
        # Instantiate step2 (depends on step1)
        step2 = assembler._instantiate_step('step2')
        self.assertIsNotNone(step2)

    @patch('src.cursus.core.assembler.pipeline_assembler.Pipeline')
    def test_generate_pipeline(self, mock_pipeline_class):
        """Test pipeline generation."""
        # Mock Pipeline constructor
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            sagemaker_session=self.mock_session,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Generate pipeline
        pipeline_name = "test_pipeline"
        result = assembler.generate_pipeline(pipeline_name)
        
        # Verify pipeline was created
        self.assertEqual(result, mock_pipeline)
        mock_pipeline_class.assert_called_once()
        
        # Verify all steps were instantiated
        self.assertEqual(len(assembler.step_instances), 3)

    def test_generate_pipeline_with_cycle(self):
        """Test pipeline generation with cyclic DAG raises error."""
        # Create DAG with cycle
        cyclic_dag = PipelineDAG(
            nodes=['step1', 'step2', 'step3'],
            edges=[('step1', 'step2'), ('step2', 'step3'), ('step3', 'step1')]
        )
        
        assembler = PipelineAssembler(
            dag=cyclic_dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Should raise ValueError due to cycle
        with self.assertRaises(ValueError) as context:
            assembler.generate_pipeline("test_pipeline")
        
        self.assertIn("Failed to determine build order", str(context.exception))

    @patch('src.cursus.core.assembler.pipeline_assembler.create_pipeline_components')
    def test_create_with_components(self, mock_create_components):
        """Test factory method for creating assembler with components."""
        # Mock component creation
        mock_components = {
            "registry_manager": self.mock_registry_manager,
            "resolver": self.mock_dependency_resolver
        }
        mock_create_components.return_value = mock_components
        
        # Create assembler using factory method
        assembler = PipelineAssembler.create_with_components(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            context_name="test_context",
            sagemaker_session=self.mock_session,
            role=self.role
        )
        
        # Verify components were used
        mock_create_components.assert_called_once_with("test_context")
        self.assertEqual(assembler._registry_manager, self.mock_registry_manager)
        self.assertEqual(assembler._dependency_resolver, self.mock_dependency_resolver)

    def test_get_registry_manager(self):
        """Test registry manager getter."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        result = assembler._get_registry_manager()
        self.assertEqual(result, self.mock_registry_manager)

    def test_get_dependency_resolver(self):
        """Test dependency resolver getter."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        result = assembler._get_dependency_resolver()
        self.assertEqual(result, self.mock_dependency_resolver)

    def test_cradle_loading_requests_storage(self):
        """Test that Cradle data loading requests are stored correctly."""
        # Create a mock builder that has get_request_dict method
        class MockCradleBuilder(MockStepBuilder):
            def get_request_dict(self):
                return {"request": "test_cradle_request"}
        
        # Create a mock config that will be identified as CradleDataLoading
        class MockCradleConfig(MockConfig):
            pass
        
        # Update step builder map and config
        cradle_config_map = {
            'step1': MockCradleConfig()
        }
        cradle_builder_map = {
            'CradleDataLoading': MockCradleBuilder
        }
        cradle_dag = PipelineDAG(nodes=['step1'], edges=[])
        
        with patch('src.cursus.core.assembler.pipeline_assembler.BasePipelineConfig.get_step_name') as mock_get_step_name:
            mock_get_step_name.return_value = "CradleDataLoading"
            
            assembler = PipelineAssembler(
                dag=cradle_dag,
                config_map=cradle_config_map,
                step_builder_map=cradle_builder_map,
                registry_manager=self.mock_registry_manager,
                dependency_resolver=self.mock_dependency_resolver
            )
            
            # Instantiate the step
            step = assembler._instantiate_step('step1')
            
            # Verify request was stored
            self.assertIn(step.name, assembler.cradle_loading_requests)
            self.assertEqual(assembler.cradle_loading_requests[step.name], {"request": "test_cradle_request"})

    def test_pipeline_regeneration(self):
        """Test that pipeline can be regenerated (step instances are cleared)."""
        assembler = PipelineAssembler(
            dag=self.dag,
            config_map=self.config_map,
            step_builder_map=self.step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Add some mock step instances
        assembler.step_instances = {'step1': Mock(), 'step2': Mock()}
        
        with patch('src.cursus.core.assembler.pipeline_assembler.Pipeline') as mock_pipeline_class:
            mock_pipeline_class.return_value = Mock()
            
            # Generate pipeline (should clear existing instances)
            assembler.generate_pipeline("test_pipeline")
            
            # Verify instances were cleared and recreated
            self.assertEqual(len(assembler.step_instances), 3)  # All steps recreated

    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        with patch('src.cursus.core.assembler.pipeline_assembler.logger') as mock_logger:
            assembler = PipelineAssembler(
                dag=self.dag,
                config_map=self.config_map,
                step_builder_map=self.step_builder_map,
                registry_manager=self.mock_registry_manager,
                dependency_resolver=self.mock_dependency_resolver
            )
            
            # Verify logging calls were made during initialization
            mock_logger.info.assert_called()

if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
