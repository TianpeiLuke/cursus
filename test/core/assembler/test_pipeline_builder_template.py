import unittest
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
        return config_class_name.replace('Config', 'Step')

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
        'Base': MockBasePipelineConfig,
        'TestConfig1': MockBasePipelineConfig,
        'TestConfig2': MockBasePipelineConfig,
    }
    
    def _validate_configuration(self) -> None:
        """Simple validation for testing."""
        if 'Base' not in self.configs:
            raise ValueError("Base configuration required")
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create a simple test DAG."""
        dag = PipelineDAG()
        dag.add_node('step1')
        dag.add_node('step2')
        dag.add_node('step3')
        dag.add_edge('step1', 'step2')
        dag.add_edge('step2', 'step3')
        return dag
    
    def _create_config_map(self):
        """Create config map for testing."""
        return {
            'step1': self.configs.get('TestConfig1', MagicMock(spec=MockBasePipelineConfig)),
            'step2': self.configs.get('TestConfig2', MagicMock(spec=MockBasePipelineConfig)),
            'step3': self.configs.get('Base', MagicMock(spec=MockBasePipelineConfig)),
        }
    
    def _create_step_builder_map(self):
        """Create step builder map for testing."""
        return {
            'TestStep1': MagicMock(spec=type),
            'TestStep2': MagicMock(spec=type),
            'TestStep3': MagicMock(spec=type),
        }

class TestPipelineTemplateBase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file content
        self.config_data = {
            'Base': {
                'pipeline_name': 'test-pipeline',
                'pipeline_version': '1.0',
                'pipeline_s3_loc': 's3://test-bucket/test-pipeline'
            },
            'TestConfig1': {
                'some_param': 'value1'
            },
            'TestConfig2': {
                'some_param': 'value2'
            }
        }
        
        # Mock file operations
        self.mock_open = patch('builtins.open', create=True)
        self.mock_file = self.mock_open.start()
        self.mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.config_data)
        
        # Mock load_configs function from the correct import path
        self.load_configs_patch = patch('src.cursus.steps.configs.utils.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        
        # Create mock configs
        self.mock_base_config = MagicMock(spec=MockBasePipelineConfig)
        self.mock_base_config.pipeline_name = 'test-pipeline'
        self.mock_base_config.pipeline_version = '1.0'
        self.mock_base_config.pipeline_s3_loc = 's3://test-bucket/test-pipeline'
        
        self.mock_configs = {
            'Base': self.mock_base_config,
            'TestConfig1': MagicMock(spec=MockBasePipelineConfig),
            'TestConfig2': MagicMock(spec=MockBasePipelineConfig),
        }
        self.mock_load_configs.return_value = self.mock_configs
        
        # Mock build_complete_config_classes
        self.build_complete_patch = patch('src.cursus.steps.configs.utils.build_complete_config_classes')
        self.mock_build_complete = self.build_complete_patch.start()
        self.mock_build_complete.return_value = {}
        
        # Mock create_pipeline_components
        self.components_patch = patch('src.cursus.core.assembler.pipeline_template_base.create_pipeline_components')
        self.mock_create_components = self.components_patch.start()
        
        self.mock_registry_manager = MagicMock(spec=MockRegistryManager)
        self.mock_dependency_resolver = MagicMock(spec=MockUnifiedDependencyResolver)
        
        self.mock_create_components.return_value = {
            'registry_manager': self.mock_registry_manager,
            'resolver': self.mock_dependency_resolver
        }
        
        # Mock PipelineAssembler
        self.assembler_patch = patch('src.cursus.core.assembler.pipeline_template_base.PipelineAssembler')
        self.mock_assembler_cls = self.assembler_patch.start()
        self.mock_assembler = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_assembler.generate_pipeline.return_value = self.mock_pipeline
        self.mock_assembler_cls.return_value = self.mock_assembler
        
        # Mock generate_pipeline_name
        self.name_gen_patch = patch('src.cursus.core.assembler.pipeline_template_base.generate_pipeline_name')
        self.mock_generate_name = self.name_gen_patch.start()
        self.mock_generate_name.return_value = 'test-pipeline-v1-0'

    def tearDown(self):
        """Clean up patches after each test."""
        self.mock_open.stop()
        self.load_configs_patch.stop()
        self.build_complete_patch.stop()
        self.components_patch.stop()
        self.assembler_patch.stop()
        self.name_gen_patch.stop()

    def test_initialization(self):
        """Test that the template initializes correctly."""
        template = ConcretePipelineTemplate(
            config_path='test_config.json',
            sagemaker_session=MagicMock(),
            role='test-role'
        )
        
        # Verify attributes were set correctly
        self.assertEqual(template.config_path, 'test_config.json')
        self.assertEqual(template.configs, self.mock_configs)
        self.assertEqual(template.base_config, self.mock_base_config)
        self.assertIsNotNone(template._registry_manager)
        self.assertIsNotNone(template._dependency_resolver)
        
        # Verify load_configs was called with correct parameters
        self.mock_load_configs.assert_called_once()
        
        # Verify components were created
        self.mock_create_components.assert_called_once_with('test-pipeline')
        
        # Verify loaded_config_data was set
        self.assertIsNotNone(template.loaded_config_data)
        
        # Verify pipeline_metadata was initialized
        self.assertEqual(template.pipeline_metadata, {})

    def test_initialization_with_provided_components(self):
        """Test initialization with provided dependency components."""
        custom_registry = MagicMock(spec=MockRegistryManager)
        custom_resolver = MagicMock(spec=MockUnifiedDependencyResolver)
        
        template = ConcretePipelineTemplate(
            config_path='test_config.json',
            registry_manager=custom_registry,
            dependency_resolver=custom_resolver
        )
        
        # Verify provided components were used
        self.assertEqual(template._registry_manager, custom_registry)
        self.assertEqual(template._dependency_resolver, custom_resolver)
        
        # Verify create_pipeline_components was not called
        self.mock_create_components.assert_not_called()

    def test_config_loading(self):
        """Test that configurations are loaded correctly."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Verify build_complete_config_classes was called
        self.mock_build_complete.assert_called_once()
        
        # Verify load_configs was called with merged config classes
        call_args = self.mock_load_configs.call_args
        self.assertEqual(call_args[0][0], 'test_config.json')
        # The second argument should be the merged config classes
        config_classes = call_args[0][1]
        self.assertIn('Base', config_classes)
        self.assertIn('TestConfig1', config_classes)
        self.assertIn('TestConfig2', config_classes)

    def test_base_config_validation(self):
        """Test that missing base config raises error."""
        # Mock load_configs to return configs without Base
        self.mock_load_configs.return_value = {'TestConfig1': MagicMock()}
        
        with self.assertRaises(ValueError) as context:
            ConcretePipelineTemplate(config_path='test_config.json')
        
        self.assertIn("Base configuration not found", str(context.exception))

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline."""
        template = ConcretePipelineTemplate(
            config_path='test_config.json',
            sagemaker_session=MagicMock(),
            role='test-role'
        )
        
        # Call generate_pipeline
        pipeline = template.generate_pipeline()
        
        # Verify PipelineAssembler was created with correct parameters
        self.mock_assembler_cls.assert_called_once()
        call_kwargs = self.mock_assembler_cls.call_args[1]
        
        # Verify DAG was created
        self.assertIsNotNone(call_kwargs['dag'])
        
        # Verify config_map was created
        self.assertIsNotNone(call_kwargs['config_map'])
        
        # Verify step_builder_map was created
        self.assertIsNotNone(call_kwargs['step_builder_map'])
        
        # Verify dependency components were passed
        self.assertEqual(call_kwargs['registry_manager'], self.mock_registry_manager)
        self.assertEqual(call_kwargs['dependency_resolver'], self.mock_dependency_resolver)
        
        # Verify generate_pipeline was called on assembler
        self.mock_assembler.generate_pipeline.assert_called_once_with('test-pipeline-v1-0')
        
        # Verify pipeline was returned
        self.assertEqual(pipeline, self.mock_pipeline)

    def test_pipeline_name_generation(self):
        """Test pipeline name generation."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Test default name generation
        name = template._get_pipeline_name()
        self.mock_generate_name.assert_called_with('test-pipeline', '1.0')
        self.assertEqual(name, 'test-pipeline-v1-0')

    def test_pipeline_name_explicit_override(self):
        """Test explicit pipeline name override."""
        self.mock_base_config.explicit_pipeline_name = 'custom-pipeline-name'
        
        template = ConcretePipelineTemplate(config_path='test_config.json')
        name = template._get_pipeline_name()
        
        # Should return explicit name without calling generator
        self.assertEqual(name, 'custom-pipeline-name')
        self.mock_generate_name.assert_not_called()

    def test_store_pipeline_metadata(self):
        """Test that pipeline metadata is stored correctly."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Mock assembler with metadata
        mock_assembler = MagicMock()
        mock_assembler.cradle_loading_requests = {'step1': {'request': 'data'}}
        mock_assembler.step_instances = {'step1': MagicMock(), 'step2': MagicMock()}
        
        # Call _store_pipeline_metadata
        template._store_pipeline_metadata(mock_assembler)
        
        # Verify metadata was stored
        self.assertEqual(template.pipeline_metadata['cradle_loading_requests'], {'step1': {'request': 'data'}})
        self.assertEqual(template.pipeline_metadata['step_instances'], mock_assembler.step_instances)

    def test_create_with_components_class_method(self):
        """Test create_with_components class method."""
        # Call class method
        template = ConcretePipelineTemplate.create_with_components(
            config_path='test_config.json',
            context_name='custom-context',
            sagemaker_session=MagicMock()
        )
        
        # Verify create_pipeline_components was called with context
        self.mock_create_components.assert_called_with('custom-context')
        
        # Verify template was created with components
        self.assertIsInstance(template, ConcretePipelineTemplate)

    def test_build_with_context_class_method(self):
        """Test build_with_context class method."""
        # Mock dependency_resolution_context
        with patch('src.cursus.core.assembler.pipeline_template_base.dependency_resolution_context') as mock_context:
            mock_context.return_value.__enter__.return_value = {
                'registry_manager': self.mock_registry_manager,
                'resolver': self.mock_dependency_resolver
            }
            
            # Call class method
            pipeline = ConcretePipelineTemplate.build_with_context(
                config_path='test_config.json'
            )
            
            # Verify context manager was used
            mock_context.assert_called_once_with(clear_on_exit=True)
            
            # Verify pipeline was returned
            self.assertEqual(pipeline, self.mock_pipeline)

    def test_build_in_thread_class_method(self):
        """Test build_in_thread class method."""
        # Mock get_thread_components
        with patch('src.cursus.core.assembler.pipeline_template_base.get_thread_components') as mock_thread:
            mock_thread.return_value = {
                'registry_manager': self.mock_registry_manager,
                'resolver': self.mock_dependency_resolver
            }
            
            # Call class method
            pipeline = ConcretePipelineTemplate.build_in_thread(
                config_path='test_config.json'
            )
            
            # Verify thread components were used
            mock_thread.assert_called_once()
            
            # Verify pipeline was returned
            self.assertEqual(pipeline, self.mock_pipeline)

    def test_fill_execution_document(self):
        """Test fill_execution_document method."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Test default implementation
        doc = {'existing': 'data'}
        result = template.fill_execution_document(doc)
        
        # Should return unchanged document
        self.assertEqual(result, doc)

    def test_initialization_with_notebook_root(self):
        """Test initialization with custom notebook_root."""
        custom_root = Path('/custom/notebook/root')
        template = ConcretePipelineTemplate(
            config_path='test_config.json',
            notebook_root=custom_root
        )
        
        # Verify notebook_root was set
        self.assertEqual(template.notebook_root, custom_root)

    def test_initialization_default_notebook_root(self):
        """Test initialization with default notebook_root."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Verify default notebook_root was set to current working directory
        self.assertEqual(template.notebook_root, Path.cwd())

    def test_config_loading_error_handling(self):
        """Test error handling when config file loading fails."""
        # Mock file operations to raise an exception
        self.mock_file.return_value.__enter__.return_value.read.side_effect = Exception("File read error")
        
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Verify loaded_config_data is None when file loading fails
        self.assertIsNone(template.loaded_config_data)

    def test_pipeline_name_fallback_values(self):
        """Test pipeline name generation with fallback values."""
        # Remove pipeline_name and pipeline_version from base config
        del self.mock_base_config.pipeline_name
        del self.mock_base_config.pipeline_version
        
        template = ConcretePipelineTemplate(config_path='test_config.json')
        name = template._get_pipeline_name()
        
        # Should use fallback values
        self.mock_generate_name.assert_called_with('mods', '1.0')

    def test_get_pipeline_parameters_default(self):
        """Test default pipeline parameters."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        params = template._get_pipeline_parameters()
        
        # Default implementation should return empty list
        self.assertEqual(params, [])

    def test_store_pipeline_metadata_without_attributes(self):
        """Test storing pipeline metadata when assembler doesn't have expected attributes."""
        template = ConcretePipelineTemplate(config_path='test_config.json')
        
        # Mock assembler without cradle_loading_requests or step_instances
        mock_assembler = MagicMock()
        # Remove the attributes
        if hasattr(mock_assembler, 'cradle_loading_requests'):
            del mock_assembler.cradle_loading_requests
        if hasattr(mock_assembler, 'step_instances'):
            del mock_assembler.step_instances
        
        # Call _store_pipeline_metadata
        template._store_pipeline_metadata(mock_assembler)
        
        # Verify metadata dict is still empty since attributes don't exist
        self.assertEqual(template.pipeline_metadata, {})

    def test_config_classes_validation(self):
        """Test that CONFIG_CLASSES must be defined."""
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
        
        with self.assertRaises(ValueError) as context:
            InvalidTemplate(config_path='test_config.json')
        
        self.assertIn("CONFIG_CLASSES must be defined", str(context.exception))

    def test_initialization_partial_components(self):
        """Test initialization with only one component provided."""
        custom_registry = MagicMock(spec=MockRegistryManager)
        
        template = ConcretePipelineTemplate(
            config_path='test_config.json',
            registry_manager=custom_registry
            # dependency_resolver not provided
        )
        
        # Verify provided component was used
        self.assertEqual(template._registry_manager, custom_registry)
        
        # Verify missing component was created
        self.assertIsNotNone(template._dependency_resolver)
        
        # Verify create_pipeline_components was called to get missing component
        self.mock_create_components.assert_called_once_with('test-pipeline')

class TestPipelineAssembler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for PipelineAssembler tests."""
        # Mock PipelineDAG
        self.mock_dag = MagicMock(spec=PipelineDAG)
        self.mock_dag.nodes = ['step1', 'step2', 'step3']
        self.mock_dag.edges = [('step1', 'step2'), ('step2', 'step3')]
        self.mock_dag.topological_sort.return_value = ['step1', 'step2', 'step3']
        self.mock_dag.get_dependencies.side_effect = lambda node: {
            'step1': [],
            'step2': ['step1'],
            'step3': ['step2']
        }[node]
        
        # Mock configs
        self.mock_config1 = MagicMock(spec=MockBasePipelineConfig)
        self.mock_config2 = MagicMock(spec=MockBasePipelineConfig)
        self.mock_config3 = MagicMock(spec=MockBasePipelineConfig)
        
        self.mock_config_map = {
            'step1': self.mock_config1,
            'step2': self.mock_config2,
            'step3': self.mock_config3
        }
        
        # Mock step builders
        self.mock_builder_cls = MagicMock()
        self.mock_builder = MagicMock()
        self.mock_builder_cls.return_value = self.mock_builder
        
        self.mock_step_builder_map = {
            'TestStep1': self.mock_builder_cls,
            'TestStep2': self.mock_builder_cls,
            'TestStep3': self.mock_builder_cls,
        }
        
        # Mock dependency components
        self.mock_registry_manager = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_registry_manager.get_registry.return_value = self.mock_registry
        
        self.mock_dependency_resolver = MagicMock(spec=MockUnifiedDependencyResolver)
        
        # Mock CONFIG_STEP_REGISTRY
        self.registry_patch = patch('src.cursus.core.assembler.pipeline_assembler.CONFIG_STEP_REGISTRY')
        self.mock_step_registry = self.registry_patch.start()
        self.mock_step_registry.get.side_effect = lambda x: {
            'MockConfig': 'TestStep1',
            'BasePipelineConfig': 'TestStep2'
        }.get(x, 'TestStep3')
        
        # Mock create_dependency_resolver
        self.resolver_patch = patch('src.cursus.core.assembler.pipeline_assembler.create_dependency_resolver')
        self.mock_create_resolver = self.resolver_patch.start()
        self.mock_create_resolver.return_value = self.mock_dependency_resolver
        
        # Mock Pipeline
        self.pipeline_patch = patch('src.cursus.core.assembler.pipeline_assembler.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline

    def tearDown(self):
        """Clean up patches."""
        self.registry_patch.stop()
        self.resolver_patch.stop()
        self.pipeline_patch.stop()

    def test_assembler_initialization(self):
        """Test PipelineAssembler initialization."""
        assembler = PipelineAssembler(
            dag=self.mock_dag,
            config_map=self.mock_config_map,
            step_builder_map=self.mock_step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Verify attributes
        self.assertEqual(assembler.dag, self.mock_dag)
        self.assertEqual(assembler.config_map, self.mock_config_map)
        self.assertEqual(assembler.step_builder_map, self.mock_step_builder_map)
        self.assertEqual(assembler._registry_manager, self.mock_registry_manager)
        self.assertEqual(assembler._dependency_resolver, self.mock_dependency_resolver)
        
        # Verify step builders were initialized
        self.assertEqual(len(assembler.step_builders), 3)

    def test_assembler_validation_missing_configs(self):
        """Test assembler validation with missing configs."""
        # Remove a config
        incomplete_config_map = {'step1': self.mock_config1}
        
        with self.assertRaises(ValueError) as context:
            PipelineAssembler(
                dag=self.mock_dag,
                config_map=incomplete_config_map,
                step_builder_map=self.mock_step_builder_map
            )
        
        self.assertIn("Missing configs for nodes", str(context.exception))

    def test_assembler_validation_missing_builders(self):
        """Test assembler validation with missing step builders."""
        # Remove a step builder
        incomplete_builder_map = {'TestStep1': self.mock_builder_cls}
        
        with self.assertRaises(ValueError) as context:
            PipelineAssembler(
                dag=self.mock_dag,
                config_map=self.mock_config_map,
                step_builder_map=incomplete_builder_map
            )
        
        self.assertIn("Missing step builder for step type", str(context.exception))

    def test_generate_pipeline(self):
        """Test pipeline generation."""
        assembler = PipelineAssembler(
            dag=self.mock_dag,
            config_map=self.mock_config_map,
            step_builder_map=self.mock_step_builder_map,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock step creation
        mock_step1 = MagicMock()
        mock_step2 = MagicMock()
        mock_step3 = MagicMock()
        self.mock_builder.create_step.side_effect = [mock_step1, mock_step2, mock_step3]
        
        # Generate pipeline
        pipeline = assembler.generate_pipeline('test-pipeline')
        
        # Verify pipeline was created
        self.mock_pipeline_cls.assert_called_once_with(
            name='test-pipeline',
            parameters=[],
            steps=[mock_step1, mock_step2, mock_step3],
            sagemaker_session=None
        )
        
        # Verify step instances were stored
        self.assertEqual(len(assembler.step_instances), 3)
        self.assertEqual(assembler.step_instances['step1'], mock_step1)
        self.assertEqual(assembler.step_instances['step2'], mock_step2)
        self.assertEqual(assembler.step_instances['step3'], mock_step3)

    def test_create_with_components_class_method(self):
        """Test create_with_components class method."""
        # Mock create_pipeline_components
        with patch('src.cursus.core.assembler.pipeline_assembler.create_pipeline_components') as mock_create:
            mock_create.return_value = {
                'registry_manager': self.mock_registry_manager,
                'resolver': self.mock_dependency_resolver
            }
            
            assembler = PipelineAssembler.create_with_components(
                dag=self.mock_dag,
                config_map=self.mock_config_map,
                step_builder_map=self.mock_step_builder_map,
                context_name='test-context'
            )
            
            # Verify create_pipeline_components was called
            mock_create.assert_called_once_with('test-context')
            
            # Verify assembler was created with components
            self.assertEqual(assembler._registry_manager, self.mock_registry_manager)
            self.assertEqual(assembler._dependency_resolver, self.mock_dependency_resolver)

if __name__ == '__main__':
    unittest.main()
