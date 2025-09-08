"""
Unit tests for workspace pipeline assembler.

Tests the WorkspacePipelineAssembler for workspace-aware pipeline assembly.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.workspace.core.assembler import WorkspacePipelineAssembler
from cursus.workspace.core.config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from cursus.api.dag.base_dag import PipelineDAG

class TestWorkspacePipelineAssembler(unittest.TestCase):
    """Test cases for WorkspacePipelineAssembler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = self.temp_dir
        
        # Create sample workspace configuration
        steps = [
            WorkspaceStepDefinition(
                step_name='preprocessing',
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'input_path': '/data/input', 'output_path': '/data/processed'},
                workspace_root=self.temp_workspace
            ),
            WorkspaceStepDefinition(
                step_name='training',
                developer_id='dev2',
                step_type='XGBoostTraining',
                config_data={'model_params': {'max_depth': 6}},
                workspace_root=self.temp_workspace,
                dependencies=['preprocessing']
            )
        ]
        
        self.sample_workspace_config = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=self.temp_workspace,
            steps=steps,
            global_config={'region': 'us-west-2'}
        )
        
        # Create mock workspace registry
        self.mock_workspace_registry = Mock()
        
        # Mock builder classes
        mock_preprocessing_builder = Mock()
        mock_preprocessing_builder.__name__ = 'PreprocessingBuilder'
        mock_training_builder = Mock()
        mock_training_builder.__name__ = 'TrainingBuilder'
        
        # Mock config classes
        mock_preprocessing_config = Mock()
        mock_preprocessing_config.__name__ = 'PreprocessingConfig'
        mock_training_config = Mock()
        mock_training_config.__name__ = 'TrainingConfig'
        
        # Setup registry responses
        def find_builder_class(step_name, developer_id=None):
            if step_name == 'preprocessing':
                return mock_preprocessing_builder
            elif step_name == 'training':
                return mock_training_builder
            return None
        
        def find_config_class(step_name, developer_id=None):
            if step_name == 'preprocessing':
                return mock_preprocessing_config
            elif step_name == 'training':
                return mock_training_config
            return None
        
        self.mock_workspace_registry.find_builder_class.side_effect = find_builder_class
        self.mock_workspace_registry.find_config_class.side_effect = find_config_class
        
        # Mock validation
        self.mock_workspace_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': [
                {'step_name': 'preprocessing', 'component_type': 'builder'},
                {'step_name': 'training', 'component_type': 'builder'}
            ]
        }
        
        self.mock_workspace_registry.get_workspace_summary.return_value = {
            'workspace_root': '/test/workspace',
            'total_components': 4,
            'developers': ['dev1', 'dev2']
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_assembler_initialization_with_workspace_manager(self):
        """Test assembler initialization with workspace manager (Phase 2 optimization)."""
        mock_manager = Mock()
        mock_manager.discovery_manager = Mock()
        
        assembler = WorkspacePipelineAssembler(
            workspace_root=self.temp_workspace,
            workspace_manager=mock_manager
        )
        
        self.assertEqual(assembler.workspace_root, self.temp_workspace)
        self.assertEqual(assembler.workspace_manager, mock_manager)
        # The registry is created internally
        self.assertIsNotNone(assembler.workspace_registry)
    
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_assembler_initialization_without_workspace_manager(self, mock_manager_class):
        """Test assembler initialization without workspace manager."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        self.assertEqual(assembler.workspace_root, self.temp_workspace)
        self.assertEqual(assembler.workspace_manager, mock_manager)
        # The registry is created internally
        self.assertIsNotNone(assembler.workspace_registry)
        mock_manager_class.assert_called_once_with(self.temp_workspace)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_resolve_workspace_configs(self, mock_manager_class, mock_registry_class):
        """Test resolving workspace configurations."""
        # Mock config classes that can be instantiated
        mock_preprocessing_config_class = Mock()
        mock_preprocessing_config_class.__name__ = 'PreprocessingConfig'
        mock_preprocessing_config_instance = {'input_path': '/data/input', 'output_path': '/data/processed'}
        mock_preprocessing_config_class.return_value = mock_preprocessing_config_instance

        mock_training_config_class = Mock()
        mock_training_config_class.__name__ = 'TrainingConfig'
        mock_training_config_instance = {'model_params': {'max_depth': 6}}
        mock_training_config_class.return_value = mock_training_config_instance

        def find_config_class(step_name, developer_id=None):
            if step_name == 'preprocessing':
                return mock_preprocessing_config_class
            elif step_name == 'training':
                return mock_training_config_class
            return None

        # Create mock registry and manager
        mock_registry = Mock()
        mock_registry.find_config_class.side_effect = find_config_class
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager.discovery_manager = Mock()
        mock_manager_class.return_value = mock_manager

        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Replace the registry instance with our mock
        assembler.workspace_registry = mock_registry
        
        config_map = assembler._resolve_workspace_configs(self.sample_workspace_config)

        self.assertIn('preprocessing', config_map)
        self.assertIn('training', config_map)
        self.assertEqual(config_map['preprocessing'], mock_preprocessing_config_instance)
        self.assertEqual(config_map['training'], mock_training_config_instance)
        
        # Verify config classes were called with correct data
        mock_preprocessing_config_class.assert_called_once_with(
            input_path='/data/input', 
            output_path='/data/processed'
        )
        mock_training_config_class.assert_called_once_with(
            model_params={'max_depth': 6}
        )
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_resolve_workspace_configs_fallback(self, mock_manager_class, mock_registry_class):
        """Test resolving workspace configs with fallback to raw data."""
        mock_registry = Mock()
        mock_registry.find_config_class.return_value = None  # No config class found
        mock_registry_class.return_value = mock_registry
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        config_map = assembler._resolve_workspace_configs(self.sample_workspace_config)
        
        # Should fallback to raw config data
        self.assertEqual(config_map['preprocessing'], {'input_path': '/data/input', 'output_path': '/data/processed'})
        self.assertEqual(config_map['training'], {'model_params': {'max_depth': 6}})
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_resolve_workspace_builders(self, mock_manager_class, mock_registry_class):
        """Test resolving workspace builders."""
        mock_preprocessing_builder = Mock()
        mock_preprocessing_builder.__name__ = 'PreprocessingBuilder'
        mock_training_builder = Mock()
        mock_training_builder.__name__ = 'TrainingBuilder'
        
        def find_builder_class(step_name, developer_id=None):
            if step_name == 'preprocessing':
                return mock_preprocessing_builder
            elif step_name == 'training':
                return mock_training_builder
            return None
        
        # Create mock registry and manager
        mock_registry = Mock()
        mock_registry.find_builder_class.side_effect = find_builder_class
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager.discovery_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Replace the registry instance with our mock
        assembler.workspace_registry = mock_registry
        
        builder_map = assembler._resolve_workspace_builders(self.sample_workspace_config)
        
        self.assertIn('DataPreprocessing', builder_map)
        self.assertIn('XGBoostTraining', builder_map)
        self.assertEqual(builder_map['DataPreprocessing'], mock_preprocessing_builder)
        self.assertEqual(builder_map['XGBoostTraining'], mock_training_builder)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_validate_workspace_components(self, mock_manager_class, mock_registry_class):
        """Test workspace component validation."""
        # Mock builder and config classes with __name__ attributes
        mock_builder = Mock()
        mock_builder.__name__ = 'TestBuilder'
        mock_config = Mock()
        mock_config.__name__ = 'TestConfig'
        
        # Create mock registry and manager
        mock_registry = Mock()
        mock_registry.find_builder_class.return_value = mock_builder
        mock_registry.find_config_class.return_value = mock_config
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager.discovery_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Replace the registry instance with our mock
        assembler.workspace_registry = mock_registry
        
        # Mock the workspace-specific validation methods
        with patch.object(assembler, '_validate_developer_consistency') as mock_dev_validation, \
             patch.object(assembler, '_validate_step_type_consistency') as mock_type_validation:
            
            mock_dev_validation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_type_validation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            result = assembler.validate_workspace_components(self.sample_workspace_config)
            
            self.assertTrue(result['valid'])
            self.assertTrue(result['workspace_valid'])
            self.assertTrue(result['overall_valid'])
            self.assertIn('workspace_validation', result)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_validate_developer_consistency(self, mock_manager_class, mock_registry_class):
        """Test developer consistency validation."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        result = assembler._validate_developer_consistency(self.sample_workspace_config)
        
        self.assertTrue(result['valid'])
        self.assertIn('developer_stats', result)
        self.assertIn('dev1', result['developer_stats'])
        self.assertIn('dev2', result['developer_stats'])
        self.assertEqual(result['developer_stats']['dev1']['step_count'], 1)
        self.assertEqual(result['developer_stats']['dev2']['step_count'], 1)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_validate_step_type_consistency(self, mock_manager_class, mock_registry_class):
        """Test step type consistency validation."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        result = assembler._validate_step_type_consistency(self.sample_workspace_config)
        
        self.assertTrue(result['valid'])
        self.assertIn('step_type_stats', result)
        self.assertIn('DataPreprocessing', result['step_type_stats'])
        self.assertIn('XGBoostTraining', result['step_type_stats'])
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_create_dag_from_workspace_config(self, mock_manager_class, mock_registry_class):
        """Test creating DAG from workspace configuration."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        dag = assembler._create_dag_from_workspace_config(self.sample_workspace_config)
        
        self.assertIsInstance(dag, PipelineDAG)
        self.assertIn('preprocessing', dag.nodes)
        self.assertIn('training', dag.nodes)
        self.assertIn(('preprocessing', 'training'), dag.edges)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_assemble_workspace_pipeline_success(self, mock_manager_class, mock_registry_class):
        """Test successful workspace pipeline assembly."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_registry_class.return_value = mock_registry
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Mock all the required methods
        with patch.object(assembler, 'validate_workspace_components') as mock_validate, \
             patch.object(assembler, '_resolve_workspace_configs') as mock_resolve_configs, \
             patch.object(assembler, '_resolve_workspace_builders') as mock_resolve_builders, \
             patch.object(assembler, '_create_dag_from_workspace_config') as mock_create_dag, \
             patch.object(assembler, '_initialize_step_builders') as mock_init_builders, \
             patch.object(assembler, 'generate_pipeline') as mock_generate:
            
            mock_validate.return_value = {'overall_valid': True}
            mock_resolve_configs.return_value = {'preprocessing': Mock(), 'training': Mock()}
            mock_resolve_builders.return_value = {'DataPreprocessing': Mock(), 'XGBoostTraining': Mock()}
            mock_create_dag.return_value = PipelineDAG()
            mock_pipeline = Mock()
            mock_generate.return_value = mock_pipeline
            
            result = assembler.assemble_workspace_pipeline(self.sample_workspace_config)
            
            self.assertEqual(result, mock_pipeline)
            mock_validate.assert_called_once()
            mock_resolve_configs.assert_called_once()
            mock_resolve_builders.assert_called_once()
            mock_create_dag.assert_called_once()
            mock_generate.assert_called_once_with('test_pipeline')
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_assemble_workspace_pipeline_validation_failure(self, mock_manager_class, mock_registry_class):
        """Test workspace pipeline assembly with validation failure."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        with patch.object(assembler, 'validate_workspace_components') as mock_validate:
            mock_validate.return_value = {'overall_valid': False, 'errors': ['Test error']}
            
            with self.assertRaises(ValueError) as context:
                assembler.assemble_workspace_pipeline(self.sample_workspace_config)
            
            self.assertIn("Workspace component validation failed", str(context.exception))
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_config(self, mock_manager_class, mock_registry_class):
        """Test creating assembler from workspace configuration."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler.from_workspace_config(
            workspace_config=self.sample_workspace_config,
            role='test-role'
        )
        
        self.assertIsInstance(assembler, WorkspacePipelineAssembler)
        self.assertEqual(assembler.workspace_root, self.temp_workspace)
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_config_file_json(self, mock_manager_class, mock_registry_class):
        """Test creating assembler from JSON configuration file."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        # Create temporary JSON file
        json_file = Path(self.temp_workspace) / 'config.json'
        self.sample_workspace_config.to_json_file(str(json_file))
        
        with patch('src.cursus.workspace.core.config.WorkspacePipelineDefinition.from_json_file') as mock_load:
            mock_load.return_value = self.sample_workspace_config
            
            assembler = WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path=str(json_file),
                role='test-role'
            )
            
            self.assertIsInstance(assembler, WorkspacePipelineAssembler)
            mock_load.assert_called_once_with(str(json_file))
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_config_file_yaml(self, mock_manager_class, mock_registry_class):
        """Test creating assembler from YAML configuration file."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        yaml_file = Path(self.temp_workspace) / 'config.yaml'
        
        with patch('src.cursus.workspace.core.config.WorkspacePipelineDefinition.from_yaml_file') as mock_load:
            mock_load.return_value = self.sample_workspace_config
            
            assembler = WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path=str(yaml_file),
                role='test-role'
            )
            
            self.assertIsInstance(assembler, WorkspacePipelineAssembler)
            mock_load.assert_called_once_with(str(yaml_file))
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_from_workspace_config_file_unsupported(self, mock_manager_class, mock_registry_class):
        """Test creating assembler from unsupported file format."""
        mock_registry_class.return_value = Mock()
        mock_manager_class.return_value = Mock()
        
        with self.assertRaises(ValueError) as context:
            WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path='/path/to/config.txt'
            )
        
        self.assertIn("Unsupported config file format", str(context.exception))
    
    @patch('src.cursus.workspace.core.registry.WorkspaceComponentRegistry')
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_get_workspace_summary(self, mock_manager_class, mock_registry_class):
        """Test getting workspace summary."""
        mock_registry = Mock()
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': self.temp_workspace,
            'total_components': 4
        }
        mock_registry_class.return_value = mock_registry
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Mock some assembler state
        assembler.dag = PipelineDAG()
        assembler.dag.add_node('test_node')
        assembler.config_map = {'step1': Mock()}
        assembler.step_builder_map = {'type1': Mock()}
        assembler.step_instances = {'step1': Mock()}
        
        summary = assembler.get_workspace_summary()
        
        self.assertEqual(summary['workspace_root'], self.temp_workspace)
        self.assertIn('registry_summary', summary)
        self.assertIn('assembly_status', summary)
        self.assertEqual(summary['assembly_status']['dag_nodes'], 1)
        self.assertEqual(summary['assembly_status']['config_count'], 1)
        self.assertEqual(summary['assembly_status']['builder_count'], 1)
        self.assertEqual(summary['assembly_status']['step_instances'], 1)
    
    @patch('src.cursus.workspace.core.manager.WorkspaceManager')
    def test_preview_workspace_assembly(self, mock_manager_class):
        """Test previewing workspace assembly."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {'valid': True}
        
        # Mock builder and config classes
        mock_builder = Mock()
        mock_builder.__name__ = 'TestBuilder'
        mock_config = Mock()
        mock_config.__name__ = 'TestConfig'
        
        mock_registry.find_builder_class.return_value = mock_builder
        mock_registry.find_config_class.return_value = mock_config
        mock_manager_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=self.temp_workspace)
        
        # Replace the registry instance with our mock
        assembler.workspace_registry = mock_registry
        
        with patch.object(assembler, 'validate_workspace_components') as mock_validate:
            mock_validate.return_value = {'valid': True}
            
            preview = assembler.preview_workspace_assembly(self.sample_workspace_config)
            
            self.assertIn('workspace_config', preview)
            self.assertIn('component_resolution', preview)
            self.assertIn('validation_results', preview)
            self.assertIn('assembly_plan', preview)
            
            self.assertEqual(preview['workspace_config']['pipeline_name'], 'test_pipeline')
            self.assertEqual(preview['workspace_config']['step_count'], 2)
            
            # Check component resolution - should have entries now
            self.assertIsInstance(preview['component_resolution'], dict)
            
            # Check assembly plan
            self.assertTrue(preview['assembly_plan']['dag_valid'])
            self.assertIn('build_order', preview['assembly_plan'])

if __name__ == '__main__':
    unittest.main()
