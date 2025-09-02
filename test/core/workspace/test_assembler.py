"""
Unit tests for workspace pipeline assembler.

Tests the WorkspacePipelineAssembler for workspace-aware pipeline assembly.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.core.workspace.assembler import WorkspacePipelineAssembler
from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from src.cursus.api.dag.base_dag import PipelineDAG


class TestWorkspacePipelineAssembler:
    """Test cases for WorkspacePipelineAssembler."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_workspace_config(self, temp_workspace):
        """Create a sample workspace configuration."""
        steps = [
            WorkspaceStepDefinition(
                step_name='preprocessing',
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'input_path': '/data/input', 'output_path': '/data/processed'},
                workspace_root=temp_workspace
            ),
            WorkspaceStepDefinition(
                step_name='training',
                developer_id='dev2',
                step_type='XGBoostTraining',
                config_data={'model_params': {'max_depth': 6}},
                workspace_root=temp_workspace,
                dependencies=['preprocessing']
            )
        ]
        
        return WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root=temp_workspace,
            steps=steps,
            global_config={'region': 'us-west-2'}
        )
    
    @pytest.fixture
    def mock_workspace_registry(self):
        """Create a mock workspace registry."""
        mock_registry = Mock()
        
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
        
        mock_registry.find_builder_class.side_effect = find_builder_class
        mock_registry.find_config_class.side_effect = find_config_class
        
        # Mock validation
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': [
                {'step_name': 'preprocessing', 'component_type': 'builder'},
                {'step_name': 'training', 'component_type': 'builder'}
            ]
        }
        
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': '/test/workspace',
            'total_components': 4,
            'developers': ['dev1', 'dev2']
        }
        
        return mock_registry
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_assembler_initialization(self, mock_registry_class, temp_workspace):
        """Test assembler initialization."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        assert assembler.workspace_root == temp_workspace
        assert assembler.workspace_registry == mock_registry
        mock_registry_class.assert_called_once_with(temp_workspace)
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_resolve_workspace_configs(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test resolving workspace configurations."""
        mock_registry = Mock()

        # Mock config classes that can be instantiated
        mock_preprocessing_config_class = Mock()
        mock_preprocessing_config_instance = {'input_path': '/data/input', 'output_path': '/data/processed'}
        mock_preprocessing_config_class.return_value = mock_preprocessing_config_instance

        mock_training_config_class = Mock()
        mock_training_config_instance = {'model_params': {'max_depth': 6}}
        mock_training_config_class.return_value = mock_training_config_instance

        def find_config_class(step_name, developer_id=None):
            if step_name == 'preprocessing':
                return mock_preprocessing_config_class
            elif step_name == 'training':
                return mock_training_config_class
            return None

        mock_registry.find_config_class.side_effect = find_config_class
        mock_registry_class.return_value = mock_registry

        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        config_map = assembler._resolve_workspace_configs(sample_workspace_config)

        assert 'preprocessing' in config_map
        assert 'training' in config_map
        assert config_map['preprocessing'] == mock_preprocessing_config_instance
        assert config_map['training'] == mock_training_config_instance
        
        # Verify config classes were called with correct data
        mock_preprocessing_config_class.assert_called_once_with(
            input_path='/data/input', 
            output_path='/data/processed'
        )
        mock_training_config_class.assert_called_once_with(
            model_params={'max_depth': 6}
        )
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_resolve_workspace_configs_fallback(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test resolving workspace configs with fallback to raw data."""
        mock_registry = Mock()
        mock_registry.find_config_class.return_value = None  # No config class found
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        config_map = assembler._resolve_workspace_configs(sample_workspace_config)
        
        # Should fallback to raw config data
        assert config_map['preprocessing'] == {'input_path': '/data/input', 'output_path': '/data/processed'}
        assert config_map['training'] == {'model_params': {'max_depth': 6}}
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_resolve_workspace_builders(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test resolving workspace builders."""
        mock_registry = Mock()
        
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
        
        mock_registry.find_builder_class.side_effect = find_builder_class
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        builder_map = assembler._resolve_workspace_builders(sample_workspace_config)
        
        assert 'DataPreprocessing' in builder_map
        assert 'XGBoostTraining' in builder_map
        assert builder_map['DataPreprocessing'] == mock_preprocessing_builder
        assert builder_map['XGBoostTraining'] == mock_training_builder
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_validate_workspace_components(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test workspace component validation."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        # Mock the workspace-specific validation methods
        with patch.object(assembler, '_validate_developer_consistency') as mock_dev_validation, \
             patch.object(assembler, '_validate_step_type_consistency') as mock_type_validation:
            
            mock_dev_validation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            mock_type_validation.return_value = {'valid': True, 'errors': [], 'warnings': []}
            
            result = assembler.validate_workspace_components(sample_workspace_config)
            
            assert result['valid'] is True
            assert result['workspace_valid'] is True
            assert result['overall_valid'] is True
            assert 'workspace_validation' in result
    
    def test_validate_developer_consistency(self, temp_workspace, sample_workspace_config):
        """Test developer consistency validation."""
        with patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry'):
            assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
            result = assembler._validate_developer_consistency(sample_workspace_config)
            
            assert result['valid'] is True
            assert 'developer_stats' in result
            assert 'dev1' in result['developer_stats']
            assert 'dev2' in result['developer_stats']
            assert result['developer_stats']['dev1']['step_count'] == 1
            assert result['developer_stats']['dev2']['step_count'] == 1
    
    def test_validate_step_type_consistency(self, temp_workspace, sample_workspace_config):
        """Test step type consistency validation."""
        with patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry'):
            assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
            result = assembler._validate_step_type_consistency(sample_workspace_config)
            
            assert result['valid'] is True
            assert 'step_type_stats' in result
            assert 'DataPreprocessing' in result['step_type_stats']
            assert 'XGBoostTraining' in result['step_type_stats']
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_create_dag_from_workspace_config(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test creating DAG from workspace configuration."""
        mock_registry_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        dag = assembler._create_dag_from_workspace_config(sample_workspace_config)
        
        assert isinstance(dag, PipelineDAG)
        assert 'preprocessing' in dag.nodes
        assert 'training' in dag.nodes
        assert ('preprocessing', 'training') in dag.edges
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_assemble_workspace_pipeline_success(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test successful workspace pipeline assembly."""
        mock_registry = Mock()
        mock_registry.validate_component_availability.return_value = {
            'valid': True,
            'missing_components': [],
            'available_components': []
        }
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
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
            
            result = assembler.assemble_workspace_pipeline(sample_workspace_config)
            
            assert result == mock_pipeline
            mock_validate.assert_called_once()
            mock_resolve_configs.assert_called_once()
            mock_resolve_builders.assert_called_once()
            mock_create_dag.assert_called_once()
            mock_generate.assert_called_once_with('test_pipeline')
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_assemble_workspace_pipeline_validation_failure(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test workspace pipeline assembly with validation failure."""
        mock_registry_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        with patch.object(assembler, 'validate_workspace_components') as mock_validate:
            mock_validate.return_value = {'overall_valid': False, 'errors': ['Test error']}
            
            with pytest.raises(ValueError, match="Workspace component validation failed"):
                assembler.assemble_workspace_pipeline(sample_workspace_config)
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_from_workspace_config(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test creating assembler from workspace configuration."""
        mock_registry_class.return_value = Mock()
        
        assembler = WorkspacePipelineAssembler.from_workspace_config(
            workspace_config=sample_workspace_config,
            role='test-role'
        )
        
        assert isinstance(assembler, WorkspacePipelineAssembler)
        assert assembler.workspace_root == temp_workspace
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_from_workspace_config_file_json(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test creating assembler from JSON configuration file."""
        mock_registry_class.return_value = Mock()
        
        # Create temporary JSON file
        json_file = Path(temp_workspace) / 'config.json'
        sample_workspace_config.to_json_file(str(json_file))
        
        with patch('src.cursus.core.workspace.config.WorkspacePipelineDefinition.from_json_file') as mock_load:
            mock_load.return_value = sample_workspace_config
            
            assembler = WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path=str(json_file),
                role='test-role'
            )
            
            assert isinstance(assembler, WorkspacePipelineAssembler)
            mock_load.assert_called_once_with(str(json_file))
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_from_workspace_config_file_yaml(self, mock_registry_class, temp_workspace, sample_workspace_config):
        """Test creating assembler from YAML configuration file."""
        mock_registry_class.return_value = Mock()
        
        yaml_file = Path(temp_workspace) / 'config.yaml'
        
        with patch('src.cursus.core.workspace.config.WorkspacePipelineDefinition.from_yaml_file') as mock_load:
            mock_load.return_value = sample_workspace_config
            
            assembler = WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path=str(yaml_file),
                role='test-role'
            )
            
            assert isinstance(assembler, WorkspacePipelineAssembler)
            mock_load.assert_called_once_with(str(yaml_file))
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_from_workspace_config_file_unsupported(self, mock_registry_class, temp_workspace):
        """Test creating assembler from unsupported file format."""
        mock_registry_class.return_value = Mock()
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            WorkspacePipelineAssembler.from_workspace_config_file(
                config_file_path='/path/to/config.txt'
            )
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_get_workspace_summary(self, mock_registry_class, temp_workspace):
        """Test getting workspace summary."""
        mock_registry = Mock()
        mock_registry.get_workspace_summary.return_value = {
            'workspace_root': temp_workspace,
            'total_components': 4
        }
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        # Mock some assembler state
        assembler.dag = PipelineDAG()
        assembler.dag.add_node('test_node')
        assembler.config_map = {'step1': Mock()}
        assembler.step_builder_map = {'type1': Mock()}
        assembler.step_instances = {'step1': Mock()}
        
        summary = assembler.get_workspace_summary()
        
        assert summary['workspace_root'] == temp_workspace
        assert 'registry_summary' in summary
        assert 'assembly_status' in summary
        assert summary['assembly_status']['dag_nodes'] == 1
        assert summary['assembly_status']['config_count'] == 1
        assert summary['assembly_status']['builder_count'] == 1
        assert summary['assembly_status']['step_instances'] == 1
    
    @patch('src.cursus.core.workspace.assembler.WorkspaceComponentRegistry')
    def test_preview_workspace_assembly(self, mock_registry_class, temp_workspace, sample_workspace_config):
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
        mock_registry_class.return_value = mock_registry
        
        assembler = WorkspacePipelineAssembler(workspace_root=temp_workspace)
        
        with patch.object(assembler, 'validate_workspace_components') as mock_validate:
            mock_validate.return_value = {'valid': True}
            
            preview = assembler.preview_workspace_assembly(sample_workspace_config)
            
            assert 'workspace_config' in preview
            assert 'component_resolution' in preview
            assert 'validation_results' in preview
            assert 'assembly_plan' in preview
            
            assert preview['workspace_config']['pipeline_name'] == 'test_pipeline'
            assert preview['workspace_config']['step_count'] == 2
            
            # Check component resolution
            assert 'dev1:preprocessing' in preview['component_resolution']
            assert 'dev2:training' in preview['component_resolution']
            
            # Check assembly plan
            assert preview['assembly_plan']['dag_valid'] is True
            assert 'build_order' in preview['assembly_plan']


if __name__ == '__main__':
    pytest.main([__file__])
