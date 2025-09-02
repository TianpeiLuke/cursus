"""
Unit tests for workspace configuration models.

Tests the Pydantic V2 models for workspace step definitions and pipeline configurations.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.cursus.core.workspace.config import WorkspaceStepDefinition, WorkspacePipelineDefinition


class TestWorkspaceStepDefinition:
    """Test cases for WorkspaceStepDefinition model."""
    
    def test_valid_step_definition(self):
        """Test creating a valid workspace step definition."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace',
            'dependencies': ['dep1', 'dep2']
        }
        
        step = WorkspaceStepDefinition(**step_data)
        
        assert step.step_name == 'test_step'
        assert step.developer_id == 'dev1'
        assert step.step_type == 'XGBoostTraining'
        assert step.config_data == {'param1': 'value1'}
        assert step.workspace_root == '/path/to/workspace'
        assert step.dependencies == ['dep1', 'dep2']
    
    def test_step_definition_with_defaults(self):
        """Test creating step definition with default values."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        step = WorkspaceStepDefinition(**step_data)
        
        assert step.dependencies == []  # Default empty list
    
    def test_invalid_step_name(self):
        """Test validation of step_name field."""
        step_data = {
            'step_name': '',  # Invalid empty string
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with pytest.raises(ValueError, match="step_name must be a non-empty string"):
            WorkspaceStepDefinition(**step_data)
    
    def test_invalid_developer_id(self):
        """Test validation of developer_id field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': '',  # Invalid empty string
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with pytest.raises(ValueError, match="developer_id must be a non-empty string"):
            WorkspaceStepDefinition(**step_data)
    
    def test_invalid_step_type(self):
        """Test validation of step_type field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': '',  # Invalid empty string
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with pytest.raises(ValueError, match="step_type must be a non-empty string"):
            WorkspaceStepDefinition(**step_data)
    
    def test_invalid_workspace_root(self):
        """Test validation of workspace_root field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': ''  # Invalid empty string
        }
        
        with pytest.raises(ValueError, match="workspace_root must be a non-empty string"):
            WorkspaceStepDefinition(**step_data)
    
    def test_get_workspace_path(self):
        """Test get_workspace_path method."""
        step = WorkspaceStepDefinition(
            step_name='test_step',
            developer_id='dev1',
            step_type='XGBoostTraining',
            config_data={'param1': 'value1'},
            workspace_root='/path/to/workspace'
        )
        
        # Test with relative path
        path = step.get_workspace_path('subdir/file.txt')
        assert path == '/path/to/workspace/subdir/file.txt'
        
        # Test without relative path
        path = step.get_workspace_path()
        assert path == '/path/to/workspace'
    
    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace',
            'dependencies': ['dep1']
        }
        
        step = WorkspaceStepDefinition(**step_data)
        
        # Test model_dump
        dumped = step.model_dump()
        assert dumped == step_data
        
        # Test recreation from dumped data
        step2 = WorkspaceStepDefinition(**dumped)
        assert step2.step_name == step.step_name
        assert step2.developer_id == step.developer_id


class TestWorkspacePipelineDefinition:
    """Test cases for WorkspacePipelineDefinition model."""
    
    def create_sample_steps(self) -> list:
        """Create sample steps for testing."""
        return [
            WorkspaceStepDefinition(
                step_name='step1',
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'param1': 'value1'},
                workspace_root='/workspace'
            ),
            WorkspaceStepDefinition(
                step_name='step2',
                developer_id='dev2',
                step_type='XGBoostTraining',
                config_data={'param2': 'value2'},
                workspace_root='/workspace',
                dependencies=['step1']
            )
        ]
    
    def test_valid_pipeline_definition(self):
        """Test creating a valid workspace pipeline definition."""
        steps = self.create_sample_steps()
        
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        assert pipeline.pipeline_name == 'test_pipeline'
        assert pipeline.workspace_root == '/workspace'
        assert len(pipeline.steps) == 2
        assert pipeline.global_config == {'global_param': 'global_value'}
    
    def test_pipeline_with_defaults(self):
        """Test creating pipeline with default values."""
        steps = self.create_sample_steps()
        
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        assert pipeline.global_config == {}  # Default empty dict
    
    def test_invalid_pipeline_name(self):
        """Test validation of pipeline_name field."""
        steps = self.create_sample_steps()
        
        with pytest.raises(ValueError, match="pipeline_name must be a non-empty string"):
            WorkspacePipelineDefinition(
                pipeline_name='',  # Invalid empty string
                workspace_root='/workspace',
                steps=steps
            )
    
    def test_invalid_workspace_root(self):
        """Test validation of workspace_root field."""
        steps = self.create_sample_steps()
        
        with pytest.raises(ValueError, match="workspace_root must be a non-empty string"):
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='',  # Invalid empty string
                steps=steps
            )
    
    def test_empty_steps_list(self):
        """Test validation of empty steps list."""
        with pytest.raises(ValueError, match="steps list cannot be empty"):
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='/workspace',
                steps=[]  # Invalid empty list
            )
    
    def test_duplicate_step_names(self):
        """Test validation of duplicate step names."""
        steps = [
            WorkspaceStepDefinition(
                step_name='duplicate_step',  # Same name
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'param1': 'value1'},
                workspace_root='/workspace'
            ),
            WorkspaceStepDefinition(
                step_name='duplicate_step',  # Same name
                developer_id='dev2',
                step_type='XGBoostTraining',
                config_data={'param2': 'value2'},
                workspace_root='/workspace'
            )
        ]
        
        with pytest.raises(ValueError, match="Duplicate step names found in pipeline"):
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='/workspace',
                steps=steps
            )
    
    def test_validate_workspace_dependencies_valid(self):
        """Test dependency validation with valid dependencies."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        result = pipeline.validate_workspace_dependencies()
        
        assert result['valid'] is True
        assert result['errors'] == []
        assert 'step1' in result['dependency_graph']
        assert 'step2' in result['dependency_graph']
        assert result['dependency_graph']['step2'] == ['step1']
    
    def test_validate_workspace_dependencies_missing(self):
        """Test dependency validation with missing dependencies."""
        steps = [
            WorkspaceStepDefinition(
                step_name='step1',
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'param1': 'value1'},
                workspace_root='/workspace',
                dependencies=['missing_step']  # Missing dependency
            )
        ]
        
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        result = pipeline.validate_workspace_dependencies()
        
        assert result['valid'] is False
        assert len(result['errors']) == 1
        assert 'missing_step' in result['errors'][0]
    
    def test_validate_circular_dependencies(self):
        """Test detection of circular dependencies."""
        steps = [
            WorkspaceStepDefinition(
                step_name='step1',
                developer_id='dev1',
                step_type='DataPreprocessing',
                config_data={'param1': 'value1'},
                workspace_root='/workspace',
                dependencies=['step2']  # Circular dependency
            ),
            WorkspaceStepDefinition(
                step_name='step2',
                developer_id='dev2',
                step_type='XGBoostTraining',
                config_data={'param2': 'value2'},
                workspace_root='/workspace',
                dependencies=['step1']  # Circular dependency
            )
        ]
        
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        result = pipeline.validate_workspace_dependencies()
        
        assert result['valid'] is False
        assert any('Circular dependencies' in error for error in result['errors'])
    
    def test_to_pipeline_config(self):
        """Test conversion to pipeline configuration format."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        config = pipeline.to_pipeline_config()
        
        assert config['pipeline_name'] == 'test_pipeline'
        assert config['workspace_root'] == '/workspace'
        assert config['global_config'] == {'global_param': 'global_value'}
        assert 'steps' in config
        assert 'step1' in config['steps']
        assert 'step2' in config['steps']
        assert config['steps']['step1']['developer_id'] == 'dev1'
        assert config['steps']['step2']['dependencies'] == ['step1']
    
    def test_get_developers(self):
        """Test getting list of developers."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        developers = pipeline.get_developers()
        
        assert set(developers) == {'dev1', 'dev2'}
    
    def test_get_steps_by_developer(self):
        """Test getting steps by developer."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        dev1_steps = pipeline.get_steps_by_developer('dev1')
        dev2_steps = pipeline.get_steps_by_developer('dev2')
        
        assert len(dev1_steps) == 1
        assert dev1_steps[0].step_name == 'step1'
        assert len(dev2_steps) == 1
        assert dev2_steps[0].step_name == 'step2'
    
    def test_get_step_by_name(self):
        """Test getting step by name."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        step1 = pipeline.get_step_by_name('step1')
        step_missing = pipeline.get_step_by_name('missing_step')
        
        assert step1 is not None
        assert step1.step_name == 'step1'
        assert step1.developer_id == 'dev1'
        assert step_missing is None
    
    def test_json_file_operations(self):
        """Test JSON file save and load operations."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test save to JSON
            pipeline.to_json_file(temp_file)
            
            # Test load from JSON
            loaded_pipeline = WorkspacePipelineDefinition.from_json_file(temp_file)
            
            assert loaded_pipeline.pipeline_name == pipeline.pipeline_name
            assert loaded_pipeline.workspace_root == pipeline.workspace_root
            assert len(loaded_pipeline.steps) == len(pipeline.steps)
            assert loaded_pipeline.global_config == pipeline.global_config
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_yaml_file_operations(self):
        """Test YAML file save and load operations."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            # Test save to YAML
            pipeline.to_yaml_file(temp_file)
            
            # Test load from YAML
            loaded_pipeline = WorkspacePipelineDefinition.from_yaml_file(temp_file)
            
            assert loaded_pipeline.pipeline_name == pipeline.pipeline_name
            assert loaded_pipeline.workspace_root == pipeline.workspace_root
            assert len(loaded_pipeline.steps) == len(pipeline.steps)
            assert loaded_pipeline.global_config == pipeline.global_config
            
        finally:
            Path(temp_file).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__])
