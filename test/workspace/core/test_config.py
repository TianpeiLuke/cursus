"""
Unit tests for workspace configuration models.

Tests the Pydantic V2 models for workspace step definitions and pipeline configurations.
"""

import unittest
import json
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from cursus.workspace.core.config import WorkspaceStepDefinition, WorkspacePipelineDefinition

class TestWorkspaceStepDefinition(unittest.TestCase):
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
        
        self.assertEqual(step.step_name, 'test_step')
        self.assertEqual(step.developer_id, 'dev1')
        self.assertEqual(step.step_type, 'XGBoostTraining')
        self.assertEqual(step.config_data, {'param1': 'value1'})
        self.assertEqual(step.workspace_root, '/path/to/workspace')
        self.assertEqual(step.dependencies, ['dep1', 'dep2'])
    
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
        
        self.assertEqual(step.dependencies, [])  # Default empty list
    
    def test_invalid_step_name(self):
        """Test validation of step_name field."""
        step_data = {
            'step_name': '',  # Invalid empty string
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with self.assertRaises(ValueError) as context:
            WorkspaceStepDefinition(**step_data)
        
        self.assertIn("step_name must be a non-empty string", str(context.exception))
    
    def test_invalid_developer_id(self):
        """Test validation of developer_id field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': '',  # Invalid empty string
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with self.assertRaises(ValueError) as context:
            WorkspaceStepDefinition(**step_data)
        
        self.assertIn("developer_id must be a non-empty string", str(context.exception))
    
    def test_invalid_step_type(self):
        """Test validation of step_type field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': '',  # Invalid empty string
            'config_data': {'param1': 'value1'},
            'workspace_root': '/path/to/workspace'
        }
        
        with self.assertRaises(ValueError) as context:
            WorkspaceStepDefinition(**step_data)
        
        self.assertIn("step_type must be a non-empty string", str(context.exception))
    
    def test_invalid_workspace_root(self):
        """Test validation of workspace_root field."""
        step_data = {
            'step_name': 'test_step',
            'developer_id': 'dev1',
            'step_type': 'XGBoostTraining',
            'config_data': {'param1': 'value1'},
            'workspace_root': ''  # Invalid empty string
        }
        
        with self.assertRaises(ValueError) as context:
            WorkspaceStepDefinition(**step_data)
        
        self.assertIn("workspace_root must be a non-empty string", str(context.exception))
    
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
        self.assertEqual(path, '/path/to/workspace/subdir/file.txt')
        
        # Test without relative path
        path = step.get_workspace_path()
        self.assertEqual(path, '/path/to/workspace')
    
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
        self.assertEqual(dumped, step_data)
        
        # Test recreation from dumped data
        step2 = WorkspaceStepDefinition(**dumped)
        self.assertEqual(step2.step_name, step.step_name)
        self.assertEqual(step2.developer_id, step.developer_id)

class TestWorkspacePipelineDefinition(unittest.TestCase):
    """Test cases for WorkspacePipelineDefinition model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
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
        
        self.assertEqual(pipeline.pipeline_name, 'test_pipeline')
        self.assertEqual(pipeline.workspace_root, '/workspace')
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.global_config, {'global_param': 'global_value'})
    
    def test_pipeline_with_defaults(self):
        """Test creating pipeline with default values."""
        steps = self.create_sample_steps()
        
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        self.assertEqual(pipeline.global_config, {})  # Default empty dict
    
    def test_invalid_pipeline_name(self):
        """Test validation of pipeline_name field."""
        steps = self.create_sample_steps()
        
        with self.assertRaises(ValueError) as context:
            WorkspacePipelineDefinition(
                pipeline_name='',  # Invalid empty string
                workspace_root='/workspace',
                steps=steps
            )
        
        self.assertIn("pipeline_name must be a non-empty string", str(context.exception))
    
    def test_invalid_workspace_root(self):
        """Test validation of workspace_root field."""
        steps = self.create_sample_steps()
        
        with self.assertRaises(ValueError) as context:
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='',  # Invalid empty string
                steps=steps
            )
        
        self.assertIn("workspace_root must be a non-empty string", str(context.exception))
    
    def test_empty_steps_list(self):
        """Test validation of empty steps list."""
        with self.assertRaises(ValueError) as context:
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='/workspace',
                steps=[]  # Invalid empty list
            )
        
        self.assertIn("steps list cannot be empty", str(context.exception))
    
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
        
        with self.assertRaises(ValueError) as context:
            WorkspacePipelineDefinition(
                pipeline_name='test_pipeline',
                workspace_root='/workspace',
                steps=steps
            )
        
        self.assertIn("Duplicate step names found in pipeline", str(context.exception))
    
    def test_validate_workspace_dependencies_valid(self):
        """Test dependency validation with valid dependencies."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        result = pipeline.validate_workspace_dependencies()
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])
        self.assertIn('step1', result['dependency_graph'])
        self.assertIn('step2', result['dependency_graph'])
        self.assertEqual(result['dependency_graph']['step2'], ['step1'])
    
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
        
        self.assertFalse(result['valid'])
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('missing_step', result['errors'][0])
    
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
        
        self.assertFalse(result['valid'])
        self.assertTrue(any('Circular dependencies' in error for error in result['errors']))
    
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
        
        self.assertEqual(config['pipeline_name'], 'test_pipeline')
        self.assertEqual(config['workspace_root'], '/workspace')
        self.assertEqual(config['global_config'], {'global_param': 'global_value'})
        self.assertIn('steps', config)
        self.assertIn('step1', config['steps'])
        self.assertIn('step2', config['steps'])
        self.assertEqual(config['steps']['step1']['developer_id'], 'dev1')
        self.assertEqual(config['steps']['step2']['dependencies'], ['step1'])
    
    def test_get_developers(self):
        """Test getting list of developers."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        developers = pipeline.get_developers()
        
        self.assertEqual(set(developers), {'dev1', 'dev2'})
    
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
        
        self.assertEqual(len(dev1_steps), 1)
        self.assertEqual(dev1_steps[0].step_name, 'step1')
        self.assertEqual(len(dev2_steps), 1)
        self.assertEqual(dev2_steps[0].step_name, 'step2')
    
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
        
        self.assertIsNotNone(step1)
        self.assertEqual(step1.step_name, 'step1')
        self.assertEqual(step1.developer_id, 'dev1')
        self.assertIsNone(step_missing)
    
    def test_validate_with_consolidated_managers(self):
        """Test validation with consolidated managers (Phase 2 optimization)."""
        mock_manager = Mock()
        mock_lifecycle_manager = Mock()
        mock_isolation_manager = Mock()
        mock_discovery_manager = Mock()
        mock_integration_manager = Mock()
        
        # Setup manager delegation
        mock_manager.lifecycle_manager = mock_lifecycle_manager
        mock_manager.isolation_manager = mock_isolation_manager
        mock_manager.discovery_manager = mock_discovery_manager
        mock_manager.integration_manager = mock_integration_manager
        
        # Mock validation responses
        mock_lifecycle_manager.validate_pipeline_lifecycle.return_value = {
            'valid': True, 'errors': [], 'warnings': []
        }
        mock_isolation_manager.validate_pipeline_isolation.return_value = {
            'valid': True, 'errors': [], 'warnings': []
        }
        mock_discovery_manager.validate_pipeline_dependencies.return_value = {
            'valid': True, 'errors': [], 'warnings': []
        }
        mock_integration_manager.validate_pipeline_integration.return_value = {
            'valid': True, 'errors': [], 'warnings': []
        }
        
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps
        )
        
        result = pipeline.validate_with_consolidated_managers(mock_manager)
        
        self.assertIn('validations', result)
        self.assertIn('lifecycle', result['validations'])
        self.assertIn('isolation', result['validations'])
        self.assertIn('discovery', result['validations'])
        self.assertIn('integration', result['validations'])
        self.assertIn('overall_valid', result)
        self.assertTrue(result['overall_valid'])
        
        # Verify manager methods were called
        mock_lifecycle_manager.validate_pipeline_lifecycle.assert_called_once()
        mock_isolation_manager.validate_pipeline_isolation.assert_called_once()
        mock_discovery_manager.validate_pipeline_dependencies.assert_called_once()
        mock_integration_manager.validate_pipeline_integration.assert_called_once()
    
    def test_json_file_operations(self):
        """Test JSON file save and load operations."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        temp_file = Path(self.temp_dir) / 'test_config.json'
        
        # Test save to JSON
        pipeline.to_json_file(str(temp_file))
        
        # Test load from JSON
        loaded_pipeline = WorkspacePipelineDefinition.from_json_file(str(temp_file))
        
        self.assertEqual(loaded_pipeline.pipeline_name, pipeline.pipeline_name)
        self.assertEqual(loaded_pipeline.workspace_root, pipeline.workspace_root)
        self.assertEqual(len(loaded_pipeline.steps), len(pipeline.steps))
        self.assertEqual(loaded_pipeline.global_config, pipeline.global_config)
    
    def test_yaml_file_operations(self):
        """Test YAML file save and load operations."""
        steps = self.create_sample_steps()
        pipeline = WorkspacePipelineDefinition(
            pipeline_name='test_pipeline',
            workspace_root='/workspace',
            steps=steps,
            global_config={'global_param': 'global_value'}
        )
        
        temp_file = Path(self.temp_dir) / 'test_config.yaml'
        
        # Test save to YAML
        pipeline.to_yaml_file(str(temp_file))
        
        # Test load from YAML
        loaded_pipeline = WorkspacePipelineDefinition.from_yaml_file(str(temp_file))
        
        self.assertEqual(loaded_pipeline.pipeline_name, pipeline.pipeline_name)
        self.assertEqual(loaded_pipeline.workspace_root, pipeline.workspace_root)
        self.assertEqual(len(loaded_pipeline.steps), len(pipeline.steps))
        self.assertEqual(loaded_pipeline.global_config, pipeline.global_config)

if __name__ == '__main__':
    unittest.main()
