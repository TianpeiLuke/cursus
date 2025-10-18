"""
Comprehensive tests for Script Testing API.

Tests validate:
- Main API functions with proper error handling
- Script execution workflow and dependency management
- Config-based input extraction and validation
- Package dependency installation and management
- Integration with existing cursus infrastructure

Following pytest best practices:
- Read source code first to understand actual implementation behavior
- Set expected responses correctly based on actual source script
- Use proper mock structure without Mock(spec=...) issues
- Test all functions systematically with realistic scenarios
- Examine possible failure patterns before running tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import argparse
import ast
import subprocess
import sys
from typing import Dict, Any

# Import the module under test
from cursus.validation.script_testing.api import (
    run_dag_scripts,
    collect_script_inputs_using_dag_factory,
    get_validated_scripts_from_config,
    collect_script_inputs,
    extract_script_path_from_config,
    extract_environment_variables_from_config,
    extract_job_arguments_from_config,
    execute_scripts_in_order,
    execute_single_script,
    install_script_dependencies,
    parse_script_imports,
    is_package_installed,
    install_package,
    import_and_execute_script,
    execute_scripts_with_registry_coordination,
    ScriptTestResult
)

# Import resolve_script_dependencies from the correct module
from cursus.validation.script_testing.script_dependency_matcher import (
    resolve_script_dependencies,
    resolve_script_dependencies_with_registry
)

# Import dependencies for mocking
from cursus.api.dag.base_dag import PipelineDAG
from cursus.step_catalog import StepCatalog


class TestScriptTestResult:
    """Test ScriptTestResult data class."""
    
    def test_script_test_result_success(self):
        """Test successful result creation."""
        result = ScriptTestResult(
            success=True,
            output_files={'model': '/output/model.pkl'},
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.output_files == {'model': '/output/model.pkl'}
        assert result.error_message is None
        assert result.execution_time == 1.5
    
    def test_script_test_result_failure(self):
        """Test failure result creation."""
        result = ScriptTestResult(
            success=False,
            error_message="Script execution failed"
        )
        
        assert result.success is False
        assert result.output_files == {}
        assert result.error_message == "Script execution failed"
        assert result.execution_time is None
    
    def test_script_test_result_defaults(self):
        """Test default values."""
        result = ScriptTestResult(success=True)
        
        assert result.success is True
        assert result.output_files == {}
        assert result.error_message is None
        assert result.execution_time is None


class TestRunDagScripts:
    """Test main API entry point."""
    
    @patch('cursus.validation.script_testing.api.Path')
    @patch('cursus.validation.script_testing.api.StepCatalog')
    @patch('cursus.validation.script_testing.script_dependency_matcher.resolve_script_dependencies_with_registry')
    @patch('cursus.validation.script_testing.api.execute_scripts_with_registry_coordination')
    def test_run_dag_scripts_with_dependency_resolution(self, mock_execute_registry, mock_resolve_registry, mock_step_catalog, mock_path):
        """Test main function with dependency resolution enabled (registry approach)."""
        # Create mock DAG that passes isinstance check
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['DataPreprocessing', 'XGBoostTraining']
        mock_dag.topological_sort.return_value = ['DataPreprocessing', 'XGBoostTraining']
        
        # Setup mocks
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = Mock()
        
        mock_step_catalog_instance = Mock()
        mock_step_catalog.return_value = mock_step_catalog_instance
        
        # Mock registry creation
        with patch('cursus.validation.script_testing.script_execution_registry.create_script_execution_registry') as mock_create_registry:
            mock_registry = Mock()
            mock_registry.get_execution_summary.return_value = {'completed_scripts': ['DataPreprocessing', 'XGBoostTraining']}
            mock_registry.get_message_passing_history.return_value = []
            mock_create_registry.return_value = mock_registry
            
            # Mock registry-coordinated dependency resolution
            mock_resolve_registry.return_value = {
                'DataPreprocessing': {
                    'input_paths': {'raw_data': '/data/raw.csv'}, 
                    'output_paths': {'processed_data': '/data/processed.csv'},
                    'script_path': '/scripts/preprocess.py'
                },
                'XGBoostTraining': {
                    'input_paths': {'training_data': '/data/processed.csv'}, 
                    'output_paths': {'model': '/models/model.pkl'},
                    'script_path': '/scripts/train.py'
                }
            }
            
            # Mock registry execution with successful results
            mock_execute_registry.return_value = {
                'pipeline_success': True,
                'script_results': {
                    'DataPreprocessing': ScriptTestResult(success=True),
                    'XGBoostTraining': ScriptTestResult(success=True)
                },
                'execution_order': ['DataPreprocessing', 'XGBoostTraining'],
                'total_scripts': 2,
                'successful_scripts': 2
            }
            
            # Execute test
            result = run_dag_scripts(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                test_workspace_dir='/test/workspace'
            )
            
            # Validate results
            assert result['pipeline_success'] is True
            assert result['total_scripts'] == 2
            assert result['successful_scripts'] == 2
            assert 'execution_summary' in result
            assert 'message_passing_history' in result
            
            # Validate calls
            mock_path.assert_called_once_with('/test/workspace')
            mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_step_catalog.assert_called_once()
            mock_create_registry.assert_called_once_with(mock_dag, mock_step_catalog_instance)
            mock_resolve_registry.assert_called_once_with(mock_dag, '/config/pipeline.json', mock_step_catalog_instance, mock_registry)
            mock_execute_registry.assert_called_once_with(mock_dag, mock_registry)
    
    @patch('cursus.validation.script_testing.api.Path')
    @patch('cursus.validation.script_testing.api.StepCatalog')
    @patch('cursus.validation.script_testing.script_dependency_matcher.prepare_script_testing_inputs')
    @patch('cursus.validation.script_testing.script_dependency_matcher.collect_user_inputs_with_registry_coordination')
    @patch('cursus.validation.script_testing.api.execute_scripts_with_registry_coordination')
    def test_run_dag_scripts_without_dependency_resolution(self, mock_execute_registry, mock_collect_registry, mock_prepare, mock_step_catalog, mock_path):
        """Test main function with dependency resolution disabled (registry-only approach)."""
        # Create mock DAG that passes isinstance check
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['DataPreprocessing']
        mock_dag.topological_sort.return_value = ['DataPreprocessing']
        
        # Setup mocks
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = Mock()
        
        mock_step_catalog_instance = Mock()
        mock_step_catalog.return_value = mock_step_catalog_instance
        
        # Mock registry creation - patch the actual import location
        with patch('cursus.validation.script_testing.script_execution_registry.create_script_execution_registry') as mock_create_registry:
            mock_registry = Mock()
            mock_registry.get_execution_summary.return_value = {'completed_scripts': ['DataPreprocessing']}
            mock_registry.get_message_passing_history.return_value = []
            mock_create_registry.return_value = mock_registry
            
            # Mock prepare_script_testing_inputs
            mock_prepared_data = {
                'node_specs': {},
                'dependency_matches': {'DataPreprocessing': {}},
                'config_data': {},
                'execution_order': ['DataPreprocessing']
            }
            mock_prepare.return_value = mock_prepared_data
            
            # Mock registry coordination input collection
            mock_collect_registry.return_value = {
                'DataPreprocessing': {'input_paths': {}, 'output_paths': {}}
            }
            
            # Mock registry execution
            mock_execute_registry.return_value = {
                'pipeline_success': True,
                'script_results': {},
                'execution_order': ['DataPreprocessing'],
                'total_scripts': 1,
                'successful_scripts': 1
            }
            
            # Execute test
            result = run_dag_scripts(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                use_dependency_resolution=False
            )
            
            # Validate results
            assert result['pipeline_success'] is True
            assert result['total_scripts'] == 1
            assert result['successful_scripts'] == 1
            assert 'execution_summary' in result
            assert 'message_passing_history' in result
            
            # Validate calls
            mock_path.assert_called_once_with('test/integration/script_testing')
            mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_step_catalog.assert_called_once()
            mock_create_registry.assert_called_once_with(mock_dag, mock_step_catalog_instance)
            mock_prepare.assert_called_once_with(mock_dag, '/config/pipeline.json', mock_step_catalog_instance)
            mock_registry.initialize_from_dependency_matcher.assert_called_once()
            mock_collect_registry.assert_called_once()
            mock_execute_registry.assert_called_once_with(mock_dag, mock_registry)
    
    def test_run_dag_scripts_invalid_dag(self):
        """Test with invalid DAG input."""
        with pytest.raises(RuntimeError, match="Failed to test DAG scripts"):
            run_dag_scripts(
                dag="not_a_dag",
                config_path='/config/pipeline.json'
            )
    
    def test_run_dag_scripts_empty_dag(self):
        """Test with empty DAG."""
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = []
        
        with pytest.raises(RuntimeError, match="Failed to test DAG scripts"):
            run_dag_scripts(
                dag=mock_dag,
                config_path='/config/pipeline.json'
            )
    
    @patch('cursus.validation.script_testing.api.Path')
    @patch('cursus.validation.script_testing.api.StepCatalog')
    @patch('cursus.validation.script_testing.script_dependency_matcher.resolve_script_dependencies')
    def test_run_dag_scripts_with_exception(self, mock_resolve, mock_step_catalog, mock_path):
        """Test exception handling."""
        # Create mock DAG that passes isinstance check
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['DataPreprocessing']
        
        # Setup mocks
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.mkdir = Mock()
        
        mock_step_catalog_instance = Mock()
        mock_step_catalog.return_value = mock_step_catalog_instance
        mock_resolve.side_effect = Exception("Dependency resolution failed")
        
        # Execute test and expect RuntimeError
        with pytest.raises(RuntimeError, match="Failed to test DAG scripts"):
            run_dag_scripts(
                dag=mock_dag,
                config_path='/config/pipeline.json'
            )


class TestCollectScriptInputsUsingDagFactory:
    """Test script input collection using DAG factory patterns."""
    
    @patch('cursus.validation.script_testing.api.DAGConfigFactory')
    @patch('cursus.validation.script_testing.api.build_complete_config_classes')
    @patch('cursus.validation.script_testing.api.load_configs')
    @patch('cursus.validation.script_testing.api.get_validated_scripts_from_config')
    @patch('cursus.validation.script_testing.api.collect_script_inputs')
    def test_collect_script_inputs_success(self, mock_collect, mock_get_validated, mock_load_configs, 
                                         mock_build_classes, mock_dag_factory):
        """Test successful script input collection."""
        # Create mock DAG
        mock_dag = Mock()
        
        # Setup mocks
        mock_dag_factory_instance = Mock()
        mock_dag_factory.return_value = mock_dag_factory_instance
        
        mock_build_classes.return_value = {'DataPreprocessing': Mock()}
        mock_load_configs.return_value = {'DataPreprocessing': Mock()}
        mock_get_validated.return_value = ['DataPreprocessing']
        mock_collect.return_value = {
            'script_path': '/scripts/preprocess.py',
            'environment_variables': {'ENV': 'test'},
            'job_arguments': Mock()
        }
        
        # Execute test
        result = collect_script_inputs_using_dag_factory(mock_dag, '/config/pipeline.json')
        
        # Validate results
        assert 'DataPreprocessing' in result
        assert result['DataPreprocessing']['script_path'] == '/scripts/preprocess.py'
        
        # Validate calls
        mock_dag_factory.assert_called_once_with(mock_dag)
        mock_build_classes.assert_called_once()
        mock_load_configs.assert_called_once()
        mock_get_validated.assert_called_once()
        mock_collect.assert_called_once()
    
    @patch('cursus.validation.script_testing.api.DAGConfigFactory')
    @patch('cursus.validation.script_testing.api.build_complete_config_classes')
    def test_collect_script_inputs_with_exception(self, mock_build_classes, mock_dag_factory):
        """Test exception handling."""
        mock_dag = Mock()
        mock_dag_factory.side_effect = Exception("DAG factory failed")
        
        with pytest.raises(ValueError, match="Input collection failed"):
            collect_script_inputs_using_dag_factory(mock_dag, '/config/pipeline.json')


class TestGetValidatedScriptsFromConfig:
    """Test script validation from config."""
    
    def test_get_validated_scripts_with_training_entry_point(self):
        """Test validation with training entry point."""
        mock_dag = Mock()
        mock_dag.nodes = ['DataPreprocessing', 'XGBoostTraining']
        
        # Create mock configs with proper hasattr behavior
        mock_config1 = Mock()
        mock_config1.training_entry_point = 'train.py'
        
        mock_config2 = Mock()
        # Remove attributes to simulate hasattr() returning False
        del mock_config2.training_entry_point
        del mock_config2.inference_entry_point
        del mock_config2.source_dir
        del mock_config2.entry_point
        
        configs = {
            'DataPreprocessing': mock_config1,
            'XGBoostTraining': mock_config2
        }
        
        # Execute test
        result = get_validated_scripts_from_config(mock_dag, configs)
        
        # Validate results - only DataPreprocessing has entry point
        assert result == ['DataPreprocessing']
    
    def test_get_validated_scripts_with_inference_entry_point(self):
        """Test validation with inference entry point."""
        mock_dag = Mock()
        mock_dag.nodes = ['ModelInference']
        
        mock_config = Mock()
        mock_config.inference_entry_point = 'inference.py'
        
        configs = {'ModelInference': mock_config}
        
        # Execute test
        result = get_validated_scripts_from_config(mock_dag, configs)
        
        # Validate results
        assert result == ['ModelInference']
    
    def test_get_validated_scripts_with_source_dir_entry_point(self):
        """Test validation with source_dir and entry_point."""
        mock_dag = Mock()
        mock_dag.nodes = ['CustomScript']
        
        mock_config = Mock()
        mock_config.source_dir = '/src'
        mock_config.entry_point = 'main.py'
        
        configs = {'CustomScript': mock_config}
        
        # Execute test
        result = get_validated_scripts_from_config(mock_dag, configs)
        
        # Validate results
        assert result == ['CustomScript']
    
    def test_get_validated_scripts_no_entry_points(self):
        """Test validation with no entry points."""
        mock_dag = Mock()
        mock_dag.nodes = ['NoEntryPoint']
        
        mock_config = Mock()
        # Remove attributes to simulate hasattr() returning False
        del mock_config.training_entry_point
        del mock_config.inference_entry_point
        del mock_config.source_dir
        del mock_config.entry_point
        
        configs = {'NoEntryPoint': mock_config}
        
        # Execute test
        result = get_validated_scripts_from_config(mock_dag, configs)
        
        # Validate results - should be empty
        assert result == []


class TestCollectScriptInputs:
    """Test script input collection from config."""
    
    @patch('cursus.validation.script_testing.api.extract_script_path_from_config')
    @patch('cursus.validation.script_testing.api.extract_environment_variables_from_config')
    @patch('cursus.validation.script_testing.api.extract_job_arguments_from_config')
    def test_collect_script_inputs_success(self, mock_extract_job_args, mock_extract_env_vars, mock_extract_script_path):
        """Test successful script input collection."""
        mock_config = Mock()
        
        # Setup mocks
        mock_extract_script_path.return_value = '/scripts/train.py'
        mock_extract_env_vars.return_value = {'ENV': 'test'}
        mock_extract_job_args.return_value = Mock()
        
        # Execute test
        result = collect_script_inputs(mock_config)
        
        # Validate results
        assert result['script_path'] == '/scripts/train.py'
        assert result['environment_variables'] == {'ENV': 'test'}
        assert result['job_arguments'] is not None
        
        # Validate calls
        mock_extract_script_path.assert_called_once_with(mock_config)
        mock_extract_env_vars.assert_called_once_with(mock_config)
        mock_extract_job_args.assert_called_once_with(mock_config)


class TestExtractScriptPathFromConfig:
    """Test script path extraction from config."""
    
    @patch('os.path.exists')
    def test_extract_script_path_training_entry_point(self, mock_exists):
        """Test extraction with training_entry_point."""
        mock_config = Mock()
        mock_config.training_entry_point = 'train.py'
        mock_config.effective_source_dir = '/src'
        # Remove other entry point attributes to ensure training_entry_point is used first
        del mock_config.inference_entry_point
        del mock_config.entry_point
        # Remove resolve_hybrid_path to prevent it from being called
        del mock_config.resolve_hybrid_path
        
        mock_exists.return_value = True
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results - based on actual implementation behavior
        assert result == '/src/train.py'
        mock_exists.assert_called_with('/src/train.py')
    
    @patch('os.path.exists')
    def test_extract_script_path_inference_entry_point(self, mock_exists):
        """Test extraction with inference_entry_point."""
        mock_config = Mock()
        mock_config.inference_entry_point = 'inference.py'
        mock_config.effective_source_dir = '/src'
        # Remove training_entry_point to ensure inference_entry_point is used
        del mock_config.training_entry_point
        del mock_config.entry_point
        # Remove resolve_hybrid_path to prevent it from being called
        del mock_config.resolve_hybrid_path
        
        mock_exists.return_value = True
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results
        assert result == '/src/inference.py'
    
    @patch('os.path.exists')
    def test_extract_script_path_with_hybrid_resolution(self, mock_exists):
        """Test extraction with hybrid path resolution."""
        mock_config = Mock()
        mock_config.entry_point = 'main.py'
        mock_config.effective_source_dir = '/src'
        mock_config.resolve_hybrid_path.return_value = '/resolved/main.py'
        # Remove other entry point attributes to ensure entry_point is used
        del mock_config.training_entry_point
        del mock_config.inference_entry_point
        
        mock_exists.side_effect = lambda path: path == '/resolved/main.py'
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results
        assert result == '/resolved/main.py'
        mock_config.resolve_hybrid_path.assert_called_once_with('/src/main.py')
    
    @patch('os.path.exists')
    def test_extract_script_path_hybrid_resolution_fails(self, mock_exists):
        """Test extraction when hybrid resolution fails."""
        mock_config = Mock()
        mock_config.entry_point = 'main.py'
        mock_config.effective_source_dir = '/src'
        mock_config.resolve_hybrid_path.side_effect = AttributeError("project_root_folder missing")
        # Remove other entry point attributes to ensure entry_point is used
        del mock_config.training_entry_point
        del mock_config.inference_entry_point
        
        mock_exists.return_value = True
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results - should fallback to direct path
        assert result == '/src/main.py'
    
    @patch('os.path.exists')
    def test_extract_script_path_file_not_exists(self, mock_exists):
        """Test extraction when file doesn't exist."""
        mock_config = Mock()
        mock_config.training_entry_point = 'train.py'
        mock_config.effective_source_dir = '/src'
        
        mock_exists.return_value = False
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results - should return path even if doesn't exist
        assert result == '/src/train.py'
    
    def test_extract_script_path_no_entry_point(self):
        """Test extraction with no entry point."""
        mock_config = Mock()
        # Remove all entry point attributes to simulate hasattr() returning False
        del mock_config.training_entry_point
        del mock_config.inference_entry_point
        del mock_config.entry_point
        
        # Execute test
        result = extract_script_path_from_config(mock_config)
        
        # Validate results
        assert result is None


class TestExtractEnvironmentVariablesFromConfig:
    """Test environment variable extraction from config."""
    
    def test_extract_environment_variables_with_model_dump(self):
        """Test extraction using model_dump()."""
        mock_config = Mock()
        mock_config.model_dump.return_value = {
            'framework_version': '1.0.0',
            'py_version': '3.8',
            'region': 'us-west-2'
        }
        mock_config.pipeline_name = 'test-pipeline'
        
        # Execute test
        result = extract_environment_variables_from_config(mock_config)
        
        # Validate results
        assert result['FRAMEWORK_VERSION'] == '1.0.0'
        assert result['PY_VERSION'] == '3.8'
        assert result['REGION'] == 'us-west-2'
        assert result['PIPELINE_NAME'] == 'test-pipeline'
    
    def test_extract_environment_variables_model_dump_fails(self):
        """Test extraction when model_dump() fails."""
        mock_config = Mock()
        mock_config.model_dump.side_effect = AttributeError("model_dump failed")
        mock_config.framework_version = '1.0.0'
        mock_config.region = 'us-west-2'
        
        # Execute test
        result = extract_environment_variables_from_config(mock_config)
        
        # Validate results - should use direct attribute access
        assert result['FRAMEWORK_VERSION'] == '1.0.0'
        assert result['REGION'] == 'us-west-2'
    
    def test_extract_environment_variables_no_attributes(self):
        """Test extraction with no relevant attributes."""
        # Create a mock that properly simulates hasattr() returning False
        mock_config = Mock()
        mock_config.model_dump.return_value = {}
        
        # Remove all attributes that the function checks for
        for attr in ['framework_version', 'py_version', 'region', 'aws_region', 
                     'model_class', 'service_name', 'author', 'bucket', 'role',
                     'pipeline_name', 'pipeline_s3_loc']:
            if hasattr(mock_config, attr):
                delattr(mock_config, attr)

        # Execute test
        result = extract_environment_variables_from_config(mock_config)

        # Validate results - should be empty since hasattr() will return False
        assert result == {}


class TestExtractJobArgumentsFromConfig:
    """Test job arguments extraction from config."""
    
    def test_extract_job_arguments_success(self):
        """Test successful job arguments extraction."""
        mock_config = Mock()
        mock_config.training_instance_type = 'ml.m5.large'
        mock_config.training_instance_count = 1
        mock_config.training_volume_size = 30
        mock_config.framework_version = '1.0.0'
        mock_config.py_version = '3.8'
        mock_config.job_type = 'training'
        
        # Execute test
        result = extract_job_arguments_from_config(mock_config)
        
        # Validate results
        assert isinstance(result, argparse.Namespace)
        assert result.instance_type == 'ml.m5.large'
        assert result.instance_count == 1
        assert result.volume_size == 30
        assert result.framework_version == '1.0.0'
        assert result.py_version == '3.8'
        assert result.job_type == 'training'
    
    def test_extract_job_arguments_missing_attributes(self):
        """Test extraction with missing attributes."""
        mock_config = Mock()
        mock_config.training_instance_type = 'ml.m5.large'
        # Remove other attributes to simulate hasattr() returning False
        del mock_config.training_instance_count
        del mock_config.training_volume_size
        del mock_config.framework_version
        del mock_config.py_version
        del mock_config.job_type
        
        # Execute test
        result = extract_job_arguments_from_config(mock_config)
        
        # Validate results
        assert isinstance(result, argparse.Namespace)
        assert result.instance_type == 'ml.m5.large'
        assert result.job_type == 'training'  # Default value
    
    def test_extract_job_arguments_no_attributes(self):
        """Test extraction with no relevant attributes."""
        mock_config = Mock()
        # Remove all relevant attributes to simulate hasattr() returning False
        del mock_config.training_instance_type
        del mock_config.training_instance_count
        del mock_config.training_volume_size
        del mock_config.framework_version
        del mock_config.py_version
        del mock_config.job_type
        
        # Execute test
        result = extract_job_arguments_from_config(mock_config)
        
        # Validate results
        assert isinstance(result, argparse.Namespace)
        assert result.job_type == 'training'  # Default value


class TestExecuteScriptsInOrder:
    """Test script execution in order."""
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_in_order_success(self, mock_execute_single):
        """Test successful script execution."""
        execution_order = ['DataPreprocessing', 'XGBoostTraining']
        user_inputs = {
            'DataPreprocessing': {
                'input_paths': {'raw_data': '/data/raw.csv'},
                'output_paths': {'processed_data': '/data/processed.csv'},
                'environment_variables': {'ENV': 'test'},
                'job_arguments': Mock(),
                'script_path': '/scripts/preprocess.py'
            },
            'XGBoostTraining': {
                'input_paths': {'training_data': '/data/processed.csv'},
                'output_paths': {'model': '/models/model.pkl'},
                'environment_variables': {'ENV': 'test'},
                'job_arguments': Mock(),
                'script_path': '/scripts/train.py'
            }
        }
        
        # Setup mock
        mock_execute_single.return_value = ScriptTestResult(success=True)
        
        # Execute test
        result = execute_scripts_in_order(execution_order, user_inputs)
        
        # Validate results
        assert result['pipeline_success'] is True
        assert result['total_scripts'] == 2
        assert result['successful_scripts'] == 2
        assert result['execution_order'] == execution_order
        assert len(result['script_results']) == 2
        
        # Validate calls
        assert mock_execute_single.call_count == 2
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_in_order_with_failure(self, mock_execute_single):
        """Test script execution with one failure."""
        execution_order = ['DataPreprocessing', 'XGBoostTraining']
        user_inputs = {
            'DataPreprocessing': {
                'script_path': '/scripts/preprocess.py',
                'input_paths': {},
                'output_paths': {},
                'environment_variables': {},
                'job_arguments': Mock()
            },
            'XGBoostTraining': {
                'script_path': '/scripts/train.py',
                'input_paths': {},
                'output_paths': {},
                'environment_variables': {},
                'job_arguments': Mock()
            }
        }
        
        # Setup mock - first succeeds, second fails
        mock_execute_single.side_effect = [
            ScriptTestResult(success=True),
            ScriptTestResult(success=False, error_message="Training failed")
        ]
        
        # Execute test
        result = execute_scripts_in_order(execution_order, user_inputs)
        
        # Validate results
        assert result['pipeline_success'] is False
        assert result['total_scripts'] == 2
        assert result['successful_scripts'] == 1
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_in_order_missing_script_path(self, mock_execute_single):
        """Test script execution with missing script path."""
        execution_order = ['DataPreprocessing']
        user_inputs = {
            'DataPreprocessing': {
                'input_paths': {},
                'output_paths': {},
                'environment_variables': {},
                'job_arguments': Mock()
                # No script_path
            }
        }
        
        # Execute test
        result = execute_scripts_in_order(execution_order, user_inputs)
        
        # Validate results - script should be skipped
        assert result['pipeline_success'] is True  # No scripts executed, so no failures
        assert result['total_scripts'] == 1
        assert result['successful_scripts'] == 0
        
        # Validate execute_single_script was not called
        mock_execute_single.assert_not_called()
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_in_order_with_exception(self, mock_execute_single):
        """Test script execution with exception."""
        execution_order = ['DataPreprocessing']
        user_inputs = {
            'DataPreprocessing': {
                'script_path': '/scripts/preprocess.py',
                'input_paths': {},
                'output_paths': {},
                'environment_variables': {},
                'job_arguments': Mock()
            }
        }
        
        # Setup mock to raise exception
        mock_execute_single.side_effect = Exception("Execution failed")
        
        # Execute test
        result = execute_scripts_in_order(execution_order, user_inputs)
        
        # Validate results
        assert result['pipeline_success'] is False
        assert result['total_scripts'] == 1
        assert result['successful_scripts'] == 0
        
        # Check that error was captured
        script_result = result['script_results']['DataPreprocessing']
        assert script_result.success is False
        assert "Execution failed" in script_result.error_message


class TestResolveScriptDependencies:
    """Test script dependency resolution."""
    
    def test_resolve_script_dependencies_success(self):
        """Test successful dependency resolution."""
        # Create proper mock DAG
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['DataPreprocessing', 'XGBoostTraining']
        mock_dag.topological_sort.return_value = ['DataPreprocessing', 'XGBoostTraining']
        
        config_path = '/config/pipeline.json'
        step_catalog = Mock()
        
        # Mock the prepare_script_testing_inputs function
        with patch('cursus.validation.script_testing.script_dependency_matcher.prepare_script_testing_inputs') as mock_prepare:
            mock_prepare.return_value = {
                'node_specs': {},
                'dependency_matches': {
                    'XGBoostTraining': {
                        'training_data': {
                            'provider_node': 'DataPreprocessing',
                            'provider_output': 'processed_data',
                            'compatibility_score': 0.95
                        }
                    }
                },
                'config_data': {},
                'execution_order': ['DataPreprocessing', 'XGBoostTraining']
            }
            
            # Mock the user input collection
            with patch('cursus.validation.script_testing.script_dependency_matcher.collect_user_inputs_with_dependency_resolution') as mock_collect:
                mock_collect.return_value = {
                    'XGBoostTraining': {
                        'input_paths': {'training_data': '/data/processed.csv'},
                        'output_paths': {'model': '/models/model.pkl'}
                    }
                }
                
                # Execute test with correct function signature
                result = resolve_script_dependencies(mock_dag, config_path, step_catalog)
                
                # Validate results
                assert 'XGBoostTraining' in result
                assert result['XGBoostTraining']['input_paths']['training_data'] == '/data/processed.csv'
    
    def test_resolve_script_dependencies_with_exception(self):
        """Test dependency resolution with exception."""
        # Create proper mock DAG
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['XGBoostTraining']
        
        config_path = '/config/pipeline.json'
        step_catalog = Mock()
        
        # Mock prepare_script_testing_inputs to raise exception
        with patch('cursus.validation.script_testing.script_dependency_matcher.prepare_script_testing_inputs') as mock_prepare:
            mock_prepare.side_effect = Exception("Preparation failed")
            
            # Execute test and expect RuntimeError
            with pytest.raises(RuntimeError, match="Failed to resolve script dependencies"):
                resolve_script_dependencies(mock_dag, config_path, step_catalog)


class TestExecuteSingleScript:
    """Test single script execution."""
    
    @patch('cursus.validation.script_testing.api.install_script_dependencies')
    @patch('cursus.validation.script_testing.api.import_and_execute_script')
    def test_execute_single_script_success(self, mock_import_execute, mock_install_deps):
        """Test successful single script execution."""
        # Setup mocks
        mock_install_deps.return_value = None
        mock_import_execute.return_value = {
            'outputs': {'model': '/output/model.pkl'},
            'execution_time': 1.5
        }
        
        # Execute test
        result = execute_single_script(
            script_path='/scripts/train.py',
            input_paths={'training_data': '/data/train.csv'},
            output_paths={'model': '/output/model.pkl'},
            environ_vars={'ENV': 'test'},
            job_args=Mock()
        )
        
        # Validate results
        assert result.success is True
        assert result.output_files == {'model': '/output/model.pkl'}
        assert result.execution_time == 1.5
        assert result.error_message is None
        
        # Validate calls
        mock_install_deps.assert_called_once_with('/scripts/train.py')
        mock_import_execute.assert_called_once()
    
    @patch('cursus.validation.script_testing.api.install_script_dependencies')
    @patch('cursus.validation.script_testing.api.import_and_execute_script')
    def test_execute_single_script_failure(self, mock_import_execute, mock_install_deps):
        """Test single script execution failure."""
        # Setup mocks
        mock_install_deps.return_value = None
        mock_import_execute.side_effect = Exception("Script execution failed")
        
        # Execute test
        result = execute_single_script(
            script_path='/scripts/train.py',
            input_paths={'training_data': '/data/train.csv'},
            output_paths={'model': '/output/model.pkl'},
            environ_vars={'ENV': 'test'},
            job_args=Mock()
        )
        
        # Validate results
        assert result.success is False
        assert result.output_files == {}
        assert result.error_message == "Script execution failed"
        assert result.execution_time is None
        
        # Validate calls
        mock_install_deps.assert_called_once_with('/scripts/train.py')
        mock_import_execute.assert_called_once()
    
    @patch('cursus.validation.script_testing.api.install_script_dependencies')
    def test_execute_single_script_dependency_installation_failure(self, mock_install_deps):
        """Test single script execution with dependency installation failure."""
        # Setup mock to raise exception during dependency installation
        mock_install_deps.side_effect = Exception("Dependency installation failed")
        
        # Execute test
        result = execute_single_script(
            script_path='/scripts/train.py',
            input_paths={'training_data': '/data/train.csv'},
            output_paths={'model': '/output/model.pkl'},
            environ_vars={'ENV': 'test'},
            job_args=Mock()
        )
        
        # Validate results
        assert result.success is False
        assert "Dependency installation failed" in result.error_message


class TestInstallScriptDependencies:
    """Test script dependency installation."""
    
    @patch('cursus.validation.script_testing.api.parse_script_imports')
    @patch('cursus.validation.script_testing.api.is_package_installed')
    @patch('cursus.validation.script_testing.api.install_package')
    def test_install_script_dependencies_success(self, mock_install_package, mock_is_installed, mock_parse_imports):
        """Test successful dependency installation."""
        # Setup mocks
        mock_parse_imports.return_value = ['numpy', 'pandas', 'sklearn']
        mock_is_installed.side_effect = lambda pkg: pkg == 'numpy'  # numpy already installed
        mock_install_package.return_value = None
        
        # Execute test
        install_script_dependencies('/scripts/train.py')
        
        # Validate calls
        mock_parse_imports.assert_called_once_with('/scripts/train.py')
        assert mock_is_installed.call_count == 3
        assert mock_install_package.call_count == 2  # pandas and sklearn need installation
        mock_install_package.assert_any_call('pandas')
        mock_install_package.assert_any_call('sklearn')
    
    @patch('cursus.validation.script_testing.api.parse_script_imports')
    def test_install_script_dependencies_parse_failure(self, mock_parse_imports):
        """Test dependency installation with parse failure."""
        # Setup mock to raise exception
        mock_parse_imports.side_effect = Exception("Parse failed")
        
        # Execute test - should not raise exception, just log warning
        install_script_dependencies('/scripts/train.py')
        
        # Validate calls
        mock_parse_imports.assert_called_once_with('/scripts/train.py')
    
    @patch('cursus.validation.script_testing.api.parse_script_imports')
    @patch('cursus.validation.script_testing.api.is_package_installed')
    @patch('cursus.validation.script_testing.api.install_package')
    def test_install_script_dependencies_install_failure(self, mock_install_package, mock_is_installed, mock_parse_imports):
        """Test dependency installation with install failure."""
        # Setup mocks
        mock_parse_imports.return_value = ['numpy']
        mock_is_installed.return_value = False
        mock_install_package.side_effect = Exception("Install failed")
        
        # Execute test - should not raise exception, just log warning
        install_script_dependencies('/scripts/train.py')
        
        # Validate calls
        mock_parse_imports.assert_called_once_with('/scripts/train.py')
        mock_is_installed.assert_called_once_with('numpy')
        mock_install_package.assert_called_once_with('numpy')


class TestParseScriptImports:
    """Test script import parsing."""
    
    @patch('builtins.open', new_callable=mock_open, read_data="""
import numpy as np
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import sys
import json
""")
    @patch('ast.parse')
    def test_parse_script_imports_success(self, mock_ast_parse, mock_file):
        """Test successful import parsing."""
        # Create mock AST nodes with proper name attributes
        mock_name1 = Mock()
        mock_name1.name = 'numpy'
        mock_name2 = Mock()
        mock_name2.name = 'pandas'
        mock_name3 = Mock()
        mock_name3.name = 'os'
        mock_name4 = Mock()
        mock_name4.name = 'sys'
        mock_name5 = Mock()
        mock_name5.name = 'json'
        
        import_node1 = Mock(spec=ast.Import)
        import_node1.names = [mock_name1, mock_name2]
        
        import_from_node1 = Mock(spec=ast.ImportFrom)
        import_from_node1.module = 'sklearn.metrics'
        
        import_from_node2 = Mock(spec=ast.ImportFrom)
        import_from_node2.module = 'sklearn.model_selection'
        
        import_node2 = Mock(spec=ast.Import)
        import_node2.names = [mock_name3, mock_name4, mock_name5]
        
        # Create mock tree
        mock_tree = Mock()
        mock_ast_parse.return_value = mock_tree
        
        # Mock ast.walk to return our nodes
        with patch('ast.walk', return_value=[import_node1, import_from_node1, import_from_node2, import_node2]):
            # Execute test
            result = parse_script_imports('/scripts/train.py')
        
        # Validate results - should exclude standard library modules
        expected_packages = ['numpy', 'pandas', 'sklearn']
        assert set(result) == set(expected_packages)
        
        # Validate calls - ast.parse may be called multiple times by other mocks, so just check it was called
        mock_file.assert_called_once_with('/scripts/train.py', 'r')
        assert mock_ast_parse.called
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_parse_script_imports_file_not_found(self, mock_file):
        """Test import parsing with file not found."""
        # Execute test
        result = parse_script_imports('/scripts/nonexistent.py')
        
        # Validate results - should return empty list
        assert result == []
    
    @patch('builtins.open', new_callable=mock_open, read_data="invalid python code !!!")
    @patch('ast.parse', side_effect=SyntaxError("Invalid syntax"))
    def test_parse_script_imports_syntax_error(self, mock_ast_parse, mock_file):
        """Test import parsing with syntax error."""
        # Execute test
        result = parse_script_imports('/scripts/invalid.py')
        
        # Validate results - should return empty list
        assert result == []


class TestIsPackageInstalled:
    """Test package installation check."""
    
    @patch('importlib.util.find_spec')
    def test_is_package_installed_true(self, mock_find_spec):
        """Test package is installed."""
        mock_find_spec.return_value = Mock()  # Package found
        
        # Execute test
        result = is_package_installed('numpy')
        
        # Validate results
        assert result is True
        mock_find_spec.assert_called_once_with('numpy')
    
    @patch('importlib.util.find_spec')
    def test_is_package_installed_false(self, mock_find_spec):
        """Test package is not installed."""
        mock_find_spec.side_effect = ImportError("No module named 'nonexistent'")
        
        # Execute test
        result = is_package_installed('nonexistent')
        
        # Validate results
        assert result is False
        mock_find_spec.assert_called_once_with('nonexistent')


class TestInstallPackage:
    """Test package installation."""
    
    @patch('subprocess.check_call')
    def test_install_package_success(self, mock_check_call):
        """Test successful package installation."""
        mock_check_call.return_value = None
        
        # Execute test
        install_package('numpy')
        
        # Validate calls
        mock_check_call.assert_called_once_with([sys.executable, '-m', 'pip', 'install', 'numpy'])
    
    @patch('subprocess.check_call')
    def test_install_package_failure(self, mock_check_call):
        """Test package installation failure."""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, 'pip install')
        
        # Execute test and expect exception
        with pytest.raises(subprocess.CalledProcessError):
            install_package('nonexistent')


class TestImportAndExecuteScript:
    """Test script import and execution."""
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    @patch('time.time')
    def test_import_and_execute_script_success(self, mock_time, mock_module_from_spec, mock_spec_from_file):
        """Test successful script import and execution."""
        # Setup mocks
        mock_time.side_effect = [0.0, 1.5]  # start_time, end_time
        
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module.main.return_value = {'model_path': '/output/model.pkl'}
        mock_module_from_spec.return_value = mock_module
        
        # Execute test
        result = import_and_execute_script(
            script_path='/scripts/train.py',
            input_paths={'training_data': '/data/train.csv'},
            output_paths={'model': '/output/model.pkl'},
            environ_vars={'ENV': 'test'},
            job_args=Mock()
        )
        
        # Validate results
        assert result['outputs'] == {'model_path': '/output/model.pkl'}
        assert result['execution_time'] == 1.5
        
        # Validate calls
        mock_spec_from_file.assert_called_once_with("script_module", '/scripts/train.py')
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_loader.exec_module.assert_called_once_with(mock_module)
        mock_module.main.assert_called_once()
    
    @patch('importlib.util.spec_from_file_location')
    def test_import_and_execute_script_no_spec(self, mock_spec_from_file):
        """Test script import with no spec."""
        mock_spec_from_file.return_value = None
        
        # Execute test and expect ImportError
        with pytest.raises(ImportError, match="Cannot load script from"):
            import_and_execute_script(
                script_path='/scripts/train.py',
                input_paths={},
                output_paths={},
                environ_vars={},
                job_args=Mock()
            )
    
    @patch('importlib.util.spec_from_file_location')
    def test_import_and_execute_script_no_loader(self, mock_spec_from_file):
        """Test script import with no loader."""
        mock_spec = Mock()
        mock_spec.loader = None
        mock_spec_from_file.return_value = mock_spec
        
        # Execute test and expect ImportError
        with pytest.raises(ImportError, match="Cannot load script from"):
            import_and_execute_script(
                script_path='/scripts/train.py',
                input_paths={},
                output_paths={},
                environ_vars={},
                job_args=Mock()
            )
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_and_execute_script_no_main_function(self, mock_module_from_spec, mock_spec_from_file):
        """Test script execution with no main function."""
        # Setup mocks
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        # No main function
        del mock_module.main
        mock_module_from_spec.return_value = mock_module
        
        # Execute test and expect ValueError
        with pytest.raises(ValueError, match="does not have a main function"):
            import_and_execute_script(
                script_path='/scripts/train.py',
                input_paths={},
                output_paths={},
                environ_vars={},
                job_args=Mock()
            )
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    @patch('time.time')
    def test_import_and_execute_script_main_function_exception(self, mock_time, mock_module_from_spec, mock_spec_from_file):
        """Test script execution with main function exception."""
        # Setup mocks
        mock_time.side_effect = [0.0, 1.0]
        
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module.main.side_effect = Exception("Main function failed")
        mock_module_from_spec.return_value = mock_module
        
        # Execute test and expect exception
        with pytest.raises(Exception, match="Main function failed"):
            import_and_execute_script(
                script_path='/scripts/train.py',
                input_paths={},
                output_paths={},
                environ_vars={},
                job_args=Mock()
            )
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    @patch('time.time')
    def test_import_and_execute_script_non_dict_result(self, mock_time, mock_module_from_spec, mock_spec_from_file):
        """Test script execution with non-dict result."""
        # Setup mocks
        mock_time.side_effect = [0.0, 1.0]
        
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_module.main.return_value = "string result"  # Non-dict result
        mock_module_from_spec.return_value = mock_module
        
        # Execute test
        result = import_and_execute_script(
            script_path='/scripts/train.py',
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args=Mock()
        )
        
        # Validate results - should wrap non-dict result
        assert result['outputs'] == {'result': 'string result'}
        assert result['execution_time'] == 1.0


class TestExecuteScriptsWithRegistryCoordination:
    """Test registry-coordinated script execution."""
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_with_registry_coordination_success(self, mock_execute_single):
        """Test successful registry-coordinated script execution."""
        # Create mock DAG
        mock_dag = Mock()
        mock_dag.topological_sort.return_value = ['DataPreprocessing', 'XGBoostTraining']
        
        # Create mock registry with proper side_effect configuration
        mock_registry = Mock()
        
        # Configure get_ready_node_inputs to return different values for each call
        script_inputs = [
            {
                'script_path': '/scripts/preprocess.py',
                'input_paths': {'raw_data': '/data/raw.csv'},
                'output_paths': {'processed_data': '/data/processed.csv'},
                'environment_variables': {'ENV': 'test'},
                'job_arguments': Mock()
            },
            {
                'script_path': '/scripts/train.py',
                'input_paths': {'training_data': '/data/processed.csv'},
                'output_paths': {'model': '/models/model.pkl'},
                'environment_variables': {'ENV': 'test'},
                'job_arguments': Mock()
            }
        ]
        
        # Configure get_ready_node_inputs to always return valid script inputs for both nodes
        # This is needed because the function is called multiple times:
        # 1. During script execution loop
        # 2. During pipeline success calculation
        def get_ready_inputs_side_effect(node_name):
            if node_name == 'DataPreprocessing':
                return script_inputs[0]
            elif node_name == 'XGBoostTraining':
                return script_inputs[1]
            else:
                return None
        
        mock_registry.get_ready_node_inputs.side_effect = get_ready_inputs_side_effect
        
        # Setup mock execute_single_script
        mock_execute_single.return_value = ScriptTestResult(success=True)
        
        # Execute test
        result = execute_scripts_with_registry_coordination(mock_dag, mock_registry)
        
        # Validate results - Based on actual source code pipeline_success calculation:
        # successful_scripts == len([n for n in execution_order if registry.get_ready_node_inputs(n)])
        # Since both nodes return valid script inputs, pipeline should be successful
        assert result['pipeline_success'] is True
        assert result['total_scripts'] == 2
        assert result['successful_scripts'] == 2
        assert len(result['script_results']) == 2
        
        # Validate calls
        assert mock_execute_single.call_count == 2
        assert mock_registry.commit_execution_results.call_count == 2
    
    @patch('cursus.validation.script_testing.api.execute_single_script')
    def test_execute_scripts_with_registry_coordination_missing_script(self, mock_execute_single):
        """Test registry-coordinated execution with missing script configuration."""
        # Create mock DAG
        mock_dag = Mock()
        mock_dag.topological_sort.return_value = ['DataPreprocessing']
        
        # Create mock registry that returns None for script inputs
        mock_registry = Mock()
        mock_registry.get_ready_node_inputs.return_value = None
        
        # Execute test
        result = execute_scripts_with_registry_coordination(mock_dag, mock_registry)
        
        # Validate results - should skip the script
        assert result['pipeline_success'] is True  # No scripts to execute
        assert result['total_scripts'] == 1
        assert result['successful_scripts'] == 0
        
        # Validate execute_single_script was not called
        mock_execute_single.assert_not_called()


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases following pytest best practices."""
    
    def test_run_dag_scripts_with_provided_step_catalog(self):
        """Test run_dag_scripts with provided step catalog."""
        mock_dag = Mock(spec=PipelineDAG)  # Fix isinstance check
        mock_dag.nodes = ['DataPreprocessing']
        mock_dag.topological_sort.return_value = ['DataPreprocessing']
        
        mock_step_catalog = Mock()
        
        with patch('cursus.validation.script_testing.api.Path') as mock_path, \
             patch('cursus.validation.script_testing.script_dependency_matcher.resolve_script_dependencies_with_registry') as mock_resolve_registry, \
             patch('cursus.validation.script_testing.api.execute_scripts_with_registry_coordination') as mock_execute_registry:
            
            # Setup mocks
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.mkdir = Mock()
            
            # Mock registry creation
            with patch('cursus.validation.script_testing.script_execution_registry.create_script_execution_registry') as mock_create_registry:
                mock_registry = Mock()
                mock_registry.get_execution_summary.return_value = {'completed_scripts': ['DataPreprocessing']}
                mock_registry.get_message_passing_history.return_value = []
                mock_create_registry.return_value = mock_registry
                
                # Mock registry-coordinated dependency resolution
                mock_resolve_registry.return_value = {
                    'DataPreprocessing': {
                        'input_paths': {'raw_data': '/data/raw.csv'}, 
                        'output_paths': {'processed_data': '/data/processed.csv'},
                        'script_path': '/scripts/preprocess.py'
                    }
                }
                
                # Mock registry execution
                mock_execute_registry.return_value = {
                    'pipeline_success': True,
                    'script_results': {'DataPreprocessing': ScriptTestResult(success=True)},
                    'execution_order': ['DataPreprocessing'],
                    'total_scripts': 1,
                    'successful_scripts': 1
                }
                
                # Execute test
                result = run_dag_scripts(
                    dag=mock_dag,
                    config_path='/config/pipeline.json',
                    step_catalog=mock_step_catalog
                )
                
                # Validate that provided step catalog was used
                mock_create_registry.assert_called_once_with(mock_dag, mock_step_catalog)
                mock_resolve_registry.assert_called_once_with(mock_dag, '/config/pipeline.json', mock_step_catalog, mock_registry)
                assert result['pipeline_success'] is True
    
    def test_extract_script_path_no_effective_source_dir(self):
        """Test script path extraction without effective_source_dir."""
        mock_config = Mock()
        mock_config.training_entry_point = 'train.py'
        # Remove effective_source_dir attribute to simulate hasattr() returning False
        del mock_config.effective_source_dir
        # Remove resolve_hybrid_path to prevent it from being called
        del mock_config.resolve_hybrid_path
        
        with patch('os.path.exists', return_value=True):
            # Execute test
            result = extract_script_path_from_config(mock_config)
            
            # Validate results - should use entry_point directly
            assert result == 'train.py'
    
    def test_extract_environment_variables_attribute_error(self):
        """Test environment variable extraction with attribute errors."""
        mock_config = Mock()
        mock_config.model_dump.return_value = {'framework_version': '1.0.0'}
        mock_config.framework_version = '1.0.0'
        # Remove region attribute to simulate hasattr() returning False
        del mock_config.region
        
        # Execute test
        result = extract_environment_variables_from_config(mock_config)
        
        # Validate results - should handle AttributeError gracefully
        assert 'FRAMEWORK_VERSION' in result
        assert 'REGION' not in result


if __name__ == '__main__':
    pytest.main([__file__])
