"""
Tests for Script Execution Registry

This module tests the ScriptExecutionRegistry implementation, which serves as the
central state coordinator for DAG execution with sequential message passing.

Test Coverage:
- All 6 integration points
- Sequential state management
- Message passing algorithm
- State consistency validation
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, Any

from cursus.validation.script_testing.script_execution_registry import (
    ScriptExecutionRegistry,
    DAGStateConsistency,
    create_script_execution_registry
)
from cursus.validation.script_testing.api import ScriptTestResult


class TestScriptExecutionRegistry:
    """Test the ScriptExecutionRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock DAG
        self.mock_dag = Mock()
        self.mock_dag.nodes = ['DataPrep', 'Training', 'Evaluation']
        self.mock_dag.topological_sort.return_value = ['DataPrep', 'Training', 'Evaluation']
        self.mock_dag.get_dependencies.side_effect = lambda node: {
            'DataPrep': set(),
            'Training': {'DataPrep'},
            'Evaluation': {'Training'}
        }.get(node, set())
        
        # Create mock step catalog
        self.mock_step_catalog = Mock()
        
        # Create registry instance
        self.registry = ScriptExecutionRegistry(self.mock_dag, self.mock_step_catalog)
    
    def test_initialization(self):
        """Test registry initialization."""
        assert self.registry.dag == self.mock_dag
        assert self.registry.step_catalog == self.mock_step_catalog
        assert self.registry.execution_order == ['DataPrep', 'Training', 'Evaluation']
        
        # Check initial state
        assert len(self.registry._state['node_configs']) == 0
        assert len(self.registry._state['resolved_inputs']) == 0
        assert len(self.registry._state['execution_outputs']) == 0
        assert len(self.registry._state['dependency_graph']) == 0
        assert len(self.registry._state['message_log']) == 0
        
        # Check execution status initialization
        for node in self.mock_dag.nodes:
            assert self.registry._state['execution_status'][node] == 'pending'
    
    def test_integration_point_1_initialize_from_dependency_matcher(self):
        """Test Integration Point 1: Initialize from dependency matcher."""
        prepared_data = {
            'config_data': {
                'DataPrep': {'script_path': '/scripts/preprocess.py'},
                'Training': {'script_path': '/scripts/train.py'}
            },
            'dependency_matches': {
                'Training': {'training_data': {'provider_node': 'DataPrep'}}
            },
            'node_specs': {
                'DataPrep': Mock(),
                'Training': Mock()
            },
            'execution_order': ['DataPrep', 'Training', 'Evaluation']
        }
        
        self.registry.initialize_from_dependency_matcher(prepared_data)
        
        # Verify state initialization
        assert self.registry._state['node_configs'] == prepared_data['config_data']
        assert self.registry._state['dependency_graph'] == prepared_data['dependency_matches']
        assert hasattr(self.registry, '_node_specs')
        assert self.registry._node_specs == prepared_data['node_specs']
        assert self.registry.execution_order == prepared_data['execution_order']
        
        # Verify execution status reset
        for node in self.mock_dag.nodes:
            assert self.registry._state['execution_status'][node] == 'pending'
    
    def test_integration_point_2_get_dependency_outputs_for_node(self):
        """Test Integration Point 2: Get dependency outputs for node."""
        # Setup execution outputs
        self.registry._state['execution_outputs'] = {
            'DataPrep': {
                'processed_data': '/data/processed.csv',
                'metadata': '/data/metadata.json'
            }
        }
        
        # Get dependency outputs for Training node
        outputs = self.registry.get_dependency_outputs_for_node('Training')
        
        # Verify outputs include both direct and prefixed mappings
        expected_outputs = {
            'processed_data': '/data/processed.csv',
            'metadata': '/data/metadata.json',
            'DataPrep_processed_data': '/data/processed.csv',
            'DataPrep_metadata': '/data/metadata.json'
        }
        assert outputs == expected_outputs
    
    def test_integration_point_2_no_dependencies(self):
        """Test Integration Point 2 with no dependencies."""
        outputs = self.registry.get_dependency_outputs_for_node('DataPrep')
        assert outputs == {}
    
    def test_integration_point_3_get_node_config_for_resolver(self):
        """Test Integration Point 3: Get node config for resolver."""
        # Setup node configs and specs
        self.registry._state['node_configs'] = {
            'DataPrep': {'script_path': '/scripts/preprocess.py', 'env_vars': {}}
        }
        self.registry._node_specs = {
            'DataPrep': Mock()
        }
        
        config = self.registry.get_node_config_for_resolver('DataPrep')
        
        # Verify config includes both config data and spec
        assert config['script_path'] == '/scripts/preprocess.py'
        assert config['env_vars'] == {}
        assert 'spec' in config
        assert config['spec'] == self.registry._node_specs['DataPrep']
    
    def test_integration_point_3_missing_node(self):
        """Test Integration Point 3 with missing node."""
        config = self.registry.get_node_config_for_resolver('NonExistent')
        assert config == {}
    
    def test_integration_point_4_store_resolved_inputs(self):
        """Test Integration Point 4: Store resolved inputs."""
        resolved_inputs = {
            'script_path': '/scripts/preprocess.py',
            'input_paths': {'raw_data': '/data/raw.csv'},
            'output_paths': {'processed_data': '/data/processed.csv'}
        }
        
        self.registry.store_resolved_inputs('DataPrep', resolved_inputs)
        
        # Verify storage
        assert self.registry._state['resolved_inputs']['DataPrep'] == resolved_inputs
        assert self.registry._state['execution_status']['DataPrep'] == 'ready'
    
    def test_integration_point_5_get_ready_node_inputs(self):
        """Test Integration Point 5: Get ready node inputs."""
        # Setup resolved inputs
        resolved_inputs = {
            'script_path': '/scripts/train.py',
            'input_paths': {'training_data': '/data/processed.csv'},
            'output_paths': {'model': '/models/model.pkl'}
        }
        self.registry._state['resolved_inputs']['Training'] = resolved_inputs
        
        inputs = self.registry.get_ready_node_inputs('Training')
        assert inputs == resolved_inputs
    
    def test_integration_point_5_missing_node(self):
        """Test Integration Point 5 with missing node."""
        inputs = self.registry.get_ready_node_inputs('NonExistent')
        assert inputs == {}
    
    def test_integration_point_6_commit_execution_results_success(self):
        """Test Integration Point 6: Commit successful execution results."""
        result = ScriptTestResult(
            success=True,
            output_files={'model': '/models/model.pkl', 'metrics': '/results/metrics.json'},
            error_message=None
        )
        
        self.registry.commit_execution_results('Training', result)
        
        # Verify state updates
        assert self.registry._state['execution_outputs']['Training'] == result.output_files
        assert self.registry._state['execution_status']['Training'] == 'completed'
    
    def test_integration_point_6_commit_execution_results_failure(self):
        """Test Integration Point 6: Commit failed execution results."""
        result = ScriptTestResult(
            success=False,
            output_files={},
            error_message="Script execution failed"
        )
        
        self.registry.commit_execution_results('Training', result)
        
        # Verify state updates
        assert 'Training' not in self.registry._state['execution_outputs']
        assert self.registry._state['execution_status']['Training'] == 'failed'
    
    def test_sequential_state_update(self):
        """Test sequential state update generator."""
        # Setup resolved inputs for all nodes
        for node in self.mock_dag.nodes:
            self.registry._state['resolved_inputs'][node] = {
                'script_path': f'/scripts/{node.lower()}.py',
                'input_paths': {},
                'output_paths': {}
            }
        
        # Collect all yielded states
        states = list(self.registry.sequential_state_update())
        
        # Verify correct order and structure
        assert len(states) == 3
        node_names = [state[0] for state in states]
        assert node_names == ['DataPrep', 'Training', 'Evaluation']
        
        # Verify each state is a tuple of (node_name, node_state)
        for node_name, node_state in states:
            assert isinstance(node_name, str)
            assert isinstance(node_state, dict)
            assert 'script_path' in node_state
    
    def test_message_passing_algorithm(self):
        """Test core message passing algorithm."""
        # Setup dependency outputs
        dep_outputs = {
            'processed_data': '/data/processed.csv',
            'model': '/models/base_model.pkl'
        }
        
        # Setup expected inputs for current node
        self.registry._node_specs = {
            'Training': Mock()
        }
        self.registry._node_specs['Training'].dependencies = {
            'training_data': Mock(),
            'base_model': Mock()
        }
        
        # Test message passing
        message_updates = self.registry._apply_message_passing('DataPrep', 'Training', dep_outputs)
        
        # Verify mappings
        assert 'DataPrep_processed_data' in message_updates
        assert 'DataPrep_model' in message_updates
        assert message_updates['DataPrep_processed_data'] == '/data/processed.csv'
        assert message_updates['DataPrep_model'] == '/models/base_model.pkl'
    
    def test_semantic_mapping(self):
        """Test semantic mapping in message passing."""
        expected_inputs = {'model_path', 'training_data', 'config_file'}
        
        # Test various semantic mappings
        assert self.registry._get_semantic_mapping('model', expected_inputs) == 'model_path'
        assert self.registry._get_semantic_mapping('processed_data', expected_inputs) == 'training_data'
        assert self.registry._get_semantic_mapping('unknown_output', expected_inputs) is None
    
    def test_get_expected_inputs(self):
        """Test getting expected inputs from specifications."""
        # Setup node spec
        mock_spec = Mock()
        mock_spec.dependencies = {
            'training_data': Mock(),
            'model_config': Mock()
        }
        self.registry._node_specs = {'Training': mock_spec}
        
        # Setup contract
        mock_contract = Mock()
        mock_contract.expected_input_paths = {
            'validation_data': '/opt/ml/input/data/validation',
            'hyperparams': '/opt/ml/input/config/hyperparameters.json'
        }
        self.mock_step_catalog.load_contract_class.return_value = mock_contract
        
        expected_inputs = self.registry._get_expected_inputs('Training')
        
        # Verify inputs from both spec and contract
        assert 'training_data' in expected_inputs
        assert 'model_config' in expected_inputs
        assert 'validation_data' in expected_inputs
        assert 'hyperparams' in expected_inputs
    
    def test_get_expected_inputs_no_contract(self):
        """Test getting expected inputs when contract is unavailable."""
        # Setup node spec only
        mock_spec = Mock()
        mock_spec.dependencies = {'training_data': Mock()}
        self.registry._node_specs = {'Training': mock_spec}
        
        # Make contract loading fail
        self.mock_step_catalog.load_contract_class.side_effect = Exception("Contract not found")
        
        expected_inputs = self.registry._get_expected_inputs('Training')
        
        # Verify graceful fallback
        assert expected_inputs == {'training_data'}
    
    def test_is_node_completed(self):
        """Test node completion check."""
        # Setup execution status
        self.registry._state['execution_status']['DataPrep'] = 'completed'
        self.registry._state['execution_status']['Training'] = 'failed'
        self.registry._state['execution_status']['Evaluation'] = 'pending'
        
        # Test completion checks
        assert self.registry._is_node_completed('DataPrep') is True
        assert self.registry._is_node_completed('Training') is False
        assert self.registry._is_node_completed('Evaluation') is False
    
    def test_get_execution_summary(self):
        """Test execution summary generation."""
        # Setup various execution states
        self.registry._state['node_configs'] = {'DataPrep': {}, 'Training': {}}
        self.registry._state['execution_status'] = {
            'DataPrep': 'completed',
            'Training': 'failed',
            'Evaluation': 'ready'
        }
        self.registry._state['message_log'] = [{'test': 'message'}]
        
        summary = self.registry.get_execution_summary()
        
        # Verify summary content
        assert summary['registered_scripts'] == ['DataPrep', 'Training']
        assert summary['completed_scripts'] == ['DataPrep']
        assert summary['failed_scripts'] == ['Training']
        assert summary['ready_scripts'] == ['Evaluation']
        assert summary['pending_scripts'] == []
        assert summary['message_count'] == 1
        assert summary['total_nodes'] == 3
    
    def test_get_message_passing_history(self):
        """Test message passing history retrieval."""
        # Setup message log
        test_messages = [
            {'from_node': 'DataPrep', 'to_node': 'Training', 'timestamp': time.time()},
            {'from_node': 'Training', 'to_node': 'Evaluation', 'timestamp': time.time()}
        ]
        self.registry._state['message_log'] = test_messages
        
        history = self.registry.get_message_passing_history()
        
        # Verify history is a copy
        assert history == test_messages
        assert history is not self.registry._state['message_log']
    
    def test_get_node_status(self):
        """Test individual node status retrieval."""
        self.registry._state['execution_status']['DataPrep'] = 'completed'
        
        assert self.registry.get_node_status('DataPrep') == 'completed'
        assert self.registry.get_node_status('NonExistent') == 'unknown'
    
    def test_get_node_outputs(self):
        """Test individual node outputs retrieval."""
        outputs = {'model': '/models/model.pkl'}
        self.registry._state['execution_outputs']['Training'] = outputs
        
        assert self.registry.get_node_outputs('Training') == outputs
        assert self.registry.get_node_outputs('NonExistent') == {}
    
    def test_clear_registry(self):
        """Test registry clearing."""
        # Setup some state
        self.registry._state['node_configs']['DataPrep'] = {'test': 'config'}
        self.registry._state['execution_status']['DataPrep'] = 'completed'
        
        self.registry.clear_registry()
        
        # Verify state is cleared
        assert len(self.registry._state['node_configs']) == 0
        assert len(self.registry._state['resolved_inputs']) == 0
        assert len(self.registry._state['execution_outputs']) == 0
        assert len(self.registry._state['message_log']) == 0
        
        # Verify execution status is reinitialized
        for node in self.mock_dag.nodes:
            assert self.registry._state['execution_status'][node] == 'pending'


class TestDAGStateConsistency:
    """Test the DAGStateConsistency validation class."""
    
    def test_validate_execution_order_valid(self):
        """Test validation of valid execution order."""
        mock_dag = Mock()
        mock_dag.get_dependencies.side_effect = lambda node: {
            'DataPrep': set(),
            'Training': {'DataPrep'},
            'Evaluation': {'Training'}
        }.get(node, set())
        
        execution_order = ['DataPrep', 'Training', 'Evaluation']
        
        # Should not raise exception
        DAGStateConsistency.validate_execution_order(mock_dag, execution_order)
    
    def test_validate_execution_order_invalid(self):
        """Test validation of invalid execution order."""
        mock_dag = Mock()
        mock_dag.get_dependencies.side_effect = lambda node: {
            'DataPrep': set(),
            'Training': {'DataPrep'},
            'Evaluation': {'Training'}
        }.get(node, set())
        
        # Invalid order - Training before DataPrep
        execution_order = ['Training', 'DataPrep', 'Evaluation']
        
        with pytest.raises(ValueError, match="Invalid execution order"):
            DAGStateConsistency.validate_execution_order(mock_dag, execution_order)
    
    def test_ensure_state_consistency_valid(self):
        """Test state consistency validation for valid state."""
        mock_registry = Mock()
        mock_registry.dag.get_dependencies.return_value = {'DataPrep'}
        mock_registry._is_node_completed.return_value = True
        
        # Should not raise exception
        DAGStateConsistency.ensure_state_consistency(mock_registry, 'Training')
    
    def test_ensure_state_consistency_invalid(self):
        """Test state consistency validation for invalid state."""
        mock_registry = Mock()
        mock_registry.dag.get_dependencies.return_value = {'DataPrep'}
        mock_registry._is_node_completed.return_value = False
        
        with pytest.raises(RuntimeError, match="State inconsistency"):
            DAGStateConsistency.ensure_state_consistency(mock_registry, 'Training')


class TestCreateScriptExecutionRegistry:
    """Test the factory function for creating registry instances."""
    
    def test_create_with_step_catalog(self):
        """Test creating registry with provided step catalog."""
        mock_dag = Mock()
        mock_dag.nodes = ['DataPrep']
        mock_dag.topological_sort.return_value = ['DataPrep']
        mock_dag.get_dependencies.return_value = set()
        
        mock_step_catalog = Mock()
        
        registry = create_script_execution_registry(mock_dag, mock_step_catalog)
        
        assert isinstance(registry, ScriptExecutionRegistry)
        assert registry.dag == mock_dag
        assert registry.step_catalog == mock_step_catalog
    
    @patch('cursus.validation.script_testing.script_execution_registry.StepCatalog')
    def test_create_without_step_catalog(self, mock_step_catalog_class):
        """Test creating registry without step catalog (creates default)."""
        mock_dag = Mock()
        mock_dag.nodes = ['DataPrep']
        mock_dag.topological_sort.return_value = ['DataPrep']
        mock_dag.get_dependencies.return_value = set()
        
        mock_step_catalog_instance = Mock()
        mock_step_catalog_class.return_value = mock_step_catalog_instance
        
        registry = create_script_execution_registry(mock_dag)
        
        assert isinstance(registry, ScriptExecutionRegistry)
        assert registry.dag == mock_dag
        mock_step_catalog_class.assert_called_once()


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        # Create realistic DAG
        self.mock_dag = Mock()
        self.mock_dag.nodes = ['DataPreprocessing', 'XGBoostTraining', 'ModelEvaluation']
        self.mock_dag.topological_sort.return_value = ['DataPreprocessing', 'XGBoostTraining', 'ModelEvaluation']
        self.mock_dag.get_dependencies.side_effect = lambda node: {
            'DataPreprocessing': set(),
            'XGBoostTraining': {'DataPreprocessing'},
            'ModelEvaluation': {'XGBoostTraining'}
        }.get(node, set())
        
        self.registry = ScriptExecutionRegistry(self.mock_dag)
    
    def test_complete_pipeline_execution_flow(self):
        """Test complete pipeline execution flow with message passing."""
        # Phase 1: Initialize from dependency matcher
        prepared_data = {
            'config_data': {
                'DataPreprocessing': {'script_path': '/scripts/preprocess.py'},
                'XGBoostTraining': {'script_path': '/scripts/train.py'},
                'ModelEvaluation': {'script_path': '/scripts/evaluate.py'}
            },
            'dependency_matches': {
                'XGBoostTraining': {'training_data': {'provider_node': 'DataPreprocessing'}},
                'ModelEvaluation': {'model': {'provider_node': 'XGBoostTraining'}}
            },
            'node_specs': {}
        }
        
        self.registry.initialize_from_dependency_matcher(prepared_data)
        
        # Phase 2: Process each node in sequence
        execution_results = {}
        
        for node_name, node_state in self.registry.sequential_state_update():
            # Simulate script execution
            if node_name == 'DataPreprocessing':
                result = ScriptTestResult(
                    success=True,
                    output_files={'processed_data': '/data/processed.csv'},
                    error_message=None
                )
            elif node_name == 'XGBoostTraining':
                result = ScriptTestResult(
                    success=True,
                    output_files={'model': '/models/xgboost_model.pkl'},
                    error_message=None
                )
            else:  # ModelEvaluation
                result = ScriptTestResult(
                    success=True,
                    output_files={'metrics': '/results/evaluation_metrics.json'},
                    error_message=None
                )
            
            # Commit results
            self.registry.commit_execution_results(node_name, result)
            execution_results[node_name] = result
        
        # Verify final state
        summary = self.registry.get_execution_summary()
        assert len(summary['completed_scripts']) == 3
        assert len(summary['failed_scripts']) == 0
        
        # Verify message passing occurred
        assert len(self.registry.get_message_passing_history()) > 0
        
        # Verify outputs are available
        assert self.registry.get_node_outputs('DataPreprocessing')['processed_data'] == '/data/processed.csv'
        assert self.registry.get_node_outputs('XGBoostTraining')['model'] == '/models/xgboost_model.pkl'
    
    def test_pipeline_execution_with_failure(self):
        """Test pipeline execution with node failure."""
        # Initialize registry
        prepared_data = {
            'config_data': {
                'DataPreprocessing': {'script_path': '/scripts/preprocess.py'},
                'XGBoostTraining': {'script_path': '/scripts/train.py'}
            },
            'dependency_matches': {},
            'node_specs': {}
        }
        
        self.registry.initialize_from_dependency_matcher(prepared_data)
        
        # Simulate DataPreprocessing success
        success_result = ScriptTestResult(
            success=True,
            output_files={'processed_data': '/data/processed.csv'},
            error_message=None
        )
        self.registry.commit_execution_results('DataPreprocessing', success_result)
        
        # Simulate XGBoostTraining failure
        failure_result = ScriptTestResult(
            success=False,
            output_files={},
            error_message="Training failed due to insufficient data"
        )
        self.registry.commit_execution_results('XGBoostTraining', failure_result)
        
        # Verify state
        summary = self.registry.get_execution_summary()
        assert 'DataPreprocessing' in summary['completed_scripts']
        assert 'XGBoostTraining' in summary['failed_scripts']
        
        # Verify failed node has no outputs
        assert self.registry.get_node_outputs('XGBoostTraining') == {}
