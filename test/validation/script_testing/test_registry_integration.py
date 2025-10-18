"""
Test Direct Registry Integration with ScriptTestingInputCollector

This test demonstrates how the ScriptTestingInputCollector integrates directly
with the ScriptExecutionRegistry for field value population and message passing.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Import the components
from cursus.validation.script_testing.input_collector import ScriptTestingInputCollector
from cursus.validation.script_testing.script_execution_registry import ScriptExecutionRegistry
from cursus.api.dag.base_dag import PipelineDAG
from cursus.step_catalog import StepCatalog


class TestRegistryIntegration:
    """Test Direct Registry Integration functionality."""
    
    def test_registry_integrated_input_collection(self):
        """Test registry-integrated input collection with field population."""
        # Create mock DAG
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['DataPreprocessing', 'XGBoostTraining']
        mock_dag.topological_sort.return_value = ['DataPreprocessing', 'XGBoostTraining']
        mock_dag.get_dependencies.side_effect = lambda node: {
            'DataPreprocessing': set(),
            'XGBoostTraining': {'DataPreprocessing'}
        }.get(node, set())
        
        # Create mock registry
        mock_registry = Mock(spec=ScriptExecutionRegistry)
        mock_registry.execution_order = ['DataPreprocessing', 'XGBoostTraining']
        
        # Mock registry integration points
        mock_registry.get_node_config_for_resolver.side_effect = [
            {'spec': Mock()},  # DataPreprocessing config
            {'spec': Mock()}   # XGBoostTraining config
        ]
        
        mock_registry.get_dependency_outputs_for_node.side_effect = [
            {},  # DataPreprocessing has no dependencies
            {'processed_data': '/data/processed.csv'}  # XGBoostTraining gets output from DataPreprocessing
        ]
        
        mock_registry.get_execution_summary.return_value = {
            'completed_scripts': [],
            'total_nodes': 2
        }
        
        mock_registry.get_node_status.return_value = 'pending'
        
        # Create input collector with registry integration
        with patch('cursus.validation.script_testing.input_collector.build_complete_config_classes') as mock_build_classes, \
             patch('cursus.validation.script_testing.input_collector.load_configs') as mock_load_configs:
            
            # Mock config loading
            mock_build_classes.return_value = {}
            mock_load_configs.return_value = {
                'DataPreprocessing': Mock(training_entry_point='preprocess.py', effective_source_dir='/scripts'),
                'XGBoostTraining': Mock(training_entry_point='train.py', effective_source_dir='/scripts')
            }
            
            # Initialize collector with registry
            collector = ScriptTestingInputCollector(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                registry=mock_registry,
                use_dependency_resolution=False
            )
            
            # Execute registry-integrated collection
            result = collector.collect_script_inputs_for_dag()
            
            # Validate results
            assert len(result) == 2
            assert 'DataPreprocessing' in result
            assert 'XGBoostTraining' in result
            
            # Validate DataPreprocessing inputs (no dependencies)
            data_prep_inputs = result['DataPreprocessing']
            assert 'input_paths' in data_prep_inputs
            assert 'output_paths' in data_prep_inputs
            assert 'environment_variables' in data_prep_inputs
            assert 'script_path' in data_prep_inputs
            
            # Validate XGBoostTraining inputs (has dependency outputs)
            xgb_inputs = result['XGBoostTraining']
            assert 'processed_data' in xgb_inputs['input_paths']  # Message passing from DataPreprocessing
            assert xgb_inputs['input_paths']['processed_data'] == '/data/processed.csv'
            
            # Validate registry integration points were called
            assert mock_registry.get_node_config_for_resolver.call_count == 2
            assert mock_registry.get_dependency_outputs_for_node.call_count == 2
            assert mock_registry.store_resolved_inputs.call_count == 2
    
    def test_registry_enhanced_environment_variables(self):
        """Test registry-enhanced environment variable population."""
        # Create mock DAG
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['TestScript']
        mock_dag.topological_sort.return_value = ['TestScript']
        mock_dag.get_dependencies.return_value = set()
        
        # Create mock registry
        mock_registry = Mock(spec=ScriptExecutionRegistry)
        mock_registry.execution_order = ['TestScript']
        mock_registry.get_node_config_for_resolver.return_value = {}
        mock_registry.get_dependency_outputs_for_node.return_value = {}
        mock_registry.get_execution_summary.return_value = {
            'completed_scripts': [],
            'total_nodes': 1
        }
        mock_registry.get_node_status.return_value = 'pending'
        
        # Create input collector with registry
        with patch('cursus.validation.script_testing.input_collector.build_complete_config_classes') as mock_build_classes, \
             patch('cursus.validation.script_testing.input_collector.load_configs') as mock_load_configs:
            
            # Mock config with environment variables
            mock_config = Mock()
            mock_config.training_entry_point = 'test.py'
            mock_config.effective_source_dir = '/scripts'
            mock_config.__dict__ = {
                'framework_version': '1.0.0',
                'region': 'us-west-2',
                'instance_type': 'ml.m5.large'
            }
            
            mock_build_classes.return_value = {}
            mock_load_configs.return_value = {'TestScript': mock_config}
            
            collector = ScriptTestingInputCollector(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                registry=mock_registry,
                use_dependency_resolution=False
            )
            
            # Execute collection
            result = collector.collect_script_inputs_for_dag()
            
            # Validate registry-enhanced environment variables
            env_vars = result['TestScript']['environment_variables']
            
            # Base config variables
            assert env_vars['FRAMEWORK_VERSION'] == '1.0.0'
            assert env_vars['REGION'] == 'us-west-2'
            assert env_vars['INSTANCE_TYPE'] == 'ml.m5.large'
            
            # Registry-specific variables
            assert 'PIPELINE_EXECUTION_ID' in env_vars
            assert env_vars['NODE_EXECUTION_ORDER'] == '0'  # First node
            assert env_vars['TOTAL_PIPELINE_NODES'] == '1'
            assert env_vars['REGISTRY_MODE'] == 'enabled'
    
    def test_semantic_mapping_with_message_passing(self):
        """Test semantic mapping between dependency outputs and node inputs."""
        # Create mock DAG with dependencies
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['ModelTraining', 'ModelEvaluation']
        mock_dag.topological_sort.return_value = ['ModelTraining', 'ModelEvaluation']
        mock_dag.get_dependencies.side_effect = lambda node: {
            'ModelTraining': set(),
            'ModelEvaluation': {'ModelTraining'}
        }.get(node, set())
        
        # Create mock registry with semantic outputs
        mock_registry = Mock(spec=ScriptExecutionRegistry)
        mock_registry.execution_order = ['ModelTraining', 'ModelEvaluation']
        mock_registry.get_node_config_for_resolver.return_value = {}
        mock_registry.get_dependency_outputs_for_node.side_effect = [
            {},  # ModelTraining has no dependencies
            {
                'model': '/models/trained_model.pkl',  # Should map to 'model_path'
                'processed_data': '/data/train_data.csv'  # Should map to 'training_data'
            }
        ]
        mock_registry.get_execution_summary.return_value = {'completed_scripts': [], 'total_nodes': 2}
        mock_registry.get_node_status.return_value = 'pending'
        
        # Create input collector
        with patch('cursus.validation.script_testing.input_collector.build_complete_config_classes') as mock_build_classes, \
             patch('cursus.validation.script_testing.input_collector.load_configs') as mock_load_configs:
            
            mock_build_classes.return_value = {}
            
            # Create proper mock configs with all required attributes
            mock_training_config = Mock()
            mock_training_config.training_entry_point = 'train.py'
            mock_training_config.effective_source_dir = '/scripts'
            mock_training_config.__dict__ = {'framework_version': '1.0.0'}
            
            mock_eval_config = Mock()
            mock_eval_config.training_entry_point = 'evaluate.py'
            mock_eval_config.effective_source_dir = '/scripts'
            mock_eval_config.__dict__ = {'framework_version': '1.0.0'}
            
            mock_load_configs.return_value = {
                'ModelTraining': mock_training_config,
                'ModelEvaluation': mock_eval_config
            }
            
            collector = ScriptTestingInputCollector(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                registry=mock_registry,
                use_dependency_resolution=False
            )
            
            # Execute collection
            result = collector.collect_script_inputs_for_dag()
            
            # Validate semantic mapping
            eval_inputs = result['ModelEvaluation']['input_paths']
            
            # Direct mapping
            assert eval_inputs['model'] == '/models/trained_model.pkl'
            assert eval_inputs['processed_data'] == '/data/train_data.csv'
            
            # Semantic mapping
            assert eval_inputs['model_path'] == '/models/trained_model.pkl'  # model → model_path
            assert eval_inputs['training_data'] == '/data/train_data.csv'    # processed_data → training_data
    
    def test_fallback_to_manual_collection_on_registry_error(self):
        """Test fallback to manual collection when registry integration fails."""
        # Create mock DAG
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ['TestScript']
        mock_dag.topological_sort.return_value = ['TestScript']
        
        # Create mock registry that raises errors
        mock_registry = Mock(spec=ScriptExecutionRegistry)
        mock_registry.get_node_config_for_resolver.side_effect = Exception("Registry error")
        
        # Create input collector
        with patch('cursus.validation.script_testing.input_collector.build_complete_config_classes') as mock_build_classes, \
             patch('cursus.validation.script_testing.input_collector.load_configs') as mock_load_configs:
            
            mock_config = Mock()
            mock_config.training_entry_point = 'test.py'
            mock_config.__dict__ = {'framework_version': '1.0.0'}
            
            mock_build_classes.return_value = {}
            mock_load_configs.return_value = {'TestScript': mock_config}
            
            collector = ScriptTestingInputCollector(
                dag=mock_dag,
                config_path='/config/pipeline.json',
                registry=mock_registry,
                use_dependency_resolution=False
            )
            
            # Execute collection (should fallback to manual)
            result = collector.collect_script_inputs_for_dag()
            
            # Validate fallback worked
            assert len(result) == 1
            assert 'TestScript' in result
            assert 'input_paths' in result['TestScript']
            assert 'environment_variables' in result['TestScript']


def test_integration_modes():
    """Test different integration modes of ScriptTestingInputCollector."""
    mock_dag = Mock(spec=PipelineDAG)
    mock_dag.nodes = ['TestScript']
    
    with patch('cursus.validation.script_testing.input_collector.build_complete_config_classes'), \
         patch('cursus.validation.script_testing.input_collector.load_configs'):
        
        # Mode 1: Registry-integrated
        mock_registry = Mock()
        collector1 = ScriptTestingInputCollector(
            dag=mock_dag,
            config_path='/config/pipeline.json',
            registry=mock_registry,
            use_dependency_resolution=False
        )
        assert collector1.registry is not None
        
        # Mode 2: Two-phase dependency resolution
        collector2 = ScriptTestingInputCollector(
            dag=mock_dag,
            config_path='/config/pipeline.json',
            registry=None,
            use_dependency_resolution=True
        )
        assert collector2.registry is None
        assert collector2.use_dependency_resolution is True
        
        # Mode 3: Manual collection (legacy)
        collector3 = ScriptTestingInputCollector(
            dag=mock_dag,
            config_path='/config/pipeline.json',
            registry=None,
            use_dependency_resolution=False
        )
        assert collector3.registry is None
        assert collector3.use_dependency_resolution is False


if __name__ == '__main__':
    pytest.main([__file__])
