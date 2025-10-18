"""
Comprehensive tests for the Two-Phase Script Dependency Resolution System.

Tests validate:
- Direct reuse of pipeline assembler patterns
- Maximum component reuse from existing cursus infrastructure
- Two-phase architecture functionality
- User override capability
- Performance and automation benefits
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any
import argparse

# Import the module under test
from cursus.validation.script_testing.script_dependency_matcher import (
    prepare_script_testing_inputs,
    collect_user_inputs_with_dependency_resolution,
    resolve_script_dependencies,
    validate_dependency_resolution_result,
    get_dependency_resolution_summary
)

# Import dependencies for mocking
from cursus.api.dag.base_dag import PipelineDAG
from cursus.step_catalog import StepCatalog


class TestPrepareScriptTestingInputs:
    """Test Phase 1: Automatic dependency analysis using pipeline assembler patterns."""
    
    def test_prepare_with_valid_dag_and_specs(self):
        """Test Phase 1 with valid DAG and specifications."""
        # Create test DAG
        dag = PipelineDAG()
        dag.add_node('TrainingScript')
        dag.add_node('ValidationScript')
        dag.add_edge('TrainingScript', 'ValidationScript')
        
        # Mock step catalog and specifications
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        
        mock_spec = Mock()
        mock_spec.dependencies = {'training_data': Mock()}
        mock_spec.outputs = {'model': Mock()}
        mock_spec_discovery.load_spec_class.return_value = mock_spec
        
        # Mock dependency resolver
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver') as mock_resolver_factory:
            mock_resolver = Mock()
            mock_resolver._calculate_compatibility.return_value = 0.8
            mock_resolver_factory.return_value = mock_resolver
            
            # Mock config loading
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes') as mock_build_classes:
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs') as mock_load_configs:
                    with patch('cursus.validation.script_testing.script_dependency_matcher.collect_script_inputs') as mock_collect_inputs:
                        mock_build_classes.return_value = {'TrainingScript': Mock}
                        mock_load_configs.return_value = {'TrainingScript': Mock()}
                        mock_collect_inputs.return_value = {
                            'script_path': '/path/to/script.py',
                            'environment_variables': {'ENV': 'test'},
                            'job_arguments': argparse.Namespace(epochs=100)
                        }
                        
                        # Execute test
                        result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                        
                        # Validate results
                        assert 'node_specs' in result
                        assert 'dependency_matches' in result
                        assert 'config_data' in result
                        assert 'execution_order' in result
                        assert result['execution_order'] == ['TrainingScript', 'ValidationScript']
    
    def test_prepare_with_no_specifications(self):
        """Test Phase 1 when no specifications are found."""
        dag = PipelineDAG()
        dag.add_node('UnknownScript')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        mock_spec_discovery.load_spec_class.return_value = None
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver'):
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                    
                    assert len(result['node_specs']) == 0
                    assert len(result['dependency_matches']) == 0
    
    def test_prepare_with_config_extraction_failure(self):
        """Test Phase 1 when config extraction fails."""
        dag = PipelineDAG()
        dag.add_node('TestScript')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        mock_spec_discovery.load_spec_class.return_value = None
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver'):
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes') as mock_build:
                mock_build.side_effect = Exception("Config loading failed")
                
                result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                
                assert result['config_data'] == {}
                assert 'execution_order' in result


class TestCollectUserInputsWithDependencyResolution:
    """Test Phase 2: Interactive input collection with automatic dependency resolution."""
    
    def test_collect_with_auto_resolved_dependencies(self):
        """Test Phase 2 with automatic dependency resolution."""
        # Prepare test data
        prepared_data = {
            'execution_order': ['Script1', 'Script2'],
            'dependency_matches': {
                'Script2': {
                    'input_data': {
                        'provider_node': 'Script1',
                        'provider_output': 'output_data',
                        'compatibility_score': 0.9,
                        'match_type': 'specification_match'
                    }
                }
            },
            'node_specs': {
                'Script1': Mock(dependencies={}, outputs={'output_data': Mock()}),
                'Script2': Mock(dependencies={'input_data': Mock()}, outputs={'result': Mock()})
            },
            'config_data': {
                'Script1': {
                    'script_path': '/path/to/script1.py',
                    'environment_variables': {'ENV': 'test'},
                    'job_arguments': argparse.Namespace()
                },
                'Script2': {
                    'script_path': '/path/to/script2.py',
                    'environment_variables': {'ENV': 'test'},
                    'job_arguments': argparse.Namespace()
                }
            }
        }
        
        # Mock user input
        with patch('builtins.input', side_effect=[
            '/output/script1_data.csv',  # Script1 output (output_data)
            '',  # Script2 input override (empty = keep auto-resolved)
            '/output/script2_result.csv'  # Script2 output (result)
        ]):
            result = collect_user_inputs_with_dependency_resolution(prepared_data)
            
            # Validate results
            assert len(result) == 2
            assert 'Script1' in result
            assert 'Script2' in result
            
            # Check Script2 has auto-resolved input
            script2_inputs = result['Script2']
            assert 'input_paths' in script2_inputs
            assert 'input_data' in script2_inputs['input_paths']
            assert script2_inputs['input_paths']['input_data'] == '/output/script1_data.csv'
    
    def test_collect_with_user_override(self):
        """Test Phase 2 with user override of auto-resolved paths."""
        prepared_data = {
            'execution_order': ['Script1', 'Script2'],
            'dependency_matches': {
                'Script2': {
                    'input_data': {
                        'provider_node': 'Script1',
                        'provider_output': 'output_data',
                        'compatibility_score': 0.9,
                        'match_type': 'specification_match'
                    }
                }
            },
            'node_specs': {
                'Script1': Mock(dependencies={}, outputs={'output_data': Mock()}),
                'Script2': Mock(dependencies={'input_data': Mock()}, outputs={'result': Mock()})
            },
            'config_data': {
                'Script1': {'script_path': '/path/to/script1.py', 'environment_variables': {}, 'job_arguments': argparse.Namespace()},
                'Script2': {'script_path': '/path/to/script2.py', 'environment_variables': {}, 'job_arguments': argparse.Namespace()}
            }
        }
        
        # Mock user input with override
        with patch('builtins.input', side_effect=[
            '/output/script1_data.csv',  # Script1 output
            '/custom/override_path.csv',  # Script2 input override
            '/output/script2_result.csv'  # Script2 output
        ]):
            result = collect_user_inputs_with_dependency_resolution(prepared_data)
            
            # Check that user override was applied
            script2_inputs = result['Script2']
            assert script2_inputs['input_paths']['input_data'] == '/custom/override_path.csv'


class TestResolveScriptDependencies:
    """Test main entry point: Two-phase dependency resolution system."""
    
    @patch('cursus.validation.script_testing.script_dependency_matcher.collect_user_inputs_with_dependency_resolution')
    @patch('cursus.validation.script_testing.script_dependency_matcher.prepare_script_testing_inputs')
    def test_resolve_complete_workflow(self, mock_prepare, mock_collect):
        """Test complete two-phase workflow."""
        # Setup mocks
        mock_prepare.return_value = {
            'dependency_matches': {'Script1': {'dep1': {}}},
            'execution_order': ['Script1']
        }
        mock_collect.return_value = {
            'Script1': {
                'input_paths': {'dep1': '/path/to/input'},
                'output_paths': {'output1': '/path/to/output'},
                'environment_variables': {},
                'job_arguments': argparse.Namespace()
            }
        }
        
        # Create test inputs
        dag = PipelineDAG()
        dag.add_node('Script1')
        step_catalog = Mock(spec=StepCatalog)
        
        # Execute test
        result = resolve_script_dependencies(dag, 'config.json', step_catalog)
        
        # Validate results
        assert 'Script1' in result
        assert mock_prepare.called
        assert mock_collect.called
    
    def test_resolve_with_exception(self):
        """Test resolve_script_dependencies with exception handling."""
        dag = PipelineDAG()
        step_catalog = Mock(spec=StepCatalog)
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.prepare_script_testing_inputs') as mock_prepare:
            mock_prepare.side_effect = Exception("Test error")
            
            with pytest.raises(RuntimeError, match="Failed to resolve script dependencies"):
                resolve_script_dependencies(dag, 'config.json', step_catalog)


class TestValidateDependencyResolutionResult:
    """Test validation of dependency resolution results."""
    
    def test_validate_valid_result(self):
        """Test validation with valid user inputs."""
        user_inputs = {
            'Script1': {
                'input_paths': {'input1': '/path/to/input'},
                'output_paths': {'output1': '/path/to/output'},
                'environment_variables': {'ENV': 'test'},
                'job_arguments': argparse.Namespace()
            }
        }
        
        result = validate_dependency_resolution_result(user_inputs)
        assert result is True
    
    def test_validate_missing_required_field(self):
        """Test validation with missing required field."""
        user_inputs = {
            'Script1': {
                'input_paths': {'input1': '/path/to/input'},
                # Missing 'output_paths'
                'environment_variables': {'ENV': 'test'},
                'job_arguments': argparse.Namespace()
            }
        }
        
        result = validate_dependency_resolution_result(user_inputs)
        assert result is False
    
    def test_validate_with_exception(self):
        """Test validation with exception handling."""
        # Invalid input that will cause exception
        user_inputs = None
        
        result = validate_dependency_resolution_result(user_inputs)
        assert result is False


class TestGetDependencyResolutionSummary:
    """Test summary generation for dependency resolution process."""
    
    def test_get_summary_with_automation(self):
        """Test summary generation with automatic dependency resolution."""
        prepared_data = {
            'dependency_matches': {
                'Script1': {},
                'Script2': {'dep1': {}, 'dep2': {}}
            }
        }
        
        user_inputs = {
            'Script1': {
                'input_paths': {'manual_input': '/path/to/manual'},
                'output_paths': {'output1': '/path/to/output1'}
            },
            'Script2': {
                'input_paths': {'dep1': '/auto/path1', 'dep2': '/auto/path2', 'manual_input': '/path/to/manual2'},
                'output_paths': {'output2': '/path/to/output2'}
            }
        }
        
        summary = get_dependency_resolution_summary(prepared_data, user_inputs)
        
        assert summary['total_nodes'] == 2
        assert summary['total_dependencies'] == 4  # 1 + 3 input paths
        assert summary['auto_resolved_dependencies'] == 2  # dep1 + dep2
        assert summary['manual_dependencies'] == 2  # manual inputs
        assert summary['automation_rate_percentage'] == 50.0  # 2/4 * 100
        assert summary['nodes_with_auto_resolution'] == 1  # Only Script2 has matches
    
    def test_get_summary_no_automation(self):
        """Test summary generation with no automatic dependency resolution."""
        prepared_data = {
            'dependency_matches': {
                'Script1': {},
                'Script2': {}
            }
        }
        
        user_inputs = {
            'Script1': {
                'input_paths': {'manual_input': '/path/to/manual'},
                'output_paths': {'output1': '/path/to/output1'}
            }
        }
        
        summary = get_dependency_resolution_summary(prepared_data, user_inputs)
        
        assert summary['total_nodes'] == 1
        assert summary['total_dependencies'] == 1
        assert summary['auto_resolved_dependencies'] == 0
        assert summary['manual_dependencies'] == 1
        assert summary['automation_rate_percentage'] == 0.0
        assert summary['nodes_with_auto_resolution'] == 0


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_xgboost_training_pipeline_scenario(self):
        """Test realistic XGBoost training pipeline scenario."""
        # Create realistic DAG
        dag = PipelineDAG()
        dag.add_node('DataPreprocessing')
        dag.add_node('XGBoostTraining')
        dag.add_node('ModelEvaluation')
        dag.add_edge('DataPreprocessing', 'XGBoostTraining')
        dag.add_edge('XGBoostTraining', 'ModelEvaluation')
        
        # Mock realistic specifications
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        
        def mock_load_spec(node_name):
            if node_name == 'DataPreprocessing':
                spec = Mock()
                spec.dependencies = {}
                spec.outputs = {'processed_data': Mock()}
                return spec
            elif node_name == 'XGBoostTraining':
                spec = Mock()
                spec.dependencies = {'training_data': Mock()}
                spec.outputs = {'model': Mock()}
                return spec
            elif node_name == 'ModelEvaluation':
                spec = Mock()
                spec.dependencies = {'model': Mock(), 'test_data': Mock()}
                spec.outputs = {'metrics': Mock()}
                return spec
            return None
        
        mock_spec_discovery.load_spec_class.side_effect = mock_load_spec
        
        # Mock dependency resolver with realistic compatibility scores
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver') as mock_resolver_factory:
            mock_resolver = Mock()
            
            def mock_compatibility(dep_spec, output_spec, provider_spec):
                # Simulate high compatibility for matching data types
                return 0.9
            
            mock_resolver._calculate_compatibility.side_effect = mock_compatibility
            mock_resolver_factory.return_value = mock_resolver
            
            # Mock config loading
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs') as mock_load_configs:
                    with patch('cursus.validation.script_testing.script_dependency_matcher.collect_script_inputs') as mock_collect_inputs:
                        mock_load_configs.return_value = {
                            'DataPreprocessing': Mock(),
                            'XGBoostTraining': Mock(),
                            'ModelEvaluation': Mock()
                        }
                        
                        def mock_collect(config):
                            return {
                                'script_path': f'/scripts/{config.__class__.__name__.lower()}.py',
                                'environment_variables': {'FRAMEWORK': 'xgboost'},
                                'job_arguments': argparse.Namespace(epochs=100)
                            }
                        
                        mock_collect_inputs.side_effect = mock_collect
                        
                        # Execute Phase 1
                        result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                        
                        # Validate realistic dependency matching
                        assert len(result['node_specs']) == 3
                        assert 'XGBoostTraining' in result['dependency_matches']
                        assert 'ModelEvaluation' in result['dependency_matches']
                        
                        # Check that XGBoostTraining has training_data dependency matched
                        xgb_matches = result['dependency_matches']['XGBoostTraining']
                        assert 'training_data' in xgb_matches
                        assert xgb_matches['training_data']['provider_node'] == 'DataPreprocessing'
                        assert xgb_matches['training_data']['provider_output'] == 'processed_data'
                        assert xgb_matches['training_data']['compatibility_score'] == 0.9
                        
                        # Check that ModelEvaluation has model dependency matched
                        # Note: The algorithm finds the first compatible provider, which could be any node
                        # since our mock returns 0.9 for all compatibility checks
                        eval_matches = result['dependency_matches']['ModelEvaluation']
                        assert 'model' in eval_matches
                        # The provider could be any node since all have 0.9 compatibility
                        assert eval_matches['model']['provider_node'] in ['DataPreprocessing', 'XGBoostTraining']
                        assert eval_matches['model']['compatibility_score'] == 0.9


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_prepare_with_spec_loading_exception(self):
        """Test Phase 1 when spec loading throws exception."""
        dag = PipelineDAG()
        dag.add_node('ProblematicScript')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        mock_spec_discovery.load_spec_class.side_effect = Exception("Spec loading failed")
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver'):
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    # Should not raise exception, should handle gracefully
                    result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                    
                    assert len(result['node_specs']) == 0
                    assert 'execution_order' in result
    
    def test_prepare_with_compatibility_calculation_exception(self):
        """Test Phase 1 when compatibility calculation fails."""
        dag = PipelineDAG()
        dag.add_node('Script1')
        dag.add_node('Script2')
        dag.add_edge('Script1', 'Script2')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        
        mock_spec1 = Mock()
        mock_spec1.dependencies = {}
        mock_spec1.outputs = {'output1': Mock()}
        mock_spec2 = Mock()
        mock_spec2.dependencies = {'input1': Mock()}
        mock_spec2.outputs = {}
        
        def mock_load_spec(node_name):
            return mock_spec1 if node_name == 'Script1' else mock_spec2
        
        mock_spec_discovery.load_spec_class.side_effect = mock_load_spec
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver') as mock_resolver_factory:
            mock_resolver = Mock()
            mock_resolver._calculate_compatibility.side_effect = Exception("Compatibility failed")
            mock_resolver_factory.return_value = mock_resolver
            
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    # Should handle exception gracefully
                    result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                    
                    # Should still return valid structure
                    assert 'dependency_matches' in result
                    assert len(result['dependency_matches']['Script2']) == 0  # No matches due to exception


class TestRealWorldScenarios:
    """Test realistic scenarios based on actual usage patterns."""
    
    def test_empty_dag(self):
        """Test with empty DAG."""
        dag = PipelineDAG()
        step_catalog = Mock()
        mock_spec_discovery = Mock()
        step_catalog.spec_discovery = mock_spec_discovery
        mock_spec_discovery.load_spec_class.return_value = None
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver'):
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    result = prepare_script_testing_inputs(dag, 'config.json', step_catalog)
                    
                    assert result['node_specs'] == {}
                    assert result['dependency_matches'] == {}
                    assert result['config_data'] == {}
                    assert result['execution_order'] == []
    
    def test_single_node_dag(self):
        """Test with single node DAG (no dependencies)."""
        dag = PipelineDAG()
        dag.add_node('SingleScript')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        
        mock_spec = Mock()
        mock_spec.dependencies = {}
        mock_spec.outputs = {'result': Mock()}
        mock_spec_discovery.load_spec_class.return_value = mock_spec
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver'):
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    with patch('cursus.validation.script_testing.script_dependency_matcher.collect_script_inputs'):
                        result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                        
                        assert len(result['node_specs']) == 1
                        assert 'SingleScript' in result['node_specs']
                        assert result['dependency_matches']['SingleScript'] == {}  # No dependencies
                        assert result['execution_order'] == ['SingleScript']
    
    def test_linear_pipeline(self):
        """Test with linear pipeline (A -> B -> C)."""
        dag = PipelineDAG()
        dag.add_node('A')
        dag.add_node('B')
        dag.add_node('C')
        dag.add_edge('A', 'B')
        dag.add_edge('B', 'C')
        
        mock_step_catalog = Mock()
        mock_spec_discovery = Mock()
        mock_step_catalog.spec_discovery = mock_spec_discovery
        
        def mock_load_spec(node_name):
            if node_name == 'A':
                spec = Mock()
                spec.dependencies = {}
                spec.outputs = {'data': Mock()}
                return spec
            elif node_name == 'B':
                spec = Mock()
                spec.dependencies = {'input_data': Mock()}
                spec.outputs = {'processed_data': Mock()}
                return spec
            elif node_name == 'C':
                spec = Mock()
                spec.dependencies = {'final_input': Mock()}
                spec.outputs = {'result': Mock()}
                return spec
            return None
        
        mock_spec_discovery.load_spec_class.side_effect = mock_load_spec
        
        with patch('cursus.validation.script_testing.script_dependency_matcher.create_dependency_resolver') as mock_resolver_factory:
            mock_resolver = Mock()
            mock_resolver._calculate_compatibility.return_value = 0.8
            mock_resolver_factory.return_value = mock_resolver
            
            with patch('cursus.validation.script_testing.script_dependency_matcher.build_complete_config_classes'):
                with patch('cursus.validation.script_testing.script_dependency_matcher.load_configs'):
                    with patch('cursus.validation.script_testing.script_dependency_matcher.collect_script_inputs'):
                        result = prepare_script_testing_inputs(dag, 'config.json', mock_step_catalog)
                        
                        # Validate linear dependency chain
                        assert result['execution_order'] == ['A', 'B', 'C']
                        assert len(result['dependency_matches']['B']) == 1
                        assert len(result['dependency_matches']['C']) == 1
                        
                        # B should depend on A
                        b_matches = result['dependency_matches']['B']
                        assert 'input_data' in b_matches
                        assert b_matches['input_data']['provider_node'] == 'A'
                        
                        # C should depend on B (but algorithm finds first compatible provider)
                        # Since all providers return 0.8 compatibility, it could be A or B
                        c_matches = result['dependency_matches']['C']
                        assert 'final_input' in c_matches
                        # The provider could be A or B since both have 0.8 compatibility
                        assert c_matches['final_input']['provider_node'] in ['A', 'B']


if __name__ == '__main__':
    pytest.main([__file__])
