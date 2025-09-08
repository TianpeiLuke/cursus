"""
Unit tests for runtime testing system

Tests the RuntimeTester class and its methods for script validation,
data compatibility testing, and pipeline flow validation.
"""

import sys
from pathlib import Path

)

import unittest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration
)
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from cursus.api.dag.base_dag import PipelineDAG

class TestRuntimeTester(unittest.TestCase):
    """Test RuntimeTester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple DAG for testing
        self.test_dag = PipelineDAG(
            nodes=["script_a", "script_b"],
            edges=[("script_a", "script_b")]
        )
        
        # Create script specs
        self.script_spec_a = ScriptExecutionSpec.create_default("script_a", "script_a_step", self.temp_dir)
        self.script_spec_b = ScriptExecutionSpec.create_default("script_b", "script_b_step", self.temp_dir)
        
        # Create pipeline spec
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.test_dag,
            script_specs={"script_a": self.script_spec_a, "script_b": self.script_spec_b},
            test_workspace_root=self.temp_dir
        )
        
        # Create runtime configuration
        self.config = RuntimeTestingConfiguration(pipeline_spec=self.pipeline_spec)
        
        # Create RuntimeTester instance
        self.tester = RuntimeTester(self.config)
    
    def test_runtime_tester_initialization(self):
        """Test RuntimeTester initialization"""
        self.assertEqual(self.tester.config, self.config)
        self.assertEqual(self.tester.pipeline_spec, self.pipeline_spec)
        self.assertEqual(self.tester.workspace_dir, Path(self.temp_dir))
        self.assertIsInstance(self.tester.builder, PipelineTestingSpecBuilder)
    
    def test_script_functionality_validation_with_spec(self):
        """Test script import and main function validation using ScriptExecutionSpec"""
        script_spec = self.script_spec_a
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args
        }
        
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        with patch('pandas.DataFrame.to_csv'), \
                             patch('pathlib.Path.mkdir'), \
                             patch('pathlib.Path.exists', return_value=False):
                            
                            result = self.tester.test_script_with_spec(script_spec, main_params)
                            
                            self.assertIsInstance(result, ScriptTestResult)
                            self.assertTrue(result.success)
                            self.assertTrue(result.has_main_function)
                            self.assertEqual(result.script_name, "script_a")
                            
                            # Verify that main function was actually called
                            mock_module.main.assert_called_once_with(**main_params)
    
    def test_script_missing_main_function_with_spec(self):
        """Test script without main function fails validation with spec"""
        script_spec = self.script_spec_a
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args
        }
        
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                # No main function
                del mock_module.main
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    result = self.tester.test_script_with_spec(script_spec, main_params)
                    
                    self.assertFalse(result.success)
                    self.assertFalse(result.has_main_function)
                    self.assertIn("missing main() function", result.error_message)
    
    def test_data_compatibility_with_specs(self):
        """Test data compatibility between scripts using ScriptExecutionSpecs"""
        spec_a = self.script_spec_a
        spec_b = self.script_spec_b
        
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {"data_input": "/test/input"},
                "output_paths": {"data_output": "/test/output"},
                "environ_vars": {"LABEL_FIELD": "label"},
                "job_args": {"job_type": "testing"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                # Mock successful script A execution
                mock_test_script.side_effect = [
                    ScriptTestResult(script_name="script_a", success=True, execution_time=0.1),
                    ScriptTestResult(script_name="script_b", success=True, execution_time=0.1)
                ]
                
                with patch('pathlib.Path.glob') as mock_glob:
                    mock_glob.return_value = [Path("/test/output/data.csv")]
                    
                    result = self.tester.test_data_compatibility_with_specs(spec_a, spec_b)
                    
                    self.assertIsInstance(result, DataCompatibilityResult)
                    self.assertEqual(result.script_a, "script_a")
                    self.assertEqual(result.script_b, "script_b")
                    self.assertTrue(result.compatible)
    
    def test_data_compatibility_script_a_fails(self):
        """Test data compatibility when script A fails"""
        spec_a = self.script_spec_a
        spec_b = self.script_spec_b
        
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {}
            
            with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                # Mock script A failure
                mock_test_script.return_value = ScriptTestResult(
                    script_name="script_a", 
                    success=False, 
                    error_message="Script failed",
                    execution_time=0.1
                )
                
                result = self.tester.test_data_compatibility_with_specs(spec_a, spec_b)
                
                self.assertFalse(result.compatible)
                self.assertIn("Script A failed", result.compatibility_issues[0])
    
    def test_pipeline_flow_with_spec(self):
        """Test end-to-end pipeline flow using PipelineTestingSpec"""
        pipeline_spec = self.pipeline_spec
        
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {"data_input": "/test/input"},
                "output_paths": {"data_output": "/test/output"},
                "environ_vars": {"LABEL_FIELD": "label"},
                "job_args": {"job_type": "testing"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                # Mock successful script tests
                mock_test_script.side_effect = [
                    ScriptTestResult(script_name="script_a", success=True, execution_time=0.1),
                    ScriptTestResult(script_name="script_b", success=True, execution_time=0.1)
                ]
                
                with patch.object(self.tester, 'test_data_compatibility_with_specs') as mock_test_compat:
                    # Mock successful data compatibility
                    mock_test_compat.return_value = DataCompatibilityResult(
                        script_a="script_a", script_b="script_b", compatible=True
                    )
                    
                    result = self.tester.test_pipeline_flow_with_spec(pipeline_spec)
                    
                    self.assertTrue(result["pipeline_success"])
                    self.assertEqual(len(result["script_results"]), 2)
                    self.assertEqual(len(result["data_flow_results"]), 1)
                    self.assertEqual(len(result["errors"]), 0)
    
    def test_pipeline_flow_empty_dag(self):
        """Test pipeline flow with empty DAG"""
        empty_dag = PipelineDAG(nodes=[], edges=[])
        empty_pipeline_spec = PipelineTestingSpec(
            dag=empty_dag,
            script_specs={},
            test_workspace_root=self.temp_dir
        )
        
        result = self.tester.test_pipeline_flow_with_spec(empty_pipeline_spec)
        
        self.assertFalse(result["pipeline_success"])
        self.assertIn("No nodes found in pipeline DAG", result["errors"])
    
    def test_backward_compatibility_test_script(self):
        """Test backward compatibility with original test_script method"""
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        with patch('pandas.DataFrame.to_csv'), \
                             patch('pathlib.Path.mkdir'):
                            
                            result = self.tester.test_script("test_script")
                            
                            self.assertIsInstance(result, ScriptTestResult)
                            self.assertTrue(result.success)
                            self.assertTrue(result.has_main_function)
                            self.assertEqual(result.script_name, "test_script")
                            
                            # Verify that main function was actually called
                            mock_module.main.assert_called_once()
    
    def test_backward_compatibility_test_data_compatibility(self):
        """Test backward compatibility with original test_data_compatibility method"""
        sample_data = {"col1": [1, 2], "col2": ["a", "b"]}
        
        with patch.object(self.tester, '_execute_script_with_data') as mock_exec:
            # Mock successful script executions
            mock_exec.side_effect = [
                ScriptTestResult(script_name="script_a", success=True, execution_time=0.1),
                ScriptTestResult(script_name="script_b", success=True, execution_time=0.1)
            ]
            
            with patch('pandas.DataFrame.to_csv'), \
                 patch('pandas.read_csv') as mock_read, \
                 patch('pathlib.Path.exists', return_value=True):
                
                mock_read.return_value = pd.DataFrame(sample_data)
                
                result = self.tester.test_data_compatibility("script_a", "script_b", sample_data)
                
                self.assertIsInstance(result, DataCompatibilityResult)
                self.assertEqual(result.script_a, "script_a")
                self.assertEqual(result.script_b, "script_b")
                self.assertTrue(result.compatible)
    
    def test_backward_compatibility_test_pipeline_flow(self):
        """Test backward compatibility with original test_pipeline_flow method"""
        pipeline_config = {
            "steps": {
                "step1": {"script": "script1.py"},
                "step2": {"script": "script2.py"}
            }
        }
        
        with patch.object(self.tester, 'test_script') as mock_test_script:
            # Mock successful script tests
            mock_test_script.side_effect = [
                ScriptTestResult(script_name="step1", success=True, execution_time=0.1),
                ScriptTestResult(script_name="step2", success=True, execution_time=0.1)
            ]
            
            with patch.object(self.tester, 'test_data_compatibility') as mock_test_compat:
                # Mock successful data compatibility
                mock_test_compat.return_value = DataCompatibilityResult(
                    script_a="step1", script_b="step2", compatible=True
                )
                
                result = self.tester.test_pipeline_flow(pipeline_config)
                
                self.assertTrue(result["pipeline_success"])
                self.assertEqual(len(result["script_results"]), 2)
                self.assertEqual(len(result["data_flow_results"]), 1)
                self.assertEqual(len(result["errors"]), 0)
    
    def test_find_script_path(self):
        """Test script path discovery logic"""
        with patch('pathlib.Path.exists') as mock_exists:
            # Mock exists to return True for the first path
            mock_exists.side_effect = lambda: True
            
            result = self.tester._find_script_path("test_script")
            self.assertEqual(result, "src/cursus/steps/scripts/test_script.py")
    
    def test_find_script_path_not_found(self):
        """Test script path discovery when script doesn't exist"""
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError) as context:
                self.tester._find_script_path("nonexistent_script")
            
            self.assertIn("Script not found: nonexistent_script", str(context.exception))
    
    def test_execute_script_with_data(self):
        """Test executing script with test data"""
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('pathlib.Path.mkdir'):
                        result = self.tester._execute_script_with_data(
                            "test_script", 
                            "/input/path", 
                            "/output/path"
                        )
                        
                        self.assertIsInstance(result, ScriptTestResult)
                        self.assertTrue(result.success)
                        self.assertTrue(result.has_main_function)
                        self.assertEqual(result.script_name, "test_script")
                        
                        # Verify main function was called with correct parameters
                        mock_module.main.assert_called_once()
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        sample_data = self.tester._generate_sample_data()
        
        self.assertIsInstance(sample_data, dict)
        self.assertIn("feature1", sample_data)
        self.assertIn("feature2", sample_data)
        self.assertIn("label", sample_data)
        self.assertEqual(len(sample_data["feature1"]), 5)
        self.assertEqual(len(sample_data["feature2"]), 5)
        self.assertEqual(len(sample_data["label"]), 5)
    
    def test_clear_error_feedback(self):
        """Test error messages are clear and actionable"""
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.side_effect = FileNotFoundError("Script not found: nonexistent_script")
            
            result = self.tester.test_script("nonexistent_script")
            
            self.assertFalse(result.success)
            self.assertIn("Script not found: nonexistent_script", result.error_message)
            self.assertEqual(result.script_name, "nonexistent_script")
    
    def test_performance_requirements(self):
        """Test that script testing completes quickly"""
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.return_value = "test_script.py"
            
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        with patch('pandas.DataFrame.to_csv'), \
                             patch('pathlib.Path.mkdir'):
                            
                            result = self.tester.test_script("test_script")
                            
                            # Should complete very quickly (much less than 100ms)
                            self.assertLess(result.execution_time, 0.1)

class TestRuntimeTesterIntegration(unittest.TestCase):
    """Integration tests for RuntimeTester"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a complex DAG for testing
        self.complex_dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval")
            ]
        )
        
        # Create script specs for all nodes
        self.script_specs = {}
        for node in self.complex_dag.nodes:
            self.script_specs[node] = ScriptExecutionSpec.create_default(
                node, f"{node}_step", self.temp_dir
            )
        
        # Create pipeline spec
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.complex_dag,
            script_specs=self.script_specs,
            test_workspace_root=self.temp_dir
        )
        
        # Create configuration
        self.config = RuntimeTestingConfiguration(pipeline_spec=self.pipeline_spec)
        self.tester = RuntimeTester(self.config)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from script testing to pipeline validation"""
        with patch.object(self.tester, '_find_script_path', return_value="test.py"):
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
                            mock_params.return_value = {
                                "input_paths": {"data_input": "/test/input"},
                                "output_paths": {"data_output": "/test/output"},
                                "environ_vars": {"LABEL_FIELD": "label"},
                                "job_args": {"job_type": "testing"}
                            }
                            
                            with patch('pathlib.Path.glob') as mock_glob:
                                mock_glob.return_value = [Path("/test/output/data.csv")]
                                
                                # Test individual script functionality
                                script_result = self.tester.test_script_with_spec(
                                    self.script_specs["data_prep"],
                                    mock_params.return_value
                                )
                                self.assertTrue(script_result.success)
                                
                                # Test complete pipeline
                                pipeline_result = self.tester.test_pipeline_flow_with_spec(self.pipeline_spec)
                                self.assertTrue(pipeline_result["pipeline_success"])

if __name__ == '__main__':
    unittest.main()
