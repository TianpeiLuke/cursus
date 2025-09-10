"""
Unit tests for runtime testing system

Tests the RuntimeTester class and its methods for script validation,
data compatibility testing, and pipeline flow validation.
Updated to include tests for enhanced Phase 2/3 functionality.
"""

import sys
from pathlib import Path

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

# Import logical name matching components for enhanced testing
try:
    from cursus.validation.runtime.logical_name_matching import (
        PathSpec,
        PathMatch,
        MatchType,
        EnhancedScriptExecutionSpec,
        PathMatcher,
        TopologicalExecutor,
        LogicalNameMatchingTester,
        EnhancedDataCompatibilityResult
    )
    LOGICAL_MATCHING_AVAILABLE = True
except ImportError:
    LOGICAL_MATCHING_AVAILABLE = False

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
    
    def test_backward_compatibility_test_script_with_spec(self):
        """Test script testing using test_script_with_spec method"""
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
    
    def test_backward_compatibility_test_data_compatibility_with_specs(self):
        """Test data compatibility using ScriptExecutionSpecs (current implementation)"""
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
                # Mock successful script executions
                mock_test_script.side_effect = [
                    ScriptTestResult(script_name="script_a", success=True, execution_time=0.1),
                    ScriptTestResult(script_name="script_b", success=True, execution_time=0.1)
                ]
                
                with patch.object(self.tester, '_find_valid_output_files') as mock_find_files:
                    mock_find_files.return_value = [Path("/test/output/data.csv")]
                    
                    result = self.tester.test_data_compatibility_with_specs(spec_a, spec_b)
                    
                    self.assertIsInstance(result, DataCompatibilityResult)
                    self.assertEqual(result.script_a, "script_a")
                    self.assertEqual(result.script_b, "script_b")
                    self.assertTrue(result.compatible)
    
    def test_pipeline_flow_with_spec_comprehensive(self):
        """Test comprehensive pipeline flow using PipelineTestingSpec (current implementation)"""
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
        """Test executing script with test data using test_script_with_spec"""
        script_spec = ScriptExecutionSpec.create_default("test_script", "test_step", self.temp_dir)
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
                        
                        with patch('pathlib.Path.mkdir'):
                            result = self.tester.test_script_with_spec(script_spec, main_params)
                            
                            self.assertIsInstance(result, ScriptTestResult)
                            self.assertTrue(result.success)
                            self.assertTrue(result.has_main_function)
                            self.assertEqual(result.script_name, "test_script")
                            
                            # Verify main function was called with correct parameters
                            mock_module.main.assert_called_once_with(**main_params)
    
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
        script_spec = ScriptExecutionSpec.create_default("nonexistent_script", "nonexistent_step", self.temp_dir)
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args
        }
        
        with patch.object(self.tester, '_find_script_path') as mock_find:
            mock_find.side_effect = FileNotFoundError("Script not found: nonexistent_script")
            
            result = self.tester.test_script_with_spec(script_spec, main_params)
            
            self.assertFalse(result.success)
            self.assertIn("Script not found: nonexistent_script", result.error_message)
            self.assertEqual(result.script_name, "nonexistent_script")
    
    def test_performance_requirements(self):
        """Test that script testing completes quickly"""
        script_spec = ScriptExecutionSpec.create_default("test_script", "test_step", self.temp_dir)
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
                             patch('pathlib.Path.mkdir'):
                            
                            result = self.tester.test_script_with_spec(script_spec, main_params)
                            
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

class TestEnhancedFileFormatSupport(unittest.TestCase):
    """Test enhanced file format support in RuntimeTester"""
    
    def setUp(self):
        """Set up test fixtures for enhanced file format testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dag = PipelineDAG(nodes=["script_a"], edges=[])
        self.script_spec = ScriptExecutionSpec.create_default("script_a", "script_a_step", self.temp_dir)
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.test_dag,
            script_specs={"script_a": self.script_spec},
            test_workspace_root=self.temp_dir
        )
        self.config = RuntimeTestingConfiguration(pipeline_spec=self.pipeline_spec)
        self.tester = RuntimeTester(self.config)
    
    def test_find_valid_output_files_csv_only(self):
        """Test finding valid output files with CSV-only approach (Phase 1)"""
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Create test files
        (output_dir / "data.csv").touch()
        (output_dir / "model.pkl").touch()
        (output_dir / "temp_file.tmp").touch()
        (output_dir / ".hidden").touch()
        
        with patch.object(self.tester, '_is_enhanced_mode', return_value=False):
            valid_files = self.tester._find_valid_output_files(output_dir)
            
            # Should only find CSV files in Phase 1
            self.assertEqual(len(valid_files), 1)
            self.assertEqual(valid_files[0].name, "data.csv")
    
    def test_find_valid_output_files_enhanced_mode(self):
        """Test finding valid output files with enhanced format support (Phase 2+)"""
        output_dir = Path(self.temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Create test files
        (output_dir / "data.csv").touch()
        (output_dir / "model.pkl").touch()
        (output_dir / "features.parquet").touch()
        (output_dir / "config.json").touch()
        (output_dir / "temp_file.tmp").touch()
        (output_dir / ".hidden").touch()
        (output_dir / "system.log").touch()
        
        with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
            valid_files = self.tester._find_valid_output_files(output_dir)
            
            # Should find all valid files except temp/system files
            valid_names = {f.name for f in valid_files}
            expected_names = {"data.csv", "model.pkl", "features.parquet", "config.json"}
            self.assertEqual(valid_names, expected_names)
    
    def test_is_temp_or_system_file(self):
        """Test identification of temporary and system files"""
        test_cases = [
            ("data.csv", False),
            ("model.pkl", False),
            ("temp_file.tmp", True),
            (".hidden", True),
            ("system.log", True),
            ("cache.cache", True),
            ("backup.bak", True),
            ("~temp.txt", True),
            ("file.swp", True),
            ("normal_file.json", False),
            ("features.parquet", False)
        ]
        
        for filename, expected_is_temp in test_cases:
            with self.subTest(filename=filename):
                result = self.tester._is_temp_or_system_file(Path(filename))
                self.assertEqual(result, expected_is_temp, 
                               f"File {filename} should {'be' if expected_is_temp else 'not be'} temp/system")
    
    def test_enhanced_mode_detection(self):
        """Test enhanced mode detection logic"""
        # Test with logical matching available
        if LOGICAL_MATCHING_AVAILABLE:
            with patch.object(self.tester.config, 'enable_enhanced_features', True):
                self.assertTrue(self.tester._is_enhanced_mode())
            
            with patch.object(self.tester.config, 'enable_enhanced_features', False):
                self.assertFalse(self.tester._is_enhanced_mode())
        else:
            # Should always be False if logical matching not available
            self.assertFalse(self.tester._is_enhanced_mode())


@unittest.skipUnless(LOGICAL_MATCHING_AVAILABLE, "Logical name matching not available")
class TestLogicalNameMatchingIntegration(unittest.TestCase):
    """Test logical name matching integration in RuntimeTester"""
    
    def setUp(self):
        """Set up test fixtures for logical name matching integration"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create DAG with logical name dependencies
        self.test_dag = PipelineDAG(
            nodes=["tabular_preprocessing", "xgboost_training"],
            edges=[("tabular_preprocessing", "xgboost_training")]
        )
        
        # Create enhanced script specs with logical names
        self.preprocessing_spec = EnhancedScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing_step",
            workspace_dir=self.temp_dir,
            input_paths={"raw_data": "/input/raw.csv"},
            output_paths={"processed_data": "/output/processed.csv"},
            environ_vars={"PREPROCESSING_MODE": "standard"},
            job_args={"batch_size": "1000"},
            logical_names={"processed_data": ["clean_data", "training_ready_data"]},
            aliases={"processed_data": "prep_output"}
        )
        
        self.training_spec = EnhancedScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training_step",
            workspace_dir=self.temp_dir,
            input_paths={
                "training_data": "/input/training.csv",
                "hyperparameter_s3": "/config/hyperparams.json"  # Independent input
            },
            output_paths={"model_output": "/output/model.pkl"},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"},
            logical_names={"training_data": ["processed_data", "clean_data"]},
            aliases={"training_data": "train_input"}
        )
        
        self.script_specs = {
            "tabular_preprocessing": self.preprocessing_spec,
            "xgboost_training": self.training_spec
        }
        
        # Create pipeline spec with enhanced features enabled
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.test_dag,
            script_specs=self.script_specs,
            test_workspace_root=self.temp_dir
        )
        
        self.config = RuntimeTestingConfiguration(
            pipeline_spec=self.pipeline_spec,
            enable_enhanced_features=True
        )
        self.tester = RuntimeTester(self.config)
    
    def test_enhanced_data_compatibility_with_logical_matching(self):
        """Test enhanced data compatibility using logical name matching"""
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {"raw_data": "/input/raw.csv"},
                "output_paths": {"processed_data": "/output/processed.csv"},
                "environ_vars": {"PREPROCESSING_MODE": "standard"},
                "job_args": {"batch_size": "1000"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                # Mock successful script executions
                mock_test_script.side_effect = [
                    ScriptTestResult(script_name="tabular_preprocessing", success=True, execution_time=0.1),
                    ScriptTestResult(script_name="xgboost_training", success=True, execution_time=0.1)
                ]
                
                with patch('pathlib.Path.glob') as mock_glob:
                    mock_glob.return_value = [Path("/output/processed.csv")]
                    
                    with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
                        result = self.tester.test_data_compatibility_with_specs(
                            self.preprocessing_spec, 
                            self.training_spec
                        )
                        
                        # Should be enhanced result with logical matching details
                        self.assertIsInstance(result, EnhancedDataCompatibilityResult)
                        self.assertTrue(result.compatible)
                        self.assertEqual(result.script_a, "tabular_preprocessing")
                        self.assertEqual(result.script_b, "xgboost_training")
                        
                        # Check logical matching details
                        self.assertIsNotNone(result.logical_matches)
                        self.assertGreater(len(result.logical_matches), 0)
    
    def test_independent_input_handling(self):
        """Test that independent inputs (like hyperparameter_s3) are handled correctly"""
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {
                    "training_data": "/input/training.csv",
                    "hyperparameter_s3": "/config/hyperparams.json"
                },
                "output_paths": {"model_output": "/output/model.pkl"},
                "environ_vars": {"MODEL_TYPE": "xgboost"},
                "job_args": {"max_depth": "6"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                mock_test_script.side_effect = [
                    ScriptTestResult(script_name="tabular_preprocessing", success=True, execution_time=0.1),
                    ScriptTestResult(script_name="xgboost_training", success=True, execution_time=0.1)
                ]
                
                with patch('pathlib.Path.glob') as mock_glob:
                    mock_glob.return_value = [Path("/output/processed.csv")]
                    
                    with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
                        result = self.tester.test_data_compatibility_with_specs(
                            self.preprocessing_spec, 
                            self.training_spec
                        )
                        
                        # Should still be compatible despite independent input
                        self.assertTrue(result.compatible)
                        
                        # Independent inputs should be preserved
                        if hasattr(result, 'independent_inputs'):
                            self.assertIn("hyperparameter_s3", result.independent_inputs)
    
    def test_fallback_to_basic_mode(self):
        """Test fallback to basic mode when enhanced features fail"""
        with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
            with patch('cursus.validation.runtime.logical_name_matching.LogicalNameMatchingTester') as mock_tester:
                # Mock enhanced tester failure
                mock_tester.side_effect = Exception("Enhanced mode failed")
                
                with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
                    mock_params.return_value = {}
                    
                    with patch.object(self.tester, 'test_script_with_spec') as mock_test_script:
                        mock_test_script.side_effect = [
                            ScriptTestResult(script_name="tabular_preprocessing", success=True, execution_time=0.1),
                            ScriptTestResult(script_name="xgboost_training", success=True, execution_time=0.1)
                        ]
                        
                        with patch('pathlib.Path.glob') as mock_glob:
                            mock_glob.return_value = [Path("/output/processed.csv")]
                            
                            result = self.tester.test_data_compatibility_with_specs(
                                self.preprocessing_spec, 
                                self.training_spec
                            )
                            
                            # Should fallback to basic DataCompatibilityResult
                            self.assertIsInstance(result, DataCompatibilityResult)
                            self.assertNotIsInstance(result, EnhancedDataCompatibilityResult)


@unittest.skipUnless(LOGICAL_MATCHING_AVAILABLE, "Logical name matching not available")
class TestTopologicalExecution(unittest.TestCase):
    """Test topological execution capabilities"""
    
    def setUp(self):
        """Set up test fixtures for topological execution testing"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create complex DAG for topological testing
        self.complex_dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval")
            ]
        )
        
        # Create enhanced script specs
        self.script_specs = {}
        for i, node in enumerate(self.complex_dag.nodes):
            self.script_specs[node] = EnhancedScriptExecutionSpec(
                script_name=node,
                step_name=f"{node}_step",
                workspace_dir=self.temp_dir,
                input_paths={"input": f"/input/{node}.csv"},
                output_paths={"output": f"/output/{node}.csv"},
                environ_vars={"STEP": node},
                job_args={"step_id": str(i)},
                logical_names={},
                aliases={}
            )
        
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.complex_dag,
            script_specs=self.script_specs,
            test_workspace_root=self.temp_dir
        )
        
        self.config = RuntimeTestingConfiguration(
            pipeline_spec=self.pipeline_spec,
            enable_enhanced_features=True
        )
        self.tester = RuntimeTester(self.config)
    
    def test_topological_execution_order(self):
        """Test that pipeline execution follows topological order"""
        execution_order = []
        
        def mock_test_script(spec, params):
            execution_order.append(spec.script_name)
            return ScriptTestResult(script_name=spec.script_name, success=True, execution_time=0.1)
        
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {"input": "/test/input.csv"},
                "output_paths": {"output": "/test/output.csv"},
                "environ_vars": {"STEP": "test"},
                "job_args": {"step_id": "0"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec', side_effect=mock_test_script):
                with patch.object(self.tester, 'test_data_compatibility_with_specs') as mock_compat:
                    mock_compat.return_value = DataCompatibilityResult(
                        script_a="", script_b="", compatible=True
                    )
                    
                    with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
                        result = self.tester.test_pipeline_flow_with_spec(self.pipeline_spec)
                        
                        # Verify execution order follows topological sort
                        expected_order = ["data_prep", "feature_eng", "model_train", "model_eval"]
                        self.assertEqual(execution_order, expected_order)
                        self.assertTrue(result["pipeline_success"])
    
    def test_topological_execution_with_failure(self):
        """Test topological execution handles failures gracefully"""
        def mock_test_script(spec, params):
            if spec.script_name == "feature_eng":
                return ScriptTestResult(
                    script_name=spec.script_name, 
                    success=False, 
                    error_message="Feature engineering failed",
                    execution_time=0.1
                )
            return ScriptTestResult(script_name=spec.script_name, success=True, execution_time=0.1)
        
        with patch.object(self.tester.builder, 'get_script_main_params') as mock_params:
            mock_params.return_value = {
                "input_paths": {"input": "/test/input.csv"},
                "output_paths": {"output": "/test/output.csv"},
                "environ_vars": {"STEP": "test"},
                "job_args": {"step_id": "0"}
            }
            
            with patch.object(self.tester, 'test_script_with_spec', side_effect=mock_test_script):
                with patch.object(self.tester, '_is_enhanced_mode', return_value=True):
                    result = self.tester.test_pipeline_flow_with_spec(self.pipeline_spec)
                    
                    # Pipeline should fail due to feature_eng failure
                    self.assertFalse(result["pipeline_success"])
                    self.assertGreater(len(result["errors"]), 0)
                    
                    # Should have results for scripts that were tested
                    script_names = [r.script_name for r in result["script_results"]]
                    self.assertIn("data_prep", script_names)
                    self.assertIn("feature_eng", script_names)


class TestRuntimeTesterErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in RuntimeTester"""
    
    def setUp(self):
        """Set up test fixtures for error handling tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dag = PipelineDAG(nodes=["script_a"], edges=[])
        self.script_spec = ScriptExecutionSpec.create_default("script_a", "script_a_step", self.temp_dir)
        self.pipeline_spec = PipelineTestingSpec(
            dag=self.test_dag,
            script_specs={"script_a": self.script_spec},
            test_workspace_root=self.temp_dir
        )
        self.config = RuntimeTestingConfiguration(pipeline_spec=self.pipeline_spec)
        self.tester = RuntimeTester(self.config)
    
    def test_script_import_error(self):
        """Test handling of script import errors"""
        script_spec = ScriptExecutionSpec.create_default("test_script", "test_step", self.temp_dir)
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args
        }
        
        with patch.object(self.tester, '_find_script_path', return_value="test_script.py"):
            with patch('importlib.util.spec_from_file_location', side_effect=ImportError("Module not found")):
                result = self.tester.test_script_with_spec(script_spec, main_params)
                
                self.assertFalse(result.success)
                self.assertIn("Module not found", result.error_message)
                self.assertEqual(result.script_name, "test_script")
    
    def test_script_execution_error(self):
        """Test handling of script execution errors"""
        script_spec = ScriptExecutionSpec.create_default("test_script", "test_step", self.temp_dir)
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args
        }
        
        with patch.object(self.tester, '_find_script_path', return_value="test_script.py"):
            with patch('importlib.util.spec_from_file_location') as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock(side_effect=RuntimeError("Script execution failed"))
                
                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj
                
                with patch('importlib.util.module_from_spec', return_value=mock_module):
                    with patch('inspect.signature') as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            'input_paths', 'output_paths', 'environ_vars', 'job_args'
                        ]
                        
                        result = self.tester.test_script_with_spec(script_spec, main_params)
                        
                        self.assertFalse(result.success)
                        self.assertIn("Script execution failed", result.error_message)
    
    def test_invalid_pipeline_spec(self):
        """Test handling of invalid pipeline specifications"""
        # Create invalid pipeline spec with missing script spec
        invalid_dag = PipelineDAG(nodes=["missing_script"], edges=[])
        invalid_pipeline_spec = PipelineTestingSpec(
            dag=invalid_dag,
            script_specs={},  # Missing script spec
            test_workspace_root=self.temp_dir
        )
        
        result = self.tester.test_pipeline_flow_with_spec(invalid_pipeline_spec)
        
        self.assertFalse(result["pipeline_success"])
        self.assertGreater(len(result["errors"]), 0)
        self.assertIn("missing_script", str(result["errors"]))
    
    def test_workspace_permission_error(self):
        """Test handling of workspace permission errors"""
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with patch.object(self.tester, '_find_script_path', return_value="test_script.py"):
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
                            
                            script_spec = ScriptExecutionSpec.create_default("test_script", "test_step", self.temp_dir)
                            main_params = {
                                "input_paths": script_spec.input_paths,
                                "output_paths": script_spec.output_paths,
                                "environ_vars": script_spec.environ_vars,
                                "job_args": script_spec.job_args
                            }
                            
                            result = self.tester.test_script_with_spec(script_spec, main_params)
                            
                            self.assertFalse(result.success)
                            self.assertIn("Permission denied", result.error_message)


if __name__ == '__main__':
    unittest.main()
