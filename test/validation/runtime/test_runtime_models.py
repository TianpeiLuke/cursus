"""
Unit tests for runtime testing models

Tests the Pydantic models used in runtime testing including ScriptExecutionSpec,
PipelineTestingSpec, and RuntimeTestingConfiguration.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import unittest
import tempfile
import json
from datetime import datetime

from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration
)
from cursus.api.dag.base_dag import PipelineDAG

class TestScriptTestResult(unittest.TestCase):
    """Test ScriptTestResult model"""
    
    def test_script_test_result_creation(self):
        """Test creating a ScriptTestResult"""
        result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=0.5,
            has_main_function=True
        )
        
        self.assertEqual(result.script_name, "test_script")
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 0.5)
        self.assertTrue(result.has_main_function)
        self.assertIsNone(result.error_message)
    
    def test_script_test_result_with_error(self):
        """Test creating a ScriptTestResult with error"""
        result = ScriptTestResult(
            script_name="broken_script",
            success=False,
            error_message="Script failed to execute",
            execution_time=0.1,
            has_main_function=False
        )
        
        self.assertEqual(result.script_name, "broken_script")
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Script failed to execute")
        self.assertEqual(result.execution_time, 0.1)
        self.assertFalse(result.has_main_function)

class TestDataCompatibilityResult(unittest.TestCase):
    """Test DataCompatibilityResult model"""
    
    def test_data_compatibility_result_creation(self):
        """Test creating a DataCompatibilityResult"""
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            data_format_a="csv",
            data_format_b="csv"
        )
        
        self.assertEqual(result.script_a, "script_a")
        self.assertEqual(result.script_b, "script_b")
        self.assertTrue(result.compatible)
        self.assertEqual(result.data_format_a, "csv")
        self.assertEqual(result.data_format_b, "csv")
        self.assertEqual(result.compatibility_issues, [])
    
    def test_data_compatibility_result_with_issues(self):
        """Test creating a DataCompatibilityResult with compatibility issues"""
        issues = ["Column mismatch", "Type error"]
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            compatibility_issues=issues
        )
        
        self.assertEqual(result.script_a, "script_a")
        self.assertEqual(result.script_b, "script_b")
        self.assertFalse(result.compatible)
        self.assertEqual(result.compatibility_issues, issues)

class TestScriptExecutionSpec(unittest.TestCase):
    """Test ScriptExecutionSpec model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.specs_dir = Path(self.temp_dir) / ".specs"
        self.specs_dir.mkdir(parents=True, exist_ok=True)
    
    def test_script_execution_spec_creation(self):
        """Test creating a ScriptExecutionSpec"""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_paths={"data_input": "/path/to/input"},
            output_paths={"data_output": "/path/to/output"},
            environ_vars={"LABEL_FIELD": "label"},
            job_args={"job_type": "testing"}
        )
        
        self.assertEqual(spec.script_name, "test_script")
        self.assertEqual(spec.step_name, "test_step")
        self.assertEqual(spec.input_paths["data_input"], "/path/to/input")
        self.assertEqual(spec.output_paths["data_output"], "/path/to/output")
        self.assertEqual(spec.environ_vars["LABEL_FIELD"], "label")
        self.assertEqual(spec.job_args["job_type"], "testing")
    
    def test_create_default_spec(self):
        """Test creating a default ScriptExecutionSpec"""
        spec = ScriptExecutionSpec.create_default(
            script_name="default_script",
            step_name="default_step",
            test_data_dir="test/data"
        )
        
        self.assertEqual(spec.script_name, "default_script")
        self.assertEqual(spec.step_name, "default_step")
        self.assertEqual(spec.input_paths["data_input"], "test/data/default_script/input")
        self.assertEqual(spec.output_paths["data_output"], "test/data/default_script/output")
        self.assertEqual(spec.environ_vars["LABEL_FIELD"], "label")
        self.assertEqual(spec.job_args["job_type"], "testing")
    
    def test_save_and_load_spec(self):
        """Test saving and loading ScriptExecutionSpec"""
        original_spec = ScriptExecutionSpec(
            script_name="save_test",
            step_name="save_step",
            input_paths={"data_input": "/test/input"},
            output_paths={"data_output": "/test/output"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"}
        )
        
        # Save the spec
        saved_path = original_spec.save_to_file(str(self.specs_dir))
        self.assertTrue(Path(saved_path).exists())
        
        # Load the spec
        loaded_spec = ScriptExecutionSpec.load_from_file("save_test", str(self.specs_dir))
        
        # Verify loaded spec matches original
        self.assertEqual(loaded_spec.script_name, original_spec.script_name)
        self.assertEqual(loaded_spec.step_name, original_spec.step_name)
        self.assertEqual(loaded_spec.input_paths, original_spec.input_paths)
        self.assertEqual(loaded_spec.output_paths, original_spec.output_paths)
        self.assertEqual(loaded_spec.environ_vars, original_spec.environ_vars)
        self.assertEqual(loaded_spec.job_args, original_spec.job_args)
        self.assertIsNotNone(loaded_spec.last_updated)
    
    def test_load_nonexistent_spec(self):
        """Test loading a non-existent ScriptExecutionSpec"""
        with self.assertRaises(FileNotFoundError):
            ScriptExecutionSpec.load_from_file("nonexistent", str(self.specs_dir))
    
    def test_filename_generation(self):
        """Test that filename is generated correctly"""
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        saved_path = spec.save_to_file(str(self.specs_dir))
        
        expected_filename = "test_script_runtime_test_spec.json"
        self.assertTrue(saved_path.endswith(expected_filename))

class TestPipelineTestingSpec(unittest.TestCase):
    """Test PipelineTestingSpec model"""
    
    def test_pipeline_testing_spec_creation(self):
        """Test creating a PipelineTestingSpec"""
        dag = PipelineDAG(
            nodes=["script_a", "script_b"],
            edges=[("script_a", "script_b")]
        )
        
        spec_a = ScriptExecutionSpec.create_default("script_a", "step_a")
        spec_b = ScriptExecutionSpec.create_default("script_b", "step_b")
        
        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={"script_a": spec_a, "script_b": spec_b},
            test_workspace_root="test/workspace"
        )
        
        self.assertEqual(pipeline_spec.dag.nodes, ["script_a", "script_b"])
        self.assertEqual(pipeline_spec.dag.edges, [("script_a", "script_b")])
        self.assertEqual(len(pipeline_spec.script_specs), 2)
        self.assertIn("script_a", pipeline_spec.script_specs)
        self.assertIn("script_b", pipeline_spec.script_specs)
        self.assertEqual(pipeline_spec.test_workspace_root, "test/workspace")

class TestRuntimeTestingConfiguration(unittest.TestCase):
    """Test RuntimeTestingConfiguration model"""
    
    def test_runtime_testing_configuration_creation(self):
        """Test creating a RuntimeTestingConfiguration"""
        dag = PipelineDAG(nodes=["test_script"], edges=[])
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={"test_script": spec}
        )
        
        config = RuntimeTestingConfiguration(
            pipeline_spec=pipeline_spec,
            test_individual_scripts=True,
            test_data_compatibility=True,
            test_pipeline_flow=True,
            use_workspace_aware=False
        )
        
        self.assertEqual(config.pipeline_spec, pipeline_spec)
        self.assertTrue(config.test_individual_scripts)
        self.assertTrue(config.test_data_compatibility)
        self.assertTrue(config.test_pipeline_flow)
        self.assertFalse(config.use_workspace_aware)
    
    def test_runtime_testing_configuration_defaults(self):
        """Test RuntimeTestingConfiguration with default values"""
        dag = PipelineDAG(nodes=["test_script"], edges=[])
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={"test_script": spec}
        )
        
        config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
        
        # Test default values
        self.assertTrue(config.test_individual_scripts)
        self.assertTrue(config.test_data_compatibility)
        self.assertTrue(config.test_pipeline_flow)
        self.assertFalse(config.use_workspace_aware)

if __name__ == '__main__':
    unittest.main()
