"""
Unit tests for runtime spec builder

Tests the PipelineTestingSpecBuilder class used for building pipeline testing
specifications from DAG structures.
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
from unittest.mock import Mock, patch, MagicMock

from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from cursus.validation.runtime.runtime_models import (
    ScriptExecutionSpec,
    PipelineTestingSpec
)
from cursus.api.dag.base_dag import PipelineDAG


class TestPipelineTestingSpecBuilder(unittest.TestCase):
    """Test PipelineTestingSpecBuilder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple DAG for testing
        self.test_dag = PipelineDAG(
            nodes=["script_a", "script_b", "script_c"],
            edges=[("script_a", "script_b"), ("script_b", "script_c")]
        )
        
        # Create builder with temporary directory
        self.builder = PipelineTestingSpecBuilder(test_data_dir=self.temp_dir)
    
    def test_builder_initialization(self):
        """Test PipelineTestingSpecBuilder initialization"""
        builder = PipelineTestingSpecBuilder()
        
        self.assertEqual(str(builder.test_data_dir), "test/integration/runtime")
        self.assertTrue(builder.specs_dir.name == ".specs")
    
    def test_builder_initialization_with_params(self):
        """Test PipelineTestingSpecBuilder initialization with parameters"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        builder = PipelineTestingSpecBuilder(test_data_dir=temp_dir)
        
        self.assertEqual(str(builder.test_data_dir), temp_dir)
        self.assertTrue(builder.specs_dir.name == ".specs")
    
    def test_build_from_dag(self):
        """Test building PipelineTestingSpec from DAG"""
        pipeline_spec = self.builder.build_from_dag(self.test_dag, validate=False)
        
        self.assertIsInstance(pipeline_spec, PipelineTestingSpec)
        self.assertEqual(pipeline_spec.dag, self.test_dag)
        self.assertEqual(len(pipeline_spec.script_specs), 3)
        
        # Verify all scripts have specs
        for script_name in ["script_a", "script_b", "script_c"]:
            self.assertIn(script_name, pipeline_spec.script_specs)
            spec = pipeline_spec.script_specs[script_name]
            self.assertEqual(spec.script_name, script_name)
    
    def test_load_or_create_script_spec_new(self):
        """Test loading or creating script spec when none exists"""
        spec = self.builder._load_or_create_script_spec("new_script")
        
        self.assertIsInstance(spec, ScriptExecutionSpec)
        self.assertEqual(spec.script_name, "new_script")
        self.assertEqual(spec.step_name, "new_script")
        self.assertIn("data_input", spec.input_paths)
        self.assertIn("data_output", spec.output_paths)
    
    def test_load_or_create_script_spec_existing(self):
        """Test loading existing script spec"""
        # Create and save a spec first
        original_spec = ScriptExecutionSpec(
            script_name="existing_script",
            step_name="existing_step",
            input_paths={"custom_input": "/custom/path"},
            output_paths={"custom_output": "/custom/output"},
            environ_vars={"CUSTOM_VAR": "custom_value"},
            job_args={"custom_arg": "custom_value"}
        )
        original_spec.save_to_file(str(self.builder.specs_dir))
        
        # Load the spec using the builder
        loaded_spec = self.builder._load_or_create_script_spec("existing_script")
        
        self.assertEqual(loaded_spec.script_name, "existing_script")
        self.assertEqual(loaded_spec.step_name, "existing_step")
        self.assertEqual(loaded_spec.input_paths["custom_input"], "/custom/path")
        self.assertEqual(loaded_spec.environ_vars["CUSTOM_VAR"], "custom_value")
    
    def test_save_script_spec(self):
        """Test saving script spec"""
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        
        # Save the spec
        self.builder.save_script_spec(spec)
        
        # Verify it was saved
        saved_specs = self.builder.list_saved_specs()
        self.assertIn("test_script", saved_specs)
    
    def test_update_script_spec(self):
        """Test updating script spec"""
        # Create initial spec
        original_spec = ScriptExecutionSpec.create_default("update_test", "update_step")
        self.builder.save_script_spec(original_spec)
        
        # Update the spec
        updated_spec = self.builder.update_script_spec(
            "update_test",
            input_paths={"new_input": "/new/path"},
            environ_vars={"NEW_VAR": "new_value"}
        )
        
        self.assertEqual(updated_spec.input_paths["new_input"], "/new/path")
        self.assertEqual(updated_spec.environ_vars["NEW_VAR"], "new_value")
    
    def test_list_saved_specs(self):
        """Test listing saved specs"""
        # Create and save some specs
        spec1 = ScriptExecutionSpec.create_default("spec1", "step1")
        spec2 = ScriptExecutionSpec.create_default("spec2", "step2")
        
        self.builder.save_script_spec(spec1)
        self.builder.save_script_spec(spec2)
        
        saved_specs = self.builder.list_saved_specs()
        
        self.assertIn("spec1", saved_specs)
        self.assertIn("spec2", saved_specs)
        self.assertEqual(len(saved_specs), 2)
    
    def test_get_script_spec_by_name(self):
        """Test getting script spec by name"""
        # Create and save a spec
        original_spec = ScriptExecutionSpec.create_default("get_test", "get_step")
        self.builder.save_script_spec(original_spec)
        
        # Get the spec by name
        retrieved_spec = self.builder.get_script_spec_by_name("get_test")
        
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.script_name, "get_test")
        self.assertEqual(retrieved_spec.step_name, "get_step")
    
    def test_get_script_spec_by_name_nonexistent(self):
        """Test getting non-existent script spec returns None"""
        retrieved_spec = self.builder.get_script_spec_by_name("nonexistent")
        self.assertIsNone(retrieved_spec)
    
    def test_match_step_to_spec_direct_match(self):
        """Test direct step to spec matching"""
        available_specs = ["script_a", "script_b", "script_c"]
        
        match = self.builder.match_step_to_spec("script_a", available_specs)
        self.assertEqual(match, "script_a")
    
    def test_match_step_to_spec_variation_match(self):
        """Test step to spec matching with variations"""
        available_specs = ["script_a", "script_b", "script_c"]
        
        # Test lowercase variation
        match = self.builder.match_step_to_spec("SCRIPT_A", available_specs)
        self.assertEqual(match, "script_a")
    
    def test_match_step_to_spec_no_match(self):
        """Test step to spec matching with no match"""
        available_specs = ["script_a", "script_b", "script_c"]
        
        match = self.builder.match_step_to_spec("completely_different", available_specs)
        self.assertIsNone(match)
    
    def test_is_spec_complete_valid(self):
        """Test spec completeness validation for valid spec"""
        spec = ScriptExecutionSpec(
            script_name="complete_script",
            step_name="complete_step",
            input_paths={"data_input": "/valid/input"},
            output_paths={"data_output": "/valid/output"},
            environ_vars={"LABEL_FIELD": "label"},
            job_args={"job_type": "testing"}
        )
        
        is_complete = self.builder._is_spec_complete(spec)
        self.assertTrue(is_complete)
    
    def test_is_spec_complete_invalid(self):
        """Test spec completeness validation for invalid spec"""
        spec = ScriptExecutionSpec(
            script_name="incomplete_script",
            step_name="incomplete_step",
            input_paths={},  # Empty paths
            output_paths={},  # Empty paths
            environ_vars={},
            job_args={}
        )
        
        is_complete = self.builder._is_spec_complete(spec)
        self.assertFalse(is_complete)
    
    def test_validate_specs_completeness_valid(self):
        """Test validation with complete specs"""
        dag_nodes = ["script_a", "script_b"]
        missing_specs = []
        incomplete_specs = []
        
        # Should not raise exception
        try:
            self.builder._validate_specs_completeness(dag_nodes, missing_specs, incomplete_specs)
        except ValueError:
            self.fail("_validate_specs_completeness raised ValueError unexpectedly")
    
    def test_validate_specs_completeness_invalid(self):
        """Test validation with incomplete specs"""
        dag_nodes = ["script_a", "script_b"]
        missing_specs = ["script_a"]
        incomplete_specs = ["script_b"]
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.builder._validate_specs_completeness(dag_nodes, missing_specs, incomplete_specs)
        
        error_message = str(context.exception)
        self.assertIn("Missing ScriptExecutionSpec", error_message)
        self.assertIn("Incomplete ScriptExecutionSpec", error_message)
    
    def test_get_script_main_params(self):
        """Test getting script main parameters"""
        spec = ScriptExecutionSpec(
            script_name="param_test",
            step_name="param_step",
            input_paths={"data_input": "/test/input"},
            output_paths={"data_output": "/test/output"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"}
        )
        
        params = self.builder.get_script_main_params(spec)
        
        self.assertEqual(params["input_paths"]["data_input"], "/test/input")
        self.assertEqual(params["output_paths"]["data_output"], "/test/output")
        self.assertEqual(params["environ_vars"]["TEST_VAR"], "test_value")
        self.assertEqual(params["job_args"].test_arg, "test_value")
    
    @patch('builtins.input')
    def test_update_script_spec_interactive(self, mock_input):
        """Test interactive script spec update"""
        # Mock user inputs
        mock_input.side_effect = [
            "/interactive/input",  # input path
            "/interactive/output",  # output path
            "",  # environment variables (use defaults)
            ""   # job arguments (use defaults)
        ]
        
        # Create a spec with empty paths
        spec = ScriptExecutionSpec(
            script_name="interactive_test",
            step_name="interactive_step",
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args={}
        )
        self.builder.save_script_spec(spec)
        
        # Update interactively
        updated_spec = self.builder.update_script_spec_interactive("interactive_test")
        
        self.assertEqual(updated_spec.input_paths["data_input"], "/interactive/input")
        self.assertEqual(updated_spec.output_paths["data_output"], "/interactive/output")
        self.assertEqual(updated_spec.environ_vars["LABEL_FIELD"], "label")
        self.assertEqual(updated_spec.job_args["job_type"], "testing")


class TestPipelineTestingSpecBuilderIntegration(unittest.TestCase):
    """Integration tests for PipelineTestingSpecBuilder"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = PipelineTestingSpecBuilder(test_data_dir=self.temp_dir)
    
    def test_end_to_end_spec_building(self):
        """Test complete end-to-end spec building workflow"""
        # Create a complex DAG
        complex_dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval")
            ]
        )
        
        # Build pipeline spec (without validation to avoid completeness issues)
        pipeline_spec = self.builder.build_from_dag(complex_dag, validate=False)
        
        # Verify the complete spec
        self.assertEqual(len(pipeline_spec.script_specs), 4)
        self.assertEqual(len(pipeline_spec.dag.nodes), 4)
        self.assertEqual(len(pipeline_spec.dag.edges), 3)
        
        # Verify each script has a spec
        for node in complex_dag.nodes:
            self.assertIn(node, pipeline_spec.script_specs)
            spec = pipeline_spec.script_specs[node]
            self.assertEqual(spec.script_name, node)
            self.assertEqual(spec.step_name, node)
    
    def test_spec_persistence_and_updates(self):
        """Test specification persistence and update workflow"""
        # Create initial DAG and spec
        dag = PipelineDAG(nodes=["script_a"], edges=[])
        initial_spec = self.builder.build_from_dag(dag, validate=False)
        
        # Modify script spec and save
        script_spec = initial_spec.script_specs["script_a"]
        script_spec.environ_vars["UPDATED_VAR"] = "updated_value"
        self.builder.save_script_spec(script_spec)
        
        # Build new spec (should load updated script spec)
        updated_spec = self.builder.build_from_dag(dag, validate=False)
        
        # Verify the update was preserved
        self.assertEqual(
            updated_spec.script_specs["script_a"].environ_vars["UPDATED_VAR"],
            "updated_value"
        )


if __name__ == '__main__':
    unittest.main()
