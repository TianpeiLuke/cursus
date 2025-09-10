"""
Unit tests for runtime testing models

Tests the Pydantic models used in runtime testing including ScriptExecutionSpec,
PipelineTestingSpec, and RuntimeTestingConfiguration.
"""

import sys
from pathlib import Path

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

# Import enhanced models for testing
try:
    from cursus.validation.runtime.logical_name_matching import (
        EnhancedScriptExecutionSpec,
        EnhancedDataCompatibilityResult,
        PathSpec,
        PathMatch,
        MatchType
    )
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False

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

@unittest.skipUnless(ENHANCED_MODELS_AVAILABLE, "Enhanced models not available")
class TestEnhancedScriptExecutionSpec(unittest.TestCase):
    """Test EnhancedScriptExecutionSpec model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_enhanced_script_execution_spec_creation(self):
        """Test creating an EnhancedScriptExecutionSpec with PathSpecs"""
        from cursus.validation.runtime.logical_name_matching import PathSpec
        
        input_spec = PathSpec(
            logical_name="training_data",
            path="/input/data.csv",
            aliases=["processed_data", "clean_data"]
        )
        
        output_spec = PathSpec(
            logical_name="model_output",
            path="/output/model.pkl",
            aliases=["trained_model"]
        )
        
        spec = EnhancedScriptExecutionSpec(
            script_name="enhanced_script",
            step_name="enhanced_step",
            input_path_specs={"training_data": input_spec},
            output_path_specs={"model_output": output_spec},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"}
        )
        
        self.assertEqual(spec.script_name, "enhanced_script")
        self.assertEqual(spec.step_name, "enhanced_step")
        self.assertEqual(spec.input_path_specs["training_data"].logical_name, "training_data")
        self.assertEqual(spec.input_path_specs["training_data"].path, "/input/data.csv")
        self.assertEqual(spec.input_path_specs["training_data"].aliases, ["processed_data", "clean_data"])
        self.assertEqual(spec.output_path_specs["model_output"].logical_name, "model_output")
        self.assertEqual(spec.environ_vars["MODEL_TYPE"], "xgboost")
        self.assertEqual(spec.job_args["max_depth"], "6")
    
    def test_enhanced_spec_backward_compatibility_properties(self):
        """Test backward compatibility properties for input_paths and output_paths"""
        from cursus.validation.runtime.logical_name_matching import PathSpec
        
        input_spec = PathSpec(logical_name="data_input", path="/input/data.csv")
        output_spec = PathSpec(logical_name="data_output", path="/output/data.csv")
        
        spec = EnhancedScriptExecutionSpec(
            script_name="compat_test",
            step_name="compat_step",
            input_path_specs={"data_input": input_spec},
            output_path_specs={"data_output": output_spec}
        )
        
        # Test backward compatibility properties
        self.assertEqual(spec.input_paths["data_input"], "/input/data.csv")
        self.assertEqual(spec.output_paths["data_output"], "/output/data.csv")
    
    def test_enhanced_spec_from_script_execution_spec(self):
        """Test creating EnhancedScriptExecutionSpec from basic ScriptExecutionSpec"""
        # Create basic spec
        basic_spec = ScriptExecutionSpec(
            script_name="convert_test",
            step_name="convert_step",
            input_paths={"data_input": "/input/data.csv"},
            output_paths={"data_output": "/output/data.csv"},
            environ_vars={"TEST_VAR": "test"},
            job_args={"test_arg": "value"}
        )
        
        # Convert to enhanced spec
        enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
            basic_spec,
            input_aliases={"data_input": ["raw_data"]},
            output_aliases={"data_output": ["processed_data"]}
        )
        
        self.assertEqual(enhanced_spec.script_name, "convert_test")
        self.assertEqual(enhanced_spec.step_name, "convert_step")
        self.assertEqual(enhanced_spec.input_paths["data_input"], "/input/data.csv")
        self.assertEqual(enhanced_spec.output_paths["data_output"], "/output/data.csv")
        self.assertEqual(enhanced_spec.input_path_specs["data_input"].aliases, ["raw_data"])
        self.assertEqual(enhanced_spec.output_path_specs["data_output"].aliases, ["processed_data"])


@unittest.skipUnless(ENHANCED_MODELS_AVAILABLE, "Enhanced models not available")
class TestEnhancedDataCompatibilityResult(unittest.TestCase):
    """Test EnhancedDataCompatibilityResult model"""
    
    def test_enhanced_data_compatibility_result_creation(self):
        """Test creating an EnhancedDataCompatibilityResult"""
        # Create sample path matches
        path_matches = [
            PathMatch(
                source_logical_name="processed_data",
                dest_logical_name="training_data",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=0.95,
                matched_source_name="processed_data",
                matched_dest_name="training_data"
            )
        ]
        
        result = EnhancedDataCompatibilityResult(
            script_a="tabular_preprocessing",
            script_b="xgboost_training",
            compatible=True,
            path_matches=path_matches,
            matching_details={"total_matches": 1, "high_confidence_matches": 1}
        )
        
        self.assertEqual(result.script_a, "tabular_preprocessing")
        self.assertEqual(result.script_b, "xgboost_training")
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.path_matches), 1)
        self.assertIsNotNone(result.matching_details)
        
        # Check the path match details
        match = result.path_matches[0]
        self.assertEqual(match.source_logical_name, "processed_data")
        self.assertEqual(match.dest_logical_name, "training_data")
        self.assertEqual(match.match_type, MatchType.EXACT_LOGICAL)
        self.assertEqual(match.confidence, 0.95)
    
    def test_enhanced_result_with_no_matches(self):
        """Test EnhancedDataCompatibilityResult with no path matches"""
        result = EnhancedDataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            path_matches=[],
            matching_details={"total_matches": 0, "recommendations": ["Check logical names"]},
            compatibility_issues=["No compatible outputs found"]
        )
        
        self.assertFalse(result.compatible)
        self.assertEqual(len(result.path_matches), 0)
        self.assertIsNotNone(result.matching_details)
        self.assertIn("No compatible outputs found", result.compatibility_issues)
    
    def test_enhanced_result_multiple_matches(self):
        """Test EnhancedDataCompatibilityResult with multiple path matches"""
        path_matches = [
            PathMatch(
                source_logical_name="processed_data",
                dest_logical_name="training_data",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=0.95,
                matched_source_name="processed_data",
                matched_dest_name="training_data"
            ),
            PathMatch(
                source_logical_name="feature_data",
                dest_logical_name="feature_input",
                match_type=MatchType.SEMANTIC,
                confidence=0.8,
                matched_source_name="feature_data",
                matched_dest_name="feature_input"
            )
        ]
        
        result = EnhancedDataCompatibilityResult(
            script_a="preprocessing",
            script_b="training",
            compatible=True,
            path_matches=path_matches,
            matching_details={"total_matches": 2, "high_confidence_matches": 1}
        )
        
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.path_matches), 2)
        
        # Check match types
        match_types = [match.match_type for match in result.path_matches]
        self.assertIn(MatchType.EXACT_LOGICAL, match_types)
        self.assertIn(MatchType.SEMANTIC, match_types)
        
        # Check confidence scores
        confidences = [match.confidence for match in result.path_matches]
        self.assertIn(0.95, confidences)
        self.assertIn(0.8, confidences)
    
    def test_enhanced_result_inheritance_from_basic(self):
        """Test that EnhancedDataCompatibilityResult has basic fields"""
        result = EnhancedDataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            path_matches=[],
            matching_details={"total_matches": 0}
        )
        
        # Should have all basic fields
        self.assertEqual(result.script_a, "script_a")
        self.assertEqual(result.script_b, "script_b")
        self.assertTrue(result.compatible)
        self.assertEqual(result.compatibility_issues, [])  # Default from base class
        
        # Should also have enhanced fields
        self.assertEqual(result.path_matches, [])
        self.assertIsNotNone(result.matching_details)


@unittest.skipUnless(ENHANCED_MODELS_AVAILABLE, "Enhanced models not available")
class TestPathSpecAndPathMatch(unittest.TestCase):
    """Test PathSpec and PathMatch models"""
    
    def test_path_spec_creation(self):
        """Test creating a PathSpec with all fields"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/data/processed.csv",
            aliases=["prep_output", "clean_data"]
        )
        
        self.assertEqual(spec.logical_name, "processed_data")
        self.assertEqual(spec.path, "/data/processed.csv")
        self.assertEqual(spec.aliases, ["prep_output", "clean_data"])
    
    def test_path_spec_with_minimal_fields(self):
        """Test PathSpec with minimal required fields"""
        spec = PathSpec(
            logical_name="data_input",
            path="/data/input.csv"
        )
        
        self.assertEqual(spec.logical_name, "data_input")
        self.assertEqual(spec.path, "/data/input.csv")
        self.assertEqual(spec.aliases, [])  # Default empty list
    
    def test_path_spec_matches_name_or_alias(self):
        """Test PathSpec matches_name_or_alias method"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/data/processed.csv",
            aliases=["clean_data", "prep_output"]
        )
        
        # Should match logical name
        self.assertTrue(spec.matches_name_or_alias("processed_data"))
        
        # Should match aliases
        self.assertTrue(spec.matches_name_or_alias("clean_data"))
        self.assertTrue(spec.matches_name_or_alias("prep_output"))
        
        # Should not match unrelated names
        self.assertFalse(spec.matches_name_or_alias("raw_data"))
        self.assertFalse(spec.matches_name_or_alias("other_data"))
    
    def test_path_match_creation(self):
        """Test creating a PathMatch"""
        match = PathMatch(
            source_logical_name="processed_data",
            dest_logical_name="training_data",
            match_type=MatchType.EXACT_LOGICAL,
            confidence=0.95,
            matched_source_name="processed_data",
            matched_dest_name="training_data"
        )
        
        self.assertEqual(match.source_logical_name, "processed_data")
        self.assertEqual(match.dest_logical_name, "training_data")
        self.assertEqual(match.match_type, MatchType.EXACT_LOGICAL)
        self.assertEqual(match.confidence, 0.95)
        self.assertEqual(match.matched_source_name, "processed_data")
        self.assertEqual(match.matched_dest_name, "training_data")
    
    def test_match_type_enum_values(self):
        """Test MatchType enum values"""
        # Test that all expected match types exist
        expected_types = [
            MatchType.EXACT_LOGICAL,
            MatchType.LOGICAL_TO_ALIAS,
            MatchType.ALIAS_TO_LOGICAL,
            MatchType.ALIAS_TO_ALIAS,
            MatchType.SEMANTIC
        ]
        
        for match_type in expected_types:
            self.assertIsInstance(match_type, MatchType)
    
    def test_path_match_with_different_match_types(self):
        """Test PathMatch with different match types"""
        # Test alias to logical match
        match = PathMatch(
            source_logical_name="model_data",
            dest_logical_name="trained_model",
            match_type=MatchType.ALIAS_TO_LOGICAL,
            confidence=0.75,
            matched_source_name="model_output",  # alias
            matched_dest_name="trained_model"   # logical name
        )
        
        self.assertEqual(match.match_type, MatchType.ALIAS_TO_LOGICAL)
        self.assertEqual(match.confidence, 0.75)
        
        # Test semantic match
        semantic_match = PathMatch(
            source_logical_name="feature_data",
            dest_logical_name="feature_input",
            match_type=MatchType.SEMANTIC,
            confidence=0.8,
            matched_source_name="feature_data",
            matched_dest_name="feature_input",
            semantic_details={"similarity_score": 0.8, "method": "word_embedding"}
        )
        
        self.assertEqual(semantic_match.match_type, MatchType.SEMANTIC)
        self.assertEqual(semantic_match.confidence, 0.8)
        self.assertIsNotNone(semantic_match.semantic_details)


if __name__ == '__main__':
    unittest.main()
