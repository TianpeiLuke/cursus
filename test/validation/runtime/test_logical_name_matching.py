"""
Unit tests for logical name matching system

Tests the logical name matching components including PathSpec, PathMatcher,
EnhancedScriptExecutionSpec, TopologicalExecutor, and LogicalNameMatchingTester.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.api.dag.base_dag import PipelineDAG


class TestPathSpec(unittest.TestCase):
    """Test PathSpec model"""
    
    def test_path_spec_creation(self):
        """Test creating a PathSpec"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"]
        )
        
        self.assertEqual(spec.logical_name, "processed_data")
        self.assertEqual(spec.path, "/path/to/data")
        self.assertEqual(spec.aliases, ["input_data", "training_data"])
    
    def test_matches_name_or_alias_logical_name(self):
        """Test matching against logical name"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"]
        )
        
        self.assertTrue(spec.matches_name_or_alias("processed_data"))
        self.assertFalse(spec.matches_name_or_alias("other_data"))
    
    def test_matches_name_or_alias_aliases(self):
        """Test matching against aliases"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"]
        )
        
        self.assertTrue(spec.matches_name_or_alias("input_data"))
        self.assertTrue(spec.matches_name_or_alias("training_data"))
        self.assertFalse(spec.matches_name_or_alias("test_data"))


class TestPathMatch(unittest.TestCase):
    """Test PathMatch model"""
    
    def test_path_match_creation(self):
        """Test creating a PathMatch"""
        match = PathMatch(
            source_logical_name="output_data",
            dest_logical_name="input_data",
            match_type=MatchType.EXACT_LOGICAL,
            confidence=1.0,
            matched_source_name="output_data",
            matched_dest_name="input_data"
        )
        
        self.assertEqual(match.source_logical_name, "output_data")
        self.assertEqual(match.dest_logical_name, "input_data")
        self.assertEqual(match.match_type, MatchType.EXACT_LOGICAL)
        self.assertEqual(match.confidence, 1.0)
        self.assertEqual(match.matched_source_name, "output_data")
        self.assertEqual(match.matched_dest_name, "input_data")


class TestEnhancedScriptExecutionSpec(unittest.TestCase):
    """Test EnhancedScriptExecutionSpec model"""
    
    def test_enhanced_spec_creation(self):
        """Test creating an EnhancedScriptExecutionSpec"""
        input_specs = {
            "data_input": PathSpec(
                logical_name="training_data",
                path="/input/path",
                aliases=["input_data", "processed_data"]
            )
        }
        output_specs = {
            "data_output": PathSpec(
                logical_name="model_output",
                path="/output/path",
                aliases=["trained_model", "artifacts"]
            )
        }
        
        spec = EnhancedScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_path_specs=input_specs,
            output_path_specs=output_specs,
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"}
        )
        
        self.assertEqual(spec.script_name, "test_script")
        self.assertEqual(spec.step_name, "test_step")
        self.assertEqual(len(spec.input_path_specs), 1)
        self.assertEqual(len(spec.output_path_specs), 1)
        self.assertEqual(spec.environ_vars["TEST_VAR"], "test_value")
        self.assertEqual(spec.job_args["test_arg"], "test_value")
    
    def test_backward_compatibility_properties(self):
        """Test backward compatibility properties"""
        input_specs = {
            "data_input": PathSpec(logical_name="training_data", path="/input/path")
        }
        output_specs = {
            "data_output": PathSpec(logical_name="model_output", path="/output/path")
        }
        
        spec = EnhancedScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_path_specs=input_specs,
            output_path_specs=output_specs
        )
        
        # Test backward compatibility properties
        input_paths = spec.input_paths
        output_paths = spec.output_paths
        
        self.assertEqual(input_paths["data_input"], "/input/path")
        self.assertEqual(output_paths["data_output"], "/output/path")
    
    def test_from_script_execution_spec(self):
        """Test creating enhanced spec from original spec"""
        original_spec = ScriptExecutionSpec(
            script_name="original_script",
            step_name="original_step",
            input_paths={"data_input": "/input/path"},
            output_paths={"data_output": "/output/path"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"}
        )
        
        input_aliases = {"data_input": ["training_data", "processed_data"]}
        output_aliases = {"data_output": ["model_output", "artifacts"]}
        
        enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
            original_spec, input_aliases, output_aliases
        )
        
        self.assertEqual(enhanced_spec.script_name, "original_script")
        self.assertEqual(enhanced_spec.step_name, "original_step")
        self.assertEqual(enhanced_spec.input_path_specs["data_input"].logical_name, "data_input")
        self.assertEqual(enhanced_spec.input_path_specs["data_input"].aliases, ["training_data", "processed_data"])
        self.assertEqual(enhanced_spec.output_path_specs["data_output"].logical_name, "data_output")
        self.assertEqual(enhanced_spec.output_path_specs["data_output"].aliases, ["model_output", "artifacts"])


class TestPathMatcher(unittest.TestCase):
    """Test PathMatcher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.path_matcher = PathMatcher(semantic_threshold=0.7)
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_path_matcher_initialization(self, mock_semantic_matcher):
        """Test PathMatcher initialization"""
        matcher = PathMatcher(semantic_threshold=0.8)
        
        self.assertEqual(matcher.semantic_threshold, 0.8)
        mock_semantic_matcher.assert_called_once()
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_find_path_matches_exact_logical(self, mock_semantic_matcher):
        """Test finding exact logical name matches"""
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            }
        )
        
        # Create destination spec
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="processed_data", path="/input/path")
            }
        )
        
        matches = self.path_matcher.find_path_matches(source_spec, dest_spec)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source_logical_name, "output1")
        self.assertEqual(matches[0].dest_logical_name, "input1")
        self.assertEqual(matches[0].match_type, MatchType.EXACT_LOGICAL)
        self.assertEqual(matches[0].confidence, 1.0)
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_find_path_matches_logical_to_alias(self, mock_semantic_matcher):
        """Test finding logical name to alias matches"""
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            }
        )
        
        # Create destination spec with alias
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(
                    logical_name="training_data", 
                    path="/input/path",
                    aliases=["processed_data", "input_data"]
                )
            }
        )
        
        matches = self.path_matcher.find_path_matches(source_spec, dest_spec)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_type, MatchType.LOGICAL_TO_ALIAS)
        self.assertEqual(matches[0].confidence, 0.95)
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_find_path_matches_semantic(self, mock_semantic_matcher):
        """Test finding semantic similarity matches"""
        # Mock semantic matcher
        mock_semantic_matcher.return_value.calculate_similarity.return_value = 0.8
        mock_semantic_matcher.return_value.explain_similarity.return_value = {"method": "semantic"}
        
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            }
        )
        
        # Create destination spec with similar name
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="training_data", path="/input/path")
            }
        )
        
        # Create a new PathMatcher instance to use the mocked SemanticMatcher
        path_matcher = PathMatcher(semantic_threshold=0.7)
        matches = path_matcher.find_path_matches(source_spec, dest_spec)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_type, MatchType.SEMANTIC)
        self.assertEqual(matches[0].confidence, 0.8)
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_find_path_matches_no_matches(self, mock_semantic_matcher):
        """Test finding no matches when similarity is below threshold"""
        # Mock semantic matcher to return low similarity
        mock_semantic_matcher.return_value.calculate_similarity.return_value = 0.3
        
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="completely_different", path="/output/path")
            }
        )
        
        # Create destination spec
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="unrelated_data", path="/input/path")
            }
        )
        
        matches = self.path_matcher.find_path_matches(source_spec, dest_spec)
        
        self.assertEqual(len(matches), 0)
    
    def test_generate_matching_report_no_matches(self):
        """Test generating matching report with no matches"""
        report = self.path_matcher.generate_matching_report([])
        
        self.assertEqual(report["total_matches"], 0)
        self.assertEqual(report["match_summary"], "No matches found")
        self.assertIn("Check logical names and aliases for compatibility", report["recommendations"])
    
    def test_generate_matching_report_with_matches(self):
        """Test generating matching report with matches"""
        matches = [
            PathMatch(
                source_logical_name="output1",
                dest_logical_name="input1",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=1.0,
                matched_source_name="processed_data",
                matched_dest_name="processed_data"
            ),
            PathMatch(
                source_logical_name="output2",
                dest_logical_name="input2",
                match_type=MatchType.SEMANTIC,
                confidence=0.8,
                matched_source_name="model_output",
                matched_dest_name="model_artifacts"
            )
        ]
        
        report = self.path_matcher.generate_matching_report(matches)
        
        self.assertEqual(report["total_matches"], 2)
        self.assertEqual(report["high_confidence_matches"], 2)
        self.assertEqual(report["average_confidence"], 0.9)
        self.assertEqual(report["matches_by_type"]["exact_logical"], 1)
        self.assertEqual(report["matches_by_type"]["semantic"], 1)
        self.assertEqual(len(report["best_matches"]), 2)


class TestTopologicalExecutor(unittest.TestCase):
    """Test TopologicalExecutor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dag = PipelineDAG(
            nodes=["step_a", "step_b", "step_c"],
            edges=[("step_a", "step_b"), ("step_b", "step_c")]
        )
        self.executor = TopologicalExecutor()
    
    def test_get_execution_order(self):
        """Test getting topological execution order"""
        with patch.object(self.dag, 'topological_sort', return_value=["step_a", "step_b", "step_c"]):
            order = self.executor.get_execution_order(self.dag)
            self.assertEqual(order, ["step_a", "step_b", "step_c"])
    
    def test_get_execution_order_with_error(self):
        """Test getting execution order with DAG error"""
        with patch.object(self.dag, 'topological_sort', side_effect=ValueError("Cycle detected")):
            with self.assertRaises(ValueError) as context:
                self.executor.get_execution_order(self.dag)
            
            self.assertIn("DAG topology error", str(context.exception))
    
    def test_validate_dag_structure(self):
        """Test DAG structure validation"""
        script_specs = {
            "step_a": Mock(),
            "step_b": Mock(),
            "step_c": Mock()
        }
        
        errors = self.executor.validate_dag_structure(self.dag, script_specs)
        self.assertEqual(len(errors), 0)
    
    def test_validate_dag_structure_missing_specs(self):
        """Test DAG structure validation with missing specs"""
        script_specs = {
            "step_a": Mock(),
            # Missing step_b and step_c
        }
        
        errors = self.executor.validate_dag_structure(self.dag, script_specs)
        self.assertEqual(len(errors), 2)
        self.assertIn("No ScriptExecutionSpec found for DAG node: step_b", errors)
        self.assertIn("No ScriptExecutionSpec found for DAG node: step_c", errors)
    
    def test_validate_dag_structure_extra_specs(self):
        """Test DAG structure validation with extra specs"""
        script_specs = {
            "step_a": Mock(),
            "step_b": Mock(),
            "step_c": Mock(),
            "step_d": Mock()  # Extra spec not in DAG
        }
        
        errors = self.executor.validate_dag_structure(self.dag, script_specs)
        self.assertEqual(len(errors), 1)
        self.assertIn("ScriptExecutionSpec 'step_d' not found in DAG nodes", errors)


class TestLogicalNameMatchingTester(unittest.TestCase):
    """Test LogicalNameMatchingTester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = LogicalNameMatchingTester(semantic_threshold=0.7)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    @patch('cursus.validation.runtime.logical_name_matching.PathMatcher')
    def test_test_data_compatibility_with_logical_matching_success(self, mock_path_matcher_class):
        """Test successful data compatibility with logical matching"""
        # Mock path matcher
        mock_path_matcher = Mock()
        mock_path_matcher_class.return_value = mock_path_matcher
        mock_path_matcher.find_path_matches.return_value = [
            PathMatch(
                source_logical_name="output1",
                dest_logical_name="input1",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=1.0,
                matched_source_name="processed_data",
                matched_dest_name="processed_data"
            )
        ]
        mock_path_matcher.generate_matching_report.return_value = {
            "total_matches": 1,
            "high_confidence_matches": 1
        }
        
        # Create test specs
        spec_a = EnhancedScriptExecutionSpec(
            script_name="script_a",
            step_name="step_a",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            }
        )
        
        spec_b = EnhancedScriptExecutionSpec(
            script_name="script_b",
            step_name="step_b",
            input_path_specs={
                "input1": PathSpec(logical_name="processed_data", path="/input/path")
            }
        )
        
        # Create mock output files
        output_files = [self.temp_dir / "output.csv"]
        output_files[0].touch()
        
        result = self.tester.test_data_compatibility_with_logical_matching(
            spec_a, spec_b, output_files
        )
        
        self.assertIsInstance(result, EnhancedDataCompatibilityResult)
        self.assertEqual(result.script_a, "script_a")
        self.assertEqual(result.script_b, "script_b")
        self.assertTrue(result.compatible)
        self.assertEqual(len(result.path_matches), 1)
    
    @patch('cursus.validation.runtime.logical_name_matching.PathMatcher')
    def test_test_data_compatibility_no_matches(self, mock_path_matcher_class):
        """Test data compatibility with no logical matches"""
        # Mock path matcher to return no matches
        mock_path_matcher = Mock()
        mock_path_matcher_class.return_value = mock_path_matcher
        mock_path_matcher.find_path_matches.return_value = []
        mock_path_matcher.generate_matching_report.return_value = {
            "total_matches": 0
        }
        
        # Create test specs
        spec_a = EnhancedScriptExecutionSpec(
            script_name="script_a",
            step_name="step_a",
            output_path_specs={
                "output1": PathSpec(logical_name="different_data", path="/output/path")
            }
        )
        
        spec_b = EnhancedScriptExecutionSpec(
            script_name="script_b",
            step_name="step_b",
            input_path_specs={
                "input1": PathSpec(logical_name="unrelated_data", path="/input/path")
            }
        )
        
        output_files = [self.temp_dir / "output.csv"]
        output_files[0].touch()
        
        result = self.tester.test_data_compatibility_with_logical_matching(
            spec_a, spec_b, output_files
        )
        
        self.assertFalse(result.compatible)
        self.assertIn("No matching logical names found between source outputs and destination inputs", result.compatibility_issues)
    
    def test_test_pipeline_with_topological_order_success(self):
        """Test successful pipeline testing with topological order"""
        # Create test DAG
        dag = PipelineDAG(
            nodes=["step_a", "step_b"],
            edges=[("step_a", "step_b")]
        )
        
        # Create test specs
        script_specs = {
            "step_a": EnhancedScriptExecutionSpec(
                script_name="script_a",
                step_name="step_a",
                output_path_specs={
                    "output1": PathSpec(logical_name="processed_data", path="/output/path")
                }
            ),
            "step_b": EnhancedScriptExecutionSpec(
                script_name="script_b",
                step_name="step_b",
                input_path_specs={
                    "input1": PathSpec(logical_name="processed_data", path="/input/path")
                }
            )
        }
        
        # Mock script tester function
        def mock_script_tester(spec):
            from cursus.validation.runtime.runtime_models import ScriptTestResult
            return ScriptTestResult(
                script_name=spec.script_name,
                success=True,
                execution_time=0.1
            )
        
        with patch.object(dag, 'topological_sort', return_value=["step_a", "step_b"]):
            with patch.object(self.tester.path_matcher, 'find_path_matches') as mock_find_matches:
                mock_find_matches.return_value = [
                    PathMatch(
                        source_logical_name="output1",
                        dest_logical_name="input1",
                        match_type=MatchType.EXACT_LOGICAL,
                        confidence=1.0,
                        matched_source_name="processed_data",
                        matched_dest_name="processed_data"
                    )
                ]
                
                with patch.object(self.tester.path_matcher, 'generate_matching_report') as mock_report:
                    mock_report.return_value = {"total_matches": 1}
                    
                    result = self.tester.test_pipeline_with_topological_execution(
                        dag, script_specs, mock_script_tester
                    )
        
        self.assertTrue(result["pipeline_success"])
        self.assertEqual(result["execution_order"], ["step_a", "step_b"])
        self.assertEqual(len(result["logical_matching_results"]), 1)
    
    def test_test_pipeline_with_topological_order_dag_error(self):
        """Test pipeline testing with DAG topology error"""
        dag = PipelineDAG(nodes=["step_a"], edges=[])
        script_specs = {"step_a": Mock()}
        
        with patch.object(dag, 'topological_sort', side_effect=ValueError("Cycle detected")):
            result = self.tester.test_pipeline_with_topological_execution(
                dag, script_specs, Mock()
            )
        
        self.assertFalse(result["pipeline_success"])
        self.assertIn("DAG topology error", result["errors"][0])
    
    def test_find_best_file_for_logical_name(self):
        """Test finding best file for logical name"""
        # Create test files
        file1 = self.temp_dir / "processed_data.csv"
        file2 = self.temp_dir / "other_data.csv"
        file1.touch()
        file2.touch()
        
        output_files = [file1, file2]
        
        # Test exact match
        best_file = self.tester._find_best_file_for_logical_name("processed_data", output_files)
        self.assertEqual(best_file, file1)
        
        # Test no match (should return most recent)
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1000  # Mock modification time
            best_file = self.tester._find_best_file_for_logical_name("unrelated", output_files)
            self.assertIn(best_file, output_files)
    
    def test_detect_primary_format(self):
        """Test detecting primary file format"""
        # Create test files with different extensions
        files = [
            self.temp_dir / "file1.csv",
            self.temp_dir / "file2.csv",
            self.temp_dir / "file3.json"
        ]
        
        for file in files:
            file.touch()
        
        # CSV should be primary (most common)
        primary_format = self.tester._detect_primary_format(files)
        self.assertEqual(primary_format, ".csv")
        
        # Test with no files
        primary_format = self.tester._detect_primary_format([])
        self.assertEqual(primary_format, "unknown")


class TestLogicalNameMatchingIntegration(unittest.TestCase):
    """Integration tests for logical name matching system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tester = LogicalNameMatchingTester(semantic_threshold=0.7)
    
    @patch('cursus.validation.runtime.logical_name_matching.SemanticMatcher')
    def test_end_to_end_matching_workflow(self, mock_semantic_matcher_class):
        """Test complete end-to-end matching workflow"""
        # Mock semantic matcher to return low similarity for hyperparameter_s3
        mock_semantic_matcher = Mock()
        mock_semantic_matcher_class.return_value = mock_semantic_matcher
        
        def mock_similarity(name1, name2):
            # Only return high similarity for processed_data <-> training_data
            if ("processed_data" in name1 and "training_data" in name2) or \
               ("training_data" in name1 and "processed_data" in name2):
                return 0.9
            return 0.3  # Low similarity for hyperparameter_s3
        
        mock_semantic_matcher.calculate_similarity.side_effect = mock_similarity
        mock_semantic_matcher.explain_similarity.return_value = {"method": "semantic", "similarity": 0.9}
        
        # Create realistic specs
        preprocessing_spec = EnhancedScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing",
            output_path_specs={
                "processed_data": PathSpec(
                    logical_name="processed_data",
                    path="/preprocessing/output",
                    aliases=["clean_data", "training_data"]
                )
            }
        )
        
        training_spec = EnhancedScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training",
            input_path_specs={
                "training_data": PathSpec(
                    logical_name="training_data",
                    path="/training/input",
                    aliases=["processed_data", "input_data"]
                ),
                "hyperparameter_s3": PathSpec(
                    logical_name="hyperparameter_s3",
                    path="s3://bucket/hyperparams.json",
                    aliases=["hyperparams", "config"]
                )
            }
        )
        
        # Test path matching
        path_matcher = PathMatcher(semantic_threshold=0.7)
        matches = path_matcher.find_path_matches(preprocessing_spec, training_spec)
        
        # Should find alias match for processed_data -> training_data
        # Note: May find multiple matches (alias + semantic), but alias should be highest confidence
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0].match_type, MatchType.LOGICAL_TO_ALIAS)
        self.assertEqual(matches[0].confidence, 0.95)
        
        # hyperparameter_s3 should remain independent (no match)
        matched_inputs = {match.dest_logical_name for match in matches}
        self.assertIn("training_data", matched_inputs)
        self.assertNotIn("hyperparameter_s3", matched_inputs)


if __name__ == '__main__':
    unittest.main()
