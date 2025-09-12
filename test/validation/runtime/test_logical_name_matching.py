"""
Pytest tests for logical name matching system

Tests the logical name matching components including PathSpec, PathMatcher,
EnhancedScriptExecutionSpec, TopologicalExecutor, and LogicalNameMatchingTester.
"""

import pytest
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
    EnhancedDataCompatibilityResult,
)
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.api.dag.base_dag import PipelineDAG


class TestPathSpec:
    """Test PathSpec model"""

    def test_path_spec_creation(self):
        """Test creating a PathSpec"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"],
        )

        assert spec.logical_name == "processed_data"
        assert spec.path == "/path/to/data"
        assert spec.aliases == ["input_data", "training_data"]

    def test_matches_name_or_alias_logical_name(self):
        """Test matching against logical name"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"],
        )

        assert spec.matches_name_or_alias("processed_data") is True
        assert spec.matches_name_or_alias("other_data") is False

    def test_matches_name_or_alias_aliases(self):
        """Test matching against aliases"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/path/to/data",
            aliases=["input_data", "training_data"],
        )

        assert spec.matches_name_or_alias("input_data") is True
        assert spec.matches_name_or_alias("training_data") is True
        assert spec.matches_name_or_alias("test_data") is False


class TestPathMatch:
    """Test PathMatch model"""

    def test_path_match_creation(self):
        """Test creating a PathMatch"""
        match = PathMatch(
            source_logical_name="output_data",
            dest_logical_name="input_data",
            match_type=MatchType.EXACT_LOGICAL,
            confidence=1.0,
            matched_source_name="output_data",
            matched_dest_name="input_data",
        )

        assert match.source_logical_name == "output_data"
        assert match.dest_logical_name == "input_data"
        assert match.match_type == MatchType.EXACT_LOGICAL
        assert match.confidence == 1.0
        assert match.matched_source_name == "output_data"
        assert match.matched_dest_name == "input_data"


class TestEnhancedScriptExecutionSpec:
    """Test EnhancedScriptExecutionSpec model"""

    def test_enhanced_spec_creation(self):
        """Test creating an EnhancedScriptExecutionSpec"""
        input_specs = {
            "data_input": PathSpec(
                logical_name="training_data",
                path="/input/path",
                aliases=["input_data", "processed_data"],
            )
        }
        output_specs = {
            "data_output": PathSpec(
                logical_name="model_output",
                path="/output/path",
                aliases=["trained_model", "artifacts"],
            )
        }

        spec = EnhancedScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_path_specs=input_specs,
            output_path_specs=output_specs,
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"},
        )

        assert spec.script_name == "test_script"
        assert spec.step_name == "test_step"
        assert len(spec.input_path_specs) == 1
        assert len(spec.output_path_specs) == 1
        assert spec.environ_vars["TEST_VAR"] == "test_value"
        assert spec.job_args["test_arg"] == "test_value"

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
            output_path_specs=output_specs,
        )

        # Test backward compatibility properties
        input_paths = spec.input_paths
        output_paths = spec.output_paths

        assert input_paths["data_input"] == "/input/path"
        assert output_paths["data_output"] == "/output/path"

    def test_from_script_execution_spec(self):
        """Test creating enhanced spec from original spec"""
        original_spec = ScriptExecutionSpec(
            script_name="original_script",
            step_name="original_step",
            input_paths={"data_input": "/input/path"},
            output_paths={"data_output": "/output/path"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"},
        )

        input_aliases = {"data_input": ["training_data", "processed_data"]}
        output_aliases = {"data_output": ["model_output", "artifacts"]}

        enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
            original_spec, input_aliases, output_aliases
        )

        assert enhanced_spec.script_name == "original_script"
        assert enhanced_spec.step_name == "original_step"
        assert enhanced_spec.input_path_specs["data_input"].logical_name == "data_input"
        assert enhanced_spec.input_path_specs["data_input"].aliases == [
            "training_data",
            "processed_data",
        ]
        assert (
            enhanced_spec.output_path_specs["data_output"].logical_name == "data_output"
        )
        assert enhanced_spec.output_path_specs["data_output"].aliases == [
            "model_output",
            "artifacts",
        ]


class TestPathMatcher:
    """Test PathMatcher class"""

    @pytest.fixture
    def path_matcher(self):
        """Create PathMatcher instance"""
        return PathMatcher(semantic_threshold=0.7)

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_path_matcher_initialization(self, mock_semantic_matcher):
        """Test PathMatcher initialization"""
        matcher = PathMatcher(semantic_threshold=0.8)

        assert matcher.semantic_threshold == 0.8
        mock_semantic_matcher.assert_called_once()

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_find_path_matches_exact_logical(self, mock_semantic_matcher, path_matcher):
        """Test finding exact logical name matches"""
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            },
        )

        # Create destination spec
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="processed_data", path="/input/path")
            },
        )

        matches = path_matcher.find_path_matches(source_spec, dest_spec)

        assert len(matches) == 1
        assert matches[0].source_logical_name == "output1"
        assert matches[0].dest_logical_name == "input1"
        assert matches[0].match_type == MatchType.EXACT_LOGICAL
        assert matches[0].confidence == 1.0

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_find_path_matches_logical_to_alias(
        self, mock_semantic_matcher, path_matcher
    ):
        """Test finding logical name to alias matches"""
        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            },
        )

        # Create destination spec with alias
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(
                    logical_name="training_data",
                    path="/input/path",
                    aliases=["processed_data", "input_data"],
                )
            },
        )

        matches = path_matcher.find_path_matches(source_spec, dest_spec)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.LOGICAL_TO_ALIAS
        assert matches[0].confidence == 0.95

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_find_path_matches_semantic(
        self, mock_semantic_matcher_class, path_matcher
    ):
        """Test finding semantic similarity matches"""
        # Mock semantic matcher
        mock_semantic_matcher = Mock()
        mock_semantic_matcher.calculate_similarity.return_value = 0.8
        mock_semantic_matcher.explain_similarity.return_value = {"method": "semantic"}
        mock_semantic_matcher_class.return_value = mock_semantic_matcher

        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            },
        )

        # Create destination spec with similar name
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="training_data", path="/input/path")
            },
        )

        # Create a new PathMatcher instance to use the mocked SemanticMatcher
        path_matcher = PathMatcher(semantic_threshold=0.7)
        matches = path_matcher.find_path_matches(source_spec, dest_spec)

        assert len(matches) == 1
        assert matches[0].match_type == MatchType.SEMANTIC
        assert matches[0].confidence == 0.8

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_find_path_matches_no_matches(
        self, mock_semantic_matcher_class, path_matcher
    ):
        """Test finding no matches when similarity is below threshold"""
        # Mock semantic matcher to return low similarity
        mock_semantic_matcher = Mock()
        mock_semantic_matcher.calculate_similarity.return_value = 0.3
        mock_semantic_matcher_class.return_value = mock_semantic_matcher

        # Create source spec
        source_spec = EnhancedScriptExecutionSpec(
            script_name="source_script",
            step_name="source_step",
            output_path_specs={
                "output1": PathSpec(
                    logical_name="completely_different", path="/output/path"
                )
            },
        )

        # Create destination spec
        dest_spec = EnhancedScriptExecutionSpec(
            script_name="dest_script",
            step_name="dest_step",
            input_path_specs={
                "input1": PathSpec(logical_name="unrelated_data", path="/input/path")
            },
        )

        matches = path_matcher.find_path_matches(source_spec, dest_spec)

        assert len(matches) == 0

    def test_generate_matching_report_no_matches(self, path_matcher):
        """Test generating matching report with no matches"""
        report = path_matcher.generate_matching_report([])

        assert report["total_matches"] == 0
        assert report["match_summary"] == "No matches found"
        assert (
            "Check logical names and aliases for compatibility"
            in report["recommendations"]
        )

    def test_generate_matching_report_with_matches(self, path_matcher):
        """Test generating matching report with matches"""
        matches = [
            PathMatch(
                source_logical_name="output1",
                dest_logical_name="input1",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=1.0,
                matched_source_name="processed_data",
                matched_dest_name="processed_data",
            ),
            PathMatch(
                source_logical_name="output2",
                dest_logical_name="input2",
                match_type=MatchType.SEMANTIC,
                confidence=0.8,
                matched_source_name="model_output",
                matched_dest_name="model_artifacts",
            ),
        ]

        report = path_matcher.generate_matching_report(matches)

        assert report["total_matches"] == 2
        assert report["high_confidence_matches"] == 2
        assert report["average_confidence"] == 0.9
        assert report["matches_by_type"]["exact_logical"] == 1
        assert report["matches_by_type"]["semantic"] == 1
        assert len(report["best_matches"]) == 2


class TestTopologicalExecutor:
    """Test TopologicalExecutor class"""

    @pytest.fixture
    def dag(self):
        """Create test DAG"""
        return PipelineDAG(
            nodes=["step_a", "step_b", "step_c"],
            edges=[("step_a", "step_b"), ("step_b", "step_c")],
        )

    @pytest.fixture
    def executor(self):
        """Create TopologicalExecutor instance"""
        return TopologicalExecutor()

    def test_get_execution_order(self, dag, executor):
        """Test getting topological execution order"""
        with patch.object(
            dag, "topological_sort", return_value=["step_a", "step_b", "step_c"]
        ):
            order = executor.get_execution_order(dag)
            assert order == ["step_a", "step_b", "step_c"]

    def test_get_execution_order_with_error(self, dag, executor):
        """Test getting execution order with DAG error"""
        with patch.object(
            dag, "topological_sort", side_effect=ValueError("Cycle detected")
        ):
            with pytest.raises(ValueError) as exc_info:
                executor.get_execution_order(dag)

            assert "DAG topology error" in str(exc_info.value)

    def test_validate_dag_structure(self, dag, executor):
        """Test DAG structure validation"""
        script_specs = {"step_a": Mock(), "step_b": Mock(), "step_c": Mock()}

        errors = executor.validate_dag_structure(dag, script_specs)
        assert len(errors) == 0

    def test_validate_dag_structure_missing_specs(self, dag, executor):
        """Test DAG structure validation with missing specs"""
        script_specs = {
            "step_a": Mock(),
            # Missing step_b and step_c
        }

        errors = executor.validate_dag_structure(dag, script_specs)
        assert len(errors) == 2
        assert "No ScriptExecutionSpec found for DAG node: step_b" in errors
        assert "No ScriptExecutionSpec found for DAG node: step_c" in errors

    def test_validate_dag_structure_extra_specs(self, dag, executor):
        """Test DAG structure validation with extra specs"""
        script_specs = {
            "step_a": Mock(),
            "step_b": Mock(),
            "step_c": Mock(),
            "step_d": Mock(),  # Extra spec not in DAG
        }

        errors = executor.validate_dag_structure(dag, script_specs)
        assert len(errors) == 1
        assert "ScriptExecutionSpec 'step_d' not found in DAG nodes" in errors


class TestLogicalNameMatchingTester:
    """Test LogicalNameMatchingTester class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def tester(self):
        """Create LogicalNameMatchingTester instance"""
        return LogicalNameMatchingTester(semantic_threshold=0.7)

    @patch("cursus.validation.runtime.logical_name_matching.PathMatcher")
    def test_test_data_compatibility_with_logical_matching_success(
        self, mock_path_matcher_class, tester, temp_dir
    ):
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
                matched_dest_name="processed_data",
            )
        ]
        mock_path_matcher.generate_matching_report.return_value = {
            "total_matches": 1,
            "high_confidence_matches": 1,
        }

        # Create test specs
        spec_a = EnhancedScriptExecutionSpec(
            script_name="script_a",
            step_name="step_a",
            output_path_specs={
                "output1": PathSpec(logical_name="processed_data", path="/output/path")
            },
        )

        spec_b = EnhancedScriptExecutionSpec(
            script_name="script_b",
            step_name="step_b",
            input_path_specs={
                "input1": PathSpec(logical_name="processed_data", path="/input/path")
            },
        )

        # Create mock output files
        output_files = [temp_dir / "output.csv"]
        output_files[0].touch()

        result = tester.test_data_compatibility_with_logical_matching(
            spec_a, spec_b, output_files
        )

        assert isinstance(result, EnhancedDataCompatibilityResult)
        assert result.script_a == "script_a"
        assert result.script_b == "script_b"
        assert result.compatible is True
        assert len(result.path_matches) == 1

    @patch("cursus.validation.runtime.logical_name_matching.PathMatcher")
    def test_test_data_compatibility_no_matches(
        self, mock_path_matcher_class, tester, temp_dir
    ):
        """Test data compatibility with no logical matches"""
        # Mock path matcher to return no matches
        mock_path_matcher = Mock()
        mock_path_matcher_class.return_value = mock_path_matcher
        mock_path_matcher.find_path_matches.return_value = []
        mock_path_matcher.generate_matching_report.return_value = {"total_matches": 0}

        # Create test specs
        spec_a = EnhancedScriptExecutionSpec(
            script_name="script_a",
            step_name="step_a",
            output_path_specs={
                "output1": PathSpec(logical_name="different_data", path="/output/path")
            },
        )

        spec_b = EnhancedScriptExecutionSpec(
            script_name="script_b",
            step_name="step_b",
            input_path_specs={
                "input1": PathSpec(logical_name="unrelated_data", path="/input/path")
            },
        )

        output_files = [temp_dir / "output.csv"]
        output_files[0].touch()

        result = tester.test_data_compatibility_with_logical_matching(
            spec_a, spec_b, output_files
        )

        assert result.compatible is False
        assert (
            "No matching logical names found between source outputs and destination inputs"
            in result.compatibility_issues
        )

    def test_test_pipeline_with_topological_order_success(self, tester):
        """Test successful pipeline testing with topological order"""
        # Create test DAG
        dag = PipelineDAG(nodes=["step_a", "step_b"], edges=[("step_a", "step_b")])

        # Create test specs
        script_specs = {
            "step_a": EnhancedScriptExecutionSpec(
                script_name="script_a",
                step_name="step_a",
                output_path_specs={
                    "output1": PathSpec(
                        logical_name="processed_data", path="/output/path"
                    )
                },
            ),
            "step_b": EnhancedScriptExecutionSpec(
                script_name="script_b",
                step_name="step_b",
                input_path_specs={
                    "input1": PathSpec(
                        logical_name="processed_data", path="/input/path"
                    )
                },
            ),
        }

        # Mock script tester function
        def mock_script_tester(spec):
            from cursus.validation.runtime.runtime_models import ScriptTestResult

            return ScriptTestResult(
                script_name=spec.script_name, success=True, execution_time=0.1
            )

        with patch.object(dag, "topological_sort", return_value=["step_a", "step_b"]):
            with patch.object(
                tester.path_matcher, "find_path_matches"
            ) as mock_find_matches:
                mock_find_matches.return_value = [
                    PathMatch(
                        source_logical_name="output1",
                        dest_logical_name="input1",
                        match_type=MatchType.EXACT_LOGICAL,
                        confidence=1.0,
                        matched_source_name="processed_data",
                        matched_dest_name="processed_data",
                    )
                ]

                with patch.object(
                    tester.path_matcher, "generate_matching_report"
                ) as mock_report:
                    mock_report.return_value = {"total_matches": 1}

                    result = tester.test_pipeline_with_topological_execution(
                        dag, script_specs, mock_script_tester
                    )

        assert result["pipeline_success"] is True
        assert result["execution_order"] == ["step_a", "step_b"]
        assert len(result["logical_matching_results"]) == 1

    def test_test_pipeline_with_topological_order_dag_error(self, tester):
        """Test pipeline testing with DAG topology error"""
        dag = PipelineDAG(nodes=["step_a"], edges=[])
        script_specs = {"step_a": Mock()}

        with patch.object(
            dag, "topological_sort", side_effect=ValueError("Cycle detected")
        ):
            result = tester.test_pipeline_with_topological_execution(
                dag, script_specs, Mock()
            )

        assert result["pipeline_success"] is False
        assert "DAG topology error" in result["errors"][0]

    def test_find_best_file_for_logical_name(self, tester, temp_dir):
        """Test finding best file for logical name"""
        # Create test files
        file1 = temp_dir / "processed_data.csv"
        file2 = temp_dir / "other_data.csv"
        file1.touch()
        file2.touch()

        output_files = [file1, file2]

        # Test exact match
        best_file = tester._find_best_file_for_logical_name(
            "processed_data", output_files
        )
        assert best_file == file1

        # Test no match (should return most recent)
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_mtime = 1000  # Mock modification time
            best_file = tester._find_best_file_for_logical_name(
                "unrelated", output_files
            )
            assert best_file in output_files

    def test_detect_primary_format(self, tester, temp_dir):
        """Test detecting primary file format"""
        # Create test files with different extensions
        files = [
            temp_dir / "file1.csv",
            temp_dir / "file2.csv",
            temp_dir / "file3.json",
        ]

        for file in files:
            file.touch()

        # CSV should be primary (most common)
        primary_format = tester._detect_primary_format(files)
        assert primary_format == ".csv"

        # Test with no files
        primary_format = tester._detect_primary_format([])
        assert primary_format == "unknown"


class TestLogicalNameMatchingIntegration:
    """Integration tests for logical name matching system"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def tester(self):
        """Create LogicalNameMatchingTester instance"""
        return LogicalNameMatchingTester(semantic_threshold=0.7)

    @patch("cursus.validation.runtime.logical_name_matching.SemanticMatcher")
    def test_end_to_end_matching_workflow(self, mock_semantic_matcher_class, tester):
        """Test complete end-to-end matching workflow"""
        # Mock semantic matcher to return low similarity for hyperparameter_s3
        mock_semantic_matcher = Mock()
        mock_semantic_matcher_class.return_value = mock_semantic_matcher

        def mock_similarity(name1, name2):
            # Only return high similarity for processed_data <-> training_data
            if ("processed_data" in name1 and "training_data" in name2) or (
                "training_data" in name1 and "processed_data" in name2
            ):
                return 0.9
            return 0.3  # Low similarity for hyperparameter_s3

        mock_semantic_matcher.calculate_similarity.side_effect = mock_similarity
        mock_semantic_matcher.explain_similarity.return_value = {
            "method": "semantic",
            "similarity": 0.9,
        }

        # Create realistic specs
        preprocessing_spec = EnhancedScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing",
            output_path_specs={
                "processed_data": PathSpec(
                    logical_name="processed_data",
                    path="/preprocessing/output",
                    aliases=["clean_data", "training_data"],
                )
            },
        )

        training_spec = EnhancedScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training",
            input_path_specs={
                "training_data": PathSpec(
                    logical_name="training_data",
                    path="/training/input",
                    aliases=["processed_data", "input_data"],
                ),
                "hyperparameter_s3": PathSpec(
                    logical_name="hyperparameter_s3",
                    path="s3://bucket/hyperparams.json",
                    aliases=["hyperparams", "config"],
                ),
            },
        )

        # Test path matching
        path_matcher = PathMatcher(semantic_threshold=0.7)
        matches = path_matcher.find_path_matches(preprocessing_spec, training_spec)

        # Should find alias match for processed_data -> training_data
        # Note: May find multiple matches (alias + semantic), but alias should be highest confidence
        assert len(matches) >= 1
        assert matches[0].match_type == MatchType.LOGICAL_TO_ALIAS
        assert matches[0].confidence == 0.95

        # hyperparameter_s3 should remain independent (no match)
        matched_inputs = {match.dest_logical_name for match in matches}
        assert "training_data" in matched_inputs
        assert "hyperparameter_s3" not in matched_inputs


class TestMatchTypeEnum:
    """Test MatchType enum"""

    def test_match_type_enum_values(self):
        """Test MatchType enum values"""
        # Test that all expected match types exist
        expected_types = [
            MatchType.EXACT_LOGICAL,
            MatchType.LOGICAL_TO_ALIAS,
            MatchType.ALIAS_TO_LOGICAL,
            MatchType.ALIAS_TO_ALIAS,
            MatchType.SEMANTIC,
        ]

        for match_type in expected_types:
            assert isinstance(match_type, MatchType)

    def test_match_type_string_values(self):
        """Test MatchType string values"""
        assert MatchType.EXACT_LOGICAL.value == "exact_logical"
        assert MatchType.LOGICAL_TO_ALIAS.value == "logical_to_alias"
        assert MatchType.ALIAS_TO_LOGICAL.value == "alias_to_logical"
        assert MatchType.ALIAS_TO_ALIAS.value == "alias_to_alias"
        assert MatchType.SEMANTIC.value == "semantic"


class TestEnhancedDataCompatibilityResult:
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
                matched_dest_name="training_data",
            )
        ]

        result = EnhancedDataCompatibilityResult(
            script_a="tabular_preprocessing",
            script_b="xgboost_training",
            compatible=True,
            path_matches=path_matches,
            matching_details={"total_matches": 1, "high_confidence_matches": 1},
        )

        assert result.script_a == "tabular_preprocessing"
        assert result.script_b == "xgboost_training"
        assert result.compatible is True
        assert len(result.path_matches) == 1
        assert result.matching_details is not None

        # Check the path match details
        match = result.path_matches[0]
        assert match.source_logical_name == "processed_data"
        assert match.dest_logical_name == "training_data"
        assert match.match_type == MatchType.EXACT_LOGICAL
        assert match.confidence == 0.95

    def test_enhanced_result_with_no_matches(self):
        """Test EnhancedDataCompatibilityResult with no path matches"""
        result = EnhancedDataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            path_matches=[],
            matching_details={
                "total_matches": 0,
                "recommendations": ["Check logical names"],
            },
            compatibility_issues=["No compatible outputs found"],
        )

        assert result.compatible is False
        assert len(result.path_matches) == 0
        assert result.matching_details is not None
        assert "No compatible outputs found" in result.compatibility_issues

    def test_enhanced_result_multiple_matches(self):
        """Test EnhancedDataCompatibilityResult with multiple path matches"""
        path_matches = [
            PathMatch(
                source_logical_name="processed_data",
                dest_logical_name="training_data",
                match_type=MatchType.EXACT_LOGICAL,
                confidence=0.95,
                matched_source_name="processed_data",
                matched_dest_name="training_data",
            ),
            PathMatch(
                source_logical_name="feature_data",
                dest_logical_name="feature_input",
                match_type=MatchType.SEMANTIC,
                confidence=0.8,
                matched_source_name="feature_data",
                matched_dest_name="feature_input",
            ),
        ]

        result = EnhancedDataCompatibilityResult(
            script_a="preprocessing",
            script_b="training",
            compatible=True,
            path_matches=path_matches,
            matching_details={"total_matches": 2, "high_confidence_matches": 1},
        )

        assert result.compatible is True
        assert len(result.path_matches) == 2

        # Check match types
        match_types = [match.match_type for match in result.path_matches]
        assert MatchType.EXACT_LOGICAL in match_types
        assert MatchType.SEMANTIC in match_types

        # Check confidence scores
        confidences = [match.confidence for match in result.path_matches]
        assert 0.95 in confidences
        assert 0.8 in confidences

    def test_enhanced_result_inheritance_from_basic(self):
        """Test that EnhancedDataCompatibilityResult has basic fields"""
        result = EnhancedDataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            path_matches=[],
            matching_details={"total_matches": 0},
        )

        # Should have all basic fields
        assert result.script_a == "script_a"
        assert result.script_b == "script_b"
        assert result.compatible is True
        assert result.compatibility_issues == []  # Default from base class

        # Should also have enhanced fields
        assert result.path_matches == []
        assert result.matching_details is not None
