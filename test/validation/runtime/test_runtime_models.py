"""
Pytest tests for runtime testing models

Tests the Pydantic models used in runtime testing including ScriptExecutionSpec,
PipelineTestingSpec, and RuntimeTestingConfiguration.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration,
)
from cursus.api.dag.base_dag import PipelineDAG

# Import enhanced models for testing
try:
    from cursus.validation.runtime.logical_name_matching import (
        EnhancedScriptExecutionSpec,
        EnhancedDataCompatibilityResult,
        PathSpec,
        PathMatch,
        MatchType,
    )

    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False


class TestScriptTestResult:
    """Test ScriptTestResult model"""

    def test_script_test_result_creation(self):
        """Test creating a ScriptTestResult"""
        result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=0.5,
            has_main_function=True,
        )

        assert result.script_name == "test_script"
        assert result.success is True
        assert result.execution_time == 0.5
        assert result.has_main_function is True
        assert result.error_message is None

    def test_script_test_result_with_error(self):
        """Test creating a ScriptTestResult with error"""
        result = ScriptTestResult(
            script_name="broken_script",
            success=False,
            error_message="Script failed to execute",
            execution_time=0.1,
            has_main_function=False,
        )

        assert result.script_name == "broken_script"
        assert result.success is False
        assert result.error_message == "Script failed to execute"
        assert result.execution_time == 0.1
        assert result.has_main_function is False


class TestDataCompatibilityResult:
    """Test DataCompatibilityResult model"""

    def test_data_compatibility_result_creation(self):
        """Test creating a DataCompatibilityResult"""
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            data_format_a="csv",
            data_format_b="csv",
        )

        assert result.script_a == "script_a"
        assert result.script_b == "script_b"
        assert result.compatible is True
        assert result.data_format_a == "csv"
        assert result.data_format_b == "csv"
        assert result.compatibility_issues == []

    def test_data_compatibility_result_with_issues(self):
        """Test creating a DataCompatibilityResult with compatibility issues"""
        issues = ["Column mismatch", "Type error"]
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            compatibility_issues=issues,
        )

        assert result.script_a == "script_a"
        assert result.script_b == "script_b"
        assert result.compatible is False
        assert result.compatibility_issues == issues


class TestScriptExecutionSpec:
    """Test ScriptExecutionSpec model"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def specs_dir(self, temp_dir):
        """Create specs directory"""
        specs_dir = Path(temp_dir) / ".specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        return specs_dir

    def test_script_execution_spec_creation(self):
        """Test creating a ScriptExecutionSpec"""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_paths={"data_input": "/path/to/input"},
            output_paths={"data_output": "/path/to/output"},
            environ_vars={"LABEL_FIELD": "label"},
            job_args={"job_type": "testing"},
        )

        assert spec.script_name == "test_script"
        assert spec.step_name == "test_step"
        assert spec.input_paths["data_input"] == "/path/to/input"
        assert spec.output_paths["data_output"] == "/path/to/output"
        assert spec.environ_vars["LABEL_FIELD"] == "label"
        assert spec.job_args["job_type"] == "testing"

    def test_create_default_spec(self):
        """Test creating a default ScriptExecutionSpec"""
        spec = ScriptExecutionSpec.create_default(
            script_name="default_script",
            step_name="default_step",
            test_data_dir="test/data",
        )

        assert spec.script_name == "default_script"
        assert spec.step_name == "default_step"
        assert spec.input_paths["data_input"] == "test/data/default_script/input"
        assert spec.output_paths["data_output"] == "test/data/default_script/output"
        assert spec.environ_vars["LABEL_FIELD"] == "label"
        assert spec.job_args["job_type"] == "testing"

    def test_save_and_load_spec(self, specs_dir):
        """Test saving and loading ScriptExecutionSpec"""
        original_spec = ScriptExecutionSpec(
            script_name="save_test",
            step_name="save_step",
            input_paths={"data_input": "/test/input"},
            output_paths={"data_output": "/test/output"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"},
        )

        # Save the spec
        saved_path = original_spec.save_to_file(str(specs_dir))
        assert Path(saved_path).exists()

        # Load the spec
        loaded_spec = ScriptExecutionSpec.load_from_file("save_test", str(specs_dir))

        # Verify loaded spec matches original
        assert loaded_spec.script_name == original_spec.script_name
        assert loaded_spec.step_name == original_spec.step_name
        assert loaded_spec.input_paths == original_spec.input_paths
        assert loaded_spec.output_paths == original_spec.output_paths
        assert loaded_spec.environ_vars == original_spec.environ_vars
        assert loaded_spec.job_args == original_spec.job_args
        assert loaded_spec.last_updated is not None

    def test_load_nonexistent_spec(self, specs_dir):
        """Test loading a non-existent ScriptExecutionSpec"""
        with pytest.raises(FileNotFoundError):
            ScriptExecutionSpec.load_from_file("nonexistent", str(specs_dir))

    def test_filename_generation(self, specs_dir):
        """Test that filename is generated correctly"""
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        saved_path = spec.save_to_file(str(specs_dir))

        expected_filename = "test_script_runtime_test_spec.json"
        assert saved_path.endswith(expected_filename)


class TestPipelineTestingSpec:
    """Test PipelineTestingSpec model"""

    def test_pipeline_testing_spec_creation(self):
        """Test creating a PipelineTestingSpec"""
        dag = PipelineDAG(
            nodes=["script_a", "script_b"], edges=[("script_a", "script_b")]
        )

        spec_a = ScriptExecutionSpec.create_default("script_a", "step_a")
        spec_b = ScriptExecutionSpec.create_default("script_b", "step_b")

        pipeline_spec = PipelineTestingSpec(
            dag=dag,
            script_specs={"script_a": spec_a, "script_b": spec_b},
            test_workspace_root="test/workspace",
        )

        assert pipeline_spec.dag.nodes == ["script_a", "script_b"]
        assert pipeline_spec.dag.edges == [("script_a", "script_b")]
        assert len(pipeline_spec.script_specs) == 2
        assert "script_a" in pipeline_spec.script_specs
        assert "script_b" in pipeline_spec.script_specs
        assert pipeline_spec.test_workspace_root == "test/workspace"


class TestRuntimeTestingConfiguration:
    """Test RuntimeTestingConfiguration model"""

    def test_runtime_testing_configuration_creation(self):
        """Test creating a RuntimeTestingConfiguration"""
        dag = PipelineDAG(nodes=["test_script"], edges=[])
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        pipeline_spec = PipelineTestingSpec(dag=dag, script_specs={"test_script": spec})

        config = RuntimeTestingConfiguration(
            pipeline_spec=pipeline_spec,
            test_individual_scripts=True,
            test_data_compatibility=True,
            test_pipeline_flow=True,
            use_workspace_aware=False,
        )

        assert config.pipeline_spec == pipeline_spec
        assert config.test_individual_scripts is True
        assert config.test_data_compatibility is True
        assert config.test_pipeline_flow is True
        assert config.use_workspace_aware is False

    def test_runtime_testing_configuration_defaults(self):
        """Test RuntimeTestingConfiguration with default values"""
        dag = PipelineDAG(nodes=["test_script"], edges=[])
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")
        pipeline_spec = PipelineTestingSpec(dag=dag, script_specs={"test_script": spec})

        config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

        # Test default values
        assert config.test_individual_scripts is True
        assert config.test_data_compatibility is True
        assert config.test_pipeline_flow is True
        assert config.use_workspace_aware is False


@pytest.mark.skipif(
    not ENHANCED_MODELS_AVAILABLE, reason="Enhanced models not available"
)
class TestEnhancedScriptExecutionSpec:
    """Test EnhancedScriptExecutionSpec model"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_enhanced_script_execution_spec_creation(self):
        """Test creating an EnhancedScriptExecutionSpec with PathSpecs"""
        input_spec = PathSpec(
            logical_name="training_data",
            path="/input/data.csv",
            aliases=["processed_data", "clean_data"],
        )

        output_spec = PathSpec(
            logical_name="model_output",
            path="/output/model.pkl",
            aliases=["trained_model"],
        )

        spec = EnhancedScriptExecutionSpec(
            script_name="enhanced_script",
            step_name="enhanced_step",
            input_path_specs={"training_data": input_spec},
            output_path_specs={"model_output": output_spec},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"},
        )

        assert spec.script_name == "enhanced_script"
        assert spec.step_name == "enhanced_step"
        assert spec.input_path_specs["training_data"].logical_name == "training_data"
        assert spec.input_path_specs["training_data"].path == "/input/data.csv"
        assert spec.input_path_specs["training_data"].aliases == [
            "processed_data",
            "clean_data",
        ]
        assert spec.output_path_specs["model_output"].logical_name == "model_output"
        assert spec.environ_vars["MODEL_TYPE"] == "xgboost"
        assert spec.job_args["max_depth"] == "6"

    def test_enhanced_spec_backward_compatibility_properties(self):
        """Test backward compatibility properties for input_paths and output_paths"""
        input_spec = PathSpec(logical_name="data_input", path="/input/data.csv")
        output_spec = PathSpec(logical_name="data_output", path="/output/data.csv")

        spec = EnhancedScriptExecutionSpec(
            script_name="compat_test",
            step_name="compat_step",
            input_path_specs={"data_input": input_spec},
            output_path_specs={"data_output": output_spec},
        )

        # Test backward compatibility properties
        assert spec.input_paths["data_input"] == "/input/data.csv"
        assert spec.output_paths["data_output"] == "/output/data.csv"

    def test_enhanced_spec_from_script_execution_spec(self):
        """Test creating EnhancedScriptExecutionSpec from basic ScriptExecutionSpec"""
        # Create basic spec
        basic_spec = ScriptExecutionSpec(
            script_name="convert_test",
            step_name="convert_step",
            input_paths={"data_input": "/input/data.csv"},
            output_paths={"data_output": "/output/data.csv"},
            environ_vars={"TEST_VAR": "test"},
            job_args={"test_arg": "value"},
        )

        # Convert to enhanced spec
        enhanced_spec = EnhancedScriptExecutionSpec.from_script_execution_spec(
            basic_spec,
            input_aliases={"data_input": ["raw_data"]},
            output_aliases={"data_output": ["processed_data"]},
        )

        assert enhanced_spec.script_name == "convert_test"
        assert enhanced_spec.step_name == "convert_step"
        assert enhanced_spec.input_paths["data_input"] == "/input/data.csv"
        assert enhanced_spec.output_paths["data_output"] == "/output/data.csv"
        assert enhanced_spec.input_path_specs["data_input"].aliases == ["raw_data"]
        assert enhanced_spec.output_path_specs["data_output"].aliases == [
            "processed_data"
        ]


@pytest.mark.skipif(
    not ENHANCED_MODELS_AVAILABLE, reason="Enhanced models not available"
)
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


@pytest.mark.skipif(
    not ENHANCED_MODELS_AVAILABLE, reason="Enhanced models not available"
)
class TestPathSpecAndPathMatch:
    """Test PathSpec and PathMatch models"""

    def test_path_spec_creation(self):
        """Test creating a PathSpec with all fields"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/data/processed.csv",
            aliases=["prep_output", "clean_data"],
        )

        assert spec.logical_name == "processed_data"
        assert spec.path == "/data/processed.csv"
        assert spec.aliases == ["prep_output", "clean_data"]

    def test_path_spec_with_minimal_fields(self):
        """Test PathSpec with minimal required fields"""
        spec = PathSpec(logical_name="data_input", path="/data/input.csv")

        assert spec.logical_name == "data_input"
        assert spec.path == "/data/input.csv"
        assert spec.aliases == []  # Default empty list

    def test_path_spec_matches_name_or_alias(self):
        """Test PathSpec matches_name_or_alias method"""
        spec = PathSpec(
            logical_name="processed_data",
            path="/data/processed.csv",
            aliases=["clean_data", "prep_output"],
        )

        # Should match logical name
        assert spec.matches_name_or_alias("processed_data") is True

        # Should match aliases
        assert spec.matches_name_or_alias("clean_data") is True
        assert spec.matches_name_or_alias("prep_output") is True

        # Should not match unrelated names
        assert spec.matches_name_or_alias("raw_data") is False
        assert spec.matches_name_or_alias("other_data") is False

    def test_path_match_creation(self):
        """Test creating a PathMatch"""
        match = PathMatch(
            source_logical_name="processed_data",
            dest_logical_name="training_data",
            match_type=MatchType.EXACT_LOGICAL,
            confidence=0.95,
            matched_source_name="processed_data",
            matched_dest_name="training_data",
        )

        assert match.source_logical_name == "processed_data"
        assert match.dest_logical_name == "training_data"
        assert match.match_type == MatchType.EXACT_LOGICAL
        assert match.confidence == 0.95
        assert match.matched_source_name == "processed_data"
        assert match.matched_dest_name == "training_data"

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

    def test_path_match_with_different_match_types(self):
        """Test PathMatch with different match types"""
        # Test alias to logical match
        match = PathMatch(
            source_logical_name="model_data",
            dest_logical_name="trained_model",
            match_type=MatchType.ALIAS_TO_LOGICAL,
            confidence=0.75,
            matched_source_name="model_output",  # alias
            matched_dest_name="trained_model",  # logical name
        )

        assert match.match_type == MatchType.ALIAS_TO_LOGICAL
        assert match.confidence == 0.75

        # Test semantic match
        semantic_match = PathMatch(
            source_logical_name="feature_data",
            dest_logical_name="feature_input",
            match_type=MatchType.SEMANTIC,
            confidence=0.8,
            matched_source_name="feature_data",
            matched_dest_name="feature_input",
            semantic_details={"similarity_score": 0.8, "method": "word_embedding"},
        )

        assert semantic_match.match_type == MatchType.SEMANTIC
        assert semantic_match.confidence == 0.8
        assert semantic_match.semantic_details is not None


class TestModelSerialization:
    """Test model serialization and deserialization"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_script_execution_spec_json_serialization(self, temp_dir):
        """Test ScriptExecutionSpec JSON serialization"""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_paths={"data_input": "/input/data.csv"},
            output_paths={"data_output": "/output/data.csv"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": 123},
        )

        # Test model_dump
        spec_dict = spec.model_dump()
        assert spec_dict["script_name"] == "test_script"
        assert spec_dict["input_paths"]["data_input"] == "/input/data.csv"
        assert spec_dict["job_args"]["test_arg"] == 123

        # Test JSON serialization
        json_str = json.dumps(spec_dict)
        loaded_dict = json.loads(json_str)

        # Create new spec from loaded dict
        new_spec = ScriptExecutionSpec(**loaded_dict)
        assert new_spec.script_name == spec.script_name
        assert new_spec.input_paths == spec.input_paths
        assert new_spec.job_args == spec.job_args

    def test_script_test_result_serialization(self):
        """Test ScriptTestResult serialization"""
        result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=1.5,
            has_main_function=True,
            error_message=None,
        )

        result_dict = result.model_dump()
        assert result_dict["script_name"] == "test_script"
        assert result_dict["success"] is True
        assert result_dict["execution_time"] == 1.5
        assert result_dict["has_main_function"] is True
        assert result_dict["error_message"] is None

        # Test reconstruction
        new_result = ScriptTestResult(**result_dict)
        assert new_result.script_name == result.script_name
        assert new_result.success == result.success
        assert new_result.execution_time == result.execution_time

    def test_data_compatibility_result_serialization(self):
        """Test DataCompatibilityResult serialization"""
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            compatibility_issues=["Column mismatch", "Type error"],
            data_format_a="csv",
            data_format_b="json",
        )

        result_dict = result.model_dump()
        assert result_dict["script_a"] == "script_a"
        assert result_dict["compatible"] is False
        assert len(result_dict["compatibility_issues"]) == 2
        assert result_dict["data_format_a"] == "csv"

        # Test reconstruction
        new_result = DataCompatibilityResult(**result_dict)
        assert new_result.script_a == result.script_a
        assert new_result.compatible == result.compatible
        assert new_result.compatibility_issues == result.compatibility_issues


class TestModelValidation:
    """Test model validation and error handling"""

    def test_script_execution_spec_required_fields(self):
        """Test ScriptExecutionSpec required field validation"""
        # Missing script_name should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ScriptExecutionSpec(
                step_name="test_step"
                # Missing script_name
            )

        # Missing step_name should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ScriptExecutionSpec(
                script_name="test_script"
                # Missing step_name
            )

    def test_script_test_result_required_fields(self):
        """Test ScriptTestResult required field validation"""
        # Missing script_name should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ScriptTestResult(
                success=True
                # Missing script_name
            )

        # Missing success should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ScriptTestResult(
                script_name="test_script"
                # Missing success
            )

    def test_data_compatibility_result_required_fields(self):
        """Test DataCompatibilityResult required field validation"""
        # Missing script_a should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            DataCompatibilityResult(
                script_b="script_b",
                compatible=True,
                # Missing script_a
            )

    def test_script_execution_spec_default_values(self):
        """Test ScriptExecutionSpec default values"""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            # All other fields should have defaults
        )

        assert spec.script_path is None
        assert spec.input_paths == {}
        assert spec.output_paths == {}
        assert spec.environ_vars == {}
        assert spec.job_args == {}
        assert spec.last_updated is None
        assert spec.user_notes is None

    def test_script_test_result_default_values(self):
        """Test ScriptTestResult default values"""
        result = ScriptTestResult(
            script_name="test_script",
            success=True,
            # Other fields should have defaults
        )

        assert result.error_message is None
        assert result.execution_time == 0.0
        assert result.has_main_function is False

    def test_data_compatibility_result_default_values(self):
        """Test DataCompatibilityResult default values"""
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            # Other fields should have defaults
        )

        assert result.compatibility_issues == []
        assert result.data_format_a is None
        assert result.data_format_b is None


class TestModelIntegration:
    """Test model integration and relationships"""

    def test_pipeline_testing_spec_with_multiple_scripts(self):
        """Test PipelineTestingSpec with multiple script specs"""
        # Create complex DAG
        dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval"),
            ],
        )

        # Create script specs for all nodes
        script_specs = {}
        for node in dag.nodes:
            script_specs[node] = ScriptExecutionSpec.create_default(
                script_name=node, step_name=f"{node}_step"
            )

        # Create pipeline spec
        pipeline_spec = PipelineTestingSpec(
            dag=dag, script_specs=script_specs, test_workspace_root="test/integration"
        )

        assert len(pipeline_spec.script_specs) == 4
        assert all(node in pipeline_spec.script_specs for node in dag.nodes)
        assert pipeline_spec.test_workspace_root == "test/integration"

        # Verify each script spec is properly configured
        for node, spec in pipeline_spec.script_specs.items():
            assert spec.script_name == node
            assert spec.step_name == f"{node}_step"
            assert "data_input" in spec.input_paths
            assert "data_output" in spec.output_paths

    def test_runtime_testing_configuration_integration(self):
        """Test RuntimeTestingConfiguration with complete pipeline"""
        # Create complex DAG
        dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train"],
            edges=[("data_prep", "feature_eng"), ("feature_eng", "model_train")],
        )

        # Create script specs
        script_specs = {}
        for node in dag.nodes:
            script_specs[node] = ScriptExecutionSpec.create_default(
                script_name=node, step_name=f"{node}_step"
            )

        # Create pipeline spec
        pipeline_spec = PipelineTestingSpec(
            dag=dag, script_specs=script_specs, test_workspace_root="test/integration"
        )

        # Create configuration
        config = RuntimeTestingConfiguration(
            pipeline_spec=pipeline_spec,
            test_individual_scripts=True,
            test_data_compatibility=True,
            test_pipeline_flow=True,
            use_workspace_aware=True,
        )

        # Verify configuration
        assert config.pipeline_spec == pipeline_spec
        assert config.test_individual_scripts is True
        assert config.test_data_compatibility is True
        assert config.test_pipeline_flow is True
        assert config.use_workspace_aware is True

        # Verify pipeline spec integration
        assert len(config.pipeline_spec.script_specs) == 3
        assert config.pipeline_spec.test_workspace_root == "test/integration"

        # Verify DAG structure
        assert len(config.pipeline_spec.dag.nodes) == 3
        assert len(config.pipeline_spec.dag.edges) == 2

    def test_model_relationships_and_dependencies(self):
        """Test relationships between different models"""
        # Create script test results
        script_results = [
            ScriptTestResult(
                script_name="data_prep",
                success=True,
                execution_time=0.5,
                has_main_function=True,
            ),
            ScriptTestResult(
                script_name="feature_eng",
                success=True,
                execution_time=1.2,
                has_main_function=True,
            ),
        ]

        # Create data compatibility result
        compatibility_result = DataCompatibilityResult(
            script_a="data_prep",
            script_b="feature_eng",
            compatible=True,
            data_format_a="csv",
            data_format_b="csv",
        )

        # Verify relationships
        assert script_results[0].script_name == compatibility_result.script_a
        assert script_results[1].script_name == compatibility_result.script_b
        assert all(result.success for result in script_results)
        assert compatibility_result.compatible is True

        # Test aggregated results
        total_execution_time = sum(result.execution_time for result in script_results)
        assert total_execution_time == 1.7

        successful_scripts = [r for r in script_results if r.success]
        assert len(successful_scripts) == 2


class TestModelEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_script_execution_spec(self):
        """Test ScriptExecutionSpec with minimal data"""
        spec = ScriptExecutionSpec(
            script_name="minimal_script", step_name="minimal_step"
        )

        assert spec.script_name == "minimal_script"
        assert spec.step_name == "minimal_step"
        assert spec.input_paths == {}
        assert spec.output_paths == {}
        assert spec.environ_vars == {}
        assert spec.job_args == {}

    def test_script_test_result_zero_execution_time(self):
        """Test ScriptTestResult with zero execution time"""
        result = ScriptTestResult(
            script_name="instant_script", success=True, execution_time=0.0
        )

        assert result.execution_time == 0.0
        assert result.success is True

    def test_data_compatibility_result_empty_issues(self):
        """Test DataCompatibilityResult with empty compatibility issues"""
        result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            compatibility_issues=[],
        )

        assert result.compatible is True
        assert len(result.compatibility_issues) == 0

    def test_pipeline_testing_spec_single_node(self):
        """Test PipelineTestingSpec with single node DAG"""
        dag = PipelineDAG(nodes=["single_script"], edges=[])
        spec = ScriptExecutionSpec.create_default("single_script", "single_step")

        pipeline_spec = PipelineTestingSpec(
            dag=dag, script_specs={"single_script": spec}
        )

        assert len(pipeline_spec.dag.nodes) == 1
        assert len(pipeline_spec.dag.edges) == 0
        assert len(pipeline_spec.script_specs) == 1

    def test_large_job_args_and_environ_vars(self):
        """Test ScriptExecutionSpec with large job_args and environ_vars"""
        large_job_args = {f"arg_{i}": f"value_{i}" for i in range(100)}
        large_environ_vars = {f"ENV_VAR_{i}": f"env_value_{i}" for i in range(50)}

        spec = ScriptExecutionSpec(
            script_name="large_config_script",
            step_name="large_config_step",
            job_args=large_job_args,
            environ_vars=large_environ_vars,
        )

        assert len(spec.job_args) == 100
        assert len(spec.environ_vars) == 50
        assert spec.job_args["arg_42"] == "value_42"
        assert spec.environ_vars["ENV_VAR_25"] == "env_value_25"

    def test_special_characters_in_paths(self):
        """Test ScriptExecutionSpec with special characters in paths"""
        spec = ScriptExecutionSpec(
            script_name="special_chars_script",
            step_name="special_chars_step",
            input_paths={
                "data_with_spaces": "/path with spaces/data.csv",
                "data_with_unicode": "/path/with/üñíçødé/data.csv",
                "data_with_symbols": "/path/with/@#$%/data.csv",
            },
            output_paths={"output_special": "/output/with-dashes_and.dots/result.json"},
        )

        assert spec.input_paths["data_with_spaces"] == "/path with spaces/data.csv"
        assert spec.input_paths["data_with_unicode"] == "/path/with/üñíçødé/data.csv"
        assert spec.input_paths["data_with_symbols"] == "/path/with/@#$%/data.csv"
        assert (
            spec.output_paths["output_special"]
            == "/output/with-dashes_and.dots/result.json"
        )


class TestModelPerformance:
    """Test model performance and memory usage"""

    def test_large_pipeline_spec_creation(self):
        """Test creating PipelineTestingSpec with many nodes"""
        # Create DAG with many nodes
        num_nodes = 100
        nodes = [f"script_{i}" for i in range(num_nodes)]
        edges = [(f"script_{i}", f"script_{i+1}") for i in range(num_nodes - 1)]

        dag = PipelineDAG(nodes=nodes, edges=edges)

        # Create script specs for all nodes
        script_specs = {}
        for node in nodes:
            script_specs[node] = ScriptExecutionSpec.create_default(
                script_name=node, step_name=f"{node}_step"
            )

        # Create pipeline spec
        pipeline_spec = PipelineTestingSpec(dag=dag, script_specs=script_specs)

        assert len(pipeline_spec.script_specs) == num_nodes
        assert len(pipeline_spec.dag.nodes) == num_nodes
        assert len(pipeline_spec.dag.edges) == num_nodes - 1

    def test_model_serialization_performance(self):
        """Test serialization performance with large models"""
        # Create large script spec
        large_spec = ScriptExecutionSpec(
            script_name="performance_test_script",
            step_name="performance_test_step",
            input_paths={f"input_{i}": f"/path/to/input_{i}.csv" for i in range(50)},
            output_paths={f"output_{i}": f"/path/to/output_{i}.csv" for i in range(50)},
            environ_vars={f"ENV_{i}": f"value_{i}" for i in range(100)},
            job_args={f"arg_{i}": f"value_{i}" for i in range(100)},
        )

        # Test serialization
        spec_dict = large_spec.model_dump()
        assert len(spec_dict["input_paths"]) == 50
        assert len(spec_dict["output_paths"]) == 50
        assert len(spec_dict["environ_vars"]) == 100
        assert len(spec_dict["job_args"]) == 100

        # Test JSON serialization
        json_str = json.dumps(spec_dict)
        assert len(json_str) > 1000  # Should be a substantial JSON string

        # Test deserialization
        loaded_dict = json.loads(json_str)
        new_spec = ScriptExecutionSpec(**loaded_dict)
        assert new_spec.script_name == large_spec.script_name
        assert len(new_spec.input_paths) == 50
        assert len(new_spec.output_paths) == 50
