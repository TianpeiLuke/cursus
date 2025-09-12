"""
Pytest tests for runtime spec builder

Tests the PipelineTestingSpecBuilder class used for building pipeline testing
specifications from DAG structures.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from cursus.validation.runtime.runtime_models import (
    ScriptExecutionSpec,
    PipelineTestingSpec,
)
from cursus.api.dag.base_dag import PipelineDAG


class TestPipelineTestingSpecBuilder:
    """Test PipelineTestingSpecBuilder class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create a simple DAG for testing"""
        return PipelineDAG(
            nodes=["script_a", "script_b", "script_c"],
            edges=[("script_a", "script_b"), ("script_b", "script_c")],
        )

    @pytest.fixture
    def builder(self, temp_dir):
        """Create builder with temporary directory"""
        return PipelineTestingSpecBuilder(test_data_dir=temp_dir)

    def test_builder_initialization(self):
        """Test PipelineTestingSpecBuilder initialization"""
        builder = PipelineTestingSpecBuilder()

        assert str(builder.test_data_dir) == "test/integration/runtime"
        assert builder.specs_dir.name == ".specs"

    def test_builder_initialization_with_params(self):
        """Test PipelineTestingSpecBuilder initialization with parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = PipelineTestingSpecBuilder(test_data_dir=temp_dir)

            assert str(builder.test_data_dir) == temp_dir
            assert builder.specs_dir.name == ".specs"

    def test_build_from_dag(self, builder, test_dag):
        """Test building PipelineTestingSpec from DAG"""
        pipeline_spec = builder.build_from_dag(test_dag, validate=False)

        assert isinstance(pipeline_spec, PipelineTestingSpec)
        assert pipeline_spec.dag == test_dag
        assert len(pipeline_spec.script_specs) == 3

        # Verify all scripts have specs
        for script_name in ["script_a", "script_b", "script_c"]:
            assert script_name in pipeline_spec.script_specs
            spec = pipeline_spec.script_specs[script_name]
            assert spec.script_name == script_name

    def test_load_or_create_script_spec_new(self, builder):
        """Test loading or creating script spec when none exists"""
        spec = builder._load_or_create_script_spec("new_script")

        assert isinstance(spec, ScriptExecutionSpec)
        assert spec.script_name == "new_script"
        assert spec.step_name == "new_script"
        assert "data_input" in spec.input_paths
        assert "data_output" in spec.output_paths

    def test_load_or_create_script_spec_existing(self, builder):
        """Test loading existing script spec"""
        # Create and save a spec first
        original_spec = ScriptExecutionSpec(
            script_name="existing_script",
            step_name="existing_step",
            input_paths={"custom_input": "/custom/path"},
            output_paths={"custom_output": "/custom/output"},
            environ_vars={"CUSTOM_VAR": "custom_value"},
            job_args={"custom_arg": "custom_value"},
        )
        original_spec.save_to_file(str(builder.specs_dir))

        # Load the spec using the builder
        loaded_spec = builder._load_or_create_script_spec("existing_script")

        assert loaded_spec.script_name == "existing_script"
        assert loaded_spec.step_name == "existing_step"
        assert loaded_spec.input_paths["custom_input"] == "/custom/path"
        assert loaded_spec.environ_vars["CUSTOM_VAR"] == "custom_value"

    def test_save_script_spec(self, builder):
        """Test saving script spec"""
        spec = ScriptExecutionSpec.create_default("test_script", "test_step")

        # Save the spec
        builder.save_script_spec(spec)

        # Verify it was saved
        saved_specs = builder.list_saved_specs()
        assert "test_script" in saved_specs

    def test_update_script_spec(self, builder):
        """Test updating script spec"""
        # Create initial spec
        original_spec = ScriptExecutionSpec.create_default("update_test", "update_step")
        builder.save_script_spec(original_spec)

        # Update the spec
        updated_spec = builder.update_script_spec(
            "update_test",
            input_paths={"new_input": "/new/path"},
            environ_vars={"NEW_VAR": "new_value"},
        )

        assert updated_spec.input_paths["new_input"] == "/new/path"
        assert updated_spec.environ_vars["NEW_VAR"] == "new_value"

    def test_list_saved_specs(self, builder):
        """Test listing saved specs"""
        # Create and save some specs
        spec1 = ScriptExecutionSpec.create_default("spec1", "step1")
        spec2 = ScriptExecutionSpec.create_default("spec2", "step2")

        builder.save_script_spec(spec1)
        builder.save_script_spec(spec2)

        saved_specs = builder.list_saved_specs()

        assert "spec1" in saved_specs
        assert "spec2" in saved_specs
        assert len(saved_specs) == 2

    def test_get_script_spec_by_name(self, builder):
        """Test getting script spec by name"""
        # Create and save a spec
        original_spec = ScriptExecutionSpec.create_default("get_test", "get_step")
        builder.save_script_spec(original_spec)

        # Get the spec by name
        retrieved_spec = builder.get_script_spec_by_name("get_test")

        assert retrieved_spec is not None
        assert retrieved_spec.script_name == "get_test"
        assert retrieved_spec.step_name == "get_step"

    def test_get_script_spec_by_name_nonexistent(self, builder):
        """Test getting non-existent script spec returns None"""
        retrieved_spec = builder.get_script_spec_by_name("nonexistent")
        assert retrieved_spec is None

    def test_match_step_to_spec_direct_match(self, builder):
        """Test direct step to spec matching"""
        available_specs = ["script_a", "script_b", "script_c"]

        match = builder.match_step_to_spec("script_a", available_specs)
        assert match == "script_a"

    def test_match_step_to_spec_variation_match(self, builder):
        """Test step to spec matching with variations"""
        available_specs = ["script_a", "script_b", "script_c"]

        # Test lowercase variation
        match = builder.match_step_to_spec("SCRIPT_A", available_specs)
        assert match == "script_a"

    def test_match_step_to_spec_no_match(self, builder):
        """Test step to spec matching with no match"""
        available_specs = ["script_a", "script_b", "script_c"]

        match = builder.match_step_to_spec("completely_different", available_specs)
        assert match is None

    def test_is_spec_complete_valid(self, builder):
        """Test spec completeness validation for valid spec"""
        spec = ScriptExecutionSpec(
            script_name="complete_script",
            step_name="complete_step",
            input_paths={"data_input": "/valid/input"},
            output_paths={"data_output": "/valid/output"},
            environ_vars={"LABEL_FIELD": "label"},
            job_args={"job_type": "testing"},
        )

        is_complete = builder._is_spec_complete(spec)
        assert is_complete is True

    def test_is_spec_complete_invalid(self, builder):
        """Test spec completeness validation for invalid spec"""
        spec = ScriptExecutionSpec(
            script_name="incomplete_script",
            step_name="incomplete_step",
            input_paths={},  # Empty paths
            output_paths={},  # Empty paths
            environ_vars={},
            job_args={},
        )

        is_complete = builder._is_spec_complete(spec)
        assert is_complete is False

    def test_validate_specs_completeness_valid(self, builder):
        """Test validation with complete specs"""
        dag_nodes = ["script_a", "script_b"]
        missing_specs = []
        incomplete_specs = []

        # Should not raise exception
        try:
            builder._validate_specs_completeness(
                dag_nodes, missing_specs, incomplete_specs
            )
        except ValueError:
            pytest.fail("_validate_specs_completeness raised ValueError unexpectedly")

    def test_validate_specs_completeness_invalid(self, builder):
        """Test validation with incomplete specs"""
        dag_nodes = ["script_a", "script_b"]
        missing_specs = ["script_a"]
        incomplete_specs = ["script_b"]

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            builder._validate_specs_completeness(
                dag_nodes, missing_specs, incomplete_specs
            )

        error_message = str(exc_info.value)
        assert "Missing ScriptExecutionSpec" in error_message
        assert "Incomplete ScriptExecutionSpec" in error_message

    def test_get_script_main_params(self, builder):
        """Test getting script main parameters"""
        spec = ScriptExecutionSpec(
            script_name="param_test",
            step_name="param_step",
            input_paths={"data_input": "/test/input"},
            output_paths={"data_output": "/test/output"},
            environ_vars={"TEST_VAR": "test_value"},
            job_args={"test_arg": "test_value"},
        )

        params = builder.get_script_main_params(spec)

        assert params["input_paths"]["data_input"] == "/test/input"
        assert params["output_paths"]["data_output"] == "/test/output"
        assert params["environ_vars"]["TEST_VAR"] == "test_value"
        assert params["job_args"].test_arg == "test_value"

    @patch("builtins.input")
    def test_update_script_spec_interactive(self, mock_input, builder):
        """Test interactive script spec update"""
        # Mock user inputs
        mock_input.side_effect = [
            "/interactive/input",  # input path
            "/interactive/output",  # output path
            "",  # environment variables (use defaults)
            "",  # job arguments (use defaults)
        ]

        # Create a spec with empty paths
        spec = ScriptExecutionSpec(
            script_name="interactive_test",
            step_name="interactive_step",
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args={},
        )
        builder.save_script_spec(spec)

        # Update interactively
        updated_spec = builder.update_script_spec_interactive("interactive_test")

        assert updated_spec.input_paths["data_input"] == "/interactive/input"
        assert updated_spec.output_paths["data_output"] == "/interactive/output"
        assert updated_spec.environ_vars["LABEL_FIELD"] == "label"
        assert updated_spec.job_args["job_type"] == "testing"


class TestPipelineTestingSpecBuilderIntegration:
    """Integration tests for PipelineTestingSpecBuilder"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def builder(self, temp_dir):
        """Create builder with temporary directory"""
        return PipelineTestingSpecBuilder(test_data_dir=temp_dir)

    def test_end_to_end_spec_building(self, builder):
        """Test complete end-to-end spec building workflow"""
        # Create a complex DAG
        complex_dag = PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval"),
            ],
        )

        # Build pipeline spec (without validation to avoid completeness issues)
        pipeline_spec = builder.build_from_dag(complex_dag, validate=False)

        # Verify the complete spec
        assert len(pipeline_spec.script_specs) == 4
        assert len(pipeline_spec.dag.nodes) == 4
        assert len(pipeline_spec.dag.edges) == 3

        # Verify each script has a spec
        for node in complex_dag.nodes:
            assert node in pipeline_spec.script_specs
            spec = pipeline_spec.script_specs[node]
            assert spec.script_name == node
            assert spec.step_name == node

    def test_spec_persistence_and_updates(self, builder):
        """Test specification persistence and update workflow"""
        # Create initial DAG and spec
        dag = PipelineDAG(nodes=["script_a"], edges=[])
        initial_spec = builder.build_from_dag(dag, validate=False)

        # Modify script spec and save
        script_spec = initial_spec.script_specs["script_a"]
        script_spec.environ_vars["UPDATED_VAR"] = "updated_value"
        builder.save_script_spec(script_spec)

        # Build new spec (should load updated script spec)
        updated_spec = builder.build_from_dag(dag, validate=False)

        # Verify the update was preserved
        assert (
            updated_spec.script_specs["script_a"].environ_vars["UPDATED_VAR"]
            == "updated_value"
        )
