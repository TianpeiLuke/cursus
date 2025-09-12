"""Pytest tests for the enhanced step names registry module with workspace awareness."""

import pytest
import os
from contextlib import nullcontext
from unittest.mock import patch, MagicMock
from cursus.registry.step_names import (
    # Core registry data structures (now dynamic)
    STEP_NAMES,
    CONFIG_STEP_REGISTRY,
    BUILDER_STEP_NAMES,
    SPEC_STEP_TYPES,
    # Original helper functions (now workspace-aware)
    get_config_class_name,
    get_builder_step_name,
    get_spec_step_type,
    get_spec_step_type_with_job_type,
    get_step_name_from_spec_type,
    get_all_step_names,
    validate_step_name,
    validate_spec_type,
    get_step_description,
    list_all_step_info,
    get_canonical_name_from_file_name,
    validate_file_name,
    # NEW: Workspace context management
    set_workspace_context,
    get_workspace_context,
    clear_workspace_context,
    workspace_context,
    # NEW: Workspace-aware registry functions
    get_step_names,
    get_config_step_registry,
    get_builder_step_names,
    get_spec_step_types,
    # NEW: Workspace management functions
    list_available_workspaces,
    get_workspace_step_count,
    has_workspace_conflicts,
    # Internal functions for testing
    _get_registry_manager,
    _create_fallback_manager,
)


class TestStepNames:
    """Test case for step names registry functions."""

    def test_step_names_structure(self):
        """Test that STEP_NAMES has the expected structure."""
        assert isinstance(STEP_NAMES, dict)
        assert len(STEP_NAMES) > 0

        # Check that each entry has the required fields
        for step_name, info in STEP_NAMES.items():
            assert isinstance(step_name, str)
            assert isinstance(info, dict)

            required_fields = [
                "config_class",
                "builder_step_name",
                "spec_type",
                "description",
            ]
            for field in required_fields:
                assert field in info, f"Missing field '{field}' in {step_name}"

    def test_generated_registries(self):
        """Test that generated registries are consistent with STEP_NAMES."""
        # Test CONFIG_STEP_REGISTRY
        assert isinstance(CONFIG_STEP_REGISTRY, dict)
        for step_name, info in STEP_NAMES.items():
            config_class = info["config_class"]
            assert config_class in CONFIG_STEP_REGISTRY
            assert CONFIG_STEP_REGISTRY[config_class] == step_name

        # Test BUILDER_STEP_NAMES
        assert isinstance(BUILDER_STEP_NAMES, dict)
        for step_name, info in STEP_NAMES.items():
            assert step_name in BUILDER_STEP_NAMES
            assert BUILDER_STEP_NAMES[step_name] == info["builder_step_name"]

        # Test SPEC_STEP_TYPES
        assert isinstance(SPEC_STEP_TYPES, dict)
        for step_name, info in STEP_NAMES.items():
            assert step_name in SPEC_STEP_TYPES
            assert SPEC_STEP_TYPES[step_name] == info["spec_type"]

    def test_get_config_class_name(self):
        """Test get_config_class_name function."""
        for step_name, info in STEP_NAMES.items():
            config_class = get_config_class_name(step_name)
            assert config_class == info["config_class"]

        # Test with invalid step name
        with pytest.raises(ValueError):
            get_config_class_name("InvalidStepName")

    def test_get_builder_step_name(self):
        """Test get_builder_step_name function."""
        for step_name, info in STEP_NAMES.items():
            builder_name = get_builder_step_name(step_name)
            assert builder_name == info["builder_step_name"]

        # Test with invalid step name
        with pytest.raises(ValueError):
            get_builder_step_name("InvalidStepName")

    def test_get_spec_step_type(self):
        """Test get_spec_step_type function."""
        for step_name, info in STEP_NAMES.items():
            spec_type = get_spec_step_type(step_name)
            assert spec_type == info["spec_type"]

        # Test with invalid step name
        with pytest.raises(ValueError):
            get_spec_step_type("InvalidStepName")

    def test_get_spec_step_type_with_job_type(self):
        """Test get_spec_step_type_with_job_type function."""
        # Test without job type
        for step_name, info in STEP_NAMES.items():
            spec_type = get_spec_step_type_with_job_type(step_name)
            assert spec_type == info["spec_type"]

        # Test with job type
        step_name = "CradleDataLoading"
        spec_type_with_job = get_spec_step_type_with_job_type(step_name, "training")
        expected = f"{STEP_NAMES[step_name]['spec_type']}_Training"
        assert spec_type_with_job == expected

        # Test with invalid step name
        with pytest.raises(ValueError):
            get_spec_step_type_with_job_type("InvalidStepName")

    def test_get_step_name_from_spec_type(self):
        """Test get_step_name_from_spec_type function."""
        # Test with basic spec types
        for step_name, info in STEP_NAMES.items():
            spec_type = info["spec_type"]
            retrieved_name = get_step_name_from_spec_type(spec_type)
            assert retrieved_name == step_name

        # Test with job type variants
        spec_type_with_job = "TabularPreprocessing_Training"
        retrieved_name = get_step_name_from_spec_type(spec_type_with_job)
        assert retrieved_name == "TabularPreprocessing"

        # Test with unknown spec type
        unknown_spec = get_step_name_from_spec_type("UnknownSpecType")
        assert unknown_spec == "UnknownSpecType"

    def test_get_all_step_names(self):
        """Test get_all_step_names function."""
        all_names = get_all_step_names()

        assert isinstance(all_names, list)
        assert set(all_names) == set(STEP_NAMES.keys())

    def test_validate_step_name(self):
        """Test validate_step_name function."""
        # Test with valid step names
        for step_name in STEP_NAMES.keys():
            assert validate_step_name(step_name)

        # Test with invalid step names
        invalid_names = ["InvalidStep", "NonExistentStep", ""]
        for invalid_name in invalid_names:
            assert not validate_step_name(invalid_name)

    def test_validate_spec_type(self):
        """Test validate_spec_type function."""
        # Test with valid spec types
        for info in STEP_NAMES.values():
            spec_type = info["spec_type"]
            assert validate_spec_type(spec_type)

        # Test with job type variants
        assert validate_spec_type("TabularPreprocessing_Training")
        assert validate_spec_type("XGBoostTraining_Evaluation")

        # Test with invalid spec types
        invalid_specs = ["InvalidSpec", "NonExistentSpec", ""]
        for invalid_spec in invalid_specs:
            assert not validate_spec_type(invalid_spec)

    def test_get_step_description(self):
        """Test get_step_description function."""
        for step_name, info in STEP_NAMES.items():
            description = get_step_description(step_name)
            assert description == info["description"]

        # Test with invalid step name
        with pytest.raises(ValueError):
            get_step_description("InvalidStepName")

    def test_list_all_step_info(self):
        """Test list_all_step_info function."""
        all_info = list_all_step_info()

        assert isinstance(all_info, dict)
        assert all_info == STEP_NAMES

        # Verify it's a copy, not the original
        assert all_info is not STEP_NAMES

    def test_registry_contains_expected_steps(self):
        """Test that the registry contains expected step types."""
        expected_steps = [
            "Base",
            "Processing",
            "CradleDataLoading",
            "TabularPreprocessing",
            "PyTorchTraining",
            "XGBoostTraining",
            "XGBoostModelEval",
            "PyTorchModel",
            "XGBoostModel",
            "Package",
            "Registration",
            "Payload",
            "BatchTransform",
            "ModelCalibration",
            "RiskTableMapping",
            "CurrencyConversion",
            "DummyTraining",
            "HyperparameterPrep",
        ]

        for expected_step in expected_steps:
            assert expected_step in STEP_NAMES
            assert validate_step_name(expected_step)

    def test_config_class_uniqueness(self):
        """Test that config class names are unique across steps."""
        config_classes = [info["config_class"] for info in STEP_NAMES.values()]
        assert len(config_classes) == len(set(config_classes))

    def test_spec_type_uniqueness(self):
        """Test that spec types are unique across steps."""
        spec_types = [info["spec_type"] for info in STEP_NAMES.values()]
        assert len(spec_types) == len(set(spec_types))

    def test_descriptions_are_present(self):
        """Test that all steps have non-empty descriptions."""
        for step_name, info in STEP_NAMES.items():
            description = info["description"]

            assert isinstance(description, str)
            assert len(description) > 0
            assert description.strip() != ""

    def test_job_type_capitalization(self):
        """Test that job types are properly capitalized in spec types."""
        test_cases = [
            ("training", "Training"),
            ("evaluation", "Evaluation"),
            ("inference", "Inference"),
            ("test", "Test"),
        ]

        step_name = "CradleDataLoading"  # Use any valid step name
        for job_type, expected_suffix in test_cases:
            spec_type = get_spec_step_type_with_job_type(step_name, job_type)
            assert spec_type.endswith(f"_{expected_suffix}")

    def test_get_canonical_name_from_file_name(self):
        """Test get_canonical_name_from_file_name function."""
        # Test cases that should work with the registry
        test_cases = [
            # Basic cases
            ("xgboost_training", "XGBoostTraining"),
            ("pytorch_training", "PyTorchTraining"),
            ("tabular_preprocessing", "TabularPreprocessing"),
            ("dummy_training", "DummyTraining"),
            ("currency_conversion", "CurrencyConversion"),
            ("risk_table_mapping", "RiskTableMapping"),
            ("model_calibration", "ModelCalibration"),
            # The specific case that was failing - this is the main fix
            ("xgboost_model_evaluation", "XGBoostModelEval"),
            # Other model evaluation cases
            ("pytorch_model", "PyTorchModel"),
            ("xgboost_model", "XGBoostModel"),
            # Deployment steps
            ("package", "Package"),
            ("registration", "Registration"),
            ("payload", "Payload"),
            # Transform steps
            ("batch_transform", "BatchTransform"),
            # Data loading
            ("cradle_data_loading", "CradleDataLoading"),
        ]

        for file_name, expected_canonical in test_cases:
            with (
                pytest.raises(ValueError)
                if expected_canonical not in STEP_NAMES
                else nullcontext()
            ):
                canonical_name = get_canonical_name_from_file_name(file_name)
                if expected_canonical in STEP_NAMES:
                    assert (
                        canonical_name == expected_canonical
                    ), f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'"

    def test_get_canonical_name_from_file_name_with_job_types(self):
        """Test get_canonical_name_from_file_name with job type suffixes."""
        # Test cases with job type suffixes that should be stripped
        test_cases = [
            ("xgboost_training_training", "XGBoostTraining"),
            ("pytorch_training_validation", "PyTorchTraining"),
            ("tabular_preprocessing_testing", "TabularPreprocessing"),
            ("dummy_training_calibration", "DummyTraining"),
        ]

        for file_name, expected_canonical in test_cases:
            with (
                pytest.raises(ValueError)
                if expected_canonical not in STEP_NAMES
                else nullcontext()
            ):
                canonical_name = get_canonical_name_from_file_name(file_name)
                if expected_canonical in STEP_NAMES:
                    assert (
                        canonical_name == expected_canonical
                    ), f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'"

    def test_get_canonical_name_from_file_name_abbreviations(self):
        """Test get_canonical_name_from_file_name with abbreviations."""
        # Test cases with abbreviations
        test_cases = [
            ("xgb_training", "XGBoostTraining"),
            ("xgb_model", "XGBoostModel"),
            # The main fix: full xgboost should also work
            ("xgboost_training", "XGBoostTraining"),
            ("xgboost_model", "XGBoostModel"),
        ]

        for file_name, expected_canonical in test_cases:
            with (
                pytest.raises(ValueError)
                if expected_canonical not in STEP_NAMES
                else nullcontext()
            ):
                canonical_name = get_canonical_name_from_file_name(file_name)
                if expected_canonical in STEP_NAMES:
                    assert (
                        canonical_name == expected_canonical
                    ), f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'"

    def test_get_canonical_name_from_file_name_invalid(self):
        """Test get_canonical_name_from_file_name with invalid inputs."""
        invalid_cases = [
            "",  # Empty string
            "completely_unknown_step",
            "invalid_step_name",
            "nonexistent_builder",
        ]

        for invalid_name in invalid_cases:
            with pytest.raises(ValueError):
                get_canonical_name_from_file_name(invalid_name)

    def test_validate_file_name(self):
        """Test validate_file_name function."""
        # Test with valid file names
        valid_names = [
            "xgboost_training",
            "pytorch_training",
            "xgboost_model_evaluation",  # The main case we fixed
            "tabular_preprocessing",
            "dummy_training",
        ]

        for valid_name in valid_names:
            # Only test if the corresponding canonical name exists
            try:
                canonical = get_canonical_name_from_file_name(valid_name)
                if canonical in STEP_NAMES:
                    assert validate_file_name(valid_name)
            except ValueError:
                # If canonical name doesn't exist, validation should return False
                assert not validate_file_name(valid_name)

        # Test with invalid file names
        invalid_names = ["", "completely_unknown_step", "invalid_step_name"]

        for invalid_name in invalid_names:
            assert not validate_file_name(invalid_name)

    def test_xgboost_model_evaluation_registry_integration(self):
        """Test the complete registry integration for xgboost_model_evaluation."""
        # This is the specific test for the bug fix
        file_name = "xgboost_model_evaluation"

        # Only run this test if XGBoostModelEval exists in the registry
        if "XGBoostModelEval" not in STEP_NAMES:
            pytest.skip("XGBoostModelEval not in registry")

        # Step 1: Convert file name to canonical name
        canonical_name = get_canonical_name_from_file_name(file_name)
        assert canonical_name == "XGBoostModelEval"

        # Step 2: Get config class name from canonical name
        config_class_name = get_config_class_name(canonical_name)
        assert config_class_name == "XGBoostModelEvalConfig"

        # Step 3: Verify the registry entry exists
        assert canonical_name in STEP_NAMES

        # Step 4: Verify all registry mappings are consistent
        step_info = STEP_NAMES[canonical_name]
        assert step_info["config_class"] == "XGBoostModelEvalConfig"
        assert step_info["builder_step_name"] == "XGBoostModelEvalStepBuilder"
        assert step_info["spec_type"] == "XGBoostModelEval"
        assert step_info["sagemaker_step_type"] == "Processing"


class TestWorkspaceContextManagement:
    """Test case for workspace context management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing workspace context
        clear_workspace_context()

    def teardown_method(self):
        """Clean up after tests."""
        # Clear workspace context after each test
        clear_workspace_context()
        # Clear environment variable if set
        if "CURSUS_WORKSPACE_ID" in os.environ:
            del os.environ["CURSUS_WORKSPACE_ID"]

    def test_workspace_context_basic_operations(self):
        """Test basic workspace context operations."""
        # Initially no context
        assert get_workspace_context() is None

        # Set workspace context
        set_workspace_context("test_workspace")
        assert get_workspace_context() == "test_workspace"

        # Clear workspace context
        clear_workspace_context()
        assert get_workspace_context() is None

    def test_workspace_context_environment_variable(self):
        """Test workspace context from environment variable."""
        # Set environment variable
        os.environ["CURSUS_WORKSPACE_ID"] = "env_workspace"

        # Should return environment variable value
        assert get_workspace_context() == "env_workspace"

        # Explicit context should override environment
        set_workspace_context("explicit_workspace")
        assert get_workspace_context() == "explicit_workspace"

        # Clear explicit context, should fall back to environment
        clear_workspace_context()
        assert get_workspace_context() == "env_workspace"

    def test_workspace_context_manager(self):
        """Test workspace context manager."""
        # Set initial context
        set_workspace_context("initial_workspace")

        # Use context manager
        with workspace_context("temp_workspace"):
            assert get_workspace_context() == "temp_workspace"

        # Should restore original context
        assert get_workspace_context() == "initial_workspace"

        # Test with no initial context
        clear_workspace_context()
        with workspace_context("temp_workspace"):
            assert get_workspace_context() == "temp_workspace"

        # Should be None after context manager
        assert get_workspace_context() is None

    def test_workspace_aware_functions(self):
        """Test that functions work with workspace context."""
        # Test without workspace context
        step_names_core = get_step_names()
        assert isinstance(step_names_core, dict)

        # Test with workspace context
        set_workspace_context("test_workspace")
        step_names_workspace = get_step_names()
        assert isinstance(step_names_workspace, dict)

        # Test explicit workspace parameter
        step_names_explicit = get_step_names("explicit_workspace")
        assert isinstance(step_names_explicit, dict)

        # Test other workspace-aware functions
        config_registry = get_config_step_registry("test_workspace")
        assert isinstance(config_registry, dict)

        builder_names = get_builder_step_names("test_workspace")
        assert isinstance(builder_names, dict)

        spec_types = get_spec_step_types("test_workspace")
        assert isinstance(spec_types, dict)

    def test_workspace_aware_helper_functions(self):
        """Test that helper functions work with workspace context."""
        # Test with workspace context
        set_workspace_context("test_workspace")

        # These should work the same as before
        config_class = get_config_class_name("XGBoostTraining")
        assert config_class == "XGBoostTrainingConfig"

        # Test with explicit workspace parameter
        config_class_explicit = get_config_class_name(
            "XGBoostTraining", "explicit_workspace"
        )
        assert config_class_explicit == "XGBoostTrainingConfig"

        # Test other functions
        builder_name = get_builder_step_name("XGBoostTraining", "test_workspace")
        assert builder_name == "XGBoostTrainingStepBuilder"

        spec_type = get_spec_step_type("XGBoostTraining", "test_workspace")
        assert spec_type == "XGBoostTraining"

        # Test validation functions
        assert validate_step_name("XGBoostTraining", "test_workspace")
        assert not validate_step_name("NonExistentStep", "test_workspace")


class TestRegistryManagerAndFallback:
    """Test case for registry manager and fallback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_workspace_context()

    def teardown_method(self):
        """Clean up after tests."""
        clear_workspace_context()

    def test_registry_manager_initialization(self):
        """Test registry manager initialization."""
        manager = _get_registry_manager()
        assert manager is not None

        # Should have required methods
        assert hasattr(manager, "create_legacy_step_names_dict")

    def test_fallback_manager_creation(self):
        """Test fallback manager creation."""
        fallback_manager = _create_fallback_manager()
        assert fallback_manager is not None

        # Should have required methods
        assert hasattr(fallback_manager, "create_legacy_step_names_dict")
        assert hasattr(fallback_manager, "get_step_definition")
        assert hasattr(fallback_manager, "has_step")
        assert hasattr(fallback_manager, "list_steps")

        # Test fallback functionality
        step_names = fallback_manager.create_legacy_step_names_dict()
        assert isinstance(step_names, dict)
        assert len(step_names) > 0

        # Test specific step lookup
        assert fallback_manager.has_step("XGBoostTraining")
        assert not fallback_manager.has_step("NonExistentStep")

        step_list = fallback_manager.list_steps()
        assert isinstance(step_list, list)
        assert "XGBoostTraining" in step_list

    def test_hybrid_registry_fallback_behavior(self):
        """Test that fallback manager provides expected functionality."""
        # Test the fallback manager directly
        fallback_manager = _create_fallback_manager()
        assert fallback_manager is not None

        # Should provide basic functionality
        step_names = fallback_manager.create_legacy_step_names_dict()
        assert isinstance(step_names, dict)
        assert len(step_names) > 0

        # Should contain expected steps
        assert "XGBoostTraining" in step_names
        assert "PyTorchTraining" in step_names

        # Test that it behaves like the original registry
        xgb_info = step_names["XGBoostTraining"]
        assert xgb_info["config_class"] == "XGBoostTrainingConfig"
        assert xgb_info["builder_step_name"] == "XGBoostTrainingStepBuilder"

    def test_workspace_management_functions(self):
        """Test workspace management functions."""
        # These functions should not fail even if hybrid registry is not available
        workspaces = list_available_workspaces()
        assert isinstance(workspaces, list)

        step_count = get_workspace_step_count("test_workspace")
        assert isinstance(step_count, int)
        assert step_count >= 0

        conflicts = has_workspace_conflicts()
        assert isinstance(conflicts, bool)


class TestBackwardCompatibility:
    """Test case for backward compatibility with original implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_workspace_context()

    def teardown_method(self):
        """Clean up after tests."""
        clear_workspace_context()

    def test_module_level_variables_exist(self):
        """Test that module-level variables still exist and work."""
        # These should exist and be dictionaries
        assert isinstance(STEP_NAMES, dict)
        assert isinstance(CONFIG_STEP_REGISTRY, dict)
        assert isinstance(BUILDER_STEP_NAMES, dict)
        assert isinstance(SPEC_STEP_TYPES, dict)

        # Should have content
        assert len(STEP_NAMES) > 0
        assert len(CONFIG_STEP_REGISTRY) > 0
        assert len(BUILDER_STEP_NAMES) > 0
        assert len(SPEC_STEP_TYPES) > 0

    def test_original_function_signatures(self):
        """Test that original function signatures still work."""
        # These should work without workspace parameters
        config_class = get_config_class_name("XGBoostTraining")
        assert config_class == "XGBoostTrainingConfig"

        builder_name = get_builder_step_name("XGBoostTraining")
        assert builder_name == "XGBoostTrainingStepBuilder"

        spec_type = get_spec_step_type("XGBoostTraining")
        assert spec_type == "XGBoostTraining"

        # Validation functions
        assert validate_step_name("XGBoostTraining")
        assert not validate_step_name("NonExistentStep")

        # File name resolution
        canonical_name = get_canonical_name_from_file_name("xgboost_training")
        assert canonical_name == "XGBoostTraining"

    def test_error_messages_consistency(self):
        """Test that error messages are consistent with original implementation."""
        # Test invalid step name error
        with pytest.raises(ValueError) as exc_info:
            get_config_class_name("InvalidStepName")

        error_message = str(exc_info.value)
        assert "Unknown step name" in error_message
        assert "InvalidStepName" in error_message
        assert "Available steps" in error_message

    def test_data_structure_consistency(self):
        """Test that data structures maintain expected format."""
        # STEP_NAMES should have expected structure
        for step_name, info in STEP_NAMES.items():
            assert isinstance(step_name, str)
            assert isinstance(info, dict)

            # Required fields
            required_fields = [
                "config_class",
                "builder_step_name",
                "spec_type",
                "description",
            ]
            for field in required_fields:
                assert field in info
                assert isinstance(info[field], str)

        # Generated registries should be consistent
        for config_class, step_name in CONFIG_STEP_REGISTRY.items():
            assert step_name in STEP_NAMES
            assert STEP_NAMES[step_name]["config_class"] == config_class

        for step_name, builder_name in BUILDER_STEP_NAMES.items():
            assert step_name in STEP_NAMES
            assert STEP_NAMES[step_name]["builder_step_name"] == builder_name

        for step_name, spec_type in SPEC_STEP_TYPES.items():
            assert step_name in STEP_NAMES
            assert STEP_NAMES[step_name]["spec_type"] == spec_type


class TestEnhancedFunctionality:
    """Test case for enhanced functionality added in Phase 5."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_workspace_context()

    def teardown_method(self):
        """Clean up after tests."""
        clear_workspace_context()

    def test_enhanced_file_name_resolution(self):
        """Test enhanced file name resolution with workspace context."""
        # Test with workspace context
        set_workspace_context("test_workspace")

        canonical_name = get_canonical_name_from_file_name("xgboost_model_evaluation")
        assert canonical_name == "XGBoostModelEval"

        # Test with explicit workspace parameter
        canonical_name_explicit = get_canonical_name_from_file_name(
            "xgboost_model_evaluation", "explicit_workspace"
        )
        assert canonical_name_explicit == "XGBoostModelEval"

    def test_workspace_aware_error_messages(self):
        """Test that error messages include workspace context information."""
        set_workspace_context("test_workspace")

        with pytest.raises(ValueError) as exc_info:
            get_canonical_name_from_file_name("completely_unknown_step")

        error_message = str(exc_info.value)
        # Should include workspace context in error message
        assert "workspace: test_workspace" in error_message

    def test_environment_variable_integration(self):
        """Test integration with CURSUS_WORKSPACE_ID environment variable."""
        # Set environment variable
        os.environ["CURSUS_WORKSPACE_ID"] = "env_test_workspace"

        try:
            # Should use environment variable
            context = get_workspace_context()
            assert context == "env_test_workspace"

            # Functions should use environment context
            step_names = get_step_names()  # Should use env context
            assert isinstance(step_names, dict)

        finally:
            # Clean up
            if "CURSUS_WORKSPACE_ID" in os.environ:
                del os.environ["CURSUS_WORKSPACE_ID"]
