"""Unit tests for the enhanced step names registry module with workspace awareness."""

import unittest
import os
from unittest.mock import patch, MagicMock
from src.cursus.registry.step_names import (
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
    _create_fallback_manager
)


class TestStepNames(unittest.TestCase):
    """Test case for step names registry functions."""

    def test_step_names_structure(self):
        """Test that STEP_NAMES has the expected structure."""
        self.assertIsInstance(STEP_NAMES, dict)
        self.assertGreater(len(STEP_NAMES), 0)
        
        # Check that each entry has the required fields
        for step_name, info in STEP_NAMES.items():
            self.assertIsInstance(step_name, str)
            self.assertIsInstance(info, dict)
            
            required_fields = ["config_class", "builder_step_name", "spec_type", "description"]
            for field in required_fields:
                self.assertIn(field, info, f"Missing field '{field}' in {step_name}")

    def test_generated_registries(self):
        """Test that generated registries are consistent with STEP_NAMES."""
        # Test CONFIG_STEP_REGISTRY
        self.assertIsInstance(CONFIG_STEP_REGISTRY, dict)
        for step_name, info in STEP_NAMES.items():
            config_class = info["config_class"]
            self.assertIn(config_class, CONFIG_STEP_REGISTRY)
            self.assertEqual(CONFIG_STEP_REGISTRY[config_class], step_name)
        
        # Test BUILDER_STEP_NAMES
        self.assertIsInstance(BUILDER_STEP_NAMES, dict)
        for step_name, info in STEP_NAMES.items():
            self.assertIn(step_name, BUILDER_STEP_NAMES)
            self.assertEqual(BUILDER_STEP_NAMES[step_name], info["builder_step_name"])
        
        # Test SPEC_STEP_TYPES
        self.assertIsInstance(SPEC_STEP_TYPES, dict)
        for step_name, info in STEP_NAMES.items():
            self.assertIn(step_name, SPEC_STEP_TYPES)
            self.assertEqual(SPEC_STEP_TYPES[step_name], info["spec_type"])

    def test_get_config_class_name(self):
        """Test get_config_class_name function."""
        for step_name, info in STEP_NAMES.items():
            config_class = get_config_class_name(step_name)
            self.assertEqual(config_class, info["config_class"])
        
        # Test with invalid step name
        with self.assertRaises(ValueError):
            get_config_class_name("InvalidStepName")

    def test_get_builder_step_name(self):
        """Test get_builder_step_name function."""
        for step_name, info in STEP_NAMES.items():
            builder_name = get_builder_step_name(step_name)
            self.assertEqual(builder_name, info["builder_step_name"])
        
        # Test with invalid step name
        with self.assertRaises(ValueError):
            get_builder_step_name("InvalidStepName")

    def test_get_spec_step_type(self):
        """Test get_spec_step_type function."""
        for step_name, info in STEP_NAMES.items():
            spec_type = get_spec_step_type(step_name)
            self.assertEqual(spec_type, info["spec_type"])
        
        # Test with invalid step name
        with self.assertRaises(ValueError):
            get_spec_step_type("InvalidStepName")

    def test_get_spec_step_type_with_job_type(self):
        """Test get_spec_step_type_with_job_type function."""
        # Test without job type
        for step_name, info in STEP_NAMES.items():
            spec_type = get_spec_step_type_with_job_type(step_name)
            self.assertEqual(spec_type, info["spec_type"])
        
        # Test with job type
        step_name = "CradleDataLoading"
        spec_type_with_job = get_spec_step_type_with_job_type(step_name, "training")
        expected = f"{STEP_NAMES[step_name]['spec_type']}_Training"
        self.assertEqual(spec_type_with_job, expected)
        
        # Test with invalid step name
        with self.assertRaises(ValueError):
            get_spec_step_type_with_job_type("InvalidStepName")

    def test_get_step_name_from_spec_type(self):
        """Test get_step_name_from_spec_type function."""
        # Test with basic spec types
        for step_name, info in STEP_NAMES.items():
            spec_type = info["spec_type"]
            retrieved_name = get_step_name_from_spec_type(spec_type)
            self.assertEqual(retrieved_name, step_name)
        
        # Test with job type variants
        spec_type_with_job = "TabularPreprocessing_Training"
        retrieved_name = get_step_name_from_spec_type(spec_type_with_job)
        self.assertEqual(retrieved_name, "TabularPreprocessing")
        
        # Test with unknown spec type
        unknown_spec = get_step_name_from_spec_type("UnknownSpecType")
        self.assertEqual(unknown_spec, "UnknownSpecType")

    def test_get_all_step_names(self):
        """Test get_all_step_names function."""
        all_names = get_all_step_names()
        
        self.assertIsInstance(all_names, list)
        self.assertEqual(set(all_names), set(STEP_NAMES.keys()))

    def test_validate_step_name(self):
        """Test validate_step_name function."""
        # Test with valid step names
        for step_name in STEP_NAMES.keys():
            self.assertTrue(validate_step_name(step_name))
        
        # Test with invalid step names
        invalid_names = ["InvalidStep", "NonExistentStep", ""]
        for invalid_name in invalid_names:
            self.assertFalse(validate_step_name(invalid_name))

    def test_validate_spec_type(self):
        """Test validate_spec_type function."""
        # Test with valid spec types
        for info in STEP_NAMES.values():
            spec_type = info["spec_type"]
            self.assertTrue(validate_spec_type(spec_type))
        
        # Test with job type variants
        self.assertTrue(validate_spec_type("TabularPreprocessing_Training"))
        self.assertTrue(validate_spec_type("XGBoostTraining_Evaluation"))
        
        # Test with invalid spec types
        invalid_specs = ["InvalidSpec", "NonExistentSpec", ""]
        for invalid_spec in invalid_specs:
            self.assertFalse(validate_spec_type(invalid_spec))

    def test_get_step_description(self):
        """Test get_step_description function."""
        for step_name, info in STEP_NAMES.items():
            description = get_step_description(step_name)
            self.assertEqual(description, info["description"])
        
        # Test with invalid step name
        with self.assertRaises(ValueError):
            get_step_description("InvalidStepName")

    def test_list_all_step_info(self):
        """Test list_all_step_info function."""
        all_info = list_all_step_info()
        
        self.assertIsInstance(all_info, dict)
        self.assertEqual(all_info, STEP_NAMES)
        
        # Verify it's a copy, not the original
        self.assertIsNot(all_info, STEP_NAMES)

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
            "HyperparameterPrep"
        ]
        
        for expected_step in expected_steps:
            self.assertIn(expected_step, STEP_NAMES)
            self.assertTrue(validate_step_name(expected_step))

    def test_config_class_uniqueness(self):
        """Test that config class names are unique across steps."""
        config_classes = [info["config_class"] for info in STEP_NAMES.values()]
        self.assertEqual(len(config_classes), len(set(config_classes)))

    def test_spec_type_uniqueness(self):
        """Test that spec types are unique across steps."""
        spec_types = [info["spec_type"] for info in STEP_NAMES.values()]
        self.assertEqual(len(spec_types), len(set(spec_types)))

    def test_descriptions_are_present(self):
        """Test that all steps have non-empty descriptions."""
        for step_name, info in STEP_NAMES.items():
            description = info["description"]
            
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)
            self.assertNotEqual(description.strip(), "")

    def test_job_type_capitalization(self):
        """Test that job types are properly capitalized in spec types."""
        test_cases = [
            ("training", "Training"),
            ("evaluation", "Evaluation"),
            ("inference", "Inference"),
            ("test", "Test")
        ]
        
        step_name = "CradleDataLoading"  # Use any valid step name
        for job_type, expected_suffix in test_cases:
            spec_type = get_spec_step_type_with_job_type(step_name, job_type)
            self.assertTrue(spec_type.endswith(f"_{expected_suffix}"))

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
            with self.subTest(file_name=file_name):
                canonical_name = get_canonical_name_from_file_name(file_name)
                self.assertEqual(canonical_name, expected_canonical,
                               f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'")

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
            with self.subTest(file_name=file_name):
                canonical_name = get_canonical_name_from_file_name(file_name)
                self.assertEqual(canonical_name, expected_canonical,
                               f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'")

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
            with self.subTest(file_name=file_name):
                canonical_name = get_canonical_name_from_file_name(file_name)
                self.assertEqual(canonical_name, expected_canonical,
                               f"Failed to map '{file_name}' to '{expected_canonical}', got '{canonical_name}'")

    def test_get_canonical_name_from_file_name_invalid(self):
        """Test get_canonical_name_from_file_name with invalid inputs."""
        invalid_cases = [
            "",  # Empty string
            "completely_unknown_step",
            "invalid_step_name",
            "nonexistent_builder"
        ]
        
        for invalid_name in invalid_cases:
            with self.subTest(file_name=invalid_name):
                with self.assertRaises(ValueError):
                    get_canonical_name_from_file_name(invalid_name)

    def test_validate_file_name(self):
        """Test validate_file_name function."""
        # Test with valid file names
        valid_names = [
            "xgboost_training",
            "pytorch_training", 
            "xgboost_model_evaluation",  # The main case we fixed
            "tabular_preprocessing",
            "dummy_training"
        ]
        
        for valid_name in valid_names:
            with self.subTest(file_name=valid_name):
                self.assertTrue(validate_file_name(valid_name))
        
        # Test with invalid file names
        invalid_names = [
            "",
            "completely_unknown_step",
            "invalid_step_name"
        ]
        
        for invalid_name in invalid_names:
            with self.subTest(file_name=invalid_name):
                self.assertFalse(validate_file_name(invalid_name))

    def test_xgboost_model_evaluation_registry_integration(self):
        """Test the complete registry integration for xgboost_model_evaluation."""
        # This is the specific test for the bug fix
        file_name = "xgboost_model_evaluation"
        
        # Step 1: Convert file name to canonical name
        canonical_name = get_canonical_name_from_file_name(file_name)
        self.assertEqual(canonical_name, "XGBoostModelEval")
        
        # Step 2: Get config class name from canonical name
        config_class_name = get_config_class_name(canonical_name)
        self.assertEqual(config_class_name, "XGBoostModelEvalConfig")
        
        # Step 3: Verify the registry entry exists
        self.assertIn(canonical_name, STEP_NAMES)
        
        # Step 4: Verify all registry mappings are consistent
        step_info = STEP_NAMES[canonical_name]
        self.assertEqual(step_info["config_class"], "XGBoostModelEvalConfig")
        self.assertEqual(step_info["builder_step_name"], "XGBoostModelEvalStepBuilder")
        self.assertEqual(step_info["spec_type"], "XGBoostModelEval")
        self.assertEqual(step_info["sagemaker_step_type"], "Processing")


class TestWorkspaceContextManagement(unittest.TestCase):
    """Test case for workspace context management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing workspace context
        clear_workspace_context()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear workspace context after each test
        clear_workspace_context()
        # Clear environment variable if set
        if 'CURSUS_WORKSPACE_ID' in os.environ:
            del os.environ['CURSUS_WORKSPACE_ID']
    
    def test_workspace_context_basic_operations(self):
        """Test basic workspace context operations."""
        # Initially no context
        self.assertIsNone(get_workspace_context())
        
        # Set workspace context
        set_workspace_context("test_workspace")
        self.assertEqual(get_workspace_context(), "test_workspace")
        
        # Clear workspace context
        clear_workspace_context()
        self.assertIsNone(get_workspace_context())
    
    def test_workspace_context_environment_variable(self):
        """Test workspace context from environment variable."""
        # Set environment variable
        os.environ['CURSUS_WORKSPACE_ID'] = "env_workspace"
        
        # Should return environment variable value
        self.assertEqual(get_workspace_context(), "env_workspace")
        
        # Explicit context should override environment
        set_workspace_context("explicit_workspace")
        self.assertEqual(get_workspace_context(), "explicit_workspace")
        
        # Clear explicit context, should fall back to environment
        clear_workspace_context()
        self.assertEqual(get_workspace_context(), "env_workspace")
    
    def test_workspace_context_manager(self):
        """Test workspace context manager."""
        # Set initial context
        set_workspace_context("initial_workspace")
        
        # Use context manager
        with workspace_context("temp_workspace"):
            self.assertEqual(get_workspace_context(), "temp_workspace")
        
        # Should restore original context
        self.assertEqual(get_workspace_context(), "initial_workspace")
        
        # Test with no initial context
        clear_workspace_context()
        with workspace_context("temp_workspace"):
            self.assertEqual(get_workspace_context(), "temp_workspace")
        
        # Should be None after context manager
        self.assertIsNone(get_workspace_context())
    
    def test_workspace_aware_functions(self):
        """Test that functions work with workspace context."""
        # Test without workspace context
        step_names_core = get_step_names()
        self.assertIsInstance(step_names_core, dict)
        
        # Test with workspace context
        set_workspace_context("test_workspace")
        step_names_workspace = get_step_names()
        self.assertIsInstance(step_names_workspace, dict)
        
        # Test explicit workspace parameter
        step_names_explicit = get_step_names("explicit_workspace")
        self.assertIsInstance(step_names_explicit, dict)
        
        # Test other workspace-aware functions
        config_registry = get_config_step_registry("test_workspace")
        self.assertIsInstance(config_registry, dict)
        
        builder_names = get_builder_step_names("test_workspace")
        self.assertIsInstance(builder_names, dict)
        
        spec_types = get_spec_step_types("test_workspace")
        self.assertIsInstance(spec_types, dict)
    
    def test_workspace_aware_helper_functions(self):
        """Test that helper functions work with workspace context."""
        # Test with workspace context
        set_workspace_context("test_workspace")
        
        # These should work the same as before
        config_class = get_config_class_name("XGBoostTraining")
        self.assertEqual(config_class, "XGBoostTrainingConfig")
        
        # Test with explicit workspace parameter
        config_class_explicit = get_config_class_name("XGBoostTraining", "explicit_workspace")
        self.assertEqual(config_class_explicit, "XGBoostTrainingConfig")
        
        # Test other functions
        builder_name = get_builder_step_name("XGBoostTraining", "test_workspace")
        self.assertEqual(builder_name, "XGBoostTrainingStepBuilder")
        
        spec_type = get_spec_step_type("XGBoostTraining", "test_workspace")
        self.assertEqual(spec_type, "XGBoostTraining")
        
        # Test validation functions
        self.assertTrue(validate_step_name("XGBoostTraining", "test_workspace"))
        self.assertFalse(validate_step_name("NonExistentStep", "test_workspace"))


class TestRegistryManagerAndFallback(unittest.TestCase):
    """Test case for registry manager and fallback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        clear_workspace_context()
    
    def tearDown(self):
        """Clean up after tests."""
        clear_workspace_context()
    
    def test_registry_manager_initialization(self):
        """Test registry manager initialization."""
        manager = _get_registry_manager()
        self.assertIsNotNone(manager)
        
        # Should have required methods
        self.assertTrue(hasattr(manager, 'create_legacy_step_names_dict'))
    
    def test_fallback_manager_creation(self):
        """Test fallback manager creation."""
        fallback_manager = _create_fallback_manager()
        self.assertIsNotNone(fallback_manager)
        
        # Should have required methods
        self.assertTrue(hasattr(fallback_manager, 'create_legacy_step_names_dict'))
        self.assertTrue(hasattr(fallback_manager, 'get_step_definition'))
        self.assertTrue(hasattr(fallback_manager, 'has_step'))
        self.assertTrue(hasattr(fallback_manager, 'list_steps'))
        
        # Test fallback functionality
        step_names = fallback_manager.create_legacy_step_names_dict()
        self.assertIsInstance(step_names, dict)
        self.assertGreater(len(step_names), 0)
        
        # Test specific step lookup
        self.assertTrue(fallback_manager.has_step("XGBoostTraining"))
        self.assertFalse(fallback_manager.has_step("NonExistentStep"))
        
        step_list = fallback_manager.list_steps()
        self.assertIsInstance(step_list, list)
        self.assertIn("XGBoostTraining", step_list)
    
    def test_hybrid_registry_fallback_behavior(self):
        """Test that fallback manager provides expected functionality."""
        # Test the fallback manager directly
        fallback_manager = _create_fallback_manager()
        self.assertIsNotNone(fallback_manager)
        
        # Should provide basic functionality
        step_names = fallback_manager.create_legacy_step_names_dict()
        self.assertIsInstance(step_names, dict)
        self.assertGreater(len(step_names), 0)
        
        # Should contain expected steps
        self.assertIn("XGBoostTraining", step_names)
        self.assertIn("PyTorchTraining", step_names)
        
        # Test that it behaves like the original registry
        xgb_info = step_names["XGBoostTraining"]
        self.assertEqual(xgb_info["config_class"], "XGBoostTrainingConfig")
        self.assertEqual(xgb_info["builder_step_name"], "XGBoostTrainingStepBuilder")
    
    def test_workspace_management_functions(self):
        """Test workspace management functions."""
        # These functions should not fail even if hybrid registry is not available
        workspaces = list_available_workspaces()
        self.assertIsInstance(workspaces, list)
        
        step_count = get_workspace_step_count("test_workspace")
        self.assertIsInstance(step_count, int)
        self.assertGreaterEqual(step_count, 0)
        
        conflicts = has_workspace_conflicts()
        self.assertIsInstance(conflicts, bool)


class TestBackwardCompatibility(unittest.TestCase):
    """Test case for backward compatibility with original implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        clear_workspace_context()
    
    def tearDown(self):
        """Clean up after tests."""
        clear_workspace_context()
    
    def test_module_level_variables_exist(self):
        """Test that module-level variables still exist and work."""
        # These should exist and be dictionaries
        self.assertIsInstance(STEP_NAMES, dict)
        self.assertIsInstance(CONFIG_STEP_REGISTRY, dict)
        self.assertIsInstance(BUILDER_STEP_NAMES, dict)
        self.assertIsInstance(SPEC_STEP_TYPES, dict)
        
        # Should have content
        self.assertGreater(len(STEP_NAMES), 0)
        self.assertGreater(len(CONFIG_STEP_REGISTRY), 0)
        self.assertGreater(len(BUILDER_STEP_NAMES), 0)
        self.assertGreater(len(SPEC_STEP_TYPES), 0)
    
    def test_original_function_signatures(self):
        """Test that original function signatures still work."""
        # These should work without workspace parameters
        config_class = get_config_class_name("XGBoostTraining")
        self.assertEqual(config_class, "XGBoostTrainingConfig")
        
        builder_name = get_builder_step_name("XGBoostTraining")
        self.assertEqual(builder_name, "XGBoostTrainingStepBuilder")
        
        spec_type = get_spec_step_type("XGBoostTraining")
        self.assertEqual(spec_type, "XGBoostTraining")
        
        # Validation functions
        self.assertTrue(validate_step_name("XGBoostTraining"))
        self.assertFalse(validate_step_name("NonExistentStep"))
        
        # File name resolution
        canonical_name = get_canonical_name_from_file_name("xgboost_training")
        self.assertEqual(canonical_name, "XGBoostTraining")
    
    def test_error_messages_consistency(self):
        """Test that error messages are consistent with original implementation."""
        # Test invalid step name error
        with self.assertRaises(ValueError) as context:
            get_config_class_name("InvalidStepName")
        
        error_message = str(context.exception)
        self.assertIn("Unknown step name", error_message)
        self.assertIn("InvalidStepName", error_message)
        self.assertIn("Available steps", error_message)
    
    def test_data_structure_consistency(self):
        """Test that data structures maintain expected format."""
        # STEP_NAMES should have expected structure
        for step_name, info in STEP_NAMES.items():
            self.assertIsInstance(step_name, str)
            self.assertIsInstance(info, dict)
            
            # Required fields
            required_fields = ["config_class", "builder_step_name", "spec_type", "description"]
            for field in required_fields:
                self.assertIn(field, info)
                self.assertIsInstance(info[field], str)
        
        # Generated registries should be consistent
        for config_class, step_name in CONFIG_STEP_REGISTRY.items():
            self.assertIn(step_name, STEP_NAMES)
            self.assertEqual(STEP_NAMES[step_name]["config_class"], config_class)
        
        for step_name, builder_name in BUILDER_STEP_NAMES.items():
            self.assertIn(step_name, STEP_NAMES)
            self.assertEqual(STEP_NAMES[step_name]["builder_step_name"], builder_name)
        
        for step_name, spec_type in SPEC_STEP_TYPES.items():
            self.assertIn(step_name, STEP_NAMES)
            self.assertEqual(STEP_NAMES[step_name]["spec_type"], spec_type)


class TestEnhancedFunctionality(unittest.TestCase):
    """Test case for enhanced functionality added in Phase 5."""
    
    def setUp(self):
        """Set up test fixtures."""
        clear_workspace_context()
    
    def tearDown(self):
        """Clean up after tests."""
        clear_workspace_context()
    
    def test_enhanced_file_name_resolution(self):
        """Test enhanced file name resolution with workspace context."""
        # Test with workspace context
        set_workspace_context("test_workspace")
        
        canonical_name = get_canonical_name_from_file_name("xgboost_model_evaluation")
        self.assertEqual(canonical_name, "XGBoostModelEval")
        
        # Test with explicit workspace parameter
        canonical_name_explicit = get_canonical_name_from_file_name(
            "xgboost_model_evaluation", "explicit_workspace"
        )
        self.assertEqual(canonical_name_explicit, "XGBoostModelEval")
    
    def test_workspace_aware_error_messages(self):
        """Test that error messages include workspace context information."""
        set_workspace_context("test_workspace")
        
        with self.assertRaises(ValueError) as context:
            get_canonical_name_from_file_name("completely_unknown_step")
        
        error_message = str(context.exception)
        # Should include workspace context in error message
        self.assertIn("workspace: test_workspace", error_message)
    
    def test_environment_variable_integration(self):
        """Test integration with CURSUS_WORKSPACE_ID environment variable."""
        # Set environment variable
        os.environ['CURSUS_WORKSPACE_ID'] = "env_test_workspace"
        
        try:
            # Should use environment variable
            context = get_workspace_context()
            self.assertEqual(context, "env_test_workspace")
            
            # Functions should use environment context
            step_names = get_step_names()  # Should use env context
            self.assertIsInstance(step_names, dict)
            
        finally:
            # Clean up
            if 'CURSUS_WORKSPACE_ID' in os.environ:
                del os.environ['CURSUS_WORKSPACE_ID']


if __name__ == '__main__':
    unittest.main()
