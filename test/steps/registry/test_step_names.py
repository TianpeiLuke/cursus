"""Unit tests for the step names registry module."""

import unittest
from src.cursus.registry.step_names import (
    STEP_NAMES,
    CONFIG_STEP_REGISTRY,
    BUILDER_STEP_NAMES,
    SPEC_STEP_TYPES,
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
    validate_file_name
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


if __name__ == '__main__':
    unittest.main()
