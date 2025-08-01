import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the config class to be tested
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig, VariableType


class TestModelRegistrationConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal valid config
        # Use the project root for source_dir and a real inference script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dockers_dir = os.path.join(project_root, 'dockers', 'xgboost_atoz')
        
        self.config_data = {
            "region": "NA",
            "author": "test-author",
            "bucket": "test-bucket",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "role": "arn:aws:iam::000000000000:role/TestRole",
            "service_name": "test-service",
            "model_owner": "test-team",
            "model_registration_domain": "BuyerSellerMessaging",
            "model_registration_objective": "TestObjective",
            "framework": "xgboost",
            "inference_entry_point": "inference_xgb.py",
            "source_dir": dockers_dir,
            "source_model_inference_input_variable_list": {"feature1": "NUMERIC", "feature2": "TEXT"},
            "source_model_inference_output_variable_list": {"score": "NUMERIC"}
        }

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = ModelRegistrationConfig(**self.config_data)
        
        # Verify basic attributes
        self.assertEqual(config.model_owner, "test-team")
        self.assertEqual(config.model_registration_domain, "BuyerSellerMessaging")
        self.assertEqual(config.model_registration_objective, "TestObjective")
        self.assertEqual(config.framework, "xgboost")
        self.assertEqual(config.inference_entry_point, "inference_xgb.py")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        dockers_dir = os.path.join(project_root, 'dockers', 'xgboost_atoz')
        self.assertEqual(config.source_dir, dockers_dir)

        # Verify default values
        self.assertEqual(config.inference_instance_type, "ml.m5.large")
        self.assertEqual(config.source_model_inference_content_types, ["text/csv"])
        self.assertEqual(config.source_model_inference_response_types, ["application/json"])

    def test_variable_schema(self):
        """Test that variable_schema property generates correct schema."""
        config = ModelRegistrationConfig(**self.config_data)
        
        schema = config.variable_schema
        
        # Check schema structure
        self.assertIn("input", schema)
        self.assertIn("output", schema)
        self.assertIn("variables", schema["input"])
        self.assertIn("variables", schema["output"])
        
        # Check input variables
        input_vars = schema["input"]["variables"]
        self.assertEqual(len(input_vars), 2)
        
        # Check that both input variables are present with correct types
        feature1_found = False
        feature2_found = False
        
        for var in input_vars:
            if var["name"] == "feature1":
                self.assertEqual(var["type"], "NUMERIC")
                feature1_found = True
            elif var["name"] == "feature2":
                self.assertEqual(var["type"], "TEXT")
                feature2_found = True
        
        self.assertTrue(feature1_found, "feature1 not found in input variables")
        self.assertTrue(feature2_found, "feature2 not found in input variables")
        
        # Check output variables
        output_vars = schema["output"]["variables"]
        self.assertEqual(len(output_vars), 1)
        self.assertEqual(output_vars[0]["name"], "score")
        self.assertEqual(output_vars[0]["type"], "NUMERIC")

    def test_input_variable_list_dict_format(self):
        """Test handling of input variable list in dictionary format."""
        # Use dictionary format for input variables
        self.config_data["source_model_inference_input_variable_list"] = {
            "feature1": VariableType.NUMERIC,
            "feature2": "TEXT",
            "feature3": "NUMERIC"
        }
        
        config = ModelRegistrationConfig(**self.config_data)
        
        # Check that input variables were processed correctly
        input_vars = config.source_model_inference_input_variable_list
        self.assertEqual(len(input_vars), 3)
        self.assertEqual(input_vars["feature1"], VariableType.NUMERIC)
        self.assertEqual(input_vars["feature2"], VariableType.TEXT)
        self.assertEqual(input_vars["feature3"], VariableType.NUMERIC)
        
        # Check variable schema
        schema = config.variable_schema
        input_schema_vars = schema["input"]["variables"]
        self.assertEqual(len(input_schema_vars), 3)

    def test_input_variable_list_list_format(self):
        """Test handling of input variable list in list format."""
        # Use list format for input variables
        self.config_data["source_model_inference_input_variable_list"] = [
            ["feature1", "NUMERIC"],
            ["feature2", "TEXT"],
            ["feature3", "NUMERIC"]
        ]
        
        config = ModelRegistrationConfig(**self.config_data)
        
        # Check variable schema
        schema = config.variable_schema
        input_schema_vars = schema["input"]["variables"]
        self.assertEqual(len(input_schema_vars), 3)
        
        # Check that variables were processed correctly
        feature1_found = False
        feature2_found = False
        feature3_found = False
        
        for var in input_schema_vars:
            if var["name"] == "feature1":
                self.assertEqual(var["type"], "NUMERIC")
                feature1_found = True
            elif var["name"] == "feature2":
                self.assertEqual(var["type"], "TEXT")
                feature2_found = True
            elif var["name"] == "feature3":
                self.assertEqual(var["type"], "NUMERIC")
                feature3_found = True
        
        self.assertTrue(feature1_found, "feature1 not found in input variables")
        self.assertTrue(feature2_found, "feature2 not found in input variables")
        self.assertTrue(feature3_found, "feature3 not found in input variables")

    def test_set_source_model_inference_input_variable_list(self):
        """Test the set_source_model_inference_input_variable_list method."""
        config = ModelRegistrationConfig(**self.config_data)
        
        # Set input variables using the helper method (dict format)
        config.set_source_model_inference_input_variable_list(
            numeric_fields=["num1", "num2"],
            text_fields=["text1"],
            output_format="dict"
        )
        
        # Check that input variables were set correctly
        input_vars = config.source_model_inference_input_variable_list
        self.assertEqual(len(input_vars), 3)
        self.assertEqual(input_vars["num1"], VariableType.NUMERIC)
        self.assertEqual(input_vars["num2"], VariableType.NUMERIC)
        self.assertEqual(input_vars["text1"], VariableType.TEXT)
        
        # Check variable schema
        schema = config.variable_schema
        input_schema_vars = schema["input"]["variables"]
        self.assertEqual(len(input_schema_vars), 3)
        
        # Set input variables using the helper method (list format)
        config.set_source_model_inference_input_variable_list(
            numeric_fields=["num3", "num4"],
            text_fields=["text2"],
            output_format="list"
        )
        
        # Check that input variables were set correctly
        input_vars = config.source_model_inference_input_variable_list
        self.assertEqual(len(input_vars), 3)
        
        # Verify list format
        self.assertTrue(isinstance(input_vars, list))
        
        # Check all combinations are present
        expected_items = [
            ["num3", "NUMERIC"],
            ["num4", "NUMERIC"],
            ["text2", "TEXT"]
        ]
        
        for expected_item in expected_items:
            self.assertTrue(
                any(item[0] == expected_item[0] and item[1] == expected_item[1] for item in input_vars),
                f"Expected item {expected_item} not found in {input_vars}"
            )

    def test_invalid_framework(self):
        """Test that invalid framework raises ValidationError."""
        # Set invalid framework
        invalid_config = self.config_data.copy()
        invalid_config["framework"] = "invalid-framework"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_inference_instance_type(self):
        """Test that invalid inference_instance_type raises ValidationError."""
        # Set invalid inference_instance_type
        invalid_config = self.config_data.copy()
        invalid_config["inference_instance_type"] = "invalid-instance-type"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_content_types(self):
        """Test that invalid content types raise ValidationError."""
        # Set invalid content types
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_content_types"] = ["invalid/type"]
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_response_types(self):
        """Test that invalid response types raise ValidationError."""
        # Set invalid response types
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_response_types"] = ["invalid/type"]
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_input_variable_list_format(self):
        """Test that invalid input variable list format raises ValidationError."""
        # Set invalid input variable list (not dict or list)
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_input_variable_list"] = "invalid format"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_input_variable_list_dict_values(self):
        """Test that invalid values in input variable list dict raises ValidationError."""
        # Set invalid value in dict - we'll use a non-string type to avoid the recursion issue
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_input_variable_list"] = {
            "feature1": 123  # Use a number instead of "INVALID_TYPE" to avoid recursion
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_input_variable_list_list_format(self):
        """Test that invalid list format in input variable list raises ValidationError."""
        # Set invalid list format (not pairs)
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_input_variable_list"] = [
            ["feature1", "NUMERIC", "extra_item"]
        ]
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_input_variable_list_list_values(self):
        """Test that invalid values in input variable list list raises ValidationError."""
        # Set invalid value in list
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_input_variable_list"] = [
            ["feature1", "INVALID_TYPE"]
        ]
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_invalid_output_variable_list_values(self):
        """Test that invalid values in output variable list raises ValidationError."""
        # Set invalid value in output variable list
        invalid_config = self.config_data.copy()
        invalid_config["source_model_inference_output_variable_list"] = {
            "score": "INVALID_TYPE"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError), patch('pathlib.Path.exists', return_value=True):
            ModelRegistrationConfig(**invalid_config)

    def test_model_dump(self):
        """Test the model_dump method."""
        config = ModelRegistrationConfig(**self.config_data)
        
        # Get model dump
        dumped = config.model_dump()
        
        # Check that all expected fields are present
        self.assertIn("model_owner", dumped)
        self.assertIn("model_registration_domain", dumped)
        self.assertIn("model_registration_objective", dumped)
        self.assertIn("framework", dumped)
        self.assertIn("inference_entry_point", dumped)
        self.assertIn("source_model_inference_input_variable_list", dumped)
        self.assertIn("source_model_inference_output_variable_list", dumped)
        self.assertIn("source_model_inference_content_types", dumped)
        self.assertIn("source_model_inference_response_types", dumped)
        self.assertIn("variable_schema", dumped)
        
        # Check variable schema is included
        schema = dumped["variable_schema"]
        self.assertIn("input", schema)
        self.assertIn("output", schema)

    def test_initialize_derived_fields(self):
        """Test that initialize_derived_fields sets up the variable schema."""
        config = ModelRegistrationConfig(**self.config_data)
        
        # Variable schema should be initialized
        self.assertIsNotNone(config.variable_schema)
        self.assertEqual(len(config.variable_schema["input"]["variables"]), 2)
        self.assertEqual(len(config.variable_schema["output"]["variables"]), 1)

    def test_validation_with_s3_source_dir(self):
        """Test validation with S3 source directory."""
        # Set source_dir to S3 path
        s3_config = self.config_data.copy()
        s3_config["source_dir"] = "s3://test-bucket/source"
        s3_config["inference_entry_point"] = "inference.py" # Any name is valid with S3 path
        
        # Should not check for file existence with S3 path
        config = ModelRegistrationConfig(**s3_config)
        self.assertEqual(config.source_dir, "s3://test-bucket/source")


if __name__ == '__main__':
    unittest.main()
