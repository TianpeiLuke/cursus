# test/pipeline_scripts/test_mims_payload.py
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import tarfile
import json
from pathlib import Path
import logging

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the functions and main entrypoint from the script to be tested
from src.pipeline_scripts.mims_payload import (
    VariableType,
    create_model_variable_list,
    extract_hyperparameters_from_tarball,
    get_environment_content_types,
    get_environment_default_numeric_value,
    get_environment_default_text_value,
    get_environment_special_fields,
    get_field_default_value,
    generate_csv_payload,
    generate_json_payload,
    generate_sample_payloads,
    save_payloads,
    main as payload_main
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)

class TestMimsPayloadHelpers(unittest.TestCase):
    """Unit tests for the individual helper functions in the payload script."""

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.base_dir = Path(tempfile.mkdtemp())
        self.input_model_dir = self.base_dir / "input" / "model"
        self.output_dir = self.base_dir / "output"
        self.payload_sample_dir = self.output_dir / "payload_sample"
        self.payload_metadata_dir = self.output_dir / "payload_metadata"
        self.working_dir = self.base_dir / "work"
        
        # Create the directory structure
        self.input_model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.payload_sample_dir.mkdir(parents=True, exist_ok=True)
        self.payload_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_hyperparameters_tarball(self, hyperparams):
        """Helper to create a model.tar.gz with hyperparameters."""
        model_tar_path = self.input_model_dir / "model.tar.gz"
        
        # Create a temporary directory to hold files for the tarball
        temp_dir = self.base_dir / "temp_tar_contents"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write hyperparameters to a JSON file
        hyperparams_path = temp_dir / "hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f)
        
        # Create the tarball
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(hyperparams_path, arcname="hyperparameters.json")
        
        return model_tar_path

    def test_create_model_variable_list(self):
        """Test creating a model variable list from field lists."""
        full_field_list = ["id", "feature1", "feature2", "category1", "label"]
        tab_field_list = ["feature1", "feature2"]
        cat_field_list = ["category1"]
        
        var_list = create_model_variable_list(
            full_field_list, tab_field_list, cat_field_list, "label", "id"
        )
        
        # Check that the variable list has the correct structure and types
        self.assertEqual(len(var_list), 3)  # 3 fields excluding id and label
        self.assertEqual(var_list[0][0], "feature1")
        self.assertEqual(var_list[0][1], "NUMERIC")
        self.assertEqual(var_list[1][0], "feature2")
        self.assertEqual(var_list[1][1], "NUMERIC")
        self.assertEqual(var_list[2][0], "category1")
        self.assertEqual(var_list[2][1], "TEXT")

    def test_extract_hyperparameters_from_tarball(self):
        """Test extracting hyperparameters from a model.tar.gz file."""
        # Create a test hyperparameters file
        test_hyperparams = {
            "full_field_list": ["id", "feature1", "category1", "label"],
            "tab_field_list": ["feature1"],
            "cat_field_list": ["category1"],
            "label_name": "label",
            "id_name": "id",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0"
        }
        self._create_hyperparameters_tarball(test_hyperparams)
        
        # Patch the module-level constants to use our test directories
        with patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'INPUT_MODEL_DIR', str(self.input_model_dir)), \
             patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'WORKING_DIRECTORY', Path(str(self.working_dir))):
            
            # Extract the hyperparameters
            hyperparams = extract_hyperparameters_from_tarball()
            
            # Verify the extracted hyperparameters
            self.assertEqual(hyperparams["pipeline_name"], "test_pipeline")
            self.assertEqual(hyperparams["full_field_list"], ["id", "feature1", "category1", "label"])
            self.assertEqual(hyperparams["tab_field_list"], ["feature1"])

    def test_get_environment_content_types(self):
        """Test getting content types from environment variables."""
        # Test default value
        with patch.dict('os.environ', {}, clear=True):
            content_types = get_environment_content_types()
            self.assertEqual(content_types, ["application/json"])
        
        # Test custom value
        with patch.dict('os.environ', {"CONTENT_TYPES": "text/csv,application/json"}, clear=True):
            content_types = get_environment_content_types()
            self.assertEqual(content_types, ["text/csv", "application/json"])

    def test_get_environment_default_numeric_value(self):
        """Test getting default numeric value from environment variables."""
        # Test default value
        with patch.dict('os.environ', {}, clear=True):
            value = get_environment_default_numeric_value()
            self.assertEqual(value, 0.0)
        
        # Test custom value
        with patch.dict('os.environ', {"DEFAULT_NUMERIC_VALUE": "42.5"}, clear=True):
            value = get_environment_default_numeric_value()
            self.assertEqual(value, 42.5)
        
        # Test invalid value
        with patch.dict('os.environ', {"DEFAULT_NUMERIC_VALUE": "not_a_number"}, clear=True):
            value = get_environment_default_numeric_value()
            self.assertEqual(value, 0.0)  # Should fall back to default

    def test_get_environment_default_text_value(self):
        """Test getting default text value from environment variables."""
        # Test default value
        with patch.dict('os.environ', {}, clear=True):
            value = get_environment_default_text_value()
            self.assertEqual(value, "DEFAULT_TEXT")
        
        # Test custom value
        with patch.dict('os.environ', {"DEFAULT_TEXT_VALUE": "CUSTOM_TEXT"}, clear=True):
            value = get_environment_default_text_value()
            self.assertEqual(value, "CUSTOM_TEXT")

    def test_get_environment_special_fields(self):
        """Test getting special field values from environment variables."""
        # Test empty case
        with patch.dict('os.environ', {}, clear=True):
            special_fields = get_environment_special_fields()
            self.assertEqual(special_fields, {})
        
        # Test with special fields
        with patch.dict('os.environ', {
            "SPECIAL_FIELD_email": "user@example.com",
            "SPECIAL_FIELD_timestamp": "{timestamp}",
            "REGULAR_FIELD": "should_be_ignored"
        }, clear=True):
            special_fields = get_environment_special_fields()
            self.assertEqual(len(special_fields), 2)
            self.assertEqual(special_fields["email"], "user@example.com")
            self.assertEqual(special_fields["timestamp"], "{timestamp}")
            self.assertNotIn("REGULAR_FIELD", special_fields)

    def test_get_field_default_value(self):
        """Test getting default value for a field based on its type."""
        # Test numeric field
        value = get_field_default_value("feature1", "NUMERIC", 42.0, "DEFAULT_TEXT", {})
        self.assertEqual(value, "42.0")
        
        # Test text field
        value = get_field_default_value("category1", "TEXT", 42.0, "DEFAULT_TEXT", {})
        self.assertEqual(value, "DEFAULT_TEXT")
        
        # Test text field with special value
        special_fields = {"category1": "SPECIAL_VALUE"}
        value = get_field_default_value("category1", "TEXT", 42.0, "DEFAULT_TEXT", special_fields)
        self.assertEqual(value, "SPECIAL_VALUE")
        
        # Test text field with timestamp template
        special_fields = {"category1": "Date: {timestamp}"}
        value = get_field_default_value("category1", "TEXT", 42.0, "DEFAULT_TEXT", special_fields)
        self.assertTrue(value.startswith("Date: "))
        self.assertGreater(len(value), 6)  # Should have timestamp appended
        
        # Test invalid variable type
        with self.assertRaises(ValueError):
            get_field_default_value("feature1", "INVALID_TYPE", 42.0, "DEFAULT_TEXT", {})

    def test_generate_csv_payload(self):
        """Test generating CSV format payload."""
        # Test with list format
        input_vars = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"]
        ]
        csv_payload = generate_csv_payload(
            input_vars, 42.0, "DEFAULT_TEXT", {}
        )
        self.assertEqual(csv_payload, "42.0,42.0,DEFAULT_TEXT")
        
        # Test with dictionary format
        input_vars = {
            "feature1": "NUMERIC",
            "feature2": "NUMERIC",
            "category1": "TEXT"
        }
        csv_payload = generate_csv_payload(
            input_vars, 42.0, "DEFAULT_TEXT", {}
        )
        # Order might vary in dictionary, so check parts
        self.assertIn("42.0", csv_payload)
        self.assertIn("DEFAULT_TEXT", csv_payload)
        self.assertEqual(csv_payload.count(","), 2)  # Should have 2 commas for 3 values

    def test_generate_json_payload(self):
        """Test generating JSON format payload."""
        # Test with list format
        input_vars = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"]
        ]
        json_payload = generate_json_payload(
            input_vars, 42.0, "DEFAULT_TEXT", {}
        )
        payload_dict = json.loads(json_payload)
        self.assertEqual(payload_dict["feature1"], "42.0")
        self.assertEqual(payload_dict["feature2"], "42.0")
        self.assertEqual(payload_dict["category1"], "DEFAULT_TEXT")
        
        # Test with dictionary format
        input_vars = {
            "feature1": "NUMERIC",
            "feature2": "NUMERIC",
            "category1": "TEXT"
        }
        json_payload = generate_json_payload(
            input_vars, 42.0, "DEFAULT_TEXT", {}
        )
        payload_dict = json.loads(json_payload)
        self.assertEqual(payload_dict["feature1"], "42.0")
        self.assertEqual(payload_dict["feature2"], "42.0")
        self.assertEqual(payload_dict["category1"], "DEFAULT_TEXT")

    def test_generate_sample_payloads(self):
        """Test generating sample payloads for different content types."""
        input_vars = [
            ["feature1", "NUMERIC"],
            ["category1", "TEXT"]
        ]
        content_types = ["text/csv", "application/json"]
        
        payloads = generate_sample_payloads(
            input_vars, content_types, 42.0, "DEFAULT_TEXT", {}
        )
        
        self.assertEqual(len(payloads), 2)
        
        # Check CSV payload
        csv_payload = next(p for p in payloads if p["content_type"] == "text/csv")
        self.assertEqual(csv_payload["payload"], "42.0,DEFAULT_TEXT")
        
        # Check JSON payload
        json_payload = next(p for p in payloads if p["content_type"] == "application/json")
        payload_dict = json.loads(json_payload["payload"])
        self.assertEqual(payload_dict["feature1"], "42.0")
        self.assertEqual(payload_dict["category1"], "DEFAULT_TEXT")
        
        # Test with unsupported content type
        with self.assertRaises(ValueError):
            generate_sample_payloads(
                input_vars, ["unsupported/type"], 42.0, "DEFAULT_TEXT", {}
            )

    def test_save_payloads(self):
        """Test saving payloads to files."""
        input_vars = [
            ["feature1", "NUMERIC"],
            ["category1", "TEXT"]
        ]
        content_types = ["text/csv", "application/json"]
        
        file_paths = save_payloads(
            self.payload_sample_dir,
            input_vars,
            content_types,
            42.0,
            "DEFAULT_TEXT",
            {}
        )
        
        self.assertEqual(len(file_paths), 2)
        
        # Check that files were created
        csv_file = next(p for p in file_paths if "text_csv" in p)
        json_file = next(p for p in file_paths if "application_json" in p)
        
        self.assertTrue(os.path.exists(csv_file))
        self.assertTrue(os.path.exists(json_file))
        
        # Check file contents
        with open(csv_file, 'r') as f:
            csv_content = f.read()
            self.assertEqual(csv_content, "42.0,DEFAULT_TEXT")
        
        with open(json_file, 'r') as f:
            json_content = f.read()
            payload_dict = json.loads(json_content)
            self.assertEqual(payload_dict["feature1"], "42.0")
            self.assertEqual(payload_dict["category1"], "DEFAULT_TEXT")


class TestMimsPayloadMainFlow(unittest.TestCase):
    """Integration-style tests for the main() function of the payload script."""

    def setUp(self):
        """Set up a temporary directory structure mimicking the SageMaker environment."""
        self.base_dir = Path(tempfile.mkdtemp())
        
        # Define mock paths within the temporary directory
        self.input_model_dir = self.base_dir / "input" / "model"
        self.output_dir = self.base_dir / "output"
        self.payload_sample_dir = self.output_dir / "payload_sample"
        self.payload_metadata_dir = self.output_dir / "payload_metadata"
        self.working_dir = self.base_dir / "work"
        
        # Create the directory structure
        self.input_model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.base_dir)

    def _create_hyperparameters_tarball(self, hyperparams):
        """Helper to create a model.tar.gz with hyperparameters."""
        model_tar_path = self.input_model_dir / "model.tar.gz"
        
        # Create a temporary directory to hold files for the tarball
        temp_dir = self.base_dir / "temp_tar_contents"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write hyperparameters to a JSON file
        hyperparams_path = temp_dir / "hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f)
        
        # Create the tarball
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(hyperparams_path, arcname="hyperparameters.json")
        
        return model_tar_path

    def test_main_flow(self):
        """Test the main flow of the payload script."""
        # Create test hyperparameters
        test_hyperparams = {
            "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "label",
            "id_name": "id",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "model_registration_objective": "test_objective"
        }
        self._create_hyperparameters_tarball(test_hyperparams)
        
        # Set up environment variables
        env_vars = {
            "CONTENT_TYPES": "text/csv,application/json",
            "DEFAULT_NUMERIC_VALUE": "42.0",
            "DEFAULT_TEXT_VALUE": "TEST_TEXT",
            "SPECIAL_FIELD_category1": "SPECIAL_CATEGORY"
        }
        
        # Patch the constants and environment variables
        with patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'INPUT_MODEL_DIR', str(self.input_model_dir)), \
             patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'OUTPUT_DIR', Path(str(self.output_dir))), \
             patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'PAYLOAD_SAMPLE_DIR', Path(str(self.payload_sample_dir))), \
             patch.object(sys.modules['src.pipeline_scripts.mims_payload'], 'WORKING_DIRECTORY', Path(str(self.working_dir))), \
             patch.dict('os.environ', env_vars, clear=True):
            
            # Run the main function
            payload_main()
            
            # Check that output directories were created
            self.assertTrue(os.path.exists(self.payload_sample_dir))
            
            # Check that payload files were created
            csv_files = list(self.payload_sample_dir.glob("*csv*"))
            json_files = list(self.payload_sample_dir.glob("*json*"))
            self.assertGreaterEqual(len(csv_files), 1)
            self.assertGreaterEqual(len(json_files), 1)
            
            # Check that payload archive was created
            archive_path = self.output_dir / "payload.tar.gz"
            self.assertTrue(os.path.exists(archive_path))

    def test_main_flow_missing_model_tarball(self):
        """Test the main flow when the model.tar.gz file is missing."""
        # Don't create the model.tar.gz file
        
        # Patch the constants
        with patch('src.pipeline_scripts.mims_payload.INPUT_MODEL_DIR', str(self.input_model_dir)), \
             patch('src.pipeline_scripts.mims_payload.OUTPUT_DIR', str(self.output_dir)), \
             patch('src.pipeline_scripts.mims_payload.PAYLOAD_SAMPLE_DIR', str(self.payload_sample_dir)), \
             patch('src.pipeline_scripts.mims_payload.WORKING_DIRECTORY', str(self.working_dir)):
            
            # Run the main function and expect an exception
            with self.assertRaises(FileNotFoundError):
                payload_main()

    def test_main_flow_missing_hyperparameters(self):
        """Test the main flow when hyperparameters.json is missing from the tarball."""
        # Create an empty tarball without hyperparameters.json
        model_tar_path = self.input_model_dir / "model.tar.gz"
        with tarfile.open(model_tar_path, "w:gz") as tar:
            # Create an empty file to add to the tarball
            empty_file = self.base_dir / "empty.txt"
            empty_file.touch()
            tar.add(empty_file, arcname="empty.txt")
        
        # Patch the constants
        with patch('src.pipeline_scripts.mims_payload.INPUT_MODEL_DIR', str(self.input_model_dir)), \
             patch('src.pipeline_scripts.mims_payload.OUTPUT_DIR', str(self.output_dir)), \
             patch('src.pipeline_scripts.mims_payload.PAYLOAD_SAMPLE_DIR', str(self.payload_sample_dir)), \
             patch('src.pipeline_scripts.mims_payload.WORKING_DIRECTORY', str(self.working_dir)):
            
            # Run the main function and expect an exception
            with self.assertRaises(FileNotFoundError):
                payload_main()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
