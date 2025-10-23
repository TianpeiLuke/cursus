# test/pipeline_scripts/test_mims_payload.py
import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import tarfile
import json
from pathlib import Path
import logging

# Import the functions and main entrypoint from the script to be tested
from cursus.steps.scripts.payload import (
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
    main as payload_main,
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)


class TestMimsPayloadHelpers:
    """Unit tests for the individual helper functions in the payload script."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up a temporary directory for each test."""
        base_dir = Path(tempfile.mkdtemp())
        input_model_dir = base_dir / "input" / "model"
        output_dir = base_dir / "output"
        payload_sample_dir = output_dir / "payload_sample"
        payload_metadata_dir = output_dir / "payload_metadata"
        working_dir = base_dir / "work"

        # Create the directory structure
        input_model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload_sample_dir.mkdir(parents=True, exist_ok=True)
        payload_metadata_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)

        yield {
            'base_dir': base_dir,
            'input_model_dir': input_model_dir,
            'output_dir': output_dir,
            'payload_sample_dir': payload_sample_dir,
            'payload_metadata_dir': payload_metadata_dir,
            'working_dir': working_dir
        }
        
        shutil.rmtree(base_dir)

    def _create_hyperparameters_tarball(self, dirs, hyperparams):
        """Helper to create a model.tar.gz with hyperparameters."""
        base_dir = dirs['base_dir']
        input_model_dir = dirs['input_model_dir']
        
        model_tar_path = input_model_dir / "model.tar.gz"

        # Create a temporary directory to hold files for the tarball
        temp_dir = base_dir / "temp_tar_contents"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write hyperparameters to a JSON file
        hyperparams_path = temp_dir / "hyperparameters.json"
        with open(hyperparams_path, "w") as f:
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
        assert len(var_list) == 3  # 3 fields excluding id and label
        assert var_list[0][0] == "feature1"
        assert var_list[0][1] == "NUMERIC"
        assert var_list[1][0] == "feature2"
        assert var_list[1][1] == "NUMERIC"
        assert var_list[2][0] == "category1"
        assert var_list[2][1] == "TEXT"

    def test_extract_hyperparameters_from_tarball(self, setup_dirs):
        """Test extracting hyperparameters from a model.tar.gz file."""
        dirs = setup_dirs
        
        # Create a test hyperparameters file
        test_hyperparams = {
            "full_field_list": ["id", "feature1", "category1", "label"],
            "tab_field_list": ["feature1"],
            "cat_field_list": ["category1"],
            "label_name": "label",
            "id_name": "id",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
        }
        self._create_hyperparameters_tarball(dirs, test_hyperparams)

        # Extract the hyperparameters using the new function signature
        hyperparams = extract_hyperparameters_from_tarball(
            dirs['input_model_dir'], dirs['working_dir']
        )

        # Verify the extracted hyperparameters
        assert hyperparams["pipeline_name"] == "test_pipeline"
        assert hyperparams["full_field_list"] == ["id", "feature1", "category1", "label"]
        assert hyperparams["tab_field_list"] == ["feature1"]

    def test_get_environment_content_types(self):
        """Test getting content types from environment variables."""
        # Test default value
        environ_vars = {}
        content_types = get_environment_content_types(environ_vars)
        assert content_types == ["application/json"]

        # Test custom value
        environ_vars = {"CONTENT_TYPES": "text/csv,application/json"}
        content_types = get_environment_content_types(environ_vars)
        assert content_types == ["text/csv", "application/json"]

    def test_get_environment_default_numeric_value(self):
        """Test getting default numeric value from environment variables."""
        # Test default value
        environ_vars = {}
        value = get_environment_default_numeric_value(environ_vars)
        assert value == 0.0

        # Test custom value
        environ_vars = {"DEFAULT_NUMERIC_VALUE": "42.5"}
        value = get_environment_default_numeric_value(environ_vars)
        assert value == 42.5

        # Test invalid value
        environ_vars = {"DEFAULT_NUMERIC_VALUE": "not_a_number"}
        value = get_environment_default_numeric_value(environ_vars)
        assert value == 0.0  # Should fall back to default

    def test_get_environment_default_text_value(self):
        """Test getting default text value from environment variables."""
        # Test default value
        environ_vars = {}
        value = get_environment_default_text_value(environ_vars)
        assert value == "DEFAULT_TEXT"

        # Test custom value
        environ_vars = {"DEFAULT_TEXT_VALUE": "CUSTOM_TEXT"}
        value = get_environment_default_text_value(environ_vars)
        assert value == "CUSTOM_TEXT"

    def test_get_environment_special_fields(self):
        """Test getting special field values from environment variables."""
        # Test empty case
        environ_vars = {}
        special_fields = get_environment_special_fields(environ_vars)
        assert special_fields == {}

        # Test with special fields
        environ_vars = {
            "SPECIAL_FIELD_email": "user@example.com",
            "SPECIAL_FIELD_timestamp": "{timestamp}",
            "REGULAR_FIELD": "should_be_ignored",
        }
        special_fields = get_environment_special_fields(environ_vars)
        assert len(special_fields) == 2
        assert special_fields["email"] == "user@example.com"
        assert special_fields["timestamp"] == "{timestamp}"
        assert "REGULAR_FIELD" not in special_fields

    def test_get_field_default_value(self):
        """Test getting default value for a field based on its type."""
        # Test numeric field
        value = get_field_default_value("feature1", "NUMERIC", 42.0, "DEFAULT_TEXT", {})
        assert value == "42.0"

        # Test text field
        value = get_field_default_value("category1", "TEXT", 42.0, "DEFAULT_TEXT", {})
        assert value == "DEFAULT_TEXT"

        # Test text field with special value
        special_fields = {"category1": "SPECIAL_VALUE"}
        value = get_field_default_value(
            "category1", "TEXT", 42.0, "DEFAULT_TEXT", special_fields
        )
        assert value == "SPECIAL_VALUE"

        # Test text field with timestamp template
        special_fields = {"category1": "Date: {timestamp}"}
        value = get_field_default_value(
            "category1", "TEXT", 42.0, "DEFAULT_TEXT", special_fields
        )
        assert value.startswith("Date: ")
        assert len(value) > 6  # Should have timestamp appended

        # Test invalid variable type
        with pytest.raises(ValueError):
            get_field_default_value(
                "feature1", "INVALID_TYPE", 42.0, "DEFAULT_TEXT", {}
            )

    def test_generate_csv_payload(self):
        """Test generating CSV format payload."""
        # Test with list format
        input_vars = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"],
        ]
        csv_payload = generate_csv_payload(input_vars, 42.0, "DEFAULT_TEXT", {})
        assert csv_payload == "42.0,42.0,DEFAULT_TEXT"

        # Test with dictionary format
        input_vars = {"feature1": "NUMERIC", "feature2": "NUMERIC", "category1": "TEXT"}
        csv_payload = generate_csv_payload(input_vars, 42.0, "DEFAULT_TEXT", {})
        # Order might vary in dictionary, so check parts
        assert "42.0" in csv_payload
        assert "DEFAULT_TEXT" in csv_payload
        assert csv_payload.count(",") == 2  # Should have 2 commas for 3 values

    def test_generate_json_payload(self):
        """Test generating JSON format payload."""
        # Test with list format
        input_vars = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"],
        ]
        json_payload = generate_json_payload(input_vars, 42.0, "DEFAULT_TEXT", {})
        payload_dict = json.loads(json_payload)
        assert payload_dict["feature1"] == "42.0"
        assert payload_dict["feature2"] == "42.0"
        assert payload_dict["category1"] == "DEFAULT_TEXT"

        # Test with dictionary format
        input_vars = {"feature1": "NUMERIC", "feature2": "NUMERIC", "category1": "TEXT"}
        json_payload = generate_json_payload(input_vars, 42.0, "DEFAULT_TEXT", {})
        payload_dict = json.loads(json_payload)
        assert payload_dict["feature1"] == "42.0"
        assert payload_dict["feature2"] == "42.0"
        assert payload_dict["category1"] == "DEFAULT_TEXT"

    def test_generate_sample_payloads(self):
        """Test generating sample payloads for different content types."""
        input_vars = [["feature1", "NUMERIC"], ["category1", "TEXT"]]
        content_types = ["text/csv", "application/json"]

        payloads = generate_sample_payloads(
            input_vars, content_types, 42.0, "DEFAULT_TEXT", {}
        )

        assert len(payloads) == 2

        # Check CSV payload
        csv_payload = next(p for p in payloads if p["content_type"] == "text/csv")
        assert csv_payload["payload"] == "42.0,DEFAULT_TEXT"

        # Check JSON payload
        json_payload = next(
            p for p in payloads if p["content_type"] == "application/json"
        )
        payload_dict = json.loads(json_payload["payload"])
        assert payload_dict["feature1"] == "42.0"
        assert payload_dict["category1"] == "DEFAULT_TEXT"

        # Test with unsupported content type
        with pytest.raises(ValueError):
            generate_sample_payloads(
                input_vars, ["unsupported/type"], 42.0, "DEFAULT_TEXT", {}
            )

    def test_save_payloads(self, setup_dirs):
        """Test saving payloads to files."""
        dirs = setup_dirs
        
        input_vars = [["feature1", "NUMERIC"], ["category1", "TEXT"]]
        content_types = ["text/csv", "application/json"]

        file_paths = save_payloads(
            dirs['payload_sample_dir'], input_vars, content_types, 42.0, "DEFAULT_TEXT", {}
        )

        assert len(file_paths) == 2

        # Check that files were created
        csv_file = next(p for p in file_paths if "text_csv" in p)
        json_file = next(p for p in file_paths if "application_json" in p)

        assert os.path.exists(csv_file)
        assert os.path.exists(json_file)

        # Check file contents
        with open(csv_file, "r") as f:
            csv_content = f.read()
            assert csv_content == "42.0,DEFAULT_TEXT"

        with open(json_file, "r") as f:
            json_content = f.read()
            payload_dict = json.loads(json_content)
            assert payload_dict["feature1"] == "42.0"
            assert payload_dict["category1"] == "DEFAULT_TEXT"


class TestMimsPayloadMainFlow:
    """Integration-style tests for the main() function of the payload script."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up a temporary directory structure mimicking the SageMaker environment."""
        base_dir = Path(tempfile.mkdtemp())

        # Define mock paths within the temporary directory
        input_model_dir = base_dir / "input" / "model"
        output_dir = base_dir / "output"
        payload_sample_dir = output_dir / "payload_sample"
        payload_metadata_dir = output_dir / "payload_metadata"
        working_dir = base_dir / "work"

        # Create the directory structure
        input_model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        yield {
            'base_dir': base_dir,
            'input_model_dir': input_model_dir,
            'output_dir': output_dir,
            'payload_sample_dir': payload_sample_dir,
            'payload_metadata_dir': payload_metadata_dir,
            'working_dir': working_dir
        }
        
        shutil.rmtree(base_dir)

    def _create_hyperparameters_tarball(self, dirs, hyperparams):
        """Helper to create a model.tar.gz with hyperparameters."""
        base_dir = dirs['base_dir']
        input_model_dir = dirs['input_model_dir']
        
        model_tar_path = input_model_dir / "model.tar.gz"

        # Create a temporary directory to hold files for the tarball
        temp_dir = base_dir / "temp_tar_contents"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write hyperparameters to a JSON file
        hyperparams_path = temp_dir / "hyperparameters.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f)

        # Create the tarball
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(hyperparams_path, arcname="hyperparameters.json")

        return model_tar_path

    def test_main_flow(self, setup_dirs):
        """Test the main flow of the payload script."""
        dirs = setup_dirs
        
        # Create test hyperparameters
        test_hyperparams = {
            "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "label",
            "id_name": "id",
            "pipeline_name": "test_pipeline",
            "pipeline_version": "1.0.0",
            "model_registration_objective": "test_objective",
        }
        self._create_hyperparameters_tarball(dirs, test_hyperparams)

        # Set up input and output paths
        input_paths = {"model_input": str(dirs['input_model_dir'])}
        output_paths = {"output_dir": str(dirs['output_dir'])}
        environ_vars = {
            "CONTENT_TYPES": "text/csv,application/json",
            "DEFAULT_NUMERIC_VALUE": "42.0",
            "DEFAULT_TEXT_VALUE": "TEST_TEXT",
            "SPECIAL_FIELD_category1": "SPECIAL_CATEGORY",
            "WORKING_DIRECTORY": str(dirs['working_dir']),
        }

        # Run the main function
        result = payload_main(input_paths, output_paths, environ_vars)

        # The script creates payload_sample_dir in the working directory, not output directory
        actual_payload_sample_dir = dirs['working_dir'] / "payload_sample"

        # Check that output directories were created
        assert os.path.exists(actual_payload_sample_dir)

        # Check that payload files were created
        csv_files = list(actual_payload_sample_dir.glob("*csv*"))
        json_files = list(actual_payload_sample_dir.glob("*json*"))
        assert len(csv_files) >= 1
        assert len(json_files) >= 1

        # Check that payload archive was created
        archive_path = dirs['output_dir'] / "payload.tar.gz"
        assert os.path.exists(archive_path)
        assert result == str(archive_path)

    def test_main_flow_missing_model_tarball(self, setup_dirs):
        """Test the main flow when the model.tar.gz file is missing."""
        dirs = setup_dirs
        
        # Don't create the model.tar.gz file

        # Set up input and output paths
        input_paths = {"model_input": str(dirs['input_model_dir'])}
        output_paths = {"output_dir": str(dirs['output_dir'])}
        environ_vars = {"WORKING_DIRECTORY": str(dirs['working_dir'])}

        # Run the main function and expect an exception
        with pytest.raises(FileNotFoundError):
            payload_main(input_paths, output_paths, environ_vars)

    def test_main_flow_missing_hyperparameters(self, setup_dirs):
        """Test the main flow when hyperparameters.json is missing from the tarball."""
        dirs = setup_dirs
        
        # Create an empty tarball without hyperparameters.json
        model_tar_path = dirs['input_model_dir'] / "model.tar.gz"
        with tarfile.open(model_tar_path, "w:gz") as tar:
            # Create an empty file to add to the tarball
            empty_file = dirs['base_dir'] / "empty.txt"
            empty_file.touch()
            tar.add(empty_file, arcname="empty.txt")

        # Set up input and output paths
        input_paths = {"model_input": str(dirs['input_model_dir'])}
        output_paths = {"output_dir": str(dirs['output_dir'])}
        environ_vars = {"WORKING_DIRECTORY": str(dirs['working_dir'])}

        # Run the main function and expect an exception
        with pytest.raises(FileNotFoundError):
            payload_main(input_paths, output_paths, environ_vars)
