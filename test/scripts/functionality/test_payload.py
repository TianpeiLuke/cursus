# tests/scripts/functionality/test_payload.py
"""
Comprehensive tests for the payload generation script.

Tests cover:
- Multi-modal model support (tabular, bimodal, trimodal)
- Custom payload loading (JSON, CSV, Parquet)
- Intelligent text generation with 3-tier priority
- Field defaults configuration (FIELD_DEFAULTS + SPECIAL_FIELD_*)
- Validation with feature_columns.txt
- CSV field ordering for XGBoost/LightGBM

SOURCE CODE VERIFICATION (2025-11-29):
- generate_csv_payload(input_vars, default_numeric_value, default_text_value, hyperparams=None, field_defaults=None, model_dir=None)
- generate_json_payload(input_vars, default_numeric_value, default_text_value, hyperparams=None, field_defaults=None)
- generate_text_sample(field_name, field_defaults, default_text_value="Sample text for inference testing")
- pandas imported conditionally inside functions, not at module level
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import os
import tempfile
import shutil
import tarfile
import json
from pathlib import Path
import logging

# Import functions from the payload script
from cursus.steps.scripts.payload import (
    VariableType,
    detect_model_type,
    get_field_defaults,
    generate_text_sample,
    load_custom_payload,
    get_required_fields_from_model,
    validate_payload_completeness,
    log_payload_field_mapping,
    create_model_variable_list,
    extract_hyperparameters_from_tarball,
    get_environment_content_types,
    get_environment_default_numeric_value,
    get_environment_default_text_value,
    generate_csv_payload,
    generate_json_payload,
    generate_sample_payloads,
    save_payloads,
    create_payload_archive,
    main as payload_main,
)

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    original_level = logging.root.level
    logging.disable(logging.CRITICAL)
    yield
    logging.root.level = original_level


class TestMultiModalSupport:
    """Tests for multi-modal model detection and handling."""

    def test_detect_tabular_model(self):
        """Test detection of tabular-only models."""
        hyperparams = {
            "full_field_list": ["id", "feature1", "feature2", "label"],
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": [],
        }
        model_type = detect_model_type(hyperparams)
        assert model_type == "tabular"

    def test_detect_bimodal_model(self):
        """Test detection of bimodal (text + tabular) models."""
        hyperparams = {
            "text_name": "product_description",
            "tab_field_list": ["feature1", "feature2"],
        }
        model_type = detect_model_type(hyperparams)
        assert model_type == "bimodal"

    def test_detect_trimodal_model_by_fields(self):
        """Test detection of trimodal models by field presence."""
        hyperparams = {
            "primary_text_name": "chat_history",
            "secondary_text_name": "shiptrack_events",
            "tab_field_list": ["feature1"],
        }
        model_type = detect_model_type(hyperparams)
        assert model_type == "trimodal"

    def test_detect_trimodal_model_by_class(self):
        """Test detection of trimodal models by model_class."""
        hyperparams = {
            "model_class": "TrimodalBERT",
            "tab_field_list": ["feature1"],
        }
        model_type = detect_model_type(hyperparams)
        assert model_type == "trimodal"


class TestFieldDefaults:
    """Tests for field defaults configuration."""

    def test_get_field_defaults_empty(self):
        """Test with no field defaults configured."""
        environ_vars = {}
        defaults = get_field_defaults(environ_vars)
        assert defaults == {}

    def test_get_field_defaults_from_json(self):
        """Test loading field defaults from FIELD_DEFAULTS JSON."""
        environ_vars = {
            "FIELD_DEFAULTS": '{"field1": "value1", "field2": "value2"}'
        }
        defaults = get_field_defaults(environ_vars)
        assert defaults == {"field1": "value1", "field2": "value2"}

    def test_get_field_defaults_from_special_fields(self):
        """Test loading field defaults from SPECIAL_FIELD_* variables."""
        environ_vars = {
            "SPECIAL_FIELD_timestamp": "{timestamp}",
            "SPECIAL_FIELD_user_id": "test_user",
        }
        defaults = get_field_defaults(environ_vars)
        assert defaults == {"timestamp": "{timestamp}", "user_id": "test_user"}

    def test_get_field_defaults_priority(self):
        """Test that SPECIAL_FIELD_* overrides FIELD_DEFAULTS."""
        environ_vars = {
            "FIELD_DEFAULTS": '{"timestamp": "old_value", "field2": "value2"}',
            "SPECIAL_FIELD_timestamp": "{timestamp}",  # Should override
        }
        defaults = get_field_defaults(environ_vars)
        assert defaults["timestamp"] == "{timestamp}"  # Override
        assert defaults["field2"] == "value2"  # From JSON

    def test_get_field_defaults_invalid_json(self):
        """Test handling of invalid JSON in FIELD_DEFAULTS."""
        environ_vars = {"FIELD_DEFAULTS": "not valid json"}
        defaults = get_field_defaults(environ_vars)
        assert defaults == {}  # Should return empty dict


class TestIntelligentTextGeneration:
    """Tests for intelligent text field generation with 3-tier priority."""

    def test_generate_text_user_provided_exact_match(self):
        """Test Priority 1: User-provided value (exact match)."""
        field_defaults = {"chat_history": "Custom chat text"}
        text = generate_text_sample("chat_history", field_defaults, "fallback")
        assert text == "Custom chat text"

    def test_generate_text_user_provided_case_insensitive(self):
        """Test Priority 1: User-provided value (case-insensitive match)."""
        field_defaults = {"ChAt_HiStOrY": "Custom chat text"}
        text = generate_text_sample("chat_history", field_defaults, "fallback")
        assert text == "Custom chat text"

    def test_generate_text_timestamp_template(self):
        """Test template expansion for {timestamp}."""
        field_defaults = {"timestamp": "Time: {timestamp}"}
        text = generate_text_sample("timestamp", field_defaults, "fallback")
        assert text.startswith("Time: ")
        assert len(text) > 6  # Should have timestamp appended

    def test_generate_text_pattern_chat(self):
        """Test Priority 2: Pattern-based default for chat fields."""
        text = generate_text_sample("chat_history", {}, "fallback")
        assert "Hello" in text or "help" in text

    def test_generate_text_pattern_tracking(self):
        """Test Priority 2: Pattern-based default for tracking fields."""
        text = generate_text_sample("shiptrack_events", {}, "fallback")
        assert "shipped" in text.lower() or "|" in text

    def test_generate_text_pattern_description(self):
        """Test Priority 2: Pattern-based default for description fields."""
        text = generate_text_sample("product_description", {}, "fallback")
        assert "description" in text.lower()

    def test_generate_text_generic_fallback(self):
        """Test Priority 3: Generic fallback for unknown fields."""
        text = generate_text_sample("unknown_field", {}, "FALLBACK_TEXT")
        assert text == "FALLBACK_TEXT"


class TestCustomPayloadLoading:
    """Tests for custom payload loading functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_load_custom_payload_json_file(self, temp_dir):
        """Test loading custom payload from JSON file."""
        json_file = temp_dir / "sample.json"
        payload_data = {"field1": "value1", "field2": "value2"}
        with open(json_file, "w") as f:
            json.dump(payload_data, f)

        payload = load_custom_payload(json_file, "application/json")
        assert payload == payload_data

    def test_load_custom_payload_json_directory(self, temp_dir):
        """Test loading custom payload from directory with JSON file."""
        json_file = temp_dir / "sample.json"
        payload_data = {"field1": "value1", "field2": "value2"}
        with open(json_file, "w") as f:
            json.dump(payload_data, f)

        payload = load_custom_payload(temp_dir, "application/json")
        assert payload == payload_data

    def test_load_custom_payload_nonexistent(self, temp_dir):
        """Test loading from nonexistent path returns None."""
        nonexistent = temp_dir / "nonexistent.json"
        payload = load_custom_payload(nonexistent, "application/json")
        assert payload is None

    def test_load_custom_payload_empty_directory(self, temp_dir):
        """Test loading from empty directory returns None."""
        payload = load_custom_payload(temp_dir, "application/json")
        assert payload is None


class TestValidation:
    """Tests for payload validation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_validate_tabular_payload(self, temp_dir):
        """Test validation of tabular model payload."""
        hyperparams = {
            "id_name": "id",
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
        }
        var_type_list = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"],
        ]
        payload = {
            "id": "123",
            "feature1": "1.0",
            "feature2": "2.0",
            "category1": "cat",
        }

        is_valid, missing = validate_payload_completeness(
            payload, hyperparams, var_type_list, temp_dir
        )
        assert is_valid
        assert missing == []

    def test_validate_bimodal_payload(self, temp_dir):
        """Test validation of bimodal model payload."""
        hyperparams = {
            "id_name": "id",
            "text_name": "description",
            "tab_field_list": ["feature1"],
        }
        var_type_list = [["feature1", "NUMERIC"]]
        payload = {"id": "123", "description": "text", "feature1": "1.0"}

        is_valid, missing = validate_payload_completeness(
            payload, hyperparams, var_type_list, temp_dir
        )
        assert is_valid
        assert missing == []

    def test_validate_trimodal_payload(self, temp_dir):
        """Test validation of trimodal model payload."""
        hyperparams = {
            "id_name": "id",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack",
            "tab_field_list": ["feature1"],
        }
        var_type_list = [["feature1", "NUMERIC"]]
        payload = {
            "id": "123",
            "chat": "text1",
            "shiptrack": "text2",
            "feature1": "1.0",
        }

        is_valid, missing = validate_payload_completeness(
            payload, hyperparams, var_type_list, temp_dir
        )
        assert is_valid
        assert missing == []

    def test_validate_missing_fields(self, temp_dir):
        """Test validation catches missing required fields."""
        hyperparams = {"id_name": "id", "tab_field_list": ["feature1", "feature2"]}
        var_type_list = [["feature1", "NUMERIC"], ["feature2", "NUMERIC"]]
        payload = {"id": "123", "feature1": "1.0"}  # Missing feature2

        is_valid, missing = validate_payload_completeness(
            payload, hyperparams, var_type_list, temp_dir
        )
        assert not is_valid
        assert "feature2" in missing


class TestFeatureColumnsIntegration:
    """Tests for feature_columns.txt integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_get_required_fields_with_feature_columns(self, temp_dir):
        """Test field extraction using feature_columns.txt."""
        # Create feature_columns.txt
        feature_file = temp_dir / "feature_columns.txt"
        feature_file.write_text("0,feature1\n1,feature2\n2,category1\n")

        hyperparams = {}
        var_type_list = [
            ["feature1", "NUMERIC"],
            ["feature2", "NUMERIC"],
            ["category1", "TEXT"],
        ]

        required = get_required_fields_from_model(
            temp_dir, hyperparams, var_type_list
        )

        assert required["source"] == "feature_columns.txt"
        assert required["tabular_fields"] == ["feature1", "feature2", "category1"]
        assert required["field_order"] == ["feature1", "feature2", "category1"]

    def test_get_required_fields_from_hyperparams(self, temp_dir):
        """Test field extraction fallback to hyperparameters."""
        hyperparams = {
            "id_name": "id",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack",
        }
        var_type_list = [["feature1", "NUMERIC"]]

        required = get_required_fields_from_model(
            temp_dir, hyperparams, var_type_list
        )

        assert required["source"] == "hyperparameters.json"
        assert required["model_type"] == "trimodal"
        assert required["id_field"] == "id"
        assert "chat" in required["text_fields"].values()
        assert "shiptrack" in required["text_fields"].values()
        # Field order should be: id -> text fields -> tabular fields
        assert required["field_order"][0] == "id"
        assert "feature1" in required["field_order"]


class TestPayloadGeneration:
    """Tests for payload generation with all model types."""

    def test_generate_json_tabular(self):
        """Test JSON generation for tabular model."""
        input_vars = [["feature1", "NUMERIC"], ["category1", "TEXT"]]
        json_payload = generate_json_payload(input_vars, 1.0, "TEXT")

        payload_dict = json.loads(json_payload)
        assert "feature1" in payload_dict
        assert "category1" in payload_dict

    def test_generate_json_bimodal(self):
        """Test JSON generation for bimodal model."""
        hyperparams = {"id_name": "id", "text_name": "description"}
        input_vars = [["feature1", "NUMERIC"]]
        field_defaults = {}

        json_payload = generate_json_payload(
            input_vars, 1.0, "TEXT", hyperparams, field_defaults
        )

        payload_dict = json.loads(json_payload)
        assert "id" in payload_dict
        assert "description" in payload_dict
        assert "feature1" in payload_dict

    def test_generate_json_trimodal(self):
        """Test JSON generation for trimodal model."""
        hyperparams = {
            "id_name": "id",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack",
        }
        input_vars = [["feature1", "NUMERIC"]]
        field_defaults = {}

        json_payload = generate_json_payload(
            input_vars, 1.0, "TEXT", hyperparams, field_defaults
        )

        payload_dict = json.loads(json_payload)
        assert "id" in payload_dict
        assert "chat" in payload_dict
        assert "shiptrack" in payload_dict
        assert "feature1" in payload_dict

    def test_generate_csv_tabular(self):
        """Test CSV generation for tabular model."""
        input_vars = [["feature1", "NUMERIC"], ["category1", "TEXT"]]

        csv_payload = generate_csv_payload(input_vars, 1.0, "TEXT")

        # Should have 2 values
        values = csv_payload.split(",")
        assert len(values) == 2

    def test_generate_csv_multimodal(self):
        """Test CSV generation for multi-modal models includes all fields in order.
        
        Note: Current CSV implementation doesn't escape commas in text fields,
        so text fields with commas will be split into multiple values.
        This test verifies the field order structure.
        """
        hyperparams = {
            "id_name": "id",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack",
        }
        input_vars = [["feature1", "NUMERIC"]]
        # Provide custom text without commas to get predictable CSV
        field_defaults = {
            "chat": "Hello I need help with my order",
            "shiptrack": "Package shipped and delivered"
        }

        csv_payload = generate_csv_payload(
            input_vars, 1.0, "TEXT", hyperparams, field_defaults
        )

        # Should have: id, chat, shiptrack, feature1 = 4 fields
        values = csv_payload.split(",")
        assert len(values) == 4
        # Verify it starts with an ID (TEST_ID_...)
        assert values[0].startswith("TEST_ID_")
        # Verify the custom text values are present
        assert "Hello I need help with my order" in values[1]
        assert "Package shipped and delivered" in values[2]
        # Verify numeric value is last
        assert values[3] == "1.0"

    def test_generate_csv_empty_input(self):
        """Test CSV generation with empty input list."""
        input_vars = []
        csv_payload = generate_csv_payload(input_vars, 1.0, "TEXT")

        # Should return empty string for no fields
        assert csv_payload == ""

    def test_generate_json_empty_input(self):
        """Test JSON generation with empty input list."""
        input_vars = []
        json_payload = generate_json_payload(input_vars, 1.0, "TEXT")

        payload_dict = json.loads(json_payload)
        # Should return empty dict for no fields
        assert payload_dict == {}


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_model_variable_list(self):
        """Test creating model variable list."""
        full_field_list = ["id", "feature1", "feature2", "category1", "label"]
        tab_field_list = ["feature1", "feature2"]
        cat_field_list = ["category1"]

        var_list = create_model_variable_list(
            full_field_list, tab_field_list, cat_field_list, "label", "id"
        )

        assert len(var_list) == 3
        assert ["feature1", "NUMERIC"] in var_list
        assert ["feature2", "NUMERIC"] in var_list
        assert ["category1", "TEXT"] in var_list

    def test_get_environment_content_types(self):
        """Test getting content types from environment."""
        environ_vars = {"CONTENT_TYPES": "application/json,text/csv"}
        content_types = get_environment_content_types(environ_vars)
        assert content_types == ["application/json", "text/csv"]

    def test_get_environment_default_numeric_value(self):
        """Test getting default numeric value."""
        environ_vars = {"DEFAULT_NUMERIC_VALUE": "42.0"}
        value = get_environment_default_numeric_value(environ_vars)
        assert value == 42.0

    def test_get_environment_default_text_value(self):
        """Test getting default text value."""
        environ_vars = {"DEFAULT_TEXT_VALUE": "CUSTOM"}
        value = get_environment_default_text_value(environ_vars)
        assert value == "CUSTOM"


class TestMainIntegration:
    """Integration tests for main() function with new features."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up temporary directory structure."""
        base_dir = Path(tempfile.mkdtemp())
        input_model_dir = base_dir / "input" / "model"
        output_dir = base_dir / "output"
        working_dir = base_dir / "work"

        input_model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)

        yield {
            "base_dir": base_dir,
            "input_model_dir": input_model_dir,
            "output_dir": output_dir,
            "working_dir": working_dir,
        }

        if base_dir.exists():
            shutil.rmtree(base_dir)

    def _create_model_with_hyperparams(self, dirs, hyperparams, include_feature_columns=False):
        """Helper to create model artifacts."""
        input_model_dir = dirs["input_model_dir"]
        model_tar_path = input_model_dir / "model.tar.gz"

        temp_dir = dirs["base_dir"] / "temp_tar_contents"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create hyperparameters.json
        hyperparams_path = temp_dir / "hyperparameters.json"
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f)

        # Create tar with hyperparameters
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(hyperparams_path, arcname="hyperparameters.json")
            
            # Optionally add feature_columns.txt
            if include_feature_columns:
                feature_file = temp_dir / "feature_columns.txt"
                feature_file.write_text("0,feature1\n1,feature2\n2,category1\n")
                tar.add(feature_file, arcname="feature_columns.txt")

    def test_main_tabular_model(self, setup_dirs):
        """Test main flow with tabular model."""
        dirs = setup_dirs
        hyperparams = {
            "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "label",
            "id_name": "id",
        }
        self._create_model_with_hyperparams(dirs, hyperparams)

        input_paths = {"model_input": str(dirs["input_model_dir"])}
        output_paths = {"output_dir": str(dirs["output_dir"])}
        environ_vars = {
            "CONTENT_TYPES": "application/json,text/csv",
            "WORKING_DIRECTORY": str(dirs["working_dir"]),
        }

        result = payload_main(input_paths, output_paths, environ_vars)

        assert Path(result).exists()
        assert result.endswith("payload.tar.gz")

    def test_main_trimodal_model(self, setup_dirs):
        """Test main flow with trimodal model."""
        dirs = setup_dirs
        hyperparams = {
            "full_field_list": ["id", "chat", "shiptrack", "feature1", "label"],
            "tab_field_list": ["feature1"],
            "cat_field_list": [],
            "label_name": "label",
            "id_name": "id",
            "primary_text_name": "chat",
            "secondary_text_name": "shiptrack",
            "model_class": "TrimodalBERT",
        }
        self._create_model_with_hyperparams(dirs, hyperparams)

        input_paths = {"model_input": str(dirs["input_model_dir"])}
        output_paths = {"output_dir": str(dirs["output_dir"])}
        environ_vars = {
            "CONTENT_TYPES": "application/json",
            "FIELD_DEFAULTS": '{"chat": "Custom chat", "shiptrack": "Custom tracking"}',
            "WORKING_DIRECTORY": str(dirs["working_dir"]),
        }

        result = payload_main(input_paths, output_paths, environ_vars)

        assert Path(result).exists()

    def test_main_with_feature_columns(self, setup_dirs):
        """Test main flow with feature_columns.txt for XGBoost/LightGBM."""
        dirs = setup_dirs
        hyperparams = {
            "full_field_list": ["feature1", "feature2", "category1", "label"],
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "label",
        }
        self._create_model_with_hyperparams(dirs, hyperparams, include_feature_columns=True)

        input_paths = {"model_input": str(dirs["input_model_dir"])}
        output_paths = {"output_dir": str(dirs["output_dir"])}
        environ_vars = {
            "CONTENT_TYPES": "text/csv",
            "WORKING_DIRECTORY": str(dirs["working_dir"]),
        }

        result = payload_main(input_paths, output_paths, environ_vars)

        assert Path(result).exists()
        # CSV should use feature_columns.txt ordering


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
