import pytest
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List
import logging
import tempfile
import os

from cursus.core.base.contract_base import (
    ScriptContract,
    ValidationResult,
    ScriptAnalyzer,
)


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_init_valid(self):
        """Test initialization with valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_init_invalid_with_errors(self):
        """Test initialization with invalid result and errors."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]

        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        assert not result.is_valid
        assert result.errors == errors
        assert result.warnings == warnings

    def test_success_class_method(self):
        """Test success class method."""
        result = ValidationResult.success("Test success")

        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_error_class_method_with_list(self):
        """Test error class method with list of errors."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult.error(errors)

        assert not result.is_valid
        assert result.errors == errors

    def test_error_class_method_with_string(self):
        """Test error class method with single error string."""
        error = "Single error"
        result = ValidationResult.error(error)

        assert not result.is_valid
        assert result.errors == [error]

    def test_combine_class_method(self):
        """Test combine class method."""
        result1 = ValidationResult(is_valid=True, warnings=["Warning 1"])
        result2 = ValidationResult(
            is_valid=False, errors=["Error 1"], warnings=["Warning 2"]
        )
        result3 = ValidationResult(is_valid=False, errors=["Error 2"])

        combined = ValidationResult.combine([result1, result2, result3])

        assert not combined.is_valid
        assert combined.errors == ["Error 1", "Error 2"]
        assert combined.warnings == ["Warning 1", "Warning 2"]

    def test_combine_all_valid(self):
        """Test combine with all valid results."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True, warnings=["Warning 1"])

        combined = ValidationResult.combine([result1, result2])

        assert combined.is_valid
        assert combined.errors == []
        assert combined.warnings == ["Warning 1"]


class TestScriptContract:
    """Test cases for ScriptContract class."""

    @pytest.fixture
    def valid_contract_data(self):
        """Set up test fixtures."""
        return {
            "entry_point": "train.py",
            "expected_input_paths": {
                "training_data": "/opt/ml/processing/input/train",
                "validation_data": "/opt/ml/processing/input/validation",
            },
            "expected_output_paths": {
                "model": "/opt/ml/processing/output/model",
                "metrics": "/opt/ml/processing/output/metrics",
            },
            "required_env_vars": ["MODEL_TYPE", "LEARNING_RATE"],
            "optional_env_vars": {"BATCH_SIZE": "32", "EPOCHS": "10"},
            "expected_arguments": {
                "input-path": "/opt/ml/processing/input/train",
                "output-path": "/opt/ml/processing/output/model",
            },
            "framework_requirements": {"xgboost": "1.5.0"},
            "description": "Training script for XGBoost model",
        }

    def test_init_with_valid_data(self, valid_contract_data):
        """Test initialization with valid data."""
        contract = ScriptContract(**valid_contract_data)

        assert contract.entry_point == "train.py"
        assert (
            contract.expected_input_paths["training_data"]
            == "/opt/ml/processing/input/train"
        )
        assert (
            contract.expected_output_paths["model"] == "/opt/ml/processing/output/model"
        )
        assert contract.required_env_vars == ["MODEL_TYPE", "LEARNING_RATE"]
        assert contract.optional_env_vars["BATCH_SIZE"] == "32"
        assert (
            contract.expected_arguments["input-path"]
            == "/opt/ml/processing/input/train"
        )
        assert contract.framework_requirements["xgboost"] == "1.5.0"
        assert contract.description == "Training script for XGBoost model"

    def test_init_with_minimal_data(self):
        """Test initialization with minimal required data."""
        minimal_data = {
            "entry_point": "script.py",
            "expected_input_paths": {},
            "expected_output_paths": {},
            "required_env_vars": [],
        }

        contract = ScriptContract(**minimal_data)

        assert contract.entry_point == "script.py"
        assert contract.expected_input_paths == {}
        assert contract.expected_output_paths == {}
        assert contract.required_env_vars == []
        assert contract.optional_env_vars == {}
        assert contract.expected_arguments == {}
        assert contract.framework_requirements == {}
        assert contract.description == ""

    def test_validate_entry_point_invalid(self, valid_contract_data):
        """Test entry point validation with invalid file."""
        invalid_data = valid_contract_data.copy()
        invalid_data["entry_point"] = "script.txt"

        with pytest.raises(ValueError) as exc_info:
            ScriptContract(**invalid_data)

        assert "Entry point must be a Python file" in str(exc_info.value)

    def test_validate_input_paths_invalid(self, valid_contract_data):
        """Test input path validation with invalid paths."""
        invalid_data = valid_contract_data.copy()
        invalid_data["expected_input_paths"] = {"data": "/invalid/path"}

        with pytest.raises(ValueError) as exc_info:
            ScriptContract(**invalid_data)

        assert "must start with /opt/ml/processing/input" in str(exc_info.value)

    def test_validate_input_paths_generated_payload_samples(self, valid_contract_data):
        """Test input path validation for GeneratedPayloadSamples."""
        valid_data = valid_contract_data.copy()
        valid_data["expected_input_paths"] = {
            "GeneratedPayloadSamples": "/opt/ml/processing/payload/samples"
        }

        # Should not raise an exception
        contract = ScriptContract(**valid_data)
        assert (
            contract.expected_input_paths["GeneratedPayloadSamples"]
            == "/opt/ml/processing/payload/samples"
        )

    def test_validate_output_paths_invalid(self, valid_contract_data):
        """Test output path validation with invalid paths."""
        invalid_data = valid_contract_data.copy()
        invalid_data["expected_output_paths"] = {"result": "/invalid/path"}

        with pytest.raises(ValueError) as exc_info:
            ScriptContract(**invalid_data)

        assert "must start with /opt/ml/processing/output" in str(exc_info.value)

    def test_validate_arguments_invalid_characters(self, valid_contract_data):
        """Test argument validation with invalid characters."""
        invalid_data = valid_contract_data.copy()
        invalid_data["expected_arguments"] = {"input_path!": "/some/path"}

        with pytest.raises(ValueError) as exc_info:
            ScriptContract(**invalid_data)

        assert "contains invalid characters" in str(exc_info.value)

    def test_validate_arguments_uppercase(self, valid_contract_data):
        """Test argument validation with uppercase characters."""
        invalid_data = valid_contract_data.copy()
        invalid_data["expected_arguments"] = {"INPUT-PATH": "/some/path"}

        with pytest.raises(ValueError) as exc_info:
            ScriptContract(**invalid_data)

        assert "should be lowercase" in str(exc_info.value)

    def test_validate_implementation_file_not_found(self, valid_contract_data):
        """Test implementation validation with non-existent file."""
        contract = ScriptContract(**valid_contract_data)

        result = contract.validate_implementation("/nonexistent/script.py")

        assert not result.is_valid
        assert "Script file not found" in result.errors[0]

    @patch("cursus.core.base.contract_base.ScriptAnalyzer")
    def test_validate_implementation_success(
        self, mock_analyzer_class, valid_contract_data
    ):
        """Test successful implementation validation."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {
            "/opt/ml/processing/input/train",
            "/opt/ml/processing/input/validation",
        }
        mock_analyzer.get_output_paths.return_value = {
            "/opt/ml/processing/output/model",
            "/opt/ml/processing/output/metrics",
        }
        mock_analyzer.get_env_var_usage.return_value = {"MODEL_TYPE", "LEARNING_RATE"}
        mock_analyzer.get_argument_usage.return_value = {"input-path", "output-path"}
        mock_analyzer_class.return_value = mock_analyzer

        contract = ScriptContract(**valid_contract_data)

        with patch("os.path.exists", return_value=True):
            result = contract.validate_implementation("/test/script.py")

        assert result.is_valid
        assert result.errors == []

    @patch("cursus.core.base.contract_base.ScriptAnalyzer")
    def test_validate_implementation_missing_paths(
        self, mock_analyzer_class, valid_contract_data
    ):
        """Test implementation validation with missing paths."""
        # Mock analyzer with missing paths
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {
            "/opt/ml/processing/input/train"
        }  # Missing validation
        mock_analyzer.get_output_paths.return_value = {
            "/opt/ml/processing/output/model"
        }  # Missing metrics
        mock_analyzer.get_env_var_usage.return_value = {
            "MODEL_TYPE"
        }  # Missing LEARNING_RATE
        mock_analyzer.get_argument_usage.return_value = {
            "input-path"
        }  # Missing output-path
        mock_analyzer_class.return_value = mock_analyzer

        contract = ScriptContract(**valid_contract_data)

        with patch("os.path.exists", return_value=True):
            result = contract.validate_implementation("/test/script.py")

        assert not result.is_valid
        assert any("validation" in error for error in result.errors)
        assert any("metrics" in error for error in result.errors)
        assert any("LEARNING_RATE" in error for error in result.errors)

    @patch("cursus.core.base.contract_base.ScriptAnalyzer")
    def test_validate_implementation_with_warnings(
        self, mock_analyzer_class, valid_contract_data
    ):
        """Test implementation validation with warnings."""
        # Mock analyzer with extra paths
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {
            "/opt/ml/processing/input/train",
            "/opt/ml/processing/input/validation",
            "/opt/ml/processing/input/extra",  # Extra path
        }
        mock_analyzer.get_output_paths.return_value = {
            "/opt/ml/processing/output/model",
            "/opt/ml/processing/output/metrics",
        }
        mock_analyzer.get_env_var_usage.return_value = {"MODEL_TYPE", "LEARNING_RATE"}
        mock_analyzer.get_argument_usage.return_value = {
            "input-path"
        }  # Missing output-path
        mock_analyzer_class.return_value = mock_analyzer

        contract = ScriptContract(**valid_contract_data)

        with patch("os.path.exists", return_value=True):
            result = contract.validate_implementation("/test/script.py")

        assert result.is_valid
        assert any("undeclared input path" in warning for warning in result.warnings)
        assert any("output-path" in warning for warning in result.warnings)


class TestScriptAnalyzer:
    """Test cases for ScriptAnalyzer class."""

    @pytest.fixture
    def sample_script(self):
        """Set up test fixtures."""
        return """
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Input paths
    train_data = "/opt/ml/processing/input/train"
    val_data = "/opt/ml/processing/input/validation"
    
    # Output paths
    model_path = "/opt/ml/processing/output/model"
    metrics_path = "/opt/ml/processing/output/metrics"
    
    # Environment variables
    model_type = os.environ["MODEL_TYPE"]
    learning_rate = os.environ.get("LEARNING_RATE", "0.1")
    batch_size = os.getenv("BATCH_SIZE", "32")
    
    print(f"Training with {model_type}")

if __name__ == "__main__":
    main()
"""

    def test_get_input_paths(self, sample_script):
        """Test extraction of input paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)
                input_paths = analyzer.get_input_paths()

                expected_paths = {
                    "/opt/ml/processing/input/train",
                    "/opt/ml/processing/input/validation",
                }
                assert input_paths == expected_paths
            finally:
                os.unlink(f.name)

    def test_get_output_paths(self, sample_script):
        """Test extraction of output paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)
                output_paths = analyzer.get_output_paths()

                expected_paths = {
                    "/opt/ml/processing/output/model",
                    "/opt/ml/processing/output/metrics",
                }
                assert output_paths == expected_paths
            finally:
                os.unlink(f.name)

    def test_get_env_var_usage(self, sample_script):
        """Test extraction of environment variable usage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)
                env_vars = analyzer.get_env_var_usage()

                expected_vars = {"MODEL_TYPE", "LEARNING_RATE", "BATCH_SIZE"}
                assert env_vars == expected_vars
            finally:
                os.unlink(f.name)

    def test_get_argument_usage(self, sample_script):
        """Test extraction of argument usage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)
                arguments = analyzer.get_argument_usage()

                expected_args = {
                    "input-path",
                    "output-path",
                    "v",
                }  # Script analyzer finds short form "v"
                assert arguments == expected_args
            finally:
                os.unlink(f.name)

    def test_ast_tree_lazy_loading(self, sample_script):
        """Test that AST tree is lazily loaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)

                # AST should not be loaded yet
                assert analyzer._ast_tree is None

                # Access AST tree
                ast_tree = analyzer.ast_tree

                # Should now be loaded
                assert ast_tree is not None
                assert analyzer._ast_tree == ast_tree

                # Second access should return same object
                ast_tree2 = analyzer.ast_tree
                assert ast_tree == ast_tree2
            finally:
                os.unlink(f.name)

    def test_caching_behavior(self, sample_script):
        """Test that analysis results are cached."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_script)
            f.flush()

            try:
                analyzer = ScriptAnalyzer(f.name)

                # First call should populate cache
                input_paths1 = analyzer.get_input_paths()
                assert analyzer._input_paths is not None

                # Second call should use cache
                input_paths2 = analyzer.get_input_paths()
                assert input_paths1 == input_paths2

                # Same for other methods
                env_vars1 = analyzer.get_env_var_usage()
                env_vars2 = analyzer.get_env_var_usage()
                assert env_vars1 == env_vars2
            finally:
                os.unlink(f.name)
