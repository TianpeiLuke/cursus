"""
Test suite for hybrid registry utilities.

Tests the utility functions in cursus.registry.hybrid.utils.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.registry.hybrid.utils import (
    RegistryLoadError,
    load_registry_module,
    get_step_names_from_module,
    from_legacy_format,
    to_legacy_format,
    convert_registry_dict,
    validate_registry_type,
    validate_step_name,
    validate_workspace_id,
    validate_registry_data,
    format_registry_error,
    format_step_not_found_error,
    format_registry_load_error,
    format_validation_error,
)
from cursus.registry.hybrid.models import StepDefinition, RegistryType


class TestRegistryLoading(unittest.TestCase):
    """Test registry loading functions."""

    def test_load_registry_module_success(self):
        """Test successful registry module loading."""
        # Create temporary registry file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
STEP_NAMES = {
    "TestStep": {
        "config_class": "TestStepConfig",
        "builder_step_name": "TestStepBuilder",
        "spec_type": "TestStep",
        "sagemaker_step_type": "Processing",
        "description": "Test step"
    }
}
"""
            )
            temp_file = f.name

        try:
            # Load the module
            module = load_registry_module(temp_file)

            # Verify module was loaded correctly
            self.assertTrue(hasattr(module, "STEP_NAMES"))
            self.assertIn("TestStep", module.STEP_NAMES)
            self.assertEqual(
                module.STEP_NAMES["TestStep"]["config_class"], "TestStepConfig"
            )

        finally:
            # Clean up
            Path(temp_file).unlink()

    def test_load_registry_module_file_not_found(self):
        """Test loading non-existent registry file."""
        with self.assertRaises(RegistryLoadError) as exc_info:
            load_registry_module("nonexistent.py")

        self.assertIn("Registry file not found", str(exc_info.exception))

    def test_load_registry_module_invalid_python(self):
        """Test loading invalid Python file."""
        # Create temporary file with invalid Python
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("invalid python syntax !!!")
            temp_file = f.name

        try:
            with self.assertRaises(RegistryLoadError) as exc_info:
                load_registry_module(temp_file)

            self.assertIn("Failed to load registry from", str(exc_info.exception))

        finally:
            Path(temp_file).unlink()

    def test_get_step_names_from_module(self):
        """Test extracting STEP_NAMES from module."""
        mock_module = Mock()
        mock_module.STEP_NAMES = {"TestStep": {"config_class": "TestConfig"}}

        result = get_step_names_from_module(mock_module)
        self.assertEqual(result, {"TestStep": {"config_class": "TestConfig"}})

    def test_get_step_names_from_module_missing(self):
        """Test extracting STEP_NAMES from module without STEP_NAMES."""
        mock_module = Mock()
        # Remove STEP_NAMES attribute to simulate missing attribute
        del mock_module.STEP_NAMES

        result = get_step_names_from_module(mock_module)
        self.assertEqual(result, {})


class TestStepDefinitionConversion(unittest.TestCase):
    """Test step definition conversion functions."""

    def setUp(self):
        """Set up test data."""
        self.sample_step_info = {
            "config_class": "TestStepConfig",
            "builder_step_name": "TestStepBuilder",
            "spec_type": "TestStep",
            "sagemaker_step_type": "Processing",
            "description": "Test step description",
            "framework": "pytorch",
            "job_types": ["training", "inference"],
        }

    def test_from_legacy_format_basic(self):
        """Test converting from legacy format to StepDefinition."""
        result = from_legacy_format(
            "TestStep", self.sample_step_info, registry_type="core", workspace_id=None
        )

        self.assertIsInstance(result, StepDefinition)
        self.assertEqual(result.name, "TestStep")
        self.assertEqual(result.config_class, "TestStepConfig")
        self.assertEqual(result.builder_step_name, "TestStepBuilder")
        self.assertEqual(result.spec_type, "TestStep")
        self.assertEqual(result.sagemaker_step_type, "Processing")
        self.assertEqual(result.description, "Test step description")
        self.assertEqual(result.framework, "pytorch")
        self.assertEqual(result.job_types, ["training", "inference"])
        self.assertEqual(result.registry_type, RegistryType.CORE)
        self.assertIsNone(result.workspace_id)

    def test_from_legacy_format_workspace(self):
        """Test converting with workspace context."""
        result = from_legacy_format(
            "TestStep",
            self.sample_step_info,
            registry_type="workspace",
            workspace_id="test_workspace",
        )

        self.assertEqual(result.registry_type, RegistryType.WORKSPACE)
        self.assertEqual(result.workspace_id, "test_workspace")

    def test_to_legacy_format(self):
        """Test converting from StepDefinition to legacy format."""
        step_def = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step description",
            framework="pytorch",
            job_types=["training", "inference"],
            registry_type=RegistryType.CORE,
        )

        result = to_legacy_format(step_def)

        expected = {
            "config_class": "TestStepConfig",
            "builder_step_name": "TestStepBuilder",
            "spec_type": "TestStep",
            "sagemaker_step_type": "Processing",
            "description": "Test step description",
            "framework": "pytorch",
            "job_types": ["training", "inference"],
        }

        self.assertEqual(result, expected)

    def test_to_legacy_format_with_metadata(self):
        """Test converting with metadata."""
        step_def = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            registry_type=RegistryType.CORE,
            metadata={"custom_field": "custom_value"},
        )

        result = to_legacy_format(step_def)

        # Should include metadata
        self.assertEqual(result["custom_field"], "custom_value")

    def test_convert_registry_dict(self):
        """Test batch conversion of registry dictionary."""
        registry_dict = {
            "TestStep1": self.sample_step_info,
            "TestStep2": {
                "config_class": "TestStep2Config",
                "builder_step_name": "TestStep2Builder",
                "spec_type": "TestStep2",
                "sagemaker_step_type": "Training",
            },
        }

        result = convert_registry_dict(
            registry_dict, registry_type="workspace", workspace_id="test_workspace"
        )

        self.assertEqual(len(result), 2)
        self.assertIn("TestStep1", result)
        self.assertIn("TestStep2", result)

        # Check first step
        step1 = result["TestStep1"]
        self.assertIsInstance(step1, StepDefinition)
        self.assertEqual(step1.name, "TestStep1")
        self.assertEqual(step1.registry_type, RegistryType.WORKSPACE)
        self.assertEqual(step1.workspace_id, "test_workspace")

        # Check second step
        step2 = result["TestStep2"]
        self.assertEqual(step2.name, "TestStep2")
        self.assertEqual(step2.config_class, "TestStep2Config")


class TestValidationFunctions(unittest.TestCase):
    """Test validation functions."""

    def test_validate_registry_type_valid(self):
        """Test validation of valid registry types."""
        valid_types = ["core", "workspace", "override"]

        for registry_type in valid_types:
            result = validate_registry_type(registry_type)
            self.assertEqual(result, registry_type)

    def test_validate_registry_type_invalid(self):
        """Test validation of invalid registry type."""
        with self.assertRaises(ValueError) as exc_info:
            validate_registry_type("invalid_type")

        self.assertIn("registry_type must be one of", str(exc_info.exception))

    def test_validate_step_name_valid(self):
        """Test validation of valid step names."""
        valid_names = ["TestStep", "XGBoostTraining", "test_step", "Test-Step"]

        for step_name in valid_names:
            result = validate_step_name(step_name)
            self.assertEqual(result, step_name.strip())

    def test_validate_step_name_empty(self):
        """Test validation of empty step name."""
        with self.assertRaises(ValueError) as exc_info:
            validate_step_name("")

        self.assertIn("Step name cannot be empty", str(exc_info.exception))

    def test_validate_step_name_invalid_characters(self):
        """Test validation of step name with invalid characters."""
        with self.assertRaises(ValueError) as exc_info:
            validate_step_name("Test@Step!")

        self.assertIn("contains invalid characters", str(exc_info.exception))

    def test_validate_workspace_id_valid(self):
        """Test validation of valid workspace IDs."""
        result = validate_workspace_id("test_workspace")
        self.assertEqual(result, "test_workspace")

    def test_validate_workspace_id_none(self):
        """Test validation of None workspace ID."""
        result = validate_workspace_id(None)
        self.assertIsNone(result)

    def test_validate_workspace_id_invalid(self):
        """Test validation of invalid workspace ID."""
        with self.assertRaises(ValueError) as exc_info:
            validate_workspace_id("invalid@workspace!")

        self.assertIn("contains invalid characters", str(exc_info.exception))

    def test_validate_registry_data_success(self):
        """Test successful registry data validation."""
        result = validate_registry_data(
            registry_type="core", step_name="TestStep", workspace_id=None
        )
        self.assertTrue(result)

    def test_validate_registry_data_with_workspace(self):
        """Test registry data validation with workspace."""
        result = validate_registry_data(
            registry_type="workspace",
            step_name="TestStep",
            workspace_id="test_workspace",
        )
        self.assertTrue(result)

    def test_validate_registry_data_invalid_type(self):
        """Test registry data validation with invalid type."""
        with self.assertRaises(ValueError):
            validate_registry_data(registry_type="invalid", step_name="TestStep")

    def test_validate_registry_data_invalid_step_name(self):
        """Test registry data validation with invalid step name."""
        with self.assertRaises(ValueError):
            validate_registry_data(registry_type="core", step_name="Invalid@Step!")


class TestErrorFormatting(unittest.TestCase):
    """Test error formatting functions."""

    def test_format_registry_error_step_not_found(self):
        """Test step not found error formatting."""
        result = format_registry_error(
            "step_not_found",
            step_name="TestStep",
            workspace_context="test_workspace",
            available_steps=["Step1", "Step2"],
        )

        self.assertIn("Step 'TestStep' not found", result)
        self.assertIn("workspace: test_workspace", result)
        self.assertIn("Available steps: Step1, Step2", result)

    def test_format_registry_error_registry_load(self):
        """Test registry load error formatting."""
        result = format_registry_error(
            "registry_load",
            registry_path="/path/to/registry.py",
            error_details="File not found",
        )

        self.assertIn("Failed to load registry from '/path/to/registry.py'", result)
        self.assertIn("File not found", result)

    def test_format_registry_error_validation(self):
        """Test validation error formatting."""
        result = format_registry_error(
            "validation",
            component_name="TestStep",
            validation_issues=["Missing config_class", "Invalid priority"],
        )

        self.assertIn("Validation failed for 'TestStep'", result)
        self.assertIn("1. Missing config_class", result)
        self.assertIn("2. Invalid priority", result)

    def test_format_step_not_found_error(self):
        """Test step not found error formatting function."""
        result = format_step_not_found_error(
            step_name="TestStep",
            workspace_context="test_workspace",
            available_steps=["Step1", "Step2"],
        )

        self.assertIn("Step 'TestStep' not found", result)
        self.assertIn("workspace: test_workspace", result)
        self.assertIn("Available steps: Step1, Step2", result)

    def test_format_registry_load_error(self):
        """Test registry load error formatting function."""
        result = format_registry_load_error(
            registry_path="/path/to/registry.py", error_details="Permission denied"
        )

        self.assertIn("Failed to load registry from '/path/to/registry.py'", result)
        self.assertIn("Permission denied", result)

    def test_format_validation_error(self):
        """Test validation error formatting function."""
        result = format_validation_error(
            component_name="TestStep",
            validation_issues=["Missing required field", "Invalid format"],
        )

        self.assertIn("Validation failed for 'TestStep'", result)
        self.assertIn("1. Missing required field", result)
        self.assertIn("2. Invalid format", result)

    def test_format_registry_error_unknown_type(self):
        """Test formatting unknown error type."""
        result = format_registry_error("unknown_error", error="Something went wrong")

        self.assertIn("Registry error: Something went wrong", result)


class TestRegistryLoadError(unittest.TestCase):
    """Test RegistryLoadError exception."""

    def test_registry_load_error_creation(self):
        """Test creating RegistryLoadError."""
        error = RegistryLoadError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)

    def test_registry_load_error_inheritance(self):
        """Test RegistryLoadError inheritance."""
        from cursus.registry.exceptions import RegistryError

        error = RegistryLoadError("Test error")
        self.assertIsInstance(error, RegistryError)


if __name__ == "__main__":
    unittest.main()
