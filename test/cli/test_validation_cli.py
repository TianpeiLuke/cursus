"""
Unit tests for the validation CLI module.

This module tests all functionality of the validation command-line interface,
including argument parsing, command execution, and output formatting.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from io import StringIO
import argparse

from cursus.cli.validation_cli import (
    print_violations,
    validate_registry,
    validate_file_name,
    validate_step_name,
    validate_logical_name,
    main,
)
from cursus.validation.naming.naming_standard_validator import NamingViolation


class TestPrintViolations(unittest.TestCase):
    """Test the print_violations function."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch("sys.stdout", new_callable=StringIO)
        self.mock_stdout = self.patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()

    def test_print_violations_no_violations(self):
        """Test printing when there are no violations."""
        violations = []
        print_violations(violations)

        output = self.mock_stdout.getvalue()
        self.assertIn("✅ No violations found!", output)

    def test_print_violations_single_violation(self):
        """Test printing a single violation."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            expected="expected_value",
            actual="actual_value",
        )
        violations = [violation]

        print_violations(violations)

        output = self.mock_stdout.getvalue()
        self.assertIn("❌ Found 1 naming violations:", output)
        self.assertIn("📁 TestComponent:", output)
        self.assertIn("Test message", output)

    def test_print_violations_multiple_violations_same_component(self):
        """Test printing multiple violations from the same component."""
        violations = [
            NamingViolation(
                component="TestComponent", violation_type="type1", message="Message 1"
            ),
            NamingViolation(
                component="TestComponent", violation_type="type2", message="Message 2"
            ),
        ]

        print_violations(violations)

        output = self.mock_stdout.getvalue()
        self.assertIn("❌ Found 2 naming violations:", output)
        self.assertIn("📁 TestComponent:", output)
        self.assertIn("Message 1", output)
        self.assertIn("Message 2", output)

    def test_print_violations_multiple_components(self):
        """Test printing violations from multiple components."""
        violations = [
            NamingViolation(
                component="Component1", violation_type="type1", message="Message 1"
            ),
            NamingViolation(
                component="Component2", violation_type="type2", message="Message 2"
            ),
        ]

        print_violations(violations)

        output = self.mock_stdout.getvalue()
        self.assertIn("❌ Found 2 naming violations:", output)
        self.assertIn("📁 Component1:", output)
        self.assertIn("📁 Component2:", output)
        self.assertIn("Message 1", output)
        self.assertIn("Message 2", output)

    def test_print_violations_with_suggestions_verbose(self):
        """Test printing violations with suggestions in verbose mode."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            suggestions=["suggestion1", "suggestion2"],
        )
        violations = [violation]

        print_violations(violations, verbose=True)

        output = self.mock_stdout.getvalue()
        self.assertIn("💡 Suggestions: suggestion1, suggestion2", output)

    def test_print_violations_with_suggestions_not_verbose(self):
        """Test printing violations with suggestions in non-verbose mode."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            suggestions=["suggestion1", "suggestion2"],
        )
        violations = [violation]

        print_violations(violations, verbose=False)

        output = self.mock_stdout.getvalue()
        self.assertNotIn("💡 Suggestions:", output)


class TestValidationFunctions(unittest.TestCase):
    """Test the individual validation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.stdout_patcher = patch("sys.stdout", new_callable=StringIO)
        self.mock_stdout = self.stdout_patcher.start()

        self.validator_patcher = patch(
            "cursus.cli.validation_cli.NamingStandardValidator"
        )
        self.mock_validator_class = self.validator_patcher.start()
        self.mock_validator = MagicMock()
        self.mock_validator_class.return_value = self.mock_validator

    def tearDown(self):
        """Clean up after tests."""
        self.stdout_patcher.stop()
        self.validator_patcher.stop()

    def test_validate_registry_no_violations(self):
        """Test registry validation with no violations."""
        self.mock_validator.validate_all_registry_entries.return_value = []

        result = validate_registry()

        self.assertEqual(result, 0)
        self.mock_validator.validate_all_registry_entries.assert_called_once()
        output = self.mock_stdout.getvalue()
        self.assertIn("🔍 Validating registry entries...", output)
        self.assertIn("✅ No violations found!", output)

    def test_validate_registry_with_violations(self):
        """Test registry validation with violations."""
        violations = [
            NamingViolation("Component1", "type1", "Message 1"),
            NamingViolation("Component2", "type2", "Message 2"),
        ]
        self.mock_validator.validate_all_registry_entries.return_value = violations

        result = validate_registry()

        self.assertEqual(result, 2)
        self.mock_validator.validate_all_registry_entries.assert_called_once()
        output = self.mock_stdout.getvalue()
        self.assertIn("❌ Found 2 naming violations:", output)

    def test_validate_file_name_no_violations(self):
        """Test file name validation with no violations."""
        self.mock_validator.validate_file_naming.return_value = []

        result = validate_file_name("test_file.py", "builder")

        self.assertEqual(result, 0)
        self.mock_validator.validate_file_naming.assert_called_once_with(
            "test_file.py", "builder"
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("🔍 Validating file name: test_file.py (type: builder)", output)

    def test_validate_file_name_with_violations(self):
        """Test file name validation with violations."""
        violations = [
            NamingViolation("test_file.py", "invalid_name", "Invalid file name")
        ]
        self.mock_validator.validate_file_naming.return_value = violations

        result = validate_file_name("test_file.py", "builder")

        self.assertEqual(result, 1)
        self.mock_validator.validate_file_naming.assert_called_once_with(
            "test_file.py", "builder"
        )

    def test_validate_step_name_no_violations(self):
        """Test step name validation with no violations."""
        self.mock_validator._validate_canonical_step_name.return_value = []

        result = validate_step_name("XGBoostTraining")

        self.assertEqual(result, 0)
        self.mock_validator._validate_canonical_step_name.assert_called_once_with(
            "XGBoostTraining", "CLI"
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("🔍 Validating step name: XGBoostTraining", output)

    def test_validate_step_name_with_violations(self):
        """Test step name validation with violations."""
        violations = [
            NamingViolation("XGBoostTraining", "invalid_case", "Invalid case")
        ]
        self.mock_validator._validate_canonical_step_name.return_value = violations

        result = validate_step_name("xgboost_training")

        self.assertEqual(result, 1)
        self.mock_validator._validate_canonical_step_name.assert_called_once_with(
            "xgboost_training", "CLI"
        )

    def test_validate_logical_name_no_violations(self):
        """Test logical name validation with no violations."""
        self.mock_validator._validate_logical_name.return_value = []

        result = validate_logical_name("input_data")

        self.assertEqual(result, 0)
        self.mock_validator._validate_logical_name.assert_called_once_with(
            "input_data", "CLI"
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("🔍 Validating logical name: input_data", output)

    def test_validate_logical_name_with_violations(self):
        """Test logical name validation with violations."""
        violations = [
            NamingViolation("inputData", "invalid_case", "Should be snake_case")
        ]
        self.mock_validator._validate_logical_name.return_value = violations

        result = validate_logical_name("inputData")

        self.assertEqual(result, 1)
        self.mock_validator._validate_logical_name.assert_called_once_with(
            "inputData", "CLI"
        )


class TestMainFunction(unittest.TestCase):
    """Test the main CLI function."""

    def setUp(self):
        """Set up test fixtures."""
        self.stdout_patcher = patch("sys.stdout", new_callable=StringIO)
        self.mock_stdout = self.stdout_patcher.start()

        # Mock the validation functions
        self.validate_registry_patcher = patch(
            "cursus.cli.validation_cli.validate_registry"
        )
        self.mock_validate_registry = self.validate_registry_patcher.start()

        self.validate_file_name_patcher = patch(
            "cursus.cli.validation_cli.validate_file_name"
        )
        self.mock_validate_file_name = self.validate_file_name_patcher.start()

        self.validate_step_name_patcher = patch(
            "cursus.cli.validation_cli.validate_step_name"
        )
        self.mock_validate_step_name = self.validate_step_name_patcher.start()

        self.validate_logical_name_patcher = patch(
            "cursus.cli.validation_cli.validate_logical_name"
        )
        self.mock_validate_logical_name = self.validate_logical_name_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.stdout_patcher.stop()
        self.validate_registry_patcher.stop()
        self.validate_file_name_patcher.stop()
        self.validate_step_name_patcher.stop()
        self.validate_logical_name_patcher.stop()

    @patch("sys.argv", ["validation_cli.py"])
    def test_main_no_command(self):
        """Test main function with no command provided."""
        with patch("argparse.ArgumentParser.print_help") as mock_help:
            result = main()
            self.assertEqual(result, 1)
            mock_help.assert_called_once()

    @patch("sys.argv", ["validation_cli.py", "registry"])
    def test_main_registry_command_success(self):
        """Test main function with registry command - success case."""
        self.mock_validate_registry.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_registry.assert_called_once_with(False)
        output = self.mock_stdout.getvalue()
        self.assertIn("✅ All naming conventions checks passed!", output)

    @patch("sys.argv", ["validation_cli.py", "-v", "registry"])
    def test_main_registry_command_verbose(self):
        """Test main function with registry command in verbose mode."""
        self.mock_validate_registry.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_registry.assert_called_once_with(True)

    @patch("sys.argv", ["validation_cli.py", "registry"])
    def test_main_registry_command_with_violations(self):
        """Test main function with registry command - violations found."""
        self.mock_validate_registry.return_value = 3

        result = main()

        self.assertEqual(result, 1)
        self.mock_validate_registry.assert_called_once_with(False)
        output = self.mock_stdout.getvalue()
        self.assertIn("⚠️  Found 3 violation(s)", output)

    @patch("sys.argv", ["validation_cli.py", "file", "test_file.py", "builder"])
    def test_main_file_command_success(self):
        """Test main function with file command - success case."""
        self.mock_validate_file_name.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_file_name.assert_called_once_with(
            "test_file.py", "builder", False
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("✅ All naming conventions checks passed!", output)

    @patch("sys.argv", ["validation_cli.py", "-v", "file", "test_file.py", "builder"])
    def test_main_file_command_verbose(self):
        """Test main function with file command in verbose mode."""
        self.mock_validate_file_name.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_file_name.assert_called_once_with(
            "test_file.py", "builder", True
        )

    @patch("sys.argv", ["validation_cli.py", "file", "invalid_file.py", "config"])
    def test_main_file_command_with_violations(self):
        """Test main function with file command - violations found."""
        self.mock_validate_file_name.return_value = 2

        result = main()

        self.assertEqual(result, 1)
        self.mock_validate_file_name.assert_called_once_with(
            "invalid_file.py", "config", False
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("⚠️  Found 2 violation(s)", output)

    @patch("sys.argv", ["validation_cli.py", "step", "XGBoostTraining"])
    def test_main_step_command_success(self):
        """Test main function with step command - success case."""
        self.mock_validate_step_name.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_step_name.assert_called_once_with("XGBoostTraining", False)
        output = self.mock_stdout.getvalue()
        self.assertIn("✅ All naming conventions checks passed!", output)

    @patch("sys.argv", ["validation_cli.py", "step", "invalid_step_name"])
    def test_main_step_command_with_violations(self):
        """Test main function with step command - violations found."""
        self.mock_validate_step_name.return_value = 1

        result = main()

        self.assertEqual(result, 1)
        self.mock_validate_step_name.assert_called_once_with("invalid_step_name", False)
        output = self.mock_stdout.getvalue()
        self.assertIn("⚠️  Found 1 violation(s)", output)

    @patch("sys.argv", ["validation_cli.py", "logical", "input_data"])
    def test_main_logical_command_success(self):
        """Test main function with logical command - success case."""
        self.mock_validate_logical_name.return_value = 0

        result = main()

        self.assertEqual(result, 0)
        self.mock_validate_logical_name.assert_called_once_with("input_data", False)
        output = self.mock_stdout.getvalue()
        self.assertIn("✅ All naming conventions checks passed!", output)

    @patch("sys.argv", ["validation_cli.py", "logical", "invalidLogicalName"])
    def test_main_logical_command_with_violations(self):
        """Test main function with logical command - violations found."""
        self.mock_validate_logical_name.return_value = 1

        result = main()

        self.assertEqual(result, 1)
        self.mock_validate_logical_name.assert_called_once_with(
            "invalidLogicalName", False
        )
        output = self.mock_stdout.getvalue()
        self.assertIn("⚠️  Found 1 violation(s)", output)

    @patch("sys.argv", ["validation_cli.py", "invalid_command"])
    def test_main_invalid_command(self):
        """Test main function with invalid command."""
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(
            cm.exception.code, 2
        )  # argparse exits with code 2 for invalid arguments

    @patch("sys.argv", ["validation_cli.py", "registry"])
    def test_main_exception_handling(self):
        """Test main function exception handling."""
        self.mock_validate_registry.side_effect = Exception("Test exception")

        result = main()

        self.assertEqual(result, 1)
        output = self.mock_stdout.getvalue()
        self.assertIn("❌ Error during validation: Test exception", output)

    @patch("sys.argv", ["validation_cli.py", "-v", "registry"])
    def test_main_exception_handling_verbose(self):
        """Test main function exception handling in verbose mode."""
        self.mock_validate_registry.side_effect = Exception("Test exception")

        with patch("traceback.print_exc") as mock_traceback:
            result = main()

            self.assertEqual(result, 1)
            mock_traceback.assert_called_once()
            output = self.mock_stdout.getvalue()
            self.assertIn("❌ Error during validation: Test exception", output)


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing functionality."""

    def test_file_command_valid_file_types(self):
        """Test that file command accepts valid file types."""
        valid_types = ["builder", "config", "spec", "contract"]

        for file_type in valid_types:
            with patch("sys.argv", ["validation_cli.py", "file", "test.py", file_type]):
                with patch(
                    "cursus.cli.validation_cli.validate_file_name", return_value=0
                ):
                    try:
                        result = main()
                        self.assertEqual(result, 0)
                    except SystemExit:
                        self.fail(f"File type '{file_type}' should be valid")

    def test_help_message_contains_examples(self):
        """Test that help message contains usage examples."""
        with patch("sys.argv", ["validation_cli.py", "--help"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                try:
                    main()
                except SystemExit:
                    pass  # argparse calls sys.exit after printing help

                output = mock_stdout.getvalue()
                self.assertIn("Examples:", output)
                self.assertIn("python -m cursus.cli.validation_cli registry", output)
                self.assertIn("python -m cursus.cli.validation_cli file", output)
                self.assertIn("python -m cursus.cli.validation_cli step", output)
                self.assertIn("python -m cursus.cli.validation_cli logical", output)


if __name__ == "__main__":
    unittest.main()
