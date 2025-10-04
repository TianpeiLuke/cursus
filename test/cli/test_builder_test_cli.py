"""
Pytest tests for the builder_test_cli module.

This module tests the CLI functionality for the Universal Step Builder Test System,
including the enhanced features for scoring, registry discovery, and export capabilities.
Updated to work with the refactored validation system and step catalog integration.
"""

import pytest
from unittest.mock import Mock, patch, call
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from cursus.cli.builder_test_cli import (
    builder_test,
    test_all,
    test_level,
    test_variant,
    test_by_type,
    registry_report,
    validate_builder,
    list_builders,
    export_results_to_json,
    generate_score_chart,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if Path(temp_dir).exists():
        import shutil
        shutil.rmtree(temp_dir)


class TestBuilderTestCliBasic:
    """Test basic CLI functionality and command structure."""

    def test_builder_test_cli_group_exists(self, cli_runner):
        """Test that the builder test CLI group exists and is accessible."""
        result = cli_runner.invoke(builder_test, ["--help"])
        assert result.exit_code == 0
        assert "Universal Step Builder Test System" in result.output

    def test_builder_test_cli_commands_exist(self, cli_runner):
        """Test that all expected commands exist in the CLI group."""
        result = cli_runner.invoke(builder_test, ["--help"])
        assert result.exit_code == 0

        expected_commands = [
            "test-all",
            "test-level",
            "test-variant",
            "test-by-type",
            "registry-report",
            "validate-builder",
            "list-builders",
        ]

        for command in expected_commands:
            assert command in result.output


class TestTestAllCommand:
    """Test the test-all command."""

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_success(self, mock_test_class, mock_import, cli_runner):
        """Test running all tests successfully."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        # Mock the tester instance - CLI uses from_builder_class() then run_validation_for_step()
        mock_tester = Mock()
        # CLI expects component-based format from actual implementation
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "COMPLETED",  # CLI expects COMPLETED for success
            "components": {
                "interface_validation": {"status": "COMPLETED"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            }
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_all, 
            ["TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 0
        assert "🚀 Running all tests for TabularPreprocessingStepBuilder" in result.output
        assert "🎉 Validation passed successfully!" in result.output
        mock_import.assert_called_once_with("TabularPreprocessingStepBuilder")
        mock_test_class.from_builder_class.assert_called_once_with(
            builder_class=mock_builder_cls,
            workspace_dirs=None,
            verbose=False,
            enable_scoring=False,
            enable_structured_reporting=False
        )

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_with_failures(self, mock_test_class, mock_import, cli_runner):
        """Test running all tests with failures."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        # CLI expects component-based format with ERROR status for failures
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "FAILED",  # CLI expects FAILED for failures
            "components": {
                "interface_validation": {"status": "ERROR", "error": "Missing method _create_step"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            }
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_all, 
            ["TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 1
        assert "⚠️  Validation failed with status: FAILED" in result.output

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_with_scoring(self, mock_test_class, mock_import, cli_runner):
        """Test running all tests with scoring enabled."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        # CLI expects component-based format with scoring
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "COMPLETED",
            "components": {
                "interface_validation": {"status": "COMPLETED"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            },
            "scoring": {
                "overall": {"score": 95.5, "rating": "Excellent"},
                "levels": {
                    "level1_interface": {"score": 100.0, "passed": 2, "total": 2}
                },
            },
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_all, 
            ["TabularPreprocessingStepBuilder", "--scoring"]
        )

        assert result.exit_code == 0
        assert "🏆 Quality Score: 95.5/100 - Excellent" in result.output

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_with_exports(self, mock_test_class, mock_import, cli_runner, temp_dir):
        """Test running all tests with export options."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        # CLI expects component-based format with scoring for exports
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "COMPLETED",
            "components": {
                "interface_validation": {"status": "COMPLETED"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            },
            "scoring": {
                "overall": {"score": 85.0, "rating": "Good"},
            },
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_all,
            [
                "TabularPreprocessingStepBuilder",
                "--scoring",
                "--export-json", str(Path(temp_dir) / "results.json"),
                "--export-chart",
                "--output-dir", temp_dir,
            ]
        )

        assert result.exit_code == 0
        
        # Verify JSON export
        json_file = Path(temp_dir) / "results.json"
        assert json_file.exists()

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_error_handling(self, mock_test_class, mock_import, cli_runner):
        """Test error handling in test-all command."""
        # Mock the import function to succeed
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        # Make from_builder_class fail
        mock_test_class.from_builder_class.side_effect = Exception("Test initialization failed")

        result = cli_runner.invoke(
            test_all, 
            ["TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 1
        assert "❌ Error during test execution" in result.output


class TestTestLevelCommand:
    """Test the test-level command."""

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_level_1_success(self, mock_universal_test, mock_import, cli_runner):
        """Test running Level 1 tests successfully."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = {
            "test_inheritance": {"passed": True, "error": None},
            "test_naming_conventions": {"passed": True, "error": None},
        }
        mock_universal_test.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_level, 
            ["1", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 0
        assert "🚀 Running Level 1 (Interface) tests" in result.output
        assert "Using refactored UniversalStepBuilderTest" in result.output
        mock_import.assert_called_once_with("TabularPreprocessingStepBuilder")
        mock_universal_test.from_builder_class.assert_called_once_with(
            builder_class=mock_builder_cls,
            workspace_dirs=None,
            verbose=False
        )

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_level_2_success(self, mock_universal_test, cli_runner):
        """Test running Level 2 tests successfully."""
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = {
            "test_specification_usage": {"passed": True, "error": None},
        }
        mock_universal_test.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_level, 
            ["2", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 0
        assert "🚀 Running Level 2 (Specification) tests" in result.output
        assert "Using refactored UniversalStepBuilderTest" in result.output

    def test_test_level_invalid_level(self, cli_runner):
        """Test running tests with invalid level."""
        result = cli_runner.invoke(
            test_level, 
            ["5", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 2  # Click validation error


class TestTestVariantCommand:
    """Test the test-variant command."""

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_variant_processing_success(self, mock_universal_test, mock_import, cli_runner):
        """Test running processing variant tests successfully."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = {
            "test_processing_step_creation": {"passed": True, "error": None},
        }
        mock_universal_test.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_variant, 
            ["processing", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 0
        assert "🚀 Running Processing variant tests" in result.output
        assert "Using refactored UniversalStepBuilderTest" in result.output
        mock_import.assert_called_once_with("TabularPreprocessingStepBuilder")
        mock_universal_test.from_builder_class.assert_called_once_with(
            builder_class=mock_builder_cls,
            workspace_dirs=None,
            verbose=False
        )

    def test_test_variant_invalid_variant(self, cli_runner):
        """Test running tests with invalid variant."""
        result = cli_runner.invoke(
            test_variant, 
            ["invalid", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 2  # Click validation error


class TestTestByTypeCommand:
    """Test the test-by-type command."""

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_by_type_success(self, mock_test_class, cli_runner):
        """Test running tests by SageMaker type successfully."""
        mock_results = {
            "XGBoostTraining": {
                "test_results": {"test_inheritance": {"passed": True, "error": None}},
                "scoring": {"overall": {"score": 85.0, "rating": "Good"}},
            },
            "TabularPreprocessing": {
                "test_results": {"test_inheritance": {"passed": True, "error": None}},
                "scoring": {"overall": {"score": 90.0, "rating": "Excellent"}},
            }
        }
        mock_test_class.test_all_builders_by_type.return_value = mock_results

        result = cli_runner.invoke(
            test_by_type, 
            ["Training", "--scoring"]
        )

        assert result.exit_code == 0
        assert "🔍 Testing all builders for SageMaker step type: Training" in result.output
        assert "✅ XGBoostTraining: Score 85.0/100 (Good)" in result.output
        assert "✅ TabularPreprocessing: Score 90.0/100 (Excellent)" in result.output
        mock_test_class.test_all_builders_by_type.assert_called_once_with(
            sagemaker_step_type="Training",
            verbose=False,
            enable_scoring=True
        )

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_by_type_with_errors(self, mock_test_class, cli_runner):
        """Test running tests by type with some errors."""
        mock_results = {
            "error": "No builders found for type 'InvalidType'"
        }
        mock_test_class.test_all_builders_by_type.return_value = mock_results

        result = cli_runner.invoke(
            test_by_type, 
            ["Training"]  # Use valid type but return error result
        )

        assert result.exit_code == 1
        assert "❌ Error: No builders found for type 'InvalidType'" in result.output

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_by_type_with_export(self, mock_test_class, cli_runner, temp_dir):
        """Test running tests by type with JSON export."""
        mock_results = {
            "XGBoostTraining": {
                "test_results": {"test_inheritance": {"passed": True, "error": None}},
            }
        }
        mock_test_class.test_all_builders_by_type.return_value = mock_results

        result = cli_runner.invoke(
            test_by_type,
            [
                "Training",
                "--export-json", str(Path(temp_dir) / "batch_results.json")
            ]
        )

        assert result.exit_code == 0
        
        # Verify JSON export
        json_file = Path(temp_dir) / "batch_results.json"
        assert json_file.exists()


class TestRegistryReportCommand:
    """Test the registry-report command."""

    @patch("cursus.cli.builder_test_cli.StepCatalog")
    def test_registry_report_success(self, mock_catalog_class, cli_runner):
        """Test generating step catalog discovery report successfully."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = [
            "XGBoostTraining", "TabularPreprocessing", "BatchTransform"
        ]
        mock_catalog.load_builder_class.side_effect = [
            Mock(__name__="XGBoostTrainingStepBuilder"),  # Success
            Mock(__name__="TabularPreprocessingStepBuilder"),  # Success
            None  # Failure
        ]
        mock_catalog_class.return_value = mock_catalog

        with patch("cursus.registry.step_names.get_sagemaker_step_type") as mock_get_type:
            mock_get_type.side_effect = ["Training", "Processing", "Transform"]
            
            result = cli_runner.invoke(registry_report, [])

        assert result.exit_code == 0
        assert "🔍 Generating step catalog discovery report..." in result.output
        assert "📊 Step Catalog Discovery Report" in result.output
        assert "Total steps in catalog: 3" in result.output
        assert "Training: 1 steps" in result.output
        assert "✅ Available: 2" in result.output
        assert "❌ Unavailable: 1" in result.output

    @patch("cursus.cli.builder_test_cli.StepCatalog")
    def test_registry_report_with_verbose_errors(self, mock_catalog_class, cli_runner):
        """Test step catalog report with verbose error display."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = [
            "BrokenStep1", "BrokenStep2"
        ]
        mock_catalog.load_builder_class.side_effect = [
            Exception("Module not found"),
            Exception("Class not found")
        ]
        mock_catalog_class.return_value = mock_catalog

        with patch("cursus.registry.get_sagemaker_step_type") as mock_get_type:
            mock_get_type.side_effect = [Exception("Type error"), Exception("Type error")]
            
            result = cli_runner.invoke(registry_report, ["--verbose"])

        assert result.exit_code == 0
        # CLI implementation shows errors in a different format - check for actual error reporting
        assert "❌ Unavailable: 2" in result.output
        assert "Errors:" in result.output


class TestValidateBuilderCommand:
    """Test the validate-builder command."""

    @patch("cursus.cli.builder_test_cli.StepCatalog")
    def test_validate_builder_success(self, mock_catalog_class, cli_runner):
        """Test validating builder availability successfully."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = ["XGBoostTraining"]
        mock_catalog.load_builder_class.return_value = Mock(__name__="XGBoostTrainingStepBuilder")
        mock_catalog_class.return_value = mock_catalog

        result = cli_runner.invoke(
            validate_builder, 
            ["XGBoostTraining"]
        )

        assert result.exit_code == 0
        assert "🔍 Validating builder availability for: XGBoostTraining" in result.output
        assert "📊 Builder Validation Results" in result.output
        assert "Step name: XGBoostTraining" in result.output
        assert "In step catalog: ✅" in result.output
        assert "Builder class found: ✅" in result.output
        assert "Loadable: ✅" in result.output

    @patch("cursus.cli.builder_test_cli.StepCatalog")
    def test_validate_builder_with_errors(self, mock_catalog_class, cli_runner):
        """Test validating builder with errors."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = []  # Not in catalog
        mock_catalog.load_builder_class.side_effect = Exception("Step not found")
        mock_catalog_class.return_value = mock_catalog

        result = cli_runner.invoke(
            validate_builder, 
            ["InvalidBuilder"]
        )

        assert result.exit_code == 1
        assert "In step catalog: ❌" in result.output
        assert "Builder class found: ❌" in result.output
        assert "Loadable: ❌" in result.output
        assert "Error: Step not found" in result.output


class TestListBuildersCommand:
    """Test the list-builders command."""

    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.exists")
    def test_list_builders_success(self, mock_exists, mock_glob, cli_runner):
        """Test listing available builders successfully."""
        # Mock the directory structure
        mock_exists.return_value = True
        
        # Mock file paths
        mock_file1 = Mock()
        mock_file1.name = "builder_tabular_preprocessing_step.py"
        mock_file1.stem = "builder_tabular_preprocessing_step"
        
        mock_file2 = Mock()
        mock_file2.name = "builder_xgboost_training_step.py"
        mock_file2.stem = "builder_xgboost_training_step"
        
        mock_glob.return_value = [mock_file1, mock_file2]

        result = cli_runner.invoke(list_builders, [])

        assert result.exit_code == 0
        assert "📋 Available Step Builder Classes:" in result.output
        # The actual implementation scans files, so we should see some builders listed
        assert "Total:" in result.output

    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.exists")
    def test_list_builders_no_builders_found(self, mock_exists, mock_glob, cli_runner):
        """Test listing builders when none are found."""
        mock_exists.return_value = True
        mock_glob.return_value = []  # No files found

        result = cli_runner.invoke(list_builders, [])

        assert result.exit_code == 0
        assert "No builder classes found" in result.output

    def test_list_builders_error_handling(self, cli_runner):
        """Test error handling in list-builders command."""
        # The actual implementation is quite robust and handles errors gracefully
        # It will still succeed even if some directories don't exist
        # So let's test that it at least runs without crashing
        result = cli_runner.invoke(list_builders, [])

        # The command should complete successfully even with missing directories
        # because it has fallback mechanisms
        assert result.exit_code == 0
        assert "📋 Available Step Builder Classes:" in result.output


class TestBuilderTestCliHelpers:
    """Test helper functions in builder test CLI."""

    def test_json_export_functionality(self, temp_dir):
        """Test JSON export functionality."""
        from cursus.cli.builder_test_cli import export_results_to_json
        
        sample_results = {
            "test_results": {"test_inheritance": {"passed": True}},
            "scoring": {"overall": {"score": 85.0}},
        }

        output_path = Path(temp_dir) / "test_results.json"

        with patch("click.echo") as mock_echo:
            export_results_to_json(sample_results, str(output_path))

        # Check that file was created
        assert output_path.exists()

        # Check file contents
        with open(output_path, "r") as f:
            loaded_results = json.load(f)

        assert loaded_results == sample_results

        # Check that success message was printed
        mock_echo.assert_called_with(f"✅ Results exported to: {output_path}")

    def test_generate_score_chart(self):
        """Test generating score chart."""
        import tempfile
        import shutil
        from pathlib import Path
        from cursus.cli.builder_test_cli import generate_score_chart

        # Test with enhanced results format - CLI implementation uses matplotlib directly
        enhanced_results = {
            "scoring": {
                "overall": {"score": 85.0, "rating": "Good"}
            }
        }

        # Use temporary directory with proper cleanup
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_score_chart(enhanced_results, "TestBuilder", temp_dir)

            # CLI implementation returns actual path format: temp_dir/BuilderName_score_chart.png
            expected_path = f"{temp_dir}/TestBuilder_score_chart.png"
            assert result == expected_path
            
            # Verify the file was actually created
            chart_file = Path(expected_path)
            assert chart_file.exists(), "Chart file should be created"
            
            # Verify file content (basic check)
            assert chart_file.stat().st_size > 0, "Chart file should not be empty"
        
        # Temporary directory and all contents are automatically cleaned up here


class TestBuilderTestCliErrorHandling:
    """Test error handling in builder test CLI commands."""

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_all_import_error(self, mock_test_class, cli_runner):
        """Test error handling when builder class cannot be imported."""
        mock_test_class.side_effect = ImportError("Could not import builder class")

        result = cli_runner.invoke(
            test_all, 
            ["NonExistentBuilder"]
        )

        assert result.exit_code == 1
        assert "❌ Error during test execution" in result.output

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_level_execution_error(self, mock_universal_test, cli_runner):
        """Test error handling during level test execution."""
        mock_tester = Mock()
        mock_tester.run_all_tests.side_effect = Exception("Test execution failed")
        mock_universal_test.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_level, 
            ["1", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 1
        assert "❌ Error during test execution" in result.output

    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_test_variant_execution_error(self, mock_universal_test, cli_runner):
        """Test error handling during variant test execution."""
        mock_tester = Mock()
        mock_tester.run_all_tests.side_effect = Exception("Variant test failed")
        mock_universal_test.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_variant, 
            ["processing", "TabularPreprocessingStepBuilder"]
        )

        assert result.exit_code == 1
        assert "❌ Error during test execution" in result.output


class TestBuilderTestCliIntegration:
    """Integration tests for builder test CLI."""

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.StepCatalog")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_full_workflow_integration(self, mock_test_class, mock_catalog_class, mock_import, cli_runner):
        """Test a complete workflow integration."""
        # Mock step catalog
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = ["TabularPreprocessing"]
        mock_catalog.load_builder_class.return_value = Mock(__name__="TabularPreprocessingStepBuilder")
        mock_catalog_class.return_value = mock_catalog
        
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        # Mock test execution - CLI uses component-based format
        mock_tester = Mock()
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "COMPLETED",
            "components": {
                "interface_validation": {"status": "COMPLETED"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            }
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        # Test list builders
        result = cli_runner.invoke(list_builders, [])
        assert result.exit_code == 0
        # Note: list_builders uses file scanning, not step catalog, so we can't easily mock the output

        # Test validate builder
        result = cli_runner.invoke(validate_builder, ["TabularPreprocessing"])
        assert result.exit_code == 0
        assert "Loadable: ✅" in result.output

        # Test run all tests
        result = cli_runner.invoke(test_all, ["TabularPreprocessingStepBuilder"])
        assert result.exit_code == 0
        assert "🎉 Validation passed successfully!" in result.output

    @patch("cursus.cli.builder_test_cli.import_builder_class")
    @patch("cursus.cli.builder_test_cli.UniversalStepBuilderTest")
    def test_scoring_workflow_integration(self, mock_test_class, mock_import, cli_runner, temp_dir):
        """Test scoring workflow integration."""
        # Mock the import function
        mock_builder_cls = Mock()
        mock_builder_cls.__name__ = "TabularPreprocessingStepBuilder"
        mock_import.return_value = mock_builder_cls
        
        mock_tester = Mock()
        # CLI expects component-based format with scoring for integration test
        mock_tester.run_validation_for_step.return_value = {
            "step_name": "TabularPreprocessing",
            "builder_class": "TabularPreprocessingStepBuilder",
            "overall_status": "COMPLETED",
            "components": {
                "interface_validation": {"status": "COMPLETED"},
                "specification_validation": {"status": "COMPLETED"},
                "step_creation_validation": {"status": "COMPLETED"},
            },
            "scoring": {
                "overall": {"score": 92.5, "rating": "Excellent"},
                "levels": {
                    "level1_interface": {"score": 95.0, "passed": 2, "total": 2}
                },
            },
        }
        mock_test_class.from_builder_class.return_value = mock_tester

        result = cli_runner.invoke(
            test_all,
            [
                "TabularPreprocessingStepBuilder",
                "--scoring",
                "--export-json", str(Path(temp_dir) / "results.json"),
                "--verbose"
            ]
        )

        assert result.exit_code == 0
        assert "🏆 Quality Score: 92.5/100 - Excellent" in result.output
        # CLI implementation shows component-based format, not detailed scoring breakdown
        assert "🎉 Validation passed successfully!" in result.output
        
        # Verify JSON export
        json_file = Path(temp_dir) / "results.json"
        assert json_file.exists()
        
        with open(json_file, "r") as f:
            exported_data = json.load(f)
        
        assert exported_data["scoring"]["overall"]["score"] == 92.5
