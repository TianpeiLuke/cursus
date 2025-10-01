"""
Test suite for UnifiedAlignmentTester level validation.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_reporter import ValidationResult
from cursus.validation.alignment.utils import (
    AlignmentIssue,
    SeverityLevel,
    AlignmentLevel,
)


@pytest.fixture
def tester():
    """Set up UnifiedAlignmentTester fixture."""
    return UnifiedAlignmentTester()


class TestLevelValidation:
    """Test individual level validation in UnifiedAlignmentTester."""

    def test_run_level_validation_level1(self, tester):
        """Test running level 1 validation."""
        with patch.object(tester, "_run_level1_validation") as mock_level1:
            mock_level1.return_value = None

            report = tester.run_level_validation(1, ["test_script"])

            mock_level1.assert_called_once_with(["test_script"])
            assert report is not None

    def test_run_level_validation_level2(self, tester):
        """Test running level 2 validation."""
        with patch.object(tester, "_run_level2_validation") as mock_level2:
            mock_level2.return_value = None

            report = tester.run_level_validation(2, ["test_script"])

            mock_level2.assert_called_once_with(["test_script"])
            assert report is not None

    def test_run_level_validation_level3(self, tester):
        """Test running level 3 validation."""
        with patch.object(tester, "_run_level3_validation") as mock_level3:
            mock_level3.return_value = None

            report = tester.run_level_validation(3, ["test_script"])

            mock_level3.assert_called_once_with(["test_script"])
            assert report is not None

    def test_run_level_validation_level4(self, tester):
        """Test running level 4 validation."""
        with patch.object(tester, "_run_level4_validation") as mock_level4:
            mock_level4.return_value = None

            report = tester.run_level_validation(4, ["test_script"])

            mock_level4.assert_called_once_with(["test_script"])
            assert report is not None

    def test_run_level_validation_invalid_level(self, tester):
        """Test running validation with invalid level."""
        with pytest.raises(ValueError) as exc_info:
            tester.run_level_validation(5)

        assert "Invalid alignment level: 5" in str(exc_info.value)

    def test_level1_validation_with_mock_results(self, tester):
        """Test level 1 validation with mocked results."""
        mock_results = {"test_script": {"passed": True, "issues": []}}

        with patch.object(
            tester.level1_tester, "validate_all_scripts"
        ) as mock_validate:
            mock_validate.return_value = mock_results

            tester._run_level1_validation(["test_script"])

            mock_validate.assert_called_once_with(["test_script"])
            assert len(tester.report.level1_results) == 1
            assert "test_script" in tester.report.level1_results

    def test_level1_validation_with_issues(self, tester):
        """Test level 1 validation with issues."""
        mock_results = {
            "test_script": {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "path_usage",
                        "message": "Undeclared path usage",
                        "details": {"path": "/opt/ml/input"},
                        "recommendation": "Add path to contract",
                    }
                ],
            }
        }

        with patch.object(
            tester.level1_tester, "validate_all_scripts"
        ) as mock_validate:
            mock_validate.return_value = mock_results

            tester._run_level1_validation(["test_script"])

            result = tester.report.level1_results["test_script"]
            assert result.passed is False
            assert len(result.issues) == 1
            assert result.issues[0].level == SeverityLevel.ERROR

    def test_level1_validation_with_exception(self, tester):
        """Test level 1 validation with exception."""
        with patch.object(
            tester.level1_tester, "validate_all_scripts"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            tester._run_level1_validation(["test_script"])

            # Should create error result
            assert "validation_error" in tester.report.level1_results
            error_result = tester.report.level1_results["validation_error"]
            assert error_result.passed is False
            assert len(error_result.issues) > 0

    def test_validate_specific_script(self, tester):
        """Test validating a specific script across all levels."""
        # Mock all level testers
        with patch.object(
            tester.level1_tester, "validate_script"
        ) as mock_l1, patch.object(
            tester.level2_tester, "validate_contract"
        ) as mock_l2, patch.object(
            tester.level3_tester, "validate_specification"
        ) as mock_l3, patch.object(
            tester.level4_tester, "validate_builder"
        ) as mock_l4:

            # Set up mock returns
            mock_l1.return_value = {"passed": True, "issues": []}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}

            result = tester.validate_specific_script("test_script")

            assert result["script_name"] == "test_script"
            assert result["overall_status"] == "PASSING"
            assert result["level1"]["passed"] is True
            assert result["level2"]["passed"] is True
            assert result["level3"]["passed"] is True
            assert result["level4"]["passed"] is True

            # Verify scoring information is included (new functionality)
            assert "scoring" in result
            assert "overall_score" in result["scoring"]
            assert "quality_rating" in result["scoring"]
            assert "level_scores" in result["scoring"]

    def test_validate_specific_script_with_failures(self, tester):
        """Test validating a specific script with some failures."""
        with patch.object(
            tester.level1_tester, "validate_script"
        ) as mock_l1, patch.object(
            tester.level2_tester, "validate_contract"
        ) as mock_l2, patch.object(
            tester.level3_tester, "validate_specification"
        ) as mock_l3, patch.object(
            tester.level4_tester, "validate_builder"
        ) as mock_l4:

            # Set up mock returns with some failures
            mock_l1.return_value = {"passed": False, "issues": [{"severity": "ERROR"}]}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}

            result = tester.validate_specific_script("test_script")

            assert result["overall_status"] == "FAILING"
            assert result["level1"]["passed"] is False

    def test_validate_specific_script_with_exception(self, tester):
        """Test validating a specific script when an exception occurs."""
        with patch.object(tester.level1_tester, "validate_script") as mock_l1:
            mock_l1.side_effect = Exception("Validation error")

            result = tester.validate_specific_script("test_script")

            assert result["overall_status"] == "ERROR"
            assert "error" in result

    def test_level3_validation_mode_configuration(self):
        """Test that Level 3 validation mode is properly configured."""
        # Test strict mode
        strict_tester = UnifiedAlignmentTester(level3_validation_mode="strict")
        assert strict_tester.level3_config is not None

        # Test relaxed mode (default)
        relaxed_tester = UnifiedAlignmentTester(level3_validation_mode="relaxed")
        assert relaxed_tester.level3_config is not None

        # Test permissive mode
        permissive_tester = UnifiedAlignmentTester(level3_validation_mode="permissive")
        assert permissive_tester.level3_config is not None

        # Test invalid mode (should default to relaxed)
        invalid_tester = UnifiedAlignmentTester(level3_validation_mode="invalid")
        assert invalid_tester.level3_config is not None

    def test_json_serialization_in_level_validation(self, tester):
        """Test that level validation results can be JSON serialized."""
        import json

        # Mock results with complex objects
        mock_results = {
            "test_script": {
                "passed": True,
                "issues": [],
                "complex_data": {
                    "type_object": str,
                    "property_object": property(lambda self: "test"),
                    "nested": {"inner_type": int},
                },
            }
        }

        with patch.object(
            tester.level1_tester, "validate_all_scripts"
        ) as mock_validate:
            mock_validate.return_value = mock_results

            tester._run_level1_validation(["test_script"])

            # Should be able to export to JSON without errors
            json_output = tester.export_report(format="json", generate_chart=False)
            parsed_json = json.loads(json_output)

            assert isinstance(parsed_json, dict)

    def test_cli_compatibility_methods(self, tester):
        """Test methods that are specifically used by the CLI."""
        # Test get_validation_summary
        test_result = ValidationResult(test_name="cli_test", passed=True)
        tester.report.add_level1_result("cli_test", test_result)

        summary = tester.get_validation_summary()

        assert "overall_status" in summary
        assert "total_tests" in summary
        assert "pass_rate" in summary
        assert "level_breakdown" in summary

        # Verify scoring information is included (new functionality)
        assert "scoring" in summary
        assert "overall_score" in summary["scoring"]
        assert "quality_rating" in summary["scoring"]

        # Test get_critical_issues
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="cli_test",
            message="Critical issue for CLI",
            details={"cli": "test"},
        )

        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        tester.report.add_level1_result("critical_test", critical_result)

        critical_issues = tester.get_critical_issues()
        assert len(critical_issues) > 0
        assert critical_issues[0]["level"] == "CRITICAL"

    def test_alignment_status_matrix_with_real_data(self, tester):
        """Test alignment status matrix with realistic validation data."""
        with patch.object(tester, "discover_scripts") as mock_discover:
            mock_discover.return_value = ["payload", "package", "dummy_training"]

            # Add mixed results
            passing_result = ValidationResult(test_name="payload_test", passed=True)
            failing_result = ValidationResult(test_name="package_test", passed=False)

            tester.report.add_level1_result("payload", passing_result)
            tester.report.add_level2_result("payload", failing_result)
            tester.report.add_level3_result("package", passing_result)
            tester.report.add_level4_result("package", failing_result)

            matrix = tester.get_alignment_status_matrix()

            # Verify structure
            assert "payload" in matrix
            assert "package" in matrix
            assert "dummy_training" in matrix

            # Verify status values
            assert matrix["payload"]["level1"] == "PASSING"
            assert matrix["payload"]["level2"] == "FAILING"
            assert matrix["payload"]["level3"] == "UNKNOWN"
            assert matrix["payload"]["level4"] == "UNKNOWN"

            assert matrix["package"]["level1"] == "UNKNOWN"
            assert matrix["package"]["level2"] == "UNKNOWN"
            assert matrix["package"]["level3"] == "PASSING"
            assert matrix["package"]["level4"] == "FAILING"

            # Scripts with no results should be UNKNOWN
            for level in ["level1", "level2", "level3", "level4"]:
                assert matrix["dummy_training"][level] == "UNKNOWN"

    def test_get_validation_summary(self, tester):
        """Test getting validation summary."""
        # Add some mock results to the report
        mock_result = ValidationResult(test_name="test", passed=True)
        tester.report.add_level1_result("test", mock_result)

        summary = tester.get_validation_summary()

        assert "overall_status" in summary
        assert "total_tests" in summary
        assert "pass_rate" in summary
        assert "level_breakdown" in summary

        # Verify scoring information is included (new functionality)
        assert "scoring" in summary
        assert "overall_score" in summary["scoring"]
        assert "quality_rating" in summary["scoring"]

    def test_discover_scripts(self, tester):
        """Test script discovery."""
        # Mock the step catalog's list_available_steps method and get_step_info
        with patch.object(tester.step_catalog, "list_available_steps") as mock_list_steps, \
             patch.object(tester.step_catalog, "get_step_info") as mock_get_step_info:
            
            mock_list_steps.return_value = ["script1", "script2"]
            
            # Mock step info to indicate these steps have script components
            def mock_step_info_side_effect(step_name):
                mock_step_info = MagicMock()
                mock_step_info.file_components = {"script": MagicMock()}
                return mock_step_info
            
            mock_get_step_info.side_effect = mock_step_info_side_effect

            scripts = tester.discover_scripts()

            assert len(scripts) == 2
            assert "script1" in scripts
            assert "script2" in scripts
            mock_list_steps.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
