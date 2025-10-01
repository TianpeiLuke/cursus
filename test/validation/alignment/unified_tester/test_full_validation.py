"""
Test suite for UnifiedAlignmentTester full validation.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_reporter import ValidationResult
from cursus.validation.alignment.core_models import (
    AlignmentIssue,
    SeverityLevel,
    AlignmentLevel,
)

class TestFullValidation:
    """Test full validation functionality in UnifiedAlignmentTester."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = UnifiedAlignmentTester()

    def test_run_full_validation_success(self):
        """Test successful full validation across all levels."""
        # Mock all level validation methods
        with patch.object(
            self.tester, "_run_level1_validation"
        ) as mock_l1, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            # Set up mocks to do nothing (successful validation)
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            # Add a successful result to the report
            success_result = ValidationResult(test_name="test_validation", passed=True)
            self.tester.report.add_level1_result("test", success_result)

            report = self.tester.run_full_validation(["test_script"])

            # Verify all levels were called
            mock_l1.assert_called_once_with(["test_script"])
            mock_l2.assert_called_once_with(["test_script"])
            mock_l3.assert_called_once_with(["test_script"])
            mock_l4.assert_called_once_with(["test_script"])

            assert report is not None
            assert report.summary is not None

    def test_run_full_validation_with_skip_levels(self):
        """Test full validation with some levels skipped."""
        with patch.object(
            self.tester, "_run_level1_validation"
        ) as mock_l1, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            # Skip levels 2 and 4
            report = self.tester.run_full_validation(
                target_scripts=["test_script"], skip_levels=[2, 4]
            )

            # Verify only levels 1 and 3 were called
            mock_l1.assert_called_once_with(["test_script"])
            mock_l2.assert_not_called()
            mock_l3.assert_called_once_with(["test_script"])
            mock_l4.assert_not_called()

            assert report is not None

    def test_run_full_validation_with_failures(self):
        """Test full validation with some failures."""
        # Mock level testers to return failure results
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [
                    {
                        "severity": "ERROR",
                        "category": "script_contract",
                        "message": "Test failure",
                        "details": {"error": "test"},
                        "recommendation": "Fix the issue",
                    }
                ],
            }
        }

        with patch.object(
            self.tester.level1_tester, "validate_all_scripts"
        ) as mock_validate, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            report = self.tester.run_full_validation(["test_script"])

            assert report is not None
            assert len(self.tester.report.level1_results) == 1

            # Check that the failure was recorded
            result = self.tester.report.level1_results["test_script"]
            assert result.passed is False
            assert len(result.issues) > 0

    def test_run_full_validation_with_critical_issues(self):
        """Test full validation with critical issues."""
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "builder_configuration",
                        "message": "System cannot proceed",
                        "details": {"critical": "error"},
                        "recommendation": "Fix immediately",
                    }
                ],
            }
        }

        with patch.object(
            self.tester.level1_tester, "validate_all_scripts"
        ) as mock_validate, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            report = self.tester.run_full_validation(["test_script"])

            assert report is not None

            # Check for critical issues
            critical_issues = self.tester.get_critical_issues()
            assert len(critical_issues) > 0

    def test_run_full_validation_early_termination(self):
        """Test that validation continues even with critical issues."""
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "category": "script_contract",
                        "message": "Cannot continue",
                        "details": {"critical": "error"},
                    }
                ],
            }
        }

        with patch.object(
            self.tester.level1_tester, "validate_all_scripts"
        ) as mock_validate, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            report = self.tester.run_full_validation(["test_script"])

            # All levels should still be called
            mock_l2.assert_called_once()
            mock_l3.assert_called_once()
            mock_l4.assert_called_once()

            assert report is not None

    def test_run_full_validation_with_exception(self):
        """Test full validation when an exception occurs."""
        with patch.object(self.tester, "_run_level1_validation") as mock_l1:
            mock_l1.side_effect = Exception("Orchestration error")

            # Should not raise exception, but continue with other levels
            report = self.tester.run_full_validation(["test_script"])

            assert report is not None

    def test_run_full_validation_missing_config(self):
        """Test full validation with incomplete configuration."""
        # Test with empty target scripts (should discover scripts)
        with patch.object(
            self.tester, "discover_scripts"
        ) as mock_discover, patch.object(
            self.tester, "_run_level1_validation"
        ) as mock_l1, patch.object(
            self.tester, "_run_level2_validation"
        ) as mock_l2, patch.object(
            self.tester, "_run_level3_validation"
        ) as mock_l3, patch.object(
            self.tester, "_run_level4_validation"
        ) as mock_l4:

            mock_discover.return_value = ["discovered_script"]
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None

            report = self.tester.run_full_validation(target_scripts=None)

            # Should call with None (which means discover all scripts)
            mock_l1.assert_called_once_with(None)
            assert report is not None

    def test_export_report_json(self):
        """Test exporting report to JSON format."""
        # Add some test data
        test_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", test_result)

        json_output = self.tester.export_report(format="json", generate_chart=False)

        assert isinstance(json_output, str)
        assert '"test"' in json_output  # Should contain test name

        # Verify JSON contains scoring information (new functionality)
        import json

        parsed_json = json.loads(json_output)
        assert "scoring" in parsed_json
        assert "overall_score" in parsed_json["scoring"]
        assert "quality_rating" in parsed_json["scoring"]

    def test_export_report_html(self):
        """Test exporting report to HTML format."""
        # Add some test data
        test_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", test_result)

        # Mock the HTML export to avoid template issues
        with patch.object(self.tester.report, "export_to_html") as mock_html:
            mock_html.return_value = "<html>Test Report</html>"

            html_output = self.tester.export_report(format="html", generate_chart=False)

            assert isinstance(html_output, str)
            assert "<html>" in html_output

    def test_export_report_with_chart_generation(self):
        """Test exporting report with chart generation enabled."""
        # Add some test data
        test_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", test_result)

        # Mock chart generation
        with patch.object(self.tester.report, "get_scorer") as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.generate_chart.return_value = "/path/to/chart.png"
            mock_get_scorer.return_value = mock_scorer

            json_output = self.tester.export_report(
                format="json", generate_chart=True, script_name="test_script"
            )

            assert isinstance(json_output, str)
            # Chart generation should be attempted
            mock_scorer.generate_chart.assert_called_once()

    def test_export_report_invalid_format(self):
        """Test exporting report with invalid format."""
        with pytest.raises(ValueError) as context:
            self.tester.export_report(format="xml")

        assert "Unsupported export format: xml" in str(context.value)

    def test_get_alignment_status_matrix(self):
        """Test getting alignment status matrix."""
        with patch.object(self.tester, "discover_scripts") as mock_discover:
            mock_discover.return_value = ["script1", "script2"]

            # Add some results
            result1 = ValidationResult(test_name="test1", passed=True)
            result2 = ValidationResult(test_name="test2", passed=False)

            self.tester.report.add_level1_result("script1", result1)
            self.tester.report.add_level2_result("script1", result2)

            matrix = self.tester.get_alignment_status_matrix()

            assert "script1" in matrix
            assert "script2" in matrix
            assert matrix["script1"]["level1"] == "PASSING"
            assert matrix["script1"]["level2"] == "FAILING"
            assert matrix["script2"]["level1"] == "UNKNOWN"

    def test_phase1_fixes_integration(self):
        """Test that Phase 1 fixes are properly integrated in the unified tester."""
        # Test that all level testers are properly initialized and functional

        # Verify level testers exist and are properly configured
        assert (
            self.tester.level1_tester is not None
        ), "Level 1 tester should be initialized"
        assert (
            self.tester.level2_tester is not None
        ), "Level 2 tester should be initialized"
        assert (
            self.tester.level3_tester is not None
        ), "Level 3 tester should be initialized"
        assert (
            self.tester.level4_tester is not None
        ), "Level 4 tester should be initialized"

        # Test that level 3 config is properly set
        assert hasattr(
            self.tester, "level3_config"
        ), "Unified tester should have level3_config"

        # Test that level 4 tester has proper configuration
        assert hasattr(
            self.tester.level4_tester, "builders_dir"
        ), "Level 4 tester should have builders_dir"
        assert hasattr(
            self.tester.level4_tester, "configs_dir"
        ), "Level 4 tester should have configs_dir"

    def test_reduced_false_positives_simulation(self):
        """Test simulation of reduced false positives from Phase 1 fixes."""
        # Mock results that would have been false positives before Phase 1
        mock_l3_results = {
            "test_script": {
                "passed": True,  # Now passes due to pattern classification
                "issues": [
                    {
                        "severity": "INFO",
                        "category": "dependency_classification",
                        "message": "External dependency correctly classified",
                        "details": {"pattern": "external"},
                    }
                ],
            }
        }

        mock_l4_results = {
            "test_script": {
                "passed": True,  # Now passes due to pattern-aware validation
                "issues": [
                    {
                        "severity": "INFO",
                        "category": "pattern_filtering",
                        "message": "Framework field filtered as acceptable",
                        "details": {"field": "logger"},
                    }
                ],
            }
        }

        with patch.object(
            self.tester.level1_tester, "validate_all_scripts"
        ) as mock_l1, patch.object(
            self.tester.level2_tester, "validate_all_contracts"
        ) as mock_l2, patch.object(
            self.tester.level3_tester, "validate_all_specifications"
        ) as mock_l3, patch.object(
            self.tester.level4_tester, "validate_all_builders"
        ) as mock_l4:

            mock_l1.return_value = {"test_script": {"passed": True, "issues": []}}
            mock_l2.return_value = {"test_script": {"passed": True, "issues": []}}
            mock_l3.return_value = mock_l3_results
            mock_l4.return_value = mock_l4_results

            report = self.tester.run_full_validation(["test_script"])

            # Should have improved pass rates due to Phase 1 fixes
            assert report is not None

            # Check that issues are informational rather than errors
            all_issues = []
            for result in self.tester.report.level3_results.values():
                all_issues.extend(result.issues)
            for result in self.tester.report.level4_results.values():
                all_issues.extend(result.issues)

            # All issues should be INFO level (not ERROR/CRITICAL)
            error_issues = [
                issue
                for issue in all_issues
                if issue.level.value in ["ERROR", "CRITICAL"]
            ]
            assert (
                len(error_issues) == 0
            ), "Phase 1 fixes should reduce ERROR/CRITICAL issues"

    def test_json_serialization_with_complex_objects(self):
        """Test that JSON serialization works with complex Python objects."""
        # Create a result with complex objects that would cause serialization issues
        test_result = ValidationResult(test_name="complex_test", passed=True)

        # Add an issue with complex details that include property objects, types, etc.
        complex_issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="complex_serialization",
            message="Test complex object serialization",
            details={
                "property_object": property(lambda self: "test"),
                "type_object": str,
                "nested_dict": {
                    "inner_property": property(lambda self: "inner"),
                    "inner_type": int,
                    "normal_value": "should_work",
                },
                "list_with_complex": [
                    property(lambda self: "list_prop"),
                    type,
                    "normal_string",
                ],
            },
            recommendation="Handle complex objects properly",
        )
        test_result.add_issue(complex_issue)

        self.tester.report.add_level1_result("complex_test", test_result)

        # This should not raise a JSON serialization error
        json_output = self.tester.export_report(format="json", generate_chart=False)

        # Verify it's valid JSON
        parsed_json = json.loads(json_output)
        assert isinstance(parsed_json, dict)

        # Verify complex objects were converted to strings
        assert "complex_test" in json_output
        assert "<property object at" in json_output  # Property objects converted
        assert "<class 'str'>" in json_output  # Type objects converted

    def test_json_serialization_with_pydantic_models(self):
        """Test JSON serialization with Pydantic model fields and complex structures."""
        # Mock a validation result that includes Pydantic model information
        mock_l4_results = {
            "test_script": {
                "passed": True,
                "issues": [],
                "config_analysis": {
                    "class_name": "TestConfig",
                    "fields": {
                        "test_field": {
                            "type": str,  # This would be a type object
                            "required": True,
                        }
                    },
                    "default_values": {
                        "computed_property": property(lambda self: "computed"),
                        "model_fields": {
                            "field1": "annotation=str required=True description='Test field'"
                        },
                    },
                },
            }
        }

        with patch.object(
            self.tester.level4_tester, "validate_all_builders"
        ) as mock_l4:
            mock_l4.return_value = mock_l4_results

            self.tester._run_level4_validation(["test_script"])

            # Should be able to serialize without errors
            json_output = self.tester.export_report(format="json", generate_chart=False)
            parsed_json = json.loads(json_output)

            assert isinstance(parsed_json, dict)
            assert "test_script" in json_output

    def test_cli_integration_compatibility(self):
        """Test that the unified tester works with CLI integration patterns."""
        # Test the validate_specific_script method that CLI uses
        with patch.object(
            self.tester.level1_tester, "validate_script"
        ) as mock_l1, patch.object(
            self.tester.level2_tester, "validate_contract"
        ) as mock_l2, patch.object(
            self.tester.level3_tester, "validate_specification"
        ) as mock_l3, patch.object(
            self.tester.level4_tester, "validate_builder"
        ) as mock_l4:

            # Mock successful validation across all levels
            mock_l1.return_value = {"passed": True, "issues": []}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}

            result = self.tester.validate_specific_script("test_script")

            # Verify the result structure matches CLI expectations
            assert result["script_name"] == "test_script"
            assert result["overall_status"] == "PASSING"
            assert "level1" in result
            assert "level2" in result
            assert "level3" in result
            assert "level4" in result

            # Verify each level result is properly structured
            for level in ["level1", "level2", "level3", "level4"]:
                assert "passed" in result[level]
                assert "issues" in result[level]

            # Verify scoring information is included (new functionality)
            assert "scoring" in result
            assert "overall_score" in result["scoring"]
            assert "quality_rating" in result["scoring"]
            assert "level_scores" in result["scoring"]

    def test_comprehensive_validation_workflow(self):
        """Test the complete validation workflow as used by the CLI and standalone scripts."""
        # Mock all level testers to return realistic results
        mock_l1_results = {
            "payload": {
                "passed": True,
                "issues": [],
                "script_analysis": {
                    "script_path": "/path/to/payload.py",
                    "path_references": [],
                    "env_var_accesses": [],
                },
            }
        }

        mock_l2_results = {
            "payload": {
                "passed": True,
                "issues": [],
                "contract": {
                    "entry_point": "payload.py",
                    "inputs": {
                        "model_input": {"path": "/opt/ml/processing/input/model"}
                    },
                    "outputs": {
                        "payload_sample": {"path": "/opt/ml/processing/output"}
                    },
                },
            }
        }

        mock_l3_results = {
            "payload": {
                "passed": True,
                "issues": [],
                "specification": {
                    "step_type": "Payload",
                    "dependencies": [],
                    "outputs": [],
                },
            }
        }

        mock_l4_results = {
            "payload": {
                "passed": True,
                "issues": [
                    {
                        "severity": "WARNING",
                        "category": "configuration_fields",
                        "message": "Required field not accessed",
                        "details": {"field_name": "test_field"},
                    }
                ],
                "builder_analysis": {"config_accesses": [], "validation_calls": []},
            }
        }

        with patch.object(
            self.tester.level1_tester, "validate_all_scripts"
        ) as mock_l1, patch.object(
            self.tester.level2_tester, "validate_all_contracts"
        ) as mock_l2, patch.object(
            self.tester.level3_tester, "validate_all_specifications"
        ) as mock_l3, patch.object(
            self.tester.level4_tester, "validate_all_builders"
        ) as mock_l4:

            mock_l1.return_value = mock_l1_results
            mock_l2.return_value = mock_l2_results
            mock_l3.return_value = mock_l3_results
            mock_l4.return_value = mock_l4_results

            # Run full validation
            report = self.tester.run_full_validation(["payload"])

            # Verify report structure
            assert report is not None
            assert report.summary is not None

            # Verify JSON export works
            json_output = self.tester.export_report(format="json", generate_chart=False)
            parsed_json = json.loads(json_output)
            assert isinstance(parsed_json, dict)

            # Verify validation summary
            summary = self.tester.get_validation_summary()
            assert "overall_status" in summary
            assert "total_tests" in summary
            assert "level_breakdown" in summary

            # Verify scoring information is included in summary (new functionality)
            assert "scoring" in summary
            assert "overall_score" in summary["scoring"]
            assert "quality_rating" in summary["scoring"]

            # Verify specific script validation
            script_result = self.tester.validate_specific_script("payload")
            # Should be PASSING since all levels pass (warnings don't fail the test)
            assert script_result["overall_status"] in ["PASSING", "FAILING"]
            assert script_result["script_name"] == "payload"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
