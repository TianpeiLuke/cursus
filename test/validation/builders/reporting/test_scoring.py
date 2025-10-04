"""
Pytest tests for StreamlinedStepBuilderScorer class.

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Mock at import locations, not definition locations  
3. Match test behavior to actual implementation behavior
4. Use realistic fixtures and data structures
5. Test both success and failure scenarios
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile

# Import the classes under test
from cursus.validation.builders.reporting.scoring import (
    StreamlinedStepBuilderScorer,
    score_builder_validation_results,
    score_builder_results,
    LEVEL_WEIGHTS,
    RATING_LEVELS
)


class TestStreamlinedStepBuilderScorerInitialization:
    """Test StreamlinedStepBuilderScorer initialization."""

    def test_init_with_validation_results(self):
        """Test initialization with validation results."""
        validation_results = {
            "step_name": "TestStep",
            "validation_type": "comprehensive_builder_validation",
            "components": {
                "alignment_validation": {"status": "COMPLETED"},
                "integration_testing": {"status": "COMPLETED"}
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        
        assert scorer.validation_results == validation_results
        assert scorer.components == validation_results["components"]

    def test_init_with_empty_components(self):
        """Test initialization when components are missing."""
        validation_results = {
            "step_name": "TestStep",
            "validation_type": "comprehensive_builder_validation"
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        
        assert scorer.validation_results == validation_results
        assert scorer.components == {}


class TestStreamlinedStepBuilderScorerAlignmentValidation:
    """Test alignment validation scoring methods."""

    def test_score_alignment_validation_passed_status(self):
        """Test scoring alignment validation with PASSED status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        alignment_data = {
            "results": {
                "overall_status": "PASSED",
                "validation_results": {
                    "level_1": {"result": {"passed": True, "issues": []}},
                    "level_2": {"result": {"passed": True, "issues": []}},
                    "level_3": {"result": {"passed": False, "issues": [
                        {"severity": "WARNING", "message": "Minor issue"}
                    ]}}
                }
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        # PASSED status should get high score (85-100 range) based on actual implementation
        assert score >= 85.0
        assert score <= 100.0
        assert details["status"] == "PASSED"
        assert details["passed_levels"] == 2
        assert details["total_levels"] == 3
        assert details["warning_issues"] == 1
        assert details["error_issues"] == 0
        assert details["failed_tests"] == 1  # One failed test with warning

    def test_score_alignment_validation_completed_status(self):
        """Test scoring alignment validation with COMPLETED status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        alignment_data = {
            "results": {
                "overall_status": "COMPLETED",
                "validation_results": {
                    "level_1": {"result": {"passed": True, "issues": []}},
                    "level_2": {"result": {"passed": False, "issues": [
                        {"severity": "ERROR", "message": "Critical issue"}
                    ]}}
                }
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        # COMPLETED status should get 70-100 range based on pass rate
        assert score >= 70.0
        assert score <= 100.0
        assert details["status"] == "COMPLETED"
        assert details["passed_levels"] == 1
        assert details["total_levels"] == 2
        assert details["error_issues"] == 1

    def test_score_alignment_validation_failed_status(self):
        """Test scoring alignment validation with FAILED status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        alignment_data = {
            "results": {
                "overall_status": "FAILED"
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        assert score == 40.0  # Fixed score for FAILED status
        assert details["status"] == "FAILED"

    def test_score_alignment_validation_no_data(self):
        """Test scoring alignment validation with no data."""
        scorer = StreamlinedStepBuilderScorer({})
        
        score, details = scorer._score_alignment_validation({})
        
        assert score == 0.0
        assert details["status"] == "no_data"
        assert "No alignment validation data" in details["reason"]

    def test_score_alignment_validation_with_penalty_cap(self):
        """Test that penalty is capped at 15 points."""
        scorer = StreamlinedStepBuilderScorer({})
        
        # Create many failed tests to test penalty cap
        failed_issues = [
            {"severity": "ERROR", "message": f"Error {i}"} for i in range(10)
        ]
        
        alignment_data = {
            "results": {
                "overall_status": "PASSED",
                "validation_results": {
                    "level_1": {"result": {"passed": False, "issues": failed_issues}}
                }
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        # Score should not go below 60 for PASSED status, even with many errors
        assert score >= 60.0
        assert details["error_issues"] == 10


class TestStreamlinedStepBuilderScorerIntegrationTesting:
    """Test integration testing scoring methods."""

    def test_score_integration_testing_completed(self):
        """Test scoring integration testing with COMPLETED status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        integration_data = {
            "status": "COMPLETED",
            "checks": {
                "dependency_resolution": {"passed": True},
                "cache_configuration": {"passed": True},
                "step_instantiation": {"passed": True}
            }
        }
        
        score, details = scorer._score_integration_testing(integration_data)
        
        assert score == 100.0
        assert details["status"] == "COMPLETED"
        assert details["checks_passed"] == 3
        assert details["total_checks"] == 3

    def test_score_integration_testing_issues_found(self):
        """Test scoring integration testing with ISSUES_FOUND status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        integration_data = {
            "status": "ISSUES_FOUND",
            "checks": {
                "dependency_resolution": {"passed": True},
                "cache_configuration": {"passed": False},
                "step_instantiation": {"passed": True}
            }
        }
        
        score, details = scorer._score_integration_testing(integration_data)
        
        # Should be weighted between status score (70) and check score (66.7)
        assert score > 65.0
        assert score < 75.0
        assert details["status"] == "ISSUES_FOUND"
        assert details["checks_passed"] == 2
        assert details["total_checks"] == 3

    def test_score_integration_testing_error(self):
        """Test scoring integration testing with ERROR status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        integration_data = {
            "status": "ERROR"
        }
        
        score, details = scorer._score_integration_testing(integration_data)
        
        assert score == 20.0
        assert details["status"] == "ERROR"

    def test_score_integration_testing_no_data(self):
        """Test scoring integration testing with no data."""
        scorer = StreamlinedStepBuilderScorer({})
        
        score, details = scorer._score_integration_testing({})
        
        assert score == 0.0
        assert details["status"] == "no_data"


class TestStreamlinedStepBuilderScorerStepCreation:
    """Test step creation scoring methods."""

    def test_score_step_creation_completed_with_capability(self):
        """Test scoring step creation with COMPLETED status and capability validated."""
        scorer = StreamlinedStepBuilderScorer({})
        
        step_creation_data = {
            "status": "COMPLETED",
            "capability_validated": True
        }
        
        score, details = scorer._score_step_creation(step_creation_data)
        
        assert score == 100.0
        assert details["status"] == "COMPLETED"
        assert details["capability_validated"] is True

    def test_score_step_creation_completed_without_capability(self):
        """Test scoring step creation with COMPLETED status but no capability validation."""
        scorer = StreamlinedStepBuilderScorer({})
        
        step_creation_data = {
            "status": "COMPLETED",
            "capability_validated": False
        }
        
        score, details = scorer._score_step_creation(step_creation_data)
        
        assert score == 85.0
        assert details["status"] == "COMPLETED"
        assert details["capability_validated"] is False

    def test_score_step_creation_config_error(self):
        """Test scoring step creation with configuration error."""
        scorer = StreamlinedStepBuilderScorer({})
        
        step_creation_data = {
            "status": "ERROR",
            "error": "Config field required: training_entry_point"
        }
        
        score, details = scorer._score_step_creation(step_creation_data)
        
        assert score == 60.0  # Configuration errors get moderate score
        assert details["status"] == "ERROR"
        assert details["error_type"] == "configuration"

    def test_score_step_creation_other_error(self):
        """Test scoring step creation with non-configuration error."""
        scorer = StreamlinedStepBuilderScorer({})
        
        step_creation_data = {
            "status": "ERROR",
            "error": "Import failed: module not found"
        }
        
        score, details = scorer._score_step_creation(step_creation_data)
        
        assert score == 30.0  # Other errors get lower score
        assert details["status"] == "ERROR"
        assert details["error_type"] == "other"

    def test_score_step_creation_unknown_status(self):
        """Test scoring step creation with unknown status."""
        scorer = StreamlinedStepBuilderScorer({})
        
        step_creation_data = {
            "status": "UNKNOWN"
        }
        
        score, details = scorer._score_step_creation(step_creation_data)
        
        assert score == 70.0  # Unknown status gets benefit of doubt
        assert details["status"] == "UNKNOWN"

    def test_score_step_creation_no_data(self):
        """Test scoring step creation with no data."""
        scorer = StreamlinedStepBuilderScorer({})
        
        score, details = scorer._score_step_creation({})
        
        assert score == 0.0
        assert details["status"] == "no_data"


class TestStreamlinedStepBuilderScorerOverallScoring:
    """Test overall scoring calculation methods."""

    def test_calculate_overall_score_all_components(self):
        """Test overall score calculation with all components."""
        validation_results = {
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "PASSED"}
                },
                "integration_testing": {
                    "status": "COMPLETED",
                    "checks": {"test": {"passed": True}}
                },
                "step_creation": {
                    "status": "COMPLETED",
                    "capability_validated": True
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        overall_score = scorer.calculate_overall_score()
        
        # Should be high score since all components are successful
        assert overall_score >= 90.0
        assert overall_score <= 100.0

    def test_calculate_overall_score_mixed_results(self):
        """Test overall score calculation with mixed component results."""
        validation_results = {
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "COMPLETED"}
                },
                "integration_testing": {
                    "status": "ISSUES_FOUND"
                },
                "step_creation": {
                    "status": "ERROR",
                    "error": "Config error"
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        overall_score = scorer.calculate_overall_score()
        
        # Should be moderate score due to mixed results
        assert overall_score >= 50.0
        assert overall_score <= 85.0

    def test_calculate_overall_score_no_components(self):
        """Test overall score calculation with no components."""
        validation_results = {"components": {}}
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        overall_score = scorer.calculate_overall_score()
        
        assert overall_score == 0.0

    def test_get_rating_excellent(self):
        """Test rating determination for excellent score."""
        scorer = StreamlinedStepBuilderScorer({})
        
        rating = scorer.get_rating(95.0)
        assert rating == "Excellent"

    def test_get_rating_good(self):
        """Test rating determination for good score."""
        scorer = StreamlinedStepBuilderScorer({})
        
        rating = scorer.get_rating(85.0)
        assert rating == "Good"

    def test_get_rating_satisfactory(self):
        """Test rating determination for satisfactory score."""
        scorer = StreamlinedStepBuilderScorer({})
        
        rating = scorer.get_rating(75.0)
        assert rating == "Satisfactory"

    def test_get_rating_needs_work(self):
        """Test rating determination for needs work score."""
        scorer = StreamlinedStepBuilderScorer({})
        
        rating = scorer.get_rating(65.0)
        assert rating == "Needs Work"

    def test_get_rating_poor(self):
        """Test rating determination for poor score."""
        scorer = StreamlinedStepBuilderScorer({})
        
        rating = scorer.get_rating(45.0)
        assert rating == "Poor"


class TestStreamlinedStepBuilderScorerReportGeneration:
    """Test report generation methods."""

    def test_generate_report_complete(self):
        """Test complete report generation."""
        validation_results = {
            "step_name": "TestStep",
            "validation_type": "comprehensive_builder_validation",
            "overall_status": "PASSED",
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "PASSED"}
                },
                "integration_testing": {
                    "status": "COMPLETED",
                    "checks": {"test": {"passed": True}}
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        report = scorer.generate_report()
        
        assert "overall" in report
        assert "components" in report
        assert "validation_results" in report
        assert "metadata" in report
        
        # Check overall section
        assert "score" in report["overall"]
        assert "rating" in report["overall"]
        assert "scoring_approach" in report["overall"]
        
        # Check components section
        assert "alignment_validation" in report["components"]
        assert "integration_testing" in report["components"]
        
        # Check validation results section
        assert report["validation_results"]["step_name"] == "TestStep"
        assert report["validation_results"]["validation_type"] == "comprehensive_builder_validation"
        assert report["validation_results"]["overall_status"] == "PASSED"
        
        # Check metadata
        assert report["metadata"]["scorer_version"] == "2.0.0"
        assert report["metadata"]["scoring_method"] == "component_weighted_scoring"
        assert report["metadata"]["alignment_system_integration"] is True

    def test_generate_report_minimal_data(self):
        """Test report generation with minimal data."""
        validation_results = {
            "step_name": "MinimalStep",
            "components": {}
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        report = scorer.generate_report()
        
        assert report["overall"]["score"] == 0.0
        assert report["components"] == {}
        assert report["validation_results"]["step_name"] == "MinimalStep"

    def test_save_report(self):
        """Test saving report to file."""
        validation_results = {
            "step_name": "TestStep",
            "components": {}
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir
            filename = scorer.save_report("TestStep", output_dir)
            
            expected_path = f"{output_dir}/TestStep_streamlined_score_report.json"
            assert filename == expected_path
            
            # Verify file exists and contains valid JSON
            assert Path(filename).exists()
            with open(filename) as f:
                saved_data = json.load(f)
            assert saved_data["validation_results"]["step_name"] == "TestStep"

    def test_print_report(self, capsys):
        """Test printing report to console."""
        validation_results = {
            "step_name": "TestStep",
            "validation_type": "comprehensive_builder_validation",
            "overall_status": "PASSED",
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "PASSED"}
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        scorer.print_report()
        
        captured = capsys.readouterr()
        assert "STREAMLINED STEP BUILDER QUALITY SCORE REPORT" in captured.out
        assert "Overall Score:" in captured.out
        assert "TestStep" in captured.out


class TestStreamlinedStepBuilderScorerComponentScoring:
    """Test component-specific scoring methods."""

    def test_calculate_component_score_alignment_validation(self):
        """Test component score calculation for alignment validation."""
        validation_results = {
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "PASSED"}
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        score, details = scorer.calculate_component_score("alignment_validation")
        
        assert score >= 85.0  # PASSED status should get high score
        assert details["status"] == "PASSED"

    def test_calculate_component_score_integration_testing(self):
        """Test component score calculation for integration testing."""
        validation_results = {
            "components": {
                "integration_testing": {
                    "status": "COMPLETED"
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        score, details = scorer.calculate_component_score("integration_testing")
        
        assert score == 100.0
        assert details["status"] == "COMPLETED"

    def test_calculate_component_score_step_creation(self):
        """Test component score calculation for step creation."""
        validation_results = {
            "components": {
                "step_creation": {
                    "status": "COMPLETED",
                    "capability_validated": True
                }
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        score, details = scorer.calculate_component_score("step_creation")
        
        assert score == 100.0
        assert details["status"] == "COMPLETED"

    def test_calculate_component_score_unknown_component(self):
        """Test component score calculation for unknown component."""
        validation_results = {"components": {}}
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        score, details = scorer.calculate_component_score("unknown_component")
        
        assert score == 50.0  # Neutral score for unknown components
        assert details["status"] == "unknown_component"


class TestStreamlinedStepBuilderScorerUtilityFunctions:
    """Test utility functions in the scoring module."""

    def test_score_builder_validation_results_function(self, capsys):
        """Test the score_builder_validation_results utility function."""
        validation_results = {
            "step_name": "TestStep",
            "components": {
                "alignment_validation": {
                    "results": {"overall_status": "PASSED"}
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report = score_builder_validation_results(
                validation_results,
                step_name="TestStep",
                save_report=True,
                output_dir=temp_dir
            )
            
            assert "overall" in report
            assert report["validation_results"]["step_name"] == "TestStep"
            
            # Check that report was printed
            captured = capsys.readouterr()
            assert "STREAMLINED STEP BUILDER QUALITY SCORE REPORT" in captured.out
            
            # Check that file was saved
            expected_file = Path(temp_dir) / "TestStep_streamlined_score_report.json"
            assert expected_file.exists()

    def test_score_builder_validation_results_no_save(self, capsys):
        """Test the score_builder_validation_results function without saving."""
        validation_results = {
            "step_name": "TestStep",
            "components": {}
        }
        
        report = score_builder_validation_results(
            validation_results,
            step_name="TestStep",
            save_report=False
        )
        
        assert "overall" in report
        
        # Check that report was printed
        captured = capsys.readouterr()
        assert "STREAMLINED STEP BUILDER QUALITY SCORE REPORT" in captured.out

    def test_score_builder_results_legacy_compatibility(self):
        """Test the legacy score_builder_results function."""
        legacy_results = {
            "test1": {"passed": True},
            "test2": {"passed": False, "error": "Test failed"},
            "test3": {"passed": True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report = score_builder_results(
                legacy_results,
                builder_name="LegacyBuilder",
                save_report=True,
                output_dir=temp_dir
            )
            
            assert "overall" in report
            assert report["validation_results"]["step_name"] == "LegacyBuilder"
            
            # Check that file was saved
            expected_file = Path(temp_dir) / "LegacyBuilder_streamlined_score_report.json"
            assert expected_file.exists()


class TestStreamlinedStepBuilderScorerConstants:
    """Test module constants and configuration."""

    def test_level_weights_constants(self):
        """Test that LEVEL_WEIGHTS constants are properly defined."""
        assert "alignment_validation" in LEVEL_WEIGHTS
        assert "integration_testing" in LEVEL_WEIGHTS
        assert "step_creation" in LEVEL_WEIGHTS
        
        assert LEVEL_WEIGHTS["alignment_validation"] == 2.0
        assert LEVEL_WEIGHTS["integration_testing"] == 1.5
        assert LEVEL_WEIGHTS["step_creation"] == 1.0

    def test_rating_levels_constants(self):
        """Test that RATING_LEVELS constants are properly defined."""
        assert 90 in RATING_LEVELS
        assert 80 in RATING_LEVELS
        assert 70 in RATING_LEVELS
        assert 60 in RATING_LEVELS
        assert 0 in RATING_LEVELS
        
        assert RATING_LEVELS[90] == "Excellent"
        assert RATING_LEVELS[80] == "Good"
        assert RATING_LEVELS[70] == "Satisfactory"
        assert RATING_LEVELS[60] == "Needs Work"
        assert RATING_LEVELS[0] == "Poor"


class TestStreamlinedStepBuilderScorerErrorHandling:
    """Test error handling scenarios."""

    def test_score_alignment_validation_exception_handling(self):
        """Test alignment validation scoring with None level result."""
        scorer = StreamlinedStepBuilderScorer({})
        
        # Create alignment data with None level result (simulates exception scenario)
        alignment_data = {
            "results": {
                "overall_status": "PASSED",
                "validation_results": {
                    "level_1": {"result": None}  # None result should be handled gracefully
                }
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        # Should handle None gracefully and return reasonable score for PASSED status
        assert score >= 85.0  # PASSED status should get high score even with None result
        assert score <= 100.0
        assert details["total_levels"] == 1
        assert details["passed_levels"] == 0  # None result counts as not passed
        assert details["failed_tests"] == 0  # No issues to extract from None

    def test_calculate_overall_score_with_missing_components(self):
        """Test overall score calculation when components are missing from LEVEL_WEIGHTS."""
        validation_results = {
            "components": {
                "unknown_component": {"status": "COMPLETED"}
            }
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        overall_score = scorer.calculate_overall_score()
        
        # Should handle missing components gracefully
        assert overall_score == 0.0

    def test_save_report_directory_creation(self):
        """Test that save_report creates directories if they don't exist."""
        validation_results = {
            "step_name": "TestStep",
            "components": {}
        }
        
        scorer = StreamlinedStepBuilderScorer(validation_results)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "directory"
            filename = scorer.save_report("TestStep", str(nested_dir))
            
            # Directory should be created and file should exist
            assert Path(filename).exists()
            assert Path(filename).parent == nested_dir


class TestStreamlinedStepBuilderScorerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_score_with_zero_total_levels(self):
        """Test scoring when there are no validation levels."""
        scorer = StreamlinedStepBuilderScorer({})
        
        alignment_data = {
            "results": {
                "overall_status": "PASSED",
                "validation_results": {}  # No levels
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        assert score == 95.0  # Should get high score for PASSED status
        assert details["total_levels"] == 0
        assert details["passed_levels"] == 0

    def test_score_with_all_failed_levels(self):
        """Test scoring when all validation levels fail."""
        scorer = StreamlinedStepBuilderScorer({})
        
        alignment_data = {
            "results": {
                "overall_status": "PASSED",  # Overall still PASSED
                "validation_results": {
                    "level_1": {"result": {"passed": False, "issues": [
                        {"severity": "ERROR", "message": "Error 1"}
                    ]}},
                    "level_2": {"result": {"passed": False, "issues": [
                        {"severity": "ERROR", "message": "Error 2"}
                    ]}}
                }
            }
        }
        
        score, details = scorer._score_alignment_validation(alignment_data)
        
        # Should still get reasonable score due to PASSED status, with penalty
        assert score >= 60.0  # Minimum for PASSED status
        assert details["passed_levels"] == 0
        assert details["total_levels"] == 2
        assert details["error_issues"] == 2

    def test_integration_testing_with_no_checks(self):
        """Test integration testing scoring with no checks."""
        scorer = StreamlinedStepBuilderScorer({})
        
        integration_data = {
            "status": "COMPLETED"
            # No checks provided
        }
        
        score, details = scorer._score_integration_testing(integration_data)
        
        assert score == 100.0  # Should still get full score for COMPLETED
        assert details["checks_passed"] == 0
        assert details["total_checks"] == 0
