"""
Test suite for ValidationResult model.
"""

import pytest
from datetime import datetime

from cursus.validation.alignment.alignment_reporter import ValidationResult
from cursus.validation.alignment.alignment_utils import (
    AlignmentIssue,
    SeverityLevel,
    AlignmentLevel,
)


@pytest.fixture
def sample_issue():
    """Set up sample issue fixture."""
    return AlignmentIssue(
        level=SeverityLevel.ERROR,
        category="test_category",
        message="Test issue",
        details={"key": "value"},
    )


class TestValidationResult:
    """Test ValidationResult model."""

    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(test_name="test_script_contract", passed=True)

        assert result.test_name == "test_script_contract"
        assert result.passed is True
        assert len(result.issues) == 0
        assert result.details == {}
        assert isinstance(result.timestamp, datetime)

    def test_add_issue(self, sample_issue):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(test_name="test_with_issues", passed=True)

        # Add a warning issue - should not change passed status
        warning_issue = AlignmentIssue(
            level=SeverityLevel.WARNING, category="test", message="Warning message"
        )
        result.add_issue(warning_issue)

        assert result.passed is True  # Still passing
        assert len(result.issues) == 1

        # Add an error issue - should change passed status
        result.add_issue(sample_issue)

        assert result.passed is False  # Now failing
        assert len(result.issues) == 2

    def test_get_severity_level(self, sample_issue):
        """Test getting highest severity level."""
        result = ValidationResult(test_name="test_severity", passed=True)

        # No issues - should return None
        assert result.get_severity_level() is None

        # Add warning issue
        warning_issue = AlignmentIssue(
            level=SeverityLevel.WARNING, category="test", message="Warning"
        )
        result.add_issue(warning_issue)
        assert result.get_severity_level() == SeverityLevel.WARNING

        # Add error issue - should return ERROR as highest
        result.add_issue(sample_issue)
        assert result.get_severity_level() == SeverityLevel.ERROR

    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        result = ValidationResult(test_name="test_critical", passed=True)

        assert result.has_critical_issues() is False

        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL, category="test", message="Critical issue"
        )
        result.add_issue(critical_issue)

        assert result.has_critical_issues() is True

    def test_has_errors(self, sample_issue):
        """Test checking for error issues."""
        result = ValidationResult(test_name="test_errors", passed=True)

        assert result.has_errors() is False

        # Add error issue
        result.add_issue(sample_issue)

        assert result.has_errors() is True

    def test_to_dict(self, sample_issue):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            test_name="test_dict", passed=False, details={"config": "test_config"}
        )
        result.add_issue(sample_issue)

        result_dict = result.to_dict()

        assert result_dict["test_name"] == "test_dict"
        assert result_dict["passed"] is False
        assert result_dict["details"] == {"config": "test_config"}
        assert len(result_dict["issues"]) == 1
        assert result_dict["severity_level"] == "ERROR"
        assert "timestamp" in result_dict


if __name__ == "__main__":
    pytest.main([__file__])
