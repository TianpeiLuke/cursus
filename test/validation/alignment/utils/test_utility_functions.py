"""
Test suite for alignment utility functions.
"""

import pytest

from cursus.validation.alignment.utils.utils import (
    normalize_path,
    is_sagemaker_path,
    extract_logical_name_from_path,
    format_alignment_issue,
    group_issues_by_severity,
    get_highest_severity,
)
from cursus.validation.alignment import (
    create_alignment_issue,
    SeverityLevel,
    AlignmentLevel,
    AlignmentIssue,
)


class TestUtilityFunctions:
    """Test alignment utility functions."""

    def test_normalize_path_basic(self):
        """Test basic path normalization."""
        # Test absolute path
        result = normalize_path("/opt/ml/input/data")
        assert result == "/opt/ml/input/data"

        # Test relative path
        result = normalize_path("./data/train")
        assert result == "data/train"

        # Test path with double slashes
        result = normalize_path("/opt//ml/input//data")
        assert result == "/opt/ml/input/data"

        # Test Windows path normalization
        result = normalize_path("C:\\data\\train")
        assert result == "C:/data/train"

    def test_is_sagemaker_path_positive(self):
        """Test SageMaker path detection - positive cases."""
        sagemaker_paths = [
            "/opt/ml/processing/input/data",
            "/opt/ml/processing/output",
            "/opt/ml/input/data/train",
            "/opt/ml/model",
            "/opt/ml/output/model.tar.gz",
        ]

        for path in sagemaker_paths:
            assert is_sagemaker_path(
                path
            ), f"Path {path} should be detected as SageMaker path"

    def test_is_sagemaker_path_negative(self):
        """Test SageMaker path detection - negative cases."""
        non_sagemaker_paths = [
            "/home/user/data",
            "/tmp/data",
            "/var/log",
            "data/train",
            "/opt/other/path",
        ]

        for path in non_sagemaker_paths:
            assert not is_sagemaker_path(
                path
            ), f"Path {path} should not be detected as SageMaker path"

    def test_extract_logical_name_from_path(self):
        """Test logical name extraction from SageMaker paths."""
        test_cases = [
            ("/opt/ml/processing/input/train", "train"),
            ("/opt/ml/processing/output/model", "model"),
            ("/opt/ml/input/data/validation", "validation"),
            ("/opt/ml/model/artifacts", "artifacts"),
            ("/opt/ml/output/results", "results"),
        ]

        for path, expected in test_cases:
            result = extract_logical_name_from_path(path)
            assert (
                result == expected
            ), f"Expected {expected} for path {path}, got {result}"

    def test_extract_logical_name_from_path_invalid(self):
        """Test logical name extraction from non-SageMaker paths."""
        invalid_paths = [
            "/home/user/data",
            "/tmp/file.txt",
            "relative/path",
            "/opt/ml/processing/input/",  # No logical name part
            "/opt/ml/processing/output/",  # No logical name part
        ]

        for path in invalid_paths:
            result = extract_logical_name_from_path(path)
            assert result is None, f"Expected None for path {path}, got {result}"

    def test_format_alignment_issue(self):
        """Test alignment issue formatting."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test",
            message="Test error message",
            details={"file": "test.py", "line": 42},
            recommendation="Fix the issue",
        )

        formatted = format_alignment_issue(issue)
        assert "ERROR" in formatted
        assert "Test error message" in formatted
        assert "Fix the issue" in formatted

    def test_group_issues_by_severity(self):
        """Test grouping issues by severity level."""
        issues = [
            AlignmentIssue(
                level=SeverityLevel.ERROR, category="test", message="Error 1"
            ),
            AlignmentIssue(
                level=SeverityLevel.WARNING, category="test", message="Warning 1"
            ),
            AlignmentIssue(
                level=SeverityLevel.ERROR, category="test", message="Error 2"
            ),
            AlignmentIssue(level=SeverityLevel.INFO, category="test", message="Info 1"),
        ]

        grouped = group_issues_by_severity(issues)

        assert len(grouped[SeverityLevel.ERROR]) == 2
        assert len(grouped[SeverityLevel.WARNING]) == 1
        assert len(grouped[SeverityLevel.INFO]) == 1
        assert len(grouped[SeverityLevel.CRITICAL]) == 0

    def test_get_highest_severity(self):
        """Test getting highest severity level."""
        # Test with mixed severity levels
        issues = [
            AlignmentIssue(
                level=SeverityLevel.WARNING, category="test", message="Warning"
            ),
            AlignmentIssue(level=SeverityLevel.ERROR, category="test", message="Error"),
            AlignmentIssue(level=SeverityLevel.INFO, category="test", message="Info"),
        ]

        highest = get_highest_severity(issues)
        assert highest == SeverityLevel.ERROR

        # Test with critical issue
        issues.append(
            AlignmentIssue(
                level=SeverityLevel.CRITICAL, category="test", message="Critical"
            )
        )
        highest = get_highest_severity(issues)
        assert highest == SeverityLevel.CRITICAL

        # Test with empty list
        highest = get_highest_severity([])
        assert highest is None

    def test_create_alignment_issue(self):
        """Test alignment issue creation helper."""
        issue = create_alignment_issue(
            level=SeverityLevel.WARNING,
            category="path_validation",
            message="Hardcoded path detected",
            details={"path": "/opt/ml/input", "line": 10},
            recommendation="Use environment variables",
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT,
        )

        assert issue.level == SeverityLevel.WARNING
        assert issue.category == "path_validation"
        assert issue.message == "Hardcoded path detected"
        assert issue.details["path"] == "/opt/ml/input"
        assert issue.recommendation == "Use environment variables"
        assert issue.alignment_level == AlignmentLevel.SCRIPT_CONTRACT

    def test_create_alignment_issue_minimal(self):
        """Test alignment issue creation with minimal parameters."""
        issue = create_alignment_issue(
            level=SeverityLevel.INFO, category="general", message="Info message"
        )

        assert issue.level == SeverityLevel.INFO
        assert issue.category == "general"
        assert issue.message == "Info message"
        assert issue.details == {}
        assert issue.recommendation is None
        assert issue.alignment_level is None


if __name__ == "__main__":
    pytest.main([__file__])
