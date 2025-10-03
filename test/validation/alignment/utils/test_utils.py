"""
Test module for validation alignment utilities.

Tests the utility functions used throughout the validation alignment system
including file operations, path resolution, and helper functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import os

from cursus.validation.alignment.utils.utils import (
    normalize_path,
    extract_logical_name_from_path,
    is_sagemaker_path,
    format_alignment_issue,
    group_issues_by_severity,
    get_highest_severity,
    validate_environment_setup,
    get_validation_summary_stats
)
from cursus.validation.alignment.utils.validation_models import ValidationIssue, IssueLevel


class TestValidationAlignmentUtils:
    """Test cases for validation alignment utility functions."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workspace structure
            workspace_path = Path(temp_dir) / "test_workspace"
            workspace_path.mkdir()
            
            # Create step builder files
            step_dir = workspace_path / "steps"
            step_dir.mkdir()
            
            # Create a sample step builder file
            builder_file = step_dir / "processing_step_builder.py"
            builder_content = '''
class ProcessingStepBuilder:
    def validate_configuration(self):
        pass
    
    def _get_inputs(self):
        return {}
    
    def create_step(self):
        pass
    
    def _create_processor(self):
        return {"processor_type": "ScriptProcessor"}
'''
            builder_file.write_text(builder_content)
            
            yield workspace_path

    def test_normalize_path(self):
        """Test normalize_path function."""
        test_cases = [
            ("/path/to/file.py", "/path/to/file.py"),
            ("relative/path/file.py", "relative/path/file.py"),
            ("/path//double//slash.py", "/path/double/slash.py"),
            ("/path/./current/dir.py", "/path/current/dir.py"),
            ("/path/../parent/dir.py", "/parent/dir.py"),
            ("", ""),
        ]
        
        for input_path, expected_path in test_cases:
            result = normalize_path(input_path)
            assert isinstance(result, str)
            # Basic normalization should occur
            assert "//" not in result or input_path == ""

    def test_extract_logical_name_from_path(self):
        """Test extract_logical_name_from_path function."""
        test_cases = [
            ("/opt/ml/processing/input/training_data", "training_data"),
            ("/opt/ml/processing/output/model_artifacts", "model_artifacts"),
            ("/opt/ml/input/data/simple_file", "simple_file"),
            ("/opt/ml/model/with_underscores_file", "with_underscores_file"),
            ("", None),
            ("/regular/path/file.txt", None),  # Non-SageMaker path
        ]
        
        for input_path, expected_name in test_cases:
            result = extract_logical_name_from_path(input_path)
            if expected_name is None:
                assert result is None
            else:
                assert result == expected_name

    def test_is_sagemaker_path(self):
        """Test is_sagemaker_path function."""
        test_cases = [
            ("/opt/ml/processing/input", True),
            ("/opt/ml/processing/output", True),
            ("/opt/ml/input/data/training", True),
            ("/opt/ml/output/model", True),
            ("/opt/ml/model", True),
            ("/opt/ml/code", True),
            ("/regular/local/path", False),
            ("relative/path", False),
            ("", False),
            ("s3://bucket/path/file.csv", False),  # S3 paths are not SageMaker container paths
        ]
        
        for path, expected_result in test_cases:
            result = is_sagemaker_path(path)
            assert result == expected_result

    def test_format_alignment_issue(self):
        """Test format_alignment_issue function."""
        # Create a sample ValidationIssue
        issue = ValidationIssue(
            level=IssueLevel.ERROR,
            message="Test error message",
            details={"step_name": "test_step", "method": "test_method"}
        )
        
        formatted = format_alignment_issue(issue)
        assert isinstance(formatted, str)
        assert "ERROR" in formatted
        assert "Test error message" in formatted

    def test_group_issues_by_severity(self):
        """Test group_issues_by_severity function."""
        issues = [
            ValidationIssue(level=IssueLevel.ERROR, message="Error 1"),
            ValidationIssue(level=IssueLevel.WARNING, message="Warning 1"),
            ValidationIssue(level=IssueLevel.ERROR, message="Error 2"),
            ValidationIssue(level=IssueLevel.INFO, message="Info 1"),
        ]
        
        grouped = group_issues_by_severity(issues)
        assert isinstance(grouped, dict)
        assert IssueLevel.ERROR in grouped
        assert IssueLevel.WARNING in grouped
        assert IssueLevel.INFO in grouped
        assert len(grouped[IssueLevel.ERROR]) == 2
        assert len(grouped[IssueLevel.WARNING]) == 1
        assert len(grouped[IssueLevel.INFO]) == 1

    def test_get_highest_severity(self):
        """Test get_highest_severity function."""
        # Test with mixed severity issues
        issues = [
            ValidationIssue(level=IssueLevel.WARNING, message="Warning"),
            ValidationIssue(level=IssueLevel.ERROR, message="Error"),
            ValidationIssue(level=IssueLevel.INFO, message="Info"),
        ]
        
        highest = get_highest_severity(issues)
        assert highest == IssueLevel.ERROR
        
        # Test with empty list
        highest_empty = get_highest_severity([])
        assert highest_empty is None
        
        # Test with single issue
        single_issue = [ValidationIssue(level=IssueLevel.WARNING, message="Warning")]
        highest_single = get_highest_severity(single_issue)
        assert highest_single == IssueLevel.WARNING

    def test_validate_environment_setup(self):
        """Test validate_environment_setup function."""
        issues = validate_environment_setup()
        assert isinstance(issues, list)
        # Should return a list (may be empty if environment is properly set up)
        for issue in issues:
            assert isinstance(issue, str)

    def test_get_validation_summary_stats(self):
        """Test get_validation_summary_stats function."""
        issues = [
            ValidationIssue(level=IssueLevel.ERROR, message="Error 1"),
            ValidationIssue(level=IssueLevel.WARNING, message="Warning 1"),
            ValidationIssue(level=IssueLevel.WARNING, message="Warning 2"),
            ValidationIssue(level=IssueLevel.INFO, message="Info 1"),
        ]
        
        stats = get_validation_summary_stats(issues)
        assert isinstance(stats, dict)
        assert "total_issues" in stats
        assert "by_severity" in stats
        assert "highest_severity" in stats
        assert "has_errors" in stats
        assert "has_errors" in stats
        
        assert stats["total_issues"] == 4
        assert stats["by_severity"]["ERROR"] == 1
        assert stats["by_severity"]["WARNING"] == 2
        assert stats["by_severity"]["INFO"] == 1
        assert stats["highest_severity"] == "ERROR"
        assert stats["has_errors"] == True

    def test_normalize_path_edge_cases(self):
        """Test normalize_path with edge cases."""
        # Test with None input
        result = normalize_path(None)
        assert result == "" or result is None
        
        # Test with very long path
        long_path = "/very/long/path/" + "subdir/" * 50 + "file.txt"
        result = normalize_path(long_path)
        assert isinstance(result, str)

    def test_extract_logical_name_edge_cases(self):
        """Test extract_logical_name_from_path with edge cases."""
        # Test with path containing no file extension
        result = extract_logical_name_from_path("/path/to/filename")
        assert result == "filename" or result is None
        
        # Test with path containing multiple dots
        result = extract_logical_name_from_path("/path/to/file.name.with.dots.txt")
        assert isinstance(result, str) or result is None

    def test_is_sagemaker_path_edge_cases(self):
        """Test is_sagemaker_path with edge cases."""
        # Test with None input
        result = is_sagemaker_path(None)
        assert result is False
        
        # Test with partial SageMaker paths
        partial_paths = [
            "/opt/ml",
            "s3://",
            "/opt/ml/",
            "s3://bucket",
        ]
        
        for path in partial_paths:
            result = is_sagemaker_path(path)
            assert isinstance(result, bool)

    def test_format_alignment_issue_edge_cases(self):
        """Test format_alignment_issue with edge cases."""
        # Test with minimal issue
        minimal_issue = ValidationIssue(
            level=IssueLevel.INFO,
            message="Minimal message"
        )
        
        formatted = format_alignment_issue(minimal_issue)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_group_issues_by_severity_edge_cases(self):
        """Test group_issues_by_severity with edge cases."""
        # Test with empty list
        grouped_empty = group_issues_by_severity([])
        assert isinstance(grouped_empty, dict)
        assert len(grouped_empty) == 3  # ERROR, WARNING, INFO
        # All groups should be empty
        for level in IssueLevel:
            assert len(grouped_empty[level]) == 0
        
        # Test with single issue
        single_issue = [ValidationIssue(level=IssueLevel.ERROR, message="Single error")]
        grouped_single = group_issues_by_severity(single_issue)
        assert isinstance(grouped_single, dict)
        assert IssueLevel.ERROR in grouped_single
        assert len(grouped_single[IssueLevel.ERROR]) == 1
        assert len(grouped_single[IssueLevel.WARNING]) == 0
        assert len(grouped_single[IssueLevel.INFO]) == 0
        # CRITICAL level has been removed - no longer testing for it

    def test_get_validation_summary_stats_edge_cases(self):
        """Test get_validation_summary_stats with edge cases."""
        # Test with empty list
        stats_empty = get_validation_summary_stats([])
        assert isinstance(stats_empty, dict)
        assert stats_empty["total_issues"] == 0
        assert stats_empty["by_severity"]["ERROR"] == 0
        assert stats_empty["by_severity"]["WARNING"] == 0
        assert stats_empty["by_severity"]["INFO"] == 0
        assert stats_empty["highest_severity"] is None
        assert stats_empty["has_errors"] == False
        
        # Test with only one type of issue
        error_only = [ValidationIssue(level=IssueLevel.ERROR, message="Error")]
        stats_error = get_validation_summary_stats(error_only)
        assert stats_error["total_issues"] == 1
        assert stats_error["by_severity"]["ERROR"] == 1
        assert stats_error["by_severity"]["WARNING"] == 0
        assert stats_error["by_severity"]["INFO"] == 0
        assert stats_error["highest_severity"] == "ERROR"
        assert stats_error["has_errors"] == True

    def test_validate_environment_setup_integration(self):
        """Test validate_environment_setup integration."""
        issues = validate_environment_setup()
        assert isinstance(issues, list)
        
        # All issues should be strings
        for issue in issues:
            assert isinstance(issue, str)
            assert len(issue) > 0

    def test_utility_functions_error_handling(self):
        """Test that utility functions handle errors gracefully."""
        # Test functions with invalid inputs
        try:
            normalize_path(123)  # Invalid type
        except (TypeError, AttributeError):
            pass  # Expected to handle gracefully or raise appropriate error
        
        try:
            extract_logical_name_from_path(123)  # Invalid type
        except (TypeError, AttributeError):
            pass  # Expected to handle gracefully or raise appropriate error
        
        try:
            is_sagemaker_path(123)  # Invalid type
        except (TypeError, AttributeError):
            pass  # Expected to handle gracefully or raise appropriate error
