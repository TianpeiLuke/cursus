"""
Test suite for AlignmentIssue model.
"""

import pytest

from cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)

class TestAlignmentIssue:
    """Test AlignmentIssue model."""
    
    def test_alignment_issue_creation(self):
        """Test basic AlignmentIssue creation."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test_category",
            message="Test issue",
            details={"key": "value"}
        )
        
        assert issue.level == SeverityLevel.ERROR
        assert issue.category == "test_category"
        assert issue.message == "Test issue"
        assert issue.details == {"key": "value"}
        assert issue.recommendation is None
        assert issue.alignment_level is None
    
    def test_alignment_issue_with_recommendation(self):
        """Test AlignmentIssue creation with recommendation."""
        issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="path_validation",
            message="Hardcoded path found",
            details={"path": "/opt/ml/input", "line": 42},
            recommendation="Use environment variables instead"
        )
        
        assert issue.recommendation == "Use environment variables instead"
        assert issue.details["path"] == "/opt/ml/input"
        assert issue.details["line"] == 42
    
    def test_alignment_issue_with_alignment_level(self):
        """Test AlignmentIssue creation with alignment level."""
        issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="script_contract",
            message="Script contract mismatch",
            details={"script": "train.py"},
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
        )
        
        assert issue.alignment_level == AlignmentLevel.SCRIPT_CONTRACT
        assert issue.level == SeverityLevel.CRITICAL
    
    def test_alignment_issue_defaults(self):
        """Test AlignmentIssue default values."""
        issue = AlignmentIssue(
            level=SeverityLevel.INFO,
            category="general",
            message="Info message"
        )
        
        assert issue.details == {}
        assert issue.recommendation is None
        assert issue.alignment_level is None
    
    def test_alignment_issue_string_representation(self):
        """Test AlignmentIssue string representation."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="validation",
            message="Test error message",
            details={"context": "test"}
        )
        
        str_repr = str(issue)
        assert "level=<SeverityLevel.ERROR: 'ERROR'>" in str_repr
        assert "Test error message" in str_repr
    
    def test_alignment_issue_validation(self):
        """Test AlignmentIssue validation with invalid data."""
        # Test with invalid severity level
        with pytest.raises(ValueError):
            AlignmentIssue(
                level="INVALID",  # Should be SeverityLevel enum
                category="test",
                message="Test"
            )
    
    def test_alignment_issue_serialization(self):
        """Test AlignmentIssue serialization to dict."""
        issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="test_category",
            message="Test message",
            details={"key": "value"},
            recommendation="Fix this",
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
        )
        
        issue_dict = issue.model_dump()
        
        assert issue_dict["level"] == SeverityLevel.WARNING
        assert issue_dict["category"] == "test_category"
        assert issue_dict["message"] == "Test message"
        assert issue_dict["details"] == {"key": "value"}
        assert issue_dict["recommendation"] == "Fix this"
        assert issue_dict["alignment_level"] == AlignmentLevel.SCRIPT_CONTRACT
    
    def test_alignment_issue_json_serialization(self):
        """Test AlignmentIssue JSON serialization."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test",
            message="Test message"
        )
        
        json_str = issue.model_dump_json()
        assert isinstance(json_str, str)
        assert "ERROR" in json_str
        assert "test" in json_str
        assert "Test message" in json_str

if __name__ == '__main__':
    pytest.main([__file__])
