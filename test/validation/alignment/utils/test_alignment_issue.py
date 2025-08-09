"""
Test suite for AlignmentIssue model.
"""

import unittest

from src.cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)


class TestAlignmentIssue(unittest.TestCase):
    """Test AlignmentIssue model."""
    
    def test_alignment_issue_creation(self):
        """Test basic AlignmentIssue creation."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test_category",
            message="Test issue",
            details={"key": "value"}
        )
        
        self.assertEqual(issue.level, SeverityLevel.ERROR)
        self.assertEqual(issue.category, "test_category")
        self.assertEqual(issue.message, "Test issue")
        self.assertEqual(issue.details, {"key": "value"})
        self.assertIsNone(issue.recommendation)
        self.assertIsNone(issue.alignment_level)
    
    def test_alignment_issue_with_recommendation(self):
        """Test AlignmentIssue creation with recommendation."""
        issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="path_validation",
            message="Hardcoded path found",
            details={"path": "/opt/ml/input", "line": 42},
            recommendation="Use environment variables instead"
        )
        
        self.assertEqual(issue.recommendation, "Use environment variables instead")
        self.assertEqual(issue.details["path"], "/opt/ml/input")
        self.assertEqual(issue.details["line"], 42)
    
    def test_alignment_issue_with_alignment_level(self):
        """Test AlignmentIssue creation with alignment level."""
        issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="script_contract",
            message="Script contract mismatch",
            details={"script": "train.py"},
            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
        )
        
        self.assertEqual(issue.alignment_level, AlignmentLevel.SCRIPT_CONTRACT)
        self.assertEqual(issue.level, SeverityLevel.CRITICAL)
    
    def test_alignment_issue_defaults(self):
        """Test AlignmentIssue default values."""
        issue = AlignmentIssue(
            level=SeverityLevel.INFO,
            category="general",
            message="Info message"
        )
        
        self.assertEqual(issue.details, {})
        self.assertIsNone(issue.recommendation)
        self.assertIsNone(issue.alignment_level)
    
    def test_alignment_issue_string_representation(self):
        """Test AlignmentIssue string representation."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="validation",
            message="Test error message",
            details={"context": "test"}
        )
        
        str_repr = str(issue)
        self.assertIn("level=<SeverityLevel.ERROR: 'ERROR'>", str_repr)
        self.assertIn("Test error message", str_repr)
    
    def test_alignment_issue_validation(self):
        """Test AlignmentIssue validation with invalid data."""
        # Test with invalid severity level
        with self.assertRaises(ValueError):
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
        
        issue_dict = issue.dict()
        
        self.assertEqual(issue_dict["level"], SeverityLevel.WARNING)
        self.assertEqual(issue_dict["category"], "test_category")
        self.assertEqual(issue_dict["message"], "Test message")
        self.assertEqual(issue_dict["details"], {"key": "value"})
        self.assertEqual(issue_dict["recommendation"], "Fix this")
        self.assertEqual(issue_dict["alignment_level"], AlignmentLevel.SCRIPT_CONTRACT)
    
    def test_alignment_issue_json_serialization(self):
        """Test AlignmentIssue JSON serialization."""
        issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test",
            message="Test message"
        )
        
        json_str = issue.json()
        self.assertIsInstance(json_str, str)
        self.assertIn("ERROR", json_str)
        self.assertIn("test", json_str)
        self.assertIn("Test message", json_str)


if __name__ == '__main__':
    unittest.main()
