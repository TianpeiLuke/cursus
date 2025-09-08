"""
Test suite for ValidationResult model.
"""

import unittest
from datetime import datetime

from cursus.validation.alignment.alignment_reporter import ValidationResult
from cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)

class TestValidationResult(unittest.TestCase):
    """Test ValidationResult model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test_category",
            message="Test issue",
            details={"key": "value"}
        )
    
    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(
            test_name="test_script_contract",
            passed=True
        )
        
        self.assertEqual(result.test_name, "test_script_contract")
        self.assertTrue(result.passed)
        self.assertEqual(len(result.issues), 0)
        self.assertEqual(result.details, {})
        self.assertIsInstance(result.timestamp, datetime)
    
    def test_add_issue(self):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(
            test_name="test_with_issues",
            passed=True
        )
        
        # Add a warning issue - should not change passed status
        warning_issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="test",
            message="Warning message"
        )
        result.add_issue(warning_issue)
        
        self.assertTrue(result.passed)  # Still passing
        self.assertEqual(len(result.issues), 1)
        
        # Add an error issue - should change passed status
        result.add_issue(self.sample_issue)
        
        self.assertFalse(result.passed)  # Now failing
        self.assertEqual(len(result.issues), 2)
    
    def test_get_severity_level(self):
        """Test getting highest severity level."""
        result = ValidationResult(
            test_name="test_severity",
            passed=True
        )
        
        # No issues - should return None
        self.assertIsNone(result.get_severity_level())
        
        # Add warning issue
        warning_issue = AlignmentIssue(
            level=SeverityLevel.WARNING,
            category="test",
            message="Warning"
        )
        result.add_issue(warning_issue)
        self.assertEqual(result.get_severity_level(), SeverityLevel.WARNING)
        
        # Add error issue - should return ERROR as highest
        result.add_issue(self.sample_issue)
        self.assertEqual(result.get_severity_level(), SeverityLevel.ERROR)
    
    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        result = ValidationResult(
            test_name="test_critical",
            passed=True
        )
        
        self.assertFalse(result.has_critical_issues())
        
        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="test",
            message="Critical issue"
        )
        result.add_issue(critical_issue)
        
        self.assertTrue(result.has_critical_issues())
    
    def test_has_errors(self):
        """Test checking for error issues."""
        result = ValidationResult(
            test_name="test_errors",
            passed=True
        )
        
        self.assertFalse(result.has_errors())
        
        # Add error issue
        result.add_issue(self.sample_issue)
        
        self.assertTrue(result.has_errors())
    
    def test_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            test_name="test_dict",
            passed=False,
            details={"config": "test_config"}
        )
        result.add_issue(self.sample_issue)
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['test_name'], "test_dict")
        self.assertFalse(result_dict['passed'])
        self.assertEqual(result_dict['details'], {"config": "test_config"})
        self.assertEqual(len(result_dict['issues']), 1)
        self.assertEqual(result_dict['severity_level'], "ERROR")
        self.assertIn('timestamp', result_dict)

if __name__ == '__main__':
    unittest.main()
