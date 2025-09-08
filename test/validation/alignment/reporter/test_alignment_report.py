"""
Test suite for AlignmentReport class.
"""

import unittest
import json

from cursus.validation.alignment.alignment_reporter import (
    AlignmentReport, ValidationResult, AlignmentSummary
)
from cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)

class TestAlignmentReport(unittest.TestCase):
    """Test AlignmentReport class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_issue = AlignmentIssue(
            level=SeverityLevel.ERROR,
            category="test_category",
            message="Test issue",
            details={"key": "value"}
        )
        
        self.sample_result = ValidationResult(
            test_name="test_validation",
            passed=False
        )
        self.sample_result.add_issue(self.sample_issue)
    
    def test_alignment_report_creation(self):
        """Test basic AlignmentReport creation."""
        report = AlignmentReport()
        
        self.assertEqual(len(report.level1_results), 0)
        self.assertEqual(len(report.level2_results), 0)
        self.assertEqual(len(report.level3_results), 0)
        self.assertEqual(len(report.level4_results), 0)
        self.assertIsNone(report.summary)
        self.assertEqual(len(report.recommendations), 0)
        self.assertEqual(report.metadata, {})
    
    def test_add_result(self):
        """Test adding results to different levels."""
        report = AlignmentReport()
        
        # Add results to each level
        report.add_level1_result("script_test", self.sample_result)
        report.add_level2_result("contract_test", self.sample_result)
        report.add_level3_result("spec_test", self.sample_result)
        report.add_level4_result("builder_test", self.sample_result)
        
        self.assertEqual(len(report.level1_results), 1)
        self.assertEqual(len(report.level2_results), 1)
        self.assertEqual(len(report.level3_results), 1)
        self.assertEqual(len(report.level4_results), 1)
        
        self.assertIn("script_test", report.level1_results)
        self.assertIn("contract_test", report.level2_results)
        self.assertIn("spec_test", report.level3_results)
        self.assertIn("builder_test", report.level4_results)
    
    def test_get_all_results(self):
        """Test getting all results across levels."""
        report = AlignmentReport()
        
        report.add_level1_result("test1", self.sample_result)
        report.add_level2_result("test2", self.sample_result)
        
        all_results = report.get_all_results()
        
        self.assertEqual(len(all_results), 2)
        self.assertIn("test1", all_results)
        self.assertIn("test2", all_results)
    
    def test_get_summary(self):
        """Test generating summary."""
        report = AlignmentReport()
        report.add_level1_result("test1", self.sample_result)
        
        summary = report.generate_summary()
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary.total_tests, 1)
        self.assertEqual(summary.failed_tests, 1)
        self.assertEqual(summary.passed_tests, 0)
        self.assertEqual(summary.total_issues, 1)
        self.assertEqual(summary.error_issues, 1)
    
    def test_get_issues_by_level(self):
        """Test getting issues by severity level."""
        report = AlignmentReport()
        
        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="critical_test",
            message="Critical issue"
        )
        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        
        report.add_level1_result("critical_test", critical_result)
        report.add_level2_result("error_test", self.sample_result)
        
        critical_issues = report.get_critical_issues()
        error_issues = report.get_error_issues()
        
        self.assertEqual(len(critical_issues), 1)
        self.assertEqual(len(error_issues), 1)
        self.assertEqual(critical_issues[0].level, SeverityLevel.CRITICAL)
        self.assertEqual(error_issues[0].level, SeverityLevel.ERROR)
    
    def test_has_critical_issues(self):
        """Test checking for critical issues."""
        report = AlignmentReport()
        
        # No critical issues initially
        self.assertFalse(report.has_critical_issues())
        
        # Add critical issue
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="critical_test",
            message="Critical issue"
        )
        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        report.add_level1_result("critical_test", critical_result)
        
        self.assertTrue(report.has_critical_issues())
    
    def test_export_to_json(self):
        """Test exporting report to JSON."""
        report = AlignmentReport()
        report.add_level1_result("test1", self.sample_result)
        
        json_output = report.export_to_json()
        
        self.assertIsInstance(json_output, str)
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        self.assertIn('summary', data)
        self.assertIn('level1_results', data)
        self.assertIn('recommendations', data)
        self.assertIn('metadata', data)
    
    def test_export_to_html(self):
        """Test exporting report to HTML."""
        report = AlignmentReport()
        report.add_level1_result("test1", self.sample_result)
        
        html_output = report.export_to_html()
        
        self.assertIsInstance(html_output, str)
        self.assertIn('<html>', html_output)
        self.assertIn('Alignment Validation Report', html_output)
        self.assertIn('test1', html_output)

if __name__ == '__main__':
    unittest.main()
