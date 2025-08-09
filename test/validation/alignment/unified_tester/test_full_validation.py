"""
Test suite for UnifiedAlignmentTester full validation.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from src.cursus.validation.alignment.alignment_reporter import ValidationResult
from src.cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)


class TestFullValidation(unittest.TestCase):
    """Test full validation functionality in UnifiedAlignmentTester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = UnifiedAlignmentTester()
    
    def test_run_full_validation_success(self):
        """Test successful full validation across all levels."""
        # Mock all level validation methods
        with patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
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
            
            self.assertIsNotNone(report)
            self.assertIsNotNone(report.summary)
    
    def test_run_full_validation_with_skip_levels(self):
        """Test full validation with some levels skipped."""
        with patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            # Skip levels 2 and 4
            report = self.tester.run_full_validation(
                target_scripts=["test_script"],
                skip_levels=[2, 4]
            )
            
            # Verify only levels 1 and 3 were called
            mock_l1.assert_called_once_with(["test_script"])
            mock_l2.assert_not_called()
            mock_l3.assert_called_once_with(["test_script"])
            mock_l4.assert_not_called()
            
            self.assertIsNotNone(report)
    
    def test_run_full_validation_with_failures(self):
        """Test full validation with some failures."""
        # Mock level testers to return failure results
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [{
                    "severity": "ERROR",
                    "category": "script_contract",
                    "message": "Test failure",
                    "details": {"error": "test"},
                    "recommendation": "Fix the issue"
                }]
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            report = self.tester.run_full_validation(["test_script"])
            
            self.assertIsNotNone(report)
            self.assertEqual(len(self.tester.report.level1_results), 1)
            
            # Check that the failure was recorded
            result = self.tester.report.level1_results["test_script"]
            self.assertFalse(result.passed)
            self.assertGreater(len(result.issues), 0)
    
    def test_run_full_validation_with_critical_issues(self):
        """Test full validation with critical issues."""
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [{
                    "severity": "CRITICAL",
                    "category": "builder_configuration",
                    "message": "System cannot proceed",
                    "details": {"critical": "error"},
                    "recommendation": "Fix immediately"
                }]
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            report = self.tester.run_full_validation(["test_script"])
            
            self.assertIsNotNone(report)
            
            # Check for critical issues
            critical_issues = self.tester.get_critical_issues()
            self.assertGreater(len(critical_issues), 0)
    
    def test_run_full_validation_early_termination(self):
        """Test that validation continues even with critical issues."""
        mock_l1_results = {
            "test_script": {
                "passed": False,
                "issues": [{
                    "severity": "CRITICAL",
                    "category": "script_contract",
                    "message": "Cannot continue",
                    "details": {"critical": "error"}
                }]
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_validate.return_value = mock_l1_results
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            report = self.tester.run_full_validation(["test_script"])
            
            # All levels should still be called
            mock_l2.assert_called_once()
            mock_l3.assert_called_once()
            mock_l4.assert_called_once()
            
            self.assertIsNotNone(report)
    
    def test_run_full_validation_with_exception(self):
        """Test full validation when an exception occurs."""
        with patch.object(self.tester, '_run_level1_validation') as mock_l1:
            mock_l1.side_effect = Exception("Orchestration error")
            
            # Should not raise exception, but continue with other levels
            report = self.tester.run_full_validation(["test_script"])
            
            self.assertIsNotNone(report)
    
    def test_run_full_validation_missing_config(self):
        """Test full validation with incomplete configuration."""
        # Test with empty target scripts (should discover scripts)
        with patch.object(self.tester, 'discover_scripts') as mock_discover, \
             patch.object(self.tester, '_run_level1_validation') as mock_l1, \
             patch.object(self.tester, '_run_level2_validation') as mock_l2, \
             patch.object(self.tester, '_run_level3_validation') as mock_l3, \
             patch.object(self.tester, '_run_level4_validation') as mock_l4:
            
            mock_discover.return_value = ["discovered_script"]
            mock_l1.return_value = None
            mock_l2.return_value = None
            mock_l3.return_value = None
            mock_l4.return_value = None
            
            report = self.tester.run_full_validation(target_scripts=None)
            
            # Should call with None (which means discover all scripts)
            mock_l1.assert_called_once_with(None)
            self.assertIsNotNone(report)
    
    def test_export_report_json(self):
        """Test exporting report to JSON format."""
        # Add some test data
        test_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", test_result)
        
        json_output = self.tester.export_report(format='json')
        
        self.assertIsInstance(json_output, str)
        self.assertIn('"test"', json_output)  # Should contain test name
    
    def test_export_report_html(self):
        """Test exporting report to HTML format."""
        # Add some test data
        test_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", test_result)
        
        # Mock the HTML export to avoid template issues
        with patch.object(self.tester.report, 'export_to_html') as mock_html:
            mock_html.return_value = "<html>Test Report</html>"
            
            html_output = self.tester.export_report(format='html')
            
            self.assertIsInstance(html_output, str)
            self.assertIn('<html>', html_output)
    
    def test_export_report_invalid_format(self):
        """Test exporting report with invalid format."""
        with self.assertRaises(ValueError) as context:
            self.tester.export_report(format='xml')
        
        self.assertIn("Unsupported export format: xml", str(context.exception))
    
    def test_get_alignment_status_matrix(self):
        """Test getting alignment status matrix."""
        with patch.object(self.tester, 'discover_scripts') as mock_discover:
            mock_discover.return_value = ["script1", "script2"]
            
            # Add some results
            result1 = ValidationResult(test_name="test1", passed=True)
            result2 = ValidationResult(test_name="test2", passed=False)
            
            self.tester.report.add_level1_result("script1", result1)
            self.tester.report.add_level2_result("script1", result2)
            
            matrix = self.tester.get_alignment_status_matrix()
            
            self.assertIn("script1", matrix)
            self.assertIn("script2", matrix)
            self.assertEqual(matrix["script1"]["level1"], "PASSING")
            self.assertEqual(matrix["script1"]["level2"], "FAILING")
            self.assertEqual(matrix["script2"]["level1"], "UNKNOWN")


if __name__ == '__main__':
    unittest.main()
