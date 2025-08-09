"""
Test suite for UnifiedAlignmentTester level validation.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from src.cursus.validation.alignment.alignment_reporter import ValidationResult
from src.cursus.validation.alignment.alignment_utils import (
    AlignmentIssue, SeverityLevel, AlignmentLevel
)


class TestLevelValidation(unittest.TestCase):
    """Test individual level validation in UnifiedAlignmentTester."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = UnifiedAlignmentTester()
    
    def test_run_level_validation_level1(self):
        """Test running level 1 validation."""
        with patch.object(self.tester, '_run_level1_validation') as mock_level1:
            mock_level1.return_value = None
            
            report = self.tester.run_level_validation(1, ["test_script"])
            
            mock_level1.assert_called_once_with(["test_script"])
            self.assertIsNotNone(report)
    
    def test_run_level_validation_level2(self):
        """Test running level 2 validation."""
        with patch.object(self.tester, '_run_level2_validation') as mock_level2:
            mock_level2.return_value = None
            
            report = self.tester.run_level_validation(2, ["test_script"])
            
            mock_level2.assert_called_once_with(["test_script"])
            self.assertIsNotNone(report)
    
    def test_run_level_validation_level3(self):
        """Test running level 3 validation."""
        with patch.object(self.tester, '_run_level3_validation') as mock_level3:
            mock_level3.return_value = None
            
            report = self.tester.run_level_validation(3, ["test_script"])
            
            mock_level3.assert_called_once_with(["test_script"])
            self.assertIsNotNone(report)
    
    def test_run_level_validation_level4(self):
        """Test running level 4 validation."""
        with patch.object(self.tester, '_run_level4_validation') as mock_level4:
            mock_level4.return_value = None
            
            report = self.tester.run_level_validation(4, ["test_script"])
            
            mock_level4.assert_called_once_with(["test_script"])
            self.assertIsNotNone(report)
    
    def test_run_level_validation_invalid_level(self):
        """Test running validation with invalid level."""
        with self.assertRaises(ValueError) as context:
            self.tester.run_level_validation(5)
        
        self.assertIn("Invalid alignment level: 5", str(context.exception))
    
    def test_level1_validation_with_mock_results(self):
        """Test level 1 validation with mocked results."""
        mock_results = {
            "test_script": {
                "passed": True,
                "issues": []
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate:
            mock_validate.return_value = mock_results
            
            self.tester._run_level1_validation(["test_script"])
            
            mock_validate.assert_called_once_with(["test_script"])
            self.assertEqual(len(self.tester.report.level1_results), 1)
            self.assertIn("test_script", self.tester.report.level1_results)
    
    def test_level1_validation_with_issues(self):
        """Test level 1 validation with issues."""
        mock_results = {
            "test_script": {
                "passed": False,
                "issues": [{
                    "severity": "ERROR",
                    "category": "path_usage",
                    "message": "Undeclared path usage",
                    "details": {"path": "/opt/ml/input"},
                    "recommendation": "Add path to contract"
                }]
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate:
            mock_validate.return_value = mock_results
            
            self.tester._run_level1_validation(["test_script"])
            
            result = self.tester.report.level1_results["test_script"]
            self.assertFalse(result.passed)
            self.assertEqual(len(result.issues), 1)
            self.assertEqual(result.issues[0].level, SeverityLevel.ERROR)
    
    def test_level1_validation_with_exception(self):
        """Test level 1 validation with exception."""
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")
            
            self.tester._run_level1_validation(["test_script"])
            
            # Should create error result
            self.assertIn("validation_error", self.tester.report.level1_results)
            error_result = self.tester.report.level1_results["validation_error"]
            self.assertFalse(error_result.passed)
            self.assertGreater(len(error_result.issues), 0)
    
    def test_validate_specific_script(self):
        """Test validating a specific script across all levels."""
        # Mock all level testers
        with patch.object(self.tester.level1_tester, 'validate_script') as mock_l1, \
             patch.object(self.tester.level2_tester, 'validate_contract') as mock_l2, \
             patch.object(self.tester.level3_tester, 'validate_specification') as mock_l3, \
             patch.object(self.tester.level4_tester, 'validate_builder') as mock_l4:
            
            # Set up mock returns
            mock_l1.return_value = {"passed": True, "issues": []}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}
            
            result = self.tester.validate_specific_script("test_script")
            
            self.assertEqual(result['script_name'], "test_script")
            self.assertEqual(result['overall_status'], "PASSING")
            self.assertTrue(result['level1']['passed'])
            self.assertTrue(result['level2']['passed'])
            self.assertTrue(result['level3']['passed'])
            self.assertTrue(result['level4']['passed'])
    
    def test_validate_specific_script_with_failures(self):
        """Test validating a specific script with some failures."""
        with patch.object(self.tester.level1_tester, 'validate_script') as mock_l1, \
             patch.object(self.tester.level2_tester, 'validate_contract') as mock_l2, \
             patch.object(self.tester.level3_tester, 'validate_specification') as mock_l3, \
             patch.object(self.tester.level4_tester, 'validate_builder') as mock_l4:
            
            # Set up mock returns with some failures
            mock_l1.return_value = {"passed": False, "issues": [{"severity": "ERROR"}]}
            mock_l2.return_value = {"passed": True, "issues": []}
            mock_l3.return_value = {"passed": True, "issues": []}
            mock_l4.return_value = {"passed": True, "issues": []}
            
            result = self.tester.validate_specific_script("test_script")
            
            self.assertEqual(result['overall_status'], "FAILING")
            self.assertFalse(result['level1']['passed'])
    
    def test_get_validation_summary(self):
        """Test getting validation summary."""
        # Add some mock results to the report
        mock_result = ValidationResult(test_name="test", passed=True)
        self.tester.report.add_level1_result("test", mock_result)
        
        summary = self.tester.get_validation_summary()
        
        self.assertIn('overall_status', summary)
        self.assertIn('total_tests', summary)
        self.assertIn('pass_rate', summary)
        self.assertIn('level_breakdown', summary)
    
    def test_discover_scripts(self):
        """Test script discovery."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob:
            
            mock_exists.return_value = True
            mock_script1 = MagicMock()
            mock_script1.name = "script1.py"
            mock_script1.stem = "script1"
            mock_script2 = MagicMock()
            mock_script2.name = "script2.py"
            mock_script2.stem = "script2"
            
            mock_glob.return_value = [mock_script1, mock_script2]
            
            scripts = self.tester.discover_scripts()
            
            self.assertEqual(len(scripts), 2)
            self.assertIn("script1", scripts)
            self.assertIn("script2", scripts)


if __name__ == '__main__':
    unittest.main()
