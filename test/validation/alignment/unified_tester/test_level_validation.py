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
    
    def test_validate_specific_script_with_exception(self):
        """Test validating a specific script when an exception occurs."""
        with patch.object(self.tester.level1_tester, 'validate_script') as mock_l1:
            mock_l1.side_effect = Exception("Validation error")
            
            result = self.tester.validate_specific_script("test_script")
            
            self.assertEqual(result['overall_status'], "ERROR")
            self.assertIn('error', result)
    
    def test_level3_validation_mode_configuration(self):
        """Test that Level 3 validation mode is properly configured."""
        # Test strict mode
        strict_tester = UnifiedAlignmentTester(level3_validation_mode="strict")
        self.assertIsNotNone(strict_tester.level3_config)
        
        # Test relaxed mode (default)
        relaxed_tester = UnifiedAlignmentTester(level3_validation_mode="relaxed")
        self.assertIsNotNone(relaxed_tester.level3_config)
        
        # Test permissive mode
        permissive_tester = UnifiedAlignmentTester(level3_validation_mode="permissive")
        self.assertIsNotNone(permissive_tester.level3_config)
        
        # Test invalid mode (should default to relaxed)
        invalid_tester = UnifiedAlignmentTester(level3_validation_mode="invalid")
        self.assertIsNotNone(invalid_tester.level3_config)
    
    def test_json_serialization_in_level_validation(self):
        """Test that level validation results can be JSON serialized."""
        import json
        
        # Mock results with complex objects
        mock_results = {
            "test_script": {
                "passed": True,
                "issues": [],
                "complex_data": {
                    "type_object": str,
                    "property_object": property(lambda self: "test"),
                    "nested": {
                        "inner_type": int
                    }
                }
            }
        }
        
        with patch.object(self.tester.level1_tester, 'validate_all_scripts') as mock_validate:
            mock_validate.return_value = mock_results
            
            self.tester._run_level1_validation(["test_script"])
            
            # Should be able to export to JSON without errors
            json_output = self.tester.export_report(format='json')
            parsed_json = json.loads(json_output)
            
            self.assertIsInstance(parsed_json, dict)
    
    def test_cli_compatibility_methods(self):
        """Test methods that are specifically used by the CLI."""
        # Test get_validation_summary
        test_result = ValidationResult(test_name="cli_test", passed=True)
        self.tester.report.add_level1_result("cli_test", test_result)
        
        summary = self.tester.get_validation_summary()
        
        self.assertIn('overall_status', summary)
        self.assertIn('total_tests', summary)
        self.assertIn('pass_rate', summary)
        self.assertIn('level_breakdown', summary)
        
        # Test get_critical_issues
        critical_issue = AlignmentIssue(
            level=SeverityLevel.CRITICAL,
            category="cli_test",
            message="Critical issue for CLI",
            details={"cli": "test"}
        )
        
        critical_result = ValidationResult(test_name="critical_test", passed=False)
        critical_result.add_issue(critical_issue)
        self.tester.report.add_level1_result("critical_test", critical_result)
        
        critical_issues = self.tester.get_critical_issues()
        self.assertGreater(len(critical_issues), 0)
        self.assertEqual(critical_issues[0]['level'], 'CRITICAL')
    
    def test_alignment_status_matrix_with_real_data(self):
        """Test alignment status matrix with realistic validation data."""
        with patch.object(self.tester, 'discover_scripts') as mock_discover:
            mock_discover.return_value = ["payload", "package", "dummy_training"]
            
            # Add mixed results
            passing_result = ValidationResult(test_name="payload_test", passed=True)
            failing_result = ValidationResult(test_name="package_test", passed=False)
            
            self.tester.report.add_level1_result("payload", passing_result)
            self.tester.report.add_level2_result("payload", failing_result)
            self.tester.report.add_level3_result("package", passing_result)
            self.tester.report.add_level4_result("package", failing_result)
            
            matrix = self.tester.get_alignment_status_matrix()
            
            # Verify structure
            self.assertIn("payload", matrix)
            self.assertIn("package", matrix)
            self.assertIn("dummy_training", matrix)
            
            # Verify status values
            self.assertEqual(matrix["payload"]["level1"], "PASSING")
            self.assertEqual(matrix["payload"]["level2"], "FAILING")
            self.assertEqual(matrix["payload"]["level3"], "UNKNOWN")
            self.assertEqual(matrix["payload"]["level4"], "UNKNOWN")
            
            self.assertEqual(matrix["package"]["level1"], "UNKNOWN")
            self.assertEqual(matrix["package"]["level2"], "UNKNOWN")
            self.assertEqual(matrix["package"]["level3"], "PASSING")
            self.assertEqual(matrix["package"]["level4"], "FAILING")
            
            # Scripts with no results should be UNKNOWN
            for level in ["level1", "level2", "level3", "level4"]:
                self.assertEqual(matrix["dummy_training"][level], "UNKNOWN")
    
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
