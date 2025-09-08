"""
Tests for InterfaceViolation class.

This module tests the InterfaceViolation data structure used to represent
validation violations in the interface standard validator.
"""

import unittest

from cursus.validation.interface.interface_standard_validator import InterfaceViolation

class TestInterfaceViolation(unittest.TestCase):
    """Tests for InterfaceViolation class."""
    
    def test_violation_creation(self):
        """Test creating an interface violation."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test violation message",
            expected="expected_value",
            actual="actual_value",
            suggestions=["suggestion1", "suggestion2"]
        )
        
        self.assertEqual(violation.component, "TestComponent")
        self.assertEqual(violation.violation_type, "test_violation")
        self.assertEqual(violation.message, "Test violation message")
        self.assertEqual(violation.expected, "expected_value")
        self.assertEqual(violation.actual, "actual_value")
        self.assertEqual(violation.suggestions, ["suggestion1", "suggestion2"])
    
    def test_violation_str_with_expected_actual(self):
        """Test string representation with expected and actual values."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message",
            expected="expected",
            actual="actual",
            suggestions=["fix it"]
        )
        
        result = str(violation)
        self.assertIn("TestComponent: Test message", result)
        self.assertIn("Expected: expected", result)
        self.assertIn("Actual: actual", result)
        self.assertIn("Suggestions: fix it", result)
    
    def test_violation_str_without_expected_actual(self):
        """Test string representation without expected and actual values."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message"
        )
        
        result = str(violation)
        self.assertEqual(result, "TestComponent: Test message")
    
    def test_violation_default_suggestions(self):
        """Test that suggestions default to empty list."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message"
        )
        
        self.assertEqual(violation.suggestions, [])

if __name__ == '__main__':
    unittest.main()
