"""
Pytest tests for the NamingViolation class.
"""

import pytest

from cursus.validation.naming.naming_standard_validator import NamingViolation

class TestNamingViolation:
    """Test the NamingViolation class."""
    
    def test_naming_violation_creation(self):
        """Test creating a NamingViolation object."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            expected="expected_value",
            actual="actual_value",
            suggestions=["suggestion1", "suggestion2"]
        )
        
        assert violation.component == "TestComponent"
        assert violation.violation_type == "test_type"
        assert violation.message == "Test message"
        assert violation.expected == "expected_value"
        assert violation.actual == "actual_value"
        assert violation.suggestions == ["suggestion1", "suggestion2"]
    
    def test_naming_violation_str_with_expected_actual(self):
        """Test string representation with expected and actual values."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            expected="expected_value",
            actual="actual_value"
        )
        
        expected_str = "TestComponent: Test message (Expected: expected_value, Actual: actual_value)"
        assert str(violation) == expected_str
    
    def test_naming_violation_str_with_suggestions(self):
        """Test string representation with suggestions."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message",
            suggestions=["suggestion1", "suggestion2"]
        )
        
        expected_str = "TestComponent: Test message Suggestions: suggestion1, suggestion2"
        assert str(violation) == expected_str
    
    def test_naming_violation_str_minimal(self):
        """Test string representation with minimal information."""
        violation = NamingViolation(
            component="TestComponent",
            violation_type="test_type",
            message="Test message"
        )
        
        expected_str = "TestComponent: Test message"
        assert str(violation) == expected_str
