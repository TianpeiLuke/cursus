"""
Pytest tests for InterfaceViolation class.

This module tests the InterfaceViolation data structure used to represent
validation violations in the interface standard validator.
"""

import pytest

from cursus.validation.interface.interface_standard_validator import InterfaceViolation


class TestInterfaceViolation:
    """Tests for InterfaceViolation class."""

    def test_violation_creation(self):
        """Test creating an interface violation."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test violation message",
            expected="expected_value",
            actual="actual_value",
            suggestions=["suggestion1", "suggestion2"],
        )

        assert violation.component == "TestComponent"
        assert violation.violation_type == "test_violation"
        assert violation.message == "Test violation message"
        assert violation.expected == "expected_value"
        assert violation.actual == "actual_value"
        assert violation.suggestions == ["suggestion1", "suggestion2"]

    def test_violation_str_with_expected_actual(self):
        """Test string representation with expected and actual values."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message",
            expected="expected",
            actual="actual",
            suggestions=["fix it"],
        )

        result = str(violation)
        assert "TestComponent: Test message" in result
        assert "Expected: expected" in result
        assert "Actual: actual" in result
        assert "Suggestions: fix it" in result

    def test_violation_str_without_expected_actual(self):
        """Test string representation without expected and actual values."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message",
        )

        result = str(violation)
        assert result == "TestComponent: Test message"

    def test_violation_default_suggestions(self):
        """Test that suggestions default to empty list."""
        violation = InterfaceViolation(
            component="TestComponent",
            violation_type="test_violation",
            message="Test message",
        )

        assert violation.suggestions == []
