"""
Pytest tests for basic NamingStandardValidator functionality.
"""

import pytest
from unittest.mock import Mock

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestNamingStandardValidator:
    """Test the NamingStandardValidator class."""

    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator, NamingStandardValidator)
        assert validator.violations == []

    def test_clear_violations(self, validator):
        """Test clearing violations."""
        validator.violations = [Mock()]
        validator.clear_violations()
        assert validator.violations == []
