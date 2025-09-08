"""
Unit tests for basic NamingStandardValidator functionality.
"""

import unittest
from unittest.mock import Mock

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestNamingStandardValidator(unittest.TestCase):
    """Test the NamingStandardValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator, NamingStandardValidator)
        self.assertEqual(self.validator.violations, [])
    
    def test_clear_violations(self):
        """Test clearing violations."""
        self.validator.violations = [Mock()]
        self.validator.clear_violations()
        self.assertEqual(self.validator.violations, [])

if __name__ == '__main__':
    unittest.main()
