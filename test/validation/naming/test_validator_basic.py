"""
Unit tests for basic NamingStandardValidator functionality.
"""

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from unittest.mock import Mock

from src.cursus.validation.naming.naming_standard_validator import NamingStandardValidator


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
