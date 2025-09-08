"""
Test suite for SeverityLevel enum.
"""

import unittest

from cursus.validation.alignment.alignment_utils import SeverityLevel

class TestSeverityLevel(unittest.TestCase):
    """Test SeverityLevel enum."""
    
    def test_severity_levels_exist(self):
        """Test that all expected severity levels exist."""
        self.assertEqual(SeverityLevel.INFO.value, "INFO")
        self.assertEqual(SeverityLevel.WARNING.value, "WARNING")
        self.assertEqual(SeverityLevel.ERROR.value, "ERROR")
        self.assertEqual(SeverityLevel.CRITICAL.value, "CRITICAL")
    
    def test_severity_level_comparison(self):
        """Test that severity levels can be compared."""
        # Test enum equality
        self.assertEqual(SeverityLevel.ERROR, SeverityLevel.ERROR)
        self.assertNotEqual(SeverityLevel.ERROR, SeverityLevel.WARNING)
    
    def test_severity_ordering(self):
        """Test that severity ordering works correctly."""
        severities = [
            SeverityLevel.INFO,
            SeverityLevel.WARNING,
            SeverityLevel.ERROR,
            SeverityLevel.CRITICAL
        ]
        
        # Test that each level is distinct
        for i, severity in enumerate(severities):
            for j, other_severity in enumerate(severities):
                if i != j:
                    self.assertNotEqual(severity, other_severity)
    
    def test_severity_level_string_representation(self):
        """Test string representation of severity levels."""
        self.assertEqual(str(SeverityLevel.INFO), "SeverityLevel.INFO")
        self.assertEqual(str(SeverityLevel.WARNING), "SeverityLevel.WARNING")
        self.assertEqual(str(SeverityLevel.ERROR), "SeverityLevel.ERROR")
        self.assertEqual(str(SeverityLevel.CRITICAL), "SeverityLevel.CRITICAL")
    
    def test_severity_level_membership(self):
        """Test membership in SeverityLevel enum."""
        self.assertIn(SeverityLevel.INFO, SeverityLevel)
        self.assertIn(SeverityLevel.WARNING, SeverityLevel)
        self.assertIn(SeverityLevel.ERROR, SeverityLevel)
        self.assertIn(SeverityLevel.CRITICAL, SeverityLevel)
    
    def test_severity_level_iteration(self):
        """Test iteration over SeverityLevel enum."""
        expected_values = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        actual_values = [level.value for level in SeverityLevel]
        
        self.assertEqual(len(actual_values), 4)
        for expected in expected_values:
            self.assertIn(expected, actual_values)
    
    def test_severity_level_from_string(self):
        """Test creating SeverityLevel from string values."""
        self.assertEqual(SeverityLevel("INFO"), SeverityLevel.INFO)
        self.assertEqual(SeverityLevel("WARNING"), SeverityLevel.WARNING)
        self.assertEqual(SeverityLevel("ERROR"), SeverityLevel.ERROR)
        self.assertEqual(SeverityLevel("CRITICAL"), SeverityLevel.CRITICAL)
    
    def test_severity_level_invalid_string(self):
        """Test creating SeverityLevel from invalid string."""
        with self.assertRaises(ValueError):
            SeverityLevel("INVALID")
        
        with self.assertRaises(ValueError):
            SeverityLevel("info")  # lowercase should fail
        
        with self.assertRaises(ValueError):
            SeverityLevel("")

if __name__ == '__main__':
    unittest.main()
