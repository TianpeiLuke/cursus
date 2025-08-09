"""
Test suite for SeverityLevel enum.
"""

import unittest

from src.cursus.validation.alignment.alignment_utils import SeverityLevel


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


if __name__ == '__main__':
    unittest.main()
