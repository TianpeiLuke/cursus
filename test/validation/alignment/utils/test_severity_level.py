"""
Test suite for SeverityLevel enum.
"""

import pytest

from cursus.validation.alignment.alignment_utils import SeverityLevel

class TestSeverityLevel:
    """Test SeverityLevel enum."""
    
    def test_severity_levels_exist(self):
        """Test that all expected severity levels exist."""
        assert SeverityLevel.INFO.value == "INFO"
        assert SeverityLevel.WARNING.value == "WARNING"
        assert SeverityLevel.ERROR.value == "ERROR"
        assert SeverityLevel.CRITICAL.value == "CRITICAL"
    
    def test_severity_level_comparison(self):
        """Test that severity levels can be compared."""
        # Test enum equality
        assert SeverityLevel.ERROR == SeverityLevel.ERROR
        assert SeverityLevel.ERROR != SeverityLevel.WARNING
    
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
                    assert severity != other_severity
    
    def test_severity_level_string_representation(self):
        """Test string representation of severity levels."""
        assert str(SeverityLevel.INFO) == "SeverityLevel.INFO"
        assert str(SeverityLevel.WARNING) == "SeverityLevel.WARNING"
        assert str(SeverityLevel.ERROR) == "SeverityLevel.ERROR"
        assert str(SeverityLevel.CRITICAL) == "SeverityLevel.CRITICAL"
    
    def test_severity_level_membership(self):
        """Test membership in SeverityLevel enum."""
        assert SeverityLevel.INFO in SeverityLevel
        assert SeverityLevel.WARNING in SeverityLevel
        assert SeverityLevel.ERROR in SeverityLevel
        assert SeverityLevel.CRITICAL in SeverityLevel
    
    def test_severity_level_iteration(self):
        """Test iteration over SeverityLevel enum."""
        expected_values = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        actual_values = [level.value for level in SeverityLevel]
        
        assert len(actual_values) == 4
        for expected in expected_values:
            assert expected in actual_values
    
    def test_severity_level_from_string(self):
        """Test creating SeverityLevel from string values."""
        assert SeverityLevel("INFO") == SeverityLevel.INFO
        assert SeverityLevel("WARNING") == SeverityLevel.WARNING
        assert SeverityLevel("ERROR") == SeverityLevel.ERROR
        assert SeverityLevel("CRITICAL") == SeverityLevel.CRITICAL
    
    def test_severity_level_invalid_string(self):
        """Test creating SeverityLevel from invalid string."""
        with pytest.raises(ValueError):
            SeverityLevel("INVALID")
        
        with pytest.raises(ValueError):
            SeverityLevel("info")  # lowercase should fail
        
        with pytest.raises(ValueError):
            SeverityLevel("")


if __name__ == '__main__':
    pytest.main([__file__])
