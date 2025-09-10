"""
Test suite for AlignmentLevel enum.
"""

import pytest

from cursus.validation.alignment.alignment_utils import AlignmentLevel

class TestAlignmentLevel:
    """Test AlignmentLevel enum."""
    
    def test_alignment_levels_exist(self):
        """Test that all expected alignment levels exist."""
        assert AlignmentLevel.SCRIPT_CONTRACT.value == 1
        assert AlignmentLevel.CONTRACT_SPECIFICATION.value == 2
        assert AlignmentLevel.SPECIFICATION_DEPENDENCY.value == 3
        assert AlignmentLevel.BUILDER_CONFIGURATION.value == 4
    
    def test_alignment_level_ordering(self):
        """Test that alignment levels have correct ordering."""
        levels = [
            AlignmentLevel.SCRIPT_CONTRACT,
            AlignmentLevel.CONTRACT_SPECIFICATION,
            AlignmentLevel.SPECIFICATION_DEPENDENCY,
            AlignmentLevel.BUILDER_CONFIGURATION
        ]
        
        # Test that values are in ascending order
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value
    
    def test_alignment_level_comparison(self):
        """Test that alignment levels can be compared."""
        assert AlignmentLevel.SCRIPT_CONTRACT == AlignmentLevel.SCRIPT_CONTRACT
        assert AlignmentLevel.SCRIPT_CONTRACT != AlignmentLevel.CONTRACT_SPECIFICATION
        assert AlignmentLevel.SCRIPT_CONTRACT.value < AlignmentLevel.CONTRACT_SPECIFICATION.value
    
    def test_alignment_level_string_representation(self):
        """Test string representation of alignment levels."""
        assert str(AlignmentLevel.SCRIPT_CONTRACT) == "AlignmentLevel.SCRIPT_CONTRACT"
        assert str(AlignmentLevel.CONTRACT_SPECIFICATION) == "AlignmentLevel.CONTRACT_SPECIFICATION"
        assert str(AlignmentLevel.SPECIFICATION_DEPENDENCY) == "AlignmentLevel.SPECIFICATION_DEPENDENCY"
        assert str(AlignmentLevel.BUILDER_CONFIGURATION) == "AlignmentLevel.BUILDER_CONFIGURATION"
    
    def test_alignment_level_membership(self):
        """Test membership in AlignmentLevel enum."""
        assert AlignmentLevel.SCRIPT_CONTRACT in AlignmentLevel
        assert AlignmentLevel.CONTRACT_SPECIFICATION in AlignmentLevel
        assert AlignmentLevel.SPECIFICATION_DEPENDENCY in AlignmentLevel
        assert AlignmentLevel.BUILDER_CONFIGURATION in AlignmentLevel
    
    def test_alignment_level_iteration(self):
        """Test iteration over AlignmentLevel enum."""
        expected_values = [1, 2, 3, 4]
        actual_values = [level.value for level in AlignmentLevel]
        
        assert len(actual_values) == 4
        assert sorted(actual_values) == expected_values
    
    def test_alignment_level_from_value(self):
        """Test creating AlignmentLevel from numeric values."""
        assert AlignmentLevel(1) == AlignmentLevel.SCRIPT_CONTRACT
        assert AlignmentLevel(2) == AlignmentLevel.CONTRACT_SPECIFICATION
        assert AlignmentLevel(3) == AlignmentLevel.SPECIFICATION_DEPENDENCY
        assert AlignmentLevel(4) == AlignmentLevel.BUILDER_CONFIGURATION
    
    def test_alignment_level_invalid_value(self):
        """Test creating AlignmentLevel from invalid values."""
        with pytest.raises(ValueError):
            AlignmentLevel(0)
        
        with pytest.raises(ValueError):
            AlignmentLevel(5)
        
        with pytest.raises(ValueError):
            AlignmentLevel("SCRIPT_CONTRACT")  # string should fail


if __name__ == '__main__':
    pytest.main([__file__])
