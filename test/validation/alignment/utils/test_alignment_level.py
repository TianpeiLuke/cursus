"""
Test suite for AlignmentLevel enum.
"""

import unittest

from src.cursus.validation.alignment.alignment_utils import AlignmentLevel


class TestAlignmentLevel(unittest.TestCase):
    """Test AlignmentLevel enum."""
    
    def test_alignment_levels_exist(self):
        """Test that all expected alignment levels exist."""
        self.assertEqual(AlignmentLevel.SCRIPT_CONTRACT.value, 1)
        self.assertEqual(AlignmentLevel.CONTRACT_SPECIFICATION.value, 2)
        self.assertEqual(AlignmentLevel.SPECIFICATION_DEPENDENCY.value, 3)
        self.assertEqual(AlignmentLevel.BUILDER_CONFIGURATION.value, 4)
    
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
            self.assertLess(levels[i].value, levels[i + 1].value)
    
    def test_alignment_level_comparison(self):
        """Test that alignment levels can be compared."""
        self.assertEqual(AlignmentLevel.SCRIPT_CONTRACT, AlignmentLevel.SCRIPT_CONTRACT)
        self.assertNotEqual(AlignmentLevel.SCRIPT_CONTRACT, AlignmentLevel.CONTRACT_SPECIFICATION)
        self.assertLess(AlignmentLevel.SCRIPT_CONTRACT.value, AlignmentLevel.CONTRACT_SPECIFICATION.value)
    
    def test_alignment_level_string_representation(self):
        """Test string representation of alignment levels."""
        self.assertEqual(str(AlignmentLevel.SCRIPT_CONTRACT), "AlignmentLevel.SCRIPT_CONTRACT")
        self.assertEqual(str(AlignmentLevel.CONTRACT_SPECIFICATION), "AlignmentLevel.CONTRACT_SPECIFICATION")
        self.assertEqual(str(AlignmentLevel.SPECIFICATION_DEPENDENCY), "AlignmentLevel.SPECIFICATION_DEPENDENCY")
        self.assertEqual(str(AlignmentLevel.BUILDER_CONFIGURATION), "AlignmentLevel.BUILDER_CONFIGURATION")
    
    def test_alignment_level_membership(self):
        """Test membership in AlignmentLevel enum."""
        self.assertIn(AlignmentLevel.SCRIPT_CONTRACT, AlignmentLevel)
        self.assertIn(AlignmentLevel.CONTRACT_SPECIFICATION, AlignmentLevel)
        self.assertIn(AlignmentLevel.SPECIFICATION_DEPENDENCY, AlignmentLevel)
        self.assertIn(AlignmentLevel.BUILDER_CONFIGURATION, AlignmentLevel)
    
    def test_alignment_level_iteration(self):
        """Test iteration over AlignmentLevel enum."""
        expected_values = [1, 2, 3, 4]
        actual_values = [level.value for level in AlignmentLevel]
        
        self.assertEqual(len(actual_values), 4)
        self.assertEqual(sorted(actual_values), expected_values)
    
    def test_alignment_level_from_value(self):
        """Test creating AlignmentLevel from numeric values."""
        self.assertEqual(AlignmentLevel(1), AlignmentLevel.SCRIPT_CONTRACT)
        self.assertEqual(AlignmentLevel(2), AlignmentLevel.CONTRACT_SPECIFICATION)
        self.assertEqual(AlignmentLevel(3), AlignmentLevel.SPECIFICATION_DEPENDENCY)
        self.assertEqual(AlignmentLevel(4), AlignmentLevel.BUILDER_CONFIGURATION)
    
    def test_alignment_level_invalid_value(self):
        """Test creating AlignmentLevel from invalid values."""
        with self.assertRaises(ValueError):
            AlignmentLevel(0)
        
        with self.assertRaises(ValueError):
            AlignmentLevel(5)
        
        with self.assertRaises(ValueError):
            AlignmentLevel("SCRIPT_CONTRACT")  # string should fail


if __name__ == '__main__':
    unittest.main()
