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


if __name__ == '__main__':
    unittest.main()
