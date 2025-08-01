"""
Unit tests for the name_generator module.

This module tests the pipeline name generation, validation, and sanitization 
functionality to ensure conformance with SageMaker constraints.
"""

import unittest
from src.pipeline_api.name_generator import (
    generate_random_word,
    generate_pipeline_name,
    validate_pipeline_name,
    sanitize_pipeline_name
)

class TestNameGenerator(unittest.TestCase):
    """Tests for the name_generator module."""

    def test_generate_random_word_length(self):
        """Test that generate_random_word returns a word of the expected length."""
        word = generate_random_word(5)
        self.assertEqual(len(word), 5)
        
        word = generate_random_word(10)
        self.assertEqual(len(word), 10)
        
    def test_validate_pipeline_name(self):
        """Test that validate_pipeline_name correctly validates pipeline names."""
        # Valid names
        self.assertTrue(validate_pipeline_name("valid-name"))
        self.assertTrue(validate_pipeline_name("valid-name-123"))
        self.assertTrue(validate_pipeline_name("a"))
        self.assertTrue(validate_pipeline_name("123"))
        self.assertTrue(validate_pipeline_name("a" * 255))  # Maximum length
        
        # Invalid names
        self.assertFalse(validate_pipeline_name(""))  # Empty
        self.assertFalse(validate_pipeline_name("-leading-hyphen"))  # Leading hyphen
        self.assertFalse(validate_pipeline_name("invalid.name"))  # Contains dot
        self.assertFalse(validate_pipeline_name("invalid_name"))  # Contains underscore
        self.assertFalse(validate_pipeline_name("invalid@name"))  # Contains special char
        self.assertFalse(validate_pipeline_name("a" * 256))  # Too long
        
    def test_sanitize_pipeline_name(self):
        """Test that sanitize_pipeline_name correctly sanitizes pipeline names."""
        # Names that should be unchanged
        self.assertEqual(sanitize_pipeline_name("valid-name"), "valid-name")
        self.assertEqual(sanitize_pipeline_name("valid-name-123"), "valid-name-123")
        
        # Names that should be sanitized
        self.assertEqual(sanitize_pipeline_name("invalid.name"), "invalid-name")
        self.assertEqual(sanitize_pipeline_name("invalid_name"), "invalid-name")
        self.assertEqual(sanitize_pipeline_name("invalid@name"), "invalidname")
        self.assertEqual(sanitize_pipeline_name("-leading-hyphen"), "p-leading-hyphen")
        self.assertEqual(sanitize_pipeline_name("double--hyphen"), "double-hyphen")
        self.assertEqual(sanitize_pipeline_name("version.1.0.0"), "version-1-0-0")
        
        # Edge cases
        self.assertEqual(sanitize_pipeline_name(""), "")
        self.assertEqual(sanitize_pipeline_name("."), "p-")  # p- because it needs to start with alphanumeric
        self.assertEqual(sanitize_pipeline_name("a" * 256), "a" * 255)  # Truncated
        
    def test_generate_pipeline_name(self):
        """Test that generate_pipeline_name generates valid pipeline names."""
        # Test with simple names
        name = generate_pipeline_name("test", "1.0")
        self.assertTrue(validate_pipeline_name(name))
        
        # Test with problematic names
        name = generate_pipeline_name("test.project", "1.0.0")
        self.assertTrue(validate_pipeline_name(name))
        self.assertNotIn(".", name)  # Should replace dots with hyphens
        
        # Test with long base name
        long_name = "x" * 250
        name = generate_pipeline_name(long_name, "1.0")
        self.assertTrue(validate_pipeline_name(name))
        self.assertLessEqual(len(name), 255)  # Should be truncated
        
        # Test with special characters
        name = generate_pipeline_name("test@project", "1.0")
        self.assertTrue(validate_pipeline_name(name))
        self.assertNotIn("@", name)  # Special chars should be removed

if __name__ == '__main__':
    unittest.main()
