"""
Unit tests for the name_generator module.

This module tests the pipeline name generation, validation, and sanitization 
functionality to ensure conformance with SageMaker constraints.
"""

import pytest
from cursus.core.compiler.name_generator import (
    generate_random_word,
    generate_pipeline_name,
    validate_pipeline_name,
    sanitize_pipeline_name
)


class TestNameGenerator:
    """Tests for the name_generator module."""

    def test_generate_random_word_length(self):
        """Test that generate_random_word returns a word of the expected length."""
        word = generate_random_word(5)
        assert len(word) == 5
        
        word = generate_random_word(10)
        assert len(word) == 10
        
    def test_validate_pipeline_name(self):
        """Test that validate_pipeline_name correctly validates pipeline names."""
        # Valid names
        assert validate_pipeline_name("valid-name")
        assert validate_pipeline_name("valid-name-123")
        assert validate_pipeline_name("a")
        assert validate_pipeline_name("123")
        assert validate_pipeline_name("a" * 255)  # Maximum length
        
        # Invalid names
        assert not validate_pipeline_name("")  # Empty
        assert not validate_pipeline_name("-leading-hyphen")  # Leading hyphen
        assert not validate_pipeline_name("invalid.name")  # Contains dot
        assert not validate_pipeline_name("invalid_name")  # Contains underscore
        assert not validate_pipeline_name("invalid@name")  # Contains special char
        assert not validate_pipeline_name("a" * 256)  # Too long
        
    def test_sanitize_pipeline_name(self):
        """Test that sanitize_pipeline_name correctly sanitizes pipeline names."""
        # Names that should be unchanged
        assert sanitize_pipeline_name("valid-name") == "valid-name"
        assert sanitize_pipeline_name("valid-name-123") == "valid-name-123"
        
        # Names that should be sanitized
        assert sanitize_pipeline_name("invalid.name") == "invalid-name"
        assert sanitize_pipeline_name("invalid_name") == "invalid-name"
        assert sanitize_pipeline_name("invalid@name") == "invalidname"
        assert sanitize_pipeline_name("-leading-hyphen") == "p-leading-hyphen"
        assert sanitize_pipeline_name("double--hyphen") == "double-hyphen"
        assert sanitize_pipeline_name("version.1.0.0") == "version-1-0-0"
        
        # Edge cases
        assert sanitize_pipeline_name("") == ""
        assert sanitize_pipeline_name(".") == "p"  # p because trailing hyphens are removed
        assert sanitize_pipeline_name("a" * 256) == "a" * 255  # Truncated
        
    def test_generate_pipeline_name(self):
        """Test that generate_pipeline_name generates valid pipeline names."""
        # Test with simple names
        name = generate_pipeline_name("test", "1.0")
        assert validate_pipeline_name(name)
        
        # Test with problematic names
        name = generate_pipeline_name("test.project", "1.0.0")
        assert validate_pipeline_name(name)
        assert "." not in name  # Should replace dots with hyphens
        
        # Test with long base name
        long_name = "x" * 250
        name = generate_pipeline_name(long_name, "1.0")
        assert validate_pipeline_name(name)
        assert len(name) <= 255  # Should be truncated
        
        # Test with special characters
        name = generate_pipeline_name("test@project", "1.0")
        assert validate_pipeline_name(name)
        assert "@" not in name  # Special chars should be removed
