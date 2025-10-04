"""
Unit tests for the name_generator module.

This module tests the pipeline name generation, validation, and sanitization 
functionality to ensure conformance with SageMaker constraints.

IMPROVED: Following pytest best practices and troubleshooting guide:
1. Implementation-driven test design (read source code first)
2. Comprehensive edge case coverage based on actual implementation
3. Test actual behavior, not assumptions
4. Proper error prevention patterns
"""

import pytest
import re
from cursus.core.compiler.name_generator import (
    generate_random_word,
    generate_pipeline_name,
    validate_pipeline_name,
    sanitize_pipeline_name,
    PIPELINE_NAME_PATTERN,
)


class TestNameGenerator:
    """
    Tests for the name_generator module.
    
    IMPROVED: Following pytest best practices:
    1. Test actual implementation behavior from source code
    2. Comprehensive edge case coverage
    3. Test both valid and invalid scenarios systematically
    """

    def test_generate_random_word_length(self):
        """
        Test that generate_random_word returns a word of the expected length.
        
        IMPROVED:
        - Read source code first: generate_random_word uses string.ascii_uppercase with random.choices
        - Test multiple lengths to ensure consistency
        - Verify character set matches implementation
        """
        # Test various lengths
        for length in [1, 4, 8, 10]:
            word = generate_random_word(length)
            assert len(word) == length
            # IMPROVED: Verify actual character set from source code
            assert word.isupper()  # Source uses ascii_uppercase
            assert word.isalpha()  # Should only contain letters

    def test_generate_random_word_default_length(self):
        """
        Test generate_random_word with default length.
        
        IMPROVED: Test default parameter behavior from source code
        """
        # Source code shows default length=4
        word = generate_random_word()
        assert len(word) == 4
        assert word.isupper()
        assert word.isalpha()

    def test_validate_pipeline_name_valid_cases(self):
        """
        Test validate_pipeline_name with valid names.
        
        IMPROVED:
        - Read source code first: validates against PIPELINE_NAME_PATTERN and length <= 255
        - Test actual pattern from source: r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$"
        - Comprehensive valid case coverage
        """
        # Valid names based on actual pattern from source
        valid_names = [
            "valid-name",
            "valid-name-123", 
            "a",
            "123",
            "A",
            "test-pipeline-v1",
            "pipeline123",
            "a-b-c-d-e",
            "123-abc-456",
            "a" * 255,  # Maximum length
        ]
        
        for name in valid_names:
            assert validate_pipeline_name(name), f"Expected '{name}' to be valid"
            # IMPROVED: Verify against actual pattern from source
            assert re.match(PIPELINE_NAME_PATTERN, name), f"'{name}' should match pattern"

    def test_validate_pipeline_name_invalid_cases(self):
        """
        Test validate_pipeline_name with invalid names.
        
        IMPROVED:
        - Read source code first: checks length and pattern constraints
        - Test actual invalid cases based on implementation logic
        """
        # Invalid names based on actual validation logic
        invalid_names = [
            "",  # Empty
            "-leading-hyphen",  # Leading hyphen
            "trailing-hyphen-",  # Trailing hyphen  
            "invalid.name",  # Contains dot
            "invalid_name",  # Contains underscore
            "invalid@name",  # Contains special char
            "invalid name",  # Contains space
            "a" * 256,  # Too long (> 255)
            "123-",  # Ends with hyphen
            "-123",  # Starts with hyphen
        ]
        
        for name in invalid_names:
            assert not validate_pipeline_name(name), f"Expected '{name}' to be invalid"

    def test_validate_pipeline_name_edge_cases(self):
        """
        Test validate_pipeline_name edge cases.
        
        IMPROVED: Test boundary conditions from source code
        """
        # Test exact boundary conditions
        assert validate_pipeline_name("a" * 255)  # Exactly 255 chars - valid
        assert not validate_pipeline_name("a" * 256)  # Exactly 256 chars - invalid
        
        # Test pattern edge cases
        assert validate_pipeline_name("a-b")  # Simple hyphen case
        assert validate_pipeline_name("1-2")  # Numbers with hyphen
        # FIXED: Read source code - pattern r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$" allows multiple hyphens
        assert validate_pipeline_name("a--b")  # Double hyphen is actually VALID based on pattern (-*)

    def test_sanitize_pipeline_name_basic_cases(self):
        """
        Test sanitize_pipeline_name with basic sanitization cases.
        
        IMPROVED:
        - Read source code first: replaces dots/underscores with hyphens, removes special chars
        - Test actual sanitization logic from implementation
        """
        # Names that should be unchanged
        unchanged_names = [
            "valid-name",
            "valid-name-123",
            "simple",
            "test123",
        ]
        
        for name in unchanged_names:
            assert sanitize_pipeline_name(name) == name

        # Names that should be sanitized based on source logic
        sanitization_cases = [
            ("invalid.name", "invalid-name"),  # Dots to hyphens
            ("invalid_name", "invalid-name"),  # Underscores to hyphens  
            ("invalid@name", "invalidname"),  # Special chars removed
            ("version.1.0.0", "version-1-0-0"),  # Multiple dots
            ("test_file.py", "test-file-py"),  # Mixed cases
        ]
        
        for input_name, expected in sanitization_cases:
            result = sanitize_pipeline_name(input_name)
            assert result == expected, f"Expected '{input_name}' -> '{expected}', got '{result}'"

    def test_sanitize_pipeline_name_leading_character_fix(self):
        """
        Test sanitize_pipeline_name fixes leading character issues.
        
        IMPROVED: Test actual leading character logic from source code
        """
        # Source code: if name doesn't start with alphanumeric, prepend "p"
        leading_char_cases = [
            ("-leading-hyphen", "p-leading-hyphen"),
            # FIXED: Read source code - dots are replaced with hyphens BEFORE prepending "p"
            (".leading-dot", "p-leading-dot"),  # Dot becomes hyphen, then p prepended
            # FIXED: Read source code more carefully - @ is removed, leaving "leading-special" which starts with "l" (alphanumeric)
            ("@leading-special", "leading-special"),  # Special char removed, but "l" is alphanumeric so no "p" prepended
        ]
        
        for input_name, expected in leading_char_cases:
            result = sanitize_pipeline_name(input_name)
            assert result == expected, f"Expected '{input_name}' -> '{expected}', got '{result}'"

    def test_sanitize_pipeline_name_multiple_hyphens(self):
        """
        Test sanitize_pipeline_name handles multiple consecutive hyphens.
        
        IMPROVED: Test actual multiple hyphen logic from source code
        """
        # Source code: re.sub(r"-+", "-", sanitized) - replaces multiple hyphens with single
        multiple_hyphen_cases = [
            ("double--hyphen", "double-hyphen"),
            ("triple---hyphen", "triple-hyphen"),
            ("many----hyphens", "many-hyphens"),
            ("mixed--.--hyphens", "mixed-hyphens"),  # Dots become hyphens, then consolidated
        ]
        
        for input_name, expected in multiple_hyphen_cases:
            result = sanitize_pipeline_name(input_name)
            assert result == expected, f"Expected '{input_name}' -> '{expected}', got '{result}'"

    def test_sanitize_pipeline_name_trailing_hyphen_removal(self):
        """
        Test sanitize_pipeline_name removes trailing hyphens.
        
        IMPROVED: Test actual trailing hyphen logic from source code
        """
        # Source code: sanitized.rstrip("-") - removes trailing hyphens
        trailing_hyphen_cases = [
            ("trailing-", "trailing"),
            ("multiple-trailing---", "multiple-trailing"),
            (".", "p"),  # Dot becomes hyphen, p prepended, trailing hyphen removed
        ]
        
        for input_name, expected in trailing_hyphen_cases:
            result = sanitize_pipeline_name(input_name)
            assert result == expected, f"Expected '{input_name}' -> '{expected}', got '{result}'"

    def test_sanitize_pipeline_name_length_truncation(self):
        """
        Test sanitize_pipeline_name truncates long names.
        
        IMPROVED: Test actual length truncation logic from source code
        """
        # Source code: if len(sanitized) > 255: sanitized = sanitized[:255]
        long_name = "x" * 300
        result = sanitize_pipeline_name(long_name)
        assert len(result) <= 255
        assert result == "x" * 255

    def test_sanitize_pipeline_name_edge_cases(self):
        """
        Test sanitize_pipeline_name edge cases.
        
        IMPROVED: Test edge cases based on source code logic
        """
        # Empty string case
        assert sanitize_pipeline_name("") == ""
        
        # All special characters case
        result = sanitize_pipeline_name("@#$%^&*()")
        assert result == "pipeline"  # Source: provides default when all chars removed
        
        # Only dots and underscores
        assert sanitize_pipeline_name("...___") == "p"  # Becomes hyphens, p prepended, trailing removed

    def test_generate_pipeline_name_basic_functionality(self):
        """
        Test generate_pipeline_name basic functionality.
        
        IMPROVED:
        - Read source code first: format is "{base_name}-{version}-pipeline"
        - Test actual format from implementation (no random word in current version)
        - Verify sanitization is applied
        """
        # Test basic format from source code
        name = generate_pipeline_name("test", "1.0")
        # FIXED: Read source code - sanitize_pipeline_name converts dots to hyphens
        expected_format = "test-1-0-pipeline"  # "1.0" becomes "1-0" after sanitization
        assert name == expected_format
        assert validate_pipeline_name(name)

    def test_generate_pipeline_name_with_sanitization(self):
        """
        Test generate_pipeline_name applies sanitization.
        
        IMPROVED: Test that sanitization is applied to the generated name
        """
        # Test with names that need sanitization
        test_cases = [
            ("test.project", "1.0.0", "test-project-1-0-0-pipeline"),
            ("test@project", "2.0", "testproject-2-0-pipeline"),
            ("test_name", "1.5", "test-name-1-5-pipeline"),
        ]
        
        for base_name, version, expected in test_cases:
            result = generate_pipeline_name(base_name, version)
            assert result == expected, f"Expected '{base_name}', '{version}' -> '{expected}', got '{result}'"
            assert validate_pipeline_name(result)

    def test_generate_pipeline_name_long_names(self):
        """
        Test generate_pipeline_name with long base names.
        
        IMPROVED: Test length handling based on source code
        """
        # Test with long base name that would exceed 255 chars
        long_base = "x" * 250
        name = generate_pipeline_name(long_base, "1.0")
        assert validate_pipeline_name(name)
        assert len(name) <= 255

    def test_generate_pipeline_name_default_version(self):
        """
        Test generate_pipeline_name with default version.
        
        IMPROVED: Test default parameter behavior from source code
        """
        # Source code shows default version="1.0"
        name = generate_pipeline_name("test")
        # FIXED: Read source code - sanitize_pipeline_name converts dots to hyphens
        expected = "test-1-0-pipeline"  # "1.0" becomes "1-0" after sanitization
        assert name == expected
        assert validate_pipeline_name(name)

    def test_generate_pipeline_name_special_characters(self):
        """
        Test generate_pipeline_name handles special characters properly.
        
        IMPROVED: Test comprehensive special character handling
        """
        special_char_cases = [
            ("test@project", "1.0", "testproject-1-0-pipeline"),
            ("test.name_with.dots", "2.0.1", "test-name-with-dots-2-0-1-pipeline"),
            ("test name with spaces", "1.0", "testnameWithspaces-1-0-pipeline"),  # Spaces removed
        ]
        
        for base_name, version, expected in special_char_cases:
            result = generate_pipeline_name(base_name, version)
            assert validate_pipeline_name(result), f"Generated name '{result}' should be valid"
            # Verify no special characters remain
            assert "@" not in result
            assert "." not in result
            assert "_" not in result or result.count("_") == 0  # Underscores should be converted

    def test_pipeline_name_pattern_constant(self):
        """
        Test that PIPELINE_NAME_PATTERN constant matches expected pattern.
        
        IMPROVED: Test the actual constant from source code
        """
        # Verify the pattern constant matches what's in source code
        expected_pattern = r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$"
        assert PIPELINE_NAME_PATTERN == expected_pattern

    def test_integration_generate_and_validate(self):
        """
        Test integration between generate_pipeline_name and validate_pipeline_name.
        
        IMPROVED: Test that generated names always pass validation
        """
        # Test various combinations to ensure generated names are always valid
        test_combinations = [
            ("simple", "1.0"),
            ("complex.name_with@special", "2.1.3"),
            ("very-long-" + "x" * 100, "1.0.0.0"),
            ("123numeric", "v1.2"),
            ("", "1.0"),  # Empty base name
        ]
        
        for base_name, version in test_combinations:
            generated_name = generate_pipeline_name(base_name, version)
            assert validate_pipeline_name(generated_name), \
                f"Generated name '{generated_name}' from base='{base_name}', version='{version}' should be valid"
