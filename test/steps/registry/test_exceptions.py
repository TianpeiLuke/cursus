"""Unit tests for the registry exceptions module."""

import unittest
from src.cursus.steps.registry.exceptions import RegistryError


class TestRegistryExceptions(unittest.TestCase):
    """Test case for registry exception classes."""

    def test_registry_error_basic(self):
        """Test basic RegistryError functionality."""
        message = "Test error message"
        error = RegistryError(message)
        
        self.assertEqual(str(error), message)
        self.assertEqual(error.unresolvable_types, [])
        self.assertEqual(error.available_builders, [])

    def test_registry_error_with_unresolvable_types(self):
        """Test RegistryError with unresolvable types."""
        message = "Cannot resolve step types"
        unresolvable_types = ["UnknownStep1", "UnknownStep2"]
        
        error = RegistryError(message, unresolvable_types=unresolvable_types)
        
        self.assertEqual(error.unresolvable_types, unresolvable_types)
        self.assertIn("UnknownStep1", str(error))
        self.assertIn("UnknownStep2", str(error))
        self.assertIn("Unresolvable step types", str(error))

    def test_registry_error_with_available_builders(self):
        """Test RegistryError with available builders."""
        message = "Builder not found"
        available_builders = ["Builder1", "Builder2", "Builder3"]
        
        error = RegistryError(message, available_builders=available_builders)
        
        self.assertEqual(error.available_builders, available_builders)
        self.assertIn("Builder1", str(error))
        self.assertIn("Builder2", str(error))
        self.assertIn("Builder3", str(error))
        self.assertIn("Available builders", str(error))

    def test_registry_error_with_both_parameters(self):
        """Test RegistryError with both unresolvable types and available builders."""
        message = "Complex error"
        unresolvable_types = ["BadStep"]
        available_builders = ["GoodBuilder1", "GoodBuilder2"]
        
        error = RegistryError(
            message, 
            unresolvable_types=unresolvable_types,
            available_builders=available_builders
        )
        
        error_str = str(error)
        self.assertIn(message, error_str)
        self.assertIn("BadStep", error_str)
        self.assertIn("GoodBuilder1", error_str)
        self.assertIn("GoodBuilder2", error_str)
        self.assertIn("Unresolvable step types", error_str)
        self.assertIn("Available builders", error_str)

    def test_registry_error_inheritance(self):
        """Test that RegistryError properly inherits from Exception."""
        error = RegistryError("Test message")
        
        self.assertIsInstance(error, Exception)
        self.assertTrue(issubclass(RegistryError, Exception))

    def test_registry_error_empty_lists(self):
        """Test RegistryError with empty lists."""
        message = "Test with empty lists"
        error = RegistryError(
            message,
            unresolvable_types=[],
            available_builders=[]
        )
        
        # Should only show the main message, not the empty list sections
        error_str = str(error)
        self.assertEqual(error_str, message)
        self.assertNotIn("Unresolvable step types", error_str)
        self.assertNotIn("Available builders", error_str)


if __name__ == '__main__':
    unittest.main()
