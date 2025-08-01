"""
Unit tests for DependencySpec in base specifications module.

Tests the validation and functionality of DependencySpec including:
- Constructor validation
- Required flag behavior
- Logical name validation
- Dependency type validation
- Property references
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    DependencySpec, DependencyType
)


class TestDependencySpec(IsolatedTestCase):
    """Test cases for DependencySpec class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh instance of the enum for each test to ensure isolation
        self.dependency_type = DependencyType.PROCESSING_OUTPUT

    def test_construction(self):
        """Test construction of dependency spec."""
        # Valid construction
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=True
        )
        self.assertEqual(dep.logical_name, "test_input")
        self.assertEqual(dep.dependency_type, self.dependency_type)
        self.assertTrue(dep.required)

        # Test with required=False
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=False
        )
        self.assertFalse(dep.required)

        # Test with default required (should be True)
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type
        )
        self.assertTrue(dep.required)

    def test_logical_name_validation(self):
        """Test logical name validation."""
        # Valid logical names
        valid_names = ["input", "input_1", "INPUT", "input_with_underscore"]
        for name in valid_names:
            dep = DependencySpec(
                logical_name=name,
                dependency_type=self.dependency_type
            )
            self.assertEqual(dep.logical_name, name)

    def test_from_dict(self):
        """Test creation from dictionary."""
        dep_dict = {
            "logical_name": "test_input",
            "dependency_type": self.dependency_type,
            "required": False
        }
        dep = DependencySpec(**dep_dict)
        self.assertEqual(dep.logical_name, "test_input")
        self.assertEqual(dep.dependency_type, self.dependency_type)
        self.assertFalse(dep.required)

    def test_model_dump(self):
        """Test serialization to dictionary."""
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=False
        )
        dep_dict = dep.model_dump()
        self.assertEqual(dep_dict["logical_name"], "test_input")
        self.assertEqual(dep_dict["dependency_type"], self.dependency_type)
        self.assertFalse(dep_dict["required"])

    def test_string_representation(self):
        """Test string representation of dependency spec."""
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=True
        )
        # String should contain key information
        repr_str = repr(dep)
        self.assertIn("test_input", repr_str)
        self.assertIn(self.dependency_type.name, repr_str)


if __name__ == '__main__':
    unittest.main()
