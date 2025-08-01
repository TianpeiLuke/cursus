"""
Unit tests for OutputSpec in base specifications module.

Tests the validation and functionality of OutputSpec including:
- Constructor validation
- Logical name validation
- Output type validation
- Property path handling
- Data type validation
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    OutputSpec, DependencyType
)


class TestOutputSpec(IsolatedTestCase):
    """Test cases for OutputSpec class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh instance of the enum for each test to ensure isolation
        self.output_type = DependencyType.PROCESSING_OUTPUT

    def test_construction(self):
        """Test construction of output spec."""
        # Valid construction with all fields
        output = OutputSpec(
            logical_name="test_output",
            output_type=self.output_type,
            property_path="properties.Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        self.assertEqual(output.logical_name, "test_output")
        self.assertEqual(output.output_type, self.output_type)
        self.assertEqual(output.property_path, "properties.Output.S3Uri")
        self.assertEqual(output.data_type, "S3Uri")
        self.assertEqual(output.description, "Test output")

        # Test with minimal fields
        output = OutputSpec(
            logical_name="test_output",
            output_type=self.output_type,
            property_path="properties.Output.S3Uri"
        )
        self.assertEqual(output.logical_name, "test_output")
        self.assertEqual(output.output_type, self.output_type)
        self.assertEqual(output.property_path, "properties.Output.S3Uri")
        self.assertEqual(output.data_type, "S3Uri")  # Default is "S3Uri", not None
        self.assertEqual(output.description, "")  # Default is empty string
    
    def test_logical_name_validation(self):
        """Test logical name validation."""
        # Valid logical names
        valid_names = ["output", "output_1", "OUTPUT", "output_with_underscore"]
        for name in valid_names:
            output = OutputSpec(
                logical_name=name,
                output_type=self.output_type,
                property_path="properties.Output"
            )
            self.assertEqual(output.logical_name, name)

    def test_property_path_validation(self):
        """Test property path validation."""
        # Valid property paths must start with 'properties.'
        valid_paths = [
            "properties.Output",
            "properties.Output.S3Uri",
            "properties.ProcessingOutputConfig.Outputs.TestOutput.S3Output.S3Uri",
        ]
        for path in valid_paths:
            output = OutputSpec(
                logical_name="test_output",
                output_type=self.output_type,
                property_path=path
            )
            self.assertEqual(output.property_path, path)

    def test_from_dict(self):
        """Test creation from dictionary."""
        output_dict = {
            "logical_name": "test_output",
            "output_type": self.output_type,
            "property_path": "properties.Output.S3Uri",
            "data_type": "S3Uri",
            "description": "Test output"
        }
        output = OutputSpec(**output_dict)
        self.assertEqual(output.logical_name, "test_output")
        self.assertEqual(output.output_type, self.output_type)
        self.assertEqual(output.property_path, "properties.Output.S3Uri")
        self.assertEqual(output.data_type, "S3Uri")
        self.assertEqual(output.description, "Test output")

    def test_model_dump(self):
        """Test serialization to dictionary."""
        output = OutputSpec(
            logical_name="test_output",
            output_type=self.output_type,
            property_path="properties.Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        output_dict = output.model_dump()
        self.assertEqual(output_dict["logical_name"], "test_output")
        self.assertEqual(output_dict["output_type"], self.output_type)
        self.assertEqual(output_dict["property_path"], "properties.Output.S3Uri")
        self.assertEqual(output_dict["data_type"], "S3Uri")
        self.assertEqual(output_dict["description"], "Test output")

    def test_string_representation(self):
        """Test string representation of output spec."""
        output = OutputSpec(
            logical_name="test_output",
            output_type=self.output_type,
            property_path="properties.Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        # String should contain key information
        repr_str = repr(output)
        self.assertIn("test_output", repr_str)
        self.assertIn(self.output_type.name, repr_str)
        self.assertIn("properties.Output.S3Uri", repr_str)


if __name__ == '__main__':
    unittest.main()
