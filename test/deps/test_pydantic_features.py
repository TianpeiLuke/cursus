"""
Unit tests for Pydantic features in base specifications module.

Tests the validation and functionality of Pydantic features including:
- JSON serialization/deserialization
- Validation of nested models
- Schema generation
- Model copying and updating
"""

import unittest
import json
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)


class TestPydanticFeatures(IsolatedTestCase):
    """Test cases for Pydantic features used in specifications."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Use string values for input but expect enum instances for comparison
        self.node_type_internal_input = "internal"
        self.node_type_internal = NodeType.INTERNAL
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        # Create test specification with nested objects
        self.dependency_spec = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=True
        )
        
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        
        self.step_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,  # Use string input for node_type
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )

    def test_json_serialization(self):
        """Test JSON serialization of specifications."""
        # Convert to JSON
        json_str = self.step_spec.model_dump_json()
        
        # Should be valid JSON
        parsed_json = json.loads(json_str)
        
        # Check key fields
        self.assertEqual(parsed_json["step_type"], "TestStep")
        self.assertEqual(parsed_json["node_type"], "internal")  # Changed to INTERNAL
        
        # Dependencies and outputs are dictionaries in JSON
        self.assertEqual(len(parsed_json["dependencies"]), 1)
        self.assertEqual(len(parsed_json["outputs"]), 1)
        
        # Access the first item in the dictionary
        dependency_key = next(iter(parsed_json["dependencies"]))
        output_key = next(iter(parsed_json["outputs"]))
        
        self.assertEqual(parsed_json["dependencies"][dependency_key]["logical_name"], "test_input")
        self.assertEqual(parsed_json["outputs"][output_key]["logical_name"], "test_output")
    
    def test_json_deserialization(self):
        """Test JSON deserialization to specifications."""
        # First serialize to JSON
        json_str = self.step_spec.model_dump_json()
        
        # Deserialize back to model
        spec = StepSpecification.model_validate_json(json_str)
        
        # Verify model fields
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(spec.node_type, self.node_type_internal)
        self.assertEqual(len(spec.dependencies), 1)
        self.assertEqual(len(spec.outputs), 1)
        
        # Access values from the dictionary
        dependency_value = next(iter(spec.dependencies.values()))
        output_value = next(iter(spec.outputs.values()))
        
        self.assertEqual(dependency_value.logical_name, "test_input")
        self.assertEqual(output_value.logical_name, "test_output")
    
    def test_model_schema(self):
        """Test model schema generation."""
        # Get schema for step specification
        schema = StepSpecification.model_json_schema()
        
        # Verify schema properties
        self.assertIn("properties", schema)
        self.assertIn("step_type", schema["properties"])
        self.assertIn("node_type", schema["properties"])
        self.assertIn("dependencies", schema["properties"])
        self.assertIn("outputs", schema["properties"])
        
        # Verify required fields
        self.assertIn("required", schema)
        required = schema["required"]
        self.assertIn("step_type", required)
        self.assertIn("node_type", required)
    
    def test_model_copy(self):
        """Test model copying."""
        # Create a copy of the step spec
        copied_spec = self.step_spec.model_copy()
        
        # Should be a different object but with same field values
        self.assertIsNot(copied_spec, self.step_spec)
        self.assertEqual(copied_spec.step_type, self.step_spec.step_type)
        self.assertEqual(copied_spec.node_type, self.step_spec.node_type)
        self.assertEqual(len(copied_spec.dependencies), len(self.step_spec.dependencies))
        self.assertEqual(len(copied_spec.outputs), len(self.step_spec.outputs))
        
        # Get first items from dictionaries
        orig_dep_key = next(iter(self.step_spec.dependencies.keys()))
        orig_out_key = next(iter(self.step_spec.outputs.keys()))
        copy_dep_key = next(iter(copied_spec.dependencies.keys()))
        copy_out_key = next(iter(copied_spec.outputs.keys()))
        
        # Verify the keys are the same
        self.assertEqual(orig_dep_key, copy_dep_key)
        self.assertEqual(orig_out_key, copy_out_key)
        
        # Check values are equal (but we don't assert they're different objects, as the
        # actual implementation might not do a deep copy of nested objects)
        self.assertEqual(copied_spec.dependencies[copy_dep_key].logical_name, 
                       self.step_spec.dependencies[orig_dep_key].logical_name)
        self.assertEqual(copied_spec.outputs[copy_out_key].logical_name, 
                       self.step_spec.outputs[orig_out_key].logical_name)
    
    def test_model_update(self):
        """Test updating model fields."""
        # Create update data
        update_data = {
            "step_type": "UpdatedStep",
            "outputs": {}  # Empty the outputs dict
        }
        
        # Update the model
        updated_spec = self.step_spec.model_copy(update=update_data)
        
        # Verify updated fields
        self.assertEqual(updated_spec.step_type, "UpdatedStep")
        self.assertEqual(len(updated_spec.outputs), 0)
        
        # Fields not specified in update should remain the same
        self.assertEqual(updated_spec.node_type, self.step_spec.node_type)
        self.assertEqual(len(updated_spec.dependencies), len(self.step_spec.dependencies))
        
        # Check dependency values are preserved
        orig_dep_key = next(iter(self.step_spec.dependencies.keys()))
        updated_dep_key = next(iter(updated_spec.dependencies.keys()))
        self.assertEqual(updated_spec.dependencies[updated_dep_key].logical_name, 
                         self.step_spec.dependencies[orig_dep_key].logical_name)


if __name__ == '__main__':
    unittest.main()
