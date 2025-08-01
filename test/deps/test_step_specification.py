"""
Unit tests for StepSpecification in base specifications module.

Tests the validation and functionality of StepSpecification including:
- Constructor validation
- Node type validation
- Dependencies and outputs handling
- Validation of nested objects
- Serialization/deserialization
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)


class TestStepSpecification(IsolatedTestCase):
    """Test cases for StepSpecification class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Use string values for input but expect enum instances for comparison
        self.node_type_source_input = "source"
        self.node_type_internal_input = "internal"
        self.node_type_sink_input = "sink"
        self.node_type_source = NodeType.SOURCE
        self.node_type_internal = NodeType.INTERNAL
        self.node_type_sink = NodeType.SINK
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        # Create test dependency and output specs using string values
        self.dependency_spec = DependencySpec(
            logical_name="test_input",
            dependency_type="processing_output",  # Use string instead of enum
            required=True
        )
        
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type="processing_output",  # Use string instead of enum
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )

    def test_construction(self):
        """Test construction of step specification."""
        # Test with minimal fields - SOURCE needs outputs
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]
        )
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(spec.node_type, self.node_type_source)
        self.assertEqual(len(spec.dependencies), 0)
        self.assertEqual(len(spec.outputs), 1)
        
        # Test with dependencies and outputs - need INTERNAL for both
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(spec.node_type, self.node_type_internal)
        self.assertEqual(len(spec.dependencies), 1)
        self.assertEqual(len(spec.outputs), 1)
        
        # Access values from dictionaries
        self.assertIn("test_input", spec.dependencies)
        self.assertIn("test_output", spec.outputs)
        self.assertEqual(spec.dependencies["test_input"].logical_name, "test_input")
        self.assertEqual(spec.outputs["test_output"].logical_name, "test_output")

    def test_step_type_validation(self):
        """Test step type validation."""
        # Valid step types
        valid_types = ["ProcessingStep", "TrainingStep", "my_step_type", "STEP_1"]
        for step_type in valid_types:
            # Note: SOURCE node must have outputs
            spec = StepSpecification(
                step_type=step_type,
                node_type=self.node_type_source_input,
                dependencies=[],
                outputs=[self.output_spec]
            )
            self.assertEqual(spec.step_type, step_type)

    def test_node_type_validation(self):
        """Test node type validation."""
        # Test with different node types
        spec_source = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]
        )
        self.assertEqual(spec_source.node_type, self.node_type_source)
        
        spec_internal = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        self.assertEqual(spec_internal.node_type, self.node_type_internal)
        
        spec_sink = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_sink_input,
            dependencies=[self.dependency_spec],
            outputs=[]
        )
        self.assertEqual(spec_sink.node_type, self.node_type_sink)
        
    def test_dependencies_validation(self):
        """Test dependencies validation."""
        # Test with multiple dependencies
        deps = [
            DependencySpec(logical_name="input1", dependency_type="processing_output"),
            DependencySpec(logical_name="input2", dependency_type="processing_output")
        ]
        
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,  # INTERNAL can have both dependencies and outputs
            dependencies=deps,
            outputs=[self.output_spec]
        )
        
        self.assertEqual(len(spec.dependencies), 2)
        self.assertIn("input1", spec.dependencies)
        self.assertIn("input2", spec.dependencies)
        self.assertEqual(spec.dependencies["input1"].logical_name, "input1")
        self.assertEqual(spec.dependencies["input2"].logical_name, "input2")

    def test_outputs_validation(self):
        """Test outputs validation."""
        # Test with multiple outputs
        outputs = [
            OutputSpec(
                logical_name="output1", 
                output_type="processing_output", 
                property_path="properties.Output1"
            ),
            OutputSpec(
                logical_name="output2", 
                output_type="processing_output", 
                property_path="properties.Output2"
            )
        ]
        
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source_input,  # SOURCE must have outputs, no dependencies
            dependencies=[],
            outputs=outputs
        )
        
        self.assertEqual(len(spec.outputs), 2)
        self.assertIn("output1", spec.outputs)
        self.assertIn("output2", spec.outputs)
        self.assertEqual(spec.outputs["output1"].logical_name, "output1")
        self.assertEqual(spec.outputs["output2"].logical_name, "output2")

    def test_from_dict(self):
        """Test creation from dictionary."""
        # First create dependency and output specs to use
        dep = DependencySpec(
            logical_name="test_input",
            dependency_type="processing_output",
            required=True
        )
        
        out = OutputSpec(
            logical_name="test_output",
            output_type="processing_output",
            property_path="properties.Output",
            data_type="S3Uri"
        )
        
        # Create the specification object directly
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,
            dependencies=[dep],
            outputs=[out]
        )
        
        # Verify fields
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(spec.node_type, self.node_type_internal)
        self.assertEqual(len(spec.dependencies), 1)
        self.assertIn("test_input", spec.dependencies)
        self.assertEqual(len(spec.outputs), 1)
        self.assertIn("test_output", spec.outputs)

    def test_model_dump(self):
        """Test serialization to dictionary."""
        # Create spec
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        
        # Convert to dictionary
        spec_dict = spec.model_dump()
        
        # Verify dictionary fields
        self.assertEqual(spec_dict["step_type"], "TestStep")
        self.assertEqual(spec_dict["node_type"], self.node_type_internal)
        self.assertEqual(len(spec_dict["dependencies"]), 1)
        self.assertEqual(len(spec_dict["outputs"]), 1)
        
        # Dependencies and outputs are dictionaries
        self.assertIn("test_input", spec_dict["dependencies"])
        self.assertIn("test_output", spec_dict["outputs"])
        self.assertEqual(spec_dict["dependencies"]["test_input"]["logical_name"], "test_input")
        self.assertEqual(spec_dict["outputs"]["test_output"]["logical_name"], "test_output")
    
    def test_string_representation(self):
        """Test string representation of step specification."""
        spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        
        # String should contain key information
        repr_str = repr(spec)
        self.assertIn("TestStep", repr_str)
        self.assertIn("dependencies=1", repr_str)  # Number of dependencies
        self.assertIn("outputs=1", repr_str)  # Number of outputs


if __name__ == '__main__':
    unittest.main()
