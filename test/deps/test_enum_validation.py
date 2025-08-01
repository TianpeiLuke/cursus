"""
Unit tests for enum validation in base specifications module.

Tests the validation and functionality of enums including:
- NodeType validation
- DependencyType validation 
- Enum serialization/deserialization
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)


class TestEnumValidation(IsolatedTestCase):
    """Test cases for enum validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Just reset global state to ensure isolation
        super().setUp()

    def test_node_type_values(self):
        """Test that NodeType enum has expected values."""
        # Check enum values - lowercase in the actual implementation
        self.assertEqual(NodeType.SOURCE.value, "source")
        self.assertEqual(NodeType.INTERNAL.value, "internal")
        self.assertEqual(NodeType.SINK.value, "sink")
        self.assertEqual(NodeType.SINGULAR.value, "singular")
        
        # Check direct usage
        self.assertEqual(NodeType.SOURCE, NodeType.SOURCE)
        self.assertNotEqual(NodeType.SOURCE, NodeType.SINK)
    
    def test_dependency_type_values(self):
        """Test that DependencyType enum has expected values."""
        # Check some key enum values - lowercase in the actual implementation
        self.assertEqual(DependencyType.PROCESSING_OUTPUT.value, "processing_output")
        self.assertEqual(DependencyType.MODEL_ARTIFACTS.value, "model_artifacts")
        
        # Check direct usage
        self.assertEqual(DependencyType.PROCESSING_OUTPUT, DependencyType.PROCESSING_OUTPUT)
        self.assertNotEqual(DependencyType.PROCESSING_OUTPUT, DependencyType.MODEL_ARTIFACTS)
    
    def test_node_type_in_step_specification(self):
        """Test using NodeType in StepSpecification."""
        # Create specs with different node types using string inputs
        source_spec = StepSpecification(
            step_type="SourceStep",
            node_type="source",
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="test_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output"
            )]  # SOURCE node must have outputs
        )
        self.assertEqual(source_spec.node_type, NodeType.SOURCE)
        
        internal_spec = StepSpecification(
            step_type="ProcessingStep",
            node_type="internal",
            dependencies=[DependencySpec(
                logical_name="input",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )],
            outputs=[OutputSpec(
                logical_name="output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output"
            )]
        )
        self.assertEqual(internal_spec.node_type, NodeType.INTERNAL)
        
        sink_spec = StepSpecification(
            step_type="SinkStep",
            node_type="sink",
            dependencies=[DependencySpec(
                logical_name="input",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )],
            outputs=[]
        )
        self.assertEqual(sink_spec.node_type, NodeType.SINK)
    
    def test_dependency_type_in_dependency_spec(self):
        """Test using DependencyType in DependencySpec."""
        # Create dependency specs with different types
        processing_dep = DependencySpec(
            logical_name="proc_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT
        )
        self.assertEqual(processing_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
        
        model_dep = DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS
        )
        self.assertEqual(model_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
    
    def test_dependency_type_in_output_spec(self):
        """Test using DependencyType in OutputSpec."""
        # Create output specs with different types
        processing_output = OutputSpec(
            logical_name="proc_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.Output"
        )
        self.assertEqual(processing_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        model_output = OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.Output"
        )
        self.assertEqual(model_output.output_type, DependencyType.MODEL_ARTIFACTS)
    
    def test_enum_serialization(self):
        """Test serialization of enums in specifications."""
        # Create spec with enums using string input for node_type
        spec = StepSpecification(
            step_type="TestStep",
            node_type="internal",  # Use string input for node_type
            dependencies=[DependencySpec(
                logical_name="input",
                dependency_type=DependencyType.PROCESSING_OUTPUT
            )],
            outputs=[OutputSpec(
                logical_name="output",
                output_type=DependencyType.MODEL_ARTIFACTS,
                property_path="properties.Output"
            )]
        )
        
        # Serialize to dictionary
        spec_dict = spec.model_dump()
        
        # Verify enum values are preserved
        self.assertEqual(spec_dict["node_type"], NodeType.INTERNAL)
        
        # Get first values from dictionaries
        dep_key = next(iter(spec_dict["dependencies"].keys()))
        out_key = next(iter(spec_dict["outputs"].keys()))
        
        self.assertEqual(spec_dict["dependencies"][dep_key]["dependency_type"], DependencyType.PROCESSING_OUTPUT)
        self.assertEqual(spec_dict["outputs"][out_key]["output_type"], DependencyType.MODEL_ARTIFACTS)


if __name__ == '__main__':
    unittest.main()
