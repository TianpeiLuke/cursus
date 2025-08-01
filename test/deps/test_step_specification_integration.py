"""
Unit tests for StepSpecification integration with other components.

Tests the integration aspects of StepSpecification including:
- Integration with property references
- Integration with script contracts
- Integration with dependency resolvers
- End-to-end step construction with nested components
"""

import unittest
import json
from typing import List, Dict
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)
from src.pipeline_deps.property_reference import PropertyReference


class TestStepSpecificationIntegration(IsolatedTestCase):
    """Test integration aspects of StepSpecification."""

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
        
        # Create test specification components
        self.dependency_spec = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            required=True
        )
        
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        
        # Create an output spec for property references
        self.source_output_spec = OutputSpec(
            logical_name="source_output",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['DataOutput'].S3Output.S3Uri",
            data_type="S3Uri"
        )

    def test_property_reference_integration(self):
        """Test integration with property references."""
        # Create a property reference
        prop_ref = PropertyReference(
            step_name="source_step",
            output_spec=self.source_output_spec  # Use the output spec we created in setUp
        )
        
        # Create step specification that would use this reference
        spec = StepSpecification(
            step_type="ProcessingStep",
            node_type=self.node_type_internal_input,
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        
        # Simulate resolving a property reference
        # We'll call its string representation to get the reference string
        resolved_value = str(prop_ref)
        self.assertIn("source_step", resolved_value)
        self.assertIn("source_output", resolved_value)
        
        # Also test repr method
        repr_value = repr(prop_ref)
        self.assertIn("source_step", repr_value)
        self.assertIn("source_output", repr_value)
    
    def test_end_to_end_step_creation(self):
        """Test end-to-end step specification creation with all components."""
        # Create a complex step specification with multiple dependencies and outputs
        dependencies = [
            DependencySpec(
                logical_name="input_data",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                required=True
            ),
            DependencySpec(
                logical_name="model",
                dependency_type=DependencyType.MODEL_ARTIFACTS,
                required=False
            )
        ]
        
        outputs = [
            OutputSpec(
                logical_name="processed_data",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
                data_type="S3Uri"
            ),
            OutputSpec(
                logical_name="metrics",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['Metrics'].S3Output.S3Uri",
                data_type="Metrics"
            )
        ]
        
        # Create the specification
        spec = StepSpecification(
            step_type="ComplexProcessingStep",
            node_type=self.node_type_internal_input,
            dependencies=dependencies,
            outputs=outputs
        )
        
        # Verify basic properties
        self.assertEqual(spec.step_type, "ComplexProcessingStep")
        self.assertEqual(spec.node_type, self.node_type_internal)
        self.assertEqual(len(spec.dependencies), 2)
        self.assertEqual(len(spec.outputs), 2)
            
        # Convert to JSON to test serialization of the complete object
        try:
            json_str = spec.model_dump_json()
            parsed = json.loads(json_str)
            
            # Verify key components in JSON
            self.assertEqual(parsed["step_type"], "ComplexProcessingStep")
            self.assertEqual(len(parsed["dependencies"]), 2)
            self.assertEqual(len(parsed["outputs"]), 2)
            
            # Check if dependency names are in the dictionary
            dep_keys = parsed["dependencies"].keys()
            self.assertIn("input_data", dep_keys)
            self.assertIn("model", dep_keys)
            
            # Check if output names are in the dictionary
            out_keys = parsed["outputs"].keys()
            self.assertIn("processed_data", out_keys)
            self.assertIn("metrics", out_keys)
            
        except Exception as e:
            # Handle the case where model_dump_json might not be available
            # or custom fields aren't properly serialized
            self.fail(f"JSON serialization failed: {str(e)}")
    
    def test_dependency_resolution_simulation(self):
        """Test simulation of dependency resolution."""
        # Create source step with output
        source_spec = StepSpecification(
            step_type="SourceStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="source_output",
                output_type=self.dependency_type,
                property_path="properties.Output.S3Uri"
            )]
        )
        
        # Create sink step with dependency
        sink_spec = StepSpecification(
            step_type="SinkStep",
            node_type=self.node_type_sink_input,
            dependencies=[DependencySpec(
                logical_name="sink_input",
                dependency_type=self.dependency_type
            )],
            outputs=[]
        )
        
        # Manually simulate dependency resolution
        # In a real scenario, this would be done by a dependency resolver
        
        # 1. Get output from source step
        source_output = next(iter(source_spec.outputs.values()))
        
        # 2. Get dependency from sink step
        sink_dependency = next(iter(sink_spec.dependencies.values()))
        
        # 3. Check if they can be connected based on type
        self.assertEqual(source_output.output_type, sink_dependency.dependency_type)
        
        # 4. Simulate creating a connection/reference
        connection = {
            "from_step": "source",
            "from_output": source_output.logical_name,
            "to_step": "sink",
            "to_dependency": sink_dependency.logical_name
        }
        
        # Verify connection details
        self.assertEqual(connection["from_output"], "source_output")
        self.assertEqual(connection["to_dependency"], "sink_input")


if __name__ == '__main__':
    unittest.main()
