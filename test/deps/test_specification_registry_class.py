"""
Unit tests for SpecificationRegistry class.

Tests the functionality of specification registry including:
- Registry creation
- Registering specifications
- Retrieving specifications by name and type
- Context isolation
"""

import unittest
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)
from src.pipeline_deps.specification_registry import SpecificationRegistry


class TestSpecificationRegistry(IsolatedTestCase):
    """Test cases for SpecificationRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Use string values for input but keep enum instances for comparison
        self.node_type_source_input = "source"
        self.node_type_internal_input = "internal"
        self.node_type_source = NodeType.SOURCE
        self.node_type_internal = NodeType.INTERNAL
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        # Create test specification
        self.dependency_spec = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type
        )
        
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.Output.S3Uri"
        )
        
        self.spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_internal_input,  # INTERNAL can have both dependencies and outputs
            dependencies=[self.dependency_spec],
            outputs=[self.output_spec]
        )
        
        # Create registry for testing
        self.registry = SpecificationRegistry(context_name="test_context")
    
    def test_registry_creation(self):
        """Test creation of registry."""
        self.assertEqual(self.registry.context_name, "test_context")
        self.assertEqual(len(self.registry._specifications), 0)
    
    def test_register_specification(self):
        """Test registering a specification."""
        # Register specification
        step_name = "test_step"
        self.registry.register(step_name, self.spec)
        
        # Check if registered
        self.assertEqual(len(self.registry._specifications), 1)
        self.assertIn(step_name, self.registry._specifications)
        self.assertEqual(self.registry._specifications[step_name], self.spec)
    
    def test_register_multiple_specifications(self):
        """Test registering multiple specifications."""
        # Register first spec
        first_name = "first_step"
        self.registry.register(first_name, self.spec)
        
        # Create and register second spec
        second_spec = StepSpecification(
            step_type="SecondStep",
            node_type=self.node_type_source_input,  # SOURCE must have outputs
            dependencies=[],
            outputs=[self.output_spec]
        )
        second_name = "second_step"
        self.registry.register(second_name, second_spec)
        
        # Check if both registered
        self.assertEqual(len(self.registry._specifications), 2)
        self.assertIn(first_name, self.registry._specifications)
        self.assertIn(second_name, self.registry._specifications)
        self.assertEqual(self.registry._specifications[first_name], self.spec)
        self.assertEqual(self.registry._specifications[second_name], second_spec)
    
    def test_get_specification(self):
        """Test retrieving specification by name."""
        # Register specification
        step_name = "test_step"
        self.registry.register(step_name, self.spec)
        
        # Get specification by name
        spec = self.registry.get_specification(step_name)
        
        # Check if correct
        self.assertEqual(spec, self.spec)
        self.assertEqual(spec.step_type, "TestStep")
        self.assertEqual(len(spec.dependencies), 1)
        self.assertEqual(len(spec.outputs), 1)
        
        # Access items from dictionaries
        output_name = next(iter(spec.outputs.keys()))
        self.assertEqual(spec.outputs[output_name].logical_name, "test_output")
    
    def test_get_specification_by_type(self):
        """Test retrieving specification by type."""
        # Create specs with same type
        spec1 = StepSpecification(
            step_type="SharedType",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]  # SOURCE must have outputs
        )
        spec2 = StepSpecification(
            step_type="SharedType",  # Same type as spec1
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]  # SOURCE must have outputs
        )
        spec3 = StepSpecification(
            step_type="UniqueType",  # Different type
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]  # SOURCE must have outputs
        )
        
        # Register specifications
        self.registry.register("step1", spec1)
        self.registry.register("step2", spec2)
        self.registry.register("step3", spec3)
        
        # Get specifications by type
        shared_specs = self.registry.get_specifications_by_type("SharedType")
        unique_specs = self.registry.get_specifications_by_type("UniqueType")
        
        # Check results
        self.assertEqual(len(shared_specs), 2)  # Should be 2 specs of type "SharedType"
        self.assertEqual(len(unique_specs), 1)  # Should be 1 spec of type "UniqueType"
    
    def test_context_isolation(self):
        """Test that registries with different contexts don't interfere."""
        # Create two registries with different contexts
        registry1 = SpecificationRegistry(context_name="context1")
        registry2 = SpecificationRegistry(context_name="context2")
        
        # Register specification in first registry
        registry1.register("step", self.spec)
        
        # Second registry should still be empty
        self.assertEqual(len(registry2._specifications), 0)
        
        # Register different specification in second registry
        other_spec = StepSpecification(
            step_type="OtherStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]  # SOURCE must have outputs
        )
        registry2.register("other_step", other_spec)
        
        # Check isolation
        self.assertEqual(len(registry1._specifications), 1)
        self.assertEqual(len(registry2._specifications), 1)
        self.assertIn("step", registry1._specifications)
        self.assertIn("other_step", registry2._specifications)
        self.assertNotIn("step", registry2._specifications)
        self.assertNotIn("other_step", registry1._specifications)
    
    def test_list_operations(self):
        """Test listing operations."""
        # Register specifications
        self.registry.register("step1", self.spec)
        
        other_spec = StepSpecification(
            step_type="OtherType",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[self.output_spec]
        )
        self.registry.register("step2", other_spec)
        
        # Test list_step_names
        step_names = self.registry.list_step_names()
        self.assertEqual(len(step_names), 2)
        self.assertIn("step1", step_names)
        self.assertIn("step2", step_names)
        
        # Test list_step_types
        step_types = self.registry.list_step_types()
        self.assertEqual(len(step_types), 2)
        self.assertIn("TestStep", step_types)
        self.assertIn("OtherType", step_types)
    
    def test_find_compatible_outputs(self):
        """Test finding compatible outputs."""
        # Register source step
        source_spec = StepSpecification(
            step_type="SourceStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="source_output",
                output_type=self.dependency_type,
                property_path="properties.Output.S3Uri",
                data_type="S3Uri"
            )]
        )
        self.registry.register("source", source_spec)
        
        # Create dependency spec to search for
        dep_spec = DependencySpec(
            logical_name="test_input",
            dependency_type=self.dependency_type,
            data_type="S3Uri",
            compatible_sources=["SourceStep"]  # Match source step type
        )
        
        # Find compatible outputs
        compatible = self.registry.find_compatible_outputs(dep_spec)
        
        # Should find the source output
        self.assertEqual(len(compatible), 1)
        self.assertEqual(compatible[0][0], "source")  # Step name
        self.assertEqual(compatible[0][1], "source_output")  # Output name


if __name__ == '__main__':
    unittest.main()
