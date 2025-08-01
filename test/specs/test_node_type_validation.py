#!/usr/bin/env python3
"""
Unit tests for enhanced step specifications with node types.

This module provides comprehensive unit tests for the node type system,
including validation logic, constraint enforcement, and specification correctness.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, 
    DependencyType, NodeType
)
from src.pipeline_deps.specification_registry import SpecificationRegistry


class TestNodeTypeSystem(unittest.TestCase):
    """Test cases for the node type system and validation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dependency = DependencySpec(
            logical_name="test_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            data_type="S3Uri",
            description="Test dependency"
        )
        
        self.mock_output = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.TestOutput.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
    
    def test_source_node_validation_success(self):
        """Test that SOURCE nodes with no dependencies and outputs pass validation."""
        spec = StepSpecification(
            step_type="TestSource",
            node_type=NodeType.SOURCE,
            dependencies=[],  # No dependencies for SOURCE
            outputs=[self.mock_output]  # Must have outputs
        )
        
        errors = spec.validate()
        self.assertEqual(len(errors), 0, "SOURCE node should pass validation")
        self.assertEqual(spec.node_type, NodeType.SOURCE)
    
    def test_source_node_validation_fail_with_dependencies(self):
        """Test that SOURCE nodes with dependencies fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSource",
                node_type=NodeType.SOURCE,
                dependencies=[self.mock_dependency],  # Should not have dependencies
                outputs=[self.mock_output]
            )
        
        self.assertIn("SOURCE node", str(context.exception))
        self.assertIn("cannot have dependencies", str(context.exception))
    
    def test_source_node_validation_fail_without_outputs(self):
        """Test that SOURCE nodes without outputs fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSource",
                node_type=NodeType.SOURCE,
                dependencies=[],
                outputs=[]  # Should have outputs
            )
        
        self.assertIn("SOURCE node", str(context.exception))
        self.assertIn("must have outputs", str(context.exception))
    
    def test_internal_node_validation_success(self):
        """Test that INTERNAL nodes with both dependencies and outputs pass validation."""
        spec = StepSpecification(
            step_type="TestInternal",
            node_type=NodeType.INTERNAL,
            dependencies=[self.mock_dependency],  # Must have dependencies
            outputs=[self.mock_output]  # Must have outputs
        )
        
        errors = spec.validate()
        self.assertEqual(len(errors), 0, "INTERNAL node should pass validation")
        self.assertEqual(spec.node_type, NodeType.INTERNAL)
    
    def test_internal_node_validation_fail_without_dependencies(self):
        """Test that INTERNAL nodes without dependencies fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestInternal",
                node_type=NodeType.INTERNAL,
                dependencies=[],  # Should have dependencies
                outputs=[self.mock_output]
            )
        
        self.assertIn("INTERNAL node", str(context.exception))
        self.assertIn("must have dependencies", str(context.exception))
    
    def test_internal_node_validation_fail_without_outputs(self):
        """Test that INTERNAL nodes without outputs fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestInternal",
                node_type=NodeType.INTERNAL,
                dependencies=[self.mock_dependency],
                outputs=[]  # Should have outputs
            )
        
        self.assertIn("INTERNAL node", str(context.exception))
        self.assertIn("must have outputs", str(context.exception))
    
    def test_sink_node_validation_success(self):
        """Test that SINK nodes with dependencies and no outputs pass validation."""
        spec = StepSpecification(
            step_type="TestSink",
            node_type=NodeType.SINK,
            dependencies=[self.mock_dependency],  # Must have dependencies
            outputs=[]  # Should not have outputs
        )
        
        errors = spec.validate()
        self.assertEqual(len(errors), 0, "SINK node should pass validation")
        self.assertEqual(spec.node_type, NodeType.SINK)
    
    def test_sink_node_validation_fail_without_dependencies(self):
        """Test that SINK nodes without dependencies fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSink",
                node_type=NodeType.SINK,
                dependencies=[],  # Should have dependencies
                outputs=[]
            )
        
        self.assertIn("SINK node", str(context.exception))
        self.assertIn("must have dependencies", str(context.exception))
    
    def test_sink_node_validation_fail_with_outputs(self):
        """Test that SINK nodes with outputs fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSink",
                node_type=NodeType.SINK,
                dependencies=[self.mock_dependency],
                outputs=[self.mock_output]  # Should not have outputs
            )
        
        self.assertIn("SINK node", str(context.exception))
        self.assertIn("cannot have outputs", str(context.exception))
    
    def test_singular_node_validation_success(self):
        """Test that SINGULAR nodes with no dependencies and no outputs pass validation."""
        spec = StepSpecification(
            step_type="TestSingular",
            node_type=NodeType.SINGULAR,
            dependencies=[],  # Should not have dependencies
            outputs=[]  # Should not have outputs
        )
        
        errors = spec.validate()
        self.assertEqual(len(errors), 0, "SINGULAR node should pass validation")
        self.assertEqual(spec.node_type, NodeType.SINGULAR)
    
    def test_singular_node_validation_fail_with_dependencies(self):
        """Test that SINGULAR nodes with dependencies fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSingular",
                node_type=NodeType.SINGULAR,
                dependencies=[self.mock_dependency],  # Should not have dependencies
                outputs=[]
            )
        
        self.assertIn("SINGULAR node", str(context.exception))
        self.assertIn("cannot have dependencies", str(context.exception))
    
    def test_singular_node_validation_fail_with_outputs(self):
        """Test that SINGULAR nodes with outputs fail validation."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestSingular",
                node_type=NodeType.SINGULAR,
                dependencies=[],
                outputs=[self.mock_output]  # Should not have outputs
            )
        
        self.assertIn("SINGULAR node", str(context.exception))
        self.assertIn("cannot have outputs", str(context.exception))
    
    def test_invalid_node_type_raises_error(self):
        """Test that invalid node types raise ValueError."""
        with self.assertRaises(ValueError) as context:
            StepSpecification(
                step_type="TestInvalid",
                node_type="invalid_type",  # Should be NodeType enum
                dependencies=[],
                outputs=[]
            )
        
        self.assertIn("Input should be 'source', 'internal', 'sink' or 'singular'", str(context.exception))


class TestSpecificationRegistry(unittest.TestCase):
    """Test cases for the specification registry with node types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        
        # Create mock specifications for different node types
        self.source_spec = Mock(spec=StepSpecification)
        self.source_spec.step_type = "TestSource"
        self.source_spec.node_type = NodeType.SOURCE
        self.source_spec.validate.return_value = []
        self.source_spec.dependencies = {}
        self.source_spec.outputs = {"output1": Mock()}
        
        self.internal_spec = Mock(spec=StepSpecification)
        self.internal_spec.step_type = "TestInternal"
        self.internal_spec.node_type = NodeType.INTERNAL
        self.internal_spec.validate.return_value = []
        self.internal_spec.dependencies = {"input1": Mock()}
        self.internal_spec.outputs = {"output1": Mock()}
        
        self.sink_spec = Mock(spec=StepSpecification)
        self.sink_spec.step_type = "TestSink"
        self.sink_spec.node_type = NodeType.SINK
        self.sink_spec.validate.return_value = []
        self.sink_spec.dependencies = {"input1": Mock()}
        self.sink_spec.outputs = {}
    
    def test_register_valid_specifications(self):
        """Test registering valid specifications of different node types."""
        # Register specifications
        self.registry.register("source_step", self.source_spec)
        self.registry.register("internal_step", self.internal_spec)
        self.registry.register("sink_step", self.sink_spec)
        
        # Verify registration
        self.assertEqual(len(self.registry.list_step_names()), 3)
        self.assertIn("source_step", self.registry.list_step_names())
        self.assertIn("internal_step", self.registry.list_step_names())
        self.assertIn("sink_step", self.registry.list_step_names())
    
    def test_register_invalid_specification_fails(self):
        """Test that registering invalid specifications raises error."""
        invalid_spec = Mock(spec=StepSpecification)
        invalid_spec.validate.return_value = ["Validation error"]
        
        with self.assertRaises(ValueError) as context:
            self.registry.register("invalid_step", invalid_spec)
        
        self.assertIn("Invalid specification", str(context.exception))
    
    @patch('src.pipeline_deps.base_specifications.logger')
    def test_registration_logging(self, mock_logger):
        """Test that registration logs appropriate messages."""
        self.registry.register("test_step", self.source_spec)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("Registered specification", call_args)
        self.assertIn("test_step", call_args)
        self.assertIn("TestSource", call_args)


class TestRealSpecifications(unittest.TestCase):
    """Test cases for actual step specifications."""
    
    def setUp(self):
        """Set up real specifications for testing."""
        # Import real specifications
        try:
            from pipeline_step_specs.data_loading_spec import DATA_LOADING_SPEC
            from pipeline_step_specs.preprocessing_spec import PREPROCESSING_SPEC
            from pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
            from pipeline_step_specs.packaging_spec import PACKAGING_SPEC
            from pipeline_step_specs.payload_spec import PAYLOAD_SPEC
            from pipeline_step_specs.registration_spec import REGISTRATION_SPEC
            from pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC
            
            self.real_specs = {
                "DATA_LOADING": (DATA_LOADING_SPEC, NodeType.SOURCE),
                "PREPROCESSING": (PREPROCESSING_SPEC, NodeType.INTERNAL),
                "XGBOOST_TRAINING": (XGBOOST_TRAINING_SPEC, NodeType.INTERNAL),
                "PACKAGING": (PACKAGING_SPEC, NodeType.INTERNAL),
                "PAYLOAD": (PAYLOAD_SPEC, NodeType.INTERNAL),
                "REGISTRATION": (REGISTRATION_SPEC, NodeType.SINK),
                "MODEL_EVAL": (MODEL_EVAL_SPEC, NodeType.INTERNAL),
            }
            self.specs_available = True
        except ImportError:
            self.specs_available = False
    
    @unittest.skipUnless(True, "Real specifications not available")
    def test_real_specifications_node_types(self):
        """Test that real specifications have correct node types."""
        if not self.specs_available:
            self.skipTest("Real specifications not available")
        
        for spec_name, (spec, expected_node_type) in self.real_specs.items():
            with self.subTest(specification=spec_name):
                self.assertEqual(
                    spec.node_type, 
                    expected_node_type,
                    f"{spec_name} should have node type {expected_node_type.value}"
                )
    
    @unittest.skipUnless(True, "Real specifications not available")
    def test_real_specifications_validation(self):
        """Test that real specifications pass validation."""
        if not self.specs_available:
            self.skipTest("Real specifications not available")
        
        for spec_name, (spec, _) in self.real_specs.items():
            with self.subTest(specification=spec_name):
                errors = spec.validate()
                self.assertEqual(
                    len(errors), 
                    0,
                    f"{spec_name} validation failed with errors: {errors}"
                )


if __name__ == "__main__":
    # Run tests with high verbosity
    unittest.main(verbosity=2)
