#!/usr/bin/env python3
"""
Unit tests for the Cradle Data Loading Step Specification.

This module provides tests for the data loading step specification,
including validation of outputs and node type.
"""

import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.pipeline_deps.base_specifications import (
    DependencyType, NodeType
)
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_step_specs.data_loading_spec import DATA_LOADING_SPEC


class TestDataLoadingSpec(unittest.TestCase):
    """Test cases for the Cradle Data Loading Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("data_loading_step", DATA_LOADING_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("data_loading_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "CradleDataLoading")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(DATA_LOADING_SPEC.node_type, NodeType.SOURCE)
    
    def test_no_dependencies(self):
        """Test that the specification has no dependencies (as it's a SOURCE node)."""
        dependencies = DATA_LOADING_SPEC.dependencies
        self.assertEqual(len(dependencies), 0, "SOURCE node should not have dependencies")
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check output names
        self.assertIsNotNone(DATA_LOADING_SPEC.get_output("DATA"))
        self.assertIsNotNone(DATA_LOADING_SPEC.get_output("METADATA"))
        self.assertIsNotNone(DATA_LOADING_SPEC.get_output("SIGNATURE"))
        
        # Check output types
        data_output = DATA_LOADING_SPEC.get_output("DATA")
        self.assertEqual(data_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        metadata_output = DATA_LOADING_SPEC.get_output("METADATA")
        self.assertEqual(metadata_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        signature_output = DATA_LOADING_SPEC.get_output("SIGNATURE")
        self.assertEqual(signature_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            data_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri"
        )
        self.assertEqual(
            metadata_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri"
        )
        self.assertEqual(
            signature_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri"
        )
    
    def test_output_data_types(self):
        """Test that the outputs have the correct data types."""
        data_output = DATA_LOADING_SPEC.get_output("DATA")
        self.assertEqual(data_output.data_type, "S3Uri")
        
        metadata_output = DATA_LOADING_SPEC.get_output("METADATA")
        self.assertEqual(metadata_output.data_type, "S3Uri")
        
        signature_output = DATA_LOADING_SPEC.get_output("SIGNATURE")
        self.assertEqual(signature_output.data_type, "S3Uri")
    
    def test_output_descriptions(self):
        """Test that the outputs have appropriate descriptions."""
        data_output = DATA_LOADING_SPEC.get_output("DATA")
        self.assertIn("data output", data_output.description.lower())
        
        metadata_output = DATA_LOADING_SPEC.get_output("METADATA")
        self.assertIn("metadata", metadata_output.description.lower())
        
        signature_output = DATA_LOADING_SPEC.get_output("SIGNATURE")
        self.assertIn("signature", signature_output.description.lower())
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = DATA_LOADING_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
