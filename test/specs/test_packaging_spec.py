#!/usr/bin/env python3
"""
Unit tests for the MIMS Packaging Step Specification.

This module provides tests for the packaging step specification,
including validation of dependencies, outputs, and node type.
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
from src.pipeline_step_specs.packaging_spec import PACKAGING_SPEC


class TestPackagingSpec(unittest.TestCase):
    """Test cases for the MIMS Packaging Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("packaging_step", PACKAGING_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("packaging_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "Package")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PACKAGING_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = PACKAGING_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that model_input and inference_scripts_input are required
        self.assertIn("model_input", required_dep_names)
        self.assertIn("inference_scripts_input", required_dep_names)
        
        # Check dependency types
        model_input_dep = PACKAGING_SPEC.get_dependency("model_input")
        self.assertEqual(model_input_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        self.assertTrue(model_input_dep.required)
        
        inference_scripts_dep = PACKAGING_SPEC.get_dependency("inference_scripts_input")
        self.assertEqual(inference_scripts_dep.dependency_type, DependencyType.CUSTOM_PROPERTY)
        self.assertTrue(inference_scripts_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check output names
        self.assertIsNotNone(PACKAGING_SPEC.get_output("packaged_model_output"))
        self.assertIsNotNone(PACKAGING_SPEC.get_output("PackagedModel"))
        
        # Check output types
        packaged_model = PACKAGING_SPEC.get_output("packaged_model_output")
        self.assertEqual(packaged_model.output_type, DependencyType.MODEL_ARTIFACTS)
        
        # Check property paths
        self.assertEqual(
            packaged_model.property_path,
            "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"
        )
    
    def test_output_aliases(self):
        """Test that the output aliases point to the same property paths."""
        packaged_model = PACKAGING_SPEC.get_output("packaged_model_output")
        packaged_model_alias = PACKAGING_SPEC.get_output("PackagedModel")
        self.assertEqual(packaged_model.property_path, packaged_model_alias.property_path)
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        model_input_dep = PACKAGING_SPEC.get_dependency("model_input")
        inference_scripts_dep = PACKAGING_SPEC.get_dependency("inference_scripts_input")
        
        # Check model_input compatible sources
        self.assertIn("XGBoostTraining", model_input_dep.compatible_sources)
        self.assertIn("TrainingStep", model_input_dep.compatible_sources)
        self.assertIn("ModelStep", model_input_dep.compatible_sources)
        
        # Check inference_scripts_input compatible sources
        self.assertIn("ProcessingStep", inference_scripts_dep.compatible_sources)
        self.assertIn("ScriptStep", inference_scripts_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        model_input_dep = PACKAGING_SPEC.get_dependency("model_input")
        inference_scripts_dep = PACKAGING_SPEC.get_dependency("inference_scripts_input")
        
        # Check model_input semantic keywords
        self.assertIn("model", model_input_dep.semantic_keywords)
        self.assertIn("artifacts", model_input_dep.semantic_keywords)
        self.assertIn("trained", model_input_dep.semantic_keywords)
        self.assertIn("output", model_input_dep.semantic_keywords)
        self.assertIn("modelartifacts", model_input_dep.semantic_keywords)
        
        # Check inference_scripts_input semantic keywords
        self.assertIn("inference", inference_scripts_dep.semantic_keywords)
        self.assertIn("scripts", inference_scripts_dep.semantic_keywords)
        self.assertIn("code", inference_scripts_dep.semantic_keywords)
        self.assertIn("inferencescripts", inference_scripts_dep.semantic_keywords)
    
    def test_data_types(self):
        """Test that the dependencies and outputs have the correct data types."""
        # Check dependency data types
        model_input_dep = PACKAGING_SPEC.get_dependency("model_input")
        self.assertEqual(model_input_dep.data_type, "S3Uri")
        
        inference_scripts_dep = PACKAGING_SPEC.get_dependency("inference_scripts_input")
        self.assertEqual(inference_scripts_dep.data_type, "S3Uri")
        
        # Check output data types
        packaged_model = PACKAGING_SPEC.get_output("packaged_model_output")
        self.assertEqual(packaged_model.data_type, "S3Uri")
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PACKAGING_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
