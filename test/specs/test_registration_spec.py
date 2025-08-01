#!/usr/bin/env python3
"""
Unit tests for the Model Registration Step Specification.

This module provides tests for the registration step specification,
including validation of dependencies and node type.
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
from src.pipeline_step_specs.registration_spec import REGISTRATION_SPEC


class TestRegistrationSpec(unittest.TestCase):
    """Test cases for the Model Registration Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("registration_step", REGISTRATION_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("registration_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "ModelRegistration")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(REGISTRATION_SPEC.node_type, NodeType.SINK)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = REGISTRATION_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that PackagedModel and GeneratedPayloadSamples are required
        self.assertIn("PackagedModel", required_dep_names)
        self.assertIn("GeneratedPayloadSamples", required_dep_names)
        
        # Check dependency types
        packaged_model_dep = REGISTRATION_SPEC.get_dependency("PackagedModel")
        self.assertEqual(packaged_model_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        self.assertTrue(packaged_model_dep.required)
        
        payload_samples_dep = REGISTRATION_SPEC.get_dependency("GeneratedPayloadSamples")
        self.assertEqual(payload_samples_dep.dependency_type, DependencyType.PAYLOAD_SAMPLES)
        self.assertTrue(payload_samples_dep.required)
    
    def test_no_outputs(self):
        """Test that the specification has no outputs (as it's a SINK node)."""
        outputs = REGISTRATION_SPEC.outputs
        self.assertEqual(len(outputs), 0, "SINK node should not have outputs")
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        packaged_model_dep = REGISTRATION_SPEC.get_dependency("PackagedModel")
        payload_samples_dep = REGISTRATION_SPEC.get_dependency("GeneratedPayloadSamples")
        
        # Check PackagedModel compatible sources
        self.assertIn("PackagingStep", packaged_model_dep.compatible_sources)
        self.assertIn("Package", packaged_model_dep.compatible_sources)
        self.assertIn("ProcessingStep", packaged_model_dep.compatible_sources)
        
        # Check GeneratedPayloadSamples compatible sources
        self.assertIn("PayloadTestStep", payload_samples_dep.compatible_sources)
        self.assertIn("PayloadStep", payload_samples_dep.compatible_sources)
        self.assertIn("ProcessingStep", payload_samples_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        packaged_model_dep = REGISTRATION_SPEC.get_dependency("PackagedModel")
        payload_samples_dep = REGISTRATION_SPEC.get_dependency("GeneratedPayloadSamples")
        
        # Check PackagedModel semantic keywords
        self.assertIn("model", packaged_model_dep.semantic_keywords)
        self.assertIn("package", packaged_model_dep.semantic_keywords)
        self.assertIn("packaged", packaged_model_dep.semantic_keywords)
        self.assertIn("artifacts", packaged_model_dep.semantic_keywords)
        self.assertIn("tar", packaged_model_dep.semantic_keywords)
        
        # Check GeneratedPayloadSamples semantic keywords
        self.assertIn("payload", payload_samples_dep.semantic_keywords)
        self.assertIn("samples", payload_samples_dep.semantic_keywords)
        self.assertIn("test", payload_samples_dep.semantic_keywords)
        self.assertIn("generated", payload_samples_dep.semantic_keywords)
        self.assertIn("inference", payload_samples_dep.semantic_keywords)
    
    def test_data_types(self):
        """Test that the dependencies have the correct data types."""
        packaged_model_dep = REGISTRATION_SPEC.get_dependency("PackagedModel")
        self.assertEqual(packaged_model_dep.data_type, "S3Uri")
        
        payload_samples_dep = REGISTRATION_SPEC.get_dependency("GeneratedPayloadSamples")
        self.assertEqual(payload_samples_dep.data_type, "S3Uri")
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = REGISTRATION_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
