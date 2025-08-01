#!/usr/bin/env python3
"""
Unit tests for the PyTorch Model Step Specification.

This module provides tests for the PyTorch model step specification,
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
from src.pipeline_step_specs.pytorch_model_spec import PYTORCH_MODEL_SPEC


class TestPyTorchModelSpec(unittest.TestCase):
    """Test cases for the PyTorch Model Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("pytorch_model_step", PYTORCH_MODEL_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("pytorch_model_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "PyTorchModel")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PYTORCH_MODEL_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = PYTORCH_MODEL_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that model_data is required
        self.assertIn("model_data", required_dep_names)
        
        # Check dependency type
        model_data_dep = PYTORCH_MODEL_SPEC.get_dependency("model_data")
        self.assertEqual(model_data_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        self.assertTrue(model_data_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check primary output names
        self.assertIsNotNone(PYTORCH_MODEL_SPEC.get_output("model"))
        self.assertIsNotNone(PYTORCH_MODEL_SPEC.get_output("model_artifacts_path"))
        
        # Check output types
        model_output = PYTORCH_MODEL_SPEC.get_output("model")
        self.assertEqual(model_output.output_type, DependencyType.CUSTOM_PROPERTY)
        
        model_artifacts_path = PYTORCH_MODEL_SPEC.get_output("model_artifacts_path")
        self.assertEqual(model_artifacts_path.output_type, DependencyType.MODEL_ARTIFACTS)
        
        # Check property paths
        self.assertEqual(
            model_output.property_path,
            "properties.ModelName"
        )
        self.assertEqual(
            model_artifacts_path.property_path,
            "properties.ModelArtifacts.S3ModelArtifacts"
        )
    
    def test_output_aliases(self):
        """Test that the output aliases point to the same property paths."""
        # Model name aliases
        model_output = PYTORCH_MODEL_SPEC.get_output("model")
        model_name_alias = PYTORCH_MODEL_SPEC.get_output("ModelName")
        self.assertEqual(model_output.property_path, model_name_alias.property_path)
        
        # Model artifacts path aliases
        model_artifacts_path = PYTORCH_MODEL_SPEC.get_output("model_artifacts_path")
        artifacts_aliases = [
            "ModelArtifactsPath", "model_input"
        ]
        
        for alias in artifacts_aliases:
            alias_output = PYTORCH_MODEL_SPEC.get_output(alias)
            self.assertEqual(
                model_artifacts_path.property_path, 
                alias_output.property_path,
                f"Alias {alias} should have the same property path as model_artifacts_path"
            )
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        model_data_dep = PYTORCH_MODEL_SPEC.get_dependency("model_data")
        
        # Check model_data compatible sources
        self.assertIn("PyTorchTraining", model_data_dep.compatible_sources)
        self.assertIn("ProcessingStep", model_data_dep.compatible_sources)
        self.assertIn("ModelArtifactsStep", model_data_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        model_data_dep = PYTORCH_MODEL_SPEC.get_dependency("model_data")
        
        # Check model_data semantic keywords
        self.assertIn("model", model_data_dep.semantic_keywords)
        self.assertIn("artifacts", model_data_dep.semantic_keywords)
        self.assertIn("pytorch", model_data_dep.semantic_keywords)
        self.assertIn("training", model_data_dep.semantic_keywords)
        self.assertIn("output", model_data_dep.semantic_keywords)
        self.assertIn("model_data", model_data_dep.semantic_keywords)
    
    def test_data_types(self):
        """Test that the dependencies and outputs have the correct data types."""
        # Check dependency data types
        model_data_dep = PYTORCH_MODEL_SPEC.get_dependency("model_data")
        self.assertEqual(model_data_dep.data_type, "S3Uri")
        
        # Check output data types
        model_output = PYTORCH_MODEL_SPEC.get_output("model")
        self.assertEqual(model_output.data_type, "String")
        
        model_artifacts_path = PYTORCH_MODEL_SPEC.get_output("model_artifacts_path")
        self.assertEqual(model_artifacts_path.data_type, "S3Uri")
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PYTORCH_MODEL_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
