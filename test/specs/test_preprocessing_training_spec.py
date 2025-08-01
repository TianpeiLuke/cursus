#!/usr/bin/env python3
"""
Unit tests for the Tabular Preprocessing Training Step Specification.

This module provides tests for the training preprocessing step specification,
including validation of dependencies, outputs, and job-specific semantic keywords.
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
from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC


class TestPreprocessingTrainingSpec(unittest.TestCase):
    """Test cases for the Tabular Preprocessing Training Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("preprocessing_training_step", PREPROCESSING_TRAINING_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("preprocessing_training_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "TabularPreprocessing_Training")
    
    def test_step_type(self):
        """Test that the specification has the correct job-specific step type."""
        self.assertEqual(PREPROCESSING_TRAINING_SPEC.step_type, "TabularPreprocessing_Training")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PREPROCESSING_TRAINING_SPEC.node_type, NodeType.INTERNAL)
    
    def test_dependencies(self):
        """Test that the specification has the correct dependencies."""
        dependencies = PREPROCESSING_TRAINING_SPEC.dependencies
        self.assertEqual(len(dependencies), 3, "Should have 3 dependencies: DATA, METADATA, SIGNATURE")
        
        # Check dependency names
        dep_names = [dep.logical_name for dep in dependencies.values()]
        self.assertIn("DATA", dep_names)
        self.assertIn("METADATA", dep_names)
        self.assertIn("SIGNATURE", dep_names)
        
        # Check DATA dependency is required
        data_dep = next(dep for dep in dependencies.values() if dep.logical_name == "DATA")
        self.assertTrue(data_dep.required, "DATA dependency should be required")
        
        # Check METADATA and SIGNATURE are optional
        metadata_dep = next(dep for dep in dependencies.values() if dep.logical_name == "METADATA")
        self.assertFalse(metadata_dep.required, "METADATA dependency should be optional")
        
        signature_dep = next(dep for dep in dependencies.values() if dep.logical_name == "SIGNATURE")
        self.assertFalse(signature_dep.required, "SIGNATURE dependency should be optional")
    
    def test_dependency_sources(self):
        """Test that dependencies point to the correct training data loading source."""
        for dep in PREPROCESSING_TRAINING_SPEC.dependencies.values():
            self.assertIn("CradleDataLoading_Training", dep.compatible_sources)
            self.assertEqual(len(dep.compatible_sources), 1, 
                           f"Dependency {dep.logical_name} should only be compatible with training data loading")
    
    def test_training_dependency_keywords(self):
        """Test that dependencies have training-specific semantic keywords."""
        for dep in PREPROCESSING_TRAINING_SPEC.dependencies.values():
            keywords = dep.semantic_keywords or []
            self.assertIn("training", keywords, f"Dependency {dep.logical_name} should have 'training' keyword")
            self.assertIn("train", keywords, f"Dependency {dep.logical_name} should have 'train' keyword")
            self.assertIn("model_training", keywords, f"Dependency {dep.logical_name} should have 'model_training' keyword")
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check main outputs exist
        self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC.get_output("processed_data"))
        self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC.get_output("ProcessedTabularData"))
        self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC.get_output("full_data"))
        self.assertIsNotNone(PREPROCESSING_TRAINING_SPEC.get_output("FullData"))
        
        # Check output types
        processed_output = PREPROCESSING_TRAINING_SPEC.get_output("processed_data")
        self.assertEqual(processed_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            processed_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
        )
        
        full_data_output = PREPROCESSING_TRAINING_SPEC.get_output("full_data")
        self.assertEqual(
            full_data_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri"
        )
    
    def test_training_specific_descriptions(self):
        """Test that outputs have training-specific descriptions."""
        processed_output = PREPROCESSING_TRAINING_SPEC.get_output("processed_data")
        self.assertIn("training", processed_output.description.lower())
        
        full_data_output = PREPROCESSING_TRAINING_SPEC.get_output("full_data")
        self.assertIn("training", full_data_output.description.lower())
    
    def test_training_output_keywords(self):
        """Test that outputs have training-specific descriptions (semantic keywords are only on dependencies)."""
        # Note: OutputSpec doesn't have semantic_keywords, only DependencySpec does
        # We validate job type differentiation through descriptions instead
        processed_output = PREPROCESSING_TRAINING_SPEC.get_output("processed_data")
        self.assertIn("training", processed_output.description.lower())
        
        full_data_output = PREPROCESSING_TRAINING_SPEC.get_output("full_data")
        self.assertIn("training", full_data_output.description.lower())
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PREPROCESSING_TRAINING_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")
    
    def test_unique_from_generic(self):
        """Test that this specification is distinct from the generic preprocessing spec."""
        from src.pipeline_step_specs.preprocessing_spec import PREPROCESSING_SPEC
        
        # Different step types
        self.assertNotEqual(PREPROCESSING_TRAINING_SPEC.step_type, PREPROCESSING_SPEC.step_type)
        
        # Training spec should have specific compatible sources
        training_data_dep = next(dep for dep in PREPROCESSING_TRAINING_SPEC.dependencies.values() if dep.logical_name == "DATA")
        generic_data_dep = next(dep for dep in PREPROCESSING_SPEC.dependencies.values() if dep.logical_name == "DATA")
        
        # Training spec should only be compatible with training data loading
        self.assertEqual(training_data_dep.compatible_sources, ["CradleDataLoading_Training"])
        
        # Generic spec should be compatible with multiple sources
        self.assertGreater(len(generic_data_dep.compatible_sources), 1)
    
    def test_dependency_resolution_compatibility(self):
        """Test compatibility with training data loading specification."""
        from src.pipeline_step_specs.data_loading_training_spec import DATA_LOADING_TRAINING_SPEC
        
        # Get outputs from training data loading
        data_loading_outputs = [output.logical_name for output in DATA_LOADING_TRAINING_SPEC.outputs.values()]
        
        # Check that all preprocessing dependencies can be satisfied
        for dep in PREPROCESSING_TRAINING_SPEC.dependencies.values():
            self.assertIn(dep.logical_name, data_loading_outputs,
                         f"Dependency {dep.logical_name} cannot be satisfied by training data loading outputs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
