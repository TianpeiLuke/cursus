#!/usr/bin/env python3
"""
Unit tests for the Tabular Preprocessing Step Specification.

This module provides tests for the preprocessing step specification,
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
from src.pipeline_step_specs.preprocessing_spec import PREPROCESSING_SPEC


class TestPreprocessingSpec(unittest.TestCase):
    """Test cases for the Tabular Preprocessing Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("preprocessing_step", PREPROCESSING_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("preprocessing_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "TabularPreprocessing")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PREPROCESSING_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = PREPROCESSING_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that DATA is required
        self.assertIn("DATA", required_dep_names)
        
        # Check dependency type
        data_dep = PREPROCESSING_SPEC.get_dependency("DATA")
        self.assertEqual(data_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
        self.assertTrue(data_dep.required)
    
    def test_optional_dependencies(self):
        """Test that the specification has the optional dependencies."""
        optional_deps = PREPROCESSING_SPEC.list_optional_dependencies()
        optional_dep_names = [dep.logical_name for dep in optional_deps]
        
        # Check that METADATA and SIGNATURE are optional
        self.assertIn("METADATA", optional_dep_names)
        self.assertIn("SIGNATURE", optional_dep_names)
        
        # Check dependency types
        metadata_dep = PREPROCESSING_SPEC.get_dependency("METADATA")
        self.assertEqual(metadata_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
        self.assertFalse(metadata_dep.required)
        
        signature_dep = PREPROCESSING_SPEC.get_dependency("SIGNATURE")
        self.assertEqual(signature_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
        self.assertFalse(signature_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check output names
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("processed_data"))
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("ProcessedTabularData"))
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("full_data"))
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("FullData"))
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("calibration_data"))
        self.assertIsNotNone(PREPROCESSING_SPEC.get_output("CalibrationData"))
        
        # Check output types
        processed_data = PREPROCESSING_SPEC.get_output("processed_data")
        self.assertEqual(processed_data.output_type, DependencyType.PROCESSING_OUTPUT)
        
        full_data = PREPROCESSING_SPEC.get_output("full_data")
        self.assertEqual(full_data.output_type, DependencyType.PROCESSING_OUTPUT)
        
        calibration_data = PREPROCESSING_SPEC.get_output("calibration_data")
        self.assertEqual(calibration_data.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            processed_data.property_path,
            "properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
        )
        self.assertEqual(
            full_data.property_path,
            "properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri"
        )
        self.assertEqual(
            calibration_data.property_path,
            "properties.ProcessingOutputConfig.Outputs['CalibrationData'].S3Output.S3Uri"
        )
    
    def test_output_aliases(self):
        """Test that the output aliases point to the same property paths."""
        processed_data = PREPROCESSING_SPEC.get_output("processed_data")
        processed_data_alias = PREPROCESSING_SPEC.get_output("ProcessedTabularData")
        self.assertEqual(processed_data.property_path, processed_data_alias.property_path)
        
        full_data = PREPROCESSING_SPEC.get_output("full_data")
        full_data_alias = PREPROCESSING_SPEC.get_output("FullData")
        self.assertEqual(full_data.property_path, full_data_alias.property_path)
        
        calibration_data = PREPROCESSING_SPEC.get_output("calibration_data")
        calibration_data_alias = PREPROCESSING_SPEC.get_output("CalibrationData")
        self.assertEqual(calibration_data.property_path, calibration_data_alias.property_path)
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        data_dep = PREPROCESSING_SPEC.get_dependency("DATA")
        metadata_dep = PREPROCESSING_SPEC.get_dependency("METADATA")
        signature_dep = PREPROCESSING_SPEC.get_dependency("SIGNATURE")
        
        # Check DATA compatible sources
        self.assertIn("CradleDataLoading", data_dep.compatible_sources)
        self.assertIn("DataLoad", data_dep.compatible_sources)
        self.assertIn("ProcessingStep", data_dep.compatible_sources)
        
        # Check METADATA compatible sources
        self.assertIn("CradleDataLoading", metadata_dep.compatible_sources)
        self.assertIn("DataLoad", metadata_dep.compatible_sources)
        self.assertIn("ProcessingStep", metadata_dep.compatible_sources)
        
        # Check SIGNATURE compatible sources
        self.assertIn("CradleDataLoading", signature_dep.compatible_sources)
        self.assertIn("DataLoad", signature_dep.compatible_sources)
        self.assertIn("ProcessingStep", signature_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        data_dep = PREPROCESSING_SPEC.get_dependency("DATA")
        metadata_dep = PREPROCESSING_SPEC.get_dependency("METADATA")
        signature_dep = PREPROCESSING_SPEC.get_dependency("SIGNATURE")
        
        # Check DATA semantic keywords
        self.assertIn("data", data_dep.semantic_keywords)
        self.assertIn("input", data_dep.semantic_keywords)
        self.assertIn("raw", data_dep.semantic_keywords)
        self.assertIn("dataset", data_dep.semantic_keywords)
        self.assertIn("source", data_dep.semantic_keywords)
        self.assertIn("tabular", data_dep.semantic_keywords)
        
        # Check METADATA semantic keywords
        self.assertIn("metadata", metadata_dep.semantic_keywords)
        self.assertIn("schema", metadata_dep.semantic_keywords)
        self.assertIn("info", metadata_dep.semantic_keywords)
        self.assertIn("description", metadata_dep.semantic_keywords)
        
        # Check SIGNATURE semantic keywords
        self.assertIn("signature", signature_dep.semantic_keywords)
        self.assertIn("validation", signature_dep.semantic_keywords)
        self.assertIn("checksum", signature_dep.semantic_keywords)
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PREPROCESSING_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
