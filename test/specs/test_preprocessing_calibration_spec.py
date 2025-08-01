#!/usr/bin/env python3
"""
Unit tests for the Tabular Preprocessing Calibration Step Specification.

This module provides tests for the calibration preprocessing step specification,
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
from src.pipeline_step_specs.preprocessing_calibration_spec import PREPROCESSING_CALIBRATION_SPEC


class TestPreprocessingCalibrationSpec(unittest.TestCase):
    """Test cases for the Tabular Preprocessing Calibration Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("preprocessing_calibration_step", PREPROCESSING_CALIBRATION_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("preprocessing_calibration_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "TabularPreprocessing_Calibration")
    
    def test_step_type(self):
        """Test that the specification has the correct job-specific step type."""
        self.assertEqual(PREPROCESSING_CALIBRATION_SPEC.step_type, "TabularPreprocessing_Calibration")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PREPROCESSING_CALIBRATION_SPEC.node_type, NodeType.INTERNAL)
    
    def test_dependencies(self):
        """Test that the specification has the correct dependencies."""
        dependencies = PREPROCESSING_CALIBRATION_SPEC.dependencies
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
        """Test that dependencies point to the correct calibration data loading source."""
        for dep in PREPROCESSING_CALIBRATION_SPEC.dependencies.values():
            self.assertIn("CradleDataLoading_Calibration", dep.compatible_sources)
            self.assertEqual(len(dep.compatible_sources), 1, 
                           f"Dependency {dep.logical_name} should only be compatible with calibration data loading")
    
    def test_calibration_dependency_keywords(self):
        """Test that dependencies have calibration-specific semantic keywords."""
        for dep in PREPROCESSING_CALIBRATION_SPEC.dependencies.values():
            keywords = dep.semantic_keywords or []
            self.assertIn("calibration", keywords, f"Dependency {dep.logical_name} should have 'calibration' keyword")
            self.assertIn("calib", keywords, f"Dependency {dep.logical_name} should have 'calib' keyword")
            self.assertIn("eval", keywords, f"Dependency {dep.logical_name} should have 'eval' keyword")
            self.assertIn("evaluation", keywords, f"Dependency {dep.logical_name} should have 'evaluation' keyword")
            self.assertIn("model_eval", keywords, f"Dependency {dep.logical_name} should have 'model_eval' keyword")
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check main outputs exist
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("processed_data"))
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("ProcessedTabularData"))
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("full_data"))
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("FullData"))
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("calibration_data"))
        self.assertIsNotNone(PREPROCESSING_CALIBRATION_SPEC.get_output("CalibrationData"))
        
        # Check output types
        processed_output = PREPROCESSING_CALIBRATION_SPEC.get_output("processed_data")
        self.assertEqual(processed_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            processed_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
        )
        
        calibration_output = PREPROCESSING_CALIBRATION_SPEC.get_output("calibration_data")
        self.assertEqual(
            calibration_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['CalibrationData'].S3Output.S3Uri"
        )
    
    def test_calibration_specific_descriptions(self):
        """Test that outputs have calibration-specific descriptions."""
        processed_output = PREPROCESSING_CALIBRATION_SPEC.get_output("processed_data")
        self.assertIn("calibration", processed_output.description.lower())
        
        calibration_output = PREPROCESSING_CALIBRATION_SPEC.get_output("calibration_data")
        self.assertIn("calibration", calibration_output.description.lower())
    
    def test_calibration_output_keywords(self):
        """Test that outputs have calibration-specific descriptions (semantic keywords are only on dependencies)."""
        # Note: OutputSpec doesn't have semantic_keywords, only DependencySpec does
        # We validate job type differentiation through descriptions instead
        processed_output = PREPROCESSING_CALIBRATION_SPEC.get_output("processed_data")
        self.assertIn("calibration", processed_output.description.lower())
        
        calibration_output = PREPROCESSING_CALIBRATION_SPEC.get_output("calibration_data")
        self.assertIn("calibration", calibration_output.description.lower())
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PREPROCESSING_CALIBRATION_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")
    
    def test_unique_from_training(self):
        """Test that this specification is distinct from the training preprocessing spec."""
        from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
        
        # Different step types
        self.assertNotEqual(PREPROCESSING_CALIBRATION_SPEC.step_type, PREPROCESSING_TRAINING_SPEC.step_type)
        
        # Different compatible sources
        calib_data_dep = next(dep for dep in PREPROCESSING_CALIBRATION_SPEC.dependencies.values() if dep.logical_name == "DATA")
        training_data_dep = next(dep for dep in PREPROCESSING_TRAINING_SPEC.dependencies.values() if dep.logical_name == "DATA")
        
        self.assertEqual(calib_data_dep.compatible_sources, ["CradleDataLoading_Calibration"])
        self.assertEqual(training_data_dep.compatible_sources, ["CradleDataLoading_Training"])
        
        # Different semantic keywords
        calib_keywords = calib_data_dep.semantic_keywords or []
        training_keywords = training_data_dep.semantic_keywords or []
        
        self.assertIn("calibration", calib_keywords)
        self.assertIn("evaluation", calib_keywords)
        self.assertNotIn("training", calib_keywords)
        
        self.assertIn("training", training_keywords)
        self.assertIn("model_training", training_keywords)
        self.assertNotIn("calibration", training_keywords)
    
    def test_calibration_specific_outputs(self):
        """Test that calibration spec has calibration-specific outputs not in training spec."""
        from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
        
        # Calibration spec should have CalibrationData output
        calib_output = PREPROCESSING_CALIBRATION_SPEC.get_output("CalibrationData")
        self.assertIsNotNone(calib_output)
        
        # Training spec should not have CalibrationData output
        training_calib_output = PREPROCESSING_TRAINING_SPEC.get_output("CalibrationData")
        self.assertIsNone(training_calib_output)
    
    def test_dependency_resolution_compatibility(self):
        """Test compatibility with calibration data loading specification."""
        from src.pipeline_step_specs.data_loading_calibration_spec import DATA_LOADING_CALIBRATION_SPEC
        
        # Get outputs from calibration data loading
        data_loading_outputs = [output.logical_name for output in DATA_LOADING_CALIBRATION_SPEC.outputs.values()]
        
        # Check that all preprocessing dependencies can be satisfied
        for dep in PREPROCESSING_CALIBRATION_SPEC.dependencies.values():
            self.assertIn(dep.logical_name, data_loading_outputs,
                         f"Dependency {dep.logical_name} cannot be satisfied by calibration data loading outputs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
