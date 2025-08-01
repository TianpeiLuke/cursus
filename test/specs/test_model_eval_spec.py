#!/usr/bin/env python3
"""
Unit tests for the XGBoost Model Evaluation Step Specification.

This module provides tests for the model evaluation step specification,
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
from src.pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC


class TestModelEvalSpec(unittest.TestCase):
    """Test cases for the XGBoost Model Evaluation Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("model_eval_step", MODEL_EVAL_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("model_eval_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "XGBoostModelEvaluation")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(MODEL_EVAL_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = MODEL_EVAL_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that model_input and eval_data_input are required
        self.assertIn("model_input", required_dep_names)
        self.assertIn("eval_data_input", required_dep_names)
        
        # Check dependency types
        model_input_dep = MODEL_EVAL_SPEC.get_dependency("model_input")
        self.assertEqual(model_input_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        
        eval_data_dep = MODEL_EVAL_SPEC.get_dependency("eval_data_input")
        self.assertEqual(eval_data_dep.dependency_type, DependencyType.PROCESSING_OUTPUT)
    
    def test_optional_dependencies(self):
        """Test that the specification has the optional dependencies."""
        optional_deps = MODEL_EVAL_SPEC.list_optional_dependencies()
        optional_dep_names = [dep.logical_name for dep in optional_deps]
        
        # Check that hyperparameters_input is optional
        self.assertIn("hyperparameters_input", optional_dep_names)
        
        # Check dependency type
        hyperparams_dep = MODEL_EVAL_SPEC.get_dependency("hyperparameters_input")
        self.assertEqual(hyperparams_dep.dependency_type, DependencyType.HYPERPARAMETERS)
        self.assertFalse(hyperparams_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check output names
        self.assertIsNotNone(MODEL_EVAL_SPEC.get_output("eval_output"))
        self.assertIsNotNone(MODEL_EVAL_SPEC.get_output("metrics_output"))
        self.assertIsNotNone(MODEL_EVAL_SPEC.get_output("EvaluationResults"))
        self.assertIsNotNone(MODEL_EVAL_SPEC.get_output("EvaluationMetrics"))
        
        # Check output types
        eval_output = MODEL_EVAL_SPEC.get_output("eval_output")
        self.assertEqual(eval_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        metrics_output = MODEL_EVAL_SPEC.get_output("metrics_output")
        self.assertEqual(metrics_output.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            eval_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['EvaluationResults'].S3Output.S3Uri"
        )
        self.assertEqual(
            metrics_output.property_path,
            "properties.ProcessingOutputConfig.Outputs['EvaluationMetrics'].S3Output.S3Uri"
        )
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        model_input_dep = MODEL_EVAL_SPEC.get_dependency("model_input")
        eval_data_dep = MODEL_EVAL_SPEC.get_dependency("eval_data_input")
        
        # Check model_input compatible sources
        self.assertIn("XGBoostTraining", model_input_dep.compatible_sources)
        self.assertIn("TrainingStep", model_input_dep.compatible_sources)
        self.assertIn("ModelStep", model_input_dep.compatible_sources)
        
        # Check eval_data_input compatible sources
        self.assertIn("TabularPreprocessing", eval_data_dep.compatible_sources)
        self.assertIn("ProcessingStep", eval_data_dep.compatible_sources)
        self.assertIn("DataLoad", eval_data_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        model_input_dep = MODEL_EVAL_SPEC.get_dependency("model_input")
        eval_data_dep = MODEL_EVAL_SPEC.get_dependency("eval_data_input")
        
        # Check model_input semantic keywords
        self.assertIn("model", model_input_dep.semantic_keywords)
        self.assertIn("artifacts", model_input_dep.semantic_keywords)
        
        # Check eval_data_input semantic keywords
        self.assertIn("data", eval_data_dep.semantic_keywords)
        self.assertIn("evaluation", eval_data_dep.semantic_keywords)
        self.assertIn("calibration", eval_data_dep.semantic_keywords)
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = MODEL_EVAL_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
