"""
Integration tests for RegistryManager with pipeline scenarios.

Tests the integration of RegistryManager with complex pipeline scenarios including:
- Multi-pipeline isolation
- Pipeline context switching
- Pipeline cleanup scenarios
"""

import unittest
from src.cursus.core.deps import RegistryManager
from src.cursus.core.base.specification_base import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from ..core.deps.test_helpers import IsolatedTestCase


class TestRegistryManagerPipelineIntegration(IsolatedTestCase):
    """Test RegistryManager integration with pipeline scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.manager = RegistryManager()
        
        # Create pipeline step specifications
        self.data_output = OutputSpec(
            logical_name="raw_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.data",
            data_type="S3Uri"
        )
        
        self.model_output = OutputSpec(
            logical_name="trained_model",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.model",
            data_type="S3Uri"
        )
        
        self.data_spec = StepSpecification(
            step_type="DataStep",
            node_type="source",
            dependencies=[],
            outputs=[self.data_output]
        )
        
        self.training_spec = StepSpecification(
            step_type="TrainingStep",
            node_type="internal",
            dependencies=[DependencySpec(
                logical_name="input_data",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                required=True
            )],
            outputs=[self.model_output]
        )
    
    def test_multi_pipeline_isolation(self):
        """Test isolation between multiple pipeline contexts."""
        # Create training pipeline
        training_registry = self.manager.get_registry("training_pipeline")
        training_registry.register("data_step", self.data_spec)
        training_registry.register("training_step", self.training_spec)
        
        # Create inference pipeline
        inference_registry = self.manager.get_registry("inference_pipeline")
        inference_registry.register("inference_data", self.data_spec)
        
        # Verify isolation
        training_steps = training_registry.list_step_names()
        inference_steps = inference_registry.list_step_names()
        
        self.assertEqual(len(training_steps), 2)
        self.assertEqual(len(inference_steps), 1)
        
        self.assertIn("data_step", training_steps)
        self.assertIn("training_step", training_steps)
        self.assertIn("inference_data", inference_steps)
        
        # Verify no cross-contamination
        self.assertNotIn("inference_data", training_steps)
        self.assertNotIn("data_step", inference_steps)
        self.assertNotIn("training_step", inference_steps)
    
    def test_pipeline_context_switching(self):
        """Test switching between pipeline contexts."""
        # Create multiple pipeline contexts
        contexts = ["dev", "staging", "prod"]
        
        for context in contexts:
            registry = self.manager.get_registry(f"{context}_pipeline")
            registry.register("data_step", self.data_spec)
            registry.register("training_step", self.training_spec)
        
        # Verify all contexts exist
        all_contexts = self.manager.list_contexts()
        for context in contexts:
            self.assertIn(f"{context}_pipeline", all_contexts)
        
        # Test context switching
        for context in contexts:
            registry = self.manager.get_registry(f"{context}_pipeline")
            steps = registry.list_step_names()
            self.assertEqual(len(steps), 2)
            self.assertIn("data_step", steps)
            self.assertIn("training_step", steps)
    
    def test_pipeline_cleanup_scenarios(self):
        """Test various pipeline cleanup scenarios."""
        # Create multiple pipelines
        pipelines = ["pipeline_a", "pipeline_b", "pipeline_c"]
        
        for pipeline in pipelines:
            registry = self.manager.get_registry(pipeline)
            registry.register("step1", self.data_spec)
            registry.register("step2", self.training_spec)
        
        # Verify all created
        self.assertEqual(len(self.manager.list_contexts()), 3)
        
        # Clear specific pipeline
        result = self.manager.clear_context("pipeline_b")
        self.assertTrue(result)
        
        remaining_contexts = self.manager.list_contexts()
        self.assertEqual(len(remaining_contexts), 2)
        self.assertIn("pipeline_a", remaining_contexts)
        self.assertIn("pipeline_c", remaining_contexts)
        self.assertNotIn("pipeline_b", remaining_contexts)
        
        # Clear all remaining
        self.manager.clear_all_contexts()
        self.assertEqual(len(self.manager.list_contexts()), 0)


if __name__ == '__main__':
    unittest.main()
