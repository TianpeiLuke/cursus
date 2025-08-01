"""
Tests for advanced monitoring capabilities of RegistryManager.
"""

import unittest
from src.pipeline_deps import RegistryManager
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestAdvancedMonitoring(IsolatedTestCase):
    """Test advanced monitoring and statistics features."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.manager = RegistryManager()
        
        # Create test specification
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="test_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri",
                data_type="S3Uri"
            )]
        )
    
    def test_detailed_context_stats(self):
        """Test detailed context statistics."""
        # Create multiple registries with varying complexity
        pipeline1 = self.manager.get_registry("pipeline1")
        pipeline2 = self.manager.get_registry("pipeline2")
        
        # Add different specs to pipeline1
        pipeline1.register("step1", self.test_spec)
        pipeline1.register("step2", self.test_spec)
        
        # Add different specs to pipeline2
        different_spec = StepSpecification(
            step_type="DifferentStep",
            node_type=NodeType.SINK,
            dependencies=[DependencySpec(
                logical_name="input",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                required=True
            )],
            outputs=[]
        )
        
        pipeline2.register("step3", self.test_spec)  # Same type as in pipeline1
        pipeline2.register("step4", different_spec)  # Different type
        
        # Get detailed stats
        stats = self.manager.get_context_stats()
        
        # Verify detailed statistics
        self.assertEqual(stats["pipeline1"]["step_count"], 2)
        self.assertEqual(stats["pipeline1"]["step_type_count"], 1)  # Same type used twice
        
        self.assertEqual(stats["pipeline2"]["step_count"], 2)
        self.assertEqual(stats["pipeline2"]["step_type_count"], 2)  # Two different types
    
    def test_cross_context_monitoring(self):
        """Test monitoring across multiple contexts."""
        # Create multiple contexts
        contexts = ["pipeline1", "pipeline2", "pipeline3"]
        for ctx in contexts:
            registry = self.manager.get_registry(ctx)
            registry.register("test_step", self.test_spec)
        
        # Add extra specs to specific contexts
        self.manager.get_registry("pipeline1").register("extra_step", self.test_spec)
        self.manager.get_registry("pipeline3").register("extra_step", self.test_spec)
        
        # Get all stats
        stats = self.manager.get_context_stats()
        
        # Verify cross-context statistics
        registered_contexts = list(stats.keys())
        for ctx in contexts:
            self.assertIn(ctx, registered_contexts)
        
        self.assertEqual(stats["pipeline1"]["step_count"], 2)
        self.assertEqual(stats["pipeline2"]["step_count"], 1)
        self.assertEqual(stats["pipeline3"]["step_count"], 2)
        
        # Aggregate statistics
        total_steps = sum(stat["step_count"] for stat in stats.values())
        self.assertEqual(total_steps, 5)  # 2 + 1 + 2
    
    def test_resource_usage_monitoring(self):
        """Test monitoring resource usage with many registries."""
        # Create many registries to test resource handling
        for i in range(50):
            registry = self.manager.get_registry(f"pipeline_{i}")
            registry.register("step", self.test_spec)
        
        # Check context count
        contexts = self.manager.list_contexts()
        self.assertEqual(len(contexts), 50)
        
        # Get stats for all contexts
        stats = self.manager.get_context_stats()
        self.assertEqual(len(stats), 50)
        
        # Check total steps across all contexts
        total_steps = sum(stat["step_count"] for stat in stats.values())
        self.assertEqual(total_steps, 50)


if __name__ == '__main__':
    unittest.main()
