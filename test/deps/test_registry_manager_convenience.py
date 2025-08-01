"""
Tests for convenience functions of RegistryManager.
"""

import unittest
from src.pipeline_deps import (
    RegistryManager, get_registry, get_pipeline_registry, 
    get_default_registry, list_contexts, clear_context, get_context_stats
)
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestConvenienceFunctions(IsolatedTestCase):
    """Test convenience functions for registry management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Call parent setUp to reset global state
        super().setUp()
        
        # Create a fresh manager for each test
        self.manager = RegistryManager()
        
        # Create fresh instances of the enums for each test to ensure isolation
        self.node_type_source = NodeType.SOURCE
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri",
            data_type="S3Uri"
        )
        
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source,
            dependencies=[],
            outputs=[output_spec]
        )
    
    def test_get_registry_function(self):
        """Test get_registry convenience function."""
        # Get registry using convenience function
        registry = get_registry(self.manager, "test_pipeline")
        
        # Verify it works
        self.assertTrue(hasattr(registry, 'context_name'))
        self.assertEqual(registry.context_name, "test_pipeline")
        
        # Verify it uses the provided manager
        self.assertIn("test_pipeline", self.manager.list_contexts())
    
    def test_get_pipeline_registry_backward_compatibility(self):
        """Test backward compatibility function."""
        # Use old function name
        registry = get_pipeline_registry(self.manager, "my_pipeline")
        
        # Should work the same as get_registry
        self.assertTrue(hasattr(registry, 'context_name'))
        self.assertEqual(registry.context_name, "my_pipeline")
    
    def test_get_default_registry_backward_compatibility(self):
        """Test backward compatibility for default registry."""
        # Get default registry
        registry = get_default_registry(self.manager)
        
        # Should be default context
        self.assertTrue(hasattr(registry, 'context_name'))
        self.assertEqual(registry.context_name, "default")
    
    def test_list_contexts_function(self):
        """Test list_contexts convenience function."""
        # Initially empty
        self.assertEqual(len(list_contexts(self.manager)), 0)
        
        # Create some registries
        get_registry(self.manager, "pipeline_1")
        get_registry(self.manager, "pipeline_2")
        
        # Verify listing
        contexts = list_contexts(self.manager)
        self.assertEqual(len(contexts), 2)
        self.assertIn("pipeline_1", contexts)
        self.assertIn("pipeline_2", contexts)
    
    def test_clear_context_function(self):
        """Test clear_context convenience function."""
        # Create registry
        registry = get_registry(self.manager, "test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Verify it exists
        self.assertIn("test_pipeline", list_contexts(self.manager))
        
        # Clear using convenience function
        result = clear_context(self.manager, "test_pipeline")
        
        # Verify clearing
        self.assertTrue(result)
        self.assertNotIn("test_pipeline", list_contexts(self.manager))
    
    def test_get_context_stats_function(self):
        """Test get_context_stats convenience function."""
        # Create registry with spec
        registry = get_registry(self.manager, "test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Get stats using convenience function
        stats = get_context_stats(self.manager)
        
        # Verify stats
        self.assertIn("test_pipeline", stats)
        self.assertEqual(stats["test_pipeline"]["step_count"], 1)
    
    def test_multiple_contexts_isolation(self):
        """Test that multiple contexts remain isolated through convenience functions."""
        # Create multiple registries
        registry1 = get_registry(self.manager, "training")
        registry2 = get_registry(self.manager, "inference")
        registry3 = get_pipeline_registry(self.manager, "evaluation")  # Using backward compatibility
        
        # Register different specs
        registry1.register("train_step", self.test_spec)
        registry2.register("infer_step", self.test_spec)
        registry3.register("eval_step", self.test_spec)
        
        # Verify isolation
        self.assertIn("train_step", registry1.list_step_names())
        self.assertNotIn("train_step", registry2.list_step_names())
        self.assertNotIn("train_step", registry3.list_step_names())
        
        self.assertIn("infer_step", registry2.list_step_names())
        self.assertNotIn("infer_step", registry1.list_step_names())
        self.assertNotIn("infer_step", registry3.list_step_names())
        
        self.assertIn("eval_step", registry3.list_step_names())
        self.assertNotIn("eval_step", registry1.list_step_names())
        self.assertNotIn("eval_step", registry2.list_step_names())


if __name__ == '__main__':
    unittest.main()
