"""
Tests for RegistryManager - multi-registry management functionality.
"""

import unittest
from src.pipeline_deps import RegistryManager, get_registry, get_pipeline_registry, get_default_registry, list_contexts, clear_context, get_context_stats
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestRegistryManager(IsolatedTestCase):
    """Test cases for RegistryManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Call parent setUp to reset global state
        super().setUp()
        
        # Create a fresh manager for each test
        self.manager = RegistryManager()
        
        # Use string values for input but expect enum instances for comparison
        self.node_type_source_input = "source"
        self.node_type_source = NodeType.SOURCE
        self.dependency_type = DependencyType.PROCESSING_OUTPUT
        
        # Create test specification
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=self.dependency_type,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Test output"
        )
        
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type=self.node_type_source_input,
            dependencies=[],
            outputs=[output_spec]
        )
        
    def tearDown(self):
        """Clean up after tests."""
        # Clear all contexts to avoid state leakage between tests
        self.manager.clear_all_contexts()
    
    def test_manager_initialization(self):
        """Test registry manager initialization."""
        manager = RegistryManager()
        self.assertEqual(len(manager.list_contexts()), 0)
        self.assertIsInstance(manager, RegistryManager)
    
    def test_get_registry_creates_new(self):
        """Test that get_registry creates new registries when needed."""
        # Get registry for new context
        registry = self.manager.get_registry("test_pipeline")
        
        # Verify it was created
        self.assertTrue(hasattr(registry, 'context_name'))
        self.assertEqual(registry.context_name, "test_pipeline")
        self.assertIn("test_pipeline", self.manager.list_contexts())
    
    def test_get_registry_returns_existing(self):
        """Test that get_registry returns existing registries."""
        # Create registry
        registry1 = self.manager.get_registry("test_pipeline")
        registry1.register("test_step", self.test_spec)
        
        # Get same registry again
        registry2 = self.manager.get_registry("test_pipeline")
        
        # Should be the same instance
        self.assertIs(registry1, registry2)
        self.assertIn("test_step", registry2.list_step_names())
    
    def test_get_registry_no_create(self):
        """Test get_registry with create_if_missing=False."""
        # Try to get non-existent registry without creating
        registry = self.manager.get_registry("nonexistent", create_if_missing=False)
        
        # Should return None
        self.assertIsNone(registry)
        self.assertNotIn("nonexistent", self.manager.list_contexts())
    
    def test_registry_isolation(self):
        """Test that registries are properly isolated."""
        # Create two registries
        registry1 = self.manager.get_registry("pipeline_1")
        registry2 = self.manager.get_registry("pipeline_2")
        
        # Register different specs
        registry1.register("step1", self.test_spec)
        
        # Verify isolation
        self.assertIn("step1", registry1.list_step_names())
        self.assertNotIn("step1", registry2.list_step_names())
        
        # Verify they are different instances
        self.assertIsNot(registry1, registry2)
    
    def test_list_contexts(self):
        """Test listing all contexts."""
        # Initially empty
        self.assertEqual(len(self.manager.list_contexts()), 0)
        
        # Create some registries
        self.manager.get_registry("pipeline_1")
        self.manager.get_registry("pipeline_2")
        self.manager.get_registry("pipeline_3")
        
        # Verify listing
        contexts = self.manager.list_contexts()
        self.assertEqual(len(contexts), 3)
        self.assertIn("pipeline_1", contexts)
        self.assertIn("pipeline_2", contexts)
        self.assertIn("pipeline_3", contexts)
    
    def test_clear_context(self):
        """Test clearing specific contexts."""
        # Create registry and add spec
        registry = self.manager.get_registry("test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Verify it exists
        self.assertIn("test_pipeline", self.manager.list_contexts())
        
        # Clear it
        result = self.manager.clear_context("test_pipeline")
        
        # Verify clearing
        self.assertTrue(result)
        self.assertNotIn("test_pipeline", self.manager.list_contexts())
        
        # Try to clear non-existent context
        result = self.manager.clear_context("nonexistent")
        self.assertFalse(result)
    
    def test_clear_all_contexts(self):
        """Test clearing all contexts."""
        # Create multiple registries
        self.manager.get_registry("pipeline_1")
        self.manager.get_registry("pipeline_2")
        self.manager.get_registry("pipeline_3")
        
        # Verify they exist
        self.assertEqual(len(self.manager.list_contexts()), 3)
        
        # Clear all
        self.manager.clear_all_contexts()
        
        # Verify all cleared
        self.assertEqual(len(self.manager.list_contexts()), 0)
    
    def test_get_context_stats(self):
        """Test getting context statistics."""
        # Create registries with different numbers of specs
        registry1 = self.manager.get_registry("pipeline_1")
        registry1.register("step1", self.test_spec)
        
        registry2 = self.manager.get_registry("pipeline_2")
        registry2.register("step2a", self.test_spec)
        registry2.register("step2b", self.test_spec)
        
        # Get stats
        stats = self.manager.get_context_stats()
        
        # Verify stats
        self.assertIn("pipeline_1", stats)
        self.assertIn("pipeline_2", stats)
        
        self.assertEqual(stats["pipeline_1"]["step_count"], 1)
        self.assertEqual(stats["pipeline_2"]["step_count"], 2)
        
        self.assertEqual(stats["pipeline_1"]["step_type_count"], 1)
        self.assertEqual(stats["pipeline_2"]["step_type_count"], 1)  # Same step type
    
    def test_manager_string_representation(self):
        """Test string representation of manager."""
        # Empty manager
        repr_str = repr(self.manager)
        self.assertIn("contexts=0", repr_str)
        
        # Manager with contexts
        self.manager.get_registry("test1")
        self.manager.get_registry("test2")
        
        repr_str = repr(self.manager)
        self.assertIn("contexts=2", repr_str)


class TestConvenienceFunctions(IsolatedTestCase):
    """Test convenience functions for registry management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Call parent setUp to reset global state
        super().setUp()
        
        # Create a fresh manager for each test
        self.manager = RegistryManager()
        
        # Use string values for input but expect enum instances for comparison
        self.node_type_source_input = "source"
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
            node_type=self.node_type_source_input,
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
