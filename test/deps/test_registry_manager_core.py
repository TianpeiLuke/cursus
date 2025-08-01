"""
Tests for core functionality of RegistryManager.
"""

import unittest
from src.pipeline_deps import RegistryManager
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestRegistryManager(IsolatedTestCase):
    """Test cases for core RegistryManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Call parent setUp to reset global state
        super().setUp()
        
        # Create a fresh manager for each test
        self.manager = RegistryManager()
        
        # Create fresh instances of the enums for each test to ensure isolation
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
            node_type=self.node_type_source,
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


if __name__ == '__main__':
    unittest.main()
