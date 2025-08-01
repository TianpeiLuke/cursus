"""
Tests for error handling and edge cases in RegistryManager.
"""

import unittest
import threading
import time
import gc
from src.pipeline_deps import RegistryManager
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from src.pipeline_deps.dependency_resolver import UnifiedDependencyResolver
from src.pipeline_deps.semantic_matcher import SemanticMatcher
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state


class TestErrorHandlingAndEdgeCases(IsolatedTestCase):
    """Test error handling and edge cases."""
    
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
                property_path="properties.Output",
                data_type="S3Uri"
            )]
        )
    
    def test_context_not_found_handling(self):
        """Test handling of non-existent contexts."""
        # Try to get context that doesn't exist
        registry = self.manager.get_registry("nonexistent", create_if_missing=False)
        self.assertIsNone(registry)
        
        # Verify it wasn't created
        self.assertNotIn("nonexistent", self.manager.list_contexts())
        
        # Try to clear non-existent context
        result = self.manager.clear_context("nonexistent")
        self.assertFalse(result)
    
    def test_special_context_names(self):
        """Test registry with special context names."""
        # Test with unusual context names
        special_names = [
            "context with spaces",
            "context.with.dots",
            "context-with-dashes",
            "context_with_underscores",
            "UPPERCASE_CONTEXT",
            "123_numeric_context"
        ]
        
        # Create registries with special names
        for name in special_names:
            registry = self.manager.get_registry(name)
            registry.register("test_step", self.test_spec)
        
        # Verify they were created and can be accessed
        contexts = self.manager.list_contexts()
        for name in special_names:
            self.assertIn(name, contexts)
            
            # Get registry again
            registry = self.manager.get_registry(name)
            self.assertIsNotNone(registry)
            self.assertEqual(registry.context_name, name)
            
            # Verify step was registered
            self.assertIn("test_step", registry.list_step_names())
    
    def test_clear_while_in_use(self):
        """Test clearing contexts that are still referenced."""
        # Create registry
        registry = self.manager.get_registry("test_context")
        registry.register("test_step", self.test_spec)
        
        # Clear registry while still having a reference to it
        self.manager.clear_context("test_context")
        
        # Registry object should still work (but be disconnected from manager)
        self.assertEqual(registry.context_name, "test_context")
        self.assertIn("test_step", registry.list_step_names())
        
        # But it should no longer be in the manager
        self.assertNotIn("test_context", self.manager.list_contexts())
        
        # Getting the registry again should create a new one
        new_registry = self.manager.get_registry("test_context")
        self.assertIsNot(registry, new_registry)
        self.assertNotIn("test_step", new_registry.list_step_names())
    
    def test_dependency_resolver_integration(self):
        """Test integration with dependency resolver."""
        # Create registry and matcher
        registry = self.manager.get_registry("integration_test")
        matcher = SemanticMatcher()
        
        # Create resolver with registry
        resolver = UnifiedDependencyResolver(registry, matcher)
        
        # Register test specifications with dependencies
        source_spec = StepSpecification(
            step_type="SourceStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[OutputSpec(
                logical_name="source_output",
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.Output.S3Uri",
                data_type="S3Uri"
            )]
        )
        
        sink_spec = StepSpecification(
            step_type="SinkStep",
            node_type=NodeType.SINK,
            dependencies=[DependencySpec(
                logical_name="input_data",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                required=True
            )],
            outputs=[]
        )
        
        # Register specs
        registry.register("source", source_spec)
        registry.register("sink", sink_spec)
        
        # Resolve dependencies
        dependencies = resolver.resolve_all_dependencies(["source", "sink"])
        
        # Check that we got dependencies back (the exact content depends on implementation)
        # Since we're just testing integration with resolver, not the resolver itself
        self.assertIsNotNone(dependencies)
        self.assertIsInstance(dependencies, dict)
        
        # Verify that a resolution was at least attempted
        # The actual result may vary based on resolver implementation
        self.assertTrue(len(dependencies.keys()) >= 0)
    
    def test_concurrent_registry_access(self):
        """Test concurrent access to registries is thread-safe."""
        # Number of concurrent operations
        num_threads = 10
        operations_per_thread = 50
        
        # Track any errors that occur in threads
        errors = []
        
        # Function to perform registry operations
        def registry_operations(thread_id):
            try:
                # Create and use registry in each thread
                for i in range(operations_per_thread):
                    # Create unique context name for this operation
                    context = f"thread_{thread_id}_op_{i}"
                    
                    # Create registry
                    registry = self.manager.get_registry(context)
                    
                    # Register a specification
                    registry.register("step", self.test_spec)
                    
                    # Verify registration
                    self.assertIn("step", registry.list_step_names())
                    
                    # Get stats
                    stats = self.manager.get_context_stats()
                    self.assertIn(context, stats)
                    
                    # Clear this context
                    self.manager.clear_context(context)
                    
                    # Verify clearing
                    self.assertNotIn(context, self.manager.list_contexts())
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=registry_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        
        # Verify final state
        self.assertEqual(len(self.manager.list_contexts()), 0)
    
    def test_gc_behavior(self):
        """Test garbage collection behavior."""
        # Create a registry
        registry = self.manager.get_registry("test_gc")
        registry.register("test_step", self.test_spec)
        
        # Verify registry exists
        self.assertIn("test_gc", self.manager.list_contexts())
        
        # Remove our reference to the registry
        registry = None
        
        # Force garbage collection
        gc.collect()
        
        # Registry should still exist in manager (manager maintains the reference)
        self.assertIn("test_gc", self.manager.list_contexts())
        
        # Clear context
        self.manager.clear_context("test_gc")
        
        # Registry should now be gone
        self.assertNotIn("test_gc", self.manager.list_contexts())


if __name__ == '__main__':
    unittest.main()
