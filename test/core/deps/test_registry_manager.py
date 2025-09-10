"""
Comprehensive tests for RegistryManager functionality.

Tests the complete functionality of registry manager including:
- Core registry management operations
- Context patterns and isolation
- Convenience functions and backward compatibility
- Error handling and edge cases
- Monitoring and statistics
- Pipeline integration scenarios
"""

import pytest
from cursus.core.deps import RegistryManager, get_registry, get_pipeline_registry, get_default_registry, list_contexts, clear_context, get_context_stats
from cursus.core.deps.specification_registry import SpecificationRegistry
from cursus.core.base.specification_base import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from .test_helpers import reset_all_global_state

class TestRegistryManagerCore:
    """Test cases for RegistryManager."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        reset_all_global_state()
        
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
        
        yield
        
        # Clean up after tests
        self.manager.clear_all_contexts()
        reset_all_global_state()
    
    def test_manager_initialization(self):
        """Test registry manager initialization."""
        manager = RegistryManager()
        assert len(manager.list_contexts()) == 0
        assert isinstance(manager, RegistryManager)
    
    def test_get_registry_creates_new(self):
        """Test that get_registry creates new registries when needed."""
        # Get registry for new context
        registry = self.manager.get_registry("test_pipeline")
        
        # Verify it was created
        assert hasattr(registry, 'context_name')
        assert registry.context_name == "test_pipeline"
        assert "test_pipeline" in self.manager.list_contexts()
    
    def test_get_registry_returns_existing(self):
        """Test that get_registry returns existing registries."""
        # Create registry
        registry1 = self.manager.get_registry("test_pipeline")
        registry1.register("test_step", self.test_spec)
        
        # Get same registry again
        registry2 = self.manager.get_registry("test_pipeline")
        
        # Should be the same instance
        assert registry1 is registry2
        assert "test_step" in registry2.list_step_names()
    
    def test_get_registry_no_create(self):
        """Test get_registry with create_if_missing=False."""
        # Try to get non-existent registry without creating
        registry = self.manager.get_registry("nonexistent", create_if_missing=False)
        
        # Should return None
        assert registry is None
        assert "nonexistent" not in self.manager.list_contexts()
    
    def test_registry_isolation(self):
        """Test that registries are properly isolated."""
        # Create two registries
        registry1 = self.manager.get_registry("pipeline_1")
        registry2 = self.manager.get_registry("pipeline_2")
        
        # Register different specs
        registry1.register("step1", self.test_spec)
        
        # Verify isolation
        assert "step1" in registry1.list_step_names()
        assert "step1" not in registry2.list_step_names()
        
        # Verify they are different instances
        assert registry1 is not registry2
    
    def test_list_contexts(self):
        """Test listing all contexts."""
        # Initially empty
        assert len(self.manager.list_contexts()) == 0
        
        # Create some registries
        self.manager.get_registry("pipeline_1")
        self.manager.get_registry("pipeline_2")
        self.manager.get_registry("pipeline_3")
        
        # Verify listing
        contexts = self.manager.list_contexts()
        assert len(contexts) == 3
        assert "pipeline_1" in contexts
        assert "pipeline_2" in contexts
        assert "pipeline_3" in contexts
    
    def test_clear_context(self):
        """Test clearing specific contexts."""
        # Create registry and add spec
        registry = self.manager.get_registry("test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Verify it exists
        assert "test_pipeline" in self.manager.list_contexts()
        
        # Clear it
        result = self.manager.clear_context("test_pipeline")
        
        # Verify clearing
        assert result is True
        assert "test_pipeline" not in self.manager.list_contexts()
        
        # Try to clear non-existent context
        result = self.manager.clear_context("nonexistent")
        assert result is False
    
    def test_clear_all_contexts(self):
        """Test clearing all contexts."""
        # Create multiple registries
        self.manager.get_registry("pipeline_1")
        self.manager.get_registry("pipeline_2")
        self.manager.get_registry("pipeline_3")
        
        # Verify they exist
        assert len(self.manager.list_contexts()) == 3
        
        # Clear all
        self.manager.clear_all_contexts()
        
        # Verify all cleared
        assert len(self.manager.list_contexts()) == 0
    
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
        assert "pipeline_1" in stats
        assert "pipeline_2" in stats
        
        assert stats["pipeline_1"]["step_count"] == 1
        assert stats["pipeline_2"]["step_count"] == 2
        
        assert stats["pipeline_1"]["step_type_count"] == 1
        assert stats["pipeline_2"]["step_type_count"] == 1  # Same step type
    
    def test_manager_string_representation(self):
        """Test string representation of manager."""
        # Empty manager
        repr_str = repr(self.manager)
        assert "contexts=0" in repr_str
        
        # Manager with contexts
        self.manager.get_registry("test1")
        self.manager.get_registry("test2")
        
        repr_str = repr(self.manager)
        assert "contexts=2" in repr_str

class TestConvenienceFunctions:
    """Test convenience functions for registry management."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        reset_all_global_state()
        
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
        
        yield
        reset_all_global_state()
    
    def test_get_registry_function(self):
        """Test get_registry convenience function."""
        # Get registry using convenience function
        registry = get_registry(self.manager, "test_pipeline")
        
        # Verify it works
        assert hasattr(registry, 'context_name')
        assert registry.context_name == "test_pipeline"
        
        # Verify it uses the provided manager
        assert "test_pipeline" in self.manager.list_contexts()
    
    def test_get_pipeline_registry_backward_compatibility(self):
        """Test backward compatibility function."""
        # Use old function name
        registry = get_pipeline_registry(self.manager, "my_pipeline")
        
        # Should work the same as get_registry
        assert hasattr(registry, 'context_name')
        assert registry.context_name == "my_pipeline"
    
    def test_get_default_registry_backward_compatibility(self):
        """Test backward compatibility for default registry."""
        # Get default registry
        registry = get_default_registry(self.manager)
        
        # Should be default context
        assert hasattr(registry, 'context_name')
        assert registry.context_name == "default"
    
    def test_list_contexts_function(self):
        """Test list_contexts convenience function."""
        # Initially empty
        assert len(list_contexts(self.manager)) == 0
        
        # Create some registries
        get_registry(self.manager, "pipeline_1")
        get_registry(self.manager, "pipeline_2")
        
        # Verify listing
        contexts = list_contexts(self.manager)
        assert len(contexts) == 2
        assert "pipeline_1" in contexts
        assert "pipeline_2" in contexts
    
    def test_clear_context_function(self):
        """Test clear_context convenience function."""
        # Create registry
        registry = get_registry(self.manager, "test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Verify it exists
        assert "test_pipeline" in list_contexts(self.manager)
        
        # Clear using convenience function
        result = clear_context(self.manager, "test_pipeline")
        
        # Verify clearing
        assert result is True
        assert "test_pipeline" not in list_contexts(self.manager)
    
    def test_get_context_stats_function(self):
        """Test get_context_stats convenience function."""
        # Create registry with spec
        registry = get_registry(self.manager, "test_pipeline")
        registry.register("test_step", self.test_spec)
        
        # Get stats using convenience function
        stats = get_context_stats(self.manager)
        
        # Verify stats
        assert "test_pipeline" in stats
        assert stats["test_pipeline"]["step_count"] == 1
    
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
        assert "train_step" in registry1.list_step_names()
        assert "train_step" not in registry2.list_step_names()
        assert "train_step" not in registry3.list_step_names()
        
        assert "infer_step" in registry2.list_step_names()
        assert "infer_step" not in registry1.list_step_names()
        assert "infer_step" not in registry3.list_step_names()
        
        assert "eval_step" in registry3.list_step_names()
        assert "eval_step" not in registry1.list_step_names()
        assert "eval_step" not in registry2.list_step_names()

class TestRegistryManagerErrorHandling:
    """Test error handling in RegistryManager."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        reset_all_global_state()
        
        self.manager = RegistryManager()
        
        # Create test specification
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        yield
        
        # Clean up after tests
        self.manager.clear_all_contexts()
        reset_all_global_state()
    
    def test_invalid_context_name_handling(self):
        """Test handling of invalid context names."""
        # Test empty context name - should work but create empty context
        registry = self.manager.get_registry("")
        assert registry is not None
        
        # Test None context name - implementation may handle this gracefully
        try:
            registry = self.manager.get_registry(None)
            # If no exception, verify it creates a registry
            assert registry is not None
        except (TypeError, AttributeError):
            # If exception is raised, that's also acceptable
            pass
    
    def test_registry_operations_on_cleared_context(self):
        """Test operations on cleared contexts."""
        # Create and populate registry
        registry = self.manager.get_registry("test_context")
        registry.register("test_step", self.test_spec)
        
        # Clear the context
        self.manager.clear_context("test_context")
        
        # Verify context is cleared
        assert "test_context" not in self.manager.list_contexts()
        
        # Getting the same context should create a new empty registry
        new_registry = self.manager.get_registry("test_context")
        assert len(new_registry.list_step_names()) == 0
        assert registry is not new_registry
    
    def test_concurrent_access_safety(self):
        """Test thread safety of registry operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_registry(context_name):
            try:
                registry = self.manager.get_registry(f"context_{context_name}")
                registry.register(f"step_{context_name}", self.test_spec)
                results.append(context_name)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_registry, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert len(self.manager.list_contexts()) == 10

class TestRegistryManagerMonitoring:
    """Test monitoring capabilities of RegistryManager."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        reset_all_global_state()
        
        self.manager = RegistryManager()
        
        # Create test specifications
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        self.test_spec = StepSpecification(
            step_type="TestStep",
            node_type="source",
            dependencies=[],
            outputs=[self.output_spec]
        )
        
        yield
        
        # Clean up after tests
        self.manager.clear_all_contexts()
        reset_all_global_state()
    
    def test_context_statistics_detailed(self):
        """Test detailed context statistics."""
        # Create contexts with different characteristics
        registry1 = self.manager.get_registry("small_pipeline")
        registry1.register("step1", self.test_spec)
        
        registry2 = self.manager.get_registry("large_pipeline")
        for i in range(5):
            registry2.register(f"step_{i}", self.test_spec)
        
        # Get detailed stats
        stats = self.manager.get_context_stats()
        
        # Verify detailed statistics
        assert "small_pipeline" in stats
        assert "large_pipeline" in stats
        
        small_stats = stats["small_pipeline"]
        large_stats = stats["large_pipeline"]
        
        assert small_stats["step_count"] == 1
        assert large_stats["step_count"] == 5
        
        assert small_stats["step_type_count"] == 1
        assert large_stats["step_type_count"] == 1  # All same type
        
        # Basic stats should be present
        assert "step_count" in small_stats
        assert "step_type_count" in small_stats
    
    def test_memory_usage_monitoring(self):
        """Test memory usage patterns."""
        
        # Get initial memory usage
        initial_contexts = len(self.manager.list_contexts())
        
        # Create many contexts
        for i in range(100):
            registry = self.manager.get_registry(f"context_{i}")
            registry.register(f"step_{i}", self.test_spec)
        
        # Verify contexts were created
        assert len(self.manager.list_contexts()) == 100
        
        # Clear all contexts
        self.manager.clear_all_contexts()
        
        # Verify cleanup
        assert len(self.manager.list_contexts()) == 0
