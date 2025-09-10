"""
Tests for factory functions and component creation.

Tests the factory module functionality including:
- Pipeline component creation
- Thread-local component management
- Dependency resolution context management
- Component wiring and integration
"""

import pytest
import threading
import time
from contextlib import contextmanager
from cursus.core.deps.factory import (
    create_pipeline_components, 
    get_thread_components, 
    dependency_resolution_context
)
from cursus.core.deps import (
    SemanticMatcher, 
    SpecificationRegistry, 
    RegistryManager, 
    UnifiedDependencyResolver
)
from cursus.core.base.specification_base import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)
from .test_helpers import reset_all_global_state

class TestFactoryFunctions:
    """Test factory function functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        reset_all_global_state()
        
        # Create test specification for testing
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
        reset_all_global_state()
    
    def test_create_pipeline_components_default(self):
        """Test creating pipeline components with default context."""
        components = create_pipeline_components()
        
        # Verify all components are created
        assert "semantic_matcher" in components
        assert "registry_manager" in components
        assert "registry" in components
        assert "resolver" in components
        
        # Verify component types
        assert isinstance(components["semantic_matcher"], SemanticMatcher)
        assert isinstance(components["registry_manager"], RegistryManager)
        assert isinstance(components["registry"], SpecificationRegistry)
        assert isinstance(components["resolver"], UnifiedDependencyResolver)
        
        # Verify default context
        assert components["registry"].context_name == "default"
    
    def test_create_pipeline_components_custom_context(self):
        """Test creating pipeline components with custom context."""
        context_name = "test_pipeline"
        components = create_pipeline_components(context_name)
        
        # Verify custom context is used
        assert components["registry"].context_name == context_name
        
        # Verify registry manager contains the context
        assert context_name in components["registry_manager"].list_contexts()
    
    def test_create_pipeline_components_wiring(self):
        """Test that components are properly wired together."""
        components = create_pipeline_components("test_context")
        
        # Test that resolver can use the registry
        registry = components["registry"]
        resolver = components["resolver"]
        
        # Register a test specification
        registry.register("test_step", self.test_spec)
        
        # Verify resolver can find the specification
        found_specs = registry.get_specifications_by_type("TestStep")
        assert len(found_specs) == 1
        assert found_specs[0].step_type == "TestStep"
    
    def test_create_pipeline_components_isolation(self):
        """Test that different component sets are isolated."""
        components1 = create_pipeline_components("context1")
        components2 = create_pipeline_components("context2")
        
        # Verify different instances
        assert components1["registry_manager"] is not components2["registry_manager"]
        assert components1["registry"] is not components2["registry"]
        assert components1["resolver"] is not components2["resolver"]
        
        # Verify different contexts
        assert components1["registry"].context_name == "context1"
        assert components2["registry"].context_name == "context2"
        
        # Test isolation by registering different specs
        components1["registry"].register("step1", self.test_spec)
        
        # Verify isolation
        assert "step1" in components1["registry"].list_step_names()
        assert "step1" not in components2["registry"].list_step_names()

class TestThreadLocalComponents:
    """Test thread-local component management."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        reset_all_global_state()
        
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
        reset_all_global_state()
    
    def test_get_thread_components_single_thread(self):
        """Test getting thread components in single thread."""
        components1 = get_thread_components()
        components2 = get_thread_components()
        
        # Should return same instances within same thread
        assert components1["registry_manager"] is components2["registry_manager"]
        assert components1["registry"] is components2["registry"]
        assert components1["resolver"] is components2["resolver"]
        
        # Verify components are properly initialized
        assert isinstance(components1["semantic_matcher"], SemanticMatcher)
        assert isinstance(components1["registry_manager"], RegistryManager)
        assert isinstance(components1["registry"], SpecificationRegistry)
        assert isinstance(components1["resolver"], UnifiedDependencyResolver)
    
    def test_get_thread_components_multi_thread(self):
        """Test thread isolation of components."""
        results = {}
        errors = []
        
        def thread_worker(thread_id):
            try:
                components = get_thread_components()
                # Register a spec specific to this thread
                components["registry"].register(f"step_{thread_id}", self.test_spec)
                results[thread_id] = {
                    "components": components,
                    "step_names": components["registry"].list_step_names()
                }
            except Exception as e:
                errors.append((thread_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all threads got results
        assert len(results) == 5
        
        # Verify thread isolation
        for thread_id, result in results.items():
            step_names = result["step_names"]
            assert len(step_names) == 1
            assert f"step_{thread_id}" in step_names
            
            # Verify no cross-contamination
            for other_id in results:
                if other_id != thread_id:
                    assert f"step_{other_id}" not in step_names
        
        # Verify different component instances across threads
        component_instances = [result["components"]["registry"] for result in results.values()]
        for i, comp1 in enumerate(component_instances):
            for j, comp2 in enumerate(component_instances):
                if i != j:
                    assert comp1 is not comp2
    
    def test_thread_components_persistence(self):
        """Test that thread components persist across multiple calls."""
        # First call
        components1 = get_thread_components()
        components1["registry"].register("persistent_step", self.test_spec)
        
        # Second call
        components2 = get_thread_components()
        
        # Should be same instances
        assert components1["registry"] is components2["registry"]
        
        # Should maintain state
        assert "persistent_step" in components2["registry"].list_step_names()

class TestDependencyResolutionContext:
    """Test dependency resolution context management."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        reset_all_global_state()
        
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
        reset_all_global_state()
    
    def test_dependency_resolution_context_basic(self):
        """Test basic context manager functionality."""
        with dependency_resolution_context() as components:
            # Verify components are provided
            assert "semantic_matcher" in components
            assert "registry_manager" in components
            assert "registry" in components
            assert "resolver" in components
            
            # Verify component types
            assert isinstance(components["semantic_matcher"], SemanticMatcher)
            assert isinstance(components["registry_manager"], RegistryManager)
            assert isinstance(components["registry"], SpecificationRegistry)
            assert isinstance(components["resolver"], UnifiedDependencyResolver)
            
            # Test functionality within context
            components["registry"].register("test_step", self.test_spec)
            assert "test_step" in components["registry"].list_step_names()
    
    def test_dependency_resolution_context_cleanup(self):
        """Test context cleanup on exit."""
        registry_manager = None
        resolver = None
        
        with dependency_resolution_context(clear_on_exit=True) as components:
            registry_manager = components["registry_manager"]
            resolver = components["resolver"]
            
            # Add some data
            components["registry"].register("test_step", self.test_spec)
            assert "test_step" in components["registry"].list_step_names()
            
            # Verify context exists
            assert "default" in registry_manager.list_contexts()
        
        # After context exit, should be cleaned up
        assert len(registry_manager.list_contexts()) == 0
    
    def test_dependency_resolution_context_no_cleanup(self):
        """Test context without cleanup."""
        registry_manager = None
        
        with dependency_resolution_context(clear_on_exit=False) as components:
            registry_manager = components["registry_manager"]
            
            # Add some data
            components["registry"].register("test_step", self.test_spec)
            assert "test_step" in components["registry"].list_step_names()
            
            # Verify context exists
            assert "default" in registry_manager.list_contexts()
        
        # After context exit, should still exist
        assert "default" in registry_manager.list_contexts()
        
        # Clean up manually for test isolation
        registry_manager.clear_all_contexts()
    
    def test_dependency_resolution_context_exception_handling(self):
        """Test context cleanup on exception."""
        registry_manager = None
        
        try:
            with dependency_resolution_context(clear_on_exit=True) as components:
                registry_manager = components["registry_manager"]
                
                # Add some data
                components["registry"].register("test_step", self.test_spec)
                assert "default" in registry_manager.list_contexts()
                
                # Raise exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception
        
        # Should still be cleaned up despite exception
        assert len(registry_manager.list_contexts()) == 0
    
    def test_dependency_resolution_context_nested(self):
        """Test nested context usage."""
        with dependency_resolution_context() as outer_components:
            outer_components["registry"].register("outer_step", self.test_spec)
            
            with dependency_resolution_context() as inner_components:
                inner_components["registry"].register("inner_step", self.test_spec)
                
                # Verify isolation
                assert "outer_step" in outer_components["registry"].list_step_names()
                assert "inner_step" in inner_components["registry"].list_step_names()
                
                # Verify different instances
                assert outer_components["registry"] is not inner_components["registry"]
                assert "inner_step" not in outer_components["registry"].list_step_names()
                assert "outer_step" not in inner_components["registry"].list_step_names()

class TestFactoryIntegration:
    """Test integration scenarios with factory components."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        reset_all_global_state()
        
        # Create test specifications
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
        
        yield
        reset_all_global_state()
    
    def test_end_to_end_pipeline_creation(self):
        """Test complete pipeline creation and dependency resolution."""
        with dependency_resolution_context() as components:
            registry = components["registry"]
            resolver = components["resolver"]
            
            # Register pipeline steps
            registry.register("data_step", self.data_spec)
            registry.register("training_step", self.training_spec)
            
            # Test dependency resolution
            available_steps = ["data_step", "training_step"]
            dependencies = resolver.resolve_all_dependencies(available_steps)
            
            # Verify resolution worked
            assert "training_step" in dependencies
            training_deps = dependencies["training_step"]
            assert len(training_deps) == 1
            assert "input_data" in training_deps
            
            # Verify the property reference points to the correct step
            prop_ref = training_deps["input_data"]
            assert prop_ref.step_name == "data_step"
    
    def test_multi_context_pipeline_isolation(self):
        """Test isolation between multiple pipeline contexts."""
        # Create two separate pipeline contexts
        training_components = create_pipeline_components("training")
        inference_components = create_pipeline_components("inference")
        
        # Register different specs in each
        training_components["registry"].register("train_data", self.data_spec)
        training_components["registry"].register("train_model", self.training_spec)
        
        inference_components["registry"].register("infer_data", self.data_spec)
        
        # Verify isolation
        training_steps = training_components["registry"].list_step_names()
        inference_steps = inference_components["registry"].list_step_names()
        
        assert len(training_steps) == 2
        assert len(inference_steps) == 1
        
        assert "train_data" in training_steps
        assert "train_model" in training_steps
        assert "infer_data" in inference_steps
        
        # Verify no cross-contamination
        assert "infer_data" not in training_steps
        assert "train_data" not in inference_steps
        assert "train_model" not in inference_steps
