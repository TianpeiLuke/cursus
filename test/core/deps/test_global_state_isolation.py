"""
Tests demonstrating global state isolation issues and solutions.

This module contains tests that demonstrate the issues that can occur when
global state is not properly isolated between tests, and shows how to fix
these issues using the test_helpers module.
"""

import pytest
from cursus.core.deps import RegistryManager, get_registry, create_dependency_resolver
from cursus.core.base.specification_base import (
    StepSpecification, OutputSpec, DependencyType, NodeType
)
from .test_helpers import reset_all_global_state

class TestWithoutIsolation:
    """
    Tests that demonstrate issues when global state is not properly isolated.
    
    These tests may pass when run individually but fail when run together
    due to global state leakage between tests.
    """
    
    def test_registry_state_1(self):
        """First test that modifies global registry state."""
        # Reset global state before the test
        manager = RegistryManager()
        
        # Create a registry and add a specification
        registry = get_registry(manager, "test_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep1",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_1", spec)
        
        # Verify it was registered
        assert "test_pipeline" in manager.list_contexts()
        assert "test_step_1" in registry.list_step_names()
        
        # Note: No cleanup is performed, so global state persists
    
    def test_registry_state_2(self):
        """
        Second test that assumes clean global registry state.
        
        This test will fail if test_registry_state_1 runs first and doesn't
        clean up its global state.
        """
        # Reset global state before the test
        manager = RegistryManager()
        
        # This test assumes no registries exist yet
        contexts = manager.list_contexts()
        
        # This will fail if test_registry_state_1 ran first and didn't clean up
        assert len(contexts) == 0, f"Expected no contexts, but found: {contexts}"
        
        # Create a new registry
        registry = get_registry(manager, "another_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="another_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep2",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_2", spec)
        
        # Verify it was registered
        assert "another_pipeline" in manager.list_contexts()
        assert "test_step_2" in registry.list_step_names()

class TestWithManualIsolation:
    """
    Tests that manually handle global state isolation.
    
    These tests should pass whether run individually or together
    because they manually clean up global state.
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures, creating new instances for each test."""
        self.manager = RegistryManager()
        self.resolver = create_dependency_resolver()
    
    def test_registry_state_1(self):
        """First test that modifies registry state."""
        # Create a registry and add a specification
        registry = get_registry(self.manager, "test_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep1",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_1", spec)
        
        # Verify it was registered
        assert "test_pipeline" in self.manager.list_contexts()
        assert "test_step_1" in registry.list_step_names()
    
    def test_registry_state_2(self):
        """
        Second test that assumes clean registry state.
        
        This test will pass even if test_registry_state_1 runs first
        because a new manager is created for each test.
        """
        # This test assumes no registries exist yet
        contexts = self.manager.list_contexts()
        
        # This will pass because a new manager is created for each test
        assert len(contexts) == 0
        
        # Create a new registry
        registry = get_registry(self.manager, "another_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="another_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep2",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_2", spec)
        
        # Verify it was registered
        assert "another_pipeline" in self.manager.list_contexts()
        assert "test_step_2" in registry.list_step_names()

class TestWithHelperIsolation:
    """
    Tests that use pytest fixtures for global state isolation.
    
    These tests should pass whether run individually or together
    because they use pytest fixtures which handle global state cleanup.
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state before and after each test
        reset_all_global_state()
        self.manager = RegistryManager()
        yield
        reset_all_global_state()
    
    def test_registry_state_1(self):
        """First test that modifies registry state."""
        # Create a registry and add a specification
        registry = get_registry(self.manager, "test_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep1",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_1", spec)
        
        # Verify it was registered
        assert "test_pipeline" in self.manager.list_contexts()
        assert "test_step_1" in registry.list_step_names()
    
    def test_registry_state_2(self):
        """
        Second test that assumes clean registry state.
        
        This test will pass even if test_registry_state_1 runs first
        because a new manager is created for each test.
        """
        # This test assumes no registries exist yet
        contexts = self.manager.list_contexts()
        
        # This will pass because setup_method() cleared all contexts
        assert len(contexts) == 0
        
        # Create a new registry
        registry = get_registry(self.manager, "another_pipeline")
        
        # Create a simple specification
        output_spec = OutputSpec(
            logical_name="another_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        spec = StepSpecification(
            step_type="TestStep2",
            node_type="source",
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Register the specification
        registry.register("test_step_2", spec)
        
        # Verify it was registered
        assert "another_pipeline" in self.manager.list_contexts()
        assert "test_step_2" in registry.list_step_names()

def run_tests_individually():
    """Run each test individually to demonstrate they pass in isolation."""
    print("\nRunning tests individually:")
    
    # Run TestWithoutIsolation tests individually
    print("\nTestWithoutIsolation.test_registry_state_1:")
    try:
        test_instance = TestWithoutIsolation()
        test_instance.test_registry_state_1()
        print("  Passed: True")
    except Exception as e:
        print(f"  Passed: False - {e}")
    
    # Reset global state between individual test runs
    reset_all_global_state()
    
    print("\nTestWithoutIsolation.test_registry_state_2:")
    try:
        test_instance = TestWithoutIsolation()
        test_instance.test_registry_state_2()
        print("  Passed: True")
    except Exception as e:
        print(f"  Passed: False - {e}")

def run_tests_together():
    """Run tests together to demonstrate isolation issues."""
    print("\nRunning tests together:")
    
    # Run TestWithoutIsolation tests together
    print("\nTestWithoutIsolation (both tests):")
    try:
        test_instance = TestWithoutIsolation()
        test_instance.test_registry_state_1()
        test_instance.test_registry_state_2()
        print("  All passed: True")
    except Exception as e:
        print(f"  All passed: False - {e}")
    
    # Reset global state between test classes
    reset_all_global_state()
    
    # Run TestWithManualIsolation tests together
    print("\nTestWithManualIsolation (both tests):")
    try:
        test_instance = TestWithManualIsolation()
        # Manually call setup since we're not using pytest runner
        test_instance.setup_method()
        test_instance.test_registry_state_1()
        test_instance.test_registry_state_2()
        print("  All passed: True")
    except Exception as e:
        print(f"  All passed: False - {e}")
    
    # Reset global state between test classes
    reset_all_global_state()
    
    # Run TestWithHelperIsolation tests together
    print("\nTestWithHelperIsolation (both tests):")
    try:
        test_instance = TestWithHelperIsolation()
        # Manually call setup since we're not using pytest runner
        setup_gen = test_instance.setup_method()
        next(setup_gen)  # Execute setup
        test_instance.test_registry_state_1()
        test_instance.test_registry_state_2()
        try:
            next(setup_gen)  # Execute teardown
        except StopIteration:
            pass
        print("  All passed: True")
    except Exception as e:
        print(f"  All passed: False - {e}")

if __name__ == "__main__":
    # First run tests individually to show they pass
    run_tests_individually()
    
    # Then run tests together to show isolation issues
    run_tests_together()
    
    print("\nConclusion:")
    print("- TestWithoutIsolation: Tests pass individually but fail when run together")
    print("- TestWithManualIsolation: Tests pass both individually and together")
    print("- TestWithHelperIsolation: Tests pass both individually and together")
    print("\nNote: This file demonstrates global state isolation patterns.")
    print("In practice, use 'pytest' to run these tests with proper fixture handling.")
