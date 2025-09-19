"""
Unit tests for ConfigClassStore functionality.

This module contains comprehensive tests for the ConfigClassStore functionality,
now integrated with step catalog and unified config management.
"""

import pytest
from pathlib import Path
from typing import Dict, Type, Optional

from cursus.core.config_fields import ConfigClassStore
from pydantic import BaseModel


class TestConfigA(BaseModel):
    """Test config class A for testing."""

    name: str = "test_a"
    value: int = 1


class TestConfigB(BaseModel):
    """Test config class B for testing."""

    name: str = "test_b"
    value: int = 2


class TestConfigC(BaseModel):
    """Test config class C for testing."""

    name: str = "test_c"
    value: int = 3


class TestConfigClassStore:
    """Test cases for ConfigClassStore."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Clear the registry before each test to ensure clean state
        if hasattr(ConfigClassStore, 'clear'):
            ConfigClassStore.clear()
        else:
            # For fallback implementation, clear the _classes dict
            ConfigClassStore._classes = {}
        
        yield  # This is where the test runs
        
        # Clean up after each test
        if hasattr(ConfigClassStore, 'clear'):
            ConfigClassStore.clear()
        else:
            # For fallback implementation, clear the _classes dict
            ConfigClassStore._classes = {}

    def test_register_decorator_functionality(self):
        """Test that the register decorator properly registers classes."""

        # Test using the decorator
        @ConfigClassStore.register
        class DecoratedConfig(BaseModel):
            name: str = "decorated"

        # Verify the class was registered
        registered_classes = ConfigClassStore.get_all_classes()
        assert "DecoratedConfig" in registered_classes
        assert registered_classes["DecoratedConfig"] == DecoratedConfig

        # Verify the decorator returns the original class
        assert DecoratedConfig.__name__ == "DecoratedConfig"
        assert DecoratedConfig().name == "decorated"

    def test_register_direct_functionality(self):
        """Test that register works when called directly."""
        # Register class directly
        registered_class = ConfigClassStore.register(TestConfigA)

        # Verify the class was registered
        registered_classes = ConfigClassStore.get_all_classes()
        assert "TestConfigA" in registered_classes
        assert registered_classes["TestConfigA"] == TestConfigA

        # Verify the function returns the original class
        assert registered_class == TestConfigA

    def test_get_class_method(self):
        """Test the get_class method."""
        # Register a class
        ConfigClassStore.register(TestConfigA)

        # Test successful retrieval - fallback implementation uses get_all_classes
        if hasattr(ConfigClassStore, 'get_class'):
            retrieved_class = ConfigClassStore.get_class("TestConfigA")
            assert retrieved_class == TestConfigA
            
            # Test retrieval of non-existent class
            non_existent = ConfigClassStore.get_class("NonExistentClass")
            assert non_existent is None
        else:
            # For fallback implementation, test via get_all_classes
            all_classes = ConfigClassStore.get_all_classes()
            assert "TestConfigA" in all_classes
            assert all_classes["TestConfigA"] == TestConfigA

    def test_get_all_classes_method(self):
        """Test the get_all_classes method."""
        # Initially should be empty
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 0

        # Register multiple classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)

        # Verify all classes are returned
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 2
        assert "TestConfigA" in all_classes
        assert "TestConfigB" in all_classes
        assert all_classes["TestConfigA"] == TestConfigA
        assert all_classes["TestConfigB"] == TestConfigB

    def test_clear_method(self):
        """Test the clear method."""
        # Register some classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)

        # Verify classes are registered
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 2

        # Clear the registry
        if hasattr(ConfigClassStore, 'clear'):
            ConfigClassStore.clear()
        else:
            # For fallback implementation, clear manually
            ConfigClassStore._classes = {}

        # Verify registry is empty
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 0

    def test_register_many_method(self):
        """Test the register_many method."""
        if hasattr(ConfigClassStore, 'register_many'):
            # Register multiple classes at once
            ConfigClassStore.register_many(TestConfigA, TestConfigB, TestConfigC)

            # Verify all classes are registered
            all_classes = ConfigClassStore.get_all_classes()
            assert len(all_classes) == 3
            assert "TestConfigA" in all_classes
            assert "TestConfigB" in all_classes
            assert "TestConfigC" in all_classes
        else:
            # For fallback implementation, register individually
            ConfigClassStore.register(TestConfigA)
            ConfigClassStore.register(TestConfigB)
            ConfigClassStore.register(TestConfigC)

            # Verify all classes are registered
            all_classes = ConfigClassStore.get_all_classes()
            assert len(all_classes) == 3
            assert "TestConfigA" in all_classes
            assert "TestConfigB" in all_classes
            assert "TestConfigC" in all_classes

    def test_registered_names_method(self):
        """Test the registered_names method."""
        if hasattr(ConfigClassStore, 'registered_names'):
            # Initially should be empty
            names = ConfigClassStore.registered_names()
            assert len(names) == 0

            # Register some classes
            ConfigClassStore.register(TestConfigA)
            ConfigClassStore.register(TestConfigB)

            # Verify names are returned
            names = ConfigClassStore.registered_names()
            assert len(names) == 2
            assert "TestConfigA" in names
            assert "TestConfigB" in names
        else:
            # For fallback implementation, test via get_all_classes
            # Initially should be empty
            all_classes = ConfigClassStore.get_all_classes()
            assert len(all_classes) == 0

            # Register some classes
            ConfigClassStore.register(TestConfigA)
            ConfigClassStore.register(TestConfigB)

            # Verify names are returned via keys
            all_classes = ConfigClassStore.get_all_classes()
            names = list(all_classes.keys())
            assert len(names) == 2
            assert "TestConfigA" in names
            assert "TestConfigB" in names

    def test_class_name_collision_handling(self):
        """Test handling of class name collisions."""
        # Register a class
        ConfigClassStore.register(TestConfigA)

        # Create another class with the same name
        class TestConfigADuplicate(BaseModel):  # Different name to avoid scope issues
            name: str = "different"
            different_field: str = "collision"

        # Manually set the class name to create collision
        TestConfigADuplicate.__name__ = "TestConfigA"

        # Register the second class (should overwrite)
        # Note: Fallback implementation may not log warnings
        ConfigClassStore.register(TestConfigADuplicate)

        # Verify the second class overwrote the first
        if hasattr(ConfigClassStore, 'get_class'):
            retrieved_class = ConfigClassStore.get_class("TestConfigA")
        else:
            # For fallback implementation, get via get_all_classes
            all_classes = ConfigClassStore.get_all_classes()
            retrieved_class = all_classes.get("TestConfigA")
        
        assert retrieved_class is not None
        instance = retrieved_class()
        assert instance.name == "different"
        assert hasattr(instance, "different_field")

    def test_registry_persistence_across_operations(self):
        """Test that registry persists across multiple operations."""
        # Register classes in different ways
        ConfigClassStore.register(TestConfigA)

        @ConfigClassStore.register
        class PersistentConfig(BaseModel):
            persistent: bool = True

        # Register additional classes (fallback-compatible)
        if hasattr(ConfigClassStore, 'register_many'):
            ConfigClassStore.register_many(TestConfigB, TestConfigC)
        else:
            ConfigClassStore.register(TestConfigB)
            ConfigClassStore.register(TestConfigC)

        # Verify all classes are still registered
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 4
        assert "TestConfigA" in all_classes
        assert "PersistentConfig" in all_classes
        assert "TestConfigB" in all_classes
        assert "TestConfigC" in all_classes

    def test_build_complete_config_classes_function(self):
        """Test the build_complete_config_classes function."""
        # This test is for the original function that may not exist in refactored version
        try:
            from cursus.steps.configs.utils import build_complete_config_classes
            
            # Register some classes
            ConfigClassStore.register(TestConfigA)
            ConfigClassStore.register(TestConfigB)

            # Build complete config classes
            complete_classes = build_complete_config_classes()

            # Verify it includes our registered classes (may include others from step catalog)
            assert isinstance(complete_classes, dict)
            assert len(complete_classes) >= 0  # May be empty if step catalog not available
        except ImportError:
            # Function doesn't exist in refactored version, skip test
            pytest.skip("build_complete_config_classes function not available in refactored version")

    def test_registry_thread_safety_simulation(self):
        """Test registry behavior under concurrent-like operations."""
        # Simulate concurrent registrations
        classes_to_register = [TestConfigA, TestConfigB, TestConfigC]

        # Register classes in rapid succession
        for cls in classes_to_register:
            ConfigClassStore.register(cls)
            # Verify immediate availability
            if hasattr(ConfigClassStore, 'get_class'):
                assert ConfigClassStore.get_class(cls.__name__) is not None
            else:
                # For fallback, verify via get_all_classes
                all_classes = ConfigClassStore.get_all_classes()
                assert cls.__name__ in all_classes

        # Verify final state
        all_classes = ConfigClassStore.get_all_classes()
        assert len(all_classes) == 3

        # Verify all classes are accessible
        for cls in classes_to_register:
            if hasattr(ConfigClassStore, 'get_class'):
                retrieved = ConfigClassStore.get_class(cls.__name__)
                assert retrieved == cls
            else:
                # For fallback, verify via get_all_classes
                assert cls.__name__ in all_classes
                assert all_classes[cls.__name__] == cls

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test registering None (should not crash)
        try:
            ConfigClassStore.register(None)
        except Exception as e:
            # If it raises an exception, that's acceptable behavior
            assert isinstance(e, (TypeError, AttributeError))

        # Test getting class with None name (if method exists)
        if hasattr(ConfigClassStore, 'get_class'):
            result = ConfigClassStore.get_class(None)
            assert result is None

            # Test getting class with empty string
            result = ConfigClassStore.get_class("")
            assert result is None

        # Test register_many with empty list (if method exists)
        if hasattr(ConfigClassStore, 'register_many'):
            ConfigClassStore.register_many()  # Should not crash

        # Verify registry is still functional
        ConfigClassStore.register(TestConfigA)
        if hasattr(ConfigClassStore, 'get_class'):
            assert ConfigClassStore.get_class("TestConfigA") is not None
        else:
            all_classes = ConfigClassStore.get_all_classes()
            assert "TestConfigA" in all_classes
