"""
Unit tests for ConfigClassStore class.

This module contains comprehensive tests for the ConfigClassStore class,
addressing the critical gap identified in the test coverage analysis.
"""

import unittest
import sys
from pathlib import Path
from typing import Dict, Type, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cursus.core.config_fields.config_class_store import ConfigClassStore
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


class TestConfigClassStore(unittest.TestCase):
    """Test cases for ConfigClassStore."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear the registry before each test to ensure clean state
        ConfigClassStore.clear()

    def tearDown(self):
        """Clean up after each test."""
        # Clear the registry after each test
        ConfigClassStore.clear()

    def test_register_decorator_functionality(self):
        """Test that the register decorator properly registers classes."""
        # Test using the decorator
        @ConfigClassStore.register
        class DecoratedConfig(BaseModel):
            name: str = "decorated"

        # Verify the class was registered
        registered_classes = ConfigClassStore.get_all_classes()
        self.assertIn("DecoratedConfig", registered_classes)
        self.assertEqual(registered_classes["DecoratedConfig"], DecoratedConfig)

        # Verify the decorator returns the original class
        self.assertEqual(DecoratedConfig.__name__, "DecoratedConfig")
        self.assertEqual(DecoratedConfig().name, "decorated")

    def test_register_direct_functionality(self):
        """Test that register works when called directly."""
        # Register class directly
        registered_class = ConfigClassStore.register(TestConfigA)
        
        # Verify the class was registered
        registered_classes = ConfigClassStore.get_all_classes()
        self.assertIn("TestConfigA", registered_classes)
        self.assertEqual(registered_classes["TestConfigA"], TestConfigA)
        
        # Verify the function returns the original class
        self.assertEqual(registered_class, TestConfigA)

    def test_get_class_method(self):
        """Test the get_class method."""
        # Register a class
        ConfigClassStore.register(TestConfigA)
        
        # Test successful retrieval
        retrieved_class = ConfigClassStore.get_class("TestConfigA")
        self.assertEqual(retrieved_class, TestConfigA)
        
        # Test retrieval of non-existent class
        non_existent = ConfigClassStore.get_class("NonExistentClass")
        self.assertIsNone(non_existent)

    def test_get_all_classes_method(self):
        """Test the get_all_classes method."""
        # Initially should be empty
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 0)
        
        # Register multiple classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)
        
        # Verify all classes are returned
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 2)
        self.assertIn("TestConfigA", all_classes)
        self.assertIn("TestConfigB", all_classes)
        self.assertEqual(all_classes["TestConfigA"], TestConfigA)
        self.assertEqual(all_classes["TestConfigB"], TestConfigB)

    def test_clear_method(self):
        """Test the clear method."""
        # Register some classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)
        
        # Verify classes are registered
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 2)
        
        # Clear the registry
        ConfigClassStore.clear()
        
        # Verify registry is empty
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 0)

    def test_register_many_method(self):
        """Test the register_many method."""
        # Register multiple classes at once
        ConfigClassStore.register_many(TestConfigA, TestConfigB, TestConfigC)
        
        # Verify all classes are registered
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 3)
        self.assertIn("TestConfigA", all_classes)
        self.assertIn("TestConfigB", all_classes)
        self.assertIn("TestConfigC", all_classes)

    def test_registered_names_method(self):
        """Test the registered_names method."""
        # Initially should be empty
        names = ConfigClassStore.registered_names()
        self.assertEqual(len(names), 0)
        
        # Register some classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)
        
        # Verify names are returned
        names = ConfigClassStore.registered_names()
        self.assertEqual(len(names), 2)
        self.assertIn("TestConfigA", names)
        self.assertIn("TestConfigB", names)

    def test_class_name_collision_handling(self):
        """Test handling of class name collisions."""
        # Register a class
        ConfigClassStore.register(TestConfigA)
        
        # Create another class with the same name
        class TestConfigA(BaseModel):  # Same name, different class
            name: str = "different"
            different_field: str = "collision"
        
        # Register the second class (should overwrite with warning)
        with self.assertLogs(level='WARNING') as log:
            ConfigClassStore.register(TestConfigA)
        
        # Verify warning was logged
        self.assertTrue(any("already registered" in message for message in log.output))
        
        # Verify the second class overwrote the first
        retrieved_class = ConfigClassStore.get_class("TestConfigA")
        instance = retrieved_class()
        self.assertEqual(instance.name, "different")
        self.assertTrue(hasattr(instance, "different_field"))

    def test_registry_persistence_across_operations(self):
        """Test that registry persists across multiple operations."""
        # Register classes in different ways
        ConfigClassStore.register(TestConfigA)
        
        @ConfigClassStore.register
        class PersistentConfig(BaseModel):
            persistent: bool = True
        
        ConfigClassStore.register_many(TestConfigB, TestConfigC)
        
        # Verify all classes are still registered
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 4)
        self.assertIn("TestConfigA", all_classes)
        self.assertIn("PersistentConfig", all_classes)
        self.assertIn("TestConfigB", all_classes)
        self.assertIn("TestConfigC", all_classes)

    def test_build_complete_config_classes_function(self):
        """Test the build_complete_config_classes function."""
        from src.cursus.core.config_fields.config_class_store import build_complete_config_classes
        
        # Register some classes
        ConfigClassStore.register(TestConfigA)
        ConfigClassStore.register(TestConfigB)
        
        # Build complete config classes
        complete_classes = build_complete_config_classes()
        
        # Verify it returns the registered classes
        self.assertIn("TestConfigA", complete_classes)
        self.assertIn("TestConfigB", complete_classes)
        self.assertEqual(complete_classes["TestConfigA"], TestConfigA)
        self.assertEqual(complete_classes["TestConfigB"], TestConfigB)

    def test_registry_thread_safety_simulation(self):
        """Test registry behavior under concurrent-like operations."""
        # Simulate concurrent registrations
        classes_to_register = [TestConfigA, TestConfigB, TestConfigC]
        
        # Register classes in rapid succession
        for cls in classes_to_register:
            ConfigClassStore.register(cls)
            # Verify immediate availability
            self.assertIsNotNone(ConfigClassStore.get_class(cls.__name__))
        
        # Verify final state
        all_classes = ConfigClassStore.get_all_classes()
        self.assertEqual(len(all_classes), 3)
        
        # Verify all classes are accessible
        for cls in classes_to_register:
            retrieved = ConfigClassStore.get_class(cls.__name__)
            self.assertEqual(retrieved, cls)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test registering None (should not crash)
        try:
            ConfigClassStore.register(None)
        except Exception as e:
            # If it raises an exception, that's acceptable behavior
            self.assertIsInstance(e, (TypeError, AttributeError))
        
        # Test getting class with None name
        result = ConfigClassStore.get_class(None)
        self.assertIsNone(result)
        
        # Test getting class with empty string
        result = ConfigClassStore.get_class("")
        self.assertIsNone(result)
        
        # Test register_many with empty list
        ConfigClassStore.register_many()  # Should not crash
        
        # Verify registry is still functional
        ConfigClassStore.register(TestConfigA)
        self.assertIsNotNone(ConfigClassStore.get_class("TestConfigA"))


if __name__ == '__main__':
    unittest.main()
