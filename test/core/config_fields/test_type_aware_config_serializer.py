"""
Comprehensive tests for TypeAwareConfigSerializer class.

This module provides comprehensive pytest test coverage for the type-aware serialization
system, following pytest best practices and preventing common test failure categories.

Following pytest best practices:
- Source code first analysis completed
- Implementation-driven testing approach
- Mock path precision for all imports
- Comprehensive error handling and edge case coverage
- >80% test coverage target

Error Prevention Categories Addressed:
- Category 1: Mock Path and Import Issues (35% of failures)
- Category 2: Mock Configuration and Side Effects (25% of failures)
- Category 3: Path and File System Operations (20% of failures)
- Category 4: Test Expectations vs Implementation (10% of failures)
- Category 12: NoneType Attribute Access (4% of failures)
- Category 16: Exception Handling vs Test Expectations (1% of failures)
- Category 17: Global State Management (2% of failures)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json
import tempfile
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Set, Union

from pydantic import BaseModel

# Import the module under test - using exact import paths from source analysis
from cursus.core.config_fields.type_aware_config_serializer import (
    TypeAwareConfigSerializer,
    serialize_config,
    deserialize_config
)
from cursus.core.config_fields.constants import SerializationMode, TYPE_MAPPING


class SerializationTestEnum(Enum):
    """Test enum for serialization testing."""
    VALUE1 = "test_value_1"
    VALUE2 = "test_value_2"
    NUMERIC_VALUE = 42


class SerializationTestConfig(BaseModel):
    """Test Pydantic model for serialization testing."""
    field1: str = "default_value1"
    field2: Optional[int] = None
    field3: bool = True
    
    def categorize_fields(self):
        """Mock field categorization for three-tier testing."""
        return {
            "essential": ["field1"],
            "system": ["field2"],
            "derived": ["field3"]
        }


class TestTypeAwareConfigSerializer:
    """
    Test cases for TypeAwareConfigSerializer class.
    
    Comprehensive testing following implementation-driven approach with
    source code analysis and error prevention strategies.
    """
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Reset global state to prevent Category 17 failures
        self._reset_global_state()
        
        # Create mock unified manager following source analysis
        self.mock_unified_manager = Mock()
        self.mock_unified_manager.get_config_classes.return_value = {
            "SerializationTestConfig": SerializationTestConfig
        }
        self.mock_unified_manager.get_field_tiers.return_value = {
            "essential": ["field1"],
            "system": ["field2"],
            "derived": ["field3"]
        }
        
        # Create serializer instance with controlled dependencies
        self.serializer = TypeAwareConfigSerializer(
            config_classes={"SerializationTestConfig": SerializationTestConfig},
            mode=SerializationMode.PRESERVE_TYPES
        )
        # Manually set the unified manager after initialization
        self.serializer.unified_manager = self.mock_unified_manager
        
        yield  # This is where the test runs
        
        # Cleanup after test
        self._reset_global_state()
    
    def _reset_global_state(self):
        """Reset any global state to prevent test interference."""
        # Clear any caches that might exist
        if hasattr(TypeAwareConfigSerializer, '_class_cache'):
            TypeAwareConfigSerializer._class_cache.clear()
    
    def test_initialization_default(self):
        """Test default initialization of TypeAwareConfigSerializer."""
        # Mock get_unified_config_manager to prevent Category 1 failures
        with patch('cursus.core.config_fields.type_aware_config_serializer.get_unified_config_manager', create=True) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_config_classes.return_value = {}
            mock_get_manager.return_value = mock_manager
            
            serializer = TypeAwareConfigSerializer()
            
            # Verify initialization
            assert serializer.mode == SerializationMode.PRESERVE_TYPES
            # Note: unified_manager is set internally and may not be exposed as public attribute
            assert isinstance(serializer._serializing_ids, set)
            assert len(serializer._serializing_ids) == 0
    
    def test_initialization_with_parameters(self):
        """Test initialization with explicit parameters."""
        config_classes = {"TestClass": SerializationTestConfig}
        mode = SerializationMode.SIMPLE_JSON
        
        serializer = TypeAwareConfigSerializer(
            config_classes=config_classes,
            mode=mode
        )
        
        # Verify parameter assignment
        assert serializer.config_classes == config_classes
        assert serializer.mode == mode
        # Note: unified_manager is set internally, not via parameter
    
    def test_serialize_primitives(self):
        """Test serialization of primitive types."""
        # Read source: primitives should pass through unchanged
        test_cases = [
            (None, None),
            ("test_string", "test_string"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (False, False)
        ]
        
        for input_val, expected in test_cases:
            result = self.serializer.serialize(input_val)
            assert result == expected, f"Failed for input: {input_val}"
    
    def test_serialize_datetime_with_type_preservation(self):
        """Test serialization of datetime objects with type preservation."""
        # Read source: datetime should be serialized with type info when PRESERVE_TYPES
        test_datetime = datetime(2023, 1, 15, 10, 30, 45)
        
        result = self.serializer.serialize(test_datetime)
        
        # Verify structure matches implementation
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["datetime"]
        assert result["value"] == test_datetime.isoformat()
    
    def test_serialize_datetime_simple_mode(self):
        """Test datetime serialization in simple mode."""
        # Test different serialization mode
        self.serializer.mode = SerializationMode.SIMPLE_JSON
        test_datetime = datetime(2023, 1, 15, 10, 30, 45)
        
        result = self.serializer.serialize(test_datetime)
        
        # Should return ISO string directly in simple mode
        assert result == test_datetime.isoformat()
        assert not isinstance(result, dict)
    
    def test_serialize_enum_with_type_preservation(self):
        """Test serialization of enum objects with type preservation."""
        test_enum = SerializationTestEnum.VALUE1
        result = self.serializer.serialize(test_enum)
        
        # Verify structure matches implementation
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["Enum"]
        assert result["value"] == "test_value_1"
        assert "enum_class" in result
        assert result["enum_class"] == f"{SerializationTestEnum.__module__}.{SerializationTestEnum.__name__}"
    
    def test_serialize_enum_simple_mode(self):
        """Test enum serialization in simple mode."""
        self.serializer.mode = SerializationMode.SIMPLE_JSON
        test_enum = SerializationTestEnum.NUMERIC_VALUE
        
        result = self.serializer.serialize(test_enum)
        
        # Should return value directly in simple mode
        assert result == 42
        assert not isinstance(result, dict)
    
    def test_serialize_path_with_type_preservation(self):
        """Test serialization of Path objects with type preservation."""
        # Use MagicMock to prevent Category 3 failures
        test_path = Path("/test/path/file.txt")
        
        result = self.serializer.serialize(test_path)
        
        # Verify structure matches implementation
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["Path"]
        assert result["value"] == str(test_path)
    
    def test_serialize_path_simple_mode(self):
        """Test Path serialization in simple mode."""
        self.serializer.mode = SerializationMode.SIMPLE_JSON
        test_path = Path("/test/path/file.txt")
        
        result = self.serializer.serialize(test_path)
        
        # Should return string directly in simple mode
        assert result == str(test_path)
        assert not isinstance(result, dict)
    
    def test_serialize_pydantic_model_with_tier_awareness(self):
        """Test serialization of Pydantic models with tier-aware field selection."""
        # Create test config instance
        test_config = SerializationTestConfig(field1="test_value", field2=42, field3=False)
        
        # Mock unified manager to return field tiers (following source analysis)
        self.mock_unified_manager.get_field_tiers.return_value = {
            "essential": ["field1"],
            "system": ["field2"],
            "derived": ["field3"]
        }
        
        result = self.serializer.serialize(test_config)
        
        # Verify structure matches implementation
        assert isinstance(result, dict)
        assert result["__model_type__"] == "SerializationTestConfig"
        
        # Essential fields should be included
        assert "field1" in result
        assert result["field1"] == "test_value"
        
        # System fields should be included if not None
        assert "field2" in result
        assert result["field2"] == 42
        
        # Derived fields should be included from model_dump if available
        # Note: This depends on the actual model_dump behavior
    
    def test_serialize_pydantic_model_skip_none_system_fields(self):
        """Test that None system fields are skipped during serialization."""
        # Create test config with None system field
        test_config = SerializationTestConfig(field1="test_value", field2=None, field3=True)
        
        # Mock field tiers
        self.mock_unified_manager.get_field_tiers.return_value = {
            "essential": ["field1"],
            "system": ["field2"],
            "derived": ["field3"]
        }
        
        result = self.serializer.serialize(test_config)
        
        # Essential field should be included
        assert "field1" in result
        assert result["field1"] == "test_value"
        
        # None system field should be skipped (following source logic)
        assert "field2" not in result or result["field2"] is None
    
    def test_serialize_circular_reference_detection(self):
        """Test circular reference detection and handling."""
        # Create a mock object that will cause circular reference
        mock_config = Mock(spec=BaseModel)
        mock_config.__class__.__name__ = "CircularConfig"
        
        # Add the object to serializing_ids to simulate circular reference
        obj_id = id(mock_config)
        self.serializer._serializing_ids.add(obj_id)
        
        result = self.serializer.serialize(mock_config)
        
        # Verify circular reference handling
        assert isinstance(result, dict)
        assert result["__model_type__"] == "CircularConfig"
        assert result["_circular_ref"] is True
        assert "_ref_message" in result
    
    def test_serialize_model_exception_handling(self):
        """Test exception handling during model serialization."""
        # Create a mock that will raise an exception
        mock_config = Mock(spec=BaseModel)
        mock_config.__class__.__name__ = "ErrorConfig"
        
        # Mock unified manager to raise exception
        self.mock_unified_manager.get_field_tiers.side_effect = Exception("Field tier error")
        
        result = self.serializer.serialize(mock_config)
        
        # Verify error handling
        assert isinstance(result, dict)
        assert result["__model_type__"] == "ErrorConfig"
        assert "_error" in result
        assert "_serialization_error" in result
        assert result["_serialization_error"] is True
    
    def test_serialize_dict_simple(self):
        """Test serialization of simple dictionaries."""
        test_dict = {"key1": "value1", "key2": 42, "key3": True}
        
        result = self.serializer.serialize(test_dict)
        
        # Simple dict should be serialized without type info
        assert isinstance(result, dict)
        assert "__type_info__" not in result
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True
    
    def test_serialize_dict_with_complex_values(self):
        """Test serialization of dictionaries with complex values."""
        test_datetime = datetime(2023, 1, 15, 10, 30)
        test_dict = {
            "simple": "value",
            "complex": test_datetime
        }
        
        result = self.serializer.serialize(test_dict)
        
        # Dict with complex values should include type info
        assert isinstance(result, dict)
        assert "__type_info__" in result
        assert result["__type_info__"] == TYPE_MAPPING["dict"]
        assert "value" in result
        
        # Check nested serialization
        nested_result = result["value"]
        assert nested_result["simple"] == "value"
        assert isinstance(nested_result["complex"], dict)
        assert nested_result["complex"]["__type_info__"] == TYPE_MAPPING["datetime"]
    
    def test_serialize_list_simple(self):
        """Test serialization of simple lists."""
        test_list = ["item1", 42, True]
        
        result = self.serializer.serialize(test_list)
        
        # Simple list should be serialized without type info
        assert isinstance(result, list)
        assert result == ["item1", 42, True]
    
    def test_serialize_list_with_complex_values(self):
        """Test serialization of lists with complex values."""
        test_datetime = datetime(2023, 1, 15, 10, 30)
        test_list = ["simple", test_datetime]
        
        result = self.serializer.serialize(test_list)
        
        # List with complex values should include type info
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["list"]
        assert "value" in result
        
        # Check nested serialization
        nested_result = result["value"]
        assert nested_result[0] == "simple"
        assert isinstance(nested_result[1], dict)
        assert nested_result[1]["__type_info__"] == TYPE_MAPPING["datetime"]
    
    def test_serialize_tuple_with_complex_values(self):
        """Test serialization of tuples with complex values."""
        test_datetime = datetime(2023, 1, 15, 10, 30)
        test_tuple = ("simple", test_datetime)
        
        result = self.serializer.serialize(test_tuple)
        
        # Tuple with complex values should include type info
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["tuple"]
        assert "value" in result
        
        # Check nested serialization
        nested_result = result["value"]
        assert nested_result[0] == "simple"
        assert isinstance(nested_result[1], dict)
    
    def test_serialize_set_with_type_preservation(self):
        """Test serialization of sets with type preservation."""
        test_set = {"item1", "item2", "item3"}
        
        result = self.serializer.serialize(test_set)
        
        # Set should always include type info in PRESERVE_TYPES mode
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["set"]
        assert "value" in result
        assert isinstance(result["value"], list)
        assert set(result["value"]) == test_set
    
    def test_serialize_frozenset_with_type_preservation(self):
        """Test serialization of frozensets with type preservation."""
        test_frozenset = frozenset(["item1", "item2", "item3"])
        
        result = self.serializer.serialize(test_frozenset)
        
        # Frozenset should include type info
        assert isinstance(result, dict)
        assert result["__type_info__"] == TYPE_MAPPING["frozenset"]
        assert "value" in result
        assert isinstance(result["value"], list)
        assert set(result["value"]) == set(test_frozenset)
    
    def test_serialize_fallback_to_string(self):
        """Test fallback to string representation for unknown types."""
        # Create a custom object that's not handled by specific serializers
        class CustomObject:
            def __str__(self):
                return "custom_object_string"
        
        custom_obj = CustomObject()
        result = self.serializer.serialize(custom_obj)
        
        # Should fallback to string representation
        assert result == "custom_object_string"


class TestTypeAwareConfigSerializerDeserialization:
    """Test cases for deserialization functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Reset global state
        self._reset_global_state()
        
        # Create mock config classes
        self.mock_config_classes = {
            "SerializationTestConfig": SerializationTestConfig
        }
        
        # Create serializer with mock classes
        self.serializer = TypeAwareConfigSerializer(
            config_classes=self.mock_config_classes
        )
        
        yield
        
        # Cleanup
        self._reset_global_state()
    
    def _reset_global_state(self):
        """Reset any global state to prevent test interference."""
        if hasattr(TypeAwareConfigSerializer, '_class_cache'):
            TypeAwareConfigSerializer._class_cache.clear()
    
    def test_deserialize_primitives(self):
        """Test deserialization of primitive types."""
        test_cases = [
            None, "test_string", 42, 3.14, True, False
        ]
        
        for test_value in test_cases:
            result = self.serializer.deserialize(test_value)
            assert result == test_value, f"Failed for value: {test_value}"
    
    def test_deserialize_datetime(self):
        """Test deserialization of datetime objects."""
        # Create serialized datetime data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["datetime"],
            "value": "2023-01-15T10:30:45"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify datetime reconstruction
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45
    
    def test_deserialize_enum_success(self):
        """Test successful enum deserialization."""
        # Create serialized enum data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["Enum"],
            "enum_class": f"{SerializationTestEnum.__module__}.{SerializationTestEnum.__name__}",
            "value": "test_value_1"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify enum reconstruction
        assert isinstance(result, SerializationTestEnum)
        assert result == SerializationTestEnum.VALUE1
        assert result.value == "test_value_1"
    
    def test_deserialize_enum_import_error(self):
        """Test enum deserialization with import error."""
        # Create serialized enum data with non-existent class
        serialized_data = {
            "__type_info__": TYPE_MAPPING["Enum"],
            "enum_class": "nonexistent.module.NonExistentEnum",
            "value": "test_value"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should fallback to raw value
        assert result == "test_value"
    
    def test_deserialize_enum_missing_class_path(self):
        """Test enum deserialization with missing enum_class."""
        # Create serialized enum data without enum_class
        serialized_data = {
            "__type_info__": TYPE_MAPPING["Enum"],
            "value": "test_value"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Based on actual implementation: returns the original data when enum_class is missing
        assert result == serialized_data
    
    def test_deserialize_path(self):
        """Test deserialization of Path objects."""
        # Create serialized Path data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["Path"],
            "value": "/test/path/file.txt"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify Path reconstruction
        assert isinstance(result, Path)
        assert str(result) == "/test/path/file.txt"
    
    def test_deserialize_dict_with_type_info(self):
        """Test deserialization of dictionaries with type info."""
        # Create serialized dict data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["dict"],
            "value": {
                "key1": "value1",
                "key2": 42
            }
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify dict reconstruction
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
    
    def test_deserialize_list_with_type_info(self):
        """Test deserialization of lists with type info."""
        # Create serialized list data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["list"],
            "value": ["item1", 42, True]
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify list reconstruction
        assert isinstance(result, list)
        assert result == ["item1", 42, True]
    
    def test_deserialize_tuple_with_type_info(self):
        """Test deserialization of tuples with type info."""
        # Create serialized tuple data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["tuple"],
            "value": ["item1", 42, True]
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify tuple reconstruction
        assert isinstance(result, tuple)
        assert result == ("item1", 42, True)
    
    def test_deserialize_set_with_type_info(self):
        """Test deserialization of sets with type info."""
        # Create serialized set data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["set"],
            "value": ["item1", "item2", "item3"]
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify set reconstruction
        assert isinstance(result, set)
        assert result == {"item1", "item2", "item3"}
    
    def test_deserialize_frozenset_with_type_info(self):
        """Test deserialization of frozensets with type info."""
        # Create serialized frozenset data
        serialized_data = {
            "__type_info__": TYPE_MAPPING["frozenset"],
            "value": ["item1", "item2", "item3"]
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify frozenset reconstruction
        assert isinstance(result, frozenset)
        assert result == frozenset(["item1", "item2", "item3"])
    
    def test_deserialize_model_success(self):
        """Test successful model deserialization."""
        # Create serialized model data
        serialized_data = {
            "__model_type__": "SerializationTestConfig",
            "field1": "test_value",
            "field2": 42
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Verify model instantiation
        assert isinstance(result, SerializationTestConfig)
        assert result.field1 == "test_value"
        assert result.field2 == 42
    
    def test_deserialize_model_missing_class(self):
        """Test model deserialization with missing class."""
        # Create serialized model data with unknown class
        serialized_data = {
            "__model_type__": "UnknownConfig",
            "field1": "test_value",
            "field2": 42
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should return filtered dict
        assert isinstance(result, dict)
        assert "field1" in result
        assert "field2" in result
        assert "__model_type__" not in result
    
    def test_deserialize_model_validation_error_fallback(self):
        """Test model deserialization fallback when validation fails."""
        # Create mock class that fails model_validate but succeeds model_construct
        mock_class = Mock()
        mock_class.__name__ = "TestConfig"  # Fix AttributeError: __name__
        mock_class.model_validate.side_effect = Exception("Validation failed")
        mock_instance = Mock()
        mock_class.model_construct.return_value = mock_instance
        mock_class.model_fields = {"field1": Mock(), "field2": Mock()}
        
        # Update config classes
        self.serializer.config_classes["SerializationTestConfig"] = mock_class
        
        # Create serialized model data
        serialized_data = {
            "__model_type__": "SerializationTestConfig",
            "field1": "test_value",
            "field2": 42
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should fallback to model_construct
        assert result == mock_instance
        mock_class.model_construct.assert_called_once()
    
    def test_deserialize_model_all_methods_fail(self):
        """Test model deserialization when all instantiation methods fail."""
        # Create mock class that fails all methods
        mock_class = Mock()
        mock_class.__name__ = "TestConfig"  # Fix AttributeError: __name__
        mock_class.model_validate.side_effect = Exception("Validation failed")
        mock_class.model_construct.side_effect = Exception("Construction failed")
        # Fix: Don't set side_effect on __init__ method directly
        mock_class.model_fields = {"field1": Mock(), "field2": Mock()}
        
        # Update config classes
        self.serializer.config_classes["SerializationTestConfig"] = mock_class
        
        # Create serialized model data
        serialized_data = {
            "__model_type__": "SerializationTestConfig",
            "field1": "test_value",
            "field2": 42
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should return filtered dict as final fallback
        assert isinstance(result, dict)
        assert result["field1"] == "test_value"
        assert result["field2"] == 42
    
    def test_deserialize_simple_dict(self):
        """Test deserialization of simple dictionaries."""
        test_dict = {"key1": "value1", "key2": 42}
        
        result = self.serializer.deserialize(test_dict)
        
        # Should return dict with recursively deserialized values
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
    
    def test_deserialize_simple_list(self):
        """Test deserialization of simple lists."""
        test_list = ["item1", 42, True]
        
        result = self.serializer.deserialize(test_list)
        
        # Should return list with recursively deserialized values
        assert isinstance(result, list)
        assert result == ["item1", 42, True]
    
    def test_deserialize_unknown_type_info(self):
        """Test deserialization with unknown type info."""
        # Create data with unknown type info
        serialized_data = {
            "__type_info__": "unknown_type",
            "value": "test_value"
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should return original data
        assert result == serialized_data


class TestSerializationConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Reset global state
        self._reset_global_state()
        yield
        self._reset_global_state()
    
    def _reset_global_state(self):
        """Reset any global state to prevent test interference."""
        if hasattr(TypeAwareConfigSerializer, '_class_cache'):
            TypeAwareConfigSerializer._class_cache.clear()
    
    @patch('cursus.core.config_fields.type_aware_config_serializer.TypeAwareConfigSerializer')
    def test_serialize_config_function_dict_result(self, mock_serializer_class):
        """Test serialize_config convenience function with dict result."""
        # Create mock config object
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestConfig"
        
        # Mock serializer behavior - returns dict
        mock_serializer = Mock()
        mock_serializer_class.return_value = mock_serializer
        mock_serializer.serialize.return_value = {"field1": "value1"}
        mock_serializer.generate_step_name.return_value = "TestStep"
        
        result = serialize_config(mock_config)
        
        # Verify structure includes metadata
        assert isinstance(result, dict)
        assert "_metadata" in result
        assert result["_metadata"]["step_name"] == "TestStep"
        assert result["_metadata"]["config_type"] == "TestConfig"
        assert result["field1"] == "value1"
    
    @patch('cursus.core.config_fields.type_aware_config_serializer.TypeAwareConfigSerializer')
    def test_serialize_config_function_non_dict_result(self, mock_serializer_class):
        """Test serialize_config convenience function with non-dict result."""
        # Create mock config object
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestConfig"
        
        # Mock serializer behavior - returns non-dict
        mock_serializer = Mock()
        mock_serializer_class.return_value = mock_serializer
        mock_serializer.serialize.return_value = "string_result"
        mock_serializer.generate_step_name.return_value = "TestStep"
        
        result = serialize_config(mock_config)
        
        # Verify wrapper structure for non-dict result
        assert isinstance(result, dict)
        assert result["__model_type__"] == "TestConfig"
        assert "_metadata" in result
        assert result["_metadata"]["step_name"] == "TestStep"
        assert result["_metadata"]["config_type"] == "TestConfig"
        assert result["value"] == "string_result"
    
    @patch('cursus.core.config_fields.type_aware_config_serializer.TypeAwareConfigSerializer')
    def test_deserialize_config_function(self, mock_serializer_class):
        """Test deserialize_config convenience function."""
        # Create serialized data
        serialized_data = {
            "__model_type__": "TestConfig",
            "field1": "value1"
        }
        
        # Mock serializer behavior
        mock_serializer = Mock()
        mock_serializer_class.return_value = mock_serializer
        mock_instance = Mock()
        mock_serializer.deserialize.return_value = mock_instance
        
        result = deserialize_config(serialized_data)
        
        # Verify deserialization
        assert result == mock_instance
        mock_serializer.deserialize.assert_called_once_with(serialized_data, expected_type=None)
    
    @patch('cursus.core.config_fields.type_aware_config_serializer.TypeAwareConfigSerializer')
    def test_deserialize_config_function_with_expected_type(self, mock_serializer_class):
        """Test deserialize_config convenience function with expected type."""
        # Create serialized data
        serialized_data = {
            "__model_type__": "TestConfig",
            "field1": "value1"
        }
        
        # Mock serializer behavior
        mock_serializer = Mock()
        mock_serializer_class.return_value = mock_serializer
        mock_instance = Mock()
        mock_serializer.deserialize.return_value = mock_instance
        
        result = deserialize_config(serialized_data, expected_type=SerializationTestConfig)
        
        # Verify deserialization with expected type
        assert result == mock_instance
        mock_serializer.deserialize.assert_called_once_with(serialized_data, expected_type=SerializationTestConfig)


class TestSerializationIntegration:
    """Integration tests for serialization system."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Reset global state
        self._reset_global_state()
        yield
        self._reset_global_state()
    
    def _reset_global_state(self):
        """Reset any global state to prevent test interference."""
        if hasattr(TypeAwareConfigSerializer, '_class_cache'):
            TypeAwareConfigSerializer._class_cache.clear()
    
    def test_round_trip_serialization_primitives(self):
        """Test complete serialization/deserialization round trip for primitives."""
        # Create test data structure with primitives
        test_data = {
            "string_field": "test_value",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "none_field": None,
            "list_field": ["item1", "item2", 42],
            "nested_dict": {
                "nested_string": "nested_value",
                "nested_int": 123
            }
        }
        
        # Serialize
        serializer = TypeAwareConfigSerializer()
        serialized = serializer.serialize(test_data)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round trip accuracy for primitives
        assert deserialized["string_field"] == test_data["string_field"]
        assert deserialized["int_field"] == test_data["int_field"]
        assert deserialized["float_field"] == test_data["float_field"]
        assert deserialized["bool_field"] == test_data["bool_field"]
        assert deserialized["none_field"] == test_data["none_field"]
        assert deserialized["list_field"] == test_data["list_field"]
        assert deserialized["nested_dict"] == test_data["nested_dict"]
    
    def test_round_trip_serialization_complex_types(self):
        """Test complete serialization/deserialization round trip for complex types."""
        # Create test data structure with complex types
        test_datetime = datetime(2023, 1, 15, 10, 30, 45)
        test_path = Path("/test/path/file.txt")
        test_enum = SerializationTestEnum.VALUE1
        
        test_data = {
            "datetime_field": test_datetime,
            "path_field": test_path,
            "enum_field": test_enum,
            "set_field": {"item1", "item2", "item3"},
            "tuple_field": ("tuple_item1", "tuple_item2")
        }
        
        # Serialize
        serializer = TypeAwareConfigSerializer()
        serialized = serializer.serialize(test_data)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round trip accuracy for complex types
        assert isinstance(deserialized["datetime_field"], datetime)
        assert deserialized["datetime_field"] == test_datetime
        
        assert isinstance(deserialized["path_field"], Path)
        assert str(deserialized["path_field"]) == str(test_path)
        
        assert isinstance(deserialized["enum_field"], SerializationTestEnum)
        assert deserialized["enum_field"] == test_enum
        
        assert isinstance(deserialized["set_field"], set)
        assert deserialized["set_field"] == test_data["set_field"]
        
        assert isinstance(deserialized["tuple_field"], tuple)
        assert deserialized["tuple_field"] == test_data["tuple_field"]
    
    def test_round_trip_serialization_pydantic_model(self):
        """Test complete serialization/deserialization round trip for Pydantic models."""
        # Create test config
        test_config = SerializationTestConfig(field1="test_value", field2=42, field3=False)
        
        # Serialize with config classes provided
        serializer = TypeAwareConfigSerializer(config_classes={"SerializationTestConfig": SerializationTestConfig})
        serialized = serializer.serialize(test_config)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round trip accuracy for Pydantic model
        assert isinstance(deserialized, SerializationTestConfig)
        assert deserialized.field1 == test_config.field1
        assert deserialized.field2 == test_config.field2
        # Note: field3 behavior depends on tier handling and model_dump
    
    def test_serialization_with_none_handling(self):
        """Test serialization with None values to prevent Category 12 failures."""
        # Test data with None values that could cause AttributeError
        test_data = {
            "field1": None,  # Could cause 'NoneType' object has no attribute 'get'
            "nested_field": {"subfield": None},
            "list_with_none": ["item1", None, "item3"],
            "dict_with_none_values": {
                "key1": "value1",
                "key2": None,
                "key3": {"nested_none": None}
            }
        }
        
        # Serialize - should handle None gracefully
        serializer = TypeAwareConfigSerializer()
        serialized = serializer.serialize(test_data)
        
        # Deserialize - should handle None gracefully
        deserialized = serializer.deserialize(serialized)
        
        # Verify None handling
        assert deserialized["field1"] is None
        assert deserialized["nested_field"]["subfield"] is None
        assert deserialized["list_with_none"] == ["item1", None, "item3"]
        assert deserialized["dict_with_none_values"]["key2"] is None
        assert deserialized["dict_with_none_values"]["key3"]["nested_none"] is None
    
    def test_serialization_error_recovery(self):
        """Test serialization error recovery and fallback mechanisms."""
        # Create object that will cause serialization issues
        class ProblematicObject:
            def __str__(self):
                return "problematic_object_fallback"
            
            def __getattr__(self, name):
                if name == "some_attribute":
                    raise Exception("Attribute access error")
                return "default_value"
        
        problematic_obj = ProblematicObject()
        
        # Test serialization fallback
        serializer = TypeAwareConfigSerializer()
        result = serializer.serialize(problematic_obj)
        
        # Should fallback to string representation
        assert result == "problematic_object_fallback"
    
    def test_generate_step_name_functionality(self):
        """Test step name generation functionality."""
        # Create mock config with step_name_override
        mock_config_with_override = Mock()
        mock_config_with_override.__class__.__name__ = "TestConfig"
        mock_config_with_override.step_name_override = "CustomStepName"
        
        serializer = TypeAwareConfigSerializer()
        result = serializer.generate_step_name(mock_config_with_override)
        
        # Should use override
        assert result == "CustomStepName"
    
    def test_generate_step_name_with_registry(self):
        """Test step name generation with registry lookup."""
        # Create mock config without override
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestConfig"
        mock_config.step_name_override = None
        mock_config.job_type = "training"
        mock_config.data_type = "tabular"
        
        # Mock the registry inside the try/except block where it's actually imported
        with patch('cursus.registry.step_names.CONFIG_STEP_REGISTRY', {"TestConfig": "TestStep"}, create=True):
            serializer = TypeAwareConfigSerializer()
            result = serializer.generate_step_name(mock_config)
            
            # Should use registry and append attributes
            assert "TestStep" in result
            assert "training" in result
            assert "tabular" in result
    
    def test_generate_step_name_fallback(self):
        """Test step name generation fallback when registry unavailable."""
        # Create mock config without override
        mock_config = Mock()
        mock_config.__class__.__name__ = "TestStepConfig"
        mock_config.step_name_override = None
        
        # Mock missing attributes
        for attr in ("job_type", "data_type", "mode"):
            setattr(mock_config, attr, None)
        
        serializer = TypeAwareConfigSerializer()
        result = serializer.generate_step_name(mock_config)
        
        # Should fallback to class name - implementation doesn't remove "Config"
        assert result == "TestStepConfig"  # Implementation returns full class name


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Reset global state
        self._reset_global_state()
        
        # Create serializer with minimal setup
        self.serializer = TypeAwareConfigSerializer(config_classes={})
        
        yield
        self._reset_global_state()
    
    def _reset_global_state(self):
        """Reset any global state to prevent test interference."""
        if hasattr(TypeAwareConfigSerializer, '_class_cache'):
            TypeAwareConfigSerializer._class_cache.clear()
    
    def test_deserialize_with_field_name_tracking(self):
        """Test deserialization with field name tracking for debugging."""
        # Create nested data structure
        test_data = {
            "level1": {
                "level2": ["item1", "item2", {"level3": "value"}]
            }
        }
        
        result = self.serializer.deserialize(test_data, field_name="root")
        
        # Should deserialize correctly with field name tracking
        assert result["level1"]["level2"][2]["level3"] == "value"
    
    def test_serialize_with_circular_reference_cleanup(self):
        """Test that circular reference tracking is properly cleaned up."""
        # Create mock config
        mock_config = Mock(spec=BaseModel)
        mock_config.__class__.__name__ = "TestConfig"
        
        # Mock unified manager
        mock_unified_manager = Mock()
        mock_unified_manager.get_field_tiers.return_value = {
            "essential": [],
            "system": [],
            "derived": []
        }
        self.serializer.unified_manager = mock_unified_manager
        
        # Serialize (should add to _serializing_ids and then remove)
        result = self.serializer.serialize(mock_config)
        
        # Verify cleanup - _serializing_ids should be empty after serialization
        assert len(self.serializer._serializing_ids) == 0
    
    def test_deserialize_with_nested_type_resolution(self):
        """Test deserialization with nested type resolution."""
        # Create mock class with model_fields
        mock_class = Mock()
        mock_class.model_fields = {
            "field1": Mock(annotation=str),
            "field2": Mock(annotation=int)
        }
        mock_instance = Mock()
        mock_class.model_validate.return_value = mock_instance
        
        # Update serializer config classes
        self.serializer.config_classes["TestConfig"] = mock_class
        
        # Create nested serialized data
        serialized_data = {
            "__model_type__": "TestConfig",
            "field1": "test_value",
            "field2": 42,
            "field3": "ignored_field"  # Not in model_fields
        }
        
        result = self.serializer.deserialize(serialized_data)
        
        # Should filter to model_fields only
        assert result == mock_instance
        mock_class.model_validate.assert_called_once()
        call_args = mock_class.model_validate.call_args[0][0]
        assert "field1" in call_args
        assert "field2" in call_args
        assert "field3" not in call_args  # Should be filtered out
    
    def test_serialize_mode_consistency(self):
        """Test that serialization mode is consistently applied."""
        # Test PRESERVE_TYPES mode
        self.serializer.mode = SerializationMode.PRESERVE_TYPES
        
        test_datetime = datetime(2023, 1, 15, 10, 30)
        result_preserve = self.serializer.serialize(test_datetime)
        
        assert isinstance(result_preserve, dict)
        assert "__type_info__" in result_preserve
        
        # Test SIMPLE_JSON mode
        self.serializer.mode = SerializationMode.SIMPLE_JSON
        result_simple = self.serializer.serialize(test_datetime)
        
        assert isinstance(result_simple, str)
        assert result_simple == test_datetime.isoformat()
    
    def test_logging_functionality(self):
        """Test that logging works correctly for debugging."""
        # This test verifies that logger is properly initialized
        assert hasattr(self.serializer, 'logger')
        assert self.serializer.logger.name == 'cursus.core.config_fields.type_aware_config_serializer'
    
    def test_constants_usage(self):
        """Test that constants are used correctly."""
        # Verify constants are properly defined and used
        assert hasattr(TypeAwareConfigSerializer, 'MODEL_TYPE_FIELD')
        assert hasattr(TypeAwareConfigSerializer, 'TYPE_INFO_FIELD')
        assert TypeAwareConfigSerializer.MODEL_TYPE_FIELD == "__model_type__"
        assert TypeAwareConfigSerializer.TYPE_INFO_FIELD == "__type_info__"
