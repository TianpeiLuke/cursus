"""Pytest tests for the registry exceptions module."""

import pytest
from cursus.registry.exceptions import RegistryError


class TestRegistryExceptions:
    """Test case for registry exception classes."""

    def test_registry_error_basic(self):
        """Test basic RegistryError functionality."""
        message = "Test error message"
        error = RegistryError(message)
        
        assert str(error) == message
        assert error.unresolvable_types == []
        assert error.available_builders == []

    def test_registry_error_with_unresolvable_types(self):
        """Test RegistryError with unresolvable types."""
        message = "Cannot resolve step types"
        unresolvable_types = ["UnknownStep1", "UnknownStep2"]
        
        error = RegistryError(message, unresolvable_types=unresolvable_types)
        
        assert error.unresolvable_types == unresolvable_types
        assert "UnknownStep1" in str(error)
        assert "UnknownStep2" in str(error)
        assert "Unresolvable step types" in str(error)

    def test_registry_error_with_available_builders(self):
        """Test RegistryError with available builders."""
        message = "Builder not found"
        available_builders = ["Builder1", "Builder2", "Builder3"]
        
        error = RegistryError(message, available_builders=available_builders)
        
        assert error.available_builders == available_builders
        assert "Builder1" in str(error)
        assert "Builder2" in str(error)
        assert "Builder3" in str(error)
        assert "Available builders" in str(error)

    def test_registry_error_with_both_parameters(self):
        """Test RegistryError with both unresolvable types and available builders."""
        message = "Complex error"
        unresolvable_types = ["BadStep"]
        available_builders = ["GoodBuilder1", "GoodBuilder2"]
        
        error = RegistryError(
            message, 
            unresolvable_types=unresolvable_types,
            available_builders=available_builders
        )
        
        error_str = str(error)
        assert message in error_str
        assert "BadStep" in error_str
        assert "GoodBuilder1" in error_str
        assert "GoodBuilder2" in error_str
        assert "Unresolvable step types" in error_str
        assert "Available builders" in error_str

    def test_registry_error_inheritance(self):
        """Test that RegistryError properly inherits from Exception."""
        error = RegistryError("Test message")
        
        assert isinstance(error, Exception)
        assert issubclass(RegistryError, Exception)

    def test_registry_error_empty_lists(self):
        """Test RegistryError with empty lists."""
        message = "Test with empty lists"
        error = RegistryError(
            message,
            unresolvable_types=[],
            available_builders=[]
        )
        
        # Should only show the main message, not the empty list sections
        error_str = str(error)
        assert error_str == message
        assert "Unresolvable step types" not in error_str
        assert "Available builders" not in error_str
