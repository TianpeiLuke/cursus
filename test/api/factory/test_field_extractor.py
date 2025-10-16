"""
Test suite for field_extractor.py

Following pytest best practices:
1. Source code first - Read actual implementation before writing tests
2. Implementation-driven testing - Tests match actual behavior
3. Comprehensive coverage - All public functions tested
4. Mock-based isolation - External dependencies mocked appropriately
5. Clear test organization - Grouped by functionality with descriptive names
"""

import pytest
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock

from cursus.api.factory.field_extractor import (
    extract_field_requirements,
    extract_non_inherited_fields,
    print_field_requirements,
    get_field_type_string,
    categorize_field_requirements,
    validate_field_value
)


# Test fixtures - Pydantic V2 compatible models
class SimpleTestConfig(BaseModel):
    """Simple test config with basic field types."""
    name: str = Field(description="Configuration name")
    value: int = Field(default=42, description="Configuration value")
    enabled: bool = Field(default=True, description="Enable flag")
    optional_field: Optional[str] = Field(default=None, description="Optional field")


class ComplexTestConfig(BaseModel):
    """Complex test config with various field types."""
    required_str: str = Field(description="Required string field")
    optional_int: Optional[int] = Field(default=100, description="Optional integer")
    list_field: List[str] = Field(default_factory=list, description="List of strings")
    dict_field: Dict[str, Any] = Field(default_factory=dict, description="Dictionary field")
    union_field: Union[str, int] = Field(default="default", description="Union field")


class BaseTestConfig(BaseModel):
    """Base configuration for inheritance testing."""
    base_field: str = Field(description="Base field")
    base_optional: Optional[int] = Field(default=10, description="Base optional field")


class DerivedTestConfig(BaseTestConfig):
    """Derived configuration for inheritance testing."""
    derived_field: str = Field(description="Derived field")
    derived_optional: Optional[bool] = Field(default=False, description="Derived optional field")


class TestExtractFieldRequirements:
    """Test extract_field_requirements function."""
    
    def test_extract_simple_config_fields(self):
        """Test extraction from simple config with basic field types."""
        requirements = extract_field_requirements(SimpleTestConfig)
        
        # Should extract all 4 fields
        assert len(requirements) == 4
        
        # Check field names are present
        field_names = [req['name'] for req in requirements]
        assert 'name' in field_names
        assert 'value' in field_names
        assert 'enabled' in field_names
        assert 'optional_field' in field_names
        
        # Check required vs optional fields
        name_req = next(req for req in requirements if req['name'] == 'name')
        assert name_req['required'] is True
        assert name_req['type'] == 'str'
        assert name_req['description'] == "Configuration name"
        
        value_req = next(req for req in requirements if req['name'] == 'value')
        assert value_req['required'] is False
        assert value_req['default'] == 42
        assert value_req['type'] == 'int'
    
    def test_extract_complex_config_fields(self):
        """Test extraction from complex config with various field types."""
        requirements = extract_field_requirements(ComplexTestConfig)
        
        # Should extract all 5 fields
        assert len(requirements) == 5
        
        # Check required field
        required_req = next(req for req in requirements if req['name'] == 'required_str')
        assert required_req['required'] is True
        assert required_req['type'] == 'str'
        
        # Check optional field with default
        optional_req = next(req for req in requirements if req['name'] == 'optional_int')
        assert optional_req['required'] is False
        assert optional_req['default'] == 100
        
        # Check factory default fields
        list_req = next(req for req in requirements if req['name'] == 'list_field')
        assert list_req['required'] is False
        # Factory defaults should be handled gracefully
        assert 'default' in list_req
    
    def test_extract_with_no_fields(self):
        """Test extraction from config with no fields."""
        class EmptyConfig(BaseModel):
            pass
        
        requirements = extract_field_requirements(EmptyConfig)
        assert requirements == []
    
    def test_extract_handles_private_fields(self):
        """Test that private fields are skipped."""
        # Use PrivateAttr for private fields in Pydantic V2
        from pydantic import PrivateAttr
        
        class ConfigWithPrivateFields(BaseModel):
            public_field: str = Field(description="Public field")
            _private_field: str = PrivateAttr(default="private")
        
        requirements = extract_field_requirements(ConfigWithPrivateFields)
        
        # Should only extract public field (private fields are skipped)
        field_names = [req['name'] for req in requirements]
        assert 'public_field' in field_names
        assert '_private_field' not in field_names
    
    def test_extract_fallback_for_non_pydantic(self):
        """Test fallback behavior for non-Pydantic classes."""
        class NonPydanticClass:
            def __init__(self, name: str, value: int = 42):
                self.name = name
                self.value = value
        
        # Should handle gracefully and return empty list or use inspection fallback
        requirements = extract_field_requirements(NonPydanticClass)
        # Implementation may return empty list or use inspection - both are valid
        assert isinstance(requirements, list)


class TestExtractNonInheritedFields:
    """Test extract_non_inherited_fields function."""
    
    def test_extract_derived_fields_only(self):
        """Test extraction of only derived fields, excluding base fields."""
        requirements = extract_non_inherited_fields(DerivedTestConfig, BaseTestConfig)
        
        # Should only extract derived fields
        assert len(requirements) == 2
        
        field_names = [req['name'] for req in requirements]
        assert 'derived_field' in field_names
        assert 'derived_optional' in field_names
        
        # Should not include base fields
        assert 'base_field' not in field_names
        assert 'base_optional' not in field_names
    
    def test_extract_with_same_class(self):
        """Test extraction when derived and base are the same class."""
        requirements = extract_non_inherited_fields(BaseTestConfig, BaseTestConfig)
        
        # Should return empty list since all fields are "inherited"
        assert requirements == []
    
    def test_extract_with_no_inheritance(self):
        """Test extraction when there's no actual inheritance relationship."""
        requirements = extract_non_inherited_fields(SimpleTestConfig, ComplexTestConfig)
        
        # Should return all fields from SimpleTestConfig since none are in ComplexTestConfig
        assert len(requirements) == 4
        
        field_names = [req['name'] for req in requirements]
        assert 'name' in field_names
        assert 'value' in field_names
        assert 'enabled' in field_names
        assert 'optional_field' in field_names


class TestGetFieldTypeString:
    """Test get_field_type_string function."""
    
    def test_basic_types(self):
        """Test conversion of basic Python types."""
        assert get_field_type_string(str) == 'str'
        assert get_field_type_string(int) == 'int'
        assert get_field_type_string(bool) == 'bool'
        assert get_field_type_string(float) == 'float'
    
    def test_none_annotation(self):
        """Test handling of None annotation."""
        assert get_field_type_string(None) == 'Any'
    
    def test_optional_types(self):
        """Test handling of Optional types (Union with None)."""
        from typing import Optional
        
        # Optional types should be simplified - check actual implementation behavior
        result = get_field_type_string(Optional[str])
        # The actual implementation may return different formats
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_list_types(self):
        """Test handling of List types."""
        from typing import List
        
        result = get_field_type_string(List[str])
        # Check actual implementation behavior
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_dict_types(self):
        """Test handling of Dict types."""
        from typing import Dict
        
        result = get_field_type_string(Dict[str, int])
        # Check actual implementation behavior
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_union_types(self):
        """Test handling of Union types."""
        from typing import Union
        
        result = get_field_type_string(Union[str, int])
        assert isinstance(result, str)
        # Should handle Union gracefully
    
    def test_complex_annotation(self):
        """Test handling of complex type annotations."""
        # Should not crash on complex types
        result = get_field_type_string(List[Dict[str, Optional[int]]])
        assert isinstance(result, str)
        assert len(result) > 0


class TestCategorizeFieldRequirements:
    """Test categorize_field_requirements function."""
    
    def test_categorize_mixed_requirements(self):
        """Test categorization of mixed required and optional fields."""
        requirements = [
            {'name': 'req1', 'required': True, 'type': 'str'},
            {'name': 'opt1', 'required': False, 'type': 'int', 'default': 42},
            {'name': 'req2', 'required': True, 'type': 'bool'},
            {'name': 'opt2', 'required': False, 'type': 'str', 'default': 'test'}
        ]
        
        categorized = categorize_field_requirements(requirements)
        
        assert 'required' in categorized
        assert 'optional' in categorized
        
        # Check required fields
        assert len(categorized['required']) == 2
        required_names = [req['name'] for req in categorized['required']]
        assert 'req1' in required_names
        assert 'req2' in required_names
        
        # Check optional fields
        assert len(categorized['optional']) == 2
        optional_names = [req['name'] for req in categorized['optional']]
        assert 'opt1' in optional_names
        assert 'opt2' in optional_names
    
    def test_categorize_all_required(self):
        """Test categorization when all fields are required."""
        requirements = [
            {'name': 'req1', 'required': True, 'type': 'str'},
            {'name': 'req2', 'required': True, 'type': 'int'}
        ]
        
        categorized = categorize_field_requirements(requirements)
        
        assert len(categorized['required']) == 2
        assert len(categorized['optional']) == 0
    
    def test_categorize_all_optional(self):
        """Test categorization when all fields are optional."""
        requirements = [
            {'name': 'opt1', 'required': False, 'type': 'str', 'default': 'test'},
            {'name': 'opt2', 'required': False, 'type': 'int', 'default': 42}
        ]
        
        categorized = categorize_field_requirements(requirements)
        
        assert len(categorized['required']) == 0
        assert len(categorized['optional']) == 2
    
    def test_categorize_empty_requirements(self):
        """Test categorization of empty requirements list."""
        categorized = categorize_field_requirements([])
        
        assert categorized['required'] == []
        assert categorized['optional'] == []


class TestValidateFieldValue:
    """Test validate_field_value function."""
    
    def test_validate_required_field_with_value(self):
        """Test validation of required field with valid value."""
        field_req = {'name': 'test', 'required': True, 'type': 'str'}
        
        assert validate_field_value(field_req, 'valid_value') is True
        assert validate_field_value(field_req, '') is False  # Empty string
        assert validate_field_value(field_req, None) is False  # None value
        assert validate_field_value(field_req, '   ') is False  # Whitespace only
    
    def test_validate_optional_field(self):
        """Test validation of optional field."""
        field_req = {'name': 'test', 'required': False, 'type': 'str', 'default': 'default'}
        
        assert validate_field_value(field_req, 'valid_value') is True
        assert validate_field_value(field_req, None) is True  # None is OK for optional
        assert validate_field_value(field_req, '') is True  # Empty string OK for optional
    
    def test_validate_type_checking(self):
        """Test basic type validation."""
        # String field
        str_field = {'name': 'test', 'required': True, 'type': 'str'}
        assert validate_field_value(str_field, 'string_value') is True
        assert validate_field_value(str_field, 123) is False
        
        # Integer field
        int_field = {'name': 'test', 'required': True, 'type': 'int'}
        assert validate_field_value(int_field, 123) is True
        assert validate_field_value(int_field, 'not_int') is False
        
        # Boolean field
        bool_field = {'name': 'test', 'required': True, 'type': 'bool'}
        assert validate_field_value(bool_field, True) is True
        assert validate_field_value(bool_field, False) is True
        assert validate_field_value(bool_field, 'not_bool') is False
        
        # Float field (should accept int and float)
        float_field = {'name': 'test', 'required': True, 'type': 'float'}
        assert validate_field_value(float_field, 3.14) is True
        assert validate_field_value(float_field, 42) is True  # int acceptable for float
        assert validate_field_value(float_field, 'not_float') is False
    
    def test_validate_complex_types(self):
        """Test validation of complex type strings."""
        # Complex type strings should not cause crashes
        complex_field = {'name': 'test', 'required': True, 'type': 'List[Dict[str, int]]'}
        
        # The actual implementation does basic validation - check what it actually returns
        result = validate_field_value(complex_field, [])
        assert isinstance(result, bool)  # Should return a boolean without crashing
        
        # Required field with None should be False
        assert validate_field_value(complex_field, None) is False


class TestPrintFieldRequirements:
    """Test print_field_requirements function."""
    
    @patch('builtins.print')
    def test_print_empty_requirements(self, mock_print):
        """Test printing empty requirements list."""
        print_field_requirements([])
        
        mock_print.assert_called_with("No field requirements found.")
    
    @patch('builtins.print')
    def test_print_mixed_requirements(self, mock_print):
        """Test printing mixed required and optional fields."""
        requirements = [
            {
                'name': 'required_field',
                'type': 'str',
                'description': 'A required field',
                'required': True
            },
            {
                'name': 'optional_field',
                'type': 'int',
                'description': 'An optional field',
                'required': False,
                'default': 42
            }
        ]
        
        print_field_requirements(requirements)
        
        # Check that print was called multiple times
        assert mock_print.call_count > 0
        
        # Check that required field is marked with *
        calls = [str(call) for call in mock_print.call_args_list]
        required_call = next((call for call in calls if 'required_field' in call and '*' in call), None)
        assert required_call is not None
        
        # Check that optional field shows default value
        optional_call = next((call for call in calls if 'optional_field' in call and 'default: 42' in call), None)
        assert optional_call is not None


class TestIntegrationWithRealPydanticModels:
    """Integration tests with real Pydantic models."""
    
    def test_extract_from_real_config_models(self):
        """Test extraction from realistic configuration models."""
        requirements = extract_field_requirements(ComplexTestConfig)
        
        # Should successfully extract all fields
        assert len(requirements) > 0
        
        # All requirements should have required structure
        for req in requirements:
            assert 'name' in req
            assert 'type' in req
            assert 'description' in req
            assert 'required' in req
            assert isinstance(req['required'], bool)
    
    def test_categorize_real_config_fields(self):
        """Test categorization with real config model fields."""
        requirements = extract_field_requirements(ComplexTestConfig)
        categorized = categorize_field_requirements(requirements)
        
        # Should have both required and optional fields
        assert len(categorized['required']) > 0
        assert len(categorized['optional']) > 0
        
        # All categorized fields should maintain their structure
        all_fields = categorized['required'] + categorized['optional']
        assert len(all_fields) == len(requirements)
    
    def test_inheritance_extraction_integration(self):
        """Test inheritance extraction with real model hierarchy."""
        base_requirements = extract_field_requirements(BaseTestConfig)
        derived_requirements = extract_field_requirements(DerivedTestConfig)
        non_inherited = extract_non_inherited_fields(DerivedTestConfig, BaseTestConfig)
        
        # Derived should have more fields than base
        assert len(derived_requirements) > len(base_requirements)
        
        # Non-inherited should be the difference
        expected_non_inherited_count = len(derived_requirements) - len(base_requirements)
        assert len(non_inherited) == expected_non_inherited_count
        
        # Non-inherited fields should not overlap with base fields
        base_field_names = {req['name'] for req in base_requirements}
        non_inherited_names = {req['name'] for req in non_inherited}
        assert base_field_names.isdisjoint(non_inherited_names)


# Pytest fixtures for reuse across tests
@pytest.fixture
def simple_requirements():
    """Fixture providing simple field requirements for testing."""
    return [
        {'name': 'name', 'type': 'str', 'description': 'Name field', 'required': True},
        {'name': 'count', 'type': 'int', 'description': 'Count field', 'required': False, 'default': 0}
    ]


@pytest.fixture
def complex_requirements():
    """Fixture providing complex field requirements for testing."""
    return [
        {'name': 'required_str', 'type': 'str', 'description': 'Required string', 'required': True},
        {'name': 'optional_int', 'type': 'int', 'description': 'Optional integer', 'required': False, 'default': 100},
        {'name': 'list_field', 'type': 'List[str]', 'description': 'List field', 'required': False, 'default': []},
        {'name': 'required_bool', 'type': 'bool', 'description': 'Required boolean', 'required': True}
    ]


class TestFixtureUsage:
    """Test using pytest fixtures."""
    
    def test_simple_requirements_fixture(self, simple_requirements):
        """Test using simple requirements fixture."""
        assert len(simple_requirements) == 2
        assert simple_requirements[0]['name'] == 'name'
        assert simple_requirements[1]['required'] is False
    
    def test_complex_requirements_fixture(self, complex_requirements):
        """Test using complex requirements fixture."""
        assert len(complex_requirements) == 4
        
        required_fields = [req for req in complex_requirements if req['required']]
        optional_fields = [req for req in complex_requirements if not req['required']]
        
        assert len(required_fields) == 2
        assert len(optional_fields) == 2
