"""
Field Requirement Extraction Utilities

This module provides utilities for extracting field requirements from Pydantic configuration classes.
The extracted requirements are returned as simple dictionaries that are easy to print and use
in interactive configuration workflows.

Key Functions:
- extract_field_requirements: Extract field requirements from Pydantic class
- extract_non_inherited_fields: Extract only non-inherited fields from derived class
- print_field_requirements: Print field requirements in user-friendly format
- get_field_type_string: Convert field type annotation to readable string
"""

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel
import inspect


def extract_field_requirements(config_class: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Extract field requirements directly from Pydantic V2 class definition.
    
    Args:
        config_class: Pydantic V2 model class to extract fields from
        
    Returns:
        List of field requirement dictionaries with format:
        {
            'name': str,           # Field name
            'type': str,           # Field type as string
            'description': str,    # Field description from Pydantic Field()
            'required': bool,      # True for required fields, False for optional
            'default': Any         # Default value (only for optional fields)
        }
    """
    requirements = []
    
    # Pydantic V2+ compatible field access
    try:
        # Try to get model fields - compatible with V2 and future versions
        fields = getattr(config_class, 'model_fields', None)
        if fields is not None:
            for field_name, field_info in fields.items():
                # Skip private fields
                if field_name.startswith('_'):
                    continue
                    
                # Pydantic V2+ field info structure
                is_required = field_info.is_required() if hasattr(field_info, 'is_required') else True
                
                # Get default value - handle different default types
                default_value = None
                if not is_required:
                    if hasattr(field_info, 'default') and field_info.default is not None:
                        default_value = field_info.default
                    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        try:
                            default_value = field_info.default_factory()
                        except Exception:
                            factory_name = getattr(field_info.default_factory, '__name__', 'unknown')
                            default_value = f"<factory: {factory_name}>"
                
                # Get description from field info
                description = f"Configuration for {field_name}"
                if hasattr(field_info, 'description') and field_info.description:
                    description = field_info.description
                elif hasattr(field_info, 'json_schema_extra') and isinstance(field_info.json_schema_extra, dict):
                    description = field_info.json_schema_extra.get('description', description)
                
                # Get field annotation
                annotation = getattr(field_info, 'annotation', None)
                
                requirements.append({
                    'name': field_name,
                    'type': get_field_type_string(annotation),
                    'description': description,
                    'required': is_required,
                    'default': default_value
                })
        else:
            # Fallback for non-Pydantic classes or future compatibility
            raise AttributeError("model_fields not available")
    except (AttributeError, TypeError):
        # Enhanced fallback for future compatibility
        # Fallback: try to inspect the class directly for non-Pydantic classes
        try:
            signature = inspect.signature(config_class.__init__)
            for param_name, param in signature.parameters.items():
                if param_name in ('self', 'args', 'kwargs'):
                    continue
                    
                requirements.append({
                    'name': param_name,
                    'type': get_field_type_string(param.annotation),
                    'description': f"Configuration for {param_name}",
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None
                })
        except Exception:
            # If all else fails, return empty list
            pass
    
    return requirements


def extract_non_inherited_fields(derived_class: Type[BaseModel], base_class: Type[BaseModel]) -> List[Dict[str, Any]]:
    """
    Extract fields from derived class that are not inherited from base class.
    
    Args:
        derived_class: The derived Pydantic V2 model class
        base_class: The base Pydantic V2 model class to exclude fields from
        
    Returns:
        List of field requirement dictionaries for non-inherited fields only
    """
    # Get base class field names to exclude (Pydantic V2+ compatible)
    base_fields = set()
    try:
        model_fields = getattr(base_class, 'model_fields', None)
        if model_fields is not None:
            base_fields = set(model_fields.keys())
    except (AttributeError, TypeError):
        # Future compatibility fallback
        pass
    
    # Extract all fields from derived class
    all_requirements = extract_field_requirements(derived_class)
    
    # Filter out inherited base fields
    non_inherited_requirements = []
    for req in all_requirements:
        if req['name'] not in base_fields:
            non_inherited_requirements.append(req)
    
    return non_inherited_requirements


def print_field_requirements(requirements: List[Dict[str, Any]]) -> None:
    """
    Print field requirements in user-friendly format.
    
    Args:
        requirements: List of field requirement dictionaries
    """
    if not requirements:
        print("No field requirements found.")
        return
    
    print("Field Requirements:")
    print("-" * 50)
    
    for req in requirements:
        marker = "*" if req['required'] else " "
        default_info = f" (default: {req.get('default')})" if not req['required'] and 'default' in req else ""
        print(f"{marker} {req['name']} ({req['type']}){default_info}")
        print(f"    {req['description']}")
        print()


def get_field_type_string(annotation: Any) -> str:
    """
    Convert field type annotation to readable string.
    
    Args:
        annotation: Type annotation from Pydantic field
        
    Returns:
        Human-readable string representation of the type
    """
    if annotation is None:
        return "Any"
    
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    
    # Handle typing module types
    type_str = str(annotation)
    
    # Clean up common typing patterns
    type_str = type_str.replace('typing.', '')
    type_str = type_str.replace('<class \'', '').replace('\'>', '')
    
    # Handle Union types (Optional is Union[T, None])
    if 'Union[' in type_str:
        # Simplify Optional[T] to T (optional)
        if type_str.endswith(', NoneType]') or type_str.endswith(', None]'):
            type_str = type_str.replace('Union[', '').replace(', NoneType]', '').replace(', None]', '')
            type_str += ' (optional)'
    
    # Handle List, Dict, etc.
    type_str = type_str.replace('List[', 'list[').replace('Dict[', 'dict[')
    
    return type_str


def categorize_field_requirements(requirements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize field requirements into required and optional groups.
    
    Args:
        requirements: List of field requirement dictionaries
        
    Returns:
        Dictionary with 'required' and 'optional' keys containing respective field lists
    """
    categorized = {
        'required': [],
        'optional': []
    }
    
    for req in requirements:
        if req['required']:
            categorized['required'].append(req)
        else:
            categorized['optional'].append(req)
    
    return categorized


def validate_field_value(field_req: Dict[str, Any], value: Any) -> bool:
    """
    Basic validation of field value against field requirement.
    
    Args:
        field_req: Field requirement dictionary
        value: Value to validate
        
    Returns:
        True if value is valid for the field, False otherwise
    """
    # Check if required field has a value
    if field_req['required'] and (value is None or (isinstance(value, str) and not value.strip())):
        return False
    
    # Basic type checking (simplified)
    field_type = field_req['type'].lower()
    
    if 'str' in field_type and value is not None and not isinstance(value, str):
        return False
    elif 'int' in field_type and value is not None and not isinstance(value, int):
        return False
    elif 'float' in field_type and value is not None and not isinstance(value, (int, float)):
        return False
    elif 'bool' in field_type and value is not None and not isinstance(value, bool):
        return False
    
    return True
