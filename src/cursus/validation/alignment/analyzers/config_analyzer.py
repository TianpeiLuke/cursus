"""
Configuration Analysis Engine

Analyzes configuration classes to extract field information, types, defaults, and requirements.
Handles both Pydantic v1 and v2 configurations with robust field detection.
"""

import sys
import importlib.util
from typing import Dict, Any, Set, Optional
from pathlib import Path


class ConfigurationAnalyzer:
    """
    Analyzes configuration classes to extract comprehensive field information.
    
    Supports:
    - Pydantic v1 and v2 configurations
    - Type annotation analysis
    - Default value detection
    - Required/optional field classification
    """
    
    def __init__(self, configs_dir: str):
        """
        Initialize the configuration analyzer.
        
        Args:
            configs_dir: Directory containing configuration files
        """
        self.configs_dir = Path(configs_dir)
    
    def load_config_from_python(self, config_path: Path, builder_name: str) -> Dict[str, Any]:
        """
        Load configuration from Python file with robust import handling.
        
        Args:
            config_path: Path to the configuration file
            builder_name: Name of the builder (for class name inference)
            
        Returns:
            Configuration analysis dictionary
        """
        try:
            # Try to import the module directly
            module_name = f"config_{builder_name}_step"
            
            # Add both the configs directory and the project root to sys.path temporarily
            configs_dir_str = str(self.configs_dir)
            project_root_str = str(self.configs_dir.parent.parent.parent)
            
            paths_to_add = []
            if configs_dir_str not in sys.path:
                sys.path.insert(0, configs_dir_str)
                paths_to_add.append(configs_dir_str)
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)
                paths_to_add.append(project_root_str)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, config_path)
                module = importlib.util.module_from_spec(spec)
                
                # Set up the module's __package__ to help with relative imports
                # Use the correct package path based on the project structure
                module.__package__ = 'cursus.steps.configs'
                
                # Add the module to sys.modules with the correct package structure
                sys.modules[module_name] = module
                sys.modules[f'cursus.steps.configs.{module_name}'] = module
                
                spec.loader.exec_module(module)
            finally:
                # Clean up sys.path
                for path in paths_to_add:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Use systematic approach with step registry
            config_class = None
            config_class_name = None
            
            # Strategy 1: Use step registry to get the correct config class name
            try:
                # Import the registry functions - use absolute import path
                project_root = str(self.configs_dir.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                from src.cursus.registry.step_names import get_canonical_name_from_file_name, get_config_class_name
                
                # Get canonical step name from builder name (script name)
                canonical_name = get_canonical_name_from_file_name(builder_name)
                
                # Get the correct config class name from registry
                registry_config_class_name = get_config_class_name(canonical_name)
                
                # Try to find this class in the module
                if hasattr(module, registry_config_class_name):
                    config_class = getattr(module, registry_config_class_name)
                    config_class_name = registry_config_class_name
                    
            except Exception as registry_error:
                # If registry approach fails, fall back to pattern matching
                pass
            
            # Strategy 2: Fallback to pattern matching if registry approach failed
            if config_class is None:
                possible_names = [
                    f"{builder_name.title().replace('_', '')}Config",
                    f"{''.join(word.capitalize() for word in builder_name.split('_'))}Config",
                    f"{''.join(word.capitalize() for word in builder_name.split('_'))}StepConfig",  # StepConfig pattern
                    f"CurrencyConversionConfig",  # Specific case
                    f"DummyTrainingConfig",       # Specific case
                    f"BatchTransformStepConfig",  # Specific case
                    f"XGBoostModelEvalConfig"     # Specific case for xgboost_model_evaluation
                ]
                
                for name in possible_names:
                    if hasattr(module, name):
                        config_class = getattr(module, name)
                        config_class_name = name
                        break
            
            if config_class is None:
                # List all classes in the module for debugging
                classes = [name for name in dir(module) if isinstance(getattr(module, name), type)]
                raise ValueError(f"Configuration class not found in {config_path}. Available classes: {classes}")
            
            # Analyze the configuration class
            return self.analyze_config_class(config_class, config_class_name)
            
        except Exception as e:
            # Return a simplified analysis if we can't load the module
            return {
                'class_name': f"{builder_name}Config",
                'fields': {},
                'required_fields': set(),
                'optional_fields': set(),
                'default_values': {},
                'load_error': str(e)
            }
    
    def analyze_config_class(self, config_class, class_name: str) -> Dict[str, Any]:
        """
        Analyze configuration class to extract comprehensive field information.
        
        Args:
            config_class: The configuration class to analyze
            class_name: Name of the configuration class
            
        Returns:
            Dictionary containing field analysis results
        """
        analysis = {
            'class_name': class_name,
            'fields': {},
            'required_fields': [],
            'optional_fields': [],
            'default_values': {}
        }
        
        # Get all annotations from the class hierarchy (including inherited fields)
        all_annotations = {}
        
        # Walk through the MRO (Method Resolution Order) to get all annotations
        for cls in reversed(config_class.__mro__):
            if hasattr(cls, '__annotations__'):
                all_annotations.update(cls.__annotations__)
        
        # Process all annotations (direct and inherited)
        for field_name, field_type in all_annotations.items():
            # Skip private fields and special methods
            if field_name.startswith('_'):
                continue
                
            # Determine if field is optional based on type annotation
            is_optional = self._is_optional_field(field_type, field_name, config_class)
            
            analysis['fields'][field_name] = {
                'type': str(field_type),
                'required': not is_optional
            }
            
            if is_optional:
                analysis['optional_fields'].append(field_name)
            else:
                analysis['required_fields'].append(field_name)
        
        # Check for Pydantic model fields (v2 style) - includes inherited fields
        if hasattr(config_class, 'model_fields'):
            for field_name, field_info in config_class.model_fields.items():
                # Skip if we already processed this field from annotations
                if field_name in analysis['fields']:
                    # Update the required status based on Pydantic field info
                    if hasattr(field_info, 'is_required'):
                        is_required = field_info.is_required()
                        analysis['fields'][field_name]['required'] = is_required
                        
                        # Update the lists
                        if is_required:
                            if field_name in analysis['optional_fields']:
                                analysis['optional_fields'].remove(field_name)
                            if field_name not in analysis['required_fields']:
                                analysis['required_fields'].append(field_name)
                        else:
                            if field_name in analysis['required_fields']:
                                analysis['required_fields'].remove(field_name)
                            if field_name not in analysis['optional_fields']:
                                analysis['optional_fields'].append(field_name)
                else:
                    # Add field that wasn't in annotations
                    is_required = hasattr(field_info, 'is_required') and field_info.is_required()
                    analysis['fields'][field_name] = {
                        'type': str(getattr(field_info, 'annotation', 'Any')),
                        'required': is_required
                    }
                    
                    if is_required:
                        analysis['required_fields'].append(field_name)
                    else:
                        analysis['optional_fields'].append(field_name)
                
                # Extract default values from Pydantic field info
                if hasattr(field_info, 'default') and field_info.default is not ...:
                    analysis['default_values'][field_name] = field_info.default
        
        # Check for Pydantic v1 style fields
        elif hasattr(config_class, '__fields__'):
            for field_name, field_info in config_class.__fields__.items():
                # Skip if we already processed this field from annotations
                if field_name in analysis['fields']:
                    # Update the required status based on Pydantic field info
                    if hasattr(field_info, 'required'):
                        is_required = field_info.required
                        analysis['fields'][field_name]['required'] = is_required
                        
                        # Update the lists
                        if is_required:
                            if field_name in analysis['optional_fields']:
                                analysis['optional_fields'].remove(field_name)
                            if field_name not in analysis['required_fields']:
                                analysis['required_fields'].append(field_name)
                        else:
                            if field_name in analysis['required_fields']:
                                analysis['required_fields'].remove(field_name)
                            if field_name not in analysis['optional_fields']:
                                analysis['optional_fields'].append(field_name)
                else:
                    # Add field that wasn't in annotations
                    is_required = hasattr(field_info, 'required') and field_info.required
                    analysis['fields'][field_name] = {
                        'type': str(getattr(field_info, 'type_', 'Any')),
                        'required': is_required
                    }
                    
                    if is_required:
                        analysis['required_fields'].append(field_name)
                    else:
                        analysis['optional_fields'].append(field_name)
                
                # Extract default values from Pydantic field info
                if hasattr(field_info, 'default') and field_info.default is not ...:
                    analysis['default_values'][field_name] = field_info.default
        
        # Check for properties and other attributes (including inherited)
        for attr_name in dir(config_class):
            if not attr_name.startswith('_'):
                attr_value = getattr(config_class, attr_name, None)
                
                # Check if it's a property
                if isinstance(attr_value, property):
                    # Add property as an optional field if not already present
                    if attr_name not in analysis['fields']:
                        analysis['fields'][attr_name] = {
                            'type': 'property',
                            'required': False  # Properties are typically computed, so optional
                        }
                        analysis['optional_fields'].append(attr_name)
                
                # Check for default values (non-callable, non-descriptor attributes)
                elif (attr_name in analysis['fields'] and 
                      not callable(attr_value) and 
                      not hasattr(attr_value, '__get__')):  # Skip descriptors
                    analysis['default_values'][attr_name] = attr_value
                    # If a field has a default value, it's optional
                    if attr_name in analysis['required_fields']:
                        analysis['required_fields'].remove(attr_name)
                        if attr_name not in analysis['optional_fields']:
                            analysis['optional_fields'].append(attr_name)
                        if attr_name in analysis['fields']:
                            analysis['fields'][attr_name]['required'] = False
        
        return analysis
    
    def _is_optional_field(self, field_type, field_name: str, config_class) -> bool:
        """
        Determine if a field is optional based on its type annotation and Field definition.
        
        Supports both Pydantic v1 and v2 field detection with comprehensive type analysis.
        
        Args:
            field_type: The type annotation for the field
            field_name: Name of the field
            config_class: The configuration class
            
        Returns:
            True if the field is optional, False if required
        """
        import typing
        
        # First priority: Check Pydantic field info for definitive answer
        try:
            if hasattr(config_class, 'model_fields'):
                # Pydantic v2 style
                field_info = config_class.model_fields.get(field_name)
                if field_info and hasattr(field_info, 'is_required'):
                    # Use Pydantic's own determination of required/optional
                    return not field_info.is_required()
            elif hasattr(config_class, '__fields__'):
                # Pydantic v1 style
                field_info = config_class.__fields__.get(field_name)
                if field_info and hasattr(field_info, 'required'):
                    return not field_info.required
        except Exception:
            # If Pydantic info fails, fall back to other methods
            pass
        
        # Second priority: Check for Optional[Type] or Union[Type, None] patterns in type annotation
        type_str = str(field_type)
        if 'Optional[' in type_str or 'Union[' in type_str:
            # Handle Optional[Type] which is Union[Type, None]
            if hasattr(typing, 'get_origin') and hasattr(typing, 'get_args'):
                origin = typing.get_origin(field_type)
                if origin is typing.Union:
                    args = typing.get_args(field_type)
                    # Check if None is one of the union types
                    if type(None) in args:
                        return True
            # Fallback string-based check
            elif 'NoneType' in type_str or ', None' in type_str or 'None]' in type_str:
                return True
        
        # Third priority: Check if the field has a class-level default value
        if hasattr(config_class, field_name):
            default_value = getattr(config_class, field_name)
            # If it's not a callable (method) and not a Field descriptor, it's a default
            if not callable(default_value):
                return True
        
        # If none of the above, assume required
        return False
    
    def get_configuration_schema(self, config_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration schema in a standardized format.
        
        Args:
            config_analysis: Result from analyze_config_class
            
        Returns:
            Standardized configuration schema
        """
        return {
            'configuration': {
                'required': config_analysis.get('required_fields', []),
                'optional': config_analysis.get('optional_fields', []),
                'fields': config_analysis.get('fields', {}),
                'defaults': config_analysis.get('default_values', {})
            }
        }
