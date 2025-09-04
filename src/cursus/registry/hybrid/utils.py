"""
Consolidated Shared Utilities for Hybrid Registry System

This module contains all shared utility components that eliminate code redundancy
across the hybrid registry system. These utilities provide common functionality
used by all registry components.

Components:
- RegistryLoader: Common loading logic for registry modules
- StepDefinitionConverter: Format conversions between legacy and new formats
- RegistryValidationUtils: Shared validation logic
- RegistryErrorFormatter: Consistent error message formatting
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..exceptions import RegistryError


class RegistryLoadError(RegistryError):
    """Error loading registry from file."""
    pass


class RegistryLoader:
    """
    Shared utility for loading registry modules from files.
    
    Eliminates redundant loading logic between CoreStepRegistry and LocalStepRegistry.
    """
    
    @staticmethod
    def load_registry_module(file_path: str, module_name: str = None) -> Any:
        """
        Common registry loading logic.
        
        Args:
            file_path: Path to the registry file
            module_name: Name for the loaded module
            
        Returns:
            Loaded module object
            
        Raises:
            RegistryLoadError: If module loading fails
        """
        from ..exceptions import RegistryLoadError
        
        try:
            if not Path(file_path).exists():
                raise RegistryLoadError(f"Registry file not found: {file_path}")
            
            module_name = module_name or f"registry_module_{hash(file_path)}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise RegistryLoadError(f"Could not create module spec from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            if isinstance(e, RegistryLoadError):
                raise
            raise RegistryLoadError(f"Failed to load registry module from {file_path}: {e}")
    
    @staticmethod
    def validate_registry_structure(module: Any, required_attributes: List[str]) -> None:
        """
        Validate that a registry module has required attributes.
        
        Args:
            module: Loaded registry module
            required_attributes: List of required attribute names
            
        Raises:
            RegistryLoadError: If required attributes are missing
        """
        from ..exceptions import RegistryLoadError
        
        missing_attributes = []
        for attr_name in required_attributes:
            if not hasattr(module, attr_name):
                missing_attributes.append(attr_name)
        
        if missing_attributes:
            raise RegistryLoadError(f"Registry module missing required attributes: {missing_attributes}")
    
    @staticmethod
    def safe_get_attribute(module: Any, attr_name: str, default: Any = None) -> Any:
        """
        Safely get an attribute from a module with a default value.
        
        Args:
            module: Module to get attribute from
            attr_name: Name of the attribute
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        try:
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
            else:
                return default
        except Exception:
            return default
    
    @staticmethod
    def validate_registry_file(file_path: str) -> bool:
        """
        Validate that a registry file exists and is readable.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            return path.exists() and path.is_file() and path.suffix == '.py'
        except Exception:
            return False
    
    @staticmethod
    def get_registry_attributes(module: Any, expected_attributes: List[str] = None) -> Dict[str, Any]:
        """
        Extract expected attributes from a registry module.
        
        Args:
            module: Loaded registry module
            expected_attributes: List of attribute names to extract
            
        Returns:
            Dictionary of attribute name to value mappings
        """
        if expected_attributes is None:
            # Default to looking for STEP_NAMES
            expected_attributes = ['STEP_NAMES']
        
        attributes = {}
        for attr_name in expected_attributes:
            attr_value = getattr(module, attr_name, {})
            attributes[attr_name] = attr_value
        
        # If only STEP_NAMES was requested, return it directly
        if expected_attributes == ['STEP_NAMES']:
            return attributes.get('STEP_NAMES', {})
        
        return attributes


class HybridStepDefinition:
    """Alias for backward compatibility."""
    pass


class StepDefinitionConverter:
    """
    Shared utility for converting between step definition formats.
    
    Eliminates redundant conversion logic across registry components.
    """
    
    @staticmethod
    def from_legacy_format(step_name: str, 
                          step_info: Dict[str, Any], 
                          registry_type: str = 'core', 
                          workspace_id: str = None) -> 'StepDefinition':
        """
        Convert legacy STEP_NAMES format to StepDefinition.
        
        Args:
            step_name: Name of the step
            step_info: Legacy step information dictionary
            registry_type: Type of registry ('core', 'workspace', 'override')
            workspace_id: Workspace identifier for workspace steps
            
        Returns:
            StepDefinition object
        """
        from .models import StepDefinition
        
        # Extract standard fields
        definition_data = {
            'name': step_name,
            'registry_type': registry_type,
            'config_class': step_info.get('config_class'),
            'spec_type': step_info.get('spec_type'),
            'sagemaker_step_type': step_info.get('sagemaker_step_type'),
            'builder_step_name': step_info.get('builder_step_name'),
            'description': step_info.get('description'),
            'framework': step_info.get('framework'),
            'job_types': step_info.get('job_types', [])
        }
        
        # Add workspace-specific fields
        if workspace_id:
            definition_data['workspace_id'] = workspace_id
        
        # Extract conflict resolution metadata if present
        if 'priority' in step_info:
            definition_data['priority'] = step_info['priority']
        if 'compatibility_tags' in step_info:
            definition_data['compatibility_tags'] = step_info['compatibility_tags']
        if 'framework_version' in step_info:
            definition_data['framework_version'] = step_info['framework_version']
        if 'environment_tags' in step_info:
            definition_data['environment_tags'] = step_info['environment_tags']
        if 'conflict_resolution_strategy' in step_info:
            definition_data['conflict_resolution_strategy'] = step_info['conflict_resolution_strategy']
        
        # Store any additional metadata (excluding fields we've already handled)
        handled_fields = {
            'config_class', 'spec_type', 'sagemaker_step_type', 'builder_step_name', 
            'description', 'framework', 'job_types', 'priority', 'compatibility_tags',
            'framework_version', 'environment_tags', 'conflict_resolution_strategy'
        }
        metadata = {}
        for key, value in step_info.items():
            if key not in handled_fields:
                metadata[key] = value
        if metadata:
            definition_data['metadata'] = metadata
        
        return StepDefinition(**definition_data)
    
    @staticmethod
    def to_legacy_format(definition: 'StepDefinition') -> Dict[str, Any]:
        """
        Convert StepDefinition to legacy STEP_NAMES format.
        
        Args:
            definition: StepDefinition object
            
        Returns:
            Legacy format dictionary
        """
        legacy_dict = {}
        
        # Standard fields
        if definition.sagemaker_step_type:
            legacy_dict['sagemaker_step_type'] = definition.sagemaker_step_type
        if definition.builder_step_name:
            legacy_dict['builder_step_name'] = definition.builder_step_name
        if definition.description:
            legacy_dict['description'] = definition.description
        if definition.framework:
            legacy_dict['framework'] = definition.framework
        if definition.job_types:
            legacy_dict['job_types'] = definition.job_types
        
        # Conflict resolution metadata
        if hasattr(definition, 'priority') and definition.priority != 100:
            legacy_dict['priority'] = definition.priority
        if hasattr(definition, 'compatibility_tags') and definition.compatibility_tags:
            legacy_dict['compatibility_tags'] = definition.compatibility_tags
        if hasattr(definition, 'framework_version') and definition.framework_version:
            legacy_dict['framework_version'] = definition.framework_version
        if hasattr(definition, 'environment_tags') and definition.environment_tags:
            legacy_dict['environment_tags'] = definition.environment_tags
        if hasattr(definition, 'conflict_resolution_strategy') and definition.conflict_resolution_strategy != 'workspace_priority':
            legacy_dict['conflict_resolution_strategy'] = definition.conflict_resolution_strategy
        
        # Additional metadata - handle Mock objects safely
        if hasattr(definition, 'metadata') and definition.metadata:
            try:
                # Check if metadata is iterable and has keys method
                if hasattr(definition.metadata, 'keys') and callable(definition.metadata.keys):
                    legacy_dict.update(definition.metadata)
            except (TypeError, AttributeError):
                # Skip if metadata is not a proper dict-like object
                pass
        
        return legacy_dict
    
    @staticmethod
    def convert_registry_dict(registry_dict: Dict[str, Dict[str, Any]], 
                            registry_type: str = 'core',
                            workspace_id: str = None) -> Dict[str, 'StepDefinition']:
        """
        Convert a complete registry dictionary to StepDefinition objects.
        
        Args:
            registry_dict: Dictionary of step_name -> step_info
            registry_type: Type of registry
            workspace_id: Workspace identifier
            
        Returns:
            Dictionary of step_name -> StepDefinition
        """
        definitions = {}
        for step_name, step_info in registry_dict.items():
            definitions[step_name] = StepDefinitionConverter.from_legacy_format(
                step_name, step_info, registry_type, workspace_id
            )
        return definitions
    
    @staticmethod
    def batch_convert_from_legacy(legacy_definitions: List[Any], 
                                registry_type: str = 'core',
                                workspace_id: str = None) -> List['StepDefinition']:
        """
        Batch convert multiple legacy definitions to StepDefinition objects.
        
        Args:
            legacy_definitions: List of legacy step definition dictionaries or step names
            registry_type: Type of registry
            workspace_id: Workspace identifier
            
        Returns:
            List of StepDefinition objects
        """
        definitions = []
        for legacy_def in legacy_definitions:
            if isinstance(legacy_def, str):
                # If it's just a step name, create a minimal definition
                step_name = legacy_def
                step_info = {'sagemaker_step_type': 'Processing', 'builder_step_name': f'{step_name}Builder'}
            else:
                # It's a dictionary
                step_name = legacy_def.get('name', 'unknown')
                step_info = legacy_def
            
            definitions.append(StepDefinitionConverter.from_legacy_format(
                step_name, step_info, registry_type, workspace_id
            ))
        return definitions
    
    @staticmethod
    def batch_convert_to_legacy(definitions: List['StepDefinition']) -> List[Dict[str, Any]]:
        """
        Batch convert multiple StepDefinition objects to legacy format.
        
        Args:
            definitions: List of StepDefinition objects
            
        Returns:
            List of legacy format dictionaries
        """
        legacy_definitions = []
        for definition in definitions:
            legacy_definitions.append(StepDefinitionConverter.to_legacy_format(definition))
        return legacy_definitions


class RegistryValidationUtils:
    """
    Shared utility for registry validation logic.
    
    Eliminates redundant validation patterns across registry components.
    """
    
    @staticmethod
    def validate_registry_type(registry_type: str) -> str:
        """
        Shared registry type validation.
        
        Args:
            registry_type: Registry type to validate
            
        Returns:
            Validated registry type
            
        Raises:
            ValueError: If registry type is invalid
        """
        allowed_types = {'core', 'workspace', 'override'}
        if registry_type not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}, got: {registry_type}")
        return registry_type
    
    @staticmethod
    def validate_step_name(step_name: str) -> str:
        """
        Shared step name validation.
        
        Args:
            step_name: Step name to validate
            
        Returns:
            Validated step name
            
        Raises:
            ValueError: If step name is invalid
        """
        if not step_name or not step_name.strip():
            raise ValueError("Step name cannot be empty or whitespace")
        
        # Check for valid identifier characters
        if not step_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Step name '{step_name}' contains invalid characters")
        
        return step_name.strip()
    
    @staticmethod
    def validate_workspace_id(workspace_id: str) -> str:
        """
        Shared workspace ID validation.
        
        Args:
            workspace_id: Workspace ID to validate
            
        Returns:
            Validated workspace ID
            
        Raises:
            ValueError: If workspace ID is invalid
        """
        if not workspace_id or not workspace_id.strip():
            raise ValueError("Workspace ID cannot be empty or whitespace")
        
        # Check for valid directory name characters
        if not workspace_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Workspace ID '{workspace_id}' contains invalid characters")
        
        return workspace_id.strip()
    
    @staticmethod
    def validate_step_definition_fields(definition_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shared step definition field validation.
        
        Args:
            definition_data: Step definition data to validate
            
        Returns:
            Validated definition data
            
        Raises:
            ValueError: If any field is invalid
        """
        required_fields = {'name', 'registry_type'}
        missing_fields = required_fields - set(definition_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate individual fields
        RegistryValidationUtils.validate_step_name(definition_data['name'])
        RegistryValidationUtils.validate_registry_type(definition_data['registry_type'])
        
        if 'workspace_id' in definition_data and definition_data['workspace_id']:
            RegistryValidationUtils.validate_workspace_id(definition_data['workspace_id'])
        
        return definition_data
    
    @staticmethod
    def validate_registry_consistency(definitions: Dict[str, 'StepDefinition']) -> List[str]:
        """
        Shared registry consistency validation.
        
        Args:
            definitions: Dictionary of step definitions to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check for duplicate builder names
        builder_names = {}
        for step_name, definition in definitions.items():
            if definition.builder_step_name:
                if definition.builder_step_name in builder_names:
                    issues.append(
                        f"Duplicate builder name '{definition.builder_step_name}' "
                        f"used by steps '{step_name}' and '{builder_names[definition.builder_step_name]}'"
                    )
                else:
                    builder_names[definition.builder_step_name] = step_name
        
        # Check for missing required fields
        for step_name, definition in definitions.items():
            if not definition.sagemaker_step_type:
                issues.append(f"Step '{step_name}' missing sagemaker_step_type")
            if not definition.builder_step_name:
                issues.append(f"Step '{step_name}' missing builder_step_name")
        
        return issues
    
    @staticmethod
    def validate_step_definition_completeness(definition: 'StepDefinition') -> 'RegistryValidationResult':
        """
        Validate completeness of a step definition.
        
        Args:
            definition: Step definition to validate
            
        Returns:
            RegistryValidationResult with validation results
        """
        from .models import RegistryValidationResult
        
        errors = []
        warnings = []
        
        # Check required fields
        if not definition.name:
            errors.append("Step name is required")
        if not definition.sagemaker_step_type:
            errors.append("SageMaker step type is required")
        if not definition.builder_step_name:
            errors.append("Builder step name is required")
        
        # Check optional but recommended fields
        if not definition.description:
            warnings.append("Step description is recommended")
        if not definition.framework:
            warnings.append("Framework specification is recommended")
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            registry_type=definition.registry_type,
            issues=errors + warnings,
            errors=errors,
            warnings=warnings,
            step_count=1
        )
    
    @staticmethod
    def validate_workspace_registry_structure(registry_data: Dict[str, Any]) -> 'RegistryValidationResult':
        """
        Validate workspace registry structure.
        
        Args:
            registry_data: Registry data to validate
            
        Returns:
            RegistryValidationResult with validation results
        """
        from .models import RegistryValidationResult
        
        errors = []
        warnings = []
        
        # Check for required sections
        if 'LOCAL_STEPS' not in registry_data:
            errors.append("LOCAL_STEPS section is required")
        if 'WORKSPACE_METADATA' not in registry_data:
            warnings.append("WORKSPACE_METADATA section is recommended")
        
        # Validate LOCAL_STEPS structure
        if 'LOCAL_STEPS' in registry_data:
            local_steps = registry_data['LOCAL_STEPS']
            if not isinstance(local_steps, dict):
                errors.append("LOCAL_STEPS must be a dictionary")
            else:
                for step_name, step_info in local_steps.items():
                    if not isinstance(step_info, dict):
                        errors.append(f"Step '{step_name}' must be a dictionary")
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            registry_type="workspace",
            issues=errors + warnings,
            errors=errors,
            warnings=warnings,
            step_count=len(registry_data.get('LOCAL_STEPS', {}))
        )
    
    @staticmethod
    def validate_conflict_resolution_metadata(metadata: Dict[str, Any]) -> 'RegistryValidationResult':
        """
        Validate conflict resolution metadata.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            RegistryValidationResult with validation results
        """
        from .models import RegistryValidationResult
        
        errors = []
        warnings = []
        
        # Validate priority
        if 'priority' in metadata:
            priority = metadata['priority']
            if not isinstance(priority, int) or priority < 0:
                errors.append("Priority must be a non-negative integer")
        
        # Validate strategy
        if 'conflict_resolution_strategy' in metadata:
            strategy = metadata['conflict_resolution_strategy']
            allowed_strategies = {'workspace_priority', 'framework_match', 'environment_match', 'manual'}
            if strategy not in allowed_strategies:
                errors.append(f"Invalid conflict resolution strategy: {strategy}")
        
        # Validate framework version
        if 'framework_version' in metadata:
            version = metadata['framework_version']
            if not isinstance(version, str) or not version.strip():
                errors.append("Framework version must be a non-empty string")
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            step_count=1
        )
    
    @staticmethod
    def format_registry_error(error_type: str, details: str, suggestions: List[str] = None) -> str:
        """
        Format registry error messages.
        
        Args:
            error_type: Type of error
            details: Error details
            suggestions: Optional suggestions for fixing the error
            
        Returns:
            Formatted error message
        """
        error_msg = f"Registry Error in {error_type}: {details}"
        if suggestions:
            error_msg += f". Suggestions: {'; '.join(suggestions)}"
        return error_msg


class RegistryErrorFormatter:
    """
    Shared utility for consistent error message formatting.
    
    Eliminates redundant error handling patterns across registry components.
    """
    
    @staticmethod
    def format_step_not_found_error(step_name: str, 
                                   workspace_context: str = None,
                                   available_steps: List[str] = None,
                                   suggestions: List[str] = None) -> str:
        """
        Format consistent step not found error messages.
        
        Args:
            step_name: Name of the step that wasn't found
            workspace_context: Current workspace context
            available_steps: List of available step names
            suggestions: List of suggested alternatives
            
        Returns:
            Formatted error message
        """
        context_info = f" (workspace: {workspace_context})" if workspace_context else " (core registry)"
        
        error_msg = f"Step '{step_name}' not found{context_info}"
        
        if suggestions:
            error_msg += f". Did you mean: {', '.join(suggestions[:3])}"
        
        if available_steps:
            error_msg += f". Available steps: {', '.join(sorted(available_steps))}"
        
        return error_msg
    
    @staticmethod
    def format_registry_load_error(registry_path: str, 
                                  error_details: str,
                                  suggestions: List[str] = None) -> str:
        """
        Format consistent registry loading error messages.
        
        Args:
            registry_path: Path to the registry that failed to load
            error_details: Detailed error information
            suggestions: List of suggested fixes
            
        Returns:
            Formatted error message
        """
        error_msg = f"Failed to load registry from '{registry_path}': {error_details}"
        
        if suggestions:
            error_msg += f". Suggestions: {'; '.join(suggestions)}"
        
        return error_msg
    
    @staticmethod
    def format_conflict_resolution_error(step_name: str,
                                       conflicting_namespaces: List[str],
                                       resolution_context: str = None) -> str:
        """
        Format consistent conflict resolution error messages.
        
        Args:
            step_name: Name of the conflicting step
            conflicting_namespaces: List of namespaces with conflicts
            resolution_context: Context information for resolution
            
        Returns:
            Formatted error message
        """
        context_info = f" in context '{resolution_context}'" if resolution_context else ""
        
        error_msg = (
            f"Cannot resolve step '{step_name}'{context_info}. "
            f"Multiple definitions found in namespaces: {', '.join(conflicting_namespaces)}. "
            f"Use qualified names (e.g., '{conflicting_namespaces[0]}.{step_name}') or "
            f"set resolution context to disambiguate."
        )
        
        return error_msg
    
    @staticmethod
    def format_validation_error(component_type: str,
                              component_name: str,
                              validation_issues: List[str]) -> str:
        """
        Format consistent validation error messages.
        
        Args:
            component_type: Type of component being validated
            component_name: Name of the component
            validation_issues: List of validation issues
            
        Returns:
            Formatted error message
        """
        error_msg = f"{component_type} '{component_name}' validation failed:"
        
        for i, issue in enumerate(validation_issues, 1):
            error_msg += f"\n  {i}. {issue}"
        
        return error_msg
    
    @staticmethod
    def format_registry_error(error_type: str, details: str, suggestions: List[str] = None) -> str:
        """
        Format registry error messages.
        
        Args:
            error_type: Type of error
            details: Error details
            suggestions: Optional suggestions for fixing the error
            
        Returns:
            Formatted error message
        """
        error_msg = f"Registry Error in {error_type}: {details}"
        if suggestions:
            error_msg += f". Suggestions: {'; '.join(suggestions)}"
        return error_msg
    
    @staticmethod
    def format_conflict_error(step_name: str, conflicting_sources: List[str]) -> str:
        """
        Format conflict error messages.
        
        Args:
            step_name: Name of the conflicting step
            conflicting_sources: List of conflicting sources
            
        Returns:
            Formatted error message
        """
        return f"Step Name Conflict: {step_name}"
    
    @staticmethod
    def format_resolution_error(step_name: str, error_details: str, suggestions: List[str] = None) -> str:
        """
        Format resolution error messages.
        
        Args:
            step_name: Name of the step that failed to resolve
            error_details: Error details
            suggestions: Optional suggestions for fixing the error
            
        Returns:
            Formatted error message
        """
        error_msg = f"Failed to resolve step '{step_name}': {error_details}"
        if suggestions:
            error_msg += f". Suggestions: {'; '.join(suggestions)}"
        return error_msg
