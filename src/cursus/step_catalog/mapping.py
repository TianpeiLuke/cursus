"""
Step Catalog Mapping Module - Config-to-Builder Resolution and Legacy Support.

This module contains the mapping functionality extracted from StepCatalog to improve
maintainability and separation of concerns. It handles:
- Config-to-builder resolution
- Legacy alias support
- Pipeline construction interface
- Registry integration for mapping operations

PHASE 1 ENHANCEMENT: Replaces StepBuilderRegistry mapping functionality.
"""

import logging
import re
from typing import Dict, List, Optional, Type, Any

logger = logging.getLogger(__name__)


class StepCatalogMapper:
    """
    Handles all mapping operations for the Step Catalog system.
    
    This class encapsulates the mapping logic that was previously spread across
    StepBuilderRegistry and provides a clean interface for config-to-builder
    resolution, legacy alias handling, and pipeline construction support.
    """
    
    # Legacy aliases for backward compatibility (moved from StepBuilderRegistry)
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package",  # Legacy name from before standardization
        "MIMSPayload": "Payload",  # Legacy name from before standardization
        "ModelRegistration": "Registration",  # Legacy name from before standardization
        "PytorchTraining": "PyTorchTraining",  # Case sensitivity difference
        "PytorchModel": "PyTorchModel",  # Case sensitivity difference
    }
    
    def __init__(self, step_catalog):
        """
        Initialize the mapper with a reference to the step catalog.
        
        Args:
            step_catalog: Reference to the StepCatalog instance
        """
        self.step_catalog = step_catalog
        self.logger = logging.getLogger(__name__)
        
        # Cache for registry functions to avoid repeated imports
        self._registry_functions = {}
    
    def get_builder_for_config(self, config, node_name: str = None) -> Optional[Type]:
        """
        Map config instance directly to builder class.
        
        This method replaces StepBuilderRegistry.get_builder_for_config() functionality
        while using the registry system as Single Source of Truth.
        
        Args:
            config: Configuration instance (BasePipelineConfig)
            node_name: Optional DAG node name for context
            
        Returns:
            Builder class type or None if not found
        """
        try:
            config_class_name = type(config).__name__
            job_type = getattr(config, "job_type", None)
            
            # Use registry system as Single Source of Truth
            canonical_name = self._resolve_canonical_name_from_registry(
                config_class_name, node_name, job_type
            )
            
            # Use step catalog's discovery to load builder class
            return self.step_catalog.load_builder_class(canonical_name)
            
        except Exception as e:
            self.logger.error(f"Error getting builder for config {config_class_name}: {e}")
            return None
    
    def get_builder_for_step_type(self, step_type: str) -> Optional[Type]:
        """
        Get builder class for step type with legacy alias support and job type variant fallback.

        This method replaces StepBuilderRegistry.get_builder_for_step_type() functionality.

        Args:
            step_type: Step type name (may be legacy alias or compound name with job type)

        Returns:
            Builder class type or None if not found
        """
        try:
            # Handle legacy aliases first
            canonical_step_type = self.resolve_legacy_aliases(step_type)
            
            # Try exact match first
            builder_class = self.step_catalog.load_builder_class(canonical_step_type)
            if builder_class:
                return builder_class
                
            # JOB TYPE FALLBACK: If compound name fails, try base name
            if "_" in canonical_step_type:
                base_step_type = canonical_step_type.rsplit("_", 1)[0]
                job_type = canonical_step_type.rsplit("_", 1)[1]
                
                self.logger.info(f"Trying base step type '{base_step_type}' for compound '{canonical_step_type}' (job_type: {job_type})")
                builder_class = self.step_catalog.load_builder_class(base_step_type)
                if builder_class:
                    return builder_class
                    
            return None

        except Exception as e:
            self.logger.error(f"Error getting builder for step type {step_type}: {e}")
            return None
    
    def resolve_legacy_aliases(self, step_type: str) -> str:
        """
        Resolve legacy aliases to canonical names.
        
        Args:
            step_type: Step type (may be legacy alias)
            
        Returns:
            Canonical step type name
        """
        return self.LEGACY_ALIASES.get(step_type, step_type)
    
    def is_step_type_supported(self, step_type: str) -> bool:
        """
        Check if step type is supported (including legacy aliases).
        
        Args:
            step_type: Step type name
            
        Returns:
            True if supported, False otherwise
        """
        try:
            canonical_step_type = self.resolve_legacy_aliases(step_type)
            self.step_catalog._ensure_index_built()
            return canonical_step_type in self.step_catalog._step_index
            
        except Exception as e:
            self.logger.error(f"Error checking step type support for {step_type}: {e}")
            return False
    
    def validate_builder_availability(self, step_types: List[str]) -> Dict[str, bool]:
        """
        Validate that builders are available for step types.
        
        Args:
            step_types: List of step types to validate
            
        Returns:
            Dictionary mapping step types to availability status
        """
        results = {}
        for step_type in step_types:
            try:
                builder_class = self.get_builder_for_step_type(step_type)
                results[step_type] = builder_class is not None
            except Exception:
                results[step_type] = False
        return results
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """
        Get possible config class names for a step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            List of possible configuration class names
        """
        try:
            canonical_step_type = self.resolve_legacy_aliases(step_type)
            
            # Try to get from step info first
            step_info = self.step_catalog.get_step_info(canonical_step_type)
            if step_info and step_info.registry_data.get('config_class'):
                return [step_info.registry_data['config_class']]
            
            # Fallback to naming patterns
            return [f"{step_type}Config", f"{step_type}StepConfig"]
            
        except Exception as e:
            self.logger.error(f"Error getting config types for step type {step_type}: {e}")
            return []
    
    def list_supported_step_types(self) -> List[str]:
        """
        List all supported step types including legacy aliases.
        
        Returns:
            List of supported step type names
        """
        try:
            self.step_catalog._ensure_index_built()
            discovered_types = list(self.step_catalog._step_index.keys())
            legacy_types = list(self.LEGACY_ALIASES.keys())
            return sorted(discovered_types + legacy_types)
            
        except Exception as e:
            self.logger.error(f"Error listing supported step types: {e}")
            return []
    
    def _resolve_canonical_name_from_registry(self, config_class_name: str, 
                                            node_name: str = None, job_type: str = None) -> str:
        """
        Use registry system for canonical name resolution.
        
        Args:
            config_class_name: Configuration class name
            node_name: Optional DAG node name
            job_type: Optional job type
            
        Returns:
            Canonical step name
            
        Raises:
            ValueError: If config class cannot be resolved
        """
        try:
            from ..registry.step_names import get_config_step_registry
            
            # Extract job type from node name if provided and job_type not explicitly provided
            if node_name and not job_type:
                _, extracted_job_type = self._extract_job_type(node_name)
                job_type = extracted_job_type
            
            config_registry = get_config_step_registry()
            canonical_name = config_registry.get(config_class_name)
            
            if not canonical_name:
                # Fallback logic for compatibility (moved from StepBuilderRegistry)
                canonical_name = self._fallback_config_to_step_type(config_class_name)
            
            # Handle job type variants
            if job_type:
                return f"{canonical_name}_{job_type}"
            
            return canonical_name
            
        except Exception as e:
            raise ValueError(f"Cannot resolve canonical name for config class: {config_class_name}") from e
    
    def _extract_job_type(self, node_name: str):
        """
        Extract job type information from a node name.
        
        Args:
            node_name: Node name from DAG
            
        Returns:
            Tuple of (base_name, job_type)
        """
        # Pattern: BaseType_JobType (e.g., CradleDataLoading_training)
        match = re.match(r"^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$", node_name)
        if match:
            base_name, job_type = match.groups()
            return base_name, job_type
        
        # If no pattern match, return the original name with no job type
        return node_name, None
    
    def _fallback_config_to_step_type(self, config_class_name: str) -> str:
        """
        Fallback logic for config class to step type conversion.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            Step type name
        """
        self.logger.warning(
            f"Config class '{config_class_name}' not found in registry, using fallback logic"
        )
        
        # Remove common suffixes
        step_type = config_class_name
        
        # Remove 'Config' suffix
        if step_type.endswith("Config"):
            step_type = step_type[:-6]
        
        # Remove 'Step' suffix if present
        if step_type.endswith("Step"):
            step_type = step_type[:-4]
        
        # Handle common naming patterns
        if step_type == "CradleDataLoad":
            step_type = "CradleDataLoading"
        
        return step_type
    
    def get_registry_function(self, func_name: str):
        """
        Lazy load registry functions to avoid circular imports.
        
        Args:
            func_name: Name of the registry function to load
            
        Returns:
            Registry function
        """
        if func_name not in self._registry_functions:
            try:
                from ..registry.step_names import (
                    get_step_names, get_config_step_registry, validate_step_name
                )
                self._registry_functions.update({
                    'get_step_names': get_step_names,
                    'get_config_step_registry': get_config_step_registry,
                    'validate_step_name': validate_step_name,
                })
            except ImportError as e:
                self.logger.warning(f"Could not import registry functions: {e}")
                return None
        
        return self._registry_functions.get(func_name)
    
    def validate_step_name_with_registry(self, step_name: str) -> bool:
        """
        Use registry system for step name validation.
        
        Args:
            step_name: Step name to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            validate_func = self.get_registry_function('validate_step_name')
            if validate_func:
                return validate_func(step_name)
            return False
        except Exception as e:
            self.logger.error(f"Error validating step name {step_name}: {e}")
            return False


class PipelineConstructionInterface:
    """
    Pipeline construction interface for Step Catalog.
    
    This class provides the pipeline-specific methods that were previously
    in StepBuilderRegistry, now integrated with the Step Catalog system.
    """
    
    def __init__(self, mapper: StepCatalogMapper):
        """
        Initialize with a reference to the mapper.
        
        Args:
            mapper: StepCatalogMapper instance
        """
        self.mapper = mapper
        self.logger = logging.getLogger(__name__)
    
    def get_builder_map(self) -> Dict[str, Type]:
        """
        Get a complete builder map for pipeline construction.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        try:
            builder_map = {}
            step_types = self.mapper.list_supported_step_types()
            
            for step_type in step_types:
                builder_class = self.mapper.get_builder_for_step_type(step_type)
                if builder_class:
                    builder_map[step_type] = builder_class
            
            return builder_map
            
        except Exception as e:
            self.logger.error(f"Error building builder map: {e}")
            return {}
    
    def validate_dag_compatibility(self, step_types: List[str]) -> Dict[str, Any]:
        """
        Validate DAG compatibility with available builders.
        
        Args:
            step_types: List of step types in the DAG
            
        Returns:
            Dictionary with validation results
        """
        try:
            availability = self.mapper.validate_builder_availability(step_types)
            
            missing_builders = [
                step_type for step_type, available in availability.items() 
                if not available
            ]
            
            return {
                'compatible': len(missing_builders) == 0,
                'missing_builders': missing_builders,
                'available_builders': [
                    step_type for step_type, available in availability.items() 
                    if available
                ],
                'total_steps': len(step_types),
                'supported_steps': len(step_types) - len(missing_builders)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating DAG compatibility: {e}")
            return {
                'compatible': False,
                'missing_builders': step_types,
                'available_builders': [],
                'total_steps': len(step_types),
                'supported_steps': 0
            }
    
    def get_step_builder_suggestions(self, config_class_name: str) -> List[str]:
        """
        Get suggestions for step builders based on config class name.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            List of suggested step type names
        """
        try:
            # Try direct resolution first
            try:
                canonical_name = self.mapper._resolve_canonical_name_from_registry(config_class_name)
                return [canonical_name]
            except ValueError:
                pass
            
            # Fallback to pattern matching
            suggestions = []
            all_step_types = self.mapper.list_supported_step_types()
            
            # Remove 'Config' suffix for matching
            base_name = config_class_name
            if base_name.endswith("Config"):
                base_name = base_name[:-6]
            
            # Find similar step types
            for step_type in all_step_types:
                if base_name.lower() in step_type.lower() or step_type.lower() in base_name.lower():
                    suggestions.append(step_type)
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting step builder suggestions for {config_class_name}: {e}")
            return []
