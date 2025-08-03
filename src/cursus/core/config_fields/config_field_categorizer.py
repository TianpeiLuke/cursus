"""
Configuration field categorizer for organizing fields across multiple configurations.

This module provides a rule-based categorizer for configuration fields,
implementing the Declarative Over Imperative principle with explicit rules.
"""

import json
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Set, Optional, Tuple

from pydantic import BaseModel

from .constants import (
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC,
    NON_STATIC_FIELD_PATTERNS,
    NON_STATIC_FIELD_EXCEPTIONS,
    CategoryType
)
from .type_aware_config_serializer import serialize_config

class ConfigFieldCategorizer:
    """
    Responsible for categorizing configuration fields based on their characteristics.
    
    Analyzes field values and metadata across configs to determine proper placement.
    Uses explicit rules with clear precedence for categorization decisions,
    implementing the Declarative Over Imperative principle.
    
    Implements simplified categorization structure with just shared and specific sections.
    """
    
    def __init__(self, config_list: List[Any], processing_step_config_base_class: Optional[type] = None):
        """
        Initialize with list of config objects to categorize.
        
        Args:
            config_list: List of configuration objects to analyze
            processing_step_config_base_class: Base class for processing step configs
        """
        self.config_list = config_list
        self.logger = logging.getLogger(__name__)
        
        # Determine the base class for processing steps
        self.processing_base_class = processing_step_config_base_class
        if self.processing_base_class is None:
            # Try to infer the base class from imports
            try:
                from ..pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
                self.processing_base_class = ProcessingStepConfigBase
            except ImportError:
                self.logger.warning("Could not import ProcessingStepConfigBase. "
                                    "Processing steps will not be properly identified.")
                # Use a fallback approach - assume no processing configs
                self.processing_base_class = object
        
        # Categorize configs
        self.processing_configs = [c for c in config_list 
                                  if isinstance(c, self.processing_base_class)]
        self.non_processing_configs = [c for c in config_list 
                                      if not isinstance(c, self.processing_base_class)]
        
        # Collect field information and categorize
        self.logger.info(f"Collecting field information for {len(config_list)} configs "
                        f"({len(self.processing_configs)} processing configs)")
        self.field_info = self._collect_field_info()
        self.categorization = self._categorize_fields()
    
    def _collect_field_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive information about all fields across configs.
        
        Implements the Single Source of Truth principle by gathering all information
        in one place for consistent categorization decisions.
        
        Returns:
            dict: Field information including values, sources, types, etc.
        """
        field_info = {
            'values': defaultdict(set),             # field_name -> set of values (as JSON strings)
            'sources': defaultdict(list),           # field_name -> list of step names
            'processing_sources': defaultdict(list), # field_name -> list of processing step names
            'non_processing_sources': defaultdict(list), # field_name -> list of non-processing step names
            'is_static': defaultdict(bool),         # field_name -> bool (is this field likely static)
            'is_special': defaultdict(bool),        # field_name -> bool (is this a special field)
            'is_cross_type': defaultdict(bool),     # field_name -> bool (appears in both processing/non-processing)
            'raw_values': defaultdict(dict)         # field_name -> {step_name: actual value}
        }
        
        # Collect information from all configs
        for config in self.config_list:
            serialized = serialize_config(config)
            
            # Extract step name from metadata
            if "_metadata" not in serialized:
                self.logger.warning(f"Config {config.__class__.__name__} does not have _metadata. "
                                   "Using class name as step name.")
                step_name = config.__class__.__name__
            else:
                step_name = serialized["_metadata"].get("step_name", config.__class__.__name__)
            
            # Process each field - ensure serialized is a dictionary
            if not isinstance(serialized, dict):
                self.logger.warning(f"Serialized config for {config.__class__.__name__} is not a dictionary, got {type(serialized)}")
                continue
                
            for field_name, value in serialized.items():
                if field_name == "_metadata":
                    continue
                    
                # Track raw value
                field_info['raw_values'][field_name][step_name] = value
                
                # Track serialized value for comparison
                try:
                    value_str = json.dumps(value, sort_keys=True)
                    field_info['values'][field_name].add(value_str)
                except (TypeError, ValueError):
                    # If not JSON serializable, use object ID as placeholder
                    field_info['values'][field_name].add(f"__non_serializable_{id(value)}__")
                
                # Track sources
                field_info['sources'][field_name].append(step_name)
                
                # Track processing/non-processing sources
                if isinstance(config, self.processing_base_class):
                    field_info['processing_sources'][field_name].append(step_name)
                else:
                    field_info['non_processing_sources'][field_name].append(step_name)
                
                # Determine if cross-type
                is_processing = bool(field_info['processing_sources'][field_name])
                is_non_processing = bool(field_info['non_processing_sources'][field_name])
                field_info['is_cross_type'][field_name] = is_processing and is_non_processing
                
                # Check if special
                field_info['is_special'][field_name] = self._is_special_field(field_name, value, config)
                
                # Check if static
                field_info['is_static'][field_name] = self._is_likely_static(field_name, value)
        
        # Log statistics about field collection
        self.logger.info(f"Collected information for {len(field_info['sources'])} unique fields")
        self.logger.debug(f"Fields with multiple values: "
                          f"{[f for f, v in field_info['values'].items() if len(v) > 1]}")
        self.logger.debug(f"Cross-type fields: {[f for f, v in field_info['is_cross_type'].items() if v]}")
        self.logger.debug(f"Special fields: {[f for f, v in field_info['is_special'].items() if v]}")
                
        return field_info
    
    def _is_special_field(self, field_name: str, value: Any, config: Any) -> bool:
        """
        Determine if a field should be treated as special.
        
        Special fields are always kept in specific sections.
        
        Args:
            field_name: Name of the field
            value: Value of the field
            config: The config containing this field
            
        Returns:
            bool: True if the field is special
        """
        # Check against known special fields
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            return True
            
        # Check if it's a Pydantic model
        if isinstance(value, BaseModel):
            return True
            
        # Check for fields with nested complex structures
        if isinstance(value, dict) and any(isinstance(v, (dict, list)) for v in value.values()):
            # Complex nested structure should be considered special
            return True
            
        return False
    
    def _is_likely_static(self, field_name: str, value: Any, config=None) -> bool:
        """
        Determine if a field is likely static based on name and value.
        
        Static fields are those that don't change at runtime.
        
        Args:
            field_name: Name of the field
            value: Value of the field
            config: Optional config the field belongs to (not used, kept for backwards compatibility)
            
        Returns:
            bool: True if the field is likely static
        """
        # Fields in the exceptions list are considered static
        if field_name in NON_STATIC_FIELD_EXCEPTIONS:
            return True
            
        # Special fields are never static
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            return False
            
        # Pydantic models are never static
        if isinstance(value, BaseModel):
            return False
        
        # Check name patterns that suggest non-static fields
        if any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS):
            return False
            
        # Check complex values
        if isinstance(value, dict) and len(value) > 3:
            return False
        if isinstance(value, list) and len(value) > 5:
            return False
            
        # Default to static
        return True
        
    def _categorize_fields(self) -> Dict[str, Any]:
        """
        Apply categorization rules to all fields.
        
        Implements the Declarative Over Imperative principle with explicit rules.
        Uses the simplified structure with just 'shared' and 'specific' sections.
        
        Returns:
            dict: Field categorization results
        """
        # Simplified structure with just shared and specific
        categorization = {
            'shared': {},
            'specific': defaultdict(dict)
        }
        
        # Apply categorization rules to each field
        for field_name in self.field_info['sources']:
            # Explicit categorization following Explicit Over Implicit principle
            category = self._categorize_field(field_name)
            
            self.logger.debug(f"Field '{field_name}' categorized as {category}")
            
            # Place field in the appropriate category in simplified structure
            self._place_field(field_name, category, categorization)
        
        # Log statistics about categorization
        self.logger.info(f"Shared fields: {len(categorization['shared'])}")
        self.logger.info(f"Specific steps: {len(categorization['specific'])}")
        
        return categorization
    
    def _categorize_field(self, field_name: str) -> CategoryType:
        """
        Determine the category for a field based on simplified explicit rules.
        
        This implements the simplified categorization with just SHARED and SPECIFIC categories.
        
        Args:
            field_name: Name of the field to categorize
            
        Returns:
            CategoryType: Category for the field (SHARED or SPECIFIC)
        """
        info = self.field_info
        
        # Rule 1: Special fields always go to specific sections
        if info['is_special'][field_name]:
            self.logger.debug(f"Rule 1: Field '{field_name}' is special")
            return CategoryType.SPECIFIC
                
        # Rule 2: Fields that only appear in one config are specific
        if len(info['sources'][field_name]) <= 1:
            self.logger.debug(f"Rule 2: Field '{field_name}' only appears in one config")
            return CategoryType.SPECIFIC
                
        # Rule 3: Fields with different values across configs are specific
        if len(info['values'][field_name]) > 1:
            self.logger.debug(f"Rule 3: Field '{field_name}' has different values across configs")
            return CategoryType.SPECIFIC
                
        # Rule 4: Non-static fields are specific
        if not info['is_static'][field_name]:
            self.logger.debug(f"Rule 4: Field '{field_name}' is non-static")
            return CategoryType.SPECIFIC
                
        # Rule 5: Fields with identical values across all configs go to shared
        if len(info['sources'][field_name]) == len(self.config_list) and len(info['values'][field_name]) == 1:
            self.logger.debug(f"Rule 5: Field '{field_name}' has identical values in ALL configs")
            return CategoryType.SHARED
            
        # Default case: if we can't determine clearly, be safe and make it specific
        self.logger.debug(f"Default rule: Field '{field_name}' using safe default rule")
        return CategoryType.SPECIFIC
    
    def _place_field(self, field_name: str, category: CategoryType, 
                    categorization: Dict[str, Any]) -> None:
        """
        Place a field into the appropriate category in the simplified categorization structure.
        
        Args:
            field_name: Name of the field
            category: Category to place the field in (SHARED or SPECIFIC)
            categorization: Categorization structure to update
        """
        info = self.field_info
        
        # Handle each category
        if category == CategoryType.SHARED:
            # Use the common value for all configs
            value_str = next(iter(info['values'][field_name]))
            try:
                categorization['shared'][field_name] = json.loads(value_str)
            except json.JSONDecodeError:
                # Handle non-serializable values
                self.logger.warning(f"Could not deserialize value for shared field '{field_name}'. "
                                    "Using raw value from first config.")
                step_name = info['sources'][field_name][0]
                categorization['shared'][field_name] = info['raw_values'][field_name][step_name]
                    
        else:  # CategoryType.SPECIFIC
            # Add to each config that has this field
            for config in self.config_list:
                step_name = None
                # Get step name from serialized config
                serialized = serialize_config(config)
                if "_metadata" in serialized:
                    step_name = serialized["_metadata"].get("step_name")
                
                if step_name is None:
                    step_name = config.__class__.__name__
                
                # Check if this config has the field
                for field, sources in info['raw_values'].items():
                    if field == field_name and step_name in sources:
                        if step_name not in categorization['specific']:
                            categorization['specific'][step_name] = {}
                        value = sources[step_name]
                        categorization['specific'][step_name][field_name] = value
    
    def get_category_for_field(self, field_name: str, config: Optional[Any] = None) -> Optional[CategoryType]:
        """
        Get the category for a specific field, optionally in a specific config.
        
        Args:
            field_name: Name of the field
            config: Optional config instance
            
        Returns:
            CategoryType: Category for the field or None if field not found
        """
        if field_name not in self.field_info['sources']:
            return None
            
        if config is None:
            # Return general category
            return self._categorize_field(field_name)
        else:
            # Check if this config has this field
            serialized = serialize_config(config)
            if field_name not in serialized or field_name == "_metadata":
                return None
                
            # Get category for this specific instance
            category = self._categorize_field(field_name)
            
            # In simplified model, we only have SHARED and SPECIFIC
            return category
                
    def get_categorized_fields(self) -> Dict[str, Any]:
        """
        Get the categorization result.
        
        Returns:
            dict: Field categorization
        """
        return self.categorization
        
    def print_categorization_stats(self) -> None:
        """
        Print statistics about field categorization for the simplified structure.
        """
        shared_count = len(self.categorization['shared'])
        specific_count = sum(
            len(fields) for fields in self.categorization['specific'].values()
        )
        
        total = shared_count + specific_count
        
        print(f"Field categorization statistics:")
        print(f"  Shared: {shared_count} ({shared_count / total:.1%})")
        print(f"  Specific: {specific_count} ({specific_count / total:.1%})")
        print(f"  Total: {total}")
