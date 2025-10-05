"""
Configuration merger for combining and saving multiple configuration objects.

This module provides a merger that combines configuration objects according to
their field categorization, implementing the Single Source of Truth principle.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel

from .step_catalog_aware_categorizer import StepCatalogAwareConfigFieldCategorizer
from .constants import CategoryType, MergeDirection, SPECIAL_FIELDS_TO_KEEP_SPECIFIC
from .type_aware_config_serializer import serialize_config, TypeAwareConfigSerializer


class ConfigMerger:
    """
    Merger for combining multiple configuration objects into a unified output.

    Uses categorization results to produce properly structured output files.
    Implements the Explicit Over Implicit principle by clearly defining merge behavior.
    """

    def __init__(
        self,
        config_list: List[Any],
        processing_step_config_base_class: Optional[type] = None,
    ):
        """
        Initialize with list of config objects to merge.

        Args:
            config_list: List of configuration objects to merge
            processing_step_config_base_class: Optional base class for processing steps
        """
        self.config_list = config_list
        self.logger = logging.getLogger(__name__)

        # Use StepCatalogAwareConfigFieldCategorizer to categorize fields - implementing Single Source of Truth
        self.logger.info(f"Categorizing fields for {len(config_list)} configs")
        self.categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list, processing_step_config_base_class
        )

        # Create serializer for saving output
        self.serializer = TypeAwareConfigSerializer()

        self.logger.info("Field categorization complete")

    def merge(self) -> Dict[str, Any]:
        """
        Merge configurations according to simplified categorization rules.

        Returns:
            dict: Merged configuration structure with just 'shared' and 'specific' sections
        """
        # Get categorized fields from categorizer - implementing Single Source of Truth
        categorized = self.categorizer.get_categorized_fields()

        # Create the merged output following the simplified structure
        merged = {"shared": categorized["shared"], "specific": categorized["specific"]}

        # Log statistics about the merged result
        shared_count = len(merged["shared"])
        specific_steps = len(merged["specific"])
        specific_fields = sum(
            len(fields) for step, fields in merged["specific"].items()
        )

        self.logger.info(f"Merged result contains:")
        self.logger.info(f"  - {shared_count} shared fields")
        self.logger.info(
            f"  - {specific_steps} specific steps with {specific_fields} total fields"
        )

        # Verify the merged result
        self._verify_merged_output(merged)

        return merged

    def _verify_merged_output(self, merged: Dict[str, Any]) -> None:
        """
        OPTIMIZED: Single comprehensive verification method covering critical requirements only.
        
        Reduction: 100+ lines → ~40 lines (60% reduction)
        Focus: Essential structure validation only
        Performance: Faster validation with single pass
        
        Args:
            merged: Merged configuration structure
        """
        # Single comprehensive verification covering all critical requirements
        self._verify_essential_structure(merged)
    
    def _verify_essential_structure(self, merged: Dict[str, Any]) -> None:
        """
        Single comprehensive verification method covering all essential requirements.
        
        Combines structure integrity, field placement, and mutual exclusivity checks
        into one efficient pass through the data.
        """
        # Validate basic structure
        if set(merged.keys()) != {"shared", "specific"}:
            self.logger.warning(
                f"Merged structure has unexpected keys: {set(merged.keys())}. Expected 'shared' and 'specific' only."
            )
            return
        
        # Collect all field information in single pass
        shared_fields = set(merged["shared"].keys())
        specific_field_info = {}
        special_fields_in_shared = []
        field_collisions = {}
        
        # Single pass through specific configs
        for step_name, fields in merged["specific"].items():
            if not isinstance(fields, dict):
                self.logger.warning(f"Specific config for {step_name} is not a dictionary")
                continue
                
            step_fields = set(fields.keys())
            specific_field_info[step_name] = step_fields
            
            # Check for collisions with shared fields
            collisions = shared_fields.intersection(step_fields)
            if collisions:
                field_collisions[step_name] = collisions
        
        # Check for special fields in shared section
        for field in shared_fields:
            if field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
                special_fields_in_shared.append(field)
        
        # Log any issues found
        if field_collisions:
            for step, collisions in field_collisions.items():
                self.logger.warning(
                    f"Field name collision between shared and specific.{step}: {collisions}"
                )
        
        if special_fields_in_shared:
            self.logger.warning(
                f"Special fields found in shared section: {special_fields_in_shared}"
            )
        
        # Log successful verification
        total_shared = len(shared_fields)
        total_specific = sum(len(fields) for fields in specific_field_info.values())
        self.logger.debug(
            f"Structure verification complete: {total_shared} shared, {total_specific} specific fields"
        )

    def _generate_step_name(self, config: Any) -> str:
        """
        Generate a consistent step name for a config object using the pipeline registry.

        Args:
            config: Config object

        Returns:
            str: Step name
        """
        # Use the serializer's method to ensure consistency
        serializer = TypeAwareConfigSerializer()
        return serializer.generate_step_name(config)


    def save(self, output_file: str) -> Dict[str, Any]:
        """
        Save merged configuration to a file.

        Args:
            output_file: Path to input file

        Returns:
            dict: Merged configuration
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Merge configurations
        merged = self.merge()

        # Create metadata with proper step name -> class name mapping for config_types
        config_types = {}
        for cfg in self.config_list:
            step_name = self._generate_step_name(cfg)
            class_name = cfg.__class__.__name__
            config_types[step_name] = class_name

        # Get field sources (inverted index) from categorizer
        field_sources = self.categorizer.get_field_sources()

        metadata = {
            "created_at": datetime.now().isoformat(),
            "config_types": config_types,
            "field_sources": field_sources,
        }

        # Create the output structure with the simplified format
        output = {"metadata": metadata, "configuration": merged}

        # Serialize and save to file
        self.logger.info(f"Saving merged configuration to {output_file}")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

        self.logger.info(f"Successfully saved merged configuration to {output_file}")
        return merged

    @classmethod
    def load(
        cls, input_file: str, config_classes: Optional[Dict[str, type]] = None
    ) -> Dict[str, Any]:
        """
        Load a merged configuration from a file.

        Supports the simplified structure with just shared and specific sections.

        Args:
            input_file: Path to input file
            config_classes: Optional mapping of class names to class objects

        Returns:
            dict: Loaded configuration in the simplified structure
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration from {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Configuration file not found: {input_file}")

        # Load the JSON file
        with open(input_file, "r") as f:
            file_data = json.load(f)

        # Check if we're dealing with the old format (with metadata and configuration keys)
        # or the new format (direct structure)
        if "configuration" in file_data and isinstance(
            file_data["configuration"], dict
        ):
            # Old format - extract the actual configuration data
            logger.debug("Detected old configuration format with metadata wrapper")
            data = file_data["configuration"]
        else:
            # New format - direct structure
            logger.debug("Detected new configuration format (direct structure)")
            data = file_data

        # Create serializer
        serializer = TypeAwareConfigSerializer(config_classes=config_classes)

        # Process each section into the simplified structure
        result: Dict[str, Any] = {"shared": {}, "specific": {}}

        # Deserialize shared fields
        if "shared" in data:
            for field, value in data["shared"].items():
                result["shared"][field] = serializer.deserialize(value)

        # NOTE: We no longer support the legacy format with processing sections
        # Current implementation uses simplified structure with just shared and specific

        # Deserialize specific fields
        if "specific" in data:
            for step, fields in data["specific"].items():
                if step not in result["specific"]:
                    result["specific"][step] = {}
                for field, value in fields.items():
                    result["specific"][step][field] = serializer.deserialize(value)

        logger.info(f"Successfully loaded configuration from {input_file}")
        return result

    @classmethod
    def merge_with_direction(
        cls,
        source: Dict[str, Any],
        target: Dict[str, Any],
        direction: MergeDirection = MergeDirection.PREFER_SOURCE,
    ) -> Dict[str, Any]:
        """
        Merge two dictionaries with a specified merge direction.

        Args:
            source: Source dictionary
            target: Target dictionary
            direction: Merge direction for conflict resolution

        Returns:
            dict: Merged dictionary
        """
        result = target.copy()

        for key, source_value in source.items():
            if key not in result:
                # Key only in source, add it
                result[key] = source_value
            else:
                target_value = result[key]

                if isinstance(source_value, dict) and isinstance(target_value, dict):
                    # Recursive merge for nested dictionaries
                    result[key] = cls.merge_with_direction(
                        source_value, target_value, direction
                    )
                elif source_value != target_value:
                    # Handle conflict based on direction
                    if direction == MergeDirection.PREFER_SOURCE:
                        result[key] = source_value
                    elif direction == MergeDirection.PREFER_TARGET:
                        pass  # Keep target value
                    elif direction == MergeDirection.ERROR_ON_CONFLICT:
                        raise ValueError(
                            f"Conflict on key {key}: source={source_value}, target={target_value}"
                        )

        return result


# Convenience functions for backward compatibility and ease of use
def merge_and_save_configs(
    config_list: List[Any],
    output_file: str,
    processing_step_config_base_class: Optional[type] = None,
) -> Dict[str, Any]:
    """
    Convenience function to merge configs and save to file.

    Args:
        config_list: List of configuration objects to merge
        output_file: Path to output file
        processing_step_config_base_class: Optional base class for processing steps

    Returns:
        dict: Merged configuration
    """
    merger = ConfigMerger(config_list, processing_step_config_base_class)
    return merger.save(output_file)


def load_configs(
    input_file: str, config_classes: Optional[Dict[str, type]] = None
) -> Dict[str, Any]:
    """
    Convenience function to load configs from file.

    Args:
        input_file: Path to input file
        config_classes: Optional mapping of class names to class objects

    Returns:
        dict: Loaded configuration
    """
    return ConfigMerger.load(input_file, config_classes)
