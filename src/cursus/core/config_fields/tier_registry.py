"""
ConfigFieldTierRegistry module.

This module defines the ConfigFieldTierRegistry class which serves as the central
registry for field tier classifications in the three-tier configuration architecture.
"""

from typing import Dict, Any, Optional, Set


class ConfigFieldTierRegistry:
    """
    Registry for field tier classifications.

    This class implements the registry for classifying configuration fields into tiers:
    1. Essential User Inputs (Tier 1)
    2. System Inputs (Tier 2)
    3. Derived Inputs (Tier 3)

    The registry provides methods to get and set tier classifications for fields.
    """

    # Default tier classifications based on field analysis
    DEFAULT_TIER_REGISTRY = {
        # Essential User Inputs (Tier 1)
        "region_list": 1,
        "region_selection": 1,
        "full_field_list": 1,
        "cat_field_list": 1,
        "tab_field_list": 1,
        "label_name": 1,
        "id_name": 1,
        "marketplace_id_col": 1,
        "multiclass_categories": 1,
        "class_weights": 1,
        "model_class": 1,
        "num_round": 1,
        "max_depth": 1,
        "min_child_weight": 1,
        "service_name": 1,
        "pipeline_version": 1,
        "framework_version": 1,
        "current_date": 1,
        "source_dir": 1,
        "training_start_datetime": 1,
        "training_end_datetime": 1,
        "calibration_start_datetime": 1,
        "calibration_end_datetime": 1,
        "model_owner": 1,
        "model_registration_domain": 1,
        "expected_tps": 1,
        "max_latency_ms": 1,
        "max_error_rate": 1,
        "author": 1,
        "region": 1,
        "bucket": 1,
        
        # System Inputs (Tier 2)
        "metric_choices": 2,
        "device": 2,
        "header": 2,
        "batch_size": 2,
        "lr": 2,
        "max_epochs": 2,
        "optimizer": 2,
        "py_version": 2,
        "processing_framework_version": 2,
        "processing_instance_type_large": 2,
        "processing_instance_type_small": 2,
        "processing_instance_count": 2,
        "processing_volume_size": 2,
        "test_val_ratio": 2,
        "training_instance_count": 2,
        "training_volume_size": 2,
        "training_instance_type": 2,
        "inference_instance_type": 2,
        "processing_entry_point": 2,
        "model_eval_processing_entry_point": 2,
        "model_eval_job_type": 2,
        "packaging_entry_point": 2,
        "training_entry_point": 2,
        "calibration_method": 2,
        "score_field": 2,
        "score_field_prefix": 2,
        "use_large_processing_instance": 2,
        "eval_metric_choices": 2,
        "max_acceptable_error_rate": 2,
        "default_numeric_value": 2,
        "default_text_value": 2,
        "special_field_values": 2,
        "source_model_inference_content_types": 2,
        "source_model_inference_response_types": 2,
        
        # All other fields default to Tier 3 (derived)
    }
    
    @classmethod
    def get_tier(cls, field_name: str) -> int:
        """
        Get tier classification for a field.
        
        Args:
            field_name: The name of the field to get the tier for
            
        Returns:
            int: Tier classification (1, 2, or 3)
        """
        return cls.DEFAULT_TIER_REGISTRY.get(field_name, 3)  # Default to Tier 3
        
    @classmethod
    def register_field(cls, field_name: str, tier: int) -> None:
        """
        Register a field with a specific tier.
        
        Args:
            field_name: The name of the field to register
            tier: The tier to assign (1, 2, or 3)
            
        Raises:
            ValueError: If tier is not 1, 2, or 3
        """
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
            
        cls.DEFAULT_TIER_REGISTRY[field_name] = tier
        
    @classmethod
    def register_fields(cls, tier_mapping: Dict[str, int]) -> None:
        """
        Register multiple fields with their tiers.
        
        Args:
            tier_mapping: Dictionary mapping field names to tier classifications
            
        Raises:
            ValueError: If any tier is not 1, 2, or 3
        """
        for field_name, tier in tier_mapping.items():
            if tier not in [1, 2, 3]:
                raise ValueError(f"Tier must be 1, 2, or 3, got {tier} for field {field_name}")
                
        cls.DEFAULT_TIER_REGISTRY.update(tier_mapping)
        
    @classmethod
    def get_fields_by_tier(cls, tier: int) -> Set[str]:
        """
        Get all fields assigned to a specific tier.
        
        Args:
            tier: Tier classification (1, 2, or 3)
            
        Returns:
            Set[str]: Set of field names assigned to the specified tier
            
        Raises:
            ValueError: If tier is not 1, 2, or 3
        """
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
            
        return {field for field, t in cls.DEFAULT_TIER_REGISTRY.items() if t == tier}
        
    @classmethod
    def reset_to_defaults(cls) -> None:
        """
        Reset the registry to default tier classifications.
        
        This method is primarily intended for testing purposes.
        """
        # Make a deep copy of the original defaults
        cls.DEFAULT_TIER_REGISTRY = {
            k: v for k, v in ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.items()
        }
