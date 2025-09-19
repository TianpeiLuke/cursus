"""
TierRegistry Adapter - Backward compatibility adapter for eliminated TierRegistry.

This adapter provides backward compatibility for code that still uses TierRegistry
while redirecting to the UnifiedConfigManager's config-class-based tier methods.

MIGRATION: TierRegistry eliminated (150 lines → 0 lines)
- Replaced hardcoded tier mappings with config class methods
- Uses config instances' own categorize_fields() methods
- Maintains backward compatibility through adapter pattern
"""

import logging
from typing import Dict, Set, Optional, Any
from pydantic import BaseModel

from .unified_config_manager import get_unified_config_manager

logger = logging.getLogger(__name__)


class ConfigFieldTierRegistryAdapter:
    """
    Backward compatibility adapter for eliminated TierRegistry.
    
    Redirects tier queries to config classes' own categorize_fields() methods
    via the UnifiedConfigManager.
    """
    
    # Legacy tier mapping for fallback (minimal set)
    FALLBACK_TIER_MAPPING = {
        # Essential fields (Tier 1)
        "region": 1,
        "pipeline_name": 1,
        "full_field_list": 1,
        "label_name": 1,
        "id_name": 1,
        # System fields (Tier 2)  
        "instance_type": 2,
        "framework_version": 2,
        "processing_entry_point": 2,
        # All others default to Tier 3
    }
    
    @classmethod
    def get_tier(cls, field_name: str, config_instance: Optional[BaseModel] = None) -> int:
        """
        Get tier classification for a field.
        
        Args:
            field_name: The name of the field to get the tier for
            config_instance: Optional config instance for context-aware classification
            
        Returns:
            int: Tier classification (1, 2, or 3)
        """
        if config_instance:
            try:
                # Use config's own categorization if available
                manager = get_unified_config_manager()
                field_tiers = manager.get_field_tiers(config_instance)
                
                # Map tier names to numbers
                for tier_name, fields in field_tiers.items():
                    if field_name in fields:
                        if tier_name.lower() in ['essential', 'tier1']:
                            return 1
                        elif tier_name.lower() in ['system', 'tier2']:
                            return 2
                        else:
                            return 3
                            
            except Exception as e:
                logger.warning(f"Failed to get tier from config instance: {e}")
        
        # Fallback to legacy mapping
        return cls.FALLBACK_TIER_MAPPING.get(field_name, 3)
    
    @classmethod
    def register_field(cls, field_name: str, tier: int) -> None:
        """
        Register a field with a specific tier (deprecated).
        
        Args:
            field_name: The name of the field to register
            tier: The tier to assign (1, 2, or 3)
        """
        logger.warning(
            f"TierRegistry.register_field() is deprecated. "
            f"Field tiers should be defined in config classes' categorize_fields() methods."
        )
        
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        
        # Update fallback mapping for backward compatibility
        cls.FALLBACK_TIER_MAPPING[field_name] = tier
    
    @classmethod
    def register_fields(cls, tier_mapping: Dict[str, int]) -> None:
        """
        Register multiple fields with their tiers (deprecated).
        
        Args:
            tier_mapping: Dictionary mapping field names to tier classifications
        """
        logger.warning(
            f"TierRegistry.register_fields() is deprecated. "
            f"Field tiers should be defined in config classes' categorize_fields() methods."
        )
        
        for field_name, tier in tier_mapping.items():
            if tier not in [1, 2, 3]:
                raise ValueError(f"Tier must be 1, 2, or 3, got {tier} for field {field_name}")
        
        # Update fallback mapping for backward compatibility
        cls.FALLBACK_TIER_MAPPING.update(tier_mapping)
    
    @classmethod
    def get_fields_by_tier(cls, tier: int, config_instance: Optional[BaseModel] = None) -> Set[str]:
        """
        Get all fields assigned to a specific tier.
        
        Args:
            tier: Tier classification (1, 2, or 3)
            config_instance: Optional config instance for context-aware classification
            
        Returns:
            Set[str]: Set of field names assigned to the specified tier
        """
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
        
        if config_instance:
            try:
                # Use config's own categorization if available
                manager = get_unified_config_manager()
                field_tiers = manager.get_field_tiers(config_instance)
                
                # Map tier numbers to names and collect fields
                tier_names = {
                    1: ['essential', 'tier1'],
                    2: ['system', 'tier2'], 
                    3: ['derived', 'tier3']
                }
                
                result_fields = set()
                for tier_name, fields in field_tiers.items():
                    if tier_name.lower() in tier_names[tier]:
                        result_fields.update(fields)
                
                if result_fields:
                    return result_fields
                    
            except Exception as e:
                logger.warning(f"Failed to get fields by tier from config instance: {e}")
        
        # Fallback to legacy mapping
        return {field for field, t in cls.FALLBACK_TIER_MAPPING.items() if t == tier}
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """
        Reset the registry to default tier classifications (deprecated).
        """
        logger.warning(
            f"TierRegistry.reset_to_defaults() is deprecated. "
            f"Field tiers are now managed by config classes themselves."
        )
        # Reset fallback mapping to minimal set
        cls.FALLBACK_TIER_MAPPING = {
            "region": 1,
            "pipeline_name": 1,
            "full_field_list": 1,
            "label_name": 1,
            "id_name": 1,
            "instance_type": 2,
            "framework_version": 2,
            "processing_entry_point": 2,
        }


# Backward compatibility alias
ConfigFieldTierRegistry = ConfigFieldTierRegistryAdapter
