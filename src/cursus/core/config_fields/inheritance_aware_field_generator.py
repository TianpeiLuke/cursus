"""
Inheritance-Aware Field Generator

This module provides centralized inheritance-aware field generation functionality,
consolidating logic previously scattered across UI modules.

Key Features:
- 4-tier field system with inheritance awareness (essential, system, inherited, derived)
- Integration with unified_config_manager for tier-aware categorization
- Step catalog integration for enhanced field processing
- Workspace and framework-aware field mappings
- Performance-optimized field enhancement algorithms

Consolidates:
- config_ui's get_inheritance_aware_form_fields method (~80 lines)
- Manual inheritance logic from UI modules
- Field tier enhancement scattered across modules
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from .unified_config_manager import get_unified_config_manager
from .step_catalog_aware_categorizer import StepCatalogAwareConfigFieldCategorizer

logger = logging.getLogger(__name__)


class InheritanceAwareFieldGenerator:
    """
    Centralized inheritance-aware field generation.
    
    Provides systematic inheritance-aware field enhancement using the existing
    unified_config_manager infrastructure instead of manual inheritance logic.
    """
    
    def __init__(self, workspace_dirs: Optional[List[str]] = None, project_id: Optional[str] = None):
        """
        Initialize inheritance-aware field generator.
        
        Args:
            workspace_dirs: Optional workspace directories for step catalog integration
            project_id: Optional project ID for workspace-specific processing
        """
        self.workspace_dirs = workspace_dirs
        self.project_id = project_id
        self.unified_manager = get_unified_config_manager(workspace_dirs)
        
        logger.info(f"Initialized InheritanceAwareFieldGenerator with project_id: {project_id}")
    
    def get_inheritance_aware_form_fields(self,
                                        config_class_name: str,
                                        config_class: Optional[Type[BaseModel]] = None,
                                        inheritance_analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate form fields with Smart Default Value Inheritance awareness.
        
        CONSOLIDATED: This method replaces manual inheritance logic from config_ui
        with systematic tier-aware field enhancement using unified_config_manager.
        
        Creates the enhanced 4-tier field system:
        - Tier 1 (essential): Required fields with no defaults (NEW to this config)
        - Tier 2 (system): Optional fields with defaults (NEW to this config)
        - Tier 3 (inherited): Fields inherited from parent configs (NEW TIER)
        - Tier 4 (derived): Computed fields (hidden from UI)
        
        Args:
            config_class_name: Name of the configuration class
            config_class: Optional config class (will be discovered if not provided)
            inheritance_analysis: Optional inheritance analysis from StepCatalog
            
        Returns:
            List of enhanced field definitions with inheritance information
        """
        logger.info(f"🔍 Generating inheritance-aware fields for {config_class_name}")
        
        # Get config class if not provided
        if config_class is None:
            config_classes = self.unified_manager.get_config_classes(self.project_id)
            config_class = config_classes.get(config_class_name)
            
            if not config_class:
                logger.warning(f"Config class {config_class_name} not found for inheritance-aware field generation")
                return []
        
        # Use unified_config_manager's tier-aware field categorization
        enhanced_fields = self._get_tier_aware_fields_with_inheritance(
            config_class, config_class_name, inheritance_analysis
        )
        
        # Log inheritance statistics
        inherited_count = len([f for f in enhanced_fields if f.get('tier') == 'inherited'])
        essential_count = len([f for f in enhanced_fields if f.get('tier') == 'essential'])
        system_count = len([f for f in enhanced_fields if f.get('tier') == 'system'])
        
        logger.info(f"✅ Generated inheritance-aware fields for {config_class_name}: "
                   f"{len(enhanced_fields)} total ({inherited_count} inherited, {essential_count} essential, {system_count} system)")
        
        return enhanced_fields
    
    def _get_tier_aware_fields_with_inheritance(self,
                                              config_class: Type[BaseModel],
                                              config_class_name: str,
                                              inheritance_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get tier-aware fields with inheritance enhancement using unified_config_manager.
        
        CONSOLIDATED: Replaces manual field extraction and inheritance logic with
        systematic approach using existing unified_config_manager capabilities.
        
        Args:
            config_class: Configuration class to analyze
            config_class_name: Name of the configuration class
            inheritance_analysis: Optional inheritance analysis
            
        Returns:
            List of enhanced field definitions with tier and inheritance information
        """
        # Create a temporary config instance for tier analysis
        try:
            # Try to create minimal instance for tier extraction
            temp_instance = self._create_minimal_config_instance(config_class)
            
            if temp_instance is None:
                # Fallback to class-based field extraction
                return self._extract_fields_from_class_definition(config_class, inheritance_analysis)
            
            # Use unified_config_manager's tier-aware categorization
            field_tiers = self.unified_manager.get_field_tiers(temp_instance)
            
            # Extract parent values if inheritance analysis is provided
            parent_values = {}
            immediate_parent = None
            if inheritance_analysis and inheritance_analysis.get('inheritance_enabled'):
                parent_values = inheritance_analysis.get('parent_values', {})
                immediate_parent = inheritance_analysis.get('immediate_parent')
                logger.info(f"🔍 Inheritance enabled: {len(parent_values)} parent values from {immediate_parent}")
            
            # Build enhanced fields using tier information
            enhanced_fields = []
            
            # Process each tier
            for tier_name, field_names in field_tiers.items():
                for field_name in field_names:
                    field_info = self._get_field_info_from_config(temp_instance, field_name)
                    
                    if field_info:
                        # Enhance field with inheritance information
                        enhanced_field = self._enhance_field_with_inheritance(
                            field_info, field_name, tier_name, parent_values, immediate_parent
                        )
                        enhanced_fields.append(enhanced_field)
            
            return enhanced_fields
            
        except Exception as e:
            logger.warning(f"Tier-aware field extraction failed for {config_class_name}: {e}")
            # Fallback to class-based extraction
            return self._extract_fields_from_class_definition(config_class, inheritance_analysis)
    
    def _create_minimal_config_instance(self, config_class: Type[BaseModel]) -> Optional[BaseModel]:
        """
        Create minimal config instance for tier extraction.
        
        Args:
            config_class: Configuration class to instantiate
            
        Returns:
            Minimal config instance or None if creation fails
        """
        try:
            # Try to create with empty args first
            return config_class()
        except Exception:
            try:
                # Try with minimal required fields
                required_fields = {}
                for field_name, field_info in config_class.model_fields.items():
                    if field_info.is_required():
                        # Provide minimal default values based on field type
                        field_type = field_info.annotation
                        if field_type == str:
                            required_fields[field_name] = "default"
                        elif field_type == int:
                            required_fields[field_name] = 0
                        elif field_type == bool:
                            required_fields[field_name] = False
                        elif field_type == list:
                            required_fields[field_name] = []
                        elif field_type == dict:
                            required_fields[field_name] = {}
                
                return config_class(**required_fields)
            except Exception as e:
                logger.debug(f"Could not create minimal instance of {config_class.__name__}: {e}")
                return None
    
    def _get_field_info_from_config(self, config_instance: BaseModel, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract field information from config instance.
        
        Args:
            config_instance: Config instance to extract from
            field_name: Name of field to extract
            
        Returns:
            Field information dictionary or None if field not found
        """
        try:
            if not hasattr(config_instance, field_name):
                return None
            
            field_info = config_instance.model_fields.get(field_name)
            if not field_info:
                return None
            
            # Extract field metadata
            field_value = getattr(config_instance, field_name, None)
            field_type = field_info.annotation
            field_required = field_info.is_required()
            field_default = getattr(field_info, 'default', None)
            field_description = getattr(field_info, 'description', '')
            
            return {
                'name': field_name,
                'type': str(field_type),
                'required': field_required,
                'default': field_default,
                'description': field_description,
                'value': field_value
            }
            
        except Exception as e:
            logger.debug(f"Could not extract field info for {field_name}: {e}")
            return None
    
    def _enhance_field_with_inheritance(self,
                                      field_info: Dict[str, Any],
                                      field_name: str,
                                      tier_name: str,
                                      parent_values: Dict[str, Any],
                                      immediate_parent: Optional[str]) -> Dict[str, Any]:
        """
        Enhance field with inheritance information.
        
        Args:
            field_info: Base field information
            field_name: Name of the field
            tier_name: Tier name from unified_config_manager
            parent_values: Parent values from inheritance analysis
            immediate_parent: Name of immediate parent config
            
        Returns:
            Enhanced field definition with inheritance metadata
        """
        # Start with base field info
        enhanced_field = field_info.copy()
        
        # Map tier names to standard tier system
        tier_mapping = {
            'essential': 'essential',
            'system': 'system',
            'derived': 'derived',
            'tier1': 'essential',
            'tier2': 'system',
            'tier3': 'derived'
        }
        
        standard_tier = tier_mapping.get(tier_name.lower(), 'system')
        
        # Determine smart tier with inheritance awareness
        if field_name in parent_values:
            # Tier 3: Inherited field - pre-populated with parent value
            enhanced_field.update({
                "tier": 'inherited',
                "required": False,  # Override: not required since we have parent value
                "default": parent_values[field_name],
                "is_pre_populated": True,
                "inherited_from": immediate_parent,
                "inheritance_note": f"Auto-filled from {immediate_parent}" if immediate_parent else "Auto-filled from parent",
                "can_override": True,
                "original_tier": standard_tier  # Preserve original tier
            })
        else:
            # Keep original tier and add inheritance metadata
            enhanced_field.update({
                "tier": standard_tier,
                "is_pre_populated": False,
                "inherited_from": None,
                "inheritance_note": None,
                "can_override": False,
                "original_tier": standard_tier
            })
        
        return enhanced_field
    
    def _extract_fields_from_class_definition(self,
                                            config_class: Type[BaseModel],
                                            inheritance_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback field extraction from class definition.
        
        Args:
            config_class: Configuration class to analyze
            inheritance_analysis: Optional inheritance analysis
            
        Returns:
            List of field definitions extracted from class
        """
        logger.info(f"Using fallback class-based field extraction for {config_class.__name__}")
        
        fields = []
        parent_values = {}
        immediate_parent = None
        
        if inheritance_analysis and inheritance_analysis.get('inheritance_enabled'):
            parent_values = inheritance_analysis.get('parent_values', {})
            immediate_parent = inheritance_analysis.get('immediate_parent')
        
        # Extract fields from Pydantic model definition
        for field_name, field_info in config_class.model_fields.items():
            try:
                field_type = field_info.annotation
                field_required = field_info.is_required()
                field_default = getattr(field_info, 'default', None)
                field_description = getattr(field_info, 'description', '')
                
                # Create basic field definition
                field_def = {
                    'name': field_name,
                    'type': str(field_type),
                    'required': field_required,
                    'default': field_default,
                    'description': field_description
                }
                
                # Add inheritance information
                if field_name in parent_values:
                    field_def.update({
                        "tier": 'inherited',
                        "required": False,
                        "default": parent_values[field_name],
                        "is_pre_populated": True,
                        "inherited_from": immediate_parent,
                        "inheritance_note": f"Auto-filled from {immediate_parent}" if immediate_parent else "Auto-filled from parent",
                        "can_override": True,
                        "original_tier": 'essential' if field_required else 'system'
                    })
                else:
                    field_def.update({
                        "tier": 'essential' if field_required else 'system',
                        "is_pre_populated": False,
                        "inherited_from": None,
                        "inheritance_note": None,
                        "can_override": False,
                        "original_tier": 'essential' if field_required else 'system'
                    })
                
                fields.append(field_def)
                
            except Exception as e:
                logger.debug(f"Could not extract field {field_name}: {e}")
                continue
        
        return fields
    
    def get_enhanced_field_categorization(self,
                                        config_list: List[BaseModel],
                                        processing_step_config_base_class: Optional[Type] = None) -> Dict[str, Any]:
        """
        Get enhanced field categorization using step catalog aware categorizer.
        
        Args:
            config_list: List of configuration objects to categorize
            processing_step_config_base_class: Optional base class for processing steps
            
        Returns:
            Enhanced categorization with inheritance and workspace awareness
        """
        logger.info(f"Getting enhanced field categorization for {len(config_list)} configs")
        
        # Create step catalog aware categorizer
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=config_list,
            processing_step_config_base_class=processing_step_config_base_class,
            project_id=self.project_id,
            workspace_root=self.workspace_dirs[0] if self.workspace_dirs else None
        )
        
        # Get enhanced categorization
        categorization = categorizer.categorize_with_enhanced_metadata()
        
        logger.info(f"Enhanced categorization completed with {len(categorization.get('shared', {}))} shared fields")
        
        return categorization


# Global instance for backward compatibility
_field_generator = None

def get_inheritance_aware_field_generator(workspace_dirs: Optional[List[str]] = None, 
                                        project_id: Optional[str] = None) -> InheritanceAwareFieldGenerator:
    """
    Get global inheritance-aware field generator instance.
    
    Args:
        workspace_dirs: Optional workspace directories for step catalog integration
        project_id: Optional project ID for workspace-specific processing
        
    Returns:
        InheritanceAwareFieldGenerator instance
    """
    global _field_generator
    if _field_generator is None:
        _field_generator = InheritanceAwareFieldGenerator(workspace_dirs, project_id)
    return _field_generator


# Convenience function for direct usage
def get_inheritance_aware_form_fields(config_class_name: str,
                                    config_class: Optional[Type[BaseModel]] = None,
                                    inheritance_analysis: Optional[Dict[str, Any]] = None,
                                    workspace_dirs: Optional[List[str]] = None,
                                    project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to get inheritance-aware form fields.
    
    CONSOLIDATED: This function replaces the manual inheritance logic from config_ui
    with systematic tier-aware field enhancement using unified_config_manager.
    
    Args:
        config_class_name: Name of the configuration class
        config_class: Optional config class (will be discovered if not provided)
        inheritance_analysis: Optional inheritance analysis from StepCatalog
        workspace_dirs: Optional workspace directories for step catalog integration
        project_id: Optional project ID for workspace-specific processing
        
    Returns:
        List of enhanced field definitions with inheritance information
    """
    generator = get_inheritance_aware_field_generator(workspace_dirs, project_id)
    return generator.get_inheritance_aware_form_fields(
        config_class_name, config_class, inheritance_analysis
    )
