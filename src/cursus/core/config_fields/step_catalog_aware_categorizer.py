"""
Step Catalog Aware Config Field Categorizer.

Enhanced categorizer with workspace and framework awareness.

This module extends the existing ConfigFieldCategorizer with step catalog integration
to provide workspace-specific field categorization and framework-specific field handling
while preserving all existing categorization rules and logic.

Workspace: Project-specific field categorization
Framework: Framework-specific field handling  
Preserved: All existing categorization rules and logic
"""

import logging
from typing import Dict, List, Any, Optional, Set, Type
from pathlib import Path

from .config_field_categorizer import ConfigFieldCategorizer
from .unified_config_manager import get_unified_config_manager

logger = logging.getLogger(__name__)


class StepCatalogAwareConfigFieldCategorizer(ConfigFieldCategorizer):
    """
    Enhanced categorizer with workspace and framework awareness.
    
    Workspace: Project-specific field categorization
    Framework: Framework-specific field handling
    Preserved: All existing categorization rules and logic
    """
    
    def __init__(
        self, 
        config_list: List[Any], 
        processing_step_config_base_class: Optional[Type] = None,
        project_id: Optional[str] = None,
        step_catalog: Optional[Any] = None,
        workspace_root: Optional[Path] = None
    ):
        """
        Initialize step catalog aware categorizer.
        
        Args:
            config_list: List of configuration objects to categorize
            processing_step_config_base_class: Optional base class for processing steps
            project_id: Optional project ID for workspace-specific categorization
            step_catalog: Optional step catalog instance for enhanced processing
            workspace_root: Optional workspace root for step catalog integration
        """
        # Initialize base categorizer
        super().__init__(config_list, processing_step_config_base_class)
        
        # Enhanced attributes
        self.project_id = project_id
        self.step_catalog = step_catalog
        self.workspace_root = workspace_root
        self.unified_manager = None
        
        # Initialize unified manager if available
        try:
            self.unified_manager = get_unified_config_manager(workspace_root)
            logger.debug("Initialized with unified config manager")
        except Exception as e:
            logger.debug(f"Could not initialize unified config manager: {e}")
        
        # Workspace-specific field mappings
        self._workspace_field_mappings = {}
        self._framework_field_mappings = {}
        
        # Initialize enhanced mappings
        self._initialize_enhanced_mappings()
    
    def _initialize_enhanced_mappings(self) -> None:
        """Initialize workspace and framework-specific field mappings."""
        try:
            # Get workspace-specific field mappings if project_id provided
            if self.project_id and self.unified_manager:
                self._workspace_field_mappings = self._get_workspace_field_mappings()
            
            # Get framework-specific field mappings
            self._framework_field_mappings = self._get_framework_field_mappings()
            
            logger.debug(f"Initialized enhanced mappings: workspace={len(self._workspace_field_mappings)}, framework={len(self._framework_field_mappings)}")
            
        except Exception as e:
            logger.debug(f"Could not initialize enhanced mappings: {e}")
    
    def _get_workspace_field_mappings(self) -> Dict[str, str]:
        """
        Get workspace-specific field mappings.
        
        Returns:
            Dictionary mapping field names to workspace-specific categories
        """
        workspace_mappings = {}
        
        try:
            # Get config classes for the workspace
            if self.unified_manager:
                config_classes = self.unified_manager.get_config_classes(self.project_id)
                
                # Analyze workspace-specific patterns
                for class_name, config_class in config_classes.items():
                    if hasattr(config_class, 'get_workspace_field_mappings'):
                        class_mappings = config_class.get_workspace_field_mappings(self.project_id)
                        workspace_mappings.update(class_mappings)
                    
                    # Check for workspace-specific field annotations
                    if hasattr(config_class, 'model_fields'):
                        for field_name, field_info in config_class.model_fields.items():
                            # Look for workspace-specific annotations
                            if hasattr(field_info, 'json_schema_extra'):
                                extra = field_info.json_schema_extra or {}
                                if 'workspace_category' in extra:
                                    workspace_mappings[field_name] = extra['workspace_category']
        
        except Exception as e:
            logger.debug(f"Could not get workspace field mappings: {e}")
        
        return workspace_mappings
    
    def _get_framework_field_mappings(self) -> Dict[str, str]:
        """
        Get framework-specific field mappings.
        
        Returns:
            Dictionary mapping field names to framework-specific categories
        """
        framework_mappings = {
            # SageMaker-specific fields
            'sagemaker_session': 'framework_specific',
            'sagemaker_config': 'framework_specific',
            'role_arn': 'framework_specific',
            'security_group_ids': 'framework_specific',
            'subnets': 'framework_specific',
            'kms_key': 'framework_specific',
            
            # Docker/Container-specific fields
            'image_uri': 'framework_specific',
            'container_entry_point': 'framework_specific',
            'container_arguments': 'framework_specific',
            'environment_variables': 'framework_specific',
            
            # Kubernetes-specific fields
            'namespace': 'framework_specific',
            'service_account': 'framework_specific',
            'pod_template': 'framework_specific',
            
            # Cloud provider-specific fields
            'aws_region': 'cloud_specific',
            'azure_region': 'cloud_specific',
            'gcp_project': 'cloud_specific',
            'cloud_credentials': 'cloud_specific',
            
            # ML framework-specific fields
            'pytorch_version': 'ml_framework',
            'tensorflow_version': 'ml_framework',
            'xgboost_version': 'ml_framework',
            'sklearn_version': 'ml_framework',
            'cuda_version': 'ml_framework',
        }
        
        return framework_mappings
    
    def _categorize_field_with_step_catalog_context(
        self, 
        field_name: str, 
        field_values: List[Any], 
        config_names: List[str]
    ) -> str:
        """
        Categorize field with step catalog context.
        
        Args:
            field_name: Name of the field to categorize
            field_values: List of values for this field across configs
            config_names: List of config names that have this field
            
        Returns:
            Category name for the field
        """
        # Check workspace-specific mappings first
        if field_name in self._workspace_field_mappings:
            workspace_category = self._workspace_field_mappings[field_name]
            logger.debug(f"Field {field_name} categorized as {workspace_category} (workspace-specific)")
            return workspace_category
        
        # Check framework-specific mappings
        if field_name in self._framework_field_mappings:
            framework_category = self._framework_field_mappings[field_name]
            logger.debug(f"Field {field_name} categorized as {framework_category} (framework-specific)")
            return framework_category
        
        # Use enhanced tier-aware categorization if unified manager available
        if self.unified_manager:
            try:
                # Get field tiers from config instances
                for config in self.config_list:
                    if hasattr(config, field_name):
                        field_tiers = self.unified_manager.get_field_tiers(config)
                        
                        # Map tier information to categories
                        for tier_name, fields in field_tiers.items():
                            if field_name in fields:
                                if tier_name.lower() in ['essential', 'tier1']:
                                    return 'shared'  # Essential fields are typically shared
                                elif tier_name.lower() in ['system', 'tier2']:
                                    return 'specific'  # System fields are typically specific
                                elif tier_name.lower() in ['derived', 'tier3']:
                                    return 'specific'  # Derived fields are typically specific
                                
            except Exception as e:
                logger.debug(f"Could not use tier-aware categorization: {e}")
        
        # Fall back to base categorization logic
        return super()._categorize_field(field_name, field_values, config_names)
    
    def _categorize_field(
        self, 
        field_name: str, 
        field_values: List[Any], 
        config_names: List[str]
    ) -> str:
        """
        Override base categorization to include step catalog context.
        
        Args:
            field_name: Name of the field to categorize
            field_values: List of values for this field across configs
            config_names: List of config names that have this field
            
        Returns:
            Category name for the field
        """
        # Use enhanced categorization with step catalog context
        enhanced_category = self._categorize_field_with_step_catalog_context(
            field_name, field_values, config_names
        )
        
        # If enhanced categorization didn't provide a definitive answer,
        # fall back to base categorization logic
        if enhanced_category in ['framework_specific', 'cloud_specific', 'ml_framework']:
            # These are all specific categories
            return 'specific'
        elif enhanced_category in ['shared', 'specific']:
            return enhanced_category
        else:
            # Use base categorization as final fallback
            return super()._categorize_field(field_name, field_values, config_names)
    
    def get_enhanced_categorization_info(self) -> Dict[str, Any]:
        """
        Get enhanced categorization information.
        
        Returns:
            Dictionary with enhanced categorization details
        """
        info = {
            'project_id': self.project_id,
            'workspace_field_mappings_count': len(self._workspace_field_mappings),
            'framework_field_mappings_count': len(self._framework_field_mappings),
            'unified_manager_available': self.unified_manager is not None,
            'step_catalog_available': self.step_catalog is not None,
        }
        
        # Add step catalog information if available
        if self.step_catalog:
            try:
                info['step_catalog_info'] = {
                    'catalog_type': type(self.step_catalog).__name__,
                    'workspace_root': str(getattr(self.step_catalog, 'workspace_root', 'unknown')),
                }
            except Exception as e:
                logger.debug(f"Could not get step catalog info: {e}")
        
        return info
    
    def categorize_with_enhanced_metadata(self) -> Dict[str, Any]:
        """
        Perform categorization with enhanced metadata.
        
        Returns:
            Categorization result with enhanced metadata
        """
        # Perform standard categorization
        result = self.categorize()
        
        # Add enhanced metadata
        result['enhanced_metadata'] = self.get_enhanced_categorization_info()
        
        return result


def create_step_catalog_aware_categorizer(
    config_list: List[Any],
    processing_step_config_base_class: Optional[Type] = None,
    project_id: Optional[str] = None,
    step_catalog: Optional[Any] = None,
    workspace_root: Optional[Path] = None
) -> StepCatalogAwareConfigFieldCategorizer:
    """
    Factory function to create a step catalog aware categorizer.
    
    Args:
        config_list: List of configuration objects to categorize
        processing_step_config_base_class: Optional base class for processing steps
        project_id: Optional project ID for workspace-specific categorization
        step_catalog: Optional step catalog instance for enhanced processing
        workspace_root: Optional workspace root for step catalog integration
        
    Returns:
        StepCatalogAwareConfigFieldCategorizer instance
    """
    return StepCatalogAwareConfigFieldCategorizer(
        config_list=config_list,
        processing_step_config_base_class=processing_step_config_base_class,
        project_id=project_id,
        step_catalog=step_catalog,
        workspace_root=workspace_root
    )
