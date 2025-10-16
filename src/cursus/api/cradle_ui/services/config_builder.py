"""
Configuration builder service for Cradle Data Load Config UI

Simplified service using factory system for configuration generation.
Preserves only unique cradle-specific validation and export functionality.
"""

from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

# Direct factory import - no wrapper methods
from ...factory import ConfigurationGenerator
from ....core.base.config_base import BasePipelineConfig
from ....steps.configs.config_cradle_data_loading_step import CradleDataLoadingConfig

logger = logging.getLogger(__name__)


class ConfigBuilderService:
    """Simplified service using factory system for configuration generation."""
    
    def __init__(self):
        """Initialize with direct factory usage."""
        pass
    
    def validate_and_build_config(self, ui_data: Dict[str, Any]) -> CradleDataLoadingConfig:
        """
        Simplified config building using factory directly.
        
        Args:
            ui_data: Complete UI data from all steps
            
        Returns:
            Built and validated CradleDataLoadingConfig
            
        Raises:
            ValueError: If validation fails or required data is missing
        """
        try:
            # Extract configs
            base_config_data = ui_data.get('base_config', {})
            step_config_data = ui_data.get('step_config', {})
            
            # Use factory directly - no wrapper generator
            if base_config_data:
                generator = ConfigurationGenerator(base_config=BasePipelineConfig(**base_config_data))
                config = generator.generate_config_instance(CradleDataLoadingConfig, step_config_data)
            else:
                config = CradleDataLoadingConfig(**{**base_config_data, **step_config_data})
            
            # Only preserve unique cradle validation
            self._validate_built_config(config)
            return config
            
        except Exception as e:
            logger.error(f"Error building config: {str(e)}")
            raise ValueError(f"Failed to build configuration: {str(e)}")
    
    def _validate_built_config(self, config: CradleDataLoadingConfig) -> None:
        """
        Only cradle-specific validation (unique business logic only).
        
        Args:
            config: Built configuration to validate
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Keep only unique cradle validation rules
            if not config.job_type:
                raise ValueError("Job type is required")
            
            if not config.data_sources_spec.data_sources:
                raise ValueError("At least one data source is required")
            
            if not config.transform_spec.transform_sql:
                raise ValueError("Transform SQL is required")
            
            if not config.output_spec.output_schema:
                raise ValueError("Output schema is required")
            
            if not config.cradle_job_spec.cradle_account:
                raise ValueError("Cradle account is required")
            
        except Exception as e:
            logger.error(f"Config validation failed: {str(e)}")
            raise
    
    def export_config(
        self, 
        config: Dict[str, Any], 
        format: str = "json", 
        include_comments: bool = True
    ) -> str:
        """
        Export configuration in the specified format.
        
        Args:
            config: Configuration dictionary to export
            format: Export format ('json' or 'python')
            include_comments: Whether to include comments in the output
            
        Returns:
            Exported configuration as string
        """
        try:
            if format.lower() == "json":
                return self._export_as_json(config, include_comments)
            elif format.lower() == "python":
                return self._export_as_python(config, include_comments)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting config as {format}: {str(e)}")
            raise ValueError(f"Failed to export configuration: {str(e)}")
    
    def _export_as_json(self, config: Dict[str, Any], include_comments: bool) -> str:
        """Export configuration as JSON."""
        try:
            export_config = dict(config)  # Simple copy
            
            if include_comments:
                export_config["_metadata"] = {
                    "generated_by": "Cradle Data Load Config UI",
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            
            return json.dumps(export_config, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error exporting as JSON: {str(e)}")
            raise
    
    def _export_as_python(self, config: Dict[str, Any], include_comments: bool) -> str:
        """Export configuration as Python code."""
        try:
            lines = []
            
            if include_comments:
                lines.extend([
                    "# Cradle Data Load Configuration",
                    f"# Generated by Cradle Data Load Config UI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "from cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadingConfig",
                    "",
                ])
            
            lines.append("# Create the configuration")
            lines.append(f"config = {repr(config)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error exporting as Python: {str(e)}")
            raise
