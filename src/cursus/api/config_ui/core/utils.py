"""
Factory Functions and Utilities

Provides factory functions for easy widget creation and utility functions
for configuration management.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
    from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    from .import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.core.base.config_base import BasePipelineConfig
    from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from .core import UniversalConfigCore

# Handle widget imports with the same pattern
try:
    from ..widgets.widget import UniversalConfigWidget, MultiStepWizard
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    from .import_utils import ensure_cursus_path
    ensure_cursus_path()
    
    from cursus.api.config_ui.widgets.widget import UniversalConfigWidget, MultiStepWizard

logger = logging.getLogger(__name__)


def create_config_widget(config_class_name: str, 
                        base_config: Optional[BasePipelineConfig] = None,
                        workspace_dirs: Optional[List[Union[str, Path]]] = None,
                        **kwargs) -> UniversalConfigWidget:
    """
    Factory function to create configuration widgets for any config type.
    
    Args:
        config_class_name: Name of the configuration class
        base_config: Optional base configuration for pre-population
        workspace_dirs: Optional workspace directories for step catalog
        **kwargs: Additional arguments for config creation
        
    Returns:
        UniversalConfigWidget instance
        
    Example:
        >>> base_config = BasePipelineConfig(author="john", bucket="my-bucket", ...)
        >>> widget = create_config_widget("XGBoostTrainingConfig", base_config=base_config)
        >>> widget.display()
    """
    logger.info(f"Creating config widget for: {config_class_name}")
    
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    return core.create_config_widget(config_class_name, base_config, **kwargs)


def create_pipeline_config_widget(dag: Any, 
                                 base_config: BasePipelineConfig,
                                 processing_config: Optional[ProcessingStepConfigBase] = None,
                                 workspace_dirs: Optional[List[Union[str, Path]]] = None,
                                 **kwargs) -> MultiStepWizard:
    """
    Factory function for DAG-driven pipeline configuration widgets.
    
    Args:
        dag: Pipeline DAG definition
        base_config: Base pipeline configuration
        processing_config: Optional processing configuration
        workspace_dirs: Optional workspace directories for step catalog
        **kwargs: Additional arguments (e.g., hyperparameters)
        
    Returns:
        MultiStepWizard instance
        
    Example:
        >>> from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag
        >>> pipeline_dag = create_xgboost_complete_e2e_dag()
        >>> base_config = BasePipelineConfig(...)
        >>> processing_config = ProcessingStepConfigBase.from_base_config(base_config, ...)
        >>> wizard = create_pipeline_config_widget(pipeline_dag, base_config, processing_config)
        >>> wizard.display()
        >>> # After completion:
        >>> config_list = wizard.get_completed_configs()
        >>> merged_config = merge_and_save_configs(config_list, 'config.json')
    """
    logger.info("Creating DAG-driven pipeline configuration widget")
    
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    return core.create_pipeline_config_widget(dag, base_config, processing_config, **kwargs)


def discover_available_configs(workspace_dirs: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """
    Discover all available configuration classes.
    
    Args:
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        Dictionary mapping config class names to config classes
        
    Example:
        >>> configs = discover_available_configs()
        >>> print(f"Available configs: {list(configs.keys())}")
    """
    logger.info("Discovering available configuration classes")
    
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    config_classes = core.discover_config_classes()
    
    logger.info(f"Discovered {len(config_classes)} configuration classes")
    return config_classes


def validate_config_instance(config_instance: BasePipelineConfig) -> Dict[str, Any]:
    """
    Validate a configuration instance.
    
    Args:
        config_instance: Configuration instance to validate
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> config = BasePipelineConfig(author="test", bucket="test-bucket", ...)
        >>> result = validate_config_instance(config)
        >>> if result["valid"]:
        >>>     print("Configuration is valid!")
    """
    logger.info(f"Validating configuration instance: {type(config_instance).__name__}")
    
    try:
        # Use Pydantic validation if available
        if hasattr(config_instance, 'model_validate'):
            # For Pydantic v2
            validated = config_instance.model_validate(config_instance.model_dump())
            return {
                "valid": True,
                "config_type": type(config_instance).__name__,
                "validated_instance": validated,
                "errors": []
            }
        elif hasattr(config_instance, 'validate'):
            # For Pydantic v1
            validated = config_instance.validate(config_instance.dict())
            return {
                "valid": True,
                "config_type": type(config_instance).__name__,
                "validated_instance": validated,
                "errors": []
            }
        else:
            # Basic validation
            return {
                "valid": True,
                "config_type": type(config_instance).__name__,
                "validated_instance": config_instance,
                "errors": [],
                "note": "Basic validation only (no Pydantic validation available)"
            }
            
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return {
            "valid": False,
            "config_type": type(config_instance).__name__,
            "validated_instance": None,
            "errors": [str(e)]
        }


def get_config_info(config_class_name: str, 
                   workspace_dirs: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """
    Get detailed information about a configuration class.
    
    Args:
        config_class_name: Name of the configuration class
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        Dictionary with configuration class information
        
    Example:
        >>> info = get_config_info("BasePipelineConfig")
        >>> print(f"Fields: {[f['name'] for f in info['fields']]}")
    """
    logger.info(f"Getting configuration info for: {config_class_name}")
    
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    config_classes = core.discover_config_classes()
    
    if config_class_name not in config_classes:
        return {
            "found": False,
            "config_class_name": config_class_name,
            "available_classes": list(config_classes.keys())
        }
    
    config_class = config_classes[config_class_name]
    fields = core._get_form_fields(config_class)
    inheritance_chain = core._get_inheritance_chain(config_class)
    
    return {
        "found": True,
        "config_class_name": config_class_name,
        "config_class": config_class,
        "fields": fields,
        "inheritance_chain": inheritance_chain,
        "field_count": len(fields),
        "required_fields": [f["name"] for f in fields if f["required"]],
        "optional_fields": [f["name"] for f in fields if not f["required"]],
        "has_from_base_config": hasattr(config_class, 'from_base_config'),
        "docstring": config_class.__doc__ or "No documentation available"
    }


def create_example_base_config() -> BasePipelineConfig:
    """
    Create an example base configuration for testing and demonstration.
    
    Returns:
        Example BasePipelineConfig instance
        
    Example:
        >>> base_config = create_example_base_config()
        >>> widget = create_config_widget("ProcessingStepConfigBase", base_config=base_config)
    """
    logger.info("Creating example base configuration")
    
    return BasePipelineConfig(
        author="example-user",
        bucket="example-pipeline-bucket",
        role="arn:aws:iam::123456789012:role/ExampleRole",
        region="NA",
        service_name="ExampleService",
        pipeline_version="1.0.0",
        project_root_folder="cursus",
        model_class="example",
        current_date="2025-10-07"
    )


def create_example_processing_config(base_config: Optional[BasePipelineConfig] = None) -> ProcessingStepConfigBase:
    """
    Create an example processing configuration for testing and demonstration.
    
    Args:
        base_config: Optional base configuration to use
        
    Returns:
        Example ProcessingStepConfigBase instance
        
    Example:
        >>> base_config = create_example_base_config()
        >>> processing_config = create_example_processing_config(base_config)
    """
    logger.info("Creating example processing configuration")
    
    if base_config is None:
        base_config = create_example_base_config()
    
    return ProcessingStepConfigBase.from_base_config(
        base_config,
        processing_source_dir="/example/processing",
        processing_instance_type_large="ml.m5.12xlarge",
        processing_instance_type_small="ml.m5.4xlarge",
        processing_framework_version="1.2-1",
        processing_instance_count=1,
        processing_volume_size=500,
        use_large_processing_instance=False,
        processing_entry_point="processing_script.py"
    )


def export_config_to_dict(config_instance: BasePipelineConfig) -> Dict[str, Any]:
    """
    Export configuration instance to dictionary format.
    
    Args:
        config_instance: Configuration instance to export
        
    Returns:
        Dictionary representation of the configuration
        
    Example:
        >>> config = create_example_base_config()
        >>> config_dict = export_config_to_dict(config)
        >>> print(json.dumps(config_dict, indent=2))
    """
    logger.info(f"Exporting configuration to dict: {type(config_instance).__name__}")
    
    try:
        if hasattr(config_instance, 'model_dump'):
            # Pydantic v2
            return config_instance.model_dump()
        elif hasattr(config_instance, 'dict'):
            # Pydantic v1
            return config_instance.dict()
        else:
            # Fallback to __dict__
            return config_instance.__dict__
    except Exception as e:
        logger.error(f"Failed to export config to dict: {e}")
        return {"error": str(e), "config_type": type(config_instance).__name__}


def import_config_from_dict(config_dict: Dict[str, Any], 
                           config_class_name: str,
                           workspace_dirs: Optional[List[Union[str, Path]]] = None) -> BasePipelineConfig:
    """
    Import configuration from dictionary format.
    
    Args:
        config_dict: Dictionary representation of configuration
        config_class_name: Name of the configuration class
        workspace_dirs: Optional workspace directories for step catalog
        
    Returns:
        Configuration instance
        
    Example:
        >>> config_dict = {"author": "test", "bucket": "test-bucket", ...}
        >>> config = import_config_from_dict(config_dict, "BasePipelineConfig")
    """
    logger.info(f"Importing configuration from dict: {config_class_name}")
    
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    config_classes = core.discover_config_classes()
    
    if config_class_name not in config_classes:
        available_classes = list(config_classes.keys())
        raise ValueError(
            f"Configuration class '{config_class_name}' not found. "
            f"Available classes: {available_classes}"
        )
    
    config_class = config_classes[config_class_name]
    
    try:
        return config_class(**config_dict)
    except Exception as e:
        logger.error(f"Failed to import config from dict: {e}")
        raise ValueError(f"Failed to create {config_class_name} from dictionary: {e}")


# Utility constants
SUPPORTED_FIELD_TYPES = ["text", "number", "checkbox", "list", "keyvalue"]

DEFAULT_WORKSPACE_DIRS = [
    "src/cursus",
    "cursus",
    "."
]

# Version info
__version__ = "1.0.0"
