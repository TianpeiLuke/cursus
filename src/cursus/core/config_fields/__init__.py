"""
Configuration Field Manager Package.

This package provides robust tools for managing configuration fields, including:
- Field categorization for configuration organization
- Type-aware serialization and deserialization
- Configuration class registration
- Configuration merging and loading
- Three-tier configuration architecture components

Primary API functions:
- merge_and_save_configs: Merge and save multiple config objects to a unified JSON file
- load_configs: Load config objects from a saved JSON file
- serialize_config: Convert a config object to a JSON-serializable dict with type metadata
- deserialize_config: Convert a serialized dict back to a config object

New Three-Tier Architecture Components:
- ConfigFieldTierRegistry: Registry for field tier classifications (Tier 1, 2, 3)
- DefaultValuesProvider: Provider for default values (Tier 2)
- FieldDerivationEngine: Engine for deriving field values (Tier 3)
- Essential Input Models: Pydantic models for Data, Model, and Registration configurations

Usage:
    ```python
    from ..config_field_manager import merge_and_save_configs, load_configs, ConfigClassStore    
    # Register config classes for type-aware serialization
    @ConfigClassStore.register
    class MyConfig:
        ...
        
    # Merge and save configs
    configs = [MyConfig(...), AnotherConfig(...)]
    merge_and_save_configs(configs, "output.json")
    
    # Load configs
    loaded_configs = load_configs("output.json")
    
    # Using the three-tier architecture
    from ..config_field_manager import (        ConfigFieldTierRegistry, DefaultValuesProvider, 
        FieldDerivationEngine, DataConfig, ModelConfig, RegistrationConfig
    )
    
    # Apply defaults and derive fields
    DefaultValuesProvider.apply_defaults(config)
    field_engine = FieldDerivationEngine()
    field_engine.derive_fields(config)
    ```
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Type, Union, Tuple, Set
from pathlib import Path

from .config_merger import ConfigMerger
from .type_aware_config_serializer import (
    TypeAwareConfigSerializer,
    serialize_config as _serialize_config,
    deserialize_config as _deserialize_config,
)
from .config_field_categorizer import ConfigFieldCategorizer
from .circular_reference_tracker import CircularReferenceTracker
from .tier_registry import ConfigFieldTierRegistry

# Import step catalog adapters for config class functionality
try:
    from ...step_catalog.adapters.config_class_detector import (
        ConfigClassStoreAdapter as ConfigClassStore,
        ConfigClassDetectorAdapter as ConfigClassDetector,
        detect_config_classes_from_json,
        build_complete_config_classes,
    )
except ImportError:
    # Fallback for environments where step catalog is not available
    class ConfigClassStore:
        """Fallback ConfigClassStore for environments without step catalog."""
        _classes = {}
        
        @classmethod
        def register(cls, config_class):
            cls._classes[config_class.__name__] = config_class
            return config_class
        
        @classmethod
        def get_all_classes(cls):
            return cls._classes.copy()
    
    # Fallback ConfigClassDetector
    class ConfigClassDetector:
        """Fallback ConfigClassDetector for environments without step catalog."""
        MODEL_TYPE_FIELD = "__model_type__"
        METADATA_FIELD = "metadata"
        CONFIG_TYPES_FIELD = "config_types"
        CONFIGURATION_FIELD = "configuration"
        SPECIFIC_FIELD = "specific"
        
        @classmethod
        def detect_from_json(cls, config_file_path: str):
            return ConfigClassStore.get_all_classes()
        
        @classmethod
        def from_config_store(cls, config_file_path: str):
            return cls.detect_from_json(config_file_path)
        
        @classmethod
        def _extract_class_names(cls, data, logger):
            return set()
    
    def detect_config_classes_from_json(config_file_path: str):
        """Fallback function for detecting config classes from JSON."""
        return ConfigClassDetector.detect_from_json(config_file_path)
    
    def build_complete_config_classes():
        """Fallback function for building complete config classes."""
        return ConfigClassStore.get_all_classes()

# Import below modules when they are available
# from .default_values_provider import DefaultValuesProvider
# from .field_derivation_engine import FieldDerivationEngine
# from .essential_input_models import (
#     DataConfig,
#     ModelConfig,
#     RegistrationConfig,
#     EssentialInputs
# )


__all__ = [
    # Original exports
    "merge_and_save_configs",
    "load_configs",
    "serialize_config",
    "deserialize_config",
    "ConfigClassStore",  # Export for use as a decorator
    "register_config_class",  # Convenient alias for the decorator
    "CircularReferenceTracker",  # For advanced circular reference handling
    # Three-tier architecture components
    "ConfigFieldTierRegistry",
    # Config class detection functionality
    "ConfigClassDetector",
    "detect_config_classes_from_json",
    "build_complete_config_classes",
    # The following modules are not currently available
    # 'DefaultValuesProvider',
    # 'FieldDerivationEngine',
    # 'DataConfig',
    # 'ModelConfig',
    # 'RegistrationConfig',
    # 'EssentialInputs'
]


# Create logger
logger = logging.getLogger(__name__)


def merge_and_save_configs(
    config_list: List[Any],
    output_file: str,
    processing_step_config_base_class: Optional[type] = None,
    project_id: Optional[str] = None,  # NEW: Workspace awareness
    step_catalog: Optional[Any] = None,  # NEW: Step catalog integration
    enhanced_metadata: bool = False,  # NEW: Enhanced metadata option
) -> Dict[str, Any]:
    """
    Merge and save multiple configs to a single JSON file.
    
    ENHANCED: Workspace-aware merging with step catalog integration.
    
    Backward Compatible: Original signature preserved
    Enhanced: Optional workspace and step catalog parameters
    Format: Exact same JSON output structure maintained

    This function uses the ConfigFieldCategorizer to analyze fields across all configurations,
    organizing them into shared and specific sections based on values and usage patterns.

    The output follows the simplified structure:
    ```
    {
      "metadata": {
        "created_at": "ISO timestamp",
        "config_types": {
          "StepName1": "ConfigClassName1",
          "StepName2": "ConfigClassName2",
          ...
        },
        "project_id": "project_id",  # NEW: If provided
        "step_catalog_info": {...},  # NEW: If step_catalog provided
        "framework_version": "...",  # NEW: If enhanced_metadata enabled
      },
      "configuration": {
        "shared": {
          "common_field1": "common_value1",
          ...
        },
        "specific": {
          "StepName1": {
            "specific_field1": "specific_value1",
            ...
          },
          "StepName2": {
            "specific_field2": "specific_value2",
            ...
          }
        }
      }
    }
    ```

    Args:
        config_list: List of configuration objects to merge and save
        output_file: Path to the output JSON file
        processing_step_config_base_class: Optional base class to identify processing step configs
        project_id: Optional project ID for workspace-aware processing
        step_catalog: Optional step catalog instance for enhanced processing
        enhanced_metadata: Whether to include enhanced metadata (framework info, etc.)

    Returns:
        dict: The merged configuration

    Raises:
        ValueError: If config_list is empty or contains invalid configs
        IOError: If there's an issue writing to the output file
        TypeError: If configs are not serializable
    """
    # Validate inputs
    if not config_list:
        raise ValueError("Config list cannot be empty")

    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create enhanced merger if step catalog provided
        if step_catalog is not None:
            try:
                from .step_catalog_aware_merger import StepCatalogAwareConfigMerger
                logger.info(f"Using step catalog aware merger for {len(config_list)} configs")
                merger = StepCatalogAwareConfigMerger(
                    config_list, 
                    processing_step_config_base_class,
                    step_catalog=step_catalog,
                    project_id=project_id
                )
            except ImportError:
                logger.warning("StepCatalogAwareConfigMerger not available, using standard merger")
                merger = ConfigMerger(config_list, processing_step_config_base_class)
        else:
            # Use standard merger for backward compatibility
            merger = ConfigMerger(config_list, processing_step_config_base_class)

        # Save configs with enhanced options
        logger.info(f"Merging and saving {len(config_list)} configs to {output_file}")
        if hasattr(merger, 'save_with_enhanced_metadata') and (project_id or enhanced_metadata):
            merged = merger.save_with_enhanced_metadata(
                output_file, 
                project_id=project_id,
                enhanced_metadata=enhanced_metadata
            )
        else:
            merged = merger.save(output_file)
        
        # Add enhanced metadata if requested and not already added
        if enhanced_metadata and 'metadata' in merged:
            _add_enhanced_metadata(merged['metadata'], project_id, step_catalog)
        
        logger.info(f"Successfully saved merged configs to {output_file}")
        return merged
        
    except Exception as e:
        logger.error(f"Error merging and saving configs: {str(e)}")
        raise


def _add_enhanced_metadata(metadata: Dict[str, Any], project_id: Optional[str], step_catalog: Optional[Any]) -> None:
    """Add enhanced metadata to the configuration metadata section."""
    try:
        # Add project information
        if project_id:
            metadata['project_id'] = project_id
        
        # Add step catalog information
        if step_catalog:
            try:
                metadata['step_catalog_info'] = {
                    'catalog_type': type(step_catalog).__name__,
                    'workspace_root': str(getattr(step_catalog, 'workspace_root', 'unknown')),
                    'discovery_method': 'step_catalog_integration'
                }
            except Exception as e:
                logger.debug(f"Could not add step catalog info: {e}")
        
        # Add framework version information
        try:
            import cursus
            metadata['framework_version'] = getattr(cursus, '__version__', 'unknown')
        except ImportError:
            metadata['framework_version'] = 'unknown'
        
        # Add enhanced processing timestamp
        from datetime import datetime
        metadata['enhanced_processing_timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
    except Exception as e:
        logger.debug(f"Could not add enhanced metadata: {e}")


def load_configs(
    input_file: str, 
    config_classes: Optional[Dict[str, Type]] = None,
    project_id: Optional[str] = None,  # NEW: Workspace awareness
    auto_detect_project: bool = True,  # NEW: Automatic project detection
    enhanced_discovery: bool = True,  # NEW: Enhanced discovery with step catalog
) -> Dict[str, Any]:
    """
    Load multiple configs from a JSON file.
    
    ENHANCED: Workspace-aware loading with automatic project detection.
    
    Auto-Detection: Extract project_id from file metadata
    Workspace: Project-specific config class discovery
    Fallbacks: Multiple discovery strategies with graceful degradation

    This function loads configurations from a JSON file that was previously saved using
    merge_and_save_configs. It reconstructs the configuration objects based on the
    type information stored in the file, using the simplified structure with shared
    and specific fields.

    Args:
        input_file: Path to the input JSON file
        config_classes: Optional dictionary mapping class names to class types
                       If not provided, enhanced discovery will be used
        project_id: Optional project ID for workspace-specific loading
        auto_detect_project: Whether to auto-detect project_id from file metadata
        enhanced_discovery: Whether to use enhanced discovery with step catalog integration

    Returns:
        dict: A dictionary with the following structure:
            {
                "shared": {shared_field1: value1, ...},
                "specific": {
                    "StepName1": {specific_field1: value1, ...},
                    "StepName2": {specific_field2: value2, ...},
                    ...
                }
            }

    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the input file is not valid JSON
        KeyError: If required keys are missing from the file
        TypeError: If deserialization fails due to type mismatches
    """
    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        # Auto-detect project_id from file metadata if enabled and not provided
        detected_project_id = project_id
        if auto_detect_project and not project_id:
            detected_project_id = _extract_project_id_from_file(input_file)
            if detected_project_id:
                logger.info(f"Auto-detected project_id: {detected_project_id}")

        # Get config classes using enhanced discovery if enabled
        if config_classes is None and enhanced_discovery:
            all_config_classes = _get_enhanced_config_classes(detected_project_id)
        else:
            # Fallback to ConfigClassStore or provided classes
            all_config_classes = config_classes or ConfigClassStore.get_all_classes()

        if not all_config_classes:
            logger.warning(
                "No config classes available - trying legacy discovery as fallback"
            )
            # Final fallback to basic discovery
            all_config_classes = _get_basic_config_classes()

        # Use enhanced loader if available
        if enhanced_discovery:
            try:
                from .enhanced_config_loader import EnhancedConfigLoader
                loader = EnhancedConfigLoader(
                    config_classes=all_config_classes,
                    project_id=detected_project_id
                )
                logger.info(f"Using enhanced config loader for {input_file}")
                loaded_configs = loader.load(input_file)
            except ImportError:
                logger.debug("EnhancedConfigLoader not available, using standard loader")
                loaded_configs = ConfigMerger.load(input_file, all_config_classes)
        else:
            # Use standard ConfigMerger for backward compatibility
            loaded_configs = ConfigMerger.load(input_file, all_config_classes)

        logger.info(
            f"Successfully loaded configs from {input_file} with {len(loaded_configs.get('specific', {}))} specific configs"
        )

        return loaded_configs
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Missing required key in input file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading configs: {str(e)}")
        raise


def _extract_project_id_from_file(input_file: str) -> Optional[str]:
    """
    Extract project_id from file metadata if available.
    
    Args:
        input_file: Path to the config file
        
    Returns:
        Project ID if found in metadata, None otherwise
    """
    try:
        with open(input_file, 'r') as f:
            file_data = json.load(f)
        
        # Check for project_id in metadata
        if 'metadata' in file_data:
            metadata = file_data['metadata']
            
            # Check various possible locations for project_id
            project_id = (
                metadata.get('project_id') or
                metadata.get('workspace_id') or
                metadata.get('step_catalog_info', {}).get('project_id')
            )
            
            if project_id:
                logger.debug(f"Extracted project_id from file metadata: {project_id}")
                return project_id
        
        return None
        
    except Exception as e:
        logger.debug(f"Could not extract project_id from file: {e}")
        return None


def _get_enhanced_config_classes(project_id: Optional[str] = None) -> Dict[str, Type]:
    """
    Get config classes using enhanced discovery with step catalog integration.
    
    Args:
        project_id: Optional project ID for workspace-specific discovery
        
    Returns:
        Dictionary mapping class names to class types
    """
    try:
        # Try to use unified config manager for enhanced discovery
        from .unified_config_manager import get_unified_config_manager
        manager = get_unified_config_manager()
        config_classes = manager.get_config_classes(project_id)
        
        if config_classes:
            logger.info(f"Enhanced discovery found {len(config_classes)} config classes")
            return config_classes
            
    except ImportError:
        logger.debug("UnifiedConfigManager not available")
    except Exception as e:
        logger.debug(f"Enhanced discovery failed: {e}")
    
    # Fallback to step catalog integration from utils
    try:
        from ...steps.configs.utils import build_complete_config_classes
        config_classes = build_complete_config_classes(project_id)
        
        if config_classes:
            logger.info(f"Step catalog discovery found {len(config_classes)} config classes")
            return config_classes
            
    except ImportError:
        logger.debug("Step catalog utils not available")
    except Exception as e:
        logger.debug(f"Step catalog discovery failed: {e}")
    
    # Final fallback to ConfigClassStore
    return ConfigClassStore.get_all_classes()


def _get_basic_config_classes() -> Dict[str, Type]:
    """
    Get basic config classes as final fallback.
    
    Returns:
        Dictionary with basic config classes
    """
    try:
        from ...core.base.config_base import BasePipelineConfig
        from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
        from ...core.base.hyperparameters_base import ModelHyperparameters
        
        return {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "ModelHyperparameters": ModelHyperparameters,
        }
    except ImportError as e:
        logger.error(f"Could not import basic config classes: {e}")
        return {}


def serialize_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a configuration object to a JSON-serializable dictionary.

    This function serializes a configuration object, preserving its type information
    and special fields. It embeds metadata including the step name derived from
    attributes like 'job_type', 'data_type', and 'mode'.

    Args:
        config: The configuration object to serialize

    Returns:
        dict: A serialized representation of the config

    Raises:
        TypeError: If the config is not serializable
    """
    try:
        return _serialize_config(config)
    except Exception as e:
        logger.error(f"Error serializing config: {str(e)}")
        raise TypeError(
            f"Failed to serialize config of type {type(config).__name__}: {str(e)}"
        )


def deserialize_config(
    data: Dict[str, Any], config_classes: Optional[Dict[str, Type]] = None
) -> Any:
    """
    Deserialize a dictionary back into a configuration object.

    This function deserializes a dictionary into a configuration object based on
    type information embedded in the dictionary. If the dictionary contains the
    __model_type__ field, it will attempt to reconstruct
    the original object type using the step catalog system.

    Args:
        data: The serialized dictionary
        config_classes: Optional dictionary mapping class names to class types
                       If not provided, all classes registered with ConfigClassStore will be used

    Returns:
        Any: The deserialized configuration object

    Raises:
        TypeError: If the data cannot be deserialized to the specified type
    """
    # Get config classes from store or use provided ones
    all_config_classes = config_classes or ConfigClassStore.get_all_classes()

    try:
        serializer = TypeAwareConfigSerializer(all_config_classes)
        return serializer.deserialize(data)
    except Exception as e:
        logger.error(f"Error deserializing config: {str(e)}")
        raise TypeError(f"Failed to deserialize config: {str(e)}")


# Convenient alias for the ConfigClassStore.register decorator
def register_config_class(cls: Any) -> Any:
    """
    Register a configuration class with the ConfigClassStore.

    This is a convenient alias for ConfigClassStore.register decorator.

    Args:
        cls: The class to register

    Returns:
        The class, allowing this to be used as a decorator
    """
    return ConfigClassStore.register(cls)
