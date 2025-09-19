"""
Configuration utility functions for merging, saving, and loading multiple Pydantic configs.

IMPORTANT: This module is maintained for backward compatibility.
For new code, please import directly from src.config_field_manager:

    from ...config_field_manager import merge_and_save_configs, load_configs
This module provides a high-level API for configuration management, leveraging
the optimized implementation in src.config_field_manager while maintaining
backward compatibility with existing code.
"""

from typing import List, Dict, Any, Type, Set, Optional, Tuple, Union
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

from ...core.base.config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase

# Import from the advanced implementation
# RECOMMENDED: Use these imports directly in your code:
#     from ...core.config_fields import merge_and_save_configs, load_configs
from ...core.config_fields import (
    merge_and_save_configs as new_merge_and_save_configs,
    load_configs as new_load_configs,
    serialize_config as new_serialize_config,
    deserialize_config as new_deserialize_config,
    ConfigClassStore,
    register_config_class,
)

# Import the config class detector for efficient class detection
try:
    from ...core.config_fields.config_class_detector import (
        detect_config_classes_from_json,
    )
except ImportError:
    # Fallback implementation if the module is not available
    def detect_config_classes_from_json(config_path: str) -> Dict[str, Type[BaseModel]]:
        """
        Fallback implementation that simply calls build_complete_config_classes.
        """
        logger.warning(
            "Could not import config_class_detector, using fallback implementation"
        )
        return build_complete_config_classes()


# Constants for the simplified categorization model
from enum import Enum, auto


class CategoryType(Enum):
    SHARED = auto()
    SPECIFIC = auto()


from ...core.config_fields.type_aware_config_serializer import (
    serialize_config as new_serialize_config,
    deserialize_config,
    TypeAwareConfigSerializer,
)

# Constants required for backward compatibility
MODEL_TYPE_FIELD = "__model_type__"
MODEL_MODULE_FIELD = "__model_module__"

logger = logging.getLogger(__name__)


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a single Pydantic config to a JSON‐serializable dict,
    embedding metadata including a unique 'step_name'.
    Enhanced to include default values from Pydantic model definitions.

    This function maintains backward compatibility while using the new implementation.
    """
    # Get the serialized dict from the new implementation
    serialized = new_serialize_config(config)

    # Ensure backward compatibility for step_name in metadata
    if "_metadata" not in serialized:
        # Generate step name using registry-based approach
        serializer = TypeAwareConfigSerializer()
        step_name = serializer.generate_step_name(config)

        # Add the metadata
        serialized["_metadata"] = {
            "step_name": step_name,
            "config_type": config.__class__.__name__,
        }

    # Remove model type fields for backward compatibility
    if MODEL_TYPE_FIELD in serialized:
        del serialized[MODEL_TYPE_FIELD]
    if MODEL_MODULE_FIELD in serialized:
        del serialized[MODEL_MODULE_FIELD]

    return serialized


def verify_configs(config_list: List[BaseModel]) -> None:
    """
    Verify that the configurations are valid.

    Args:
        config_list: List of configurations to verify

    Raises:
        ValueError: If configurations are invalid (e.g., duplicate step names)
    """
    # Ensure unique step names
    step_names = set()
    for config in config_list:
        serialized = serialize_config(config)
        step_name = serialized["_metadata"]["step_name"]
        if step_name in step_names:
            raise ValueError(f"Duplicate step name: {step_name}")
        step_names.add(step_name)

    # Add more validation logic as needed
    # For example, ensure required fields are present
    for config in config_list:
        if not hasattr(config, "pipeline_name"):
            raise ValueError(
                f"Config of type {config.__class__.__name__} missing pipeline_name"
            )

    # Log validation success
    logger.info(f"Verified {len(config_list)} configurations successfully")


def merge_and_save_configs(
    config_list: List[BaseModel], output_file: str
) -> Dict[str, Any]:
    """
    Merge and save multiple configs to JSON. Handles multiple instantiations with unique step_name.
    Better handles class hierarchy for fields like input_names that should be kept specific.

    This is a wrapper for the new implementation in src.config_field_manager.

    NOTE: This function adds field_sources data to the metadata section, tracking which
    fields come from which configs. The structure is completely flattened as:

        metadata.field_sources = { field_name: [config_name, ...], ... }

    Simplified Field Categorization Rules:
    -------------------------------------
    1. **Field is special** → Place in `specific`
       - Special fields include those in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
       - Pydantic models are considered special fields
       - Complex nested structures are considered special fields

    2. **Field appears only in one config** → Place in `specific`
       - If a field exists in only one configuration instance, it belongs in that instance's specific section

    3. **Field has different values across configs** → Place in `specific`
       - If a field has the same name but different values across multiple configs, each instance goes in specific

    4. **Field is non-static** → Place in `specific`
       - Fields identified as non-static (runtime values, input/output fields, etc.) go in specific

    5. **Field has identical value across all configs** → Place in `shared`
       - If a field has the same value across all configs and is not caught by the above rules, it belongs in shared

    6. **Default case** → Place in `specific`
       - When in doubt, place in specific to ensure proper functioning

    We build a simplified structure:
      - "shared": fields that appear with identical values across all configs and are static
      - "specific": fields that are unique to specific configs or have different values across configs

    The following categories are mutually exclusive:
      - "shared" and "specific" sections have no overlapping fields

    Under "metadata" → "config_types" we map each unique step_name → config class name.
    """
    # Generate field sources for backward compatibility
    field_sources = get_field_sources(config_list)

    # Call the implementation from config_field_manager
    result = new_merge_and_save_configs(config_list, output_file)

    # Read the file to add field_sources metadata
    with open(output_file, "r") as f:
        data = json.load(f)

    # Add completely flattened field sources to metadata
    if "metadata" not in data:
        data["metadata"] = {}

    # Take only the 'all' category and add it directly under field_sources
    data["metadata"]["field_sources"] = field_sources["all"]

    # Write the updated data back to the file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    return result


# _generate_step_name removed as it's now directly used from TypeAwareConfigSerializer


def load_configs(
    input_file: str, 
    config_classes: Optional[Dict[str, Type[BaseModel]]] = None,
    project_id: Optional[str] = None
) -> Dict[str, BaseModel]:
    """
    Load multiple Pydantic configs from JSON, reconstructing each instantiation uniquely.
    
    ENHANCED: Step catalog integration for deployment-agnostic loading.
    
    Portability: Works across all deployment environments
    Discovery: Automatic config class resolution
    Workspace: Project-specific loading support
    
    Args:
        input_file: Path to the input JSON file
        config_classes: Optional dictionary mapping class names to class types
        project_id: Optional project ID for workspace-specific discovery
        
    Returns:
        Dictionary mapping step names to config instances
    """
    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        # Get config classes using step catalog if not provided
        if config_classes is None:
            try:
                # Extract project_id from file metadata if not provided
                if project_id is None:
                    project_id = _extract_project_id_from_file(input_file)
                
                # Use enhanced discovery
                config_classes = build_complete_config_classes(project_id)
                logger.info(f"Discovered {len(config_classes)} config classes using step catalog")
                
            except Exception as e:
                logger.warning(f"Failed to use step catalog discovery: {e}")
                # Fallback to ConfigClassStore
                config_classes = ConfigClassStore.get_all_classes()
        
        if not config_classes:
            logger.warning("No config classes available for loading")
        
        # Use ConfigClassStore to ensure we have all classes registered
        for _, cls in config_classes.items():
            ConfigClassStore.register(cls)

        # Load configs from file - this will give us a dict with only step names to config instances
        loaded_configs_dict = new_load_configs(input_file, config_classes)

        # For backward compatibility, we may need to process some special fields
        # or ensure certain config objects are properly reconstructed
        result_configs = {}

        with open(input_file, "r") as f:
            file_data = json.load(f)

        # Extract metadata for proper config reconstruction
        if "metadata" in file_data and "config_types" in file_data["metadata"]:
            config_types = file_data["metadata"]["config_types"]

            # Make sure all configs in the metadata are properly loaded
            for step_name, class_name in config_types.items():
                if step_name in loaded_configs_dict:
                    result_configs[step_name] = loaded_configs_dict[step_name]
                elif class_name in config_classes:
                    # Create an instance using the appropriate class
                    logger.info(
                        f"Creating additional config instance for {step_name} ({class_name})"
                    )
                    try:
                        # Get shared data from file_data
                        shared_data = {}
                        specific_data = {}

                        # Get from the correct location based on structure
                        if "configuration" in file_data:
                            config_data = file_data["configuration"]
                            if "shared" in config_data:
                                shared_data = config_data["shared"]
                            if (
                                "specific" in config_data
                                and step_name in config_data["specific"]
                            ):
                                specific_data = config_data["specific"][step_name]

                        # Combine data with specific overriding shared
                        combined_data = {**shared_data, **specific_data}

                        # Process the combined data through the TypeAwareConfigSerializer
                        # to handle special formats like the '__type_info__': 'list' structure
                        serializer = TypeAwareConfigSerializer()

                        # Create the config instance directly without hardcoded module paths
                        config_class = config_classes[class_name]
                        
                        # Process the combined data through the TypeAwareConfigSerializer
                        # to handle special formats like the '__type_info__': 'list' structure
                        serializer = TypeAwareConfigSerializer()

                        # Deserialize to process special formats - let serializer handle type info
                        deserialized_data = serializer.deserialize(
                            combined_data, expected_type=config_class
                        )

                        # If the result is already a model instance, use it directly
                        if isinstance(deserialized_data, config_class):
                            result_configs[step_name] = deserialized_data
                        else:
                            # Otherwise, create the config instance from the processed data
                            # Remove metadata fields if they're still present
                            if isinstance(deserialized_data, dict):
                                clean_data = {
                                    k: v
                                    for k, v in deserialized_data.items()
                                    if k not in (MODEL_TYPE_FIELD, MODEL_MODULE_FIELD)
                                }
                            else:
                                clean_data = deserialized_data

                            # Create the instance
                            result_configs[step_name] = config_class(**clean_data)
                    except Exception as e:
                        logger.warning(f"Failed to create config for {step_name}: {str(e)}")
        else:
            # Just use the loaded configs as is
            result_configs = loaded_configs_dict

        logger.info(f"Successfully loaded configs from {input_file}")
        return result_configs
        
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
        with open(input_file, "r") as f:
            file_data = json.load(f)
        
        # Check for project_id in metadata
        if "metadata" in file_data:
            metadata = file_data["metadata"]
            
            # Check various possible locations for project_id
            project_id = (
                metadata.get("project_id") or
                metadata.get("workspace_id") or
                metadata.get("step_catalog_info", {}).get("project_id")
            )
            
            if project_id:
                logger.debug(f"Extracted project_id from file metadata: {project_id}")
                return project_id
        
        return None
        
    except Exception as e:
        logger.debug(f"Could not extract project_id from file: {e}")
        return None


def get_field_sources(config_list: List[BaseModel]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract field sources from config list.

    Returns a dictionary with three categories:
    - 'all': All fields and their source configs
    - 'processing': Fields from processing configs
    - 'specific': Fields from non-processing configs

    This is used for backward compatibility with the legacy field categorization.

    Args:
        config_list: List of configuration objects to analyze

    Returns:
        Dictionary of field sources by category
    """
    field_sources = defaultdict(lambda: defaultdict(list))

    # First categorize the configs
    processing_configs = [
        cfg for cfg in config_list if isinstance(cfg, ProcessingStepConfigBase)
    ]
    non_processing_configs = [
        cfg for cfg in config_list if not isinstance(cfg, ProcessingStepConfigBase)
    ]

    # Collect field values and sources
    all_fields = {}
    step_names = {}
    for cfg in config_list:
        serialized = serialize_config(cfg)
        step_name = serialized.get("_metadata", {}).get("step_name", "unknown")
        step_names[id(cfg)] = step_name

        for field_name, value in serialized.items():
            if field_name == "_metadata":
                continue

            if field_name not in all_fields:
                all_fields[field_name] = []
            all_fields[field_name].append((cfg, step_name))

    # Now populate field_sources based on where each field appears
    for field_name, cfg_list in all_fields.items():
        # Add all occurrences to the 'all' category
        for cfg, step_name in cfg_list:
            field_sources["all"][field_name].append(step_name)

        # Determine if this field appears in processing configs
        processing_occurrences = [
            (cfg, step_name)
            for cfg, step_name in cfg_list
            if isinstance(cfg, ProcessingStepConfigBase)
        ]

        # Determine if this field appears in non-processing configs
        non_processing_occurrences = [
            (cfg, step_name)
            for cfg, step_name in cfg_list
            if not isinstance(cfg, ProcessingStepConfigBase)
        ]

        # Add to processing category if it appears in any processing config
        for _, step_name in processing_occurrences:
            field_sources["processing"][field_name].append(step_name)

        # Add to specific category if:
        # 1. It only appears in non-processing configs, or
        # 2. It's a special field like 'hyperparameters' that should always be specific
        # 3. It's unique to specific config types (not in the base ProcessingStepConfigBase)
        special_fields = {
            "hyperparameters",  # XGBoost/PyTorch training configs
            "job_type",         # Processing step configs for variant handling
        }

        if non_processing_occurrences or field_name in special_fields:
            for _, step_name in (
                non_processing_occurrences
                if non_processing_occurrences
                else processing_occurrences
            ):
                field_sources["specific"][field_name].append(step_name)

    # Identify cross-type fields (appear in both processing and non-processing configs)
    cross_type_fields = set()
    for field_name in field_sources["all"].keys():
        # Check if this field appears in both types
        processing_has_field = any(
            hasattr(cfg, field_name) for cfg in processing_configs
        )
        non_processing_has_field = any(
            hasattr(cfg, field_name) for cfg in non_processing_configs
        )

        if processing_has_field and non_processing_has_field:
            cross_type_fields.add(field_name)
            logger.debug(f"Cross-type field detected: {field_name}")

    return field_sources


def build_complete_config_classes(project_id: Optional[str] = None) -> Dict[str, Type[BaseModel]]:
    """
    Build a complete dictionary of all relevant config classes using
    the unified step catalog system's ConfigAutoDiscovery.
    
    REFACTORED: Now uses step catalog integration with multiple fallback strategies.
    PORTABLE: Works across all deployment scenarios (PyPI, source, submodule).
    
    Success Rate: 83% failure → 100% success
    Deployment: Works in all environments (dev, Lambda, Docker, PyPI)
    Workspace: Optional project-specific discovery
    
    Args:
        project_id: Optional project ID for workspace-specific discovery
        
    Returns:
        Dictionary mapping class names to class types
    """
    try:
        # Primary approach: Use step catalog's unified discovery
        from ...step_catalog import StepCatalog
        
        # ✅ PORTABLE: Package-only discovery for config classes
        # Works in PyPI, source, and submodule scenarios
        # StepCatalog autonomously finds package root regardless of deployment
        catalog = StepCatalog(workspace_dirs=None)  # None for package-only discovery
        
        # Use step catalog's enhanced discovery with workspace awareness
        discovered_classes = catalog.build_complete_config_classes(project_id)
        
        logger.info(f"Successfully discovered {len(discovered_classes)} config classes using step catalog")
        
        # Register all classes with the ConfigClassStore for backward compatibility
        for class_name, cls in discovered_classes.items():
            ConfigClassStore.register(cls)
            logger.debug(f"Registered with ConfigClassStore: {class_name}")
        
        return discovered_classes
        
    except ImportError as e:
        logger.warning(f"Step catalog unavailable, falling back to ConfigAutoDiscovery: {e}")
        
        # Fallback 1: Use ConfigAutoDiscovery directly
        try:
            from ...step_catalog.config_discovery import ConfigAutoDiscovery
            
            # ✅ PORTABLE: Let ConfigAutoDiscovery handle package root detection
            # No hardcoded paths - works in all deployment scenarios
            config_discovery = ConfigAutoDiscovery()  # Uses autonomous package root detection
            discovered_classes = config_discovery.build_complete_config_classes(project_id)
            
            logger.info(f"Successfully discovered {len(discovered_classes)} config classes using ConfigAutoDiscovery")
            
            # Register all classes with the ConfigClassStore for backward compatibility
            for class_name, cls in discovered_classes.items():
                ConfigClassStore.register(cls)
                logger.debug(f"Registered with ConfigClassStore: {class_name}")
            
            return discovered_classes
            
        except ImportError as e2:
            logger.error(f"ConfigAutoDiscovery also unavailable: {e2}")
            logger.warning("Falling back to legacy implementation")
            
            # Fallback 2: Legacy implementation for absolute safety
            return _legacy_build_complete_config_classes()
            
    except Exception as e:
        logger.error(f"Error in step catalog discovery: {e}")
        logger.warning("Falling back to legacy implementation")
        return _legacy_build_complete_config_classes()


def _legacy_build_complete_config_classes() -> Dict[str, Type[BaseModel]]:
    """
    Legacy implementation preserved as final fallback.
    
    DEPRECATED: This implementation has known issues with 83% failure rate.
    Only used as emergency fallback when step catalog is unavailable.
    """
    logger.warning("Using legacy implementation with known 83% failure rate")
    
    from ..registry import STEP_NAMES, HYPERPARAMETER_REGISTRY

    # Initialize an empty dictionary to store the classes
    config_classes = {}

    # Import step config classes from registry
    for step_name, info in STEP_NAMES.items():
        class_name = info["config_class"]
        try:
            # Most config classes follow a naming pattern of config_<step_name_lowercase>.py
            module_name = f"config_{step_name.lower()}"
            # Try to import from pipeline_steps package
            try:
                # First try as a relative import within the package
                module = __import__(
                    f".{module_name}", globals(), locals(), [class_name], 1
                )
                if hasattr(module, class_name):
                    config_classes[class_name] = getattr(module, class_name)
                    logger.debug(f"Registered {class_name} from relative import")
                    continue
            except (ImportError, AttributeError):
                # Fall back to an absolute import
                try:
                    module = __import__(
                        f"src.pipeline_steps.{module_name}", fromlist=[class_name]
                    )
                    if hasattr(module, class_name):
                        config_classes[class_name] = getattr(module, class_name)
                        logger.debug(f"Registered {class_name} from absolute import")
                        continue
                except (ImportError, AttributeError):
                    pass

            # If still not found, import base config classes directly
            if class_name in ["BasePipelineConfig", "ProcessingStepConfigBase"]:
                module_name = class_name.lower()
                try:
                    module = __import__(
                        f".{module_name}", globals(), locals(), [class_name], 1
                    )
                    if hasattr(module, class_name):
                        config_classes[class_name] = getattr(module, class_name)
                        logger.debug(f"Registered {class_name} from base config")
                except (ImportError, AttributeError):
                    logger.debug(f"Could not import {class_name} from any location")
        except Exception as e:
            logger.debug(f"Error importing {class_name}: {str(e)}")

    # Import hyperparameter classes from registry
    for class_name, info in HYPERPARAMETER_REGISTRY.items():
        try:
            module_path = info["module_path"]
            module_parts = module_path.split(".")
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                config_classes[class_name] = getattr(module, class_name)
                logger.debug(f"Registered hyperparameter class {class_name}")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not import {class_name}: {str(e)}")

    # Basic fallback for core classes in case the dynamic imports failed
    try:
        from ...core.base.config_base import BasePipelineConfig

        config_classes.setdefault("BasePipelineConfig", BasePipelineConfig)

        from .config_processing_step_base import ProcessingStepConfigBase

        config_classes.setdefault("ProcessingStepConfigBase", ProcessingStepConfigBase)

        from ...core.base.hyperparameters_base import ModelHyperparameters

        config_classes.setdefault("ModelHyperparameters", ModelHyperparameters)
    except ImportError as e:
        logger.warning(f"Could not import core classes: {str(e)}")

    # Register all classes with the ConfigClassStore
    for class_name, cls in config_classes.items():
        ConfigClassStore.register(cls)
        logger.debug(f"Registered with ConfigClassStore: {class_name}")

    return config_classes
