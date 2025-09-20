"""
Pipeline Catalog - MODS Pipelines Module

This module provides MODS-enhanced pipeline functionality using the new API approach
that eliminates code duplication between regular pipelines and MODS pipelines.

Instead of maintaining separate MODS pipeline implementations, this module now:
- Imports regular pipelines from pipeline_catalog.pipelines
- Applies MODS decorators using the mods_api
- Extracts metadata from configuration automatically
- Maintains backward compatibility

Key Components:
- MODS API: Converts regular pipelines to MODS-enhanced versions
- Configuration-based metadata extraction
- Dynamic pipeline creation and registration
- Backward compatibility with existing MODS pipeline interfaces

Example Usage:
    ```python
    from cursus.pipeline_catalog.mods_pipelines import create_mods_pipeline_from_config
    from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
    
    # Create MODS-enhanced pipeline
    MODSPipeline = create_mods_pipeline_from_config(
        XGBoostE2EComprehensivePipeline,
        config_path="config.json"
    )
    
    # Use like any other pipeline
    pipeline_instance = MODSPipeline(
        config_path="config.json",
        sagemaker_session=session,
        execution_role=role
    )
    pipeline = pipeline_instance.generate_pipeline()
    ```
"""

from typing import Dict, List, Any, Type, Optional
import importlib
from pathlib import Path

# Import the MODS API
from ..mods_api import (
    create_mods_pipeline,
    create_mods_pipeline_from_config,
    create_mods_pipeline_by_name,
    get_mods_pipeline_factory,
    PIPELINE_REGISTRY,
    # Convenience functions
    create_mods_xgboost_e2e_comprehensive,
    create_mods_pytorch_e2e_standard,
    create_mods_dummy_e2e_basic,
)

# MODS Pipeline registry for dynamic discovery
_MODS_PIPELINE_REGISTRY: Dict[str, Any] = {}


def register_mods_pipeline(pipeline_id: str, pipeline_module: Any) -> None:
    """Register a MODS pipeline in the local registry."""
    _MODS_PIPELINE_REGISTRY[pipeline_id] = pipeline_module


def get_registered_mods_pipelines() -> Dict[str, Any]:
    """Get all registered MODS pipelines."""
    return _MODS_PIPELINE_REGISTRY.copy()


def discover_mods_pipelines() -> List[str]:
    """
    Discover all available MODS pipeline modules.
    
    Note: With the new API approach, MODS pipelines are created dynamically
    from regular pipelines, so this now returns available pipeline types
    from the PIPELINE_REGISTRY.
    """
    return list(PIPELINE_REGISTRY.keys())


def load_mods_pipeline(pipeline_id: str) -> Any:
    """
    Dynamically load a MODS pipeline module.
    
    With the new API approach, this creates a MODS-enhanced pipeline
    from the corresponding regular pipeline.
    """
    try:
        # Check if it's a known pipeline type
        if pipeline_id in PIPELINE_REGISTRY:
            # Create MODS pipeline using the API
            mods_pipeline_class = create_mods_pipeline_by_name(pipeline_id)
            register_mods_pipeline(pipeline_id, mods_pipeline_class)
            return mods_pipeline_class
        else:
            # Try to load as a module (for backward compatibility)
            module = importlib.import_module(
                f".{pipeline_id}", package=__name__
            )
            register_mods_pipeline(pipeline_id, module)
            return module
    except (ImportError, ValueError) as e:
        raise ImportError(f"Failed to load MODS pipeline {pipeline_id}: {e}")


def create_all_mods_pipelines(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    pipeline_version: Optional[str] = None
) -> Dict[str, Type]:
    """
    Create MODS-enhanced versions of all available pipelines.
    
    Args:
        config_path: Path to configuration file
        config_dict: Configuration dictionary
        author: Override author from config
        pipeline_description: Override description from config
        pipeline_version: Override version from config
        
    Returns:
        Dict mapping pipeline names to MODS-enhanced pipeline classes
        
    Example:
        ```python
        # Create all MODS pipelines from config
        mods_pipelines = create_all_mods_pipelines(config_path="config.json")
        
        # Use a specific pipeline
        XGBoostMODS = mods_pipelines['xgb_e2e_comprehensive']
        pipeline_instance = XGBoostMODS(
            config_path="config.json",
            sagemaker_session=session,
            execution_role=role
        )
        ```
    """
    mods_pipelines = {}
    
    for pipeline_name in PIPELINE_REGISTRY.keys():
        try:
            mods_pipeline_class = create_mods_pipeline_by_name(
                pipeline_name=pipeline_name,
                config_path=config_path,
                config_dict=config_dict,
                author=author,
                pipeline_description=pipeline_description,
                pipeline_version=pipeline_version
            )
            mods_pipelines[pipeline_name] = mods_pipeline_class
            register_mods_pipeline(pipeline_name, mods_pipeline_class)
        except Exception as e:
            # Skip pipelines that can't be created
            print(f"Warning: Could not create MODS pipeline for {pipeline_name}: {e}")
            continue
    
    return mods_pipelines


# Auto-discover and register MODS pipelines on import
def _auto_register_mods_pipelines():
    """
    Automatically register all available MODS pipelines.
    
    With the new API approach, this registers the available pipeline types
    rather than loading individual modules.
    """
    for pipeline_id in discover_mods_pipelines():
        try:
            # Register the pipeline type (not the instance)
            _MODS_PIPELINE_REGISTRY[pipeline_id] = pipeline_id
        except Exception:
            # Skip pipelines that can't be registered
            pass


# Perform auto-registration
_auto_register_mods_pipelines()

# Export the main API functions
__all__ = [
    # Core MODS API
    "create_mods_pipeline",
    "create_mods_pipeline_from_config", 
    "create_mods_pipeline_by_name",
    "get_mods_pipeline_factory",
    
    # Convenience functions
    "create_mods_xgboost_e2e_comprehensive",
    "create_mods_pytorch_e2e_standard", 
    "create_mods_dummy_e2e_basic",
    
    # Registry functions
    "register_mods_pipeline",
    "get_registered_mods_pipelines",
    "discover_mods_pipelines",
    "load_mods_pipeline",
    "create_all_mods_pipelines",
    
    # Constants
    "PIPELINE_REGISTRY",
]
