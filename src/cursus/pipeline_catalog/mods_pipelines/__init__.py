"""
Pipeline Catalog - MODS Pipelines Module

This module contains all MODS-compatible pipeline implementations organized in a flat structure
following Zettelkasten knowledge management principles.

MODS pipelines are atomic, independent units with:
- Enhanced DAGMetadata integration
- MODS compiler compatibility
- Connection-based relationships
- Multi-dimensional tagging
- Registry synchronization
"""

from typing import Dict, List, Any
import importlib
import pkgutil
from pathlib import Path

# MODS Pipeline registry for dynamic discovery
_MODS_PIPELINE_REGISTRY: Dict[str, Any] = {}

def register_mods_pipeline(pipeline_id: str, pipeline_module: Any) -> None:
    """Register a MODS pipeline in the local registry."""
    _MODS_PIPELINE_REGISTRY[pipeline_id] = pipeline_module

def get_registered_mods_pipelines() -> Dict[str, Any]:
    """Get all registered MODS pipelines."""
    return _MODS_PIPELINE_REGISTRY.copy()

def discover_mods_pipelines() -> List[str]:
    """Discover all available MODS pipeline modules."""
    pipeline_modules = []
    
    # Get the current package path
    package_path = Path(__file__).parent
    
    # Discover all Python files in the mods_pipelines directory
    for file_path in package_path.glob("*.py"):
        if file_path.name != "__init__.py":
            module_name = file_path.stem
            pipeline_modules.append(module_name)
    
    return pipeline_modules

def load_mods_pipeline(pipeline_id: str) -> Any:
    """Dynamically load a MODS pipeline module."""
    try:
        module = importlib.import_module(f"src.cursus.pipeline_catalog.mods_pipelines.{pipeline_id}")
        register_mods_pipeline(pipeline_id, module)
        return module
    except ImportError as e:
        raise ImportError(f"Failed to load MODS pipeline {pipeline_id}: {e}")

# Auto-discover and register MODS pipelines on import
def _auto_register_mods_pipelines():
    """Automatically register all available MODS pipelines."""
    for pipeline_id in discover_mods_pipelines():
        try:
            load_mods_pipeline(pipeline_id)
        except ImportError:
            # Skip pipelines that can't be loaded
            pass

# Perform auto-registration
_auto_register_mods_pipelines()

__all__ = [
    "register_mods_pipeline",
    "get_registered_mods_pipelines", 
    "discover_mods_pipelines",
    "load_mods_pipeline"
]
