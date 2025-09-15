"""
Pipeline Catalog - Pipelines Module

This module contains all standard pipeline implementations organized in a flat structure
following Zettelkasten knowledge management principles.

Each pipeline is an atomic, independent unit with:
- Enhanced DAGMetadata integration
- Connection-based relationships
- Multi-dimensional tagging
- Registry synchronization
"""

from typing import Dict, List, Any
import importlib
import pkgutil
from pathlib import Path

# Pipeline registry for dynamic discovery
_PIPELINE_REGISTRY: Dict[str, Any] = {}


def register_pipeline(pipeline_id: str, pipeline_module: Any) -> None:
    """Register a pipeline in the local registry."""
    _PIPELINE_REGISTRY[pipeline_id] = pipeline_module


def get_registered_pipelines() -> Dict[str, Any]:
    """Get all registered pipelines."""
    return _PIPELINE_REGISTRY.copy()


def discover_pipelines() -> List[str]:
    """Discover all available pipeline modules."""
    pipeline_modules = []

    # Get the current package path
    package_path = Path(__file__).parent

    # Discover all Python files in the pipelines directory
    for file_path in package_path.glob("*.py"):
        if file_path.name != "__init__.py":
            module_name = file_path.stem
            pipeline_modules.append(module_name)

    return pipeline_modules


def load_pipeline(pipeline_id: str) -> Any:
    """Dynamically load a pipeline module."""
    try:
        module = importlib.import_module(
            f".{pipeline_id}", package=__name__
        )
        register_pipeline(pipeline_id, module)
        return module
    except ImportError as e:
        raise ImportError(f"Failed to load pipeline {pipeline_id}: {e}")


# Auto-discover and register pipelines on import
def _auto_register_pipelines():
    """Automatically register all available pipelines."""
    for pipeline_id in discover_pipelines():
        try:
            load_pipeline(pipeline_id)
        except ImportError:
            # Skip pipelines that can't be loaded
            pass


# Perform auto-registration
_auto_register_pipelines()

__all__ = [
    "register_pipeline",
    "get_registered_pipelines",
    "discover_pipelines",
    "load_pipeline",
]
