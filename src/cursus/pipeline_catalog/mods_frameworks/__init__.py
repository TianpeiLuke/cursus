"""
MODS Pipeline Frameworks

This module contains MODS-specific pipeline implementations that use the
MODSPipelineDAGCompiler instead of the standard PipelineDAGCompiler.

MODS (Model Operations Data Science) pipelines provide enhanced functionality
including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

All MODS pipelines use the same shared DAG definitions as their standard
counterparts, ensuring consistency while providing MODS-specific features.

Directory Structure:
- xgboost/: MODS XGBoost pipeline variants
- pytorch/: MODS PyTorch pipeline variants

Usage:
    from cursus.pipeline_catalog.mods_frameworks.xgboost.simple_mods import create_pipeline
    
    # Create MODS pipeline (automatically registers with MODS global registry)
    pipeline, report, dag_compiler = create_pipeline(
        config_path="config.json",
        session=pipeline_session,
        role=role
    )
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# MODS availability check
_MODS_AVAILABLE = None

def is_mods_available() -> bool:
    """
    Check if MODS is available in the current environment.
    
    Returns:
        bool: True if MODS is available, False otherwise
    """
    global _MODS_AVAILABLE
    
    if _MODS_AVAILABLE is None:
        try:
            from cursus.mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler
            _MODS_AVAILABLE = True
            logger.debug("MODS is available")
        except ImportError:
            _MODS_AVAILABLE = False
            logger.debug("MODS is not available")
    
    return _MODS_AVAILABLE


def get_mods_compiler_class():
    """
    Get the MODS compiler class if available.
    
    Returns:
        MODSPipelineDAGCompiler class if available, None otherwise
    """
    if not is_mods_available():
        return None
    
    try:
        from cursus.mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler
        return MODSPipelineDAGCompiler
    except ImportError:
        logger.warning("Failed to import MODSPipelineDAGCompiler")
        return None


def check_mods_requirements():
    """
    Check MODS requirements and raise appropriate error if not available.
    
    Raises:
        ImportError: If MODS is not available with helpful message
    """
    if not is_mods_available():
        raise ImportError(
            "MODS is not available in this environment. "
            "MODS pipelines require the MODS package to be installed. "
            "Please install MODS or use standard pipelines instead."
        )


class MODSPipelineError(Exception):
    """Base exception for MODS pipeline errors."""
    pass


class MODSNotAvailableError(MODSPipelineError):
    """Raised when MODS functionality is requested but not available."""
    pass


def safe_mods_operation(func):
    """
    Decorator to safely execute MODS operations with fallback.
    
    Args:
        func: Function to execute
        
    Returns:
        Decorated function that handles MODS availability
    """
    def wrapper(*args, **kwargs):
        try:
            check_mods_requirements()
            return func(*args, **kwargs)
        except ImportError as e:
            raise MODSNotAvailableError(str(e))
    
    return wrapper
