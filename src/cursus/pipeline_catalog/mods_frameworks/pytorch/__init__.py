"""
MODS PyTorch Pipeline Variants

This module contains MODS-specific PyTorch pipeline implementations that use
the MODSPipelineDAGCompiler for enhanced functionality and MODS global registry
integration.

Available MODS PyTorch Pipelines:
- training/basic_training_mods: Basic PyTorch training with MODS features
- end_to_end/standard_e2e_mods: Standard PyTorch end-to-end workflow

All pipelines use the same shared DAG definitions as their standard counterparts
but provide MODS-specific enhancements including automatic template registration
and enhanced metadata handling.
"""

from .. import is_mods_available, check_mods_requirements, MODSNotAvailableError

# Re-export common MODS utilities for convenience
__all__ = [
    'is_mods_available',
    'check_mods_requirements', 
    'MODSNotAvailableError'
]
