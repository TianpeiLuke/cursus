"""
MODS XGBoost Pipeline Variants

This module contains MODS-specific XGBoost pipeline implementations that use
the MODSPipelineDAGCompiler for enhanced functionality and MODS global registry
integration.

Available MODS XGBoost Pipelines:
- simple_mods: Simple XGBoost training with MODS features
- training/with_calibration_mods: XGBoost training with calibration
- training/with_evaluation_mods: XGBoost training with evaluation
- end_to_end/complete_e2e_mods: Complete end-to-end XGBoost workflow

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
