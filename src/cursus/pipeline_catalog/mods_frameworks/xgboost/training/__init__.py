"""
MODS XGBoost Training Pipelines

This module contains MODS-specific XGBoost training pipeline implementations.

Available Pipelines:
- with_calibration_mods: XGBoost training with model calibration
- with_evaluation_mods: XGBoost training with model evaluation
"""

from ... import is_mods_available, check_mods_requirements, MODSNotAvailableError

__all__ = [
    'is_mods_available',
    'check_mods_requirements',
    'MODSNotAvailableError'
]
