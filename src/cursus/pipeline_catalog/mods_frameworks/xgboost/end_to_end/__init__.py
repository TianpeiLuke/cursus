"""
MODS XGBoost End-to-End Pipelines

This module contains MODS-specific XGBoost end-to-end pipeline implementations.

Available Pipelines:
- complete_e2e_mods: Complete XGBoost end-to-end workflow with MODS features
"""

from ... import is_mods_available, check_mods_requirements, MODSNotAvailableError

__all__ = [
    'is_mods_available',
    'check_mods_requirements',
    'MODSNotAvailableError'
]
