"""
Script Testing Utilities Module

This module contains utility functions and helpers for script testing,
including script discovery and result formatting.
"""

from .result_formatter import ResultFormatter

# Note: Script discovery functionality is available in cursus.step_catalog module
# We reuse existing script discovery rather than reimplementing it

__all__ = [
    "ResultFormatter",
]
