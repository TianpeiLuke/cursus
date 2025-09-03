"""
Base Validation Result

Simple module that exports BaseValidationResult from unified_result_structures
for backward compatibility and cleaner imports.

This module provides a clean import path for the BaseValidationResult class
that is used across the workspace validation system.
"""

from .unified_result_structures import BaseValidationResult

__all__ = ['BaseValidationResult']
