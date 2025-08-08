"""
Interface validation module for the Cursus pipeline framework.

This module provides validation for interface compliance as defined in the
standardization rules document. It validates that step builders implement
required interfaces correctly.
"""

from .interface_standard_validator import InterfaceStandardValidator, InterfaceViolation

__all__ = [
    'InterfaceStandardValidator',
    'InterfaceViolation',
]
