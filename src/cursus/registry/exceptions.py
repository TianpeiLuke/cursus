"""
Exception classes for the Pipeline Registry.

This module defines custom exceptions used in the Pipeline Registry
to provide clear, actionable error messages.
"""

from typing import List, Optional


class RegistryError(Exception):
    """Raised when step builder registry errors occur."""
    
    def __init__(self, message: str, unresolvable_types: Optional[List[str]] = None,
                 available_builders: Optional[List[str]] = None):
        super().__init__(message)
        self.unresolvable_types = unresolvable_types or []
        self.available_builders = available_builders or []
        
    def __str__(self) -> str:
        msg = super().__str__()
        if self.unresolvable_types:
            msg += f"\nUnresolvable step types: {self.unresolvable_types}"
        if self.available_builders:
            msg += f"\nAvailable builders: {self.available_builders}"
        return msg
