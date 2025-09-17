"""
Base classes and interfaces for execution document generation.

This module defines the base interfaces that all execution document helpers
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ExecutionDocumentHelper(ABC):
    """Base class for execution document helpers."""
    
    @abstractmethod
    def can_handle_step(self, step_name: str, config: Any) -> bool:
        """
        Check if this helper can handle the given step.
        
        Args:
            step_name: Name of the step
            config: Configuration object for the step
            
        Returns:
            True if this helper can process the step, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_step_config(self, step_name: str, config: Any) -> Dict[str, Any]:
        """
        Extract step configuration for execution document.
        
        Args:
            step_name: Name of the step
            config: Configuration object for the step
            
        Returns:
            Dictionary containing the step configuration for execution document
        """
        pass


class ExecutionDocumentGenerationError(Exception):
    """Base exception for execution document generation errors."""
    pass


class ConfigurationNotFoundError(ExecutionDocumentGenerationError):
    """Raised when configuration cannot be found for a step."""
    pass


class UnsupportedStepTypeError(ExecutionDocumentGenerationError):
    """Raised when step type is not supported for execution document generation."""
    pass
