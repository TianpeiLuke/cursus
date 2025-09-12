"""
Exception classes for the Pipeline API.

This module defines custom exceptions used throughout the Pipeline API
to provide clear, actionable error messages for users.
"""

from typing import List, Dict, Any, Optional

# Import RegistryError from pipeline_registry
from ...registry.exceptions import RegistryError


class PipelineAPIError(Exception):
    """Base exception for all Pipeline API errors."""

    pass


class ConfigurationError(PipelineAPIError):
    """Raised when configuration-related errors occur."""

    def __init__(
        self,
        message: str,
        missing_configs: Optional[List[str]] = None,
        available_configs: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.missing_configs = missing_configs or []
        self.available_configs = available_configs or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.missing_configs:
            msg += f"\nMissing configurations: {self.missing_configs}"
        if self.available_configs:
            msg += f"\nAvailable configurations: {self.available_configs}"
        return msg


# RegistryError moved to pipeline_registry.exceptions


class AmbiguityError(PipelineAPIError):
    """Raised when multiple configurations could match a DAG node."""

    def __init__(
        self,
        message: str,
        node_name: Optional[str] = None,
        candidates: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.node_name = node_name
        self.candidates = candidates or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.candidates:
            msg += f"\nCandidates for node '{self.node_name}':"
            for candidate in self.candidates:
                if isinstance(candidate, tuple) and len(candidate) >= 2:
                    # Handle tuple format (config, confidence, method)
                    config = candidate[0]
                    confidence = candidate[1]
                    config_type = type(config).__name__
                    job_type = getattr(config, "job_type", "N/A")
                    msg += f"\n  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})"
                elif isinstance(candidate, dict):
                    # Handle dictionary format
                    config_type = candidate.get("config_type", "Unknown")
                    confidence = candidate.get("confidence", 0.0)
                    job_type = candidate.get("job_type", "N/A")
                    msg += f"\n  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})"
        return msg


class ValidationError(PipelineAPIError):
    """Raised when DAG-config validation fails."""

    def __init__(
        self, message: str, validation_errors: Optional[Dict[str, List[str]]] = None
    ):
        super().__init__(message)
        self.validation_errors = validation_errors or {}

    def __str__(self) -> str:
        msg = super().__str__()
        if self.validation_errors:
            msg += "\nValidation errors:"
            for category, errors in self.validation_errors.items():
                msg += f"\n  {category}:"
                for error in errors:
                    msg += f"\n    - {error}"
        return msg


class ResolutionError(PipelineAPIError):
    """Raised when DAG node resolution fails."""

    def __init__(
        self,
        message: str,
        failed_nodes: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.failed_nodes = failed_nodes or []
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        msg = super().__str__()
        if self.failed_nodes:
            msg += f"\nFailed to resolve nodes: {self.failed_nodes}"
        if self.suggestions:
            msg += "\nSuggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  - {suggestion}"
        return msg
