"""
Phase 3: Simplified Local Registry Infrastructure

This module provides the core hybrid registry system components for Phase 3 implementation.
Focuses on essential functionality with reduced redundancy.

Architecture:
- utils.py: Simple function-based utilities
- models.py: Pydantic data models
- manager.py: Unified registry manager
- setup.py: Workspace initialization utilities
"""

# Data Models
from .models import (
    StepDefinition,
    ResolutionContext,
    StepResolutionResult,
    RegistryValidationResult,
    ConflictAnalysis,
)

# Registry Management
from .manager import (
    UnifiedRegistryManager,
    CoreStepRegistry,
    LocalStepRegistry,
    HybridRegistryManager,
)

# Shared Utilities
from .utils import (
    load_registry_module,
    from_legacy_format,
    to_legacy_format,
    convert_registry_dict,
    validate_registry_data,
    format_step_not_found_error,
    format_registry_load_error,
)

__all__ = [
    # Data Models
    "StepDefinition",
    "ResolutionContext",
    "StepResolutionResult",
    "RegistryValidationResult",
    "ConflictAnalysis",
    # Registry Management
    "UnifiedRegistryManager",
    "CoreStepRegistry",
    "LocalStepRegistry",
    "HybridRegistryManager",
    # Shared Utilities
    "load_registry_module",
    "from_legacy_format",
    "to_legacy_format",
    "convert_registry_dict",
    "validate_registry_data",
    "format_step_not_found_error",
    "format_registry_load_error",
]
