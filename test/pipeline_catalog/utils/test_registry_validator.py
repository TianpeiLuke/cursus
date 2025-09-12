"""
Unit tests for RegistryValidator class.

This file serves as an entry point that imports all registry validator tests
from the split test modules for better organization.

The tests have been split into focused modules:
- test_registry_validator_models.py: Model classes and enums
- test_registry_validator_core.py: Core validation methods
- test_registry_validator_integration.py: Integration tests and Zettelkasten principles
"""

# Import all test classes from split modules to maintain test discovery
from .test_registry_validator_models import (
    TestValidationSeverity,
    TestValidationIssue,
    TestSpecializedValidationIssues,
    TestValidationReport,
)

from .test_registry_validator_core import TestRegistryValidatorCore

from .test_registry_validator_integration import (
    TestRegistryValidatorZettelkasten,
    TestRegistryValidatorIntegration,
)

# Re-export all test classes for pytest discovery
__all__ = [
    "TestValidationSeverity",
    "TestValidationIssue",
    "TestSpecializedValidationIssues",
    "TestValidationReport",
    "TestRegistryValidatorCore",
    "TestRegistryValidatorZettelkasten",
    "TestRegistryValidatorIntegration",
]
