"""
Unit tests for registry synchronization infrastructure.

Tests the DAGMetadataRegistrySync system that provides bidirectional
synchronization between enhanced DAG metadata and the JSON-based
connection registry.

This file serves as an entry point that imports tests from focused modules:
- test_registry_sync_models: Tests for basic models and utility functions
- test_registry_sync_core: Tests for main DAGMetadataRegistrySync class functionality
- test_registry_sync_integration: Integration tests and complex workflows
"""

# Import all test classes from split modules to maintain pytest discovery
from .test_registry_sync_models import (
    TestRegistryValidationError,
    TestRegistrySyncUtilityFunctions,
)

from .test_registry_sync_core import TestDAGMetadataRegistrySync

from .test_registry_sync_integration import TestRegistrySyncIntegration

# Re-export all test classes for pytest discovery
__all__ = [
    # Model and utility tests
    "TestRegistryValidationError",
    "TestRegistrySyncUtilityFunctions",
    # Core functionality tests
    "TestDAGMetadataRegistrySync",
    # Integration tests
    "TestRegistrySyncIntegration",
]
