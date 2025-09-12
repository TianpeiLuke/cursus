"""
Unit tests for enhanced metadata system.

Tests the EnhancedDAGMetadata system that integrates Zettelkasten knowledge
management principles with the existing pipeline metadata infrastructure.

This file serves as an entry point that imports tests from focused modules:
- test_enhanced_metadata_models: Tests for enum classes and ZettelkastenMetadata
- test_enhanced_metadata_core: Tests for main EnhancedDAGMetadata class functionality
- test_enhanced_metadata_integration: Integration tests and complex scenarios
"""

# Import all test classes from split modules to maintain pytest discovery
from .test_enhanced_metadata_models import (
    TestComplexityLevel,
    TestPipelineFramework,
    TestZettelkastenMetadata,
)

from .test_enhanced_metadata_core import (
    TestEnhancedDAGMetadata,
    TestDAGMetadataAdapter,
    TestValidateEnhancedDAGMetadata,
)

from .test_enhanced_metadata_integration import TestEnhancedDAGMetadataIntegration

# Re-export all test classes for pytest discovery
__all__ = [
    # Model tests
    "TestComplexityLevel",
    "TestPipelineFramework",
    "TestZettelkastenMetadata",
    # Core functionality tests
    "TestEnhancedDAGMetadata",
    "TestDAGMetadataAdapter",
    "TestValidateEnhancedDAGMetadata",
    # Integration tests
    "TestEnhancedDAGMetadataIntegration",
]
