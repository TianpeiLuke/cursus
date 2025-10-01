"""
Universal Step Builder Validation Framework.

This package provides comprehensive testing and validation capabilities for
step builders in the cursus pipeline system. It includes multiple test levels
that validate different aspects of step builder implementation.

Main Components:
- UniversalStepBuilderTest: Main test suite combining all test levels
- InterfaceTests: Level 1 - Basic interface compliance
- SpecificationTests: Level 2 - Specification and contract compliance  
- PathMappingTests: Level 3 - Path mapping and property paths
- IntegrationTests: Level 4 - System integration
- StepBuilderScorer: Scoring system for test results

Usage:
    from cursus.validation.builders import UniversalStepBuilderTest
    
    # Test a step builder
    tester = UniversalStepBuilderTest(MyStepBuilder)
    results = tester.run_all_tests()
"""

# Core testing framework
from .universal_test import UniversalStepBuilderTest
from .core.interface_tests import InterfaceTests
from .core.specification_tests import SpecificationTests
from .core.step_creation_tests import StepCreationTests
from .core.integration_tests import IntegrationTests

# Reporting and visualization
from .reporting import (
    StepBuilderScorer, 
    score_builder_results,
    BuilderTestResultsStorage,
    EnhancedReportGenerator,
    StepTypeColorScheme,
    EnhancedStatusDisplay
)

# Factory classes
from .factories import StepTypeTestFrameworkFactory

# Discovery utilities
from .discovery import (
    RegistryStepDiscovery,
    get_training_steps_from_registry,
    get_transform_steps_from_registry,
    get_createmodel_steps_from_registry,
    get_processing_steps_from_registry,
    get_builder_class_path,
    load_builder_class,
)

# Step-type-specific test variants
from .variants.processing_test import ProcessingStepBuilderTest
from .variants.training_test import TrainingStepBuilderTest
from .variants.transform_test import TransformStepBuilderTest
from .variants.createmodel_test import CreateModelStepBuilderTest

# Legacy compatibility
try:
    from .core.base_test import UniversalStepBuilderTestBase
except ImportError:
    UniversalStepBuilderTestBase = None

__all__ = [
    "UniversalStepBuilderTest",
    "InterfaceTests",
    "SpecificationTests",
    "StepCreationTests",
    "IntegrationTests",
    "StepBuilderScorer",
    "score_builder_results",
    "UniversalStepBuilderTestBase",
    # Enhanced universal tester system
    "UniversalStepBuilderTestFactory",
    # Step-type-specific test variants
    "ProcessingStepBuilderTest",
    "TrainingStepBuilderTest",
    "TransformStepBuilderTest",
    "CreateModelStepBuilderTest",
    # Registry-based discovery utilities
    "RegistryStepDiscovery",
    "get_training_steps_from_registry",
    "get_transform_steps_from_registry",
    "get_createmodel_steps_from_registry",
    "get_processing_steps_from_registry",
    "get_builder_class_path",
    "load_builder_class",
]
