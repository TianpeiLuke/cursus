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

from .universal_test import UniversalStepBuilderTest
from .interface_tests import InterfaceTests
from .specification_tests import SpecificationTests
from .step_creation_tests import StepCreationTests
from .integration_tests import IntegrationTests
from .scoring import StepBuilderScorer, score_builder_results
from .base_test import UniversalStepBuilderTestBase
# Enhanced universal tester system
from .test_factory import UniversalStepBuilderTestFactory
from .step_info_detector import StepInfoDetector
from .mock_factory import StepTypeMockFactory
from .generic_test import GenericStepBuilderTest
from .variants.processing_test import ProcessingStepBuilderTest
from .variants.training_test import TrainingStepBuilderTest
from .variants.transform_test import TransformStepBuilderTest
from .variants.createmodel_test import CreateModelStepBuilderTest
# Registry-based discovery utilities
from .registry_discovery import (
    RegistryStepDiscovery,
    get_training_steps_from_registry,
    get_transform_steps_from_registry,
    get_createmodel_steps_from_registry,
    get_processing_steps_from_registry,
    get_builder_class_path,
    load_builder_class
)

__all__ = [
    'UniversalStepBuilderTest',
    'InterfaceTests',
    'SpecificationTests', 
    'StepCreationTests',
    'IntegrationTests',
    'StepBuilderScorer',
    'score_builder_results',
    'UniversalStepBuilderTestBase',
    # Enhanced universal tester system
    'UniversalStepBuilderTestFactory',
    'StepInfoDetector',
    'StepTypeMockFactory',
    'GenericStepBuilderTest',
    # Step-type-specific test variants
    'ProcessingStepBuilderTest',
    'TrainingStepBuilderTest',
    'TransformStepBuilderTest',
    'CreateModelStepBuilderTest',
    # Registry-based discovery utilities
    'RegistryStepDiscovery',
    'get_training_steps_from_registry',
    'get_transform_steps_from_registry',
    'get_createmodel_steps_from_registry',
    'get_processing_steps_from_registry',
    'get_builder_class_path',
    'load_builder_class'
]
