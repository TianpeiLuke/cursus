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
from .path_mapping_tests import PathMappingTests
from .integration_tests import IntegrationTests
from .scoring import StepBuilderScorer, score_builder_results
from .base_test import UniversalStepBuilderTestBase
# Enhanced universal tester system
from .test_factory import UniversalStepBuilderTestFactory
from .step_info_detector import StepInfoDetector
from .mock_factory import StepTypeMockFactory
from .generic_test import GenericStepBuilderTest
from .variants.processing_test import ProcessingStepBuilderTest

__all__ = [
    'UniversalStepBuilderTest',
    'InterfaceTests',
    'SpecificationTests', 
    'PathMappingTests',
    'IntegrationTests',
    'StepBuilderScorer',
    'score_builder_results',
    'UniversalStepBuilderTestBase',
    # Enhanced universal tester system
    'UniversalStepBuilderTestFactory',
    'StepInfoDetector',
    'StepTypeMockFactory',
    'GenericStepBuilderTest',
    'ProcessingStepBuilderTest'
]
