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
from .processing_test import (
    UniversalProcessingBuilderTest,
    ProcessingStepBuilderValidator,
    ProcessingStepBuilderLLMAnalyzer,
    test_processing_builder,
    ProcessingStepType,
    StandardizationViolation,
    AlignmentViolation,
    LLMFeedback
)

__all__ = [
    'UniversalStepBuilderTest',
    'InterfaceTests',
    'SpecificationTests', 
    'PathMappingTests',
    'IntegrationTests',
    'StepBuilderScorer',
    'score_builder_results',
    'UniversalStepBuilderTestBase',
    'UniversalProcessingBuilderTest',
    'ProcessingStepBuilderValidator',
    'ProcessingStepBuilderLLMAnalyzer',
    'test_processing_builder',
    'ProcessingStepType',
    'StandardizationViolation',
    'AlignmentViolation',
    'LLMFeedback'
]
