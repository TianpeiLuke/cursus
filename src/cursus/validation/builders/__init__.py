"""
Refactored Builders Validation Package

Simplified package that leverages the alignment system to eliminate redundancy
while preserving unique builder testing capabilities.

This package provides comprehensive testing and validation capabilities for
step builders in the cursus pipeline system, now unified with the alignment
system to eliminate 60-70% code redundancy.

Main Components:
- UniversalStepBuilderTest: Refactored main test suite leveraging alignment system
- IntegrationTests: Unique integration testing capabilities (Level 4)
- StepBuilderScorer: Scoring system for test results
- Enhanced reporting and visualization

Key Changes:
- Eliminated redundant interface and specification tests (now handled by alignment system)
- Simplified step creation testing with capability-focused approach
- Removed redundant discovery, factory, and variant components
- Preserved unique integration testing and scoring capabilities
- Maintained 100% backward compatibility

Usage:
    # New unified approach (recommended)
    from cursus.validation.builders import UniversalStepBuilderTest
    tester = UniversalStepBuilderTest(workspace_dirs=['.'])
    results = tester.run_full_validation()
    
    # Legacy compatibility (still supported)
    tester = UniversalStepBuilderTest.from_builder_class(MyStepBuilder)
    results = tester.run_all_tests_legacy()
"""

# Core testing framework (refactored)
from .universal_test import UniversalStepBuilderTest

# Keep existing integration testing (unique value)
try:
    from .core.integration_tests import IntegrationTests
    _has_integration_tests = True
except ImportError:
    _has_integration_tests = False

# Keep existing scoring and reporting (unique value)
try:
    from .reporting.scoring import StepBuilderScorer
    from .reporting.builder_reporter import score_builder_results
    from .reporting.results_storage import BuilderTestResultsStorage
    from .reporting.report_generator import EnhancedReportGenerator
    from .reporting.step_type_color_scheme import StepTypeColorScheme
    from .reporting.enhanced_status_display import EnhancedStatusDisplay
    _has_reporting = True
except ImportError:
    _has_reporting = False

# Legacy compatibility
try:
    from .core.base_test import UniversalStepBuilderTestBase
except ImportError:
    UniversalStepBuilderTestBase = None

# Build __all__ dynamically based on what's available
__all__ = ["UniversalStepBuilderTest"]

if _has_integration_tests:
    __all__.append("IntegrationTests")

if _has_reporting:
    __all__.extend([
        "StepBuilderScorer",
        "score_builder_results", 
        "BuilderTestResultsStorage",
        "EnhancedReportGenerator",
        "StepTypeColorScheme",
        "EnhancedStatusDisplay"
    ])

if UniversalStepBuilderTestBase is not None:
    __all__.append("UniversalStepBuilderTestBase")
