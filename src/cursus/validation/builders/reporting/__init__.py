"""
Streamlined Reporting and Scoring Package for Builder Testing.

This package contains only the essential streamlined modules for generating 
reports and scoring for builder test results. Optimized to leverage the
alignment system infrastructure while eliminating over-engineering.

Essential Components:
- StreamlinedBuilderTestReporter: Unified reporting system
- StreamlinedStepBuilderScorer: Component-based scoring system

Architectural Changes:
- ✅ Streamlined reporting system (50% code reduction)
- ✅ Component-based scoring (eliminates manual test categorization)
- ✅ Alignment system integration (leverages proven infrastructure)
- ✅ Unified report formats (compatible with alignment system)
- ✅ Eliminated over-engineered components (results_storage, report_generator, etc.)
- ✅ Backward compatibility maintained
"""

# Core streamlined components (essential only)
from .builder_reporter import (
    StreamlinedBuilderTestReporter,
    StreamlinedBuilderTestReport,
    BuilderTestReporter,    # Legacy compatibility alias
    BuilderTestReport,      # Legacy compatibility alias
)

from .scoring import (
    StreamlinedStepBuilderScorer,
    score_builder_validation_results,
    score_builder_results,  # Legacy compatibility
    StepBuilderScorer,      # Legacy compatibility alias
)

__all__ = [
    # New streamlined components
    'StreamlinedBuilderTestReporter',
    'StreamlinedBuilderTestReport',
    'StreamlinedStepBuilderScorer',
    'score_builder_validation_results',
    
    # Legacy compatibility
    'BuilderTestReporter',
    'BuilderTestReport',
    'StepBuilderScorer',
    'score_builder_results',
]
