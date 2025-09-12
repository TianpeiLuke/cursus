"""
Unit tests for PipelineRecommendationEngine class.

This file serves as an entry point that imports all recommendation engine tests
from the split test modules for better organization.

The tests have been split into focused modules:
- test_recommendation_engine_models.py: Model classes and data structures
- test_recommendation_engine_core.py: Core recommendation functionality
- test_recommendation_engine_integration.py: Integration tests and composition algorithms
"""

# Import all test classes from split modules to maintain test discovery
from .test_recommendation_engine_models import (
    TestRecommendationResult,
    TestCompositionRecommendation,
)

from .test_recommendation_engine_core import TestPipelineRecommendationEngineCore

from .test_recommendation_engine_integration import (
    TestPipelineRecommendationEngineCompositions,
    TestPipelineRecommendationEngineIntegration,
)

# Re-export all test classes for pytest discovery
__all__ = [
    "TestRecommendationResult",
    "TestCompositionRecommendation",
    "TestPipelineRecommendationEngineCore",
    "TestPipelineRecommendationEngineCompositions",
    "TestPipelineRecommendationEngineIntegration",
]
