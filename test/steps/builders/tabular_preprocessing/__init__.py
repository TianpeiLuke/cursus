"""
Tabular Preprocessing Step Builder Test Suite.

This package contains tests for the TabularPreprocessingStepBuilder using the
existing validation infrastructure from src/cursus/validation/builders.

The test suite leverages:
- UniversalStepBuilderTest: Comprehensive validation across all architectural levels
- ProcessingStepBuilderTest: Processing-specific validation patterns

Usage:
    # Run all tests
    python run_test.py
    
    # Run specific test classes
    python test_tabular_preprocessing.py
    
    # Import and run programmatically
    from test_tabular_preprocessing import run_comprehensive_test
    run_comprehensive_test()
"""

__version__ = "1.0.6"
__author__ = "Cursus Development Team"

# Import test runners for easy access
from .test_tabular_preprocessing import (
    TestTabularPreprocessingWithExistingValidators,
    TestTabularPreprocessingMultipleJobTypes,
    run_comprehensive_test
)

__all__ = [
    "TestTabularPreprocessingWithExistingValidators",
    "TestTabularPreprocessingMultipleJobTypes", 
    "run_comprehensive_test"
]
