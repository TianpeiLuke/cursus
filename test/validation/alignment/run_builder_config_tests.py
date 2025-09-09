#!/usr/bin/env python3
"""
Test runner for builder-configuration alignment tests

Runs all tests for the enhanced builder-configuration validation components:
- ConfigurationAnalyzer tests
- BuilderCodeAnalyzer tests  
- BuilderConfigurationAlignmentTester tests
"""

import unittest
import sys
from pathlib import Path


# Import test modules
from analyzers.test_config_analyzer import TestConfigurationAnalyzer
from analyzers.test_builder_analyzer import TestBuilderCodeAnalyzer
from .test_builder_config_alignment import TestBuilderConfigurationAlignmentTester

def create_test_suite():
    """Create a test suite with all builder-config alignment tests"""
    suite = unittest.TestSuite()
    
    # Add ConfigurationAnalyzer tests
    suite.addTest(unittest.makeSuite(TestConfigurationAnalyzer))
    
    # Add BuilderCodeAnalyzer tests
    suite.addTest(unittest.makeSuite(TestBuilderCodeAnalyzer))
    
    # Add BuilderConfigurationAlignmentTester tests
    suite.addTest(unittest.makeSuite(TestBuilderConfigurationAlignmentTester))
    
    return suite

def run_tests():
    """Run all builder-configuration alignment tests"""
    print("=" * 70)
    print("Running Builder-Configuration Alignment Tests")
    print("=" * 70)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
