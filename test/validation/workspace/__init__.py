"""
Test package for Cursus Validation Workspace functionality

This package contains comprehensive unit tests for the multi-developer workspace
management system, including tests for:

- DeveloperWorkspaceFileResolver: Workspace-aware file discovery
- WorkspaceModuleLoader: Dynamic module loading with workspace isolation  
- WorkspaceManager: Workspace discovery, validation, and management

Test Structure:
- test_workspace_file_resolver.py: Tests for file resolution functionality
- test_workspace_module_loader.py: Tests for module loading functionality
- test_workspace_manager.py: Tests for workspace management functionality

Usage:
    # Run all workspace tests
    python -m pytest test/validation/workspace/
    
    # Run specific test module
    python -m pytest test/validation/workspace/test_workspace_manager.py
    
    # Run with coverage
    python -m pytest test/validation/workspace/ --cov=src.cursus.validation.workspace

Test Coverage:
The test suite provides comprehensive coverage of:
- Workspace mode detection and validation
- File discovery with developer/shared workspace fallback
- Dynamic module loading with Python path management
- Workspace structure creation and validation
- Configuration management (JSON/YAML)
- Error handling and edge cases
- Backward compatibility with single workspace mode
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import test modules
from .test_workspace_file_resolver import TestDeveloperWorkspaceFileResolver
from .test_workspace_module_loader import TestWorkspaceModuleLoader
from .test_workspace_manager import TestWorkspaceManager

# Test suite configuration
__version__ = "1.0.0"
__author__ = "Cursus Development Team"

# Test discovery patterns
TEST_PATTERNS = [
    "test_*.py",
    "*_test.py"
]

# Test categories
TEST_CATEGORIES = {
    "file_resolution": [
        "test_workspace_file_resolver.py"
    ],
    "module_loading": [
        "test_workspace_module_loader.py"
    ],
    "workspace_management": [
        "test_workspace_manager.py"
    ],
    "integration": [
        # Future integration tests
    ]
}


def create_test_suite():
    """
    Create a comprehensive test suite for workspace functionality.
    
    Returns:
        unittest.TestSuite: Complete test suite
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDeveloperWorkspaceFileResolver,
        TestWorkspaceModuleLoader,
        TestWorkspaceManager
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_all_tests(verbosity=2):
    """
    Run all workspace tests with specified verbosity.
    
    Args:
        verbosity: Test output verbosity level (0-2)
    
    Returns:
        unittest.TestResult: Test results
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_category_tests(category, verbosity=2):
    """
    Run tests for a specific category.
    
    Args:
        category: Test category name
        verbosity: Test output verbosity level (0-2)
    
    Returns:
        unittest.TestResult: Test results
    """
    if category not in TEST_CATEGORIES:
        raise ValueError(f"Unknown test category: {category}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Map category to test classes
    category_classes = {
        "file_resolution": [TestDeveloperWorkspaceFileResolver],
        "module_loading": [TestWorkspaceModuleLoader],
        "workspace_management": [TestWorkspaceManager]
    }
    
    test_classes = category_classes.get(category, [])
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def get_test_info():
    """
    Get information about available tests.
    
    Returns:
        dict: Test information including categories and test counts
    """
    info = {
        "total_test_classes": 3,
        "categories": list(TEST_CATEGORIES.keys()),
        "test_files": [
            "test_workspace_file_resolver.py",
            "test_workspace_module_loader.py", 
            "test_workspace_manager.py"
        ],
        "coverage_areas": [
            "Workspace mode detection and validation",
            "File discovery with workspace fallback",
            "Dynamic module loading with path management",
            "Workspace structure creation and validation",
            "Configuration management (JSON/YAML)",
            "Error handling and edge cases",
            "Backward compatibility"
        ]
    }
    
    # Count test methods
    loader = unittest.TestLoader()
    total_tests = 0
    
    for test_class in [TestDeveloperWorkspaceFileResolver, TestWorkspaceModuleLoader, TestWorkspaceManager]:
        suite = loader.loadTestsFromTestCase(test_class)
        total_tests += suite.countTestCases()
    
    info["total_test_methods"] = total_tests
    
    return info


# Convenience functions for common test scenarios
def test_file_resolution_only():
    """Run only file resolution tests."""
    return run_category_tests("file_resolution")


def test_module_loading_only():
    """Run only module loading tests."""
    return run_category_tests("module_loading")


def test_workspace_management_only():
    """Run only workspace management tests."""
    return run_category_tests("workspace_management")


# Test execution entry point
if __name__ == "__main__":
    print("Cursus Validation Workspace Test Suite")
    print("=" * 50)
    
    # Display test information
    info = get_test_info()
    print(f"Total test classes: {info['total_test_classes']}")
    print(f"Total test methods: {info['total_test_methods']}")
    print(f"Test categories: {', '.join(info['categories'])}")
    print()
    
    # Run all tests
    print("Running all tests...")
    result = run_all_tests()
    
    # Display results summary
    print("\nTest Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
