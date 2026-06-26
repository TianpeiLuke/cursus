"""
Test helpers for pipeline_deps tests to ensure proper isolation.

This module provides utility functions and fixtures to reset global state
before and after tests, ensuring proper isolation between test cases.
"""

import pytest


def reset_all_global_state():
    """
    Reset all global state for testing.

    This function resets the state of all global singletons used in the pipeline_deps
    module, ensuring that tests start with a clean state.
    """
    # Note: All components (RegistryManager, SemanticMatcher, and UnifiedDependencyResolver)
    # are now created per-test and don't require global state reset
    pass


@pytest.fixture(autouse=True)
def isolated_test_setup():
    """
    Pytest fixture that automatically resets global state before and after each test.

    This fixture ensures that tests are properly isolated from each other by
    resetting all global state before and after each test execution.
    """
    # Setup: reset global state before test
    reset_all_global_state()
    yield
    # Teardown: reset global state after test
    reset_all_global_state()


# Example usage:
"""
from .test_helpers import reset_all_global_state

class TestMyFeature:
    def test_something(self):
        # This test starts with clean global state (via isolated_test_setup fixture)
        pass
        
    def test_something_else(self):
        # This test also starts with clean global state (via isolated_test_setup fixture)
        pass

# For tests that need explicit state reset:
def test_with_explicit_reset():
    reset_all_global_state()
    # ... test code ...

# The isolated_test_setup fixture runs automatically for all tests in modules that import this
"""
