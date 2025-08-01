# Pipeline Dependencies Test Suite

This directory contains tests for the `pipeline_deps` module, which provides the core dependency management functionality for the pipeline system.

## Test Isolation Issue

The `pipeline_deps` module uses global singleton objects for key components:

- `registry_manager` in `registry_manager.py`
- `global_resolver` in `dependency_resolver.py`
- `semantic_matcher` in `semantic_matcher.py`

These global singletons maintain state that can persist between test runs, leading to a common issue where **tests pass when run individually but fail when run together**. This happens because one test modifies the global state and doesn't clean up properly, affecting subsequent tests.

## Solution: Test Helpers

The `test_helpers.py` module provides utilities to ensure proper isolation between tests:

1. `reset_all_global_state()` function - Resets all global singletons to their initial state
2. `IsolatedTestCase` base class - Automatically resets global state before and after each test

## How to Use

### Option 1: Inherit from IsolatedTestCase

The simplest approach is to inherit your test classes from `IsolatedTestCase`:

```python
from test.pipeline_deps.test_helpers import IsolatedTestCase

class TestMyFeature(IsolatedTestCase):
    def test_something(self):
        # This test starts with clean global state
        pass
```

### Option 2: Manual Reset in setUp/tearDown

If you can't inherit from `IsolatedTestCase`, manually call `reset_all_global_state()` in your `setUp` and `tearDown` methods:

```python
from unittest import TestCase
from test.pipeline_deps.test_helpers import reset_all_global_state

class TestMyFeature(TestCase):
    def setUp(self):
        reset_all_global_state()
        # Other setup code...
    
    def tearDown(self):
        # Other teardown code...
        reset_all_global_state()
```

## Demonstration

The `test_global_state_isolation.py` file demonstrates the issue and solutions:

1. `TestWithoutIsolation` - Shows tests that fail when run together due to state leakage
2. `TestWithManualIsolation` - Shows manual cleanup in setUp/tearDown
3. `TestWithHelperIsolation` - Shows using the IsolatedTestCase base class

You can run this file directly to see the demonstration:

```bash
python -m test.pipeline_deps.test_global_state_isolation
```

## Best Practices

1. **Always use IsolatedTestCase** for new test classes in the pipeline_deps module
2. **Add reset_all_global_state() calls** to existing test classes that can't be refactored
3. **Run tests both individually and together** to catch isolation issues
4. **Consider using dependency injection** instead of global singletons for new code

## Further Reading

For a detailed discussion of global singletons vs. local objects, see the design document:

- [Global Singletons vs. Local Objects](../../slipbox/pipeline_design/global_vs_local_objects.md)
