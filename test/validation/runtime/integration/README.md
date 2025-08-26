# Integration Tests for Runtime Validation Components

This directory contains integration tests for runtime validation components in the cursus pipeline system.

## Test Files

### test_workspace_manager.py
Comprehensive unit tests for the `WorkspaceManager` class located in `src/cursus/validation/runtime/integration/workspace_manager.py`.

**Test Coverage:**
- **WorkspaceConfig Model Tests**: Configuration validation and default values
- **CacheEntry Model Tests**: Cache entry data structure validation
- **WorkspaceManager Core Functionality**:
  - Workspace setup and cleanup
  - Data caching with LRU policy
  - Cache size management and retention policies
  - Workspace information retrieval
  - Cache index persistence (load/save)
  - Error handling for file operations
  - Auto-cleanup configuration

**Test Classes:**
- `TestWorkspaceConfig`: Tests for workspace configuration model
- `TestCacheEntry`: Tests for cache entry data model  
- `TestWorkspaceManager`: Comprehensive tests for workspace management functionality

**Key Test Scenarios:**
- Workspace creation with standard directory structure
- Cache data with deduplication and access tracking
- Automatic cleanup of expired cache entries
- LRU-based cache size limit enforcement
- Graceful handling of file operation failures
- Cache index persistence across manager instances
- Configuration-driven auto-cleanup behavior

## Running Tests

### Run all integration tests:
```bash
python -m pytest test/validation/runtime/integration/ -v
```

### Run specific test file:
```bash
python -m pytest test/validation/runtime/integration/test_workspace_manager.py -v
```

### Run with coverage:
```bash
python -m pytest test/validation/runtime/integration/ --cov=src.cursus.validation.runtime.integration --cov-report=html
```

## Test Dependencies

- `pytest`: Test framework
- `tempfile`: Temporary directory creation for isolated testing
- `unittest.mock`: Mocking for error condition testing
- `pathlib`: Path manipulation utilities
- `datetime`: Time-based testing for cache expiration

## Test Structure

Each test class uses pytest fixtures for:
- `temp_dir`: Isolated temporary directory for each test
- `workspace_config`: Test configuration with appropriate limits
- `workspace_manager`: Configured manager instance
- `sample_file`: Test data file for caching operations

Tests are designed to be:
- **Isolated**: Each test runs in its own temporary directory
- **Comprehensive**: Cover both success and failure scenarios
- **Fast**: Use minimal test data and efficient cleanup
- **Reliable**: Mock external dependencies and handle edge cases
