# Config Field Manager and Registry Refactoring

## Overview

We have refactored the configuration management system to follow the Single Source of Truth (SSOT) principle. This document explains the changes made and the rationale behind them.

## Key Changes

### 1. Created a New Hyperparameter Registry

- Added `src/pipeline_registry/hyperparameter_registry.py` to serve as the central source of truth for all hyperparameter classes
- This registry contains metadata about each hyperparameter class, including module path, model type, and description
- Added helper functions to look up hyperparameter classes by name or model type

### 2. Refactored `build_complete_config_classes()` in utils.py

- Previously, this function contained hardcoded class mappings, violating the SSOT principle
- Now it dynamically imports classes from the registries:
  - Step config classes from `STEP_NAMES` registry in `step_names.py`
  - Hyperparameter classes from the new `HYPERPARAMETER_REGISTRY`
- Includes fallback mechanisms for core classes to ensure backward compatibility
- Provides detailed debug logging during class registration

### 3. Updated Pipeline Registry Module

- Expanded `src/pipeline_registry/__init__.py` to expose the hyperparameter registry
- This allows for centralized access to both step and hyperparameter registries

## Benefits

1. **Reduced Duplication**: Class information is now defined in a single location
2. **Improved Maintainability**: Adding or changing classes requires updates to only one place
3. **Better Error Handling**: Provides graceful degradation when classes can't be imported
4. **Enhanced Logging**: Debug logs help trace class registration issues
5. **Centralized Metadata**: Additional metadata like descriptions and module paths are kept with class registrations

## Design Principles Applied

- **Single Source of Truth**: Each piece of information has one authoritative source
- **Separation of Concerns**: Registration is separate from usage
- **Open/Closed Principle**: System is open for extension but closed for modification
- **Dependency Inversion**: High-level modules don't depend on low-level implementation details

## Future Work

1. **Complete Deserialization**: Enhance the deserialization system to fully leverage the registries
2. **Add More Metadata**: Extend registries with additional information like version compatibility
3. **Tool Support**: Develop tools to validate and manage registry entries
4. **Registry Synchronization**: Ensure registry entries match actual class implementations

## Test Results

Tests for the refactored system are passing, with the exception of the `load_configs` test which is currently skipped as it requires further development of the deserialization functionality.
