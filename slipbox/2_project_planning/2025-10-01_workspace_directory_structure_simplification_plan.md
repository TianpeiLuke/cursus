---
tags:
  - project
  - planning
  - workspace
  - directory_structure
  - simplification
  - step_catalog
keywords:
  - workspace directory simplification
  - consistent structure format
  - package structure alignment
  - workspace-aware method refactoring
  - directory structure standardization
topics:
  - workspace directory structure
  - package structure consistency
  - workspace-aware system refactoring
  - step catalog integration
language: python
date of note: 2025-10-01
implementation_status: PLANNING_PHASE
---

# Workspace Directory Structure Simplification Implementation Plan

## Executive Summary

This implementation plan details the **simplification of workspace directory structure assumptions** across the cursus step catalog and workspace systems. The plan addresses the current complex nested directory structure (`development/projects/project_alpha/src/cursus_dev/steps`) and replaces it with a simplified structure where `workspace_dir` directly contains the same folder structure as the package structure under `cursus/steps`.

### Key Objectives

- **Simplify Workspace Structure**: Replace complex nested paths with direct structure alignment
- **Consistent Structure Format**: Ensure workspace directories mirror package structure exactly
- **Reduce Complexity**: Eliminate hardcoded path assumptions and nested directory navigation
- **Maintain Functionality**: Preserve all existing workspace-aware functionality
- **Improve Usability**: Make workspace configuration more intuitive for users

### Strategic Impact

- **Simplified User Experience**: Users can organize workspace directories to match package structure
- **Reduced Code Complexity**: Eliminate complex path navigation and hardcoded assumptions
- **Better Maintainability**: Consistent structure patterns across package and workspace
- **Enhanced Reliability**: Fewer path resolution errors and edge cases
- **Improved Performance**: Simpler directory traversal and component discovery

## Current Problem Analysis

### Complex Nested Directory Structure

**Current Assumption**:
```
/user/project/
├── development/projects/   # User workspaces
│   ├── project_alpha/
│   │   └── src/cursus_dev/steps/
│   │       ├── contracts/
│   │       ├── builders/
│   │       ├── configs/
│   │       ├── scripts/
│   │       └── specs/
│   └── project_beta/
│       └── src/cursus_dev/steps/
└── my_pipeline.py
```

**Problems with Current Structure**:
1. **Complex Path Navigation**: Requires hardcoded `development/projects/*/src/cursus_dev/steps` traversal
2. **Inconsistent with Package**: Package uses `cursus/steps/*` while workspace uses nested structure
3. **User Confusion**: Users must create complex nested directory structures
4. **Maintenance Burden**: Multiple hardcoded path assumptions across codebase
5. **Error Prone**: Complex path resolution leads to discovery failures

### Proposed Simplified Structure

**New Assumption**:
```
workspace_dir/              # User-specified workspace directory
├── contracts/              # Same as cursus/steps/contracts/
├── builders/               # Same as cursus/steps/builders/
├── configs/                # Same as cursus/steps/configs/
├── scripts/                # Same as cursus/steps/scripts/
└── specs/                  # Same as cursus/steps/specs/
```

**Benefits of Simplified Structure**:
1. **Direct Structure Alignment**: `workspace_dir` structure matches `cursus/steps` exactly
2. **Simplified Path Resolution**: `workspace_dir / 'contracts'` instead of complex nested paths
3. **User-Friendly**: Users can easily understand and create workspace directories
4. **Consistent Patterns**: Same structure patterns for package and workspace discovery
5. **Reduced Complexity**: Eliminate hardcoded path assumptions and nested navigation

## Current Implementation Analysis

### Workspace-Aware Methods Found

Based on comprehensive code analysis, the following files contain workspace-aware methods that need refactoring:

#### **Step Catalog Module (`src/cursus/step_catalog/`)**

**Files with Complex Path Assumptions**:
1. **`config_discovery.py`** - Lines 89-95, 125-135
   - `_discover_workspace_configs()` - Uses `development/projects/*/src/cursus_dev/steps/configs`
   - `_discover_workspace_hyperparams()` - Uses `development/projects/*/src/cursus_dev/steps/hyperparams`

2. **`spec_discovery.py`** - Lines 78-88, 156-166, 189-199, 221-231
   - `_discover_workspace_specs()` - Uses `development/projects/*/src/cursus_dev/steps/specs`
   - `_try_workspace_spec_import()` - Complex nested path navigation
   - `_find_specs_by_contract_in_workspace()` - Hardcoded structure assumptions
   - `_find_job_type_variants_in_workspace()` - Complex path resolution

3. **`contract_discovery.py`** - Lines 89-95, 156-166, 189-199, 221-231
   - `_discover_workspace_contracts()` - **ALREADY SIMPLIFIED** (uses direct structure)
   - `_try_workspace_contract_import()` - **ALREADY SIMPLIFIED** (uses direct structure)
   - `_find_contracts_by_entry_point_in_workspace()` - **ALREADY SIMPLIFIED**
   - `_extract_entry_points_from_workspace()` - **ALREADY SIMPLIFIED**

4. **`builder_discovery.py`** - Lines 67-85
   - `_discover_workspace_builders()` - Uses `development/projects/*/src/cursus_dev/steps/builders`

5. **`step_catalog.py`** - Lines 89-95
   - `_discover_workspace_components()` - **ALREADY SIMPLIFIED** (uses direct structure)

**Files with Adapter-Level Complexity**:
6. **`adapters/workspace_discovery.py`** - Lines 67-85, 102-125
   - `_count_workspace_components()` - Uses `src/cursus_dev/steps` hardcoded path
   - `discover_workspaces()` - Complex
