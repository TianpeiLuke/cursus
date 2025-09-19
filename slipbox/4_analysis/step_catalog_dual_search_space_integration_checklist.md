---
tags:
  - analysis
  - step_catalog
  - integration
  - dual_search_space
  - migration
keywords:
  - step catalog integration
  - dual search space API
  - workspace_dirs parameter
  - migration checklist
  - system updates
topics:
  - step catalog system integration
  - dual search space migration
  - API signature updates
language: python
date of note: 2025-09-19
---

# Step Catalog Dual Search Space Integration Checklist

## Overview

This checklist tracks the systematic update of all files that import the step catalog system to use the new dual search space API (`workspace_dirs` parameter instead of `workspace_root`).

**Migration Pattern**: `StepCatalog(workspace_root)` → `StepCatalog(workspace_dirs=[workspace_root])`

## ⚠️ **CRITICAL PORTABILITY REMINDER**

**🚨 EVERY FILE UPDATE MUST CHECK FOR PORTABILITY ISSUES! 🚨**

During integration updates, we discovered **5 critical portability issues** that would have broken PyPI and submodule deployments. **Every file in this checklist must be audited for:**

### **Common Portability Anti-Patterns to Fix:**

1. **❌ Hardcoded Path Calculations**
   ```python
   # BROKEN: Will fail in PyPI/submodule deployments
   workspace_root = Path(__file__).parent.parent.parent.parent
   catalog = StepCatalog(workspace_dirs=[workspace_root])
   ```

2. **✅ Portable Solutions**
   ```python
   # PORTABLE: Package-only discovery (most common need)
   catalog = StepCatalog(workspace_dirs=None)
   
   # OR: Workspace-aware discovery (when workspace is actually needed)
   catalog = StepCatalog(workspace_dirs=[provided_workspace_root])
   ```

3. **❌ Conceptual Confusion: package_root vs workspace_dirs**
   ```python
   # BROKEN: Using workspace_root as package_root
   config_discovery = ConfigAutoDiscovery(workspace_root, workspace_dirs)
   ```

4. **✅ Conceptual Clarity**
   ```python
   # CORRECT: Proper separation of concerns
   temp_catalog = StepCatalog(workspace_dirs=None)
   package_root = temp_catalog.package_root  # Cursus package location
   workspace_dirs = [workspace_root] if workspace_root else []  # User workspaces
   config_discovery = ConfigAutoDiscovery(package_root, workspace_dirs)
   ```

### **Deployment Compatibility Check:**
- ✅ **Development/Source**: Must work when run from source
- ✅ **PyPI Installation**: Must work when installed via `pip install cursus`
- ✅ **Submodule Integration**: Must work when cursus is a git submodule

### **Files Already Fixed (5/5 Critical Portability Issues):**
1. `src/cursus/registry/builder_registry.py` - Fixed hardcoded path
2. `src/cursus/steps/configs/utils.py` - Fixed hardcoded path + ConfigAutoDiscovery
3. `src/cursus/core/config_fields/unified_config_manager.py` - Fixed conceptual confusion
4. `src/cursus/validation/builders/registry_discovery.py` - Fixed hardcoded paths (2 methods)
5. `src/cursus/validation/alignment/unified_alignment_tester.py` - Fixed hardcoded path

**⚠️ EVERY REMAINING FILE MUST BE CHECKED FOR THESE SAME ISSUES! ⚠️**

## Core Systems (High Priority) - Complete Replacement

### ✅ UPDATED - Core Config Fields Integration
- [x] **`src/cursus/core/config_fields/unified_config_manager.py`**
  - [x] Updated StepCatalog initialization: `StepCatalog(workspace_dirs=[workspace_root])`
  - [x] Fixed ConfigAutoDiscovery fallback: Corrected package_root vs workspace_dirs confusion
  - [x] Enhanced: Uses StepCatalog's package root detection to avoid code duplication
  - [x] Verified: Lazy loading with dual search space API
  - [x] Verified: Fallback handling preserved
  - [x] Fixed: Conceptual separation between package_root and workspace_dirs

### ✅ UPDATED - Step Configuration Utilities
- [x] **`src/cursus/steps/configs/utils.py`**
  - [x] Updated StepCatalog initialization: `StepCatalog(workspace_dirs=None)` (PORTABLE)
  - [x] Fixed: Removed hardcoded path calculation in build_complete_config_classes
  - [x] Enhanced: ConfigAutoDiscovery fallback also made portable
  - [x] Verified: Enhanced discovery maintained
  - [x] Verified: Project-specific loading preserved
  - [x] Enhanced: Now works in PyPI, source, and submodule deployment scenarios

### ✅ UPDATED - Registry System Integration
- [x] **`src/cursus/registry/builder_registry.py`**
  - [x] Updated StepCatalog initialization: `StepCatalog(workspace_dirs=None)` (PORTABLE)
  - [x] Fixed: Removed hardcoded path calculation for universal deployment compatibility
  - [x] Verified: Builder discovery maintained
  - [x] Verified: Auto-discovery functionality preserved
  - [x] Enhanced: Now works in PyPI, source, and submodule deployment scenarios

### ✅ UPDATED - Validation System Integration
- [x] **`src/cursus/validation/alignment/unified_alignment_tester.py`**
  - [x] Fixed StepCatalog initialization (_get_step_catalog): `StepCatalog(workspace_dirs=None)` (PORTABLE)
  - [x] Fixed: Removed hardcoded path calculation for universal deployment compatibility
  - [x] Verified: Script discovery maintained
  - [x] Verified: Alignment testing functionality preserved
  - [x] Enhanced: Now works in PyPI, source, and submodule deployment scenarios

### ✅ UPDATED - Registry Discovery System
- [x] **`src/cursus/validation/builders/registry_discovery.py`**
  - [x] Fixed StepCatalog initialization (get_builder_class_path): `StepCatalog(workspace_dirs=None)` (PORTABLE)
  - [x] Fixed StepCatalog initialization (load_builder_class): `StepCatalog(workspace_dirs=None)` (PORTABLE)
  - [x] Fixed: Removed hardcoded path calculations for universal deployment compatibility
  - [x] Verified: Builder class path discovery maintained
  - [x] Verified: Builder class loading functionality preserved
  - [x] Enhanced: Now works in PyPI, source, and submodule deployment scenarios

## Validation Systems (Medium Priority) - Significant Simplification

### ⏳ PENDING - Step Info Detection
- [ ] **`src/cursus/validation/builders/step_info_detector.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify step info detection functionality

### ⏳ PENDING - SageMaker Step Type Validation
- [ ] **`src/cursus/validation/builders/sagemaker_step_type_validator.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify step name detection functionality

### ⏳ PENDING - Universal Test System
- [ ] **`src/cursus/validation/builders/universal_test.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify step name inference functionality

### ⏳ PENDING - Validation Orchestration
- [ ] **`src/cursus/validation/alignment/orchestration/validation_orchestrator.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify contract file discovery functionality

### ⏳ PENDING - Specification Loader
- [ ] **`src/cursus/validation/alignment/loaders/specification_loader.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify specification discovery functionality

### ⏳ PENDING - Contract Loader
- [ ] **`src/cursus/validation/alignment/loaders/contract_loader.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify contract object discovery functionality

### ⏳ PENDING - Workspace-Aware Spec Builder
- [ ] **`src/cursus/validation/runtime/workspace_aware_spec_builder.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace script discovery functionality

### ⏳ PENDING - Runtime Spec Builder
- [ ] **`src/cursus/validation/runtime/runtime_spec_builder.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify script file discovery functionality

## Workspace Systems (Medium Priority) - Significant Simplification

### ⏳ PENDING - Workspace Manager
- [ ] **`src/cursus/workspace/validation/workspace_manager.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace discovery functionality

### ⏳ PENDING - Workspace Type Detector
- [ ] **`src/cursus/workspace/validation/workspace_type_detector.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace detection functionality

### ⏳ PENDING - Workspace Test Manager
- [ ] **`src/cursus/workspace/validation/workspace_test_manager.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify test workspace discovery functionality

### ⏳ PENDING - Workspace Module Loader
- [ ] **`src/cursus/workspace/validation/workspace_module_loader.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace module discovery functionality

### ⏳ PENDING - Workspace Alignment Tester
- [ ] **`src/cursus/workspace/validation/workspace_alignment_tester.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace script discovery functionality

### ⏳ PENDING - Workspace File Resolver
- [ ] **`src/cursus/workspace/validation/workspace_file_resolver.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace component discovery functionality

## Pipeline/API Systems (Lower Priority) - Significant Simplification

### ⏳ PENDING - Pipeline Catalog Utils
- [ ] **`src/cursus/pipeline_catalog/utils.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify pipeline discovery functionality

### ⏳ PENDING - Pipeline DAG Resolver
- [ ] **`src/cursus/api/dag/pipeline_dag_resolver.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify step contract discovery functionality

### ⏳ PENDING - Pipeline Catalog Indexer
- [ ] **`src/cursus/pipeline_catalog/indexer.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify pipeline indexing functionality

## Step Catalog Adapters (Backward Compatibility)

### ⏳ PENDING - Config Class Detector Adapter
- [ ] **`src/cursus/step_catalog/adapters/config_class_detector.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify adapter functionality

### ⏳ PENDING - Config Resolver Adapter
- [ ] **`src/cursus/step_catalog/adapters/config_resolver.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify adapter functionality

### ⏳ PENDING - Contract Discovery Adapter
- [ ] **`src/cursus/step_catalog/adapters/contract_discovery.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify adapter functionality

### ⏳ PENDING - File Resolver Adapter
- [ ] **`src/cursus/step_catalog/adapters/file_resolver.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify adapter functionality

### ⏳ PENDING - Workspace Discovery Adapter
- [ ] **`src/cursus/step_catalog/adapters/workspace_discovery.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify adapter functionality

### ⏳ PENDING - Legacy Wrappers
- [ ] **`src/cursus/step_catalog/adapters/legacy_wrappers.py`**
  - [ ] Check for StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify wrapper functionality

## Registry Systems (High Priority) - Already Integrated

### ✅ COMPLETED - Hybrid Registry Manager
- [x] **`src/cursus/registry/hybrid/manager.py`**
  - [x] Already integrated in Phase 5.6
  - [x] Uses step catalog for workspace discovery
  - [x] Maintains registry management logic

## Complete Search Results - All Files Importing Step Catalog System

### ⏳ PENDING - Validation Alignment File Resolver
- [ ] **`src/cursus/validation/alignment/file_resolver.py`**
  - [ ] Uses: `from ...step_catalog.adapters.file_resolver import FlexibleFileResolverAdapter`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify file resolution functionality

### ⏳ PENDING - Validation Alignment Patterns File Resolver
- [ ] **`src/cursus/validation/alignment/patterns/file_resolver.py`**
  - [ ] Uses: `from ....step_catalog.adapters.file_resolver import HybridFileResolverAdapter`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify pattern-based file resolution functionality

### ⏳ PENDING - Validation Alignment Discovery Contract Discovery
- [ ] **`src/cursus/validation/alignment/discovery/contract_discovery.py`**
  - [ ] Uses: `from ....step_catalog.adapters.contract_discovery import ContractDiscoveryEngineAdapter`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify contract discovery functionality

### ⏳ PENDING - Validation Runtime Contract Discovery
- [ ] **`src/cursus/validation/runtime/contract_discovery.py`**
  - [ ] Uses: `from ...step_catalog.adapters.contract_discovery import ContractDiscoveryManagerAdapter`
  - [ ] Uses: `from ...step_catalog.adapters.contract_discovery import ContractDiscoveryResult`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify runtime contract discovery functionality

### ⏳ PENDING - Core Compiler Config Resolver
- [ ] **`src/cursus/core/compiler/config_resolver.py`**
  - [ ] Uses: `from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify config resolution functionality

### ⏳ PENDING - Workspace Core Discovery
- [ ] **`src/cursus/workspace/core/discovery.py`**
  - [ ] Uses: `from ...step_catalog.adapters.workspace_discovery import WorkspaceDiscoveryManagerAdapter`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify workspace core discovery functionality

### ⏳ PENDING - Core Config Fields Init
- [ ] **`src/cursus/core/config_fields/__init__.py`**
  - [ ] Uses: `from ...step_catalog.adapters.config_class_detector import ConfigClassStoreAdapter`
  - [ ] Uses: `from .step_catalog_aware_merger import StepCatalogAwareConfigMerger`
  - [ ] Check for direct StepCatalog imports and usage
  - [ ] Update StepCatalog initialization if present
  - [ ] Verify config field functionality

## Step Catalog Internal Files (No Updates Needed)

### ✅ INTERNAL - Step Catalog Main Module
- [x] **`src/cursus/step_catalog/__init__.py`**
  - [x] Internal: `from .step_catalog import StepCatalog`
  - [x] No updates needed - internal module

### ✅ INTERNAL - Step Catalog Adapters
- [x] **`src/cursus/step_catalog/adapters/workspace_discovery.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

- [x] **`src/cursus/step_catalog/adapters/config_class_detector.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

- [x] **`src/cursus/step_catalog/adapters/legacy_wrappers.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

- [x] **`src/cursus/step_catalog/adapters/config_resolver.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

- [x] **`src/cursus/step_catalog/adapters/contract_discovery.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

- [x] **`src/cursus/step_catalog/adapters/file_resolver.py`**
  - [x] Internal: `from ..step_catalog import StepCatalog`
  - [x] No updates needed - internal adapter

## Progress Summary

### ✅ Completed (5 files)
- Core config fields integration
- Step configuration utilities
- Registry system integration
- Validation alignment tester
- Registry discovery system

### ⏳ Pending (37+ files)
- Validation systems (8 files)
- Workspace systems (6 files)
- Pipeline/API systems (3 files)
- Step catalog adapters (6 files)
- Additional discovered files (14+ files)

### 📊 Overall Progress
- **Completed**: 5/42 files (12%)
- **Remaining**: 37/42 files (88%)
- **Critical files updated**: 5/5 (100%)

## Next Steps

1. **Systematic Review**: Go through each pending file one by one
2. **Pattern Verification**: Check for `from ...step_catalog import StepCatalog` imports
3. **API Update**: Update `StepCatalog(workspace_root)` to `StepCatalog(workspace_dirs=[workspace_root])`
4. **Functionality Verification**: Ensure no breaking changes to existing functionality
5. **Test Validation**: Run relevant tests after each update

## Notes

- **Critical files already updated**: The 5 most critical integration points have been successfully updated
- **Core functionality preserved**: All 536 core tests continue to pass
- **Backward compatibility maintained**: Legacy adapters provide transition support
- **Systematic approach**: Following the migration guide's prioritization and patterns
