# Step Catalog Test Organization

## Current Test Coverage Analysis

### ✅ Existing Tests (with corresponding source modules)

| Source Module | Test File | Status |
|---------------|-----------|---------|
| `step_catalog.py` | `test_step_catalog.py` | ✅ Complete |
| `builder_discovery.py` | `test_builder_discovery.py` | ✅ Complete |
| `config_discovery.py` | `test_config_discovery.py` | ✅ Complete |
| `contract_discovery.py` | `test_contract_discovery.py` | ✅ Complete |
| `mapping.py` | `test_mapping.py` | ✅ Complete |
| `models.py` | `test_models.py` | ✅ Complete |
| `spec_discovery.py` | `test_spec_discovery.py` | ✅ **NEWLY CREATED** |

### ❌ Missing Tests (need to be created)

#### Main Directory Missing Tests
None - all main modules have tests.

#### Adapters Directory Missing Tests
| Source Module | Missing Test File | Priority |
|---------------|------------------|----------|
| `adapters/config_class_detector.py` | `adapters/test_config_class_detector.py` | High |
| `adapters/config_resolver.py` | `adapters/test_config_resolver.py` | High |
| `adapters/contract_adapter.py` | `adapters/test_contract_adapter.py` | High |
| `adapters/file_resolver.py` | `adapters/test_file_resolver.py` | High |
| `adapters/legacy_wrappers.py` | `adapters/test_legacy_wrappers.py` | High |
| `adapters/workspace_discovery.py` | `adapters/test_workspace_discovery.py` | High |

### 📁 Directory Structure Comparison

#### Source Structure:
```
src/cursus/step_catalog/
├── __init__.py
├── step_catalog.py
├── builder_discovery.py
├── config_discovery.py
├── contract_discovery.py
├── mapping.py
├── models.py
├── spec_discovery.py
└── adapters/
    ├── __init__.py
    ├── config_class_detector.py
    ├── config_resolver.py
    ├── contract_adapter.py
    ├── file_resolver.py
    ├── legacy_wrappers.py
    └── workspace_discovery.py
```

#### Test Structure (Current):
```
test/step_catalog/
├── __init__.py
├── test_step_catalog.py ✅
├── test_builder_discovery.py ✅
├── test_config_discovery.py ✅
├── test_contract_discovery.py ✅
├── test_contract_discovery_debugging.py ✅
├── test_mapping.py ✅
├── test_models.py ✅
├── test_spec_discovery.py ✅ (NEWLY CREATED)
├── test_adapters.py ⚠️ (Generic - should be reorganized)
├── test_dual_search_space.py ⚠️ (Integration test)
├── test_expanded_discovery.py ⚠️ (Integration test)
├── test_integration.py ✅ (Integration test)
└── adapters/
    ├── __init__.py ✅ (NEWLY CREATED)
    ├── test_config_class_detector.py ❌ (MISSING)
    ├── test_config_resolver.py ❌ (MISSING)
    ├── test_contract_adapter.py ❌ (MISSING)
    ├── test_file_resolver.py ❌ (MISSING)
    ├── test_legacy_wrappers.py ❌ (MISSING)
    └── test_workspace_discovery.py ❌ (MISSING)
```

## Action Items

### ✅ Completed
1. Created `test_spec_discovery.py` with comprehensive tests
2. Created `test/step_catalog/adapters/__init__.py`
3. Analyzed complete test coverage

### 🔄 Next Steps
1. Create missing adapter test files (6 files)
2. Reorganize `test_adapters.py` content into specific adapter tests
3. Verify all tests pass with real implementations
4. Update integration tests if needed

### 📊 Test Coverage Summary
- **Total Source Modules**: 13 (7 main + 6 adapters)
- **Existing Tests**: 7 main modules (100% coverage)
- **Missing Tests**: 6 adapter modules (0% coverage)
- **Overall Coverage**: 54% (7/13 modules)

### 🎯 Goal
Achieve 100% test coverage by creating the 6 missing adapter test files.
