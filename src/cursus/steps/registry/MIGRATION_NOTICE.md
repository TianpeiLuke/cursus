# Registry Location Migration Notice

## Overview
The registry system has been moved from `src/cursus/steps/registry/` to `src/cursus/registry/` to better align with the hybrid registry architecture and improve code organization.

## What Changed

### File Locations
- **OLD**: `src/cursus/steps/registry/`
- **NEW**: `src/cursus/registry/`

All registry files have been moved:
- `step_names.py`
- `builder_registry.py` 
- `hyperparameter_registry.py`
- `exceptions.py`
- `step_type_test_variants.py`
- `__init__.py`

### Import Changes
- **OLD**: `from cursus.steps.registry import STEP_NAMES`
- **NEW**: `from cursus.registry import STEP_NAMES`

## Backward Compatibility

Old import paths continue to work via a compatibility shim, but will issue deprecation warnings:

```python
# This still works but shows a deprecation warning
from cursus.steps.registry import STEP_NAMES

# Preferred new import
from cursus.registry import STEP_NAMES
```

## Migration Timeline

- **Phase 0 (Week 0)**: ✅ File migration and import updates completed
- **Future**: Compatibility shim removal (TBD)

## Action Required

### For Developers
1. Update your imports to use the new location: `cursus.registry`
2. Update any documentation or scripts that reference the old location
3. Test your code to ensure it works with the new imports

### For New Code
Always use the new import paths: `from cursus.registry import ...`

## Validation

The migration has been validated to ensure:
- ✅ All registry files moved to new location
- ✅ New import paths working correctly  
- ✅ Backward compatibility maintained
- ✅ Registry functionality preserved
- ✅ 18 steps found in registry

## Support

If you encounter any issues with the registry migration:
1. Check that you're using the correct import paths
2. Verify the registry files exist in `src/cursus/registry/`
3. Run the validation script: `python validate_registry_migration.py`
4. Contact the development team for assistance

## Related Documentation

This migration is part of Phase 0 of the larger [Hybrid Registry Migration Plan](../../../slipbox/2_project_planning/2025-09-02_workspace_aware_hybrid_registry_migration_plan.md).
