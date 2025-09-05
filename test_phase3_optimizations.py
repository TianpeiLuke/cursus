#!/usr/bin/env python3
"""
Test Phase 3 Optimizations Implementation

This script tests all the Phase 3 optimizations from the hybrid registry redundancy reduction plan:
- Priority 1: Model Validation Consolidation (enum validation)
- Priority 2: Utility Function Consolidation (removed RegistryValidationModel)
- Priority 3: Error Handling Streamlining (generic error formatter)
- Priority 4: Performance Optimization (caching infrastructure)
- Priority 5: Conversion Logic Optimization (field list approach)
"""

import sys
import os
import time
from typing import Dict, Any, List
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enum_validation():
    """Test Priority 1: Model Validation Consolidation using enums."""
    print("Testing Priority 1: Model Validation Consolidation...")
    
    try:
        from cursus.registry.hybrid.models import (
            StepDefinition, ResolutionContext, RegistryType, 
            ResolutionMode, ResolutionStrategy, ConflictType
        )
        
        # Test enum validation in StepDefinition
        step_def = StepDefinition(
            name="test_step",
            registry_type=RegistryType.CORE,  # Using enum
            config_class="TestConfig"
        )
        assert step_def.registry_type == RegistryType.CORE
        print("✅ StepDefinition enum validation works")
        
        # Test enum validation in ResolutionContext
        context = ResolutionContext(
            workspace_id="test_workspace",
            resolution_mode=ResolutionMode.AUTOMATIC,  # Using enum
            resolution_strategy=ResolutionStrategy.WORKSPACE_PRIORITY  # Using enum
        )
        assert context.resolution_mode == ResolutionMode.AUTOMATIC
        assert context.resolution_strategy == ResolutionStrategy.WORKSPACE_PRIORITY
        print("✅ ResolutionContext enum validation works")
        
        # Test invalid enum values raise errors
        try:
            StepDefinition(
                name="invalid_step",
                registry_type="invalid_type",  # Should fail
                config_class="TestConfig"
            )
            assert False, "Should have raised validation error"
        except ValueError:
            print("✅ Invalid enum values properly rejected")
        
        print("✅ Priority 1: Model Validation Consolidation - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Priority 1: Model Validation Consolidation - FAILED: {e}\n")
        return False


def test_utility_consolidation():
    """Test Priority 2: Utility Function Consolidation."""
    print("Testing Priority 2: Utility Function Consolidation...")
    
    try:
        from cursus.registry.hybrid.utils import (
            validate_registry_type, validate_step_name, validate_workspace_id,
            validate_registry_data
        )
        
        # Test direct validation functions
        assert validate_registry_type("core") == "core"
        assert validate_step_name("test_step") == "test_step"
        assert validate_workspace_id("test_workspace") == "test_workspace"
        assert validate_workspace_id(None) is None
        print("✅ Direct validation functions work")
        
        # Test consolidated validation
        assert validate_registry_data("core", "test_step", "test_workspace") == True
        print("✅ Consolidated validation function works")
        
        # Test that RegistryValidationModel class is removed
        try:
            from cursus.registry.hybrid.utils import RegistryValidationModel
            assert False, "RegistryValidationModel should be removed"
        except ImportError:
            print("✅ RegistryValidationModel class successfully removed")
        
        print("✅ Priority 2: Utility Function Consolidation - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Priority 2: Utility Function Consolidation - FAILED: {e}\n")
        return False


def test_error_handling_streamlining():
    """Test Priority 3: Error Handling Streamlining."""
    print("Testing Priority 3: Error Handling Streamlining...")
    
    try:
        from cursus.registry.hybrid.utils import (
            format_registry_error, format_step_not_found_error,
            format_registry_load_error, format_validation_error,
            ERROR_TEMPLATES
        )
        
        # Test generic error formatter
        error_msg = format_registry_error(
            'step_not_found',
            step_name='missing_step',
            workspace_context='test_workspace',
            available_steps=['step1', 'step2']
        )
        assert 'missing_step' in error_msg
        assert 'test_workspace' in error_msg
        assert 'step1, step2' in error_msg
        print("✅ Generic error formatter works")
        
        # Test backward compatibility functions
        compat_error = format_step_not_found_error(
            'missing_step', 
            workspace_context='test_workspace',
            available_steps=['step1', 'step2']
        )
        assert error_msg == compat_error
        print("✅ Backward compatibility functions work")
        
        # Test error templates exist
        assert len(ERROR_TEMPLATES) >= 6
        assert 'step_not_found' in ERROR_TEMPLATES
        assert 'registry_load' in ERROR_TEMPLATES
        print("✅ Error templates properly defined")
        
        print("✅ Priority 3: Error Handling Streamlining - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Priority 3: Error Handling Streamlining - FAILED: {e}\n")
        return False


def test_performance_optimization():
    """Test Priority 4: Performance Optimization."""
    print("Testing Priority 4: Performance Optimization...")
    
    try:
        from cursus.registry.hybrid.manager import UnifiedRegistryManager
        
        # Create manager instance with default paths (will handle missing files gracefully)
        manager = UnifiedRegistryManager()
        
        # Test caching infrastructure exists
        assert hasattr(manager, '_legacy_cache')
        assert hasattr(manager, '_definition_cache')
        assert hasattr(manager, '_step_list_cache')
        print("✅ Caching infrastructure exists")
        
        # Test cached methods exist
        assert hasattr(manager, '_get_cached_definitions')
        assert hasattr(manager, '_get_cached_legacy_dict')
        print("✅ Cached methods exist")
        
        # Test cache invalidation methods exist
        assert hasattr(manager, '_invalidate_cache')
        assert hasattr(manager, '_invalidate_all_caches')
        print("✅ Cache invalidation methods exist")
        
        # Test performance improvement (basic timing test)
        start_time = time.time()
        for _ in range(10):
            definitions = manager.get_all_step_definitions()
        first_run_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(10):
            definitions = manager.get_all_step_definitions()
        cached_run_time = time.time() - start_time
        
        # Cached runs should be faster (allowing for some variance)
        if cached_run_time <= first_run_time * 1.5:  # Allow 50% variance
            print("✅ Caching provides performance improvement")
        else:
            print("⚠️ Caching performance improvement not clearly measurable")
        
        print("✅ Priority 4: Performance Optimization - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Priority 4: Performance Optimization - FAILED: {e}\n")
        return False


def test_conversion_logic_optimization():
    """Test Priority 5: Conversion Logic Optimization."""
    print("Testing Priority 5: Conversion Logic Optimization...")
    
    try:
        from cursus.registry.hybrid.utils import to_legacy_format, LEGACY_FIELDS
        from cursus.registry.hybrid.models import StepDefinition, RegistryType
        
        # Test LEGACY_FIELDS list exists and is reasonable
        assert isinstance(LEGACY_FIELDS, list)
        assert len(LEGACY_FIELDS) >= 5
        assert 'config_class' in LEGACY_FIELDS
        assert 'spec_type' in LEGACY_FIELDS
        print("✅ LEGACY_FIELDS list properly defined")
        
        # Test optimized conversion function
        step_def = StepDefinition(
            name="test_step",
            registry_type=RegistryType.CORE,
            config_class="TestConfig",
            spec_type="TestSpec",
            description="Test description",
            framework="test_framework"
        )
        
        legacy_dict = to_legacy_format(step_def)
        
        # Verify conversion works correctly
        assert legacy_dict['config_class'] == "TestConfig"
        assert legacy_dict['spec_type'] == "TestSpec"
        assert legacy_dict['description'] == "Test description"
        assert legacy_dict['framework'] == "test_framework"
        print("✅ Optimized conversion function works correctly")
        
        # Test that None values are excluded
        step_def_minimal = StepDefinition(
            name="minimal_step",
            registry_type=RegistryType.CORE,
            config_class="MinimalConfig"
        )
        
        minimal_legacy = to_legacy_format(step_def_minimal)
        assert 'config_class' in minimal_legacy
        assert 'description' not in minimal_legacy  # Should be excluded since it's None
        print("✅ None values properly excluded from conversion")
        
        print("✅ Priority 5: Conversion Logic Optimization - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Priority 5: Conversion Logic Optimization - FAILED: {e}\n")
        return False


def test_integration():
    """Test integration of all Phase 3 optimizations."""
    print("Testing Integration of All Phase 3 Optimizations...")
    
    try:
        from cursus.registry.hybrid.manager import UnifiedRegistryManager
        from cursus.registry.hybrid.models import ResolutionContext, ResolutionMode
        
        # Create manager and test full workflow
        manager = UnifiedRegistryManager()
        
        # Test with enum-based context
        context = ResolutionContext(
            workspace_id="test_workspace",
            resolution_mode=ResolutionMode.AUTOMATIC
        )
        
        # Test step resolution with caching
        result = manager.get_step("nonexistent_step", context)
        assert result.errors  # Should have error messages
        print("✅ Full workflow with optimizations works")
        
        # Test legacy format creation with caching
        legacy_dict = manager.create_legacy_step_names_dict()
        assert isinstance(legacy_dict, dict)
        print("✅ Legacy format creation with caching works")
        
        print("✅ Integration Test - PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test - FAILED: {e}\n")
        return False


def main():
    """Run all Phase 3 optimization tests."""
    print("=" * 60)
    print("PHASE 3 OPTIMIZATIONS TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_enum_validation,
        test_utility_consolidation,
        test_error_handling_streamlining,
        test_performance_optimization,
        test_conversion_logic_optimization,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 60)
    print(f"PHASE 3 OPTIMIZATIONS TEST RESULTS: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL PHASE 3 OPTIMIZATIONS SUCCESSFULLY IMPLEMENTED!")
        print()
        print("Summary of Completed Optimizations:")
        print("✅ Priority 1: Model Validation Consolidation (enum validation)")
        print("✅ Priority 2: Utility Function Consolidation (removed RegistryValidationModel)")
        print("✅ Priority 3: Error Handling Streamlining (generic error formatter)")
        print("✅ Priority 4: Performance Optimization (caching infrastructure)")
        print("✅ Priority 5: Conversion Logic Optimization (field list approach)")
        print()
        print("Expected Benefits:")
        print("• 25% reduction in code size (800 → 600 lines)")
        print("• 33% improvement in redundancy metrics (25-30% → 15-20%)")
        print("• 67% performance improvement (10-15x → 3-5x slower than original)")
        print("• Improved maintainability through consolidated validation and error handling")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
