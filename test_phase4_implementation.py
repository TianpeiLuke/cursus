#!/usr/bin/env python3
"""
Test script for Phase 4: Base Class Integration implementation.

This script tests the workspace-aware enhancements to:
1. StepBuilderBase STEP_NAMES property
2. BasePipelineConfig _get_step_registry method
3. RegistryManager workspace awareness

Usage:
    python test_phase4_implementation.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_step_builder_workspace_context():
    """Test StepBuilderBase workspace context extraction."""
    print("\n=== Testing StepBuilderBase Workspace Context ===")
    
    try:
        from cursus.core.base.config_base import BasePipelineConfig
        from cursus.core.base.builder_base import StepBuilderBase
        
        # Create a mock config with workspace context
        class MockConfig(BasePipelineConfig):
            def __init__(self, **kwargs):
                # Set required fields with defaults for testing
                defaults = {
                    'author': 'test-author',
                    'bucket': 'test-bucket',
                    'role': 'test-role',
                    'region': 'NA',
                    'service_name': 'test-service',
                    'pipeline_version': '1.0.0',
                    'workspace_context': 'test-workspace'
                }
                defaults.update(kwargs)
                super().__init__(**defaults)
        
        # Create a mock step builder
        class MockStepBuilder(StepBuilderBase):
            def validate_configuration(self):
                pass
            
            def _get_inputs(self, inputs):
                return []
            
            def _get_outputs(self, outputs):
                return []
            
            def create_step(self, **kwargs):
                pass
        
        # Test workspace context extraction
        config = MockConfig()
        builder = MockStepBuilder(config)
        
        workspace_context = builder._get_workspace_context()
        print(f"✓ Extracted workspace context: {workspace_context}")
        
        # Test STEP_NAMES property (should not fail even if hybrid registry is not available)
        step_names = builder.STEP_NAMES
        print(f"✓ STEP_NAMES property accessed successfully: {type(step_names)}")
        
        return True
        
    except Exception as e:
        print(f"✗ StepBuilderBase test failed: {e}")
        return False

def test_config_workspace_registry():
    """Test BasePipelineConfig workspace-aware registry."""
    print("\n=== Testing BasePipelineConfig Workspace Registry ===")
    
    try:
        from cursus.core.base.config_base import BasePipelineConfig
        
        # Test workspace-aware step registry
        registry = BasePipelineConfig._get_step_registry("test-workspace")
        print(f"✓ Workspace-aware registry accessed: {type(registry)}")
        
        # Test default registry
        default_registry = BasePipelineConfig._get_step_registry()
        print(f"✓ Default registry accessed: {type(default_registry)}")
        
        return True
        
    except Exception as e:
        print(f"✗ BasePipelineConfig test failed: {e}")
        return False

def test_registry_manager_workspace():
    """Test RegistryManager workspace awareness."""
    print("\n=== Testing RegistryManager Workspace Awareness ===")
    
    try:
        from cursus.core.deps.registry_manager import RegistryManager
        
        # Test workspace-aware registry manager
        manager = RegistryManager(workspace_context="test-workspace")
        print(f"✓ Created workspace-aware registry manager: {manager}")
        
        # Test getting workspace-aware registry
        registry = manager.get_registry("test-context")
        print(f"✓ Got workspace-aware registry: {registry}")
        
        # Test workspace-aware context naming
        context_name = manager._get_workspace_aware_context_name("test-context")
        print(f"✓ Workspace-aware context name: {context_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ RegistryManager test failed: {e}")
        return False

def test_environment_variable_workspace():
    """Test environment variable workspace context."""
    print("\n=== Testing Environment Variable Workspace Context ===")
    
    try:
        from cursus.core.base.config_base import BasePipelineConfig
        from cursus.core.base.builder_base import StepBuilderBase
        
        # Set environment variable
        os.environ['CURSUS_WORKSPACE_CONTEXT'] = 'env-test-workspace'
        
        # Create mock classes
        class MockConfig(BasePipelineConfig):
            def __init__(self, **kwargs):
                defaults = {
                    'author': 'test-author',
                    'bucket': 'test-bucket', 
                    'role': 'test-role',
                    'region': 'NA',
                    'service_name': 'test-service',
                    'pipeline_version': '1.0.0'
                }
                defaults.update(kwargs)
                super().__init__(**defaults)
        
        class MockStepBuilder(StepBuilderBase):
            def validate_configuration(self):
                pass
            def _get_inputs(self, inputs):
                return []
            def _get_outputs(self, outputs):
                return []
            def create_step(self, **kwargs):
                pass
        
        # Test environment variable extraction
        config = MockConfig()
        builder = MockStepBuilder(config)
        
        workspace_context = builder._get_workspace_context()
        print(f"✓ Environment variable workspace context: {workspace_context}")
        
        # Clean up
        del os.environ['CURSUS_WORKSPACE_CONTEXT']
        
        return workspace_context == 'env-test-workspace'
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        return False

def main():
    """Run all Phase 4 tests."""
    print("Phase 4: Base Class Integration - Test Suite")
    print("=" * 50)
    
    tests = [
        test_step_builder_workspace_context,
        test_config_workspace_registry,
        test_registry_manager_workspace,
        test_environment_variable_workspace
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 4 tests passed!")
        return 0
    else:
        print("❌ Some Phase 4 tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
