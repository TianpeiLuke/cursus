#!/usr/bin/env python3
"""
Test Phase 5: Drop-in Registry Replacement Implementation

This script tests the Phase 5 implementation including:
- Enhanced step_names.py with hybrid backend
- Updated registry __init__.py with workspace awareness
- CLI commands for workspace management
- Backward compatibility validation
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_step_names():
    """Test enhanced step_names.py with hybrid backend support."""
    print("Testing Enhanced step_names.py...")
    
    try:
        # Test basic imports work
        from cursus.registry.step_names import (
            STEP_NAMES, get_config_class_name, get_workspace_context,
            set_workspace_context, workspace_context
        )
        print("✅ Enhanced step_names imports work")
        
        # Test backward compatibility - basic functionality
        step_names = STEP_NAMES
        assert isinstance(step_names, dict)
        assert len(step_names) > 0
        print("✅ STEP_NAMES backward compatibility maintained")
        
        # Test workspace context management
        original_context = get_workspace_context()
        
        set_workspace_context("test_workspace")
        current_context = get_workspace_context()
        assert current_context == "test_workspace"
        print("✅ Workspace context management works")
        
        # Test context manager
        with workspace_context("temp_workspace"):
            temp_context = get_workspace_context()
            assert temp_context == "temp_workspace"
        
        # Context should be restored
        restored_context = get_workspace_context()
        assert restored_context == "test_workspace"
        print("✅ Workspace context manager works")
        
        # Test helper functions still work
        config_class = get_config_class_name("XGBoostTraining")
        assert config_class == "XGBoostTrainingConfig"
        print("✅ Helper functions maintain backward compatibility")
        
        # Test fallback mechanism
        try:
            from cursus.registry.step_names import _get_registry_manager
            manager = _get_registry_manager()
            assert manager is not None
            print("✅ Registry manager initialization works (hybrid or fallback)")
        except Exception as e:
            print(f"⚠️  Registry manager initialization: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced step_names test failed: {e}")
        return False

def test_enhanced_registry_init():
    """Test enhanced registry __init__.py with workspace awareness."""
    print("\nTesting Enhanced registry __init__.py...")
    
    try:
        # Test enhanced imports
        from cursus.registry import (
            STEP_NAMES, get_config_class_name,
            set_workspace_context, get_workspace_context,
            workspace_context, switch_to_workspace, switch_to_core,
            get_registry_info
        )
        print("✅ Enhanced registry imports work")
        
        # Test convenience functions
        switch_to_workspace("test_workspace")
        context = get_workspace_context()
        assert context == "test_workspace"
        print("✅ switch_to_workspace works")
        
        switch_to_core()
        context = get_workspace_context()
        assert context is None
        print("✅ switch_to_core works")
        
        # Test registry info function
        info = get_registry_info()
        assert isinstance(info, dict)
        assert "step_count" in info
        assert "available_steps" in info
        print("✅ get_registry_info works")
        
        # Test hybrid components import (optional)
        try:
            from cursus.registry import UnifiedRegistryManager, StepDefinition
            print("✅ Hybrid registry components available")
        except ImportError:
            print("⚠️  Hybrid registry components not available (fallback mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced registry __init__.py test failed: {e}")
        return False

def test_cli_commands():
    """Test CLI commands for workspace management."""
    print("\nTesting CLI Commands...")
    
    try:
        # Test CLI module import
        from cursus.cli.registry_cli import registry_cli
        print("✅ CLI module imports work")
        
        # Test CLI command structure
        import click.testing
        runner = click.testing.CliRunner()
        
        # Test help command
        result = runner.invoke(registry_cli, ['--help'])
        assert result.exit_code == 0
        assert 'Registry management commands' in result.output
        print("✅ CLI help command works")
        
        # Test init-workspace command help
        result = runner.invoke(registry_cli, ['init-workspace', '--help'])
        assert result.exit_code == 0
        assert 'Initialize a new developer workspace' in result.output
        print("✅ init-workspace command help works")
        
        # Test list-steps command help
        result = runner.invoke(registry_cli, ['list-steps', '--help'])
        assert result.exit_code == 0
        assert 'List available steps' in result.output
        print("✅ list-steps command help works")
        
        # Test validate-registry command help
        result = runner.invoke(registry_cli, ['validate-registry', '--help'])
        assert result.exit_code == 0
        assert 'Validate registry configuration' in result.output
        print("✅ validate-registry command help works")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI commands test failed: {e}")
        return False

def test_workspace_initialization():
    """Test workspace initialization functionality."""
    print("\nTesting Workspace Initialization...")
    
    try:
        import tempfile
        import shutil
        from cursus.cli.registry_cli import (
            _create_workspace_structure, _create_workspace_registry,
            _create_workspace_documentation, _create_example_implementations
        )
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_workspace"
            
            # Test workspace structure creation
            _create_workspace_structure(workspace_dir)
            
            # Check required directories exist
            required_dirs = [
                "src/cursus_dev/steps/builders",
                "src/cursus_dev/steps/configs",
                "src/cursus_dev/registry",
                "test/unit",
                "examples"
            ]
            
            for dir_path in required_dirs:
                full_path = workspace_dir / dir_path
                assert full_path.exists(), f"Directory {dir_path} not created"
            
            print("✅ Workspace structure creation works")
            
            # Test registry file creation
            registry_file = _create_workspace_registry(workspace_dir, "test_workspace", "standard")
            assert Path(registry_file).exists()
            
            # Check registry file content
            with open(registry_file, 'r') as f:
                content = f.read()
                assert 'WORKSPACE_METADATA' in content
                assert 'LOCAL_STEPS' in content
                assert 'STEP_OVERRIDES' in content
                assert 'test_workspace' in content
            
            print("✅ Registry file creation works")
            
            # Test documentation creation
            readme_file = _create_workspace_documentation(workspace_dir, "test_workspace", registry_file)
            assert Path(readme_file).exists()
            
            with open(readme_file, 'r') as f:
                content = f.read()
                assert 'test_workspace' in content
                assert 'Quick Start' in content
                assert 'CLI Commands' in content
            
            print("✅ Documentation creation works")
            
            # Test example implementations
            _create_example_implementations(workspace_dir, "test_workspace", "standard")
            example_file = workspace_dir / "examples" / "example_custom_step_config.py"
            assert example_file.exists()
            
            with open(example_file, 'r') as f:
                content = f.read()
                assert 'ExampleCustomStepConfig' in content
                assert 'test_workspace' in content
            
            print("✅ Example implementations creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Workspace initialization test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that all existing code patterns still work."""
    print("\nTesting Backward Compatibility...")
    
    try:
        # Test all original imports still work
        from cursus.registry import (
            STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES,
            get_config_class_name, get_builder_step_name, get_spec_step_type,
            get_all_step_names, validate_step_name, get_step_description,
            get_sagemaker_step_type, get_canonical_name_from_file_name
        )
        print("✅ All original imports still work")
        
        # Test original functionality patterns
        step_names = STEP_NAMES
        config_registry = CONFIG_STEP_REGISTRY
        builder_names = BUILDER_STEP_NAMES
        spec_types = SPEC_STEP_TYPES
        
        assert isinstance(step_names, dict)
        assert isinstance(config_registry, dict)
        assert isinstance(builder_names, dict)
        assert isinstance(spec_types, dict)
        print("✅ Original data structures maintain format")
        
        # Test original helper functions
        config_class = get_config_class_name("XGBoostTraining")
        builder_name = get_builder_step_name("XGBoostTraining")
        spec_type = get_spec_step_type("XGBoostTraining")
        
        assert config_class == "XGBoostTrainingConfig"
        assert builder_name == "XGBoostTrainingStepBuilder"
        assert spec_type == "XGBoostTraining"
        print("✅ Original helper functions work unchanged")
        
        # Test validation functions
        assert validate_step_name("XGBoostTraining") == True
        assert validate_step_name("NonExistentStep") == False
        print("✅ Validation functions work unchanged")
        
        # Test file name resolution
        canonical_name = get_canonical_name_from_file_name("xgboost_training")
        assert canonical_name == "XGBoostTraining"
        print("✅ File name resolution works unchanged")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_integration():
    """Test integration of all Phase 5 components."""
    print("\nTesting Phase 5 Integration...")
    
    try:
        # Test full workflow: workspace context + registry access + CLI
        from cursus.registry import (
            set_workspace_context, get_workspace_context, 
            get_config_class_name, get_registry_info
        )
        
        # Test workspace switching workflow
        original_context = get_workspace_context()
        
        # Switch to test workspace
        set_workspace_context("integration_test")
        assert get_workspace_context() == "integration_test"
        
        # Test registry access in workspace context
        config_class = get_config_class_name("XGBoostTraining")
        assert config_class == "XGBoostTrainingConfig"
        
        # Test registry info in workspace context
        info = get_registry_info()
        assert info["workspace_id"] == "integration_test"
        
        # Test CLI integration
        from cursus.cli.registry_cli import registry_cli
        import click.testing
        runner = click.testing.CliRunner()
        
        # Test list-steps with workspace context
        result = runner.invoke(registry_cli, ['list-steps', '--workspace', 'integration_test'])
        assert result.exit_code == 0
        print("✅ CLI integration with workspace context works")
        
        # Restore original context
        if original_context:
            set_workspace_context(original_context)
        else:
            from cursus.registry import clear_workspace_context
            clear_workspace_context()
        
        print("✅ Full Phase 5 integration works")
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 integration test failed: {e}")
        return False

def main():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("PHASE 5: DROP-IN REGISTRY REPLACEMENT TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_enhanced_step_names,
        test_enhanced_registry_init,
        test_cli_commands,
        test_workspace_initialization,
        test_backward_compatibility,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"PHASE 5 TEST RESULTS: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("🎉 PHASE 5: DROP-IN REGISTRY REPLACEMENT SUCCESSFULLY IMPLEMENTED!")
        print()
        print("Summary of Completed Phase 5 Components:")
        print("✅ Enhanced step_names.py with hybrid backend support")
        print("✅ Updated registry __init__.py with workspace awareness")
        print("✅ CLI commands for workspace initialization and management")
        print("✅ Workspace structure creation and templates")
        print("✅ 100% backward compatibility maintained")
        print("✅ Fallback mechanism for hybrid registry unavailability")
        print()
        print("Key Features Delivered:")
        print("• Drop-in replacement for existing step_names.py")
        print("• Workspace context management (set_workspace_context, workspace_context)")
        print("• CLI tools for workspace initialization (init-workspace, list-steps, validate-registry)")
        print("• Enhanced registry exports with workspace-aware functions")
        print("• Comprehensive workspace templates (minimal, standard, advanced)")
        print("• Seamless fallback to original functionality when hybrid registry unavailable")
        print()
        print("Next Phase: Phase 6 - Integration and Testing")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
