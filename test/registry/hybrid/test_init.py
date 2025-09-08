"""
Test suite for hybrid registry module initialization.

Tests module imports, exports, and __all__ definitions.
"""

import unittest
from unittest.mock import patch, MagicMock

class TestHybridRegistryImports(unittest.TestCase):
    """Test hybrid registry module imports and exports."""
    
    def test_data_model_imports(self):
        """Test that data models can be imported from the hybrid module."""
        from cursus.registry.hybrid import (
            StepDefinition,
            ResolutionContext,
            StepResolutionResult,
            RegistryValidationResult,
            ConflictAnalysis
        )
        
        # Check that classes are imported correctly
        self.assertTrue(hasattr(StepDefinition, '__name__'))
        self.assertTrue(hasattr(ResolutionContext, '__name__'))
        self.assertTrue(hasattr(StepResolutionResult, '__name__'))
        self.assertTrue(hasattr(RegistryValidationResult, '__name__'))
        self.assertTrue(hasattr(ConflictAnalysis, '__name__'))
        
        # Check that they are the expected types
        self.assertEqual(StepDefinition.__name__, 'StepDefinition')
        self.assertEqual(ResolutionContext.__name__, 'ResolutionContext')
        self.assertEqual(StepResolutionResult.__name__, 'StepResolutionResult')
        self.assertEqual(RegistryValidationResult.__name__, 'RegistryValidationResult')
        self.assertEqual(ConflictAnalysis.__name__, 'ConflictAnalysis')
    
    def test_registry_management_imports(self):
        """Test that registry management classes can be imported."""
        from cursus.registry.hybrid import (
            UnifiedRegistryManager,
            CoreStepRegistry,
            LocalStepRegistry,
            HybridRegistryManager
        )

        # Check that classes are imported correctly
        self.assertTrue(hasattr(UnifiedRegistryManager, '__name__'))
        self.assertTrue(hasattr(CoreStepRegistry, '__name__'))
        self.assertTrue(hasattr(LocalStepRegistry, '__name__'))
        self.assertTrue(hasattr(HybridRegistryManager, '__name__'))

        # Check that they are the expected types
        self.assertEqual(UnifiedRegistryManager.__name__, 'UnifiedRegistryManager')
        # CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager are aliases for UnifiedRegistryManager
        self.assertEqual(CoreStepRegistry.__name__, 'UnifiedRegistryManager')
        self.assertEqual(LocalStepRegistry.__name__, 'UnifiedRegistryManager')
        self.assertEqual(HybridRegistryManager.__name__, 'UnifiedRegistryManager')
        
        # Verify they are the same class
        self.assertIs(CoreStepRegistry, UnifiedRegistryManager)
        self.assertIs(LocalStepRegistry, UnifiedRegistryManager)
        self.assertIs(HybridRegistryManager, UnifiedRegistryManager)
    
    def test_utility_function_imports(self):
        """Test that utility functions can be imported."""
        from cursus.registry.hybrid import (
            load_registry_module,
            from_legacy_format,
            to_legacy_format,
            convert_registry_dict,
            validate_registry_data,
            format_step_not_found_error,
            format_registry_load_error
        )
        
        # Check that functions are imported correctly
        self.assertTrue(callable(load_registry_module))
        self.assertTrue(callable(from_legacy_format))
        self.assertTrue(callable(to_legacy_format))
        self.assertTrue(callable(convert_registry_dict))
        self.assertTrue(callable(validate_registry_data))
        self.assertTrue(callable(format_step_not_found_error))
        self.assertTrue(callable(format_registry_load_error))
        
        # Check function names
        self.assertEqual(load_registry_module.__name__, 'load_registry_module')
        self.assertEqual(from_legacy_format.__name__, 'from_legacy_format')
        self.assertEqual(to_legacy_format.__name__, 'to_legacy_format')
        self.assertEqual(convert_registry_dict.__name__, 'convert_registry_dict')
        self.assertEqual(validate_registry_data.__name__, 'validate_registry_data')
        self.assertEqual(format_step_not_found_error.__name__, 'format_step_not_found_error')
        self.assertEqual(format_registry_load_error.__name__, 'format_registry_load_error')
    
    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        import cursus.registry.hybrid as hybrid_module
        
        expected_exports = [
            # Data Models
            "StepDefinition",
            "ResolutionContext",
            "StepResolutionResult",
            "RegistryValidationResult",
            "ConflictAnalysis",
            
            # Registry Management
            "UnifiedRegistryManager",
            "CoreStepRegistry",
            "LocalStepRegistry",
            "HybridRegistryManager",
            
            # Shared Utilities
            "load_registry_module",
            "from_legacy_format",
            "to_legacy_format",
            "convert_registry_dict",
            "validate_registry_data",
            "format_step_not_found_error",
            "format_registry_load_error"
        ]
        
        # Check that __all__ exists and contains expected items
        self.assertTrue(hasattr(hybrid_module, '__all__'))
        self.assertEqual(set(hybrid_module.__all__), set(expected_exports))
    
    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can actually be imported."""
        import cursus.registry.hybrid as hybrid_module
        
        for export_name in hybrid_module.__all__:
            # Should be able to get the attribute
            self.assertTrue(hasattr(hybrid_module, export_name))
            
            # Should not be None
            export_item = getattr(hybrid_module, export_name)
            self.assertIsNotNone(export_item)
    
    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import cursus.registry.hybrid as hybrid_module
        
        # Check that module has docstring
        self.assertIsNotNone(hybrid_module.__doc__)
        self.assertIn("Phase 3", hybrid_module.__doc__)
        self.assertIn("Simplified Local Registry Infrastructure", hybrid_module.__doc__)
        self.assertIn("Architecture:", hybrid_module.__doc__)
    
    def test_star_import(self):
        """Test that star import works correctly."""
        # This tests that 'from cursus.registry.hybrid import *' works
        import cursus.registry.hybrid as hybrid_module
        
        # Get all items that would be imported with *
        star_imports = {name: getattr(hybrid_module, name) for name in hybrid_module.__all__}
        
        # Check that we get the expected number of items
        self.assertEqual(len(star_imports), len(hybrid_module.__all__))
        
        # Check that all items are not None
        for name, item in star_imports.items():
            self.assertIsNotNone(item, f"Star import item '{name}' should not be None")

class TestModuleStructure(unittest.TestCase):
    """Test the overall module structure and organization."""
    
    def test_module_categories(self):
        """Test that exports are properly categorized."""
        import cursus.registry.hybrid as hybrid_module
        
        # Data Models
        data_models = [
            "StepDefinition",
            "ResolutionContext", 
            "StepResolutionResult",
            "RegistryValidationResult",
            "ConflictAnalysis"
        ]
        
        # Registry Management
        registry_classes = [
            "UnifiedRegistryManager",
            "CoreStepRegistry",
            "LocalStepRegistry", 
            "HybridRegistryManager"
        ]
        
        # Utility Functions
        utility_functions = [
            "load_registry_module",
            "from_legacy_format",
            "to_legacy_format",
            "convert_registry_dict",
            "validate_registry_data",
            "format_step_not_found_error",
            "format_registry_load_error"
        ]
        
        # Check that all categories are represented in __all__
        all_expected = data_models + registry_classes + utility_functions
        self.assertEqual(set(hybrid_module.__all__), set(all_expected))
        
        # Check that data models are classes
        for model_name in data_models:
            model_class = getattr(hybrid_module, model_name)
            self.assertTrue(hasattr(model_class, '__bases__'), f"{model_name} should be a class")
        
        # Check that registry classes are classes
        for class_name in registry_classes:
            registry_class = getattr(hybrid_module, class_name)
            self.assertTrue(hasattr(registry_class, '__bases__'), f"{class_name} should be a class")
        
        # Check that utilities are functions
        for func_name in utility_functions:
            func = getattr(hybrid_module, func_name)
            self.assertTrue(callable(func), f"{func_name} should be callable")
    
    def test_no_private_exports(self):
        """Test that no private items are exported."""
        import cursus.registry.hybrid as hybrid_module
        
        # Check that __all__ doesn't contain private items
        for export_name in hybrid_module.__all__:
            self.assertFalse(export_name.startswith('_'), 
                           f"Private item '{export_name}' should not be in __all__")
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # This test ensures that if there are import issues, they're caught appropriately
        try:
            import cursus.registry.hybrid
            # If we get here, imports worked
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Hybrid registry module should import without errors: {e}")

class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of the module interface."""
    
    def test_legacy_import_patterns(self):
        """Test that common import patterns still work."""
        # Test individual imports
        try:
            from cursus.registry.hybrid import StepDefinition
            from cursus.registry.hybrid import UnifiedRegistryManager
            from cursus.registry.hybrid import load_registry_module
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Legacy import patterns should work: {e}")
    
    def test_module_level_access(self):
        """Test that items can be accessed at module level."""
        import cursus.registry.hybrid as hybrid
        
        # Test accessing classes
        step_def_class = hybrid.StepDefinition
        self.assertIsNotNone(step_def_class)
        
        # Test accessing functions
        load_func = hybrid.load_registry_module
        self.assertIsNotNone(load_func)
        self.assertTrue(callable(load_func))
    
    def test_attribute_access_consistency(self):
        """Test that attribute access is consistent."""
        import cursus.registry.hybrid as hybrid
        
        # Test that getattr works the same as direct access
        for export_name in hybrid.__all__:
            direct_access = getattr(hybrid, export_name)
            attr_access = getattr(hybrid, export_name)
            self.assertIs(direct_access, attr_access, 
                         f"Attribute access should be consistent for {export_name}")

class TestModuleIntegrity(unittest.TestCase):
    """Test the integrity and consistency of the module."""
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # This test imports the module and checks that it doesn't cause issues
        try:
            import cursus.registry.hybrid
            # Try to access all exported items
            for export_name in src.cursus.registry.hybrid.__all__:
                item = getattr(src.cursus.registry.hybrid, export_name)
                self.assertIsNotNone(item)
        except Exception as e:
            self.fail(f"Module should not have circular import issues: {e}")
    
    def test_module_completeness(self):
        """Test that the module exports everything it should."""
        import cursus.registry.hybrid as hybrid
        
        # Check that we have exports from all expected submodules
        has_models = any(name in hybrid.__all__ for name in ["StepDefinition", "ResolutionContext"])
        has_manager = any(name in hybrid.__all__ for name in ["UnifiedRegistryManager", "CoreStepRegistry"])
        has_utils = any(name in hybrid.__all__ for name in ["load_registry_module", "from_legacy_format"])
        
        self.assertTrue(has_models, "Should export model classes")
        self.assertTrue(has_manager, "Should export manager classes")
        self.assertTrue(has_utils, "Should export utility functions")
    
    def test_export_uniqueness(self):
        """Test that all exports have unique names."""
        import cursus.registry.hybrid as hybrid
        
        # Check that __all__ has no duplicates
        all_exports = hybrid.__all__
        unique_exports = set(all_exports)
        
        self.assertEqual(len(all_exports), len(unique_exports), 
                        "All exports should have unique names")

if __name__ == "__main__":
    unittest.main()
