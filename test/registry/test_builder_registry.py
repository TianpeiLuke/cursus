"""Unit tests for the StepBuilderRegistry class."""

import unittest
import logging

from src.pipeline_registry.builder_registry import StepBuilderRegistry, get_global_registry
from src.pipeline_registry.step_names import STEP_NAMES, get_all_step_names
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.config_base import BasePipelineConfig


class TestBuilderRegistry(unittest.TestCase):
    """Test case for StepBuilderRegistry."""

    def setUp(self):
        """Set up test case."""
        logging.basicConfig(level=logging.INFO)
        self.registry = StepBuilderRegistry()

    def test_registry_initialization(self):
        """Test registry initialization."""
        self.assertGreater(len(self.registry.BUILDER_REGISTRY), 0)
        self.assertGreater(len(self.registry.LEGACY_ALIASES), 0)

    def test_canonical_step_names(self):
        """Test that canonical step names are properly mapped."""
        builder_map = self.registry.get_builder_map()
        
        # Test a few key canonical names from step_names.py
        self.assertIn("Package", builder_map)
        self.assertIn("Payload", builder_map)
        self.assertIn("Registration", builder_map)
        self.assertIn("PytorchTraining", builder_map)
        self.assertIn("PytorchModel", builder_map)
        
        # Verify legacy aliases are properly handled
        self.assertTrue(self.registry.is_step_type_supported("MIMSPackaging"))
        self.assertTrue(self.registry.is_step_type_supported("MIMSPayload"))
        self.assertTrue(self.registry.is_step_type_supported("PyTorchTraining"))
        self.assertTrue(self.registry.is_step_type_supported("PyTorchModel"))
        
    def test_config_class_to_step_type(self):
        """Test _config_class_to_step_type method."""
        # Test with config classes from step registry
        for step_name, info in STEP_NAMES.items():
            config_class = info["config_class"]
            step_type = self.registry._config_class_to_step_type(config_class)
            
            # Either the returned step type should be in the builder registry directly
            # or it should be a legacy alias that maps to a canonical name
            is_supported = (step_type in self.registry.get_builder_map() or 
                            step_type in self.registry.LEGACY_ALIASES)
            
            self.assertTrue(is_supported, 
                            f"Step type '{step_type}' from config class '{config_class}' not supported")
        
        # Test fallback for unknown config class
        unknown_step = self.registry._config_class_to_step_type("UnknownConfig")
        self.assertEqual(unknown_step, "Unknown")
    
    def test_get_config_types_for_step_type(self):
        """Test get_config_types_for_step_type method."""
        # Test with step types from registry
        for step_name in get_all_step_names():
            config_types = self.registry.get_config_types_for_step_type(step_name)
            self.assertGreater(len(config_types), 0)
        
        # Test with legacy aliases
        for legacy_name in self.registry.LEGACY_ALIASES:
            config_types = self.registry.get_config_types_for_step_type(legacy_name)
            self.assertGreater(len(config_types), 0)
    
    def test_validate_registry(self):
        """Test validate_registry method."""
        validation = self.registry.validate_registry()
        
        # Should have valid entries
        self.assertGreater(len(validation['valid']), 0)
        
        # Print any invalid entries
        if validation.get('invalid'):
            logging.warning(f"Invalid registry entries: {validation['invalid']}")
        
        # Print any missing entries
        if validation.get('missing'):
            logging.warning(f"Missing registry entries: {validation['missing']}")
    
    def test_global_registry_singleton(self):
        """Test that the global registry is a singleton."""
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        self.assertIs(reg1, reg2)
    


if __name__ == '__main__':
    unittest.main()
