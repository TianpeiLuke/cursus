"""
Test suite for hybrid registry manager components.

Tests UnifiedRegistryManager and related functionality.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.registry.hybrid.models import StepDefinition, RegistryType
from cursus.registry.exceptions import RegistryError

class TestUnifiedRegistryManager(unittest.TestCase):
    """Test UnifiedRegistryManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the core registry loading to avoid circular imports
        with patch('src.cursus.registry.hybrid.manager.load_registry_module') as mock_load:
            mock_module = MagicMock()
            mock_module.STEP_NAMES = {
                "XGBoostTraining": {
                    "config_class": "XGBoostTrainingConfig",
                    "builder_step_name": "XGBoostTrainingStepBuilder",
                    "spec_type": "XGBoostTraining",
                    "sagemaker_step_type": "Training",
                    "description": "XGBoost training step"
                },
                "PyTorchTraining": {
                    "config_class": "PyTorchTrainingConfig",
                    "builder_step_name": "PyTorchTrainingStepBuilder",
                    "spec_type": "PyTorchTraining",
                    "sagemaker_step_type": "Training",
                    "description": "PyTorch training step"
                }
            }
            mock_load.return_value = mock_module
            
            from cursus.registry.hybrid.manager import UnifiedRegistryManager
            self.manager = UnifiedRegistryManager()
    
    def test_unified_registry_manager_initialization(self):
        """Test UnifiedRegistryManager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertTrue(hasattr(self.manager, '_core_steps'))
        self.assertTrue(hasattr(self.manager, '_workspace_steps'))
        self.assertTrue(hasattr(self.manager, '_workspace_overrides'))
    
    def test_get_step_definition(self):
        """Test getting step definition."""
        # Test with a known step from the core registry
        definition = self.manager.get_step_definition("XGBoostTraining")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "XGBoostTraining")
        self.assertEqual(definition.config_class, "XGBoostTrainingConfig")
        self.assertEqual(definition.registry_type, RegistryType.CORE)
    
    def test_get_step_definition_nonexistent(self):
        """Test getting non-existent step definition."""
        definition = self.manager.get_step_definition("NonExistentStep")
        self.assertIsNone(definition)
    
    def test_get_all_step_definitions(self):
        """Test getting all step definitions."""
        all_definitions = self.manager.get_all_step_definitions()
        self.assertIsInstance(all_definitions, dict)
        self.assertGreater(len(all_definitions), 0)
        
        # Should contain known steps
        self.assertIn("XGBoostTraining", all_definitions)
        self.assertIn("PyTorchTraining", all_definitions)
    
    def test_get_all_step_definitions_with_workspace(self):
        """Test getting all step definitions with workspace context."""
        all_definitions = self.manager.get_all_step_definitions("test_workspace")
        self.assertIsInstance(all_definitions, dict)
        self.assertGreater(len(all_definitions), 0)
    
    def test_create_legacy_step_names_dict(self):
        """Test creating legacy step names dictionary."""
        legacy_dict = self.manager.create_legacy_step_names_dict()
        self.assertIsInstance(legacy_dict, dict)
        self.assertGreater(len(legacy_dict), 0)
        
        # Should have expected structure
        for step_name, step_info in legacy_dict.items():
            self.assertIsInstance(step_name, str)
            self.assertIsInstance(step_info, dict)
            self.assertIn("config_class", step_info)
            self.assertIn("builder_step_name", step_info)
            self.assertIn("spec_type", step_info)
    
    def test_create_legacy_step_names_dict_with_workspace(self):
        """Test creating legacy step names dictionary with workspace."""
        legacy_dict = self.manager.create_legacy_step_names_dict("test_workspace")
        self.assertIsInstance(legacy_dict, dict)
        self.assertGreater(len(legacy_dict), 0)
    
    def test_has_step(self):
        """Test checking if step exists."""
        self.assertTrue(self.manager.has_step("XGBoostTraining"))
        self.assertFalse(self.manager.has_step("NonExistentStep"))
    
    def test_has_step_with_workspace(self):
        """Test checking if step exists with workspace context."""
        self.assertTrue(self.manager.has_step("XGBoostTraining", "test_workspace"))
        self.assertFalse(self.manager.has_step("NonExistentStep", "test_workspace"))
    
    def test_list_steps(self):
        """Test listing all step names."""
        steps = self.manager.list_steps()
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
        self.assertIn("XGBoostTraining", steps)
        self.assertIn("PyTorchTraining", steps)
    
    def test_list_steps_with_workspace(self):
        """Test listing step names with workspace context."""
        steps = self.manager.list_steps("test_workspace")
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
    
    def test_get_step_count(self):
        """Test getting step count."""
        count = self.manager.get_step_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
    
    def test_get_step_count_with_workspace(self):
        """Test getting step count with workspace context."""
        count = self.manager.get_step_count("test_workspace")
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
    
    def test_get_registry_status(self):
        """Test getting registry status."""
        status = self.manager.get_registry_status()
        self.assertIsInstance(status, dict)
        self.assertIn("core", status)
    
    def test_get_step_conflicts(self):
        """Test getting step conflicts."""
        conflicts = self.manager.get_step_conflicts()
        # Conflicts can be either a list or dict depending on implementation
        self.assertIsInstance(conflicts, (list, dict))

class TestStepDefinition(unittest.TestCase):
    """Test StepDefinition model."""
    
    def test_step_definition_creation(self):
        """Test creating StepDefinition."""
        step_def = StepDefinition(
            name="TestStep",
            config_class="TestStepConfig",
            builder_step_name="TestStepBuilder",
            spec_type="TestStep",
            sagemaker_step_type="Processing",
            description="Test step",
            registry_type=RegistryType.CORE
        )
        
        self.assertEqual(step_def.name, "TestStep")
        self.assertEqual(step_def.config_class, "TestStepConfig")
        self.assertEqual(step_def.registry_type, RegistryType.CORE)
        self.assertIsNone(step_def.workspace_id)
    
    def test_step_definition_with_workspace(self):
        """Test creating StepDefinition with workspace."""
        step_def = StepDefinition(
            name="WorkspaceStep",
            config_class="WorkspaceStepConfig",
            builder_step_name="WorkspaceStepBuilder",
            spec_type="WorkspaceStep",
            sagemaker_step_type="Processing",
            description="Workspace step",
            registry_type=RegistryType.WORKSPACE,
            workspace_id="test_workspace"
        )
        
        self.assertEqual(step_def.name, "WorkspaceStep")
        self.assertEqual(step_def.registry_type, RegistryType.WORKSPACE)
        self.assertEqual(step_def.workspace_id, "test_workspace")

if __name__ == '__main__':
    unittest.main()
