"""
Test suite for hybrid registry manager components.

Tests RegistryConfig, CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.registry.hybrid.manager import (
    RegistryConfig,
    CoreStepRegistry,
    LocalStepRegistry,
    HybridRegistryManager
)
from src.cursus.registry.exceptions import RegistryLoadError


class TestRegistryConfig(unittest.TestCase):
    """Test RegistryConfig data model."""
    
    def test_registry_config_creation_minimal(self):
        """Test creating registry config with minimal fields."""
        config = RegistryConfig()
        
        self.assertEqual(config.core_registry_path, "src/cursus/registry/step_names.py")
        self.assertEqual(config.workspaces_root, "developer_workspaces/developers")
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_size, 1000)
        self.assertTrue(config.auto_discover_workspaces)
        self.assertFalse(config.strict_validation)
    
    def test_registry_config_creation_custom(self):
        """Test creating registry config with custom values."""
        config = RegistryConfig(
            core_registry_path="custom/path/step_names.py",
            workspaces_root="custom/workspaces",
            enable_caching=False,
            cache_size=500,
            auto_discover_workspaces=False,
            strict_validation=True
        )
        
        self.assertEqual(config.core_registry_path, "custom/path/step_names.py")
        self.assertEqual(config.workspaces_root, "custom/workspaces")
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.cache_size, 500)
        self.assertFalse(config.auto_discover_workspaces)
        self.assertTrue(config.strict_validation)


class TestCoreStepRegistry(unittest.TestCase):
    """Test CoreStepRegistry component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary registry file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.temp_file.write("""
STEP_NAMES = {
    "TestStep": {
        "config_class": "TestStepConfig",
        "builder_step_name": "TestStepBuilder",
        "spec_type": "TestStep",
        "sagemaker_step_type": "Processing",
        "description": "Test step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost training step"
    }
}
""")
        self.temp_file.close()
        self.temp_path = self.temp_file.name
    
    def tearDown(self):
        """Clean up test environment."""
        Path(self.temp_path).unlink()
    
    def test_core_step_registry_initialization(self):
        """Test CoreStepRegistry initialization."""
        registry = CoreStepRegistry(self.temp_path)
        
        self.assertEqual(registry.registry_path, Path(self.temp_path))
        self.assertEqual(len(registry._step_definitions), 2)
        self.assertIn("TestStep", registry._step_definitions)
        self.assertIn("XGBoostTraining", registry._step_definitions)
    
    def test_core_step_registry_get_step_definition(self):
        """Test getting step definition from core registry."""
        registry = CoreStepRegistry(self.temp_path)
        
        definition = registry.get_step_definition("TestStep")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "TestStep")
        self.assertEqual(definition.config_class, "TestStepConfig")
        self.assertEqual(definition.registry_type, "core")
        self.assertIsNone(definition.workspace_id)
    
    def test_core_step_registry_get_nonexistent_step(self):
        """Test getting non-existent step definition."""
        registry = CoreStepRegistry(self.temp_path)
        
        definition = registry.get_step_definition("NonExistentStep")
        self.assertIsNone(definition)
    
    def test_core_step_registry_get_all_definitions(self):
        """Test getting all step definitions."""
        registry = CoreStepRegistry(self.temp_path)
        
        all_definitions = registry.get_all_step_definitions()
        self.assertEqual(len(all_definitions), 2)
        self.assertIn("TestStep", all_definitions)
        self.assertIn("XGBoostTraining", all_definitions)
        
        # Verify all are core registry type
        for definition in all_definitions.values():
            self.assertEqual(definition.registry_type, "core")
            self.assertIsNone(definition.workspace_id)
    
    def test_core_step_registry_file_not_found(self):
        """Test CoreStepRegistry with non-existent file."""
        with self.assertRaises(RegistryLoadError) as exc_info:
            CoreStepRegistry("nonexistent.py")
        
        self.assertIn("Registry Error in Core Registry Loading", str(exc_info.exception))
    
    def test_core_step_registry_invalid_format(self):
        """Test CoreStepRegistry with invalid file format."""
        # Create file without STEP_NAMES
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write("# No STEP_NAMES defined")
        temp_file.close()
        
        try:
            with self.assertRaises(RegistryLoadError):
                CoreStepRegistry(temp_file.name)
        finally:
            Path(temp_file.name).unlink()


class TestLocalStepRegistry(unittest.TestCase):
    """Test LocalStepRegistry component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary core registry
        self.core_temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.core_temp_file.write("""
STEP_NAMES = {
    "CoreStep": {
        "config_class": "CoreStepConfig",
        "builder_step_name": "CoreStepBuilder",
        "spec_type": "CoreStep",
        "sagemaker_step_type": "Processing",
        "description": "Core step"
    }
}
""")
        self.core_temp_file.close()
        
        # Create core registry
        self.core_registry = CoreStepRegistry(self.core_temp_file.name)
        
        # Create temporary workspace directory
        self.workspace_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.workspace_dir)
        
        # Create workspace registry file
        registry_dir = self.workspace_path / "src" / "cursus_dev" / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)
        
        registry_file = registry_dir / "workspace_registry.py"
        registry_file.write_text("""
LOCAL_STEPS = {
    "LocalStep": {
        "config_class": "LocalStepConfig",
        "builder_step_name": "LocalStepBuilder",
        "spec_type": "LocalStep",
        "sagemaker_step_type": "Processing",
        "description": "Local step"
    }
}

STEP_OVERRIDES = {
    "CoreStep": {
        "config_class": "OverriddenCoreStepConfig",
        "builder_step_name": "OverriddenCoreStepBuilder",
        "spec_type": "OverriddenCoreStep",
        "sagemaker_step_type": "Processing",
        "description": "Overridden core step"
    }
}

WORKSPACE_METADATA = {
    "developer_id": "test_developer",
    "version": "1.0.0",
    "description": "Test workspace"
}
""")
    
    def tearDown(self):
        """Clean up test environment."""
        Path(self.core_temp_file.name).unlink()
        import shutil
        shutil.rmtree(self.workspace_dir)
    
    def test_local_step_registry_initialization(self):
        """Test LocalStepRegistry initialization."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        self.assertEqual(registry.workspace_path, self.workspace_path)
        self.assertEqual(registry.core_registry, self.core_registry)
        self.assertEqual(registry.workspace_id, "test_developer")
    
    def test_local_step_registry_get_local_step(self):
        """Test getting local step definition."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        definition = registry.get_step_definition("LocalStep")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "LocalStep")
        self.assertEqual(definition.config_class, "LocalStepConfig")
        self.assertEqual(definition.registry_type, "workspace")
        self.assertEqual(definition.workspace_id, "test_developer")
    
    def test_local_step_registry_get_overridden_step(self):
        """Test getting overridden core step."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        definition = registry.get_step_definition("CoreStep")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "CoreStep")
        self.assertEqual(definition.config_class, "OverriddenCoreStepConfig")
        self.assertEqual(definition.registry_type, "override")
        self.assertEqual(definition.workspace_id, "test_developer")
    
    def test_local_step_registry_get_core_step_fallback(self):
        """Test fallback to core registry for non-local steps."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        # Remove LOCAL_STEPS and STEP_OVERRIDES to test fallback
        registry._local_steps = {}
        registry._step_overrides = {}
        
        definition = registry.get_step_definition("CoreStep")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "CoreStep")
        self.assertEqual(definition.config_class, "CoreStepConfig")  # Original core config
        self.assertEqual(definition.registry_type, "core")
        self.assertIsNone(definition.workspace_id)
    
    def test_local_step_registry_get_all_definitions(self):
        """Test getting all step definitions (merged)."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        all_definitions = registry.get_all_step_definitions()
        
        # Should have both local and core steps, with overrides applied
        self.assertIn("LocalStep", all_definitions)
        self.assertIn("CoreStep", all_definitions)
        
        # LocalStep should be workspace type
        local_def = all_definitions["LocalStep"]
        self.assertEqual(local_def.registry_type, "workspace")
        self.assertEqual(local_def.workspace_id, "test_developer")
        
        # CoreStep should be override type (overridden)
        core_def = all_definitions["CoreStep"]
        self.assertEqual(core_def.registry_type, "override")
        self.assertEqual(core_def.config_class, "OverriddenCoreStepConfig")
    
    def test_local_step_registry_get_local_only_definitions(self):
        """Test getting only local definitions."""
        registry = LocalStepRegistry(str(self.workspace_path), self.core_registry)
        
        local_only = registry.get_local_only_definitions()
        
        # Should only have local and override steps, not core
        self.assertIn("LocalStep", local_only)
        self.assertIn("CoreStep", local_only)  # Override counts as local
        self.assertEqual(len(local_only), 2)
        
        # Verify types
        self.assertEqual(local_only["LocalStep"].registry_type, "workspace")
        self.assertEqual(local_only["CoreStep"].registry_type, "override")


class TestHybridRegistryManager(unittest.TestCase):
    """Test HybridRegistryManager component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary core registry
        self.core_temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.core_temp_file.write("""
STEP_NAMES = {
    "CoreStep1": {
        "config_class": "CoreStep1Config",
        "builder_step_name": "CoreStep1Builder",
        "spec_type": "CoreStep1",
        "sagemaker_step_type": "Processing",
        "description": "Core step 1"
    },
    "CoreStep2": {
        "config_class": "CoreStep2Config",
        "builder_step_name": "CoreStep2Builder",
        "spec_type": "CoreStep2",
        "sagemaker_step_type": "Training",
        "description": "Core step 2"
    }
}
""")
        self.core_temp_file.close()
        
        # Create temporary workspaces directory
        self.workspaces_dir = tempfile.mkdtemp()
        self.workspaces_path = Path(self.workspaces_dir)
        
        # Create developer_1 workspace
        self._create_test_workspace("developer_1", {
            "LOCAL_STEPS": {
                "LocalStep1": {
                    "config_class": "LocalStep1Config",
                    "builder_step_name": "LocalStep1Builder",
                    "spec_type": "LocalStep1",
                    "sagemaker_step_type": "Processing",
                    "description": "Local step 1"
                }
            },
            "STEP_OVERRIDES": {},
            "WORKSPACE_METADATA": {
                "developer_id": "developer_1",
                "version": "1.0.0"
            }
        })
        
        # Create developer_2 workspace
        self._create_test_workspace("developer_2", {
            "LOCAL_STEPS": {
                "LocalStep2": {
                    "config_class": "LocalStep2Config",
                    "builder_step_name": "LocalStep2Builder",
                    "spec_type": "LocalStep2",
                    "sagemaker_step_type": "Training",
                    "description": "Local step 2"
                }
            },
            "STEP_OVERRIDES": {
                "CoreStep1": {
                    "config_class": "OverriddenCoreStep1Config",
                    "builder_step_name": "OverriddenCoreStep1Builder",
                    "spec_type": "OverriddenCoreStep1",
                    "sagemaker_step_type": "Processing",
                    "description": "Overridden core step 1"
                }
            },
            "WORKSPACE_METADATA": {
                "developer_id": "developer_2",
                "version": "1.0.0"
            }
        })
    
    def tearDown(self):
        """Clean up test environment."""
        Path(self.core_temp_file.name).unlink()
        import shutil
        shutil.rmtree(self.workspaces_dir)
    
    def _create_test_workspace(self, workspace_id: str, registry_content: Dict[str, Any]):
        """Helper to create test workspace."""
        workspace_dir = self.workspaces_path / workspace_id
        registry_dir = workspace_dir / "src" / "cursus_dev" / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)
        
        registry_file = registry_dir / "workspace_registry.py"
        
        # Convert dict to Python code
        content_lines = []
        for key, value in registry_content.items():
            if isinstance(value, dict):
                content_lines.append(f"{key} = {repr(value)}")
            else:
                content_lines.append(f"{key} = {repr(value)}")
        
        registry_file.write_text('\n'.join(content_lines))
    
    def test_hybrid_registry_manager_initialization(self):
        """Test HybridRegistryManager initialization."""
        manager = HybridRegistryManager(
            core_registry_path=self.core_temp_file.name,
            workspaces_root=str(self.workspaces_path)
        )
        
        self.assertIsNotNone(manager.core_registry)
        self.assertEqual(manager.workspaces_root, self.workspaces_path)
        self.assertEqual(len(manager._local_registries), 2)
        self.assertIn("developer_1", manager._local_registries)
        self.assertIn("developer_2", manager._local_registries)
        self.assertIsNotNone(manager.conflict_resolver)
    
    def test_hybrid_registry_manager_get_core_step(self):
        """Test getting core step definition."""
        manager = HybridRegistryManager(
            core_registry_path=self.core_temp_file.name,
            workspaces_root=str(self.workspaces_path)
        )
        
        definition = manager.get_step_definition("CoreStep1")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "CoreStep1")
        self.assertEqual(definition.config_class, "CoreStep1Config")
        self.assertEqual(definition.registry_type, "core")
    
    def test_hybrid_registry_manager_get_workspace_step(self):
        """Test getting workspace-specific step definition."""
        manager = HybridRegistryManager(
            core_registry_path=self.core_temp_file.name,
            workspaces_root=str(self.workspaces_path)
        )
        
        definition = manager.get_step_definition("LocalStep1", workspace_id="developer_1")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.name, "LocalStep1")
        self.assertEqual(definition.config_class, "LocalStep1Config")
        self.assertEqual(definition.registry_type, "workspace")


if __name__ == '__main__':
    unittest.main()
