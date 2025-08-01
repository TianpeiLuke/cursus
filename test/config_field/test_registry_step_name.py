"""
Tests for registry-based step name generation.

This module verifies that the step name generation correctly uses the pipeline registry
as the single source of truth for step names.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
from pydantic import BaseModel

from src.config_field_manager.config_merger import ConfigMerger
from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
from src.pipeline_steps.utils import serialize_config


class DummyConfig(BaseModel):
    """Dummy config for testing."""
    field1: str = "value1"
    field2: int = 123


class TestRegistryStepName(unittest.TestCase):
    """Tests for registry-based step name generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DummyConfig()
        
    @patch("src.pipeline_registry.step_names.CONFIG_STEP_REGISTRY", {"DummyConfig": "Registry_DummyStep"})
    def test_merger_uses_registry_step_name(self):
        """Test that ConfigMerger uses registry for step name generation."""
        # Create a merger with our config
        merger = ConfigMerger([self.config])
        
        # Get the step name using the method
        step_name = merger._generate_step_name(self.config)
        
        # Verify it's using the registry name
        self.assertEqual(step_name, "Registry_DummyStep")
    
    @patch("src.pipeline_registry.step_names.CONFIG_STEP_REGISTRY", {"DummyConfig": "Registry_DummyStep"})
    def test_serializer_uses_registry_step_name(self):
        """Test that TypeAwareConfigSerializer uses registry for step name generation."""
        # Create a serializer
        serializer = TypeAwareConfigSerializer()
        
        # Get the step name using the method
        step_name = serializer.generate_step_name(self.config)
        
        # Verify it's using the registry name
        self.assertEqual(step_name, "Registry_DummyStep")
    
    @patch("src.pipeline_registry.step_names.CONFIG_STEP_REGISTRY", {"DummyConfig": "Registry_DummyStep"})
    def test_utils_uses_registry_step_name(self):
        """Test that utils uses registry for step name generation."""
        # Serialize config using utils.py
        serialized = serialize_config(self.config)
        
        # Verify it's using the registry name in metadata
        self.assertEqual(serialized["_metadata"]["step_name"], "Registry_DummyStep")
    
    @patch("src.pipeline_registry.step_names.CONFIG_STEP_REGISTRY", {"DummyConfig": "Registry_DummyStep"})
    def test_save_config_uses_registry_step_name(self):
        """Test that saving configs uses registry for step name generation."""
        # Create a merger with our config
        merger = ConfigMerger([self.config])
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            merger.save(tmp_path)
            
            # Read the saved file
            with open(tmp_path, 'r') as f:
                saved_data = json.load(f)
            
            # Clean up
            os.unlink(tmp_path)
        
        # Verify config_types has the registry step name as key
        self.assertIn("Registry_DummyStep", saved_data["metadata"]["config_types"])
        self.assertEqual(saved_data["metadata"]["config_types"]["Registry_DummyStep"], "DummyConfig")


if __name__ == "__main__":
    unittest.main()
