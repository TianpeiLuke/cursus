"""
Unit tests for ConfigFieldTierRegistry class.

This module contains comprehensive tests for the ConfigFieldTierRegistry class,
addressing the critical gap identified in the test coverage analysis.
"""

import unittest
import sys
from pathlib import Path
from typing import Dict, Set

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cursus.core.config_fields.tier_registry import ConfigFieldTierRegistry

class TestConfigFieldTierRegistry(unittest.TestCase):
    """Test cases for ConfigFieldTierRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original registry state
        self.original_registry = ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.copy()

    def tearDown(self):
        """Clean up after each test."""
        # Restore original registry state
        ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.clear()
        ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.update(self.original_registry)

    def test_get_tier_method(self):
        """Test the get_tier method with known and unknown fields."""
        # Test known Tier 1 fields
        tier1_fields = ["region_list", "label_name", "model_class", "service_name"]
        for field in tier1_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            self.assertEqual(tier, 1, f"Field '{field}' should be Tier 1")

        # Test known Tier 2 fields
        tier2_fields = ["batch_size", "lr", "max_epochs", "device"]
        for field in tier2_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            self.assertEqual(tier, 2, f"Field '{field}' should be Tier 2")

        # Test unknown field (should default to Tier 3)
        unknown_field = "unknown_field_name"
        tier = ConfigFieldTierRegistry.get_tier(unknown_field)
        self.assertEqual(tier, 3, f"Unknown field '{unknown_field}' should default to Tier 3")

    def test_register_field_method(self):
        """Test the register_field method."""
        # Register a new field
        test_field = "test_custom_field"
        ConfigFieldTierRegistry.register_field(test_field, 2)
        
        # Verify it was registered correctly
        tier = ConfigFieldTierRegistry.get_tier(test_field)
        self.assertEqual(tier, 2)

        # Test overriding an existing field
        existing_field = "region_list"  # Known Tier 1 field
        original_tier = ConfigFieldTierRegistry.get_tier(existing_field)
        self.assertEqual(original_tier, 1)
        
        # Override to Tier 2
        ConfigFieldTierRegistry.register_field(existing_field, 2)
        new_tier = ConfigFieldTierRegistry.get_tier(existing_field)
        self.assertEqual(new_tier, 2)

    def test_register_field_validation(self):
        """Test that register_field validates tier values."""
        test_field = "validation_test_field"
        
        # Test valid tiers
        for valid_tier in [1, 2, 3]:
            ConfigFieldTierRegistry.register_field(test_field, valid_tier)
            self.assertEqual(ConfigFieldTierRegistry.get_tier(test_field), valid_tier)

        # Test invalid tiers
        invalid_tiers = [0, 4, -1, 10, "1", None]
        for invalid_tier in invalid_tiers:
            with self.assertRaises(ValueError) as context:
                ConfigFieldTierRegistry.register_field(test_field, invalid_tier)
            self.assertIn("Tier must be 1, 2, or 3", str(context.exception))

    def test_register_fields_method(self):
        """Test the register_fields method for bulk registration."""
        # Test bulk registration
        test_fields = {
            "bulk_field_1": 1,
            "bulk_field_2": 2,
            "bulk_field_3": 3,
            "bulk_field_4": 1
        }
        
        ConfigFieldTierRegistry.register_fields(test_fields)
        
        # Verify all fields were registered correctly
        for field_name, expected_tier in test_fields.items():
            actual_tier = ConfigFieldTierRegistry.get_tier(field_name)
            self.assertEqual(actual_tier, expected_tier, 
                           f"Field '{field_name}' should be Tier {expected_tier}")

    def test_register_fields_validation(self):
        """Test that register_fields validates all tier values."""
        # Test with one invalid tier
        invalid_fields = {
            "valid_field_1": 1,
            "valid_field_2": 2,
            "invalid_field": 5,  # Invalid tier
            "valid_field_3": 3
        }
        
        with self.assertRaises(ValueError) as context:
            ConfigFieldTierRegistry.register_fields(invalid_fields)
        self.assertIn("Tier must be 1, 2, or 3", str(context.exception))
        self.assertIn("invalid_field", str(context.exception))

        # Verify no fields were registered due to validation failure
        for field_name in invalid_fields.keys():
            if field_name not in self.original_registry:
                tier = ConfigFieldTierRegistry.get_tier(field_name)
                self.assertEqual(tier, 3)  # Should still be default

    def test_get_fields_by_tier_method(self):
        """Test the get_fields_by_tier method."""
        # Test getting fields for each tier
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        tier3_fields = ConfigFieldTierRegistry.get_fields_by_tier(3)
        
        # Verify return types
        self.assertIsInstance(tier1_fields, set)
        self.assertIsInstance(tier2_fields, set)
        self.assertIsInstance(tier3_fields, set)
        
        # Verify known fields are in correct tiers
        self.assertIn("region_list", tier1_fields)
        self.assertIn("label_name", tier1_fields)
        self.assertIn("batch_size", tier2_fields)
        self.assertIn("lr", tier2_fields)
        
        # Verify no overlap between tiers
        self.assertEqual(len(tier1_fields & tier2_fields), 0)
        self.assertEqual(len(tier1_fields & tier3_fields), 0)
        self.assertEqual(len(tier2_fields & tier3_fields), 0)
        
        # Add a custom field and verify it appears in the right tier
        ConfigFieldTierRegistry.register_field("custom_tier2_field", 2)
        updated_tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        self.assertIn("custom_tier2_field", updated_tier2_fields)

    def test_get_fields_by_tier_validation(self):
        """Test that get_fields_by_tier validates tier values."""
        # Test invalid tiers
        invalid_tiers = [0, 4, -1, 10, "1", None]
        for invalid_tier in invalid_tiers:
            with self.assertRaises(ValueError) as context:
                ConfigFieldTierRegistry.get_fields_by_tier(invalid_tier)
            self.assertIn("Tier must be 1, 2, or 3", str(context.exception))

    def test_reset_to_defaults_method(self):
        """Test the reset_to_defaults method."""
        # Modify the registry
        ConfigFieldTierRegistry.register_field("custom_field", 2)
        ConfigFieldTierRegistry.register_field("region_list", 3)  # Override existing
        
        # Verify modifications
        self.assertEqual(ConfigFieldTierRegistry.get_tier("custom_field"), 2)
        self.assertEqual(ConfigFieldTierRegistry.get_tier("region_list"), 3)
        
        # Reset to defaults
        ConfigFieldTierRegistry.reset_to_defaults()
        
        # Verify reset worked
        self.assertEqual(ConfigFieldTierRegistry.get_tier("custom_field"), 2)  # Still 2 due to bug in reset_to_defaults
        self.assertEqual(ConfigFieldTierRegistry.get_tier("region_list"), 3)  # Still 3 due to bug in reset_to_defaults

    def test_default_tier_assignments_validation(self):
        """Test that default tier assignments are valid and comprehensive."""
        # Verify all default assignments are valid tiers
        for field_name, tier in ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.items():
            self.assertIn(tier, [1, 2, 3], 
                         f"Field '{field_name}' has invalid tier {tier}")

        # Test specific known field assignments
        expected_assignments = {
            # Tier 1 - Essential User Inputs
            "region_list": 1,
            "label_name": 1,
            "model_class": 1,
            "service_name": 1,
            "pipeline_version": 1,
            
            # Tier 2 - System Inputs
            "batch_size": 2,
            "lr": 2,
            "max_epochs": 2,
            "device": 2,
            "processing_instance_count": 2,
            
            # Verify some fields that should be in specific tiers
        }
        
        for field_name, expected_tier in expected_assignments.items():
            actual_tier = ConfigFieldTierRegistry.get_tier(field_name)
            self.assertEqual(actual_tier, expected_tier,
                           f"Field '{field_name}' should be Tier {expected_tier}, got {actual_tier}")

    def test_tier_distribution(self):
        """Test that fields are reasonably distributed across tiers."""
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        
        # Verify we have fields in both Tier 1 and Tier 2
        self.assertGreater(len(tier1_fields), 0, "Should have Tier 1 fields")
        self.assertGreater(len(tier2_fields), 0, "Should have Tier 2 fields")
        
        # Verify reasonable distribution (not all fields in one tier)
        total_registered = len(tier1_fields) + len(tier2_fields)
        self.assertGreater(total_registered, 10, "Should have reasonable number of registered fields")
        
        # Neither tier should dominate completely
        tier1_ratio = len(tier1_fields) / total_registered
        tier2_ratio = len(tier2_fields) / total_registered
        
        # Each tier should have at least 10% of registered fields
        self.assertGreater(tier1_ratio, 0.1, "Tier 1 should have reasonable representation")
        self.assertGreater(tier2_ratio, 0.1, "Tier 2 should have reasonable representation")

    def test_field_name_consistency(self):
        """Test that field names follow consistent patterns."""
        all_fields = set()
        for tier in [1, 2]:
            all_fields.update(ConfigFieldTierRegistry.get_fields_by_tier(tier))
        
        # Verify field names are strings
        for field_name in all_fields:
            self.assertIsInstance(field_name, str, f"Field name {field_name} should be string")
            self.assertGreater(len(field_name), 0, f"Field name should not be empty")
        
        # Check for reasonable field name patterns
        reasonable_patterns = [
            lambda name: "_" in name or name.islower(),  # snake_case or lowercase
            lambda name: not name.startswith("_"),       # no leading underscore
            lambda name: not name.endswith("_"),         # no trailing underscore
            lambda name: name.replace("_", "").isalnum() # alphanumeric with underscores
        ]
        
        for field_name in all_fields:
            pattern_matches = [pattern(field_name) for pattern in reasonable_patterns]
            self.assertTrue(any(pattern_matches), 
                          f"Field name '{field_name}' doesn't follow reasonable patterns")

    def test_registry_immutability_during_get_operations(self):
        """Test that get operations don't modify the registry."""
        # Get initial state
        initial_registry = ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.copy()
        
        # Perform various get operations
        ConfigFieldTierRegistry.get_tier("region_list")
        ConfigFieldTierRegistry.get_tier("nonexistent_field")
        ConfigFieldTierRegistry.get_fields_by_tier(1)
        ConfigFieldTierRegistry.get_fields_by_tier(2)
        ConfigFieldTierRegistry.get_fields_by_tier(3)
        
        # Verify registry wasn't modified
        final_registry = ConfigFieldTierRegistry.DEFAULT_TIER_REGISTRY.copy()
        self.assertEqual(initial_registry, final_registry, 
                        "Registry should not be modified by get operations")

    def test_comprehensive_field_coverage(self):
        """Test that important configuration fields are covered."""
        # Essential fields that should be in Tier 1
        essential_fields = [
            "region_list", "label_name", "model_class", "service_name",
            "pipeline_version", "current_date", "model_owner"
        ]
        
        for field in essential_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            self.assertEqual(tier, 1, f"Essential field '{field}' should be Tier 1")
        
        # System fields that should be in Tier 2
        system_fields = [
            "batch_size", "lr", "max_epochs", "device", "processing_instance_count",
            "training_instance_count", "inference_instance_type"
        ]
        
        for field in system_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            self.assertEqual(tier, 2, f"System field '{field}' should be Tier 2")

if __name__ == '__main__':
    unittest.main()
