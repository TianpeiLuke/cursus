"""
Unit tests for ConfigFieldTierRegistry class.

This module contains comprehensive tests for the ConfigFieldTierRegistry class,
addressing the critical gap identified in the test coverage analysis.
"""

import pytest
from typing import Dict, Set

from cursus.core.config_fields import ConfigFieldTierRegistry


class TestConfigFieldTierRegistry:
    """Test cases for ConfigFieldTierRegistry."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Store original tier registry state
        self.original_mapping = ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.copy()
        
        yield  # This is where the test runs
        
        # Clean up after each test
        ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.clear()
        ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.update(self.original_mapping)

    def test_get_tier_method(self):
        """Test the get_tier method with known and unknown fields."""
        # Test known Tier 1 fields from the actual FALLBACK_TIER_MAPPING
        tier1_fields = ["region", "pipeline_name", "full_field_list", "label_name", "id_name"]
        for field in tier1_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            assert tier == 1, f"Field '{field}' should be Tier 1"

        # Test known Tier 2 fields from the actual FALLBACK_TIER_MAPPING
        tier2_fields = ["instance_type", "framework_version", "processing_entry_point"]
        for field in tier2_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            assert tier == 2, f"Field '{field}' should be Tier 2"

        # Test unknown field (should default to Tier 3)
        unknown_field = "unknown_field_name"
        tier = ConfigFieldTierRegistry.get_tier(unknown_field)
        assert tier == 3, f"Unknown field '{unknown_field}' should default to Tier 3"

    def test_register_field_method(self):
        """Test the register_field method."""
        # Register a new field
        test_field = "test_custom_field"
        ConfigFieldTierRegistry.register_field(test_field, 2)

        # Verify it was registered correctly
        tier = ConfigFieldTierRegistry.get_tier(test_field)
        assert tier == 2

        # Test overriding an existing field
        existing_field = "region"  # Known Tier 1 field from FALLBACK_TIER_MAPPING
        original_tier = ConfigFieldTierRegistry.get_tier(existing_field)
        assert original_tier == 1

        # Override to Tier 2
        ConfigFieldTierRegistry.register_field(existing_field, 2)
        new_tier = ConfigFieldTierRegistry.get_tier(existing_field)
        assert new_tier == 2

    def test_register_field_validation(self):
        """Test that register_field validates tier values."""
        test_field = "validation_test_field"

        # Test valid tiers
        for valid_tier in [1, 2, 3]:
            ConfigFieldTierRegistry.register_field(test_field, valid_tier)
            assert ConfigFieldTierRegistry.get_tier(test_field) == valid_tier

        # Test invalid tiers
        invalid_tiers = [0, 4, -1, 10, "1", None]
        for invalid_tier in invalid_tiers:
            with pytest.raises(ValueError, match="Tier must be 1, 2, or 3"):
                ConfigFieldTierRegistry.register_field(test_field, invalid_tier)

    def test_register_fields_method(self):
        """Test the register_fields method for bulk registration."""
        # Test bulk registration
        test_fields = {
            "bulk_field_1": 1,
            "bulk_field_2": 2,
            "bulk_field_3": 3,
            "bulk_field_4": 1,
        }

        ConfigFieldTierRegistry.register_fields(test_fields)

        # Verify all fields were registered correctly
        for field_name, expected_tier in test_fields.items():
            actual_tier = ConfigFieldTierRegistry.get_tier(field_name)
            assert actual_tier == expected_tier, f"Field '{field_name}' should be Tier {expected_tier}"

    def test_register_fields_validation(self):
        """Test that register_fields validates all tier values."""
        # Test with one invalid tier
        invalid_fields = {
            "valid_field_1": 1,
            "valid_field_2": 2,
            "invalid_field": 5,  # Invalid tier
            "valid_field_3": 3,
        }

        with pytest.raises(ValueError, match="Tier must be 1, 2, or 3"):
            ConfigFieldTierRegistry.register_fields(invalid_fields)

        # Verify no fields were registered due to validation failure
        for field_name in invalid_fields.keys():
            if field_name not in self.original_mapping:
                tier = ConfigFieldTierRegistry.get_tier(field_name)
                assert tier == 3  # Should still be default

    def test_get_fields_by_tier_method(self):
        """Test the get_fields_by_tier method."""
        # Test getting fields for each tier
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        tier3_fields = ConfigFieldTierRegistry.get_fields_by_tier(3)

        # Verify return types
        assert isinstance(tier1_fields, set)
        assert isinstance(tier2_fields, set)
        assert isinstance(tier3_fields, set)

        # Verify known fields are in correct tiers (from actual FALLBACK_TIER_MAPPING)
        assert "region" in tier1_fields
        assert "label_name" in tier1_fields
        assert "instance_type" in tier2_fields
        assert "framework_version" in tier2_fields

        # Verify no overlap between tiers
        assert len(tier1_fields & tier2_fields) == 0
        assert len(tier1_fields & tier3_fields) == 0
        assert len(tier2_fields & tier3_fields) == 0

        # Add a custom field and verify it appears in the right tier
        ConfigFieldTierRegistry.register_field("custom_tier2_field", 2)
        updated_tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        assert "custom_tier2_field" in updated_tier2_fields

    def test_get_fields_by_tier_validation(self):
        """Test that get_fields_by_tier validates tier values."""
        # Test invalid tiers
        invalid_tiers = [0, 4, -1, 10, "1", None]
        for invalid_tier in invalid_tiers:
            with pytest.raises(ValueError, match="Tier must be 1, 2, or 3"):
                ConfigFieldTierRegistry.get_fields_by_tier(invalid_tier)

    def test_reset_to_defaults_method(self):
        """Test the reset_to_defaults method."""
        # Modify the registry
        ConfigFieldTierRegistry.register_field("custom_field", 2)
        ConfigFieldTierRegistry.register_field("region", 3)  # Override existing

        # Verify modifications
        assert ConfigFieldTierRegistry.get_tier("custom_field") == 2
        assert ConfigFieldTierRegistry.get_tier("region") == 3

        # Reset to defaults
        ConfigFieldTierRegistry.reset_to_defaults()

        # Verify reset worked - custom field should be gone, region should be back to Tier 1
        assert ConfigFieldTierRegistry.get_tier("custom_field") == 3  # Should be default after reset
        assert ConfigFieldTierRegistry.get_tier("region") == 1  # Should be back to original

    def test_default_tier_assignments_validation(self):
        """Test that default tier assignments are valid and comprehensive."""
        # Verify all default assignments are valid tiers
        for field_name, tier in ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.items():
            assert tier in [1, 2, 3], f"Field '{field_name}' has invalid tier {tier}"

        # Test specific known field assignments from actual FALLBACK_TIER_MAPPING
        expected_assignments = {
            # Tier 1 - Essential User Inputs (from actual mapping)
            "region": 1,
            "pipeline_name": 1,
            "full_field_list": 1,
            "label_name": 1,
            "id_name": 1,
            # Tier 2 - System Inputs (from actual mapping)
            "instance_type": 2,
            "framework_version": 2,
            "processing_entry_point": 2,
        }

        for field_name, expected_tier in expected_assignments.items():
            actual_tier = ConfigFieldTierRegistry.get_tier(field_name)
            assert actual_tier == expected_tier, f"Field '{field_name}' should be Tier {expected_tier}, got {actual_tier}"

    def test_tier_distribution(self):
        """Test that fields are reasonably distributed across tiers."""
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)

        # Verify we have fields in both Tier 1 and Tier 2
        assert len(tier1_fields) > 0, "Should have Tier 1 fields"
        assert len(tier2_fields) > 0, "Should have Tier 2 fields"

        # Verify reasonable distribution (adapted for smaller FALLBACK_TIER_MAPPING)
        total_registered = len(tier1_fields) + len(tier2_fields)
        assert total_registered >= 5, "Should have at least 5 registered fields in FALLBACK_TIER_MAPPING"

        # Neither tier should dominate completely
        tier1_ratio = len(tier1_fields) / total_registered
        tier2_ratio = len(tier2_fields) / total_registered

        # Each tier should have at least some representation (adapted for smaller mapping)
        assert tier1_ratio > 0.0, "Tier 1 should have some representation"
        assert tier2_ratio > 0.0, "Tier 2 should have some representation"
        
        # Verify the actual expected distribution from FALLBACK_TIER_MAPPING
        # Tier 1: region, pipeline_name, full_field_list, label_name, id_name (5 fields)
        # Tier 2: instance_type, framework_version, processing_entry_point (3 fields)
        assert len(tier1_fields) == 5, "Should have exactly 5 Tier 1 fields"
        assert len(tier2_fields) == 3, "Should have exactly 3 Tier 2 fields"

    def test_field_name_consistency(self):
        """Test that field names follow consistent patterns."""
        all_fields = set()
        for tier in [1, 2]:
            all_fields.update(ConfigFieldTierRegistry.get_fields_by_tier(tier))

        # Verify field names are strings
        for field_name in all_fields:
            assert isinstance(field_name, str), f"Field name {field_name} should be string"
            assert len(field_name) > 0, "Field name should not be empty"

        # Check for reasonable field name patterns
        reasonable_patterns = [
            lambda name: "_" in name or name.islower(),  # snake_case or lowercase
            lambda name: not name.startswith("_"),  # no leading underscore
            lambda name: not name.endswith("_"),  # no trailing underscore
            lambda name: name.replace("_", "").isalnum(),  # alphanumeric with underscores
        ]

        for field_name in all_fields:
            pattern_matches = [pattern(field_name) for pattern in reasonable_patterns]
            assert any(pattern_matches), f"Field name '{field_name}' doesn't follow reasonable patterns"

    def test_registry_immutability_during_get_operations(self):
        """Test that get operations don't modify the registry."""
        # Get initial state
        initial_registry = ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.copy()

        # Perform various get operations
        ConfigFieldTierRegistry.get_tier("region_list")
        ConfigFieldTierRegistry.get_tier("nonexistent_field")
        ConfigFieldTierRegistry.get_fields_by_tier(1)
        ConfigFieldTierRegistry.get_fields_by_tier(2)
        ConfigFieldTierRegistry.get_fields_by_tier(3)

        # Verify registry wasn't modified
        final_registry = ConfigFieldTierRegistry.FALLBACK_TIER_MAPPING.copy()
        assert initial_registry == final_registry, "Registry should not be modified by get operations"

    def test_comprehensive_field_coverage(self):
        """Test that important configuration fields are covered."""
        # Essential fields that are actually in FALLBACK_TIER_MAPPING as Tier 1
        essential_fields_tier1 = [
            "label_name",  # This is in the actual mapping
            "region",
            "pipeline_name",
            "full_field_list",
            "id_name",
        ]

        for field in essential_fields_tier1:
            tier = ConfigFieldTierRegistry.get_tier(field)
            assert tier == 1, f"Essential field '{field}' should be Tier 1"

        # System fields that are actually in FALLBACK_TIER_MAPPING as Tier 2
        system_fields_tier2 = [
            "instance_type",
            "framework_version", 
            "processing_entry_point",
        ]

        for field in system_fields_tier2:
            tier = ConfigFieldTierRegistry.get_tier(field)
            assert tier == 2, f"System field '{field}' should be Tier 2"

        # Fields not in FALLBACK_TIER_MAPPING should default to Tier 3
        unknown_fields = [
            "batch_size",
            "lr", 
            "max_epochs",
            "device",
            "model_class",
            "service_name",
        ]

        for field in unknown_fields:
            tier = ConfigFieldTierRegistry.get_tier(field)
            assert tier == 3, f"Unknown field '{field}' should default to Tier 3"

    def test_api_methods_exist(self):
        """Test that all expected API methods exist."""
        # Test that the class has all expected methods
        assert hasattr(ConfigFieldTierRegistry, 'get_tier')
        assert hasattr(ConfigFieldTierRegistry, 'register_field')
        assert hasattr(ConfigFieldTierRegistry, 'register_fields')
        assert hasattr(ConfigFieldTierRegistry, 'get_fields_by_tier')
        assert hasattr(ConfigFieldTierRegistry, 'reset_to_defaults')

        # Test that methods are callable
        assert callable(ConfigFieldTierRegistry.get_tier)
        assert callable(ConfigFieldTierRegistry.register_field)
        assert callable(ConfigFieldTierRegistry.register_fields)
        assert callable(ConfigFieldTierRegistry.get_fields_by_tier)
        assert callable(ConfigFieldTierRegistry.reset_to_defaults)

    def test_basic_functionality_integration(self):
        """Test basic functionality integration without relying on internal attributes."""
        # Test the basic workflow
        test_field = "integration_test_field"
        
        # Initially should be Tier 3 (default)
        assert ConfigFieldTierRegistry.get_tier(test_field) == 3
        
        # Register as Tier 1
        ConfigFieldTierRegistry.register_field(test_field, 1)
        assert ConfigFieldTierRegistry.get_tier(test_field) == 1
        
        # Should appear in Tier 1 fields
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        assert test_field in tier1_fields
        
        # Should not appear in other tiers
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        tier3_fields = ConfigFieldTierRegistry.get_fields_by_tier(3)
        assert test_field not in tier2_fields
        assert test_field not in tier3_fields
        
        # Change to Tier 2
        ConfigFieldTierRegistry.register_field(test_field, 2)
        assert ConfigFieldTierRegistry.get_tier(test_field) == 2
        
        # Should now appear in Tier 2 fields
        tier2_fields = ConfigFieldTierRegistry.get_fields_by_tier(2)
        assert test_field in tier2_fields
        
        # Should not appear in other tiers
        tier1_fields = ConfigFieldTierRegistry.get_fields_by_tier(1)
        tier3_fields = ConfigFieldTierRegistry.get_fields_by_tier(3)
        assert test_field not in tier1_fields
        assert test_field not in tier3_fields

    def test_context_aware_tier_classification(self):
        """Test context-aware tier classification with config instances."""
        # Test that get_tier works with optional config_instance parameter
        # This tests the adapter's ability to use config instances for context-aware classification
        
        # Test without config instance (should use fallback mapping)
        tier_without_context = ConfigFieldTierRegistry.get_tier("region")
        assert tier_without_context == 1
        
        # Test with None config instance (should still use fallback mapping)
        tier_with_none_context = ConfigFieldTierRegistry.get_tier("region", None)
        assert tier_with_none_context == 1
        
        # Both should be the same since we're using fallback mapping
        assert tier_without_context == tier_with_none_context

    def test_get_fields_by_tier_with_context(self):
        """Test get_fields_by_tier with optional config_instance parameter."""
        # Test without config instance
        tier1_fields_without_context = ConfigFieldTierRegistry.get_fields_by_tier(1)
        
        # Test with None config instance
        tier1_fields_with_none_context = ConfigFieldTierRegistry.get_fields_by_tier(1, None)
        
        # Both should be the same since we're using fallback mapping
        assert tier1_fields_without_context == tier1_fields_with_none_context
        assert len(tier1_fields_without_context) == 5  # Expected number from FALLBACK_TIER_MAPPING
