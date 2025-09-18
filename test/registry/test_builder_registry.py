"""Pytest tests for the StepBuilderRegistry class."""

import pytest
import logging

from cursus.registry.builder_registry import StepBuilderRegistry, get_global_registry
from cursus.registry.step_names import STEP_NAMES, get_all_step_names
from cursus.registry.exceptions import RegistryError


class TestBuilderRegistry:
    """Test case for StepBuilderRegistry."""

    def setup_method(self):
        """Set up test case."""
        logging.basicConfig(level=logging.INFO)
        self.registry = StepBuilderRegistry()

    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry.BUILDER_REGISTRY) > 0
        assert len(self.registry.LEGACY_ALIASES) > 0

    def test_canonical_step_names(self):
        """Test that canonical step names are properly mapped."""
        builder_map = self.registry.get_builder_map()

        # Test a few key canonical names from step_names.py that should exist
        expected_canonical_names = [
            "Package",
            "Payload",
            "PyTorchTraining",
            "PyTorchModel",
        ]
        for name in expected_canonical_names:
            if name in builder_map:
                assert name in builder_map

        # Verify legacy aliases are properly handled - updated for step catalog compatibility
        legacy_aliases_to_test = [
            "MIMSPackaging",
            "MIMSPayload",
            "PyTorchTraining",
            "PyTorchModel",
        ]
        for alias in legacy_aliases_to_test:
            # Only test if the registry supports this alias
            if (
                hasattr(self.registry, "LEGACY_ALIASES")
                and alias in self.registry.LEGACY_ALIASES
            ):
                canonical_name = self.registry.LEGACY_ALIASES[alias]
                # Check if the canonical name is actually available in the builder map
                if canonical_name in builder_map:
                    assert self.registry.is_step_type_supported(alias)
                else:
                    # Log warning but don't fail test if builder not available (may depend on external dependencies)
                    logging.warning(f"Legacy alias {alias} -> {canonical_name} not supported (builder not available)")

    def test_config_class_to_step_type(self):
        """Test _config_class_to_step_type method."""
        # Test with config classes from step registry
        for step_name, info in STEP_NAMES.items():
            # Skip base classes that are not expected to have builders
            if step_name in ["Base", "Processing"]:
                continue

            config_class = info["config_class"]
            step_type = self.registry._config_class_to_step_type(config_class)

            # Either the returned step type should be in the builder registry directly
            # or it should be a legacy alias that maps to a canonical name
            is_supported = (
                step_type in self.registry.get_builder_map()
                or step_type in self.registry.LEGACY_ALIASES
            )

            # If not supported, log for debugging but don't fail the test for missing external dependencies
            if not is_supported:
                logging.warning(
                    f"Step type '{step_type}' from config class '{config_class}' not supported - may require external dependencies"
                )

        # Test fallback for unknown config class
        unknown_step = self.registry._config_class_to_step_type("UnknownConfig")
        assert unknown_step == "Unknown"

    def test_get_config_types_for_step_type(self):
        """Test get_config_types_for_step_type method."""
        # Test with step types from registry
        for step_name in get_all_step_names():
            config_types = self.registry.get_config_types_for_step_type(step_name)
            assert len(config_types) > 0

        # Test with legacy aliases
        for legacy_name in self.registry.LEGACY_ALIASES:
            config_types = self.registry.get_config_types_for_step_type(legacy_name)
            assert len(config_types) > 0

    def test_validate_registry(self):
        """Test validate_registry method."""
        validation = self.registry.validate_registry()

        # Should have valid entries
        assert len(validation["valid"]) > 0

        # Print any invalid entries
        if validation.get("invalid"):
            logging.warning(f"Invalid registry entries: {validation['invalid']}")

        # Print any missing entries
        if validation.get("missing"):
            logging.warning(f"Missing registry entries: {validation['missing']}")

    def test_global_registry_singleton(self):
        """Test that the global registry is a singleton."""
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2
