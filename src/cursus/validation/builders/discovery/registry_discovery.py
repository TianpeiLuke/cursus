"""
Registry-based step builder discovery utilities.

This module provides utilities for automatically discovering step builders
and their paths using the central registry, making tests adaptive to changes
in the step builder ecosystem.
"""

import importlib
from typing import Dict, List, Tuple, Optional, Type, Any
from pathlib import Path

from ...registry.step_names import STEP_NAMES, get_steps_by_sagemaker_type
from ...core.base.builder_base import StepBuilderBase


class RegistryStepDiscovery:
    """
    Registry-based step builder discovery utility.

    This class provides methods to automatically discover step builders
    and their module paths using the central registry, making tests
    adaptive to changes in the step builder ecosystem.
    """

    # Steps that should be excluded from testing (abstract/base steps kept for parity)
    EXCLUDED_FROM_TESTING = {
        "Processing",  # Base processing step - no concrete implementation
        "Base",  # Base step - abstract
    }

    @staticmethod
    def get_steps_by_sagemaker_type(
        sagemaker_step_type: str, exclude_abstract: bool = True
    ) -> List[str]:
        """
        Get all step names for a specific SageMaker step type from registry.

        Args:
            sagemaker_step_type: The SageMaker step type (e.g., 'Training', 'Transform', 'CreateModel')
            exclude_abstract: Whether to exclude abstract/base steps from the results

        Returns:
            List of step names that match the specified SageMaker step type
        """
        steps = get_steps_by_sagemaker_type(sagemaker_step_type)

        if exclude_abstract:
            steps = [
                step
                for step in steps
                if step not in RegistryStepDiscovery.EXCLUDED_FROM_TESTING
            ]

        return steps

    @staticmethod
    def get_testable_steps_by_sagemaker_type(sagemaker_step_type: str) -> List[str]:
        """
        Get all testable step names for a specific SageMaker step type from registry.
        Excludes abstract/base steps that shouldn't be tested.

        Args:
            sagemaker_step_type: The SageMaker step type (e.g., 'Training', 'Transform', 'CreateModel')

        Returns:
            List of testable step names that match the specified SageMaker step type
        """
        return RegistryStepDiscovery.get_steps_by_sagemaker_type(
            sagemaker_step_type, exclude_abstract=True
        )

    @staticmethod
    def is_step_testable(step_name: str) -> bool:
        """
        Check if a step should be included in testing.

        Args:
            step_name: The step name to check

        Returns:
            True if the step should be tested, False if it should be excluded
        """
        return step_name not in RegistryStepDiscovery.EXCLUDED_FROM_TESTING

    @staticmethod
    def get_builder_class_path(step_name: str) -> Tuple[str, str]:
        """
        Get the module path and class name for a step builder using step catalog.

        Args:
            step_name: The step name from the registry

        Returns:
            Tuple of (module_path, class_name)

        Raises:
            KeyError: If step_name is not found in registry
            ValueError: If registry entry is missing required information
        """
        # Use step catalog for discovery
        try:
            from ...step_catalog import StepCatalog
            
            # ✅ PORTABLE: Package-only discovery for builder class paths
            # Works in PyPI, source, and submodule scenarios
            # StepCatalog autonomously finds package root regardless of deployment
            catalog = StepCatalog(workspace_dirs=None)  # None for package-only discovery
            
            # Use catalog's get_builder_class_path method
            builder_path = catalog.get_builder_class_path(step_name)
            if builder_path:
                # Extract module path and class name from builder path
                if builder_path.startswith('cursus.'):
                    module_path, class_name = builder_path.rsplit('.', 1)
                    return module_path, class_name
                    
        except ImportError:
            raise ImportError("Step catalog not available - builder discovery disabled")
        except Exception as e:
            raise ValueError(f"Step catalog discovery failed for '{step_name}': {e}")
            
        # If we get here, step catalog didn't find the builder
        raise KeyError(f"Step '{step_name}' not found in step catalog")

    @staticmethod
    def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
        """
        Dynamically load a step builder class using step catalog.

        Args:
            step_name: The step name from the registry

        Returns:
            The loaded step builder class

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class cannot be found in the module
        """
        # Use step catalog for loading
        try:
            from ...step_catalog import StepCatalog
            
            # ✅ PORTABLE: Package-only discovery for builder class loading
            # Works in PyPI, source, and submodule scenarios
            # StepCatalog autonomously finds package root regardless of deployment
            catalog = StepCatalog(workspace_dirs=None)  # None for package-only discovery
            
            # Use catalog's load_builder_class method
            builder_class = catalog.load_builder_class(step_name)
            if builder_class:
                return builder_class
                
        except ImportError:
            raise ImportError("Step catalog not available - builder loading disabled")
        except Exception as e:
            raise ImportError(f"Step catalog loading failed for '{step_name}': {e}")
            
        # If we get here, step catalog didn't find the builder
        raise AttributeError(f"Builder class for step '{step_name}' not found in step catalog")

    @staticmethod
    def get_all_builder_classes_by_type(
        sagemaker_step_type: str,
    ) -> Dict[str, Type[StepBuilderBase]]:
        """
        Get all builder classes for a specific SageMaker step type.

        Args:
            sagemaker_step_type: The SageMaker step type

        Returns:
            Dictionary mapping step names to their builder classes
        """
        step_names = RegistryStepDiscovery.get_steps_by_sagemaker_type(
            sagemaker_step_type
        )
        builder_classes = {}

        for step_name in step_names:
            try:
                builder_class = RegistryStepDiscovery.load_builder_class(step_name)
                builder_classes[step_name] = builder_class
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load builder class for '{step_name}': {e}")
                continue

        return builder_classes

    @staticmethod
    def get_step_info_from_registry(step_name: str) -> Dict[str, Any]:
        """
        Get complete step information from registry.

        Args:
            step_name: The step name from the registry

        Returns:
            Dictionary containing all registry information for the step
        """
        if step_name not in STEP_NAMES:
            return {}

        return STEP_NAMES[step_name].copy()

    @staticmethod
    def get_all_sagemaker_step_types() -> List[str]:
        """
        Get all unique SageMaker step types from the registry.

        Returns:
            List of unique SageMaker step types
        """
        step_types = set()
        for step_info in STEP_NAMES.values():
            sagemaker_step_type = step_info.get("sagemaker_step_type")
            if sagemaker_step_type:
                step_types.add(sagemaker_step_type)

        return sorted(list(step_types))

    @staticmethod
    def validate_step_builder_availability(step_name: str) -> Dict[str, Any]:
        """
        Validate that a step builder is available and can be loaded.

        Args:
            step_name: The step name to validate

        Returns:
            Dictionary containing validation results
        """
        result = {
            "step_name": step_name,
            "in_registry": False,
            "module_exists": False,
            "class_exists": False,
            "loadable": False,
            "error": None,
        }

        # Check if step is in registry
        if step_name not in STEP_NAMES:
            result["error"] = f"Step '{step_name}' not found in registry"
            return result

        result["in_registry"] = True

        try:
            # Get module and class paths
            module_path, class_name = RegistryStepDiscovery.get_builder_class_path(
                step_name
            )

            # Use StepCatalog for validation instead of manual importlib
            try:
                from ...step_catalog import StepCatalog
                
                catalog = StepCatalog(workspace_dirs=None)  # Package-only discovery
                builder_class = catalog.load_builder_class(step_name)
                
                if builder_class is not None:
                    # Success - StepCatalog found and loaded the builder
                    result["module_exists"] = True
                    result["class_exists"] = True
                    result["loadable"] = True
                    result["builder_class"] = builder_class
                else:
                    # StepCatalog returned None - step exists in registry but builder not loadable
                    result["error"] = f"StepCatalog could not load builder for '{step_name}' (builder may not exist or have import issues)"
                    
            except Exception as e:
                result["error"] = f"StepCatalog validation failed for '{step_name}': {e}"

        except (KeyError, ValueError) as e:
            result["error"] = str(e)

        return result

    @staticmethod
    def generate_discovery_report() -> Dict[str, Any]:
        """
        Generate a comprehensive report of step builder discovery status.

        Returns:
            Dictionary containing discovery report
        """
        report = {
            "total_steps": len(STEP_NAMES),
            "sagemaker_step_types": RegistryStepDiscovery.get_all_sagemaker_step_types(),
            "step_type_counts": {},
            "availability_summary": {"available": 0, "unavailable": 0, "errors": []},
            "step_details": {},
        }

        # Count steps by type
        for step_type in report["sagemaker_step_types"]:
            steps = RegistryStepDiscovery.get_steps_by_sagemaker_type(step_type)
            report["step_type_counts"][step_type] = len(steps)

        # Validate each step
        for step_name in STEP_NAMES.keys():
            validation = RegistryStepDiscovery.validate_step_builder_availability(
                step_name
            )
            report["step_details"][step_name] = validation

            if validation["loadable"]:
                report["availability_summary"]["available"] += 1
            else:
                report["availability_summary"]["unavailable"] += 1
                if validation["error"]:
                    report["availability_summary"]["errors"].append(
                        {"step_name": step_name, "error": validation["error"]}
                    )

        return report



# Convenience functions for backward compatibility and ease of use
def get_training_steps_from_registry() -> List[str]:
    """Get all testable training step names from registry."""
    return RegistryStepDiscovery.get_testable_steps_by_sagemaker_type("Training")


def get_transform_steps_from_registry() -> List[str]:
    """Get all testable transform step names from registry."""
    return RegistryStepDiscovery.get_testable_steps_by_sagemaker_type("Transform")


def get_createmodel_steps_from_registry() -> List[str]:
    """Get all testable createmodel step names from registry."""
    return RegistryStepDiscovery.get_testable_steps_by_sagemaker_type("CreateModel")


def get_processing_steps_from_registry() -> List[str]:
    """Get all testable processing step names from registry."""
    return RegistryStepDiscovery.get_testable_steps_by_sagemaker_type("Processing")


def get_builder_class_path(step_name: str) -> Tuple[str, str]:
    """Get builder class path for a step name."""
    return RegistryStepDiscovery.get_builder_class_path(step_name)


def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
    """Load a builder class by step name."""
    return RegistryStepDiscovery.load_builder_class(step_name)
