"""
Registry-based step builder discovery utilities.

This module provides utilities for automatically discovering step builders
and their paths using the central registry, making tests adaptive to changes
in the step builder ecosystem.
"""

import importlib
from typing import Dict, List, Tuple, Optional, Type, Any
from pathlib import Path

from ...steps.registry.step_names import STEP_NAMES, get_steps_by_sagemaker_type
from ...core.base.builder_base import StepBuilderBase


class RegistryStepDiscovery:
    """
    Registry-based step builder discovery utility.
    
    This class provides methods to automatically discover step builders
    and their module paths using the central registry, making tests
    adaptive to changes in the step builder ecosystem.
    """
    
    @staticmethod
    def get_steps_by_sagemaker_type(sagemaker_step_type: str) -> List[str]:
        """
        Get all step names for a specific SageMaker step type from registry.
        
        Args:
            sagemaker_step_type: The SageMaker step type (e.g., 'Training', 'Transform', 'CreateModel')
            
        Returns:
            List of step names that match the specified SageMaker step type
        """
        return get_steps_by_sagemaker_type(sagemaker_step_type)
    
    @staticmethod
    def get_builder_class_path(step_name: str) -> Tuple[str, str]:
        """
        Get the module path and class name for a step builder from registry.
        
        Args:
            step_name: The step name from the registry
            
        Returns:
            Tuple of (module_path, class_name)
            
        Raises:
            KeyError: If step_name is not found in registry
            ValueError: If registry entry is missing required information
        """
        if step_name not in STEP_NAMES:
            raise KeyError(f"Step '{step_name}' not found in registry")
        
        step_info = STEP_NAMES[step_name]
        
        # Get builder class name
        builder_class_name = step_info.get("builder_step_name")
        if not builder_class_name:
            raise ValueError(f"No builder_step_name found for step '{step_name}'")
        
        # Convert step name to module name (CamelCase to snake_case)
        module_name = RegistryStepDiscovery._camel_to_snake(step_name)
        
        # Construct module path
        module_path = f"cursus.steps.builders.builder_{module_name}_step"
        
        return module_path, builder_class_name
    
    @staticmethod
    def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
        """
        Dynamically load a step builder class from registry information.
        
        Args:
            step_name: The step name from the registry
            
        Returns:
            The loaded step builder class
            
        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class cannot be found in the module
        """
        module_path, class_name = RegistryStepDiscovery.get_builder_class_path(step_name)
        
        try:
            # Import the module
            module = importlib.import_module(f"src.{module_path}")
            
            # Get the class from the module
            builder_class = getattr(module, class_name)
            
            return builder_class
            
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_path}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Could not find class '{class_name}' in module '{module_path}': {e}")
    
    @staticmethod
    def get_all_builder_classes_by_type(sagemaker_step_type: str) -> Dict[str, Type[StepBuilderBase]]:
        """
        Get all builder classes for a specific SageMaker step type.
        
        Args:
            sagemaker_step_type: The SageMaker step type
            
        Returns:
            Dictionary mapping step names to their builder classes
        """
        step_names = RegistryStepDiscovery.get_steps_by_sagemaker_type(sagemaker_step_type)
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
            "error": None
        }
        
        # Check if step is in registry
        if step_name not in STEP_NAMES:
            result["error"] = f"Step '{step_name}' not found in registry"
            return result
        
        result["in_registry"] = True
        
        try:
            # Get module and class paths
            module_path, class_name = RegistryStepDiscovery.get_builder_class_path(step_name)
            
            # Check if module exists
            try:
                module = importlib.import_module(f"src.{module_path}")
                result["module_exists"] = True
                
                # Check if class exists
                if hasattr(module, class_name):
                    result["class_exists"] = True
                    
                    # Try to load the class
                    builder_class = getattr(module, class_name)
                    result["loadable"] = True
                    result["builder_class"] = builder_class
                else:
                    result["error"] = f"Class '{class_name}' not found in module '{module_path}'"
                    
            except ImportError as e:
                result["error"] = f"Could not import module '{module_path}': {e}"
                
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
            "availability_summary": {
                "available": 0,
                "unavailable": 0,
                "errors": []
            },
            "step_details": {}
        }
        
        # Count steps by type
        for step_type in report["sagemaker_step_types"]:
            steps = RegistryStepDiscovery.get_steps_by_sagemaker_type(step_type)
            report["step_type_counts"][step_type] = len(steps)
        
        # Validate each step
        for step_name in STEP_NAMES.keys():
            validation = RegistryStepDiscovery.validate_step_builder_availability(step_name)
            report["step_details"][step_name] = validation
            
            if validation["loadable"]:
                report["availability_summary"]["available"] += 1
            else:
                report["availability_summary"]["unavailable"] += 1
                if validation["error"]:
                    report["availability_summary"]["errors"].append({
                        "step_name": step_name,
                        "error": validation["error"]
                    })
        
        return report
    
    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """
        Convert CamelCase to snake_case.
        
        Args:
            name: CamelCase string
            
        Returns:
            snake_case string
        """
        import re
        
        # Insert underscore before uppercase letters that follow lowercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        
        # Insert underscore before uppercase letters that follow lowercase letters or digits
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        
        return s2.lower()


# Convenience functions for backward compatibility and ease of use
def get_training_steps_from_registry() -> List[str]:
    """Get all training step names from registry."""
    return RegistryStepDiscovery.get_steps_by_sagemaker_type("Training")


def get_transform_steps_from_registry() -> List[str]:
    """Get all transform step names from registry."""
    return RegistryStepDiscovery.get_steps_by_sagemaker_type("Transform")


def get_createmodel_steps_from_registry() -> List[str]:
    """Get all createmodel step names from registry."""
    return RegistryStepDiscovery.get_steps_by_sagemaker_type("CreateModel")


def get_processing_steps_from_registry() -> List[str]:
    """Get all processing step names from registry."""
    return RegistryStepDiscovery.get_steps_by_sagemaker_type("Processing")


def get_builder_class_path(step_name: str) -> Tuple[str, str]:
    """Get builder class path for a step name."""
    return RegistryStepDiscovery.get_builder_class_path(step_name)


def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
    """Load a builder class by step name."""
    return RegistryStepDiscovery.load_builder_class(step_name)
