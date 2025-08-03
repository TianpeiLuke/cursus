"""
Step Builder Registry for the Pipeline API.

This module provides a centralized registry mapping configuration types
to step builder classes, enabling automatic resolution during pipeline construction.
This implementation uses the step_names registry as the single source of truth
for step naming and supports auto-discovery of step builders.
"""

from typing import Dict, Type, List, Optional, Any, Callable, TYPE_CHECKING
import logging
import importlib
import inspect
import pkgutil
import sys

# Use TYPE_CHECKING to break circular import
if TYPE_CHECKING:
    from ...core.base.builder_base import StepBuilderBase
    from ...core.base.config_base import BasePipelineConfig
else:
    # Placeholder for runtime
    StepBuilderBase = Any
    BasePipelineConfig = Any

from .step_names import STEP_NAMES, get_all_step_names, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES
from .exceptions import RegistryError

# Create an explicit logger for the builder registry
registry_logger = logging.getLogger("pipeline_registry.builder_registry")

# Configure the logger with a default handler if none exists
if not registry_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    registry_logger.addHandler(handler)
    registry_logger.setLevel(logging.INFO)


# Create reverse mapping from builder step names to canonical step names for efficient lookup
REVERSE_BUILDER_MAPPING = {
    info["builder_step_name"]: step_name 
    for step_name, info in STEP_NAMES.items()
}

# Decorator for auto-registering step builders
def register_builder(step_type: str = None):
    """
    Decorator to automatically register a step builder class.
    
    Args:
        step_type: Optional step type name. If not provided,
                  will be derived from the class name using the STEP_NAMES registry.
    """
    def decorator(cls):
        # Import at runtime to avoid circular import
        from ...core.base.builder_base import StepBuilderBase as RuntimeStepBuilderBase
        if not issubclass(cls, RuntimeStepBuilderBase):
            raise TypeError(f"@register_builder can only be used on StepBuilderBase subclasses: {cls.__name__}")
        
        # Determine step type if not provided
        nonlocal step_type
        if step_type is None:
            class_name = cls.__name__
            
            # First, try to find in STEP_NAMES registry (single source of truth)
            if class_name in REVERSE_BUILDER_MAPPING:
                step_type = REVERSE_BUILDER_MAPPING[class_name]
                registry_logger.debug(f"Found step type '{step_type}' for class '{class_name}' in STEP_NAMES registry")
            else:
                # Fallback to current logic for backward compatibility
                registry_logger.debug(f"Class '{class_name}' not found in STEP_NAMES registry, using fallback logic")
                if class_name.endswith('StepBuilder'):
                    step_type = class_name[:-11]  # Remove 'StepBuilder'
                else:
                    step_type = class_name
                registry_logger.warning(f"Using fallback step type '{step_type}' for class '{class_name}'. Consider adding to STEP_NAMES registry.")
        
        # Register the class
        StepBuilderRegistry.register_builder_class(step_type, cls)
        return cls
    
    return decorator


class StepBuilderRegistry:
    """
    Centralized registry mapping step types to builder classes.
    
    This registry maintains the mapping between step types and their
    corresponding step builder classes, enabling automatic resolution
    during pipeline construction. It uses the step_names registry as 
    the single source of truth for step naming.
    
    The registry has been enhanced to handle job type variants, so that
    configurations with the same class but different job types can be
    properly mapped to their step builders.
    """
    
    # Core registry mapping step types to builders - auto-populated during initialization
    BUILDER_REGISTRY = {}  
    
    # Legacy aliases for backward compatibility
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package",  # Legacy name from before standardization
        "MIMSPayload": "Payload",    # Legacy name from before standardization
        "ModelRegistration": "Registration",  # Legacy name from before standardization
        "PytorchTraining": "PyTorchTraining",  # Case sensitivity difference
        "PytorchModel": "PyTorchModel",  # Case sensitivity difference
    }
    
    @classmethod
    def register_builder_class(cls, step_type: str, builder_class: Type[StepBuilderBase]) -> None:
        """
        Register a builder class directly in the registry.
        
        Args:
            step_type: Step type name
            builder_class: Step builder class
        """
        # Import at runtime to avoid circular import
        from ...core.base.builder_base import StepBuilderBase as RuntimeStepBuilderBase
        if not issubclass(builder_class, RuntimeStepBuilderBase):
            raise ValueError(f"Builder class must extend StepBuilderBase: {builder_class}")
        
        cls.BUILDER_REGISTRY[step_type] = builder_class
        registry_logger.info(f"Registered builder: {step_type} -> {builder_class.__name__}")
    
    @classmethod
    def discover_builders(cls):
        """
        Automatically discover and register step builders.
        
        Returns:
            Dict[str, Type[StepBuilderBase]]: Dictionary of discovered builders
        """
        
        discovered_builders = {}
        
        # Get the package containing step builders
        try:
            from ..builders import __path__ as builders_path
            
            # Walk through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(builders_path):
                if module_name.startswith('builder_'):
                    try:
                        # Import the module
                        module = importlib.import_module(f"..builders.{module_name}", __name__)
                        
                        # Find builder classes in the module
                        # Import at runtime to avoid circular import
                        from ...core.base.builder_base import StepBuilderBase as RuntimeStepBuilderBase
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, RuntimeStepBuilderBase) and 
                                obj != RuntimeStepBuilderBase):
                                
                                # Use the same logic as register_builder decorator for consistency
                                step_type = None
                                
                                # First, try to find in STEP_NAMES registry (single source of truth)
                                if name in REVERSE_BUILDER_MAPPING:
                                    step_type = REVERSE_BUILDER_MAPPING[name]
                                    registry_logger.debug(f"Found step type '{step_type}' for class '{name}' in STEP_NAMES registry")
                                else:
                                    # Fallback to current logic for backward compatibility
                                    if name.endswith('StepBuilder'):
                                        step_type = name[:-11]  # Remove 'StepBuilder'
                                    else:
                                        step_type = name
                                    registry_logger.debug(f"Class '{name}' not found in STEP_NAMES registry, using fallback step type '{step_type}'")
                                
                                discovered_builders[step_type] = obj
                                registry_logger.debug(f"Discovered builder: {step_type} -> {name}")
                    
                    except (ImportError, AttributeError) as e:
                        registry_logger.warning(f"Error discovering builders in {module_name}: {e}")
        
        except ImportError:
            registry_logger.warning("Could not import builders package for builder discovery")
        
        # Manual import and registration of known step builders to ensure backward compatibility
        cls._register_known_builders(discovered_builders)
        
        return discovered_builders
    
    @classmethod
    def _register_known_builders(cls, builder_map: Dict[str, Type[StepBuilderBase]]) -> None:
        """Register known step builders to ensure backward compatibility."""
        # Import all step builders
        from ..builders.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
        from ..builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        from ..builders.builder_training_step_xgboost import XGBoostTrainingStepBuilder
        from ..builders.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
        from ..builders.builder_package_step import PackageStepBuilder
        from ..builders.builder_payload_step import PayloadStepBuilder
        from ..builders.builder_registration_step import RegistrationStepBuilder
        from ..builders.builder_training_step_pytorch import PyTorchTrainingStepBuilder
        from ..builders.builder_model_step_pytorch import PyTorchModelStepBuilder
        from ..builders.builder_model_step_xgboost import XGBoostModelStepBuilder
        from ..builders.builder_batch_transform_step import BatchTransformStepBuilder
        from ..builders.builder_model_calibration_step import ModelCalibrationStepBuilder
        from ..builders.builder_currency_conversion_step import CurrencyConversionStepBuilder
        from ..builders.builder_risk_table_mapping_step import RiskTableMappingStepBuilder
        from ..builders.builder_dummy_training_step import DummyTrainingStepBuilder
        
        # Core registry with canonical step names from the central step registry
        known_builders = {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessing": TabularPreprocessingStepBuilder,
            "XGBoostTraining": XGBoostTrainingStepBuilder,
            "XGBoostModelEval": XGBoostModelEvalStepBuilder, 
            "Package": PackageStepBuilder,  # Using standardized name
            "Payload": PayloadStepBuilder,   # Using standardized name 
            "Registration": RegistrationStepBuilder,  # Using standardized name
            "PyTorchTraining": PyTorchTrainingStepBuilder,  # Canonical: PyTorchTraining (not PytorchTraining)
            "PyTorchModel": PyTorchModelStepBuilder,  # Canonical: PyTorchModel (not PytorchModel)
            "XGBoostModel": XGBoostModelStepBuilder,
            "BatchTransform": BatchTransformStepBuilder,
            "ModelCalibration": ModelCalibrationStepBuilder,
            "CurrencyConversion": CurrencyConversionStepBuilder,
            "RiskTableMapping": RiskTableMappingStepBuilder,
            "DummyTraining": DummyTrainingStepBuilder,
        }
        
        # Add any known builders that weren't discovered automatically
        for step_type, builder_class in known_builders.items():
            if step_type not in builder_map:
                builder_map[step_type] = builder_class
                registry_logger.debug(f"Added known builder: {step_type} -> {builder_class.__name__}")
    
    def __init__(self):
        """Initialize the registry."""
        self._custom_builders = {}
        self.logger = registry_logger
        
        # Populate the registry if empty (first initialization)
        if not self.__class__.BUILDER_REGISTRY:
            # Get core builders through discovery
            discovered = self.__class__.discover_builders()
            self.__class__.BUILDER_REGISTRY = discovered
            
            # Log discovery results
            self.logger.info(f"Discovered {len(discovered)} step builders")
    
    def get_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Get the complete builder registry.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        # Combine default and custom builders
        builder_map = self.BUILDER_REGISTRY.copy()
        builder_map.update(self._custom_builders)
        return builder_map
    
    def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Type[StepBuilderBase]:
        """
        Get step builder class for a specific configuration.
        
        Args:
            config: Configuration instance
            node_name: Original DAG node name (optional)
            
        Returns:
            Step builder class
            
        Raises:
            RegistryError: If no builder found for config type
        """
        config_class_name = type(config).__name__
        job_type = getattr(config, 'job_type', None)
        
        # First try with node name if provided
        if node_name:
            try:
                # Try to find a step builder using the node name directly
                if self.is_step_type_supported(node_name):
                    self.logger.info(f"Found builder for exact node name: {node_name}")
                    return self.get_builder_for_step_type(node_name)
            except Exception as e:
                # If it fails, continue with standard approach
                self.logger.debug(f"Could not find builder using exact node name '{node_name}': {e}")
        
        # Convert config class name to step type, considering job type
        step_type = self._config_class_to_step_type(config_class_name, node_name=node_name, job_type=job_type)
        
        # Try with the full step type (including job type if present)
        builder_map = self.get_builder_map()
        if step_type in builder_map:
            self.logger.info(f"Found builder for step type: {step_type}")
            return builder_map[step_type]
        
        # If step type includes job type but no builder found, try without job type
        if '_' in step_type:
            base_step_type = step_type.rsplit('_', 1)[0]
            if base_step_type in builder_map:
                self.logger.info(f"Found builder using base step type: {base_step_type}")
                return builder_map[base_step_type]
        
        # If still no match, try registering CradleDataLoading_training as an alias for CradleDataLoading
        if step_type.startswith("CradleDataLoading_") and "CradleDataLoading" in builder_map:
            self.logger.info(f"Using CradleDataLoading builder for {step_type}")
            return builder_map["CradleDataLoading"]
        
        # Similarly for TabularPreprocessing
        if step_type.startswith("TabularPreprocessing_") and "TabularPreprocessing" in builder_map:
            self.logger.info(f"Using TabularPreprocessing builder for {step_type}")
            return builder_map["TabularPreprocessing"]
            
        # Similarly for XGBoostModelEval
        if step_type.startswith("XGBoostModelEval_") and "XGBoostModelEval" in builder_map:
            self.logger.info(f"Using XGBoostModelEval builder for {step_type}")
            return builder_map["XGBoostModelEval"]
        
        # If still not found, raise error
        available_types = list(builder_map.keys())
        raise RegistryError(
            f"No step builder found for config type '{config_class_name}' (step type: '{step_type}')",
            unresolvable_types=[step_type],
            available_builders=available_types
        )
    
    def get_builder_for_step_type(self, step_type: str) -> Type[StepBuilderBase]:
        """
        Get step builder class for a specific step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            Step builder class
            
        Raises:
            RegistryError: If no builder found for step type
        """
        # Check if the step_type is a legacy alias and convert to canonical name
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        
        builder_map = self.get_builder_map()
        if canonical_step_type not in builder_map:
            available_types = list(builder_map.keys())
            raise RegistryError(
                f"No step builder found for step type '{step_type}' (canonical: '{canonical_step_type}')",
                unresolvable_types=[step_type],
                available_builders=available_types
            )
        
        return builder_map[canonical_step_type]
    
    def register_builder(self, step_type: str, builder_class: Type[StepBuilderBase]) -> None:
        """
        Register a new step builder (for extensibility).
        
        Args:
            step_type: Step type name
            builder_class: Step builder class
        """
        # Import at runtime to avoid circular import
        from ...core.base.builder_base import StepBuilderBase as RuntimeStepBuilderBase
        if not issubclass(builder_class, RuntimeStepBuilderBase):
            raise ValueError(f"Builder class must extend StepBuilderBase: {builder_class}")
        
        self._custom_builders[step_type] = builder_class
        self.logger.info(f"Registered custom builder: {step_type} -> {builder_class.__name__}")
    
    def unregister_builder(self, step_type: str) -> None:
        """
        Unregister a custom step builder.
        
        Args:
            step_type: Step type name
        """
        if step_type in self._custom_builders:
            del self._custom_builders[step_type]
            self.logger.info(f"Unregistered custom builder: {step_type}")
        else:
            self.logger.warning(f"Attempted to unregister non-existent custom builder: {step_type}")
    
    def list_supported_step_types(self) -> List[str]:
        """
        List all supported step types.
        
        Returns:
            List of supported step type names
        """
        canonical_types = list(self.get_builder_map().keys())
        # Include legacy aliases for backward compatibility
        all_types = canonical_types + list(self.LEGACY_ALIASES.keys())
        return sorted(all_types)
    
    def is_step_type_supported(self, step_type: str) -> bool:
        """
        Check if a step type is supported.
        
        Args:
            step_type: Step type name
            
        Returns:
            True if supported, False otherwise
        """
        # Check both canonical names and legacy aliases
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        return canonical_step_type in self.get_builder_map()
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """
        Get possible configuration class names for a step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            List of possible configuration class names
        """
        # First, check if this is a legacy alias
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        
        # Try to use the central registry from step_names.py
        possible_configs = []
        
        # Look up config class in step registry
        if canonical_step_type in STEP_NAMES:
            config_class = STEP_NAMES[canonical_step_type].get("config_class")
            if config_class:
                possible_configs.append(config_class)
        
        # Add standard patterns as fallback
        if not possible_configs:
            self.logger.warning(f"Step type '{step_type}' not found in STEP_NAMES registry, using fallback patterns")
            possible_configs.append(f"{step_type}Config")
            possible_configs.append(f"{step_type}StepConfig")
        
        return possible_configs
    
    def _extract_job_type(self, node_name: str):
        """
        Extract job type information from a node name.
        
        Args:
            node_name: Node name from DAG
            
        Returns:
            Tuple of (base_name, job_type)
        """
        # Pattern 1: BaseType_JobType (e.g., CradleDataLoading_training)
        import re
        match = re.match(r'^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$', node_name)
        if match:
            base_name, job_type = match.groups()
            return base_name, job_type
        
        # If no pattern match, return the original name with no job type
        return node_name, None
    
    def _config_class_to_step_type(self, config_class_name: str, node_name: str = None, job_type: str = None) -> str:
        """
        Convert configuration class name to step type.
        
        Args:
            config_class_name: Configuration class name
            node_name: Original DAG node name (optional)
            job_type: Job type to append to step type (optional)
            
        Returns:
            Step type name
        """
        # Extract job type from node name if provided and job_type not explicitly provided
        extracted_job_type = None
        if node_name and not job_type:
            _, extracted_job_type = self._extract_job_type(node_name)
            job_type = extracted_job_type
        
        # Use the central registry from step_names.py - already imported
        if config_class_name in CONFIG_STEP_REGISTRY:
            canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
            # Check if this is a canonical name that has a legacy alias
            for legacy_name, canonical_name in self.LEGACY_ALIASES.items():
                if canonical_name == canonical_step_name:
                    base_step_type = legacy_name
                    # Append job type if provided
                    if job_type:
                        return f"{base_step_type}_{job_type}"
                    return base_step_type  # Return legacy name for backward compatibility
            
            # Just use the canonical name from the registry
            base_step_type = canonical_step_name
            # Append job type if provided
            if job_type:
                return f"{base_step_type}_{job_type}"
            return base_step_type
            
        # Fallback to the old conversion logic for compatibility
        self.logger.warning(f"Config class '{config_class_name}' not found in CONFIG_STEP_REGISTRY, using fallback logic")
        
        # Remove common suffixes
        step_type = config_class_name
        
        # Remove 'Config' suffix
        if step_type.endswith('Config'):
            step_type = step_type[:-6]
        
        # Remove 'Step' suffix if present
        if step_type.endswith('Step'):
            step_type = step_type[:-4]
        
        # Handle common naming patterns
        if step_type == "CradleDataLoad":
            base_step_type = "CradleDataLoading"
        else:
            base_step_type = step_type
        
        # Append job type if provided
        if job_type:
            return f"{base_step_type}_{job_type}"
        
        return base_step_type
    
    def validate_registry(self) -> Dict[str, List[str]]:
        """
        Validate the registry for consistency.
        
        Returns:
            Dictionary with validation results:
            - 'valid': List of valid mappings
            - 'invalid': List of invalid mappings with reasons
            - 'missing': List of step names in step registry but missing builders
        """
        results = {'valid': [], 'invalid': [], 'missing': []}
        
        builder_map = self.get_builder_map()
        
        # Validate existing mappings
        for step_type, builder_class in builder_map.items():
            try:
                # Check if builder class is valid
                # Import at runtime to avoid circular import
                from ...core.base.builder_base import StepBuilderBase as RuntimeStepBuilderBase
                if not issubclass(builder_class, RuntimeStepBuilderBase):
                    results['invalid'].append(f"{step_type}: Not a StepBuilderBase subclass")
                    continue
                
                # Check if builder can be instantiated (basic check)
                # Note: We can't fully instantiate without a config, but we can check the class
                if not hasattr(builder_class, '__init__'):
                    results['invalid'].append(f"{step_type}: Missing __init__ method")
                    continue
                
                results['valid'].append(f"{step_type} -> {builder_class.__name__}")
                
            except Exception as e:
                results['invalid'].append(f"{step_type}: {str(e)}")
        
        # Check for missing step names from the registry
        for step_name in get_all_step_names():
            if step_name not in builder_map and step_name not in self.LEGACY_ALIASES.values():
                results['missing'].append(f"{step_name}: No builder registered")
        
        return results
    
    def get_registry_stats(self) -> Dict[str, int]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        builder_map = self.get_builder_map()
        return {
            'total_builders': len(builder_map),
            'default_builders': len(self.BUILDER_REGISTRY),
            'custom_builders': len(self._custom_builders),
            'legacy_aliases': len(self.LEGACY_ALIASES),
            'step_registry_names': len(get_all_step_names()),
        }


# Global registry instance
_global_registry = None


def get_global_registry() -> StepBuilderRegistry:
    """
    Get the global step builder registry instance.
    
    Returns:
        Global StepBuilderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StepBuilderRegistry()
    return _global_registry


def register_global_builder(step_type: str, builder_class: Type[StepBuilderBase]) -> None:
    """
    Register a builder in the global registry.
    
    Args:
        step_type: Step type name
        builder_class: Step builder class
    """
    registry = get_global_registry()
    registry.register_builder(step_type, builder_class)


def list_global_step_types() -> List[str]:
    """
    List all step types in the global registry.
    
    Returns:
        List of supported step type names
    """
    registry = get_global_registry()
    return registry.list_supported_step_types()
