"""
Configuration Generator

This module provides functionality for generating final configuration instances
with proper base config inheritance. It handles the assembly of step-specific
configurations with base pipeline configurations.

Key Components:
- ConfigurationGenerator: Generates config instances with base config inheritance
- Configuration assembly with proper field inheritance
- Validation and error handling for configuration generation
"""

from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ConfigurationGenerator:
    """
    Generates final configuration instances with base config inheritance.
    
    This class handles the complex task of combining base pipeline configurations
    with step-specific configurations, ensuring proper inheritance and validation.
    """
    
    def __init__(self, 
                 base_config,  # BasePipelineConfig instance
                 base_processing_config = None):  # BaseProcessingStepConfig instance
        """
        Initialize generator with base configurations.
        
        Args:
            base_config: Base pipeline configuration instance
            base_processing_config: Base processing configuration instance (optional)
        """
        self.base_config = base_config
        self.base_processing_config = base_processing_config
    
    def generate_config_instance(self, 
                                config_class: Type[BaseModel], 
                                step_inputs: Dict[str, Any]) -> BaseModel:
        """
        Generate config instance using base config inheritance.
        
        Args:
            config_class: Configuration class to instantiate
            step_inputs: Step-specific input values
            
        Returns:
            Configured instance with base config inheritance applied
        """
        try:
            # Determine inheritance strategy based on config class hierarchy
            if self._inherits_from_processing_config(config_class):
                return self._generate_with_processing_inheritance(config_class, step_inputs)
            elif self._inherits_from_base_config(config_class):
                return self._generate_with_base_inheritance(config_class, step_inputs)
            else:
                # Standalone configuration class
                return self._generate_standalone_config(config_class, step_inputs)
                
        except Exception as e:
            logger.error(f"Failed to generate config for {config_class.__name__}: {e}")
            raise ValueError(f"Configuration generation failed for {config_class.__name__}: {e}")
    
    def generate_all_instances(self, 
                              config_class_map: Dict[str, Type[BaseModel]],
                              step_configs: Dict[str, Dict[str, Any]]) -> List[BaseModel]:
        """
        Generate all configuration instances with proper inheritance.
        
        Args:
            config_class_map: Mapping of step names to configuration classes
            step_configs: Step-specific configuration values
            
        Returns:
            List of configured instances ready for pipeline execution
        """
        generated_configs = []
        
        for step_name, config_class in config_class_map.items():
            step_inputs = step_configs.get(step_name, {})
            
            try:
                config_instance = self.generate_config_instance(config_class, step_inputs)
                generated_configs.append(config_instance)
                logger.info(f"Generated config for step: {step_name}")
                
            except Exception as e:
                logger.error(f"Failed to generate config for step {step_name}: {e}")
                raise ValueError(f"Configuration generation failed for step {step_name}: {e}")
        
        return generated_configs
    
    def _inherits_from_processing_config(self, config_class: Type[BaseModel]) -> bool:
        """
        Check if config class inherits from BaseProcessingStepConfig.
        
        Args:
            config_class: Configuration class to check
            
        Returns:
            True if class inherits from BaseProcessingStepConfig
        """
        try:
            # Check method resolution order for BaseProcessingStepConfig
            mro = getattr(config_class, '__mro__', [])
            for base_class in mro:
                if hasattr(base_class, '__name__') and 'BaseProcessingStepConfig' in base_class.__name__:
                    return True
            return False
        except Exception:
            return False
    
    def _inherits_from_base_config(self, config_class: Type[BaseModel]) -> bool:
        """
        Check if config class inherits from BasePipelineConfig.
        
        Args:
            config_class: Configuration class to check
            
        Returns:
            True if class inherits from BasePipelineConfig
        """
        try:
            # Check method resolution order for BasePipelineConfig
            mro = getattr(config_class, '__mro__', [])
            for base_class in mro:
                if hasattr(base_class, '__name__') and 'BasePipelineConfig' in base_class.__name__:
                    return True
            return False
        except Exception:
            return False
    
    def _generate_with_processing_inheritance(self, 
                                            config_class: Type[BaseModel], 
                                            step_inputs: Dict[str, Any]) -> BaseModel:
        """
        Generate config instance with processing config inheritance.
        
        Args:
            config_class: Configuration class that inherits from BaseProcessingStepConfig
            step_inputs: Step-specific input values
            
        Returns:
            Configuration instance with processing inheritance applied
        """
        # Combine base config, processing config, and step-specific inputs
        combined_inputs = {}
        
        # Start with base pipeline config fields
        if self.base_config:
            combined_inputs.update(self._extract_config_values(self.base_config))
        
        # Add base processing config fields
        if self.base_processing_config:
            combined_inputs.update(self._extract_config_values(self.base_processing_config))
        
        # Override with step-specific inputs
        combined_inputs.update(step_inputs)
        
        # Try to use from_base_config method if available
        if hasattr(config_class, 'from_base_config') and self.base_processing_config:
            try:
                return config_class.from_base_config(self.base_processing_config, **step_inputs)
            except Exception as e:
                logger.warning(f"from_base_config failed for {config_class.__name__}: {e}")
        
        # Fallback to direct instantiation
        return config_class(**combined_inputs)
    
    def _generate_with_base_inheritance(self, 
                                      config_class: Type[BaseModel], 
                                      step_inputs: Dict[str, Any]) -> BaseModel:
        """
        Generate config instance with base config inheritance.
        
        Args:
            config_class: Configuration class that inherits from BasePipelineConfig
            step_inputs: Step-specific input values
            
        Returns:
            Configuration instance with base inheritance applied
        """
        # Combine base config and step-specific inputs
        combined_inputs = {}
        
        # Start with base pipeline config fields
        if self.base_config:
            combined_inputs.update(self._extract_config_values(self.base_config))
        
        # Override with step-specific inputs
        combined_inputs.update(step_inputs)
        
        # Try to use from_base_config method if available
        if hasattr(config_class, 'from_base_config') and self.base_config:
            try:
                return config_class.from_base_config(self.base_config, **step_inputs)
            except Exception as e:
                logger.warning(f"from_base_config failed for {config_class.__name__}: {e}")
        
        # Fallback to direct instantiation
        return config_class(**combined_inputs)
    
    def _generate_standalone_config(self, 
                                   config_class: Type[BaseModel], 
                                   step_inputs: Dict[str, Any]) -> BaseModel:
        """
        Generate standalone config instance without inheritance.
        
        Args:
            config_class: Standalone configuration class
            step_inputs: Step-specific input values
            
        Returns:
            Configuration instance created from step inputs only
        """
        return config_class(**step_inputs)
    
    def _extract_config_values(self, config_instance: BaseModel) -> Dict[str, Any]:
        """
        Extract field values from a Pydantic V2 configuration instance.
        
        Args:
            config_instance: Pydantic V2 configuration instance to extract values from
            
        Returns:
            Dictionary of field names to values
        """
        try:
            # Use Pydantic V2's model_dump method to get all field values
            if hasattr(config_instance, 'model_dump'):
                return config_instance.model_dump()
            else:
                # Fallback: extract using __dict__
                return {k: v for k, v in config_instance.__dict__.items() if not k.startswith('_')}
        except Exception as e:
            logger.warning(f"Failed to extract config values: {e}")
            return {}
    
    def validate_generated_configs(self, configs: List[BaseModel]) -> Dict[str, List[str]]:
        """
        Validate generated configuration instances.
        
        Args:
            configs: List of generated configuration instances
            
        Returns:
            Dictionary mapping config class names to validation error lists
        """
        validation_results = {}
        
        for config in configs:
            config_name = config.__class__.__name__
            errors = []
            
            try:
                # Try to validate the config instance (Pydantic V2)
                if hasattr(config, 'model_validate'):
                    config.model_validate(config.model_dump())
                
            except Exception as e:
                errors.append(str(e))
            
            # Check for required fields that might be None or empty
            errors.extend(self._check_required_fields(config))
            
            if errors:
                validation_results[config_name] = errors
        
        return validation_results
    
    def _check_required_fields(self, config: BaseModel) -> List[str]:
        """
        Check for required fields that are None or empty (Pydantic V2).
        
        Args:
            config: Pydantic V2 configuration instance to check
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Get field information (Pydantic V2+ compatible)
            model_fields = getattr(config, 'model_fields', None)
            if model_fields is not None:
                for field_name, field_info in model_fields.items():
                    if hasattr(field_info, 'is_required') and field_info.is_required():
                        value = getattr(config, field_name, None)
                        if value is None or (isinstance(value, str) and not value.strip()):
                            errors.append(f"Required field '{field_name}' is missing or empty")
            
        except Exception as e:
            logger.warning(f"Failed to check required fields: {e}")
        
        return errors
    
    def get_config_summary(self, configs: List[BaseModel]) -> Dict[str, Dict[str, Any]]:
        """
        Get summary information about generated configurations.
        
        Args:
            configs: List of generated configuration instances
            
        Returns:
            Dictionary with summary information for each config
        """
        summary = {}
        
        for config in configs:
            config_name = config.__class__.__name__
            
            try:
                config_dict = self._extract_config_values(config)
                
                summary[config_name] = {
                    'class_name': config_name,
                    'field_count': len(config_dict),
                    'required_fields': self._count_required_fields(config),
                    'optional_fields': len(config_dict) - self._count_required_fields(config),
                    'has_base_inheritance': self._inherits_from_base_config(config.__class__),
                    'has_processing_inheritance': self._inherits_from_processing_config(config.__class__)
                }
                
            except Exception as e:
                summary[config_name] = {
                    'class_name': config_name,
                    'error': str(e)
                }
        
        return summary
    
    def _count_required_fields(self, config: BaseModel) -> int:
        """
        Count the number of required fields in a Pydantic V2 configuration.
        
        Args:
            config: Pydantic V2 configuration instance
            
        Returns:
            Number of required fields
        """
        try:
            model_fields = getattr(config, 'model_fields', None)
            if model_fields is not None:
                return sum(1 for field_info in model_fields.values() 
                          if hasattr(field_info, 'is_required') and field_info.is_required())
            return 0
        except Exception:
            return 0
