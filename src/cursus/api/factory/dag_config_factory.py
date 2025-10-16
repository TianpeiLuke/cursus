"""
DAG Configuration Factory

This module provides the main DAGConfigFactory class for interactive pipeline configuration
generation. It orchestrates the step-by-step workflow for collecting user inputs and
generating complete pipeline configurations.

Key Components:
- DAGConfigFactory: Main interactive factory class
- ConfigurationIncompleteError: Exception for incomplete configurations
- Interactive workflow management and validation
"""

from typing import Dict, List, Type, Any, Optional
from pydantic import BaseModel
import logging

from .config_class_mapper import ConfigClassMapper
from .configuration_generator import ConfigurationGenerator
from .field_extractor import extract_field_requirements, extract_non_inherited_fields

logger = logging.getLogger(__name__)


class ConfigurationIncompleteError(Exception):
    """Exception raised when essential configuration fields are missing."""
    pass


class DAGConfigFactory:
    """
    Interactive factory for step-by-step pipeline configuration generation.
    
    This class provides a user-friendly interface for creating pipeline configurations
    by guiding users through the process of setting base configurations and step-specific
    configurations in a structured workflow.
    
    Workflow:
    1. Analyze DAG to get config class mapping
    2. Collect base configurations first
    3. Guide user through step-specific configurations
    4. Generate final config instances with inheritance
    """
    
    def __init__(self, dag):
        """
        Initialize factory with DAG analysis.
        
        Args:
            dag: Pipeline DAG object to create configurations for
        """
        self.dag = dag
        self.config_mapper = ConfigClassMapper()
        self.config_generator = None  # Initialized after base configs are set
        
        # Direct state management
        self._config_class_map = self.config_mapper.map_dag_to_config_classes(dag)
        self.base_config = None  # BasePipelineConfig instance
        self.base_processing_config = None  # BaseProcessingStepConfig instance
        self.step_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized DAGConfigFactory for DAG with {len(self._config_class_map)} steps")
    
    def get_config_class_map(self) -> Dict[str, Type[BaseModel]]:
        """
        Get mapping of DAG node names to config classes (not instances).
        
        Returns:
            Dictionary mapping node names to configuration classes
        """
        return self._config_class_map.copy()
    
    def get_base_config_requirements(self) -> List[Dict[str, Any]]:
        """
        Get base configuration requirements directly from Pydantic class definition.
        
        Extracts field requirements directly from BasePipelineConfig Pydantic class definition.
        
        Returns:
            List of field requirement dictionaries with format:
            {
                'name': str,           # Field name
                'type': str,           # Field type as string
                'description': str,    # Field description from Pydantic Field()
                'required': bool,      # True for required fields, False for optional
                'default': Any         # Default value (only for optional fields)
            }
        """
        try:
            # Import BasePipelineConfig using correct relative import
            from ...core.base.config_base import BasePipelineConfig
            return extract_field_requirements(BasePipelineConfig)
        except ImportError:
            logger.warning("BasePipelineConfig not found, returning empty requirements")
            return []
    
    def get_base_processing_config_requirements(self) -> List[Dict[str, Any]]:
        """
        Get base processing configuration requirements.
        
        Returns only the non-inherited fields specific to BaseProcessingStepConfig.
        Inherited fields from BasePipelineConfig can be obtained by calling get_base_config_requirements().
        
        Returns:
            List of field requirement dictionaries for processing-specific fields
        """
        try:
            # Import configuration classes using correct relative imports
            from ...core.base.config_base import BasePipelineConfig
            from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
            
            # Check if any step requires processing configuration
            needs_processing_config = any(
                self._inherits_from_processing_config(config_class)
                for config_class in self._config_class_map.values()
            )
            
            if not needs_processing_config:
                return []
            
            # Extract only non-inherited fields specific to ProcessingStepConfigBase
            return extract_non_inherited_fields(ProcessingStepConfigBase, BasePipelineConfig)
            
        except ImportError:
            logger.warning("BaseProcessingStepConfig not found, returning empty requirements")
            return []
    
    def set_base_config(self, **kwargs) -> None:
        """
        Set base pipeline configuration from user inputs.
        
        Args:
            **kwargs: Base configuration field values
        """
        try:
            from ...core.base.config_base import BasePipelineConfig
            
            self.base_config = BasePipelineConfig(**kwargs)
            
            # Initialize config generator once base config is set
            self.config_generator = ConfigurationGenerator(
                base_config=self.base_config,
                base_processing_config=self.base_processing_config
            )
            
            logger.info("Base configuration set successfully")
            
        except ImportError:
            logger.error("BasePipelineConfig not available")
            raise ValueError("BasePipelineConfig class not found")
        except Exception as e:
            logger.error(f"Failed to set base configuration: {e}")
            raise ValueError(f"Invalid base configuration: {e}")
    
    def set_base_processing_config(self, **kwargs) -> None:
        """
        Set base processing configuration from user inputs.
        
        Args:
            **kwargs: Base processing configuration field values
        """
        try:
            from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
            
            # Combine base config values with processing-specific values
            combined_kwargs = {}
            if self.base_config:
                combined_kwargs.update(self.config_generator._extract_config_values(self.base_config))
            combined_kwargs.update(kwargs)
            
            self.base_processing_config = ProcessingStepConfigBase(**combined_kwargs)
            
            # Update config generator with processing config
            if self.config_generator:
                self.config_generator.base_processing_config = self.base_processing_config
            
            logger.info("Base processing configuration set successfully")
            
        except ImportError:
            logger.error("BaseProcessingStepConfig not available")
            raise ValueError("BaseProcessingStepConfig class not found")
        except Exception as e:
            logger.error(f"Failed to set base processing configuration: {e}")
            raise ValueError(f"Invalid base processing configuration: {e}")
    
    def get_pending_steps(self) -> List[str]:
        """
        Get list of steps that still need configuration.
        
        Returns:
            List of step names that haven't been configured yet
        """
        return [step_name for step_name in self._config_class_map.keys() 
                if step_name not in self.step_configs]
    
    def get_step_requirements(self, step_name: str) -> List[Dict[str, Any]]:
        """
        Get step-specific requirements excluding inherited base config fields.
        
        Extracts step-specific fields only (excludes base config fields) from the 
        step's configuration class using Pydantic field definitions.
        
        Args:
            step_name: Name of the step to get requirements for
            
        Returns:
            List of field requirement dictionaries for step-specific fields only
        """
        if step_name not in self._config_class_map:
            raise ValueError(f"Step '{step_name}' not found in DAG")
        
        config_class = self._config_class_map[step_name]
        
        # Extract step-specific fields (exclude base config fields)
        return self._extract_step_specific_fields(config_class)
    
    def set_step_config(self, step_name: str, **kwargs) -> None:
        """
        Set configuration for a specific step.
        
        Args:
            step_name: Name of the step to configure
            **kwargs: Step-specific configuration field values
        """
        if step_name not in self._config_class_map:
            raise ValueError(f"Step '{step_name}' not found in DAG")
        
        self.step_configs[step_name] = kwargs
        logger.info(f"Configuration set for step: {step_name}")
    
    def get_configuration_status(self) -> Dict[str, bool]:
        """
        Check which configurations have been filled in.
        
        Returns:
            Dictionary mapping configuration names to completion status
        """
        status = {
            'base_config': self.base_config is not None,
            'base_processing_config': self.base_processing_config is not None or not self.get_base_processing_config_requirements()
        }
        
        # Add step configuration status
        for step_name in self._config_class_map.keys():
            status[f'step_{step_name}'] = step_name in self.step_configs
        
        return status
    
    def generate_all_configs(self) -> List[BaseModel]:
        """
        Generate final list of config instances with enhanced validation guardrails.
        
        Includes enhanced validation to ensure all essential (tier 1) fields
        are provided before configuration generation as a guardrail.
        
        Returns:
            List of configured instances ready for pipeline execution
        """
        # Enhanced validation: Check all essential fields are provided
        validation_errors = self._validate_essential_fields()
        if validation_errors:
            error_msg = "Essential field validation failed:\n" + "\n".join(validation_errors)
            raise ConfigurationIncompleteError(error_msg)
        
        if not self.base_config:
            raise ValueError("Base configuration must be set before generating configs")
        
        if not self.config_generator:
            self.config_generator = ConfigurationGenerator(
                base_config=self.base_config,
                base_processing_config=self.base_processing_config
            )
        
        try:
            configs = self.config_generator.generate_all_instances(
                config_class_map=self._config_class_map,
                step_configs=self.step_configs
            )
            
            logger.info(f"Successfully generated {len(configs)} configuration instances")
            return configs
            
        except Exception as e:
            logger.error(f"Configuration generation failed: {e}")
            raise ValueError(f"Failed to generate configurations: {e}")
    
    def _validate_essential_fields(self) -> List[str]:
        """
        Validate that all essential (tier 1) fields are provided before config generation.
        
        This is a guardrail to ensure all required fields are present across:
        1. Base pipeline configuration
        2. Base processing configuration (if needed)
        3. All step-specific configurations
        
        Returns:
            List of validation error messages (empty if validation passes)
        """
        validation_errors = []
        
        # 1. Validate base configuration essential fields
        if not self.base_config:
            validation_errors.append("Base pipeline configuration is required but not set")
        else:
            # Check if all essential fields in base config are provided
            base_requirements = self.get_base_config_requirements()
            essential_base_fields = [req['name'] for req in base_requirements if req['required']]
            
            for field_name in essential_base_fields:
                field_value = getattr(self.base_config, field_name, None)
                if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                    validation_errors.append(f"Essential base config field '{field_name}' is missing or empty")
        
        # 2. Validate base processing configuration if needed
        processing_requirements = self.get_base_processing_config_requirements()
        if processing_requirements:  # Processing config is needed
            if not self.base_processing_config:
                validation_errors.append("Base processing configuration is required but not set")
            else:
                essential_processing_fields = [req['name'] for req in processing_requirements if req['required']]
                
                for field_name in essential_processing_fields:
                    field_value = getattr(self.base_processing_config, field_name, None)
                    if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                        validation_errors.append(f"Essential processing config field '{field_name}' is missing or empty")
        
        # 3. Validate step-specific essential fields
        for step_name, config_class in self._config_class_map.items():
            if step_name not in self.step_configs:
                validation_errors.append(f"Step '{step_name}' configuration is missing")
                continue
            
            step_requirements = self.get_step_requirements(step_name)
            essential_step_fields = [req['name'] for req in step_requirements if req['required']]
            provided_step_fields = self.step_configs[step_name]
            
            for field_name in essential_step_fields:
                if field_name not in provided_step_fields:
                    validation_errors.append(f"Essential field '{field_name}' missing for step '{step_name}'")
                else:
                    field_value = provided_step_fields[field_name]
                    if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                        validation_errors.append(f"Essential field '{field_name}' is empty for step '{step_name}'")
        
        return validation_errors
    
    def _extract_step_specific_fields(self, config_class: Type[BaseModel]) -> List[Dict[str, Any]]:
        """
        Extract step-specific fields excluding inherited base config fields.
        
        Args:
            config_class: Step configuration class to extract fields from
            
        Returns:
            List of field requirement dictionaries for step-specific fields only
        """
        try:
            from ...core.base.config_base import BasePipelineConfig
            from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
            
            # Determine the appropriate base class to exclude fields from
            if self._inherits_from_processing_config(config_class):
                # If step inherits from ProcessingStepConfigBase, exclude those fields
                base_class = ProcessingStepConfigBase
            else:
                # Otherwise, exclude BasePipelineConfig fields
                base_class = BasePipelineConfig
            
            return extract_non_inherited_fields(config_class, base_class)
            
        except ImportError:
            logger.warning("Base config classes not found, extracting all fields")
            return extract_field_requirements(config_class)
    
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
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the factory state.
        
        Returns:
            Dictionary with factory state summary
        """
        status = self.get_configuration_status()
        
        return {
            'dag_steps': len(self._config_class_map),
            'mapped_config_classes': list(self._config_class_map.keys()),
            'configuration_status': status,
            'completed_steps': len([k for k, v in status.items() if k.startswith('step_') and v]),
            'pending_steps': self.get_pending_steps(),
            'base_config_set': self.base_config is not None,
            'processing_config_set': self.base_processing_config is not None,
            'ready_for_generation': all(status.values())
        }
    
    def save_partial_state(self, file_path: str) -> None:
        """
        Save current factory state for later restoration.
        
        Args:
            file_path: Path to save the state file
        """
        import json
        from pathlib import Path
        
        state = {
            'step_configs': self.step_configs,
            'base_config_dict': self.config_generator._extract_config_values(self.base_config) if self.base_config else None,
            'base_processing_config_dict': self.config_generator._extract_config_values(self.base_processing_config) if self.base_processing_config else None,
            'config_class_map': {k: v.__name__ for k, v in self._config_class_map.items()}
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Factory state saved to: {file_path}")
    
    def load_partial_state(self, file_path: str) -> None:
        """
        Load previously saved factory state.
        
        Args:
            file_path: Path to the saved state file
        """
        import json
        
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        # Restore step configs
        self.step_configs = state.get('step_configs', {})
        
        # Restore base configs if available
        if state.get('base_config_dict'):
            self.set_base_config(**state['base_config_dict'])
        
        if state.get('base_processing_config_dict'):
            self.set_base_processing_config(**state['base_processing_config_dict'])
        
        logger.info(f"Factory state loaded from: {file_path}")
