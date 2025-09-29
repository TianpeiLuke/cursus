"""
Step Catalog Configuration Provider.

This module provides a simplified configuration provider that leverages the existing
step catalog system to eliminate redundancy and provide proper configuration instances
for testing instead of primitive Mock() objects.
"""

import logging
from typing import Dict, Type, Any, Optional
from types import SimpleNamespace


class StepCatalogConfigProvider:
    """
    Simplified configuration provider that leverages existing step catalog system.
    
    This class eliminates redundancy by using the step catalog's existing
    configuration discovery capabilities directly, with zero hard-coded
    configuration data.
    """
    
    def __init__(self):
        """Initialize with lazy loading for performance."""
        self._step_catalog = None
        self._config_classes = None
        self.logger = logging.getLogger(__name__)
    
    @property
    def step_catalog(self):
        """Lazy-loaded step catalog instance."""
        if self._step_catalog is None:
            try:
                from ...step_catalog import StepCatalog
                self._step_catalog = StepCatalog(workspace_dirs=None)
            except ImportError as e:
                self.logger.debug(f"Step catalog not available: {e}")
                self._step_catalog = None
        return self._step_catalog
    
    @property
    def config_classes(self) -> Dict[str, Type]:
        """Lazy-loaded configuration classes from step catalog."""
        if self._config_classes is None and self.step_catalog is not None:
            try:
                self._config_classes = self.step_catalog.build_complete_config_classes()
            except Exception as e:
                self.logger.debug(f"Failed to build config classes: {e}")
                self._config_classes = {}
        return self._config_classes or {}
    
    def get_config_for_builder(self, builder_class: Type) -> Any:
        """
        Get proper configuration for builder using step catalog discovery.
        
        Args:
            builder_class: The step builder class requiring configuration
            
        Returns:
            Configuration instance (proper config class or fallback)
        """
        builder_name = builder_class.__name__
        
        try:
            # Direct step catalog integration - no redundant logic
            config_class_name = self._map_builder_to_config_class(builder_name)
            
            if config_class_name in self.config_classes:
                config_class = self.config_classes[config_class_name]
                config_instance = self._create_config_instance(config_class, builder_class)
                
                if config_instance:
                    self.logger.debug(f"✅ Step catalog config: {config_class_name} for {builder_name}")
                    return config_instance
            
            # Simple fallback to existing mock factory (reuse existing code)
            return self._fallback_to_existing_mock_factory(builder_class)
            
        except Exception as e:
            self.logger.debug(f"Config creation failed for {builder_name}: {e}")
            return self._fallback_to_existing_mock_factory(builder_class)
    
    def _map_builder_to_config_class(self, builder_name: str) -> str:
        """Simple builder name to config class mapping."""
        if builder_name.endswith('StepBuilder'):
            base_name = builder_name[:-11]  # Remove 'StepBuilder'
            return f"{base_name}Config"
        return f"{builder_name}Config"
    
    def _create_config_instance(self, config_class: Type, builder_class: Type) -> Optional[Any]:
        """Create config instance using step catalog's from_base_config pattern."""
        try:
            # Use step catalog's existing base config creation
            base_config = self._get_base_config(builder_class)
            if base_config is None:
                return None
            
            # Get builder-specific data
            config_data = self._get_builder_config_data(builder_class)
            
            # Use existing from_base_config pattern
            return config_class.from_base_config(base_config, **config_data)
            
        except Exception as e:
            self.logger.debug(f"Failed to create {config_class.__name__}: {e}")
            return None
    
    def _get_base_config(self, builder_class: Type) -> Optional[Any]:
        """
        Get base pipeline config by leveraging existing mock factory system.
        
        This eliminates hard-coding by reusing the existing mock factory's
        base configuration generation capabilities.
        """
        try:
            # Leverage existing mock factory to create realistic base config
            from .sagemaker_step_type_validator import SageMakerStepTypeValidator
            from .mock_factory import StepTypeMockFactory
            
            # Use the builder class to get base config structure
            validator = SageMakerStepTypeValidator(builder_class)
            step_info = validator.get_step_type_info()
            factory = StepTypeMockFactory(step_info, test_mode=True)
            
            # Get mock config and extract base config if available
            mock_config = factory.create_mock_config()
            
            # Try to extract base config from mock config
            if hasattr(mock_config, 'base_config'):
                return mock_config.base_config
            elif hasattr(mock_config, '__dict__'):
                # Extract base config fields from mock config
                base_fields = [
                    'author', 'bucket', 'role', 'region', 'service_name',
                    'pipeline_version', 'model_class', 'current_date',
                    'framework_version', 'py_version', 'source_dir',
                    'project_root_folder', 'pipeline_name', 'pipeline_s3_loc'
                ]
                
                base_config_data = {}
                for field in base_fields:
                    if hasattr(mock_config, field):
                        base_config_data[field] = getattr(mock_config, field)
                
                if base_config_data:
                    # Create base config using extracted data
                    try:
                        from ...core.base.config_base import BasePipelineConfig
                        return BasePipelineConfig(**base_config_data)
                    except Exception as e:
                        self.logger.debug(f"Failed to create BasePipelineConfig: {e}")
                        # Return the mock config itself as fallback
                        return mock_config
            
            # If no base config available, return the mock config itself
            # as it may already be a valid base config
            return mock_config
            
        except Exception as e:
            self.logger.debug(f"Failed to get base config from mock factory: {e}")
            return None
    
    def _get_builder_config_data(self, builder_class: Type) -> Dict[str, Any]:
        """
        Get builder-specific configuration data by leveraging existing mock factory.
        
        This eliminates hard-coding by reusing the existing StepTypeMockFactory's
        intelligent configuration generation capabilities.
        """
        try:
            # Leverage existing mock factory's configuration intelligence
            from .sagemaker_step_type_validator import SageMakerStepTypeValidator
            from .mock_factory import StepTypeMockFactory
            
            # Get step info using existing validator
            validator = SageMakerStepTypeValidator(builder_class)
            step_info = validator.get_step_type_info()
            
            # Use existing mock factory to generate realistic config data
            factory = StepTypeMockFactory(step_info, test_mode=True)
            mock_config = factory.create_mock_config()
            
            # Extract configuration data from mock config
            if hasattr(mock_config, '__dict__'):
                # Convert mock config to dictionary, filtering out methods
                config_data = {
                    key: value for key, value in mock_config.__dict__.items()
                    if not callable(value) and not key.startswith('_')
                }
                return config_data
            elif hasattr(mock_config, 'model_dump'):
                # Handle Pydantic models
                return mock_config.model_dump()
            else:
                # Fallback: return empty dict - let from_base_config handle defaults
                return {}
                
        except Exception as e:
            self.logger.debug(f"Failed to get config data from mock factory: {e}")
            # Return empty dict - let from_base_config handle defaults
            return {}
    
    def _fallback_to_existing_mock_factory(self, builder_class: Type) -> Any:
        """Fallback to existing mock factory system (reuse existing code)."""
        try:
            from .sagemaker_step_type_validator import SageMakerStepTypeValidator
            from .mock_factory import StepTypeMockFactory
            
            validator = SageMakerStepTypeValidator(builder_class)
            step_info = validator.get_step_type_info()
            factory = StepTypeMockFactory(step_info, test_mode=True)
            
            return factory.create_mock_config()
            
        except Exception as e:
            self.logger.debug(f"Mock factory fallback failed: {e}")
            # Final fallback to simple mock
            mock_config = SimpleNamespace()
            mock_config.region = "NA"
            mock_config.pipeline_name = "test-pipeline"
            mock_config.pipeline_s3_loc = "s3://bucket/prefix"
            return mock_config
