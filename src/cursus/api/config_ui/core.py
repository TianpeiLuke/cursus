"""
Universal Configuration Engine

Core engine for universal configuration management that supports any configuration
class inheriting from BasePipelineConfig with .from_base_config() method support.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import inspect

from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class UniversalConfigCore:
    """Core engine for universal configuration management."""
    
    def __init__(self, workspace_dirs: Optional[List[Union[str, Path]]] = None):
        """
        Initialize with existing step catalog infrastructure.
        
        Args:
            workspace_dirs: Optional list of workspace directories for step catalog
        """
        self.workspace_dirs = [Path(d) if isinstance(d, str) else d for d in (workspace_dirs or [])]
        self._step_catalog = None
        self._config_classes_cache = None
        
        # Simple field type mapping for automatic form generation
        self.field_types = {
            str: "text",
            int: "number", 
            float: "number",
            bool: "checkbox",
            list: "list",
            dict: "keyvalue"
        }
        
        logger.info(f"UniversalConfigCore initialized with workspace_dirs: {self.workspace_dirs}")
    
    @property
    def step_catalog(self):
        """Lazy-loaded step catalog with error handling."""
        if self._step_catalog is None:
            try:
                from cursus.step_catalog.step_catalog import StepCatalog
                self._step_catalog = StepCatalog(workspace_dirs=self.workspace_dirs)
                logger.info("Step catalog initialized successfully")
            except ImportError as e:
                logger.warning(f"Step catalog not available: {e}")
                self._step_catalog = None
        return self._step_catalog
    
    def discover_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
        """
        Discover available configuration classes using step catalog.
        
        Returns:
            Dictionary mapping config class names to config classes
        """
        if self._config_classes_cache is not None:
            return self._config_classes_cache
            
        config_classes = {}
        
        # Try step catalog first
        if self.step_catalog:
            try:
                config_classes = self.step_catalog.discover_config_classes()
                logger.info(f"Discovered {len(config_classes)} config classes via step catalog")
            except Exception as e:
                logger.warning(f"Step catalog discovery failed: {e}")
        
        # Always include base config classes alongside step catalog discoveries
        base_config_classes = {
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase
        }
        
        # Merge base classes with discovered classes
        config_classes.update(base_config_classes)
        
        # If step catalog failed, use only base classes
        if len(config_classes) == len(base_config_classes):
            logger.info(f"Using base config classes only: {list(config_classes.keys())}")
        else:
            logger.info(f"Using {len(config_classes)} config classes: {len(config_classes) - len(base_config_classes)} from step catalog + {len(base_config_classes)} base classes")
        
        self._config_classes_cache = config_classes
        return config_classes
    
    def create_config_widget(self, 
                           config_class_name: str, 
                           base_config: Optional[BasePipelineConfig] = None, 
                           **kwargs) -> 'UniversalConfigWidget':
        """
        Create configuration widget for any config type.
        
        Args:
            config_class_name: Name of the configuration class
            base_config: Optional base configuration for pre-population
            **kwargs: Additional arguments for config creation
            
        Returns:
            UniversalConfigWidget instance
            
        Raises:
            ValueError: If configuration class is not found
        """
        # Discover config class
        config_classes = self.discover_config_classes()
        config_class = config_classes.get(config_class_name)
        
        if not config_class:
            available_classes = list(config_classes.keys())
            raise ValueError(
                f"Configuration class '{config_class_name}' not found. "
                f"Available classes: {available_classes}"
            )
        
        logger.info(f"Creating widget for config class: {config_class_name}")
        
        # Create pre-populated instance using .from_base_config()
        pre_populated = None
        pre_populated_values = {}
        
        if base_config and hasattr(config_class, 'from_base_config'):
            try:
                pre_populated = config_class.from_base_config(base_config, **kwargs)
                pre_populated_values = pre_populated.model_dump() if hasattr(pre_populated, 'model_dump') else {}
                logger.info(f"Pre-populated config using from_base_config method")
            except Exception as e:
                logger.warning(f"Failed to pre-populate config: {e}, will create form with empty fields")
                # For configs that can't be pre-populated, we'll create a form with empty fields
                # This allows users to fill in required fields through the UI
                pre_populated_values = {}
        
        # If no base_config provided, try to create empty instance for field extraction only
        if pre_populated is None:
            try:
                # Try to create with minimal required fields for field extraction
                pre_populated = None  # We'll extract fields from class definition instead
                pre_populated_values = {}
            except Exception as e:
                logger.debug(f"Cannot create empty instance of {config_class_name}: {e}")
                pre_populated = None
                pre_populated_values = {}
        
        # Generate form data
        form_data = {
            "config_class": config_class,
            "config_class_name": config_class_name,
            "fields": self._get_form_fields(config_class),
            "values": pre_populated.model_dump() if hasattr(pre_populated, 'model_dump') else {},
            "inheritance_chain": self._get_inheritance_chain(config_class),
            "pre_populated_instance": pre_populated
        }
        
        # Import here to avoid circular imports
        from .widget import UniversalConfigWidget
        return UniversalConfigWidget(form_data)
    
    def create_pipeline_config_widget(self, 
                                    dag: Any, 
                                    base_config: BasePipelineConfig,
                                    processing_config: Optional[ProcessingStepConfigBase] = None,
                                    **kwargs) -> 'MultiStepWizard':
        """
        Create DAG-driven pipeline configuration widget.
        
        Args:
            dag: Pipeline DAG definition
            base_config: Base pipeline configuration
            processing_config: Optional processing configuration
            **kwargs: Additional arguments (e.g., hyperparameters)
            
        Returns:
            MultiStepWizard instance
        """
        logger.info("Creating DAG-driven pipeline configuration widget")
        
        # Use existing StepConfigResolverAdapter for DAG resolution
        config_map = {}
        try:
            from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter
            resolver = StepConfigResolverAdapter()
            config_map = resolver.resolve_config_map(dag.nodes, {})
            logger.info(f"Resolved {len(config_map)} config requirements from DAG")
        except Exception as e:
            logger.warning(f"DAG resolution failed: {e}, using basic config structure")
        
        # Create multi-step wizard structure
        steps = []
        
        # Step 1: Base Pipeline Configuration (always first)
        steps.append({
            "title": "Base Pipeline Configuration",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig",
            "pre_populated": base_config,
            "required": True
        })
        
        # Step 2: Base Processing Configuration (always second, required for processing steps)
        if processing_config:
            steps.append({
                "title": "Processing Configuration", 
                "config_class": ProcessingStepConfigBase,
                "config_class_name": "ProcessingStepConfigBase",
                "pre_populated": processing_config,
                "required": True
            })
        else:
            steps.append({
                "title": "Processing Configuration",
                "config_class": ProcessingStepConfigBase, 
                "config_class_name": "ProcessingStepConfigBase",
                "base_config": base_config,
                "required": True
            })
        
        # Step 3+: Specialized configs from DAG
        for node_name, config_instance in config_map.items():
            if config_instance:
                config_class = type(config_instance)
                config_class_name = config_class.__name__
                
                # Try to pre-populate using from_base_config
                pre_populated_data = None
                if hasattr(config_class, 'from_base_config'):
                    try:
                        pre_populated = config_class.from_base_config(base_config, **kwargs)
                        pre_populated_data = pre_populated.model_dump() if hasattr(pre_populated, 'model_dump') else {}
                    except Exception as e:
                        logger.warning(f"Failed to pre-populate {config_class_name}: {e}")
                
                steps.append({
                    "title": config_class_name,
                    "config_class": config_class,
                    "config_class_name": config_class_name,
                    "pre_populated_data": pre_populated_data,
                    "base_config": base_config,
                    "node_name": node_name,
                    "required": True
                })
        
        logger.info(f"Created {len(steps)} steps for pipeline wizard")
        
        # Import here to avoid circular imports
        from .widget import MultiStepWizard
        return MultiStepWizard(steps, base_config=base_config, processing_config=processing_config)
    
    def _get_form_fields(self, config_class: Type[BasePipelineConfig]) -> List[Dict[str, Any]]:
        """
        Extract form fields from Pydantic model.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            List of field definitions for form generation
        """
        fields = []
        
        # Handle Pydantic v2 model_fields
        if hasattr(config_class, 'model_fields'):
            for field_name, field_info in config_class.model_fields.items():
                if not field_name.startswith("_"):
                    # Get field type
                    field_type = getattr(field_info, 'annotation', str)
                    if hasattr(field_type, '__origin__'):
                        # Handle generic types like Optional[str], List[str], etc.
                        field_type = field_type.__origin__
                    
                    # Determine if field is required
                    is_required = getattr(field_info, 'is_required', lambda: True)()
                    if callable(is_required):
                        is_required = is_required()
                    
                    # Get description
                    description = ""
                    if hasattr(field_info, 'description') and field_info.description:
                        description = field_info.description
                    
                    fields.append({
                        "name": field_name,
                        "type": self.field_types.get(field_type, "text"),
                        "required": is_required,
                        "description": description,
                        "field_info": field_info
                    })
        
        # Fallback for classes without model_fields
        else:
            # Use inspection to get fields
            try:
                signature = inspect.signature(config_class.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name != 'self':
                        field_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                        is_required = param.default == inspect.Parameter.empty
                        
                        fields.append({
                            "name": param_name,
                            "type": self.field_types.get(field_type, "text"),
                            "required": is_required,
                            "description": f"Parameter: {param_name}",
                            "field_info": param
                        })
            except Exception as e:
                logger.warning(f"Failed to extract fields from {config_class.__name__}: {e}")
        
        logger.debug(f"Extracted {len(fields)} fields from {config_class.__name__}")
        return fields
    
    def _get_inheritance_chain(self, config_class: Type[BasePipelineConfig]) -> List[str]:
        """
        Get inheritance chain for configuration class.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            List of class names in inheritance chain
        """
        chain = []
        for cls in config_class.__mro__:
            if issubclass(cls, BasePipelineConfig) and cls != BasePipelineConfig:
                chain.append(cls.__name__)
        
        logger.debug(f"Inheritance chain for {config_class.__name__}: {chain}")
        return chain
