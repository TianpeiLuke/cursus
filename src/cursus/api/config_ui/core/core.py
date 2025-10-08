"""
Universal Configuration Engine

Core engine for universal configuration management that supports any configuration
class inheriting from BasePipelineConfig with .from_base_config() method support.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import inspect

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ....core.base.config_base import BasePipelineConfig
    from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    from .import_utils import ensure_cursus_path
    ensure_cursus_path()
    
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
        from ..widgets.widget import UniversalConfigWidget
        return UniversalConfigWidget(form_data)
    
    def create_pipeline_config_widget(self, 
                                    pipeline_dag: Any, 
                                    base_config: BasePipelineConfig,
                                    processing_config: Optional[ProcessingStepConfigBase] = None,
                                    **kwargs) -> 'MultiStepWizard':
        """
        Create DAG-driven pipeline configuration widget with inheritance support.
        
        Uses the same infrastructure as DynamicPipelineTemplate but for discovery
        rather than resolution of existing configurations.
        
        Args:
            pipeline_dag: Pipeline DAG definition
            base_config: Base pipeline configuration
            processing_config: Optional processing configuration
            **kwargs: Additional arguments (e.g., hyperparameters)
            
        Returns:
            MultiStepWizard instance
        """
        logger.info("Creating DAG-driven pipeline configuration widget")
        
        # Use existing StepConfigResolverAdapter (matches production pattern)
        try:
            from cursus.step_catalog.adapters.config_resolver import StepConfigResolverAdapter
            resolver = StepConfigResolverAdapter()
        except ImportError as e:
            logger.warning(f"StepConfigResolverAdapter not available: {e}")
            resolver = None
        
        # Extract DAG nodes (matches DynamicPipelineTemplate._create_config_map pattern)
        dag_nodes = list(pipeline_dag.nodes) if hasattr(pipeline_dag, 'nodes') else []
        logger.info(f"Extracted {len(dag_nodes)} nodes from pipeline DAG")
        
        # Discover required config classes (UI-specific, not resolution)
        required_config_classes = self._discover_required_config_classes(dag_nodes, resolver)
        
        # Create multi-step wizard with inheritance support
        workflow_steps = self._create_workflow_structure(required_config_classes)
        
        logger.info(f"Created {len(workflow_steps)} workflow steps for pipeline wizard")
        
        # Import here to avoid circular imports
        from ..widgets.widget import MultiStepWizard
        return MultiStepWizard(workflow_steps, base_config=base_config, processing_config=processing_config)
    
    def _get_form_fields(self, config_class: Type[BasePipelineConfig]) -> List[Dict[str, Any]]:
        """
        Extract form fields from Pydantic model with 3-tier categorization.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            List of field definitions for form generation
        """
        # Get field categories using the config class's categorize_fields method if available
        field_categories = self._categorize_fields(config_class)
        
        # Use the enhanced method that supports 3-tier categorization
        return self._get_form_fields_with_tiers(config_class, field_categories)
    
    def _categorize_fields(self, config_class: Type[BasePipelineConfig]) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers using the config class's method if available.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            Dictionary with categorized field lists
        """
        # Try to use the config class's categorize_fields method if available
        if hasattr(config_class, 'categorize_fields'):
            try:
                # Create a temporary instance to call the method
                temp_instance = None
                if hasattr(config_class, 'model_fields'):
                    # For Pydantic models, try to create with minimal data
                    try:
                        temp_instance = config_class()
                    except Exception:
                        # If we can't create an instance, fall back to manual categorization
                        pass
                
                if temp_instance and hasattr(temp_instance, 'categorize_fields'):
                    return temp_instance.categorize_fields()
            except Exception as e:
                logger.debug(f"Could not use categorize_fields method for {config_class.__name__}: {e}")
        
        # Fallback to manual categorization
        return self._manual_field_categorization(config_class)
    
    def _manual_field_categorization(self, config_class: Type[BasePipelineConfig]) -> Dict[str, List[str]]:
        """
        Manually categorize fields into three tiers.
        
        Args:
            config_class: Configuration class to analyze
            
        Returns:
            Dictionary with categorized field lists
        """
        categories = {
            "essential": [],  # Tier 1: Required, public
            "system": [],     # Tier 2: Optional (has default), public  
            "derived": []     # Tier 3: Public properties (HIDDEN from UI)
        }
        
        # Handle Pydantic v2 model_fields
        if hasattr(config_class, 'model_fields'):
            model_fields = config_class.model_fields
            
            for field_name, field_info in model_fields.items():
                if field_name.startswith("_"):
                    continue  # Skip private fields
                    
                # Determine if field is required
                is_required = getattr(field_info, 'is_required', lambda: True)()
                if callable(is_required):
                    is_required = is_required()
                
                if is_required:
                    categories["essential"].append(field_name)
                else:
                    categories["system"].append(field_name)
            
            # Find derived properties (hidden from UI)
            for attr_name in dir(config_class):
                if (not attr_name.startswith("_") 
                    and attr_name not in model_fields
                    and isinstance(getattr(config_class, attr_name, None), property)):
                    categories["derived"].append(attr_name)
        
        # Fallback for classes without model_fields
        else:
            try:
                signature = inspect.signature(config_class.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name != 'self' and not param_name.startswith("_"):
                        is_required = param.default == inspect.Parameter.empty
                        if is_required:
                            categories["essential"].append(param_name)
                        else:
                            categories["system"].append(param_name)
            except Exception as e:
                logger.warning(f"Failed to categorize fields for {config_class.__name__}: {e}")
        
        logger.debug(f"Field categorization for {config_class.__name__}: "
                    f"Essential: {len(categories['essential'])}, "
                    f"System: {len(categories['system'])}, "
                    f"Derived: {len(categories['derived'])}")
        
        return categories
    
    def _get_form_fields_with_tiers(self, config_class: Type[BasePipelineConfig], field_categories: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract form fields with 3-tier categorization - only Tier 1 + Tier 2."""
        fields = []
        
        # Only include essential (Tier 1) and system (Tier 2) fields
        # Derived fields (Tier 3) are completely excluded from UI
        fields_to_include = field_categories["essential"] + field_categories["system"]
        
        # Handle Pydantic v2 model_fields
        if hasattr(config_class, 'model_fields'):
            for field_name, field_info in config_class.model_fields.items():
                if field_name in fields_to_include:
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
                    
                    # Get default value - handle PydanticUndefinedType properly
                    default_value = None
                    if hasattr(field_info, 'default'):
                        try:
                            # Check if default is PydanticUndefinedType or similar
                            default = field_info.default
                            if default is not None and str(type(default)) != "<class 'pydantic_core._pydantic_core.PydanticUndefinedType'>":
                                # Only use serializable defaults
                                if isinstance(default, (str, int, float, bool, list, dict)):
                                    default_value = default
                                else:
                                    # Try to convert to string for complex types
                                    try:
                                        default_value = str(default)
                                    except:
                                        default_value = None
                        except Exception as e:
                            logger.debug(f"Could not extract default for field {field_name}: {e}")
                            default_value = None
                    
                    fields.append({
                        "name": field_name,
                        "type": self.field_types.get(field_type, "text"),
                        "required": is_required,  # True for Tier 1, False for Tier 2
                        "tier": "essential" if is_required else "system",
                        "description": description,
                        "default": default_value
                    })
        
        # Fallback for classes without model_fields
        else:
            try:
                signature = inspect.signature(config_class.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name in fields_to_include:
                        field_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                        is_required = param.default == inspect.Parameter.empty
                        default_value = param.default if param.default != inspect.Parameter.empty else None
                        
                        fields.append({
                            "name": param_name,
                            "type": self.field_types.get(field_type, "text"),
                            "required": is_required,
                            "tier": "essential" if is_required else "system",
                            "description": f"Parameter: {param_name}",
                            "default": default_value
                        })
            except Exception as e:
                logger.warning(f"Failed to extract fields from {config_class.__name__}: {e}")
        
        logger.debug(f"Extracted {len(fields)} UI fields from {config_class.__name__} "
                    f"(excluded {len(field_categories['derived'])} derived fields)")
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
    
    def _discover_required_config_classes(self, dag_nodes: List[str], resolver: Optional[Any]) -> List[Dict]:
        """
        Discover what configuration classes are needed for the DAG nodes.
        
        This is different from production resolve_config_map() because:
        - Production: Maps nodes to existing config instances from saved file
        - UI: Discovers what config classes users need to create from scratch
        
        Args:
            dag_nodes: List of DAG node names (extracted same as production)
            resolver: StepConfigResolverAdapter instance
            
        Returns:
            List of required configuration class information
        """
        required_configs = []
        
        for node_name in dag_nodes:
            if resolver and hasattr(resolver, 'catalog'):
                # Use step catalog to determine required config class
                try:
                    step_info = resolver.catalog.get_step_info(node_name)
                    
                    if step_info and hasattr(step_info, 'config_class') and step_info.config_class:
                        config_class = resolver.catalog.get_config_class(step_info.config_class)
                        if config_class:
                            required_configs.append({
                                "node_name": node_name,
                                "config_class_name": step_info.config_class,
                                "config_class": config_class,
                                "inheritance_pattern": self._get_inheritance_pattern(config_class),
                                "is_specialized": self._is_specialized_config(config_class)
                            })
                            continue
                except Exception as e:
                    logger.debug(f"Failed to get step info for {node_name}: {e}")
            
            # Fallback: Try to infer from node name patterns
            inferred_config = self._infer_config_class_from_node_name(node_name, resolver)
            if inferred_config:
                required_configs.append(inferred_config)
        
        logger.info(f"Discovered {len(required_configs)} required config classes from {len(dag_nodes)} DAG nodes")
        return required_configs
    
    def _infer_config_class_from_node_name(self, node_name: str, resolver: Optional[Any]) -> Optional[Dict]:
        """
        Fallback method to infer config class from node name patterns.
        
        Uses similar pattern matching logic as StepConfigResolverAdapter
        but for discovering requirements rather than resolving instances.
        
        Args:
            node_name: DAG node name to analyze
            resolver: StepConfigResolverAdapter instance (optional)
            
        Returns:
            Configuration class information if found, None otherwise
        """
        try:
            # Get all available config classes from catalog or discovery
            available_config_classes = self.discover_config_classes()
            
            # Use resolver's pattern matching to find best match
            for class_name, config_class in available_config_classes.items():
                # Simple heuristic: check if node name contains config type keywords
                config_base = class_name.lower().replace("config", "").replace("step", "")
                if config_base in node_name.lower():
                    return {
                        "node_name": node_name,
                        "config_class_name": class_name,
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class),
                        "inferred": True
                    }
            
            # Additional pattern matching for common node name patterns
            node_lower = node_name.lower()
            if "cradle" in node_lower and "data" in node_lower:
                if "CradleDataLoadConfig" in available_config_classes:
                    config_class = available_config_classes["CradleDataLoadConfig"]
                    return {
                        "node_name": node_name,
                        "config_class_name": "CradleDataLoadConfig",
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class),
                        "inferred": True
                    }
            
            if "xgboost" in node_lower and "training" in node_lower:
                if "XGBoostTrainingConfig" in available_config_classes:
                    config_class = available_config_classes["XGBoostTrainingConfig"]
                    return {
                        "node_name": node_name,
                        "config_class_name": "XGBoostTrainingConfig",
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class),
                        "inferred": True
                    }
            
            if "preprocessing" in node_lower or "preprocess" in node_lower:
                if "TabularPreprocessingConfig" in available_config_classes:
                    config_class = available_config_classes["TabularPreprocessingConfig"]
                    return {
                        "node_name": node_name,
                        "config_class_name": "TabularPreprocessingConfig",
                        "config_class": config_class,
                        "inheritance_pattern": self._get_inheritance_pattern(config_class),
                        "is_specialized": self._is_specialized_config(config_class),
                        "inferred": True
                    }
            
            logger.debug(f"Could not infer config class for node: {node_name}")
            return None
            
        except Exception as e:
            logger.warning(f"Could not infer config class for node {node_name}: {e}")
            return None
    
    def _create_workflow_structure(self, required_configs: List[Dict]) -> List[Dict]:
        """Create logical workflow structure for configuration steps."""
        workflow_steps = []
        
        # Step 1: Always start with Base Configuration
        workflow_steps.append({
            "step_number": 1,
            "title": "Base Configuration",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig",
            "type": "base",
            "required": True
        })
        
        # Step 2: Add Processing Configuration if any configs need it
        processing_based_configs = [
            config for config in required_configs 
            if config["inheritance_pattern"] == "processing_based"
        ]
        
        if processing_based_configs:
            workflow_steps.append({
                "step_number": 2,
                "title": "Processing Configuration",
                "config_class": ProcessingStepConfigBase,
                "config_class_name": "ProcessingStepConfigBase",
                "type": "processing",
                "required": True
            })
        
        # Step 3+: Add specific configurations
        step_number = len(workflow_steps) + 1
        for config in required_configs:
            workflow_steps.append({
                "step_number": step_number,
                "title": config["config_class_name"],
                "config_class": config["config_class"],
                "config_class_name": config["config_class_name"],
                "step_name": config["node_name"],
                "type": "specific",
                "inheritance_pattern": config["inheritance_pattern"],
                "is_specialized": config["is_specialized"],
                "required": True,
                "inferred": config.get("inferred", False)
            })
            step_number += 1
        
        logger.info(f"Created workflow structure with {len(workflow_steps)} steps")
        return workflow_steps
    
    def _get_inheritance_pattern(self, config_class: Type[BasePipelineConfig]) -> str:
        """Determine inheritance pattern for a configuration class."""
        # Check if config inherits from ProcessingStepConfigBase
        for base_class in config_class.__mro__:
            if base_class.__name__ == "ProcessingStepConfigBase":
                return "processing_based"
        
        # Special handling for CradleDataLoadConfig
        if config_class.__name__ == "CradleDataLoadConfig":
            return "base_only_specialized"
        
        # Default: inherits from BasePipelineConfig only
        return "base_only"
    
    def _is_specialized_config(self, config_class: Type[BasePipelineConfig]) -> bool:
        """Check if configuration requires specialized UI."""
        specialized_configs = {
            "CradleDataLoadConfig": True,
            "ModelHyperparameters": True,
            "XGBoostModelHyperparameters": True,
            # Add other specialized configs here as needed
        }
        return specialized_configs.get(config_class.__name__, False)


# Factory function for creating configuration widgets
def create_config_widget(config_class_name: str, 
                        base_config: Optional[BasePipelineConfig] = None,
                        workspace_dirs: Optional[List[Union[str, Path]]] = None,
                        **kwargs) -> 'UniversalConfigWidget':
    """
    Factory function to create configuration widgets for any config type.
    
    Args:
        config_class_name: Name of configuration class
        base_config: Optional base configuration for pre-population
        workspace_dirs: Optional workspace directories for step catalog
        **kwargs: Additional arguments for config creation
        
    Returns:
        UniversalConfigWidget instance
    """
    core = UniversalConfigCore(workspace_dirs=workspace_dirs)
    return core.create_config_widget(config_class_name, base_config, **kwargs)
