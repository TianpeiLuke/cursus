"""
Universal Configuration Engine

Core engine for universal configuration management that supports any configuration
class inheriting from BasePipelineConfig with .from_base_config() method support.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import inspect

from ....core.base.config_base import BasePipelineConfig
from ....steps.configs.config_processing_step_base import ProcessingStepConfigBase

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
                from ....step_catalog.step_catalog import StepCatalog
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
        return UniversalConfigWidget(form_data, config_core=self)
    
    def create_pipeline_config_widget(self, 
                                    pipeline_dag: Any, 
                                    base_config: BasePipelineConfig,
                                    processing_config: Optional[ProcessingStepConfigBase] = None,
                                    **kwargs) -> 'MultiStepWizard':
        """
        Simplified widget creation using factory directly.
        
        Args:
            pipeline_dag: Pipeline DAG definition
            base_config: Base pipeline configuration
            processing_config: Optional processing configuration
            **kwargs: Additional arguments
            
        Returns:
            MultiStepWizard instance with factory integration
        """
        logger.info("Creating DAG-driven pipeline configuration widget using factory system")
        
        # Use factory directly - no wrapper logic
        from ...factory import DAGConfigFactory
        
        try:
            factory = DAGConfigFactory(pipeline_dag)
            factory.set_base_config(**base_config.model_dump())
            
            if processing_config:
                factory.set_base_processing_config(**processing_config.model_dump())
            
            logger.info("Factory-driven widget creation successful")
            
            # Import here to avoid circular imports
            from ..widgets.widget import MultiStepWizard
            return MultiStepWizard(factory=factory, base_config=base_config, processing_config=processing_config, core=self, **kwargs)
            
        except Exception as e:
            logger.error(f"Factory-driven widget creation failed: {e}")
            # Fallback to manual workflow creation for compatibility
            return self._create_fallback_widget(pipeline_dag, base_config, processing_config, **kwargs)
    
    def _create_fallback_widget(self, pipeline_dag: Any, base_config: BasePipelineConfig, processing_config: Optional[ProcessingStepConfigBase] = None, **kwargs) -> 'MultiStepWizard':
        """Fallback widget creation for compatibility."""
        logger.info("Using fallback widget creation")
        
        # Extract DAG nodes for fallback
        dag_nodes = list(pipeline_dag.nodes) if hasattr(pipeline_dag, 'nodes') else []
        
        # Use existing discovery logic as fallback
        try:
            from ....step_catalog.adapters.config_resolver import StepConfigResolverAdapter
            resolver = StepConfigResolverAdapter()
        except ImportError:
            resolver = None
        
        required_config_classes = self._discover_required_config_classes(pipeline_dag, resolver)
        workflow_steps = self._create_workflow_structure(required_config_classes)
        
        from ..widgets.widget import MultiStepWizard
        return MultiStepWizard(workflow_steps, base_config=base_config, processing_config=processing_config, core=self, **kwargs)
    
    def _get_form_fields(self, config_class: Type[BasePipelineConfig], _recursion_guard: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Extract form fields using factory field extractor.
        
        Args:
            config_class: Configuration class to analyze
            _recursion_guard: Internal parameter to prevent infinite recursion
            
        Returns:
            List of field definitions for form generation
        """
        # Import factory field extractor directly
        from ...factory import extract_field_requirements, categorize_field_requirements
        
        config_class_name = config_class.__name__
        logger.info(f"🔍 _get_form_fields called for {config_class_name} - using factory field extractor")
        
        # Special handling for CradleDataLoadingConfig - use discovery-based sub-config organization
        if config_class_name == "CradleDataLoadingConfig":
            logger.info(f"✅ {config_class_name} matches CradleDataLoadingConfig - using discovery-based sub-config organization")
            try:
                from .field_definitions import get_cradle_fields_by_sub_config
                field_blocks = get_cradle_fields_by_sub_config(config_core=self, _recursion_guard=_recursion_guard)
                
                # Convert sub-config blocks to flat field list for backward compatibility
                fields = []
                for block_name, block_fields in field_blocks.items():
                    fields.extend(block_fields)
                
                logger.info(f"✅ Successfully imported discovery-based field definitions for {config_class_name}: {len(fields)} fields from {len(field_blocks)} blocks")
                return fields
            except ImportError as e:
                logger.error(f"❌ Could not import field definitions for {config_class_name}: {e}, falling back to factory extraction")
        
        # Use factory field extraction for all other classes
        try:
            requirements = extract_field_requirements(config_class)
            logger.info(f"📊 Factory field extraction returned {len(requirements)} fields for {config_class_name}")
            return requirements
        except Exception as e:
            logger.error(f"❌ Factory field extraction failed for {config_class_name}: {e}")
            return []
    
    def get_inheritance_aware_form_fields(self,
                                        config_class_name: str,
                                        inheritance_analysis: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate form fields with Smart Default Value Inheritance awareness.

        CONSOLIDATED: This method now uses the centralized inheritance-aware field generator
        from the core config_fields module instead of manual inheritance logic.

        Args:
            config_class_name: Name of the configuration class
            inheritance_analysis: Optional inheritance analysis from StepCatalog

        Returns:
            List of enhanced field definitions with inheritance information
        """
        logger.info(f"🔄 CONSOLIDATED: Using centralized inheritance-aware field generator for {config_class_name}")
        
        try:
            # Import from core config_fields module
            from ....core.config_fields import get_inheritance_aware_form_fields
            
            # Use centralized inheritance-aware field generation
            enhanced_fields = get_inheritance_aware_form_fields(
                config_class_name=config_class_name,
                inheritance_analysis=inheritance_analysis,
                workspace_dirs=self.workspace_dirs,
                project_id=getattr(self, 'project_id', None)
            )
            
            logger.info(f"✅ Centralized field generation returned {len(enhanced_fields)} enhanced fields")
            return enhanced_fields
            
        except ImportError as e:
            logger.error(f"Could not import centralized inheritance-aware field generator: {e}")
            # Fallback to basic field extraction
            return self._get_form_fields_fallback(config_class_name)
    
    def _get_form_fields_fallback(self, config_class_name: str) -> List[Dict[str, Any]]:
        """
        Fallback form field extraction when centralized generator is not available.
        
        Args:
            config_class_name: Name of the configuration class
            
        Returns:
            Basic field definitions
        """
        logger.info(f"Using fallback form field extraction for {config_class_name}")
        
        # Get the config class
        config_classes = self.discover_config_classes()
        config_class = config_classes.get(config_class_name)

        if not config_class:
            logger.warning(f"Config class {config_class_name} not found")
            return []

        # Use existing _get_form_fields method as fallback
        return self._get_form_fields(config_class)
    
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
    
    def _discover_required_config_classes(self, pipeline_dag: Any, resolver: Optional[Any]) -> List[Dict]:
        """
        CONSOLIDATED: Use factory's get_config_class_map() directly instead of manual discovery.
        
        This method now uses DAGConfigFactory.get_config_class_map() which provides
        the exact same functionality with robust DAG node to configuration class mapping.
        
        Args:
            pipeline_dag: Pipeline DAG object (used directly by factory)
            resolver: StepConfigResolverAdapter instance (not used - factory handles resolution)
            
        Returns:
            List of required configuration class information
        """
        logger.info("🔄 CONSOLIDATED: Using factory get_config_class_map() instead of manual discovery")
        
        # Use factory directly with the actual pipeline_dag - this replaces 200+ lines of manual logic
        from ...factory import DAGConfigFactory
        
        try:
            factory = DAGConfigFactory(pipeline_dag)
            config_class_map = factory.get_config_class_map()
            
            # Convert factory mapping to expected format
            required_configs = []
            for node_name, config_class in config_class_map.items():
                required_configs.append({
                    "node_name": node_name,
                    "config_class_name": config_class.__name__,
                    "config_class": config_class,
                    "inheritance_pattern": self._get_inheritance_pattern(config_class),
                    "is_specialized": self._is_specialized_config(config_class),
                    "factory_resolved": True
                })
            
            logger.info(f"✅ Factory-based discovery: {len(required_configs)} required config classes from pipeline DAG")
            return required_configs
            
        except Exception as e:
            logger.warning(f"Factory discovery failed: {e}, falling back to manual discovery")
            # Extract DAG nodes for fallback
            dag_nodes = list(pipeline_dag.nodes) if hasattr(pipeline_dag, 'nodes') else []
            return self._manual_discovery_fallback(dag_nodes, resolver)
    
    def _manual_discovery_fallback(self, dag_nodes: List[str], resolver: Optional[Any]) -> List[Dict]:
        """
        Manual discovery fallback for compatibility when factory fails.
        
        This preserves the original manual discovery logic as a fallback mechanism
        to ensure backward compatibility when factory-based discovery fails.
        
        Args:
            dag_nodes: List of DAG node names
            resolver: StepConfigResolverAdapter instance (optional)
            
        Returns:
            List of required configuration class information
        """
        logger.info("🔄 Using manual discovery fallback")
        
        required_configs = []
        
        for node_name in dag_nodes:
            # Try manual inference (preserve existing fallback logic)
            inferred_config = self._infer_config_class_from_node_name(node_name, resolver)
            if inferred_config:
                inferred_config["factory_resolved"] = False
                inferred_config["manual_fallback"] = True
                required_configs.append(inferred_config)
        
        logger.info(f"Manual fallback discovery: {len(required_configs)} required config classes from {len(dag_nodes)} DAG nodes")
        return required_configs
    
    def _infer_config_class_from_node_name(self, node_name: str, resolver: Optional[Any]) -> Optional[Dict]:
        """
        Infer config class from node name using registry helper functions.
        
        Args:
            node_name: DAG node name to analyze
            resolver: StepConfigResolverAdapter instance (optional)
            
        Returns:
            Configuration class information if found, None otherwise
        """
        try:
            from ....registry.step_names import get_canonical_name_from_file_name, get_config_class_name
            
            # Get canonical name (handles job type suffixes automatically)
            canonical_name = get_canonical_name_from_file_name(node_name)
            config_class_name = get_config_class_name(canonical_name)
            
            # Get the actual config class
            available_config_classes = self.discover_config_classes()
            config_class = available_config_classes.get(config_class_name)
            
            if config_class:
                return {
                    "node_name": node_name,
                    "config_class_name": config_class_name,
                    "config_class": config_class,
                    "inheritance_pattern": self._get_inheritance_pattern(config_class),
                    "is_specialized": self._is_specialized_config(config_class),
                    "inferred": True
                }
                
        except (ImportError, ValueError) as e:
            logger.debug(f"Registry lookup failed for {node_name}: {e}")
        
        # Enhanced fallback: Try pattern matching for common job type suffixes
        return self._fallback_config_inference(node_name)
    
    def _fallback_config_inference(self, node_name: str) -> Optional[Dict]:
        """
        Fallback method to infer config class using pattern matching.
        
        This ensures that nodes with job type suffixes (like _training, _calibration)
        still get mapped to appropriate config classes for separate user configuration.
        
        Args:
            node_name: DAG node name to analyze
            
        Returns:
            Configuration class information if found, None otherwise
        """
        # Common patterns for job type suffixes
        job_type_patterns = [
            ("_training", ""),
            ("_calibration", ""),
            ("_evaluation", ""),
            ("_inference", ""),
            ("_validation", ""),
        ]
        
        available_config_classes = self.discover_config_classes()
        
        # Try removing job type suffixes and matching to known config classes
        for suffix, replacement in job_type_patterns:
            if node_name.endswith(suffix):
                base_name = node_name[:-len(suffix)] + replacement
                
                # Try direct mapping to config class names
                potential_config_names = [
                    f"{base_name}Config",
                    f"{base_name}StepConfig", 
                    base_name,
                ]
                
                
                for config_name in potential_config_names:
                    if config_name in available_config_classes:
                        config_class = available_config_classes[config_name]
                        logger.info(f"Fallback mapping: {node_name} → {config_name}")
                        return {
                            "node_name": node_name,
                            "config_class_name": config_name,
                            "config_class": config_class,
                            "inheritance_pattern": self._get_inheritance_pattern(config_class),
                            "is_specialized": self._is_specialized_config(config_class),
                            "inferred": True,
                            "fallback_used": True
                        }
        
        # Try exact name matching (for nodes without job type suffixes)
        potential_config_names = [
            f"{node_name}Config",
            f"{node_name}StepConfig",
            node_name,
        ]
        
        for config_name in potential_config_names:
            if config_name in available_config_classes:
                config_class = available_config_classes[config_name]
                logger.info(f"Direct mapping: {node_name} → {config_name}")
                return {
                    "node_name": node_name,
                    "config_class_name": config_name,
                    "config_class": config_class,
                    "inheritance_pattern": self._get_inheritance_pattern(config_class),
                    "is_specialized": self._is_specialized_config(config_class),
                    "inferred": True,
                    "direct_match": True
                }
        
        logger.warning(f"No config class found for node: {node_name}")
        return None
    
    def _create_workflow_structure(self, required_configs: List[Dict]) -> List[Dict]:
        """
        Create logical workflow structure for configuration steps.
        
        Each DAG node gets its own configuration step, even if they share
        the same config class type (e.g., CradleDataLoading_training and 
        CradleDataLoading_calibration both get separate configuration pages).
        """
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
        
        # Step 3+: Add specific configurations - ONE STEP PER DAG NODE
        step_number = len(workflow_steps) + 1
        for config in required_configs:
            # Create a unique title that includes the DAG node name
            # This makes it clear which instance of the config this is for
            node_name = config["node_name"]
            config_class_name = config["config_class_name"]
            
            # Create descriptive title that shows both the config type and the specific DAG node
            if node_name != config_class_name:
                title = f"{config_class_name} ({node_name})"
            else:
                title = config_class_name
            
            workflow_steps.append({
                "step_number": step_number,
                "title": title,
                "config_class": config["config_class"],
                "config_class_name": config_class_name,
                "step_name": node_name,  # This is the DAG node name
                "type": "specific",
                "inheritance_pattern": config["inheritance_pattern"],
                "is_specialized": config["is_specialized"],
                "required": True,
                "inferred": config.get("inferred", False),
                "fallback_used": config.get("fallback_used", False),
                "direct_match": config.get("direct_match", False)
            })
            step_number += 1
        
        logger.info(f"Created workflow structure with {len(workflow_steps)} steps")
        logger.info(f"Specific config steps: {len(required_configs)} (one per DAG node)")
        return workflow_steps
    
    def _get_inheritance_pattern(self, config_class: Type[BasePipelineConfig]) -> str:
        """Determine inheritance pattern for a configuration class."""
        # Check if config inherits from ProcessingStepConfigBase
        for base_class in config_class.__mro__:
            if base_class.__name__ == "ProcessingStepConfigBase":
                return "processing_based"
        
        # Special handling for CradleDataLoadingConfig
        if config_class.__name__ == "CradleDataLoadingConfig":
            return "base_only_specialized"
        
        # Default: inherits from BasePipelineConfig only
        return "base_only"
    
    def _is_specialized_config(self, config_class: Type[BasePipelineConfig]) -> bool:
        """Check if configuration requires specialized UI."""
        specialized_configs = {
            "CradleDataLoadingConfig": True,
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
