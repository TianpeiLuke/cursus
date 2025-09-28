"""
Base class for pipeline templates.

This module provides a base class for all pipeline templates,
ensuring consistent structure, proper component lifecycle management,
and best practices across different pipeline templates.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Set, Tuple, Union
from pathlib import Path
import logging
import json

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

from ..compiler.name_generator import generate_pipeline_name

from ..base import BasePipelineConfig, StepBuilderBase
from ..deps.registry_manager import RegistryManager
from ..deps.dependency_resolver import UnifiedDependencyResolver
from ..deps.semantic_matcher import SemanticMatcher
from ..deps.factory import (
    create_pipeline_components,
    dependency_resolution_context,
    get_thread_components,
)

from .pipeline_assembler import PipelineAssembler
from ...api.dag.base_dag import PipelineDAG

logger = logging.getLogger(__name__)


class PipelineTemplateBase(ABC):
    """
    Base class for all pipeline templates.

    This class provides a consistent structure and common functionality for
    all pipeline templates, enforcing best practices and ensuring proper
    component lifecycle management.

    The template follows these steps to build a pipeline:
    1. Load configurations from file
    2. Initialize component dependencies (registry_manager, dependency_resolver)
    3. Create the DAG, config_map, and step_builder_map
    4. Use PipelineAssembler to assemble the pipeline

    This provides a standardized approach for creating pipeline templates,
    reducing code duplication and enforcing best practices.
    """

    # This should be overridden by subclasses to specify the config classes
    # that are expected in the configuration file
    CONFIG_CLASSES: Dict[str, Type[BasePipelineConfig]] = {}

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
        pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
        step_catalog: Optional['StepCatalog'] = None,
    ):
        """
        Initialize base template.

        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session
            role: IAM role
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
            pipeline_parameters: Pipeline parameters from DAGCompiler (optional)
            step_catalog: Optional StepCatalog for config-to-builder resolution
        """
        self.config_path = config_path
        self.session = sagemaker_session
        self.role = role

        # Store pipeline parameters for template
        self._stored_pipeline_parameters: Optional[List[Union[str, ParameterString]]] = pipeline_parameters
        
        # Store step catalog
        self._step_catalog = step_catalog

        # Load configurations
        logger.info(f"Loading configs from: {config_path}")
        self.configs = self._load_configs(config_path)
        self.base_config = self._get_base_config()

        # Store loaded configuration data including metadata
        try:
            with open(config_path, "r") as f:
                self.loaded_config_data = json.load(f)
            logger.info(f"Loaded raw configuration data from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load raw configuration data: {e}")
            self.loaded_config_data = None

        # Store dependency components
        self._registry_manager = registry_manager
        self._dependency_resolver = dependency_resolver

        # Initialize components if not provided
        if not self._registry_manager or not self._dependency_resolver:
            self._initialize_components()

        # Validate configuration
        self._validate_configuration()

        # Initialize storage for pipeline metadata
        self.pipeline_metadata: Dict[str, Any] = {}

        logger.info(f"Initialized template for: {self._get_pipeline_name()}")

    def _load_configs(self, config_path: str) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary of configurations
        """
        if not self.CONFIG_CLASSES:
            raise ValueError("CONFIG_CLASSES must be defined by subclass")

        # Import here to avoid circular imports
        from ...steps.configs.utils import build_complete_config_classes, load_configs

        # Build a complete config classes dictionary with hyperparameter classes
        complete_classes = build_complete_config_classes()

        # Merge with template-defined CONFIG_CLASSES, giving preference to template classes
        # This ensures that template-specific classes override any defaults
        for class_name, class_type in self.CONFIG_CLASSES.items():
            complete_classes[class_name] = class_type

        # Type cast is safe since all config classes should inherit from BasePipelineConfig
        return load_configs(config_path, complete_classes)  # type: ignore[return-value]

    def _get_base_config(self) -> BasePipelineConfig:
        """
        Get base configuration.

        Returns:
            Base configuration

        Raises:
            ValueError: If base configuration not found
        """
        base_config = self.configs.get("Base")
        if not base_config:
            raise ValueError("Base configuration not found in config file")
        return base_config

    def _initialize_components(self) -> None:
        """
        Initialize dependency resolution components.

        This method creates registry manager and dependency resolver if they
        were not provided during initialization.
        """
        context_name = getattr(self.base_config, "pipeline_name", None)
        components = create_pipeline_components(context_name)

        if not self._registry_manager:
            self._registry_manager = components["registry_manager"]
            logger.info(
                f"Created registry manager for context: {context_name or 'default'}"
            )

        if not self._dependency_resolver:
            self._dependency_resolver = components["resolver"]
            logger.info(
                f"Created dependency resolver for context: {context_name or 'default'}"
            )

    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Perform lightweight validation of configuration structure and essential parameters.

        This method focuses on validating:
        1. Presence/absence of required configurations
        2. Basic parameter validation (types, ranges, etc.)
        3. Non-dependency related concerns

        NOTE: Dependency resolution validation is handled by the dependency resolver
        during pipeline building. This method should NOT duplicate dependency validation
        logic already provided by the resolver.

        Example implementation for a template that requires preprocessing configs:
        ```python
        def _validate_configuration(self) -> None:
            # Find preprocessing configs
            tp_configs = [cfg for name, cfg in self.configs.items()
                         if isinstance(cfg, PreprocessingConfig)]

            if len(tp_configs) < 2:
                raise ValueError("Expected at least two PreprocessingConfig instances")

            # Check for presence of training and calibration configs
            training_config = next((cfg for cfg in tp_configs
                                  if getattr(cfg, 'job_type', None) == 'training'), None)
            if not training_config:
                raise ValueError("No PreprocessingConfig found with job_type='training'")
        ```

        Raises:
            ValueError: If configuration structure is invalid
        """
        pass

    @abstractmethod
    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Create the DAG structure for the pipeline.

        This method should be implemented by subclasses to define the
        pipeline's DAG structure.

        Returns:
            PipelineDAG instance
        """
        pass

    @abstractmethod
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Create a mapping from step names to config instances.

        This method should be implemented by subclasses to map step names
        to their respective configurations.

        Returns:
            Dictionary mapping step names to configurations
        """
        pass

    @abstractmethod
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Create a mapping from step types to builder classes.

        This method should be implemented by subclasses to map step types
        to their builder classes.

        Returns:
            Dictionary mapping step types to builder classes
        """
        pass

    def set_pipeline_parameters(self, parameters: Optional[List[ParameterString]] = None) -> None:
        """
        Set pipeline parameters for this template.
        
        This method allows DAGCompiler to inject custom parameters that will be used
        instead of the default parameters defined in subclasses.
        
        Args:
            parameters: List of pipeline parameters to use
        """
        self._stored_pipeline_parameters = parameters
        logger.info(f"Set {len(parameters) if parameters else 0} custom pipeline parameters")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """
        Get pipeline parameters.
        
        Returns stored parameters if available, otherwise delegates to subclass implementation.
        This method is called by generate_pipeline() to get parameters for PipelineAssembler.
        
        Returns:
            List of pipeline parameters
        """
        if self._stored_pipeline_parameters is not None:
            logger.info("Using stored custom pipeline parameters")
            return self._stored_pipeline_parameters
        
        # Fallback to subclass implementation (existing behavior)
        logger.info("No stored parameters, using default implementation")
        return []  # Default empty list, subclasses can override

    def generate_pipeline(self) -> Pipeline:
        """
        Generate the SageMaker Pipeline.

        This method coordinates the pipeline generation process:
        1. Create the DAG, config_map, and step_builder_map
        2. Create the PipelineAssembler
        3. Generate the pipeline
        4. Store pipeline metadata

        Returns:
            SageMaker Pipeline
        """
        pipeline_name = self._get_pipeline_name()
        logger.info(f"Generating pipeline: {pipeline_name}")

        # Create the DAG, config_map, and step builder map
        dag = self._create_pipeline_dag()
        config_map = self._create_config_map()
        step_builder_map = self._create_step_builder_map()

        # Create the assembler with StepCatalog integration
        # Use provided step_catalog or create a new one
        if self._step_catalog is not None:
            step_catalog = self._step_catalog
            logger.info("Using provided StepCatalog instance")
        else:
            from ...step_catalog import StepCatalog
            step_catalog = StepCatalog()
            logger.info("Created new StepCatalog instance")
        
        template = PipelineAssembler(
            dag=dag,
            config_map=config_map,
            step_catalog=step_catalog,
            sagemaker_session=self.session,
            role=self.role,
            pipeline_parameters=self._get_pipeline_parameters(),
            registry_manager=self._registry_manager,
            dependency_resolver=self._dependency_resolver,
        )

        # Generate the pipeline
        pipeline = template.generate_pipeline(pipeline_name)

        # Store pipeline metadata
        self._store_pipeline_metadata(template)

        return pipeline

    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name using the rule-based generator.

        Returns:
            Pipeline name
        """
        # Check if explicit override is provided in the base config
        explicit_name = getattr(self.base_config, "explicit_pipeline_name", None)
        if explicit_name:
            from typing import cast
            return cast(str, explicit_name)

        # Get pipeline_name from base_config, with fallback
        pipeline_name = getattr(self.base_config, "pipeline_name", "mods")

        # Get pipeline_version from base_config, with fallback
        pipeline_version = getattr(self.base_config, "pipeline_version", "1.0")

        # Use the rule-based generator
        return generate_pipeline_name(pipeline_name, pipeline_version)

    def _store_pipeline_metadata(self, template: PipelineAssembler) -> None:
        """
        Store pipeline metadata from template.

        This method can be overridden by subclasses to store pipeline-specific
        metadata (excluding execution document data which is now handled separately).

        Args:
            template: PipelineAssembler instance
        """
        # Note: Cradle data loading requests storage removed as part of Phase 2 cleanup
        # Execution document metadata is now handled by the standalone execution document generator
        # (ExecutionDocumentGenerator in cursus.mods.exe_doc.generator)

        # Store general pipeline metadata (non-execution document related)
        if hasattr(template, "step_instances"):
            self.pipeline_metadata["step_instances"] = template.step_instances

    @classmethod
    def create_with_components(
        cls, config_path: str, context_name: Optional[str] = None, **kwargs: Any
    ) -> "PipelineTemplateBase":
        """
        Create template with managed dependency components.

        This factory method creates a template with properly configured
        dependency resolution components from the factory module.

        Args:
            config_path: Path to configuration file
            context_name: Optional context name for registry isolation
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Template instance with managed components
        """
        components = create_pipeline_components(context_name)
        return cls(
            config_path=config_path,
            registry_manager=components["registry_manager"],
            dependency_resolver=components["resolver"],
            **kwargs,
        )

    @classmethod
    def build_with_context(cls, config_path: str, **kwargs: Any) -> Pipeline:
        """
        Build pipeline with scoped dependency resolution context.

        This method creates a template with a dependency resolution context
        that ensures proper cleanup of resources after pipeline generation.

        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Generated pipeline
        """
        with dependency_resolution_context(clear_on_exit=True) as components:
            template = cls(
                config_path=config_path,
                registry_manager=components["registry_manager"],
                dependency_resolver=components["resolver"],
                **kwargs,
            )
            return template.generate_pipeline()

    @classmethod
    def build_in_thread(cls, config_path: str, **kwargs: Any) -> Pipeline:
        """
        Build pipeline using thread-local component instances.

        This method creates a template with thread-local component instances,
        ensuring thread safety in multi-threaded environments.

        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Generated pipeline
        """
        components = get_thread_components()
        template = cls(
            config_path=config_path,
            registry_manager=components["registry_manager"],
            dependency_resolver=components["resolver"],
            **kwargs,
        )
        return template.generate_pipeline()

    # Note: fill_execution_document() method removed as part of Phase 2 cleanup
    # Execution document generation is now handled by the standalone execution document generator
    # (ExecutionDocumentGenerator in cursus.mods.exe_doc.generator)
