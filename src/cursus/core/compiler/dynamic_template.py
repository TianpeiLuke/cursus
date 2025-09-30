"""
Dynamic Pipeline Template for the Pipeline API.

This module provides a dynamic implementation of PipelineTemplateBase that can work
with any PipelineDAG structure without requiring custom template classes.
"""

from __future__ import annotations

from typing import Dict, Type, Any, Optional, List, TYPE_CHECKING, Union
import logging

from sagemaker.workflow.parameters import ParameterString

from ...api.dag.base_dag import PipelineDAG
from ..base import StepBuilderBase, BasePipelineConfig

# Import PipelineTemplateBase directly - circular import should be resolved by now
from ..assembler.pipeline_template_base import PipelineTemplateBase

from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter as StepConfigResolver
from ...step_catalog import StepCatalog
from .validation import ValidationEngine
from .exceptions import ConfigurationError, ValidationError
from ...registry.exceptions import RegistryError

logger = logging.getLogger(__name__)


class DynamicPipelineTemplate(PipelineTemplateBase):
    """
    Dynamic pipeline template that works with any PipelineDAG.

    This template automatically implements the abstract methods of
    PipelineTemplateBase by using intelligent resolution mechanisms
    to map DAG nodes to configurations and step builders.
    """

    # Initialize CONFIG_CLASSES as empty - will be populated dynamically
    CONFIG_CLASSES: Dict[str, Type[BasePipelineConfig]] = {}

    def __init__(
        self,
        dag: PipelineDAG,
        config_path: str,
        config_resolver: Optional[StepConfigResolver] = None,
        step_catalog: Optional[StepCatalog] = None,
        skip_validation: bool = False,
        pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize dynamic template.

        Args:
            dag: PipelineDAG instance defining pipeline structure
            config_path: Path to configuration file
            config_resolver: Custom config resolver (optional)
            step_catalog: Custom step catalog (optional)
            skip_validation: Whether to skip validation (for testing)
            pipeline_parameters: Custom pipeline parameters from DAGCompiler (optional)
            **kwargs: Additional arguments for base template
        """
        # Initialize logger first so it's available in all methods
        self.logger = logging.getLogger(__name__)

        self._dag = dag
        self._config_resolver = config_resolver or StepConfigResolver()
        
        # Initialize step_catalog
        if step_catalog is not None:
            self._step_catalog = step_catalog
        else:
            self.logger.warning("step_catalog parameter is None, creating new StepCatalog()")
            try:
                self._step_catalog = StepCatalog()
            except Exception as e:
                self.logger.error(f"Failed to create new StepCatalog(): {e}")
                self._step_catalog = None
        
        self._validation_engine = ValidationEngine()

        # Store config_path as an instance attribute so it's available to _detect_config_classes
        self.config_path = config_path

        # Store if validation should be skipped (for testing purposes)
        self._skip_validation = skip_validation

        # Auto-detect required config classes based on DAG nodes
        # Don't set instance attribute - set class attribute before calling parent constructor
        cls = self.__class__
        if (
            not cls.CONFIG_CLASSES
        ):  # Only set if not already set (to avoid overwriting in instance reuse)
            cls.CONFIG_CLASSES = self._detect_config_classes()

        # Strategy 2 + 3: Early initialization with lazy loading flags
        self._resolved_config_map: Dict[str, BasePipelineConfig] = {}
        self._resolved_builder_map: Dict[str, Type[StepBuilderBase]] = {}
        self._loaded_metadata = None  # Store metadata from loaded configs
        # Lazy loading flags to preserve original logic
        self._config_map_loaded = False
        self._builder_map_loaded = False

        # Call parent constructor AFTER setting CONFIG_CLASSES
        # Pass pipeline_parameters directly to parent - parent handles storage
        super().__init__(
            config_path=config_path,
            sagemaker_session=kwargs.get("sagemaker_session"),
            role=kwargs.get("role"),
            registry_manager=kwargs.get("registry_manager"),
            dependency_resolver=kwargs.get("dependency_resolver"),
            pipeline_parameters=pipeline_parameters,  # Pass directly to parent
            step_catalog=self._step_catalog,  # ✅ CRITICAL FIX: Pass step_catalog to parent!
        )

    def _detect_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
        """
        Automatically detect required config classes from configuration file.

        This method analyzes the configuration file to determine which
        configuration classes are needed based on:
        1. Config type metadata in the configuration file
        2. Model type information in configuration entries
        3. Essential base classes needed for all pipelines

        Returns:
            Dictionary mapping config class names to config classes
        """
        # Import here to avoid circular imports
        from ...steps.configs.utils import detect_config_classes_from_json

        # Use the helper function to detect classes from the JSON file
        detected_classes = detect_config_classes_from_json(self.config_path)
        self.logger.debug(
            f"Detected {len(detected_classes)} required config classes from configuration file"
        )

        return detected_classes

    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Return the provided DAG.

        Returns:
            The PipelineDAG instance provided during initialization
        """
        return self._dag

    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Auto-map DAG nodes to configurations.

        Uses StepConfigResolver to intelligently match DAG node names
        to configuration instances from the loaded config file.

        Returns:
            Dictionary mapping DAG node names to configuration instances

        Raises:
            ConfigurationError: If nodes cannot be resolved to configurations
        """
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._config_map_loaded:
            try:
                dag_nodes = list(self._dag.nodes)
                self.logger.info(f"Resolving {len(dag_nodes)} DAG nodes to configurations")

                # Extract metadata from loaded configurations if available
                if self._loaded_metadata is None and hasattr(self, "loaded_config_data"):
                    if (
                        isinstance(self.loaded_config_data, dict)
                        and "metadata" in self.loaded_config_data
                    ):
                        self._loaded_metadata = self.loaded_config_data["metadata"]
                        self.logger.info(f"Using metadata from loaded configuration")

                # Use the config resolver to map nodes to configs
                resolved_map = self._config_resolver.resolve_config_map(
                    dag_nodes=dag_nodes,
                    available_configs=self.configs,
                    metadata=self._loaded_metadata,
                )
                
                # Update the early-initialized dict
                self._resolved_config_map.update(resolved_map)

                self.logger.info(
                    f"Successfully resolved all {len(self._resolved_config_map)} nodes"
                )

                # Log resolution details
                for node, config in self._resolved_config_map.items():
                    config_type = type(config).__name__
                    job_type = getattr(config, "job_type", "N/A")
                    self.logger.debug(f"  {node} → {config_type} (job_type: {job_type})")

                self._config_map_loaded = True

            except Exception as e:
                self.logger.error(f"Failed to resolve DAG nodes to configurations: {e}")
                raise ConfigurationError(f"Configuration resolution failed: {e}")

        return self._resolved_config_map

    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Auto-map step types to builders using StepCatalog.

        Uses StepCatalog to map configuration types to their
        corresponding step builder classes.

        Returns:
            Dictionary mapping step types to step builder classes

        Raises:
            RegistryError: If step builders cannot be found for config types
        """
        # Strategy 2 + 3: Use lazy loading flag to preserve original logic
        if not self._builder_map_loaded:
            try:
                # Check if step_catalog is None before calling get_builder_map()
                if self._step_catalog is None:
                    self.logger.error("CRITICAL: self._step_catalog is None when trying to call get_builder_map()!")
                    raise RegistryError("Step catalog is None - cannot get builder map")
                
                # Get the complete builder map from StepCatalog
                builder_map = self._step_catalog.get_builder_map()
                
                # Update the early-initialized dict
                self._resolved_builder_map.update(builder_map)

                self.logger.info(
                    f"Using {len(self._resolved_builder_map)} registered step builders from StepCatalog"
                )

                # Validate that all required builders are available
                config_map = self._create_config_map()
                missing_builders = []

                for node, config in config_map.items():
                    try:
                        # Use StepCatalog for config-to-builder resolution
                        builder_class = self._step_catalog.get_builder_for_config(
                            config, node_name=node
                        )
                        if builder_class:
                            self.logger.debug(f"  {node} → {builder_class.__name__}")
                        else:
                            missing_builders.append(f"{node} ({type(config).__name__})")
                    except Exception as e:
                        missing_builders.append(f"{node} ({type(config).__name__})")

                if missing_builders:
                    available_builders = list(self._resolved_builder_map.keys())
                    raise RegistryError(
                        f"Missing step builders for {len(missing_builders)} configurations",
                        unresolvable_types=missing_builders,
                        available_builders=available_builders,
                    )

                self._builder_map_loaded = True

            except Exception as e:
                self.logger.error(f"Failed to create step builder map: {e}")
                raise RegistryError(f"Step builder mapping failed: {e}")

        return self._resolved_builder_map

    def _validate_configuration(self) -> None:
        """
        Validate that all DAG nodes have corresponding configs.

        Performs comprehensive validation including:
        1. All DAG nodes have matching configurations
        2. All configurations have corresponding step builders
        3. Configuration-specific validation passes
        4. Dependency resolution is possible

        Raises:
            ValidationError: If validation fails
        """
        # Skip validation if requested (for testing purposes)
        if self._skip_validation:
            self.logger.info("Skipping configuration validation (requested)")
            return
        try:
            self.logger.info("Validating dynamic pipeline configuration")

            # Get resolved mappings
            dag_nodes = list(self._dag.nodes)
            config_map = self._create_config_map()
            builder_map = self._create_step_builder_map()

            # Run comprehensive validation
            validation_result = self._validation_engine.validate_dag_compatibility(
                dag_nodes=dag_nodes,
                available_configs=self.configs,
                config_map=config_map,
                builder_registry=builder_map,
            )

            if not validation_result.is_valid:
                self.logger.error("Configuration validation failed")
                self.logger.error(validation_result.detailed_report())
                # Flatten config_errors from Dict[str, List[str]] to List[str]
                flattened_config_errors = []
                for config_name, errors in validation_result.config_errors.items():
                    for error in errors:
                        flattened_config_errors.append(f"{config_name}: {error}")
                
                raise ValidationError(
                    "Dynamic pipeline configuration validation failed",
                    validation_errors={
                        "missing_configs": validation_result.missing_configs,
                        "unresolvable_builders": validation_result.unresolvable_builders,
                        "config_errors": flattened_config_errors,
                        "dependency_issues": validation_result.dependency_issues,
                    },
                )

            # Log warnings if any
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(warning)

            self.logger.info("Configuration validation passed successfully")

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Validation failed: {e}")

    def get_resolution_preview(self) -> Dict[str, Any]:
        """
        Get a preview of how DAG nodes will be resolved.

        Returns:
            Dictionary with resolution preview information
        """
        try:
            dag_nodes = list(self._dag.nodes)
            preview_data = self._config_resolver.preview_resolution(
                dag_nodes=dag_nodes,
                available_configs=self.configs,
                metadata=self._loaded_metadata,
            )

            # Convert to display format
            preview = {"nodes": len(dag_nodes), "resolutions": {}}

            for node, candidates in preview_data.items():
                resolutions = preview.get("resolutions")
                if isinstance(resolutions, dict):
                    if candidates:
                        best_candidate = candidates[0]
                        resolutions[node] = {
                            "config_type": best_candidate["config_type"],
                            "confidence": best_candidate["confidence"],
                            "method": best_candidate["method"],
                            "job_type": best_candidate["job_type"],
                            "alternatives": len(candidates) - 1,
                        }
                    else:
                        resolutions[node] = {
                            "config_type": "UNRESOLVED",
                            "confidence": 0.0,
                            "method": "none",
                            "job_type": "N/A",
                            "alternatives": 0,
                        }

            return preview

        except Exception as e:
            self.logger.error(f"Failed to generate resolution preview: {e}")
            return {"error": str(e)}

    def _store_pipeline_metadata(self, assembler: "PipelineAssembler") -> None:
        """
        Store pipeline metadata from template.

        This method stores general pipeline metadata (non-execution document related).
        Execution document metadata is now handled by the standalone execution document generator
        (ExecutionDocumentGenerator in cursus.mods.exe_doc.generator).

        Args:
            assembler: PipelineAssembler instance
        """
        # Store general pipeline metadata (non-execution document related)
        if hasattr(assembler, "step_instances"):
            self.pipeline_metadata["step_instances"] = assembler.step_instances
            self.logger.info(f"Stored {len(assembler.step_instances)} step instances")

        # Note: Cradle data loading requests and registration configs storage removed
        # as part of Phase 2 cleanup. Execution document metadata is now handled by
        # the standalone execution document generator.
        #
        # For execution document generation with Cradle data loading and registration, use:
        # from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
        # generator = ExecutionDocumentGenerator(config_path=config_path)
        # filled_doc = generator.fill_execution_document(dag, execution_doc)

    def get_step_catalog_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the step catalog.

        Returns:
            Dictionary with step catalog statistics
        """
        return {
            "supported_step_types": len(self._step_catalog.list_supported_step_types()),
            "indexed_steps": len(self._step_catalog._step_index) if hasattr(self._step_catalog, '_step_index') else 0,
        }

    def validate_before_build(self) -> bool:
        """
        Validate the configuration before building the pipeline.

        Returns:
            True if validation passes, False otherwise
        """
        try:
            self._validate_configuration()
            return True
        except ValidationError:
            return False

    def get_step_dependencies(self) -> Dict[str, list]:
        """
        Get the dependencies for each step based on the DAG.

        Returns:
            Dictionary mapping step names to their dependencies
        """
        dependencies = {}
        for node in self._dag.nodes:
            dependencies[node] = list(self._dag.get_dependencies(node))
        return dependencies

    def get_execution_order(self) -> list:
        """
        Get the topological execution order of steps.

        Returns:
            List of step names in execution order
        """
        try:
            return self._dag.topological_sort()
        except Exception as e:
            self.logger.error(f"Failed to get execution order: {e}")
            return list(self._dag.nodes)

    # NOTE: _get_pipeline_parameters() method is no longer needed!
    # Parent class (PipelineTemplateBase) handles parameter storage and retrieval automatically.
    # DAGCompiler provides default parameters when none are specified.

    # Note: All execution document methods removed as part of Phase 2 cleanup
    # Execution document generation is now handled by the standalone execution document generator
    # (ExecutionDocumentGenerator in cursus.mods.exe_doc.generator)
    #
    # Removed methods:
    # - fill_execution_document()
    # - _fill_cradle_configurations()
    # - _create_execution_doc_config()
    # - _find_registration_step_nodes()
    # - _fill_registration_configurations()
    # - _has_required_registration_fields()
