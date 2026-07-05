"""
DAG Compiler for the Pipeline API.

This module provides the main API functions for compiling PipelineDAG structures
into executable SageMaker pipelines.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    # Imported lazily at runtime inside create_template(); declared here only so the
    # "DynamicPipelineTemplate" string annotations resolve for type checkers/linters.
    from .dynamic_template import DynamicPipelineTemplate
from pathlib import Path

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.network import NetworkConfig

from ...api.dag.base_dag import PipelineDAG

# Import constants from core library (with fallback)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Could not import constants from mods_workflow_core, using local definitions"
    )
    # Define pipeline parameters locally if import fails
    PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
    KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMS_ENCRYPTION_KEY_PARAM")
    SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID")
    VPC_SUBNET = ParameterString(name="VPC_SUBNET")
    # Also create the network config
    PROCESSING_JOB_SHARED_NETWORK_CONFIG = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=[SECURITY_GROUP_ID],
        subnets=[VPC_SUBNET],
        encrypt_inter_container_traffic=True,
    )
from ...step_catalog.adapters.config_resolver import (
    StepConfigResolverAdapter as StepConfigResolver,
)
from ...step_catalog import StepCatalog
from .validation import (
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
    ValidationEngine,
)
from .exceptions import PipelineAPIError

logger = logging.getLogger(__name__)


def compile_dag_to_pipeline(
    dag: Optional[PipelineDAG] = None,
    dag_path: Optional[str] = None,
    config_path: str = None,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Compile a PipelineDAG into a complete SageMaker Pipeline.

    This is the main entry point for users who want a simple, one-call
    compilation from DAG to pipeline.

    Args:
        dag: PipelineDAG instance defining the pipeline structure (optional if dag_path provided)
        dag_path: Path to serialized DAG JSON file (optional if dag provided)
        config_path: Path to configuration file containing step configs
        sagemaker_session: SageMaker session for pipeline execution
        role: IAM role for pipeline execution
        pipeline_name: Optional pipeline name override
        **kwargs: Additional arguments passed to template constructor

    Returns:
        Generated SageMaker Pipeline ready for execution

    Raises:
        ValueError: If neither dag nor dag_path provided, or if DAG is invalid
        FileNotFoundError: If dag_path or config_path files not found
        ConfigurationError: If configuration validation fails
        RegistryError: If step builders not found for config types

    Example:
        >>> # Option 1: Using PipelineDAG instance
        >>> dag = PipelineDAG()
        >>> dag.add_node("data_load")
        >>> dag.add_node("preprocess")
        >>> dag.add_edge("data_load", "preprocess")
        >>>
        >>> pipeline = compile_dag_to_pipeline(
        ...     dag=dag,
        ...     config_path="configs/my_pipeline.json",
        ...     sagemaker_session=session,
        ...     role="arn:aws:iam::123456789012:role/SageMakerRole"
        ... )
        >>> pipeline.upsert()
        >>>
        >>> # Option 2: Loading DAG from file
        >>> pipeline = compile_dag_to_pipeline(
        ...     dag_path="saved_dags/my_pipeline.json",
        ...     config_path="configs/my_pipeline.json",
        ...     sagemaker_session=session,
        ...     role="arn:aws:iam::123456789012:role/SageMakerRole"
        ... )
    """
    try:
        # Load DAG from path if provided
        if dag_path is not None:
            from ...api.dag import import_dag_from_json

            logger.info(f"Loading DAG from file: {dag_path}")
            dag = import_dag_from_json(dag_path)
            logger.info(
                f"Successfully loaded DAG with {len(dag.nodes)} nodes from {dag_path}"
            )

        # Validate that we have a DAG
        if dag is None:
            raise ValueError("Must provide either 'dag' or 'dag_path' parameter")

        # Validate inputs
        if not isinstance(dag, PipelineDAG):
            raise ValueError("dag must be a PipelineDAG instance")

        if not dag.nodes:
            raise ValueError("DAG must contain at least one node")

        if config_path is None:
            raise ValueError("config_path is required")

        logger.info(f"Compiling DAG with {len(dag.nodes)} nodes to pipeline")

        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Create compiler
        compiler = PipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs,
        )

        # Use compile method which uses our create_template method
        pipeline = compiler.compile(dag, pipeline_name=pipeline_name)

        logger.info(f"Successfully compiled DAG to pipeline: {pipeline.name}")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to compile DAG to pipeline: {e}")
        raise PipelineAPIError(f"DAG compilation failed: {e}") from e


class PipelineDAGCompiler:
    """
    Advanced API for DAG-to-template compilation with additional control.

    This class provides more control over the compilation process, including
    validation, debugging, and customization options.
    """

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        config_resolver: Optional[StepConfigResolver] = None,
        step_catalog: Optional[StepCatalog] = None,
        pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
        project_root: Optional[Union[str, Path]] = None,
        anchor_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize compiler with configuration and session.

        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session for pipeline execution
            role: IAM role for pipeline execution
            config_resolver: Custom config resolver (optional)
            step_catalog: Custom step catalog (optional)
            pipeline_parameters: Pipeline parameters to pass to template (optional)
            project_root: Absolute path to the user's project **folder**, used as the
                highest-priority anchor for resolving step ``source_dir``/``processing_source_dir``
                across deployment scenarios (the "caller hook"). Pipelines may pass
                ``Path(__file__).parent`` from the module that defines ``generate_pipeline()``.
                When omitted, it is inferred from ``config_path`` (the project directory that
                contains the config file). Pushed process-wide so configs resolve against it
                without needing ``CURSUS_PROJECT_BASE`` or a ``project_root_folder`` field.
            anchor_file: A **file** inside the project folder — pass ``__file__`` from the
                module that defines ``generate_pipeline()`` and cursus derives the project
                root as its parent directory. This is the self-documenting form of the caller
                hook (``anchor_file=__file__``); it is equivalent to
                ``project_root=Path(__file__).parent``. If both are given, ``project_root``
                wins and a warning is logged when they disagree.
            **kwargs: Additional arguments for template constructor
        """
        self.config_path = config_path
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.template_kwargs = kwargs

        # Caller hook: push the project root for path resolution (Strategy 0). An explicit
        # anchor (project_root folder or anchor_file=__file__) wins; otherwise infer from the
        # config file's location (config-anchored fallback).
        self.project_root = self._resolve_project_root(
            project_root, config_path, anchor_file=anchor_file
        )
        if self.project_root:
            try:
                from ..utils.hybrid_path_resolution import set_project_root

                set_project_root(self.project_root)
            except Exception:  # pragma: no cover - resolution is best-effort
                pass

        # Store pipeline parameters for template creation
        # Use default parameters if none provided
        if pipeline_parameters is None:
            self.pipeline_parameters = [
                PIPELINE_EXECUTION_TEMP_DIR,
                KMS_ENCRYPTION_KEY_PARAM,
                SECURITY_GROUP_ID,
                VPC_SUBNET,
            ]
        else:
            self.pipeline_parameters = pipeline_parameters

        # Initialize components
        self.config_resolver = config_resolver or StepConfigResolver()
        self.step_catalog = step_catalog or StepCatalog()
        self.validation_engine = ValidationEngine()

        self.logger = logging.getLogger(__name__)

        # Store the last template created during compilation
        self._last_template = None

        # Validate config file exists
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    @staticmethod
    def _resolve_project_root(
        project_root: Optional[Union[str, Path]],
        config_path: str,
        anchor_file: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        """Resolve the project-root anchor for path resolution.

        Priority:
        1. Explicit ``project_root`` (the caller hook — a project **directory**, typically
           ``Path(__file__).parent``).
        2. Explicit ``anchor_file`` (a **file** inside the project, typically ``__file__``);
           its parent directory becomes the project root.
        3. Inferred from ``config_path``: walk up from the config file to the nearest project
           directory, treating a ``pipeline_config``/``pipeline_configs`` parent as the config
           dir and using its parent as the project root; otherwise the config file's directory.

        Both explicit forms are normalized by the shared ``resolve_anchor`` helper so a file
        and a directory collapse to the same project root. If both ``project_root`` and
        ``anchor_file`` are given and disagree, ``project_root`` wins and a warning is logged.

        Returns an absolute path string, or None if nothing usable.
        """
        if project_root or anchor_file:
            from ..utils.hybrid_path_resolution import resolve_anchor

            resolved_root = resolve_anchor(project_root)
            resolved_anchor = resolve_anchor(anchor_file)
            if resolved_root and resolved_anchor and resolved_root != resolved_anchor:
                logger.warning(
                    "PipelineDAGCompiler received both project_root (%s) and anchor_file "
                    "(-> %s) that disagree; using project_root.",
                    resolved_root,
                    resolved_anchor,
                )
            return resolved_root or resolved_anchor

        try:
            config_dir = Path(config_path).expanduser().resolve().parent
            # If the config lives in a recognized config dir, the project is its parent.
            if config_dir.name in ("pipeline_config", "pipeline_configs"):
                return str(config_dir.parent)
            # If a recognized config dir is the parent (config in a versioned subdir),
            # the project is one level above that.
            if config_dir.parent.name in ("pipeline_config", "pipeline_configs"):
                return str(config_dir.parent.parent)
            return str(config_dir)
        except Exception:  # pragma: no cover
            return None

    def validate_dag_compatibility(
        self, dag: Union[PipelineDAG, str]
    ) -> ValidationResult:
        """
        Validate that DAG nodes have corresponding configurations.

        Returns detailed validation results including:
        - Missing configurations
        - Unresolvable step builders
        - Configuration validation errors
        - Dependency resolution issues

        Args:
            dag: PipelineDAG instance or path to serialized DAG file

        Returns:
            ValidationResult with detailed validation information
        """
        # Load DAG from file if path provided
        if isinstance(dag, str):
            from ...api.dag import import_dag_from_json

            self.logger.info(f"Loading DAG from file for validation: {dag}")
            dag = import_dag_from_json(dag)

        try:
            self.logger.info(f"Validating DAG compatibility for {len(dag.nodes)} nodes")

            # Create a template using our create_template method
            temp_template = self.create_template(dag)

            # Get resolved mappings
            dag_nodes = list(dag.nodes)
            available_configs = temp_template.configs

            try:
                config_map = temp_template._create_config_map()
            except Exception as e:
                # If config resolution fails, create partial validation result
                return ValidationResult(
                    is_valid=False,
                    missing_configs=dag_nodes,
                    unresolvable_builders=[],
                    config_errors={"resolution": [str(e)]},
                    dependency_issues=[],
                    warnings=[],
                )

            try:
                builder_map = temp_template._create_step_builder_map()
            except Exception as e:
                # If builder resolution fails, create partial validation result
                return ValidationResult(
                    is_valid=False,
                    missing_configs=[],
                    unresolvable_builders=list(config_map.keys()),
                    config_errors={"builder_resolution": [str(e)]},
                    dependency_issues=[],
                    warnings=[],
                )

            # Run comprehensive validation
            validation_result = self.validation_engine.validate_dag_compatibility(
                dag_nodes=dag_nodes,
                available_configs=available_configs,
                config_map=config_map,
                builder_registry=builder_map,
                metadata=getattr(temp_template, "_loaded_metadata", None),
            )

            self.logger.info(f"Validation completed: {validation_result.summary()}")
            return validation_result

        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            return ValidationResult(
                is_valid=False,
                missing_configs=[],
                unresolvable_builders=[],
                config_errors={"validation_error": [str(e)]},
                dependency_issues=[],
                warnings=[],
            )

    def preview_resolution(self, dag: Union[PipelineDAG, str]) -> ResolutionPreview:
        """
        Preview how DAG nodes will be resolved to configs and builders.

        Returns a detailed preview showing:
        - Node → Configuration mappings
        - Configuration → Step Builder mappings
        - Detected step types and dependencies
        - Potential issues or ambiguities

        Args:
            dag: PipelineDAG instance or path to serialized DAG file

        Returns:
            ResolutionPreview with detailed resolution information
        """
        # Load DAG from file if path provided
        if isinstance(dag, str):
            from ...api.dag import import_dag_from_json

            self.logger.info(f"Loading DAG from file for resolution preview: {dag}")
            dag = import_dag_from_json(dag)

        try:
            self.logger.info(f"Previewing resolution for {len(dag.nodes)} DAG nodes")

            # Create a template using our create_template method
            temp_template = self.create_template(dag)

            # Get preview data
            dag_nodes = list(dag.nodes)
            available_configs = temp_template.configs

            # Get metadata from template if available
            metadata = None
            if hasattr(temp_template, "_loaded_metadata"):
                metadata = temp_template._loaded_metadata

            # Get resolution candidates
            preview_data = self.config_resolver.preview_resolution(
                dag_nodes=dag_nodes,
                available_configs=available_configs,
                metadata=metadata,
            )

            # Build preview result
            node_config_map = {}
            config_builder_map = {}
            resolution_confidence = {}
            ambiguous_resolutions = []
            recommendations = []

            # preview_resolution returns a fixed-shape dict, NOT {node: [candidates]}. Consume the
            # real shape: node_resolution maps node -> {config_type, confidence, method, job_type}
            # (or {error, error_type} for an unresolved node). The old code iterated .items() over
            # the top-level dict and indexed candidates[0], raising KeyError on every key and
            # silently falling back to an empty preview (deep dive 2026-07-03, T4).
            node_resolution = preview_data.get("node_resolution", {})
            for node, info in node_resolution.items():
                if info and "error" not in info:
                    config_type = info.get("config_type", "UNKNOWN")
                    confidence = info.get("confidence", 0.0)

                    node_config_map[node] = config_type
                    resolution_confidence[node] = confidence

                    # Get builder for this config type
                    try:
                        builder_class = self.step_catalog.get_builder_for_step_type(
                            config_type
                        )
                        if builder_class:
                            # getattr-guard: under Design-B a provider callable may lack __name__
                            # (FZ 31e1d3g1 Phase 0). A class still has it → identical output today.
                            config_builder_map[config_type] = getattr(
                                builder_class, "__name__", str(builder_class)
                            )
                        else:
                            config_builder_map[config_type] = "UNKNOWN"
                    except Exception:
                        config_builder_map[config_type] = "UNKNOWN"

                    # Add recommendations for low confidence
                    if confidence < 0.8:
                        recommendations.append(
                            f"Consider renaming '{node}' for better matching "
                            f"(confidence {confidence:.2f})"
                        )
                else:
                    node_config_map[node] = "UNRESOLVED"
                    resolution_confidence[node] = 0.0
                    recommendations.append(
                        f"Node '{node}' did not resolve to any config — add an explicit "
                        f"config key or metadata.config_types entry."
                    )
                    recommendations.append(f"Add configuration for node '{node}'")

            preview = ResolutionPreview(
                node_config_map=node_config_map,
                config_builder_map=config_builder_map,
                resolution_confidence=resolution_confidence,
                ambiguous_resolutions=ambiguous_resolutions,
                recommendations=recommendations,
            )

            self.logger.info("Resolution preview completed successfully")
            return preview

        except Exception as e:
            self.logger.error(f"Failed to generate resolution preview: {e}")
            # Return empty preview with error
            return ResolutionPreview(
                node_config_map={},
                config_builder_map={},
                resolution_confidence={},
                ambiguous_resolutions=[],
                recommendations=[f"Preview failed: {str(e)}"],
            )

    def compile(
        self,
        dag: Union[PipelineDAG, str],
        pipeline_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Compile DAG to pipeline with full control.

        Args:
            dag: PipelineDAG instance or path to serialized DAG file
            pipeline_name: Optional pipeline name override
            **kwargs: Additional arguments for template

        Returns:
            Generated SageMaker Pipeline

        Raises:
            PipelineAPIError: If compilation fails
        """
        # Load DAG from file if path provided
        if isinstance(dag, str):
            from ...api.dag import import_dag_from_json

            self.logger.info(f"Loading DAG from file for compilation: {dag}")
            dag = import_dag_from_json(dag)

        try:
            self.logger.info(f"Compiling DAG with {len(dag.nodes)} nodes to pipeline")

            # Reuse create_template. Default to skip_validation=True for performance (validation is
            # typically run separately via compile_with_report / validate_dag_compatibility), BUT
            # honor an explicit caller override instead of forcibly overriding it — a caller that
            # asks for skip_validation=False must get the alignment engine run before assembly
            # (deep dive 2026-07-03, T4). Note: config RESOLUTION is now strict regardless (it raises
            # on any unresolved/misresolved node), so skipping validation no longer means silently
            # accepting a bad config map — it only skips the separate alignment layer.
            template_kwargs = {**self.template_kwargs, **kwargs}
            template_kwargs.setdefault("skip_validation", True)

            template = self.create_template(dag, **template_kwargs)

            # Build pipeline
            from typing import cast
            from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

            pipeline = cast(SageMakerPipeline, template.generate_pipeline())

            # Store the template after generate_pipeline() has updated its internal state
            self._last_template = template

            # Override pipeline name if provided or generate a new one
            if pipeline_name:
                pipeline.name = pipeline_name
            else:
                # Import here to avoid circular import
                from .name_generator import generate_pipeline_name

                # Get pipeline_name and pipeline_version from any config (all have same values due to inheritance)
                if template.configs:
                    first_config = next(iter(template.configs.values()))
                    base_name = getattr(first_config, "pipeline_name", "cursus")
                    version = getattr(first_config, "pipeline_version", "0.0.0")
                else:
                    base_name = "cursus"
                    version = "0.0.0"

                # Generate a name using the same approach as PipelineTemplateBase
                pipeline.name = generate_pipeline_name(base_name, version)

            self.logger.info(f"Successfully compiled DAG to pipeline: {pipeline.name}")
            return pipeline

        except Exception as e:
            self.logger.error(f"Failed to compile DAG to pipeline: {e}")
            raise PipelineAPIError(f"DAG compilation failed: {e}") from e

    def compile_with_report(
        self,
        dag: Union[PipelineDAG, str],
        pipeline_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Pipeline, ConversionReport]:
        """
        Compile DAG to pipeline and return detailed compilation report.

        Args:
            dag: PipelineDAG instance or path to serialized DAG file
            pipeline_name: Optional pipeline name override
            **kwargs: Additional arguments for template

        Returns:
            Tuple of (Pipeline, ConversionReport)
        """
        # Load DAG from file if path provided
        if isinstance(dag, str):
            from ...api.dag import import_dag_from_json

            self.logger.info(
                f"Loading DAG from file for compilation with report: {dag}"
            )
            dag = import_dag_from_json(dag)

        try:
            self.logger.info("Compiling DAG with detailed reporting")

            # Compile pipeline with the loaded DAG instance
            pipeline = self.compile(dag, pipeline_name=pipeline_name, **kwargs)

            # Generate report
            dag_nodes = list(dag.nodes)
            resolution_details = {}
            total_confidence = 0.0
            warnings = []

            # Get resolution preview for report details
            preview = self.preview_resolution(dag)

            for node in dag_nodes:
                if node in preview.node_config_map:
                    config_type = preview.node_config_map[node]
                    confidence = preview.resolution_confidence.get(node, 0.0)
                    builder_type = preview.config_builder_map.get(
                        config_type, "Unknown"
                    )

                    resolution_details[node] = {
                        "config_type": config_type,
                        "builder_type": builder_type,
                        "confidence": confidence,
                    }

                    total_confidence += confidence

                    if confidence < 0.8:
                        warnings.append(
                            f"Low confidence resolution for node '{node}': {confidence:.2f}"
                        )

            avg_confidence = total_confidence / len(dag_nodes) if dag_nodes else 0.0

            # Add ambiguity warnings
            warnings.extend(preview.ambiguous_resolutions)

            report = ConversionReport(
                pipeline_name=pipeline.name,
                steps=dag_nodes,
                resolution_details=resolution_details,
                avg_confidence=avg_confidence,
                warnings=warnings,
                metadata={
                    "dag_nodes": len(dag_nodes),
                    "dag_edges": len(dag.edges),
                    "config_path": self.config_path,
                    "step_catalog_stats": {
                        "supported_step_types": len(
                            self.step_catalog.list_supported_step_types()
                        ),
                        "indexed_steps": len(self.step_catalog._step_index)
                        if hasattr(self.step_catalog, "_step_index")
                        else 0,
                    },
                },
            )

            self.logger.info(f"Compilation completed with report: {report.summary()}")

            # NVMe fix: patch pipeline.definition() to remove VolumeKmsKeyId from
            # NVMe ProcessingSteps. The SDK's Processor lacks the instance_supports_kms
            # gate that EstimatorBase has, so VolumeKmsKeyId leaks into the definition
            # for NVMe instances. We post-process the JSON output.
            import json as _json
            from sagemaker.utils import instance_supports_kms as _supports_kms

            _orig_defn = pipeline.definition

            def _nvme_aware_definition(*_args, **_kwargs):
                defn_str = _orig_defn(*_args, **_kwargs)
                defn = _json.loads(defn_str)
                for _step in defn.get("Steps", []):
                    _cluster = (
                        _step.get("Arguments", {})
                        .get("ProcessingResources", {})
                        .get("ClusterConfig", {})
                    )
                    _inst = _cluster.get("InstanceType")
                    if _inst and isinstance(_inst, str) and not _supports_kms(_inst):
                        _cluster.pop("VolumeKmsKeyId", None)
                return _json.dumps(defn)

            pipeline.definition = _nvme_aware_definition

            return pipeline, report

        except Exception as e:
            self.logger.error(f"Failed to compile DAG with report: {e}")
            raise PipelineAPIError(f"DAG compilation with report failed: {e}") from e

    def create_template(
        self, dag: PipelineDAG, **kwargs: Any
    ) -> "DynamicPipelineTemplate":
        """
        Create a pipeline template from the DAG without generating the pipeline.

        This allows inspecting or modifying the template before pipeline generation.

        Args:
            dag: PipelineDAG instance to create a template for
            **kwargs: Additional arguments for template

        Returns:
            DynamicPipelineTemplate instance ready for pipeline generation

        Raises:
            PipelineAPIError: If template creation fails
        """
        try:
            # Import here to avoid circular import
            from .dynamic_template import DynamicPipelineTemplate

            self.logger.info(f"Creating template for DAG with {len(dag.nodes)} nodes")

            # Merge kwargs with default values
            template_kwargs = {**self.template_kwargs}

            # Set default skip_validation if not provided
            if "skip_validation" not in kwargs:
                template_kwargs["skip_validation"] = (
                    False  # Enable validation by default
                )

            # Update with any other kwargs provided
            template_kwargs.update(kwargs)

            # Create dynamic template
            template = DynamicPipelineTemplate(
                dag=dag,
                config_path=self.config_path,
                config_resolver=self.config_resolver,
                step_catalog=self.step_catalog,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
                pipeline_parameters=self.pipeline_parameters,  # Pass parameters to template
                **template_kwargs,
            )

            self.logger.info("Successfully created template")
            return template

        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            raise PipelineAPIError(f"Template creation failed: {e}") from e

    def get_supported_step_types(self) -> list:
        """
        Get list of supported step types.

        Returns:
            List of supported step type names
        """
        return self.step_catalog.list_supported_step_types()

    def validate_config_file(self) -> Dict[str, Any]:
        """
        Validate the configuration file structure.

        Returns:
            Dictionary with validation results
        """
        try:
            # Create a minimal DAG to test config loading
            test_dag = PipelineDAG()
            test_dag.add_node("test_node")

            # Use create_template with skip_validation=True to just test config loading
            temp_template = self.create_template(dag=test_dag, skip_validation=True)

            configs = temp_template.configs

            return {
                "valid": True,
                "config_count": len(configs),
                "config_types": [type(config).__name__ for config in configs.values()],
                "config_names": list(configs.keys()),
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "config_count": 0,
                "config_types": [],
                "config_names": [],
            }

    def get_last_template(self) -> Optional["DynamicPipelineTemplate"]:
        """
        Get the last template used during compilation.

        This template will have its pipeline_metadata populated from the generation process.
        Use this method to get access to a template that has gone through the complete
        pipeline generation process, particularly useful for execution document generation.

        Returns:
            The last template used in compilation, or None if no compilation has occurred
        """
        return self._last_template

    def analyze_pipeline_structure(self) -> None:
        """
        Analyze and print the complete pipeline structure.

        Delegates to the template's analyze_pipeline_structure method.
        Must be called after compile() or compile_with_report().

        Raises:
            AttributeError: If called before compilation
        """
        if self._last_template is None:
            raise AttributeError(
                "No template found. Call compile() or compile_with_report() first."
            )

        self._last_template.analyze_pipeline_structure()

    # Note: compile_and_fill_execution_doc() method removed as part of Phase 2 cleanup
    # Execution document generation is now handled by the standalone execution document generator
    # (ExecutionDocumentGenerator in cursus.mods.exe_doc.generator)
    #
    # Users should now:
    # 1. Use compile() to generate the pipeline
    # 2. Use ExecutionDocumentGenerator separately to fill execution documents
