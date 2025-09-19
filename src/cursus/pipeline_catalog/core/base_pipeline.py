"""
Base Pipeline Class

This module provides a base class for all pipelines that incorporates PipelineDAGCompiler,
calls pipeline DAG generator, and maintains a unified interface while keeping pipeline 
metadata and registry integration.

The base class provides:
- Unified interface for all pipelines (both MODS and regular)
- Integration with PipelineDAGCompiler
- Pipeline metadata and registry management
- Execution document handling
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata
from .catalog_registry import CatalogRegistry

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
    from sagemaker.workflow.parameters import ParameterString
    from sagemaker.network import NetworkConfig
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Base class for all pipelines that incorporates PipelineDAGCompiler functionality.
    
    This class provides a unified interface for all pipelines, incorporating:
    - PipelineDAGCompiler integration
    - Pipeline DAG generation
    - Pipeline metadata and registry management
    - Execution document handling
    - Pipeline parameters support for runtime configuration
    
    Subclasses must implement:
    - create_dag(): Create the pipeline DAG
    - get_enhanced_dag_metadata(): Provide pipeline metadata
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        execution_role: Optional[str] = None,
        enable_mods: bool = True,
        validate: bool = True,
        pipeline_parameters: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize the pipeline with configuration and session details.

        Args:
            config_path: Path to the configuration file
            sagemaker_session: SageMaker pipeline session
            execution_role: IAM role for pipeline execution
            enable_mods: Whether to enable MODS features (default: True)
            validate: Whether to validate the DAG before compilation
            pipeline_parameters: Custom pipeline parameters (optional)
            **kwargs: Additional arguments for template constructor
        """
        # Set defaults if not provided
        if sagemaker_session is None:
            sagemaker_session = PipelineSession()

        if execution_role is None:
            # Get default role from session
            execution_role = sagemaker_session.get_caller_identity_arn()

        # Store configuration
        self.config_path = config_path
        self.sagemaker_session = sagemaker_session
        self.execution_role = execution_role
        self.enable_mods = enable_mods
        self.validate = validate
        self.template_kwargs = kwargs

        # Set up pipeline parameters - use provided parameters or create default list
        if pipeline_parameters is None:
            self.pipeline_parameters = [
                PIPELINE_EXECUTION_TEMP_DIR,
                KMS_ENCRYPTION_KEY_PARAM,
                SECURITY_GROUP_ID,
                VPC_SUBNET,
            ]
        else:
            self.pipeline_parameters = pipeline_parameters

        # Load configuration from file if provided and exists
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        elif config_path:
            logger.warning(f"Configuration file {config_path} not found, using defaults")

        # Create pipeline DAG
        self.dag = self.create_dag()
        logger.info(
            f"Created DAG with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges"
        )

        # Initialize compiler
        self.dag_compiler = self._initialize_compiler()
        logger.info("Initialized DAG compiler")

        # Store the last template for execution document handling
        self._last_template = None

    @abstractmethod
    def create_dag(self) -> PipelineDAG:
        """
        Create the pipeline DAG structure.
        
        This method must be implemented by subclasses to define the specific
        DAG structure for their pipeline.
        
        Returns:
            PipelineDAG: The directed acyclic graph for the pipeline
        """
        pass

    @abstractmethod
    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        """
        Get enhanced DAG metadata for this pipeline.
        
        This method must be implemented by subclasses to provide pipeline-specific
        metadata for registry integration and documentation.
        
        Returns:
            EnhancedDAGMetadata: Enhanced metadata with Zettelkasten properties
        """
        pass

    def _initialize_compiler(self) -> PipelineDAGCompiler:
        """
        Initialize the DAG compiler with pipeline parameters.
        
        Returns:
            PipelineDAGCompiler: The compiler instance
        """
        # Create compiler with pipeline parameters
        dag_compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            sagemaker_session=self.sagemaker_session,
            role=self.execution_role,
            pipeline_parameters=self.pipeline_parameters,
            **self.template_kwargs
        )

        logger.info(f"Initialized DAG compiler with {len(self.pipeline_parameters)} pipeline parameters")
        return dag_compiler

    def generate_pipeline(self) -> Pipeline:
        """
        Generate a SageMaker Pipeline using the DAG Compiler.

        This method provides the main interface for pipeline generation while using our
        DAG compilation process internally.

        Returns:
            Pipeline: Compiled SageMaker Pipeline
        """
        # Validate the DAG if requested
        if self.validate:
            validation = self.dag_compiler.validate_dag_compatibility(self.dag)
            if not validation.is_valid:
                logger.warning(f"DAG validation failed: {validation.summary()}")
                if validation.missing_configs:
                    logger.warning(f"Missing configs: {validation.missing_configs}")
                if validation.unresolvable_builders:
                    logger.warning(
                        f"Unresolvable builders: {validation.unresolvable_builders}"
                    )
                if validation.config_errors:
                    logger.warning(f"Config errors: {validation.config_errors}")
                if validation.dependency_issues:
                    logger.warning(f"Dependency issues: {validation.dependency_issues}")

        # Use the compiler to build the pipeline
        pipeline, report = self.dag_compiler.compile_with_report(dag=self.dag)

        # Store the template for later use
        self._last_template = self.dag_compiler.get_last_template()

        # Log compilation details
        logger.info(f"Pipeline '{pipeline.name}' created successfully")
        logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")

        return pipeline

    def validate_dag_compatibility(self) -> Dict[str, Any]:
        """
        Validate that the DAG is compatible with the configuration.

        Returns:
            Dict: Validation results
        """
        validation = self.dag_compiler.validate_dag_compatibility(self.dag)
        return {
            "is_valid": validation.is_valid,
            "missing_configs": validation.missing_configs,
            "unresolvable_builders": validation.unresolvable_builders,
            "config_errors": validation.config_errors,
            "dependency_issues": validation.dependency_issues,
            "warnings": validation.warnings,
        }

    def preview_resolution(self) -> Dict[str, Any]:
        """
        Preview how DAG nodes will be resolved to configs and builders.

        Returns:
            Dict: Preview of node resolution
        """
        preview = self.dag_compiler.preview_resolution(self.dag)
        return {
            "node_config_map": preview.node_config_map,
            "config_builder_map": preview.config_builder_map,
            "resolution_confidence": preview.resolution_confidence,
            "ambiguous_resolutions": preview.ambiguous_resolutions,
            "recommendations": preview.recommendations,
        }

    def get_last_template(self):
        """
        Get the last template used during compilation.

        Returns:
            Any: The template object
        """
        return self._last_template or self.dag_compiler.get_last_template()

    # Note: fill_execution_document() method removed to achieve complete independence
    # between pipeline generation and execution document generation modules.
    #
    # For execution document generation, use the standalone module:
    # from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
    # generator = ExecutionDocumentGenerator(config_path=config_path)
    # filled_doc = generator.fill_execution_document(dag, execution_doc)
    #
    # Or use the pipeline catalog integration:
    # from cursus.pipeline_catalog.pipeline_exe import generate_execution_document_for_pipeline
    # filled_doc = generate_execution_document_for_pipeline(pipeline_name, config_path, execution_doc)

    def sync_to_registry(self) -> bool:
        """
        Synchronize this pipeline's metadata to the catalog registry.

        Returns:
            bool: True if synchronization was successful, False otherwise
        """
        try:
            registry = CatalogRegistry()
            enhanced_metadata = self.get_enhanced_dag_metadata()

            # Add or update the pipeline node using the enhanced metadata
            success = registry.add_or_update_enhanced_node(enhanced_metadata)

            if success:
                logger.info(
                    f"Successfully synchronized {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry"
                )
            else:
                logger.warning(
                    f"Failed to synchronize {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry"
                )

            return success

        except Exception as e:
            logger.error(f"Error synchronizing to registry: {e}")
            return False

    def create_pipeline(
        self,
        config_path: Optional[str] = None,
        session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        enable_mods: Optional[bool] = None,
        validate: Optional[bool] = None,
    ) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
        """
        Create a SageMaker Pipeline with detailed reporting (compatibility method).
        
        This method provides compatibility with the functional interface used in
        existing pipeline implementations.

        Args:
            config_path: Path to the configuration file (overrides instance config)
            session: SageMaker pipeline session (overrides instance session)
            role: IAM role for pipeline execution (overrides instance role)
            enable_mods: Whether to enable MODS features (overrides instance setting)
            validate: Whether to validate the DAG before compilation (overrides instance setting)

        Returns:
            Tuple containing:
                - Pipeline: The created SageMaker pipeline
                - Dict: Conversion report with details about the compilation
                - PipelineDAGCompiler: The compiler instance
                - Any: The template instance
        """
        # Update instance settings if overrides provided
        if config_path is not None:
            self.config_path = config_path
        if session is not None:
            self.sagemaker_session = session
        if role is not None:
            self.execution_role = role
        if enable_mods is not None:
            self.enable_mods = enable_mods
        if validate is not None:
            self.validate = validate

        # Reinitialize compiler if settings changed
        if any([config_path, session, role, enable_mods]) is not None:
            self.dag_compiler = self._initialize_compiler()

        # Generate pipeline
        pipeline = self.generate_pipeline()

        # Get detailed report
        _, report = self.dag_compiler.compile_with_report(dag=self.dag)

        # Get template instance
        template_instance = self.get_last_template()

        # Sync to registry after successful pipeline creation
        self.sync_to_registry()

        return pipeline, report, self.dag_compiler, template_instance

    def save_execution_document(self, document: Dict[str, Any], output_path: str) -> None:
        """
        Save the execution document to a file.

        Args:
            document: The execution document to save
            output_path: Path where to save the document
        """
        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the document
        with open(output_path, "w") as f:
            json.dump(document, f, indent=2)

        logger.info(f"Execution document saved to: {output_path}")

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get the current pipeline configuration.
        
        Returns:
            Dict: Current configuration dictionary
        """
        return self.config.copy()

    def update_pipeline_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the pipeline configuration.
        
        Args:
            config_updates: Dictionary of configuration updates to apply
        """
        self.config.update(config_updates)
        logger.info(f"Updated pipeline configuration with {len(config_updates)} changes")

    def get_dag_info(self) -> Dict[str, Any]:
        """
        Get information about the current DAG.
        
        Returns:
            Dict: DAG information including nodes, edges, and metadata
        """
        return {
            "nodes": list(self.dag.nodes),
            "edges": list(self.dag.edges),
            "node_count": len(self.dag.nodes),
            "edge_count": len(self.dag.edges),
            "is_valid": len(self.dag.nodes) > 0,
        }

    def get_pipeline_parameters(self) -> list:
        """
        Get the current pipeline parameters.
        
        Returns:
            list: Current pipeline parameters
        """
        return self.pipeline_parameters.copy()

    def set_pipeline_parameters(self, pipeline_parameters: list) -> None:
        """
        Set custom pipeline parameters and reinitialize the compiler.
        
        Args:
            pipeline_parameters: List of pipeline parameters to use
        """
        self.pipeline_parameters = pipeline_parameters
        # Reinitialize compiler with new parameters
        self.dag_compiler = self._initialize_compiler()
        logger.info(f"Updated pipeline parameters and reinitialized compiler with {len(pipeline_parameters)} parameters")

    @staticmethod
    def create_pipeline_parameters(execution_s3_prefix: Optional[str] = None) -> list:
        """
        Create a list of pipeline parameters with optional custom execution prefix.
        
        This is a helper method for external systems to easily create pipeline parameters
        with a custom PIPELINE_EXECUTION_TEMP_DIR value.
        
        Args:
            execution_s3_prefix: Custom S3 prefix for pipeline execution (optional)
            
        Returns:
            list: List of pipeline parameters
            
        Example:
            # Create parameters with custom execution prefix
            params = BasePipeline.create_pipeline_parameters("s3://my-bucket/custom-path")
            
            # Use in pipeline initialization
            pipeline = XGBoostE2EComprehensivePipeline(
                config_path="config.json",
                pipeline_parameters=params
            )
        """
        from sagemaker.workflow.parameters import ParameterString
        
        # Create custom execution parameter if provided
        if execution_s3_prefix:
            custom_execution_param = ParameterString(
                name="EXECUTION_S3_PREFIX", 
                default_value=execution_s3_prefix
            )
            pipeline_parameters = [
                custom_execution_param,
                KMS_ENCRYPTION_KEY_PARAM,
                SECURITY_GROUP_ID,
                VPC_SUBNET,
            ]
        else:
            # Use default parameters
            pipeline_parameters = [
                PIPELINE_EXECUTION_TEMP_DIR,
                KMS_ENCRYPTION_KEY_PARAM,
                SECURITY_GROUP_ID,
                VPC_SUBNET,
            ]
        
        return pipeline_parameters
