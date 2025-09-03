"""
MODS DAG Compiler for the Pipeline API.

This module provides specialized functions for compiling PipelineDAG structures
into executable SageMaker pipelines with MODS integration, solving metaclass conflicts
when applying the MODSTemplate decorator to DynamicPipelineTemplate instances.
"""

from typing import Optional, Dict, Any, Type, Tuple
import logging
from pathlib import Path

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ...core.compiler.config_resolver import StepConfigResolver
from ...registry.builder_registry import StepBuilderRegistry
from ...core.compiler.exceptions import PipelineAPIError, ConfigurationError
from ...core.base.config_base import BasePipelineConfig

# Import MODSTemplate decorator
try:
    from mods.mods_template import MODSTemplate
except ImportError:
    # If MODS is not installed, use a basic placeholder for testing
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import MODSTemplate from mods.mods_template, using placeholder")
    
    # Define a placeholder decorator that does nothing
    def MODSTemplate(author=None, description=None, version=None):
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger(__name__)


def compile_mods_dag_to_pipeline(
    dag: PipelineDAG,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Compile a PipelineDAG into a complete SageMaker Pipeline with MODS integration.
    
    This is the main entry point for users who want a simple, one-call
    compilation from DAG to MODS-compatible pipeline.
    
    Args:
        dag: PipelineDAG instance defining the pipeline structure
        config_path: Path to configuration file containing step configs
        sagemaker_session: SageMaker session for pipeline execution
        role: IAM role for pipeline execution
        pipeline_name: Optional pipeline name override
        **kwargs: Additional arguments passed to template constructor
        
    Returns:
        Generated SageMaker Pipeline ready for execution, decorated with MODSTemplate
        
    Raises:
        ValueError: If DAG nodes don't have corresponding configurations
        ConfigurationError: If configuration validation fails
        RegistryError: If step builders not found for config types
        
    Example:
        >>> dag = create_xgboost_pipeline_dag()
        >>> pipeline = compile_mods_dag_to_pipeline(
        ...     dag=dag,
        ...     config_path="configs/my_pipeline.json",
        ...     sagemaker_session=session,
        ...     role="arn:aws:iam::123456789012:role/SageMakerRole"
        ... )
        >>> pipeline.upsert()
    """
    try:
        # Validate inputs first before accessing dag.nodes
        if not isinstance(dag, PipelineDAG):
            raise ValueError("dag must be a PipelineDAG instance")
        
        if not dag.nodes:
            raise ValueError("DAG must contain at least one node")
            
        logger.info(f"Compiling DAG with {len(dag.nodes)} nodes to MODS pipeline")
        
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Create MODS compiler
        compiler = MODSPipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
        
        # Use compile method which uses our create_template method
        pipeline = compiler.compile(dag, pipeline_name=pipeline_name)
        
        logger.info(f"Successfully compiled DAG to MODS pipeline: {pipeline.name}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to compile DAG to MODS pipeline: {e}")
        raise PipelineAPIError(f"MODS DAG compilation failed: {e}") from e


class MODSPipelineDAGCompiler(PipelineDAGCompiler):
    """
    Advanced API for DAG-to-template compilation with MODS integration.
    
    This class extends PipelineDAGCompiler to enable MODS integration with
    dynamically generated pipelines, solving the metaclass conflict issue that
    occurs when trying to apply the MODSTemplate decorator to an instance of
    DynamicPipelineTemplate.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        config_resolver: Optional[StepConfigResolver] = None,
        builder_registry: Optional[StepBuilderRegistry] = None,
        **kwargs
    ):
        """
        Initialize MODS compiler with configuration and session.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session for pipeline execution
            role: IAM role for pipeline execution
            config_resolver: Custom config resolver (optional)
            builder_registry: Custom builder registry (optional)
            **kwargs: Additional arguments for template constructor
        """
        super().__init__(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            config_resolver=config_resolver,
            builder_registry=builder_registry,
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _get_base_config(self) -> BasePipelineConfig:
        """
        Extract base configuration from the configuration file.
        
        This is used to get metadata like author, version, description for MODS.
        
        Returns:
            Base configuration object
            
        Raises:
            ConfigurationError: If base configuration cannot be found or loaded
        """
        try:
            # Create a minimal DAG to test config loading
            test_dag = PipelineDAG()
            test_dag.add_node("test_node")
            
            # Use create_template from parent with skip_validation=True to just test config loading
            temp_template = super().create_template(dag=test_dag, skip_validation=True)
            
            # Try to get base config
            if hasattr(temp_template, '_get_base_config'):
                base_config = temp_template._get_base_config()
                self.logger.info(f"Found base config: {type(base_config).__name__}")
                return base_config
            elif hasattr(temp_template, 'configs') and temp_template.configs:
                # Try to find base config in configs
                for name, config in temp_template.configs.items():
                    if name.lower() == 'base' or type(config).__name__.lower() == 'basepipelineconfig':
                        self.logger.info(f"Found base config by name: {name}")
                        return config
                        
                # If no specific base config found, return first config
                first_config_name = next(iter(temp_template.configs))
                self.logger.warning(f"No specific base config found, using first config: {first_config_name}")
                return temp_template.configs[first_config_name]
            else:
                raise ConfigurationError("No base configuration found")
        
        except Exception as e:
            self.logger.error(f"Failed to get base config: {e}")
            raise ConfigurationError(f"Base configuration loading failed: {e}") from e
    
    def create_decorated_class(
        self,
        dag=None,
        author: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> Type:
        """
        Create and return the MODSTemplate decorated DynamicPipelineTemplate class.
        
        Args:
            dag: Optional pipeline DAG (not used for class creation but might be used for metadata)
            author: Author name for MODS metadata (defaults to extracting from config)
            version: Version for MODS metadata (defaults to extracting from config)
            description: Description for MODS metadata (defaults to extracting from config)
            
        Returns:
            The DynamicPipelineTemplate class decorated with MODSTemplate
        """
        try:
            # Import here to avoid circular import
            from ...core.compiler.dynamic_template import DynamicPipelineTemplate
            
            # Extract metadata from config if not provided
            if author is None or version is None or description is None:
                try:
                    base_config = self._get_base_config()
                    
                    # Use provided values or extract from base config
                    author = author or getattr(base_config, 'author', 'Unknown')
                    version = version or getattr(base_config, 'pipeline_version', '1.0.0')
                    description = description or getattr(base_config, 'pipeline_description', 'MODS Pipeline')
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract metadata from base config: {e}")
                    # Use defaults if extraction fails
                    author = author or 'Unknown'
                    version = version or '1.0.0'
                    description = description or 'MODS Pipeline'
            
            self.logger.info(f"Creating MODS decorated template class with metadata: author={author}, version={version}, description={description}")
            
            # Decorate the DynamicPipelineTemplate class with MODSTemplate
            MODSDecoratedTemplate = MODSTemplate(
                author=author,
                version=version,
                description=description
            )(DynamicPipelineTemplate)
            
            return MODSDecoratedTemplate
            
        except Exception as e:
            self.logger.error(f"Failed to create decorated class: {e}")
            raise PipelineAPIError(f"Decorated class creation failed: {e}") from e
    
    def create_template_params(self, dag: PipelineDAG, **template_kwargs) -> Dict[str, Any]:
        """
        Create and return parameters needed to instantiate a template.
        
        Args:
            dag: Pipeline DAG to compile
            **template_kwargs: Additional template parameters
            
        Returns:
            Dictionary of parameters to instantiate a template
        """
        # Merge kwargs with default values
        params = {
            "dag": dag,
            "config_path": self.config_path,
            "config_resolver": self.config_resolver,
            "builder_registry": self.builder_registry,
            "sagemaker_session": self.sagemaker_session,
            "role": self.role,
        }
        
        # Update with any other kwargs provided
        params.update(template_kwargs)
        
        return params
    
    def create_template(self, dag: PipelineDAG, **kwargs) -> Any:
        """
        Create a MODS template instance with the given DAG.
        
        This method overrides the parent method to handle MODS integration.
        
        Args:
            dag: PipelineDAG instance to create a template for
            **kwargs: Additional arguments for template
            
        Returns:
            MODS-decorated template instance ready for pipeline generation
            
        Raises:
            PipelineAPIError: If template creation fails
        """
        try:
            self.logger.info(f"Creating MODS template for DAG with {len(dag.nodes)} nodes")
            
            # Extract metadata parameters if provided in kwargs
            author = kwargs.pop('author', None)
            version = kwargs.pop('version', None)
            description = kwargs.pop('description', None)
            
            # Get the decorated class with proper parameters
            MODSDecoratedTemplate = self.create_decorated_class(
                dag=dag,
                author=author,
                version=version,
                description=description
            )
            
            # Get the template parameters
            template_params = self.create_template_params(dag, **kwargs)
            
            # Create dynamic template from the decorated class
            template = MODSDecoratedTemplate(**template_params)
            
            self.logger.info(f"Successfully created MODS template")
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to create MODS template: {e}")
            raise PipelineAPIError(f"MODS template creation failed: {e}") from e
    
    def compile_and_fill_execution_doc(
        self, 
        dag: PipelineDAG, 
        execution_doc: Dict[str, Any],
        pipeline_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Compile a DAG to MODS pipeline and fill an execution document in one step.
        
        This method ensures proper sequencing of the pipeline generation and 
        execution document filling, addressing timing issues with template metadata.
        
        Args:
            dag: PipelineDAG instance to compile
            execution_doc: Execution document template to fill
            pipeline_name: Optional pipeline name override
            **kwargs: Additional arguments for template
            
        Returns:
            Tuple of (compiled_pipeline, filled_execution_doc)
        """
        # First compile the pipeline (this also stores the template)
        pipeline = self.compile(dag, pipeline_name=pipeline_name, **kwargs)
        
        # Now use the stored template to fill the execution document
        if self._last_template is not None:
            filled_doc = self._last_template.fill_execution_document(execution_doc)
            return pipeline, filled_doc
        else:
            self.logger.warning("No template available for execution document filling")
            return pipeline, execution_doc
