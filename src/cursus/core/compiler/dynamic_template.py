"""
Dynamic Pipeline Template for the Pipeline API.

This module provides a dynamic implementation of PipelineTemplateBase that can work
with any PipelineDAG structure without requiring custom template classes.
"""

from typing import Dict, Type, Any, Optional, List, TYPE_CHECKING
import logging

from sagemaker.workflow.parameters import ParameterString
from sagemaker.network import NetworkConfig

from ...api.dag.base_dag import PipelineDAG
from ..base import StepBuilderBase, BasePipelineConfig

# Import PipelineTemplateBase directly - circular import should be resolved by now
from ..assembler.pipeline_template_base import PipelineTemplateBase

from .config_resolver import StepConfigResolver
from ...steps.registry.builder_registry import StepBuilderRegistry
from .validation import ValidationEngine
from .exceptions import ConfigurationError, ValidationError
from ...steps.registry.exceptions import RegistryError

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
    logger.warning("Could not import constants from mods_workflow_core, using local definitions")
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
        builder_registry: Optional[StepBuilderRegistry] = None,
        skip_validation: bool = False,
        **kwargs
    ):
        """
        Initialize dynamic template.
        
        Args:
            dag: PipelineDAG instance defining pipeline structure
            config_path: Path to configuration file
            config_resolver: Custom config resolver (optional)
            builder_registry: Custom builder registry (optional)
            **kwargs: Additional arguments for base template
        """
        # Initialize logger first so it's available in all methods
        self.logger = logging.getLogger(__name__)
        
        self._dag = dag
        self._config_resolver = config_resolver or StepConfigResolver()
        self._builder_registry = builder_registry or StepBuilderRegistry()
        self._validation_engine = ValidationEngine()
        
        # Store config_path as an instance attribute so it's available to _detect_config_classes
        self.config_path = config_path
        
        # Store if validation should be skipped (for testing purposes)
        self._skip_validation = skip_validation
        
        # Auto-detect required config classes based on DAG nodes
        # Don't set instance attribute - set class attribute before calling parent constructor
        cls = self.__class__
        if not cls.CONFIG_CLASSES:  # Only set if not already set (to avoid overwriting in instance reuse)
            cls.CONFIG_CLASSES = self._detect_config_classes()
        
        # Store resolved mappings for later use
        self._resolved_config_map = None
        self._resolved_builder_map = None
        self._loaded_metadata = None  # Store metadata from loaded configs
        
        # Call parent constructor AFTER setting CONFIG_CLASSES
        super().__init__(
            config_path=config_path,
            sagemaker_session=kwargs.get('sagemaker_session'),
            role=kwargs.get('role'),
            notebook_root=kwargs.get('notebook_root'),
            registry_manager=kwargs.get('registry_manager'),
            dependency_resolver=kwargs.get('dependency_resolver')
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
        self.logger.debug(f"Detected {len(detected_classes)} required config classes from configuration file")
        
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
        if self._resolved_config_map is not None:
            return self._resolved_config_map
        
        try:
            dag_nodes = list(self._dag.nodes)
            self.logger.info(f"Resolving {len(dag_nodes)} DAG nodes to configurations")
            
            # Extract metadata from loaded configurations if available
            if self._loaded_metadata is None and hasattr(self, 'loaded_config_data'):
                if isinstance(self.loaded_config_data, dict) and 'metadata' in self.loaded_config_data:
                    self._loaded_metadata = self.loaded_config_data['metadata']
                    self.logger.info(f"Using metadata from loaded configuration")
            
            # Use the config resolver to map nodes to configs
            self._resolved_config_map = self._config_resolver.resolve_config_map(
                dag_nodes=dag_nodes,
                available_configs=self.configs,
                metadata=self._loaded_metadata
            )
            
            self.logger.info(f"Successfully resolved all {len(self._resolved_config_map)} nodes")
            
            # Log resolution details
            for node, config in self._resolved_config_map.items():
                config_type = type(config).__name__
                job_type = getattr(config, 'job_type', 'N/A')
                self.logger.debug(f"  {node} → {config_type} (job_type: {job_type})")
            
            return self._resolved_config_map
            
        except Exception as e:
            self.logger.error(f"Failed to resolve DAG nodes to configurations: {e}")
            raise ConfigurationError(f"Configuration resolution failed: {e}")
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Auto-map step types to builders using registry.
        
        Uses StepBuilderRegistry to map configuration types to their
        corresponding step builder classes.
        
        Returns:
            Dictionary mapping step types to step builder classes
            
        Raises:
            RegistryError: If step builders cannot be found for config types
        """
        if self._resolved_builder_map is not None:
            return self._resolved_builder_map
        
        try:
            # Get the complete builder registry
            self._resolved_builder_map = self._builder_registry.get_builder_map()
            
            self.logger.info(f"Using {len(self._resolved_builder_map)} registered step builders")
            
            # Validate that all required builders are available
            config_map = self._create_config_map()
            missing_builders = []
            
            for node, config in config_map.items():
                try:
                    # Pass the node name to the registry for better resolution
                    builder_class = self._builder_registry.get_builder_for_config(config, node_name=node)
                    step_type = self._builder_registry._config_class_to_step_type(
                        type(config).__name__, node_name=node, job_type=getattr(config, 'job_type', None))
                    self.logger.debug(f"  {step_type} → {builder_class.__name__}")
                except RegistryError as e:
                    missing_builders.append(f"{node} ({type(config).__name__})")
            
            if missing_builders:
                available_builders = list(self._resolved_builder_map.keys())
                raise RegistryError(
                    f"Missing step builders for {len(missing_builders)} configurations",
                    unresolvable_types=missing_builders,
                    available_builders=available_builders
                )
            
            return self._resolved_builder_map
            
        except Exception as e:
            self.logger.error(f"Failed to create step builder map: {e}")
            raise RegistryError(f"Step builder mapping failed: {e}")
    
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
                builder_registry=builder_map
            )
            
            if not validation_result.is_valid:
                self.logger.error("Configuration validation failed")
                self.logger.error(validation_result.detailed_report())
                raise ValidationError(
                    "Dynamic pipeline configuration validation failed",
                    validation_errors={
                        'missing_configs': validation_result.missing_configs,
                        'unresolvable_builders': validation_result.unresolvable_builders,
                        'config_errors': validation_result.config_errors,
                        'dependency_issues': validation_result.dependency_issues
                    }
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
                metadata=self._loaded_metadata
            )
            
            # Convert to display format
            preview = {
                'nodes': len(dag_nodes),
                'resolutions': {}
            }
            
            for node, candidates in preview_data.items():
                if candidates:
                    best_candidate = candidates[0]
                    preview['resolutions'][node] = {
                        'config_type': best_candidate['config_type'],
                        'confidence': best_candidate['confidence'],
                        'method': best_candidate['method'],
                        'job_type': best_candidate['job_type'],
                        'alternatives': len(candidates) - 1
                    }
                else:
                    preview['resolutions'][node] = {
                        'config_type': 'UNRESOLVED',
                        'confidence': 0.0,
                        'method': 'none',
                        'job_type': 'N/A',
                        'alternatives': 0
                    }
            
            return preview
            
        except Exception as e:
            self.logger.error(f"Failed to generate resolution preview: {e}")
            return {'error': str(e)}
            
    def _store_pipeline_metadata(self, assembler: "PipelineAssembler") -> None:
        """
        Store pipeline metadata from template.
        
        This method dynamically discovers and stores metadata from various step types,
        particularly focused on Cradle data loading requests and registration step configurations
        for use in filling execution documents.
        
        Args:
            assembler: PipelineAssembler instance
        """
        # Store Cradle data loading requests if available
        if hasattr(assembler, 'cradle_loading_requests'):
            self.pipeline_metadata['cradle_loading_requests'] = assembler.cradle_loading_requests
            self.logger.info(f"Stored {len(assembler.cradle_loading_requests)} Cradle loading requests")
            
        # Find and store registration steps and configurations
        try:
            # Find all registration steps in the pipeline
            registration_steps = []
            
            # Approach 1: Check step instances dictionary if available
            if hasattr(assembler, 'step_instances'):
                for step_name, step_instance in assembler.step_instances.items():
                    # Check for registration step using name pattern or type
                    if ("registration" in step_name.lower() or 
                        "registration" in str(type(step_instance)).lower()):
                        registration_steps.append(step_instance)
                        self.logger.info(f"Found registration step: {step_name}")
                        
            # Approach 2: Check steps dictionary if available
            elif hasattr(assembler, 'steps'):
                for step_name, step in assembler.steps.items():
                    # Check for registration step using name pattern
                    if "registration" in step_name.lower():
                        registration_steps.append(step)
                        self.logger.info(f"Found registration step: {step_name}")
            
            # If registration steps found, process and store configurations
            if registration_steps:
                # Find registration config
                registration_cfg = None
                for _, cfg in self.configs.items():
                    cfg_type_name = type(cfg).__name__.lower()
                    if "registration" in cfg_type_name and not "payload" in cfg_type_name:
                        registration_cfg = cfg
                        break
                
                if registration_cfg:
                    # Try to get image URI
                    try:
                        from sagemaker.image_uris import retrieve
                        
                        # Check if we have all required framework attributes
                        if all(hasattr(registration_cfg, attr) for attr in 
                               ['framework', 'aws_region', 'framework_version', 'py_version', 'inference_instance_type']):
                            try:
                                image_uri = retrieve(
                                    framework=registration_cfg.framework,
                                    region=registration_cfg.aws_region,
                                    version=registration_cfg.framework_version,
                                    py_version=registration_cfg.py_version,
                                    instance_type=registration_cfg.inference_instance_type,
                                    image_scope="inference"
                                )
                                self.logger.info(f"Retrieved image URI: {image_uri}")
                            except Exception as e:
                                self.logger.warning(f"Could not retrieve image URI: {e}")
                                image_uri = "image-uri-placeholder"  # Use placeholder
                        else:
                            self.logger.warning("Registration config missing framework attributes")
                            image_uri = "image-uri-placeholder"
                            
                        # Create execution document config
                        exec_config = self._create_execution_doc_config(image_uri)
                        
                        # Store configs for all registration steps found
                        registration_configs = {}
                        for step in registration_steps:
                            if hasattr(step, 'name'):
                                registration_configs[step.name] = exec_config
                                self.logger.info(f"Stored execution doc config for registration step: {step.name}")
                            elif isinstance(step, dict):
                                # Handle case where step might be a dict of steps
                                for name, s in step.items():
                                    if hasattr(s, 'name'):
                                        registration_configs[s.name] = exec_config
                                        self.logger.info(f"Stored execution doc config for registration step: {s.name}")
                        
                        # Store in pipeline metadata
                        self.pipeline_metadata['registration_configs'] = registration_configs
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create execution doc configs: {e}")
                        
                # Try to extract model name from registration steps
                for step in registration_steps:
                    if hasattr(step, 'model_name'):
                        self.pipeline_metadata['model_name'] = step.model_name
                        self.logger.info(f"Stored model name: {step.model_name}")
                        break
            
        except Exception as e:
            self.logger.warning(f"Error while processing registration steps: {e}")
            
        # Log property reference handling for debugging
        if hasattr(assembler, 'steps'):
            property_ref_count = 0
            for step_name, step in assembler.steps.items():
                # Check if step has inputs that might be property references
                if hasattr(step, 'inputs') and step.inputs:
                    for input_item in step.inputs:
                        if hasattr(input_item, 'source') and not isinstance(input_item.source, str):
                            property_ref_count += 1
            
            if property_ref_count > 0:
                self.logger.info(f"Pipeline contains {property_ref_count} property references that benefit from automatic handling")
    
    def get_builder_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the builder registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return self._builder_registry.get_registry_stats()
    
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
    
    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """
        Get pipeline parameters.
        
        Returns standard parameters used by most pipelines:
        - PIPELINE_EXECUTION_TEMP_DIR: S3 prefix for execution data
        - KMS_ENCRYPTION_KEY_PARAM: KMS key for encryption
        - SECURITY_GROUP_ID: Security group for network isolation
        - VPC_SUBNET: VPC subnet for network isolation
        
        Returns:
            List of pipeline parameters
        """
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]
            
    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method populates the execution document with:
        1. Cradle data loading requests (if present in the pipeline)
        2. Registration configurations (if present in the pipeline)
        
        Args:
            execution_document: Execution document to fill
            
        Returns:
            Updated execution document
        """
        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            self.logger.warning("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
            return execution_document
        
        pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

        # 1. Handle Cradle data loading requests
        self._fill_cradle_configurations(pipeline_configs)
        
        # 2. Handle Registration configurations
        self._fill_registration_configurations(pipeline_configs)
        
        return execution_document
    
    def _fill_cradle_configurations(self, pipeline_configs: Dict[str, Any]) -> None:
        """
        Fill Cradle data loading configurations in the execution document.
        
        Args:
            pipeline_configs: Dictionary of pipeline step configurations
        """
        cradle_requests = self.pipeline_metadata.get('cradle_loading_requests', {})
        
        if not cradle_requests:
            self.logger.debug("No Cradle loading requests found in metadata")
            return
            
        for step_name, request_dict in cradle_requests.items():
            if step_name not in pipeline_configs:
                self.logger.warning(f"Cradle step '{step_name}' not found in execution document")
                continue
                
            pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
            self.logger.info(f"Updated execution config for Cradle step: {step_name}")
    
    def _create_execution_doc_config(self, image_uri: str) -> Dict[str, Any]:
        """
        Create the execution document configuration dictionary.
        
        This method dynamically creates an execution document configuration
        by extracting information from registration, payload, and package configurations.
        
        Args:
            image_uri: The URI of the inference image to use
            
        Returns:
            Dictionary with execution document configuration
        """
        # Find needed configs using type name pattern matching
        registration_cfg = None
        payload_cfg = None
        package_cfg = None
        
        for _, cfg in self.configs.items():
            cfg_type_name = type(cfg).__name__.lower()
            if "registration" in cfg_type_name and not "payload" in cfg_type_name:
                registration_cfg = cfg
            elif "payload" in cfg_type_name:
                payload_cfg = cfg
            elif "package" in cfg_type_name:
                package_cfg = cfg
                
        if not registration_cfg:
            self.logger.warning("No registration configuration found for execution document")
            return {}
            
        # Create a basic configuration with required fields
        exec_config = {
            "source_model_inference_image_arn": image_uri,
        }
        
        # Add registration configuration fields
        for field in [
            "model_domain", "model_objective", "source_model_inference_content_types",
            "source_model_inference_response_types", "source_model_inference_input_variable_list",
            "source_model_inference_output_variable_list", "model_registration_region",
            "source_model_region", "aws_region", "model_owner", "region"
        ]:
            if hasattr(registration_cfg, field):
                # Map certain fields to their execution doc names
                if field == "aws_region":
                    exec_config["source_model_region"] = getattr(registration_cfg, field)
                elif field == "region":
                    exec_config["model_registration_region"] = getattr(registration_cfg, field)
                else:
                    exec_config[field] = getattr(registration_cfg, field)
        
        # Add environment variables if entry point is available
        if hasattr(registration_cfg, "inference_entry_point"):
            exec_config["source_model_environment_variable_map"] = {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": registration_cfg.inference_entry_point,
                "SAGEMAKER_REGION": getattr(registration_cfg, "aws_region", "us-east-1"),
                "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
            }
            
        # Add load testing info if payload and package configs are available
        if payload_cfg and package_cfg:
            load_testing_info = {}
            
            # Add bucket if available
            if hasattr(registration_cfg, "bucket"):
                load_testing_info["sample_payload_s3_bucket"] = registration_cfg.bucket
                
            # Add payload fields
            for field in ["sample_payload_s3_key", "expected_tps", "max_latency_in_millisecond", "max_acceptable_error_rate"]:
                if hasattr(payload_cfg, field):
                    load_testing_info[field] = getattr(payload_cfg, field)
                    
            # Add instance type
            if hasattr(package_cfg, "get_instance_type"):
                load_testing_info["instance_type_list"] = [package_cfg.get_instance_type()]
            elif hasattr(package_cfg, "processing_instance_type_small"):
                load_testing_info["instance_type_list"] = [package_cfg.processing_instance_type_small]
                
            if load_testing_info:
                exec_config["load_testing_info_map"] = load_testing_info
                
        return exec_config
        
    def _find_registration_step_nodes(self) -> List[str]:
        """
        Find nodes in the DAG that correspond to registration steps.
        
        This method uses both the resolved config map and node names to identify registration steps.
        
        Returns:
            List of node names for registration steps
        """
        # Get the resolved config map
        registration_nodes = []
        
        try:
            # Use resolved config map if available
            config_map = self._create_config_map()
            
            # Look for registration steps by config type
            for node_name, config in config_map.items():
                config_type_name = type(config).__name__.lower()
                
                # Check config type name
                if "registration" in config_type_name and not "payload" in config_type_name:
                    registration_nodes.append(node_name)
                    self.logger.info(f"Found registration step by config type: {node_name}")
                # Check node name as fallback
                elif any(pattern in node_name.lower() for pattern in ["registration", "register"]):
                    registration_nodes.append(node_name)
                    self.logger.info(f"Found registration step by name pattern: {node_name}")
        
        except Exception as e:
            self.logger.warning(f"Error finding registration nodes from config map: {e}")
            
        # If no nodes found, try using DAG nodes directly
        if not registration_nodes:
            for node in self._dag.nodes:
                if any(pattern in node.lower() for pattern in ["registration", "register"]):
                    registration_nodes.append(node)
                    self.logger.info(f"Found registration step from DAG nodes: {node}")
        
        return registration_nodes
                
    def _fill_registration_configurations(self, pipeline_configs: Dict[str, Any]) -> None:
        """
        Fill Registration configurations in the execution document.
        
        This method identifies registration steps in the DAG and updates their
        configurations in the execution document.
        
        Args:
            pipeline_configs: Dictionary of pipeline step configurations
        """
        # Find registration configs in the loaded configs
        registration_cfg = None
        payload_cfg = None
        package_cfg = None
        
        # Use the resolved config map to find registration steps
        registration_nodes = self._find_registration_step_nodes()
        if not registration_nodes:
            self.logger.debug("No registration steps found in DAG")
            return
            
        # Find registration configuration (and related configs)
        for _, cfg in self.configs.items():
            cfg_type_name = type(cfg).__name__.lower()
            if "registration" in cfg_type_name and not "payload" in cfg_type_name:
                registration_cfg = cfg
                self.logger.info(f"Found registration configuration: {type(cfg).__name__}")
            elif "payload" in cfg_type_name:
                payload_cfg = cfg
                self.logger.debug(f"Found payload configuration: {type(cfg).__name__}")
            elif "package" in cfg_type_name:
                package_cfg = cfg
                self.logger.debug(f"Found package configuration: {type(cfg).__name__}")
        
        if not registration_cfg:
            self.logger.debug("No registration configurations found")
            return
            
        # Get stored registration configs from metadata
        registration_configs = self.pipeline_metadata.get('registration_configs', {})
        
        # Generate search patterns for registration step names
        region = getattr(registration_cfg, 'region', '')
        
        search_patterns = []
        if region:
            search_patterns.extend([
                f"ModelRegistration-{region}",   # Format from error logs
                f"Registration_{region}",        # Format from template code
            ])
        
        # Add the DAG node names we found earlier
        search_patterns.extend(registration_nodes)
        
        # Always add generic fallbacks
        search_patterns.extend([
            "model_registration",            # Common generic name
            "Registration",                  # Very generic fallback
            "register_model"                 # Another common name
        ])
        
        # Search for any step name containing 'registration' as final fallback
        for step_name in pipeline_configs.keys():
            if "registration" in step_name.lower():
                if step_name not in search_patterns:
                    search_patterns.append(step_name)
        
        # Process each potential registration step
        registration_step_found = False
        for pattern in search_patterns:
            if pattern in pipeline_configs:
                # Update the execution config
                if registration_configs:
                    for step_name, config in registration_configs.items():
                        pipeline_configs[pattern]["STEP_CONFIG"] = config
                        self.logger.info(f"Updated execution config for registration step: {pattern}")
                        registration_step_found = True
                        break
                else:
                    # If no registration_configs, at least ensure STEP_CONFIG exists
                    if "STEP_CONFIG" not in pipeline_configs[pattern]:
                        pipeline_configs[pattern]["STEP_CONFIG"] = {}
                        
                    # Add STEP_TYPE if missing (MODS requirement)
                    if "STEP_TYPE" not in pipeline_configs[pattern]:
                        pipeline_configs[pattern]["STEP_TYPE"] = [
                            "PROCESSING_STEP",
                            "ModelRegistration"
                        ]
                        
                    # Try to create a config if we have registration info
                    try:
                        from sagemaker.image_uris import retrieve
                        
                        # Get image URI - try a couple of approaches
                        image_uri = None
                        
                        # Approach 1: Use framework information if available
                        if all(hasattr(registration_cfg, attr) for attr in ['framework', 'aws_region', 'framework_version', 'py_version', 'inference_instance_type']):
                            try:
                                image_uri = retrieve(
                                    framework=registration_cfg.framework,
                                    region=registration_cfg.aws_region,
                                    version=registration_cfg.framework_version,
                                    py_version=registration_cfg.py_version,
                                    instance_type=registration_cfg.inference_instance_type,
                                    image_scope="inference"
                                )
                                self.logger.info(f"Retrieved image URI: {image_uri}")
                            except Exception as e:
                                self.logger.warning(f"Could not retrieve image URI with framework info: {e}")
                        
                        # Create execution document config
                        if image_uri and self._has_required_registration_fields(registration_cfg, payload_cfg, package_cfg):
                            exec_config = self._create_execution_doc_config(image_uri)
                            pipeline_configs[pattern]["STEP_CONFIG"] = exec_config
                            self.logger.info(f"Created execution config for registration step: {pattern}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create execution doc config: {e}")
                    
                    registration_step_found = True
                    
                if registration_step_found:
                    break
                    
        # Handle model name specifically (common in all templates)
        model_name = self.pipeline_metadata.get('model_name')
        if model_name:
            for pattern in search_patterns:
                if pattern in pipeline_configs:
                    reg_config = pipeline_configs[pattern].get("STEP_CONFIG", {})
                    reg_config["MODEL_NAME"] = model_name
                    pipeline_configs[pattern]["STEP_CONFIG"] = reg_config
                    self.logger.info(f"Updated model name in registration config: {model_name}")
                    break
                    
    def _has_required_registration_fields(
        self, 
        registration_cfg: Any, 
        payload_cfg: Optional[Any] = None, 
        package_cfg: Optional[Any] = None
    ) -> bool:
        """
        Check if the registration config has all required fields for execution document.
        
        Args:
            registration_cfg: Registration configuration
            payload_cfg: Payload configuration (optional)
            package_cfg: Package configuration (optional)
            
        Returns:
            True if all required fields are present, False otherwise
        """
        # Check minimal required fields on registration config
        required_fields = [
            'model_registration_domain', 
            'model_registration_objective',
            'region'
        ]
        
        for field in required_fields:
            if not hasattr(registration_cfg, field):
                self.logger.warning(f"Registration config missing required field: {field}")
                return False
                
        return True
