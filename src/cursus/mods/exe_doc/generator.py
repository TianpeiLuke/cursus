"""
Main execution document generator class.

This module provides the core ExecutionDocumentGenerator class that orchestrates
the generation of execution documents from PipelineDAG and configuration data.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.base import BasePipelineConfig
from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter as StepConfigResolver
from .base import (
    ExecutionDocumentHelper,
    ExecutionDocumentGenerationError,
    ConfigurationNotFoundError,
    UnsupportedStepTypeError,
)
from .utils import determine_step_type, validate_execution_document_structure


logger = logging.getLogger(__name__)


class ExecutionDocumentGenerator:
    """
    Standalone execution document generator.
    
    Takes a PipelineDAG and configuration data as input, generates execution
    documents by collecting and processing step configurations independently
    from the pipeline generation system.
    """
    
    def __init__(self, 
                 config_path: str,
                 sagemaker_session: Optional[PipelineSession] = None,
                 role: Optional[str] = None,
                 config_resolver: Optional[StepConfigResolver] = None):
        """
        Initialize execution document generator.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session for AWS operations
            role: IAM role for AWS operations
            config_resolver: Custom config resolver for step name resolution
        """
        self.config_path = config_path
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.config_resolver = config_resolver or StepConfigResolver()
        self.logger = logging.getLogger(__name__)
        
        # Initialize helpers directly - no loose coupling
        from .cradle_helper import CradleDataLoadingHelper
        from .registration_helper import RegistrationHelper
        
        self.cradle_helper = CradleDataLoadingHelper()
        self.registration_helper = RegistrationHelper()
        
        # Load configurations using simplified approach
        self.configs = self._load_configs()
        
        # Keep helpers list for backward compatibility
        self.helpers: List[ExecutionDocumentHelper] = [
            self.cradle_helper,
            self.registration_helper
        ]
        
        self.logger.info(f"Initialized ExecutionDocumentGenerator with {len(self.configs)} configurations")
    
    def fill_execution_document(self, 
                              dag: PipelineDAG, 
                              execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method uses an optimized approach:
        1. First identify which steps need execution document processing
        2. Filter steps by helper type
        3. Only call helper-specific methods if relevant steps exist
        
        Args:
            dag: PipelineDAG defining the pipeline structure
            execution_document: Execution document to fill
            
        Returns:
            Updated execution document
            
        Raises:
            ExecutionDocumentGenerationError: If generation fails
        """
        self.logger.info(f"Starting execution document generation for DAG with {len(dag.nodes)} nodes")
        
        try:
            # Validate input execution document structure
            if "PIPELINE_STEP_CONFIGS" not in execution_document:
                self.logger.warning(
                    "Execution document missing 'PIPELINE_STEP_CONFIGS' key"
                )
                return execution_document

            # Step 1: Identify which steps need execution document processing
            relevant_steps = self._identify_relevant_steps(dag)
            
            if not relevant_steps:
                self.logger.info("No steps require execution document processing")
                return execution_document
            
            self.logger.info(f"Found {len(relevant_steps)} relevant steps: {relevant_steps}")
            
            pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

            # Step 2: Process cradle steps if any
            cradle_steps = self._filter_steps_by_helper(relevant_steps, self.cradle_helper)
            if cradle_steps:
                self.logger.info(f"Processing {len(cradle_steps)} cradle steps: {cradle_steps}")
                self._fill_cradle_configurations(dag, pipeline_configs)

            # Step 3: Process registration steps if any
            registration_steps = self._filter_steps_by_helper(relevant_steps, self.registration_helper)
            if registration_steps:
                self.logger.info(f"Processing {len(registration_steps)} registration steps: {registration_steps}")
                self._fill_registration_configurations(dag, pipeline_configs)

            self.logger.info("Successfully generated execution document")
            return execution_document
            
        except Exception as e:
            self.logger.error(f"Failed to generate execution document: {e}")
            raise ExecutionDocumentGenerationError(f"Execution document generation failed: {e}") from e
    
    def _load_configs(self) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations using simplified approach.
        
        Returns:
            Dictionary mapping config names to config instances
            
        Raises:
            ExecutionDocumentGenerationError: If config loading fails
        """
        try:
            from ...steps.configs.utils import load_configs
            
            # Simple config loading - let load_configs handle class discovery
            configs = load_configs(self.config_path)
            
            self.logger.info(f"Loaded {len(configs)} configurations from {self.config_path}")
            return configs
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise ExecutionDocumentGenerationError(f"Configuration loading failed: {e}") from e
    
    def _get_config_for_step(self, step_name: str) -> Optional[BasePipelineConfig]:
        """
        Get configuration for a specific step using config resolver.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Configuration for the step, or None if not found
        """
        try:
            # Use the config_resolver to map step names to configurations
            return self.config_resolver.resolve_config_for_step(step_name, self.configs)
        except Exception as e:
            self.logger.warning(f"Could not resolve config for step {step_name}: {e}")
            
            # Fallback: direct name match
            if step_name in self.configs:
                return self.configs[step_name]
            
            # Fallback: pattern matching for common naming conventions
            for config_name, config in self.configs.items():
                if self._names_match(step_name, config_name):
                    return config
            
            return None
    
    def _names_match(self, step_name: str, config_name: str) -> bool:
        """
        Check if step name and config name match using common patterns.
        
        Args:
            step_name: Name of the step
            config_name: Name of the configuration
            
        Returns:
            True if names match, False otherwise
        """
        # Normalize names by removing separators and converting to lowercase
        step_parts = set(step_name.lower().replace("_", " ").replace("-", " ").split())
        config_parts = set(config_name.lower().replace("_", " ").replace("-", " ").split())
        
        # Check for significant overlap in word parts
        common_parts = step_parts.intersection(config_parts)
        
        # Consider it a match if there's significant overlap
        # At least 50% of the smaller set should be in common
        min_parts = min(len(step_parts), len(config_parts))
        if min_parts == 0:
            return False
        
        overlap_ratio = len(common_parts) / min_parts
        return overlap_ratio >= 0.5
    
    def _identify_relevant_steps(self, dag: PipelineDAG) -> List[str]:
        """
        Identify steps in the DAG that require execution document processing.
        
        Args:
            dag: PipelineDAG instance
            
        Returns:
            List of step names that need execution document configuration
        """
        relevant_steps = []
        
        for step_name in dag.nodes:
            config = self._get_config_for_step(step_name)
            if config and self._is_execution_doc_relevant(config):
                relevant_steps.append(step_name)
                self.logger.debug(f"Step {step_name} is relevant for execution document")
        
        return relevant_steps
    
    def _is_execution_doc_relevant(self, config: BasePipelineConfig) -> bool:
        """
        Check if a configuration requires execution document processing.
        
        Args:
            config: Configuration to check
            
        Returns:
            True if config requires execution document processing, False otherwise
        """
        # Check if any helper can handle this config
        for helper in self.helpers:
            if helper.can_handle_step("", config):  # Step name not needed for this check
                return True
        
        # Fallback: check config type name for known patterns
        config_type_name = type(config).__name__.lower()
        return ("cradle" in config_type_name or 
                "registration" in config_type_name)
    
    def _filter_steps_by_helper(self, step_names: List[str], helper: ExecutionDocumentHelper) -> List[str]:
        """
        Filter steps that can be handled by a specific helper.
        
        Args:
            step_names: List of step names to filter
            helper: Helper to check against
            
        Returns:
            List of step names that can be handled by the helper
        """
        filtered_steps = []
        for step_name in step_names:
            config = self._get_config_for_step(step_name)
            if config and helper.can_handle_step(step_name, config):
                filtered_steps.append(step_name)
                self.logger.debug(f"Helper {helper.__class__.__name__} can handle step: {step_name}")
        
        return filtered_steps
    
    
    def _fill_cradle_configurations(self, dag: PipelineDAG, pipeline_configs: Dict[str, Any]) -> None:
        """
        Fill Cradle data loading configurations in the execution document.
        
        This method is ported from DynamicPipelineTemplate._fill_cradle_configurations()
        to maintain exact logic equivalence.
        
        Args:
            dag: PipelineDAG instance
            pipeline_configs: Dictionary of pipeline step configurations
        """
        # Find cradle helper to extract configurations
        cradle_helper = None
        for helper in self.helpers:
            if helper.__class__.__name__ == "CradleDataLoadingHelper":
                cradle_helper = helper
                break
        
        if not cradle_helper:
            self.logger.debug("No Cradle helper found, skipping cradle configurations")
            return
        
        # Find cradle steps in the DAG
        cradle_steps = []
        for step_name in dag.nodes:
            config = self._get_config_for_step(step_name)
            if config and cradle_helper.can_handle_step(step_name, config):
                cradle_steps.append(step_name)
        
        if not cradle_steps:
            self.logger.debug("No Cradle loading steps found in DAG")
            return
        
        # Extract configurations for each cradle step
        for step_name in cradle_steps:
            config = self._get_config_for_step(step_name)
            if config:
                # Get execution document step name using helper
                exec_step_name = cradle_helper.get_execution_step_name(step_name, config)
                
                if exec_step_name not in pipeline_configs:
                    self.logger.warning(
                        f"Cradle step '{exec_step_name}' not found in execution document"
                    )
                    continue
                
                try:
                    # Extract step configuration using the cradle helper
                    step_config = cradle_helper.extract_step_config(step_name, config)
                    pipeline_configs[exec_step_name]["STEP_CONFIG"] = step_config
                    self.logger.info(f"Updated execution config for Cradle step: {exec_step_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract cradle config for step {step_name}: {e}")
    
    def _fill_registration_configurations(self, dag: PipelineDAG, pipeline_configs: Dict[str, Any]) -> None:
        """
        Fill Registration configurations in the execution document.
        
        This method is ported from DynamicPipelineTemplate._fill_registration_configurations()
        to maintain exact logic equivalence.
        
        Args:
            dag: PipelineDAG instance
            pipeline_configs: Dictionary of pipeline step configurations
        """
        # Find registration helper to extract configurations
        registration_helper = None
        for helper in self.helpers:
            if helper.__class__.__name__ == "RegistrationHelper":
                registration_helper = helper
                break
        
        if not registration_helper:
            self.logger.debug("No Registration helper found, skipping registration configurations")
            return
        
        # Find registration configs in the loaded configs
        registration_cfg = None
        payload_cfg = None
        package_cfg = None

        # Find registration configuration (and related configs)
        for _, cfg in self.configs.items():
            cfg_type_name = type(cfg).__name__.lower()
            if "registration" in cfg_type_name and not "payload" in cfg_type_name:
                registration_cfg = cfg
                self.logger.info(
                    f"Found registration configuration: {type(cfg).__name__}"
                )
            elif "payload" in cfg_type_name:
                payload_cfg = cfg
                self.logger.debug(f"Found payload configuration: {type(cfg).__name__}")
            elif "package" in cfg_type_name:
                package_cfg = cfg
                self.logger.debug(f"Found package configuration: {type(cfg).__name__}")

        if not registration_cfg:
            self.logger.debug("No registration configurations found")
            return

        # Find registration steps in the DAG using the helper
        registration_nodes = self._find_registration_step_nodes(dag, registration_helper)
        if not registration_nodes:
            self.logger.debug("No registration steps found in DAG")
            return

        # Generate search patterns for registration step names (EXACT COPY from original)
        region = getattr(registration_cfg, "region", "")

        search_patterns = []
        if region:
            search_patterns.extend(
                [
                    f"ModelRegistration-{region}",  # Format from error logs
                    f"Registration_{region}",  # Format from template code
                ]
            )

        # Add the DAG node names we found earlier
        search_patterns.extend(registration_nodes)

        # Always add generic fallbacks
        search_patterns.extend(
            [
                "model_registration",  # Common generic name
                "Registration",  # Very generic fallback
                "register_model",  # Another common name
            ]
        )

        # Search for any step name containing 'registration' as final fallback
        for step_name in pipeline_configs.keys():
            if "registration" in step_name.lower():
                if step_name not in search_patterns:
                    search_patterns.append(step_name)

        # Process each potential registration step using execution step name transformation
        registration_step_found = False
        for pattern in search_patterns:
            # Get execution document step name using helper
            exec_step_name = registration_helper.get_execution_step_name(pattern, registration_cfg)
            
            if exec_step_name in pipeline_configs:
                # If no STEP_CONFIG, at least ensure it exists
                if "STEP_CONFIG" not in pipeline_configs[exec_step_name]:
                    pipeline_configs[exec_step_name]["STEP_CONFIG"] = {}

                # Add STEP_TYPE if missing (MODS requirement)
                if "STEP_TYPE" not in pipeline_configs[exec_step_name]:
                    pipeline_configs[exec_step_name]["STEP_TYPE"] = [
                        "PROCESSING_STEP",
                        "ModelRegistration",
                    ]

                # Try to create a config using the registration helper
                try:
                    # Use the registration helper to create execution config
                    exec_config = registration_helper.create_execution_doc_config_with_related_configs(
                        registration_cfg, payload_cfg, package_cfg
                    )
                    
                    if exec_config:
                        pipeline_configs[exec_step_name]["STEP_CONFIG"] = exec_config
                        self.logger.info(
                            f"Created execution config for registration step: {exec_step_name}"
                        )
                        registration_step_found = True

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create execution doc config: {e}"
                    )

                if registration_step_found:
                    break
    
    def _find_registration_step_nodes(self, dag: PipelineDAG, registration_helper) -> List[str]:
        """
        Find nodes in the DAG that correspond to registration steps.
        
        This method is ported from DynamicPipelineTemplate._find_registration_step_nodes()
        to maintain exact logic equivalence.
        
        Args:
            dag: PipelineDAG instance
            registration_helper: Registration helper instance
            
        Returns:
            List of node names for registration steps
        """
        registration_nodes = []

        try:
            # Look for registration steps by config type
            for node_name in dag.nodes:
                config = self._get_config_for_step(node_name)
                if config:
                    config_type_name = type(config).__name__.lower()

                    # Check config type name
                    if (
                        "registration" in config_type_name
                        and not "payload" in config_type_name
                    ):
                        registration_nodes.append(node_name)
                        self.logger.info(
                            f"Found registration step by config type: {node_name}"
                        )
                    # Check node name as fallback
                    elif any(
                        pattern in node_name.lower()
                        for pattern in ["registration", "register"]
                    ):
                        registration_nodes.append(node_name)
                        self.logger.info(
                            f"Found registration step by name pattern: {node_name}"
                        )

        except Exception as e:
            self.logger.warning(
                f"Error finding registration nodes from config map: {e}"
            )

        # If no nodes found, try using DAG nodes directly
        if not registration_nodes:
            for node in dag.nodes:
                if any(
                    pattern in node.lower() for pattern in ["registration", "register"]
                ):
                    registration_nodes.append(node)
                    self.logger.info(f"Found registration step from DAG nodes: {node}")

        return registration_nodes
