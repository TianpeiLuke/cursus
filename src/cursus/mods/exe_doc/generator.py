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
from ...core.compiler.config_resolver import StepConfigResolver
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
        
        # Load configurations
        self.configs = self._load_configs()
        
        # Initialize helpers (will be populated in subsequent phases)
        self.helpers: List[ExecutionDocumentHelper] = []
        
        self.logger.info(f"Initialized ExecutionDocumentGenerator with {len(self.configs)} configurations")
    
    def fill_execution_document(self, 
                              dag: PipelineDAG, 
                              execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method populates the execution document with:
        1. Cradle data loading requests (if present in the pipeline)
        2. Registration configurations (if present in the pipeline)
        
        This is ported from DynamicPipelineTemplate.fill_execution_document() to maintain
        exact logic equivalence.
        
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
            # Validate input execution document structure (EXACT COPY from original logic)
            if "PIPELINE_STEP_CONFIGS" not in execution_document:
                self.logger.warning(
                    "Execution document missing 'PIPELINE_STEP_CONFIGS' key"
                )
                return execution_document

            pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

            # 1. Handle Cradle data loading requests (EXACT COPY from original)
            self._fill_cradle_configurations(dag, pipeline_configs)

            # 2. Handle Registration configurations (EXACT COPY from original)
            self._fill_registration_configurations(dag, pipeline_configs)

            self.logger.info("Successfully generated execution document")
            return execution_document
            
        except Exception as e:
            self.logger.error(f"Failed to generate execution document: {e}")
            raise ExecutionDocumentGenerationError(f"Execution document generation failed: {e}") from e
    
    def add_helper(self, helper: ExecutionDocumentHelper) -> None:
        """
        Add a helper to the generator.
        
        Args:
            helper: ExecutionDocumentHelper instance to add
        """
        self.helpers.append(helper)
        self.logger.info(f"Added helper: {helper.__class__.__name__}")
    
    def _load_configs(self) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations using existing utilities.
        
        Returns:
            Dictionary mapping config names to config instances
            
        Raises:
            ExecutionDocumentGenerationError: If config loading fails
        """
        try:
            from ...steps.configs.utils import load_configs, build_complete_config_classes
            
            # Build complete config classes - this will import and register all classes
            # from the step and hyperparameter registries
            complete_classes = build_complete_config_classes()
            
            # Check if complete_classes is empty or insufficient
            if not complete_classes or len(complete_classes) < 3:  # Should have at least base classes
                self.logger.warning(f"build_complete_config_classes returned only {len(complete_classes)} classes, importing all configs directly")
                complete_classes = self._import_all_config_classes()
            
            self.logger.info(f"Using {len(complete_classes)} config classes for loading")
            
            # Load configs using the complete class registry
            configs = load_configs(self.config_path, complete_classes)
            
            self.logger.info(f"Loaded {len(configs)} configurations from {self.config_path}")
            return configs
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise ExecutionDocumentGenerationError(f"Configuration loading failed: {e}") from e
    
    def _import_all_config_classes(self) -> Dict[str, type]:
        """
        Import and register all config classes directly as a fallback.
        Uses the registry to get the correct config class names.
        
        Returns:
            Dictionary mapping class names to class types
        """
        from ...core.config_fields import ConfigClassStore
        from ...registry.step_names import CONFIG_STEP_REGISTRY
        from ...registry import HYPERPARAMETER_REGISTRY
        
        config_classes = {}
        
        # Import base classes using class names as keys (required by load_configs)
        try:
            from ...core.base.config_base import BasePipelineConfig
            config_classes["BasePipelineConfig"] = BasePipelineConfig
            ConfigClassStore.register(BasePipelineConfig)
            
            from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase
            config_classes["ProcessingStepConfigBase"] = ProcessingStepConfigBase
            ConfigClassStore.register(ProcessingStepConfigBase)
            
            self.logger.debug("Imported base config classes")
        except ImportError as e:
            self.logger.warning(f"Could not import base config classes: {e}")
        
        # Import step config classes from CONFIG_STEP_REGISTRY
        # CONFIG_STEP_REGISTRY maps config_class_name -> step_name, but we want step_name -> class
        for config_class_name, step_name in CONFIG_STEP_REGISTRY.items():
            try:
                # Generate module name from step name (convert PascalCase to snake_case)
                module_name = f"config_{self._pascal_to_snake(step_name)}"
                
                # Import using relative import
                try:
                    module = __import__(f"...steps.configs.{module_name}", 
                                      globals(), locals(), [config_class_name], 1)
                    if hasattr(module, config_class_name):
                        cls = getattr(module, config_class_name)
                        # Use class name as key, class as value (required by load_configs)
                        config_classes[config_class_name] = cls
                        ConfigClassStore.register(cls)
                        self.logger.debug(f"Imported {config_class_name} from {module_name}")
                    else:
                        self.logger.debug(f"Module {module_name} does not have class {config_class_name}")
                except ImportError as e:
                    self.logger.debug(f"Could not import {config_class_name} from {module_name}: {e}")
                
            except Exception as e:
                self.logger.debug(f"Error importing {config_class_name}: {e}")
        
        # Note: Hyperparameter classes are not relevant for execution document generation
        # They are handled separately in the hyperparameter management system
        
        self.logger.info(f"Imported {len(config_classes)} config classes directly")
        return config_classes
    
    def _pascal_to_snake(self, pascal_str: str) -> str:
        """
        Convert PascalCase to snake_case.
        
        Args:
            pascal_str: String in PascalCase
            
        Returns:
            String in snake_case
        """
        import re
        # Insert underscore before uppercase letters (except the first one)
        snake_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', pascal_str)
        return snake_str.lower()
    
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
    
    def _collect_step_configurations(self, step_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Collect execution document configurations for relevant steps.
        
        Args:
            step_names: List of step names to process
            
        Returns:
            Dictionary mapping step names to their execution document configurations
            
        Raises:
            ConfigurationNotFoundError: If configuration cannot be found for a step
            UnsupportedStepTypeError: If step type is not supported
        """
        step_configs = {}
        
        for step_name in step_names:
            config = self._get_config_for_step(step_name)
            if not config:
                raise ConfigurationNotFoundError(f"Configuration not found for step: {step_name}")
            
            helper = self._find_helper_for_config(config)
            if not helper:
                raise UnsupportedStepTypeError(f"No helper found for step: {step_name} (config type: {type(config).__name__})")
            
            try:
                step_config = helper.extract_step_config(step_name, config)
                step_configs[step_name] = step_config
                self.logger.debug(f"Extracted config for step {step_name}")
            except Exception as e:
                self.logger.error(f"Failed to extract config for step {step_name}: {e}")
                raise ExecutionDocumentGenerationError(f"Config extraction failed for step {step_name}: {e}") from e
        
        return step_configs
    
    def _find_helper_for_config(self, config: BasePipelineConfig) -> Optional[ExecutionDocumentHelper]:
        """
        Find the appropriate helper for a configuration.
        
        Args:
            config: Configuration to find helper for
            
        Returns:
            Helper that can handle the configuration, or None if not found
        """
        for helper in self.helpers:
            if helper.can_handle_step("", config):  # Step name not needed for this check
                return helper
        
        return None
    
    def _fill_document(self, 
                      execution_document: Dict[str, Any], 
                      step_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fill execution document with collected step configurations.
        
        Args:
            execution_document: Template execution document
            step_configs: Collected step configurations
            
        Returns:
            Filled execution document
        """
        # Create a copy to avoid modifying the original
        import copy
        filled_document = copy.deepcopy(execution_document)
        
        if "PIPELINE_STEP_CONFIGS" not in filled_document:
            filled_document["PIPELINE_STEP_CONFIGS"] = {}
        
        pipeline_configs = filled_document["PIPELINE_STEP_CONFIGS"]
        
        for step_name, step_config in step_configs.items():
            if step_name not in pipeline_configs:
                pipeline_configs[step_name] = {}
            
            # Set the step configuration
            pipeline_configs[step_name]["STEP_CONFIG"] = step_config
            
            # Add STEP_TYPE if not present
            if "STEP_TYPE" not in pipeline_configs[step_name]:
                config = self._get_config_for_step(step_name)
                if config:
                    pipeline_configs[step_name]["STEP_TYPE"] = determine_step_type(step_name, config)
            
            self.logger.debug(f"Filled execution document for step: {step_name}")
        
        return filled_document
    
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
            if step_name not in pipeline_configs:
                self.logger.warning(
                    f"Cradle step '{step_name}' not found in execution document"
                )
                continue
            
            config = self._get_config_for_step(step_name)
            if config:
                try:
                    # Extract step configuration using the cradle helper
                    step_config = cradle_helper.extract_step_config(step_name, config)
                    pipeline_configs[step_name]["STEP_CONFIG"] = step_config
                    self.logger.info(f"Updated execution config for Cradle step: {step_name}")
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

        # Process each potential registration step (EXACT COPY from original)
        registration_step_found = False
        for pattern in search_patterns:
            if pattern in pipeline_configs:
                # If no STEP_CONFIG, at least ensure it exists
                if "STEP_CONFIG" not in pipeline_configs[pattern]:
                    pipeline_configs[pattern]["STEP_CONFIG"] = {}

                # Add STEP_TYPE if missing (MODS requirement)
                if "STEP_TYPE" not in pipeline_configs[pattern]:
                    pipeline_configs[pattern]["STEP_TYPE"] = [
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
                        pipeline_configs[pattern]["STEP_CONFIG"] = exec_config
                        self.logger.info(
                            f"Created execution config for registration step: {pattern}"
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
