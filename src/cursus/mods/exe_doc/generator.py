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
        Main entry point for filling execution documents.
        
        Args:
            dag: PipelineDAG defining the pipeline structure
            execution_document: Template execution document to fill
            
        Returns:
            Filled execution document
            
        Raises:
            ExecutionDocumentGenerationError: If generation fails
        """
        self.logger.info(f"Starting execution document generation for DAG with {len(dag.nodes)} nodes")
        
        try:
            # Validate input execution document structure
            if not validate_execution_document_structure(execution_document):
                raise ExecutionDocumentGenerationError("Invalid execution document structure")
            
            # 1. Identify relevant steps in the DAG
            relevant_steps = self._identify_relevant_steps(dag)
            self.logger.info(f"Identified {len(relevant_steps)} relevant steps for execution document")
            
            # 2. Collect configurations for relevant steps
            step_configs = self._collect_step_configurations(relevant_steps)
            self.logger.info(f"Collected configurations for {len(step_configs)} steps")
            
            # 3. Fill execution document
            filled_document = self._fill_document(execution_document, step_configs)
            
            self.logger.info("Successfully generated execution document")
            return filled_document
            
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
