#!/usr/bin/env python
"""Builder for ModelCalibration processing step.

This module defines the ModelCalibrationStepBuilder class that builds a SageMaker
ProcessingStep for model calibration, connecting the configuration, specification, 
and script contract.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.entities import PipelineVariable

from ...core.base.builder_base import StepBuilderBase
from ..configs.config_model_calibration_step import ModelCalibrationConfig
from ..specs.model_calibration_spec import MODEL_CALIBRATION_SPEC
from ..registry.builder_registry import register_builder

logger = logging.getLogger(__name__)

@register_builder()
class ModelCalibrationStepBuilder(StepBuilderBase):
    """Builder for ModelCalibration processing step.
    
    This class builds a SageMaker ProcessingStep that calibrates model prediction
    scores to accurate probabilities. Calibration is essential for ensuring that
    prediction scores reflect true probabilities, which is crucial for reliable
    decision-making based on model outputs.
    """
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        """Initialize the ModelCalibrationStepBuilder.
        
        Args:
            config: Configuration object for this step
            sagemaker_session: SageMaker session
            role: IAM role for SageMaker execution
            notebook_root: Root directory for notebooks
            registry_manager: Registry manager for steps
            dependency_resolver: Resolver for step dependencies
            
        Raises:
            ValueError: If config is not a ModelCalibrationConfig instance
        """
        if not isinstance(config, ModelCalibrationConfig):
            raise ValueError("ModelCalibrationStepBuilder requires a ModelCalibrationConfig instance.")
            
        super().__init__(
            config=config,
            spec=MODEL_CALIBRATION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: ModelCalibrationConfig = config
        
    def validate_configuration(self) -> None:
        """Validate the provided configuration.
        
        This method performs comprehensive validation of all configuration parameters,
        ensuring they meet the requirements for the calibration step.
        
        Raises:
            ValueError: If any configuration validation fails
        """
        self.log_info("Validating ModelCalibrationConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count',
            'processing_volume_size',
            'calibration_method',
            'label_field',
            'score_field',
            'is_binary'  # Add required is_binary field
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelCalibrationConfig missing required attribute: {attr}")
        
        # Validate calibration method
        valid_methods = ['gam', 'isotonic', 'platt']
        if self.config.calibration_method.lower() not in valid_methods:
            raise ValueError(f"Invalid calibration method: {self.config.calibration_method}. "
                            f"Must be one of: {valid_methods}")
        
        # Validate numeric parameters
        if self.config.gam_splines <= 0:
            raise ValueError(f"gam_splines must be > 0, got {self.config.gam_splines}")
            
        if not 0 <= self.config.error_threshold <= 1:
            raise ValueError(f"error_threshold must be between 0 and 1, got {self.config.error_threshold}")
            
        # Check if script exists if notebook_root is provided
        if self.notebook_root:
            script_dir = Path(self.notebook_root) / self.config.processing_source_dir
            script_path = script_dir / self.config.processing_entry_point
            if not script_path.exists() and not self._is_pipeline_variable(script_path):
                self.log_warning(f"Script not found at {script_path}. It should exist before pipeline execution.")
            
        self.log_info("ModelCalibrationConfig validation succeeded.")
    
    def _is_pipeline_variable(self, value: Any) -> bool:
        """Check if a value is a PipelineVariable.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if the value is a PipelineVariable, False otherwise
        """
        return isinstance(value, PipelineVariable) or (hasattr(value, "expr") and callable(getattr(value, "expr", None)))
    
    def _normalize_s3_uri(self, uri: Union[str, PipelineVariable]) -> Union[str, PipelineVariable]:
        """Normalize S3 URI, handling PipelineVariable objects.
        
        This method handles both regular strings and PipelineVariable objects,
        providing a consistent interface for S3 URI handling throughout the builder.
        
        Args:
            uri: The S3 URI or PipelineVariable to normalize
            
        Returns:
            Union[str, PipelineVariable]: The normalized URI
            
        Raises:
            TypeError: If uri is not a string or PipelineVariable
        """
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            return uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(uri):
            return uri
            
        if not isinstance(uri, str):
            raise TypeError(f"Expected string or PipelineVariable, got {type(uri)}")
            
        # Normalize string URI
        return uri
    
    def _get_s3_directory_path(self, s3_uri: Union[str, PipelineVariable, dict]) -> Union[str, PipelineVariable, dict]:
        """Ensure S3 URI is a directory path (ends with '/').
        
        This method is important for ensuring consistent directory path handling
        when working with S3 URIs in the pipeline.
        
        Args:
            s3_uri: The S3 URI to process
            
        Returns:
            Union[str, PipelineVariable, dict]: The normalized URI with trailing slash
        """
        # Handle Pipeline step references with Get key
        if isinstance(s3_uri, dict) and 'Get' in s3_uri:
            return s3_uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(s3_uri):
            # We can't modify PipelineVariable directly, so return as is
            # The processing logic should handle directory paths appropriately
            return s3_uri
            
        # Normalize string URI
        normalized_uri = str(s3_uri)
        if not normalized_uri.endswith('/'):
            normalized_uri += '/'
            
        return normalized_uri
    
    def _validate_s3_uri(self, uri: Union[str, PipelineVariable, dict]) -> Union[str, PipelineVariable, dict]:
        """Validate that a given URI is a valid S3 URI.
        
        This method ensures that S3 URIs are properly formatted to prevent errors
        during pipeline execution.
        
        Args:
            uri: The URI to validate
            
        Returns:
            Union[str, PipelineVariable, dict]: The validated URI
            
        Raises:
            ValueError: If URI doesn't start with s3:// and is not a PipelineVariable or Get reference
        """
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            return uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(uri):
            return uri
            
        # Validate string URI
        normalized_uri = str(uri)
        if not normalized_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")
            
        return uri
    
    def _detect_circular_references(self, var: Any, visited: Optional[Set] = None) -> bool:
        """Detect circular references in PipelineVariable objects.
        
        This method checks for circular references that could cause infinite recursion
        or other issues during pipeline execution.
        
        Args:
            var: The variable to check
            visited: Set of already visited objects (used for recursion)
            
        Returns:
            bool: True if a circular reference is detected, False otherwise
        """
        if visited is None:
            visited = set()
            
        if id(var) in visited:
            return True
        
        if self._is_pipeline_variable(var):
            visited.add(id(var))
            # Check for circular references in any dependent variables
            for dep in getattr(var, "_dependencies", []):
                if self._detect_circular_references(dep, visited):
                    return True
                    
        # For dictionaries, check values for circular references
        elif isinstance(var, dict):
            for key, value in var.items():
                if key == 'Get':  # Skip Get references
                    continue
                if self._detect_circular_references(value, visited.copy()):
                    return True
        
        return False
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor.
        
        This method creates a dictionary of environment variables needed by the
        calibration script, combining base variables with calibration-specific ones.
        
        Returns:
            Dict[str, str]: Environment variables dictionary
        """
        env_vars = super()._get_environment_variables()
        
        # Add calibration-specific environment variables
        env_vars.update({
            "CALIBRATION_METHOD": self.config.calibration_method.lower(),
            "LABEL_FIELD": self.config.label_field,
            "SCORE_FIELD": self.config.score_field,
            "MONOTONIC_CONSTRAINT": str(self.config.monotonic_constraint).lower(),
            "GAM_SPLINES": str(self.config.gam_splines),
            "ERROR_THRESHOLD": str(self.config.error_threshold),
            # Add multi-class parameters
            "IS_BINARY": str(self.config.is_binary).lower(),
            "NUM_CLASSES": str(self.config.num_classes),
            "SCORE_FIELD_PREFIX": self.config.score_field_prefix
        })
        
        # Add multiclass categories if available
        if not self.config.is_binary and self.config.multiclass_categories:
            import json
            env_vars["MULTICLASS_CATEGORIES"] = json.dumps(self.config.multiclass_categories)
        
        return env_vars
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract.
        
        This method maps logical input names from the step specification to
        SageMaker ProcessingInput objects required by the processing script.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            List[ProcessingInput]: List of configured ProcessingInput objects
            
        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
        
        # Check for circular references in PipelineVariable inputs
        for input_name, input_value in inputs.items():
            if self._detect_circular_references(input_value):
                raise ValueError(f"Circular reference detected in input '{input_name}'")
        
        # Removed special handling for XGBoost training outputs - this is now handled in the calibration script
            
        processing_inputs = []
        
        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue
                
            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")
                
            # Use the input value directly - property references are handled by PipelineAssembler
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path
                )
            )
            
        return processing_inputs
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract.
        
        This method maps logical output names from the step specification to
        SageMaker ProcessingOutput objects that will be produced by the processing script.
        
        Args:
            outputs: Dictionary of output values
            
        Returns:
            List[ProcessingOutput]: List of configured ProcessingOutput objects
            
        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        processing_outputs = []
        
        # Process each output in the specification
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
            else:
                raise ValueError(f"No container path found for output: {logical_name}")
                
            # Try to find destination in outputs
            destination = None
            
            # Look in outputs by logical name
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from config
                destination = f"{self.config.pipeline_s3_loc}/model_calibration/{logical_name}"
                self.log_info("Using generated destination for '%s': %s", logical_name, destination)
            
            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination
                )
            )
            
        return processing_outputs
    
    def _get_processor(self) -> SKLearnProcessor:
        """Create and configure the processor for this step.
        
        Returns:
            SKLearnProcessor: The configured processor for the step
        """
        # Get appropriate instance type based on configuration
        instance_type = self.config.processing_instance_type_large if self.config.use_large_processing_instance else self.config.processing_instance_type_small
        
        # Get framework version with fallback
        framework_version = getattr(self.config, 'processing_framework_version', "1.0-1")
        
        return SKLearnProcessor(
            framework_version=framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),  # Use standardized method
            sagemaker_session=self.session,
            env=self._get_environment_variables()
        )
    
    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Returns None as job arguments since the calibration script now uses
        standard paths defined directly in the script.
        
        Returns:
            None since no arguments are needed
        """
        self.log_info("No command-line arguments needed for calibration script")
        return None
        
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        self.log_info("Creating ModelCalibration ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs
        inputs = {}
        
        # If dependencies are provided, extract inputs from them
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
                
        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)
        
        # Create processor and get inputs/outputs
        processor = self._get_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name using standardized method with auto-detection
        step_name = self._get_step_name()
        
        # Get full script path from config or contract
        script_path = self.config.get_script_path()
        if not script_path and self.contract:
            script_path = self.contract.entry_point
        
        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification to the step for future reference
        if hasattr(self, 'spec') and self.spec:
            setattr(step, '_spec', self.spec)
            
        self.log_info("Created ProcessingStep with name: %s", step.name)
        return step
