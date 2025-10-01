#!/usr/bin/env python
"""Builder for ModelMetricsComputation processing step.

This module defines the ModelMetricsComputationStepBuilder class that builds a SageMaker
ProcessingStep for model metrics computation, connecting the configuration, specification, 
and script contract.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.entities import PipelineVariable

from ...core.base.builder_base import StepBuilderBase
from ..configs.config_model_metrics_computation_step import ModelMetricsComputationConfig
from ..specs.model_metrics_computation_spec import MODEL_METRICS_COMPUTATION_SPEC

logger = logging.getLogger(__name__)


class ModelMetricsComputationStepBuilder(StepBuilderBase):
    """Builder for ModelMetricsComputation processing step.

    This class builds a SageMaker ProcessingStep that loads prediction data,
    computes comprehensive performance metrics, generates visualizations, and
    creates detailed reports. Supports both binary and multiclass classification
    with domain-specific metrics like dollar and count recall.
    """

    def __init__(
        self,
        config,
        sagemaker_session=None,
        role=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        """Initialize the ModelMetricsComputationStepBuilder.

        Args:
            config: Configuration object for this step
            sagemaker_session: SageMaker session
            role: IAM role for SageMaker execution
            registry_manager: Registry manager for steps
            dependency_resolver: Resolver for step dependencies

        Raises:
            ValueError: If config is not a ModelMetricsComputationConfig instance
        """
        if not isinstance(config, ModelMetricsComputationConfig):
            raise ValueError(
                "ModelMetricsComputationStepBuilder requires a ModelMetricsComputationConfig instance."
            )

        # Use the single specification for model metrics computation
        spec = MODEL_METRICS_COMPUTATION_SPEC
        self.log_info("Using ModelMetricsComputation specification")

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: ModelMetricsComputationConfig = config

    def validate_configuration(self) -> None:
        """Validate the provided configuration.

        This method performs comprehensive validation of all configuration parameters,
        ensuring they meet the requirements for the metrics computation step.

        Raises:
            ValueError: If any configuration validation fails
        """
        self.log_info("Validating ModelMetricsComputationConfig...")

        # Validate required attributes
        required_attrs = [
            "processing_entry_point",
            "id_name",
            "label_name",
            "processing_instance_count",
            "processing_volume_size",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [
                None,
                "",
            ]:
                raise ValueError(
                    f"ModelMetricsComputationConfig missing required attribute: {attr}"
                )


        # Validate input format
        valid_formats = {"auto", "csv", "parquet", "json"}
        if self.config.input_format not in valid_formats:
            raise ValueError(f"Invalid input_format: {self.config.input_format}")

        # Validate probability ranges
        if not 0.0 <= self.config.dollar_recall_fpr <= 1.0:
            raise ValueError(
                f"dollar_recall_fpr must be between 0.0 and 1.0, got {self.config.dollar_recall_fpr}"
            )

        if not 0.0 <= self.config.count_recall_cutoff <= 1.0:
            raise ValueError(
                f"count_recall_cutoff must be between 0.0 and 1.0, got {self.config.count_recall_cutoff}"
            )

        # Validate dollar recall configuration
        if self.config.compute_dollar_recall and not self.config.amount_field:
            self.log_warning(
                "compute_dollar_recall is enabled but amount_field is not set - "
                "dollar recall will be skipped if amount data is not available"
            )

        self.log_info("ModelMetricsComputationConfig validation succeeded.")

    def _is_pipeline_variable(self, value: Any) -> bool:
        """Check if a value is a PipelineVariable.

        Args:
            value: Value to check

        Returns:
            bool: True if the value is a PipelineVariable, False otherwise
        """
        return isinstance(value, PipelineVariable) or (
            hasattr(value, "expr") and callable(getattr(value, "expr", None))
        )

    def _detect_circular_references(
        self, var: Any, visited: Optional[Set] = None
    ) -> bool:
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
                if key == "Get":  # Skip Get references
                    continue
                if self._detect_circular_references(value, visited.copy()):
                    return True

        return False

    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor.

        This method creates a dictionary of environment variables needed by the
        metrics computation script, combining base variables with metrics-specific ones.

        Returns:
            Dict[str, str]: Environment variables dictionary
        """
        env_vars = super()._get_environment_variables()

        # Add metrics computation specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.config.id_name,
                "LABEL_FIELD": self.config.label_name,
                "INPUT_FORMAT": self.config.input_format,
                "COMPUTE_DOLLAR_RECALL": str(self.config.compute_dollar_recall).lower(),
                "COMPUTE_COUNT_RECALL": str(self.config.compute_count_recall).lower(),
                "DOLLAR_RECALL_FPR": str(self.config.dollar_recall_fpr),
                "COUNT_RECALL_CUTOFF": str(self.config.count_recall_cutoff),
                "GENERATE_PLOTS": str(self.config.generate_plots).lower(),
            }
        )

        # Add amount field if specified
        if self.config.amount_field:
            env_vars["AMOUNT_FIELD"] = self.config.amount_field

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
                    destination=container_path,
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
                from sagemaker.workflow.functions import Join
                base_output_path = self._get_base_output_path()
                destination = Join(on="/", values=[base_output_path, "model_metrics_computation", logical_name])
                self.log_info(
                    "Using generated destination for '%s': %s",
                    logical_name,
                    destination,
                )

            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination,
                )
            )

        return processing_outputs

    def _get_processor(self) -> SKLearnProcessor:
        """Create and configure the processor for this step.

        Returns:
            SKLearnProcessor: The configured processor for the step
        """
        # Get appropriate instance type based on configuration
        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        # Get framework version with fallback
        framework_version = getattr(
            self.config, "processing_framework_version", "1.2-1"
        )

        return SKLearnProcessor(
            framework_version=framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),  # Use standardized method
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.

        Returns:
            None since no arguments are needed for model metrics computation
        """
        self.log_info("No command-line arguments needed for model metrics computation script")
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
        self.log_info("Creating ModelMetricsComputation ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get("inputs", {})
        outputs = kwargs.get("outputs", {})
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

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

        # Get script path using modernized method with comprehensive fallbacks
        script_path = self.config.get_script_path()
        self.log_info("Using script path: %s", script_path)

        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

        # Attach specification to the step for future reference
        if hasattr(self, "spec") and self.spec:
            setattr(step, "_spec", self.spec)

        self.log_info("Created ProcessingStep with name: %s", step.name)
        return step
