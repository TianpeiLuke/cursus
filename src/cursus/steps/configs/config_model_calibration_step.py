"""
Model Calibration Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the ModelCalibration step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from typing import Optional, List, Union, Any, Dict
import json
from pathlib import Path
from pydantic import Field, model_validator, PrivateAttr

from .config_processing_step_base import ProcessingStepConfigBase
from ...core.base.hyperparameters_base import ModelHyperparameters


class ModelCalibrationConfig(ProcessingStepConfigBase):
    """
    Configuration for ModelCalibration step with self-contained derivation logic.

    This class defines the configuration parameters for the ModelCalibration step,
    which calibrates model prediction scores to accurate probabilities. Calibration
    ensures that model scores reflect true probabilities, which is crucial for
    risk-based decision-making and threshold setting.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    label_field: str = Field(description="Name of the label column")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Calibration parameters with defaults
    calibration_method: str = Field(
        default="gam",
        description="Method to use for calibration (gam, isotonic, platt)",
    )

    monotonic_constraint: bool = Field(
        default=True, description="Whether to enforce monotonicity in GAM"
    )

    gam_splines: int = Field(
        default=10, gt=0, description="Number of splines for GAM calibration"
    )

    error_threshold: float = Field(
        default=0.05, ge=0, le=1, description="Acceptable calibration error threshold"
    )

    # Multi-class support parameters with defaults
    is_binary: bool = Field(
        default=True,
        description="Whether this is a binary classification task (True) or multi-class (False)",
    )

    num_classes: int = Field(
        default=2, gt=0, description="Number of classes for classification"
    )

    score_field: str = Field(
        default="prob_class_1",
        description="Name of the score column to calibrate (for binary classification)",
    )

    score_field_prefix: str = Field(
        default="prob_class_",
        description="Prefix for probability columns in multi-class scenario",
    )

    multiclass_categories: List[Union[str, int]] = Field(
        default_factory=lambda: [0, 1],
        description="List of class names/values for multi-class calibration",
    )

    # Job type parameter for variant handling
    job_type: str = Field(
        default="calibration",
        description="Which data split to use for calibration (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    # Processing parameters - set defaults specific to calibration
    processing_entry_point: str = Field(
        default="model_calibration.py", description="Script entry point filename"
    )

    processing_source_dir: str = Field(
        default="dockers/xgboost_atoz/scripts",
        description="Directory containing the processing script",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # For now, there are no derived fields specific to model calibration beyond
    # what's inherited from the ProcessingStepConfigBase class

    model_config = ProcessingStepConfigBase.model_config

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ModelCalibrationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_config(self) -> "ModelCalibrationConfig":
        """Validate configuration and ensure defaults are set.

        Returns:
            Self: The validated configuration object

        Raises:
            ValueError: If any validation fails
        """
        # Basic validation - inherited from base class

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # Validate input/output paths in contract
        required_input_paths = ["evaluation_data"]
        for path_name in required_input_paths:
            if path_name not in contract.expected_input_paths:
                raise ValueError(
                    f"Script contract missing required input path: {path_name}"
                )

        required_output_paths = [
            "calibration_output",
            "metrics_output",
            "calibrated_data",
        ]
        for path_name in required_output_paths:
            if path_name not in contract.expected_output_paths:
                raise ValueError(
                    f"Script contract missing required output path: {path_name}"
                )

        # Validate calibration method
        valid_methods = ["gam", "isotonic", "platt"]
        if self.calibration_method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid calibration method: {self.calibration_method}. "
                f"Must be one of: {valid_methods}"
            )

        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate multi-class parameters
        if self.is_binary and self.num_classes != 2:
            raise ValueError("For binary classification, num_classes must be 2")

        if not self.is_binary and len(self.multiclass_categories) != self.num_classes:
            raise ValueError(
                f"For multi-class, multiclass_categories length ({len(self.multiclass_categories)}) must match num_classes ({self.num_classes})"
            )

        return self

    def get_script_contract(self):
        """Return the script contract for this step.

        Returns:
            ScriptContract: The contract for this step's script.
        """
        from ..contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT

        return MODEL_CALIBRATION_CONTRACT


    @classmethod
    def from_hyperparameters(
        cls,
        hyperparameters: ModelHyperparameters,
        region: str,
        pipeline_s3_loc: str,
        processing_instance_type: str = "ml.m5.xlarge",
        processing_instance_count: int = 1,
        processing_volume_size: int = 30,
        max_runtime_seconds: int = 3600,
        pipeline_name: Optional[str] = None,
        calibration_method: str = "gam",
        monotonic_constraint: bool = True,
        gam_splines: int = 10,
        error_threshold: float = 0.05,
        processing_entry_point: str = "model_calibration.py",
        processing_source_dir: str = "dockers/xgboost_atoz/pipeline_scripts",
    ) -> "ModelCalibrationConfig":
        """Create a ModelCalibrationConfig from a ModelHyperparameters instance.

        This factory method creates a calibration config using values from the provided
        hyperparameters, with options to override specific calibration parameters.

        Args:
            hyperparameters: ModelHyperparameters instance with classification settings
            region: AWS region
            pipeline_s3_loc: S3 location for pipeline artifacts
            processing_instance_type: SageMaker instance type for processing
            processing_instance_count: Number of processing instances
            processing_volume_size: EBS volume size in GB
            max_runtime_seconds: Maximum runtime in seconds
            pipeline_name: Name of the pipeline (optional)
            calibration_method: Method to use for calibration (gam, isotonic, platt)
            monotonic_constraint: Whether to enforce monotonicity in GAM
            gam_splines: Number of splines for GAM
            error_threshold: Acceptable calibration error threshold
            processing_entry_point: Script entry point filename
            processing_source_dir: Directory containing the processing script

        Returns:
            ModelCalibrationConfig: Configuration object with values from hyperparameters
        """
        return cls(
            region=region,
            pipeline_s3_loc=pipeline_s3_loc,
            processing_instance_type=processing_instance_type,
            processing_instance_count=processing_instance_count,
            processing_volume_size=processing_volume_size,
            max_runtime_seconds=max_runtime_seconds,
            pipeline_name=pipeline_name,
            calibration_method=calibration_method,
            monotonic_constraint=monotonic_constraint,
            gam_splines=gam_splines,
            error_threshold=error_threshold,
            # Values from hyperparameters
            label_field=hyperparameters.label_name,
            score_field="prob_class_1",  # Default value directly
            is_binary=hyperparameters.is_binary,
            num_classes=hyperparameters.num_classes,
            score_field_prefix="prob_class_",  # Default value directly
            multiclass_categories=hyperparameters.multiclass_categories,
            processing_entry_point=processing_entry_point,
            processing_source_dir=processing_source_dir,
        )

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing script.

        Returns:
            dict: Dictionary of environment variables to be passed to the processing script.
        """
        env = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add calibration-specific environment variables
        env.update(
            {
                "CALIBRATION_METHOD": self.calibration_method,
                "MONOTONIC_CONSTRAINT": str(self.monotonic_constraint).lower(),
                "GAM_SPLINES": str(self.gam_splines),
                "ERROR_THRESHOLD": str(self.error_threshold),
                "LABEL_FIELD": self.label_field,
                "SCORE_FIELD": self.score_field,
                "IS_BINARY": str(self.is_binary).lower(),
                "NUM_CLASSES": str(self.num_classes),
                "SCORE_FIELD_PREFIX": self.score_field_prefix,
            }
        )

        # Add multiclass categories if available and not binary
        if not self.is_binary and self.multiclass_categories:
            env["MULTICLASS_CATEGORIES"] = json.dumps(self.multiclass_categories)

        return env

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include calibration-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and calibration-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add calibration-specific fields
        calibration_fields = {
            # Tier 1 - Essential User Inputs
            "label_field": self.label_field,
            # Tier 2 - System Inputs with Defaults
            "calibration_method": self.calibration_method,
            "monotonic_constraint": self.monotonic_constraint,
            "gam_splines": self.gam_splines,
            "error_threshold": self.error_threshold,
            "is_binary": self.is_binary,
            "num_classes": self.num_classes,
            "score_field": self.score_field,
            "score_field_prefix": self.score_field_prefix,
            "job_type": self.job_type,
        }

        # Add multiclass_categories if set to non-default value
        if self.multiclass_categories != [0, 1]:
            calibration_fields["multiclass_categories"] = self.multiclass_categories

        # Combine base fields and calibration fields (calibration fields take precedence if overlap)
        init_fields = {**base_fields, **calibration_fields}

        return init_fields
