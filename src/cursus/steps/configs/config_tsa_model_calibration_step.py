"""
TSA Model Calibration Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the TSAModelCalibration step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from typing import Optional, Dict, Any
from pydantic import Field, model_validator

from .config_processing_step_base import ProcessingStepConfigBase


class TSAModelCalibrationConfig(ProcessingStepConfigBase):
    """
    Configuration for TSAModelCalibration step with self-contained derivation logic.

    This class defines the configuration parameters for the TSAModelCalibration step,
    which uses monotone B-spline calibration to convert raw TSA model prediction scores
    into well-calibrated probabilities for fraud detection. The calibration method is
    specifically designed for Time Series Analysis models with emphasis on high-score
    regions important for fraud detection.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    label_field: str = Field(
        ...,
        # default="label",
        description="Name of the ground truth label column in the evaluation dataset. "
        "Example: 'label' for is_abusive_mdr detection.",
    )

    score_field: str = Field(
        ...,
        # default="prob_class_1",
        description="Name of the raw prediction score column to calibrate. "
        "This should contain the uncalibrated model output scores. "
        "Example: 'prob_class_1' for binary classification.",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Calibration method (fixed for TSA)
    calibration_method: str = Field(
        default="bspline",
        description="Calibration method to use. Currently only 'bspline' (monotone B-spline) is supported for TSA models.",
    )

    # B-spline configuration parameters
    bspline_degree: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Degree of B-spline basis functions. Default 3 uses cubic splines for smooth calibration.",
    )

    adaptive_knots: bool = Field(
        default=True,
        description="Whether to use adaptive knot placement based on data size. "
        "True: automatically determines knot count (20-100 based on dataset size). "
        "False: uses fixed base_knots count.",
    )

    base_knots: Optional[int] = Field(
        default=None,
        ge=5,
        le=200,
        description="Fixed number of knots to use when adaptive_knots=False. "
        "If None with adaptive_knots=True, will be auto-determined (20 for <10k, 50 for 10k-1M, 100 for >1M records).",
    )

    # Quality threshold parameters
    min_records: int = Field(
        default=1000,
        ge=100,
        description="Minimum number of records required for calibration. "
        "Calibration will fail if dataset has fewer records.",
    )

    min_fraud: int = Field(
        default=10,
        ge=1,
        description="Minimum number of fraud/positive cases required for calibration. "
        "Ensures sufficient positive examples for reliable calibration.",
    )

    max_coef_threshold: float = Field(
        default=1e12,
        gt=0,
        description="Maximum acceptable coefficient magnitude for B-spline. "
        "Used to detect numerical instability in calibration.",
    )

    min_unique_values: int = Field(
        default=10,
        ge=2,
        description="Minimum number of unique calibrated predictions required. "
        "Ensures calibration provides sufficient score granularity.",
    )

    # Optimization parameters
    lambda_smooth: float = Field(
        default=1e-10,
        ge=0,
        description="Smoothness penalty for P-spline regularization. "
        "Higher values produce smoother calibration curves. Default 1e-10 provides minimal smoothing.",
    )

    max_iter: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum iterations for IRLS (Iteratively Reweighted Least Squares) optimization. "
        "Calibration will report convergence failure if this limit is reached.",
    )

    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Convergence tolerance for coefficient updates in IRLS optimization. "
        "Smaller values require tighter convergence but may increase iterations.",
    )

    # Job type parameter for variant handling
    job_type: str = Field(
        default="calibration",
        description="Which data split to use for calibration. "
        "Options: 'training', 'calibration', 'validation', 'testing'. "
        "Determines data loading strategy (nested tarball extraction vs standard loading).",
    )

    # Processing parameters - set defaults specific to TSA calibration
    processing_entry_point: str = Field(
        default="tsa_model_calibration.py",
        description="Script entry point filename for TSA calibration",
    )

    processing_source_dir: str = Field(
        default="afn_return_kickout/dockers/scripts",
        description="Directory containing the TSA calibration processing script",
    )

    # ===== Derived Fields (Tier 3) =====
    # No additional derived fields beyond what's inherited from ProcessingStepConfigBase

    model_config = ProcessingStepConfigBase.model_config

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TSAModelCalibrationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for TSA calibration

        return self

    @model_validator(mode="after")
    def validate_config(self) -> "TSAModelCalibrationConfig":
        """Validate configuration and ensure defaults are set.

        Returns:
            Self: The validated configuration object

        Raises:
            ValueError: If any validation fails
        """
        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load TSA model calibration script contract")

        # Validate input/output paths in contract
        required_input_paths = ["evaluation_data", "preprocessor_input"]
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

        # Validate calibration method (only bspline supported for TSA)
        if self.calibration_method.lower() != "bspline":
            raise ValueError(
                f"Invalid calibration method for TSA: {self.calibration_method}. "
                "Only 'bspline' is supported for TSA model calibration."
            )

        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate B-spline parameters
        if not self.adaptive_knots and self.base_knots is None:
            raise ValueError(
                "base_knots must be specified when adaptive_knots is False"
            )

        if self.bspline_degree < 1 or self.bspline_degree > 5:
            raise ValueError(
                f"bspline_degree must be between 1 and 5, got {self.bspline_degree}"
            )

        # Validate quality thresholds
        if self.min_fraud >= self.min_records:
            raise ValueError(
                f"min_fraud ({self.min_fraud}) must be less than min_records ({self.min_records})"
            )

        return self

    # get_script_contract() is inherited from BasePipelineConfig, which loads the contract from the
    # unified .step.yaml interface via the step catalog (Design B). The legacy override that imported
    # ..contracts.tsa_model_calibration_contract was removed — that module no longer exists.

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the TSA calibration processing script.

        Returns:
            dict: Dictionary of environment variables to be passed to the processing script.
        """
        env = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add TSA calibration-specific environment variables (required)
        env.update(
            {
                "CALIBRATION_METHOD": self.calibration_method,
                "LABEL_FIELD": self.label_field,
                "SCORE_FIELD": self.score_field,
            }
        )

        # Add B-spline configuration parameters (optional with defaults)
        env.update(
            {
                "BSPLINE_DEGREE": str(self.bspline_degree),
                "ADAPTIVE_KNOTS": str(self.adaptive_knots),
                "MIN_RECORDS": str(self.min_records),
                "MIN_FRAUD": str(self.min_fraud),
                "LAMBDA_SMOOTH": str(self.lambda_smooth),
                "MAX_ITER": str(self.max_iter),
                "TOLERANCE": str(self.tolerance),
                "MAX_COEF_THRESHOLD": str(self.max_coef_threshold),
                "MIN_UNIQUE_VALUES": str(self.min_unique_values),
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
            }
        )

        # Add base_knots if specified
        if self.base_knots is not None:
            env["BASE_KNOTS"] = str(self.base_knots)
        else:
            env["BASE_KNOTS"] = ""

        return env

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include TSA calibration-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and TSA calibration-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add TSA calibration-specific fields
        tsa_calibration_fields = {
            # Tier 1 - Essential User Inputs
            "label_field": self.label_field,
            "score_field": self.score_field,
            # Tier 2 - System Inputs with Defaults
            "calibration_method": self.calibration_method,
            "bspline_degree": self.bspline_degree,
            "adaptive_knots": self.adaptive_knots,
            "min_records": self.min_records,
            "min_fraud": self.min_fraud,
            "max_coef_threshold": self.max_coef_threshold,
            "min_unique_values": self.min_unique_values,
            "lambda_smooth": self.lambda_smooth,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "job_type": self.job_type,
        }

        # Add base_knots if specified (optional)
        if self.base_knots is not None:
            tsa_calibration_fields["base_knots"] = self.base_knots

        # Combine base fields and TSA calibration fields
        # (TSA calibration fields take precedence if overlap)
        init_fields = {**base_fields, **tsa_calibration_fields}

        return init_fields
