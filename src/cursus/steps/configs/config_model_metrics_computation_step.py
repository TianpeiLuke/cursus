"""
Model Metrics Computation Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the model metrics computation step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, field_validator
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.model_metrics_computation_contract import (
    MODEL_METRICS_COMPUTATION_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class ModelMetricsComputationConfig(ProcessingStepConfigBase):
    """
    Configuration for model metrics computation step with self-contained derivation logic.

    This class defines the configuration parameters for the model metrics computation step,
    which loads prediction data, computes comprehensive performance metrics, generates
    visualizations, and creates detailed reports. Supports both binary and multiclass
    classification with domain-specific metrics like dollar and count recall.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    # At least one of score_field or score_fields must be provided

    id_name: str = Field(
        ...,
        description="Name of the ID field in the prediction data (required for metrics computation).",
    )

    label_name: str = Field(
        ...,
        description="Name of the main label column (REQUIRED). "
        "For single-task mode: this is the only label field used. "
        "For multi-task mode: this represents the main task label field. "
        "Additional task labels are specified via task_label_names.",
    )

    score_field: Optional[str] = Field(
        default=None,
        description="Name of the score column to evaluate (single-task mode). "
        "Use this for backward compatibility or when evaluating a single score field.",
    )

    score_fields: Optional[List[str]] = Field(
        default=None,
        description="List of score column names to evaluate (multi-task mode). "
        "Use this when evaluating multiple score fields independently. "
        "If both score_field and score_fields are provided, score_fields takes precedence. "
        "Example: ['task1_prob', 'task2_prob', 'task3_prob']",
    )

    task_label_names: Optional[List[str]] = Field(
        default=None,
        description="List of task label field names for multi-task mode (one per task). "
        "REQUIRED when score_fields is provided (multi-task mode). "
        "Must match the length of score_fields. "
        "If not provided, labels will be inferred by removing '_prob' suffix from score field names. "
        "Example: score_fields=['task1_prob', 'task2_prob'], "
        "task_label_names=['task1_true', 'task2_true']",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="model_metrics_computation.py",
        description="Entry point script for model metrics computation.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    amount_field: Optional[str] = Field(
        default="order_amount",
        description="Name of the amount field for dollar recall computation (optional).",
    )

    input_format: str = Field(
        default="auto",
        description="Preferred input format for prediction data (auto, csv, parquet, json).",
    )

    # Computation control flags
    compute_dollar_recall: bool = Field(
        default=True,
        description="Enable dollar recall computation (requires amount_field).",
    )

    compute_count_recall: bool = Field(
        default=True,
        description="Enable count recall computation.",
    )

    generate_plots: bool = Field(
        default=True,
        description="Enable generation of performance visualization plots.",
    )

    # Metric computation parameters
    dollar_recall_fpr: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="False positive rate for dollar recall computation.",
    )

    count_recall_cutoff: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Cutoff percentile for count recall computation.",
    )

    # Processing framework - metrics computation uses scikit-learn
    processing_framework_version: str = Field(
        default="1.2-1",  # Python 3.8 compatible version
        description="Scikit-learn framework version for processing (metrics computation uses sklearn)",
    )

    # For metrics computation, we typically use smaller instances
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing (metrics computation typically needs less resources)",
    )

    # Model comparison configuration (Tier 2 - Optional with defaults)
    comparison_mode: bool = Field(
        default=False,
        description="Enable model comparison functionality to compare with previous model scores (single-task mode)",
    )

    previous_score_field: str = Field(
        default="",
        description="Name of the column containing previous model scores for comparison (single-task mode, required when comparison_mode=True)",
    )

    previous_score_fields: Optional[List[str]] = Field(
        default=None,
        description="List of columns containing previous model scores for multi-task comparison (multi-task mode). "
        "Must match the length of score_fields when provided. "
        "Example: ['task1_prev_prob', 'task2_prev_prob']",
    )

    comparison_metrics: str = Field(
        default="all",
        description="Comparison metrics to compute: 'all' for comprehensive metrics, 'basic' for essential metrics only",
    )

    statistical_tests: bool = Field(
        default=True,
        description="Enable statistical significance tests (McNemar's test, paired t-test, Wilcoxon test)",
    )

    comparison_plots: bool = Field(
        default=True,
        description="Enable comparison visualizations (side-by-side ROC/PR curves, scatter plots, distributions)",
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Currently no derived fields specific to model metrics computation
    # beyond what's inherited from the ProcessingStepConfigBase class

    # Field validators

    @field_validator("input_format")
    @classmethod
    def validate_input_format(cls, v: str) -> str:
        """Validate input format is supported."""
        valid_formats = {"auto", "csv", "parquet", "json"}
        if v.lower() not in valid_formats:
            raise ValueError(f"input_format must be one of {valid_formats}, got '{v}'")
        return v.lower()

    @field_validator("dollar_recall_fpr", "count_recall_cutoff")
    @classmethod
    def validate_probability_range(cls, v: float) -> float:
        """Validate probability values are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0, got {v}")
        return v

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ModelMetricsComputationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_metrics_computation_config(self) -> "ModelMetricsComputationConfig":
        """Additional validation specific to metrics computation configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError(
                "metrics computation step requires a processing_entry_point"
            )

        # Validate required fields from script contract
        if not self.id_name:
            raise ValueError(
                "id_name must be provided (required by model metrics computation contract)"
            )

        if not self.label_name:
            raise ValueError(
                "label_name must be provided (required for both single-task and multi-task modes)"
            )

        # Determine if we're in single-task or multi-task mode
        is_multitask = bool(self.score_fields)
        is_singletask = bool(self.score_field) and not is_multitask

        # Validate that at least one of score_field or score_fields is provided
        if not self.score_field and not self.score_fields:
            raise ValueError(
                "At least one of 'score_field' (single-task) or 'score_fields' (multi-task) must be provided"
            )

        # Validate score_fields if provided (multi-task mode)
        if self.score_fields:
            if not isinstance(self.score_fields, list):
                raise ValueError("score_fields must be a list of strings")

            if len(self.score_fields) == 0:
                raise ValueError("score_fields cannot be empty")

            # For multi-task: task_label_names is optional (can be inferred)
            # but if provided, must match score_fields length
            if self.task_label_names is not None:
                if not isinstance(self.task_label_names, list):
                    raise ValueError("task_label_names must be a list of strings")

                if len(self.task_label_names) == 0:
                    raise ValueError("task_label_names cannot be empty")

                if len(self.task_label_names) != len(self.score_fields):
                    raise ValueError(
                        f"task_label_names count ({len(self.task_label_names)}) must match "
                        f"score_fields count ({len(self.score_fields)})"
                    )

            # Validate previous_score_fields if provided (multi-task comparison)
            if self.previous_score_fields is not None:
                if not isinstance(self.previous_score_fields, list):
                    raise ValueError("previous_score_fields must be a list of strings")

                if len(self.previous_score_fields) != len(self.score_fields):
                    raise ValueError(
                        f"previous_score_fields count ({len(self.previous_score_fields)}) must match "
                        f"score_fields count ({len(self.score_fields)})"
                    )

                logger.info(
                    f"Multi-task comparison mode enabled with {len(self.previous_score_fields)} previous score fields"
                )

        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate dollar recall configuration
        if self.compute_dollar_recall and not self.amount_field:
            logger.warning(
                "compute_dollar_recall is enabled but amount_field is not set - "
                "dollar recall will be skipped if amount data is not available"
            )

        # Validate threshold parameters
        if self.dollar_recall_fpr <= 0 or self.dollar_recall_fpr >= 1:
            raise ValueError(
                f"dollar_recall_fpr must be between 0 and 1, got {self.dollar_recall_fpr}"
            )

        if self.count_recall_cutoff <= 0 or self.count_recall_cutoff >= 1:
            raise ValueError(
                f"count_recall_cutoff must be between 0 and 1, got {self.count_recall_cutoff}"
            )

        # Validate single-task comparison mode configuration
        if self.comparison_mode:
            if not self.previous_score_field or self.previous_score_field.strip() == "":
                raise ValueError(
                    "previous_score_field must be provided when comparison_mode is True (single-task comparison)"
                )

            # Validate comparison_metrics value
            valid_comparison_metrics = {"all", "basic"}
            if self.comparison_metrics not in valid_comparison_metrics:
                raise ValueError(
                    f"comparison_metrics must be one of {valid_comparison_metrics}, got '{self.comparison_metrics}'"
                )

            logger.info(
                f"Single-task comparison mode enabled with previous score field: '{self.previous_score_field}'"
            )
        else:
            logger.debug(
                "Comparison mode disabled - standard metrics computation will be performed"
            )

        if is_singletask:
            logger.debug(
                f"Single-task mode: ID field '{self.id_name}', label field '{self.label_name}', score field '{self.score_field}'"
            )
        else:
            logger.debug(
                f"Multi-task mode: ID field '{self.id_name}', {len(self.score_fields)} score fields"
            )

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model metrics computation script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add model metrics computation specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.id_name,
                "INPUT_FORMAT": self.input_format,
                "COMPUTE_DOLLAR_RECALL": str(self.compute_dollar_recall).lower(),
                "COMPUTE_COUNT_RECALL": str(self.compute_count_recall).lower(),
                "DOLLAR_RECALL_FPR": str(self.dollar_recall_fpr),
                "COUNT_RECALL_CUTOFF": str(self.count_recall_cutoff),
                "GENERATE_PLOTS": str(self.generate_plots).lower(),
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
            }
        )

        # Add label_field if provided (for single-task mode)
        if self.label_name:
            env_vars["LABEL_FIELD"] = self.label_name

        # Add score_field if provided (for single-task mode)
        if self.score_field:
            env_vars["SCORE_FIELD"] = self.score_field

        # Add SCORE_FIELDS for multi-task mode (takes precedence over SCORE_FIELD)
        if self.score_fields:
            env_vars["SCORE_FIELDS"] = ",".join(
                self.score_fields
            )  # Convert list to comma-separated string

            # Add TASK_LABEL_NAMES if provided
            if self.task_label_names:
                env_vars["TASK_LABEL_NAMES"] = ",".join(
                    self.task_label_names
                )  # Convert list to comma-separated string

        # Add amount field if specified
        if self.amount_field:
            env_vars["AMOUNT_FIELD"] = self.amount_field

        # Add single-task comparison mode environment variables
        env_vars.update(
            {
                "COMPARISON_MODE": str(self.comparison_mode).lower(),
                "COMPARISON_METRICS": self.comparison_metrics,
                "STATISTICAL_TESTS": str(self.statistical_tests).lower(),
                "COMPARISON_PLOTS": str(self.comparison_plots).lower(),
            }
        )

        # Add PREVIOUS_SCORE_FIELD for single-task comparison
        if self.previous_score_field:
            env_vars["PREVIOUS_SCORE_FIELD"] = self.previous_score_field

        # Add PREVIOUS_SCORE_FIELDS for multi-task comparison
        if self.previous_score_fields:
            env_vars["PREVIOUS_SCORE_FIELDS"] = ",".join(
                self.previous_score_fields
            )  # Convert list to comma-separated string

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The model metrics computation script contract
        """
        return MODEL_METRICS_COMPUTATION_CONTRACT

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include metrics computation specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and metrics computation specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add model metrics computation specific fields
        metrics_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "label_name": self.label_name,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "input_format": self.input_format,
            "compute_dollar_recall": self.compute_dollar_recall,
            "compute_count_recall": self.compute_count_recall,
            "generate_plots": self.generate_plots,
            "dollar_recall_fpr": self.dollar_recall_fpr,
            "count_recall_cutoff": self.count_recall_cutoff,
            "processing_framework_version": self.processing_framework_version,
            "use_large_processing_instance": self.use_large_processing_instance,
            # Tier 2 - Comparison mode fields
            "comparison_mode": self.comparison_mode,
            "previous_score_field": self.previous_score_field,
            "comparison_metrics": self.comparison_metrics,
            "statistical_tests": self.statistical_tests,
            "comparison_plots": self.comparison_plots,
        }

        # Add Tier 1 optional fields if set

        if self.score_field is not None:
            metrics_fields["score_field"] = self.score_field

        # Add score_fields if set (multi-task mode)
        if self.score_fields is not None:
            metrics_fields["score_fields"] = self.score_fields

        # Add task_label_names if set (multi-task mode)
        if self.task_label_names is not None:
            metrics_fields["task_label_names"] = self.task_label_names

        # Add previous_score_fields if set (multi-task comparison mode)
        if self.previous_score_fields is not None:
            metrics_fields["previous_score_fields"] = self.previous_score_fields

        # Only include optional fields if they're set
        if self.amount_field is not None:
            metrics_fields["amount_field"] = self.amount_field

        # Combine base fields and metrics fields (metrics fields take precedence if overlap)
        init_fields = {**base_fields, **metrics_fields}

        return init_fields
