"""
PIPER Metric Generation Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the PIPER metric generation step
using a self-contained design where derived fields are private with read-only properties.

PiperMetricGeneration is a peer / drop-in alternative to ModelMetricsComputation: it
consumes the SAME upstream ``eval_output`` dependency (a ``*ModelEval`` / ``*ModelInference``
producer), recomputes ROC/PR curves itself from the prediction data, and emits the PIPER
contract (``.metric`` JSON files + paired 2-column data CSVs) written FLAT to the output
root so PIPER can scan and render them.

Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, field_validator
from typing import Optional, Dict, List, Any, TYPE_CHECKING
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PiperMetricGenerationConfig(ProcessingStepConfigBase):
    """
    Configuration for the PIPER metric generation step with self-contained derivation logic.

    This class defines the configuration parameters for the PIPER metric generation step,
    which loads prediction data, recomputes ROC/PR curves, and emits the PIPER rendering
    contract: ``.metric`` JSON files (Graph-Line / Tabular visualization types) together
    with paired 2-column data CSVs, written FLAT to the processing output root
    (``/opt/ml/processing/output``) so PIPER can scan and render them.

    It is a peer / drop-in alternative to ModelMetricsComputation and reuses the same
    comparison machinery (``comparison_mode`` + ``previous_score_field``). The current model
    is the "variant" series (``score_field`` -> ``variant_model_id``); the previous / active
    model is the "control" series (``previous_score_field`` -> ``control_model_id``).

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

    variant_model_id: str = Field(
        ...,
        description="PIPER model identifier for the variant (current) model series. "
        "Emitted as VARIANT_MODEL_ID and used as series[].modelId for the variant "
        "series in the emitted .metric files (REQUIRED).",
    )

    score_field: Optional[str] = Field(
        default=None,
        description="Name of the score column to evaluate (single-task mode). "
        "Use this for backward compatibility or when evaluating a single score field. "
        "At least one of score_field or score_fields must be provided.",
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
        default="piper_metric_generation.py",
        description="Entry point script for PIPER metric generation.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'testing').",
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

    # Processing framework - metric generation uses scikit-learn
    processing_framework_version: str = Field(
        default="1.2-1",  # Python 3.8 compatible version
        description="Scikit-learn framework version for processing (metric generation uses sklearn)",
    )

    # For metric generation, we typically use smaller instances
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing (metric generation typically needs less resources)",
    )

    # Model comparison configuration (Tier 2 - Optional with defaults)
    comparison_mode: bool = Field(
        default=False,
        description="Enable model comparison functionality to compare with previous model scores (single-task mode)",
    )

    previous_score_field: str = Field(
        default="",
        description="Name of the column containing previous model scores for comparison (single-task mode, required when comparison_mode=True). "
        "This is the control series score field.",
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

    # ===== PIPER-specific additions (Tier 2) =====
    # These configure the PIPER rendering contract emitted by the script

    control_model_id: Optional[str] = Field(
        default=None,
        description="PIPER model identifier for the control (previous) model series. "
        "Emitted as CONTROL_MODEL_ID (only when set) and used as series[].modelId for "
        "the control series. The control series is only emitted when a control model is "
        "configured (comparison_mode / previous_score_field / control_model_id).",
    )

    # NOTE: pipeline_name is intentionally NOT declared here. It is a read-only
    # derived @property inherited from BasePipelineConfig
    # (f"{author}-{service_name}-{model_class}-{region}"). Redeclaring it as a
    # Field would be silently shadowed by the property. It is emitted as the
    # PIPELINE_NAME metadata env var from the inherited value (see
    # get_environment_variables), matching ModelWikiGeneratorConfig.

    dataset_type: str = Field(
        default="Validation",
        description="Dataset type emitted as DATASET_TYPE and used as metadata.dataset-type "
        "in the emitted .metric files.",
    )

    metrics_to_render: List[str] = Field(
        default_factory=lambda: ["auc_roc", "auc_pr", "data_statistics"],
        description="List of PIPER metrics to render. Emitted as METRICS_TO_RENDER "
        "(comma-joined). Supported values: 'auc_roc' (roc_curve.metric), "
        "'auc_pr' (pr_curve.metric), 'data_statistics' (data_preprocessing_statistic.metric).",
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Currently no derived fields specific to PIPER metric generation
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
    def initialize_derived_fields(self) -> "PiperMetricGenerationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_metric_generation_config(self) -> "PiperMetricGenerationConfig":
        """Additional validation specific to PIPER metric generation configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError(
                "PIPER metric generation step requires a processing_entry_point"
            )

        # Validate required fields from script contract
        if not self.id_name:
            raise ValueError(
                "id_name must be provided (required by PIPER metric generation contract)"
            )

        if not self.label_name:
            raise ValueError(
                "label_name must be provided (required for both single-task and multi-task modes)"
            )

        # Validate PIPER-required variant model id
        if not self.variant_model_id or self.variant_model_id.strip() == "":
            raise ValueError(
                "variant_model_id must be provided (required for the PIPER variant series modelId)"
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
                "Comparison mode disabled - single-series PIPER metric generation will be performed"
            )

        # Validate dataset_type is non-empty (used as metadata.dataset-type)
        if not self.dataset_type or self.dataset_type.strip() == "":
            raise ValueError("dataset_type must be a non-empty string")

        # Validate metrics_to_render
        if (
            not isinstance(self.metrics_to_render, list)
            or len(self.metrics_to_render) == 0
        ):
            raise ValueError("metrics_to_render must be a non-empty list of strings")

        valid_metrics_to_render = {"auc_roc", "auc_pr", "data_statistics"}
        invalid = [
            m for m in self.metrics_to_render if m not in valid_metrics_to_render
        ]
        if invalid:
            raise ValueError(
                f"metrics_to_render entries must be one of {valid_metrics_to_render}, got invalid entries {invalid}"
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
        Get environment variables for the PIPER metric generation script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add PIPER metric generation specific environment variables
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

        # ===== PIPER-specific environment variables =====
        env_vars.update(
            {
                "VARIANT_MODEL_ID": self.variant_model_id,
                "DATASET_TYPE": self.dataset_type,
                "METRICS_TO_RENDER": ",".join(self.metrics_to_render),
            }
        )

        # CONTROL_MODEL_ID only when set (control series is optional)
        if self.control_model_id:
            env_vars["CONTROL_MODEL_ID"] = self.control_model_id

        # PIPELINE_NAME from the inherited BasePipelineConfig.pipeline_name
        # derived property (always available). Matches ModelWikiGeneratorConfig.
        if self.pipeline_name:
            env_vars["PIPELINE_NAME"] = self.pipeline_name

        return env_vars

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include PIPER metric generation specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and PIPER metric generation specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add PIPER metric generation specific fields
        metrics_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "label_name": self.label_name,
            "variant_model_id": self.variant_model_id,
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
            # Tier 2 - PIPER-specific fields
            "dataset_type": self.dataset_type,
            "metrics_to_render": self.metrics_to_render,
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

        # Add PIPER-specific optional fields if set
        if self.control_model_id is not None:
            metrics_fields["control_model_id"] = self.control_model_id

        # pipeline_name is NOT an init field — it is a derived read-only property
        # inherited from BasePipelineConfig, so it must not be passed to child configs.

        # Combine base fields and metrics fields (metrics fields take precedence if overlap)
        init_fields = {**base_fields, **metrics_fields}

        return init_fields

    def get_job_arguments(self) -> Optional[List[str]]:
        """CLI args — config is the single source (FZ 31e1d3h)."""
        return self._job_type_arg()
