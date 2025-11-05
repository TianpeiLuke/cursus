"""
Temporal Feature Engineering Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for temporal feature engineering, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import contract
from ..contracts.temporal_feature_engineering_contract import (
    TEMPORAL_FEATURE_ENGINEERING_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class TemporalFeatureEngineeringConfig(ProcessingStepConfigBase):
    """
    Configuration for the Temporal Feature Engineering step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    sequence_grouping_field: str = Field(
        description="Field name used to group entities for feature computation (e.g., customerId)."
    )

    timestamp_field: str = Field(
        description="Field name containing temporal information for feature extraction."
    )

    value_fields: List[str] = Field(
        description="List of numerical fields for temporal feature extraction (e.g., ['transactionAmount', 'merchantRiskScore'])."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    feature_types: List[str] = Field(
        default=["statistical", "temporal", "behavioral"],
        description="List of feature types to extract: ['statistical', 'temporal', 'behavioral'].",
    )

    processing_entry_point: str = Field(
        default="temporal_feature_engineering.py",
        description="Relative path (within processing_source_dir) to the temporal feature engineering script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    categorical_fields: List[str] = Field(
        default=["merchantCategory", "paymentMethod"],
        description="List of categorical fields for temporal feature extraction.",
    )

    # Time window configuration
    window_sizes: List[int] = Field(
        default=[7, 14, 30, 90],
        description="List of time window sizes for aggregation features.",
    )

    aggregation_functions: List[str] = Field(
        default=["mean", "sum", "std", "min", "max", "count"],
        description="List of aggregation functions to apply in time windows.",
    )

    lag_features: List[int] = Field(
        default=[1, 7, 14, 30],
        description="List of lag periods for historical features.",
    )

    # Exponential smoothing configuration
    exponential_smoothing_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Alpha parameter for exponential smoothing (0.0 to 1.0).",
    )

    time_unit: str = Field(
        default="days",
        description="Time unit for window calculations: 'days' or 'hours'.",
    )

    # Input/Output format configuration
    input_format: str = Field(
        default="numpy",
        description="Input data format: 'numpy', 'parquet', or 'csv'.",
    )

    output_format: str = Field(
        default="numpy",
        description="Output format for temporal feature tensors: 'numpy', 'parquet', or 'csv'.",
    )

    # Distributed processing configuration
    enable_distributed_processing: bool = Field(
        default=False,
        description="Enable chunked processing for large datasets.",
    )

    chunk_size: int = Field(
        default=5000,
        ge=1000,
        description="Chunk size for distributed processing.",
    )

    max_workers: str = Field(
        default="auto",
        description="Number of parallel workers ('auto' or integer string).",
    )

    feature_parallelism: bool = Field(
        default=True,
        description="Enable parallel feature type computation.",
    )

    cache_intermediate: bool = Field(
        default=True,
        description="Cache intermediate results for reuse.",
    )

    # Quality control configuration
    enable_validation: bool = Field(
        default=True,
        description="Enable comprehensive feature quality validation.",
    )

    missing_value_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging high missing value features (0.0 to 1.0).",
    )

    correlation_threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Threshold for flagging highly correlated features (0.0 to 1.0).",
    )

    variance_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Threshold for flagging low variance features.",
    )

    outlier_detection: bool = Field(
        default=True,
        description="Enable outlier detection in feature distributions.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables dictionary for the processing step.

        Returns:
            Dictionary of environment variables
        """
        if self._environment_variables is None:
            self._environment_variables = {
                # Required environment variables
                "SEQUENCE_GROUPING_FIELD": self.sequence_grouping_field,
                "TIMESTAMP_FIELD": self.timestamp_field,
                "VALUE_FIELDS": json.dumps(self.value_fields),
                # Optional environment variables with defaults
                "FEATURE_TYPES": json.dumps(self.feature_types),
                "CATEGORICAL_FIELDS": json.dumps(self.categorical_fields),
                "WINDOW_SIZES": json.dumps(self.window_sizes),
                "AGGREGATION_FUNCTIONS": json.dumps(self.aggregation_functions),
                "LAG_FEATURES": json.dumps(self.lag_features),
                "EXPONENTIAL_SMOOTHING_ALPHA": str(self.exponential_smoothing_alpha),
                "TIME_UNIT": self.time_unit,
                "INPUT_FORMAT": self.input_format,
                "OUTPUT_FORMAT": self.output_format,
                "ENABLE_DISTRIBUTED_PROCESSING": str(
                    self.enable_distributed_processing
                ).lower(),
                "CHUNK_SIZE": str(self.chunk_size),
                "MAX_WORKERS": self.max_workers,
                "FEATURE_PARALLELISM": str(self.feature_parallelism).lower(),
                "CACHE_INTERMEDIATE": str(self.cache_intermediate).lower(),
                "ENABLE_VALIDATION": str(self.enable_validation).lower(),
                "MISSING_VALUE_THRESHOLD": str(self.missing_value_threshold),
                "CORRELATION_THRESHOLD": str(self.correlation_threshold),
                "VARIANCE_THRESHOLD": str(self.variance_threshold),
                "OUTLIER_DETECTION": str(self.outlier_detection).lower(),
            }

        return self._environment_variables

    # ===== Validators =====

    @field_validator("sequence_grouping_field", "timestamp_field")
    @classmethod
    def validate_field_names(cls, v: str) -> str:
        """Ensure field names are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Field names must be non-empty strings")
        return v.strip()

    @field_validator("value_fields")
    @classmethod
    def validate_value_fields(cls, v: List[str]) -> List[str]:
        """Ensure value_fields is a non-empty list of non-empty strings."""
        if not v:
            raise ValueError("value_fields must be a non-empty list")
        for field in v:
            if not field or not field.strip():
                raise ValueError("All value fields must be non-empty strings")
        return [field.strip() for field in v]

    @field_validator("feature_types")
    @classmethod
    def validate_feature_types(cls, v: List[str]) -> List[str]:
        """Ensure feature_types contains valid feature types."""
        if not v:
            raise ValueError("feature_types must be a non-empty list")

        allowed_types = {"statistical", "temporal", "behavioral"}
        for feature_type in v:
            if feature_type not in allowed_types:
                raise ValueError(
                    f"feature_type must be one of {allowed_types}, got '{feature_type}'"
                )
        return v

    @field_validator("categorical_fields")
    @classmethod
    def validate_categorical_fields(cls, v: List[str]) -> List[str]:
        """Ensure categorical_fields contains valid field names."""
        for field in v:
            if not field or not field.strip():
                raise ValueError("All categorical fields must be non-empty strings")
        return [field.strip() for field in v]

    @field_validator("window_sizes")
    @classmethod
    def validate_window_sizes(cls, v: List[int]) -> List[int]:
        """Ensure window_sizes contains positive integers."""
        if not v:
            raise ValueError("window_sizes must be a non-empty list")
        for size in v:
            if size <= 0:
                raise ValueError("All window sizes must be positive integers")
        return v

    @field_validator("aggregation_functions")
    @classmethod
    def validate_aggregation_functions(cls, v: List[str]) -> List[str]:
        """Ensure aggregation_functions contains valid function names."""
        if not v:
            raise ValueError("aggregation_functions must be a non-empty list")

        allowed_functions = {"mean", "sum", "std", "min", "max", "count", "median"}
        for func in v:
            if func not in allowed_functions:
                raise ValueError(
                    f"aggregation_function must be one of {allowed_functions}, got '{func}'"
                )
        return v

    @field_validator("lag_features")
    @classmethod
    def validate_lag_features(cls, v: List[int]) -> List[int]:
        """Ensure lag_features contains positive integers."""
        if not v:
            raise ValueError("lag_features must be a non-empty list")
        for lag in v:
            if lag <= 0:
                raise ValueError("All lag features must be positive integers")
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """Ensure processing_entry_point is a non‐empty relative path."""
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError(
                "processing_entry_point must be a relative path within source directory"
            )
        return v

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Ensure job_type is one of the allowed values."""
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("time_unit")
    @classmethod
    def validate_time_unit(cls, v: str) -> str:
        """Ensure time_unit is valid."""
        allowed = {"days", "hours"}
        if v not in allowed:
            raise ValueError(f"time_unit must be one of {allowed}, got '{v}'")
        return v

    @field_validator("input_format", "output_format")
    @classmethod
    def validate_formats(cls, v: str) -> str:
        """Ensure input/output formats are valid."""
        allowed = {"numpy", "parquet", "csv"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: str) -> str:
        """Ensure max_workers is 'auto' or a positive integer string."""
        if v != "auto":
            try:
                workers = int(v)
                if workers <= 0:
                    raise ValueError("max_workers must be 'auto' or a positive integer")
            except ValueError:
                raise ValueError(
                    "max_workers must be 'auto' or a positive integer string"
                )
        return v

    @field_validator("exponential_smoothing_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Ensure exponential_smoothing_alpha is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("exponential_smoothing_alpha must be between 0.0 and 1.0")
        return v

    @field_validator("missing_value_threshold", "correlation_threshold")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        """Ensure thresholds are between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v

    @field_validator("variance_threshold")
    @classmethod
    def validate_variance_threshold(cls, v: float) -> float:
        """Ensure variance_threshold is non-negative."""
        if v < 0.0:
            raise ValueError("variance_threshold must be non-negative")
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TemporalFeatureEngineeringConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The temporal feature engineering script contract
        """
        return TEMPORAL_FEATURE_ENGINEERING_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include temporal feature engineering specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add temporal feature engineering specific fields
        feature_engineering_fields = {
            "sequence_grouping_field": self.sequence_grouping_field,
            "timestamp_field": self.timestamp_field,
            "value_fields": self.value_fields,
            "feature_types": self.feature_types,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "categorical_fields": self.categorical_fields,
            "window_sizes": self.window_sizes,
            "aggregation_functions": self.aggregation_functions,
            "lag_features": self.lag_features,
            "exponential_smoothing_alpha": self.exponential_smoothing_alpha,
            "time_unit": self.time_unit,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "enable_distributed_processing": self.enable_distributed_processing,
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "feature_parallelism": self.feature_parallelism,
            "cache_intermediate": self.cache_intermediate,
            "enable_validation": self.enable_validation,
            "missing_value_threshold": self.missing_value_threshold,
            "correlation_threshold": self.correlation_threshold,
            "variance_threshold": self.variance_threshold,
            "outlier_detection": self.outlier_detection,
        }

        # Combine fields (feature engineering fields take precedence if overlap)
        init_fields = {**base_fields, **feature_engineering_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        data["environment_variables"] = self.environment_variables

        return data
