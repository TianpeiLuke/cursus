"""
Stratified Sampling Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for stratified sampling, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import Field, field_validator, model_validator
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StratifiedSamplingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Stratified Sampling step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    strata_column: str = Field(
        description="Column name to stratify by (e.g., target variable, confounding variable)."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="stratified_sampling.py",
        description="Relative path (within processing_source_dir) to the stratified sampling script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    sampling_strategy: str = Field(
        default="balanced",
        description="Sampling strategy: 'balanced' (class imbalance), 'proportional_min' (causal analysis), 'optimal' (variance optimization)",
    )

    target_sample_size: int = Field(
        default=1000,
        ge=1,
        description="Total desired sample size per split",
    )

    min_samples_per_stratum: int = Field(
        default=10,
        ge=1,
        description="Minimum samples per stratum for statistical power",
    )

    variance_column: Optional[str] = Field(
        default=None,
        description="Column for variance calculation (needed for optimal strategy)",
    )

    sampling_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        description="Multiplier for external reference counts (e.g., 5.0 for 5× oversampling).",
    )

    allow_replacement: bool = Field(
        default=False,
        description="Allow sampling with replacement when target exceeds available per stratum.",
    )

    reference_counts_json: Optional[str] = Field(
        default=None,
        description="JSON string of reference distribution {stratum: count}. Fallback when reference_counts.json sidecar file is absent.",
    )

    sampling_filter_column: Optional[str] = Field(
        default=None,
        description="Column to filter on before sampling. Only matching rows are sampled; rest pass through unchanged.",
    )

    sampling_filter_value: Optional[str] = Field(
        default=None,
        description="Value to match in filter_column for sampling subset selection.",
    )

    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====
    # (No derived properties needed - base class handles script path)

    # ===== Validators =====

    @field_validator("strata_column")
    @classmethod
    def validate_strata_column(cls, v: str) -> str:
        """
        Ensure strata_column is a non-empty string.
        """
        if not v or not v.strip():
            raise ValueError("strata_column must be a non-empty string")
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure processing_entry_point is a non‐empty relative path.
        """
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
        """
        Ensure job_type is one of the allowed values.
        """
        if not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(
                f"job_type must be lowercase alphanumeric (with underscores), got '{v}'"
            )
        return v

    @field_validator("sampling_strategy")
    @classmethod
    def validate_sampling_strategy(cls, v: str) -> str:
        """
        Ensure sampling_strategy is one of the allowed values (case-insensitive).

        Matching is case-insensitive and the stored value is normalized to the
        canonical-cased allowed value.
        """
        allowed = {"balanced", "proportional_min", "optimal", "external_proportional"}
        match = next((a for a in allowed if a.lower() == v.lower()), None)
        if match is None:
            raise ValueError(
                f"sampling_strategy must be one of {sorted(allowed)} (case-insensitive), got '{v}'"
            )
        return match

    @field_validator("variance_column")
    @classmethod
    def validate_variance_column(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure variance_column is a non-empty string if provided.
        """
        if v is not None and (not v or not v.strip()):
            raise ValueError("variance_column must be a non-empty string if provided")
        return v

    # Cross-field validation
    @model_validator(mode="after")
    def validate_strategy_requirements(self) -> "StratifiedSamplingConfig":
        """
        Validate that required fields are provided for specific strategies.
        """
        if self.sampling_strategy == "optimal" and self.variance_column is None:
            logger.warning(
                "optimal sampling strategy works best with variance_column specified. "
                "Using default variance if variance_column is not provided."
            )
        return self

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "StratifiedSamplingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        return self

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include stratified sampling specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add stratified sampling specific fields
        sampling_fields = {
            "strata_column": self.strata_column,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "sampling_strategy": self.sampling_strategy,
            "target_sample_size": self.target_sample_size,
            "min_samples_per_stratum": self.min_samples_per_stratum,
            "random_state": self.random_state,
            "sampling_multiplier": self.sampling_multiplier,
            "allow_replacement": self.allow_replacement,
        }

        # Only include optional fields if set
        if self.variance_column is not None:
            sampling_fields["variance_column"] = self.variance_column
        if self.reference_counts_json is not None:
            sampling_fields["reference_counts_json"] = self.reference_counts_json
        if self.sampling_filter_column is not None:
            sampling_fields["sampling_filter_column"] = self.sampling_filter_column
        if self.sampling_filter_value is not None:
            sampling_fields["sampling_filter_value"] = self.sampling_filter_value

        # Combine fields (sampling fields take precedence if overlap)
        init_fields = {**base_fields, **sampling_fields}

        return init_fields

    def get_job_arguments(self) -> Optional[List[str]]:
        """CLI args — config is the single source (FZ 31e1d3h)."""
        return self._job_type_arg()
