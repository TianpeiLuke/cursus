"""
Temporal Sequence Normalization Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for temporal sequence normalization, using a self-contained design where each field
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
from ..contracts.temporal_sequence_normalization_contract import (
    TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class TemporalSequenceNormalizationConfig(ProcessingStepConfigBase):
    """
    Configuration for the Temporal Sequence Normalization step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    temporal_field: str = Field(
        description="Field name containing timestamps for temporal ordering."
    )

    sequence_grouping_field: str = Field(
        description="Field name used to group records into temporal sequences (e.g., customerId)."
    )

    record_id_field: str = Field(
        description="Field name that uniquely identifies individual records (e.g., objectId)."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="temporal_sequence_normalization.py",
        description="Relative path (within processing_source_dir) to the temporal sequence normalization script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    sequence_length: int = Field(
        default=51,
        ge=1,
        description="Target sequence length for padding/truncation operations.",
    )

    sequence_separator: str = Field(
        default="~",
        description="Separator character used to split sequence values within fields.",
    )

    # Missing value handling
    missing_indicators: List[str] = Field(
        default=["", "My Text String"],
        description="List of values to treat as missing indicators in sequences.",
    )

    # Time delta configuration
    time_delta_max_seconds: int = Field(
        default=10000000,
        ge=0,
        description="Maximum time delta cap in seconds for temporal relationships.",
    )

    # Padding and truncation strategies
    padding_strategy: str = Field(
        default="pre",
        description="Padding strategy: 'pre' (pad at beginning) or 'post' (pad at end).",
    )

    truncation_strategy: str = Field(
        default="post",
        description="Truncation strategy: 'pre' (truncate from beginning) or 'post' (truncate from end).",
    )

    # Multi-sequence configuration
    enable_multi_sequence: bool = Field(
        default=False,
        description="Enable dual-sequence processing for multiple entity types.",
    )

    secondary_entity_field: str = Field(
        default="creditCardId",
        description="Secondary entity field for dual-sequence processing.",
    )

    sequence_naming_pattern: str = Field(
        default="*_seq_by_{entity}.*",
        description="Pattern for automatic sequence field detection.",
    )

    # Processing configuration
    enable_distributed_processing: bool = Field(
        default=False,
        description="Enable chunked processing for large datasets.",
    )

    chunk_size: int = Field(
        default=10000,
        ge=1000,
        description="Chunk size for distributed processing.",
    )

    max_workers: str = Field(
        default="auto",
        description="Number of parallel workers ('auto' or integer string).",
    )

    validation_strategy: str = Field(
        default="strict",
        description="Data validation strategy: 'strict' or 'lenient'.",
    )

    output_format: str = Field(
        default="numpy",
        description="Output format for normalized sequences: 'numpy', 'parquet', or 'csv'.",
    )

    include_attention_masks: bool = Field(
        default=True,
        description="Generate attention masks for padded sequences.",
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
                "SEQUENCE_LENGTH": str(self.sequence_length),
                "SEQUENCE_SEPARATOR": self.sequence_separator,
                "TEMPORAL_FIELD": self.temporal_field,
                "SEQUENCE_GROUPING_FIELD": self.sequence_grouping_field,
                "RECORD_ID_FIELD": self.record_id_field,
                # Optional environment variables with defaults
                "MISSING_INDICATORS": json.dumps(self.missing_indicators),
                "TIME_DELTA_MAX_SECONDS": str(self.time_delta_max_seconds),
                "PADDING_STRATEGY": self.padding_strategy,
                "TRUNCATION_STRATEGY": self.truncation_strategy,
                "ENABLE_MULTI_SEQUENCE": str(self.enable_multi_sequence).lower(),
                "SECONDARY_ENTITY_FIELD": self.secondary_entity_field,
                "SEQUENCE_NAMING_PATTERN": self.sequence_naming_pattern,
                "ENABLE_DISTRIBUTED_PROCESSING": str(
                    self.enable_distributed_processing
                ).lower(),
                "CHUNK_SIZE": str(self.chunk_size),
                "MAX_WORKERS": self.max_workers,
                "VALIDATION_STRATEGY": self.validation_strategy,
                "OUTPUT_FORMAT": self.output_format,
                "INCLUDE_ATTENTION_MASKS": str(self.include_attention_masks).lower(),
            }

        return self._environment_variables

    # ===== Validators =====

    @field_validator("sequence_length")
    @classmethod
    def validate_sequence_length(cls, v: int) -> int:
        """Ensure sequence_length is positive."""
        if v <= 0:
            raise ValueError("sequence_length must be positive")
        return v

    @field_validator("sequence_separator")
    @classmethod
    def validate_sequence_separator(cls, v: str) -> str:
        """Ensure sequence_separator is non-empty."""
        if not v:
            raise ValueError("sequence_separator must be non-empty")
        return v

    @field_validator("temporal_field", "sequence_grouping_field", "record_id_field")
    @classmethod
    def validate_field_names(cls, v: str) -> str:
        """Ensure field names are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Field names must be non-empty strings")
        return v.strip()

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

    @field_validator("padding_strategy", "truncation_strategy")
    @classmethod
    def validate_strategies(cls, v: str) -> str:
        """Ensure strategies are valid."""
        allowed = {"pre", "post"}
        if v not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("validation_strategy")
    @classmethod
    def validate_validation_strategy(cls, v: str) -> str:
        """Ensure validation_strategy is valid."""
        allowed = {"strict", "lenient"}
        if v not in allowed:
            raise ValueError(f"validation_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Ensure output_format is valid."""
        allowed = {"numpy", "parquet", "csv"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
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

    @field_validator("missing_indicators")
    @classmethod
    def validate_missing_indicators(cls, v: List[str]) -> List[str]:
        """Ensure missing_indicators is a non-empty list."""
        if not v:
            raise ValueError("missing_indicators must be a non-empty list")
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TemporalSequenceNormalizationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The temporal sequence normalization script contract
        """
        return TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include temporal sequence normalization specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add temporal sequence normalization specific fields
        normalization_fields = {
            "temporal_field": self.temporal_field,
            "sequence_grouping_field": self.sequence_grouping_field,
            "record_id_field": self.record_id_field,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "sequence_length": self.sequence_length,
            "sequence_separator": self.sequence_separator,
            "missing_indicators": self.missing_indicators,
            "time_delta_max_seconds": self.time_delta_max_seconds,
            "padding_strategy": self.padding_strategy,
            "truncation_strategy": self.truncation_strategy,
            "enable_multi_sequence": self.enable_multi_sequence,
            "secondary_entity_field": self.secondary_entity_field,
            "sequence_naming_pattern": self.sequence_naming_pattern,
            "enable_distributed_processing": self.enable_distributed_processing,
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "validation_strategy": self.validation_strategy,
            "output_format": self.output_format,
            "include_attention_masks": self.include_attention_masks,
        }

        # Combine fields (normalization fields take precedence if overlap)
        init_fields = {**base_fields, **normalization_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        data["environment_variables"] = self.environment_variables

        return data
