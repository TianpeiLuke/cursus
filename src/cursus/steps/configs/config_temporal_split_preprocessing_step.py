"""
Temporal Split Preprocessing Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for temporal split preprocessing, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import contract
from ..contracts.temporal_split_preprocessing_contract import (
    TEMPORAL_SPLIT_PREPROCESSING_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class TemporalSplitPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Temporal Split Preprocessing step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    job_type: str = Field(
        description="One of ['training','validation','testing','calibration']",
    )

    date_column: str = Field(
        description="Name of the date column for temporal split",
    )

    group_id_column: str = Field(
        description="Name of the group ID column for group-level splitting",
    )

    split_date: str = Field(
        description="Date for temporal split in YYYY-MM-DD format",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="temporal_split_preprocessing.py",
        description="Relative path (within processing_source_dir) to the temporal split preprocessing script.",
    )

    train_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Fraction of customers to allocate to the training set (only used if job_type=='training').",
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible customer splitting.",
    )

    output_format: str = Field(
        default="CSV",
        description="Output format for processed data ('CSV', 'TSV', or 'Parquet'). Default: CSV",
    )

    max_workers: Optional[int] = Field(
        default=4,
        ge=1,
        description="Maximum number of parallel workers for processing (default: 4).",
    )

    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for DataFrame concatenation (default: 10).",
    )

    # Task configuration - single task vs multitask
    label_field: Optional[str] = Field(
        default=None,
        description="Label field name for single-task mode. "
        "For multitask mode, this represents the main task label field. "
        "Required for single-task mode, optional for multitask mode.",
    )

    targets: Optional[List[str]] = Field(
        default=None,
        description="List of target column names for multitask mode. "
        "REQUIRED for multitask mode. Must include label_field if label_field is provided. "
        "Example: ['is_abuse', 'is_abusive_dnr', 'is_abusive_pda', 'is_abusive_rr']",
    )

    main_task_index: Optional[int] = Field(
        default=0,
        ge=0,
        description="Index of main task in targets list for label generation (default: 0). "
        "Only used in multitask mode when targets is provided.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _full_script_path: Optional[str] = PrivateAttr(default=None)
    _temporal_split_environment_variables: Optional[Dict[str, str]] = PrivateAttr(
        default=None
    )

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def full_script_path(self) -> Optional[str]:
        """
        Get full path to the temporal split preprocessing script.

        Returns:
            Full path to the script
        """
        if self._full_script_path is None:
            # Get effective source directory
            source_dir = self.effective_source_dir
            if source_dir is None:
                return None

            # Combine with entry point
            if source_dir.startswith("s3://"):
                self._full_script_path = (
                    f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
                )
            else:
                self._full_script_path = str(
                    Path(source_dir) / self.processing_entry_point
                )

        return self._full_script_path

    @property
    def temporal_split_environment_variables(self) -> Dict[str, str]:
        """
        Get temporal split preprocessing-specific environment variables.

        Returns:
            Dictionary mapping environment variable names to values
        """
        if self._temporal_split_environment_variables is None:
            env_vars = {}

            # Required temporal split parameters
            env_vars["DATE_COLUMN"] = self.date_column
            env_vars["GROUP_ID_COLUMN"] = self.group_id_column
            env_vars["SPLIT_DATE"] = self.split_date

            # Split configuration
            env_vars["TRAIN_RATIO"] = str(self.train_ratio)
            env_vars["RANDOM_SEED"] = str(self.random_seed)

            # Output format
            env_vars["OUTPUT_FORMAT"] = self.output_format

            # Advanced processing parameters
            env_vars["MAX_WORKERS"] = str(self.max_workers)
            env_vars["BATCH_SIZE"] = str(self.batch_size)

            # Task configuration - single task vs multitask
            if self.label_field:
                env_vars["LABEL_FIELD"] = self.label_field

            # For multitask mode: convert targets list to comma-separated string
            if self.targets:
                env_vars["TARGETS"] = ",".join(self.targets)

            if self.main_task_index is not None:
                env_vars["MAIN_TASK_INDEX"] = str(self.main_task_index)

            self._temporal_split_environment_variables = env_vars

        return self._temporal_split_environment_variables

    # ===== Validators =====

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
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("split_date")
    @classmethod
    def validate_split_date_format(cls, v: str) -> str:
        """
        Ensure split_date is in YYYY-MM-DD format.
        """
        import re

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"split_date must be in YYYY-MM-DD format, got '{v}'")

        # Additional validation: try to parse as date
        try:
            from datetime import datetime

            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"split_date '{v}' is not a valid date")

        return v

    @field_validator("train_ratio")
    @classmethod
    def validate_train_ratio(cls, v: float) -> float:
        """
        Ensure the train_ratio is between 0 and 1.
        """
        if not (0.0 < v < 1.0):
            raise ValueError(f"train_ratio must be strictly between 0 and 1, got {v}")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """
        Ensure output_format is one of the allowed values (case-insensitive).
        """
        allowed = {"CSV", "TSV", "Parquet"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("targets")
    @classmethod
    def validate_targets_list(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validate targets is a list of strings for multitask mode.
        """
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("targets must be a list of strings")

        if len(v) == 0:
            raise ValueError("targets cannot be empty")

        if not all(isinstance(item, str) and item.strip() for item in v):
            raise ValueError("All targets must be non-empty strings")

        return v

    @model_validator(mode="after")
    def validate_task_configuration(self) -> "TemporalSplitPreprocessingConfig":
        """
        Validate single-task vs multitask configuration.

        Design:
        - Single-task mode: label_field MUST be provided
        - Multitask mode: targets AND main_task_index MUST be provided
        - For non-training job types (validation, testing, calibration): both are optional
        """
        # Determine if we're in single-task or multitask mode
        is_multitask = bool(self.targets)
        is_singletask = bool(self.label_field) and not is_multitask

        # For training job type, enforce proper task configuration
        if self.job_type == "training":
            if is_multitask:
                # Multitask mode: targets AND main_task_index are required
                if not self.targets:
                    raise ValueError("For multitask mode, 'targets' must be provided")
                if self.main_task_index is None:
                    raise ValueError(
                        "For multitask mode, 'main_task_index' must be provided"
                    )
            elif is_singletask:
                # Single-task mode: label_field is required (already provided)
                pass
            else:
                # Neither mode is properly configured
                raise ValueError(
                    "For training job type, you must provide either:\n"
                    "  - Single-task mode: 'label_field' must be provided\n"
                    "  - Multitask mode: 'targets' and 'main_task_index' must be provided"
                )

        # For multitask mode validation (regardless of job type)
        if self.targets:
            # If label_field is provided in multitask mode, it must be included in targets
            if self.label_field and self.label_field not in self.targets:
                raise ValueError(
                    f"label_field '{self.label_field}' must be included in targets for multitask mode. "
                    f"Current targets: {self.targets}"
                )

            # main_task_index must be valid for the targets list
            if self.main_task_index is not None and self.main_task_index >= len(
                self.targets
            ):
                raise ValueError(
                    f"main_task_index ({self.main_task_index}) must be less than targets length ({len(self.targets)})"
                )

        return self

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TemporalSplitPreprocessingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize full script path if possible
        source_dir = self.effective_source_dir
        if source_dir is not None:
            if source_dir.startswith("s3://"):
                self._full_script_path = (
                    f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
                )
            else:
                self._full_script_path = str(
                    Path(source_dir) / self.processing_entry_point
                )

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The temporal split preprocessing script contract
        """
        return TEMPORAL_SPLIT_PREPROCESSING_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include temporal split preprocessing specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add temporal split preprocessing specific fields
        temporal_fields = {
            "job_type": self.job_type,
            "date_column": self.date_column,
            "group_id_column": self.group_id_column,
            "split_date": self.split_date,
            "processing_entry_point": self.processing_entry_point,
            "train_ratio": self.train_ratio,
            "random_seed": self.random_seed,
            "output_format": self.output_format,
            "batch_size": self.batch_size,
        }

        # Include max_workers (now has default value)
        temporal_fields["max_workers"] = self.max_workers

        if self.label_field is not None:
            temporal_fields["label_field"] = self.label_field

        if self.targets is not None:
            temporal_fields["targets"] = self.targets

        if self.main_task_index is not None:
            temporal_fields["main_task_index"] = self.main_task_index

        # Combine fields (temporal fields take precedence if overlap)
        init_fields = {**base_fields, **temporal_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        if self.full_script_path:
            data["full_script_path"] = self.full_script_path

        data["temporal_split_environment_variables"] = (
            self.temporal_split_environment_variables
        )

        return data
