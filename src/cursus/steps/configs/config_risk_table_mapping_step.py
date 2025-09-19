"""
Configuration for Risk Table Mapping Processing Step.

This module defines the configuration class for the risk table mapping processing step,
which is responsible for creating and applying risk tables for categorical features.

After Phase 6 refactor: Hyperparameters are now embedded in the source directory,
eliminating the need for S3 upload logic and hyperparameters_s3_uri field.
"""

from typing import List
from pydantic import Field, model_validator, field_validator
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class RiskTableMappingConfig(ProcessingStepConfigBase):
    """
    Configuration for Risk Table Mapping Processing Step.

    This class extends ProcessingStepConfigBase to include specific fields
    for risk table mapping, including categorical fields and job type.
    
    Source Directory Integration:
    After Phase 6 refactor, hyperparameters are embedded in the source directory
    structure and no longer require separate S3 upload. The expected source
    directory structure is:
    
    source_dir/
    ├── risk_table_mapping.py          # Main script (entry point)
    └── hyperparams/                   # Hyperparameters directory
        └── hyperparameters.json       # Generated hyperparameters file
    
    The hyperparameters.json file is automatically generated from the configuration
    fields (cat_field_list, label_name, smooth_factor, count_threshold) and
    embedded in the source directory for runtime access.
    """

    # Script settings
    processing_entry_point: str = Field(
        default="risk_table_mapping.py", 
        description="Script for risk table mapping (embedded in source directory)"
    )

    # Job type for the processing script
    job_type: str = Field(
        default="training",
        description="Type of job to perform. One of 'training', 'validation', 'testing', 'calibration'",
    )

    # Risk table mapping hyperparameters (embedded in source directory)
    cat_field_list: List[str] = Field(
        default=[],
        description="List of categorical fields to apply risk table mapping to (embedded in hyperparams/hyperparameters.json)",
    )

    label_name: str = Field(
        default="target", 
        description="Name of the target/label column (embedded in hyperparams/hyperparameters.json)"
    )

    smooth_factor: float = Field(
        default=0.01, 
        description="Smoothing factor for risk table calculation (embedded in hyperparams/hyperparameters.json)"
    )

    count_threshold: int = Field(
        default=5, 
        description="Minimum count threshold for risk table calculation (embedded in hyperparams/hyperparameters.json)"
    )

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job type is one of the allowed values."""
        allowed_types = ["training", "validation", "testing", "calibration"]
        if v.lower() not in allowed_types:
            raise ValueError(f"job_type must be one of {allowed_types}, got {v}")
        return v.lower()

    @model_validator(mode="after")
    def validate_risk_table_config(self) -> "RiskTableMappingConfig":
        """
        Validate risk table mapping configuration.
        
        After Phase 6 refactor: Simplified validation focusing on core configuration
        without S3 upload logic or external hyperparameters handling.
        """
        # Validate label_name is provided
        if not self.label_name:
            raise ValueError("label_name must be provided for risk table mapping")
        
        # For training job type, validate cat_field_list
        if self.job_type == "training" and not self.cat_field_list:
            logger.warning(
                "cat_field_list is empty for training job. Risk table mapping will validate fields at runtime."
            )
        
        # Validate numeric parameters
        if self.smooth_factor < 0:
            raise ValueError("smooth_factor must be non-negative")
        
        if self.count_threshold < 0:
            raise ValueError("count_threshold must be non-negative")
        
        return self
