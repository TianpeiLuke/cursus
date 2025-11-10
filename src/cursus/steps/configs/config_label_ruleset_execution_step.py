"""
Label Ruleset Execution Step Configuration

This module implements the configuration class for the Label Ruleset Execution step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import Field, PrivateAttr, model_validator, field_validator
from typing import Dict, Any, Optional
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class LabelRulesetExecutionConfig(ProcessingStepConfigBase):
    """
    Configuration for Label Ruleset Execution step using three-tier design.

    This step applies validated rulesets to processed data to generate classification
    labels using priority-based rule evaluation with execution-time field validation.
    Supports stacked preprocessing patterns by using processed_data for both input
    and output.

    Tier 1: Essential user inputs (required)
    Tier 2: System inputs with defaults (optional)
    Tier 3: Derived fields (private with property access)
    """

    # ===== Tier 1: Essential User Inputs (Required) =====
    # These fields must be provided by users with no defaults

    job_type: str = Field(
        description="One of ['training','validation','testing','calibration'] - determines which splits to process (REQUIRED)"
    )

    # ===== Tier 2: System Inputs with Defaults (Optional) =====
    # These fields have sensible defaults but can be overridden

    # Execution configuration
    fail_on_missing_fields: bool = Field(
        default=True,
        description="Whether to fail execution if required fields are missing in data (True: raises error, False: skips split with warning)",
    )

    enable_rule_match_tracking: bool = Field(
        default=True,
        description="Whether to track detailed per-rule match statistics (disable for performance optimization)",
    )

    enable_progress_logging: bool = Field(
        default=True,
        description="Whether to log detailed progress information during processing (disable for minimal logging)",
    )

    preferred_input_format: str = Field(
        default="",
        description="Preferred input format when multiple formats exist in same directory ('CSV', 'TSV', 'Parquet', or empty string for auto-detection)",
    )

    # Processing step overrides
    processing_entry_point: str = Field(
        default="label_ruleset_execution.py",
        description="Entry point script for label ruleset execution",
    )

    # ===== Tier 3: Derived Fields (Private with Property Access) =====
    # These fields are calculated from other fields

    _execution_environment_variables: Optional[Dict[str, str]] = PrivateAttr(
        default=None
    )
    _processing_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _execution_configuration: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # Public properties for derived fields

    @property
    def execution_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the label ruleset execution step."""
        if self._execution_environment_variables is None:
            self._execution_environment_variables = {
                # Validation configuration
                "FAIL_ON_MISSING_FIELDS": str(self.fail_on_missing_fields).lower(),
                # Execution configuration
                "ENABLE_RULE_MATCH_TRACKING": str(
                    self.enable_rule_match_tracking
                ).lower(),
                "ENABLE_PROGRESS_LOGGING": str(self.enable_progress_logging).lower(),
                # Format preference
                "PREFERRED_INPUT_FORMAT": self.preferred_input_format.lower()
                if self.preferred_input_format
                else "",
                # Framework configuration
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
            }

        return self._execution_environment_variables

    @property
    def processing_metadata(self) -> Dict[str, Any]:
        """Get processing step metadata."""
        if self._processing_metadata is None:
            self._processing_metadata = {
                "step_type": "label_ruleset_execution",
                "job_type": self.job_type,
                "fail_on_missing_fields": self.fail_on_missing_fields,
                "rule_match_tracking_enabled": self.enable_rule_match_tracking,
                "progress_logging_enabled": self.enable_progress_logging,
                "supports_stacked_preprocessing": True,
                "multi_format_support": ["csv", "tsv", "parquet"],
            }

        return self._processing_metadata

    @property
    def execution_configuration(self) -> Dict[str, Any]:
        """Get execution configuration details."""
        if self._execution_configuration is None:
            self._execution_configuration = {
                "job_type": self.job_type,
                "fail_on_missing_fields": self.fail_on_missing_fields,
                "rule_match_tracking": self.enable_rule_match_tracking,
                "progress_logging": self.enable_progress_logging,
                "graceful_degradation": not self.fail_on_missing_fields,
                "performance_optimized": not self.enable_rule_match_tracking,
            }

        return self._execution_configuration

    # Validators

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job_type is one of the allowed values."""
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("preferred_input_format")
    @classmethod
    def validate_preferred_input_format(cls, v: str) -> str:
        """Validate preferred_input_format is one of the allowed values."""
        if v == "":
            return v  # Empty string is valid (auto-detection)

        allowed = {"CSV", "TSV", "Parquet"}
        if v not in allowed:
            raise ValueError(
                f"preferred_input_format must be one of {allowed} or empty string, got '{v}'"
            )
        return v

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)

        # Add derived properties to output
        data["execution_environment_variables"] = self.execution_environment_variables
        data["processing_metadata"] = self.processing_metadata
        data["execution_configuration"] = self.execution_configuration

        return data

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "LabelRulesetExecutionConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize execution-specific derived fields
        _ = self.execution_environment_variables
        _ = self.processing_metadata
        _ = self.execution_configuration

        return self

    @model_validator(mode="after")
    def validate_production_readiness(self) -> "LabelRulesetExecutionConfig":
        """Validate configuration for production readiness."""
        # Warn if graceful degradation is enabled (fail_on_missing_fields=False)
        if not self.fail_on_missing_fields:
            logger.warning(
                "Graceful degradation enabled (fail_on_missing_fields=False). "
                "Splits with missing fields will be skipped. "
                "Consider setting to True for strict production validation."
            )

        # Warn if tracking is disabled
        if not self.enable_rule_match_tracking:
            logger.info(
                "Rule match tracking disabled for performance optimization. "
                "Detailed per-rule statistics will not be available."
            )

        # Warn if progress logging is disabled
        if not self.enable_progress_logging:
            logger.info(
                "Progress logging disabled. Only critical messages will be logged."
            )

        return self

    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.label_ruleset_execution_contract import (
            LABEL_RULESET_EXECUTION_CONTRACT,
        )

        return LABEL_RULESET_EXECUTION_CONTRACT

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path for the label ruleset execution step.

        Args:
            default_path: Default script path to use if not found via other methods

        Returns:
            Script path resolved from processing_entry_point and source directories
        """
        # Use the parent class implementation which handles hybrid resolution
        return super().get_script_path(default_path)

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include execution-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add execution-specific fields (Tier 1 + Tier 2)
        execution_fields = {
            # Tier 2: System fields with defaults
            "job_type": self.job_type,
            "fail_on_missing_fields": self.fail_on_missing_fields,
            "enable_rule_match_tracking": self.enable_rule_match_tracking,
            "enable_progress_logging": self.enable_progress_logging,
        }

        # Combine base fields and execution fields (execution fields take precedence if overlap)
        init_fields = {**base_fields, **execution_fields}

        return init_fields

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get all environment variables for the step builder.

        Returns:
            Dict[str, str]: Complete environment variables dictionary
        """
        return self.execution_environment_variables

    def is_production_ready(self) -> bool:
        """
        Check if configuration is production-ready.

        Returns:
            bool: True if configuration has production-ready settings
        """
        return (
            # Strict field validation enabled
            self.fail_on_missing_fields
            and
            # Tracking enabled for observability
            self.enable_rule_match_tracking
            and
            # Progress logging enabled for debugging
            self.enable_progress_logging
        )

    def get_execution_info(self) -> Dict[str, Any]:
        """
        Get detailed execution configuration information.

        Returns:
            Dict[str, Any]: Execution details and recommendations
        """
        return {
            "execution_configuration": self.execution_configuration,
            "processing_metadata": self.processing_metadata,
            "environment_variables": self.execution_environment_variables,
            "recommendations": {
                "strict_validation": self.fail_on_missing_fields,
                "observability": self.enable_rule_match_tracking,
                "debugging": self.enable_progress_logging,
                "production_ready": self.is_production_ready(),
                "supports_multi_format": True,
                "supports_stacked_preprocessing": True,
            },
            "feature_support": {
                "csv_format": True,
                "tsv_format": True,
                "parquet_format": True,
                "compressed_files": True,
                "train_val_test_splits": self.job_type == "training",
                "single_split_processing": self.job_type != "training",
                "field_validation": True,
                "data_quality_warnings": True,
                "priority_based_evaluation": True,
                "default_label_fallback": True,
            },
        }
