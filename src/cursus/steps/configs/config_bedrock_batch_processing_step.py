"""
Bedrock Batch Processing Step Configuration

This module implements the configuration class for the Bedrock Batch Processing step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import Field, PrivateAttr, model_validator, field_validator
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class BedrockBatchProcessingConfig(ProcessingStepConfigBase):
    """
    Configuration for Bedrock Batch Processing step using three-tier design.

    This step processes input data through AWS Bedrock models using batch inference
    capabilities with automatic fallback to real-time processing. Integrates with
    generated prompt templates and validation schemas from the Bedrock Prompt Template
    Generation step. Provides cost-efficient processing for large datasets.

    Tier 1: Essential user inputs (required)
    Tier 2: System inputs with defaults (optional)
    Tier 3: Derived fields (private with property access)
    """

    # ===== Tier 1: Essential User Inputs (Required) =====
    # These fields must be provided by users with no defaults

    bedrock_batch_role_arn: str = Field(
        description="IAM role ARN for batch inference jobs (e.g., 'arn:aws:iam::123456789012:role/BedrockBatchRole'). Must have permissions for Bedrock batch inference and S3 access."
    )

    # ===== Tier 2: System Inputs with Defaults (Optional) =====
    # These fields have sensible defaults but can be overridden

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration'] - determines processing behavior and output naming",
    )

    # Model configuration (inherited from BedrockProcessingConfig with batch optimizations)
    bedrock_primary_model_id: str = Field(
        default="anthropic.claude-sonnet-4-5-20250929-v1:0",
        description="Primary Bedrock model ID for processing (Claude Sonnet 4.5 default, latest stable model)",
    )

    bedrock_fallback_model_id: Optional[str] = Field(
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        description="Fallback model ID for inference profile failures (Claude Sonnet 4.0 for production reliability)",
    )

    bedrock_inference_profile_arn: Optional[str] = Field(
        default=None,
        description="Inference profile ARN for capacity management (e.g., 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123')",
    )

    bedrock_inference_profile_required_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic.claude-sonnet-4-5-20250929-v1:0",  # Claude Sonnet 4.5
            "anthropic.claude-sonnet-4-20250514-v1:0",  # Claude Sonnet 4.0
            "anthropic.claude-opus-4-1-20250805-v1:0",  # Claude Opus 4.1
        ],
        description="List of models requiring inference profiles (Claude 4.5, 4.0, and Opus 4.1 included by default)",
    )

    # API parameters (optimized for Claude 4 with batch processing)
    bedrock_max_tokens: int = Field(
        default=32768,
        ge=1,
        le=64000,
        description="Maximum tokens for Bedrock responses (32K optimal for Claude 4 - 50% of 64K maximum for reliability)",
    )

    bedrock_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (1.0 optimized for Claude 4)",
    )

    bedrock_top_p: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter (0.999 optimized for Claude 4)",
    )

    # Processing configuration (inherited from BedrockProcessingConfig)
    bedrock_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of records per processing batch (for real-time fallback mode)",
    )

    bedrock_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed Bedrock requests",
    )

    bedrock_output_column_prefix: str = Field(
        default="llm_", description="Prefix for output columns in processed data"
    )

    bedrock_skip_error_records: bool = Field(
        default=False,
        description="Whether to exclude error records from output files (statistics still track all records)",
    )

    # Concurrency configuration (for real-time fallback mode)
    bedrock_concurrency_mode: str = Field(
        default="sequential",
        description="Processing mode for real-time fallback: 'sequential' (safer) or 'concurrent' (faster)",
    )

    bedrock_max_concurrent_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of concurrent threads for concurrent processing (recommended: 3-10)",
    )

    bedrock_rate_limit_per_second: int = Field(
        default=10,
        ge=1,
        le=100,
        description="API requests per second limit for concurrent processing",
    )

    # ===== Batch-Specific Configuration =====
    # Unique to batch processing step

    bedrock_batch_mode: str = Field(
        default="auto",
        description="Batch processing mode: 'auto' (automatic selection), 'batch' (force batch), 'realtime' (force real-time)",
    )

    bedrock_batch_threshold: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Minimum records for automatic batch processing in auto mode (default: 1000)",
    )

    bedrock_batch_timeout_hours: int = Field(
        default=24,
        ge=1,
        le=72,
        description="Maximum hours for batch job completion (1-72 hours, default: 24)",
    )

    # AWS Bedrock batch limits (configurable per AWS documentation)
    bedrock_max_records_per_job: int = Field(
        default=45000,
        ge=1,
        le=50000,
        description="Maximum records per batch job (AWS limit: 50,000, default: 45,000 for safety margin)",
    )

    bedrock_max_concurrent_batch_jobs: int = Field(
        default=20,
        ge=1,
        le=20,
        description="Maximum concurrent batch jobs (AWS limit: 20)",
    )

    # Input truncation configuration
    bedrock_max_input_field_length: int = Field(
        default=400000,
        ge=100,
        le=1000000,
        description="Maximum length in characters for input fields before truncation (default: 400,000 chars ≈ 100,000 tokens)",
    )

    bedrock_truncation_enabled: bool = Field(
        default=True,
        description="Enable automatic truncation of oversized input fields to prevent error 413 'Input is too long'",
    )

    bedrock_log_truncations: bool = Field(
        default=True,
        description="Log detailed information about truncated fields for debugging and monitoring",
    )

    # Processing step overrides
    processing_entry_point: str = Field(
        default="bedrock_batch_processing.py",
        description="Entry point script for Bedrock batch processing",
    )

    # ===== Tier 3: Derived Fields (Private with Property Access) =====
    # These fields are calculated from other fields

    _effective_inference_profile_required_models: Optional[List[str]] = PrivateAttr(
        default=None
    )
    _bedrock_environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _processing_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _batch_configuration: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _cost_optimization_info: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # Public properties for derived fields

    @property
    def effective_inference_profile_required_models(self) -> List[str]:
        """Get effective list of models requiring inference profiles with auto-detection."""
        if self._effective_inference_profile_required_models is None:
            # Start with user-provided list
            models = list(self.bedrock_inference_profile_required_models)

            # Auto-detect known models that require inference profiles
            known_profile_models = [
                "anthropic.claude-sonnet-4-5-20250929-v1:0",  # Claude Sonnet 4.5
                "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude Haiku 4.5
                "anthropic.claude-sonnet-4-20250514-v1:0",  # Claude Sonnet 4.0
                "anthropic.claude-opus-4-1-20250805-v1:0",  # Claude Opus 4.1
                # Add other known models that require inference profiles
            ]

            # Add primary model if it's known to require profiles and not already in list
            if (
                self.bedrock_primary_model_id in known_profile_models
                and self.bedrock_primary_model_id not in models
            ):
                models.append(self.bedrock_primary_model_id)

            self._effective_inference_profile_required_models = models

        return self._effective_inference_profile_required_models

    @property
    def bedrock_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the Bedrock batch processing step."""
        if self._bedrock_environment_variables is None:
            self._bedrock_environment_variables = {
                # Standard Bedrock configuration (inherited from bedrock_processing.py)
                "BEDROCK_PRIMARY_MODEL_ID": self.bedrock_primary_model_id,
                "BEDROCK_FALLBACK_MODEL_ID": self.bedrock_fallback_model_id or "",
                "BEDROCK_INFERENCE_PROFILE_ARN": self.bedrock_inference_profile_arn
                or "",
                "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": json.dumps(
                    self.effective_inference_profile_required_models
                ),
                "AWS_DEFAULT_REGION": self.aws_region,
                "BEDROCK_MAX_TOKENS": str(self.bedrock_max_tokens),
                "BEDROCK_TEMPERATURE": str(self.bedrock_temperature),
                "BEDROCK_TOP_P": str(self.bedrock_top_p),
                "BEDROCK_BATCH_SIZE": str(self.bedrock_batch_size),
                "BEDROCK_MAX_RETRIES": str(self.bedrock_max_retries),
                "BEDROCK_OUTPUT_COLUMN_PREFIX": self.bedrock_output_column_prefix,
                "BEDROCK_SKIP_ERROR_RECORDS": str(
                    self.bedrock_skip_error_records
                ).lower(),
                "BEDROCK_MAX_CONCURRENT_WORKERS": str(
                    self.bedrock_max_concurrent_workers
                ),
                "BEDROCK_RATE_LIMIT_PER_SECOND": str(
                    self.bedrock_rate_limit_per_second
                ),
                "BEDROCK_CONCURRENCY_MODE": self.bedrock_concurrency_mode,
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
                # Batch-specific configuration (unique to batch processing)
                "BEDROCK_BATCH_MODE": self.bedrock_batch_mode,
                "BEDROCK_BATCH_THRESHOLD": str(self.bedrock_batch_threshold),
                "BEDROCK_BATCH_ROLE_ARN": self.bedrock_batch_role_arn,
                "BEDROCK_BATCH_TIMEOUT_HOURS": str(self.bedrock_batch_timeout_hours),
                # AWS Bedrock batch limits (configurable)
                "BEDROCK_MAX_RECORDS_PER_JOB": str(self.bedrock_max_records_per_job),
                "BEDROCK_MAX_CONCURRENT_BATCH_JOBS": str(
                    self.bedrock_max_concurrent_batch_jobs
                ),
                # Input truncation configuration
                "BEDROCK_MAX_INPUT_FIELD_LENGTH": str(
                    self.bedrock_max_input_field_length
                ),
                "BEDROCK_TRUNCATION_ENABLED": str(
                    self.bedrock_truncation_enabled
                ).lower(),
                "BEDROCK_LOG_TRUNCATIONS": str(self.bedrock_log_truncations).lower(),
                # S3 paths for batch processing (set by step builder using framework patterns)
                # These will be populated by the step builder using _get_base_output_path() and Join()
                "BEDROCK_BATCH_INPUT_S3_PATH": "",  # Will be set by step builder
                "BEDROCK_BATCH_OUTPUT_S3_PATH": "",  # Will be set by step builder
            }

        return self._bedrock_environment_variables

    @property
    def processing_metadata(self) -> Dict[str, Any]:
        """Get processing step metadata."""
        if self._processing_metadata is None:
            self._processing_metadata = {
                "step_type": "bedrock_batch_processing",
                "primary_model": self.bedrock_primary_model_id,
                "fallback_model": self.bedrock_fallback_model_id,
                "batch_mode": self.bedrock_batch_mode,
                "batch_threshold": self.bedrock_batch_threshold,
                "batch_timeout_hours": self.bedrock_batch_timeout_hours,
                "concurrency_mode": self.bedrock_concurrency_mode,
                "batch_size": self.bedrock_batch_size,
                "max_tokens": self.bedrock_max_tokens,
                "temperature": self.bedrock_temperature,
                "top_p": self.bedrock_top_p,
                "uses_inference_profile": bool(self.bedrock_inference_profile_arn),
                "inference_profile_required_models": self.effective_inference_profile_required_models,
                "output_column_prefix": self.bedrock_output_column_prefix,
                "batch_role_arn": self.bedrock_batch_role_arn,
            }

        return self._processing_metadata

    @property
    def batch_configuration(self) -> Dict[str, Any]:
        """Get batch processing configuration details."""
        if self._batch_configuration is None:
            self._batch_configuration = {
                "mode": self.bedrock_batch_mode,
                "threshold": self.bedrock_batch_threshold,
                "timeout_hours": self.bedrock_batch_timeout_hours,
                "role_arn": self.bedrock_batch_role_arn,
                "auto_selection_enabled": self.bedrock_batch_mode == "auto",
                "forced_batch_mode": self.bedrock_batch_mode == "batch",
                "forced_realtime_mode": self.bedrock_batch_mode == "realtime",
                "cost_optimization_enabled": self.bedrock_batch_mode
                in ["auto", "batch"],
                "fallback_to_realtime": True,  # Always enabled for reliability
            }

        return self._batch_configuration

    @property
    def cost_optimization_info(self) -> Dict[str, Any]:
        """Get cost optimization information."""
        if self._cost_optimization_info is None:
            # Estimate cost savings based on batch processing
            if self.bedrock_batch_mode == "auto":
                estimated_savings = "Up to 50% for datasets >= 1000 records"
                optimization_strategy = "Automatic selection based on data size"
            elif self.bedrock_batch_mode == "batch":
                estimated_savings = "Up to 50% for all datasets"
                optimization_strategy = "Forced batch processing for maximum savings"
            else:  # realtime
                estimated_savings = "0% (real-time processing only)"
                optimization_strategy = "Real-time processing for low latency"

            self._cost_optimization_info = {
                "batch_mode": self.bedrock_batch_mode,
                "threshold": self.bedrock_batch_threshold,
                "estimated_savings": estimated_savings,
                "optimization_strategy": optimization_strategy,
                "cost_efficient_for_large_datasets": self.bedrock_batch_mode
                in ["auto", "batch"],
                "automatic_fallback": True,
                "production_ready": self.is_production_ready(),
            }

        return self._cost_optimization_info

    # Validators

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job_type is one of the allowed values."""
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("bedrock_primary_model_id")
    @classmethod
    def validate_primary_model_id(cls, v: str) -> str:
        """Validate primary model ID format."""
        if not v or not v.strip():
            raise ValueError("bedrock_primary_model_id cannot be empty")

        # Basic format validation for common Bedrock model patterns
        valid_prefixes = [
            "anthropic.",
            "amazon.",
            "ai21.",
            "cohere.",
            "meta.",
            "mistral.",
            "stability.",
            "global.",  # For inference profile IDs
        ]

        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            logger.warning(
                f"Model ID '{v}' doesn't match common Bedrock patterns. Ensure it's a valid Bedrock model ID."
            )

        return v.strip()

    @field_validator("bedrock_batch_mode")
    @classmethod
    def validate_batch_mode(cls, v: str) -> str:
        """Validate batch processing mode."""
        valid_modes = ["auto", "batch", "realtime"]
        if v not in valid_modes:
            raise ValueError(
                f"bedrock_batch_mode must be one of {valid_modes}, got: {v}"
            )
        return v

    @field_validator("bedrock_batch_role_arn")
    @classmethod
    def validate_batch_role_arn(cls, v: str) -> str:
        """Validate batch role ARN format."""
        if not v or not v.strip():
            raise ValueError("bedrock_batch_role_arn cannot be empty")

        # Basic ARN format validation
        if not v.startswith("arn:aws:iam::"):
            raise ValueError(
                f"bedrock_batch_role_arn must be a valid IAM role ARN, got: {v}"
            )

        if ":role/" not in v:
            raise ValueError(f"bedrock_batch_role_arn must contain ':role/', got: {v}")

        return v.strip()

    @field_validator("bedrock_concurrency_mode")
    @classmethod
    def validate_concurrency_mode(cls, v: str) -> str:
        """Validate concurrency mode."""
        valid_modes = ["sequential", "concurrent"]
        if v not in valid_modes:
            raise ValueError(
                f"bedrock_concurrency_mode must be one of {valid_modes}, got: {v}"
            )
        return v

    @field_validator("bedrock_inference_profile_required_models")
    @classmethod
    def validate_inference_profile_models(cls, v: List[str]) -> List[str]:
        """Validate inference profile required models list."""
        if v is None:
            return []

        # Remove empty strings and duplicates
        cleaned = list(set(model.strip() for model in v if model and model.strip()))
        return cleaned

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)

        # Add derived properties to output
        data["effective_inference_profile_required_models"] = (
            self.effective_inference_profile_required_models
        )
        data["bedrock_environment_variables"] = self.bedrock_environment_variables
        data["processing_metadata"] = self.processing_metadata
        data["batch_configuration"] = self.batch_configuration
        data["cost_optimization_info"] = self.cost_optimization_info

        return data

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "BedrockBatchProcessingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize Bedrock batch-specific derived fields
        _ = self.effective_inference_profile_required_models
        _ = self.bedrock_environment_variables
        _ = self.processing_metadata
        _ = self.batch_configuration
        _ = self.cost_optimization_info

        return self

    @model_validator(mode="after")
    def validate_production_readiness(self) -> "BedrockBatchProcessingConfig":
        """Validate configuration for production readiness."""
        # Warn if using concurrent mode without fallback model
        if (
            self.bedrock_concurrency_mode == "concurrent"
            and not self.bedrock_fallback_model_id
        ):
            logger.warning(
                "Using concurrent processing without fallback model. "
                "Consider setting bedrock_fallback_model_id for production reliability."
            )

        # Warn if using inference profile without fallback
        if self.bedrock_inference_profile_arn and not self.bedrock_fallback_model_id:
            logger.warning(
                "Using inference profile without fallback model. "
                "Consider setting bedrock_fallback_model_id for production reliability."
            )

        # Validate batch processing configuration
        if self.bedrock_batch_mode in ["auto", "batch"]:
            if not self.bedrock_batch_role_arn:
                logger.warning(
                    "Batch processing enabled but no batch role ARN provided. "
                    "Batch processing will not be available."
                )

        # Validate concurrent processing parameters
        if self.bedrock_concurrency_mode == "concurrent":
            if self.bedrock_max_concurrent_workers > 10:
                logger.warning(
                    f"High concurrent worker count ({self.bedrock_max_concurrent_workers}). "
                    "Consider reducing to 3-10 workers to avoid rate limiting."
                )

            if self.bedrock_rate_limit_per_second > 50:
                logger.warning(
                    f"High rate limit ({self.bedrock_rate_limit_per_second} req/sec). "
                    "Ensure this doesn't exceed your Bedrock API limits."
                )

        return self

    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.bedrock_batch_processing_contract import (
            BEDROCK_BATCH_PROCESSING_CONTRACT,
        )

        return BEDROCK_BATCH_PROCESSING_CONTRACT

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path for the Bedrock batch processing step.

        Args:
            default_path: Default script path to use if not found via other methods

        Returns:
            Script path resolved from processing_entry_point and source directories
        """
        # Use the parent class implementation which handles hybrid resolution
        return super().get_script_path(default_path)

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include Bedrock batch-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add Bedrock batch-specific fields (Tier 1 + Tier 2)
        bedrock_fields = {
            # Tier 1: Essential fields
            "job_type": self.job_type,
            "bedrock_batch_role_arn": self.bedrock_batch_role_arn,
            # Tier 2: System fields with defaults
            "bedrock_primary_model_id": self.bedrock_primary_model_id,
            "bedrock_max_tokens": self.bedrock_max_tokens,
            "bedrock_temperature": self.bedrock_temperature,
            "bedrock_top_p": self.bedrock_top_p,
            "bedrock_batch_size": self.bedrock_batch_size,
            "bedrock_max_retries": self.bedrock_max_retries,
            "bedrock_output_column_prefix": self.bedrock_output_column_prefix,
            "bedrock_concurrency_mode": self.bedrock_concurrency_mode,
            "bedrock_max_concurrent_workers": self.bedrock_max_concurrent_workers,
            "bedrock_rate_limit_per_second": self.bedrock_rate_limit_per_second,
            "bedrock_inference_profile_required_models": self.bedrock_inference_profile_required_models,
            # Batch-specific fields
            "bedrock_batch_mode": self.bedrock_batch_mode,
            "bedrock_batch_threshold": self.bedrock_batch_threshold,
            "bedrock_batch_timeout_hours": self.bedrock_batch_timeout_hours,
        }

        # Only include optional fields if they're set
        if self.bedrock_fallback_model_id is not None:
            bedrock_fields["bedrock_fallback_model_id"] = self.bedrock_fallback_model_id

        if self.bedrock_inference_profile_arn is not None:
            bedrock_fields["bedrock_inference_profile_arn"] = (
                self.bedrock_inference_profile_arn
            )

        # Combine base fields and Bedrock fields (Bedrock fields take precedence if overlap)
        init_fields = {**base_fields, **bedrock_fields}

        return init_fields

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get all environment variables for the step builder.

        Returns:
            Dict[str, str]: Complete environment variables dictionary
        """
        return self.bedrock_environment_variables

    def is_production_ready(self) -> bool:
        """
        Check if configuration is production-ready.

        Returns:
            bool: True if configuration has production-ready settings
        """
        return (
            # Has fallback model for reliability
            self.bedrock_fallback_model_id is not None
            and
            # Has batch role ARN for batch processing
            bool(self.bedrock_batch_role_arn)
            and
            # Uses reasonable concurrency settings
            (
                self.bedrock_concurrency_mode == "sequential"
                or (
                    self.bedrock_max_concurrent_workers <= 10
                    and self.bedrock_rate_limit_per_second <= 50
                )
            )
        )

    def get_performance_estimate(self) -> Dict[str, Any]:
        """
        Get estimated performance characteristics including batch processing.

        Returns:
            Dict[str, Any]: Performance estimates and recommendations
        """
        # Base real-time performance estimate
        if self.bedrock_concurrency_mode == "sequential":
            realtime_speedup = 1.0
            realtime_throughput = f"~{60 // self.bedrock_batch_size} batches/min"
        else:
            realtime_speedup = min(
                self.bedrock_max_concurrent_workers,
                self.bedrock_rate_limit_per_second / 2,
            )
            realtime_throughput = (
                f"~{int(realtime_speedup * 60 // self.bedrock_batch_size)} batches/min"
            )

        # Batch processing estimates
        if self.bedrock_batch_mode == "batch":
            processing_mode = "Batch processing (forced)"
            cost_savings = "Up to 50%"
        elif self.bedrock_batch_mode == "auto":
            processing_mode = (
                f"Auto selection (batch for >= {self.bedrock_batch_threshold} records)"
            )
            cost_savings = "Up to 50% for large datasets"
        else:  # realtime
            processing_mode = "Real-time processing (forced)"
            cost_savings = "0%"

        return {
            "processing_mode": processing_mode,
            "batch_threshold": self.bedrock_batch_threshold,
            "cost_savings": cost_savings,
            "batch_timeout_hours": self.bedrock_batch_timeout_hours,
            "realtime_fallback": {
                "concurrency_mode": self.bedrock_concurrency_mode,
                "expected_speedup": f"{realtime_speedup:.1f}x",
                "throughput_estimate": realtime_throughput,
                "batch_size": self.bedrock_batch_size,
                "max_workers": self.bedrock_max_concurrent_workers
                if self.bedrock_concurrency_mode == "concurrent"
                else 1,
                "rate_limit": self.bedrock_rate_limit_per_second,
            },
            "production_ready": self.is_production_ready(),
        }

    def get_batch_processing_info(self) -> Dict[str, Any]:
        """
        Get detailed batch processing configuration information.

        Returns:
            Dict[str, Any]: Batch processing details and recommendations
        """
        return {
            "batch_configuration": self.batch_configuration,
            "cost_optimization": self.cost_optimization_info,
            "performance_estimate": self.get_performance_estimate(),
            "environment_variables": {
                k: v
                for k, v in self.bedrock_environment_variables.items()
                if k.startswith("BEDROCK_BATCH_")
            },
            "recommendations": {
                "optimal_for_datasets": f">= {self.bedrock_batch_threshold} records"
                if self.bedrock_batch_mode == "auto"
                else "All datasets",
                "cost_savings": self.cost_optimization_info["estimated_savings"],
                "fallback_available": True,
                "production_ready": self.is_production_ready(),
            },
        }
