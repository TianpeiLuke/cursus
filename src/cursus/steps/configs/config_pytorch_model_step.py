from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from ...core.base.config_base import BasePipelineConfig


class PyTorchModelStepConfig(
    BasePipelineConfig
):  # Renamed from PytorchModelCreationConfig for consistency
    """Configuration specific to the SageMaker PyTorch Model creation (for inference)."""

    # Renamed from inference_instance_type for consistency with builder
    instance_type: str = Field(
        default="ml.m5.large",
        description="Instance type for inference endpoint/transform job.",
    )
    # Renamed from inference_entry_point for consistency with builder
    entry_point: str = Field(
        default="inference.py", description="Entry point script for inference."
    )
    # source_dir is inherited from BasePipelineConfig, assumed to contain entry_point

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(
        default="1.12.0", description="SageMaker PyTorch framework version."
    )
    py_version: str = Field(
        default="py38",
        description="Python version for the SageMaker PyTorch container.",
    )

    # Accelerator type for inference
    accelerator_type: Optional[str] = Field(
        default=None, description="Accelerator type for inference endpoint."
    )

    # Model name
    model_name: Optional[str] = Field(
        default=None, description="Name for the SageMaker model."
    )

    # Tags for the model
    tags: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Tags for the model."
    )

    # Endpoint / Container specific settings
    initial_instance_count: int = Field(
        default=1,
        ge=1,
        description="Initial instance count for endpoint (used by EndpointConfig).",
    )
    container_startup_health_check_timeout: int = Field(
        default=300,
        ge=60,
        description="Container startup health check timeout (seconds).",
    )
    container_memory_limit: int = Field(
        default=6144, ge=1024, description="Container memory limit (MB)."
    )
    data_download_timeout: int = Field(
        default=900, ge=60, description="Model data download timeout (seconds)."
    )
    inference_memory_limit: int = Field(
        default=6144, ge=1024, description="Inference memory limit (MB)."
    )
    max_concurrent_invocations: int = Field(
        default=10, ge=1, description="Max concurrent invocations per instance."
    )  # Increased default
    max_payload_size: int = Field(
        default=6, ge=1, le=100, description="Max payload size (MB) for inference."
    )  # Increased range

    model_config = BasePipelineConfig.model_config

    @model_validator(mode="after")
    def validate_configuration(self) -> "PyTorchModelStepConfig":
        """Validate the complete configuration"""
        self._validate_memory_constraints()
        self._validate_timeouts()
        self._validate_entry_point()
        return self

    def _validate_memory_constraints(self) -> None:
        """Validate memory-related constraints"""
        if self.inference_memory_limit > self.container_memory_limit:
            raise ValueError(
                f"Inference memory limit ({self.inference_memory_limit}MB) cannot exceed "
                f"container memory limit ({self.container_memory_limit}MB)"
            )

    def _validate_timeouts(self) -> None:
        """Validate timeout-related configurations"""
        if self.container_startup_health_check_timeout > self.data_download_timeout:
            raise ValueError(
                "Container startup health check timeout should not exceed data download timeout"
            )

    def _validate_entry_point(self) -> None:
        """Validate entry point configuration (without file existence checks)"""
        # Removed file existence validation to improve configuration portability
        # File validation should happen at execution time in builders, not at config creation time
        
        # Only validate that source_dir is provided if entry_point is a relative path
        if self.entry_point and not self.entry_point.startswith("s3://"):
            if not self.source_dir:
                raise ValueError(
                    "source_dir must be provided when entry_point is a relative path"
                )

    @field_validator("inference_memory_limit")
    @classmethod
    def validate_memory_limits(cls, v: int, info) -> int:
        container_memory_limit = info.data.get("container_memory_limit")
        if container_memory_limit and v > container_memory_limit:
            raise ValueError(
                "Inference memory limit cannot exceed container memory limit"
            )
        return v

    @field_validator("instance_type")
    @classmethod
    def _validate_sagemaker_inference_instance_type(cls, v: str) -> str:
        if not v.startswith("ml."):
            raise ValueError(
                f"Invalid inference instance type: {v}. Must start with 'ml.'"
            )
        return v

    def get_model_name(self) -> str:
        """Generate a unique model name if not provided"""
        if self.model_name:
            return self.model_name

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.pipeline_name}-model-{timestamp}"

    def get_endpoint_config_name(self) -> str:
        """Generate endpoint configuration name"""
        return f"{self.get_model_name()}-config"

    def get_endpoint_name(self) -> str:
        """Generate endpoint name"""
        return f"{self.pipeline_name}-endpoint"

    # Model creation steps don't need script paths - they use the default implementation
    # from BasePipelineConfig which returns None/default_path
