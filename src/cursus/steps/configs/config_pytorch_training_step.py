from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from ...core.base.hyperparameters_base import ModelHyperparameters
from ...core.base.config_base import BasePipelineConfig


class PyTorchTrainingConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker PyTorch Training Step.
    This version is streamlined to work with specification-driven architecture.
    Input/output paths are now provided via step specifications and dependencies.
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    training_entry_point: str = Field(
        description="Entry point script for Pytorch training."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Instance configuration
    training_instance_type: str = Field(
        default="ml.g5.12xlarge", description="Instance type for training job."
    )
    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for training job."
    )
    training_volume_size: int = Field(
        default=30, ge=1, description="Volume size (GB) for training instances."
    )

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(
        default="1.12.0", description="SageMaker PyTorch framework version."
    )
    py_version: str = Field(
        default="py38",
        description="Python version for the SageMaker PyTorch container.",
    )

    # Hyperparameters handling configuration
    skip_hyperparameters_s3_uri: bool = Field(
        default=True, 
        description="Whether to skip hyperparameters_s3_uri channel during _get_inputs. "
                   "If True (default), hyperparameters are loaded from script folder. "
                   "If False, hyperparameters_s3_uri channel is created as TrainingInput."
    )

    # Hyperparameters object (optional for backward compatibility)
    hyperparameters: Optional[ModelHyperparameters] = Field(
        None, description="Model hyperparameters (optional when using external JSON files)"
    )

    model_config = BasePipelineConfig.model_config


    @field_validator("training_instance_type")
    @classmethod
    def _validate_sagemaker_training_instance_type(cls, v: str) -> str:
        valid_instances = [
            "ml.g4dn.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.16xlarge",
            "ml.p3.8xlarge",
            "ml.m5.12xlarge",
            "ml.p3.16xlarge",
        ]
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v
