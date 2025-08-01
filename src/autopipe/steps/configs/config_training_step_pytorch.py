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
    # Instance configuration
    training_instance_type: str = Field(default='ml.g5.12xlarge', description="Instance type for training job.")
    training_instance_count: int = Field(default=1, ge=1, description="Number of instances for training job.")
    training_volume_size: int = Field(default=30, ge=1, description="Volume size (GB) for training instances.")
    
    # Script mode configuration
    training_entry_point: str = Field(default='train.py', description="Entry point script for training.")

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(default="1.12.0", description="SageMaker PyTorch framework version.")
    py_version: str = Field(default="py38", description="Python version for the SageMaker PyTorch container.")

    # Hyperparameters object
    hyperparameters: Optional[ModelHyperparameters] = Field(None, description="Model hyperparameters")

    class Config(BasePipelineConfig.Config): # Inherit base config settings
        pass



    @model_validator(mode='after')
    def validate_field_lists(self) -> 'PyTorchTrainingConfig':
        """Validate field lists from hyperparameters"""
        if not self.hyperparameters:
            raise ValueError("hyperparameters must be provided")

        # Check if all fields in tab_field_list and cat_field_list are in full_field_list
        all_fields = set(self.hyperparameters.full_field_list)
        
        if not set(self.hyperparameters.tab_field_list).issubset(all_fields):
            raise ValueError("All fields in tab_field_list must be in full_field_list")
            
        if not set(self.hyperparameters.cat_field_list).issubset(all_fields):
            raise ValueError("All fields in cat_field_list must be in full_field_list")
        
        # Check if label_name and id_name are in full_field_list
        if self.hyperparameters.label_name not in all_fields:
            raise ValueError(
                f"label_name '{self.hyperparameters.label_name}' must be in full_field_list"
            )
            
        if self.hyperparameters.id_name not in all_fields:
            raise ValueError(
                f"id_name '{self.hyperparameters.id_name}' must be in full_field_list"
            )

        return self


    
    @field_validator('training_instance_type')
    @classmethod
    def _validate_sagemaker_training_instance_type(cls, v: str) -> str:
        valid_instances = [
            "ml.g4dn.16xlarge", 
            "ml.g5.12xlarge", 
            "ml.g5.16xlarge",
            "ml.p3.8xlarge", 
            "ml.m5.12xlarge",
            "ml.p3.16xlarge"
        ]
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v
