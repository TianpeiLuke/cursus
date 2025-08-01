"""
Configuration for Risk Table Mapping Processing Step.

This module defines the configuration class for the risk table mapping processing step,
which is responsible for creating and applying risk tables for categorical features.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import Field, model_validator, field_validator
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from ...core.base.hyperparameters_base import ModelHyperparameters

logger = logging.getLogger(__name__)

class RiskTableMappingConfig(ProcessingStepConfigBase):
    """
    Configuration for Risk Table Mapping Processing Step.
    
    This class extends ProcessingStepConfigBase to include specific fields 
    for risk table mapping, including categorical fields and job type.
    """
    # Script settings
    processing_entry_point: str = Field(
        default="risk_table_mapping.py",
        description="Script for risk table mapping"
    )
    
    # Job type for the processing script
    job_type: str = Field(
        default="training",
        description="Type of job to perform. One of 'training', 'validation', 'testing', 'calibration'"
    )

    # Hyperparameter settings
    cat_field_list: List[str] = Field(
        default=[],
        description="List of categorical fields to apply risk table mapping to"
    )
    
    label_name: str = Field(
        default="target", 
        description="Name of the target/label column"
    )
    
    smooth_factor: float = Field(
        default=0.01, 
        description="Smoothing factor for risk table calculation"
    )
    
    count_threshold: int = Field(
        default=5, 
        description="Minimum count threshold for risk table calculation"
    )
    
    # Hyperparameters S3 URI for uploading generated hyperparameters.json
    hyperparameters_s3_uri: Optional[str] = Field(
        default=None,
        description="S3 URI prefix for uploading hyperparameters.json"
    )
    
    # Model hyperparameters (to allow integration with ModelHyperparameters if needed)
    hyperparameters: Optional[ModelHyperparameters] = Field(
        default=None,
        description="Optional model hyperparameters for advanced configuration"
    )

    class Config(ProcessingStepConfigBase.Config):
        """Configuration settings for the model."""
        arbitrary_types_allowed = True
    
    @field_validator('job_type')
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job type is one of the allowed values."""
        allowed_types = ["training", "validation", "testing", "calibration"]
        if v.lower() not in allowed_types:
            raise ValueError(f"job_type must be one of {allowed_types}, got {v}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_hyperparameters(self) -> 'RiskTableMappingConfig':
        """
        Ensure either direct hyperparameters or ModelHyperparameters are properly set.
        If ModelHyperparameters is provided, extract relevant values.
        """
        if self.hyperparameters is not None:
            # If ModelHyperparameters is provided, update our fields
            if not self.cat_field_list and hasattr(self.hyperparameters, 'cat_field_list'):
                self.cat_field_list = self.hyperparameters.cat_field_list
                
            if self.label_name == "target" and hasattr(self.hyperparameters, 'label_name'):
                self.label_name = self.hyperparameters.label_name
                
        return self
    
    def get_hyperparameters_dict(self) -> Dict[str, Any]:
        """
        Get hyperparameters as a dictionary for serialization.
        
        Returns:
            Dict containing hyperparameters for risk table mapping
        """
        return {
            "cat_field_list": self.cat_field_list,
            "label_name": self.label_name,
            "smooth_factor": self.smooth_factor,
            "count_threshold": self.count_threshold
        }
