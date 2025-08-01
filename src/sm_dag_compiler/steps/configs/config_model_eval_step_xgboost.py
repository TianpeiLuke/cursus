"""
Model Evaluation Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the XGBoost model evaluation step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, PrivateAttr
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from ..hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters

# Import the script contract
from ..contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class XGBoostModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model evaluation step with self-contained derivation logic.
    
    This class defines the configuration parameters for the XGBoost model evaluation step,
    which calculates evaluation metrics for trained models. This is crucial for
    measuring model performance and comparing different models or configurations.
    
    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    hyperparameters: XGBoostModelHyperparameters = Field(
        ...,
        description="XGBoost model hyperparameters config, including id_name, label_name, field lists, etc."
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    
    processing_entry_point: str = Field(
        default="model_evaluation_xgb.py",
        description="Entry point script for model evaluation."
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test')."
    )

    eval_metric_choices: List[str] = Field(
        default_factory=lambda: ["auc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute"
    )

    # XGBoost specific fields
    xgboost_framework_version: str = Field(
        default="1.5-1",
        description="XGBoost framework version for processing"
    )

    # For most processing jobs, we want to use a larger instance
    use_large_processing_instance: bool = Field(
        default=True,
        description="Whether to use large instance type for processing"
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access
    
    # Currently no derived fields specific to model evaluation
    # beyond what's inherited from the ProcessingStepConfigBase class

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'XGBoostModelEvalConfig':
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        
        # No additional derived fields to initialize for now
        
        return self
        
    @model_validator(mode='after')
    def validate_eval_config(self) -> 'XGBoostModelEvalConfig':
        """Additional validation specific to evaluation configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("evaluation step requires a processing_entry_point")
            
        valid_job_types = {"training", "calibration", "validation", "test"}
        if self.job_type not in valid_job_types:
            raise ValueError(f"job_type must be one of {valid_job_types}, got '{self.job_type}'")
        
        if not isinstance(self.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("hyperparameters must be an instance of XGBoostModelHyperparameters")
        
        # Validate required fields from script contract
        contract = self.get_script_contract()
        
        # Check required environment variables from contract
        if not self.hyperparameters.id_name:
            raise ValueError("hyperparameters.id_name must be provided (required by model evaluation contract)")
            
        if not self.hyperparameters.label_name:
            raise ValueError("hyperparameters.label_name must be provided (required by model evaluation contract)")
            
        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model evaluation script.
        
        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = super().get_environment_variables() if hasattr(super(), "get_environment_variables") else {}
        
        # Add model evaluation specific environment variables
        env_vars.update({
            "ID_FIELD": self.hyperparameters.id_name,
            "LABEL_FIELD": self.hyperparameters.label_name,
            "JOB_TYPE": self.job_type
        })
        
        # Add evaluation metrics if available
        if hasattr(self.hyperparameters, 'eval_metric_list') and self.hyperparameters.eval_metric_list:
            env_vars["EVAL_METRICS"] = ",".join(self.hyperparameters.eval_metric_list)
        
        # Add eval metric choices
        if self.eval_metric_choices:
            env_vars["EVAL_METRIC_CHOICES"] = ",".join(self.eval_metric_choices)
        
        return env_vars
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The model evaluation script contract
        """
        return MODEL_EVALUATION_CONTRACT
        
    def get_script_path(self) -> str:
        """
        Get script path for XGBoost model evaluation.
        
        SPECIAL CASE: Unlike other step configs, XGBoostModelEvalStepBuilder provides 
        processing_source_dir and processing_entry_point directly to the processor.run() 
        method separately. Therefore, this method should return only the entry point name 
        without combining with source_dir.
        
        Returns:
            Script entry point name (without source_dir)
        """
        # Determine which entry point to use
        entry_point = None
        
        # First priority: Use processing_entry_point if provided
        if self.processing_entry_point:
            entry_point = self.processing_entry_point
        # Second priority: Use contract entry point
        elif hasattr(self, 'script_contract') and self.script_contract and hasattr(self.script_contract, 'entry_point'):
            entry_point = self.script_contract.entry_point
        
        # Return just the entry point name without combining with source directory
        # This is important for XGBoostModelEvalStepBuilder which handles paths differently
        return entry_point
        
    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include evaluation-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and evaluation-specific fields.
        
        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()
        
        # Add model evaluation specific fields
        eval_fields = {
            # Tier 1 - Essential User Inputs
            'hyperparameters': self.hyperparameters,
            
            # Tier 2 - System Inputs with Defaults
            'processing_entry_point': self.processing_entry_point,
            'job_type': self.job_type,
            'xgboost_framework_version': self.xgboost_framework_version,
            'use_large_processing_instance': self.use_large_processing_instance
        }
        
        # Add eval_metric_choices if set to non-default value
        default_metrics = ["auc", "average_precision", "f1_score"]
        if self.eval_metric_choices != default_metrics:
            eval_fields['eval_metric_choices'] = self.eval_metric_choices
        
        # Combine base fields and evaluation fields (evaluation fields take precedence if overlap)
        init_fields = {**base_fields, **eval_fields}
        
        return init_fields
