"""
SOPA Instruction Tuning Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker SOPA (Stage Of Pre-training
Alignment) Stage 2 instruction fine-tuning steps using a self-contained design where
derived fields are private with read-only properties.

Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)

Note: Most training/model hyperparameters (feature_columns, input_size, latent_size,
batch_size, lr, epochs, etc.) are defined in SOPAInstructionTuningHyperparameters
and serialized to a JSON file. They are NOT duplicated here. The builder loads them
from the JSON at pipeline construction time.

Only infrastructure settings, filenames, and the task selector live in this config.
"""

from pydantic import Field, model_validator, field_validator, PrivateAttr
from typing import Optional, Dict, Any, List

from ...core.base.config_base import BasePipelineConfig


class SOPAInstructionTuningConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker SOPA Instruction Tuning Step.

    This configuration collects user inputs for SOPA Stage 2 instruction fine-tuning,
    which fine-tunes a BLIP2-based model (Q-Former + Phi-3 LLM) for tabular-to-text
    instruction following tasks. Input/output paths are provided via step specifications
    and dependencies.

    Most training/model hyperparameters are defined separately in
    SOPAInstructionTuningHyperparameters and loaded from the hyperparameters JSON
    by the builder. This config only contains:
    - Infrastructure settings (instance type, count, volume)
    - Task selector
    - Filename overrides (training data, Q-Former checkpoint, tabular encoder checkpoint)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    training_entry_point: str = Field(
        description="Entry point script for SOPA instruction tuning (e.g., 'SOPA_instruction_tuning.py')."
    )

    task: str = Field(
        description="Instruction tuning task name. One of: 'return_risk', 'customer_risk', 'refund_decision'. "
        "Each task uses corresponding instruct_input_{task} and instruct_output_{task} columns from the training CSV."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # --- Instance configuration ---

    training_instance_type: str = Field(
        default="ml.p4d.24xlarge",
        description="Instance type for SOPA training job. "
        "GPU instances required for PyTorch + LLM training. "
        "P4d (A100, recommended): ml.p4d.24xlarge (8x A100, 320GB). "
        "P5 (H100): ml.p5.48xlarge (8x H100, 640GB). "
        "G5 (A10G): ml.g5.12xlarge (4x A10G, 96GB).",
    )

    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for SOPA training job."
    )

    training_volume_size: int = Field(
        default=50, ge=1, description="Volume size (GB) for training instances."
    )

    # --- Filename overrides ---
    # These are not in the hyperparameters JSON because they are config-level concerns
    # (which file to load from which directory).

    training_data_filename: str = Field(
        default="training_data.csv",
        description="Filename of the training CSV within the data_path directory. "
        "Allows customizing which file is loaded from the input channel.",
    )

    pretrained_qformer: str = Field(
        default="qformer",
        description="Name of the pre-trained Q-Former checkpoint file within the stage1_qformer_path directory.",
    )

    tabular_encoder_filename: str = Field(
        default="autoencoder_model",
        description="Filename of the tabular encoder checkpoint within the tabular_encoder_path directory.",
    )

    # --- Hyperparameters dict (populated at config generation time) ---

    hyperparameters_dict: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training/model hyperparameters from SOPAInstructionTuningHyperparameters, "
        "embedded at config generation time via model_dump(). The builder reads this "
        "dict directly instead of loading from S3. Keys: feature_columns, input_size, "
        "latent_size, batch_size, lr, epochs, etc.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    _hyperparameter_file: Optional[str] = PrivateAttr(default=None)
    _checkpoint_s3_uri: Optional[str] = PrivateAttr(default=None)

    model_config = BasePipelineConfig.model_config

    # Public read-only properties for derived fields

    @property
    def hyperparameter_file(self) -> str:
        """Get hyperparameter file path."""
        if self._hyperparameter_file is None:
            self._hyperparameter_file = (
                f"{self.pipeline_s3_loc}/hyperparameters/sopa_hyperparameters_{self.region}.json"
            )
        return self._hyperparameter_file

    @property
    def checkpoint_s3_uri(self) -> str:
        """Get S3 URI for model checkpoints."""
        if self._checkpoint_s3_uri is None:
            self._checkpoint_s3_uri = f"{self.pipeline_s3_loc}/checkpoints"
        return self._checkpoint_s3_uri

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["hyperparameter_file"] = self.hyperparameter_file
        data["checkpoint_s3_uri"] = self.checkpoint_s3_uri
        return data

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "SOPAInstructionTuningConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize SOPA instruction tuning-specific derived fields
        self._hyperparameter_file = (
            f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_sopa_hyperparameters.json"
        )
        self._checkpoint_s3_uri = f"{self.pipeline_s3_loc}/checkpoints"

        return self

    # ===== Validators =====

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        """Validate task is one of allowed instruction tuning tasks."""
        allowed = {"return_risk", "customer_risk", "refund_decision"}
        if v not in allowed:
            raise ValueError(
                f"task must be one of {allowed}, got '{v}'. "
                f"Each task uses corresponding instruct_input_{{task}} and instruct_output_{{task}} "
                f"columns from the training CSV."
            )
        return v

    @field_validator("training_instance_type")
    @classmethod
    def _validate_pytorch_instance_type(cls, v: str) -> str:
        """Validate instance type is suitable for PyTorch + LLM training."""
        # GPU instances required for SOPA training (LLM + Q-Former)
        valid_gpu_instances = [
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p3dn.24xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.12xlarge",
            "ml.g4dn.16xlarge",
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.12xlarge",
            "ml.g5.16xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
            "ml.p5.48xlarge",
        ]
        if v not in valid_gpu_instances:
            raise ValueError(
                f"Invalid training instance type for SOPA instruction tuning: {v}. "
                f"GPU instances are required for LLM + Q-Former training. "
                f"Must be one of: {', '.join(valid_gpu_instances[:5])}... "
                f"Recommended: ml.p4d.24xlarge (8x A100)."
            )
        return v

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the SOPA instruction tuning script.

        The SOPA script uses argparse for all configuration and does not
        require environment variables.

        Returns:
            Dict[str, str]: Empty dictionary (no env vars per script contract)
        """
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )
        return env_vars

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include SOPA instruction tuning-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and SOPA-specific fields.

        Note: Training/model hyperparameters (feature_columns, input_size, batch_size,
        lr, etc.) are NOT included here — they come from the hyperparameters JSON file.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (BasePipelineConfig)
        base_fields = super().get_public_init_fields()

        # Add SOPA instruction tuning-specific fields (Tier 1 and Tier 2)
        training_fields = {
            # Tier 1
            "training_entry_point": self.training_entry_point,
            "task": self.task,
            # Tier 2 - Instance
            "training_instance_type": self.training_instance_type,
            "training_instance_count": self.training_instance_count,
            "training_volume_size": self.training_volume_size,
            # Tier 2 - Filenames
            "training_data_filename": self.training_data_filename,
            "pretrained_qformer": self.pretrained_qformer,
            "tabular_encoder_filename": self.tabular_encoder_filename,
            # Tier 2 - Hyperparameters (embedded from SOPAInstructionTuningHyperparameters)
            "hyperparameters_dict": self.hyperparameters_dict,
        }

        # Combine base fields and training fields (training fields take precedence if overlap)
        init_fields = {**base_fields, **training_fields}

        return init_fields
