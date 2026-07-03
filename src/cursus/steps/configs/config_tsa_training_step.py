"""
TSA Training Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker TSA (Temporal Sequence Attention)
Training steps using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, field_validator, PrivateAttr
from typing import Optional, Dict, Any
from pathlib import Path

from ...core.base.config_base import BasePipelineConfig
from ...core.base.hyperparameters_base import ModelHyperparameters


class TSATrainingConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker TSA Training Step.
    This version is streamlined to work with specification-driven architecture.
    Input/output paths are now provided via step specifications and dependencies.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    training_entry_point: str = Field(
        description="Entry point script for TSA training (e.g., 'tsa_training.py')."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Instance configuration
    training_instance_type: str = Field(
        default="ml.p4d.24xlarge",
        description="Instance type for TSA training job. "
        "GPU instances recommended for PyTorch neural network training. "
        "P4d (A100, recommended): ml.p4d.24xlarge (8x A100, 320GB) - Best availability, excellent performance, TF32 support. "
        "P5 (H100): ml.p5.48xlarge (8x H100, 640GB) - Fastest with FP8/TF32 but limited availability. "
        "P3 (V100): ml.p3.2xlarge (1x V100, 16GB) - Cost-effective for development. "
        "G5 (A10G): ml.g5.12xlarge (4x A10G, 96GB) - Good price/performance. "
        "Multi-GPU instances automatically enable DDP (DistributedDataParallel).",
    )

    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for TSA training job."
    )

    training_volume_size: int = Field(
        default=50, ge=1, description="Volume size (GB) for training instances."
    )

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(
        default="2.1.0", description="SageMaker PyTorch framework version."
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    ca_repository_arn: str = Field(
        default="arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi",
        description="CodeArtifact repository ARN for secure PyPI access. Only used when use_secure_pypi=True.",
    )

    # KMS encryption keys for GPU training instances
    output_kms_key: Optional[str] = Field(
        default=None,
        description="KMS key ARN for encrypting training output artifacts (model.tar.gz). "
        "If None, uses pipeline-level KMS key. Set this when GPU instances "
        "require a different KMS key than processing steps.",
    )

    volume_kms_key: Optional[str] = Field(
        default=None,
        description="KMS key ARN for encrypting EBS volumes on training instances. "
        "If None, uses default encryption. Required for GPU instances (P4d, P5) "
        "that may be in a different security domain than CPU processing steps.",
    )

    # Hyperparameters handling configuration
    skip_hyperparameters_s3_uri: bool = Field(
        default=True,
        description="Whether to skip hyperparameters_s3_uri channel during _get_inputs. "
        "If True (default), hyperparameters are loaded from script folder. "
        "If False, hyperparameters_s3_uri channel is created as TrainingInput.",
    )

    # Performance optimization settings (Phase 1-3 optimizations)
    # Note: DDP (DistributedDataParallel) is automatically enabled on multi-GPU instances.
    # SageMaker sets LOCAL_RANK environment variable, which the training script detects.
    use_amp: bool = Field(
        default=True,
        description="Enable mixed precision training (AMP) for 2-3x speedup. "
        "Automatically disabled on CPU. Uses torch.cuda.amp for FP16/FP32 precision.",
    )

    enable_tf32: bool = Field(
        default=True,
        description="Enable TensorFloat-32 (TF32) for Ampere+ GPUs (A100, H100). "
        "Provides ~2x speedup over FP32 with minimal accuracy loss. "
        "Automatically enabled on compatible GPUs (compute capability >= 8.0). "
        "No effect on older GPUs (V100, T4).",
    )

    enable_fp8: bool = Field(
        default=False,
        description="Enable FP8 (8-bit float) precision for H100 GPUs only. "
        "Provides ~2x speedup over FP16 but requires H100 (compute capability 9.0). "
        "Falls back gracefully on older GPUs. Experimental feature.",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Gradient accumulation steps for simulating larger batch sizes. "
        "Effective batch size = batch_size * accumulation_steps * num_gpus. "
        "Useful for training with limited GPU memory.",
    )

    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        description="Maximum gradient norm for gradient clipping. "
        "Prevents exploding gradients and improves training stability.",
    )

    checkpoint_freq: int = Field(
        default=20,
        ge=1,
        description="Save model checkpoint every N epochs. "
        "Reduced from 10 to 20 to minimize I/O overhead during training.",
    )

    # Semi-supervised learning support
    # Note: Maximum runtime is configured at SageMaker job level via max_run parameter,
    # not in training configuration.
    job_type: Optional[str] = Field(
        default=None,
        description=(
            "Training job type for semi-supervised learning workflows:\n"
            "• None (default): Standard supervised learning - no step name suffix\n"
            "• 'pretrain': SSL pretraining phase - adds '-Pretrain' suffix\n"
            "• 'finetune': SSL fine-tuning phase - adds '-Finetune' suffix"
        ),
    )

    # Hyperparameters object (optional for backward compatibility)
    hyperparameters: Optional[ModelHyperparameters] = Field(
        None,
        description="Model hyperparameters (optional when using external JSON files)",
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
            self._hyperparameter_file = f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_tsa_hyperparameters.json"
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
    def initialize_derived_fields(self) -> "TSATrainingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize TSA training-specific derived fields
        self._hyperparameter_file = f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_tsa_hyperparameters.json"
        self._checkpoint_s3_uri = f"{self.pipeline_s3_loc}/checkpoints"

        return self

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate job_type is one of allowed values."""
        if v is None:
            return None  # Standard supervised learning

        allowed = {"pretrain", "finetune"}
        if v not in allowed:
            raise ValueError(
                f"job_type must be None (standard) or one of {allowed}, got '{v}'. "
                f"Use None for standard training, 'pretrain' for SSL pretraining, "
                f"'finetune' for SSL fine-tuning."
            )
        return v

    @field_validator("training_instance_type")
    @classmethod
    def _validate_pytorch_instance_type(cls, v: str) -> str:
        """Validate instance type is suitable for PyTorch training."""
        # GPU instances recommended for PyTorch/TSA training
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
        # CPU instances (for testing or small models)
        valid_cpu_instances = [
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.large",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ]
        valid_instances = valid_gpu_instances + valid_cpu_instances
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type for PyTorch TSA: {v}. "
                f"Must be one of: {', '.join(valid_instances[:10])}... "
                f"(GPU instances recommended for neural network training)"
            )
        return v

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the TSA training script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add TSA training specific environment variables
        env_vars.update(
            {
                "REGION": self.region,
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
                # Phase 1 Optimizations - Mixed Precision Training
                "USE_AMP": str(self.use_amp).lower(),
                "ENABLE_TF32": str(self.enable_tf32).lower(),
                "ENABLE_FP8": str(self.enable_fp8).lower(),
                # Phase 2 Optimizations - Gradient Accumulation & Clipping
                "GRADIENT_ACCUMULATION_STEPS": str(self.gradient_accumulation_steps),
                "MAX_GRAD_NORM": str(self.max_grad_norm),
                # Phase 3 Optimizations - Reduced I/O & Multi-GPU
                "CHECKPOINT_FREQ": str(self.checkpoint_freq),
            }
        )

        return env_vars

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include TSA training-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and TSA training-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (BasePipelineConfig)
        base_fields = super().get_public_init_fields()

        # Add TSA training-specific fields (Tier 1 and Tier 2)
        training_fields = {
            "training_entry_point": self.training_entry_point,
            "training_instance_type": self.training_instance_type,
            "training_instance_count": self.training_instance_count,
            "training_volume_size": self.training_volume_size,
            "framework_version": self.framework_version,
            "py_version": self.py_version,
            "ca_repository_arn": self.ca_repository_arn,
            "skip_hyperparameters_s3_uri": self.skip_hyperparameters_s3_uri,
            "use_secure_pypi": self.use_secure_pypi,
            "job_type": self.job_type,
        }

        # Add hyperparameters if present (use model_dump for Pydantic models)
        if self.hyperparameters is not None:
            training_fields["hyperparameters"] = self.hyperparameters.model_dump()

        # Combine base fields and training fields (training fields take precedence if overlap)
        init_fields = {**base_fields, **training_fields}

        return init_fields
