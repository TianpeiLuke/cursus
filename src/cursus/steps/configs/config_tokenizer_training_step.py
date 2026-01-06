"""
Configuration for Tokenizer Training Processing Step.

This module defines the configuration class for the tokenizer training processing step,
which trains a BPE tokenizer optimized for customer name data with automatic vocabulary
size tuning to achieve target compression ratio.
"""

from typing import Dict, Any, Optional
from pydantic import Field, model_validator, field_validator, PrivateAttr
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class TokenizerTrainingConfig(ProcessingStepConfigBase):
    """
    Configuration for Tokenizer Training Processing Step.

    This class extends ProcessingStepConfigBase to include specific fields
    for training a BPE tokenizer on text data with compression tuning.

    The tokenizer training script uses CompressionBPETokenizer from cursus.processing.tokenizers
    module to train a tokenizer that matches the legacy OrderTextTokenizer implementation
    with improved compression tuning capabilities.
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    text_field: str = Field(
        description="Name of the text column in input parquet file for tokenizer training"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Script settings
    processing_entry_point: str = Field(
        default="tokenizer_training.py",
        description="Script for tokenizer training (entry point in source directory)",
    )

    job_type: str = Field(
        default="training",
        description="Type of job to perform. One of 'training', 'validation', 'testing', 'calibration'",
    )

    # PyTorch specific fields
    framework_version: str = Field(
        default="2.1.2", description="PyTorch framework version for processing"
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    # Tokenizer training parameters
    target_compression: float = Field(
        default=2.5,
        gt=0.0,
        description="Target compression ratio for tokenizer (e.g., 2.5 means compressing text to 40% of original token count)",
    )

    min_frequency: int = Field(
        default=25,
        ge=0,
        description="Minimum frequency threshold for BPE merges (tokens appearing less frequently are not merged)",
    )

    max_vocab_size: int = Field(
        default=50000,
        gt=0,
        description="Maximum vocabulary size limit for the tokenizer",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)

    # ===== Properties for Derived Fields =====

    @property
    def environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the tokenizer training script.

        Returns:
            Dictionary of environment variables required by the script
        """
        if self._environment_variables is None:
            self._environment_variables = {
                "TEXT_FIELD": self.text_field,
                "TARGET_COMPRESSION": str(self.target_compression),
                "MIN_FREQUENCY": str(self.min_frequency),
                "MAX_VOCAB_SIZE": str(self.max_vocab_size),
            }

        return self._environment_variables

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job type is one of the allowed values."""
        allowed_types = ["training", "validation", "testing", "calibration"]
        if v.lower() not in allowed_types:
            raise ValueError(f"job_type must be one of {allowed_types}, got {v}")
        return v.lower()

    @field_validator("text_field")
    @classmethod
    def validate_text_field(cls, v: str) -> str:
        """Validate text_field is not empty."""
        if not v or not v.strip():
            raise ValueError("text_field cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_tokenizer_config(self) -> "TokenizerTrainingConfig":
        """
        Validate tokenizer training configuration.

        Ensures all tokenizer parameters are within valid ranges and
        the configuration is consistent.
        """
        # Validate compression ratio is reasonable
        if self.target_compression > 10.0:
            logger.warning(
                f"target_compression={self.target_compression} is unusually high. "
                "Typical values are between 1.5 and 4.0"
            )

        # Validate min_frequency is reasonable
        if self.min_frequency > 1000:
            logger.warning(
                f"min_frequency={self.min_frequency} is very high. "
                "This may result in a very small vocabulary"
            )

        # Validate max_vocab_size is reasonable
        if self.max_vocab_size < 1000:
            logger.warning(
                f"max_vocab_size={self.max_vocab_size} is very small. "
                "This may result in poor tokenization quality"
            )

        return self
