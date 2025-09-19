"""
Configuration for DummyTraining SOURCE step.

This module defines the configuration class for the DummyTraining step,
which is a SOURCE node that packages model.tar.gz and hyperparameters.json
from the source directory and makes them available for downstream steps.
"""

from pydantic import Field, model_validator
from typing import TYPE_CHECKING

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class DummyTrainingConfig(ProcessingStepConfigBase):
    """
    Configuration for DummyTraining SOURCE step.

    This step is a SOURCE node that reads model.tar.gz and hyperparameters.json
    from the source directory, packages them together, and makes them available
    for downstream packaging and registration steps.
    
    Expected source directory structure:
    source_dir/
    ├── dummy_training.py          # Main training script
    ├── models/                    # Model directory
    │   └── model.tar.gz          # Pre-trained model artifacts
    └── hyperparams/              # Hyperparameters directory
        └── hyperparameters.json  # Generated hyperparameters file
    """

    # Override with specific default for this step
    processing_entry_point: str = Field(
        default="dummy_training.py",
        description="Entry point script for dummy training SOURCE step.",
    )

    model_config = ProcessingStepConfigBase.model_config

    @model_validator(mode="after")
    def validate_config(self) -> "DummyTrainingConfig":
        """
        Validate configuration for SOURCE node.

        For SOURCE nodes, we only validate basic configuration attributes.
        File existence is checked at runtime, not configuration time.

        Returns:
            Self with validated configuration
        """
        # Basic validation - entry point is required for SOURCE nodes
        if not self.processing_entry_point:
            raise ValueError("DummyTraining SOURCE step requires a processing_entry_point")

        # Validate script contract - ensure it matches SOURCE node expectations
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # For SOURCE nodes, contract should have empty input paths
        if contract.expected_input_paths:
            raise ValueError(
                f"SOURCE node contract should have empty input paths, but found: {list(contract.expected_input_paths.keys())}"
            )

        # Ensure we have the required output path
        if "model_input" not in contract.expected_output_paths:
            raise ValueError(
                "Script contract missing required output path: model_input"
            )

        return self

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The DummyTraining script contract
        """
        return DUMMY_TRAINING_CONTRACT
