"""
Contract for dummy training step that processes a pretrained model.tar.gz with hyperparameters.

This script contract defines the expected input and output paths, environment variables,
and framework requirements for the DummyTraining step, which processes a pretrained model
by adding hyperparameters.json to it for downstream packaging and payload steps.
"""

from ...core.base.contract_base import ScriptContract

DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        # Empty - SOURCE node reads from source directory only
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"  # Matches specification logical name
    },
    expected_arguments={
        # No expected arguments - using hard-coded source directory paths
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={"boto3": ">=1.26.0", "pathlib": ">=1.0.0"},
    description="Contract for dummy training SOURCE step that packages model.tar.gz and hyperparameters.json from source directory",
)
