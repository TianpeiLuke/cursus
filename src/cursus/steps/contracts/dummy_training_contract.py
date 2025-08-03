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
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz",
        "hyperparameters_s3_uri": "/opt/ml/processing/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"  # Matches specification logical name
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that processes a pretrained model.tar.gz by unpacking it, "
                "adding a hyperparameters.json file inside, and repacking it for downstream steps"
)
