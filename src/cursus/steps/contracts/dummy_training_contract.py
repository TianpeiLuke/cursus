"""
Contract for dummy training step with flexible input modes.

This script contract defines the expected input and output paths, environment variables,
and framework requirements for the DummyTraining step. This step can operate in two modes:

1. INTERNAL mode: Accepts optional inputs from previous steps or S3
   - model_artifacts_input: Optional model.tar.gz from previous training step
   - hyperparameters_s3_uri: Optional hyperparameters.json from input channel

2. SOURCE mode (fallback): Reads from source directory when inputs not provided
   - Reads model.tar.gz from source_dir/models/
   - Reads hyperparameters.json from source_dir/hyperparams/ or code directory

The step processes the model by adding hyperparameters.json to model.tar.gz for
downstream packaging and payload steps.
"""

from ...core.base.contract_base import ScriptContract

DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "model_artifacts_input": "/opt/ml/processing/input/model_artifacts_input",  # OPTIONAL - for flexible model input
        "hyperparameters_s3_uri": "/opt/ml/processing/input/hyperparameters_s3_uri",  # OPTIONAL - for flexible hyperparameters input
    },
    expected_output_paths={
        "model_output": "/opt/ml/processing/output/model"  # Renamed from model_input for consistency
    },
    expected_arguments={
        # No expected arguments - paths provided via input_paths dict
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={"boto3": ">=1.26.0", "pathlib": ">=1.0.0"},
    description="Contract for dummy training INTERNAL step with flexible input modes. Can accept model and hyperparameters from previous steps or fall back to source directory.",
)
