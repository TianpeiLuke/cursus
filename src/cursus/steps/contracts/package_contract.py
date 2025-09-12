"""
MIMS Package Script Contract

Defines the contract for the MIMS packaging script that packages model artifacts
and inference scripts into a deployable tar.gz file.
"""

from ...core.base.contract_base import ScriptContract

PACKAGE_CONTRACT = ScriptContract(
    entry_point="package.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "inference_scripts_input": "/opt/ml/processing/input/script",
        "calibration_model": "/opt/ml/processing/input/calibration",
    },
    expected_output_paths={"packaged_model": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No required environment variables for this script
    ],
    optional_env_vars={},
    framework_requirements={
        "python": ">=3.7"
        # Uses only standard library modules: shutil, tarfile, pathlib, logging, os
    },
    description="""
    MIMS packaging script that:
    1. Extracts model artifacts from input model directory or model.tar.gz
    2. Includes calibration model if available
    3. Copies inference scripts to code directory
    4. Creates a packaged model.tar.gz file for deployment
    4. Provides detailed logging of the packaging process
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts (files or model.tar.gz)
    - /opt/ml/processing/input/script: Inference scripts to include
    - /opt/ml/processing/input/calibration: Optional calibration model artifacts
    
    Output Structure:
    - /opt/ml/processing/output/model.tar.gz: Packaged model ready for deployment
    """,
)
