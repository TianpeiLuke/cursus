"""
MIMS Packaging Step Specification.

This module defines the declarative specification for MIMS model packaging steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


# Import the contract at runtime to avoid circular imports
def _get_package_contract():
    from ..contracts.package_contract import PACKAGE_CONTRACT

    return PACKAGE_CONTRACT


# MIMS Packaging Step Specification
PACKAGE_SPEC = StepSpecification(
    step_type=get_spec_step_type("Package"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_package_contract(),
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=[
                "model",
                "artifacts",
                "trained",
                "output",
                "ModelArtifacts",
            ],
            data_type="S3Uri",
            description="Trained model artifacts to be packaged",
        ),
        DependencySpec(
            logical_name="inference_scripts_input",
            dependency_type=DependencyType.CUSTOM_PROPERTY,
            required=False,
            compatible_sources=["ProcessingStep", "ScriptStep"],
            semantic_keywords=["inference", "scripts", "code", "InferenceScripts"],
            data_type="String",
            description="Inference scripts and code for model deployment (can be local directory path or S3 URI)",
        ),
        DependencySpec(
            logical_name="calibration_model",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ModelCalibration"],
            semantic_keywords=[
                "calibration",
                "calibration_model",
                "calibrator",
                "probability",
                "calibrated",
            ],
            data_type="S3Uri",
            description="Calibration model and artifacts for probability calibration (optional)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="packaged_model",
            aliases=[
                "PackagedModel",
                "model_data",
            ],  # Added model_data alias to match XGBoost model dependency
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['packaged_model'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Packaged model ready for deployment",
        )
    ],
)
