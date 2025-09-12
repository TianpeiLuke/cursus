#!/usr/bin/env python
"""
Model Calibration Validation Step Specification.

This module defines the declarative specification for model calibration steps
specifically for validation data, including dependencies and outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type_with_job_type


# Import the contract at runtime to avoid circular imports
def _get_model_calibration_contract():
    from ..contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT

    return MODEL_CALIBRATION_CONTRACT


# Model Calibration Validation Step Specification
MODEL_CALIBRATION_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("ModelCalibration", "validation"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "PytorchTraining",
                "XGBoostTraining",
                "XGBoostModelEval",
                "ModelEvaluation",
                "TrainingEvaluation",
                "CrossValidation",
            ],
            semantic_keywords=[
                "validation",
                "val",
                "evaluate",
                "evaluation",
                "predictions",
                "scores",
                "results",
                "model_output",
                "performance",
                "validation_evaluation",
                "val_predictions",
                "validation_results",
            ],
            data_type="S3Uri",
            description="Validation evaluation dataset with ground truth labels and model predictions",
        )
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=[
                "calibration_model",
                "validation_calibration",
                "val_calibrator",
                "validation_probability_calibration",
                "validation_calibration_artifacts",
            ],
            data_type="S3Uri",
            description="Validation calibration mapping and artifacts",
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            aliases=[
                "validation_calibration_metrics",
                "validation_reliability_metrics",
                "validation_probability_metrics",
                "validation_calibration_performance",
                "validation_calibration_evaluation",
            ],
            data_type="S3Uri",
            description="Validation calibration quality metrics and visualizations",
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            aliases=[
                "validation_calibrated_predictions",
                "validation_calibrated_probabilities",
                "validation_probability_scores",
                "validation_adjusted_predictions",
                "validation_calibrated_outputs",
            ],
            data_type="S3Uri",
            description="Validation dataset with calibrated probabilities",
        ),
    },
)
