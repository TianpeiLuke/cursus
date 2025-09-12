#!/usr/bin/env python
"""
Model Calibration Calibration Step Specification.

This module defines the declarative specification for model calibration steps
specifically for calibration data, including dependencies and outputs.
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


# Model Calibration Calibration Step Specification
MODEL_CALIBRATION_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("ModelCalibration", "calibration"),
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
                "calibration",
                "evaluation",
                "predictions",
                "scores",
                "results",
                "model_output",
                "performance",
                "calibration_evaluation",
                "calibration_predictions",
                "calibration_results",
            ],
            data_type="S3Uri",
            description="Calibration evaluation dataset with ground truth labels and model predictions",
        )
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=[
                "calibration_model",
                "calibration_artifacts",
                "probability_calibration",
                "calibrator",
                "score_transformer",
                "probability_adjustment",
            ],
            data_type="S3Uri",
            description="Calibration mapping and artifacts",
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            aliases=[
                "calibration_metrics",
                "reliability_metrics",
                "probability_metrics",
                "calibration_performance",
                "calibration_evaluation",
            ],
            data_type="S3Uri",
            description="Calibration quality metrics and visualizations",
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            aliases=[
                "calibrated_predictions",
                "calibrated_probabilities",
                "probability_scores",
                "adjusted_predictions",
                "calibrated_outputs",
            ],
            data_type="S3Uri",
            description="Calibration dataset with calibrated probabilities",
        ),
    },
)
