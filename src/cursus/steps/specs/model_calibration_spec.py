#!/usr/bin/env python
"""Step Specification for Model Calibration Step.

This file defines the step specification for the model calibration processing step,
including dependencies, outputs, and other metadata needed for pipeline integration.
"""

from ...core.base.specification_base import (
    StepSpecification,
    NodeType,
    DependencySpec,
    OutputSpec,
    DependencyType,
)
from ...registry.step_names import get_spec_step_type


def _get_model_calibration_contract():
    """Get the script contract for the ModelCalibration step.

    Returns:
        ScriptContract: The contract defining input/output paths and environment variables.
    """
    from ..contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT

    return MODEL_CALIBRATION_CONTRACT


MODEL_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("ModelCalibration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "XGBoostTraining",
                "XGBoostModelEval",
                "ModelEvaluation",
                "TrainingEvaluation",
                "CrossValidation",
            ],
            semantic_keywords=[
                "evaluation",
                "predictions",
                "scores",
                "validation",
                "test",
                "results",
                "model_output",
                "performance",
                "inference",
                "output_data",
                "prediction_results",
            ],
            data_type="S3Uri",
            description="Evaluation dataset with ground truth labels and model predictions",
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
            description="Dataset with calibrated probabilities",
        ),
    },
)
