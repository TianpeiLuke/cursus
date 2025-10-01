"""
Model Metrics Computation Step Specification.

This module defines the declarative specification for model metrics computation steps,
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.model_metrics_computation_contract import MODEL_METRICS_COMPUTATION_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_model_metrics_computation_contract():
    from ..contracts.model_metrics_computation_contract import MODEL_METRICS_COMPUTATION_CONTRACT

    return MODEL_METRICS_COMPUTATION_CONTRACT


# Model Metrics Computation Step Specification
MODEL_METRICS_COMPUTATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("ModelMetricsComputation"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_metrics_computation_contract(),
    dependencies={
        "eval_output": DependencySpec(
            logical_name="eval_output",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "XGBoostModelInference",
                "XGBoostModelEval", 
                "PyTorchModelInference",
                "TabularPreprocessing",
                "CradleDataLoading",
                "RiskTableMapping",
                "CurrencyConversion",
            ],
            semantic_keywords=[
                "predictions",
                "inference",
                "evaluation",
                "results",
                "data",
                "processed",
                "eval_output",
                "prediction_data",
                "inference_results",
                "model_predictions",
                "metrics_input",
            ],
            data_type="S3Uri",
            description="Prediction data with labels and probabilities for metrics computation",
        ),
    },
    outputs={
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "metrics",
                "performance_metrics",
                "evaluation_metrics",
                "model_metrics",
                "metrics_results",
            ],
            description="Comprehensive model performance metrics (AUC, precision, recall, F1, domain metrics)",
        ),
        "plots_output": OutputSpec(
            logical_name="plots_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['plots_output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "plots",
                "visualizations",
                "charts",
                "performance_plots",
                "metrics_plots",
                "roc_curves",
                "pr_curves",
            ],
            description="Performance visualization plots (ROC curves, PR curves, distributions, threshold analysis)",
        ),
    },
)
