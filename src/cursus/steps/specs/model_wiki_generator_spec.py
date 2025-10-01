"""
Model Wiki Generator Step Specification.

This module defines the declarative specification for model wiki generator steps,
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
    from ..contracts.model_wiki_generator_contract import MODEL_WIKI_GENERATOR_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_model_wiki_generator_contract():
    from ..contracts.model_wiki_generator_contract import MODEL_WIKI_GENERATOR_CONTRACT

    return MODEL_WIKI_GENERATOR_CONTRACT


# Model Wiki Generator Step Specification
MODEL_WIKI_GENERATOR_SPEC = StepSpecification(
    step_type=get_spec_step_type("ModelWikiGenerator"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_wiki_generator_contract(),
    dependencies={
        "metrics_output": DependencySpec(
            logical_name="metrics_output",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "ModelMetricsComputation",
                "XGBoostModelEval",
                "XGBoostModelInference",
                "PyTorchModelInference",
            ],
            semantic_keywords=[
                "metrics",
                "performance",
                "evaluation",
                "results",
                "metrics_output",
                "performance_metrics",
                "evaluation_metrics",
                "model_metrics",
                "metrics_results",
                "metrics_report",
                "performance_data",
            ],
            data_type="S3Uri",
            description="Model performance metrics data (JSON format with AUC, precision, recall, F1, domain metrics)",
        ),
        "plots_output": DependencySpec(
            logical_name="plots_output",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,  # Optional - can generate wiki without plots
            compatible_sources=[
                "ModelMetricsComputation",
                "XGBoostModelEval",
                "XGBoostModelInference",
                "PyTorchModelInference",
            ],
            semantic_keywords=[
                "plots",
                "visualizations",
                "charts",
                "graphs",
                "plots_output",
                "performance_plots",
                "metrics_plots",
                "roc_curves",
                "pr_curves",
                "visualizations_output",
                "performance_charts",
            ],
            data_type="S3Uri",
            description="Performance visualization plots (ROC curves, PR curves, distributions, threshold analysis)",
        ),
    },
    outputs={
        "wiki_output": OutputSpec(
            logical_name="wiki_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['wiki_output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "documentation",
                "wiki",
                "model_documentation",
                "wiki_documentation",
                "documentation_output",
                "model_wiki",
                "generated_docs",
                "wiki_pages",
            ],
            description="Generated model documentation in multiple formats (Wiki, HTML, Markdown) with performance analysis",
        ),
    },
)
