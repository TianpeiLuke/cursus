"""
Redshift Data Loading Step Specification.

Source node that executes SQL against Redshift and outputs CSV to S3.
Optionally uploads to EDX as side effect. Same DAG role as CradleDataLoading.
"""

from ...core.base.specification_base import (
    StepSpecification,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


def _get_redshift_data_loading_contract():
    from ..contracts.redshift_data_loading_contract import (
        REDSHIFT_DATA_LOADING_CONTRACT,
    )

    return REDSHIFT_DATA_LOADING_CONTRACT


REDSHIFT_DATA_LOADING_SPEC = StepSpecification(
    step_type=get_spec_step_type("RedshiftDataLoading"),
    node_type=NodeType.SOURCE,
    script_contract=_get_redshift_data_loading_contract(),
    dependencies={},  # SOURCE NODE — no upstream dependencies
    outputs={
        "output": OutputSpec(
            logical_name="output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "DATA",
                "input_data",
                "input_path",
                "processed_data",
                "redshift_output",
                "sql_output",
                "training_data",
            ],
            description="Query results from Redshift (CSV format). Always written to S3; EDX upload is optional side effect.",
        ),
    },
)
