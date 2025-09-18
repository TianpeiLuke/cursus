# src/pipeline_steps/config_batch_transform_step.py

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, Dict, Any
from ...core.base.config_base import BasePipelineConfig


class BatchTransformStepConfig(BasePipelineConfig):
    """
    Configuration for a generic SageMaker BatchTransform step.
    Inherits all the BasePipelineConfig attributes (bucket, region, etc.)
    and adds just what's needed to drive a TransformStep.
    """

    # 1) Which slice are we scoring?
    job_type: str = Field(
        ...,
        description="One of 'training','testing','validation','calibration' to indicate which slice to transform",
    )

    # Note: Input/output locations are now defined in step specs and provided through dependencies

    # 3) Compute sizing
    transform_instance_type: str = Field(
        default="ml.m5.large", description="Instance type for the BatchTransform job"
    )
    transform_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for the BatchTransform job"
    )

    # 4) Content negotiation & splitting
    content_type: str = Field(
        default="text/csv", description="MIME type of the input data"
    )
    accept: str = Field(
        default="text/csv",
        description="Response MIME type so output_fn knows how to format",
    )
    split_type: str = Field(
        default="Line",
        description="How to split the input file (must match your container’s input_fn)",
    )
    assemble_with: Optional[str] = Field(
        default="Line",
        description="How to re‐assemble input+output when join_source='Input'",
    )

    # 5) Optional JMESPath filters
    input_filter: Optional[str] = Field(
        default="$[1:]",
        description="JMESPath filter on each input record (e.g. '$[1:]')",
    )
    output_filter: Optional[str] = Field(
        default="$[-1]",
        description="JMESPath filter on each joined record (e.g. '$[-1]')",
    )

    # 6) Join strategy
    join_source: str = Field(
        default="Input", description="Whether to join on the 'Input' or 'Output' stream"
    )

    # Note: input_names and output_names have been removed in favor of script contracts

    model_config = BasePipelineConfig.model_config

    @field_validator("job_type")
    def _validate_job_type(cls, v: str) -> str:
        allowed = {"training", "testing", "validation", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    # Note: S3 URI validator removed as batch_input_location and batch_output_location were removed

    @field_validator("transform_instance_type")
    def _validate_instance_type(cls, v: str) -> str:
        if not v.startswith("ml."):
            raise ValueError(f"invalid instance type '{v}', must start with 'ml.'")
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "BatchTransformStepConfig":
        """Validate join and assemble configurations."""
        split = self.split_type
        assemble = self.assemble_with
        join = self.join_source
        if join == "Input" and assemble and assemble != split:
            raise ValueError(
                "when join_source='Input', assemble_with must equal split_type"
            )
        return self

    # Note: set_default_names validator has been removed along with input_names and output_names
