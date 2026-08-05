"""
GraphStorm GNN Training Configuration with Self-Contained Derivation Logic.

Configuration for the GraphStormGNNTraining Training step — GraphStorm/DGL R-GCN node-classification
(or multi-task) training over a partitioned DGL heterograph, run in a bring-your-own GraphStorm ECR
container. Extends BasePipelineConfig (as PyTorchTrainingConfig does).

The crux is `training_image_uri` — the byo_container builder reads it (interface
`image_uri_field: training_image_uri`) and sets it VERBATIM as AlgorithmSpecification.TrainingImage,
with no image_uris.retrieve (the GraphStorm/DGL stack is in no AWS DLC). graphstorm/dgl/torch are
baked into the image, so there is no framework install here.
"""

from pydantic import Field, field_validator, PrivateAttr
from typing import Dict, Optional
import logging

from ...core.base.config_base import BasePipelineConfig

logger = logging.getLogger(__name__)


class GraphStormGNNTrainingConfig(BasePipelineConfig):
    """GraphStorm/DGL R-GCN training in a bring-your-own GraphStorm container."""

    # ===== Tier 1: Essential User Inputs =====
    training_entry_point: str = Field(
        description="Container entry script (the ContainerEntrypoint target in the code channel), e.g. train.py.",
    )
    training_image_uri: str = Field(
        description=(
            "Custom GraphStorm ECR image URI run VERBATIM as AlgorithmSpecification.TrainingImage "
            "(byo_container — no image_uris.retrieve). E.g. "
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm-gnn:sagemaker-gpu."
        ),
    )

    # ===== Tier 2: System Fields (overridable defaults) =====
    training_instance_type: str = Field(default="ml.g5.12xlarge")
    training_instance_count: int = Field(default=1, ge=1)
    training_volume_size: int = Field(default=125, ge=1)
    max_runtime_seconds: int = Field(default=172800, ge=60)
    training_mode: str = Field(
        default="multi_task", description="multi_task | node_classification"
    )
    num_servers: int = Field(default=1, ge=1)
    code_s3_uri: str = Field(
        default="", description="S3 prefix mounted as the `code` channel."
    )
    batch_size: int = Field(default=1024, ge=1)
    hidden_size: int = Field(default=100, ge=1)
    num_layers: int = Field(default=3, ge=1)
    fanout: str = Field(default="30,30,30")

    model_config = BasePipelineConfig.model_config

    @field_validator("training_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("multi_task", "node_classification"):
            raise ValueError(
                "training_mode must be 'multi_task' or 'node_classification'"
            )
        return v

    # ===== Tier 3: Derived Fields (private + read-only property) =====
    _batch_size_override: Optional[int] = PrivateAttr(default=None)

    @property
    def batch_size_override(self) -> Optional[int]:
        """Pre-adjust batch_size for the instance's GPU memory (ports launch_training._adjust_batch_size).
        Returns the reduced batch size only when it is smaller than the configured batch_size; else None
        (emitted as the BATCH_SIZE_OVERRIDE env var only when set)."""
        gpu_mem = {
            "ml.g5.12xlarge": 24,
            "ml.g5.2xlarge": 24,
            "ml.g5.xlarge": 24,
            "ml.g4dn.12xlarge": 16,
            "ml.g4dn.xlarge": 16,
        }.get(self.training_instance_type, 16)
        max_safe = {24: 1024, 16: 512}.get(gpu_mem, 512)
        first_fanout = int(self.fanout.split(",")[0]) if self.fanout else 30
        complexity = (
            (self.hidden_size / 100)
            * ((self.num_layers / 3) ** 0.5)
            * ((first_fanout / 30) ** 0.5)
        )
        adjusted = min(4096, max(64, int(max_safe / max(complexity, 0.5))))
        return adjusted if adjusted < self.batch_size else None

    def get_environment_variables(self) -> Dict[str, str]:
        """Env vars the container reads (TRAINING_MODE / NUM_SERVERS / optional BATCH_SIZE_OVERRIDE)."""
        env = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )
        env.update(
            {
                "TRAINING_MODE": self.training_mode,
                "NUM_SERVERS": str(self.num_servers),
            }
        )
        override = self.batch_size_override
        if override is not None:
            env["BATCH_SIZE_OVERRIDE"] = str(override)
        return env
