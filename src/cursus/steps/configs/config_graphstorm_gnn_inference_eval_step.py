"""
GraphStorm GNN Inference/Eval Configuration with Self-Contained Derivation Logic.

Configuration for the GraphStormGNNInferenceEval Processing step — GraphStorm/DGL GNN out-of-time
inference + evaluation on a custom GraphStorm GPU container (online-inference simulator →
ROC-AUC / PR-AUC / Recall@Precision report + plots). Runs the custom GraphStorm image (BYO container).

Three-tier field design. The load-bearing new field is Tier-1 `image_uri` (inherited from
BasePipelineConfig, required-in-practice here); `subgraph_s3_uri` is a RAW S3 folder streamed by the
simulator (a job-arg, not a mounted channel).
"""

from pydantic import Field, field_validator, PrivateAttr
from typing import Dict, List, Optional
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class GraphStormGNNInferenceEvalConfig(ProcessingStepConfigBase):
    """Config for the GraphStorm GNN inference/eval step. `image_uri` is inherited from
    BasePipelineConfig (the BYO GraphStorm ECR image), required-in-practice (byo_container)."""

    # ===== Tier 1: Essential User Inputs =====
    subgraph_s3_uri: str = Field(
        description="Eval subgraph pkl folder streamed by the online-inference simulator (--subgraph-s3-uri).",
    )
    id_name: str = Field(
        description="Order-id column for the report join (order_id/object_id) → ID_FIELD env.",
    )
    label_names: List[str] = Field(
        description="Label columns for metrics (e.g. is_abuse, e90_label, ...) → LABEL_FIELDS env.",
    )

    # ===== Tier 2: System Fields (overridable defaults) =====
    processing_entry_point: str = Field(default="graphstorm_gnn_inference_eval.py")
    query_type: str = Field(
        default="baseline", description="baseline | relation traversal mode."
    )
    checkpoint: str = Field(
        default="auto", description="auto | latest | epoch-N-iter-M."
    )
    model_type: str = Field(
        default="auto", description="auto | multitask | singletask conversion."
    )
    auto_tune: bool = Field(
        default=True, description="Auto-tune GPU workers via nvidia-smi VRAM probe."
    )
    num_gpus: Optional[int] = Field(
        default=None, description="Override GPU count (else auto-detected)."
    )
    num_processors: Optional[int] = Field(
        default=None, description="Workers/GPU (else auto-tuned)."
    )
    max_workers_per_gpu: int = Field(default=8, ge=1)
    instance_type: str = Field(
        default="ml.g5.12xlarge", description="First rung of the GPU fallback ladder."
    )
    gpu_instance_fallback: List[str] = Field(
        default_factory=lambda: [
            "ml.g5.12xlarge",
            "ml.g4dn.12xlarge",
            "ml.g5.2xlarge",
            "ml.g4dn.xlarge",
        ],
        description="ResourceLimitExceeded retry order.",
    )
    max_runtime_seconds: int = Field(default=86400, ge=60)

    model_config = ProcessingStepConfigBase.model_config

    @field_validator("query_type")
    @classmethod
    def _validate_query_type(cls, v: str) -> str:
        if v not in ("baseline", "relation"):
            raise ValueError(f"query_type must be 'baseline' or 'relation', got '{v}'")
        return v

    # ===== Tier 3: Derived Fields (private + read-only property) =====
    _volume_size: Optional[int] = PrivateAttr(default=None)

    @property
    def volume_size(self) -> int:
        """EBS volume: 125 GB for g4dn, else 500 GB (from the Nexus launcher)."""
        if self._volume_size is None:
            self._volume_size = 125 if "g4dn" in self.instance_type else 500
        return self._volume_size

    @property
    def default_num_gpus(self) -> int:
        """GPU count per instance (g5.12xl / g4dn.12xl = 4, else 1) — the auto-detect fallback."""
        gpu_counts = {"ml.g5.12xlarge": 4, "ml.g4dn.12xlarge": 4}
        return gpu_counts.get(self.instance_type, 1)

    def get_environment_variables(self) -> Dict[str, str]:
        """Env vars the eval container reads. ID_FIELD + LABEL_FIELDS are REQUIRED (declared in the
        interface) — the report join column + the comma-list of label columns for metrics; the rest
        are optional knobs."""
        env = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )
        env.update(
            {
                "ID_FIELD": self.id_name,
                "LABEL_FIELDS": ",".join(self.label_names),
                "QUERY_TYPE": self.query_type,
                "MAX_WORKERS_PER_GPU": str(self.max_workers_per_gpu),
                "MAX_RUNTIME_SECONDS": str(self.max_runtime_seconds),
                "AUTO_TUNE": str(self.auto_tune).lower(),
            }
        )
        return env
