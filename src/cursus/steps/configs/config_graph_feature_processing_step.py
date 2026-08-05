"""
Graph Feature Processing Configuration with Self-Contained Derivation Logic.

Configuration for the GraphFeatureProcessing Processing step — turns per-seed subgraph pickles +
labelled seeds into the GraphStorm GConstruct input tree (per-type node/edge parquets, reverse
edges, node-ID-keyed multi-task masks, gconstruct_config.json). Runs the custom GraphStorm image
(BYO container).

Three-tier field design:
1. Essential User Inputs (Tier 1) - Required fields users must provide.
2. System Fields (Tier 2) - Defaults that can be overridden.
3. Derived Fields (Tier 3) - Private, exposed via read-only properties.

The heavy feature-engineering modules (prepare_graphstorm_format.py + helpers) live in the BYO
image's code bundle, NOT this package; the structured knobs that do not fit env vars (per-type
feature dicts + the multi-task target list) are assembled here into a `config.yaml` payload the
container reads via --config.
"""

from pydantic import BaseModel, Field, field_validator, PrivateAttr
from typing import Any, Dict, List, Optional
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class TargetNodeSpec(BaseModel):
    """One multi-task target-node spec — a train/val/test split + idx parquet set is generated
    per label column on this node type."""

    node_type: str = Field(
        description="Graph node type carrying the labels (e.g. 'Order')."
    )
    id_col: str = Field(description="Seed-parquet column holding the node id.")
    label_cols: List[str] = Field(
        description="Label columns; one task/split per column."
    )


class SplitCfg(BaseModel):
    """Optional post-hoc split_for_gconstruct config (re-split parquets for parallel gconstruct)."""

    enabled: bool = Field(default=False)
    num_chunks: int = Field(default=50, ge=1)
    split_type: str = Field(default="both", description="nodes | edges | both")


class GraphFeatureProcessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the GraphFeatureProcessing step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase. `image_uri` is inherited from BasePipelineConfig
    (the BYO GraphStorm ECR image) and required-in-practice (compute.kind is byo_container).
    """

    # ===== Essential User Inputs (Tier 1) =====
    target_node_types: List[TargetNodeSpec] = Field(
        description=(
            "Multi-task target-node specs: each {node_type, id_col, label_cols}. One train/val/test "
            "split + idx parquet set is generated per label column."
        ),
    )
    subgraph_source: str = Field(
        description="Subgraph pickle format — 'nebula' or 'neptune' — selects the preprocessing branch.",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    processing_entry_point: str = Field(default="graph_feature_processing.py")
    num_chunks: int = Field(
        default=2, ge=1, description="gconstruct split fan-out (--num-chunks)."
    )
    chunk_size: int = Field(
        default=50000, ge=1, description="Subgraphs loaded/processed per chunk."
    )
    reverse_edge: bool = Field(
        default=True,
        description="Emit reverse edges for bidirectional message passing.",
    )
    train_frac: float = Field(default=0.8, ge=0.0, le=1.0)
    val_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    query_type: str = Field(
        default="relation",
        description="merged_traversal ('relation') vs customer_traversal ('baseline').",
    )
    numerical_feat_dict: Dict[str, List[str]] = Field(default_factory=dict)
    interval_edge_types: List[str] = Field(default_factory=list)
    unwanted_properties: List[str] = Field(default_factory=list)
    etype_with_feat: List[str] = Field(default_factory=list)
    bert_model: str = Field(default="bert-base-uncased")
    max_seq_length: int = Field(default=16, ge=1)
    co_train_lm: bool = Field(default=False)
    max_load_workers: int = Field(default=8, ge=1)
    max_workers: int = Field(default=30, ge=1)
    enable_purchase_collapse: bool = Field(default=False)
    split_files: Optional[SplitCfg] = Field(default=None)
    gconstruct_config_name: str = Field(default="gconstruct_config.json")
    processing_volume_size: int = Field(
        default=500, ge=1, description="Large subgraph corpora need a big EBS volume."
    )
    max_runtime_seconds: int = Field(default=86400, ge=60)

    # Where the materialized config.yaml is staged in the source dir + mounted in the container.
    config_yaml_name: str = Field(default="graph_feature_config.yaml")

    model_config = ProcessingStepConfigBase.model_config

    @field_validator("subgraph_source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v not in {"nebula", "neptune"}:
            raise ValueError(
                f"subgraph_source must be 'nebula' or 'neptune', got '{v}'"
            )
        return v

    # ===== Derived Fields (Tier 3) =====
    _config_yaml_payload: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    @property
    def config_yaml_payload(self) -> Dict[str, Any]:
        """The nested config.yaml the container reads via --config (structured knobs that do not
        fit env vars: per-type feature dicts + the multi-task target list)."""
        if self._config_yaml_payload is None:
            self._config_yaml_payload = {
                "output_dir": "/opt/ml/processing/output",
                "subgraph_source": self.subgraph_source,
                "query_type": self.query_type,
                "target_nodes": [t.model_dump() for t in self.target_node_types],
                "numerical_feat_dict": self.numerical_feat_dict,
                "interval_edge_types": self.interval_edge_types,
                "unwanted_properties": self.unwanted_properties,
                "etype_with_feat": self.etype_with_feat,
                "reverse_edge": self.reverse_edge,
                "train_frac": self.train_frac,
                "val_frac": self.val_frac,
                "chunk_size": self.chunk_size,
                "bert_model": self.bert_model,
                "max_seq_length": self.max_seq_length,
                "co_train_lm": self.co_train_lm,
                "max_load_workers": self.max_load_workers,
                "max_workers": self.max_workers,
                "enable_purchase_collapse": self.enable_purchase_collapse,
                "gconstruct_config_name": self.gconstruct_config_name,
                "split_files": self.split_files.model_dump()
                if self.split_files
                else None,
            }
        return self._config_yaml_payload

    @property
    def config_yaml_container_path(self) -> str:
        """The --config arg: where the materialized config.yaml is mounted in the container
        (the builder stages it into the source dir, which mounts under /opt/ml/processing/input/code)."""
        return f"/opt/ml/processing/input/code/{self.config_yaml_name}"
