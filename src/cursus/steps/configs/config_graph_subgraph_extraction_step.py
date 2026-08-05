"""
Graph Subgraph Extraction Configuration with Self-Contained Derivation Logic.

Configuration for the GraphSubgraphExtraction Processing step — a point-in-time k-hop
subgraph pull from a property-graph DB (NebulaGraph) for seed order IDs, run inside the custom
GraphStorm image (BYO container) and VPC-bound to reach the graph cluster.

Three-tier field design:
1. Essential User Inputs (Tier 1) - Required fields users must provide.
2. System Fields (Tier 2) - Defaults that can be overridden.
3. Derived Fields (Tier 3) - Private, exposed via read-only properties.

The compute descriptor lives in the `.step.yaml` (`compute.kind: byo_container` +
`network_mode: config`); this config supplies the `image_uri`, VPC `subnets`/`security_group_ids`,
and the graph-traversal knobs the ported script reads.
"""

from pydantic import Field, field_validator, PrivateAttr
from typing import Optional
from urllib.parse import urlparse
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class GraphSubgraphExtractionConfig(ProcessingStepConfigBase):
    """
    Configuration for the GraphSubgraphExtraction step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    `subnets` / `security_group_ids` are inherited from BasePipelineConfig (the VPC fields read
    when `compute.network_mode == 'config'`); `image_uri` is inherited too (read when
    `compute.kind == 'byo_container'`). This class adds the graph-traversal-specific fields.
    """

    # ===== Essential User Inputs (Tier 1) =====
    seed_s3_uri: str = Field(
        description="S3 URI of the seed parquet (an order_id/object_id column) to pull subgraphs for."
    )
    output_s3_uri: str = Field(
        description="S3 prefix the per-order subgraph pickles are written under (direct boto3 upload)."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    processing_entry_point: str = Field(
        default="graph_subgraph_extraction.py",
        description="Entry point script for the subgraph extraction, relative to the source directory.",
    )

    nebula_cluster: str = Field(
        default="gamma",
        description="Property-graph cluster key: 'gamma' (explicit endpoint) or 'prod' (auto-discover).",
    )

    num_parallel_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "Seed-shard fan-out width. v1 Cursus mapping keeps this at 1 (one SageMaker job) and "
            "relies on the in-job ThreadPoolExecutor; >1 is reserved for a future sharded mapping."
        ),
    )

    max_workers: int = Field(
        default=100,
        ge=1,
        description="Per-job traversal ThreadPoolExecutor width (emitted as the MAX_WORKERS env var).",
    )

    session_pool_min_size: int = Field(
        default=100,
        ge=1,
        description="Graph-DB session-pool min size (SESSION_POOL_MIN_SIZE env).",
    )
    session_pool_max_size: int = Field(
        default=300,
        ge=1,
        description="Graph-DB session-pool max size (SESSION_POOL_MAX_SIZE env).",
    )
    nebula_timeout_ms: int = Field(
        default=10000,
        ge=1,
        description="Graph-DB session/traversal timeout in ms (NEBULA_TIMEOUT_MS env).",
    )

    max_runtime_seconds: int = Field(
        default=432000,
        ge=60,
        description="StoppingCondition max runtime in seconds (default 5 days — traversals are slow).",
    )

    code_s3_uri: Optional[str] = Field(
        default=None,
        description="S3 prefix of the entrypoint.sh + query bundle mounted at /opt/ml/processing/input/code.",
    )

    # NOTE: image_uri, subnets, security_group_ids, enable_network_isolation are inherited from
    # BasePipelineConfig (image_uri = the BYO GraphStorm ECR URI; subnets/SGs = the VPC to reach
    # the graph cluster). They are required-in-practice here, enforced by the validator below.

    # ===== Derived Fields (Tier 3) =====
    _output_bucket: Optional[str] = PrivateAttr(default=None)
    _output_key_prefix: Optional[str] = PrivateAttr(default=None)

    @property
    def output_bucket(self) -> str:
        """The --bucket-name arg: the bucket of output_s3_uri."""
        if self._output_bucket is None:
            self._output_bucket = urlparse(self.output_s3_uri).netloc
        return self._output_bucket

    @property
    def output_key_prefix(self) -> str:
        """The --traversal-out-dir arg: the key prefix of output_s3_uri."""
        if self._output_key_prefix is None:
            self._output_key_prefix = urlparse(self.output_s3_uri).path.lstrip("/")
        return self._output_key_prefix

    @field_validator("nebula_cluster")
    @classmethod
    def _validate_cluster(cls, v: str) -> str:
        if v not in {"gamma", "prod"}:
            raise ValueError(f"nebula_cluster must be 'gamma' or 'prod', got '{v}'")
        return v

    @field_validator("session_pool_max_size")
    @classmethod
    def _validate_pool_sizes(cls, v: int, info) -> int:
        min_size = info.data.get("session_pool_min_size")
        if min_size is not None and v < min_size:
            raise ValueError(
                f"session_pool_max_size ({v}) must be >= session_pool_min_size ({min_size})"
            )
        return v
