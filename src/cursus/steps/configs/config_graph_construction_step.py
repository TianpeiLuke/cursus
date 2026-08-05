"""
Graph Construction Configuration with Self-Contained Derivation Logic.

Configuration for the GraphConstruction Processing step — runs GraphStorm's gconstruct to build a
partitioned DGL heterograph from the node/edge parquets + gconstruct schema emitted by
GraphFeatureProcessing. Runs the custom GraphStorm image (BYO container).

Three-tier field design (Tier-2/3 defaults from the Nexus launcher's step.get(...) fallbacks:
num_processes=20, num_parts=1, skip_nonexist_edges=True, volume_size_gb=125,
max_runtime_seconds=86400).
"""

from pydantic import Field, model_validator, PrivateAttr
from typing import List, Optional
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class GraphConstructionConfig(ProcessingStepConfigBase):
    """Config for the GraphStorm gconstruct (GraphConstruction) step. `image_uri` is inherited from
    BasePipelineConfig (the BYO GraphStorm ECR image) and required-in-practice (byo_container)."""

    # ===== Tier 1: Essential User Inputs =====
    graph_name: str = Field(
        description="GraphStorm graph name (--graph-name); names the partitioned DGL graph.",
    )

    # ===== Tier 2: System Fields (overridable defaults) =====
    num_processes: int = Field(
        default=20, ge=1, description="gconstruct worker processes (--num-processes)."
    )
    num_parts: int = Field(
        default=1, ge=1, description="DGL graph partitions (--num-parts)."
    )
    skip_nonexist_edges: bool = Field(
        default=True, description="Append --skip-nonexist-edges when true."
    )
    gconstruct_config_filename: str = Field(default="gconstruct_config.json")
    run_sanity_check: bool = Field(
        default=True, description="Run the ported sanity_check after construction."
    )
    sanity_check_full: bool = Field(
        default=False, description="Full (dangling-edge) sanity mode vs fast sampled."
    )
    processing_entry_point: str = Field(default="run_gconstruct.py")
    processing_volume_size: int = Field(default=125, ge=10, le=1000)
    max_runtime_seconds: int = Field(default=86400, ge=60)
    use_large_processing_instance: bool = Field(default=True)

    model_config = ProcessingStepConfigBase.model_config

    # ===== Tier 3: Derived Fields (private + read-only property) =====
    _gconstruct_arguments: Optional[List[str]] = PrivateAttr(default=None)

    @property
    def gconstruct_arguments(self) -> List[str]:
        """CLI args mirroring the Nexus ContainerArguments (the fixed --conf-file/--output-dir are
        handled by the script's own constant paths). Emitted as the step's job_arguments."""
        if self._gconstruct_arguments is None:
            args = [
                "--graph-name",
                self.graph_name,
                "--num-processes",
                str(self.num_processes),
                "--num-parts",
                str(self.num_parts),
            ]
            if self.skip_nonexist_edges:
                args.append("--skip-nonexist-edges")
            self._gconstruct_arguments = args
        return self._gconstruct_arguments

    @model_validator(mode="after")
    def _require_image_uri(self) -> "GraphConstructionConfig":
        # byo_container compute kind: the image is user-owned, so image_uri must be present.
        if not getattr(self, "image_uri", None):
            raise ValueError(
                "image_uri is required for GraphConstruction (byo_container compute)."
            )
        return self
