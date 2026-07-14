"""
Slipbox Knowledge Routing Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the SageMaker Processing step
that hosts the DKS knowledge+ruleset corpus and runs the internal
compile → index → route pipeline (see FZ 29h1e §4/§7).

Fields are categorized according to the three-tier design used across cursus
processing configs:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with
   read-only property access

The env-var names declared by ``slipbox_knowledge_routing.step.yaml`` resolve by
the single-source convention (``NAME`` -> ``self.name``); each optional env var has
a matching lower-cased field here so ``get_environment_variables`` can resolve it:

    ROUTING_SCORING_MODE   <- self.routing_scoring_mode
    ROUTING_THRESHOLD      <- self.routing_threshold
    ROUTING_TOP_K          <- self.routing_top_k
    EMBEDDING_MODEL_NAME   <- self.embedding_model_name
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SlipboxKnowledgeRoutingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Slipbox Knowledge Routing step with three-tier field
    categorization. Inherits from ProcessingStepConfigBase (same base as the
    TabularPreprocessing exemplar).

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide.
    # job_type selects the routing run mode (mirrors the Processing exemplars); the
    # builder passes it through the --job_type CLI flag.

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override.

    processing_entry_point: str = Field(
        default="slipbox_knowledge_routing.py",
        description="Relative path (within processing_source_dir) to the slipbox knowledge routing script.",
    )

    # Routing knobs — each maps by convention to a declared env var in the .step.yaml.

    routing_scoring_mode: str = Field(
        default="activation",
        description="Rule scoring mode for the router. 'activation' uses the activation top-k scorer (scoring.score_rules_by_activation); other modes may score by raw cosine similarity.",
    )

    routing_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a pattern match to activate a linked rule (ROUTING_THRESHOLD).",
    )

    routing_top_k: int = Field(
        default=7,
        ge=1,
        description="Maximum number of routed rules to keep per record after activation scoring (ROUTING_TOP_K).",
    )

    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name used to build the routing index. Overridden at runtime by the offline embedding_model input path when present (EMBEDDING_MODEL_NAME).",
    )

    # ===== Compute overrides =====
    # The routing index build runs SentenceTransformer.encode (torch); default to the
    # larger processing instance so the torch/sentence-transformers container has headroom.

    use_large_processing_instance: bool = Field(
        default=True,
        description="Use the large processing instance type; the SentenceTransformer encode pass benefits from the extra CPU/memory headroom.",
    )

    # ===== Source-dir / framework processor (Processing pattern 2) =====
    # This step ships a SOURCE DIR (not a self-contained script): the entry-point script
    # sits alongside a bundled knowledge closure (the DKS compile→index→route package +
    # the rule_*/pattern_*/behavior_*.md corpus) that it imports at runtime. The .step.yaml
    # therefore declares ``contract.source_dir: true`` + ``compute.kind: framework`` /
    # ``sdk_class: PyTorch`` (a FrameworkProcessor — ScriptProcessor.run has no source_dir),
    # the SAME pattern as the model-eval / model-inference steps.
    #
    # PyTorch (not SKLearn) because the routing index build runs SentenceTransformer.encode,
    # which needs torch: the PyTorch container ships torch already, so we only layer on
    # sentence-transformers via a requirements file — vs a SKLearn image that would have to
    # pip-install the whole torch/CUDA stack at container start.
    #
    #   * ``processing_source_dir`` (inherited from ProcessingStepConfigBase, default None)
    #     is set at pipeline-wiring time to the dir holding the entry-point script + the
    #     bundled knowledge closure; ``effective_source_dir`` / ``get_script_path`` resolve it.
    #   * ``framework_version`` / ``py_version`` below drive the SageMaker PyTorch container
    #     image (mirroring the pytorch model-eval step).

    framework_version: str = Field(
        default="2.1.2",
        description="PyTorch framework version for the processing container (drives the SageMaker PyTorch image, same as the pytorch model-eval step).",
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    # ===== Derived Fields (Tier 3) =====
    # (No derived properties needed beyond the base class — script path resolution is
    #  inherited from ProcessingStepConfigBase.)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Validators =====

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """Ensure processing_entry_point is a non-empty relative path."""
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non-empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError(
                "processing_entry_point must be a relative path within source directory"
            )
        return v

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Ensure job_type is lowercase alphanumeric (with underscores)."""
        if not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(
                f"job_type must be lowercase alphanumeric (with underscores), got '{v}'"
            )
        return v

    @field_validator("routing_scoring_mode")
    @classmethod
    def validate_scoring_mode(cls, v: str) -> str:
        """Ensure routing_scoring_mode is one of the allowed values (case-insensitive)."""
        allowed = {"activation", "similarity"}
        match = next((a for a in allowed if a.lower() == v.lower()), None)
        if match is None:
            raise ValueError(
                f"routing_scoring_mode must be one of {sorted(allowed)} (case-insensitive), got '{v}'"
            )
        return match

    @field_validator("embedding_model_name")
    @classmethod
    def validate_embedding_model_name(cls, v: str) -> str:
        """Ensure embedding_model_name is a non-empty string."""
        if not v or not v.strip():
            raise ValueError("embedding_model_name must be a non-empty string")
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "SlipboxKnowledgeRoutingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        return self

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include slipbox knowledge routing
        specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        base_fields = super().get_public_init_fields()

        routing_fields = {
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "routing_scoring_mode": self.routing_scoring_mode,
            "routing_threshold": self.routing_threshold,
            "routing_top_k": self.routing_top_k,
            "embedding_model_name": self.embedding_model_name,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Combine fields (routing fields take precedence if overlap)
        init_fields = {**base_fields, **routing_fields}

        return init_fields

    def get_job_arguments(self) -> Optional[List[str]]:
        """CLI args — config is the single source (FZ 31e1d3h)."""
        return self._job_type_arg()
