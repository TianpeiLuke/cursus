"""
Configuration for the TabularLookupModelBuilding step.

TabularLookupModelBuilding is a non-parametric model-building step: it reads a
tabular dataset from an upstream data-loading step and turns the dataset itself
into the model — either a ``key -> [values]`` lookup map (``model_kind="lookup"``)
or a de-duplicated key set (``model_kind="set_membership"``) — then packages it
into ``model.tar.gz`` for downstream MIMS ``Package`` + ``Payload``. There is no
training and there are no learned parameters ("the dataset IS the model").

The per-project inference handler is bundled separately by the ``Package`` step
(via its ``inference_scripts_input`` channel); this step does NOT generate an
inference handler at runtime.

This configuration follows the Three-Tier Config Design pattern:

- **Tier 1 (Essential Fields)**: required user inputs (``model_kind``, ``key_columns``).
- **Tier 2 (System Fields)**: fields with sensible defaults (``value_columns``,
  ``dedup``, ``shard_count``, ``processing_entry_point``).
- **Tier 3 (Derived Fields)**: inherited from ``ProcessingStepConfigBase``
  (``effective_source_dir``, ``effective_instance_type``, ``script_path``).
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field, PrivateAttr, field_validator, model_validator

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ALLOWED_MODEL_KINDS = {"lookup", "set_membership"}


class TabularLookupModelBuildingConfig(ProcessingStepConfigBase):
    """
    Configuration for the TabularLookupModelBuilding step (three-tier design).

    Inherits region/author/bucket/role/pipeline + processing_* fields from
    ``ProcessingStepConfigBase`` -> ``BasePipelineConfig``.
    """

    # ===== Essential User Inputs (Tier 1) =====

    model_kind: str = Field(
        description="Kind of non-parametric model to build: 'lookup' (group value_columns "
        "by key_columns into a key->values map) or 'set_membership' (de-duplicated set of "
        "key_columns). Emitted to the script as MODEL_KIND.",
    )

    key_columns: List[str] = Field(
        description="Column(s) forming the lookup/membership key (the group-by key for "
        "'lookup', or the de-dup key for 'set_membership'). Emitted as KEY_COLUMNS (JSON).",
    )

    # ===== System Fields with Defaults (Tier 2) =====

    value_columns: List[str] = Field(
        default_factory=list,
        description="Column(s) collected per key when model_kind='lookup' (e.g. ['value']). "
        "Ignored for 'set_membership'. Emitted as VALUE_COLUMNS (JSON).",
    )

    dedup: bool = Field(
        default=True,
        description="De-duplicate values within each key ('lookup') or the key set "
        "('set_membership'). Emitted as DEDUP.",
    )

    shard_count: int = Field(
        default=1,
        ge=1,
        description="Number of JSON shard files the built model is split across inside "
        "model.tar.gz (keeps individual files a reasonable size). Emitted as SHARD_COUNT.",
    )

    processing_entry_point: str = Field(
        default="tabular_lookup_model_building.py",
        description="Relative path (within processing_source_dir) to the model-building script.",
    )

    # ===== Derived Fields (Tier 3) =====

    _lookup_environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Environment-variable collector (bespoke, config-owned) =====

    def get_environment_variables(self, declared_env_vars=None) -> Dict[str, str]:
        """
        Source the container environment for the model-building script.

        The universal builder detects a config-owned ``get_environment_variables`` and
        calls it to build the container env, then merges interface defaults for any
        declared-optional var not produced here. ``declared_env_vars`` is accepted for
        signature compatibility with the base resolver and intentionally ignored.
        """
        if self._lookup_environment_variables is None:
            env_vars = {
                "MODEL_KIND": self.model_kind,
                "KEY_COLUMNS": json.dumps(self.key_columns),
                "VALUE_COLUMNS": json.dumps(self.value_columns),
                "DEDUP": "true" if self.dedup else "false",
                "SHARD_COUNT": str(self.shard_count),
            }
            self._lookup_environment_variables = env_vars

        return self._lookup_environment_variables

    # ===== Validators =====

    @field_validator("model_kind")
    @classmethod
    def validate_model_kind(cls, v: str) -> str:
        """Ensure model_kind is one of the allowed values (case-insensitive)."""
        match = next((a for a in _ALLOWED_MODEL_KINDS if a == v.lower()), None)
        if match is None:
            raise ValueError(
                f"model_kind must be one of {sorted(_ALLOWED_MODEL_KINDS)}, got '{v}'"
            )
        return match

    @field_validator("key_columns")
    @classmethod
    def validate_key_columns(cls, v: List[str]) -> List[str]:
        """Require at least one non-empty key column."""
        cleaned = [c for c in (v or []) if isinstance(c, str) and c.strip()]
        if not cleaned:
            raise ValueError(
                "key_columns must contain at least one non-empty column name"
            )
        return cleaned

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

    @model_validator(mode="after")
    def validate_config(self) -> "TabularLookupModelBuildingConfig":
        """
        Validate the INTERNAL-node configuration.

        - lookup mode needs value_columns to have something to collect;
        - entry point present + script contract loadable;
        - the contract must declare the required 'model_output' output path.
        """
        if self.model_kind == "lookup" and not self.value_columns:
            raise ValueError(
                "model_kind='lookup' requires at least one value column in value_columns"
            )

        if not self.processing_entry_point:
            raise ValueError(
                "TabularLookupModelBuilding step requires a processing_entry_point"
            )

        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        if "model_output" not in contract.expected_output_paths:
            raise ValueError(
                "Script contract missing required output path: model_output"
            )

        return self

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """Include TabularLookupModelBuilding-specific fields for child initialization."""
        base_fields = super().get_public_init_fields()
        lookup_fields = {
            "model_kind": self.model_kind,
            "key_columns": self.key_columns,
            "value_columns": self.value_columns,
            "dedup": self.dedup,
            "shard_count": self.shard_count,
            "processing_entry_point": self.processing_entry_point,
        }
        return {**base_fields, **lookup_fields}
