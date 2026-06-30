"""
EdxUploading Step Configuration.

Configuration for uploading S3 data to EDX via EdxDataLoader.
Inherits ProcessingStepConfigBase for standard processing step fields.

Supports two input modes:
1. Direct ARN: provide edx_arn, components parsed automatically
2. Component-based: provide edx_provider, edx_subject, edx_dataset, edx_manifest_key
"""

import re
from pydantic import Field, computed_field, model_validator
from typing import Optional, Dict, List

from .config_processing_step_base import ProcessingStepConfigBase


class EdxUploadingConfig(ProcessingStepConfigBase):
    """
    Configuration for the EdxUploading step (S3 → EDX via EdxDataLoader).

    Supports two input modes (same pattern as EdxDataSourceConfig in CradleDataLoading):
    1. Direct ARN input: provide edx_arn, components are parsed from it
    2. Component-based input: provide edx_provider, edx_subject, edx_dataset, edx_manifest_key
    """

    # ===== Tier 1: Essential (at least one mode must be satisfied) =====

    edx_arn: Optional[str] = Field(
        default=None,
        description=(
            "Full EDX manifest ARN. If provided, components are parsed from it.\n"
            'Format: arn:amazon:edx:iad::manifest/{provider}/{subject}/{dataset}/["{key}"]\n'
            "Example: 'arn:amazon:edx:iad::manifest/trms-abuse-analytics/munged-address/munged-address-tags/[\"munged_na\"]'"
        ),
    )

    edx_provider: Optional[str] = Field(
        default=None,
        description="EDX provider name (e.g., 'trms-abuse-analytics'). Required if edx_arn not provided.",
    )
    edx_subject: Optional[str] = Field(
        default=None,
        description="EDX subject (e.g., 'munged-address'). Required if edx_arn not provided.",
    )
    edx_dataset: Optional[str] = Field(
        default=None,
        description="EDX dataset name (e.g., 'munged-address-tags'). Required if edx_arn not provided.",
    )
    edx_manifest_key: Optional[str] = Field(
        default=None,
        description=(
            "Manifest key for this upload. Required if edx_arn not provided.\n"
            "Supports two modes:\n"
            "1. Static key: 'munged_na' → ARN ends with /[\"munged_na\"]\n"
            "2. Template key with placeholders: '{marketplace},{date},{execution_id}'\n"
            "   Placeholders resolved at runtime from edx_manifest_key_parts + environ.\n"
            "Example static: 'munged_na'\n"
            "Example template: '{marketplace},{dataset_date},{job_id}'"
        ),
    )

    # ===== Tier 2: System Inputs with Defaults =====

    job_type: Optional[str] = Field(
        default=None, description="Job type suffix for step naming (e.g., 'tagging')"
    )
    processing_entry_point: str = Field(
        default="edx_uploading.py", description="Entry point script for EDX upload"
    )
    edx_manifest_key_parts: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional dict of placeholder values for template manifest keys.\n"
            "Only used when edx_manifest_key contains {placeholders}.\n"
            "Example: {'marketplace': 'NA', 'dataset_date': '2026-05-27'}"
        ),
    )
    edx_output_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional explicit, ORDERED list of columns to upload. The script projects the "
            "concatenated dataframe to exactly this set (filling missing as empty, dropping extras) "
            "before writing the headerless TSV. This is the positional contract the downstream "
            "Cradle SCORED read consumes — it MUST match the consuming step's "
            "EdxDataSourceConfig.schema_overrides in count AND order, since EDX is headerless and "
            "parsed positionally. If omitted, the script falls back to its hardcoded "
            "CANONICAL_OUTPUT_COLUMNS default.\n"
            "Example: ['saddr', 'marketplaceId', 'orderDate', '__cohort__', "
            "'llm_strangeness_rating', 'llm_parse_status', 'llm_validation_passed', 'llm_status']"
        ),
    )

    # ===== Validation =====

    @model_validator(mode="after")
    def validate_input_mode(self) -> "EdxUploadingConfig":
        """Ensure either edx_arn OR all 4 components are provided. Parse ARN if given."""
        if self.edx_arn:
            # Mode 1: Parse components from ARN
            # Format: arn:amazon:edx:iad::manifest/{provider}/{subject}/{dataset}/["{key}"]
            match = re.match(
                r'arn:amazon:edx:\w+::manifest/([^/]+)/([^/]+)/([^/]+)/\["([^"]+)"\]',
                self.edx_arn,
            )
            if match:
                # Use object.__setattr__ to avoid triggering validate_assignment recursion
                if not self.edx_provider:
                    object.__setattr__(self, "edx_provider", match.group(1))
                if not self.edx_subject:
                    object.__setattr__(self, "edx_subject", match.group(2))
                if not self.edx_dataset:
                    object.__setattr__(self, "edx_dataset", match.group(3))
                if not self.edx_manifest_key:
                    object.__setattr__(self, "edx_manifest_key", match.group(4))
            else:
                # Try simpler split for ARNs without manifest key brackets
                parts = self.edx_arn.replace("arn:amazon:edx:iad::manifest/", "").split(
                    "/"
                )
                if len(parts) >= 3:
                    if not self.edx_provider:
                        object.__setattr__(self, "edx_provider", parts[0])
                    if not self.edx_subject:
                        object.__setattr__(self, "edx_subject", parts[1])
                    if not self.edx_dataset:
                        object.__setattr__(self, "edx_dataset", parts[2])
                    if len(parts) > 3 and not self.edx_manifest_key:
                        key_part = "/".join(parts[3:])
                        object.__setattr__(
                            self, "edx_manifest_key", key_part.strip('[]"')
                        )
        else:
            # Mode 2: Require all components
            missing = []
            if not self.edx_provider:
                missing.append("edx_provider")
            if not self.edx_subject:
                missing.append("edx_subject")
            if not self.edx_dataset:
                missing.append("edx_dataset")
            if not self.edx_manifest_key:
                missing.append("edx_manifest_key")
            if missing:
                raise ValueError(
                    f"Either edx_arn or all component fields required. Missing: {missing}"
                )

        return self

    # ===== Tier 3: Derived Properties =====

    @computed_field
    @property
    def edx_arn_base(self) -> str:
        """Construct EDX manifest ARN base from provider/subject/dataset."""
        return (
            f"arn:amazon:edx:iad::manifest/"
            f"{self.edx_provider}/{self.edx_subject}/{self.edx_dataset}"
        )

    def get_environment_variables(
        self, declared_env_vars: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """EDX-upload env vars (the single env source; moved here from the builder, FZ 31e1d3g).

        Bespoke values (constant regional-endpoint flag, computed edx_arn_base, JSON-encoded
        manifest-key-parts / output-columns), so this returns the full env dict; ``declared_env_vars``
        is accepted for the builder's names-driven contract but ignored.
        """
        import json

        env_vars: Dict[str, str] = {
            "AWS_STS_REGIONAL_ENDPOINTS": "regional",
            "AWS_DEFAULT_REGION": self.aws_region or "us-east-1",
            "EDX_DATASET_ARN": self.edx_arn_base,
            "EDX_MANIFEST_KEY": self.edx_manifest_key,
        }
        if self.edx_manifest_key_parts:
            env_vars["EDX_MANIFEST_KEY_PARTS"] = json.dumps(self.edx_manifest_key_parts)
        if self.edx_output_columns:
            env_vars["EDX_OUTPUT_COLUMNS"] = json.dumps(self.edx_output_columns)
        return env_vars
