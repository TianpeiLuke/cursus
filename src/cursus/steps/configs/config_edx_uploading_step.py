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
from typing import Optional, Dict

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
                self.edx_provider = self.edx_provider or match.group(1)
                self.edx_subject = self.edx_subject or match.group(2)
                self.edx_dataset = self.edx_dataset or match.group(3)
                self.edx_manifest_key = self.edx_manifest_key or match.group(4)
            else:
                # Try simpler split for ARNs without manifest key brackets
                parts = self.edx_arn.replace("arn:amazon:edx:iad::manifest/", "").split(
                    "/"
                )
                if len(parts) >= 3:
                    self.edx_provider = self.edx_provider or parts[0]
                    self.edx_subject = self.edx_subject or parts[1]
                    self.edx_dataset = self.edx_dataset or parts[2]
                    if len(parts) > 3 and not self.edx_manifest_key:
                        key_part = "/".join(parts[3:])
                        self.edx_manifest_key = key_part.strip('[]"')
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

    def get_script_contract(self):
        """Get the script contract for this configuration."""
        from ..contracts.edx_uploading_contract import EDX_UPLOADING_CONTRACT

        return EDX_UPLOADING_CONTRACT
