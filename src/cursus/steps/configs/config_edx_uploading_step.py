"""
EdxUploading Step Configuration.

Configuration for uploading S3 data to EDX via EdxDataLoader.
Inherits ProcessingStepConfigBase for standard processing step fields.
"""

from pydantic import Field, computed_field
from typing import Optional, Dict

from .config_processing_step_base import ProcessingStepConfigBase


class EdxUploadingConfig(ProcessingStepConfigBase):
    """Configuration for the EdxUploading step (S3 → EDX via EdxDataLoader)."""

    # ===== Tier 1: Essential (user must provide) =====

    edx_provider: str = Field(
        description="EDX provider name (e.g., 'trms-abuse-analytics')"
    )
    edx_subject: str = Field(description="EDX subject (e.g., 'munged-address')")
    edx_dataset: str = Field(
        description="EDX dataset name (e.g., 'munged-address-tags')"
    )
    edx_manifest_key: str = Field(
        description=(
            "Manifest key for this upload. Supports two modes:\n"
            "1. Static key: 'munged_na' → ARN ends with /[\"munged_na\"]\n"
            "2. Template key with placeholders: '{marketplace},{date},{execution_id}'\n"
            "   Placeholders resolved at runtime from edx_manifest_key_parts + environ.\n"
            "Example static: 'munged_na'\n"
            "Example template: '{marketplace},{dataset_date},{job_id}'"
        )
    )

    # ===== Tier 2: System Inputs with Defaults =====

    job_type: Optional[str] = Field(
        default=None, description="Job type suffix for step naming (e.g., 'tagging')"
    )
    processing_entry_point: str = Field(
        default="edx_upload.py", description="Entry point script for EDX upload"
    )
    edx_manifest_key_parts: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional dict of placeholder values for template manifest keys.\n"
            "Only used when edx_manifest_key contains {placeholders}.\n"
            "Example: {'marketplace': 'NA', 'dataset_date': '2026-05-27'}"
        ),
    )

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
