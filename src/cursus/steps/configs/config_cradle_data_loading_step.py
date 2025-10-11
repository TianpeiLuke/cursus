# File: pipeline_steps/config_cradle_data_load.py

from typing import List, Optional, Dict, Any, Set
import re
from datetime import datetime
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    PrivateAttr,
)

from ...core.base.config_base import BasePipelineConfig


def get_flattened_fields(config_obj, prefix="") -> Dict[str, List[str]]:
    """
    Recursively gather all fields from a config object and its nested objects,
    flattening them into a single list with dot notation to indicate hierarchy.

    Args:
        config_obj: Configuration object to analyze
        prefix: String prefix for nested fields (for recursive calls)

    Returns:
        Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
    """
    # Get the fields at this level
    if hasattr(config_obj, "categorize_fields"):
        categories = config_obj.categorize_fields()
    else:
        # Initialize empty categories if the object doesn't support categorization
        categories = {"essential": [], "system": [], "derived": []}

    # Add the prefix to the field names
    result = {
        "essential": [f"{prefix}{field}" for field in categories["essential"]],
        "system": [f"{prefix}{field}" for field in categories["system"]],
        "derived": [f"{prefix}{field}" for field in categories["derived"]],
    }

    # Handle nested configuration objects
    for field_name in categories["essential"] + categories["system"]:
        field_value = getattr(config_obj, field_name)

        # Skip None values
        if field_value is None:
            continue

        # Handle list of configuration objects
        if isinstance(field_value, list) and len(field_value) > 0:
            # Check if items have categorize_fields method
            if all(
                hasattr(item, "categorize_fields")
                for item in field_value
                if item is not None
            ):
                for i, item in enumerate(field_value):
                    if item is None:
                        continue
                    nested_result = get_flattened_fields(
                        item, f"{prefix}{field_name}[{i}]."
                    )
                    for cat, fields in nested_result.items():
                        result[cat].extend(fields)

        # Handle single configuration objects
        elif hasattr(field_value, "categorize_fields"):
            nested_result = get_flattened_fields(field_value, f"{prefix}{field_name}.")
            for cat, fields in nested_result.items():
                result[cat].extend(fields)

    return result


class BaseCradleComponentConfig(BaseModel):
    """
    Base class for Cradle configuration components with three-tier field classification support.

    Implements common functionality for categorizing fields and supporting inheritance.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # Model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Changed from "forbid" to "allow" to fix circular reference handling
    )

    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - fields with no defaults (required)
        2. Tier 2: System Inputs - fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes

        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Initialize categories
        categories = {
            "essential": [],  # Tier 1: Required, public
            "system": [],  # Tier 2: Optional (has default), public
            "derived": [],  # Tier 3: Public properties
        }

        # Get model fields
        model_fields = self.__class__.model_fields

        # Categorize public fields into essential (required) or system (with defaults)
        for field_name, field_info in model_fields.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Use is_required() to determine if a field is essential
            if field_info.is_required():
                categories["essential"].append(field_name)
            else:
                categories["system"].append(field_name)

        # Find derived properties (public properties that aren't in model_fields)
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and attr_name not in model_fields
                and isinstance(getattr(type(self), attr_name, None), property)
            ):
                categories["derived"].append(attr_name)

        return categories

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Get fields suitable for initializing a child config.
        Only includes fields that should be passed to child class constructors.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Use categorize_fields to get essential and system fields
        categories = self.categorize_fields()

        # Initialize result dict
        init_fields = {}

        # Add all essential fields (Tier 1)
        for field_name in categories["essential"]:
            init_fields[field_name] = getattr(self, field_name)

        # Add all system fields (Tier 2) that aren't None
        for field_name in categories["system"]:
            value = getattr(self, field_name)
            if value is not None:  # Only include non-None values
                init_fields[field_name] = value

        return init_fields


class MdsDataSourceConfig(BaseCradleComponentConfig):
    """
    Configuration for MDS data source with three-tier field classification.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    service_name: str = Field(description="Name of the MDS service")

    region: str = Field(description="Region code for MDS (e.g. 'NA', 'EU', 'FE')")

    output_schema: List[Dict[str, Any]] = Field(
        description="List of dictionaries describing each output column, "
        "e.g. [{'field_name':'objectId','field_type':'STRING'}, …]"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    org_id: int = Field(
        default=0,
        description="Organization ID (integer) for MDS. Default as 0 for regional MDS bucket.",
    )

    use_hourly_edx_data_set: bool = Field(
        default=False, description="Whether to use the hourly EDX dataset flag in MDS"
    )

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        valid = {"NA", "EU", "FE"}
        if v not in valid:
            raise ValueError(f"region must be one of {valid}, got '{v}'")
        return v


class EdxDataSourceConfig(BaseCradleComponentConfig):
    """
    Configuration for EDX data source with three-tier field classification.

    Supports two input modes:
    1. Direct ARN input: Provide edx_arn directly
    2. Component-based input: Provide edx_provider, edx_subject, edx_dataset, edx_manifest_key

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must always explicitly provide

    schema_overrides: List[Dict[str, Any]] = Field(
        description=(
            "List of dicts overriding the EDX schema, e.g. "
            "[{'field_name':'order_id','field_type':'STRING'}, …]"
        )
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # Control field that determines requirement of component fields

    edx_arn: Optional[str] = Field(
        default=None,
        description="Complete EDX manifest ARN. If provided, individual components are ignored."
    )

    # Conditionally required fields (required only when edx_arn is None)
    edx_provider: Optional[str] = Field(
        default=None,
        description="Provider portion of the EDX manifest ARN (required if edx_arn not provided)"
    )

    edx_subject: Optional[str] = Field(
        default=None,
        description="Subject portion of the EDX manifest ARN (required if edx_arn not provided)"
    )

    edx_dataset: Optional[str] = Field(
        default=None,
        description="Dataset portion of the EDX manifest ARN (required if edx_arn not provided)"
    )

    edx_manifest_key: Optional[str] = Field(
        default=None,
        description="Manifest key in format '[\"xxx\",...]' (required if edx_arn not provided)"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _edx_manifest: Optional[str] = PrivateAttr(default=None)

    @property
    def edx_manifest(self) -> str:
        """Get EDX manifest ARN from direct input or built from components."""
        if self._edx_manifest is None:
            if self.edx_arn is not None:
                # Mode 1: Direct ARN input
                self._edx_manifest = self.edx_arn
            else:
                # Mode 2: Build from components (existing logic)
                self._edx_manifest = (
                    f"arn:amazon:edx:iad::manifest/"
                    f"{self.edx_provider}/{self.edx_subject}/{self.edx_dataset}/{self.edx_manifest_key}"
                )
        return self._edx_manifest

    def categorize_fields(self) -> Dict[str, List[str]]:
        """Dynamic field categorization based on edx_arn presence."""
        categories = {
            "essential": ["schema_overrides"],  # Always required
            "system": ["edx_arn"],  # Control field
            "derived": ["edx_manifest"]  # Computed property
        }
        
        # Component fields are system-level but conditionally required
        component_fields = ["edx_provider", "edx_subject", "edx_dataset", "edx_manifest_key"]
        
        if self.edx_arn is None:
            # When no ARN provided, components become essential
            categories["essential"].extend(component_fields)
        else:
            # When ARN provided, components are just system fields
            categories["system"].extend(component_fields)
        
        return categories

    @field_validator("edx_arn")
    @classmethod
    def validate_edx_arn_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate EDX ARN format if provided."""
        if v is None:
            return v
        
        if not v.startswith("arn:amazon:edx:"):
            raise ValueError(
                f"edx_arn must start with 'arn:amazon:edx:', got '{v}'"
            )
        
        return v

    @field_validator("edx_manifest_key")
    @classmethod
    def validate_manifest_key_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate that edx_manifest_key is in the format '[...]' if provided."""
        if v is None:
            return v
        
        if not (v.startswith("[") and v.endswith("]")):
            raise ValueError(
                f"edx_manifest_key must be in format '[\"xxx\",...]', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_edx_input_mode(self) -> "EdxDataSourceConfig":
        """Ensure either edx_arn OR all component fields are provided."""
        
        has_arn = self.edx_arn is not None
        component_fields = [self.edx_provider, self.edx_subject, self.edx_dataset, self.edx_manifest_key]
        has_components = all(field is not None for field in component_fields)
        has_any_components = any(field is not None for field in component_fields)
        
        if has_arn and has_any_components:
            raise ValueError(
                "Cannot provide both edx_arn and component fields "
                "(edx_provider, edx_subject, edx_dataset, edx_manifest_key). "
                "Use either edx_arn OR the individual components."
            )
        
        if not has_arn and not has_components:
            missing_fields = [
                name for name, value in [
                    ("edx_provider", self.edx_provider),
                    ("edx_subject", self.edx_subject), 
                    ("edx_dataset", self.edx_dataset),
                    ("edx_manifest_key", self.edx_manifest_key)
                ] if value is None
            ]
            raise ValueError(
                f"When edx_arn is not provided, all component fields are required. "
                f"Missing: {missing_fields}"
            )
        
        # Initialize derived field (will be computed by property)
        self._edx_manifest = None
        return self


class AndesDataSourceConfig(BaseCradleComponentConfig):
    """
    Configuration for Andes data source with three-tier field classification.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    provider: str = Field(description="Andes provider ID (32-digit UUID or 'booker')")

    table_name: str = Field(description="Name of the Andes table")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    andes3_enabled: bool = Field(
        default=True, description="Whether the table uses Andes 3.0 with latest version"
    )

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    # Model configuration overrides
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Changed from "forbid" to "allow" to fix circular reference handling
        str_strip_whitespace=True,
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """
        Validate that the provider is either:
        1. A valid 32-character UUID
        2. The special case 'booker'
        """
        if v == "booker":
            return v

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )

        if not uuid_pattern.match(v.lower()):
            raise ValueError(
                "provider must be either 'booker' or a valid 32-digit UUID "
                "(8-4-4-4-12 format). "
                "Verify provider validity at: "
                f"https://datacentral.a2z.com/hoot/providers/{v}"
            )

        return v

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """
        Validate that the table name is not empty and follows valid format.
        """
        if not v or not v.strip():
            raise ValueError("table_name cannot be empty")

        return v

    def __str__(self) -> str:
        """String representation of the Andes config."""
        return (
            f"AndesDataSourceConfig(provider='{self.provider}', "
            f"table_name='{self.table_name}', "
            f"andes3_enabled={self.andes3_enabled})"
        )


class DataSourceConfig(BaseCradleComponentConfig):
    """
    Configuration for data sources with three-tier field classification.

    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasource.DataSource

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    data_source_name: str = Field(
        description="Logical name for this data source (e.g. 'RAW_MDS_NA' or 'TAGS')"
    )

    data_source_type: str = Field(description="One of 'MDS', 'EDX', or 'ANDES'")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    mds_data_source_properties: Optional[MdsDataSourceConfig] = Field(
        default=None, description="If data_source_type=='MDS', this must be provided"
    )

    edx_data_source_properties: Optional[EdxDataSourceConfig] = Field(
        default=None, description="If data_source_type=='EDX', this must be provided"
    )

    andes_data_source_properties: Optional[AndesDataSourceConfig] = Field(
        default=None, description="If data_source_type=='ANDES', this must be provided"
    )

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    # Override model_config to set frozen=True for this class
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Changed from "forbid" to "allow" to fix circular reference handling
        frozen=True,
    )

    @field_validator("data_source_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed: Set[str] = {"MDS", "EDX", "ANDES"}
        if v not in allowed:
            raise ValueError(f"data_source_type must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    @classmethod
    def check_properties(cls, model: "DataSourceConfig") -> "DataSourceConfig":
        """
        Ensure the appropriate properties are set based on data_source_type
        and that only one set of properties is provided.
        """
        t = model.data_source_type

        # Check required properties are present
        if t == "MDS" and model.mds_data_source_properties is None:
            raise ValueError(
                "mds_data_source_properties must be set when data_source_type=='MDS'"
            )
        if t == "EDX" and model.edx_data_source_properties is None:
            raise ValueError(
                "edx_data_source_properties must be set when data_source_type=='EDX'"
            )
        if t == "ANDES" and model.andes_data_source_properties is None:
            raise ValueError(
                "andes_data_source_properties must be set when data_source_type=='ANDES'"
            )

        # Ensure only one set of properties is provided
        properties_count = sum(
            1
            for prop in [
                model.mds_data_source_properties,
                model.edx_data_source_properties,
                model.andes_data_source_properties,
            ]
            if prop is not None
        )

        if properties_count > 1:
            raise ValueError(
                "Only one of mds_data_source_properties, edx_data_source_properties, "
                "or andes_data_source_properties should be provided"
            )

        return model


class DataSourcesSpecificationConfig(BaseCradleComponentConfig):
    """
    Configuration for data sources specification with three-tier field classification.

    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasourcesspecification.DataSourcesSpecification

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    start_date: str = Field(
        description="Start timestamp exactly 'YYYY-mm-DDTHH:MM:SS', e.g. '2025-01-01T00:00:00'"
    )

    end_date: str = Field(
        description="End timestamp exactly 'YYYY-mm-DDTHH:MM:SS', e.g. '2025-04-17T00:00:00'"
    )

    data_sources: List[DataSourceConfig] = Field(
        description="List of DataSourceConfig objects (both MDS and EDX)"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # None currently for this class

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_exact_datetime_format(cls, v: str, field) -> str:
        """
        Must match exactly "%Y-%m-%dT%H:%M:%S"
        """
        try:
            parsed = datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            raise ValueError(
                f"{field.name!r} must be in format YYYY-mm-DD'T'HH:MM:SS "
                f"(e.g. '2025-01-01T00:00:00'), got {v!r}"
            )
        if parsed.strftime("%Y-%m-%dT%H:%M:%S") != v:
            raise ValueError(
                f"{field.name!r} does not match the required format exactly; got {v!r}"
            )
        return v


class JobSplitOptionsConfig(BaseCradleComponentConfig):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.jobsplitoptions.JobSplitOptions:
      - split_job: bool
      - days_per_split: int
      - merge_sql: str

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide (when split_job=True)
    merge_sql: Optional[str] = Field(
        default=None,
        description="SQL to run after merging split results (if split_job=True). "
        "For example: 'SELECT * FROM INPUT'.",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    split_job: bool = Field(
        default=False,
        description="Whether to split the Cradle job into multiple daily runs",
    )

    days_per_split: int = Field(
        default=7, description="Number of days per split (only used if split_job=True)"
    )

    @field_validator("days_per_split")
    @classmethod
    def days_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("days_per_split must be ≥ 1")
        return v

    @model_validator(mode="after")
    @classmethod
    def require_merge_sql_if_split(
        cls, model: "JobSplitOptionsConfig"
    ) -> "JobSplitOptionsConfig":
        if model.split_job and not model.merge_sql:
            raise ValueError("If split_job=True, merge_sql must be provided")
        return model


class TransformSpecificationConfig(BaseCradleComponentConfig):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.transformspecification.TransformSpecification:
      - transform_sql: str
      - job_split_options: JobSplitOptionsConfig

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    transform_sql: str = Field(
        description="The SQL string used to join MDS and TAGS (or do any other transformation)."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    job_split_options: JobSplitOptionsConfig = Field(
        default_factory=JobSplitOptionsConfig,
        description="Options for splitting the Cradle job into multiple runs",
    )

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class


class OutputSpecificationConfig(BaseCradleComponentConfig):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.outputspecification.OutputSpecification:
      - output_schema: List[str]
      - output_format: str (e.g. 'PARQUET', 'CSV', etc.)
      - output_save_mode: str (e.g. 'ERRORIFEXISTS', 'OVERWRITE', 'APPEND', 'IGNORE')
      - output_file_count: int (0 means "auto")
      - keep_dot_in_output_schema: bool
      - include_header_in_s3_output: bool

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    output_schema: List[str] = Field(
        description="List of column names to emit (e.g. ['objectId','transactionDate',…])."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration'] to indicate which dataset this job is pulling"
    )

    # Pipeline S3 location - needed for output_path calculation
    pipeline_s3_loc: Optional[str] = Field(
        default=None,
        description="S3 location for pipeline artifacts (inherited from parent config)",
    )

    output_format: str = Field(
        default="PARQUET",
        description="Format for Cradle output: one of ['CSV','UNESCAPED_TSV','JSON','ION','PARQUET']",
    )

    output_save_mode: str = Field(
        default="ERRORIFEXISTS",
        description="One of ['ERRORIFEXISTS','OVERWRITE','APPEND','IGNORE']",
    )

    output_file_count: int = Field(
        default=0, ge=0, description="Number of output files (0 means auto‐split)"
    )

    keep_dot_in_output_schema: bool = Field(
        default=False,
        description="If False, replace '.' with '__DOT__' in the output header",
    )

    include_header_in_s3_output: bool = Field(
        default=True, description="Whether to write the header row in S3 output files"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _output_path: Optional[str] = PrivateAttr(default=None)

    @property
    def output_path(self) -> str:
        """Get output path derived from pipeline_s3_loc and job_type."""
        if self._output_path is None:
            # Use the explicitly provided pipeline_s3_loc field if available
            if self.pipeline_s3_loc:
                self._output_path = f"{self.pipeline_s3_loc}/data-load/{self.job_type}"
            else:
                # Fallback for backward compatibility
                self._output_path = f"s3://default-bucket/data-load/{self.job_type}"
        return self._output_path

    # Property validator to ensure the output_path is a valid S3 URI
    def validate_output_path(self) -> None:
        """Validate that output_path is a valid S3 URI."""
        # Make sure we have pipeline_s3_loc set before validation
        if not hasattr(self, "pipeline_s3_loc") or not self.pipeline_s3_loc:
            # Don't try to validate without pipeline_s3_loc - it will use default
            return

        if not self.output_path.startswith("s3://"):
            raise ValueError("output_path must start with 's3://'")

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"CSV", "UNESCAPED_TSV", "JSON", "ION", "PARQUET"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v

    @field_validator("output_save_mode")
    @classmethod
    def validate_save_mode(cls, v: str) -> str:
        allowed = {"ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"}
        if v not in allowed:
            raise ValueError(f"output_save_mode must be one of {allowed}")
        return v


class CradleJobSpecificationConfig(BaseCradleComponentConfig):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.cradlejobspecification.CradleJobSpecification:
      - cluster_type: str (e.g. 'SMALL', 'MEDIUM', 'LARGE')
      - cradle_account: str
      - extra_spark_job_arguments: Optional[str]
      - job_retry_count: int

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    cradle_account: str = Field(
        description="Cradle account name (e.g. 'Buyer-Abuse-RnD-Dev')"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    cluster_type: str = Field(
        default="STANDARD",
        description="Cluster size for Cradle job (e.g. 'STANDARD', 'SMALL', 'MEDIUM', 'LARGE')",
    )

    extra_spark_job_arguments: Optional[str] = Field(
        default="", description="Any extra Spark driver options (string or blank)"
    )

    job_retry_count: int = Field(
        default=1, ge=0, description="Number of times to retry on failure (default=1)"
    )

    @field_validator("cluster_type")
    @classmethod
    def validate_cluster_type(cls, v: str) -> str:
        allowed = {"STANDARD", "SMALL", "MEDIUM", "LARGE"}
        if v not in allowed:
            raise ValueError(f"cluster_type must be one of {allowed}, got '{v}'")
        return v


class CradleDataLoadingConfig(BasePipelineConfig):
    """
    Top‐level configuration for creating a CreateCradleDataLoadJobRequest with three-tier field classification.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)

    This class inherits from BasePipelineConfig (not BaseCradleComponentConfig) to maintain
    consistency with other pipeline configurations.
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    job_type: str = Field(
        description="One of ['training','validation','testing','calibration'] to indicate which dataset this job is pulling"
    )

    data_sources_spec: DataSourcesSpecificationConfig = Field(
        description="Full data‐sources specification (start/end dates plus list of sources)."
    )

    transform_spec: TransformSpecificationConfig = Field(
        description="Transform specification: SQL + job‐split options."
    )

    output_spec: OutputSpecificationConfig = Field(
        description="Output specification: schema, output format, save mode, etc."
    )

    cradle_job_spec: CradleJobSpecificationConfig = Field(
        description="Cradle job specification: cluster type, account, retry count, etc."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    s3_input_override: Optional[str] = Field(
        default=None,
        description="If set, skip Cradle data pull and use this S3 prefix directly",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are initialized in the model_validator based on other fields
    # The output_path in output_spec is a derived field that depends on job_type

    # Model configuration - inherit from BasePipelineConfig.Config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields like __model_type__ and __model_module__ for type-aware serialization
    )

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "CradleDataLoadingConfig":
        """Initialize all derived fields once after validation."""
        # Initialize base class derived fields first
        super().initialize_derived_fields()

        # Override the output_spec job_type with the parent's job_type
        # This ensures consistency between parent and child configs
        self.output_spec.job_type = self.job_type

        # Pass the pipeline_s3_loc to output_spec for output_path calculation
        if hasattr(self, "pipeline_s3_loc"):
            self.output_spec.pipeline_s3_loc = self.pipeline_s3_loc

        return self

    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - fields with no defaults (required)
        2. Tier 2: System Inputs - fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes

        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Initialize categories
        categories = {
            "essential": [],  # Tier 1: Required, public
            "system": [],  # Tier 2: Optional (has default), public
            "derived": [],  # Tier 3: Public properties
        }

        # Get model fields
        model_fields = self.__class__.model_fields

        # Categorize public fields into essential (required) or system (with defaults)
        for field_name, field_info in model_fields.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Use is_required() to determine if a field is essential
            if field_info.is_required():
                categories["essential"].append(field_name)
            else:
                categories["system"].append(field_name)

        # Find derived properties (public properties that aren't in model_fields)
        for attr_name in dir(self):
            if (
                not attr_name.startswith("_")
                and attr_name not in model_fields
                and isinstance(getattr(type(self), attr_name, None), property)
            ):
                categories["derived"].append(attr_name)

        # Add nested derived fields
        if hasattr(self.output_spec, "categorize_fields"):
            nested_categories = self.output_spec.categorize_fields()
            # Add the nested derived fields with a prefix
            for nested_field in nested_categories["derived"]:
                categories["derived"].append(f"output_spec.{nested_field}")

        return categories

    def get_all_tiered_fields(self) -> Dict[str, List[str]]:
        """
        Get a flattened list of all fields (including nested fields)
        organized by tier.

        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to
            lists of field names with dot notation for nesting
        """
        return get_flattened_fields(self)

    def check_split_and_override(self) -> None:
        """Check consistency of split settings and overrides."""
        # If splitting is enabled, merge_sql must be provided
        if (
            self.transform_spec.job_split_options.split_job
            and not self.transform_spec.job_split_options.merge_sql
        ):
            raise ValueError("When split_job=True, merge_sql must be provided")
