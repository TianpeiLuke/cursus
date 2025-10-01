"""
Model Wiki Generator Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the model wiki generator step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, field_validator, PrivateAttr
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.model_wiki_generator_contract import MODEL_WIKI_GENERATOR_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class ModelWikiGeneratorConfig(ProcessingStepConfigBase):
    """
    Configuration for model wiki generator step with self-contained derivation logic.

    This class defines the configuration parameters for the model wiki generator step,
    which loads metrics data and visualizations, generates comprehensive wiki documentation,
    and creates multi-format model documentation. Supports automated documentation creation
    for model registries and compliance requirements.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    model_name: str = Field(
        ...,
        description="Name of the model for documentation (required for wiki generation).",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    # Note: processing_entry_point is inherited from ProcessingStepConfigBase with default=None
    # We override it here to provide a specific default for wiki generation

    processing_entry_point: str = Field(
        default="model_wiki_generator.py",
        description="Entry point script for model wiki generation.",
    )

    # Model metadata with defaults
    model_use_case: str = Field(
        default="Machine Learning Model",
        description="Description of model use case for documentation.",
    )

    team_alias: str = Field(
        default="ml-team@",
        description="Team email alias for documentation.",
    )

    contact_email: str = Field(
        default="ml-team@company.com",
        description="Point of contact email for documentation.",
    )

    cti_classification: str = Field(
        default="Internal",
        description="CTI classification for the model documentation.",
    )

    # Documentation generation options
    output_formats: str = Field(
        default="wiki,html,markdown",
        description="Comma-separated list of output formats (wiki,html,markdown).",
    )

    include_technical_details: bool = Field(
        default=True,
        description="Include technical details section in documentation.",
    )

    # Optional custom content
    model_description: Optional[str] = Field(
        default=None,
        description="Custom model description text (auto-generated if not provided).",
    )

    model_purpose: str = Field(
        default="perform classification tasks",
        description="Custom model purpose description for documentation.",
    )

    # For wiki generation, we typically use smaller instances as it's mostly text processing
    use_large_processing_instance: bool = Field(
        default=False, 
        description="Whether to use large instance type for processing (wiki generation typically needs minimal resources)"
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    _model_display_name: Optional[str] = PrivateAttr(default=None)
    _output_formats_list: Optional[List[str]] = PrivateAttr(default=None)
    _effective_model_description: Optional[str] = PrivateAttr(default=None)

    # Public properties for derived fields
    # Note: pipeline_name is inherited from BasePipelineConfig

    @property
    def model_display_name(self) -> str:
        """Get display name for the model in documentation."""
        if self._model_display_name is None:
            self._model_display_name = self.model_name.replace("_", " ").title()
        return self._model_display_name

    @property
    def output_formats_list(self) -> List[str]:
        """Get list of output formats from comma-separated string."""
        if self._output_formats_list is None:
            formats = [fmt.strip().lower() for fmt in self.output_formats.split(",")]
            # Validate formats
            valid_formats = {"wiki", "html", "markdown"}
            self._output_formats_list = [fmt for fmt in formats if fmt in valid_formats]
            if not self._output_formats_list:
                self._output_formats_list = ["wiki"]  # Default fallback
        return self._output_formats_list

    @property
    def effective_model_description(self) -> str:
        """Get effective model description (custom or auto-generated)."""
        if self._effective_model_description is None:
            if self.model_description:
                self._effective_model_description = self.model_description
            else:
                self._effective_model_description = f"This is a machine learning model for {self.model_use_case.lower()}."
        return self._effective_model_description

    # Field validators

    @field_validator("output_formats")
    @classmethod
    def validate_output_formats(cls, v: str) -> str:
        """Validate output formats are supported."""
        valid_formats = {"wiki", "html", "markdown"}
        formats = [fmt.strip().lower() for fmt in v.split(",")]
        
        invalid_formats = [fmt for fmt in formats if fmt not in valid_formats]
        if invalid_formats:
            raise ValueError(
                f"Invalid output formats: {invalid_formats}. "
                f"Valid formats are: {valid_formats}"
            )
        
        if not formats:
            raise ValueError("At least one output format must be specified")
        
        return ",".join(formats)

    @field_validator("cti_classification")
    @classmethod
    def validate_cti_classification(cls, v: str) -> str:
        """Validate CTI classification values."""
        valid_classifications = {
            "public", "internal", "confidential", "restricted", 
            "Public", "Internal", "Confidential", "Restricted"
        }
        if v not in valid_classifications:
            logger.warning(
                f"CTI classification '{v}' is not in standard classifications: {valid_classifications}"
            )
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty and contains valid characters."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        
        # Check for potentially problematic characters for file naming
        import re
        if re.search(r'[<>:"/\\|?*]', v):
            logger.warning(
                f"Model name '{v}' contains characters that may cause issues in file names"
            )
        
        return v.strip()

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ModelWikiGeneratorConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize wiki generator specific derived fields
        # Access properties to trigger initialization
        _ = self.pipeline_name
        _ = self.model_display_name
        _ = self.output_formats_list
        _ = self.effective_model_description

        return self

    @model_validator(mode="after")
    def validate_wiki_generator_config(self) -> "ModelWikiGeneratorConfig":
        """Additional validation specific to wiki generator configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("wiki generator step requires a processing_entry_point")

        # Validate required fields from script contract
        if not self.model_name:
            raise ValueError(
                "model_name must be provided (required by model wiki generator contract)"
            )

        # Validate output formats
        if not self.output_formats_list:
            raise ValueError("At least one valid output format must be specified")

        # Validate email format if provided
        if self.contact_email and "@" not in self.contact_email:
            logger.warning(f"contact_email '{self.contact_email}' may not be a valid email address")

        logger.debug(
            f"Model '{self.model_name}' will generate documentation in formats: {self.output_formats_list}"
        )

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model wiki generator script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add model wiki generator specific environment variables
        env_vars.update(
            {
                "MODEL_NAME": self.model_name,
                "MODEL_USE_CASE": self.model_use_case,
                "MODEL_VERSION": self.pipeline_version,  # Use pipeline_version from base config
                "PIPELINE_NAME": self.pipeline_name,
                "AUTHOR": self.author,  # From base config
                "TEAM_ALIAS": self.team_alias,
                "CONTACT_EMAIL": self.contact_email,
                "CTI_CLASSIFICATION": self.cti_classification,
                "REGION": self.region,  # From base config
                "OUTPUT_FORMATS": self.output_formats,
                "INCLUDE_TECHNICAL_DETAILS": str(self.include_technical_details).lower(),
                "MODEL_PURPOSE": self.model_purpose,
            }
        )

        # Add optional fields if specified
        if self.model_description:
            env_vars["MODEL_DESCRIPTION"] = self.model_description

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The model wiki generator script contract
        """
        return MODEL_WIKI_GENERATOR_CONTRACT

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        
        # Add derived properties to output
        data["pipeline_name"] = self.pipeline_name
        data["model_display_name"] = self.model_display_name
        data["output_formats_list"] = self.output_formats_list
        data["effective_model_description"] = self.effective_model_description
        
        return data

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include wiki generator specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and wiki generator specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add model wiki generator specific fields (only fields not in base classes)
        wiki_fields = {
            # Tier 1 - Essential User Inputs
            "model_name": self.model_name,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "model_use_case": self.model_use_case,
            "team_alias": self.team_alias,
            "contact_email": self.contact_email,
            "cti_classification": self.cti_classification,
            "output_formats": self.output_formats,
            "include_technical_details": self.include_technical_details,
            "model_purpose": self.model_purpose,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Only include optional fields if they're set
        if self.model_description is not None:
            wiki_fields["model_description"] = self.model_description

        # Combine base fields and wiki fields (wiki fields take precedence if overlap)
        init_fields = {**base_fields, **wiki_fields}

        return init_fields
