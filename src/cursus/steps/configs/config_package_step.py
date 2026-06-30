from pydantic import Field, model_validator
from typing import Dict, List, Optional, TYPE_CHECKING

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    pass


class PackageConfig(ProcessingStepConfigBase):
    """
    Configuration for a model packaging step.

    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="package.py", description="Entry point script for packaging."
    )

    # Update to Pydantic V2 style model_config
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",  # Allow extra fields like __model_type__ and __model_module__ for type-aware serialization
    }

    @model_validator(mode="after")
    def validate_config(self) -> "PackageConfig":
        """
        Validate configuration and ensure defaults are set.

        This validator ensures that:
        1. Entry point is provided
        2. Script contract is available and valid
        3. Required input paths are defined in the script contract
        """
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("packaging step requires a processing_entry_point")

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        if "model_input" not in contract.expected_input_paths:
            raise ValueError("Script contract missing required input path: model_input")

        if "inference_scripts_input" not in contract.expected_input_paths:
            raise ValueError(
                "Script contract missing required input path: inference_scripts_input"
            )

        return self

    def get_environment_variables(
        self, declared_env_vars: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Packaging env vars (the single env source; moved here from the builder, FZ 31e1d3g).

        ``declared_env_vars`` accepted for the builder's names-driven contract but ignored — these
        are config-derived names (PIPELINE_NAME, REGION, …) emitted only when the underlying field
        is present, preserving the builder's original conditional-add behavior.
        """
        env_vars: Dict[str, str] = {}
        if getattr(self, "pipeline_name", None) is not None:
            env_vars["PIPELINE_NAME"] = self.pipeline_name
        if getattr(self, "region", None) is not None:
            env_vars["REGION"] = self.region
        for attr, env_key in [
            ("model_type", "MODEL_TYPE"),
            ("bucket", "BUCKET_NAME"),
            ("pipeline_version", "PIPELINE_VERSION"),
            ("model_objective", "MODEL_OBJECTIVE"),
        ]:
            value = getattr(self, attr, None)
            if value is not None:
                env_vars[env_key] = str(value)
        return env_vars

    def inference_scripts_source(self) -> str:
        """Local source for the packaging step's ``inference_scripts_input`` (FZ 31e1d3i).

        The packaging step always mounts inference scripts from a LOCAL path (overriding any
        dependency-resolved value). Delegates to ``effective_source_dir`` — the single comprehensive
        source-dir resolver (hybrid ``processing_source_dir`` → hybrid ``source_dir`` → legacy
        values) on ProcessingStepConfigBase — falling back to the literal ``"inference"`` only when
        no source dir is configured.

        NOTE: the original builder used ``resolved_source_dir or source_dir or "inference"``, which
        reimplemented a PARTIAL version of this resolution and silently IGNORED
        ``processing_source_dir`` (falling through to ``"inference"`` when only that was set — a
        latent bug). Using ``effective_source_dir`` fixes that and removes the duplicated chain.
        """
        return self.effective_source_dir or "inference"

    # Removed get_script_path override - now inherits modernized version from ProcessingStepConfigBase
    # which includes hybrid resolution and comprehensive fallbacks
    # The contract fallback logic was deemed unnecessary since processing_entry_point has a default value
