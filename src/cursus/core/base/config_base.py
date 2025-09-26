"""
Base Pipeline Configuration with Self-Contained Derivation Logic

This module implements the base configuration class for pipeline steps using a
self-contained design where each configuration class is responsible for its own
field derivations through private fields and read-only properties.
"""

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ValidationInfo,
    PrivateAttr,
    ConfigDict,
)
from typing import List, Optional, Dict, Any, ClassVar, TYPE_CHECKING, cast
from pathlib import Path
import json
from datetime import datetime
import logging
import inspect
from abc import ABC, abstractmethod

# Import for type hints only
if TYPE_CHECKING:
    from .contract_base import ScriptContract
else:
    # Just for type hints, won't be used at runtime if not available
    ScriptContract = Any

logger = logging.getLogger(__name__)

# Note: Removed circular import to steps.registry.step_names
# Step registry will be accessed via lazy loading when needed


class BasePipelineConfig(BaseModel, ABC):
    """Base configuration with shared pipeline attributes and self-contained derivation logic."""

    # Class variables using ClassVar for Pydantic
    _REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2",
    }

    _STEP_NAMES: ClassVar[Dict[str, str]] = {}  # Will be populated via lazy loading

    # For internal caching (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # Step catalog instance for optimized component discovery (lazy-loaded)
    _step_catalog: Optional[Any] = PrivateAttr(default=None)

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    author: str = Field(description="Author or owner of the pipeline.")

    bucket: str = Field(description="S3 bucket name for pipeline artifacts and data.")

    role: str = Field(description="IAM role for pipeline execution.")

    region: str = Field(
        description="Custom region code (NA, EU, FE) for internal logic."
    )

    service_name: str = Field(description="Service name for the pipeline.")

    pipeline_version: str = Field(
        description="Version string for the SageMaker Pipeline."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    model_class: str = Field(
        default="xgboost", description="Model class (e.g., XGBoost, PyTorch)."
    )

    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing.",
    )

    framework_version: str = Field(
        default="2.1.0", description="Default framework version (e.g., PyTorch)."
    )

    py_version: str = Field(default="py310", description="Default Python version.")

    source_dir: Optional[str] = Field(
        default=None,
        description="Common source directory for scripts if applicable. Can be overridden by step configs.",
    )

    # ===== Tier 1 Hybrid Resolution Fields =====
    # These fields are required for the hybrid path resolution system
    
    project_root_folder: str = Field(
        description="Root folder name for the user's project (required for hybrid resolution)"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    _pipeline_description: Optional[str] = PrivateAttr(default=None)
    _pipeline_s3_loc: Optional[str] = PrivateAttr(default=None)
    _effective_source_dir: Optional[str] = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields for type-aware serialization
        protected_namespaces=(),
    )

    # Public read-only properties for derived fields

    @property
    def aws_region(self) -> str:
        """Get AWS region based on region code."""
        if self._aws_region is None:
            self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        return self._aws_region

    @property
    def pipeline_name(self) -> str:
        """Get pipeline name derived from author, service_name, model_class, and region."""
        if self._pipeline_name is None:
            self._pipeline_name = (
                f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
            )
        return self._pipeline_name

    @property
    def pipeline_description(self) -> str:
        """Get pipeline description derived from service_name, model_class, and region."""
        if self._pipeline_description is None:
            self._pipeline_description = (
                f"{self.service_name} {self.model_class} Model {self.region}"
            )
        return self._pipeline_description

    @property
    def pipeline_s3_loc(self) -> str:
        """Get S3 location for pipeline artifacts."""
        if self._pipeline_s3_loc is None:
            pipeline_subdirectory = "MODS"
            pipeline_subsubdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
            self._pipeline_s3_loc = (
                f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
            )
        return self._pipeline_s3_loc

    @property
    def effective_source_dir(self) -> Optional[str]:
        """
        Get effective source directory with hybrid resolution.
        
        This base implementation works with just source_dir (which can be None).
        Processing configs override this to handle both processing_source_dir and source_dir.
        
        Resolution Priority:
        1. Hybrid resolution of source_dir
        2. Legacy value (source_dir)
        3. None if source_dir is not provided
        """
        if self._effective_source_dir is None:
            # Only proceed if source_dir is provided
            if self.source_dir:
                # Strategy 1: Hybrid resolution of source_dir
                resolved = self.resolve_hybrid_path(self.source_dir)
                if resolved and Path(resolved).exists():
                    self._effective_source_dir = resolved
                    return self._effective_source_dir
                
                # Strategy 2: Legacy fallback (current behavior)
                self._effective_source_dir = self.source_dir
            else:
                # source_dir is None - this is valid for base config
                self._effective_source_dir = None
        
        return self._effective_source_dir


    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["aws_region"] = self.aws_region
        data["pipeline_name"] = self.pipeline_name
        data["pipeline_description"] = self.pipeline_description
        data["pipeline_s3_loc"] = self.pipeline_s3_loc
        if self.effective_source_dir is not None:
            data["effective_source_dir"] = self.effective_source_dir
        
        return data

    def __str__(self) -> str:
        """
        Custom string representation that shows fields by category.
        This overrides the default __str__ method so that print(config) shows
        a nicely formatted representation with fields organized by tier.

        Returns:
            A formatted string with fields organized by tier
        """
        # Use StringIO to build the string
        from io import StringIO

        output = StringIO()

        # Get class name
        print(f"=== {self.__class__.__name__} ===", file=output)

        # Get fields categorized by tier
        categories = self.categorize_fields()

        # Print Tier 1 fields (essential user inputs)
        if categories["essential"]:
            print("\n- Essential User Inputs -", file=output)
            for field_name in sorted(categories["essential"]):
                print(f"{field_name}: {getattr(self, field_name)}", file=output)

        # Print Tier 2 fields (system inputs with defaults)
        if categories["system"]:
            print("\n- System Inputs -", file=output)
            for field_name in sorted(categories["system"]):
                value = getattr(self, field_name)
                if value is not None:  # Skip None values for cleaner output
                    print(f"{field_name}: {value}", file=output)

        # Print Tier 3 fields (derived properties)
        if categories["derived"]:
            print("\n- Derived Fields -", file=output)
            for field_name in sorted(categories["derived"]):
                try:
                    value = getattr(self, field_name)
                    if not callable(value):  # Skip methods
                        print(f"{field_name}: {value}", file=output)
                except Exception:
                    # Skip properties that cause errors
                    pass

        return output.getvalue()

    # Validators

    @field_validator("region")
    @classmethod
    def _validate_custom_region(cls, v: str) -> str:
        """Validate region code."""
        valid_regions = ["NA", "EU", "FE"]
        if v not in valid_regions:
            raise ValueError(
                f"Invalid custom region code: {v}. Must be one of {valid_regions}"
            )
        return v

    # Removed source_dir validator to improve configuration portability
    # Path validation should happen at execution time in builders, not at config creation time

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "BasePipelineConfig":
        """Initialize all derived fields once after validation."""
        # Direct assignment to private fields avoids triggering validation
        self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        self._pipeline_name = (
            f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        )
        self._pipeline_description = (
            f"{self.service_name} {self.model_class} Model {self.region}"
        )

        pipeline_subdirectory = "MODS"
        pipeline_subsubdirectory = f"{self._pipeline_name}_{self.pipeline_version}"
        self._pipeline_s3_loc = (
            f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        )

        return self

    @property
    def step_catalog(self) -> Optional[Any]:
        """
        Lazy-loaded step catalog instance for optimized component discovery.
        
        Returns:
            StepCatalog instance or None if initialization fails
        """
        if self._step_catalog is None:
            try:
                # Import StepCatalog with proper error handling
                from ...step_catalog.step_catalog import StepCatalog
                
                # Initialize with workspace awareness if possible
                workspace_dirs = self._detect_workspace_dirs()
                self._step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
                
                logger.debug(f"Initialized StepCatalog with workspace_dirs: {workspace_dirs}")
                
            except ImportError as e:
                logger.warning(f"StepCatalog not available: {e}")
                self._step_catalog = None
            except Exception as e:
                logger.error(f"Error initializing StepCatalog: {e}")
                self._step_catalog = None
        
        return self._step_catalog

    def _detect_workspace_dirs(self) -> Optional[List[Path]]:
        """
        Detect workspace directories based on current configuration context.
        
        Returns:
            List of workspace directories or None if not detected
        """
        try:
            # Try to detect workspace from config file location
            config_file = Path(inspect.getfile(self.__class__))
            
            # Check if we're in a workspace structure
            current_dir = config_file.parent
            while current_dir.parent != current_dir:
                # Look for workspace indicators
                if (current_dir / "development" / "projects").exists():
                    logger.debug(f"Detected workspace directory: {current_dir}")
                    return [current_dir]
                current_dir = current_dir.parent
            
            # No workspace detected
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting workspace directories: {e}")
            return None

    def _derive_step_name(self) -> str:
        """
        Get step name from configuration class using registry mapping.
        
        This method uses the step registry as the primary source of truth for step names,
        falling back to derivation logic only when the class is not found in the registry.
        This ensures compatibility with the step catalog system and avoids naming issues
        like "XGBoostModelEval" being incorrectly converted to "x_g_boost_model_eval".
        
        Returns:
            Step name from registry or derived name as fallback
        """
        class_name = self.__class__.__name__
        
        # Strategy 1: Use registry mapping first (most reliable)
        try:
            step_name = self.get_step_name(class_name)
            if step_name != class_name:  # Found in registry
                logger.debug(f"Found step name in registry: {class_name} → {step_name}")
                
                # Handle job_type variants if available
                if hasattr(self, 'job_type') and self.job_type:
                    step_name = f"{step_name}_{self.job_type.lower()}"
                
                return step_name
        except Exception as e:
            logger.debug(f"Registry lookup failed for {class_name}: {e}")
        
        # Strategy 2: Fallback to derivation (for classes not in registry)
        logger.debug(f"Using derivation fallback for {class_name}")
        
        # Remove 'Config' suffix if present
        if class_name.endswith('Config'):
            step_name = class_name[:-6]  # Remove 'Config'
        else:
            step_name = class_name
        
        # Improved PascalCase to snake_case conversion that handles acronyms better
        import re
        # Handle consecutive uppercase letters (acronyms) first
        step_name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', step_name)
        # Then handle normal PascalCase
        step_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', step_name)
        step_name = step_name.lower()
        
        # Handle job_type variants if available
        if hasattr(self, 'job_type') and self.job_type:
            step_name = f"{step_name}_{self.job_type.lower()}"
        
        logger.debug(f"Derived step name: {class_name} → {step_name}")
        return step_name

    def get_script_contract(self) -> Optional["ScriptContract"]:
        """
        Get script contract for this configuration using optimized step catalog discovery.

        This optimized implementation uses the step catalog system for efficient contract
        discovery with O(1) lookups and workspace awareness, falling back to legacy
        methods for backward compatibility.

        Returns:
            Script contract instance or None if not available
        """
        # Check cache first for performance
        cache_key = "script_contract"
        if cache_key in self._cache:
            return cast(Optional["ScriptContract"], self._cache[cache_key])

        # Check for hardcoded script_contract first (for backward compatibility)
        if hasattr(self, "_script_contract"):
            contract = cast(Optional["ScriptContract"], self._script_contract)
            self._cache[cache_key] = contract
            return contract

        # OPTIMIZATION: Use step catalog for efficient contract discovery
        try:
            step_catalog = self.step_catalog
            if step_catalog:
                step_name = self._derive_step_name()
                
                logger.debug(f"Attempting to load contract for step: {step_name}")
                
                # Use step catalog's optimized contract loading
                contract = step_catalog.load_contract_class(step_name)
                if contract:
                    logger.debug(f"Successfully loaded contract via step catalog for {step_name}")
                    self._cache[cache_key] = contract
                    return cast(Optional["ScriptContract"], contract)
                else:
                    logger.debug(f"No contract found via step catalog for {step_name}")
            else:
                logger.debug("Step catalog not available, falling back to legacy method")
        
        except Exception as e:
            logger.debug(f"Error using step catalog for contract discovery: {e}")

        # FALLBACK: Legacy hardcoded import method for backward compatibility
        try:
            class_name = self.__class__.__name__.replace("Config", "")

            # Try with job_type if available
            if hasattr(self, "job_type") and self.job_type:
                module_name = f"...steps.contracts.{class_name.lower()}_{self.job_type.lower()}_contract"
                contract_name = f"{class_name.upper()}_{self.job_type.upper()}_CONTRACT"

                try:
                    contract_module = __import__(module_name, fromlist=[""])
                    if hasattr(contract_module, contract_name):
                        contract = cast(Optional["ScriptContract"], getattr(contract_module, contract_name))
                        logger.debug(f"Successfully loaded contract via legacy method with job_type: {contract_name}")
                        self._cache[cache_key] = contract
                        return contract
                except (ImportError, AttributeError):
                    pass

            # Try without job_type
            module_name = f"...steps.contracts.{class_name.lower()}_contract"
            contract_name = f"{class_name.upper()}_CONTRACT"

            try:
                contract_module = __import__(module_name, fromlist=[""])
                if hasattr(contract_module, contract_name):
                    contract = cast(Optional["ScriptContract"], getattr(contract_module, contract_name))
                    logger.debug(f"Successfully loaded contract via legacy method: {contract_name}")
                    self._cache[cache_key] = contract
                    return contract
            except (ImportError, AttributeError):
                pass

        except Exception as e:
            logger.debug(f"Error in legacy contract loading: {e}")

        # Cache the None result to avoid repeated failed lookups
        self._cache[cache_key] = None
        logger.debug(f"No contract found for configuration: {self.__class__.__name__}")
        return None

    @property
    def script_contract(self) -> Optional["ScriptContract"]:
        """
        Property accessor for script contract.

        Returns:
            Script contract instance or None if not available
        """
        return self.get_script_contract()

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path for this configuration.
        
        This method provides a default implementation that returns None, since not all
        step types require scripts (e.g., Model creation steps don't need scripts).
        
        Derived classes that need script paths should override this method with their
        specific requirements:
        - Processing steps: Combine entry_point with source_dir
        - Training steps: Use contract entry_point or combine with source_dir
        - Model steps: May not need scripts at all
        
        Args:
            default_path: Default script path to use if not found via other methods
            
        Returns:
            Script path, default_path, or None if not applicable for this step type
        """
        # Default implementation returns None since not all step types need scripts
        return default_path

    def resolve_hybrid_path(self, relative_path: str) -> Optional[str]:
        """
        Resolve a path using the hybrid path resolution system.
        
        This method uses the hybrid path resolution system to find files across
        different deployment scenarios (Lambda/MODS bundled, development monorepo,
        pip-installed separated).
        
        Args:
            relative_path: Relative path from project root to target
            
        Returns:
            Resolved absolute path if found, None otherwise
        """
        if not self.project_root_folder or not relative_path:
            logger.debug("Missing project_root_folder or relative_path for hybrid resolution")
            return None
        
        try:
            from ..utils.hybrid_path_resolution import resolve_hybrid_path
            return resolve_hybrid_path(self.project_root_folder, relative_path)
        except ImportError:
            logger.debug("Hybrid path resolution not available")
            return None
        except Exception as e:
            logger.debug(f"Error in hybrid path resolution: {e}")
            return None

    @property
    def resolved_source_dir(self) -> Optional[str]:
        """
        Get resolved source directory using hybrid resolution.
        
        Returns None if source_dir is not provided, since it's optional in base class.
        Processing, training, and model step configs should ensure source_dir is provided.
        """
        if self.source_dir:
            return self.resolve_hybrid_path(self.source_dir)
        return None


    @classmethod
    def get_step_name(cls, config_class_name: str) -> str:
        """Get the step name for a configuration class using existing registry functions."""
        try:
            # Use the existing registry function with workspace awareness
            from ...registry.step_names import get_config_step_registry
            config_registry = get_config_step_registry()
            return config_registry.get(config_class_name, config_class_name)
        except ImportError:
            logger.debug("Registry not available, returning class name")
            return config_class_name

    @classmethod
    def get_config_class_name(cls, step_name: str) -> str:
        """Get the configuration class name from a step name using existing registry functions."""
        try:
            # Use the existing registry function with workspace awareness
            from ...registry.step_names import get_config_class_name
            return get_config_class_name(step_name)
        except ImportError:
            logger.debug("Registry not available, returning step name")
            return step_name

    @classmethod
    def _get_step_registry(
        cls, workspace_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Lazy load step registry with workspace context awareness.

        This method now supports workspace-aware step registry resolution by:
        1. Using hybrid registry manager for workspace-specific registries
        2. Falling back to traditional registry if hybrid is unavailable
        3. Maintaining backward compatibility with existing code

        Args:
            workspace_context: Optional workspace context for registry isolation

        Returns:
            Dict[str, str]: Step registry mapping for the specified workspace context
        """
        # Create a cache key that includes workspace context
        cache_key = f"_STEP_NAMES_{workspace_context or 'default'}"

        # Check if we already have this registry cached
        if not hasattr(cls, cache_key) or not getattr(cls, cache_key):
            try:
                # Try to use hybrid registry manager first
                try:
                    from ...registry.hybrid.manager import HybridRegistryManager

                    hybrid_manager = HybridRegistryManager()

                    # Get step registry using the actual available method
                    legacy_dict = hybrid_manager.create_legacy_step_names_dict(
                        workspace_context or "default"
                    )

                    # Convert to config step registry format (reverse mapping)
                    # Handle the case where values might be complex dictionaries
                    config_registry = {}
                    for k, v in legacy_dict.items():
                        if isinstance(v, dict):
                            # If value is a dict, use the key as both key and value for config registry
                            config_registry[k] = k
                        else:
                            # If value is a simple string, create reverse mapping
                            config_registry[str(v)] = k

                    if workspace_context:
                        logger.debug(
                            f"Loaded workspace-specific config step registry for context: {workspace_context}"
                        )
                    else:
                        logger.debug(
                            "Loaded default config step registry from hybrid registry"
                        )

                    setattr(cls, cache_key, config_registry)

                except ImportError:
                    # Fallback to traditional registry
                    logger.debug(
                        "Hybrid registry not available, falling back to traditional registry"
                    )
                    from ...registry.step_names import CONFIG_STEP_REGISTRY

                    setattr(cls, cache_key, CONFIG_STEP_REGISTRY)

            except ImportError:
                logger.warning("Could not import step registry, using empty registry")
                setattr(cls, cache_key, {})

        return cast(Dict[str, str], getattr(cls, cache_key))

    @classmethod
    def from_base_config(
        cls, base_config: "BasePipelineConfig", **kwargs: Any
    ) -> "BasePipelineConfig":
        """
        Create a new configuration instance from a base configuration.
        This is a virtual method that all derived classes can use to inherit from a parent config.

        Args:
            base_config: Parent BasePipelineConfig instance
            **kwargs: Additional arguments specific to the derived class

        Returns:
            A new instance of the derived class initialized with parent fields and additional kwargs
        """
        # Get public fields from parent
        parent_fields = base_config.get_public_init_fields()

        # Combine with additional fields (kwargs take precedence)
        config_dict = {**parent_fields, **kwargs}

        # Create new instance of the derived class (cls refers to the actual derived class)
        return cls(**config_dict)

    def categorize_fields(self) -> Dict[str, List[str]]:
        """
        Categorize all fields into three tiers:
        1. Tier 1: Essential User Inputs - public fields with no defaults (required)
        2. Tier 2: System Inputs - public fields with defaults (optional)
        3. Tier 3: Derived Fields - properties that access private attributes

        Returns:
            Dict with keys 'essential', 'system', and 'derived' mapping to lists of field names
        """
        # Initialize categories
        categories: Dict[str, List[str]] = {
            "essential": [],  # Tier 1: Required, public
            "system": [],  # Tier 2: Optional (has default), public
            "derived": [],  # Tier 3: Public properties
        }

        # Get model fields from the class (not instance) to avoid deprecation warnings
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

    def print_config(self) -> None:
        """
        Print complete configuration information organized by tiers.
        This method automatically categorizes fields by examining their characteristics:
        - Tier 1: Essential User Inputs (public fields without defaults)
        - Tier 2: System Inputs (public fields with defaults)
        - Tier 3: Derived Fields (properties that provide access to private fields)
        """
        print("\n===== CONFIGURATION =====")
        print(f"Class: {self.__class__.__name__}")

        # Get fields categorized by tier
        categories = self.categorize_fields()

        # Print Tier 1 fields (essential user inputs)
        print("\n----- Essential User Inputs (Tier 1) -----")
        for field_name in sorted(categories["essential"]):
            print(f"{field_name.title()}: {getattr(self, field_name)}")

        # Print Tier 2 fields (system inputs with defaults)
        print("\n----- System Inputs with Defaults (Tier 2) -----")
        for field_name in sorted(categories["system"]):
            value = getattr(self, field_name)
            if value is not None:  # Skip None values for cleaner output
                print(f"{field_name.title()}: {value}")

        # Print Tier 3 fields (derived properties)
        print("\n----- Derived Fields (Tier 3) -----")
        for field_name in sorted(categories["derived"]):
            try:
                value = getattr(self, field_name)
                if not callable(value):  # Skip methods
                    print(f"{field_name.title()}: {value}")
            except Exception as e:
                print(f"{field_name.title()}: <Error: {e}>")

        print("\n===================================\n")

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Get a dictionary of public fields suitable for initializing a child config.
        Only includes fields that should be passed to child class constructors.
        Both essential user inputs and system inputs with defaults or user-overridden values
        are included to ensure all user customizations are properly propagated to derived classes.

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
