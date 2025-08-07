"""
Base Pipeline Configuration with Self-Contained Derivation Logic

This module implements the base configuration class for pipeline steps using a
self-contained design where each configuration class is responsible for its own
field derivations through private fields and read-only properties.
"""

from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo, PrivateAttr
from typing import List, Optional, Dict, Any, ClassVar, TYPE_CHECKING
from pathlib import Path
import json
from datetime import datetime
import logging

# Import for type hints only
if TYPE_CHECKING:
    from .contract_base import ScriptContract
else:
    # Just for type hints, won't be used at runtime if not available
    ScriptContract = Any

logger = logging.getLogger(__name__)

# Note: Removed circular import to steps.registry.step_names
# Step registry will be accessed via lazy loading when needed


class BasePipelineConfig(BaseModel):
    """Base configuration with shared pipeline attributes and self-contained derivation logic."""
    
    # Class variables using ClassVar for Pydantic
    _REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    _STEP_NAMES: ClassVar[Dict[str, str]] = {}  # Will be populated via lazy loading
    
    # For internal caching (completely private)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    author: str = Field(
        description="Author or owner of the pipeline.")
    
    bucket: str = Field(
        description="S3 bucket name for pipeline artifacts and data.")
        
    role: str = Field(
        description="IAM role for pipeline execution.")
        
    region: str = Field(
        description="Custom region code (NA, EU, FE) for internal logic.")
    
    service_name: str = Field(
        description="Service name for the pipeline.")
    
    pipeline_version: str = Field(
        description="Version string for the SageMaker Pipeline.")
    
    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    
    model_class: str = Field(
        default='xgboost', 
        description="Model class (e.g., XGBoost, PyTorch).")
    
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing.")
    
    framework_version: str = Field(
        default='2.1.0', 
        description="Default framework version (e.g., PyTorch).")
    
    py_version: str = Field(
        default='py310', 
        description="Default Python version.")
    
    source_dir: Optional[str] = Field(
        default=None, 
        description="Common source directory for scripts if applicable. Can be overridden by step configs.")
    
    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access
    
    _aws_region: Optional[str] = PrivateAttr(default=None)
    _pipeline_name: Optional[str] = PrivateAttr(default=None)
    _pipeline_description: Optional[str] = PrivateAttr(default=None)
    _pipeline_s3_loc: Optional[str] = PrivateAttr(default=None)


    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'allow'  # Allow extra fields for type-aware serialization
        protected_namespaces = ()
    
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
            self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        return self._pipeline_name
    
    @property
    def pipeline_description(self) -> str:
        """Get pipeline description derived from service_name, model_class, and region."""
        if self._pipeline_description is None:
            self._pipeline_description = f"{self.service_name} {self.model_class} Model {self.region}"
        return self._pipeline_description
    
    @property
    def pipeline_s3_loc(self) -> str:
        """Get S3 location for pipeline artifacts."""
        if self._pipeline_s3_loc is None:
            pipeline_subdirectory = "MODS"
            pipeline_subsubdirectory = f"{self.pipeline_name}_{self.pipeline_version}"
            self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        return self._pipeline_s3_loc
    
    
    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["aws_region"] = self.aws_region
        data["pipeline_name"] = self.pipeline_name
        data["pipeline_description"] = self.pipeline_description
        data["pipeline_s3_loc"] = self.pipeline_s3_loc
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
        if categories['essential']:
            print("\n- Essential User Inputs -", file=output)
            for field_name in sorted(categories['essential']):
                print(f"{field_name}: {getattr(self, field_name)}", file=output)
        
        # Print Tier 2 fields (system inputs with defaults)
        if categories['system']:
            print("\n- System Inputs -", file=output)
            for field_name in sorted(categories['system']):
                value = getattr(self, field_name)
                if value is not None:  # Skip None values for cleaner output
                    print(f"{field_name}: {value}", file=output)
        
        # Print Tier 3 fields (derived properties)
        if categories['derived']:
            print("\n- Derived Fields -", file=output)
            for field_name in sorted(categories['derived']):
                try:
                    value = getattr(self, field_name)
                    if not callable(value):  # Skip methods
                        print(f"{field_name}: {value}", file=output)
                except Exception:
                    # Skip properties that cause errors
                    pass
        
        return output.getvalue()


    # Validators
    
    @field_validator('region')
    @classmethod
    def _validate_custom_region(cls, v: str) -> str:
        """Validate region code."""
        valid_regions = ['NA', 'EU', 'FE']
        if v not in valid_regions:
            raise ValueError(f"Invalid custom region code: {v}. Must be one of {valid_regions}")
        return v
    
    @field_validator('source_dir', check_fields=False)
    @classmethod
    def _validate_source_dir_exists(cls, v: Optional[str]) -> Optional[str]:
        """Validate that source_dir exists if it's a local path."""
        if v is not None and not v.startswith('s3://'):  # Only validate local paths
            if not Path(v).exists():
                logger.warning(f"Local source directory does not exist: {v}")
                raise ValueError(f"Local source directory does not exist: {v}")
            if not Path(v).is_dir():
                logger.warning(f"Local source_dir is not a directory: {v}")
                raise ValueError(f"Local source_dir is not a directory: {v}")
        return v
    
    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'BasePipelineConfig':
        """Initialize all derived fields once after validation."""
        # Direct assignment to private fields avoids triggering validation
        self._aws_region = self._REGION_MAPPING.get(self.region, "us-east-1")
        self._pipeline_name = f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        self._pipeline_description = f"{self.service_name} {self.model_class} Model {self.region}"
        
        pipeline_subdirectory = "MODS"
        pipeline_subsubdirectory = f"{self._pipeline_name}_{self.pipeline_version}"
        self._pipeline_s3_loc = f"s3://{self.bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        
        return self

    def get_script_contract(self) -> Optional['ScriptContract']:
        """
        Get script contract for this configuration.
        
        This base implementation returns None. Derived classes should override
        this method to return their specific script contract.
        
        Returns:
            Script contract instance or None if not available
        """
        # Check for hardcoded script_contract first (for backward compatibility)
        if hasattr(self, '_script_contract'):
            return self._script_contract
            
        # Otherwise attempt to load based on class and job_type
        try:
            class_name = self.__class__.__name__.replace('Config', '')
            
            # Try with job_type if available
            if hasattr(self, 'job_type') and self.job_type:
                module_name = f"...steps.contracts.{class_name.lower()}_{self.job_type.lower()}_contract"
                contract_name = f"{class_name.upper()}_{self.job_type.upper()}_CONTRACT"
                
                try:
                    contract_module = __import__(module_name, fromlist=[''])
                    if hasattr(contract_module, contract_name):
                        return getattr(contract_module, contract_name)
                except (ImportError, AttributeError):
                    pass
            
            # Try without job_type
            module_name = f"...steps.contracts.{class_name.lower()}_contract"
            contract_name = f"{class_name.upper()}_CONTRACT"
            
            try:
                contract_module = __import__(module_name, fromlist=[''])
                if hasattr(contract_module, contract_name):
                    return getattr(contract_module, contract_name)
            except (ImportError, AttributeError):
                pass
                
        except Exception as e:
            logger.debug(f"Error loading script contract: {e}")
            
        return None
        
    @property
    def script_contract(self) -> Optional['ScriptContract']:
        """
        Property accessor for script contract.
        
        Returns:
            Script contract instance or None if not available
        """
        return self.get_script_contract()
        
    def get_script_path(self, default_path: str = None) -> str:
        """
        Get script path, preferring contract-defined path if available.
        
        Args:
            default_path: Default script path to use if not found in contract
            
        Returns:
            Script path
        """
        # Try to get from contract
        contract = self.get_script_contract()
        if contract and hasattr(contract, 'script_path'):
            return contract.script_path
            
        # Fall back to default or hardcoded path
        if hasattr(self, 'script_path'):
            return self.script_path
            
        return default_path

    @classmethod
    def get_step_name(cls, config_class_name: str) -> str:
        """Get the step name for a configuration class."""
        step_names = cls._get_step_registry()
        return step_names.get(config_class_name, config_class_name)

    @classmethod
    def get_config_class_name(cls, step_name: str) -> str:
        """Get the configuration class name from a step name."""
        step_names = cls._get_step_registry()
        reverse_mapping = {v: k for k, v in step_names.items()}
        return reverse_mapping.get(step_name, step_name)
    
    @classmethod
    def _get_step_registry(cls) -> Dict[str, str]:
        """Lazy load step registry to avoid circular imports."""
        if not cls._STEP_NAMES:
            try:
                from ...steps.registry.step_names import CONFIG_STEP_REGISTRY
                cls._STEP_NAMES = CONFIG_STEP_REGISTRY
            except ImportError:
                logger.warning("Could not import step registry, using empty registry")
                cls._STEP_NAMES = {}
        return cls._STEP_NAMES
    
    @classmethod
    def from_base_config(cls, base_config: 'BasePipelineConfig', **kwargs) -> 'BasePipelineConfig':
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
        categories = {
            'essential': [],  # Tier 1: Required, public
            'system': [],     # Tier 2: Optional (has default), public
            'derived': []     # Tier 3: Public properties
        }
        
        # Get model fields from the class (not instance) to avoid deprecation warnings
        model_fields = self.__class__.model_fields
        
        # Categorize public fields into essential (required) or system (with defaults)
        for field_name, field_info in model_fields.items():
            # Skip private fields
            if field_name.startswith('_'):
                continue
                
            # Use is_required() to determine if a field is essential
            if field_info.is_required():
                categories['essential'].append(field_name)
            else:
                categories['system'].append(field_name)
        
        # Find derived properties (public properties that aren't in model_fields)
        for attr_name in dir(self):
            if (not attr_name.startswith('_') and 
                attr_name not in model_fields and
                isinstance(getattr(type(self), attr_name, None), property)):
                categories['derived'].append(attr_name)
        
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
        for field_name in sorted(categories['essential']):
            print(f"{field_name.title()}: {getattr(self, field_name)}")
        
        # Print Tier 2 fields (system inputs with defaults)
        print("\n----- System Inputs with Defaults (Tier 2) -----")
        for field_name in sorted(categories['system']):
            value = getattr(self, field_name)
            if value is not None:  # Skip None values for cleaner output
                print(f"{field_name.title()}: {value}")
        
        # Print Tier 3 fields (derived properties)
        print("\n----- Derived Fields (Tier 3) -----")
        for field_name in sorted(categories['derived']):
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
        for field_name in categories['essential']:
            init_fields[field_name] = getattr(self, field_name)
        
        # Add all system fields (Tier 2) that aren't None
        for field_name in categories['system']:
            value = getattr(self, field_name)
            if value is not None:  # Only include non-None values
                init_fields[field_name] = value
        
        return init_fields
