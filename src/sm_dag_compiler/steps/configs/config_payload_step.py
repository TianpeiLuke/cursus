from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING, ClassVar
from pathlib import Path
from datetime import datetime
from enum import Enum

import json
import boto3
import tarfile
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

from .config_processing_step_base import ProcessingStepConfigBase
from .config_registration_step import VariableType

# Import the script contract
from ..contracts.mims_payload_contract import MIMS_PAYLOAD_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class PayloadConfig(ProcessingStepConfigBase):
    """
    Configuration for payload generation and testing.
    
    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    # Model registration fields
    model_owner: str = Field(
        description="Team ID of model owner"
    )
    
    model_registration_domain: str = Field(
        description="Domain for model registration"
    )
    
    model_registration_objective: str = Field(
        description="Objective of model registration"
    )
    
    # Variable lists for input and output
    source_model_inference_output_variable_list: Dict[str, str] = Field(
        description="Dictionary of output variables and their types (NUMERIC or TEXT)"
    )
    
    source_model_inference_input_variable_list: Union[Dict[str, str], List[List[str]]] = Field(
        description="Input variables and their types. Can be either:\n"
                   "1. Dictionary: {'var1': 'NUMERIC', 'var2': 'TEXT'}\n"
                   "2. List of pairs: [['var1', 'NUMERIC'], ['var2', 'TEXT']]"
    )
    
    # Performance metrics
    expected_tps: int = Field(
        ge=1,
        description="Expected transactions per second"
    )
    
    max_latency_in_millisecond: int = Field(
        ge=100,
        le=10000,
        description="Maximum acceptable latency in milliseconds"
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override
    
    # Model framework settings
    framework: str = Field(
        default="xgboost",
        description="ML framework used for the model"
    )
    
    # Entry point script
    processing_entry_point: str = Field(
        default="mims_payload.py",
        description="Entry point script for payload generation"
    )
    
    # Content and response types
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']"
    )
    
    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Response type for model inference output. Must be exactly ['text/csv'] or ['application/json']"
    )
    
    # Performance thresholds
    max_acceptable_error_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable error rate (0-1)"
    )
    
    # Default values for payload generation
    default_numeric_value: float = Field(
        default=0.0,
        description="Default value for numeric fields"
    )
    
    default_text_value: str = Field(
        default="DEFAULT_TEXT",
        description="Default value for text fields"
    )
    
    # Special field values dictionary
    special_field_values: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary of special TEXT fields and their template values"
    )
    
    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    
    # S3 path configuration using PrivateAttr (not a model field)
    _sample_payload_s3_key = PrivateAttr(default=None)
    
    # Valid types for validation
    _VALID_TYPES: ClassVar[List[str]] = ["NUMERIC", "TEXT"]
    
    # Update to Pydantic V2 style model_config
    model_config = {
        'arbitrary_types_allowed': True,
        'validate_assignment': False,  # Changed from True to False to prevent recursion
        'extra': 'allow',  # Changed from 'forbid' to 'allow' to accept metadata fields during deserialization
        'json_encoders': {
            Path: str
        }
    }
    
    # Property for read-only access to sample_payload_s3_key
    @property
    def sample_payload_s3_key(self) -> Optional[str]:
        """Get the S3 key for sample payload file (read-only)"""
        return self._sample_payload_s3_key
    
    # Validators for inputs
    
    @field_validator('source_model_inference_input_variable_list')
    @classmethod
    def validate_input_variable_list(
        cls, 
        v: Union[Dict[str, str], List[List[str]]]
    ) -> Union[Dict[str, str], List[List[str]]]:
        """
        Validate input variable list format with string types.
        
        Args:
            v: Either a dictionary of variable names to types,
               or a list of [variable_name, variable_type] pairs
               
        Returns:
            Original value if valid, without modification
        """
        if not v:  # If empty
            raise ValueError("Input variable list cannot be empty")

        # Handle dictionary format
        if isinstance(v, dict):
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError(f"Key must be string, got {type(key)} for key: {key}")
                
                # Check if string value is valid
                if not isinstance(value, str):
                    raise ValueError(f"Value must be string, got {type(value)}")
                
                if value.upper() not in cls._VALID_TYPES:
                    raise ValueError(f"Value must be 'NUMERIC' or 'TEXT', got: {value}")
            return v

        # Handle list format
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("Each item must be a list of [variable_name, variable_type]")
                
                var_name, var_type = item
                if not isinstance(var_name, str):
                    raise ValueError(f"Variable name must be string, got {type(var_name)}")
                
                if not isinstance(var_type, str):
                    raise ValueError(f"Type must be string, got {type(var_type)}")
                
                if var_type.upper() not in cls._VALID_TYPES:
                    raise ValueError(f"Type must be 'NUMERIC' or 'TEXT', got: {var_type}")
            return v

        else:
            raise ValueError("Must be either a dictionary or a list of pairs")
            
    @field_validator('source_model_inference_output_variable_list')
    @classmethod
    def validate_output_variable_list(cls, v: Dict[str, str]) -> Dict[str, str]:
        """
        Validate output variable dictionary format with string types.
        
        Args:
            v: Dictionary mapping variable names to types
            
        Returns:
            Original dictionary if valid, without modification
        """
        if not v:  # If empty dictionary
            raise ValueError("Output variable list cannot be empty")
        
        for key, value in v.items():
            # Validate key is a string
            if not isinstance(key, str):
                raise ValueError(f"Key must be string, got {type(key)} for key: {key}")
        
            # Check if string value is valid
            if not isinstance(value, str):
                raise ValueError(f"Value must be string, got {type(value)}")
            
            if value.upper() not in cls._VALID_TYPES:
                raise ValueError(f"Value must be 'NUMERIC' or 'TEXT', got: {value}")
        
        return v
    
    # Model validators
        
    @model_validator(mode='after')
    def initialize_derived_fields(self) -> 'PayloadConfig':
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        
        # Don't immediately initialize _sample_payload_s3_key
        # It will be initialized on-demand by ensure_payload_path
        
        return self
        
    @model_validator(mode='after')
    def validate_special_fields(self) -> 'PayloadConfig':
        """Validate special fields configuration if provided"""
        if not self.special_field_values:
            return self
            
        # Check if all special fields are in input variable list
        invalid_fields = []
        input_vars = self.source_model_inference_input_variable_list
        
        for field_name in self.special_field_values:
            if isinstance(input_vars, dict):
                if field_name not in input_vars:
                    invalid_fields.append(field_name)
                else:
                    field_type = input_vars[field_name]
                    if field_type.upper() != "TEXT":
                        raise ValueError(
                            f"Special field '{field_name}' must be of type TEXT, "
                            f"got {field_type}"
                        )
            else:  # List format
                field_found = False
                for var_name, var_type in input_vars:
                    if var_name == field_name:
                        field_found = True
                        if var_type.upper() != "TEXT":
                            raise ValueError(
                                f"Special field '{field_name}' must be of type TEXT, "
                                f"got {var_type}"
                            )
                        break
                if not field_found:
                    invalid_fields.append(field_name)
        
        if invalid_fields:
            raise ValueError(
                f"Special fields not found in input variable list: {invalid_fields}"
            )
            
        # No model_copy - just return self directly
        return self
    
    # Methods for payload generation and paths
    
    def get_effective_source_dir(self) -> Optional[str]:
        """Get the effective source directory"""
        return self.processing_source_dir or self.source_dir
    
    def ensure_payload_path(self) -> None:
        """
        Ensure S3 key for payload is set. Only called when needed.
        
        This method generates the S3 key based on pipeline name, version,
        and registration objective, then stores it in a private field.
        """
        # Early exit if already set to avoid unnecessary work
        if self._sample_payload_s3_key:
            return

        # Generate path without using model validators
        payload_file_name = f'payload_{self.pipeline_name}_{self.pipeline_version}'
        if self.model_registration_objective:
            payload_file_name += f'_{self.model_registration_objective}'
        # Direct assignment to private field
        self._sample_payload_s3_key = f'mods/payload/{payload_file_name}.tar.gz'

    def get_full_payload_path(self) -> str:
        """Get full S3 path for payload"""
        # Ensure path is set before accessing
        if not self._sample_payload_s3_key:
            self.ensure_payload_path()
        return f"s3://{self.bucket}/{self._sample_payload_s3_key}"

    def get_field_default_value(self, field_name: str, var_type: str) -> str:
        """Get default value for a field"""
        var_type_upper = var_type.upper()
        if var_type_upper == "TEXT":
            if self.special_field_values and field_name in self.special_field_values:
                template = self.special_field_values[field_name]
                try:
                    return template.format(
                        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    )
                except KeyError as e:
                    raise ValueError(f"Invalid placeholder in template for field '{field_name}': {str(e)}")
            return self.default_text_value
        elif var_type_upper == "NUMERIC":
            return str(self.default_numeric_value)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")   
    
    def generate_csv_payload(self) -> str:
        """
        Generate CSV format payload following the order in source_model_inference_input_variable_list.
    
        Returns:
            Comma-separated string of values
        """
        values = []
        input_vars = self.source_model_inference_input_variable_list
        
        if isinstance(input_vars, dict):
            # Dictionary format
            for field_name, var_type in input_vars.items():
                values.append(self.get_field_default_value(field_name, var_type))
        else:
            # List format
            for field_name, var_type in input_vars:
                values.append(self.get_field_default_value(field_name, var_type))
                
        return ",".join(values)

    def generate_json_payload(self) -> str:
        """
        Generate JSON format payload using source_model_inference_input_variable_list.
    
        Returns:
            JSON string with field names and values
        """
        payload = {}
        input_vars = self.source_model_inference_input_variable_list
        
        if isinstance(input_vars, dict):
            # Dictionary format
            for field_name, var_type in input_vars.items():
                payload[field_name] = self.get_field_default_value(field_name, var_type)
        else:
            # List format
            for field_name, var_type in input_vars:
                payload[field_name] = self.get_field_default_value(field_name, var_type)
                
        return json.dumps(payload)
    
    def generate_sample_payloads(self) -> List[Dict[str, Union[str, dict]]]:
        """
        Generate sample payloads for each content type.
    
        Returns:
            List of dictionaries containing content type and payload
        """
        payloads = []
    
        for content_type in self.source_model_inference_content_types:
            payload_info = {
                "content_type": content_type,
                "payload": None
            }
        
            if content_type == "text/csv":
                payload_info["payload"] = self.generate_csv_payload()
            elif content_type == "application/json":
                payload_info["payload"] = self.generate_json_payload()
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            payloads.append(payload_info)
        
        return payloads

    def save_payloads(self, output_dir: Path) -> List[Path]:
        """
        Save payloads to files.
        
        Args:
            output_dir: Directory to save payload files
            
        Returns:
            List of paths to created payload files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        payloads = self.generate_sample_payloads()
        
        for i, payload_info in enumerate(payloads):
            content_type = payload_info["content_type"]
            payload = payload_info["payload"]
            
            # Determine file extension and name
            ext = ".csv" if content_type == "text/csv" else ".json"
            file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
            file_path = output_dir / file_name
            
            # Save payload
            with open(file_path, "w") as f:
                f.write(payload)
                
            file_paths.append(file_path)
            logger.info(f"Created payload file: {file_path}")
            
        return file_paths

    def upload_payloads_to_s3(self, payload_files: List[Path]) -> str:
        """
        Create tar.gz archive of payload files and upload to S3.
    
        Args:
            payload_files: List of payload file paths to upload
        
        Returns:
            S3 URI of uploaded archive
        
        Raises:
            ValueError: If no payload files provided or S3 upload fails
        """
        if not payload_files:
            raise ValueError("No payload files provided for upload")
        
        if not self.bucket:
            raise ValueError("Bucket not specified in configuration")
        
        # Ensure payload path is set
        if not self._sample_payload_s3_key:
            self.ensure_payload_path()
        
        try:
            # Create temporary directory for tar.gz creation
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "payload.tar.gz"
            
                # Create tar.gz archive
                with tarfile.open(archive_path, "w:gz") as tar:
                    for file_path in payload_files:
                        # Add file to archive with its basename as name
                        tar.add(file_path, arcname=file_path.name)
            
                # Use bucket and key from config
                bucket = self.bucket
                key = self._sample_payload_s3_key
                s3_uri = f"s3://{bucket}/{key}"
            
                logger.info(f"Uploading payloads archive to bucket: {bucket}")
                logger.info(f"Using S3 key: {key}")
            
                # Upload to S3
                s3_client = boto3.client('s3')
                s3_client.upload_file(
                    str(archive_path),
                    bucket,
                    key,
                    #ExtraArgs={'ServerSideEncryption': 'aws:kms'}
                )
                
                logger.info(f"Successfully uploaded payloads to: {s3_uri}")
                return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to upload payloads to S3: {str(e)}")
            raise

    def generate_and_upload_payloads(self) -> str:
        """
        Generate payloads, saveave them, and upload to S3.
        
        Returns:
            S3 URI of uploaded archive
            
        Raises:
            Exception: If any step fails
        """
        # Ensure S3 path is constructed
        if not self._sample_payload_s3_key:
            self.ensure_payload_path()
            logger.info(f"Constructed S3 key: {self._sample_payload_s3_key}")
            
        try:
            # Create temporary directory for payload files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save payloads to temporary directory
                logger.info("Generating and saving payload files...")
                payload_files = self.save_payloads(Path(temp_dir))
                
                # Upload to S3
                logger.info("Uploading payloads to S3...")
                s3_uri = self.upload_payloads_to_s3(payload_files)
                
                return s3_uri
                
        except Exception as e:
            logger.error(f"Failed to generate and upload payloads: {str(e)}")
            raise
            
    # Script and contract handling
            
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The MIMS payload script contract
        """
        return MIMS_PAYLOAD_CONTRACT

    def get_script_path(self) -> str:
        """
        Get script path with priority order:
        1. Use processing_entry_point if provided
        2. Fall back to script_contract.entry_point if available
        
        Always combines with effective source directory.
        
        Returns:
            Script path or None if no entry point can be determined
        """
        # Determine which entry point to use
        entry_point = None
        
        # First priority: Use processing_entry_point if provided
        if self.processing_entry_point:
            entry_point = self.processing_entry_point
        # Second priority: Use contract entry point
        else:
            contract = self.get_script_contract()
            if contract and hasattr(contract, 'entry_point'):
                entry_point = contract.entry_point
        
        if not entry_point:
            return None
        
        # Get the effective source directory
        effective_source_dir = self.get_effective_source_dir()
        if not effective_source_dir:
            return entry_point  # No source dir, just return entry point
        
        # Combine source dir with entry point
        if effective_source_dir.startswith('s3://'):
            full_path = f"{effective_source_dir.rstrip('/')}/{entry_point}"
        else:
            full_path = str(Path(effective_source_dir) / entry_point)
        
        return full_path

    # Input/output variable helpers
    
    def get_normalized_input_variables(self) -> List[List[str]]:
        """
        Get input variables normalized to list format with string types.
        Compatible with format from create_model_variable_list.
        
        Returns:
            List of [name, type] pairs with string types
        """
        input_vars = self.source_model_inference_input_variable_list
        result = []
        
        if isinstance(input_vars, dict):
            # Convert dict to list format
            for name, var_type in input_vars.items():
                type_str = str(var_type).upper()
                result.append([name, type_str])
        else:
            # Already list format, just standardize types
            for name, var_type in input_vars:
                type_str = str(var_type).upper()
                result.append([name, type_str])
        
        return result
    
    def get_input_variables_as_dict(self) -> Dict[str, str]:
        """
        Get input variables as a dictionary mapping names to string types.
        Compatible with the second return value from create_model_variable_json.
        
        Returns:
            Dictionary mapping variable names to string types
        """
        input_vars = self.source_model_inference_input_variable_list
        result = {}
        
        # Already in dict format
        if isinstance(input_vars, dict):
            for name, var_type in input_vars.items():
                result[name] = str(var_type).upper()
        
        # Convert list format to dict format
        else:
            for name, var_type in input_vars:
                result[name] = str(var_type).upper()
        
        return result

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization - simplified for string types"""
        data = super().model_dump(**kwargs)
        return data
