"""
Registration Helper for execution document generation.

This module provides the RegistrationHelper class that extracts execution
document configurations from registration step configurations.
"""

import logging
from typing import Dict, List, Any, Optional

from .base import ExecutionDocumentHelper, ExecutionDocumentGenerationError

# Import RegistrationConfig directly for proper type checking
try:
    from ...steps.configs.config_registration_step import RegistrationConfig
    REGISTRATION_CONFIG_AVAILABLE = True
except ImportError:
    REGISTRATION_CONFIG_AVAILABLE = False

# Import SageMaker utilities for image URI retrieval
try:
    from sagemaker.image_uris import retrieve as retrieve_image_uri
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegistrationHelper(ExecutionDocumentHelper):
    """
    Helper for extracting execution document configurations from registration steps.
    
    This helper ports the logic from DynamicPipelineTemplate._fill_registration_configurations()
    and _create_execution_doc_config() methods to generate execution document configurations.
    """
    
    def __init__(self):
        """Initialize the Registration helper."""
        self.logger = logging.getLogger(__name__)
        
        if not SAGEMAKER_AVAILABLE:
            self.logger.warning(
                "SageMaker not available. Image URI retrieval will not work."
            )
    
    def can_handle_step(self, step_name: str, config) -> bool:
        """
        Check if this helper can handle the given step configuration.
        
        Args:
            step_name: Name of the step
            config: Step configuration object
            
        Returns:
            True if this helper can handle the configuration, False otherwise
        """
        # Use direct isinstance check if RegistrationConfig is available
        if REGISTRATION_CONFIG_AVAILABLE:
            if isinstance(config, RegistrationConfig):
                return True
        
        # Fallback to string matching if import failed
        config_type_name = type(config).__name__.lower()
        
        # Check by config type name
        if "registration" in config_type_name and "payload" not in config_type_name:
            return True
            
        # Check by step name pattern
        if any(pattern in step_name.lower() for pattern in ["registration", "register"]):
            return True
            
        return False
    
    def get_execution_step_name(self, step_name: str, config) -> str:
        """
        Get execution document step name following step builder naming convention.
        
        Transforms step names from DAG format to execution document format:
        - "Registration" -> "Registration-NA" (adds region suffix)
        
        This follows the same logic as RegistrationStepBuilder.create_step():
        step_name = self._get_step_name() + "-" + self.config.region
        
        Args:
            step_name: Original step name from DAG (e.g., "Registration")
            config: Configuration object containing region
            
        Returns:
            Execution document step name (e.g., "Registration-NA")
        """
        # Check if config has region attribute
        if hasattr(config, 'region') and config.region:
            # Apply step builder transformation: step_name + "-" + region
            return f"{step_name}-{config.region}"
        
        # If no region, return step_name as-is
        return step_name
    
    def extract_step_config(self, step_name: str, config, all_configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract execution document configuration from registration step config.
        
        Args:
            step_name: Name of the step
            config: Registration configuration object
            all_configs: Optional dictionary of all available configurations for related config lookup
            
        Returns:
            Dictionary containing the execution document configuration
            
        Raises:
            ExecutionDocumentGenerationError: If configuration extraction fails
        """
        try:
            self.logger.info(f"Extracting registration execution document config for step: {step_name}")
            
            # Get image URI for the registration configuration
            image_uri = self._get_image_uri(config)
            
            # Create execution document configuration
            # If all_configs provided, use the exact original logic, otherwise create a simple config dict
            if all_configs:
                exec_config = self._create_execution_doc_config(image_uri, all_configs)
            else:
                # Fallback: create a simple config dict with just the registration config
                configs_dict = {"registration": config}
                exec_config = self._create_execution_doc_config(image_uri, configs_dict)
            
            self.logger.info(f"Successfully extracted registration config for step: {step_name}")
            return exec_config
            
        except Exception as e:
            self.logger.error(f"Failed to extract registration config for step {step_name}: {e}")
            raise ExecutionDocumentGenerationError(
                f"Registration configuration extraction failed for step {step_name}: {e}"
            ) from e
    
    def _get_image_uri(self, config) -> str:
        """
        Get the SageMaker image URI for the registration configuration.
        
        Args:
            config: Registration configuration object
            
        Returns:
            SageMaker image URI string
            
        Raises:
            ImportError: If SageMaker is not available
            ValueError: If required configuration fields are missing
        """
        if not SAGEMAKER_AVAILABLE:
            self.logger.warning("SageMaker not available, using placeholder image URI")
            return "image-uri-placeholder"
        
        # Check if we have all required framework attributes
        required_attrs = [
            "framework",
            "aws_region", 
            "framework_version",
            "py_version",
            "inference_instance_type",
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            self.logger.warning(
                f"Registration config missing framework attributes: {missing_attrs}"
            )
            return "image-uri-placeholder"
        
        try:
            image_uri = retrieve_image_uri(
                framework=config.framework,
                region=config.aws_region,
                version=config.framework_version,
                py_version=config.py_version,
                instance_type=config.inference_instance_type,
                image_scope="inference",
            )
            self.logger.info(f"Retrieved image URI: {image_uri}")
            return image_uri
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve image URI: {e}")
            return "image-uri-placeholder"
    
    def _create_execution_doc_config(self, image_uri: str, configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the execution document configuration dictionary.
        
        This method is ported exactly from DynamicPipelineTemplate._create_execution_doc_config().
        
        Args:
            image_uri: The URI of the inference image to use
            configs: Dictionary of all available configurations
            
        Returns:
            Dictionary with execution document configuration
        """
        # Find needed configs using type name pattern matching (EXACT COPY from original)
        registration_cfg = None
        payload_cfg = None
        package_cfg = None

        for _, cfg in configs.items():
            cfg_type_name = type(cfg).__name__.lower()
            if "registration" in cfg_type_name and not "payload" in cfg_type_name:
                registration_cfg = cfg
            elif "payload" in cfg_type_name:
                payload_cfg = cfg
            elif "package" in cfg_type_name:
                package_cfg = cfg

        if not registration_cfg:
            self.logger.warning(
                "No registration configuration found for execution document"
            )
            return {}

        # Create a basic configuration with required fields (EXACT COPY from original)
        exec_config: Dict[str, Any] = {
            "source_model_inference_image_arn": image_uri,
        }

        # Add registration configuration fields (EXACT COPY from original)
        for field in [
            "model_domain",
            "model_objective",
            "source_model_inference_content_types",
            "source_model_inference_response_types",
            "source_model_inference_input_variable_list",
            "source_model_inference_output_variable_list",
            "model_registration_region",
            "source_model_region",
            "aws_region",
            "model_owner",
            "region",
        ]:
            if hasattr(registration_cfg, field):
                # Map certain fields to their execution doc names (EXACT COPY from original)
                if field == "aws_region":
                    exec_config["source_model_region"] = getattr(
                        registration_cfg, field
                    )
                elif field == "region":
                    exec_config["model_registration_region"] = getattr(
                        registration_cfg, field
                    )
                else:
                    exec_config[field] = getattr(registration_cfg, field)

        # Add environment variables if entry point is available (EXACT COPY from original)
        if hasattr(registration_cfg, "inference_entry_point"):
            exec_config["source_model_environment_variable_map"] = {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": registration_cfg.inference_entry_point,
                "SAGEMAKER_REGION": getattr(
                    registration_cfg, "aws_region", "us-east-1"
                ),
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            }

        # Add load testing info if payload and package configs are available (EXACT COPY from original)
        if payload_cfg and package_cfg:
            load_testing_info: Dict[str, Any] = {}

            # Add bucket if available (EXACT COPY from original)
            if hasattr(registration_cfg, "bucket"):
                load_testing_info["sample_payload_s3_bucket"] = registration_cfg.bucket

            # Add payload fields (EXACT COPY from original)
            for field in [
                "sample_payload_s3_key",
                "expected_tps",
                "max_latency_in_millisecond",
                "max_acceptable_error_rate",
            ]:
                if hasattr(payload_cfg, field):
                    load_testing_info[field] = getattr(payload_cfg, field)

            # Add instance type (EXACT COPY from original)
            if hasattr(package_cfg, "get_instance_type"):
                load_testing_info["instance_type_list"] = [
                    package_cfg.get_instance_type()
                ]
            elif hasattr(package_cfg, "processing_instance_type_small"):
                load_testing_info["instance_type_list"] = [
                    package_cfg.processing_instance_type_small
                ]

            if load_testing_info:
                exec_config["load_testing_info_map"] = load_testing_info

        return exec_config
    
    def create_execution_doc_config_with_related_configs(
        self, 
        registration_config, 
        payload_config=None, 
        package_config=None
    ) -> Dict[str, Any]:
        """
        Create execution document configuration with related payload and package configs.
        
        This method extends the basic execution document configuration with load testing
        information from payload and package configurations.
        
        Args:
            registration_config: Registration configuration object
            payload_config: Optional payload configuration object
            package_config: Optional package configuration object
            
        Returns:
            Dictionary with complete execution document configuration
        """
        # Get basic execution config using the new signature
        image_uri = self._get_image_uri(registration_config)
        
        # Create configs dict for the new method signature
        configs_dict = {"registration": registration_config}
        if payload_config:
            configs_dict["payload"] = payload_config
        if package_config:
            configs_dict["package"] = package_config
            
        exec_config = self._create_execution_doc_config(image_uri, configs_dict)
        
        return exec_config
    
    def find_registration_step_patterns(self, step_names: List[str], region: str = "") -> List[str]:
        """
        Find step name patterns that likely correspond to registration steps.
        
        This method is ported from DynamicPipelineTemplate._fill_registration_configurations().
        
        Args:
            step_names: List of step names to search through
            region: Optional region string to create region-specific patterns
            
        Returns:
            List of step name patterns for registration steps
        """
        search_patterns = []
        
        # Generate search patterns based on region
        if region:
            search_patterns.extend([
                f"ModelRegistration-{region}",  # Format from error logs
                f"Registration_{region}",       # Format from template code
            ])
        
        # Add generic patterns
        search_patterns.extend([
            "model_registration",  # Common generic name
            "Registration",        # Very generic fallback
            "register_model",      # Another common name
        ])
        
        # Search for any step name containing 'registration' as final fallback
        for step_name in step_names:
            if "registration" in step_name.lower():
                if step_name not in search_patterns:
                    search_patterns.append(step_name)
        
        # Filter patterns to only include those that exist in step_names
        existing_patterns = []
        for pattern in search_patterns:
            if pattern in step_names:
                existing_patterns.append(pattern)
        
        return existing_patterns
    
    def validate_registration_config(self, config) -> bool:
        """
        Validate that the registration config has all required fields.
        
        Args:
            config: Registration configuration object
            
        Returns:
            True if all required fields are present, False otherwise
        """
        # Check minimal required fields on registration config
        required_fields = ["model_domain", "model_objective", "region"]
        
        for field in required_fields:
            if not hasattr(config, field):
                self.logger.warning(
                    f"Registration config missing required field: {field}"
                )
                return False
        
        return True
