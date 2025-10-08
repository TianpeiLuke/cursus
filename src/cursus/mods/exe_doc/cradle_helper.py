"""
Cradle Data Loading Helper for execution document generation.

This module provides the CradleDataLoadingHelper class that extracts execution
document configurations from Cradle data loading step configurations.
"""

import logging
from typing import Dict, List, Any, Optional

from .base import ExecutionDocumentHelper, ExecutionDocumentGenerationError

# Import CradleDataLoadingConfig directly for proper type checking
try:
    from ...steps.configs.config_cradle_data_loading_step import CradleDataLoadingConfig
    CRADLE_CONFIG_AVAILABLE = True
except ImportError:
    CRADLE_CONFIG_AVAILABLE = False

# Import Cradle models for request building
try:
    from com.amazon.secureaisandboxproxyservice.models.field import Field
    from com.amazon.secureaisandboxproxyservice.models.datasource import DataSource
    from com.amazon.secureaisandboxproxyservice.models.mdsdatasourceproperties import (
        MdsDataSourceProperties,
    )
    from com.amazon.secureaisandboxproxyservice.models.edxdatasourceproperties import (
        EdxDataSourceProperties,
    )
    from com.amazon.secureaisandboxproxyservice.models.andesdatasourceproperties import (
        AndesDataSourceProperties,
    )
    from com.amazon.secureaisandboxproxyservice.models.datasourcesspecification import (
        DataSourcesSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.jobsplitoptions import (
        JobSplitOptions,
    )
    from com.amazon.secureaisandboxproxyservice.models.transformspecification import (
        TransformSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.outputspecification import (
        OutputSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.cradlejobspecification import (
        CradleJobSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.createcradledataloadjobrequest import (
        CreateCradleDataLoadJobRequest,
    )

    CRADLE_MODELS_AVAILABLE = True
except ImportError:
    CRADLE_MODELS_AVAILABLE = False

# Import coral utils for request conversion
try:
    from secure_ai_sandbox_python_lib.utils import coral_utils

    CORAL_UTILS_AVAILABLE = True
except ImportError:
    CORAL_UTILS_AVAILABLE = False


logger = logging.getLogger(__name__)


class CradleDataLoadingHelper(ExecutionDocumentHelper):
    """
    Helper for extracting execution document configurations from Cradle data loading steps.
    
    This helper ports the logic from CradleDataLoadingStepBuilder._build_request() and
    get_request_dict() methods to generate execution document configurations.
    """
    
    def __init__(self):
        """Initialize the Cradle helper."""
        self.logger = logging.getLogger(__name__)
        
        if not CRADLE_MODELS_AVAILABLE:
            self.logger.warning(
                "Cradle models not available. _build_request will not work."
            )
        
        if not CORAL_UTILS_AVAILABLE:
            self.logger.warning(
                "coral_utils not available. get_request_dict will not work."
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
        # First try isinstance check if CradleDataLoadingConfig is available
        if CRADLE_CONFIG_AVAILABLE:
            try:
                if isinstance(config, CradleDataLoadingConfig):
                    return True
            except Exception:
                # If isinstance fails, continue to string matching
                pass
        
        # Fallback to string matching based on class name
        config_type_name = type(config).__name__.lower()
        return ("cradle" in config_type_name and 
                "data" in config_type_name and 
                "load" in config_type_name)
    
    def get_execution_step_name(self, step_name: str, config) -> str:
        """
        Get execution document step name following step builder naming convention.
        
        Transforms step names from DAG format to execution document format:
        - "CradleDataLoading_training" -> "CradleDataLoading-Training"
        - "CradleDataLoading_calibration" -> "CradleDataLoading-Calibration"
        
        This follows the same logic as CradleDataLoadingStepBuilder._get_step_name():
        1. Extract base name by removing job_type suffix
        2. Add hyphen separator and capitalize job_type
        
        Args:
            step_name: Original step name from DAG (e.g., "CradleDataLoading_training")
            config: Configuration object containing job_type
            
        Returns:
            Execution document step name (e.g., "CradleDataLoading-Training")
        """
        # Check if config has job_type attribute
        if hasattr(config, 'job_type') and config.job_type:
            job_type = config.job_type.lower()
            
            # Remove job_type suffix from step_name if present
            suffix_to_remove = f"_{job_type}"
            if step_name.endswith(suffix_to_remove):
                base_name = step_name[:-len(suffix_to_remove)]
            else:
                base_name = step_name
            
            # Apply step builder transformation: base_name + "-" + capitalized_job_type
            return f"{base_name}-{config.job_type.capitalize()}"
        
        # If no job_type, return step_name as-is
        return step_name
    
    def extract_step_config(self, step_name: str, config) -> Dict[str, Any]:
        """
        Extract execution document configuration from Cradle data loading step config.
        
        Args:
            step_name: Name of the step
            config: Cradle data loading configuration object
            
        Returns:
            Dictionary containing the execution document configuration
            
        Raises:
            ExecutionDocumentGenerationError: If configuration extraction fails
        """
        try:
            self.logger.info(f"Extracting Cradle execution document config for step: {step_name}")
            
            # Build the Cradle request
            request = self._build_request(config)
            
            # Convert to dictionary format for execution document
            request_dict = self._get_request_dict(request)
            
            self.logger.info(f"Successfully extracted Cradle config for step: {step_name}")
            return request_dict
            
        except Exception as e:
            self.logger.error(f"Failed to extract Cradle config for step {step_name}: {e}")
            raise ExecutionDocumentGenerationError(
                f"Cradle configuration extraction failed for step {step_name}: {e}"
            ) from e
    
    def _build_request(self, config) -> Any:
        """
        Convert config to a CreateCradleDataLoadJobRequest instance.
        
        This method is ported from CradleDataLoadingStepBuilder._build_request().
        
        Args:
            config: Cradle data loading configuration object
            
        Returns:
            CreateCradleDataLoadJobRequest: The request object for Cradle data loading
            
        Raises:
            ImportError: If the required Cradle models are not available
            ValueError: If the configuration is invalid
        """
        if not CRADLE_MODELS_AVAILABLE:
            raise ImportError("Cradle models not available. Cannot build request.")
        
        # Check if we have the necessary configuration attributes
        required_attrs = [
            "data_sources_spec",
            "transform_spec", 
            "output_spec",
            "cradle_job_spec",
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr) or getattr(config, attr) is None:
                raise ValueError(
                    f"CradleDataLoadingConfig missing required attribute: {attr}"
                )
        
        try:
            # (a) Build each DataSource from data_sources_spec.data_sources
            data_source_models: List[DataSource] = []
            for ds_cfg in config.data_sources_spec.data_sources:
                if ds_cfg.data_source_type == "MDS":
                    mds_props_cfg = ds_cfg.mds_data_source_properties
                    mds_props = MdsDataSourceProperties(
                        service_name=mds_props_cfg.service_name,
                        org_id=mds_props_cfg.org_id,
                        region=mds_props_cfg.region,
                        output_schema=[
                            Field(
                                field_name=f["field_name"], field_type=f["field_type"]
                            )
                            for f in mds_props_cfg.output_schema
                        ],
                        use_hourly_edx_data_set=mds_props_cfg.use_hourly_edx_data_set,
                    )
                    data_source_models.append(
                        DataSource(
                            data_source_name=ds_cfg.data_source_name,
                            data_source_type="MDS",
                            mds_data_source_properties=mds_props,
                            edx_data_source_properties=None,
                        )
                    )
                
                elif ds_cfg.data_source_type == "EDX":
                    edx_props_cfg = ds_cfg.edx_data_source_properties
                    edx_props = EdxDataSourceProperties(
                        edx_arn=edx_props_cfg.edx_manifest,
                        schema_overrides=[
                            Field(
                                field_name=f["field_name"], field_type=f["field_type"]
                            )
                            for f in edx_props_cfg.schema_overrides
                        ],
                    )
                    data_source_models.append(
                        DataSource(
                            data_source_name=ds_cfg.data_source_name,
                            data_source_type="EDX",
                            mds_data_source_properties=None,
                            edx_data_source_properties=edx_props,
                        )
                    )
                elif ds_cfg.data_source_type == "ANDES":
                    andes_props_cfg = ds_cfg.andes_data_source_properties
                    if andes_props_cfg.andes3_enabled:
                        self.logger.info(
                            "ANDES 3.0 is enabled for table %s",
                            andes_props_cfg.table_name,
                        )
                    andes_props = AndesDataSourceProperties(
                        provider=andes_props_cfg.provider,
                        table_name=andes_props_cfg.table_name,
                        andes3_enabled=andes_props_cfg.andes3_enabled,
                    )
                    data_source_models.append(
                        DataSource(
                            data_source_name=ds_cfg.data_source_name,
                            data_source_type="ANDES",
                            mds_data_source_properties=None,
                            edx_data_source_properties=None,
                            andes_data_source_properties=andes_props,
                        )
                    )
            
            # (b) DataSourcesSpecification
            ds_spec_cfg = config.data_sources_spec
            data_sources_spec = DataSourcesSpecification(
                start_date=ds_spec_cfg.start_date,
                end_date=ds_spec_cfg.end_date,
                data_sources=data_source_models,
            )
            
            # (c) TransformSpecification
            transform_spec_cfg = config.transform_spec
            jso = transform_spec_cfg.job_split_options
            split_opts = JobSplitOptions(
                split_job=jso.split_job,
                days_per_split=jso.days_per_split,
                merge_sql=jso.merge_sql or "",
            )
            transform_spec = TransformSpecification(
                transform_sql=transform_spec_cfg.transform_sql,
                job_split_options=split_opts,
            )
            
            # (d) OutputSpecification
            output_spec_cfg = config.output_spec
            output_spec = OutputSpecification(
                output_schema=output_spec_cfg.output_schema,
                output_path=output_spec_cfg.output_path,
                output_format=output_spec_cfg.output_format,
                output_save_mode=output_spec_cfg.output_save_mode,
                output_file_count=output_spec_cfg.output_file_count,
                keep_dot_in_output_schema=output_spec_cfg.keep_dot_in_output_schema,
                include_header_in_s3_output=output_spec_cfg.include_header_in_s3_output,
            )
            
            # (e) CradleJobSpecification
            cradle_job_spec_cfg = config.cradle_job_spec
            cradle_job_spec = CradleJobSpecification(
                cluster_type=cradle_job_spec_cfg.cluster_type,
                cradle_account=cradle_job_spec_cfg.cradle_account,
                extra_spark_job_arguments=cradle_job_spec_cfg.extra_spark_job_arguments
                or "",
                job_retry_count=cradle_job_spec_cfg.job_retry_count,
            )
            
            # (f) Build the final CreateCradleDataLoadJobRequest
            request = CreateCradleDataLoadJobRequest(
                data_sources=data_sources_spec,
                transform_specification=transform_spec,
                output_specification=output_spec,
                cradle_job_specification=cradle_job_spec,
            )
            
            return request
            
        except Exception as e:
            self.logger.error("Error building Cradle request: %s", e)
            raise ValueError(f"Failed to build Cradle request: {e}") from e
    
    def _get_request_dict(self, request) -> Dict[str, Any]:
        """
        Convert the CradleDataLoad request to a plain Python dict.
        
        This method is ported from CradleDataLoadingStepBuilder.get_request_dict().
        
        Args:
            request: CreateCradleDataLoadJobRequest object
            
        Returns:
            Dict[str, Any]: The request as a dictionary
            
        Raises:
            ImportError: If coral_utils is not available
            ValueError: If the request could not be converted
        """
        if not CORAL_UTILS_AVAILABLE:
            raise ImportError(
                "coral_utils not available. Cannot convert request to dict."
            )
        
        try:
            return coral_utils.convert_coral_to_dict(request)
        except Exception as e:
            self.logger.error("Error converting request to dict: %s", e)
            raise ValueError(f"Failed to convert request to dict: {e}") from e
