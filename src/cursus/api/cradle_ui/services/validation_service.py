"""
Validation service for Cradle Data Load Config UI

This module provides server-side validation for configuration data.
"""

from typing import Dict, List, Any, Optional
import logging
from pydantic import ValidationError

from ....steps.configs.config_cradle_data_loading_step import (
    CradleDataLoadConfig,
    DataSourcesSpecificationConfig,
    DataSourceConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    AndesDataSourceConfig,
    TransformSpecificationConfig,
    JobSplitOptionsConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig
)

logger = logging.getLogger(__name__)


class ValidationService:
    """Server-side validation service for CradleDataLoadConfig"""
    
    def validate_step_data(self, step: int, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate data for a specific step.
        
        Args:
            step: Step number (1-4)
            data: Step data to validate
            
        Returns:
            Dict mapping field names to lists of error messages
        """
        errors = {}
        
        try:
            if step == 1:
                errors.update(self._validate_data_sources_spec(data))
            elif step == 2:
                errors.update(self._validate_transform_spec(data))
            elif step == 3:
                errors.update(self._validate_output_spec(data))
            elif step == 4:
                errors.update(self._validate_cradle_job_spec(data))
            else:
                errors['step'] = [f'Invalid step number: {step}']
        except Exception as e:
            logger.error(f"Error validating step {step}: {str(e)}")
            errors['general'] = [f"Validation error: {str(e)}"]
        
        return errors
    
    def _validate_data_sources_spec(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate DataSourcesSpecificationConfig data.
        
        Args:
            data: Data sources specification data
            
        Returns:
            Dict mapping field names to lists of error messages
        """
        errors = {}
        
        try:
            # Validate time range
            start_date = data.get('startDate', '')
            end_date = data.get('endDate', '')
            
            if not start_date:
                errors['startDate'] = ['Start date is required']
            elif not self._validate_datetime_format(start_date):
                errors['startDate'] = ['Start date must be in format YYYY-MM-DDTHH:MM:SS']
            
            if not end_date:
                errors['endDate'] = ['End date is required']
            elif not self._validate_datetime_format(end_date):
                errors['endDate'] = ['End date must be in format YYYY-MM-DDTHH:MM:SS']
            
            # Validate data sources
            data_sources = data.get('dataSources', [])
            if not data_sources:
                errors['dataSources'] = ['At least one data source is required']
            else:
                for i, ds in enumerate(data_sources):
                    ds_errors = self._validate_data_source(ds)
                    if ds_errors:
                        errors[f'dataSources[{i}]'] = ds_errors
                        
        except Exception as e:
            logger.error(f"Error validating data sources spec: {str(e)}")
            errors['general'] = [f"Data sources validation error: {str(e)}"]
        
        return errors
    
    def _validate_data_source(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate individual DataSourceConfig.
        
        Args:
            data: Data source configuration data
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Validate basic fields
        if not data.get('dataSourceName'):
            errors.append('Data source name is required')
        
        data_source_type = data.get('dataSourceType')
        if not data_source_type:
            errors.append('Data source type is required')
            return errors
        
        if data_source_type not in ['MDS', 'EDX', 'ANDES']:
            errors.append(f'Invalid data source type: {data_source_type}')
            return errors
        
        # Validate type-specific properties
        try:
            if data_source_type == 'MDS':
                mds_props = data.get('mdsProperties', {})
                if not mds_props:
                    errors.append('MDS properties are required for MDS data source')
                else:
                    mds_config = MdsDataSourceConfig(**mds_props)
            elif data_source_type == 'EDX':
                edx_props = data.get('edxProperties', {})
                if not edx_props:
                    errors.append('EDX properties are required for EDX data source')
                else:
                    edx_config = EdxDataSourceConfig(**edx_props)
            elif data_source_type == 'ANDES':
                andes_props = data.get('andesProperties', {})
                if not andes_props:
                    errors.append('ANDES properties are required for ANDES data source')
                else:
                    andes_config = AndesDataSourceConfig(**andes_props)
                    
        except ValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append(f"{field}: {error['msg']}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def _validate_transform_spec(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate TransformSpecificationConfig data.
        
        Args:
            data: Transform specification data
            
        Returns:
            Dict mapping field names to lists of error messages
        """
        errors = {}
        
        try:
            # Validate transform SQL
            transform_sql = data.get('transformSql', '')
            if not transform_sql or not transform_sql.strip():
                errors['transformSql'] = ['Transform SQL is required']
            
            # Validate job split options
            job_split_options = data.get('jobSplitOptions', {})
            split_job = job_split_options.get('splitJob', False)
            
            if split_job:
                merge_sql = job_split_options.get('mergeSql', '')
                if not merge_sql or not merge_sql.strip():
                    errors['jobSplitOptions.mergeSql'] = ['Merge SQL is required when job splitting is enabled']
                
                days_per_split = job_split_options.get('daysPerSplit', 7)
                if not isinstance(days_per_split, int) or days_per_split < 1:
                    errors['jobSplitOptions.daysPerSplit'] = ['Days per split must be a positive integer']
            
        except Exception as e:
            logger.error(f"Error validating transform spec: {str(e)}")
            errors['general'] = [f"Transform validation error: {str(e)}"]
        
        return errors
    
    def _validate_output_spec(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate OutputSpecificationConfig data.
        
        Args:
            data: Output specification data
            
        Returns:
            Dict mapping field names to lists of error messages
        """
        errors = {}
        
        try:
            # Validate output schema
            output_schema = data.get('outputSchema', [])
            if not output_schema:
                errors['outputSchema'] = ['At least one output field is required']
            elif not isinstance(output_schema, list):
                errors['outputSchema'] = ['Output schema must be a list of field names']
            
            # Validate output format
            output_format = data.get('outputFormat', 'PARQUET')
            valid_formats = {'CSV', 'UNESCAPED_TSV', 'JSON', 'ION', 'PARQUET'}
            if output_format not in valid_formats:
                errors['outputFormat'] = [f'Output format must be one of: {", ".join(valid_formats)}']
            
            # Validate save mode
            output_save_mode = data.get('outputSaveMode', 'ERRORIFEXISTS')
            valid_save_modes = {'ERRORIFEXISTS', 'OVERWRITE', 'APPEND', 'IGNORE'}
            if output_save_mode not in valid_save_modes:
                errors['outputSaveMode'] = [f'Output save mode must be one of: {", ".join(valid_save_modes)}']
            
            # Validate file count
            output_file_count = data.get('outputFileCount', 0)
            if not isinstance(output_file_count, int) or output_file_count < 0:
                errors['outputFileCount'] = ['Output file count must be a non-negative integer']
            
        except Exception as e:
            logger.error(f"Error validating output spec: {str(e)}")
            errors['general'] = [f"Output validation error: {str(e)}"]
        
        return errors
    
    def _validate_cradle_job_spec(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate CradleJobSpecificationConfig data.
        
        Args:
            data: Cradle job specification data
            
        Returns:
            Dict mapping field names to lists of error messages
        """
        errors = {}
        
        try:
            # Validate cradle account
            cradle_account = data.get('cradleAccount', '')
            if not cradle_account or not cradle_account.strip():
                errors['cradleAccount'] = ['Cradle account is required']
            
            # Validate cluster type
            cluster_type = data.get('clusterType', 'STANDARD')
            valid_cluster_types = {'STANDARD', 'SMALL', 'MEDIUM', 'LARGE'}
            if cluster_type not in valid_cluster_types:
                errors['clusterType'] = [f'Cluster type must be one of: {", ".join(valid_cluster_types)}']
            
            # Validate retry count
            job_retry_count = data.get('jobRetryCount', 1)
            if not isinstance(job_retry_count, int) or job_retry_count < 0:
                errors['jobRetryCount'] = ['Job retry count must be a non-negative integer']
            
        except Exception as e:
            logger.error(f"Error validating cradle job spec: {str(e)}")
            errors['general'] = [f"Cradle job validation error: {str(e)}"]
        
        return errors
    
    def _validate_datetime_format(self, datetime_str: str) -> bool:
        """
        Validate datetime string format (YYYY-MM-DDTHH:MM:SS).
        
        Args:
            datetime_str: Datetime string to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            from datetime import datetime
            parsed = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
            return parsed.strftime("%Y-%m-%dT%H:%M:%S") == datetime_str
        except (ValueError, TypeError):
            return False
    
    def build_final_config(self, ui_data: Dict[str, Any]) -> CradleDataLoadConfig:
        """
        Build final CradleDataLoadConfig from UI data.
        
        Args:
            ui_data: Complete UI data from all steps
            
        Returns:
            CradleDataLoadConfig: Built configuration object
            
        Raises:
            ValidationError: If configuration data is invalid
            ValueError: If required data is missing
        """
        try:
            # Build data sources
            data_sources = []
            for ds_data in ui_data['dataSourcesSpec']['dataSources']:
                ds_type = ds_data['dataSourceType']
                
                if ds_type == 'MDS':
                    mds_props = MdsDataSourceConfig(**ds_data['mdsProperties'])
                    data_source = DataSourceConfig(
                        data_source_name=ds_data['dataSourceName'],
                        data_source_type=ds_type,
                        mds_data_source_properties=mds_props
                    )
                elif ds_type == 'EDX':
                    edx_props = EdxDataSourceConfig(**ds_data['edxProperties'])
                    data_source = DataSourceConfig(
                        data_source_name=ds_data['dataSourceName'],
                        data_source_type=ds_type,
                        edx_data_source_properties=edx_props
                    )
                elif ds_type == 'ANDES':
                    andes_props = AndesDataSourceConfig(**ds_data['andesProperties'])
                    data_source = DataSourceConfig(
                        data_source_name=ds_data['dataSourceName'],
                        data_source_type=ds_type,
                        andes_data_source_properties=andes_props
                    )
                else:
                    raise ValueError(f"Invalid data source type: {ds_type}")
                
                data_sources.append(data_source)
            
            # Build specifications
            data_sources_spec = DataSourcesSpecificationConfig(
                start_date=ui_data['dataSourcesSpec']['startDate'],
                end_date=ui_data['dataSourcesSpec']['endDate'],
                data_sources=data_sources
            )
            
            job_split_options = JobSplitOptionsConfig(**ui_data['transformSpec']['jobSplitOptions'])
            transform_spec = TransformSpecificationConfig(
                transform_sql=ui_data['transformSpec']['transformSql'],
                job_split_options=job_split_options
            )
            
            output_spec = OutputSpecificationConfig(**ui_data['outputSpec'])
            cradle_job_spec = CradleJobSpecificationConfig(**ui_data['cradleJobSpec'])
            
            # Build final config
            config = CradleDataLoadConfig(
                job_type=ui_data['jobType'],
                data_sources_spec=data_sources_spec,
                transform_spec=transform_spec,
                output_spec=output_spec,
                cradle_job_spec=cradle_job_spec
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Error building final config: {str(e)}")
            raise ValueError(f"Failed to build configuration: {str(e)}")
