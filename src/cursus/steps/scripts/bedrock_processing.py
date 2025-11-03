"""
Bedrock Processing Script

Processes input data through AWS Bedrock models using generated prompt templates
and validation schemas from the Bedrock Prompt Template Generation step.
Supports template-driven response processing with dynamic Pydantic model creation.
"""

import os
import json
import argparse
import pandas as pd
import boto3
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
from pydantic import BaseModel, ValidationError, create_model, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import glob
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Container path constants
CONTAINER_PATHS = {
    "INPUT_DATA_DIR": "/opt/ml/processing/input/data",
    "INPUT_TEMPLATES_DIR": "/opt/ml/processing/input/templates",
    "INPUT_SCHEMA_DIR": "/opt/ml/processing/input/schema",
    "OUTPUT_DATA_DIR": "/opt/ml/processing/output/data",
    "OUTPUT_SUMMARY_DIR": "/opt/ml/processing/output/summary"
}


class BedrockProcessor:
    """
    Bedrock processor with template-driven response processing.
    Integrates with Bedrock Prompt Template Generation step outputs.
    Supports both sequential and concurrent processing modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock_client = None
        self.response_model_class = None
        self.effective_model_id = config['primary_model_id']
        self.inference_profile_info = {}
        self.validation_schema = config.get('validation_schema', {})
        
        # Thread-local storage for concurrent processing
        self.thread_local = threading.local()
        
        # Rate limiting for concurrent requests
        self.max_concurrent_workers = config.get('max_concurrent_workers', 5)
        self.rate_limit_per_second = config.get('rate_limit_per_second', 10)
        self.concurrency_mode = config.get('concurrency_mode', 'sequential')  # sequential, concurrent
        
        # Rate limiting state
        self.request_semaphore = threading.Semaphore(self.max_concurrent_workers)
        self.last_request_times = {}
        self.time_lock = threading.Lock()
        
        self._initialize_bedrock_client()
        self._configure_inference_profile()
        self._create_response_model_from_schema()
    
    def _initialize_bedrock_client(self):
        """Initialize Bedrock client."""
        region_name = self.config.get('region_name', 'us-east-1')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        logger.info(f"Initialized Bedrock client for region: {region_name}")
    
    def _get_thread_local_bedrock_client(self):
        """Get thread-local Bedrock client for concurrent processing."""
        if not hasattr(self.thread_local, 'bedrock_client'):
            region_name = self.config.get('region_name', 'us-east-1')
            self.thread_local.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        return self.thread_local.bedrock_client
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests for concurrent processing."""
        if self.concurrency_mode == 'sequential':
            return  # No rate limiting needed for sequential processing
        
        with self.time_lock:
            current_time = time.time()
            min_interval = 1.0 / self.rate_limit_per_second
            
            thread_id = threading.current_thread().ident
            if thread_id in self.last_request_times:
                elapsed = current_time - self.last_request_times[thread_id]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            self.last_request_times[thread_id] = time.time()
    
    def _configure_inference_profile(self):
        """Configure inference profile settings based on model and environment."""
        model_id = self.config['primary_model_id']
        inference_profile_arn = self.config.get('inference_profile_arn')
        
        # Check if model requires inference profile
        inference_profile_required = json.loads(
            self.config.get('inference_profile_required_models', '[]')
        )
        
        if inference_profile_arn:
            # Use provided ARN
            self.effective_model_id = inference_profile_arn
            self.inference_profile_info = {
                'arn': inference_profile_arn,
                'method': 'arn'
            }
            logger.info(f"Using inference profile ARN: {inference_profile_arn}")
            
        elif model_id in inference_profile_required:
            # Auto-configure for known models
            if model_id == "anthropic.claude-sonnet-4-20250514-v1:0":
                # Use global profile ID for Claude 4
                self.effective_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
                self.inference_profile_info = {
                    'profile_id': 'global.anthropic.claude-sonnet-4-20250514-v1:0',
                    'original_model_id': model_id,
                    'method': 'profile_id'
                }
                logger.info(f"Auto-configured to use inference profile ID: {self.effective_model_id}")
            
            elif 'claude-4' in model_id or 'claude-sonnet-4' in model_id:
                logger.warning(f"Model {model_id} may require an inference profile. Consider setting BEDROCK_INFERENCE_PROFILE_ARN.")
        
        # If model already starts with 'global.', it's already a profile ID
        if model_id.startswith('global.'):
            self.inference_profile_info = {
                'profile_id': model_id,
                'method': 'profile_id'
            }
            logger.info(f"Using provided inference profile ID: {model_id}")
    
    def _create_response_model_from_schema(self):
        """Create Pydantic response model from validation schema."""
        if not self.validation_schema:
            logger.warning("No validation schema provided, using basic JSON parsing")
            return
        
        try:
            # Extract schema properties
            properties = self.validation_schema.get('properties', {})
            required_fields = self.validation_schema.get('required', [])
            processing_config = self.validation_schema.get('processing_config', {})
            
            if not properties:
                logger.warning("No properties found in validation schema")
                return
            
            # Create Pydantic fields dynamically
            fields = {}
            for field_name, field_schema in properties.items():
                field_type = self._convert_json_schema_type_to_python(field_schema)
                description = field_schema.get('description', f"The {field_name} value")
                
                if field_name in required_fields:
                    fields[field_name] = (field_type, Field(..., description=description))
                else:
                    fields[field_name] = (Optional[field_type], Field(None, description=description))
            
            # Create dynamic Pydantic model
            model_name = processing_config.get('response_model_name', 'BedrockResponse')
            self.response_model_class = create_model(model_name, **fields)
            
            logger.info(f"Created dynamic Pydantic model '{model_name}' with fields: {list(fields.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to create Pydantic model from schema: {e}")
            self.response_model_class = None
    
    def _convert_json_schema_type_to_python(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON schema field definition to Python type."""
        field_type = field_schema.get('type', 'string')
        
        if field_type == 'string':
            if 'enum' in field_schema:
                # Create Literal type for enum fields
                from typing import Literal
                return Literal[tuple(field_schema['enum'])]
            return str
        elif field_type == 'number':
            return float
        elif field_type == 'integer':
            return int
        elif field_type == 'boolean':
            return bool
        elif field_type == 'array':
            return list
        else:
            return str  # Default fallback
    
    def _format_prompt(self, row_data: Dict[str, Any]) -> str:
        """Format prompt using template placeholders and DataFrame row data."""
        template_vars = {}
        
        # Use input_placeholders from template configuration (preferred method)
        placeholders = self.config.get('input_placeholders', [])
        
        # Fallback to regex extraction if input_placeholders not available
        if not placeholders:
            import re
            placeholders = re.findall(r'\{(\w+)\}', self.config['user_prompt_template'])
            logger.info("Using regex fallback for placeholder extraction")
        else:
            logger.info(f"Using input_placeholders from template: {placeholders}")
        
        # Map DataFrame columns to template placeholders
        for placeholder in placeholders:
            if placeholder in row_data:
                template_vars[placeholder] = row_data[placeholder]
            else:
                # Log warning for missing placeholder data
                logger.warning(f"Placeholder '{placeholder}' not found in row data. Available columns: {list(row_data.keys())}")
                template_vars[placeholder] = f"[Missing: {placeholder}]"
        
        return self.config['user_prompt_template'].format(**template_vars)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _invoke_bedrock(self, prompt: str) -> Dict[str, Any]:
        """Invoke Bedrock with intelligent fallback strategy and retry logic."""
        # Enforce rate limiting for concurrent processing
        if self.concurrency_mode == 'concurrent':
            self._enforce_rate_limit()
        
        # Use thread-local client for concurrent processing, main client for sequential
        if self.concurrency_mode == 'concurrent':
            client = self._get_thread_local_bedrock_client()
        else:
            client = self.bedrock_client
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config['max_tokens']),
            "temperature": float(self.config['temperature']),
            "top_p": float(self.config['top_p']),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.config.get('system_prompt'):
            request_body["system"] = self.config['system_prompt']
        
        # Try primary model/profile first
        try:
            response = client.invoke_model(
                modelId=self.effective_model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            return json.loads(response['body'].read())
            
        except Exception as e:
            # Fallback to on-demand model if inference profile fails
            fallback_model = self.config.get('fallback_model_id')
            if fallback_model and 'ValidationException' in str(e):
                logger.warning(f"Inference profile failed, falling back to: {fallback_model}")
                try:
                    response = client.invoke_model(
                        modelId=fallback_model,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json"
                    )
                    return json.loads(response['body'].read())
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e
    
    def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Bedrock response using Pydantic model validation."""
        if 'content' in response and len(response['content']) > 0:
            response_text = response['content'][0].get('text', '')
        else:
            raise ValueError("No content in Bedrock response")
        
        try:
            if self.response_model_class:
                # Use Pydantic model for structured parsing
                validated_response = self.response_model_class.model_validate_json(response_text)
                
                # Convert to dictionary
                result = validated_response.model_dump()
                
                # Add validation status
                result['parse_status'] = 'success'
                result['validation_passed'] = True
                
                return result
            else:
                # Fallback to JSON parsing
                parsed_json = json.loads(response_text)
                parsed_json['parse_status'] = 'json_only'
                parsed_json['validation_passed'] = False
                return parsed_json
                
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            return {
                'raw_response': response_text,
                'validation_error': str(e),
                'parse_status': 'validation_failed',
                'validation_passed': False
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                'raw_response': response_text,
                'json_error': str(e),
                'parse_status': 'json_failed',
                'validation_passed': False
            }
    
    def process_single_case(
        self,
        row_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single case through Bedrock using template placeholders.
        
        Args:
            row_data: Dictionary containing all row data from DataFrame
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Format prompt using template placeholders
            prompt = self._format_prompt(row_data)
            
            # Invoke Bedrock
            response = self._invoke_bedrock(prompt)
            
            # Parse response with Pydantic validation
            parsed_result = self._parse_response_with_pydantic(response)
            
            # Add processing metadata
            result = {
                **parsed_result,
                'processing_status': 'success',
                'error_message': None,
                'model_info': {
                    'effective_model_id': self.effective_model_id,
                    'inference_profile_info': self.inference_profile_info
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing case: {str(e)}")
            
            # Return structured error response
            error_result = {
                'processing_status': 'error',
                'error_message': str(e),
                'raw_response': None,
                'parse_status': 'error',
                'validation_passed': False,
                'model_info': {
                    'effective_model_id': self.effective_model_id,
                    'inference_profile_info': self.inference_profile_info
                }
            }
            
            # Add default values for expected fields if Pydantic model is available
            if self.response_model_class:
                try:
                    default_fields = self.response_model_class.model_fields.keys()
                    for field in default_fields:
                        if field not in error_result:
                            error_result[field] = None
                except Exception:
                    pass
            
            return error_result
    
    def process_single_case_with_rate_limiting(
        self,
        row_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single case with rate limiting for concurrent processing.
        
        Args:
            row_data: Dictionary containing all row data from DataFrame
            
        Returns:
            Dictionary with analysis results and metadata
        """
        if self.concurrency_mode == 'concurrent':
            with self.request_semaphore:  # Limit concurrent requests
                return self.process_single_case(row_data)
        else:
            return self.process_single_case(row_data)
    
    def process_batch_concurrent(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of data through Bedrock using concurrent processing.
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with analysis results
        """
        batch_size = batch_size or self.config.get('batch_size', 10)
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        output_prefix = self.config['output_column_prefix']
        
        # Extract placeholders from template to validate DataFrame columns
        import re
        placeholders = re.findall(r'\{(\w+)\}', self.config['user_prompt_template'])
        
        # Log template placeholders and available columns
        logger.info(f"Template placeholders: {placeholders}")
        logger.info(f"Available DataFrame columns: {list(df.columns)}")
        logger.info(f"Concurrent processing mode: {self.max_concurrent_workers} workers")
        
        # Check for missing placeholders
        missing_placeholders = [p for p in placeholders if p not in df.columns]
        if missing_placeholders:
            logger.warning(f"Missing DataFrame columns for placeholders: {missing_placeholders}")
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records) with {self.max_concurrent_workers} workers")
            
            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=self.max_concurrent_workers) as executor:
                # Submit all tasks
                future_to_row = {
                    executor.submit(self.process_single_case_with_rate_limiting, row.to_dict()): (idx, row)
                    for idx, row in batch_df.iterrows()
                }
                
                batch_results = []
                for future in as_completed(future_to_row):
                    idx, original_row = future_to_row[future]
                    try:
                        result = future.result()
                        
                        # Add original row data
                        result_row = original_row.to_dict()
                        
                        # Add Bedrock results with prefix
                        for key, value in result.items():
                            if key not in ['processing_status', 'error_message', 'model_info']:
                                result_row[f"{output_prefix}{key}"] = value
                        
                        # Add processing metadata
                        result_row[f"{output_prefix}status"] = result['processing_status']
                        if result.get('error_message'):
                            result_row[f"{output_prefix}error"] = result['error_message']
                        
                        batch_results.append(result_row)
                        
                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {e}")
                        # Add error result
                        error_row = original_row.to_dict()
                        error_row[f"{output_prefix}status"] = "error"
                        error_row[f"{output_prefix}error"] = str(e)
                        batch_results.append(error_row)
            
            results.extend(batch_results)
            
            # Save intermediate results
            if save_intermediate:
                intermediate_df = pd.DataFrame(batch_results)
                output_dir = Path(CONTAINER_PATHS["OUTPUT_DATA_DIR"])
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_file = output_dir / f"batch_{batch_num:04d}_results.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed concurrent processing {len(results_df)} records")
        
        return results_df
    
    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of data through Bedrock using template placeholders.
        Automatically chooses between sequential and concurrent processing based on configuration.
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with analysis results
        """
        if self.concurrency_mode == 'concurrent':
            return self.process_batch_concurrent(df, batch_size, save_intermediate)
        else:
            return self.process_batch_sequential(df, batch_size, save_intermediate)
    
    def process_batch_sequential(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Process a batch of data through Bedrock using sequential processing.
        
        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with analysis results
        """
        batch_size = batch_size or self.config.get('batch_size', 10)
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        output_prefix = self.config['output_column_prefix']
        
        # Extract placeholders from template to validate DataFrame columns
        import re
        placeholders = re.findall(r'\{(\w+)\}', self.config['user_prompt_template'])
        
        # Log template placeholders and available columns
        logger.info(f"Template placeholders: {placeholders}")
        logger.info(f"Available DataFrame columns: {list(df.columns)}")
        logger.info("Sequential processing mode")
        
        # Check for missing placeholders
        missing_placeholders = [p for p in placeholders if p not in df.columns]
        if missing_placeholders:
            logger.warning(f"Missing DataFrame columns for placeholders: {missing_placeholders}")
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records)")
            
            batch_results = []
            for idx, row in batch_df.iterrows():
                # Convert row to dictionary for template processing
                row_data = row.to_dict()
                
                # Process single case using template placeholders
                result = self.process_single_case(row_data)
                
                # Add original row data
                result_row = row_data.copy()
                
                # Add Bedrock results with prefix
                for key, value in result.items():
                    if key not in ['processing_status', 'error_message', 'model_info']:
                        result_row[f"{output_prefix}{key}"] = value
                
                # Add processing metadata
                result_row[f"{output_prefix}status"] = result['processing_status']
                if result.get('error_message'):
                    result_row[f"{output_prefix}error"] = result['error_message']
                
                batch_results.append(result_row)
            
            results.extend(batch_results)
            
            # Save intermediate results
            if save_intermediate:
                intermediate_df = pd.DataFrame(batch_results)
                output_dir = Path(CONTAINER_PATHS["OUTPUT_DATA_DIR"])
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_file = output_dir / f"batch_{batch_num:04d}_results.parquet"
                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")
        
        results_df = pd.DataFrame(results)
        logger.info(f"Completed sequential processing {len(results_df)} records")
        
        return results_df


def load_prompt_templates(templates_path: str, log: Callable[[str], None]) -> Dict[str, Any]:
    """
    Load prompt templates from Bedrock Prompt Template Generation step output.
    
    Expected file structure from Template Generation step:
    - prompts.json: JSON file containing system_prompt, user_prompt_template, and input_placeholders
    
    Args:
        templates_path: Path to templates directory from Template Generation step
        log: Logger function
        
    Returns:
        Dictionary with 'system_prompt', 'user_prompt_template', and 'input_placeholders' keys
    """
    templates = {}
    templates_dir = Path(templates_path)
    
    if not templates_dir.exists():
        raise ValueError(f"Templates directory not found: {templates_path}")
    
    # Load prompts.json (standard output from Template Generation step)
    prompts_file = templates_dir / "prompts.json"
    if prompts_file.exists():
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                json_templates = json.load(f)
            
            if 'system_prompt' in json_templates:
                templates['system_prompt'] = json_templates['system_prompt']
                log(f"Loaded system prompt from {prompts_file}")
            
            if 'user_prompt_template' in json_templates:
                templates['user_prompt_template'] = json_templates['user_prompt_template']
                log(f"Loaded user prompt template from {prompts_file}")
            
            if 'input_placeholders' in json_templates:
                templates['input_placeholders'] = json_templates['input_placeholders']
                log(f"Loaded input placeholders from {prompts_file}: {json_templates['input_placeholders']}")
            else:
                log("No input_placeholders found in template, will use regex fallback")
                
        except Exception as e:
            raise ValueError(f"Failed to load templates from {prompts_file}: {e}")
    else:
        raise ValueError(f"Required prompts.json not found in {templates_path}")
    
    return templates


def load_validation_schema(schema_path: str, log: Callable[[str], None]) -> Dict[str, Any]:
    """
    Load validation schema from Bedrock Prompt Template Generation step output.
    
    Expected file structure from Template Generation step:
    - validation_schema_*.json: Enhanced validation schema with processing metadata
    
    Args:
        schema_path: Path to schema directory from Template Generation step
        log: Logger function
        
    Returns:
        Dictionary containing the validation schema
    """
    schema_dir = Path(schema_path)
    
    if not schema_dir.exists():
        raise ValueError(f"Schema directory not found: {schema_path}")
    
    # Look for validation schema files
    schema_files = list(schema_dir.glob("validation_schema_*.json"))
    if not schema_files:
        raise ValueError(f"No validation schema files found in {schema_path}")
    
    # Use the most recent schema file
    schema_file = sorted(schema_files)[-1]
    
    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        log(f"Loaded validation schema from {schema_file}")
        
        # Validate schema structure
        required_sections = ['properties', 'required']
        for section in required_sections:
            if section not in schema:
                raise ValueError(f"Missing required section '{section}' in validation schema")
        
        return schema
        
    except Exception as e:
        raise ValueError(f"Failed to load validation schema from {schema_file}: {e}")


def process_split_directory(
    split_name: str,
    split_input_path: Path,
    split_output_path: Path,
    processor: BedrockProcessor,
    config: Dict[str, Any],
    log: Callable[[str], None]
) -> Dict[str, Any]:
    """
    Process a single split directory (train, val, or test).
    
    Args:
        split_name: Name of the split (train, val, test)
        split_input_path: Path to input split directory
        split_output_path: Path to output split directory
        processor: BedrockProcessor instance
        config: Processing configuration
        log: Logger function
        
    Returns:
        Dictionary with processing statistics for this split
    """
    # Create output directory for this split
    split_output_path.mkdir(parents=True, exist_ok=True)
    
    # Find input files in this split directory
    input_files = list(split_input_path.glob("*.csv")) + list(split_input_path.glob("*.parquet"))
    
    if not input_files:
        log(f"No input files found in {split_input_path}")
        return {
            'split_name': split_name,
            'total_files': 0,
            'total_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'validation_passed_records': 0,
            'files_processed': []
        }
    
    log(f"Processing {split_name} split with {len(input_files)} files")
    
    split_results = []
    split_stats = {
        'split_name': split_name,
        'total_files': len(input_files),
        'total_records': 0,
        'successful_records': 0,
        'failed_records': 0,
        'validation_passed_records': 0,
        'files_processed': []
    }
    
    for input_file in input_files:
        log(f"Processing {split_name} file: {input_file}")
        
        # Load data
        if input_file.suffix == '.csv':
            df = pd.read_csv(input_file)
        else:
            df = pd.read_parquet(input_file)
        
        # Process batch
        result_df = processor.process_batch(df, save_intermediate=False)  # No intermediate saves for splits
        
        # Update statistics
        split_stats['total_records'] += len(df)
        success_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"])
        failed_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "error"])
        validation_passed_count = len(result_df[result_df.get(f"{config['output_column_prefix']}validation_passed", False) == True])
        
        split_stats['successful_records'] += success_count
        split_stats['failed_records'] += failed_count
        split_stats['validation_passed_records'] += validation_passed_count
        split_stats['files_processed'].append({
            'filename': input_file.name,
            'records': len(df),
            'successful': success_count,
            'failed': failed_count,
            'validation_passed': validation_passed_count,
            'success_rate': success_count / len(df) if len(df) > 0 else 0,
            'validation_rate': validation_passed_count / len(df) if len(df) > 0 else 0
        })
        
        # Save results maintaining original filename structure
        base_filename = input_file.stem
        
        # Save as Parquet (efficient for large datasets)
        parquet_file = split_output_path / f"{base_filename}_processed_data.parquet"
        result_df.to_parquet(parquet_file, index=False)
        
        # Save as CSV (human-readable)
        csv_file = split_output_path / f"{base_filename}_processed_data.csv"
        result_df.to_csv(csv_file, index=False)
        
        split_results.append(result_df)
        log(f"Saved {split_name} results to: {parquet_file} and {csv_file}")
    
    # Calculate split-level statistics
    split_stats['success_rate'] = (
        split_stats['successful_records'] / split_stats['total_records']
        if split_stats['total_records'] > 0 else 0
    )
    split_stats['validation_rate'] = (
        split_stats['validation_passed_records'] / split_stats['total_records']
        if split_stats['total_records'] > 0 else 0
    )
    
    log(f"Completed {split_name} split: {split_stats['total_records']} records, "
        f"{split_stats['success_rate']:.2%} success rate")
    
    return split_stats


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for Bedrock processing with template integration and job_type handling.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary containing processing results and statistics
    """
    # Use print function if no logger is provided
    log = logger or print
    
    try:
        # Get job_type from arguments
        job_type = job_args.job_type
        log(f"Processing with job_type: {job_type}")
        
        # Load prompt templates from Template Generation step (REQUIRED)
        if 'prompt_templates' not in input_paths:
            raise ValueError("prompt_templates input is required for Bedrock Processing")
        
        templates = load_prompt_templates(input_paths['prompt_templates'], log)
        log(f"Loaded templates: system_prompt={bool(templates.get('system_prompt'))}, user_prompt_template={bool(templates.get('user_prompt_template'))}")
        
        # Load validation schema from Template Generation step (REQUIRED)
        if 'validation_schema' not in input_paths:
            raise ValueError("validation_schema input is required for Bedrock Processing")
        
        validation_schema = load_validation_schema(input_paths['validation_schema'], log)
        log(f"Loaded validation schema with {len(validation_schema.get('properties', {}))} properties")
        
        # Build configuration with template integration
        # Priority: Templates (highest) > Environment Variables > Defaults (lowest)
        config = {
            'primary_model_id': environ_vars.get('BEDROCK_PRIMARY_MODEL_ID'),
            'fallback_model_id': environ_vars.get('BEDROCK_FALLBACK_MODEL_ID', ''),
            'inference_profile_arn': environ_vars.get('BEDROCK_INFERENCE_PROFILE_ARN'),
            'inference_profile_required_models': environ_vars.get('BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS', '[]'),
            'region_name': environ_vars.get('AWS_DEFAULT_REGION', 'us-east-1'),
            
            # Templates from Template Generation step (required)
            'system_prompt': templates.get('system_prompt'),
            'user_prompt_template': templates.get('user_prompt_template', 'Analyze: {input_data}'),
            'input_placeholders': templates.get('input_placeholders', []),
            
            # Validation schema for response processing
            'validation_schema': validation_schema,
            
            # API configuration
            'max_tokens': int(environ_vars.get('BEDROCK_MAX_TOKENS', '8192')),
            'temperature': float(environ_vars.get('BEDROCK_TEMPERATURE', '1.0')),
            'top_p': float(environ_vars.get('BEDROCK_TOP_P', '0.999')),
            'max_retries': int(environ_vars.get('BEDROCK_MAX_RETRIES', '3')),
            
            # Processing configuration
            'batch_size': int(environ_vars.get('BEDROCK_BATCH_SIZE', '10')),
            'output_column_prefix': environ_vars.get('BEDROCK_OUTPUT_COLUMN_PREFIX', 'llm_'),
            
            # Concurrency configuration
            'max_concurrent_workers': int(environ_vars.get('BEDROCK_MAX_CONCURRENT_WORKERS', '5')),
            'rate_limit_per_second': int(environ_vars.get('BEDROCK_RATE_LIMIT_PER_SECOND', '10')),
            'concurrency_mode': environ_vars.get('BEDROCK_CONCURRENCY_MODE', 'sequential')  # sequential, concurrent
        }
        
        # Initialize processor with template-driven configuration
        processor = BedrockProcessor(config)
        
        # Load input data
        input_path = Path(input_paths['input_data'])
        output_path = Path(output_paths['processed_data'])
        summary_path = Path(output_paths['analysis_summary'])
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing statistics
        processing_stats = {
            'job_type': job_type,
            'total_files': 0,
            'total_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'validation_passed_records': 0,
            'files_processed': [],
            'splits_processed': [],
            'model_info': processor.inference_profile_info,
            'effective_model_id': processor.effective_model_id,
            'template_integration': {
                'system_prompt_loaded': bool(templates.get('system_prompt')),
                'user_prompt_template_loaded': bool(templates.get('user_prompt_template')),
                'validation_schema_loaded': bool(validation_schema),
                'pydantic_model_created': processor.response_model_class is not None
            }
        }
        
        # Handle different job types based on TabularPreprocessing output structure
        if job_type == "training":
            # Training job type: expect train/val/test subdirectories
            log("Training job type detected - looking for train/val/test subdirectories")
            
            expected_splits = ['train', 'val', 'test']
            splits_found = []
            
            for split_name in expected_splits:
                split_input_path = input_path / split_name
                if split_input_path.exists() and split_input_path.is_dir():
                    splits_found.append(split_name)
                    log(f"Found {split_name} split directory")
            
            if not splits_found:
                # Fallback: treat as single dataset if no splits found
                log("No train/val/test subdirectories found, treating as single dataset")
                input_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.parquet"))
                
                if not input_files:
                    raise ValueError(f"No input files found in {input_path}")
                
                # Process as single dataset (fallback behavior)
                for input_file in input_files:
                    log(f"Processing file: {input_file}")
                    
                    # Load data
                    if input_file.suffix == '.csv':
                        df = pd.read_csv(input_file)
                    else:
                        df = pd.read_parquet(input_file)
                    
                    # Process batch
                    result_df = processor.process_batch(df, save_intermediate=True)
                    
                    # Update statistics
                    processing_stats['total_records'] += len(df)
                    success_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"])
                    failed_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "error"])
                    validation_passed_count = len(result_df[result_df.get(f"{config['output_column_prefix']}validation_passed", False) == True])
                    
                    processing_stats['successful_records'] += success_count
                    processing_stats['failed_records'] += failed_count
                    processing_stats['validation_passed_records'] += validation_passed_count
                    processing_stats['files_processed'].append({
                        'filename': input_file.name,
                        'records': len(df),
                        'successful': success_count,
                        'failed': failed_count,
                        'validation_passed': validation_passed_count,
                        'success_rate': success_count / len(df) if len(df) > 0 else 0,
                        'validation_rate': validation_passed_count / len(df) if len(df) > 0 else 0
                    })
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"processed_{input_file.stem}_{timestamp}"
                    
                    parquet_file = output_path / f"{base_filename}.parquet"
                    result_df.to_parquet(parquet_file, index=False)
                    
                    csv_file = output_path / f"{base_filename}.csv"
                    result_df.to_csv(csv_file, index=False)
                    
                    log(f"Saved results to: {parquet_file} and {csv_file}")
                
                processing_stats['total_files'] = len(input_files)
            else:
                # Process each split separately while preserving structure
                log(f"Processing {len(splits_found)} splits: {splits_found}")
                
                for split_name in splits_found:
                    split_input_path = input_path / split_name
                    split_output_path = output_path / split_name
                    
                    split_stats = process_split_directory(
                        split_name, split_input_path, split_output_path, 
                        processor, config, log
                    )
                    
                    # Aggregate statistics
                    processing_stats['total_files'] += split_stats['total_files']
                    processing_stats['total_records'] += split_stats['total_records']
                    processing_stats['successful_records'] += split_stats['successful_records']
                    processing_stats['failed_records'] += split_stats['failed_records']
                    processing_stats['validation_passed_records'] += split_stats['validation_passed_records']
                    processing_stats['files_processed'].extend(split_stats['files_processed'])
                    processing_stats['splits_processed'].append(split_stats)
        
        else:
            # Non-training job types: expect single dataset
            log(f"Non-training job type ({job_type}) detected - processing single dataset")
            
            input_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.parquet"))
            
            if not input_files:
                raise ValueError(f"No input files found in {input_path}")
            
            processing_stats['total_files'] = len(input_files)
            
            for input_file in input_files:
                log(f"Processing file: {input_file}")
                
                # Load data
                if input_file.suffix == '.csv':
                    df = pd.read_csv(input_file)
                else:
                    df = pd.read_parquet(input_file)
                
                # Process batch
                result_df = processor.process_batch(df, save_intermediate=True)
                
                # Update statistics
                processing_stats['total_records'] += len(df)
                success_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"])
                failed_count = len(result_df[result_df[f"{config['output_column_prefix']}status"] == "error"])
                validation_passed_count = len(result_df[result_df.get(f"{config['output_column_prefix']}validation_passed", False) == True])
                
                processing_stats['successful_records'] += success_count
                processing_stats['failed_records'] += failed_count
                processing_stats['validation_passed_records'] += validation_passed_count
                processing_stats['files_processed'].append({
                    'filename': input_file.name,
                    'records': len(df),
                    'successful': success_count,
                    'failed': failed_count,
                    'validation_passed': validation_passed_count,
                    'success_rate': success_count / len(df) if len(df) > 0 else 0,
                    'validation_rate': validation_passed_count / len(df) if len(df) > 0 else 0
                })
                
                # Save results with job_type in filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"processed_{job_type}_{input_file.stem}_{timestamp}"
                
                # Save as Parquet (efficient for large datasets)
                parquet_file = output_path / f"{base_filename}.parquet"
                result_df.to_parquet(parquet_file, index=False)
                
                # Save as CSV (human-readable)
                csv_file = output_path / f"{base_filename}.csv"
                result_df.to_csv(csv_file, index=False)
                
                log(f"Saved results to: {parquet_file} and {csv_file}")
        
        # Calculate overall statistics
        processing_stats['overall_success_rate'] = (
            processing_stats['successful_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        processing_stats['overall_validation_rate'] = (
            processing_stats['validation_passed_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        processing_stats['processing_timestamp'] = datetime.now().isoformat()
        
        # Save processing summary
        summary_file = summary_path / f"processing_summary_{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        
        log(f"Processing completed successfully for job_type: {job_type}")
        log(f"Total records: {processing_stats['total_records']}")
        log(f"Success rate: {processing_stats['overall_success_rate']:.2%}")
        log(f"Validation rate: {processing_stats['overall_validation_rate']:.2%}")
        log(f"Model used: {processing_stats['effective_model_id']}")
        
        if job_type == "training" and processing_stats['splits_processed']:
            log("Split-level statistics:")
            for split_stats in processing_stats['splits_processed']:
                log(f"  {split_stats['split_name']}: {split_stats['total_records']} records, "
                    f"{split_stats['success_rate']:.2%} success rate")
        
        return processing_stats
        
    except Exception as e:
        log(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Argument parser
        parser = argparse.ArgumentParser(description="Bedrock processing script with template integration")
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration'] - determines processing behavior and output naming"
        )
        parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
        parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for Bedrock calls")
        
        args = parser.parse_args()

        # Set up path dictionaries matching the container paths
        input_paths = {
            "input_data": CONTAINER_PATHS["INPUT_DATA_DIR"],
            "prompt_templates": CONTAINER_PATHS["INPUT_TEMPLATES_DIR"],
            "validation_schema": CONTAINER_PATHS["INPUT_SCHEMA_DIR"]
        }

        output_paths = {
            "processed_data": CONTAINER_PATHS["OUTPUT_DATA_DIR"],
            "analysis_summary": CONTAINER_PATHS["OUTPUT_SUMMARY_DIR"]
        }

        # Environment variables dictionary (template placeholders now come from Template Generation step)
        environ_vars = {
            "BEDROCK_PRIMARY_MODEL_ID": os.environ.get("BEDROCK_PRIMARY_MODEL_ID"),
            "BEDROCK_FALLBACK_MODEL_ID": os.environ.get("BEDROCK_FALLBACK_MODEL_ID", ""),
            "BEDROCK_INFERENCE_PROFILE_ARN": os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN"),
            "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": os.environ.get("BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS", "[]"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "BEDROCK_MAX_TOKENS": os.environ.get("BEDROCK_MAX_TOKENS", "8192"),
            "BEDROCK_TEMPERATURE": os.environ.get("BEDROCK_TEMPERATURE", "1.0"),
            "BEDROCK_TOP_P": os.environ.get("BEDROCK_TOP_P", "0.999"),
            "BEDROCK_BATCH_SIZE": os.environ.get("BEDROCK_BATCH_SIZE", "10"),
            "BEDROCK_MAX_RETRIES": os.environ.get("BEDROCK_MAX_RETRIES", "3"),
            "BEDROCK_OUTPUT_COLUMN_PREFIX": os.environ.get("BEDROCK_OUTPUT_COLUMN_PREFIX", "llm_"),
            
            # Concurrency Configuration:
            # BEDROCK_MAX_CONCURRENT_WORKERS: Number of concurrent threads (default: 5, recommended: 3-10)
            "BEDROCK_MAX_CONCURRENT_WORKERS": os.environ.get("BEDROCK_MAX_CONCURRENT_WORKERS", "5"),
            
            # BEDROCK_RATE_LIMIT_PER_SECOND: API requests per second limit (default: 10)
            "BEDROCK_RATE_LIMIT_PER_SECOND": os.environ.get("BEDROCK_RATE_LIMIT_PER_SECOND", "10"),
            
            # BEDROCK_CONCURRENCY_MODE: Processing mode (default: "sequential")
            # Available values:
            #   - "sequential": Single-threaded processing (safer, easier debugging)
            #   - "concurrent": Multi-threaded processing (faster, 3-10x speedup)
            # Usage examples:
            #   export BEDROCK_CONCURRENCY_MODE="concurrent"  # Enable concurrent processing
            #   export BEDROCK_CONCURRENCY_MODE="sequential"  # Disable concurrent processing (default)
            "BEDROCK_CONCURRENCY_MODE": os.environ.get("BEDROCK_CONCURRENCY_MODE", "sequential")
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        logger.info(f"Bedrock processing completed successfully. Results: {result}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in Bedrock processing script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
