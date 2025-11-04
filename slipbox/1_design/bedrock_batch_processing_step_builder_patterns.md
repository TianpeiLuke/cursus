---
tags:
  - design
  - step_builders
  - bedrock_steps
  - batch_processing
  - patterns
  - sagemaker
  - llm_processing
  - aws_bedrock
  - cost_optimization
keywords:
  - bedrock batch processing patterns
  - AWS Bedrock batch inference
  - LLM batch processing
  - cost-efficient processing
  - scalable inference
  - batch job management
topics:
  - step builder patterns
  - bedrock batch processing implementation
  - SageMaker batch LLM processing
  - AWS Bedrock batch architecture
language: python
date of note: 2025-11-03
---

# Bedrock Batch Processing Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock batch processing step builder implementations in the cursus framework. Bedrock batch processing steps create **ProcessingStep** instances that leverage AWS Bedrock's batch inference capabilities for cost-efficient, scalable Large Language Model (LLM) processing tasks. These steps maintain full compatibility with existing **Bedrock Processing** steps while providing significant cost savings (typically 50% reduction) and enhanced scalability for large datasets.

## Integration with Existing Bedrock Ecosystem

Bedrock batch processing steps are designed as drop-in replacements for standard Bedrock processing steps:

1. **Template Generation Step**: Generates structured prompt templates from category definitions
2. **Tabular Preprocessing Step**: Prepares data in train/val/test splits or single datasets
3. **Batch Processing Step**: Consumes templates and data for batch LLM inference
4. **Seamless Integration**: Identical input/output interface to standard Bedrock processing

**Integration Flow:**
```
Category Definitions → Prompt Template Generation → Prompt Templates
                                                         ↓
Tabular Data → Tabular Preprocessing → Processed Data → Bedrock Batch Processing → Categorized Results
```

## Key Architectural Differences from Real-Time Processing

### 1. Processing Mode Selection Pattern
```python
# Standard Bedrock Processing: Always real-time API calls
class BedrockProcessor:
    def process_batch(self, df):
        for row in df.iterrows():
            result = self._invoke_bedrock_realtime(row)
            results.append(result)

# Bedrock Batch Processing: Intelligent mode selection
class BedrockBatchProcessor(BedrockProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.batch_mode = config.get('batch_mode', 'auto')  # auto, batch, realtime
        self.batch_threshold = config.get('batch_threshold', 1000)
        self.batch_role_arn = config.get('batch_role_arn')
        self.s3_client = boto3.client('s3')
    
    def should_use_batch_processing(self, df):
        """Determine processing mode based on data size and configuration."""
        if self.batch_mode == 'realtime':
            return False
        elif self.batch_mode == 'batch':
            return True
        else:  # auto mode
            return len(df) >= self.batch_threshold and self.batch_role_arn is not None
    
    def process_batch(self, df):
        if self.should_use_batch_processing(df):
            return self.process_batch_inference(df)
        else:
            return super().process_batch(df)  # Fallback to real-time
```

### 2. JSONL Conversion Pattern
```python
# Standard Processing: Direct API calls with row data
def process_single_case(self, row_data):
    prompt = self._format_prompt(row_data)
    response = self.bedrock_client.invoke_model(...)
    return self._parse_response(response)

# Batch Processing: Convert to Bedrock batch JSONL format
def convert_df_to_jsonl(self, df):
    """Convert DataFrame to Bedrock batch inference JSONL format."""
    jsonl_records = []
    
    for idx, row in df.iterrows():
        # Use same template formatting logic as real-time processing
        row_data = row.to_dict()
        prompt = self._format_prompt(row_data)
        
        # Create Bedrock batch inference record
        record = {
            "recordId": f"record_{idx}",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": int(self.config['max_tokens']),
                "temperature": float(self.config['temperature']),
                "top_p": float(self.config['top_p']),
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        
        if self.config.get('system_prompt'):
            record["modelInput"]["system"] = self.config['system_prompt']
        
        jsonl_records.append(record)
    
    return jsonl_records
```

### 3. S3 Integration Pattern
```python
# Standard Processing: No S3 integration needed
# Files processed directly from container paths

# Batch Processing: S3 staging for batch jobs
class S3BatchManager:
    def __init__(self, config):
        self.s3_client = boto3.client('s3')
        self.input_bucket = config['batch_input_bucket']
        self.output_bucket = config['batch_output_bucket']
    
    def upload_jsonl_to_s3(self, jsonl_records):
        """Upload JSONL data to S3 for batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"batch-input/input_{timestamp}.jsonl"
        
        # Convert to JSONL format
        jsonl_content = "\n".join([json.dumps(record) for record in jsonl_records])
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.input_bucket,
            Key=s3_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/jsonl'
        )
        
        return f"s3://{self.input_bucket}/{s3_key}"
    
    def download_batch_results(self, output_s3_uri):
        """Download and parse batch job results from S3."""
        # Parse S3 URI and download JSONL results
        # Return list of result dictionaries
```

### 4. Batch Job Management Pattern
```python
# Standard Processing: No job management needed
# Direct API calls with immediate responses

# Batch Processing: Comprehensive job lifecycle management
class BatchJobManager:
    def __init__(self, bedrock_client, config):
        self.bedrock_client = bedrock_client
        self.config = config
    
    def create_batch_job(self, input_s3_uri):
        """Create Bedrock batch inference job."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"bedrock-batch-{timestamp}"
        
        response = self.bedrock_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.config['batch_role_arn'],
            modelId=self.config['effective_model_id'],
            inputDataConfig={
                's3InputDataConfig': {
                    's3Uri': input_s3_uri,
                    's3InputFormat': 'JSONL'
                }
            },
            outputDataConfig={
                's3OutputDataConfig': {
                    's3Uri': f"s3://{self.config['batch_output_bucket']}/batch-output/{timestamp}/"
                }
            },
            timeoutDurationInHours=self.config.get('batch_timeout_hours', 24)
        )
        
        return response['jobArn']
    
    def monitor_batch_job(self, job_arn):
        """Monitor batch job until completion with exponential backoff."""
        while True:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response['status']
            
            if status == 'Completed':
                return response
            elif status in ['Failed', 'Stopping', 'Stopped']:
                raise RuntimeError(f"Batch job failed with status: {status}")
            
            # Exponential backoff
            time.sleep(min(60, 10 * (1.5 ** (time.time() // 60))))
```

### 5. Results Reconstruction Pattern
```python
# Standard Processing: Direct DataFrame manipulation
def process_batch(self, df):
    results = []
    for idx, row in df.iterrows():
        result = self.process_single_case(row.to_dict())
        # Add original row data + LLM results
        result_row = row.to_dict()
        for key, value in result.items():
            result_row[f"{self.output_prefix}{key}"] = value
        results.append(result_row)
    return pd.DataFrame(results)

# Batch Processing: Reconstruct DataFrame from batch results
def convert_batch_results_to_df(self, batch_results, original_df):
    """Convert batch results back to DataFrame format maintaining exact compatibility."""
    processed_rows = []
    output_prefix = self.config['output_column_prefix']
    
    # Create mapping from recordId to original row index
    record_id_to_idx = {}
    for result in batch_results:
        record_id = result['recordId']
        idx = int(record_id.split('_')[1])
        record_id_to_idx[record_id] = idx
    
    # Process each result maintaining original order
    for result in batch_results:
        record_id = result['recordId']
        idx = record_id_to_idx[record_id]
        
        # Get original row data
        original_row = original_df.iloc[idx].to_dict()
        
        try:
            # Parse response using same logic as real-time processing
            if 'modelOutput' in result:
                parsed_result = self._parse_response_with_pydantic(result['modelOutput'])
                
                # Add LLM results with prefix (identical to real-time)
                for key, value in parsed_result.items():
                    if key not in ['processing_status', 'error_message', 'model_info']:
                        original_row[f"{output_prefix}{key}"] = value
                
                original_row[f"{output_prefix}status"] = "success"
            else:
                # Handle error case
                original_row[f"{output_prefix}status"] = "error"
                original_row[f"{output_prefix}error"] = result.get('error', 'Unknown error')
        
        except Exception as e:
            original_row[f"{output_prefix}status"] = "error"
            original_row[f"{output_prefix}error"] = str(e)
        
        processed_rows.append(original_row)
    
    # Sort by original index to maintain order
    processed_rows.sort(key=lambda x: record_id_to_idx.get(x.get('recordId', 'record_0'), 0))
    
    return pd.DataFrame(processed_rows)
```

## Bedrock Batch Processing Script Implementation

The core script implementation extends the existing `bedrock_processing.py` with batch inference capabilities while maintaining 100% compatibility:

### Complete Script Pattern

```python
#!/usr/bin/env python
"""
Bedrock Batch Processing Script

Extends bedrock_processing.py with AWS Bedrock batch inference capabilities.
Maintains identical input/output interface while providing cost-efficient batch processing
for large datasets with automatic fallback to real-time processing.
"""

import os
import json
import argparse
import pandas as pd
import boto3
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime

# Import existing bedrock processing components
from bedrock_processing import (
    load_prompt_templates, 
    load_validation_schema,
    BedrockProcessor,
    CONTAINER_PATHS
)

logger = logging.getLogger(__name__)

class BedrockBatchProcessor(BedrockProcessor):
    """
    Bedrock batch processor extending BedrockProcessor with batch inference capabilities.
    Maintains full compatibility while adding cost-efficient batch processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Batch-specific configuration
        self.batch_mode = config.get('batch_mode', 'auto')  # auto, batch, realtime
        self.batch_threshold = config.get('batch_threshold', 1000)
        self.batch_role_arn = config.get('batch_role_arn')
        self.batch_input_bucket = config.get('batch_input_bucket')
        self.batch_output_bucket = config.get('batch_output_bucket')
        self.batch_timeout_hours = config.get('batch_timeout_hours', 24)
        
        # Initialize S3 client for batch operations
        self.s3_client = boto3.client('s3', region_name=config.get('region_name', 'us-east-1'))
    
    def should_use_batch_processing(self, df: pd.DataFrame) -> bool:
        """Determine whether to use batch or real-time processing."""
        if self.batch_mode == 'realtime':
            return False
        elif self.batch_mode == 'batch':
            return True
        else:  # auto mode
            return (len(df) >= self.batch_threshold and 
                   self.batch_role_arn is not None and
                   self.batch_input_bucket is not None and
                   self.batch_output_bucket is not None)
    
    def convert_df_to_jsonl(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to Bedrock batch JSONL format using existing template logic."""
        jsonl_records = []
        
        for idx, row in df.iterrows():
            # Use parent class method to format prompt with template placeholders
            row_data = row.to_dict()
            prompt = self._format_prompt(row_data)
            
            # Create Bedrock batch inference record
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": int(self.config['max_tokens']),
                "temperature": float(self.config['temperature']),
                "top_p": float(self.config['top_p']),
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if self.config.get('system_prompt'):
                request_body["system"] = self.config['system_prompt']
            
            record = {
                "recordId": f"record_{idx}",
                "modelInput": request_body
            }
            
            jsonl_records.append(record)
        
        return jsonl_records
    
    def upload_jsonl_to_s3(self, jsonl_records: List[Dict[str, Any]]) -> str:
        """Upload JSONL data to S3 for batch processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"batch-input/input_{timestamp}.jsonl"
        
        # Convert to JSONL format
        jsonl_content = "\n".join([json.dumps(record) for record in jsonl_records])
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.batch_input_bucket,
            Key=s3_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/jsonl'
        )
        
        s3_uri = f"s3://{self.batch_input_bucket}/{s3_key}"
        logger.info(f"Uploaded batch input to: {s3_uri}")
        return s3_uri
    
    def create_batch_job(self, input_s3_uri: str) -> str:
        """Create Bedrock batch inference job."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"bedrock-batch-{timestamp}"
        
        output_s3_uri = f"s3://{self.batch_output_bucket}/batch-output/{timestamp}/"
        
        response = self.bedrock_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.batch_role_arn,
            modelId=self.effective_model_id,
            inputDataConfig={
                's3InputDataConfig': {
                    's3Uri': input_s3_uri,
                    's3InputFormat': 'JSONL'
                }
            },
            outputDataConfig={
                's3OutputDataConfig': {
                    's3Uri': output_s3_uri
                }
            },
            timeoutDurationInHours=self.batch_timeout_hours
        )
        
        job_arn = response['jobArn']
        logger.info(f"Created batch job: {job_name} (ARN: {job_arn})")
        logger.info(f"Output will be written to: {output_s3_uri}")
        
        return job_arn
    
    def monitor_batch_job(self, job_arn: str) -> Dict[str, Any]:
        """Monitor batch job until completion with exponential backoff."""
        logger.info(f"Monitoring batch job: {job_arn}")
        
        start_time = time.time()
        check_count = 0
        
        while True:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response['status']
            
            elapsed_time = time.time() - start_time
            logger.info(f"Job status: {status} (elapsed: {elapsed_time/60:.1f} minutes)")
            
            if status == 'Completed':
                logger.info("Batch job completed successfully")
                return response
            elif status in ['Failed', 'Stopping', 'Stopped']:
                error_msg = f"Batch job failed with status: {status}"
                if 'failureMessage' in response:
                    error_msg += f". Error: {response['failureMessage']}"
                raise RuntimeError(error_msg)
            
            # Exponential backoff with maximum wait time
            check_count += 1
            wait_time = min(60, 10 * (1.2 ** check_count))
            time.sleep(wait_time)
    
    def download_batch_results(self, job_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Download and parse batch job results from S3."""
        output_config = job_response['outputDataConfig']['s3OutputDataConfig']
        output_s3_uri = output_config['s3Uri']
        
        # Parse S3 URI
        bucket = output_s3_uri.replace('s3://', '').split('/')[0]
        prefix = '/'.join(output_s3_uri.replace('s3://', '').split('/')[1:])
        
        # List objects in output location
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            raise RuntimeError(f"No output files found at {output_s3_uri}")
        
        # Download and parse results
        all_results = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.jsonl'):
                # Download JSONL file
                response = self.s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                content = response['Body'].read().decode('utf-8')
                
                # Parse JSONL
                for line in content.strip().split('\n'):
                    if line.strip():
                        result = json.loads(line)
                        all_results.append(result)
        
        logger.info(f"Downloaded {len(all_results)} batch results")
        return all_results
    
    def convert_batch_results_to_df(
        self, 
        batch_results: List[Dict[str, Any]], 
        original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert batch results back to DataFrame format maintaining exact compatibility."""
        processed_rows = []
        output_prefix = self.config['output_column_prefix']
        
        # Create mapping from recordId to original row index
        record_id_to_idx = {}
        for result in batch_results:
            record_id = result['recordId']
            idx = int(record_id.split('_')[1])
            record_id_to_idx[record_id] = idx
        
        # Process each result
        for result in batch_results:
            record_id = result['recordId']
            idx = record_id_to_idx[record_id]
            
            # Get original row data
            original_row = original_df.iloc[idx].to_dict()
            
            try:
                # Parse response using parent class method
                if 'modelOutput' in result:
                    parsed_result = self._parse_response_with_pydantic(result['modelOutput'])
                    
                    # Add LLM results with prefix (same as parent class)
                    for key, value in parsed_result.items():
                        if key not in ['processing_status', 'error_message', 'model_info']:
                            original_row[f"{output_prefix}{key}"] = value
                    
                    # Add processing metadata
                    original_row[f"{output_prefix}status"] = parsed_result.get('processing_status', 'success')
                    if parsed_result.get('error_message'):
                        original_row[f"{output_prefix}error"] = parsed_result['error_message']
                
                else:
                    # Handle error case
                    original_row[f"{output_prefix}status"] = "error"
                    original_row[f"{output_prefix}error"] = result.get('error', 'Unknown error')
            
            except Exception as e:
                logger.error(f"Error processing result for record {record_id}: {e}")
                original_row[f"{output_prefix}status"] = "error"
                original_row[f"{output_prefix}error"] = str(e)
            
            processed_rows.append(original_row)
        
        # Sort by original index to maintain order
        processed_rows.sort(key=lambda x: record_id_to_idx.get(x.get('recordId', 'record_0'), 0))
        
        return pd.DataFrame(processed_rows)
    
    def process_batch_inference(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """Process DataFrame using Bedrock batch inference."""
        logger.info(f"Starting batch inference processing for {len(df)} records")
        
        try:
            # 1. Convert DataFrame to JSONL format
            logger.info("Converting DataFrame to JSONL format...")
            jsonl_records = self.convert_df_to_jsonl(df)
            
            # 2. Upload to S3
            logger.info("Uploading data to S3...")
            input_s3_uri = self.upload_jsonl_to_s3(jsonl_records)
            
            # 3. Create batch job
            logger.info("Creating batch inference job...")
            job_arn = self.create_batch_job(input_s3_uri)
            
            # 4. Monitor job completion
            logger.info("Monitoring job completion...")
            job_response = self.monitor_batch_job(job_arn)
            
            # 5. Download results
            logger.info("Downloading batch results...")
            batch_results = self.download_batch_results(job_response)
            
            # 6. Convert back to DataFrame
            logger.info("Converting results back to DataFrame...")
            result_df = self.convert_batch_results_to_df(batch_results, df)
            
            logger.info(f"Batch inference completed successfully for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            logger.info("Falling back to real-time processing...")
            # Fallback to parent class real-time processing
            return super().process_batch(df, batch_size, save_intermediate)
    
    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Main processing method with automatic batch/real-time selection.
        Maintains exact same interface as parent class.
        """
        if self.should_use_batch_processing(df):
            logger.info(f"Using batch processing for {len(df)} records")
            return self.process_batch_inference(df, batch_size, save_intermediate)
        else:
            logger.info(f"Using real-time processing for {len(df)} records")
            return super().process_batch(df, batch_size, save_intermediate)


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for Bedrock batch processing with identical interface to bedrock_processing.py.
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
        
        # Build configuration with template integration + batch settings
        config = {
            # Standard Bedrock configuration (same as bedrock_processing.py)
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
            
            # Concurrency configuration (inherited from parent)
            'max_concurrent_workers': int(environ_vars.get('BEDROCK_MAX_CONCURRENT_WORKERS', '5')),
            'rate_limit_per_second': int(environ_vars.get('BEDROCK_RATE_LIMIT_PER_SECOND', '10')),
            'concurrency_mode': environ_vars.get('BEDROCK_CONCURRENCY_MODE', 'sequential'),
            
            # Batch-specific configuration
            'batch_mode': environ_vars.get('BEDROCK_BATCH_MODE', 'auto'),
            'batch_threshold': int(environ_vars.get('BEDROCK_BATCH_THRESHOLD', '1000')),
            'batch_role_arn': environ_vars.get('BEDROCK_BATCH_ROLE_ARN'),
            'batch_input_bucket': environ_vars.get('BEDROCK_BATCH_INPUT_BUCKET'),
            'batch_output_bucket': environ_vars.get('BEDROCK_BATCH_OUTPUT_BUCKET'),
            'batch_timeout_hours': int(environ_vars.get('BEDROCK_BATCH_TIMEOUT_HOURS', '24'))
        }
        
        # Initialize batch processor (extends BedrockProcessor)
        processor = BedrockBatchProcessor(config)
        
        # Load input data and process using identical logic to bedrock_processing.py
        input_path = Path(input_paths['input_data'])
        output_path = Path(output_paths['processed_data'])
        summary_path = Path(output_paths['analysis_summary'])
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing statistics (same structure as bedrock_processing.py)
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
            },
            # Batch-specific statistics
            'batch_processing_used': False,
            'batch_job_info': None
        }
        
        # Handle different job types with comprehensive format support (identical logic to bedrock_processing.py)
        # CRITICAL: Must handle all TabularPreprocessing output variants:
        # 1. Training job type: train/val/test subdirectories OR single dataset fallback
        # 2. Non-training job types: single dataset with multiple file formats
        # 3. File formats: CSV (.csv), Parquet (.parquet), compressed versions (.csv.gz, .parquet.gz)
        # 4. Multiple files per directory with automatic combination
        
        if job_type == "training":
            # Training job type: expect train/val/test subdirectories from TabularPreprocessing
            log("Training job type detected - looking for train/val/test subdirectories")
            
            expected_splits = ['train', 'val', 'test']
            splits_found = []
            
            for split_name in expected_splits:
                split_input_path = input_path / split_name
                if split_input_path.exists() and split_input_path.is_dir():
                    splits_found.append(split_name)
                    log(f"Found {split_name} split directory")
            
            if not splits_found:
                # Fallback: treat as single dataset if no splits found (TabularPreprocessing fallback mode)
                log("No train/val/test subdirectories found, treating as single dataset")
                
                # Support all TabularPreprocessing output formats
                input_files = (
                    list(input_path.glob("*.csv")) + 
                    list(input_path.glob("*.parquet")) +
                    list(input_path.glob("*.csv.gz")) +
                    list(input_path.glob("*.parquet.gz")) +
                    list(input_path.glob("*_processed_data.csv")) +
                    list(input_path.glob("*_processed_data.parquet"))
                )
                
                if not input_files:
                    raise ValueError(f"No supported input files found in {input_path}. "
                                   f"Expected formats: .csv, .parquet, .csv.gz, .parquet.gz")
                
                log(f"Found {len(input_files)} input files: {[f.name for f in input_files]}")
                
                # Process each file with batch processing capability
                for input_file in input_files:
                    log(f"Processing file: {input_file}")
                    
                    # Load data with format detection
                    df = load_data_file(input_file, log)
                    
                    # Process batch (automatically selects batch vs real-time based on size)
                    result_df = processor.process_batch(df, save_intermediate=True)
                    
                    # Track batch processing usage for statistics
                    batch_used = processor.should_use_batch_processing(df)
                    if batch_used:
                        processing_stats['batch_processing_used'] = True
                        log(f"Used batch processing for {len(df)} records")
                    else:
                        log(f"Used real-time processing for {len(df)} records")
                    
                    # Update statistics (same logic as bedrock_processing.py)
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
                        'validation_rate': validation_passed_count / len(df) if len(df) > 0 else 0,
                        'batch_processing_used': batch_used
                    })
                    
                    # Save results (same format as bedrock_processing.py)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_filename = f"processed_{input_file.stem}_{timestamp}"
                    
                    parquet_file = output_path / f"{base_filename}.parquet"
                    result_df.to_parquet(parquet_file, index=False)
                    
                    csv_file = output_path / f"{base_filename}.csv"
                    result_df.to_csv(csv_file, index=False)
                    
                    log(f"Saved results to: {parquet_file} and {csv_file}")
                
                processing_stats['total_files'] = len(input_files)
            else:
                # Process each split separately with comprehensive format support
                log(f"Processing {len(splits_found)} splits: {splits_found}")
                
                for split_name in splits_found:
                    split_input_path = input_path / split_name
                    split_output_path = output_path / split_name
                    
                    # Process split with batch processing capability
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
                    
                    # Track batch processing usage across splits
                    if split_stats.get('batch_processing_used', False):
                        processing_stats['batch_processing_used'] = True
        
        else:
            # Non-training job types: expect single dataset with comprehensive format support
            log(f"Non-training job type ({job_type}) detected - processing single dataset")
            
            # Support all TabularPreprocessing output formats for non-training jobs
            input_files = (
                list(input_path.glob("*.csv")) + 
                list(input_path.glob("*.parquet")) +
                list(input_path.glob("*.csv.gz")) +
                list(input_path.glob("*.parquet.gz")) +
                list(input_path.glob(f"*{job_type}*.csv")) +
                list(input_path.glob(f"*{job_type}*.parquet")) +
                list(input_path.glob("*_processed_data.csv")) +
                list(input_path.glob("*_processed_data.parquet"))
            )
            
            if not input_files:
                raise ValueError(f"No supported input files found in {input_path} for job_type '{job_type}'. "
                               f"Expected formats: .csv, .parquet, .csv.gz, .parquet.gz")
            
            log(f"Found {len(input_files)} input files: {[f.name for f in input_files]}")
            processing_stats['total_files'] = len(input_files)
            
            for input_file in input_files:
                log(f"Processing file: {input_file}")
                
                # Load data with format detection
                df = load_data_file(input_file, log)
                
                # Process batch (automatically selects batch vs real-time based on size)
                result_df = processor.process_batch(df, save_intermediate=True)
                
                # Track batch processing usage
                batch_used = processor.should_use_batch_processing(df)
                if batch_used:
                    processing_stats['batch_processing_used'] = True
                    log(f"Used batch processing for {len(df)} records")
                else:
                    log(f"Used real-time processing for {len(df)} records")
                
                # Update statistics (same logic as bedrock_processing.py)
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
                    'validation_rate': validation_passed_count / len(df) if len(df) > 0 else 0,
                    'batch_processing_used': batch_used
                })
                
                # Save results with job_type in filename (same as bedrock_processing.py)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"processed_{job_type}_{input_file.stem}_{timestamp}"
                
                # Save as Parquet (efficient for large datasets)
                parquet_file = output_path / f"{base_filename}.parquet"
                result_df.to_parquet(parquet_file, index=False)
                
                # Save as CSV (human-readable)
                csv_file = output_path / f"{base_filename}.csv"
                result_df.to_csv(csv_file, index=False)
                
                log(f"Saved results to: {parquet_file} and {csv_file}")


def load_data_file(file_path: Path, log: Callable[[str], None]) -> pd.DataFrame:
    """
    Load data file with comprehensive format support matching TabularPreprocessing outputs.
    
    Supports all formats that TabularPreprocessing can output:
    - CSV files (.csv)
    - Parquet files (.parquet) 
    - Compressed versions (.csv.gz, .parquet.gz)
    - Various naming patterns from TabularPreprocessing
    """
    try:
        file_str = str(file_path)
        
        if file_str.endswith('.csv.gz'):
            log(f"Loading compressed CSV file: {file_path}")
            return pd.read_csv(file_path, compression='gzip')
        elif file_str.endswith('.csv'):
            log(f"Loading CSV file: {file_path}")
            return pd.read_csv(file_path)
        elif file_str.endswith('.parquet.gz'):
            log(f"Loading compressed Parquet file: {file_path}")
            # Decompress first, then read parquet
            import gzip
            import tempfile
            with gzip.open(file_path, 'rb') as f_in:
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_out:
                    f_out.write(f_in.read())
                    temp_path = f_out.name
            try:
                df = pd.read_parquet(temp_path)
                return df
            finally:
                os.unlink(temp_path)
        elif file_str.endswith('.parquet'):
            log(f"Loading Parquet file: {file_path}")
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load data file {file_path}: {e}")
        
        # Calculate overall statistics (same as bedrock_processing.py)
        processing_stats['overall_success_rate'] = (
            processing_stats['successful_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        processing_stats['overall_validation_rate'] = (
            processing_stats['validation_passed_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        processing_stats['processing_timestamp'] = datetime.now().isoformat()
        
        # Save processing summary (same format as bedrock_processing.py)
        summary_file = summary_path / f"processing_summary_{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        
        log(f"Processing completed successfully for job_type: {job_type}")
        log(f"Total records: {processing_stats['total_records']}")
        log(f"Success rate: {processing_stats['overall_success_rate']:.2%}")
        log(f"Validation rate: {processing_stats['overall_validation_rate']:.2%}")
        log(f"Model used: {processing_stats['effective_model_id']}")
        log(f"Batch processing used: {processing_stats['batch_processing_used']}")
        
        if job_type == "training" and processing_stats['splits_processed']:
            log("Split-level statistics:")
            for split_stats in processing_stats['splits_processed']:
                log(f"  {split_stats['split_name']}: {split_stats['total_records']} records, "
                    f"{split_stats['success_rate']:.2%} success rate")
        
        return processing_stats
        
    except Exception as e:
        log(f"Processing failed: {str(e)}")
        raise


# Import process_split_directory function from bedrock_processing.py
def process_split_directory(
    split_name: str,
    split_input_path: Path,
    split_output_path: Path,
    processor: BedrockBatchProcessor,
    config: Dict[str, Any],
    log: Callable[[str], None]
) -> Dict[str, Any]:
    """
    Process a single split directory (train, val, or test).
    Identical to bedrock_processing.py but uses BedrockBatchProcessor.
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
        
        # Process batch (automatically selects batch vs real-time)
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


if __name__ == "__main__":
    try:
        # Argument parser (identical to bedrock_processing.py)
        parser = argparse.ArgumentParser(description="Bedrock batch processing script with template integration")
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

        # Set up path dictionaries matching the container paths (identical to bedrock_processing.py)
        input_paths = {
            "input_data": CONTAINER_PATHS["INPUT_DATA_DIR"],
            "prompt_templates": CONTAINER_PATHS["INPUT_TEMPLATES_DIR"],
            "validation_schema": CONTAINER_PATHS["INPUT_SCHEMA_DIR"]
        }

        output_paths = {
            "processed_data": CONTAINER_PATHS["OUTPUT_DATA_DIR"],
            "analysis_summary": CONTAINER_PATHS["OUTPUT_SUMMARY_DIR"]
        }

        # Environment variables dictionary (extends bedrock_processing.py with batch settings)
        environ_vars = {
            # Standard Bedrock configuration (same as bedrock_processing.py)
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
            "BEDROCK_MAX_CONCURRENT_WORKERS": os.environ.get("BEDROCK_MAX_CONCURRENT_WORKERS", "5"),
            "BEDROCK_RATE_LIMIT_PER_SECOND": os.environ.get("BEDROCK_RATE_LIMIT_PER_SECOND", "10"),
            "BEDROCK_CONCURRENCY_MODE": os.environ.get("BEDROCK_CONCURRENCY_MODE", "sequential"),
            
            # Batch-specific configuration
            "BEDROCK_BATCH_MODE": os.environ.get("BEDROCK_BATCH_MODE", "auto"),
            "BEDROCK_BATCH_THRESHOLD": os.environ.get("BEDROCK_BATCH_THRESHOLD", "1000"),
            "BEDROCK_BATCH_ROLE_ARN": os.environ.get("BEDROCK_BATCH_ROLE_ARN"),
            "BEDROCK_BATCH_INPUT_BUCKET": os.environ.get("BEDROCK_BATCH_INPUT_BUCKET"),
            "BEDROCK_BATCH_OUTPUT_BUCKET": os.environ.get("BEDROCK_BATCH_OUTPUT_BUCKET"),
            "BEDROCK_BATCH_TIMEOUT_HOURS": os.environ.get("BEDROCK_BATCH_TIMEOUT_HOURS", "24")
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
        logger.info(f"Bedrock batch processing completed successfully. Results: {result}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in Bedrock batch processing script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
```

## Cursus Framework Integration Patterns

### S3 Path Management Using Framework Patterns

The Bedrock batch processing step builder follows the cursus framework's established patterns for S3 path management, using `_get_base_output_path()` and `Join()` functions for dynamic path construction:

```python
class BedrockBatchProcessingStepBuilder(StepBuilderBase):
    """
    Bedrock batch processing step builder following cursus framework patterns.
    """
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Create environment variables using cursus framework patterns for S3 path management.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Get base output path (S3 bucket + execution prefix) from framework
        base_output_path = self._get_base_output_path()
        
        # Create batch-specific S3 paths using Join (same pattern as _get_outputs)
        from sagemaker.workflow.functions import Join
        
        batch_input_path = Join(on="/", values=[base_output_path, "bedrock-batch", "input"])
        batch_output_path = Join(on="/", values=[base_output_path, "bedrock-batch", "output"])
        
        # Add batch-specific environment variables
        env_vars.update({
            'BEDROCK_BATCH_INPUT_S3_PATH': batch_input_path,
            'BEDROCK_BATCH_OUTPUT_S3_PATH': batch_output_path,
            # Keep existing batch configuration
            'BEDROCK_BATCH_MODE': self.config.batch_mode,
            'BEDROCK_BATCH_THRESHOLD': str(self.config.batch_threshold),
            'BEDROCK_BATCH_ROLE_ARN': self.config.batch_role_arn,
            'BEDROCK_BATCH_TIMEOUT_HOURS': str(self.config.batch_timeout_hours)
        })
        
        return env_vars
```

### Script-Level S3 Path Parsing

The batch processing script extracts S3 paths from environment variables and parses them for use with AWS Bedrock batch inference:

```python
class BedrockBatchProcessor(BedrockProcessor):
    """
    Bedrock batch processor with framework-compliant S3 path management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract S3 paths from environment variables (set by step builder)
        self.batch_input_s3_path = os.environ.get('BEDROCK_BATCH_INPUT_S3_PATH')
        self.batch_output_s3_path = os.environ.get('BEDROCK_BATCH_OUTPUT_S3_PATH')
        
        # Parse bucket and prefix from the full S3 paths
        if self.batch_input_s3_path:
            self.input_bucket, self.input_prefix = self._parse_s3_path(self.batch_input_s3_path)
        
        if self.batch_output_s3_path:
            self.output_bucket, self.output_prefix = self._parse_s3_path(self.batch_output_s3_path)
        
        # Initialize S3 client for batch operations
        self.s3_client = boto3.client('s3', region_name=config.get('region_name', 'us-east-1'))
    
    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and prefix components."""
        if isinstance(s3_path, str) and s3_path.startswith('s3://'):
            path_parts = s3_path.replace('s3://', '').split('/')
            bucket = path_parts[0]
            prefix = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''
            return bucket, prefix
        else:
            # Handle SageMaker Join objects or other dynamic paths
            # These will be resolved at runtime by SageMaker
            return str(s3_path), ''
    
    def upload_jsonl_to_s3(self, jsonl_records: List[Dict[str, Any]]) -> str:
        """Upload JSONL data to S3 using framework-provided paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use framework-provided input path with timestamp
        if self.input_prefix:
            s3_key = f"{self.input_prefix}/input_{timestamp}.jsonl"
        else:
            s3_key = f"input_{timestamp}.jsonl"
        
        # Convert to JSONL format
        jsonl_content = "\n".join([json.dumps(record) for record in jsonl_records])
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.input_bucket,
            Key=s3_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/jsonl'
        )
        
        s3_uri = f"s3://{self.input_bucket}/{s3_key}"
        logger.info(f"Uploaded batch input to: {s3_uri}")
        return s3_uri
    
    def create_batch_job(self, input_s3_uri: str) -> str:
        """Create Bedrock batch inference job using framework-provided output path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"bedrock-batch-{timestamp}"
        
        # Use framework-provided output path
        if self.output_prefix:
            output_s3_uri = f"s3://{self.output_bucket}/{self.output_prefix}/batch-output/{timestamp}/"
        else:
            output_s3_uri = f"s3://{self.output_bucket}/batch-output/{timestamp}/"
        
        response = self.bedrock_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.batch_role_arn,
            modelId=self.effective_model_id,
            inputDataConfig={
                's3InputDataConfig': {
                    's3Uri': input_s3_uri,
                    's3InputFormat': 'JSONL'
                }
            },
            outputDataConfig={
                's3OutputDataConfig': {
                    's3Uri': output_s3_uri
                }
            },
            timeoutDurationInHours=self.batch_timeout_hours
        )
        
        job_arn = response['jobArn']
        logger.info(f"Created batch job: {job_name} (ARN: {job_arn})")
        logger.info(f"Output will be written to: {output_s3_uri}")
        
        return job_arn
```

### Framework Benefits

This approach provides several key advantages:

1. **PIPELINE_EXECUTION_TEMP_DIR Support**: Automatically uses the pipeline's execution-specific temporary directory when available
2. **Organized S3 Structure**: Creates organized subdirectories under the pipeline's S3 space (`bedrock-batch/input`, `bedrock-batch/output`)
3. **No Additional Configuration**: Eliminates the need for separate bucket configuration - uses the pipeline's existing S3 bucket
4. **Framework Consistency**: Follows the exact same pattern as other cursus step builders for output path management
5. **Dynamic Path Resolution**: Supports both static S3 paths and SageMaker parameter-based dynamic paths

## Key Design Principles

### 1. **100% Compatibility**
- Identical input/output interface to `bedrock_processing.py`
- Same container paths and environment variables
- Same job type handling (training, validation, testing, calibration)
- Same output file formats and naming conventions

### 2. **Intelligent Processing Mode Selection**
- **Auto Mode**: Automatically selects batch vs real-time based on data size and configuration
- **Batch Mode**: Forces batch processing regardless of data size
- **Real-time Mode**: Forces real-time processing (identical to original script)

### 3. **Seamless Fallback Strategy**
- Automatic fallback to real-time processing if batch processing fails
- Preserves all error handling and retry logic from parent class
- Maintains same output format regardless of processing mode used

### 4. **Template Integration**
- Uses same template loading and formatting logic as `bedrock_processing.py`
- Maintains compatibility with Bedrock Prompt Template Generation step
- Preserves all Pydantic validation and response parsing

### 5. **Cost Optimization**
- Batch processing typically provides 50% cost savings
- Intelligent thresholding to avoid batch overhead for small datasets
- S3 lifecycle management for temporary batch files

### 6. **Framework Compliance**
- Uses cursus framework patterns for S3 path management
- Leverages `_get_base_output_path()` for PIPELINE_EXECUTION_TEMP_DIR support
- Follows established environment variable patterns
- Maintains consistency with other step builders

## Environment Variables

### Standard Bedrock Variables (inherited)
```bash
# Model Configuration
BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"

# API Configuration
BEDROCK_MAX_TOKENS="8192"
BEDROCK_TEMPERATURE="1.0"
BEDROCK_TOP_P="0.999"
BEDROCK_MAX_RETRIES="3"

# Processing Configuration
BEDROCK_BATCH_SIZE="10"
BEDROCK_OUTPUT_COLUMN_PREFIX="llm_"
BEDROCK_CONCURRENCY_MODE="sequential"
```

### Batch-Specific Variables (new)
```bash
# Batch Processing Mode
BEDROCK_BATCH_MODE="auto"  # auto, batch, realtime
BEDROCK_BATCH_THRESHOLD="1000"  # minimum records for batch mode

# AWS Resources for Batch Processing
BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
BEDROCK_BATCH_INPUT_S3_PATH="s3://pipeline-bucket/execution-123/bedrock-batch/input"  # Generated dynamically
BEDROCK_BATCH_OUTPUT_S3_PATH="s3://pipeline-bucket/execution-123/bedrock-batch/output"  # Generated dynamically
BEDROCK_BATCH_TIMEOUT_HOURS="24"
```

## Expected Benefits

### 1. **Cost Efficiency**
- **50% cost reduction** for large datasets through batch pricing
- **Automatic optimization** based on data size and processing requirements
- **Resource efficiency** through AWS-managed batch infrastructure

### 2. **Scalability**
- **No memory limits** - process millions of records without infrastructure constraints
- **Parallel processing** - AWS handles optimal resource allocation
- **Fault tolerance** - Built-in retry and error recovery mechanisms

### 3. **Operational Excellence**
- **Zero configuration changes** required for existing pipelines
- **Automatic fallback** ensures reliability and backward compatibility
- **Enhanced monitoring** with batch job status tracking and cost reporting

### 4. **Developer Experience**
- **Drop-in replacement** for existing bedrock processing steps
- **Identical debugging experience** with same logging and error handling
- **Flexible deployment** - works with existing SageMaker pipeline infrastructure

This design ensures that Bedrock batch processing provides significant cost and scalability benefits while maintaining complete compatibility with existing cursus framework patterns and user workflows.
