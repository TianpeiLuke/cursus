"""S3 data downloader for fetching pipeline output data."""

import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class S3DataSource(BaseModel):
    """Configuration for S3 data source."""
    bucket: str
    prefix: str
    pipeline_name: str
    execution_id: str
    step_outputs: Dict[str, List[str]] = Field(default_factory=dict)  # step_name -> list of S3 keys

class DownloadResult(BaseModel):
    """Result of S3 download operation."""
    success: bool
    local_path: Optional[Path] = None
    s3_key: Optional[str] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None

class S3DataDownloader:
    """Downloads pipeline data from S3 for testing."""
    
    def __init__(self, workspace_dir: str = "./pipeline_testing", 
                profile_name: Optional[str] = None, 
                region_name: Optional[str] = None):
        """Initialize with workspace directory and AWS credentials.
        
        Args:
            workspace_dir: Directory for test workspace
            profile_name: Optional AWS profile name to use
            region_name: Optional AWS region name
        """
        self.workspace_dir = Path(workspace_dir)
        
        # Create S3 client with specified profile and region if provided
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.s3_client = session.client('s3')
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.download_cache = {}
    
    @classmethod
    def from_env_vars(cls, workspace_dir: str = "./pipeline_testing") -> "S3DataDownloader":
        """Create an instance using AWS environment variables.
        
        Uses AWS_PROFILE and AWS_REGION environment variables if set.
        
        Args:
            workspace_dir: Directory for test workspace
            
        Returns:
            Configured S3DataDownloader instance
        """
        profile_name = os.environ.get('AWS_PROFILE')
        region_name = os.environ.get('AWS_REGION')
        
        return cls(
            workspace_dir=workspace_dir,
            profile_name=profile_name,
            region_name=region_name
        )
    
    def discover_pipeline_data(self, bucket: str, pipeline_name: str, 
                             execution_id: Optional[str] = None) -> List[S3DataSource]:
        """Discover available pipeline data in S3.
        
        Args:
            bucket: S3 bucket name
            pipeline_name: Name of pipeline to discover
            execution_id: Optional specific execution ID to locate
            
        Returns:
            List of S3DataSource objects representing available data
        """
        if execution_id:
            prefixes = [f"pipelines/{pipeline_name}/{execution_id}/"]
        else:
            # Find recent executions
            prefixes = self._find_recent_executions(bucket, pipeline_name)
        
        data_sources = []
        for prefix in prefixes:
            step_outputs = self._discover_step_outputs(bucket, prefix)
            if step_outputs:
                data_sources.append(S3DataSource(
                    bucket=bucket,
                    prefix=prefix,
                    pipeline_name=pipeline_name,
                    execution_id=execution_id or prefix.split('/')[-2],
                    step_outputs=step_outputs
                ))
        
        return data_sources
    
    def _find_recent_executions(self, bucket: str, pipeline_name: str, 
                               limit: int = 5) -> List[str]:
        """Find recent pipeline executions in S3.
        
        Args:
            bucket: S3 bucket name
            pipeline_name: Name of the pipeline
            limit: Maximum number of executions to return
            
        Returns:
            List of S3 prefixes for recent executions
        """
        prefix = f"pipelines/{pipeline_name}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            # Extract execution IDs from common prefixes
            executions = []
            for common_prefix in response.get('CommonPrefixes', []):
                execution_prefix = common_prefix['Prefix']
                executions.append(execution_prefix)
            
            # If available, sort by modification time (most recent first)
            # This requires additional head requests to get metadata
            try:
                executions_with_dates = []
                for execution_prefix in executions:
                    # Get a sample object from this prefix to check its date
                    list_resp = self.s3_client.list_objects_v2(
                        Bucket=bucket,
                        Prefix=execution_prefix,
                        MaxKeys=1
                    )
                    
                    if 'Contents' in list_resp and list_resp['Contents']:
                        obj = list_resp['Contents'][0]
                        executions_with_dates.append({
                            'prefix': execution_prefix,
                            'date': obj['LastModified']
                        })
                    else:
                        # No objects found, just use the prefix
                        executions_with_dates.append({
                            'prefix': execution_prefix,
                            'date': None
                        })
                
                # Sort by date (None values last)
                executions_with_dates.sort(
                    key=lambda x: (x['date'] is None, x['date']),
                    reverse=True
                )
                
                # Extract just the prefixes
                executions = [item['prefix'] for item in executions_with_dates]
            except Exception as e:
                # Fall back to simple alphabetical sorting if date sorting fails
                self.logger.warning(f"Error sorting executions by date: {e}")
                executions.sort(reverse=True)
                
            return executions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding executions: {e}")
            return []
    
    def _discover_step_outputs(self, bucket: str, prefix: str) -> Dict[str, List[str]]:
        """Discover step outputs within a pipeline execution.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 prefix for the execution
            
        Returns:
            Dictionary mapping step names to lists of S3 keys
        """
        step_outputs = {}
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Extract step name from path structure
                    # Expected: pipelines/{pipeline_name}/{execution_id}/{step_name}/output/...
                    path_parts = key.replace(prefix, '').split('/')
                    if len(path_parts) >= 2:
                        step_name = path_parts[0]
                        if step_name not in step_outputs:
                            step_outputs[step_name] = []
                        step_outputs[step_name].append(key)
            
        except Exception as e:
            self.logger.error(f"Error discovering step outputs: {e}")
        
        return step_outputs
    
    def download_step_data(self, data_source: S3DataSource, 
                          step_name: str, max_workers: int = 4) -> Dict[str, DownloadResult]:
        """Download all data for a specific step.
        
        Args:
            data_source: S3DataSource object with pipeline data info
            step_name: Name of step to download data for
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping S3 keys to DownloadResult objects
        """
        if step_name not in data_source.step_outputs:
            return {}
        
        step_dir = self.workspace_dir / "s3_data" / data_source.pipeline_name / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        
        s3_keys = data_source.step_outputs[step_name]
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_key = {
                executor.submit(self._download_single_file, data_source.bucket, key, step_dir): key
                for key in s3_keys
            }
            
            # Collect results
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results[key] = result
                except Exception as e:
                    results[key] = DownloadResult(
                        success=False,
                        s3_key=key,
                        error=str(e)
                    )
        
        return results
    
    def _download_single_file(self, bucket: str, s3_key: str, 
                            local_dir: Path) -> DownloadResult:
        """Download a single file from S3.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key to download
            local_dir: Local directory to download to
            
        Returns:
            DownloadResult object with download status
        """
        # Create local file path preserving S3 structure
        relative_path = s3_key.split('/')[-1]  # Just filename for simplicity
        local_path = local_dir / relative_path
        
        # Check cache first
        cache_key = f"{bucket}/{s3_key}"
        if cache_key in self.download_cache:
            cached_path = self.download_cache[cache_key]
            if cached_path.exists():
                return DownloadResult(
                    success=True,
                    local_path=cached_path,
                    s3_key=s3_key,
                    size_bytes=cached_path.stat().st_size
                )
        
        try:
            # Create directories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file with progress tracking for large files
            if self._is_large_file(bucket, s3_key, threshold_mb=10):
                self._download_with_progress(bucket, s3_key, local_path)
            else:
                self.s3_client.download_file(bucket, s3_key, str(local_path))
            
            # Cache the result
            self.download_cache[cache_key] = local_path
            
            return DownloadResult(
                success=True,
                local_path=local_path,
                s3_key=s3_key,
                size_bytes=local_path.stat().st_size
            )
            
        except Exception as e:
            return DownloadResult(
                success=False,
                s3_key=s3_key,
                error=str(e)
            )
            
    def _is_large_file(self, bucket: str, s3_key: str, threshold_mb: int = 10) -> bool:
        """Check if a file is large enough to warrant progress tracking.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            threshold_mb: Size threshold in megabytes
            
        Returns:
            True if file exceeds threshold, False otherwise
        """
        try:
            # Get object metadata
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            size_bytes = response.get('ContentLength', 0)
            size_mb = size_bytes / (1024 * 1024)
            
            return size_mb > threshold_mb
        except Exception as e:
            self.logger.warning(f"Error checking file size for {s3_key}: {e}")
            return False
    
    def _download_with_progress(self, bucket: str, s3_key: str, local_path: Path) -> None:
        """Download a file with progress tracking.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local destination path
        """
        try:
            # Get file size
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            total_size = response.get('ContentLength', 0)
            
            # Set up callback for progress tracking
            downloaded_bytes = 0
            last_logged_percent = -1
            
            def progress_callback(bytes_transferred):
                nonlocal downloaded_bytes, last_logged_percent
                downloaded_bytes += bytes_transferred
                percent = int(downloaded_bytes * 100 / total_size)
                
                # Log every 10% to avoid excessive logging
                if percent // 10 > last_logged_percent // 10:
                    last_logged_percent = percent
                    self.logger.info(f"Downloading {s3_key}: {percent}% complete")
            
            # Download file with progress tracking
            self.s3_client.download_file(
                Bucket=bucket,
                Key=s3_key,
                Filename=str(local_path),
                Callback=progress_callback
            )
            
        except Exception as e:
            self.logger.error(f"Error downloading file with progress: {e}")
            # Fall back to standard download
            self.s3_client.download_file(bucket, s3_key, str(local_path))
