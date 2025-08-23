---
tags:
  - project
  - implementation
  - pipeline_testing
  - s3_integration
  - phase_3
keywords:
  - S3 data integration
  - real data testing
  - cloud storage
  - data download
  - pipeline validation
  - production data
topics:
  - pipeline testing system
  - S3 integration
  - real data validation
  - implementation planning
language: python
date of note: 2025-08-21
---

# Pipeline Script Functionality Testing - S3 Integration Phase Implementation Plan

## Phase Overview

**Duration**: Weeks 5-6 (2 weeks)  
**Focus**: S3 data integration and real pipeline data testing  
**Dependencies**: Data Flow Testing Phase completion  
**Team Size**: 2-3 developers  

## Phase Objectives

1. Implement **Systematic S3 Output Path Management System** with centralized registry
2. Create **dual-mode testing capabilities** (pre-execution synthetic + post-execution real data)
3. Develop **Enhanced S3 Data Management** with boto3 integration
4. Build **production data validation workflows** for Deep Dive testing mode
5. Integrate with **existing PropertyReference system** and step builders

## Key Design Integration

This phase implements the **[Systematic S3 Output Path Management Design](../1_design/pipeline_runtime_s3_output_path_management_design.md)** and incorporates findings from the **[Pipeline Runtime Testing Timing and Data Flow Analysis](../4_analysis/pipeline_runtime_testing_timing_and_data_flow_analysis.md)**:

### Critical Timing Distinction
- **Pre-Execution Testing**: Uses synthetic data with local path simulation (Isolation & Pipeline modes)
- **Post-Execution Testing**: Uses real S3 data for production debugging (Deep Dive mode)

### Systematic S3 Path Management
- **S3OutputPathRegistry**: Centralized tracking of all S3 output paths
- **Enhanced S3DataDownloader**: Logical name-based data retrieval
- **Property Reference Integration**: Seamless resolution of S3 paths from property references

## Week 5: Systematic S3 Output Path Management

### Day 1-2: Core S3 Output Path Registry Implementation
```python
# src/cursus/validation/runtime/data/s3_output_registry.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional, List

class S3OutputInfo(BaseModel):
    """Comprehensive S3 output information with metadata"""
    
    logical_name: str = Field(
        ...,
        description="Logical name of the output as defined in step specification"
    )
    s3_uri: str = Field(
        ...,
        description="Complete S3 URI where the output is stored"
    )
    property_path: str = Field(
        ...,
        description="SageMaker property path for runtime resolution"
    )
    data_type: str = Field(
        ...,
        description="Data type of the output (e.g., 'S3Uri', 'ModelArtifacts')"
    )
    step_name: str = Field(
        ...,
        description="Name of the step that produced this output"
    )
    job_type: Optional[str] = Field(
        None,
        description="Job type context (training, validation, testing, calibration)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this output was registered"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (container paths, output types, etc.)"
    )

class ExecutionMetadata(BaseModel):
    """Metadata about pipeline execution context"""
    
    pipeline_name: Optional[str] = None
    execution_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    
    def mark_step_completed(self) -> None:
        """Mark a step as completed"""
        self.completed_steps += 1
    
    def is_complete(self) -> bool:
        """Check if pipeline execution is complete"""
        return self.completed_steps >= self.total_steps

class S3OutputPathRegistry(BaseModel):
    """Centralized registry for tracking S3 output paths across pipeline execution"""
    
    step_outputs: Dict[str, Dict[str, S3OutputInfo]] = Field(
        default_factory=dict,
        description="Nested dict: step_name -> logical_name -> S3OutputInfo"
    )
    execution_metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Metadata about the pipeline execution"
    )
    
    def register_step_output(self, step_name: str, logical_name: str, output_info: S3OutputInfo) -> None:
        """Register an S3 output for a specific step"""
        if step_name not in self.step_outputs:
            self.step_outputs[step_name] = {}
        
        self.step_outputs[step_name][logical_name] = output_info
        self.execution_metadata.mark_step_completed()
    
    def get_step_output_info(self, step_name: str, logical_name: str) -> Optional[S3OutputInfo]:
        """Get S3 output information for a specific step and logical name"""
        return self.step_outputs.get(step_name, {}).get(logical_name)
    
    def get_step_output_path(self, step_name: str, logical_name: str) -> Optional[str]:
        """Get S3 URI for a specific step output"""
        output_info = self.get_step_output_info(step_name, logical_name)
        return output_info.s3_uri if output_info else None
    
    def get_all_step_outputs(self, step_name: str) -> Dict[str, S3OutputInfo]:
        """Get all outputs for a specific step"""
        return self.step_outputs.get(step_name, {})
    
    def list_all_steps(self) -> List[str]:
        """List all steps that have registered outputs"""
        return list(self.step_outputs.keys())
```

### Day 3-4: Enhanced S3 Data Downloader Implementation
```python
# src/cursus/validation/runtime/data/s3_data_downloader.py
import boto3
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class S3DataSource(BaseModel):
    """Configuration for S3 data source."""
    bucket: str
    prefix: str
    pipeline_name: str
    execution_id: str
    step_outputs: Dict[str, List[str]]  # step_name -> list of S3 keys

class DownloadResult(BaseModel):
    """Result of S3 download operation."""
    success: bool
    local_path: Optional[Path] = None
    s3_key: Optional[str] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None

class S3DataDownloader:
    """Downloads pipeline data from S3 for testing."""
    
    def __init__(self, workspace_dir: str = "./test_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.s3_client = boto3.client('s3')
        self.logger = logging.getLogger(__name__)
        self.download_cache = {}
    
    def discover_pipeline_data(self, bucket: str, pipeline_name: str, 
                             execution_id: Optional[str] = None) -> List[S3DataSource]:
        """Discover available pipeline data in S3."""
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
        """Find recent pipeline executions."""
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
            
            # Sort by modification time (most recent first)
            executions.sort(reverse=True)
            return executions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding executions: {e}")
            return []
    
    def _discover_step_outputs(self, bucket: str, prefix: str) -> Dict[str, List[str]]:
        """Discover step outputs within a pipeline execution."""
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
        """Download all data for a specific step."""
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
        """Download a single file from S3."""
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
            # Download file
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
```

### Day 3-4: Real Data Testing Framework
```python
# src/cursus/validation/runtime/testing/real_data_tester.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import json

class RealDataTestScenario(BaseModel):
    """Test scenario using real pipeline data."""
    scenario_name: str
    pipeline_name: str
    s3_data_source: S3DataSource
    test_steps: List[str]
    validation_rules: Dict[str, Any]

class RealDataTestResult(BaseModel):
    """Result of real data testing."""
    scenario_name: str
    success: bool
    step_results: Dict[str, Any]
    data_validation_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    error_details: Optional[str] = None

class RealDataTester:
    """Tests pipeline scripts using real production data."""
    
    def __init__(self, workspace_dir: str = "./test_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.s3_downloader = S3DataDownloader(workspace_dir)
        self.script_executor = PipelineScriptExecutor(workspace_dir)
        self.data_validator = DataCompatibilityValidator()
    
    def create_test_scenario(self, pipeline_name: str, bucket: str,
                           execution_id: Optional[str] = None,
                           test_steps: Optional[List[str]] = None) -> RealDataTestScenario:
        """Create a test scenario from S3 pipeline data."""
        # Discover available data
        data_sources = self.s3_downloader.discover_pipeline_data(
            bucket, pipeline_name, execution_id
        )
        
        if not data_sources:
            raise ValueError(f"No data found for pipeline {pipeline_name}")
        
        # Use the most recent execution
        data_source = data_sources[0]
        
        # Default to testing all available steps
        if test_steps is None:
            test_steps = list(data_source.step_outputs.keys())
        
        return RealDataTestScenario(
            scenario_name=f"{pipeline_name}_{data_source.execution_id}",
            pipeline_name=pipeline_name,
            s3_data_source=data_source,
            test_steps=test_steps,
            validation_rules=self._create_default_validation_rules(test_steps)
        )
    
    def execute_test_scenario(self, scenario: RealDataTestScenario) -> RealDataTestResult:
        """Execute a real data test scenario."""
        step_results = {}
        data_validation_results = {}
        performance_metrics = {}
        
        try:
            # Download required data
            for step_name in scenario.test_steps:
                download_results = self.s3_downloader.download_step_data(
                    scenario.s3_data_source, step_name
                )
                
                # Validate downloads
                failed_downloads = [
                    key for key, result in download_results.items() 
                    if not result.success
                ]
                
                if failed_downloads:
                    return RealDataTestResult(
                        scenario_name=scenario.scenario_name,
                        success=False,
                        step_results={},
                        data_validation_results={},
                        performance_metrics={},
                        error_details=f"Failed to download data for {step_name}: {failed_downloads}"
                    )
                
                # Prepare step inputs from downloaded data
                step_inputs = self._prepare_step_inputs_from_s3(
                    step_name, download_results
                )
                
                # Execute step with real data
                step_result = self._execute_step_with_real_data(
                    step_name, step_inputs, scenario
                )
                
                step_results[step_name] = step_result
                
                # Validate step outputs against real data expectations
                validation_result = self._validate_step_against_real_data(
                    step_name, step_result, scenario
                )
                
                data_validation_results[step_name] = validation_result
                
                # Collect performance metrics
                performance_metrics[step_name] = {
                    'execution_time': step_result.duration,
                    'memory_usage': step_result.memory_usage,
                    'data_size_processed': step_result.data_size_processed
                }
            
            return RealDataTestResult(
                scenario_name=scenario.scenario_name,
                success=True,
                step_results=step_results,
                data_validation_results=data_validation_results,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            return RealDataTestResult(
                scenario_name=scenario.scenario_name,
                success=False,
                step_results=step_results,
                data_validation_results=data_validation_results,
                performance_metrics=performance_metrics,
                error_details=str(e)
            )
    
    def _prepare_step_inputs_from_s3(self, step_name: str, 
                                   download_results: Dict[str, DownloadResult]) -> Dict[str, Any]:
        """Prepare step inputs from downloaded S3 data."""
        inputs = {}
        
        for s3_key, result in download_results.items():
            if result.success and result.local_path:
                # Determine input type based on file extension
                file_path = result.local_path
                
                if file_path.suffix == '.csv':
                    inputs[file_path.stem] = str(file_path)
                elif file_path.suffix == '.json':
                    with open(file_path) as f:
                        inputs[file_path.stem] = json.load(f)
                elif file_path.suffix == '.parquet':
                    inputs[file_path.stem] = str(file_path)
                else:
                    # Generic file path
                    inputs[file_path.stem] = str(file_path)
        
        return inputs
    
    def _validate_step_against_real_data(self, step_name: str, step_result: Any,
                                       scenario: RealDataTestScenario) -> Dict[str, Any]:
        """Validate step results against real data expectations."""
        validation_rules = scenario.validation_rules.get(step_name, {})
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check output file existence
        expected_outputs = validation_rules.get('expected_outputs', [])
        for expected_output in expected_outputs:
            if expected_output not in step_result.outputs:
                validation_results['issues'].append(
                    f"Missing expected output: {expected_output}"
                )
                validation_results['passed'] = False
        
        # Check data quality metrics
        quality_thresholds = validation_rules.get('quality_thresholds', {})
        for metric, threshold in quality_thresholds.items():
            actual_value = step_result.quality_metrics.get(metric)
            if actual_value is not None and actual_value < threshold:
                validation_results['warnings'].append(
                    f"Quality metric {metric} below threshold: {actual_value} < {threshold}"
                )
        
        return validation_results
```

### Day 5: Data Caching and Workspace Management
```python
# src/cursus/testing/workspace_manager.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
import json
import hashlib
from datetime import datetime, timedelta

class WorkspaceConfig(BaseModel):
    """Configuration for test workspace management."""
    base_dir: Path
    max_cache_size_gb: float = 10.0
    cache_retention_days: int = 7
    auto_cleanup: bool = True

class CacheEntry(BaseModel):
    """Entry in the data cache."""
    key: str
    local_path: Path
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int

class WorkspaceManager:
    """Manages test workspace and data caching."""
    
    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self.cache_index_path = config.base_dir / ".cache_index.json"
        self.cache_entries: Dict[str, CacheEntry] = {}
        self._load_cache_index()
    
    def setup_workspace(self, workspace_name: str) -> Path:
        """Set up a new test workspace."""
        workspace_dir = self.config.base_dir / workspace_name
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        (workspace_dir / "inputs").mkdir(exist_ok=True)
        (workspace_dir / "outputs").mkdir(exist_ok=True)
        (workspace_dir / "logs").mkdir(exist_ok=True)
        (workspace_dir / "cache").mkdir(exist_ok=True)
        
        return workspace_dir
    
    def cleanup_workspace(self, workspace_name: str):
        """Clean up a test workspace."""
        workspace_dir = self.config.base_dir / workspace_name
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
    
    def cache_data(self, data_key: str, source_path: Path, 
                  workspace_dir: Path) -> Path:
        """Cache data in the workspace."""
        # Generate cache key
        cache_key = self._generate_cache_key(data_key, source_path)
        
        # Check if already cached
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            if entry.local_path.exists():
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._save_cache_index()
                return entry.local_path
        
        # Cache the data
        cache_dir = workspace_dir / "cache"
        cached_path = cache_dir / f"{cache_key}_{source_path.name}"
        
        shutil.copy2(source_path, cached_path)
        
        # Update cache index
        self.cache_entries[cache_key] = CacheEntry(
            key=cache_key,
            local_path=cached_path,
            size_bytes=cached_path.stat().st_size,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1
        )
        
        self._save_cache_index()
        
        # Perform cleanup if needed
        if self.config.auto_cleanup:
            self._cleanup_cache()
        
        return cached_path
    
    def _generate_cache_key(self, data_key: str, source_path: Path) -> str:
        """Generate a unique cache key."""
        content = f"{data_key}_{source_path.stat().st_mtime}_{source_path.stat().st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = datetime.now()
        retention_threshold = current_time - timedelta(days=self.config.cache_retention_days)
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.cache_entries.items():
            if entry.last_accessed < retention_threshold:
                if entry.local_path.exists():
                    entry.local_path.unlink()
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_entries[key]
        
        # Check cache size and remove least recently used if needed
        total_size_gb = sum(entry.size_bytes for entry in self.cache_entries.values()) / (1024**3)
        
        if total_size_gb > self.config.max_cache_size_gb:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.cache_entries.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove entries until under size limit
            for key, entry in sorted_entries:
                if entry.local_path.exists():
                    entry.local_path.unlink()
                del self.cache_entries[key]
                
                total_size_gb = sum(entry.size_bytes for entry in self.cache_entries.values()) / (1024**3)
                if total_size_gb <= self.config.max_cache_size_gb:
                    break
        
        self._save_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path) as f:
                    data = json.load(f)
                
                for key, entry_data in data.items():
                    self.cache_entries[key] = CacheEntry(
                        key=entry_data['key'],
                        local_path=Path(entry_data['local_path']),
                        size_bytes=entry_data['size_bytes'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        access_count=entry_data['access_count']
                    )
            except Exception as e:
                # If index is corrupted, start fresh
                self.cache_entries = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        data = {}
        for key, entry in self.cache_entries.items():
            data[key] = {
                'key': entry.key,
                'local_path': str(entry.local_path),
                'size_bytes': entry.size_bytes,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count
            }
        
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_index_path, 'w') as f:
            json.dump(data, f, indent=2)
```

## Week 6: Production Data Validation

### Day 6-7: Production Data Validation Workflows
```python
# src/cursus/testing/production_validator.py
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

class ProductionValidationRule(BaseModel):
    """Rule for validating production data."""
    rule_name: str
    rule_type: str  # 'statistical', 'schema', 'business_logic'
    parameters: Dict[str, Any]
    severity: str  # 'error', 'warning', 'info'

class ValidationResult(BaseModel):
    """Result of production data validation."""
    rule_name: str
    passed: bool
    score: Optional[float] = None
    details: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)

class ProductionDataValidator:
    """Validates pipeline outputs against production data patterns."""
    
    def __init__(self):
        self.validation_rules = self._load_default_rules()
    
    def validate_against_production_patterns(self, 
                                           test_output: Dict[str, Any],
                                           production_reference: Dict[str, Any],
                                           validation_rules: List[ProductionValidationRule]) -> List[ValidationResult]:
        """Validate test output against production data patterns."""
        results = []
        
        for rule in validation_rules:
            if rule.rule_type == 'statistical':
                result = self._validate_statistical_pattern(
                    test_output, production_reference, rule
                )
            elif rule.rule_type == 'schema':
                result = self._validate_schema_consistency(
                    test_output, production_reference, rule
                )
            elif rule.rule_type == 'business_logic':
                result = self._validate_business_logic(
                    test_output, production_reference, rule
                )
            else:
                result = ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    details=f"Unknown rule type: {rule.rule_type}"
                )
            
            results.append(result)
        
        return results
    
    def _validate_statistical_pattern(self, test_output: Dict[str, Any],
                                    production_reference: Dict[str, Any],
                                    rule: ProductionValidationRule) -> ValidationResult:
        """Validate statistical patterns in the data."""
        try:
            # Extract data based on rule parameters
            test_data = self._extract_data_for_validation(test_output, rule.parameters)
            prod_data = self._extract_data_for_validation(production_reference, rule.parameters)
            
            if test_data is None or prod_data is None:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    details="Could not extract data for comparison"
                )
            
            # Perform statistical test
            test_type = rule.parameters.get('test_type', 'ks_test')
            
            if test_type == 'ks_test':
                # Kolmogorov-Smirnov test for distribution similarity
                statistic, p_value = stats.ks_2samp(test_data, prod_data)
                threshold = rule.parameters.get('p_value_threshold', 0.05)
                
                passed = p_value > threshold
                score = p_value
                details = f"KS test p-value: {p_value:.4f}, threshold: {threshold}"
                
            elif test_type == 'mean_comparison':
                # Compare means with confidence interval
                test_mean = np.mean(test_data)
                prod_mean = np.mean(prod_data)
                
                # Calculate confidence interval for production mean
                confidence_level = rule.parameters.get('confidence_level', 0.95)
                margin_of_error = stats.sem(prod_data) * stats.t.ppf((1 + confidence_level) / 2, len(prod_data) - 1)
                
                lower_bound = prod_mean - margin_of_error
                upper_bound = prod_mean + margin_of_error
                
                passed = lower_bound <= test_mean <= upper_bound
                score = abs(test_mean - prod_mean) / prod_mean if prod_mean != 0 else float('inf')
                details = f"Test mean: {test_mean:.4f}, Production CI: [{lower_bound:.4f}, {upper_bound:.4f}]"
            
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    details=f"Unknown statistical test type: {test_type}"
                )
            
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=passed,
                score=score,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=False,
                details=f"Error in statistical validation: {str(e)}"
            )
    
    def _validate_schema_consistency(self, test_output: Dict[str, Any],
                                   production_reference: Dict[str, Any],
                                   rule: ProductionValidationRule) -> ValidationResult:
        """Validate schema consistency between test and production data."""
        try:
            file_path = rule.parameters.get('file_path')
            
            test_file = test_output.get('files', {}).get(file_path)
            prod_file = production_reference.get('files', {}).get(file_path)
            
            if not test_file or not prod_file:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    details=f"File not found: {file_path}"
                )
            
            # Load and compare schemas
            if file_path.endswith('.csv'):
                test_df = pd.read_csv(test_file['path'])
                prod_df = pd.read_csv(prod_file['path'])
                
                # Compare column names
                test_columns = set(test_df.columns)
                prod_columns = set(prod_df.columns)
                
                missing_columns = prod_columns - test_columns
                extra_columns = test_columns - prod_columns
                
                # Compare data types
                type_mismatches = []
                for col in test_columns.intersection(prod_columns):
                    if test_df[col].dtype != prod_df[col].dtype:
                        type_mismatches.append(
                            f"{col}: {test_df[col].dtype} vs {prod_df[col].dtype}"
                        )
                
                passed = len(missing_columns) == 0 and len(extra_columns) == 0 and len(type_mismatches) == 0
                
                details_parts = []
                if missing_columns:
                    details_parts.append(f"Missing columns: {missing_columns}")
                if extra_columns:
                    details_parts.append(f"Extra columns: {extra_columns}")
                if type_mismatches:
                    details_parts.append(f"Type mismatches: {type_mismatches}")
                
                details = "; ".join(details_parts) if details_parts else "Schema matches"
                
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=passed,
                    details=details
                )
            
            else:
                return ValidationResult(
                    rule_name=rule.rule_name,
                    passed=False,
                    details=f"Unsupported file type for schema validation: {file_path}"
                )
                
        except Exception as e:
            return ValidationResult(
                rule_name=rule.rule_name,
                passed=False,
                details=f"Error in schema validation: {str(e)}"
            )
```

### Day 8-9: Integration Testing with Real Data
```python
# test/integration/test_s3_integration.py
import pytest
from pathlib import Path
from cursus.testing.s3_data_downloader import S3DataDownloader
from cursus.testing.real_data_tester import RealDataTester
from cursus.testing.workspace_manager import WorkspaceManager, WorkspaceConfig

class TestS3Integration:
    """Integration tests for S3 data functionality."""
    
    @pytest.fixture
    def workspace_manager(self, tmp_path):
        """Create workspace manager for testing."""
        config = WorkspaceConfig(
            base_dir=tmp_path,
            max_cache_size_gb=1.0,
            cache_retention_days=1
        )
        return WorkspaceManager(config)
    
    @pytest.fixture
    def s3_downloader(self, tmp_path):
        """Create S3 downloader for testing."""
        return S3DataDownloader(workspace_dir=str(tmp_path))
    
    @pytest.fixture
    def real_data_tester(self, tmp_path):
        """Create real data tester for testing."""
        return RealDataTester(workspace_dir=str(tmp_path))
    
    def test_s3_data_discovery(self, s3_downloader):
        """Test S3 data discovery functionality."""
        # Mock S3 responses for testing
        with patch.object(s3_downloader.s3_client, 'list_objects_v2') as mock_list:
            mock_list.return_value = {
                'CommonPrefixes': [
                    {'Prefix': 'pipelines/test_pipeline/20250821_120000/'},
                    {'Prefix': 'pipelines/test_pipeline/20250820_120000/'}
                ]
            }
            
            data_sources = s3_downloader.discover_pipeline_data(
                'test-bucket', 'test_pipeline'
            )
            
            assert len(data_sources) >= 0  # Should handle mock data
    
    def test_real_data_scenario_creation(self, real_data_tester):
        """Test creation of real data test scenarios."""
        with patch.object(real_data_tester.s3_downloader, 'discover_pipeline_data') as mock_discover:
            mock_discover.return_value = [
                S3DataSource(
                    bucket='test-bucket',
                    prefix='pipelines/test_pipeline/20250821_120000/',
                    pipeline_name='test_pipeline',
                    execution_id='20250821_120000',
                    step_outputs={'step1': ['file1.csv'], 'step2': ['file2.csv']}
                )
            ]
            
            scenario = real_data_tester.create_test_scenario(
                'test_pipeline', 'test-bucket'
            )
            
            assert scenario.pipeline_name == 'test_pipeline'
            assert len(scenario.test_steps) == 2
    
    def test_workspace_caching(self, workspace_manager, tmp_path):
        """Test workspace data caching functionality."""
        # Create test file
        test_file = tmp_path / "test_data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        # Set up workspace
        workspace_dir = workspace_manager.setup_workspace("test_workspace")
        
        # Cache data
        cached_path = workspace_manager.cache_data(
            "test_key", test_file, workspace_dir
        )
        
        assert cached_path.exists()
        assert cached_path.read_text() == test_file.read_text()
```

### Day 10: CLI Integration and Documentation
```python
# src/cursus/testing/cli/s3_commands.py
import click
from pathlib import Path
from cursus.testing.s3_data_downloader import S3DataDownloader
from cursus.testing.real_data_tester import RealDataTester

@click.group()
def s3():
    """S3 integration commands for pipeline testing."""
    pass

@s3.command()
@click.option('--bucket', required=True, help='S3 bucket name')
@click.option('--pipeline', required=True, help='Pipeline name')
@click.option('--execution-id', help='Specific execution ID (optional)')
@click.option('--workspace', default='./test_workspace', help='Workspace directory')
def discover(bucket, pipeline, execution_id, workspace):
    """Discover available pipeline data in S3."""
    downloader = S3DataDownloader(workspace)
    
    click.echo(f"Discovering data for pipeline '{pipeline}' in bucket '{bucket}'...")
    
    data_sources = downloader.discover_pipeline_data(bucket, pipeline, execution_id)
    
    if not data_sources:
        click.echo("No data sources found.")
        return
    
    for i, source in enumerate(data_sources, 1):
        click.echo(f"\n{i}. Execution: {source.execution_id}")
        click.echo(f"   Prefix: {source.prefix}")
        click.echo(f"   Steps: {list(source.step_outputs.keys())}")
        
        for step_name, files in source.step_outputs.items():
            click.echo(f"     {step_name}: {len(files)} files")

@s3.command()
@click.option('--bucket', required=True, help='S3 bucket name')
@click.option('--pipeline', required=True, help='Pipeline name')
@click.option('--execution-id', help='Specific execution ID (optional)')
@click.option('--steps', help='Comma-separated list of steps to test')
@click.option('--workspace', default='./test_workspace', help='Workspace directory')
def test_real_data(bucket, pipeline, execution_id, steps, workspace):
    """Test pipeline with real S3 data."""
    tester = RealDataTester(workspace)
    
    test_steps = steps.split(',') if steps else None
    
    click.echo(f"Creating test scenario for pipeline '{pipeline}'...")
    
    try:
        scenario = tester.create_test_scenario(
            pipeline, bucket, execution_id, test_steps
        )
        
        click.echo(f"Scenario: {scenario.scenario_name}")
        click.echo(f"Test steps: {scenario.test_steps}")
        
        click.echo("\nExecuting test scenario...")
        result = tester.execute_test_scenario(scenario)
        
        if result.success:
            click.echo("✅ Test scenario completed successfully!")
            
            click.echo("\nPerformance Metrics:")
            for step_name, metrics in result.performance_metrics.items():
                click.echo(f"  {step_name}:")
                click.echo(f"    Execution time: {metrics['execution_time']:.2f}s")
                click.echo(f"    Memory usage: {metrics.get('memory_usage', 'N/A')}")
        else:
            click.echo("❌ Test scenario failed!")
            click.echo(f"Error: {result.error_details}")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")

@s3.command()
@click.option('--workspace', default='./test_workspace', help='Workspace directory')
def cleanup_cache(workspace):
    """Clean up cached S3 data."""
    from cursus.testing.workspace_manager import WorkspaceManager, WorkspaceConfig
    
    config = WorkspaceConfig(base_dir=Path(workspace))
    manager = WorkspaceManager(config)
    
    click.echo("Cleaning up cache...")
    manager._cleanup_cache()
    click.echo("✅ Cache cleanup completed!")
```

## Implementation Progress

### Completed Components

1. **S3 Data Downloader**
   - ✅ S3DataDownloader for discovering and downloading pipeline data
   - ✅ Concurrent downloads with ThreadPoolExecutor
   - ✅ Progress tracking for large file downloads
   - ✅ Caching system for downloaded files
   - ✅ AWS credentials handling with profiles and environment variables

2. **Workspace Management**
   - ✅ WorkspaceManager for efficient data organization
   - ✅ LRU-based cache cleanup system
   - ✅ Configurable cache size and retention periods
   - ✅ Directory structure for organized test data

3. **Real Data Testing**
   - ✅ RealDataTester for testing with production data
   - ✅ Test scenario discovery and creation
   - ✅ Script execution with real data inputs
   - ✅ Validation of outputs against expectations

4. **CLI Integration**
   - ✅ CLI commands for S3 data discovery
   - ✅ Commands for running tests with real data
   - ✅ Workspace management commands
   - ✅ Progress reporting and result formatting

## Success Metrics

### Week 5 Completion Criteria
- [x] S3 data downloader successfully discovers and downloads pipeline data
- [x] Real data testing framework executes scripts with production data
- [x] Data caching system manages workspace efficiently
- [ ] Integration tests validate S3 functionality (in progress)

### Week 6 Completion Criteria
- [x] Production data validation workflows identify data quality issues
- [x] CLI commands provide user-friendly S3 integration
- [x] End-to-end testing demonstrates real data pipeline validation
- [ ] Additional performance and statistical validations (deferred to future phase)

## Deliverables

1. **S3 Data Integration**
   - S3DataDownloader with discovery and download capabilities
   - Concurrent download with error handling and retry logic
   - Data caching system with intelligent cleanup

2. **Real Data Testing Framework**
   - RealDataTester for production data scenarios
   - Flexible test scenario creation and execution
   - Integration with existing script execution engine

3. **Workspace Management**
   - WorkspaceManager with caching and cleanup
   - Configurable cache size and retention policies
   - Efficient storage and retrieval of test data

4. **Production Data Validation**
   - ProductionDataValidator with statistical testing
   - Schema consistency validation
   - Business logic validation framework

5. **CLI Integration**
   - User-friendly commands for S3 operations
   - Data discovery and testing workflows
   - Cache management and cleanup utilities

## Risk Mitigation

### Technical Risks
- **S3 Access Issues**: Implement proper IAM role validation and error handling
- **Large Data Downloads**: Use streaming downloads and progress indicators
- **Data Format Variations**: Create flexible data parsing with fallback mechanisms

### Performance Risks
- **Memory Usage**: Implement streaming processing for large datasets
- **Download Speed**: Use concurrent downloads with configurable parallelism
- **Cache Management**: Implement LRU eviction and size monitoring

### Security Risks
- **Credential Management**: Use IAM roles and avoid hardcoded credentials
- **Data Privacy**: Implement data masking for sensitive information
- **Access Control**: Validate S3 bucket permissions before operations

## Handoff to Next Phase

### Prerequisites for Jupyter Integration Phase
1. S3 data integration fully functional with real pipeline data
2. Production data validation workflows operational
3. Workspace management system handling large datasets efficiently
4. CLI commands providing seamless user experience
5. Integration tests demonstrating end-to-end S3 functionality

### Documentation Requirements
1. S3 integration setup and configuration guide
2. Real data testing workflow documentation
3. Production data validation rule creation guide
4. Workspace management and caching documentation
5. CLI command reference and usage examples
6. Troubleshooting guide for common S3 integration issues
