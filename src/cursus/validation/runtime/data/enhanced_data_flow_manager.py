"""Enhanced data flow manager with S3 path management foundation and timing-aware path resolution."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from pydantic import BaseModel, Field

class DataCompatibilityReport(BaseModel):
    """Report on data compatibility between steps."""
    compatible: bool
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_summary: Dict[str, Any] = Field(default_factory=dict)

class EnhancedDataFlowManager:
    """Enhanced data flow manager with S3 path management foundation."""
    
    def __init__(self, workspace_dir: str, testing_mode: str = "pre_execution"):
        """Initialize enhanced data flow manager.
        
        Args:
            workspace_dir: Directory for test workspace
            testing_mode: "pre_execution" or "post_execution" testing mode
        """
        self.workspace_dir = Path(workspace_dir)
        self.testing_mode = testing_mode  # "pre_execution" or "post_execution"
        self.data_lineage = []
        self.s3_output_registry = None  # Will be integrated in Phase 3
        
        # Create directories
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "synthetic_data").mkdir(exist_ok=True)
        (self.workspace_dir / "s3_data").mkdir(exist_ok=True)
        (self.workspace_dir / "metadata").mkdir(exist_ok=True)
    
    def setup_step_inputs(self, step_name: str, upstream_outputs: Dict, 
                         step_contract: Optional[Any] = None) -> Dict[str, str]:
        """Enhanced input setup with timing-aware path resolution.
        
        Args:
            step_name: Name of the step
            upstream_outputs: Dictionary of upstream outputs
            step_contract: Optional step contract for validation
            
        Returns:
            Dictionary of resolved input paths
        """
        resolved_inputs = {}
        
        for logical_name, upstream_ref in upstream_outputs.items():
            if self.testing_mode == "pre_execution":
                # Pre-execution: Use local synthetic data paths
                resolved_inputs[logical_name] = self._resolve_synthetic_path(
                    step_name, logical_name, upstream_ref
                )
            else:
                # Post-execution: Prepare for real S3 path resolution (Phase 3)
                resolved_inputs[logical_name] = self._prepare_s3_path_resolution(
                    step_name, logical_name, upstream_ref
                )
            
            # Track data lineage
            self._track_data_lineage_entry(step_name, logical_name, upstream_ref)
        
        return resolved_inputs
    
    def _resolve_synthetic_path(self, step_name: str, logical_name: str, 
                               upstream_ref: Any) -> str:
        """Resolve synthetic local paths for pre-execution testing.
        
        Args:
            step_name: Name of the current step
            logical_name: Logical name of the input
            upstream_ref: Reference to upstream output
            
        Returns:
            Path to synthetic data file
        """
        if hasattr(upstream_ref, 'step_name') and hasattr(upstream_ref, 'output_spec'):
            # PropertyReference-like object
            synthetic_path = (
                self.workspace_dir / "synthetic_data" / 
                upstream_ref.step_name / f"{upstream_ref.output_spec.logical_name}.csv"
            )
            return str(synthetic_path)
        else:
            # Direct path provided
            return str(upstream_ref)
    
    def _prepare_s3_path_resolution(self, step_name: str, logical_name: str, 
                                   upstream_ref: Any) -> str:
        """Prepare for S3 path resolution (foundation for Phase 3).
        
        Args:
            step_name: Name of the current step
            logical_name: Logical name of the input
            upstream_ref: Reference to upstream output
            
        Returns:
            Placeholder S3 path (will be resolved in Phase 3)
        """
        # This will be enhanced in Phase 3 with S3OutputPathRegistry
        if hasattr(upstream_ref, 'step_name') and hasattr(upstream_ref, 'output_spec'):
            # For now, return a placeholder that Phase 3 will resolve
            return f"s3://placeholder/{upstream_ref.step_name}/{upstream_ref.output_spec.logical_name}"
        else:
            return str(upstream_ref)
    
    def _track_data_lineage_entry(self, step_name: str, logical_name: str, upstream_ref: Any):
        """Track data lineage with comprehensive metadata.
        
        Args:
            step_name: Name of the current step
            logical_name: Logical name of the input
            upstream_ref: Reference to upstream output
        """
        lineage_entry = {
            'to_step': step_name,
            'to_input': logical_name,
            'timestamp': datetime.now(),
            'testing_mode': self.testing_mode
        }
        
        if hasattr(upstream_ref, 'step_name') and hasattr(upstream_ref, 'output_spec'):
            lineage_entry.update({
                'from_step': upstream_ref.step_name,
                'from_output': upstream_ref.output_spec.logical_name,
                'property_path': getattr(upstream_ref.output_spec, 'property_path', None),
                'data_type': getattr(upstream_ref.output_spec, 'data_type', None)
            })
        
        self.data_lineage.append(lineage_entry)
    
    def create_data_lineage_report(self) -> Dict[str, Any]:
        """Create comprehensive data lineage report.
        
        Returns:
            Dictionary containing lineage report data
        """
        return {
            'lineage_entries': self.data_lineage,
            'total_transfers': len(self.data_lineage),
            'testing_mode': self.testing_mode,
            'unique_steps': len(set(
                entry.get('from_step', '') for entry in self.data_lineage
            ) | set(
                entry.get('to_step', '') for entry in self.data_lineage
            )),
            'generated_at': datetime.now().isoformat()
        }
    
    def validate_data_compatibility(self, producer_output: Dict[str, Any],
                                  consumer_input_spec: Dict[str, Any]) -> DataCompatibilityReport:
        """Validate data compatibility between producer and consumer.
        
        Args:
            producer_output: Output from producer step
            consumer_input_spec: Input specification for consumer step
            
        Returns:
            DataCompatibilityReport with validation results
        """
        issues = []
        warnings = []
        
        # Check required files exist
        for required_file in consumer_input_spec.get('required_files', []):
            if required_file not in producer_output.get('files', {}):
                issues.append(f"Missing required file: {required_file}")
        
        # Check data formats
        for file_name, file_info in producer_output.get('files', {}).items():
            expected_format = consumer_input_spec.get('file_formats', {}).get(file_name)
            if expected_format and file_info.get('format') != expected_format:
                issues.append(
                    f"Format mismatch for {file_name}: "
                    f"expected {expected_format}, got {file_info.get('format')}"
                )
        
        # Check data schemas
        schema_issues = self._validate_schemas(producer_output, consumer_input_spec)
        issues.extend(schema_issues)
        
        return DataCompatibilityReport(
            compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            data_summary=self._create_data_summary(producer_output)
        )
    
    def _validate_schemas(self, output: Dict[str, Any], 
                         input_spec: Dict[str, Any]) -> List[str]:
        """Validate data schemas match expectations.
        
        Args:
            output: Output data from producer
            input_spec: Input specification for consumer
            
        Returns:
            List of schema validation issues
        """
        issues = []
        
        for file_name, file_info in output.get('files', {}).items():
            expected_schema = input_spec.get('schemas', {}).get(file_name)
            if expected_schema and 'schema' in file_info:
                actual_schema = file_info['schema']
                
                # Check required columns
                required_cols = expected_schema.get('required_columns', [])
                actual_cols = actual_schema.get('columns', [])
                
                missing_cols = set(required_cols) - set(actual_cols)
                if missing_cols:
                    issues.append(
                        f"Missing columns in {file_name}: {missing_cols}"
                    )
                
                # Check data types
                for col, expected_type in expected_schema.get('column_types', {}).items():
                    actual_type = actual_schema.get('column_types', {}).get(col)
                    if actual_type and actual_type != expected_type:
                        issues.append(
                            f"Type mismatch in {file_name}.{col}: "
                            f"expected {expected_type}, got {actual_type}"
                        )
        
        return issues
    
    def _create_data_summary(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of output data.
        
        Args:
            output: Output data to summarize
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'total_files': len(output.get('files', {})),
            'file_sizes': {},
            'data_types': set()
        }
        
        for file_name, file_info in output.get('files', {}).items():
            summary['file_sizes'][file_name] = file_info.get('size', 0)
            if 'format' in file_info:
                summary['data_types'].add(file_info['format'])
        
        # Convert set to list for JSON serialization
        summary['data_types'] = list(summary['data_types'])
        
        return summary
    
    def generate_synthetic_data(self, step_name: str, data_spec: Dict[str, Any]) -> Dict[str, str]:
        """Generate synthetic data for testing.
        
        Args:
            step_name: Name of the step
            data_spec: Specification for data generation
            
        Returns:
            Dictionary of generated data file paths
        """
        step_data_dir = self.workspace_dir / "synthetic_data" / step_name
        step_data_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        for output_name, spec in data_spec.items():
            if spec.get('format') == 'csv':
                # Generate CSV data
                file_path = step_data_dir / f"{output_name}.csv"
                self._generate_csv_data(file_path, spec)
                generated_files[output_name] = str(file_path)
            elif spec.get('format') == 'json':
                # Generate JSON data
                file_path = step_data_dir / f"{output_name}.json"
                self._generate_json_data(file_path, spec)
                generated_files[output_name] = str(file_path)
        
        return generated_files
    
    def _generate_csv_data(self, file_path: Path, spec: Dict[str, Any]):
        """Generate synthetic CSV data.
        
        Args:
            file_path: Path to save CSV file
            spec: Data specification
        """
        import pandas as pd
        import numpy as np
        
        rows = spec.get('rows', 100)
        columns = spec.get('columns', ['col1', 'col2', 'col3'])
        
        # Generate random data
        data = {}
        for col in columns:
            if col.endswith('_id'):
                # Generate IDs
                data[col] = range(1, rows + 1)
            elif 'score' in col.lower() or 'rate' in col.lower():
                # Generate scores/rates
                data[col] = np.random.uniform(0, 1, rows)
            else:
                # Generate random strings
                data[col] = [f"value_{i}_{col}" for i in range(rows)]
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def _generate_json_data(self, file_path: Path, spec: Dict[str, Any]):
        """Generate synthetic JSON data.
        
        Args:
            file_path: Path to save JSON file
            spec: Data specification
        """
        data = {
            "generated_at": datetime.now().isoformat(),
            "spec": spec,
            "data": {
                "sample_key": "sample_value",
                "count": spec.get('count', 10),
                "items": [f"item_{i}" for i in range(spec.get('count', 10))]
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
