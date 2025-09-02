"""Data compatibility validator for pipeline step connections."""

from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field

from ..utils.error_handling import ValidationError

class DataCompatibilityReport(BaseModel):
    """Report on data compatibility between steps."""
    compatible: bool
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_summary: Dict[str, Any] = Field(default_factory=dict)
    workspace_context: Optional[Dict[str, Any]] = None

class DataSchemaInfo(BaseModel):
    """Information about data schema."""
    columns: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    num_rows: Optional[int] = None
    data_format: str = "unknown"
    has_header: bool = True
    sample_data: Optional[Dict[str, List[Any]]] = None

class DataCompatibilityValidator:
    """Validates data compatibility between pipeline steps with workspace awareness."""
    
    def __init__(self, workspace_root: str = None):
        """Initialize validator with optional workspace context."""
        self.compatibility_rules = self._load_compatibility_rules()
        self.workspace_root = workspace_root
        self.cross_workspace_validations = []
    
    def validate_step_transition(self, 
                               producer_output: Dict[str, Any],
                               consumer_input_spec: Dict[str, Any],
                               producer_workspace_info: Dict[str, Any] = None,
                               consumer_workspace_info: Dict[str, Any] = None) -> DataCompatibilityReport:
        """Validate data compatibility between producer and consumer with workspace context."""
        if not producer_output:
            raise ValidationError("Producer output cannot be empty")
        
        if not consumer_input_spec:
            raise ValidationError("Consumer input specification cannot be empty")
        
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
        
        # Phase 5: Cross-workspace validation
        workspace_context = None
        if producer_workspace_info and consumer_workspace_info:
            workspace_context = self._validate_cross_workspace_compatibility(
                producer_workspace_info, consumer_workspace_info, issues, warnings
            )
        
        return DataCompatibilityReport(
            compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            data_summary=self._create_data_summary(producer_output),
            workspace_context=workspace_context
        )
    
    def _validate_cross_workspace_compatibility(self, 
                                              producer_info: Dict[str, Any], 
                                              consumer_info: Dict[str, Any],
                                              issues: List[str],
                                              warnings: List[str]) -> Dict[str, Any]:
        """Validate cross-workspace data compatibility and add workspace-specific checks."""
        
        producer_dev = producer_info.get('developer_id')
        consumer_dev = consumer_info.get('developer_id')
        
        workspace_context = {
            'producer_developer': producer_dev,
            'consumer_developer': consumer_dev,
            'is_cross_workspace': producer_dev != consumer_dev,
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if producer_dev != consumer_dev:
            # Cross-workspace dependency detected
            warnings.append(f"Cross-workspace dependency: {producer_dev} -> {consumer_dev}")
            
            # Track cross-workspace validation
            cross_workspace_validation = {
                'producer_step': producer_info.get('step_name', 'unknown'),
                'producer_developer': producer_dev,
                'consumer_step': consumer_info.get('step_name', 'unknown'),
                'consumer_developer': consumer_dev,
                'validation_result': 'pending'
            }
            
            # Additional cross-workspace checks
            cross_workspace_issues = self._perform_cross_workspace_checks(producer_info, consumer_info)
            if cross_workspace_issues:
                issues.extend(cross_workspace_issues)
                cross_workspace_validation['validation_result'] = 'failed'
                cross_workspace_validation['issues'] = cross_workspace_issues
            else:
                cross_workspace_validation['validation_result'] = 'passed'
            
            self.cross_workspace_validations.append(cross_workspace_validation)
            
            # Add workspace-specific recommendations
            workspace_context['recommendations'] = [
                f"Verify data contract between {producer_dev} and {consumer_dev}",
                "Consider data versioning for cross-workspace dependencies",
                "Ensure consistent data formats across workspaces"
            ]
        
        return workspace_context
    
    def _perform_cross_workspace_checks(self, producer_info: Dict[str, Any], consumer_info: Dict[str, Any]) -> List[str]:
        """Perform additional validation checks for cross-workspace dependencies."""
        issues = []
        
        # Check for workspace-specific data format conventions
        producer_step_type = producer_info.get('step_type', 'unknown')
        consumer_step_type = consumer_info.get('step_type', 'unknown')
        
        # Example: Different step types might have different data format expectations
        if producer_step_type == 'processing' and consumer_step_type == 'training':
            # Processing steps typically output CSV, training steps expect specific formats
            issues.append(
                f"Cross-workspace step type compatibility warning: "
                f"{producer_step_type} -> {consumer_step_type} may require format validation"
            )
        
        # Check for workspace-specific naming conventions
        producer_dev = producer_info.get('developer_id')
        consumer_dev = consumer_info.get('developer_id')
        
        if producer_dev and consumer_dev:
            # Example: Check if developers follow different naming conventions
            if '_' in producer_dev and '-' in consumer_dev:
                issues.append(
                    f"Naming convention mismatch between workspaces: "
                    f"{producer_dev} uses underscores, {consumer_dev} uses hyphens"
                )
        
        return issues
    
    def _validate_schemas(self, output: Dict[str, Any], 
                         input_spec: Dict[str, Any]) -> List[str]:
        """Validate data schemas match expectations."""
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
                        f"Missing columns in {file_name}: {', '.join(missing_cols)}"
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
        """Create a summary of the data for reporting."""
        summary = {}
        
        for file_name, file_info in output.get('files', {}).items():
            file_summary = {
                'format': file_info.get('format', 'unknown'),
                'size': file_info.get('size', 0)
            }
            
            if 'schema' in file_info:
                file_summary['columns'] = len(file_info['schema'].get('columns', []))
                file_summary['rows'] = file_info['schema'].get('num_rows', 0)
            
            summary[file_name] = file_summary
        
        return summary
    
    def _load_compatibility_rules(self) -> Dict[str, Any]:
        """Load compatibility rules from configuration."""
        # In a real implementation, this would load from a configuration file
        # For now, return default rules
        return {
            'csv': {
                'compatible_with': ['csv', 'parquet', 'json'],
                'conversion': {
                    'parquet': {'quality': 'lossless'},
                    'json': {'quality': 'lossless'},
                }
            },
            'parquet': {
                'compatible_with': ['csv', 'parquet', 'json'],
                'conversion': {
                    'csv': {'quality': 'lossy'},
                    'json': {'quality': 'lossless'},
                }
            },
            'json': {
                'compatible_with': ['json', 'csv'],
                'conversion': {
                    'csv': {'quality': 'lossy'},
                }
            }
        }
    
    def analyze_file(self, file_path: Path) -> DataSchemaInfo:
        """Analyze a data file to extract schema information."""
        if not file_path or not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        file_format = file_path.suffix.lower().lstrip('.')
        schema_info = DataSchemaInfo(data_format=file_format)
        
        try:
            # Handle different file formats
            if file_format == 'csv':
                return self._analyze_csv_file(file_path)
            elif file_format == 'parquet':
                return self._analyze_parquet_file(file_path)
            elif file_format == 'json':
                return self._analyze_json_file(file_path)
            else:
                return schema_info
        except Exception as e:
            # Return basic info on error
            schema_info.warnings = [f"Error analyzing file: {str(e)}"]
            return schema_info
    
    def _analyze_csv_file(self, file_path: Path) -> DataSchemaInfo:
        """Analyze a CSV file to extract schema information."""
        try:
            # Read sample of CSV file
            df = pd.read_csv(file_path, nrows=100)
            
            # Build schema information
            schema_info = DataSchemaInfo(
                columns=df.columns.tolist(),
                column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                num_rows=len(df),
                data_format='csv',
                has_header=True
            )
            
            # Add sample data
            schema_info.sample_data = {
                col: df[col].head(5).tolist() for col in df.columns
            }
            
            return schema_info
        except Exception as e:
            # Return basic info on error
            return DataSchemaInfo(
                data_format='csv',
                warnings=[f"Error analyzing CSV: {str(e)}"]
            )
    
    def _analyze_parquet_file(self, file_path: Path) -> DataSchemaInfo:
        """Analyze a Parquet file to extract schema information."""
        try:
            # Read sample of Parquet file
            df = pd.read_parquet(file_path)
            
            # Build schema information
            schema_info = DataSchemaInfo(
                columns=df.columns.tolist(),
                column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
                num_rows=len(df),
                data_format='parquet',
                has_header=True
            )
            
            # Add sample data
            schema_info.sample_data = {
                col: df[col].head(5).tolist() for col in df.columns
            }
            
            return schema_info
        except Exception as e:
            # Return basic info on error
            return DataSchemaInfo(
                data_format='parquet',
                warnings=[f"Error analyzing Parquet: {str(e)}"]
            )
    
    def _analyze_json_file(self, file_path: Path) -> DataSchemaInfo:
        """Analyze a JSON file to extract schema information."""
        try:
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Tabular JSON data (array of objects)
                columns = list(data[0].keys())
                
                # Determine column types
                column_types = {}
                for col in columns:
                    sample_values = [item.get(col) for item in data[:100] if col in item]
                    if sample_values:
                        column_types[col] = self._infer_type(sample_values)
                
                schema_info = DataSchemaInfo(
                    columns=columns,
                    column_types=column_types,
                    num_rows=len(data),
                    data_format='json',
                    has_header=True
                )
                
                # Add sample data
                schema_info.sample_data = {
                    col: [item.get(col) for item in data[:5] if col in item] for col in columns
                }
                
                return schema_info
            else:
                # Non-tabular JSON
                return DataSchemaInfo(
                    data_format='json',
                    has_header=False
                )
        except Exception as e:
            # Return basic info on error
            return DataSchemaInfo(
                data_format='json',
                warnings=[f"Error analyzing JSON: {str(e)}"]
            )
    
    def _infer_type(self, values: List[Any]) -> str:
        """Infer data type from a list of values."""
        if all(isinstance(v, int) for v in values if v is not None):
            return 'int'
        elif all(isinstance(v, (int, float)) for v in values if v is not None):
            return 'float'
        elif all(isinstance(v, bool) for v in values if v is not None):
            return 'bool'
        else:
            return 'string'
    
    def check_compatibility(self, producer_schema: DataSchemaInfo, 
                          consumer_schema: DataSchemaInfo) -> DataCompatibilityReport:
        """Check compatibility between producer and consumer schemas."""
        if not producer_schema:
            raise ValidationError("Producer schema cannot be None")
        
        if not consumer_schema:
            raise ValidationError("Consumer schema cannot be None")
        
        issues = []
        warnings = []
        
        # Check format compatibility
        if producer_schema.data_format != consumer_schema.data_format:
            # Check if conversion is possible
            rules = self.compatibility_rules.get(producer_schema.data_format, {})
            compatible_formats = rules.get('compatible_with', [])
            
            if consumer_schema.data_format in compatible_formats:
                conversion_info = rules.get('conversion', {}).get(consumer_schema.data_format, {})
                if conversion_info.get('quality') == 'lossy':
                    warnings.append(f"Lossy conversion from {producer_schema.data_format} to "
                                   f"{consumer_schema.data_format}")
            else:
                issues.append(f"Incompatible formats: {producer_schema.data_format} -> "
                             f"{consumer_schema.data_format}")
        
        # Check column compatibility
        required_columns = set(consumer_schema.columns)
        available_columns = set(producer_schema.columns)
        
        missing_columns = required_columns - available_columns
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check data type compatibility for common columns
        common_columns = required_columns.intersection(available_columns)
        for col in common_columns:
            producer_type = producer_schema.column_types.get(col)
            consumer_type = consumer_schema.column_types.get(col)
            
            if producer_type and consumer_type and not self._are_types_compatible(
                producer_type, consumer_type
            ):
                issues.append(f"Incompatible types for column '{col}': "
                             f"{producer_type} -> {consumer_type}")
        
        return DataCompatibilityReport(
            compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            data_summary={
                'producer_format': producer_schema.data_format,
                'consumer_format': consumer_schema.data_format,
                'producer_columns': len(producer_schema.columns),
                'consumer_columns': len(consumer_schema.columns),
                'common_columns': len(common_columns),
                'missing_columns': len(missing_columns)
            }
        )
    
    def _are_types_compatible(self, producer_type: str, consumer_type: str) -> bool:
        """Check if two data types are compatible."""
        # Basic compatibility rules
        if producer_type == consumer_type:
            return True
        
        # Numeric type compatibility
        if producer_type in ['int', 'int64', 'int32'] and consumer_type in ['float', 'float64']:
            return True
        
        # String compatibility (strings can accept anything)
        if consumer_type in ['string', 'object', 'str']:
            return True
        
        return False
    
    def validate_workspace_data_flow(self, workspace_dag, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data flow across workspace boundaries.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance
            step_outputs: Dictionary of step outputs
            
        Returns:
            Validation summary for workspace data flow
        """
        if not self.workspace_root:
            return {"error": "No workspace context available"}
        
        validation_summary = {
            'workspace_root': self.workspace_root,
            'total_cross_workspace_dependencies': 0,
            'validation_results': [],
            'developer_compatibility_matrix': {},
            'recommendations': []
        }
        
        try:
            # Get cross-workspace dependencies from DAG
            dependency_validation = workspace_dag.validate_workspace_dependencies()
            cross_workspace_deps = dependency_validation.get('cross_workspace_dependencies', [])
            
            validation_summary['total_cross_workspace_dependencies'] = len(cross_workspace_deps)
            
            # Validate each cross-workspace dependency
            for dep in cross_workspace_deps:
                producer_step = dep['dependency_step']
                consumer_step = dep['dependent_step']
                producer_dev = dep['dependency_developer']
                consumer_dev = dep['dependent_developer']
                
                # Get step outputs if available
                producer_output = step_outputs.get(producer_step, {})
                
                # Create mock consumer input spec (in real implementation, this would come from step contracts)
                consumer_input_spec = {
                    'required_files': ['output'],
                    'file_formats': {'output': 'unknown'},
                    'schemas': {}
                }
                
                # Validate the transition
                compatibility_report = self.validate_step_transition(
                    producer_output,
                    consumer_input_spec,
                    producer_workspace_info={
                        'developer_id': producer_dev,
                        'step_name': producer_step,
                        'step_type': 'unknown'
                    },
                    consumer_workspace_info={
                        'developer_id': consumer_dev,
                        'step_name': consumer_step,
                        'step_type': 'unknown'
                    }
                )
                
                validation_summary['validation_results'].append({
                    'producer_step': producer_step,
                    'consumer_step': consumer_step,
                    'producer_developer': producer_dev,
                    'consumer_developer': consumer_dev,
                    'compatible': compatibility_report.compatible,
                    'issues': compatibility_report.issues,
                    'warnings': compatibility_report.warnings
                })
                
                # Update developer compatibility matrix
                dev_pair = f"{producer_dev}->{consumer_dev}"
                if dev_pair not in validation_summary['developer_compatibility_matrix']:
                    validation_summary['developer_compatibility_matrix'][dev_pair] = {
                        'total_dependencies': 0,
                        'compatible_dependencies': 0,
                        'issues': []
                    }
                
                matrix_entry = validation_summary['developer_compatibility_matrix'][dev_pair]
                matrix_entry['total_dependencies'] += 1
                if compatibility_report.compatible:
                    matrix_entry['compatible_dependencies'] += 1
                else:
                    matrix_entry['issues'].extend(compatibility_report.issues)
            
            # Generate recommendations
            if validation_summary['total_cross_workspace_dependencies'] > 0:
                validation_summary['recommendations'].extend([
                    "Consider establishing data contracts for cross-workspace dependencies",
                    "Implement data versioning strategy for workspace boundaries",
                    "Set up automated validation for cross-workspace data flows"
                ])
                
                # Check compatibility rates
                for dev_pair, matrix_entry in validation_summary['developer_compatibility_matrix'].items():
                    compatibility_rate = matrix_entry['compatible_dependencies'] / matrix_entry['total_dependencies']
                    if compatibility_rate < 0.8:
                        validation_summary['recommendations'].append(
                            f"Low compatibility rate ({compatibility_rate:.1%}) between {dev_pair} - review data contracts"
                        )
        
        except Exception as e:
            validation_summary['error'] = str(e)
        
        return validation_summary
    
    def get_cross_workspace_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all cross-workspace validations performed."""
        if not self.workspace_root:
            return {"error": "No workspace context available"}
        
        summary = {
            'workspace_root': self.workspace_root,
            'total_validations': len(self.cross_workspace_validations),
            'successful_validations': len([v for v in self.cross_workspace_validations if v['validation_result'] == 'passed']),
            'failed_validations': len([v for v in self.cross_workspace_validations if v['validation_result'] == 'failed']),
            'developer_pairs': {},
            'common_issues': {},
            'validations': self.cross_workspace_validations
        }
        
        # Analyze developer pairs
        for validation in self.cross_workspace_validations:
            dev_pair = f"{validation['producer_developer']}->{validation['consumer_developer']}"
            if dev_pair not in summary['developer_pairs']:
                summary['developer_pairs'][dev_pair] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0
                }
            
            pair_stats = summary['developer_pairs'][dev_pair]
            pair_stats['total'] += 1
            if validation['validation_result'] == 'passed':
                pair_stats['passed'] += 1
            else:
                pair_stats['failed'] += 1
        
        # Analyze common issues
        for validation in self.cross_workspace_validations:
            if 'issues' in validation:
                for issue in validation['issues']:
                    if issue not in summary['common_issues']:
                        summary['common_issues'][issue] = 0
                    summary['common_issues'][issue] += 1
        
        return summary
