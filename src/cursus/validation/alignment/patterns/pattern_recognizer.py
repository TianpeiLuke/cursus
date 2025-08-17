"""
Pattern Recognition Engine

Identifies acceptable architectural patterns and filters false positives
in configuration field validation results.
"""

from typing import Dict, Set, List, Any


class PatternRecognizer:
    """
    Recognizes acceptable architectural patterns to reduce false positives in validation.
    
    Provides pattern-aware filtering for configuration field issues based on:
    - Framework-provided fields
    - Inherited configuration patterns
    - Dynamic/runtime fields
    - Builder-specific patterns
    - Standard naming conventions
    """
    
    def __init__(self):
        """Initialize the pattern recognizer with known patterns."""
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize known architectural patterns."""
        # Framework-provided fields that are commonly available
        self.framework_fields = {
            'logger', 'session', 'context', 'environment', 'metadata',
            'step_name', 'step_type', 'execution_id', 'pipeline_id',
            'job_name', 'job_id', 'run_id', 'experiment_id'
        }
        
        # Common inherited fields from BasePipelineConfig and ProcessingStepConfigBase
        self.inherited_config_fields = {
            # From BasePipelineConfig (Tier 1 - Essential User Inputs)
            'author', 'bucket', 'role', 'region', 'service_name', 'pipeline_version',
            
            # From BasePipelineConfig (Tier 2 - System Inputs)
            'model_class', 'current_date', 'framework_version', 'py_version', 'source_dir',
            
            # From BasePipelineConfig (Tier 3 - Derived Properties)
            'aws_region', 'pipeline_name', 'pipeline_description', 'pipeline_s3_loc',
            
            # From ProcessingStepConfigBase
            'processing_instance_count', 'processing_volume_size', 
            'processing_instance_type_large', 'processing_instance_type_small',
            'use_large_processing_instance', 'processing_source_dir',
            'processing_entry_point', 'processing_script_arguments',
            'processing_framework_version', 'effective_source_dir',
            'effective_instance_type', 'script_path',
            
            # From TrainingStepConfigBase (for training builders)
            'training_instance_count', 'training_volume_size',
            'training_instance_type', 'training_entry_point',
            
            # Common hyperparameter fields
            'hyperparameters', 'hyperparameters_s3_uri'
        }
        
        # Environment variable access patterns (not configuration fields)
        self.environment_access_patterns = {
            'env', 'environ', 'environment_vars', 'env_vars'
        }
        
        # Patterns for inherited configuration fields
        self.inherited_patterns = [
            'base_', 'parent_', 'super_', 'common_', 'shared_'
        ]
        
        # Patterns for dynamic configuration fields
        self.dynamic_patterns = [
            'dynamic_', 'runtime_', 'computed_', 'derived_', 'auto_'
        ]
        
        # Optional convenience fields that may not be accessed
        self.convenience_fields = {
            'debug', 'verbose', 'dry_run', 'test_mode', 'profile',
            'cache_enabled', 'parallel_enabled', 'retry_count',
            'monitoring_enabled', 'metrics_enabled', 'profiling_enabled'
        }
        
        # Standard configuration field suffixes
        self.standard_suffixes = ['_config', '_settings', '_params', '_options']
    
    def is_acceptable_pattern(self, field_name: str, builder_name: str, issue_type: str) -> bool:
        """
        Determine if a configuration field issue represents an acceptable architectural pattern.
        
        Args:
            field_name: Name of the configuration field
            builder_name: Name of the builder
            issue_type: Type of issue ('undeclared_access', 'unaccessed_required')
            
        Returns:
            True if this is an acceptable pattern (should be filtered out)
        """
        # Pattern 1: Framework-provided fields
        if field_name in self.framework_fields:
            return True
        
        # Pattern 2: Common inherited configuration fields (MAJOR FIX)
        if field_name in self.inherited_config_fields:
            return True
        
        # Pattern 3: Environment variable access patterns (not config fields)
        if field_name in self.environment_access_patterns:
            return True
        
        # Pattern 4: Inherited configuration fields by prefix
        if any(field_name.startswith(pattern) for pattern in self.inherited_patterns):
            return True
        
        # Pattern 5: Dynamic configuration fields (runtime-determined)
        if any(field_name.startswith(pattern) for pattern in self.dynamic_patterns):
            return True
        
        # Pattern 6: Optional convenience fields that may not be accessed
        if issue_type == 'unaccessed_required' and field_name in self.convenience_fields:
            return True
        
        # Pattern 7: Builder-specific patterns
        builder_patterns = self._get_builder_specific_patterns(builder_name)
        
        if field_name in builder_patterns.get('acceptable_undeclared', set()):
            return issue_type == 'undeclared_access'
        
        if field_name in builder_patterns.get('acceptable_unaccessed', set()):
            return issue_type == 'unaccessed_required'
        
        # Pattern 8: Configuration fields that follow naming conventions
        if issue_type == 'undeclared_access':
            # Fields that follow standard naming conventions are likely legitimate
            if any(field_name.endswith(suffix) for suffix in self.standard_suffixes):
                return True
        
        return False
    
    def _get_builder_specific_patterns(self, builder_name: str) -> Dict[str, Set[str]]:
        """
        Get builder-specific acceptable patterns.
        
        Args:
            builder_name: Name of the builder
            
        Returns:
            Dictionary with 'acceptable_undeclared' and 'acceptable_unaccessed' sets
        """
        patterns = {
            'acceptable_undeclared': set(),
            'acceptable_unaccessed': set()
        }
        
        builder_lower = builder_name.lower()
        
        # Training builders often access framework-provided fields
        if 'training' in builder_lower:
            patterns['acceptable_undeclared'].update({
                'model_dir', 'output_dir', 'checkpoint_dir',
                'num_gpus', 'distributed', 'local_rank',
                'learning_rate', 'batch_size', 'epochs'
            })
        
        # Processing builders often have optional monitoring fields
        if 'processing' in builder_lower or 'transform' in builder_lower:
            patterns['acceptable_unaccessed'].update({
                'monitoring_enabled', 'metrics_enabled', 'profiling_enabled',
                'max_payload_size', 'batch_strategy'
            })
        
        # Evaluation builders often have optional comparison fields
        if 'evaluation' in builder_lower or 'validation' in builder_lower:
            patterns['acceptable_unaccessed'].update({
                'baseline_model', 'comparison_metrics', 'threshold_config',
                'evaluation_strategy', 'metric_definitions'
            })
        
        # Model builders often access model-specific fields
        if 'model' in builder_lower:
            patterns['acceptable_undeclared'].update({
                'model_name', 'model_version', 'model_artifacts',
                'model_package_group_name', 'model_approval_status'
            })
        
        # Package builders have specific packaging fields
        if 'package' in builder_lower or 'packaging' in builder_lower:
            patterns['acceptable_undeclared'].update({
                'package_name', 'package_version', 'package_description',
                'inference_specification', 'source_algorithm_specification'
            })
        
        # Calibration builders have calibration-specific fields
        if 'calibration' in builder_lower:
            patterns['acceptable_undeclared'].update({
                'calibration_method', 'calibration_data', 'probability_threshold'
            })
        
        return patterns
    
    def filter_configuration_issues(self, issues: List[Dict[str, Any]], builder_name: str) -> List[Dict[str, Any]]:
        """
        Filter configuration field issues using pattern recognition.
        
        Args:
            issues: List of validation issues
            builder_name: Name of the builder
            
        Returns:
            Filtered list of issues with acceptable patterns removed
        """
        filtered_issues = []
        
        for issue in issues:
            # Only filter configuration field issues
            if issue.get('category') != 'configuration_fields':
                filtered_issues.append(issue)
                continue
            
            field_name = issue.get('details', {}).get('field_name', '')
            
            # Determine issue type from message
            issue_type = 'unknown'
            message = issue.get('message', '').lower()
            if 'accesses undeclared' in message:
                issue_type = 'undeclared_access'
            elif 'not accessed' in message:
                issue_type = 'unaccessed_required'
            
            # Check if this is an acceptable pattern
            if not self.is_acceptable_pattern(field_name, builder_name, issue_type):
                filtered_issues.append(issue)
            # If it is an acceptable pattern, we skip adding it (filter it out)
        
        return filtered_issues
    
    def get_pattern_summary(self, builder_name: str) -> Dict[str, Any]:
        """
        Get a summary of applicable patterns for a builder.
        
        Args:
            builder_name: Name of the builder
            
        Returns:
            Summary of applicable patterns
        """
        builder_patterns = self._get_builder_specific_patterns(builder_name)
        
        return {
            'builder_name': builder_name,
            'framework_fields': list(self.framework_fields),
            'inherited_patterns': self.inherited_patterns,
            'dynamic_patterns': self.dynamic_patterns,
            'convenience_fields': list(self.convenience_fields),
            'builder_specific': {
                'acceptable_undeclared': list(builder_patterns['acceptable_undeclared']),
                'acceptable_unaccessed': list(builder_patterns['acceptable_unaccessed'])
            },
            'standard_suffixes': self.standard_suffixes
        }


class ValidationPatternFilter:
    """
    High-level filter for validation results using pattern recognition.
    
    Provides a clean interface for filtering validation issues across
    different validation categories.
    """
    
    def __init__(self):
        """Initialize the validation pattern filter."""
        self.pattern_recognizer = PatternRecognizer()
    
    def filter_validation_issues(self, validation_result: Dict[str, Any], builder_name: str) -> Dict[str, Any]:
        """
        Filter validation issues using pattern recognition.
        
        Args:
            validation_result: Complete validation result dictionary
            builder_name: Name of the builder being validated
            
        Returns:
            Filtered validation result with acceptable patterns removed
        """
        if 'issues' not in validation_result:
            return validation_result
        
        # Filter configuration field issues
        filtered_issues = self.pattern_recognizer.filter_configuration_issues(
            validation_result['issues'], builder_name
        )
        
        # Create new validation result with filtered issues
        filtered_result = validation_result.copy()
        filtered_result['issues'] = filtered_issues
        
        # Update pass/fail status based on filtered issues
        has_critical_or_error = any(
            issue['severity'] in ['CRITICAL', 'ERROR'] for issue in filtered_issues
        )
        filtered_result['passed'] = not has_critical_or_error
        
        # Add filtering metadata
        original_count = len(validation_result['issues'])
        filtered_count = len(filtered_issues)
        filtered_result['filtering_metadata'] = {
            'original_issue_count': original_count,
            'filtered_issue_count': filtered_count,
            'issues_filtered_out': original_count - filtered_count,
            'pattern_filter_applied': True
        }
        
        return filtered_result
