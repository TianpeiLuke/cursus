"""
SageMaker Property Path Validator

Validates SageMaker Step Property Path References based on official SageMaker documentation.
This module implements Level 2 Property Path Validation for the unified alignment tester.

Reference: https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference
"""

import re
from typing import Dict, List, Any, Optional, Tuple


class SageMakerPropertyPathValidator:
    """
    Validates SageMaker step property paths against official documentation.
    
    This validator ensures that property paths used in step specifications
    are valid for the specific SageMaker step type, preventing runtime errors
    in pipeline execution.
    """
    
    def __init__(self):
        """Initialize the property path validator."""
        self.documentation_version = "v2.92.2"
        self.documentation_url = "https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference"
        
        # Cache for property path definitions
        self._property_path_cache = {}
    
    def validate_specification_property_paths(self, specification: Dict[str, Any], 
                                            contract_name: str) -> List[Dict[str, Any]]:
        """
        Validate all property paths in a specification.
        
        Args:
            specification: Specification dictionary
            contract_name: Name of the contract being validated
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Get the step type from specification
        step_type = specification.get('step_type', '').lower()
        node_type = specification.get('node_type', '').lower()
        
        # Get valid property paths for this step type
        valid_property_paths = self._get_valid_property_paths_for_step_type(step_type, node_type)
        
        if not valid_property_paths:
            # If we don't have property path definitions for this step type, skip validation
            issues.append({
                'severity': 'INFO',
                'category': 'property_path_validation',
                'message': f'Property path validation skipped for step_type: {step_type}, node_type: {node_type}',
                'details': {
                    'contract': contract_name,
                    'step_type': step_type,
                    'node_type': node_type,
                    'reason': 'No property path definitions available for this step type'
                },
                'recommendation': 'Consider adding property path definitions for this step type'
            })
            return issues
        
        # Validate property paths in outputs
        for output in specification.get('outputs', []):
            property_path = output.get('property_path', '')
            logical_name = output.get('logical_name', '')
            
            if property_path:
                validation_result = self._validate_single_property_path(
                    property_path, step_type, node_type, valid_property_paths
                )
                
                if not validation_result['valid']:
                    issues.append({
                        'severity': 'ERROR',
                        'category': 'property_path_validation',
                        'message': f'Invalid property path in output {logical_name}: {property_path}',
                        'details': {
                            'contract': contract_name,
                            'logical_name': logical_name,
                            'property_path': property_path,
                            'step_type': step_type,
                            'node_type': node_type,
                            'error': validation_result['error'],
                            'valid_paths': validation_result['suggestions'],
                            'documentation_reference': self.documentation_url
                        },
                        'recommendation': f'Use a valid property path for {step_type}. Valid paths include: {", ".join(validation_result["suggestions"][:5])}'
                    })
                else:
                    # Valid property path - add info message
                    issues.append({
                        'severity': 'INFO',
                        'category': 'property_path_validation',
                        'message': f'Valid property path in output {logical_name}: {property_path}',
                        'details': {
                            'contract': contract_name,
                            'logical_name': logical_name,
                            'property_path': property_path,
                            'step_type': step_type,
                            'validation_source': f'SageMaker Documentation {self.documentation_version}',
                            'documentation_reference': self.documentation_url
                        },
                        'recommendation': 'Property path is correctly formatted for the step type'
                    })
        
        # Validate property paths in dependencies (if they have property references)
        for dependency in specification.get('dependencies', []):
            # Check if dependency has any property path references
            # This could be extended in the future if dependencies start using property paths
            pass
        
        # Add summary information about property path validation
        total_outputs = len(specification.get('outputs', []))
        outputs_with_paths = len([out for out in specification.get('outputs', []) if out.get('property_path')])
        
        if total_outputs > 0:
            issues.append({
                'severity': 'INFO',
                'category': 'property_path_validation_summary',
                'message': f'Property path validation completed for {contract_name}',
                'details': {
                    'contract': contract_name,
                    'step_type': step_type,
                    'node_type': node_type,
                    'total_outputs': total_outputs,
                    'outputs_with_property_paths': outputs_with_paths,
                    'validation_reference': self.documentation_url,
                    'documentation_version': self.documentation_version
                },
                'recommendation': f'Validated {outputs_with_paths}/{total_outputs} outputs with property paths against SageMaker documentation'
            })
        
        return issues
    
    def _get_valid_property_paths_for_step_type(self, step_type: str, node_type: str) -> Dict[str, List[str]]:
        """
        Get valid property paths for a specific SageMaker step type.
        
        Based on SageMaker documentation v2.92.2:
        https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference
        
        Args:
            step_type: The SageMaker step type
            node_type: The node type (if applicable)
            
        Returns:
            Dictionary mapping categories to lists of valid property paths
        """
        # Create cache key
        cache_key = f"{step_type}_{node_type}"
        
        if cache_key in self._property_path_cache:
            return self._property_path_cache[cache_key]
        
        # Normalize step type for matching
        step_type_lower = step_type.lower()
        node_type_lower = node_type.lower()
        
        property_paths = {}
        
        # TrainingStep - Properties from DescribeTrainingJob API
        if 'training' in step_type_lower or node_type_lower == 'training':
            property_paths = {
                'model_artifacts': [
                    'ModelArtifacts.S3ModelArtifacts',
                    'properties.ModelArtifacts.S3ModelArtifacts'
                ],
                'metrics': [
                    'FinalMetricDataList[*].Value',
                    'properties.FinalMetricDataList[*].Value',
                    'FinalMetricDataList[*].MetricName',
                    'properties.FinalMetricDataList[*].MetricName'
                ],
                'job_info': [
                    'TrainingJobName',
                    'properties.TrainingJobName',
                    'TrainingJobArn',
                    'properties.TrainingJobArn',
                    'TrainingJobStatus',
                    'properties.TrainingJobStatus',
                    'CreationTime',
                    'properties.CreationTime',
                    'TrainingStartTime',
                    'properties.TrainingStartTime',
                    'TrainingEndTime',
                    'properties.TrainingEndTime'
                ],
                'hyperparameters': [
                    'HyperParameters',
                    'properties.HyperParameters'
                ],
                'algorithm': [
                    'AlgorithmSpecification',
                    'properties.AlgorithmSpecification',
                    'AlgorithmSpecification.TrainingImage',
                    'properties.AlgorithmSpecification.TrainingImage'
                ],
                'resources': [
                    'ResourceConfig',
                    'properties.ResourceConfig',
                    'ResourceConfig.InstanceType',
                    'properties.ResourceConfig.InstanceType',
                    'ResourceConfig.InstanceCount',
                    'properties.ResourceConfig.InstanceCount'
                ]
            }
        
        # ProcessingStep - Properties from DescribeProcessingJob API
        elif 'processing' in step_type_lower or node_type_lower == 'processing':
            property_paths = {
                'outputs': [
                    'ProcessingOutputConfig.Outputs[*].S3Output.S3Uri',
                    'properties.ProcessingOutputConfig.Outputs[*].S3Output.S3Uri',
                    'ProcessingOutputConfig.Outputs[*].OutputName',
                    'properties.ProcessingOutputConfig.Outputs[*].OutputName'
                ],
                'inputs': [
                    'ProcessingInputs[*].S3Input.S3Uri',
                    'properties.ProcessingInputs[*].S3Input.S3Uri',
                    'ProcessingInputs[*].InputName',
                    'properties.ProcessingInputs[*].InputName'
                ],
                'job_info': [
                    'ProcessingJobName',
                    'properties.ProcessingJobName',
                    'ProcessingJobArn',
                    'properties.ProcessingJobArn',
                    'ProcessingJobStatus',
                    'properties.ProcessingJobStatus',
                    'CreationTime',
                    'properties.CreationTime',
                    'ProcessingStartTime',
                    'properties.ProcessingStartTime',
                    'ProcessingEndTime',
                    'properties.ProcessingEndTime'
                ],
                'resources': [
                    'ProcessingResources',
                    'properties.ProcessingResources',
                    'ProcessingResources.ClusterConfig.InstanceType',
                    'properties.ProcessingResources.ClusterConfig.InstanceType',
                    'ProcessingResources.ClusterConfig.InstanceCount',
                    'properties.ProcessingResources.ClusterConfig.InstanceCount'
                ]
            }
        
        # TransformStep - Properties from DescribeTransformJob API
        elif 'transform' in step_type_lower or node_type_lower == 'transform':
            property_paths = {
                'outputs': [
                    'TransformOutput.S3OutputPath',
                    'properties.TransformOutput.S3OutputPath'
                ],
                'job_info': [
                    'TransformJobName',
                    'properties.TransformJobName',
                    'TransformJobArn',
                    'properties.TransformJobArn',
                    'TransformJobStatus',
                    'properties.TransformJobStatus',
                    'CreationTime',
                    'properties.CreationTime',
                    'TransformStartTime',
                    'properties.TransformStartTime',
                    'TransformEndTime',
                    'properties.TransformEndTime'
                ],
                'model': [
                    'ModelName',
                    'properties.ModelName'
                ],
                'resources': [
                    'TransformResources',
                    'properties.TransformResources',
                    'TransformResources.InstanceType',
                    'properties.TransformResources.InstanceType',
                    'TransformResources.InstanceCount',
                    'properties.TransformResources.InstanceCount'
                ]
            }
        
        # TuningStep - Properties from DescribeHyperParameterTuningJob and ListTrainingJobsForHyperParameterTuningJob APIs
        elif 'tuning' in step_type_lower or 'hyperparameter' in step_type_lower:
            property_paths = {
                'best_training_job': [
                    'BestTrainingJob.TrainingJobName',
                    'properties.BestTrainingJob.TrainingJobName',
                    'BestTrainingJob.TrainingJobArn',
                    'properties.BestTrainingJob.TrainingJobArn',
                    'BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric',
                    'properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric'
                ],
                'training_job_summaries': [
                    'TrainingJobSummaries[*].TrainingJobName',
                    'properties.TrainingJobSummaries[*].TrainingJobName',
                    'TrainingJobSummaries[*].TrainingJobArn',
                    'properties.TrainingJobSummaries[*].TrainingJobArn',
                    'TrainingJobSummaries[*].FinalHyperParameterTuningJobObjectiveMetric',
                    'properties.TrainingJobSummaries[*].FinalHyperParameterTuningJobObjectiveMetric'
                ],
                'job_info': [
                    'HyperParameterTuningJobName',
                    'properties.HyperParameterTuningJobName',
                    'HyperParameterTuningJobArn',
                    'properties.HyperParameterTuningJobArn',
                    'HyperParameterTuningJobStatus',
                    'properties.HyperParameterTuningJobStatus',
                    'CreationTime',
                    'properties.CreationTime',
                    'HyperParameterTuningStartTime',
                    'properties.HyperParameterTuningStartTime',
                    'HyperParameterTuningEndTime',
                    'properties.HyperParameterTuningEndTime'
                ],
                'objective_metric': [
                    'ObjectiveStatusCounters',
                    'properties.ObjectiveStatusCounters',
                    'BestTrainingJob.ObjectiveStatus',
                    'properties.BestTrainingJob.ObjectiveStatus'
                ]
            }
        
        # CreateModelStep - Properties from DescribeModel API
        elif 'model' in step_type_lower and 'create' in step_type_lower:
            property_paths = {
                'model_info': [
                    'ModelName',
                    'properties.ModelName',
                    'ModelArn',
                    'properties.ModelArn',
                    'CreationTime',
                    'properties.CreationTime'
                ],
                'containers': [
                    'PrimaryContainer.ModelDataUrl',
                    'properties.PrimaryContainer.ModelDataUrl',
                    'PrimaryContainer.Image',
                    'properties.PrimaryContainer.Image',
                    'PrimaryContainer.Environment',
                    'properties.PrimaryContainer.Environment'
                ],
                'execution_role': [
                    'ExecutionRoleArn',
                    'properties.ExecutionRoleArn'
                ]
            }
        
        # LambdaStep - OutputParameters
        elif 'lambda' in step_type_lower:
            property_paths = {
                'output_parameters': [
                    'OutputParameters[*]',
                    'properties.OutputParameters[*]'
                ]
            }
        
        # CallbackStep - OutputParameters
        elif 'callback' in step_type_lower:
            property_paths = {
                'output_parameters': [
                    'OutputParameters[*]',
                    'properties.OutputParameters[*]'
                ]
            }
        
        # QualityCheckStep - Specific properties
        elif 'quality' in step_type_lower or 'qualitycheck' in step_type_lower:
            property_paths = {
                'baseline_constraints': [
                    'CalculatedBaselineConstraints',
                    'properties.CalculatedBaselineConstraints'
                ],
                'baseline_statistics': [
                    'CalculatedBaselineStatistics',
                    'properties.CalculatedBaselineStatistics'
                ],
                'drift_check': [
                    'BaselineUsedForDriftCheckStatistics',
                    'properties.BaselineUsedForDriftCheckStatistics',
                    'BaselineUsedForDriftCheckConstraints',
                    'properties.BaselineUsedForDriftCheckConstraints'
                ]
            }
        
        # ClarifyCheckStep - Specific properties
        elif 'clarify' in step_type_lower:
            property_paths = {
                'baseline_constraints': [
                    'CalculatedBaselineConstraints',
                    'properties.CalculatedBaselineConstraints'
                ],
                'drift_check': [
                    'BaselineUsedForDriftCheckConstraints',
                    'properties.BaselineUsedForDriftCheckConstraints'
                ]
            }
        
        # EMRStep - ClusterId
        elif 'emr' in step_type_lower:
            property_paths = {
                'cluster_info': [
                    'ClusterId',
                    'properties.ClusterId'
                ]
            }
        
        # Cache the result
        self._property_path_cache[cache_key] = property_paths
        
        return property_paths
    
    def _validate_single_property_path(self, property_path: str, step_type: str, node_type: str, 
                                     valid_paths: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate a single property path against the valid paths for the step type.
        
        Args:
            property_path: The property path to validate
            step_type: The SageMaker step type
            node_type: The node type
            valid_paths: Dictionary of valid property paths for the step type
            
        Returns:
            Dictionary with validation result and suggestions
        """
        # Flatten all valid paths into a single list
        all_valid_paths = []
        for category, paths in valid_paths.items():
            all_valid_paths.extend(paths)
        
        # Direct match
        if property_path in all_valid_paths:
            return {
                'valid': True,
                'error': None,
                'suggestions': all_valid_paths,
                'match_type': 'exact'
            }
        
        # Check for pattern matches (e.g., array indexing)
        for valid_path in all_valid_paths:
            if self._matches_property_path_pattern(property_path, valid_path):
                return {
                    'valid': True,
                    'error': None,
                    'suggestions': all_valid_paths,
                    'match_type': 'pattern',
                    'matched_pattern': valid_path
                }
        
        # Check for partial matches to provide better suggestions
        suggestions = self._get_property_path_suggestions(property_path, all_valid_paths)
        
        return {
            'valid': False,
            'error': f'Property path "{property_path}" is not valid for step type "{step_type}"',
            'suggestions': suggestions,
            'match_type': 'none'
        }
    
    def _matches_property_path_pattern(self, property_path: str, pattern: str) -> bool:
        """
        Check if a property path matches a pattern with wildcards.
        
        Args:
            property_path: The actual property path
            pattern: The pattern to match against (may contain [*])
            
        Returns:
            True if the property path matches the pattern
        """
        # Convert pattern to regex
        # Handle array indexing patterns like FinalMetricDataList['metric_name'].Value
        # and FinalMetricDataList[*].Value
        
        # Escape special regex characters except [*]
        escaped_pattern = re.escape(pattern)
        
        # Replace escaped [*] with regex pattern for array indexing
        escaped_pattern = escaped_pattern.replace(r'\[\*\]', r'\[[^\]]+\]')
        
        # Create full regex pattern
        regex_pattern = f'^{escaped_pattern}$'
        
        try:
            return bool(re.match(regex_pattern, property_path))
        except re.error:
            # If regex compilation fails, fall back to simple string comparison
            return property_path == pattern
    
    def _get_property_path_suggestions(self, property_path: str, all_valid_paths: List[str]) -> List[str]:
        """
        Get suggestions for a property path based on similarity to valid paths.
        
        Args:
            property_path: The invalid property path
            all_valid_paths: List of all valid property paths
            
        Returns:
            List of suggested property paths
        """
        suggestions = []
        property_path_lower = property_path.lower()
        
        # Score each valid path based on similarity
        scored_paths = []
        
        for valid_path in all_valid_paths:
            score = self._calculate_path_similarity(property_path_lower, valid_path.lower())
            scored_paths.append((score, valid_path))
        
        # Sort by score (descending) and take top suggestions
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 10 suggestions with score > 0
        for score, path in scored_paths[:10]:
            if score > 0:
                suggestions.append(path)
        
        # If no good suggestions, provide some common patterns
        if not suggestions:
            suggestions = [path for path in all_valid_paths[:5]]
        
        return suggestions
    
    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        """
        Calculate similarity between two property paths.
        
        Args:
            path1: First property path (lowercase)
            path2: Second property path (lowercase)
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Split paths into components
        components1 = path1.replace('[', '.').replace(']', '.').split('.')
        components2 = path2.replace('[', '.').replace(']', '.').split('.')
        
        # Remove empty components
        components1 = [c for c in components1 if c]
        components2 = [c for c in components2 if c]
        
        # Calculate component overlap
        common_components = set(components1) & set(components2)
        total_components = set(components1) | set(components2)
        
        if not total_components:
            return 0.0
        
        component_score = len(common_components) / len(total_components)
        
        # Calculate substring similarity
        substring_score = 0.0
        for comp1 in components1:
            for comp2 in components2:
                if comp1 in comp2 or comp2 in comp1:
                    substring_score += 1
                    break
        
        if components1:
            substring_score /= len(components1)
        
        # Combine scores
        return (component_score * 0.7) + (substring_score * 0.3)
    
    def get_step_type_documentation(self, step_type: str, node_type: str = '') -> Dict[str, Any]:
        """
        Get documentation information for a specific step type.
        
        Args:
            step_type: The SageMaker step type
            node_type: The node type (optional)
            
        Returns:
            Dictionary with documentation information
        """
        valid_paths = self._get_valid_property_paths_for_step_type(step_type, node_type)
        
        return {
            'step_type': step_type,
            'node_type': node_type,
            'documentation_url': self.documentation_url,
            'documentation_version': self.documentation_version,
            'valid_property_paths': valid_paths,
            'total_valid_paths': sum(len(paths) for paths in valid_paths.values()),
            'categories': list(valid_paths.keys())
        }
    
    def list_supported_step_types(self) -> List[Dict[str, Any]]:
        """
        List all supported step types and their documentation.
        
        Returns:
            List of supported step types with their information
        """
        supported_types = [
            {'step_type': 'training', 'node_type': 'training', 'description': 'TrainingStep - Properties from DescribeTrainingJob API'},
            {'step_type': 'processing', 'node_type': 'processing', 'description': 'ProcessingStep - Properties from DescribeProcessingJob API'},
            {'step_type': 'transform', 'node_type': 'transform', 'description': 'TransformStep - Properties from DescribeTransformJob API'},
            {'step_type': 'tuning', 'node_type': 'tuning', 'description': 'TuningStep - Properties from DescribeHyperParameterTuningJob API'},
            {'step_type': 'create_model', 'node_type': 'model', 'description': 'CreateModelStep - Properties from DescribeModel API'},
            {'step_type': 'lambda', 'node_type': 'lambda', 'description': 'LambdaStep - OutputParameters'},
            {'step_type': 'callback', 'node_type': 'callback', 'description': 'CallbackStep - OutputParameters'},
            {'step_type': 'quality_check', 'node_type': 'quality', 'description': 'QualityCheckStep - Baseline and drift check properties'},
            {'step_type': 'clarify_check', 'node_type': 'clarify', 'description': 'ClarifyCheckStep - Clarify-specific properties'},
            {'step_type': 'emr', 'node_type': 'emr', 'description': 'EMRStep - EMR cluster properties'}
        ]
        
        # Add documentation info for each type
        for step_info in supported_types:
            doc_info = self.get_step_type_documentation(step_info['step_type'], step_info['node_type'])
            step_info.update(doc_info)
        
        return supported_types


# Convenience function for easy import
def validate_property_paths(specification: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
    """
    Convenience function to validate property paths in a specification.
    
    Args:
        specification: Specification dictionary
        contract_name: Name of the contract being validated
        
    Returns:
        List of validation issues
    """
    validator = SageMakerPropertyPathValidator()
    return validator.validate_specification_property_paths(specification, contract_name)
