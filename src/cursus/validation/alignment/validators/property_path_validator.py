"""
SageMaker Property Path Validator

Validates SageMaker Step Property Path References based on official SageMaker documentation.
This module implements Level 2 Property Path Validation for the unified alignment tester.

Reference: https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference
"""

import re
from typing import Dict, List, Any, Optional, Tuple

# Import step registry functions for proper step type resolution
try:
    from ....registry.step_names import (
        get_step_name_from_spec_type,
        get_sagemaker_step_type,
        validate_step_name,
    )

    STEP_REGISTRY_AVAILABLE = True
except ImportError:
    # Fallback if registry is not available
    STEP_REGISTRY_AVAILABLE = False

    def get_step_name_from_spec_type(spec_type: str) -> str:
        return spec_type.split("_")[0] if "_" in spec_type else spec_type

    def get_sagemaker_step_type(step_name: str) -> str:
        return "Processing"  # Default fallback

    def validate_step_name(step_name: str) -> bool:
        return True


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

    def validate_specification_property_paths(
        self, specification: Dict[str, Any], contract_name: str
    ) -> List[Dict[str, Any]]:
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
        spec_step_type = specification.get("step_type", "")
        node_type = specification.get("node_type", "").lower()

        # STEP REGISTRY INTEGRATION: Resolve actual SageMaker step type
        try:
            if STEP_REGISTRY_AVAILABLE and spec_step_type:
                # Get canonical step name from spec type (e.g., "CurrencyConversion_Training" -> "CurrencyConversion")
                canonical_name = get_step_name_from_spec_type(spec_step_type)

                # Get actual SageMaker step type from registry (e.g., "CurrencyConversion" -> "Processing")
                sagemaker_step_type = get_sagemaker_step_type(canonical_name)

                # Use the resolved SageMaker step type for validation
                resolved_step_type = sagemaker_step_type.lower()

                # Add debug info
                issues.append(
                    {
                        "severity": "INFO",
                        "category": "step_type_resolution",
                        "message": f"Step type resolved via registry: {spec_step_type} -> {canonical_name} -> {sagemaker_step_type}",
                        "details": {
                            "contract": contract_name,
                            "original_spec_type": spec_step_type,
                            "canonical_name": canonical_name,
                            "resolved_sagemaker_type": sagemaker_step_type,
                            "registry_available": STEP_REGISTRY_AVAILABLE,
                        },
                        "recommendation": f"Using {sagemaker_step_type} step property paths for validation",
                    }
                )
            else:
                # Fallback to original logic if registry not available
                resolved_step_type = spec_step_type.lower()

                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "step_type_resolution",
                        "message": f"Step registry not available, using naive step type resolution: {spec_step_type}",
                        "details": {
                            "contract": contract_name,
                            "original_spec_type": spec_step_type,
                            "resolved_step_type": resolved_step_type,
                            "registry_available": STEP_REGISTRY_AVAILABLE,
                        },
                        "recommendation": "Consider fixing step registry imports for more accurate validation",
                    }
                )

        except Exception as e:
            # Fallback if registry resolution fails
            resolved_step_type = spec_step_type.lower()

            issues.append(
                {
                    "severity": "WARNING",
                    "category": "step_type_resolution",
                    "message": f"Step type resolution failed, using fallback: {str(e)}",
                    "details": {
                        "contract": contract_name,
                        "original_spec_type": spec_step_type,
                        "resolved_step_type": resolved_step_type,
                        "error": str(e),
                    },
                    "recommendation": "Check step registry configuration and imports",
                }
            )

        # Get valid property paths for the resolved step type
        valid_property_paths = self._get_valid_property_paths_for_step_type(
            resolved_step_type, node_type
        )

        if not valid_property_paths:
            # If we don't have property path definitions for this step type, skip validation
            issues.append(
                {
                    "severity": "INFO",
                    "category": "property_path_validation",
                    "message": f"Property path validation skipped for step_type: {resolved_step_type}, node_type: {node_type}",
                    "details": {
                        "contract": contract_name,
                        "step_type": resolved_step_type,
                        "node_type": node_type,
                        "reason": "No property path definitions available for this step type",
                    },
                    "recommendation": "Consider adding property path definitions for this step type",
                }
            )
            return issues

        # Validate property paths in outputs
        for output in specification.get("outputs", []):
            property_path = output.get("property_path", "")
            logical_name = output.get("logical_name", "")

            if property_path:
                validation_result = self._validate_single_property_path(
                    property_path, resolved_step_type, node_type, valid_property_paths
                )

                if not validation_result["valid"]:
                    issues.append(
                        {
                            "severity": "ERROR",
                            "category": "property_path_validation",
                            "message": f"Invalid property path in output {logical_name}: {property_path}",
                            "details": {
                                "contract": contract_name,
                                "logical_name": logical_name,
                                "property_path": property_path,
                                "step_type": resolved_step_type,
                                "node_type": node_type,
                                "error": validation_result["error"],
                                "valid_paths": validation_result["suggestions"],
                                "documentation_reference": self.documentation_url,
                            },
                            "recommendation": f'Use a valid property path for {resolved_step_type}. Valid paths include: {", ".join(validation_result["suggestions"][:5])}',
                        }
                    )
                else:
                    # Valid property path - add info message
                    issues.append(
                        {
                            "severity": "INFO",
                            "category": "property_path_validation",
                            "message": f"Valid property path in output {logical_name}: {property_path}",
                            "details": {
                                "contract": contract_name,
                                "logical_name": logical_name,
                                "property_path": property_path,
                                "step_type": resolved_step_type,
                                "validation_source": f"SageMaker Documentation {self.documentation_version}",
                                "documentation_reference": self.documentation_url,
                            },
                            "recommendation": "Property path is correctly formatted for the step type",
                        }
                    )

        # Validate property paths in dependencies (if they have property references)
        for dependency in specification.get("dependencies", []):
            # Check if dependency has any property path references
            # This could be extended in the future if dependencies start using property paths
            pass

        # Add summary information about property path validation
        total_outputs = len(specification.get("outputs", []))
        outputs_with_paths = len(
            [
                out
                for out in specification.get("outputs", [])
                if out.get("property_path")
            ]
        )

        # Always add summary, even when there are no outputs
        issues.append(
            {
                "severity": "INFO",
                "category": "property_path_validation_summary",
                "message": f"Property path validation completed for {contract_name}",
                "details": {
                    "contract": contract_name,
                    "step_type": resolved_step_type,
                    "node_type": node_type,
                    "total_outputs": total_outputs,
                    "outputs_with_property_paths": outputs_with_paths,
                    "validation_reference": self.documentation_url,
                    "documentation_version": self.documentation_version,
                },
                "recommendation": f"Validated {outputs_with_paths}/{total_outputs} outputs with property paths against SageMaker documentation",
            }
        )

        return issues

    def _get_valid_property_paths_for_step_type(
        self, step_type: str, node_type: str
    ) -> Dict[str, List[str]]:
        """
        Get valid property paths for a specific SageMaker step type.

        Based on SageMaker Property Path Reference Database:
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
        if "training" in step_type_lower or node_type_lower == "training":
            property_paths = {
                "model_artifacts": ["properties.ModelArtifacts.S3ModelArtifacts"],
                "output_config": [
                    "properties.OutputDataConfig.S3OutputPath",
                    "properties.OutputDataConfig.KmsKeyId",
                ],
                "metrics": [
                    # Support both named and wildcard access for metrics
                    "properties.FinalMetricDataList[*].Value",
                    "properties.FinalMetricDataList[*].MetricName",
                    "properties.FinalMetricDataList[*].Timestamp",
                ],
                "job_info": [
                    "properties.TrainingJobName",
                    "properties.TrainingJobArn",
                    "properties.TrainingJobStatus",
                    "properties.CreationTime",
                    "properties.TrainingStartTime",
                    "properties.TrainingEndTime",
                ],
                "algorithm": [
                    "properties.AlgorithmSpecification.TrainingImage",
                    "properties.AlgorithmSpecification.TrainingInputMode",
                ],
                "resources": [
                    "properties.ResourceConfig.InstanceType",
                    "properties.ResourceConfig.InstanceCount",
                    "properties.ResourceConfig.VolumeSizeInGB",
                ],
                "stopping_condition": [
                    "properties.StoppingCondition.MaxRuntimeInSeconds"
                ],
                "secondary_status": [
                    "properties.SecondaryStatus",
                    "properties.SecondaryStatusTransitions[*].Status",
                    "properties.SecondaryStatusTransitions[*].StartTime",
                ],
                "hyperparameters": ["properties.HyperParameters"],
            }

        # ProcessingStep - Properties from DescribeProcessingJob API
        elif "processing" in step_type_lower or node_type_lower == "processing":
            property_paths = {
                "outputs": [
                    # Support both named and indexed access
                    "properties.ProcessingOutputConfig.Outputs[*].S3Output.S3Uri",
                    "properties.ProcessingOutputConfig.Outputs[*].S3Output.LocalPath",
                    "properties.ProcessingOutputConfig.Outputs[*].S3Output.S3UploadMode",
                    "properties.ProcessingOutputConfig.Outputs[*].OutputName",
                ],
                "inputs": [
                    "properties.ProcessingInputs[*].S3Input.S3Uri",
                    "properties.ProcessingInputs[*].S3Input.LocalPath",
                    "properties.ProcessingInputs[*].InputName",
                ],
                "job_info": [
                    "properties.ProcessingJobName",
                    "properties.ProcessingJobArn",
                    "properties.ProcessingJobStatus",
                    "properties.CreationTime",
                    "properties.ProcessingStartTime",
                    "properties.ProcessingEndTime",
                ],
                "resources": [
                    "properties.ProcessingResources.ClusterConfig.InstanceType",
                    "properties.ProcessingResources.ClusterConfig.InstanceCount",
                    "properties.ProcessingResources.ClusterConfig.VolumeSizeInGB",
                ],
                "app_specification": [
                    "properties.AppSpecification.ImageUri",
                    "properties.AppSpecification.ContainerEntrypoint[*]",
                    "properties.AppSpecification.ContainerArguments[*]",
                ],
            }

        # TransformStep - Properties from DescribeTransformJob API
        elif "transform" in step_type_lower or node_type_lower == "transform":
            property_paths = {
                "outputs": [
                    "properties.TransformOutput.S3OutputPath",
                    "properties.TransformOutput.Accept",
                    "properties.TransformOutput.AssembleWith",
                    "properties.TransformOutput.KmsKeyId",
                ],
                "job_info": [
                    "properties.TransformJobName",
                    "properties.TransformJobArn",
                    "properties.TransformJobStatus",
                    "properties.CreationTime",
                    "properties.TransformStartTime",
                    "properties.TransformEndTime",
                ],
                "inputs": [
                    "properties.TransformInput.DataSource.S3DataSource.S3Uri",
                    "properties.TransformInput.ContentType",
                    "properties.TransformInput.CompressionType",
                    "properties.TransformInput.SplitType",
                ],
                "resources": [
                    "properties.TransformResources.InstanceType",
                    "properties.TransformResources.InstanceCount",
                ],
                "model": ["properties.ModelName"],
                "data_processing": [
                    "properties.DataProcessing.InputFilter",
                    "properties.DataProcessing.OutputFilter",
                    "properties.DataProcessing.JoinSource",
                ],
            }

        # TuningStep - Properties from DescribeHyperParameterTuningJob and ListTrainingJobsForHyperParameterTuningJob APIs
        elif "tuning" in step_type_lower or "hyperparameter" in step_type_lower:
            property_paths = {
                "best_training_job": [
                    "properties.BestTrainingJob.TrainingJobName",
                    "properties.BestTrainingJob.TrainingJobArn",
                    "properties.BestTrainingJob.TrainingJobStatus",
                    "properties.BestTrainingJob.CreationTime",
                    "properties.BestTrainingJob.TrainingStartTime",
                    "properties.BestTrainingJob.TrainingEndTime",
                    "properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.MetricName",
                    "properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value",
                ],
                "training_job_summaries": [
                    "properties.TrainingJobSummaries[*].TrainingJobName",
                    "properties.TrainingJobSummaries[*].TrainingJobArn",
                    "properties.TrainingJobSummaries[*].TrainingJobStatus",
                    "properties.TrainingJobSummaries[*].CreationTime",
                    "properties.TrainingJobSummaries[*].TrainingStartTime",
                    "properties.TrainingJobSummaries[*].TrainingEndTime",
                    "properties.TrainingJobSummaries[*].FinalHyperParameterTuningJobObjectiveMetric.MetricName",
                    "properties.TrainingJobSummaries[*].FinalHyperParameterTuningJobObjectiveMetric.Value",
                ],
                "job_info": [
                    "properties.HyperParameterTuningJobName",
                    "properties.HyperParameterTuningJobArn",
                    "properties.HyperParameterTuningJobStatus",
                    "properties.CreationTime",
                    "properties.HyperParameterTuningStartTime",
                    "properties.HyperParameterTuningEndTime",
                ],
                "tuning_config": [
                    "properties.HyperParameterTuningJobConfig.Strategy",
                    "properties.HyperParameterTuningJobConfig.HyperParameterTuningJobObjective.Type",
                    "properties.HyperParameterTuningJobConfig.HyperParameterTuningJobObjective.MetricName",
                ],
                "training_job_counts": [
                    "properties.TrainingJobStatusCounters.Completed",
                    "properties.TrainingJobStatusCounters.InProgress",
                    "properties.TrainingJobStatusCounters.RetryableError",
                    "properties.TrainingJobStatusCounters.NonRetryableError",
                    "properties.TrainingJobStatusCounters.Stopped",
                ],
            }

        # CreateModelStep - Properties from DescribeModel API
        elif "model" in step_type_lower and (
            "create" in step_type_lower or node_type_lower == "model"
        ):
            property_paths = {
                "model_info": [
                    "properties.ModelName",
                    "properties.ModelArn",
                    "properties.CreationTime",
                ],
                "primary_container": [
                    "properties.PrimaryContainer.Image",
                    "properties.PrimaryContainer.ModelDataUrl",
                    "properties.PrimaryContainer.Environment[*]",
                    "properties.PrimaryContainer.ContainerHostname",
                    "properties.PrimaryContainer.Mode",
                ],
                "multi_model_config": [
                    "properties.PrimaryContainer.MultiModelConfig.ModelCacheSetting"
                ],
                "containers": [
                    "properties.Containers[*].Image",
                    "properties.Containers[*].ModelDataUrl",
                    "properties.Containers[*].Environment[*]",
                    "properties.Containers[*].ContainerHostname",
                ],
                "inference_config": ["properties.InferenceExecutionConfig.Mode"],
                "vpc_config": [
                    "properties.VpcConfig.SecurityGroupIds[*]",
                    "properties.VpcConfig.Subnets[*]",
                ],
                "execution_role": ["properties.ExecutionRoleArn"],
                "network_isolation": ["properties.EnableNetworkIsolation"],
            }

        # LambdaStep - OutputParameters (no properties prefix)
        elif "lambda" in step_type_lower:
            property_paths = {"output_parameters": ["OutputParameters[*]"]}

        # CallbackStep - OutputParameters (no properties prefix)
        elif "callback" in step_type_lower:
            property_paths = {"output_parameters": ["OutputParameters[*]"]}

        # QualityCheckStep - Model Monitor Container Output
        elif "quality" in step_type_lower or "qualitycheck" in step_type_lower:
            property_paths = {
                "baseline_constraints": ["properties.CalculatedBaselineConstraints"],
                "baseline_statistics": ["properties.CalculatedBaselineStatistics"],
                "drift_check": [
                    "properties.BaselineUsedForDriftCheckStatistics",
                    "properties.BaselineUsedForDriftCheckConstraints",
                ],
            }

        # ClarifyCheckStep - Clarify Container Output
        elif "clarify" in step_type_lower:
            property_paths = {
                "baseline_constraints": ["properties.CalculatedBaselineConstraints"],
                "drift_check": ["properties.BaselineUsedForDriftCheckConstraints"],
            }

        # EMRStep - EMR Step Properties
        elif "emr" in step_type_lower:
            property_paths = {"cluster_info": ["properties.ClusterId"]}

        # Cache the result
        self._property_path_cache[cache_key] = property_paths

        return property_paths

    def _validate_single_property_path(
        self,
        property_path: str,
        step_type: str,
        node_type: str,
        valid_paths: Dict[str, List[str]],
    ) -> Dict[str, Any]:
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
                "valid": True,
                "error": None,
                "suggestions": all_valid_paths,
                "match_type": "exact",
            }

        # Check for pattern matches (e.g., array indexing)
        for valid_path in all_valid_paths:
            if self._matches_property_path_pattern(property_path, valid_path):
                return {
                    "valid": True,
                    "error": None,
                    "suggestions": all_valid_paths,
                    "match_type": "pattern",
                    "matched_pattern": valid_path,
                }

        # Check for partial matches to provide better suggestions
        suggestions = self._get_property_path_suggestions(
            property_path, all_valid_paths
        )

        return {
            "valid": False,
            "error": f'Property path "{property_path}" is not valid for step type "{step_type}"',
            "suggestions": suggestions,
            "match_type": "none",
        }

    def _matches_property_path_pattern(self, property_path: str, pattern: str) -> bool:
        """
        Check if a property path matches a pattern with wildcards.

        Supports multiple pattern types from the reference database:
        - Exact matches: properties.ModelArtifacts.S3ModelArtifacts
        - Wildcard array access: properties.FinalMetricDataList[*].Value
        - Named array access: properties.FinalMetricDataList['accuracy'].Value
        - Indexed array access: properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri

        Args:
            property_path: The actual property path
            pattern: The pattern to match against (may contain [*])

        Returns:
            True if the property path matches the pattern
        """
        # Direct exact match first
        if property_path == pattern:
            return True

        # Convert pattern to regex for advanced matching
        try:
            # Escape special regex characters except [*]
            escaped_pattern = re.escape(pattern)

            # Replace escaped [*] with regex patterns for different array access types:
            # 1. Named access: ['key_name'] or ["key_name"]
            # 2. Indexed access: [0], [1], [2], etc.
            # 3. Wildcard: [*] (original behavior)

            # Handle [*] -> match any array access pattern
            escaped_pattern = escaped_pattern.replace(
                r"\[\*\]",
                r'\[(?:[\'"][^\'\"]*[\'"]|\d+|\*)\]',  # Match ['key'], ["key"], [0], or [*]
            )

            # Create full regex pattern
            regex_pattern = f"^{escaped_pattern}$"

            return bool(re.match(regex_pattern, property_path))

        except re.error:
            # If regex compilation fails, fall back to simple string comparison
            return property_path == pattern

    def _get_property_path_suggestions(
        self, property_path: str, all_valid_paths: List[str]
    ) -> List[str]:
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
            score = self._calculate_path_similarity(
                property_path_lower, valid_path.lower()
            )
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
        components1 = path1.replace("[", ".").replace("]", ".").split(".")
        components2 = path2.replace("[", ".").replace("]", ".").split(".")

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

    def get_step_type_documentation(
        self, step_type: str, node_type: str = ""
    ) -> Dict[str, Any]:
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
            "step_type": step_type,
            "node_type": node_type,
            "documentation_url": self.documentation_url,
            "documentation_version": self.documentation_version,
            "valid_property_paths": valid_paths,
            "total_valid_paths": sum(len(paths) for paths in valid_paths.values()),
            "categories": list(valid_paths.keys()),
        }

    def list_supported_step_types(self) -> List[Dict[str, Any]]:
        """
        List all supported step types and their documentation.

        Returns:
            List of supported step types with their information
        """
        supported_types = [
            {
                "step_type": "training",
                "node_type": "training",
                "description": "TrainingStep - Properties from DescribeTrainingJob API",
            },
            {
                "step_type": "processing",
                "node_type": "processing",
                "description": "ProcessingStep - Properties from DescribeProcessingJob API",
            },
            {
                "step_type": "transform",
                "node_type": "transform",
                "description": "TransformStep - Properties from DescribeTransformJob API",
            },
            {
                "step_type": "tuning",
                "node_type": "tuning",
                "description": "TuningStep - Properties from DescribeHyperParameterTuningJob API",
            },
            {
                "step_type": "create_model",
                "node_type": "model",
                "description": "CreateModelStep - Properties from DescribeModel API",
            },
            {
                "step_type": "lambda",
                "node_type": "lambda",
                "description": "LambdaStep - OutputParameters",
            },
            {
                "step_type": "callback",
                "node_type": "callback",
                "description": "CallbackStep - OutputParameters",
            },
            {
                "step_type": "quality_check",
                "node_type": "quality",
                "description": "QualityCheckStep - Baseline and drift check properties",
            },
            {
                "step_type": "clarify_check",
                "node_type": "clarify",
                "description": "ClarifyCheckStep - Clarify-specific properties",
            },
            {
                "step_type": "emr",
                "node_type": "emr",
                "description": "EMRStep - EMR cluster properties",
            },
        ]

        # Add documentation info for each type
        for step_info in supported_types:
            doc_info = self.get_step_type_documentation(
                step_info["step_type"], step_info["node_type"]
            )
            step_info.update(doc_info)

        return supported_types


# Convenience function for easy import
def validate_property_paths(
    specification: Dict[str, Any], contract_name: str
) -> List[Dict[str, Any]]:
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
