---
tags:
  - design
  - sagemaker_integration
  - property_path_validation
  - reference_database
  - step_type_classification
keywords:
  - SageMaker property paths
  - step type validation
  - property reference patterns
  - validation database
  - SageMaker API integration
topics:
  - SageMaker property validation
  - property path patterns
  - step type reference
  - validation framework
language: python
date of note: 2025-08-12
---

# SageMaker Property Path Reference Database

## Related Documents
- **[Property Path Validation (Level 2) Implementation Plan](../2_project_planning/2025-08-12_property_path_validation_level2_implementation_plan.md)** - Complete implementation plan
- **[Level 2: Contract Specification Alignment Design](../1_design/level2_contract_specification_alignment_design.md)** - Current Level 2 validation system
- **[SageMaker Step Type Classification Design](../1_design/sagemaker_step_type_classification_design.md)** - Existing step type classification system for mapping step builders to SageMaker step types

## Overview

This document defines the comprehensive Property Path Reference Database based on the official [SageMaker Model Building Pipeline documentation](https://sagemaker.readthedocs.io/en/v2.92.2/amazon_sagemaker_model_building_pipeline.html#data-dependency-property-reference). The database provides validation patterns and reference data for all SageMaker step types to ensure property paths in specifications are valid for their respective step types.

## SageMaker Property Path Reference Database

### TrainingStep

**Source**: [DescribeTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTrainingJob.html#API_DescribeTrainingJob_ResponseSyntax)

```python
"TrainingStep": {
    "api_source": "DescribeTrainingJob",
    "description": "Training job properties from DescribeTrainingJob API",
    "valid_patterns": [
        # Model artifacts - primary output
        r"properties\.ModelArtifacts\.S3ModelArtifacts",
        
        # Training metrics - array access with metric name
        r"properties\.FinalMetricDataList\['.+'\]\.Value",
        r"properties\.FinalMetricDataList\['.+'\]\.Timestamp",
        
        # Training job metadata
        r"properties\.TrainingJobName",
        r"properties\.TrainingJobArn",
        r"properties\.TrainingJobStatus",
        r"properties\.CreationTime",
        r"properties\.TrainingStartTime",
        r"properties\.TrainingEndTime",
        
        # Output configuration
        r"properties\.OutputDataConfig\.S3OutputPath",
        r"properties\.OutputDataConfig\.KmsKeyId",
        
        # Algorithm specification
        r"properties\.AlgorithmSpecification\.TrainingImage",
        r"properties\.AlgorithmSpecification\.TrainingInputMode",
        
        # Resource configuration
        r"properties\.ResourceConfig\.InstanceType",
        r"properties\.ResourceConfig\.InstanceCount",
        r"properties\.ResourceConfig\.VolumeSizeInGB",
        
        # Stopping condition
        r"properties\.StoppingCondition\.MaxRuntimeInSeconds",
        
        # Secondary status
        r"properties\.SecondaryStatus",
        r"properties\.SecondaryStatusTransitions\[\d+\]\.Status",
        r"properties\.SecondaryStatusTransitions\[\d+\]\.StartTime"
    ],
    "common_outputs": {
        "model_artifacts": "properties.ModelArtifacts.S3ModelArtifacts",
        "training_output": "properties.OutputDataConfig.S3OutputPath",
        "training_job_name": "properties.TrainingJobName",
        "training_status": "properties.TrainingJobStatus",
        "accuracy_metric": "properties.FinalMetricDataList['accuracy'].Value",
        "loss_metric": "properties.FinalMetricDataList['loss'].Value",
        "val_accuracy": "properties.FinalMetricDataList['val:acc'].Value",
        "val_loss": "properties.FinalMetricDataList['val:loss'].Value"
    },
    "examples": [
        "properties.ModelArtifacts.S3ModelArtifacts",
        "properties.FinalMetricDataList['val:acc'].Value",
        "properties.OutputDataConfig.S3OutputPath",
        "properties.TrainingJobName"
    ]
}
```

### ProcessingStep

**Source**: [DescribeProcessingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeProcessingJob.html#API_DescribeProcessingJob_ResponseSyntax)

```python
"ProcessingStep": {
    "api_source": "DescribeProcessingJob",
    "description": "Processing job properties from DescribeProcessingJob API",
    "valid_patterns": [
        # Processing outputs - named access
        r"properties\.ProcessingOutputConfig\.Outputs\['.+'\]\.S3Output\.S3Uri",
        r"properties\.ProcessingOutputConfig\.Outputs\['.+'\]\.S3Output\.LocalPath",
        r"properties\.ProcessingOutputConfig\.Outputs\['.+'\]\.S3Output\.S3UploadMode",
        
        # Processing outputs - indexed access
        r"properties\.ProcessingOutputConfig\.Outputs\[\d+\]\.S3Output\.S3Uri",
        r"properties\.ProcessingOutputConfig\.Outputs\[\d+\]\.S3Output\.LocalPath",
        r"properties\.ProcessingOutputConfig\.Outputs\[\d+\]\.S3Output\.S3UploadMode",
        
        # Processing job metadata
        r"properties\.ProcessingJobName",
        r"properties\.ProcessingJobArn",
        r"properties\.ProcessingJobStatus",
        r"properties\.CreationTime",
        r"properties\.ProcessingStartTime",
        r"properties\.ProcessingEndTime",
        
        # Processing inputs
        r"properties\.ProcessingInputs\[\d+\]\.S3Input\.S3Uri",
        r"properties\.ProcessingInputs\[\d+\]\.S3Input\.LocalPath",
        
        # Resource configuration
        r"properties\.ProcessingResources\.ClusterConfig\.InstanceType",
        r"properties\.ProcessingResources\.ClusterConfig\.InstanceCount",
        r"properties\.ProcessingResources\.ClusterConfig\.VolumeSizeInGB",
        
        # App specification
        r"properties\.AppSpecification\.ImageUri",
        r"properties\.AppSpecification\.ContainerEntrypoint\[\d+\]",
        r"properties\.AppSpecification\.ContainerArguments\[\d+\]"
    ],
    "common_outputs": {
        "processing_output": "properties.ProcessingOutputConfig.Outputs['{output_name}'].S3Output.S3Uri",
        "indexed_output": "properties.ProcessingOutputConfig.Outputs[{index}].S3Output.S3Uri",
        "train_data": "properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri",
        "validation_data": "properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri",
        "test_data": "properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri",
        "processed_data": "properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
        "job_name": "properties.ProcessingJobName",
        "job_status": "properties.ProcessingJobStatus"
    },
    "examples": [
        "properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri",
        "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri",
        "properties.ProcessingJobName",
        "properties.ProcessingJobStatus"
    ]
}
```

### TransformStep

**Source**: [DescribeTransformJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTransformJob.html#API_DescribeTransformJob_ResponseSyntax)

```python
"TransformStep": {
    "api_source": "DescribeTransformJob",
    "description": "Transform job properties from DescribeTransformJob API",
    "valid_patterns": [
        # Transform output
        r"properties\.TransformOutput\.S3OutputPath",
        r"properties\.TransformOutput\.Accept",
        r"properties\.TransformOutput\.AssembleWith",
        r"properties\.TransformOutput\.KmsKeyId",
        
        # Transform job metadata
        r"properties\.TransformJobName",
        r"properties\.TransformJobArn",
        r"properties\.TransformJobStatus",
        r"properties\.CreationTime",
        r"properties\.TransformStartTime",
        r"properties\.TransformEndTime",
        
        # Transform input
        r"properties\.TransformInput\.DataSource\.S3DataSource\.S3Uri",
        r"properties\.TransformInput\.ContentType",
        r"properties\.TransformInput\.CompressionType",
        r"properties\.TransformInput\.SplitType",
        
        # Transform resources
        r"properties\.TransformResources\.InstanceType",
        r"properties\.TransformResources\.InstanceCount",
        
        # Model name
        r"properties\.ModelName",
        
        # Data processing
        r"properties\.DataProcessing\.InputFilter",
        r"properties\.DataProcessing\.OutputFilter",
        r"properties\.DataProcessing\.JoinSource"
    ],
    "common_outputs": {
        "transform_output": "properties.TransformOutput.S3OutputPath",
        "transform_job_name": "properties.TransformJobName",
        "transform_status": "properties.TransformJobStatus",
        "model_name": "properties.ModelName"
    },
    "examples": [
        "properties.TransformOutput.S3OutputPath",
        "properties.TransformJobName",
        "properties.TransformJobStatus"
    ]
}
```

### TuningStep

**Source**: [DescribeHyperParameterTuningJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeHyperParameterTuningJob.html#API_DescribeHyperParameterTuningJob_ResponseSyntax) and [ListTrainingJobsForHyperParameterTuningJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTrainingJobsForHyperParameterTuningJob.html#API_ListTrainingJobsForHyperParameterTuningJob_ResponseSyntax)

```python
"TuningStep": {
    "api_source": "DescribeHyperParameterTuningJob + ListTrainingJobsForHyperParameterTuningJob",
    "description": "Hyperparameter tuning job properties",
    "valid_patterns": [
        # Best training job (from DescribeHyperParameterTuningJob)
        r"properties\.BestTrainingJob\.TrainingJobName",
        r"properties\.BestTrainingJob\.TrainingJobArn",
        r"properties\.BestTrainingJob\.TrainingJobStatus",
        r"properties\.BestTrainingJob\.CreationTime",
        r"properties\.BestTrainingJob\.TrainingStartTime",
        r"properties\.BestTrainingJob\.TrainingEndTime",
        r"properties\.BestTrainingJob\.FinalHyperParameterTuningJobObjectiveMetric\.MetricName",
        r"properties\.BestTrainingJob\.FinalHyperParameterTuningJobObjectiveMetric\.Value",
        
        # Training job summaries (from ListTrainingJobsForHyperParameterTuningJob)
        r"properties\.TrainingJobSummaries\[\d+\]\.TrainingJobName",
        r"properties\.TrainingJobSummaries\[\d+\]\.TrainingJobArn",
        r"properties\.TrainingJobSummaries\[\d+\]\.TrainingJobStatus",
        r"properties\.TrainingJobSummaries\[\d+\]\.CreationTime",
        r"properties\.TrainingJobSummaries\[\d+\]\.TrainingStartTime",
        r"properties\.TrainingJobSummaries\[\d+\]\.TrainingEndTime",
        r"properties\.TrainingJobSummaries\[\d+\]\.FinalHyperParameterTuningJobObjectiveMetric\.MetricName",
        r"properties\.TrainingJobSummaries\[\d+\]\.FinalHyperParameterTuningJobObjectiveMetric\.Value",
        
        # Tuning job metadata
        r"properties\.HyperParameterTuningJobName",
        r"properties\.HyperParameterTuningJobArn",
        r"properties\.HyperParameterTuningJobStatus",
        r"properties\.CreationTime",
        r"properties\.HyperParameterTuningStartTime",
        r"properties\.HyperParameterTuningEndTime",
        
        # Tuning job config
        r"properties\.HyperParameterTuningJobConfig\.Strategy",
        r"properties\.HyperParameterTuningJobConfig\.HyperParameterTuningJobObjective\.Type",
        r"properties\.HyperParameterTuningJobConfig\.HyperParameterTuningJobObjective\.MetricName",
        
        # Training job counts
        r"properties\.TrainingJobStatusCounters\.Completed",
        r"properties\.TrainingJobStatusCounters\.InProgress",
        r"properties\.TrainingJobStatusCounters\.RetryableError",
        r"properties\.TrainingJobStatusCounters\.NonRetryableError",
        r"properties\.TrainingJobStatusCounters\.Stopped"
    ],
    "common_outputs": {
        "best_model": "properties.BestTrainingJob.TrainingJobName",
        "best_model_arn": "properties.BestTrainingJob.TrainingJobArn",
        "best_metric_value": "properties.BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value",
        "top_k_model": "properties.TrainingJobSummaries[{k}].TrainingJobName",
        "tuning_job_name": "properties.HyperParameterTuningJobName",
        "tuning_status": "properties.HyperParameterTuningJobStatus",
        "completed_jobs": "properties.TrainingJobStatusCounters.Completed"
    },
    "examples": [
        "properties.BestTrainingJob.TrainingJobName",
        "properties.TrainingJobSummaries[0].TrainingJobName",
        "properties.TrainingJobSummaries[1].TrainingJobName",
        "properties.HyperParameterTuningJobName"
    ]
}
```

### CreateModelStep

**Source**: [DescribeModel API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeModel.html#API_DescribeModel_ResponseSyntax)

```python
"CreateModelStep": {
    "api_source": "DescribeModel",
    "description": "Model creation properties from DescribeModel API",
    "valid_patterns": [
        # Model metadata
        r"properties\.ModelName",
        r"properties\.ModelArn",
        r"properties\.CreationTime",
        
        # Primary container
        r"properties\.PrimaryContainer\.Image",
        r"properties\.PrimaryContainer\.ModelDataUrl",
        r"properties\.PrimaryContainer\.Environment\['.+'\]",
        r"properties\.PrimaryContainer\.ContainerHostname",
        r"properties\.PrimaryContainer\.Mode",
        
        # Multi-model configuration
        r"properties\.PrimaryContainer\.MultiModelConfig\.ModelCacheSetting",
        
        # Containers (for multi-container models)
        r"properties\.Containers\[\d+\]\.Image",
        r"properties\.Containers\[\d+\]\.ModelDataUrl",
        r"properties\.Containers\[\d+\]\.Environment\['.+'\]",
        r"properties\.Containers\[\d+\]\.ContainerHostname",
        
        # Inference execution config
        r"properties\.InferenceExecutionConfig\.Mode",
        
        # VPC config
        r"properties\.VpcConfig\.SecurityGroupIds\[\d+\]",
        r"properties\.VpcConfig\.Subnets\[\d+\]",
        
        # Execution role
        r"properties\.ExecutionRoleArn",
        
        # Enable network isolation
        r"properties\.EnableNetworkIsolation"
    ],
    "common_outputs": {
        "model_name": "properties.ModelName",
        "model_arn": "properties.ModelArn",
        "model_data": "properties.PrimaryContainer.ModelDataUrl",
        "model_image": "properties.PrimaryContainer.Image",
        "execution_role": "properties.ExecutionRoleArn",
        "creation_time": "properties.CreationTime"
    },
    "examples": [
        "properties.ModelName",
        "properties.ModelArn",
        "properties.PrimaryContainer.ModelDataUrl",
        "properties.PrimaryContainer.Image"
    ]
}
```

### LambdaStep

**Source**: Lambda Step Output Parameters

```python
"LambdaStep": {
    "api_source": "Lambda Step Output Parameters",
    "description": "Lambda step output parameters",
    "valid_patterns": [
        # Output parameters - string access
        r"OutputParameters\['.+'\]"
    ],
    "common_outputs": {
        "lambda_output": "OutputParameters['{output_name}']",
        "string_output": "OutputParameters['output1']",
        "result_output": "OutputParameters['result']",
        "status_output": "OutputParameters['status']"
    },
    "examples": [
        "OutputParameters['output1']",
        "OutputParameters['result']",
        "OutputParameters['status']"
    ],
    "notes": [
        "Output parameters cannot be nested",
        "Nested values are treated as single string values",
        "Lambda function must return flat key-value pairs"
    ]
}
```

### CallbackStep

**Source**: Callback Step Output Parameters via [SendPipelineExecutionStepSuccess API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_SendPipelineExecutionStepSuccess.html)

```python
"CallbackStep": {
    "api_source": "SendPipelineExecutionStepSuccess",
    "description": "Callback step output parameters defined by SendPipelineExecutionStepSuccess call",
    "valid_patterns": [
        # Output parameters - string access
        r"OutputParameters\['.+'\]"
    ],
    "common_outputs": {
        "callback_output": "OutputParameters['{output_name}']",
        "approval_status": "OutputParameters['approval_status']",
        "review_result": "OutputParameters['review_result']",
        "callback_result": "OutputParameters['result']"
    },
    "examples": [
        "OutputParameters['approval_status']",
        "OutputParameters['review_result']",
        "OutputParameters['result']"
    ],
    "notes": [
        "Output parameters cannot be nested",
        "Nested values are treated as single string values",
        "Parameters defined by external callback via SendPipelineExecutionStepSuccess"
    ]
}
```

### QualityCheckStep

**Source**: Model Monitor Container Output

```python
"QualityCheckStep": {
    "api_source": "Model Monitor Container",
    "description": "Quality check step properties from Model Monitor container",
    "valid_patterns": [
        # Calculated baseline files
        r"properties\.CalculatedBaselineConstraints",
        r"properties\.CalculatedBaselineStatistics",
        
        # Baseline used for drift check
        r"properties\.BaselineUsedForDriftCheckStatistics",
        r"properties\.BaselineUsedForDriftCheckConstraints"
    ],
    "common_outputs": {
        "baseline_constraints": "properties.CalculatedBaselineConstraints",
        "baseline_statistics": "properties.CalculatedBaselineStatistics",
        "drift_check_statistics": "properties.BaselineUsedForDriftCheckStatistics",
        "drift_check_constraints": "properties.BaselineUsedForDriftCheckConstraints"
    },
    "examples": [
        "properties.CalculatedBaselineConstraints",
        "properties.CalculatedBaselineStatistics",
        "properties.BaselineUsedForDriftCheckStatistics"
    ]
}
```

### ClarifyCheckStep

**Source**: Clarify Container Output

```python
"ClarifyCheckStep": {
    "api_source": "Clarify Container",
    "description": "Clarify check step properties from Clarify container",
    "valid_patterns": [
        # Calculated baseline constraints
        r"properties\.CalculatedBaselineConstraints",
        
        # Baseline used for drift check
        r"properties\.BaselineUsedForDriftCheckConstraints"
    ],
    "common_outputs": {
        "baseline_constraints": "properties.CalculatedBaselineConstraints",
        "drift_check_constraints": "properties.BaselineUsedForDriftCheckConstraints"
    },
    "examples": [
        "properties.CalculatedBaselineConstraints",
        "properties.BaselineUsedForDriftCheckConstraints"
    ]
}
```

### EMRStep

**Source**: EMR Step Properties

```python
"EMRStep": {
    "api_source": "EMR Step Properties",
    "description": "EMR step properties",
    "valid_patterns": [
        # Cluster ID
        r"properties\.ClusterId"
    ],
    "common_outputs": {
        "cluster_id": "properties.ClusterId"
    },
    "examples": [
        "properties.ClusterId"
    ]
}
```

## Step Type to SageMaker Step Mapping

Based on the existing registry system in `src/cursus/steps/registry/step_names.py`, here's the mapping from our step types to SageMaker step types:

```python
STEP_TYPE_TO_SAGEMAKER_MAPPING = {
    # Processing-based steps
    "Processing": "ProcessingStep",
    "TabularPreprocessing": "ProcessingStep", 
    "RiskTableMapping": "ProcessingStep",
    "CurrencyConversion": "ProcessingStep",
    "XGBoostModelEval": "ProcessingStep",
    "ModelCalibration": "ProcessingStep",
    "Package": "ProcessingStep",
    "Payload": "ProcessingStep",
    "DummyTraining": "ProcessingStep",  # Uses pretrained model
    
    # Training-based steps
    "Training": "TrainingStep",
    "PyTorchTraining": "TrainingStep",
    "XGBoostTraining": "TrainingStep",
    
    # Transform-based steps
    "Transform": "TransformStep",
    "BatchTransform": "TransformStep",
    
    # Model creation steps
    "CreateModel": "CreateModelStep",
    "PyTorchModel": "CreateModelStep",
    "XGBoostModel": "CreateModelStep",
    
    # Special steps
    "Lambda": "LambdaStep",
    "HyperparameterPrep": "LambdaStep",
    
    # Custom steps (need special handling)
    "CradleDataLoading": "ProcessingStep",  # Custom processing
    "MimsModelRegistrationProcessing": "ProcessingStep",  # Custom processing
    
    # Base/Utility steps (no specific SageMaker equivalent)
    "Base": None,
    "Utility": None
}
```

## Validation Rules

### Pattern Matching Rules

1. **Exact Pattern Matching**: Property paths must match one of the valid regex patterns for the step type
2. **Case Sensitivity**: All property paths are case-sensitive
3. **Array Access**: Support both named (`['name']`) and indexed (`[0]`) array access
4. **Nested Properties**: Support deep property access with dot notation

### Common Validation Scenarios

```python
VALIDATION_SCENARIOS = {
    "valid_training_paths": [
        "properties.ModelArtifacts.S3ModelArtifacts",
        "properties.FinalMetricDataList['accuracy'].Value",
        "properties.OutputDataConfig.S3OutputPath"
    ],
    "invalid_training_paths": [
        "properties.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri",  # ProcessingStep pattern
        "properties.TransformOutput.S3OutputPath",  # TransformStep pattern
        "OutputParameters['result']"  # LambdaStep pattern
    ],
    "valid_processing_paths": [
        "properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri",
        "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri",
        "properties.ProcessingJobName"
    ],
    "invalid_processing_paths": [
        "properties.ModelArtifacts.S3ModelArtifacts",  # TrainingStep pattern
        "properties.TransformOutput.S3OutputPath",  # TransformStep pattern
        "properties.ModelName"  # CreateModelStep pattern
    ]
}
```

## Implementation Notes

### Pattern Compilation
- All regex patterns should be compiled for performance
- Use `re.IGNORECASE` flag where appropriate
- Cache compiled patterns for reuse

### Error Messages
- Provide specific error messages for each validation failure
- Include suggestions for correct patterns based on step type
- Reference official SageMaker documentation where applicable

### Extensibility
- Design database to easily add new step types
- Support for custom step types with user-defined patterns
- Version compatibility tracking for SageMaker SDK updates

## Usage Examples

### Valid Property Path Examples by Step Type

```python
# TrainingStep
"properties.ModelArtifacts.S3ModelArtifacts"
"properties.FinalMetricDataList['val:acc'].Value"
"properties.OutputDataConfig.S3OutputPath"

# ProcessingStep  
"properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
"properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
"properties.ProcessingJobName"

# TransformStep
"properties.TransformOutput.S3OutputPath"
"properties.TransformJobName"

# CreateModelStep
"properties.ModelName"
"properties.PrimaryContainer.ModelDataUrl"

# LambdaStep
"OutputParameters['result']"
"OutputParameters['status']"
```

### Common Validation Errors

```python
COMMON_ERRORS = {
    "wrong_step_type": {
        "error": "Property path 'properties.ModelArtifacts.S3ModelArtifacts' is invalid for ProcessingStep",
        "suggestion": "Use 'properties.ProcessingOutputConfig.Outputs['output_name'].S3Output.S3Uri' for ProcessingStep"
    },
    "invalid_syntax": {
        "error": "Property path 'properties.ProcessingOutputConfig.Outputs.train.S3Output.S3Uri' has invalid syntax",
        "suggestion": "Use bracket notation: 'properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri'"
    },
    "missing_properties_prefix": {
        "error": "Property path 'ModelArtifacts.S3ModelArtifacts' must start with 'properties.'",
        "suggestion": "Use 'properties.ModelArtifacts.S3ModelArtifacts'"
    }
}
```

## Conclusion

This Property Path Reference Database provides comprehensive validation patterns for all SageMaker step types based on official AWS documentation. It enables the Property Path Validation (Level 2) system to accurately validate property paths against their respective SageMaker step types, preventing runtime errors and ensuring compatibility with the SageMaker pipeline execution system.

The database is designed to be extensible, maintainable, and aligned with the official SageMaker API specifications, providing a solid foundation for property path validation across the entire codebase.

---

**Database Created**: August 12, 2025  
**Source**: SageMaker Python SDK v2.92.2 Documentation  
**Coverage**: 10 SageMaker step types with comprehensive property patterns  
**Integration**: Designed for Property Path Validation (Level 2) enhancement
