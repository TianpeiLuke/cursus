---
tags:
  - design
  - validation
  - sagemaker
  - step_types
  - requirements
keywords:
  - sagemaker step validation
  - step type requirements
  - builder validation
  - pipeline step requirements
  - validation specification
topics:
  - sagemaker step validation
  - step builder requirements
  - pipeline validation
  - step type specifications
language: python
date of note: 2025-10-01
implementation_status: SPECIFICATION
---

# SageMaker Step Validation Requirements Specification

## Executive Summary

This document provides the definitive specification for SageMaker step validation requirements based on the official SageMaker Python SDK documentation. It defines the **actual required fields and methods** for each SageMaker step type, eliminating overly strict validation patterns that don't reflect real SageMaker service requirements.

### Key Principles

1. **Service-Driven Validation**: Focus on what SageMaker service actually requires
2. **Builder-Centric Approach**: Validate what step builders need to function
3. **Practical Implementation**: Avoid theoretical patterns that don't affect pipeline execution
4. **Minimal Viable Validation**: Only validate what impacts step creation and execution

## SageMaker Step Types - Official Requirements

### **1. TrainingStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_TRAINING_REQUIREMENTS = {
    "role": "string",                      # ✅ VALIDATE: IAM role ARN format
    "image_uri": "string",                 # ✅ VALIDATE: Container image URI format
    "instance_type": "string",             # ✅ VALIDATE: Valid EC2 instance type
    "instance_count": "integer",           # ✅ VALIDATE: Positive integer
    "volume_size_in_gb": "integer",        # ✅ VALIDATE: Positive integer
    "max_runtime_in_seconds": "integer",   # ✅ VALIDATE: Positive integer
    "input_data_config": [                 # ✅ VALIDATE: S3 URI format
        {
            "ChannelName": "string",
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": "string"      # User must provide S3 path
                }
            }
        }
    ],
    "output_data_config": {                # ✅ VALIDATE: S3 URI format
        "S3OutputPath": "string"           # User must provide S3 output path
    }
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_TRAINING_PROPERTIES = {
    "TrainingJobName": "auto-generated",   # ❌ DON'T VALIDATE: SDK generates unique name
    "TrainingJobArn": "auto-generated",    # ❌ DON'T VALIDATE: Service provides ARN
    "TrainingJobStatus": "auto-generated", # ❌ DON'T VALIDATE: Service manages status
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "TrainingStartTime": "auto-generated", # ❌ DON'T VALIDATE: Service timestamp
    "TrainingEndTime": "auto-generated",   # ❌ DON'T VALIDATE: Service timestamp
    "ModelArtifacts": {                    # ❌ DON'T VALIDATE: Service generates
        "S3ModelArtifacts": "auto-generated"
    },
    "AlgorithmSpecification": {            # ❌ DON'T VALIDATE: SDK constructs from user input
        "TrainingImage": "derived-from-image_uri",
        "TrainingInputMode": "File"        # Default value
    }
}
```

#### **🔧 Required Builder Methods**
```python
class TrainingStepBuilder:
    def __init__(self, config: TrainingConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "role",                        # User must provide
            "image_uri",                   # User must provide
            "instance_type",               # User must provide
            "input_data_config",           # User must provide
            "output_data_config"           # User must provide
        ]
    
    def _create_estimator(self) -> Estimator:
        """✅ VALIDATE: This method must exist."""
        pass
    
    def create_step(self, name: str, inputs: Dict) -> TrainingStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **2. ProcessingStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_PROCESSING_REQUIREMENTS = {
    "role": "string",                      # ✅ VALIDATE: IAM role ARN format
    "image_uri": "string",                 # ✅ VALIDATE: Container image URI format
    "instance_type": "string",             # ✅ VALIDATE: Valid EC2 instance type
    "instance_count": "integer",           # ✅ VALIDATE: Positive integer
    "volume_size_in_gb": "integer",        # ✅ VALIDATE: Positive integer
    "max_runtime_in_seconds": "integer"    # ✅ VALIDATE: Positive integer
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_PROCESSING_PROPERTIES = {
    "ProcessingJobName": "auto-generated", # ❌ DON'T VALIDATE: SDK generates unique name
    "ProcessingJobArn": "auto-generated",  # ❌ DON'T VALIDATE: Service provides ARN
    "ProcessingJobStatus": "auto-generated", # ❌ DON'T VALIDATE: Service manages status
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "ProcessingStartTime": "auto-generated", # ❌ DON'T VALIDATE: Service timestamp
    "ProcessingEndTime": "auto-generated", # ❌ DON'T VALIDATE: Service timestamp
    "AppSpecification": {                  # ❌ DON'T VALIDATE: SDK constructs from user input
        "ImageUri": "derived-from-image_uri"
    }
}
```

#### **🔧 Required Builder Methods**
```python
class ProcessingStepBuilder:
    def __init__(self, config: ProcessingConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "role",                        # User must provide
            "image_uri",                   # User must provide
            "instance_type"                # User must provide
        ]
    
    def _create_processor(self) -> Processor:
        """✅ VALIDATE: This method must exist."""
        pass
    
    def create_step(self, name: str, inputs: List, outputs: List) -> ProcessingStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **3. CreateModelStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_CREATE_MODEL_REQUIREMENTS = {
    "role": "string",                      # ✅ VALIDATE: IAM role ARN format
    "model_data": "string",                # ✅ VALIDATE: S3 URI format to model artifacts
    "image_uri": "string"                  # ✅ VALIDATE: Container image URI format
    # OR (alternative)
    # "model_package_name": "string"       # ✅ VALIDATE: Model package ARN format
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_CREATE_MODEL_PROPERTIES = {
    "ModelName": "auto-generated",         # ❌ DON'T VALIDATE: SDK generates unique name
    "ModelArn": "auto-generated",          # ❌ DON'T VALIDATE: Service provides ARN
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "PrimaryContainer": {                  # ❌ DON'T VALIDATE: SDK constructs from user input
        "Image": "derived-from-image_uri",
        "ModelDataUrl": "derived-from-model_data"
    }
}
```

#### **🔧 Required Builder Methods**
```python
class CreateModelStepBuilder:
    def __init__(self, config: CreateModelConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "role",                        # User must provide
            "model_data",                  # User must provide (OR model_package_name)
            "image_uri"                    # User must provide (OR model_package_name)
        ]
    
    def _create_model(self) -> Model:
        """✅ VALIDATE: This method must exist."""
        pass
    
    def create_step(self, name: str) -> CreateModelStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **4. TransformStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_TRANSFORM_REQUIREMENTS = {
    "model_name": "string",                # ✅ VALIDATE: Reference to existing model
    "instance_type": "string",             # ✅ VALIDATE: Valid EC2 instance type
    "instance_count": "integer",           # ✅ VALIDATE: Positive integer
    "transform_input": {                   # ✅ VALIDATE: Input data configuration
        "DataSource": {
            "S3DataSource": {
                "S3Uri": "string"          # User must provide S3 input path
            }
        },
        "ContentType": "string"            # User must specify content type
    },
    "transform_output": {                  # ✅ VALIDATE: Output configuration
        "S3OutputPath": "string"           # User must provide S3 output path
    }
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_TRANSFORM_PROPERTIES = {
    "TransformJobName": "auto-generated",  # ❌ DON'T VALIDATE: SDK generates unique name
    "TransformJobArn": "auto-generated",   # ❌ DON'T VALIDATE: Service provides ARN
    "TransformJobStatus": "auto-generated", # ❌ DON'T VALIDATE: Service manages status
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "TransformStartTime": "auto-generated", # ❌ DON'T VALIDATE: Service timestamp
    "TransformEndTime": "auto-generated"   # ❌ DON'T VALIDATE: Service timestamp
}
```

#### **🔧 Required Builder Methods**
```python
class TransformStepBuilder:
    def __init__(self, config: TransformConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "model_name",                  # User must provide
            "instance_type",               # User must provide
            "transform_input",             # User must provide
            "transform_output"             # User must provide
        ]
    
    def _create_transformer(self) -> Transformer:
        """✅ VALIDATE: This method must exist."""
        pass
    
    def create_step(self, name: str) -> TransformStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **5. TuningStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_TUNING_REQUIREMENTS = {
    "estimator": "Estimator",              # ✅ VALIDATE: Base estimator object exists
    "hyperparameter_ranges": "dict",       # ✅ VALIDATE: Parameter ranges defined
    "metric_definitions": "list",          # ✅ VALIDATE: Metrics for optimization
    "objective_metric_name": "string",     # ✅ VALIDATE: Primary metric name
    "objective_type": "Maximize|Minimize", # ✅ VALIDATE: Valid optimization direction
    "max_jobs": "integer",                 # ✅ VALIDATE: Positive integer
    "max_parallel_jobs": "integer"         # ✅ VALIDATE: Positive integer
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_TUNING_PROPERTIES = {
    "HyperParameterTuningJobName": "auto-generated", # ❌ DON'T VALIDATE: SDK generates
    "HyperParameterTuningJobArn": "auto-generated",  # ❌ DON'T VALIDATE: Service provides
    "HyperParameterTuningJobStatus": "auto-generated", # ❌ DON'T VALIDATE: Service manages
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "HyperParameterTuningEndTime": "auto-generated", # ❌ DON'T VALIDATE: Service timestamp
    "BestTrainingJob": "auto-generated",   # ❌ DON'T VALIDATE: Service determines best job
    "TrainingJobStatusCounters": "auto-generated" # ❌ DON'T VALIDATE: Service tracks counters
}
```

#### **🔧 Required Builder Methods**
```python
class TuningStepBuilder:
    def __init__(self, config: TuningConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "estimator",                   # User must provide
            "hyperparameter_ranges",       # User must provide
            "objective_metric_name",       # User must provide
            "max_jobs"                     # User must provide
        ]
    
    def _create_tuner(self) -> HyperparameterTuner:
        """✅ VALIDATE: This method must exist."""
        pass
    
    def create_step(self, name: str, inputs: Dict) -> TuningStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **6. RegisterModel (StepCollection)**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_REGISTER_MODEL_REQUIREMENTS = {
    "model_package_group_name": "string",  # ✅ VALIDATE: Model package group name
    "content_types": ["string"],           # ✅ VALIDATE: Supported input content types
    "response_types": ["string"],          # ✅ VALIDATE: Supported output content types
    "inference_instances": ["string"],     # ✅ VALIDATE: Supported inference instance types
    "transform_instances": ["string"]      # ✅ VALIDATE: Supported transform instance types
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_REGISTER_MODEL_PROPERTIES = {
    "ModelPackageName": "auto-generated",  # ❌ DON'T VALIDATE: Service generates unique name
    "ModelPackageArn": "auto-generated",   # ❌ DON'T VALIDATE: Service provides ARN
    "ModelPackageStatus": "auto-generated", # ❌ DON'T VALIDATE: Service manages status
    "CreationTime": "auto-generated",      # ❌ DON'T VALIDATE: Service timestamp
    "ModelPackageVersion": "auto-generated", # ❌ DON'T VALIDATE: Service manages versioning
    "ModelApprovalStatus": "PendingManualApproval" # ❌ DON'T VALIDATE: Default value
}
```

#### **🔧 Required Builder Methods**
```python
class RegisterModelStepBuilder:
    def __init__(self, config: RegisterModelConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "model_package_group_name",    # User must provide
            "content_types",               # User must provide
            "response_types"               # User must provide
        ]
    
    def create_step_collection(self, name: str) -> RegisterModel:
        """✅ VALIDATE: This method must exist."""
        pass
```

### **7. ConditionStep**

#### **🔵 User-Provided Requirements (Must Validate)**
```python
# What users MUST provide in their step builder configuration
USER_PROVIDED_CONDITION_REQUIREMENTS = {
    "conditions": [                        # ✅ VALIDATE: List of condition objects
        {
            "Type": "Equals|GreaterThan|LessThan|In|Not|Or",
            "LeftValue": "string|number|boolean",
            "RightValue": "string|number|boolean"
        }
    ],
    "if_steps": ["Step"],                  # ✅ VALIDATE: Steps to execute if true
    "else_steps": ["Step"]                 # ✅ VALIDATE: Steps to execute if false
}
```

#### **🟢 SageMaker-Generated Properties (Do NOT Validate)**
```python
# What SageMaker automatically provides - DO NOT VALIDATE THESE
SAGEMAKER_GENERATED_CONDITION_PROPERTIES = {
    "Name": "provided-by-user-or-generated", # ❌ DON'T VALIDATE: Step name handling
    "ExecutionStatus": "auto-generated",   # ❌ DON'T VALIDATE: Service manages execution
    "ConditionOutcome": "auto-generated"   # ❌ DON'T VALIDATE: Service evaluates conditions
}
```

#### **🔧 Required Builder Methods**
```python
class ConditionStepBuilder:
    def __init__(self, config: ConditionConfig):
        # ✅ VALIDATE: These fields must exist in config
        self.required_config_fields = [
            "conditions",                  # User must provide
            "if_steps",                    # User must provide
            "else_steps"                   # User must provide
        ]
    
    def create_step(self, name: str) -> ConditionStep:
        """✅ VALIDATE: This method must exist."""
        pass
```

## Validation Requirements Matrix

### **Level 1: Service Compatibility Validation**

| Step Type | Required Fields | Required Methods | Critical Validations |
|-----------|----------------|------------------|---------------------|
| **TrainingStep** | `role`, `image_uri`, `instance_type`, `input_data_config`, `output_data_config` | `_create_estimator()`, `create_step()` | Role ARN format, S3 paths, instance type validity |
| **ProcessingStep** | `role`, `image_uri`, `instance_type` | `_create_processor()`, `create_step()` | Role ARN format, container image URI |
| **CreateModelStep** | `role`, `model_data`, `image_uri` | `_create_model()`, `create_step()` | Role ARN format, S3 model path, container image |
| **TransformStep** | `model_name`, `instance_type`, `transform_input`, `transform_output` | `_create_transformer()`, `create_step()` | Model reference, S3 paths, instance type |
| **TuningStep** | `estimator`, `hyperparameter_ranges`, `metric_definitions`, `max_jobs` | `_create_tuner()`, `create_step()` | Parameter ranges, metric definitions |
| **RegisterModel** | `model_package_group_name`, `content_types`, `response_types` | `create_step_collection()` | Content type formats, model package group |
| **ConditionStep** | `conditions`, `if_steps`, `else_steps` | `create_step()` | Condition syntax, step references |

### **Level 2: Builder Configuration Validation**

```python
# Builder Configuration Requirements
BUILDER_VALIDATION_REQUIREMENTS = {
    "required_methods": [
        "create_step",                     # All builders must implement
        "_create_{component}"              # Component-specific creation method
    ],
    "required_config_fields": {
        # Varies by step type - see individual specifications above
    },
    "optional_validations": [
        "vpc_config",                      # VPC configuration if specified
        "tags",                            # Tag format validation
        "environment",                     # Environment variable format
        "retry_policies"                   # Retry policy configuration
    ]
}
```

### **Level 3: Pipeline Integration Validation**

```python
# Pipeline Integration Requirements
PIPELINE_INTEGRATION_REQUIREMENTS = {
    "step_dependencies": {
        "depends_on": "List[Union[str, Step, StepCollection]]",  # Step dependencies
        "step_outputs": "Properties",      # Output property references
        "parameter_references": "Parameter" # Pipeline parameter usage
    },
    "naming_conventions": {
        "step_name": "^[a-zA-Z0-9\\-]{1,63}$",  # SageMaker naming rules
        "unique_names": True               # Names must be unique in pipeline
    }
}
```

## What NOT to Validate

### **❌ Overly Strict Validations to Remove**

#### **1. Script Implementation Details**
```python
# DON'T VALIDATE - These are implementation details
AVOID_VALIDATING = {
    "script_patterns": [
        "specific_function_names",         # model_fn, predict_fn are conventions
        "import_statements",               # User's choice of libraries
        "file_organization",               # How user organizes code
        "framework_specific_code"          # XGBoost vs PyTorch patterns
    ],
    "container_internals": [
        "inference_code_structure",       # Handled by containers
        "model_serialization_format",     # Framework-specific
        "file_paths_inside_container"      # Container implementation detail
    ],
    "theoretical_requirements": [
        "best_practice_patterns",         # Nice-to-have, not required
        "performance_optimizations",      # User optimization choice
        "code_style_conventions"          # Development preference
    ]
}
```

#### **2. Framework-Specific Code Patterns**
```python
# DON'T VALIDATE - Framework choice is user's decision
FRAMEWORK_AGNOSTIC_APPROACH = {
    "xgboost": {
        "avoid": ["xgb.Booster patterns", "DMatrix usage", "specific imports"],
        "validate": ["model_data S3 path", "container image URI"]
    },
    "pytorch": {
        "avoid": ["torch.load patterns", "state_dict usage", "model.eval()"],
        "validate": ["model_data S3 path", "container image URI"]
    },
    "sklearn": {
        "avoid": ["joblib.load patterns", "pickle usage", "fit/predict"],
        "validate": ["model_data S3 path", "container image URI"]
    }
}
```

## Implementation Strategy

### **Phase 1: Service Requirements Validation**
```python
# Focus on SageMaker service compatibility
class SageMakerServiceValidator:
    def validate_training_step(self, config: Dict) -> List[ValidationIssue]:
        """Validate against actual SageMaker Training Job requirements."""
        issues = []
        
        # Required fields only
        required_fields = ["role", "image_uri", "instance_type", "input_data_config", "output_data_config"]
        for field in required_fields:
            if field not in config:
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="missing_required_field",
                    message=f"Missing required field: {field}",
                    recommendation=f"Add {field} to step configuration"
                ))
        
        # Format validation for critical fields
        if "role" in config:
            if not self._is_valid_arn(config["role"]):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="invalid_arn_format",
                    message="Invalid IAM role ARN format",
                    recommendation="Provide valid IAM role ARN"
                ))
        
        return issues
```

### **Phase 2: Builder Method Validation**
```python
# Validate builder implementation
class BuilderMethodValidator:
    def validate_builder_methods(self, builder_class: type) -> List[ValidationIssue]:
        """Validate required builder methods exist."""
        issues = []
        
        required_methods = ["create_step"]
        for method in required_methods:
            if not hasattr(builder_class, method):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="missing_builder_method",
                    message=f"Builder missing required method: {method}",
                    recommendation=f"Implement {method} method in builder class"
                ))
        
        return issues
```

### **Phase 3: Pipeline Integration Validation**
```python
# Validate pipeline-level requirements
class PipelineIntegrationValidator:
    def validate_step_integration(self, step: Step, pipeline_context: Dict) -> List[ValidationIssue]:
        """Validate step integration with pipeline."""
        issues = []
        
        # Step name uniqueness
        if step.name in pipeline_context.get("existing_step_names", []):
            issues.append(ValidationIssue(
                severity="ERROR",
                category="duplicate_step_name",
                message=f"Step name '{step.name}' already exists in pipeline",
                recommendation="Use unique step name"
            ))
        
        # Dependency validation
        for dep in step.depends_on or []:
            if isinstance(dep, str) and dep not in pipeline_context.get("available_steps", []):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="invalid_dependency",
                    message=f"Step depends on non-existent step: {dep}",
                    recommendation="Ensure dependent step exists in pipeline"
                ))
        
        return issues
```

## Validation Configuration

### **Minimal Validation Configuration**
```python
# Practical validation configuration
STEP_VALIDATION_CONFIG = {
    "validation_levels": {
        "service_compatibility": {
            "enabled": True,
            "severity": "ERROR",
            "description": "Validate against SageMaker service requirements"
        },
        "builder_methods": {
            "enabled": True,
            "severity": "ERROR", 
            "description": "Validate required builder methods exist"
        },
        "pipeline_integration": {
            "enabled": True,
            "severity": "WARNING",
            "description": "Validate pipeline-level integration"
        },
        "script_patterns": {
            "enabled": False,  # DISABLED - too strict
            "severity": "INFO",
            "description": "Script implementation patterns (disabled)"
        },
        "framework_specific": {
            "enabled": False,  # DISABLED - user choice
            "severity": "INFO", 
            "description": "Framework-specific patterns (disabled)"
        }
    },
    "validation_modes": {
        "strict": ["service_compatibility", "builder_methods", "pipeline_integration"],
        "relaxed": ["service_compatibility", "builder_methods"],
        "minimal": ["service_compatibility"]
    }
}
```

## Usage Examples

### **Example 1: Training Step Validation**
```python
# Valid training step configuration
training_config = {
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-training:latest",
    "instance_type": "ml.m5.large",
    "instance_count": 1,
    "volume_size_in_gb": 30,
    "max_runtime_in_seconds": 3600,
    "input_data_config": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://my-bucket/training-data/"
                }
            }
        }
    ],
    "output_data_config": {
        "S3OutputPath": "s3://my-bucket/model-output/"
    }
}

# This passes validation - has all required fields
validator = SageMakerServiceValidator()
issues = validator.validate_training_step(training_config)
assert len(issues) == 0  # No validation errors
```

### **Example 2: CreateModel Step Validation**
```python
# Valid create model configuration
create_model_config = {
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-inference:latest",
    "model_data": "s3://my-bucket/model-artifacts/model.tar.gz"
}

# This passes validation - has all required fields
validator = SageMakerServiceValidator()
issues = validator.validate_create_model_step(create_model_config)
assert len(issues) == 0  # No validation errors

# Note: We DON'T validate inference code patterns, model_fn, etc.
# Those are implementation details handled by the container
```

## Migration Guide

### **From Current Validation to New Approach**

#### **Before (Overly Strict)**
```python
# Old validation - too strict
def validate_xgboost_training_script(script_content: str):
    issues = []
    
    # ❌ TOO STRICT - Don't validate implementation details
    if "xgb.train" not in script_content:
        issues.append("Missing xgb.train call")
    
    if "model.save_model" not in script_content:
        issues.append("Missing model save call")
    
    if "/opt/ml/model" not in script_content:
        issues.append("Missing SageMaker model path")
    
    return issues
```

#### **After (Service-Focused)**
```python
# New validation - service requirements only
def validate_training_step_config(config: Dict):
    issues = []
    
    # ✅ CORRECT - Validate service requirements only
    required_fields = ["role", "image_uri", "instance_type", "input_data_config", "output_data_config"]
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    # ✅ CORRECT - Validate format of critical fields
    if "role" in config and not config["role"].startswith("arn:aws:iam::"):
        issues.append("Invalid IAM role ARN format")
    
    return issues
```

## Conclusion

This specification provides a **practical, service-driven approach** to SageMaker step validation that:

1. **Focuses on actual SageMaker service requirements** rather than theoretical best practices
2. **Validates what step builders need** to create functional pipeline steps
3. **Eliminates overly strict validations** that don't affect pipeline execution
4. **Provides clear implementation guidance** for validation systems

By following this specification, validation systems can ensure SageMaker pipeline steps will work correctly while avoiding unnecessary restrictions on user implementation choices.

## References

- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html)
- [AWS SageMaker API Reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/)
- [SageMaker Pipeline Steps Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html)
