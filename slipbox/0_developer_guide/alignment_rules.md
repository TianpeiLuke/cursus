# Alignment Rules

This document centralizes all alignment guidance for pipeline step development. Refer to this file whenever you need to ensure consistency across script contracts, step specifications, and step builders.

## Alignment Principles

1. **Script ↔ Contract**  
   - Scripts must use exactly the paths defined in their Script Contract.  
   - Environment variable names, input/output directory structures, and file patterns must match the contract.
   - **Argument Naming Convention**: Contract arguments use CLI-style hyphens, scripts use Python-style underscores (standard argparse behavior).

2. **Contract ↔ Specification**  
   - Logical names in the Script Contract (`expected_input_paths`, `expected_output_paths`) must match dependency names in the Step Specification.  
   - Property paths in `OutputSpec` must correspond to the contract’s output paths.

3. **Specification ↔ Dependencies**  
   - Dependencies declared in the Step Specification must match upstream step outputs by logical name or alias.  
   - `DependencySpec.compatible_sources` must list all steps that produce the required output.

4. **Specification ↔ SageMaker Property Paths**  
   - Property paths in `OutputSpec` must be valid for the corresponding SageMaker step type.  
   - Property paths must follow SageMaker API patterns as defined in the [SageMaker Property Path Reference Database](sagemaker_property_path_reference_database.md).  
   - Step type classification must align with SageMaker step types (TrainingStep, ProcessingStep, TransformStep, etc.).

5. **Builder ↔ Configuration**  
   - Step Builders must pass configuration parameters to SageMaker components according to the config class.  
   - Environment variables set in the builder (`_get_processor_env_vars`) must cover all `required_env_vars` from the contract.

## Examples

### Script ↔ Contract

#### Argument Naming Convention (Argparse Pattern)

**Standard Pattern**: Contract arguments use CLI-style hyphens, scripts use Python-style underscores.

```python
# Contract Declaration (CLI convention)
"arguments": {
    "job-type": {"required": true},
    "marketplace-id-col": {"required": false},
    "default-currency": {"required": false}
}

# Script Implementation (Python convention)
parser.add_argument("--job-type", required=True)
parser.add_argument("--marketplace-id-col", required=False)
parser.add_argument("--default-currency", required=False)

# Script Usage (automatic argparse conversion)
args.job_type  # argparse converts job-type → job_type
args.marketplace_id_col  # argparse converts marketplace-id-col → marketplace_id_col
args.default_currency  # argparse converts default-currency → default_currency
```

**Validation Rule**: Arguments are considered aligned when contract hyphens match script underscores after normalization.

#### Path Usage

```python
from ...core.base.contract_base import ScriptContract

# Script path must match contract exactly
input_path = TABULAR_PREPROCESS_CONTRACT.expected_input_paths["DATA"] + "/file.csv"
assert "/opt/ml/processing/input/data" in input_path
```

### Contract ↔ Specification

```python
from ...core.base.specification_base import StepSpecification, DependencySpec

spec = StepSpecification(
    step_type="TabularPreprocessing",
    dependencies={
        "DATA": DependencySpec(
            logical_name="DATA",
            compatible_sources=["CradleDataLoading"]
        )
    },
    outputs={}
)
assert "DATA" in spec.dependencies
```

### Specification ↔ Dependencies

```python
from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec
from ...core.deps.dependency_resolver import UnifiedDependencyResolver

# Define a spec with dependency and output
spec = StepSpecification(
    step_type="XGBoostTraining",
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            compatible_sources=["TabularPreprocessing"]
        )
    },
    outputs={
        "training_data": OutputSpec(
            logical_name="training_data",
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
        )
    }
)
# Resolve using a registry and available steps list
resolver = UnifiedDependencyResolver(spec_registry)
matches = resolver.find_compatible_sources(
    spec.dependencies["training_data"],
    available_steps
)
assert any(m.step.step_type == "TabularPreprocessing" for m in matches)
```

### Specification ↔ SageMaker Property Paths

**Reference**: [SageMaker Property Path Reference Database](sagemaker_property_path_reference_database.md)

#### Valid Property Path Examples by Step Type

```python
# TrainingStep - Valid property paths
training_spec = StepSpecification(
    step_type="PyTorchTraining",  # Maps to SageMaker TrainingStep
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            property_path="properties.ModelArtifacts.S3ModelArtifacts"  # ✅ Valid for TrainingStep
        ),
        "training_metrics": OutputSpec(
            logical_name="training_metrics", 
            property_path="properties.FinalMetricDataList['val:acc'].Value"  # ✅ Valid for TrainingStep
        )
    }
)

# ProcessingStep - Valid property paths
processing_spec = StepSpecification(
    step_type="TabularPreprocessing",  # Maps to SageMaker ProcessingStep
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"  # ✅ Valid for ProcessingStep
        ),
        "train_data": OutputSpec(
            logical_name="train_data",
            property_path="properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri"  # ✅ Valid for ProcessingStep
        )
    }
)

# TransformStep - Valid property paths
transform_spec = StepSpecification(
    step_type="BatchTransform",  # Maps to SageMaker TransformStep
    outputs={
        "transform_output": OutputSpec(
            logical_name="transform_output",
            property_path="properties.TransformOutput.S3OutputPath"  # ✅ Valid for TransformStep
        )
    }
)

# CreateModelStep - Valid property paths
model_spec = StepSpecification(
    step_type="PyTorchModel",  # Maps to SageMaker CreateModelStep
    outputs={
        "model_name": OutputSpec(
            logical_name="model_name",
            property_path="properties.ModelName"  # ✅ Valid for CreateModelStep
        )
    }
)
```

#### Invalid Property Path Examples (Common Mistakes)

```python
# ❌ INVALID: Using ProcessingStep property path for TrainingStep
invalid_training_spec = StepSpecification(
    step_type="PyTorchTraining",  # Maps to SageMaker TrainingStep
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            property_path="properties.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri"  # ❌ ProcessingStep pattern
        )
    }
)

# ❌ INVALID: Using TrainingStep property path for ProcessingStep
invalid_processing_spec = StepSpecification(
    step_type="TabularPreprocessing",  # Maps to SageMaker ProcessingStep
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.ModelArtifacts.S3ModelArtifacts"  # ❌ TrainingStep pattern
        )
    }
)

# ❌ INVALID: Missing 'properties.' prefix
invalid_prefix_spec = StepSpecification(
    step_type="PyTorchTraining",
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            property_path="ModelArtifacts.S3ModelArtifacts"  # ❌ Missing 'properties.' prefix
        )
    }
)

# ❌ INVALID: Incorrect array access syntax
invalid_syntax_spec = StepSpecification(
    step_type="TabularPreprocessing",
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.ProcessingOutputConfig.Outputs.train.S3Output.S3Uri"  # ❌ Should use ['train']
        )
    }
)
```

#### Property Path Validation Rules

```python
from ...validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

# Validate property paths against SageMaker step types using the unified alignment tester
tester = UnifiedAlignmentTester()

# This will pass validation
valid_spec = StepSpecification(
    step_type="PyTorchTraining",
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    }
)
issues = tester.validate_specification_property_paths(valid_spec)
assert not any(issue.severity == "ERROR" for issue in issues)

# This will fail validation with helpful suggestions
invalid_spec = StepSpecification(
    step_type="PyTorchTraining",
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts", 
            property_path="properties.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri"
        )
    }
)
issues = tester.validate_specification_property_paths(invalid_spec)
error_issues = [issue for issue in issues if issue.severity == "ERROR"]
assert len(error_issues) > 0, "Should have validation errors for incorrect property path"
```

### Builder ↔ Configuration

```python
from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from ...steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT

builder = TabularPreprocessingStepBuilder(config_instance)
env_vars = builder._get_environment_variables()
for var in TABULAR_PREPROCESS_CONTRACT.required_env_vars:
    assert var in env_vars
```

### Real-World Examples

#### Script Implementation Example

```python
# nlp-pipeline/dockers/xgboost_atoz/pipeline_scripts/tabular_preprocess.py
from ...steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT

def main():
    # Use contract paths
    input_dir = TABULAR_PREPROCESS_CONTRACT.expected_input_paths["DATA"]
    output_dir = TABULAR_PREPROCESS_CONTRACT.expected_output_paths["processed_data"]
    # e.g. "/opt/ml/processing/input/data" and "/opt/ml/processing/output"
    print(f"Reading from {input_dir}, writing to {output_dir}")
```

#### Step Specification Example

```python
# src/cursus/steps/specs/tabular_preprocess_spec.py
from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec
from ...steps.contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT

TABULAR_PREPROCESS_SPEC = StepSpecification(
    step_type="TabularPreprocessing",
    script_contract=TABULAR_PREPROCESS_CONTRACT,
    dependencies={
        "DATA": DependencySpec(
            logical_name="DATA",
            compatible_sources=["CradleDataLoading"]
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
        )
    }
)
assert "DATA" in TABULAR_PREPROCESS_SPEC.dependencies
```

#### Builder Implementation Example

```python
# src/cursus/steps/builders/builder_tabular_preprocessing_step.py
from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

builder = TabularPreprocessingStepBuilder(config_instance)
inputs = builder._get_inputs({"DATA": "s3://bucket/data"})
assert inputs[0].destination == "/opt/ml/processing/input/data"
```

#### Model Evaluation Builder Example

```python
# src/cursus/steps/builders/builder_xgboost_model_eval_step.py
from ...steps.builders.builder_xgboost_model_eval_step import XGBoostModelEvalStepBuilder
from ...steps.configs.config_xgboost_model_eval_step import XGBoostModelEvalConfig

# Instantiate config with required fields
config = XGBoostModelEvalConfig(
    region="us-west-2",
    pipeline_s3_loc="s3://bucket/prefix",
    processing_entry_point="model_evaluation_xgb.py",
    processing_source_dir="src/pipeline_scripts",
    processing_instance_count=1,
    processing_volume_size=30,
    processing_instance_type_large="ml.m5.4xlarge",
    processing_instance_type_small="ml.m5.xlarge",
    use_large_processing_instance=False,
    pipeline_name="test-pipeline",
    job_type="evaluation",
    hyperparameters=...,
    xgboost_framework_version="1.7-1"
)
builder = XGBoostModelEvalStepBuilder(config)
env_vars = builder._get_environment_variables()
assert "LABEL_FIELD" in env_vars and "ID_FIELD" in env_vars
```

#### XGBoost Training Script Contract Example

```python
# src/cursus/steps/contracts/xgboost_train_contract.py
from ...steps.contracts.xgboost_train_contract import XGBOOST_TRAIN_CONTRACT

# Verify entry point and input paths
assert XGBOOST_TRAIN_CONTRACT.entry_point == "train_xgb.py"
assert "train_data" in XGBOOST_TRAIN_CONTRACT.expected_input_paths
```
```

## Usage

- When creating or modifying a script contract, update the corresponding Section in this file.  
- When defining a new Step Specification, validate alignment against this document.  
- Step Builders should include a validation check against these rules in unit tests.

See also:  
- [Validation Checklist](validation_checklist.md)  
- [Common Pitfalls](common_pitfalls.md)
