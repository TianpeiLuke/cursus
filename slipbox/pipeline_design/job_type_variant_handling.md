# Job Type Variant Handling

## Overview

This design document describes the comprehensive solution for handling job type variants throughout the pipeline system. Job type variants (training, calibration, validation, testing) allow creating step variants with different purposes while sharing the same underlying configuration class structure. This document covers both the configuration system aspects and the pipeline specification system integration.

## Purpose

The purpose of job type variant handling is to:

1. **Support multiple job types** for the same step class (e.g., training and calibration)
2. **Enable pipeline variants** (training-only, calibration-only, end-to-end)
3. **Support semantic dependency matching** between steps with the same job type
4. **Enable specialized behavior** for different job variants
5. **Provide clear separation** between data flows for different purposes

## Core Components

### 1. Job Type Attributes in Configuration

Configuration classes include a `job_type` attribute to indicate their variant:

```python
class CradleDataLoadConfig(ProcessingStepConfigBase):
    job_type: Optional[str] = None  # e.g., "training", "calibration", "validation", "testing"
    # ... other fields ...
```

### 2. Step Name Generation with Job Type

The step name generation algorithm is implemented in `TypeAwareConfigSerializer.generate_step_name` and appends the job type to the base step name:

```python
def generate_step_name(self, config: Any) -> str:
    """
    Generate a step name for a config, including job type and other distinguishing attributes.
    """
    # ... get base_step from registry ...
    
    # Append distinguishing attributes - essential for job type variants
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
                    
    return step_name
```

This creates distinct step names for different job type variants:
- `CradleDataLoading_training`
- `CradleDataLoading_calibration`

Following the Single Source of Truth principle, this implementation is used consistently throughout the system:
- In `config_merger.py` via the TypeAwareConfigSerializer
- In `utils.py` by directly using the TypeAwareConfigSerializer
- In step builders and serialization/deserialization components

### 3. Config Types Metadata

The `config_types` metadata maps these variant step names to their class:

```json
"config_types": {
  "CradleDataLoading_training": "CradleDataLoadConfig",
  "CradleDataLoading_calibration": "CradleDataLoadConfig",
  "XGBoostTraining": "XGBoostTrainingConfig",
  "XGBoostModelEval_calibration": "XGBoostModelEvalConfig"
}
```

### 4. Variant Specifications

Each job type variant has a dedicated step specification with distinct `step_type` identifiers and semantic keywords:

```python
# data_loading_training_spec.py
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    step_type="CradleDataLoading_Training",
    node_type=NodeType.SOURCE,
    dependencies=[],
    outputs=[
        OutputSpec(
            logical_name="DATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training data output from Cradle data loading",
            semantic_keywords=["training", "train", "data", "input", "raw", "dataset"]
        ),
        # ... METADATA and SIGNATURE outputs
    ]
)
```

### 5. Dynamic Specification Selection

Step builders dynamically select the appropriate specification based on the job type:

```python
class CradleDataLoadingStepBuilder(StepBuilderBase):
    """Builder for Cradle Data Loading processing step."""
    
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        # Determine which specification to use based on job type
        job_type = getattr(config, 'job_type', 'training').lower()
        
        # Select appropriate specification
        if job_type == 'calibration':
            spec = DATA_LOADING_CALIBRATION_SPEC
        elif job_type == 'validation':
            spec = DATA_LOADING_VALIDATION_SPEC
        elif job_type == 'testing':
            spec = DATA_LOADING_TESTING_SPEC
        else:  # Default to training
            spec = DATA_LOADING_TRAINING_SPEC
            
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
```

### 6. Semantic Keywords

Each job type uses distinct semantic keywords to ensure proper dependency matching:

| Job Type | Semantic Keywords |
|----------|------------------|
| Training | training, train, data, input, raw, dataset |
| Calibration | calibration, calib, eval, data, input |
| Validation | validation, valid, eval, data, input |
| Testing | testing, test, eval, data, input |

### 7. Environment Variable Contract

Job types are passed to processing scripts via environment variables:

```python
CRADLE_DATA_LOADING_CONTRACT = ScriptContract(
    script_path="cradle_data_loading/process.py",
    expected_output_paths={
        "DATA": "/opt/ml/processing/output/data",
        "METADATA": "/opt/ml/processing/output/metadata",
        "SIGNATURE": "/opt/ml/processing/output/signature"
    },
    required_env_vars=[
        "JOB_TYPE",  # Must be one of: training, calibration, validation, testing
        "REGION",
        "DATA_SOURCE_TYPE"
    ],
    # ...
)
```

## Pipeline Template Integration

### 1. Pipeline DAG Structure

The pipeline template creates a DAG with distinct nodes for each job type variant:

```python
def _create_pipeline_dag(self) -> PipelineDAG:
    """Create the pipeline DAG structure."""
    dag = PipelineDAG()
    
    # Add nodes with job type variants in node names
    dag.add_node("train_data_load")  # Uses CradleDataLoading_Training spec
    dag.add_node("train_preprocess") # Uses TabularPreprocessing_Training spec
    dag.add_node("xgboost_train")
    dag.add_node("model_packaging")
    dag.add_node("payload_test")
    dag.add_node("model_registration")
    dag.add_node("calib_data_load") # Uses CradleDataLoading_Calibration spec
    dag.add_node("calib_preprocess") # Uses TabularPreprocessing_Calibration spec
    
    # Add edges - now properly connected by job type
    dag.add_edge("train_data_load", "train_preprocess")
    dag.add_edge("train_preprocess", "xgboost_train")
    dag.add_edge("xgboost_train", "model_packaging")
    dag.add_edge("xgboost_train", "payload_test")
    dag.add_edge("model_packaging", "model_registration")
    dag.add_edge("payload_test", "model_registration")
    dag.add_edge("calib_data_load", "calib_preprocess")
    
    return dag
```

### 2. Config Map with Job Type Information

The template creates a config map that preserves job type information:

```python
def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
    """Create the config map with job type information preserved."""
    config_map = {}
    
    # Add training configs with job_type='training'
    train_dl_config = self._get_config_by_type(CradleDataLoadConfig, "training")
    if train_dl_config:
        config_map["train_data_load"] = train_dl_config
        
    # Add calibration configs with job_type='calibration'
    calib_dl_config = self._get_config_by_type(CradleDataLoadConfig, "calibration")
    if calib_dl_config:
        config_map["calib_data_load"] = calib_dl_config
    
    # ... additional configs
    return config_map
```

## Configuration System Integration

### 1. Specific Data Section

Each job type variant gets its own section in the `specific` part of the configuration:

```json
"specific": {
  "CradleDataLoading_training": {
    "job_type": "training",
    "input_path": "s3://bucket/training_data"
  },
  "CradleDataLoading_calibration": {
    "job_type": "calibration",
    "input_path": "s3://bucket/calibration_data"
  }
}
```

### 2. Property Reference Integration

Property references are adjusted based on job type:

```python
def to_runtime_property(self, step_instances: Dict[str, Any]) -> Any:
    """Create an actual SageMaker property reference for job type variant."""
    step = step_instances[self.step_name]
    
    # Get job type from step if available
    job_type = None
    if hasattr(step, "_job_type"):
        job_type = step._job_type
        
    # Job type specific property path logic
    if job_type:
        # Adjust property path based on job type
        if job_type.lower() == "training" and "train_data" in self.property_path:
            obj = step.properties.outputs["train_data"]
        elif job_type.lower() == "calibration" and "calib_data" in self.property_path:
            obj = step.properties.outputs["calib_data"]
        else:
            obj = self._navigate_property_path(step.properties, self.property_path)
    else:
        obj = self._navigate_property_path(step.properties, self.property_path)
    
    return obj
```

## Use Cases

### 1. End-to-End Pipelines

Create pipelines with multiple job types to handle different datasets:

```python
# Training pipeline components
cradle_data_training = CradleDataLoadConfig(job_type="training", ...)
preprocess_training = TabularPreprocessingConfig(job_type="training", ...)
training = XGBoostTrainingConfig(...)

# Calibration pipeline components
cradle_data_calibration = CradleDataLoadConfig(job_type="calibration", ...)
preprocess_calibration = TabularPreprocessingConfig(job_type="calibration", ...)
model_eval = XGBoostModelEvalConfig(job_type="calibration", ...)

# Combined pipeline
configs = [
    cradle_data_training, preprocess_training, training,
    cradle_data_calibration, preprocess_calibration, model_eval
]
```

### 2. Variant-Specific Pipelines

Create specialized pipelines for specific job types:

```python
# Training-only pipeline
training_configs = [c for c in configs if 
    getattr(c, "job_type", None) == "training" or not hasattr(c, "job_type")]

# Calibration-only pipeline
calibration_configs = [c for c in configs if 
    getattr(c, "job_type", None) == "calibration" or not hasattr(c, "job_type")]
```

## Benefits

1. **Reuse Classes**: Use the same config class for different purposes
2. **Semantic Matching**: Connect steps with the same job type automatically
3. **Pipeline Variations**: Create training-only or evaluation-only pipelines easily
4. **Data Flow Control**: Direct different datasets through specialized processing paths
5. **Improved Error Messages**: More context-aware error messages
6. **Simplified Configuration**: Automatic specification selection based on job_type

## Challenges and Solutions

### 1. Step Name Uniqueness

**Challenge**: Need to ensure step names remain unique across job types

**Solution**: Append job_type to step name during serialization

### 2. Configuration Loading

**Challenge**: Need to identify which specific section to load from

**Solution**: Use step name with job type suffix as the key in the specific section

### 3. Dependency Resolution

**Challenge**: Need to connect steps with matching job types

**Solution**: Use step names with job type suffixes in dependency definitions

### 4. Property Reference Handling

**Challenge**: Property paths may differ based on job type

**Solution**: Implement job type-specific property path logic in property references

## Implementation Details

### 1. Configuration Field Categorization

The configuration system's field categorization places job_type fields in the specific section:

```python
def _is_special_field(self, field_name, value, config):
    """
    Determine if a field should be treated as special.
    
    Special fields are always kept in specific sections.
    """
    # Check against known special fields
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return True
        
    # ... other checks ...

# The job_type field is explicitly marked as special
SPECIAL_FIELDS_TO_KEEP_SPECIFIC = {
    "hyperparameters", 
    "hyperparameters_s3_uri",
    "job_name_prefix",
    "job_type"  # Ensures job_type is always in specific section
}
```

### 2. Step Builder Factory Integration

```python
class StepBuilderFactory:
    """Factory for creating step builders."""
    
    @classmethod
    def create_builder(cls, config, sagemaker_session=None, role=None):
        """Create a step builder for the config."""
        config_type = config.__class__.__name__
        
        # Select builder class based on config type
        builder_class = BUILDER_MAP.get(config_type)
        if not builder_class:
            raise ValueError(f"No builder for config type {config_type}")
            
        # Create builder with config
        # Specification selection happens in builder constructor
        return builder_class(config, sagemaker_session=sagemaker_session, role=role)
```

### 3. Pipeline Template Integration

```python
class XGBoostEndToEndTemplate(PipelineTemplateBase):
    """Template for XGBoost end-to-end pipeline."""
    
    def _get_config_by_type(self, config_class, job_type=None):
        """Get config by type and optional job type."""
        for config in self.configs:
            if isinstance(config, config_class):
                if job_type is not None:
                    if hasattr(config, 'job_type') and config.job_type == job_type:
                        return config
                else:
                    return config
        
        if job_type:
            raise ValueError(f"No {config_class.__name__} found with job_type='{job_type}'")
        return None
```

## Future Improvements

1. **Validation Enhancements**: Add validation to ensure job types are consistent
2. **Documentation**: Expand usage examples for job type variants
3. **Helper Functions**: Create utilities for working with job type variants
4. **Automated Testing**: Add more tests specifically for job type handling
5. **Additional Job Types**: Support for more specialized job types as needed

## References

- [Job Type Variant Solution](../slipbox/project_planning/2025-07-04_job_type_variant_solution.md)
- [Registry-Based Step Name Generation](./registry_based_step_name_generation.md)
- [Config Types Format](./config_types_format.md)
- [Simplified Config Field Categorization](./simplified_config_field_categorization.md)
