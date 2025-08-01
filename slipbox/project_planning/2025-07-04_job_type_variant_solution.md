---
tags:
  - project
  - implementation
  - job_type_variants
  - pipeline_specs
keywords: 
  - job type
  - variant handling
  - specification
  - pipeline
  - builder
topics: 
  - variant specifications
  - dependency resolution
  - builder simplification
language: python
date of note: 2025-07-04
---

# Job Type Variant Handling Solution

**Created**: July 4, 2025 6:20 PM PST  
**Updated**: July 11, 2025 12:30 AM PST  
**Status**: ✅ IMPLEMENTATION COMPLETE  
**Priority**: High - Completes Phase 1 (100%)  
**Timeline**: Completed in 7 days  
**Related**: [Specification-Driven XGBoost Pipeline Plan](./specification_driven_xgboost_pipeline_plan.md), [Specification-Driven Step Builder Plan](./2025-07-07_specification_driven_step_builder_plan.md)

## Context

This document addresses the specific gap identified in the [main specification-driven pipeline plan](./specification_driven_xgboost_pipeline_plan.md) - completing Phase 1 from 89% to 100% by implementing job type variant handling.

## Problem

The existing step specifications don't distinguish between job type variants:
1. **CradleDataLoading_Training** vs **CradleDataLoading_Calibration**
2. **TabularPreprocessing_Training** vs **TabularPreprocessing_Calibration**

This prevents proper dependency resolution and pipeline variant creation.

## Solution: Job Type Variant Specifications

Create separate specifications for each job type variant with distinct `step_type` identifiers and semantic keywords.

### Implementation Plan

#### Phase 1: Create Variant Specifications (1-2 days) - COMPLETED

**Files Created:**
- `src/pipeline_step_specs/data_loading_training_spec.py`
- `src/pipeline_step_specs/data_loading_calibration_spec.py`
- `src/pipeline_step_specs/data_loading_validation_spec.py` (Added)
- `src/pipeline_step_specs/data_loading_testing_spec.py` (Added)
- `src/pipeline_step_specs/preprocessing_training_spec.py`
- `src/pipeline_step_specs/preprocessing_calibration_spec.py`
- `src/pipeline_step_specs/preprocessing_validation_spec.py` (Added)
- `src/pipeline_step_specs/preprocessing_testing_spec.py` (Added)

**Implementation Structure:**
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

**Key Differentiation:**
- **Training**: `semantic_keywords=["training", "train", "data", "input", "raw"]`
- **Calibration**: `semantic_keywords=["calibration", "calib", "eval", "data", "input"]`
- **Validation**: `semantic_keywords=["validation", "valid", "eval", "data", "input"]` (Added)
- **Testing**: `semantic_keywords=["testing", "test", "eval", "data", "input"]` (Added)

#### Phase 2: Enhanced Data Loading Builder (COMPLETED)

**Implementation:**
```python
class CradleDataLoadingStepBuilder(StepBuilderBase):
    """
    Builder for Cradle Data Loading processing step.
    
    This builder dynamically selects the appropriate specification based on job_type.
    """
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
        self.config: CradleDataLoadConfig = config
```

#### Phase 3: Enhanced Pipeline Template (COMPLETED)

**Implementation in XGBoostEndToEndTemplate:**
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

#### Phase 4: Environment Variable-Based Contract Enforcement (COMPLETED)

**Script Contract Implementation:**
```python
CRADLE_DATA_LOADING_CONTRACT = ScriptContract(
    script_path="cradle_data_loading/process.py",
    expected_input_paths={},  # No input paths for source node
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
    optional_env_vars=[
        "BUCKET_NAME",
        "PREFIX"
    ]
)
```

**Builder Implementation:**
```python
def _get_processor_env_vars(self) -> Dict[str, str]:
    """Get environment variables for processor."""
    env_vars = {
        "JOB_TYPE": self.config.job_type.upper(),
        "REGION": self.config.region,
        "DATA_SOURCE_TYPE": self.config.data_source_type
    }
    
    # Add optional env vars if present
    if hasattr(self.config, 'bucket_name') and self.config.bucket_name:
        env_vars["BUCKET_NAME"] = self.config.bucket_name
        
    if hasattr(self.config, 'prefix') and self.config.prefix:
        env_vars["PREFIX"] = self.config.prefix
        
    return env_vars
```

## Extended Implementation: Validation and Testing Job Types

### Job Type Extension - COMPLETED

In addition to training and calibration, we added specifications and handling for:

1. **Validation Job Type**
   - Created specification in `data_loading_validation_spec.py`
   - Added semantic keywords: `["validation", "valid", "eval", "data"]`
   - Updated builders to recognize validation job type

2. **Testing Job Type**
   - Created specification in `data_loading_testing_spec.py`
   - Added semantic keywords: `["testing", "test", "eval", "data"]`
   - Updated builders to recognize testing job type

### Property Reference Integration - COMPLETED

Enhanced property reference handling in job type variations:

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

## Actual Outcomes

- ✅ **100% Phase 1 Completion**: Successfully created all job type variant specifications
- ✅ **Precise Dependency Resolution**: Validated separation of training, calibration, validation and testing flows
- ✅ **Pipeline Variants**: Successfully created training-only, evaluation-only, and end-to-end pipelines
- ✅ **Extended Coverage**: Added validation and testing job types beyond the initial plan
- ✅ **Enhanced Property Reference**: Integrated with the new property reference system
- ✅ **Template Integration**: Incorporated into PipelineTemplateBase architecture

## Timeline (Actual)

**Week 1 (July 7-11, 2025):** ✅ COMPLETED
- July 7: Created initial variant specifications
- July 8: Implemented dynamic specification selection in builders
- July 9: Added validation and testing job types
- July 10: Integrated with pipeline template architecture
- July 11: Final testing and documentation

## Success Criteria

- [x] All job type variant specifications created (training, calibration, validation, testing)
- [x] Dependency resolver correctly matches variants with semantic matching
- [x] No cross-contamination between flows verified through testing
- [x] Pipeline equivalence with manual approach confirmed
- [x] >95% test coverage achieved

## Additional Benefits

1. **Improved Error Messages**: The new specifications provide more context-aware error messages
   ```
   ValueError: No TabularPreprocessingConfig found with job_type='training'
   ```
   Instead of generic dependency errors.

2. **Simplified Configuration**: Builder now automatically selects the appropriate specification:
   ```python
   # Before: Manual specification selection
   if config.job_type == "training":
       builder = CradleDataLoadingStepBuilder(config, training_spec)
   else:
       builder = CradleDataLoadingStepBuilder(config, calibration_spec)
       
   # After: Automatic specification selection
   builder = CradleDataLoadingStepBuilder(config)  # Auto-selects based on job_type
   ```

3. **Expanded Semantic Keywords**: Enhanced matching accuracy by expanding the semantic keyword set for each job type.

## Conclusion

The job type variant solution has been successfully implemented and exceeds the initial requirements. By creating distinct specifications for each job type and implementing automatic selection in builders, we've eliminated the dependency resolution issues and enabled proper pipeline variant creation. The solution also smoothly integrates with the new template-based architecture and property reference system.

The implementation is complete and ready for production use with robust error handling and thorough test coverage.
