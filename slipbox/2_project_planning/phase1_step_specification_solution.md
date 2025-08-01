# Phase 1 Step Specification Solution: Job Type Variant Handling

## Problem Statement

According to the project plan, the only remaining gap in Phase 1 was **job type variant handling** for:

1. **CradleDataLoading_Training** vs **CradleDataLoading_Calibration**
2. **TabularPreprocessing_Training** vs **TabularPreprocessing_Calibration**

The challenge was to provide a clean, maintainable solution that allows the same step builder classes to handle different job types (training vs calibration) while maintaining proper dependency resolution and script contract validation.

## Solution Overview

The solution integrates **Script Contract Validation** into the **Step Specification System**, providing:

1. **Enhanced StepSpecification Class** with script contract integration
2. **Job Type Variant Support** through semantic keywords and compatible sources
3. **Automated Script Validation** against predefined contracts
4. **Backward Compatibility** with existing specifications

## Implementation Details

### 1. Enhanced Base Specifications

**File**: `src/pipeline_deps/base_specifications.py`

Key enhancements:
- Added `script_contract` field to `StepSpecification` class
- Implemented `validate_script_compliance()` method
- Maintained backward compatibility with existing code
- Proper handling of TYPE_CHECKING for circular import avoidance

```python
class StepSpecification(BaseModel):
    # ... existing fields ...
    script_contract: Optional['ScriptContract'] = Field(
        default=None,
        description="Optional script contract for validation"
    )
    
    def validate_script_compliance(self, script_path: str) -> 'ValidationResult':
        """Validate script implementation against contract."""
        if not self.script_contract:
            from ..pipeline_script_contracts.base_script_contract import ValidationResult
            return ValidationResult.success("No script contract defined")
        return self.script_contract.validate_implementation(script_path)
```

### 2. Job Type Variant Handling

**Training vs Calibration Differentiation**:

The solution uses **semantic keywords** and **compatible sources** to distinguish between training and calibration variants:

#### CradleDataLoading Variants

- **Training**: `semantic_keywords=["training", "train", "model_training"]`
- **Calibration**: `semantic_keywords=["calibration", "eval", "evaluation", "model_evaluation"]`

#### TabularPreprocessing Variants

- **Training**: `semantic_keywords=["training", "train", "model_training", "preprocessed"]`
- **Calibration**: `semantic_keywords=["calibration", "eval", "evaluation", "model_evaluation", "preprocessed"]`

### 3. Updated Step Specifications

#### Model Evaluation Specification
**File**: `src/pipeline_step_specs/model_eval_spec.py`

```python
# Import the contract at runtime to avoid circular imports
def _get_model_evaluation_contract():
    from ..pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT
    return MODEL_EVALUATION_CONTRACT

MODEL_EVAL_SPEC = StepSpecification(
    step_type="XGBoostModelEvaluation",
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),
    # ... dependencies and outputs ...
)
```

#### Preprocessing Training Specification
**File**: `src/pipeline_step_specs/preprocessing_training_spec.py`

```python
def _get_tabular_preprocess_contract():
    from ..pipeline_script_contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
    return TABULAR_PREPROCESS_CONTRACT

PREPROCESSING_TRAINING_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Training",
    node_type=NodeType.INTERNAL,
    script_contract=_get_tabular_preprocess_contract(),
    # ... dependencies and outputs ...
)
```

#### XGBoost Training Specification
**File**: `src/pipeline_step_specs/xgboost_training_spec.py`

```python
def _get_xgboost_train_contract():
    from ..pipeline_script_contracts.xgboost_train_contract import XGBOOST_TRAIN_CONTRACT
    return XGBOOST_TRAIN_CONTRACT

XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    script_contract=_get_xgboost_train_contract(),
    # ... dependencies and outputs ...
)
```

### 4. Comprehensive Testing

**File**: `test/pipeline_deps/test_base_specifications.py`

Added comprehensive test suites:

1. **TestScriptContractIntegration**: Tests script contract functionality
2. **TestStepSpecificationIntegration**: Tests updated specifications
3. **Backward Compatibility Tests**: Ensures existing code continues to work
4. **Circular Import Prevention Tests**: Verifies no import issues

All tests pass successfully:
```
test/pipeline_deps/test_base_specifications.py::TestScriptContractIntegration PASSED
test/pipeline_deps/test_base_specifications.py::TestStepSpecificationIntegration PASSED
```

## How Job Type Variants Are Handled

### 1. Semantic Matching

The dependency resolver uses semantic keywords to match appropriate data sources:

```python
# Training pipeline
training_dep = DependencySpec(
    logical_name="DATA",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    semantic_keywords=["training", "train", "model_training"],
    compatible_sources=["CradleDataLoading_Training"]
)

# Calibration pipeline  
calibration_dep = DependencySpec(
    logical_name="DATA", 
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    semantic_keywords=["calibration", "eval", "model_evaluation"],
    compatible_sources=["CradleDataLoading_Calibration"]
)
```

### 2. Compatible Source Filtering

The registry system filters compatible outputs based on:
- **Step type matching**: `CradleDataLoading_Training` vs `CradleDataLoading_Calibration`
- **Semantic keyword overlap**: Training keywords vs calibration keywords
- **Dependency type compatibility**: All use `PROCESSING_OUTPUT`

### 3. Script Contract Validation

Each step specification includes appropriate script contracts:
- **TabularPreprocessing**: Uses `TABULAR_PREPROCESS_CONTRACT`
- **XGBoostTraining**: Uses `XGBOOST_TRAIN_CONTRACT`  
- **ModelEvaluation**: Uses `MODEL_EVALUATION_CONTRACT`

## Benefits of This Solution

### 1. **Clean Separation of Concerns**
- Job type variants handled through semantic matching
- Script validation separated from dependency resolution
- Clear distinction between training and calibration workflows

### 2. **Maintainability**
- Single step builder classes handle multiple job types
- Centralized contract definitions
- Consistent validation across all steps

### 3. **Extensibility**
- Easy to add new job types (e.g., inference, testing)
- Simple to add new semantic keywords
- Straightforward contract updates

### 4. **Robustness**
- Comprehensive test coverage
- Backward compatibility maintained
- Circular import prevention
- Type safety with Pydantic V2

## Usage Example

```python
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
from src.pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC

# Register specifications
registry = SpecificationRegistry()
registry.register("preprocessing_training", PREPROCESSING_TRAINING_SPEC)
registry.register("model_eval", MODEL_EVAL_SPEC)

# Find compatible outputs for training data
training_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    semantic_keywords=["training", "processed"],
    compatible_sources=["TabularPreprocessing_Training"]
)

compatible_outputs = registry.find_compatible_outputs(training_dep)
# Returns: [('preprocessing_training', 'processed_data', OutputSpec(...), score)]

# Validate script compliance
result = PREPROCESSING_TRAINING_SPEC.validate_script_compliance("src/pipeline_scripts/tabular_preprocess.py")
print(f"Script compliance: {result.is_valid}")
```

## Conclusion

This solution successfully addresses the job type variant handling gap by:

1. **Integrating script contracts** into step specifications for automated validation
2. **Using semantic keywords** to distinguish between training and calibration workflows
3. **Maintaining backward compatibility** with existing code
4. **Providing comprehensive testing** to ensure reliability

The implementation is clean, maintainable, and extensible, setting a solid foundation for Phase 2 development.

## Next Steps

With Phase 1 complete, the system is ready for:
1. **Phase 2**: Enhanced dependency resolution with smart matching
2. **Phase 3**: Full pipeline builder integration
3. **Production deployment** with validated step specifications

All job type variant handling requirements have been successfully implemented and tested.
