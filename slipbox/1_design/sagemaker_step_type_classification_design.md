---
title: "SageMaker Step Type Classification Design"
date: "2025-01-08"
author: "System Architect"
status: "Draft"
version: "1.0"
tags:
  - "design"
  - "step-registry"
  - "sagemaker"
  - "testing"
  - "automation"
---

# SageMaker Step Type Classification Design

## Overview

This design document proposes adding a new field to the `STEP_NAMES` registry to classify step builders by the type of SageMaker workflow step they create. This classification will enable automated testing frameworks to apply step-type-specific validation rules and requirements.

## Problem Statement

Currently, the Universal Builder Test framework treats all step builders uniformly, but different SageMaker step types have distinct:

1. **Input/Output Requirements**: Processing steps use ProcessingInput/ProcessingOutput, Training steps use TrainingInput, Transform steps use TransformInput, etc.
2. **Configuration Patterns**: Different instance types, framework versions, and resource requirements
3. **Validation Rules**: Each step type has specific validation requirements
4. **Testing Approaches**: Different step types need different test scenarios

The current registry structure doesn't capture this fundamental distinction, making it difficult to:
- Apply appropriate validation rules automatically
- Generate step-type-specific test cases
- Provide targeted feedback and recommendations
- Ensure compliance with SageMaker step requirements

## Proposed Solution

### 1. Registry Enhancement

Add a new `sagemaker_step_type` field to the `STEP_NAMES` registry that classifies each step builder by the SageMaker workflow step it creates:

```python
STEP_NAMES = {
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStepBuilder",
        "spec_type": "TabularPreprocessing",
        "sagemaker_step_type": "Processing",  # NEW FIELD
        "description": "Tabular data preprocessing step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",  # NEW FIELD
        "description": "XGBoost model training step"
    },
    # ... more entries
}
```

### 2. SageMaker Step Type Classification

The following classification scheme maps step builders to their corresponding SageMaker workflow step types:

#### Processing Steps
- **SageMaker Type**: `"Processing"`
- **Creates**: `sagemaker.workflow.steps.ProcessingStep`
- **Input/Output**: `ProcessingInput` / `ProcessingOutput`
- **Step Builders**:
  - `TabularPreprocessingStepBuilder`
  - `RiskTableMappingStepBuilder`
  - `CurrencyConversionStepBuilder`
  - `XGBoostModelEvalStepBuilder`
  - `ModelCalibrationStepBuilder`
  - `PackageStepBuilder`
  - `PayloadStepBuilder`
  - `CradleDataLoadingStepBuilder`

#### Training Steps
- **SageMaker Type**: `"Training"`
- **Creates**: `sagemaker.workflow.steps.TrainingStep`
- **Input/Output**: `TrainingInput` / Model artifacts
- **Step Builders**:
  - `XGBoostTrainingStepBuilder`
  - `PyTorchTrainingStepBuilder`
  - `DummyTrainingStepBuilder`

#### Transform Steps
- **SageMaker Type**: `"Transform"`
- **Creates**: `sagemaker.workflow.steps.TransformStep`
- **Input/Output**: `TransformInput` / Transform results
- **Step Builders**:
  - `BatchTransformStepBuilder`

#### CreateModel Steps
- **SageMaker Type**: `"CreateModel"`
- **Creates**: `sagemaker.workflow.steps.CreateModelStep`
- **Input/Output**: Model artifacts / Model endpoint
- **Step Builders**:
  - `XGBoostModelStepBuilder`
  - `PyTorchModelStepBuilder`

#### Registration Steps
- **SageMaker Type**: `"RegisterModel"`
- **Creates**: `sagemaker.workflow.steps.RegisterModel`
- **Input/Output**: Model artifacts / Registered model
- **Step Builders**:
  - `RegistrationStepBuilder`

### 3. Registry API Enhancements

Add new helper functions to support step type classification:

```python
def get_sagemaker_step_type(step_name: str) -> str:
    """Get SageMaker step type for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["sagemaker_step_type"]

def get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]:
    """Get all step names that create a specific SageMaker step type."""
    return [
        step_name for step_name, info in STEP_NAMES.items()
        if info["sagemaker_step_type"] == sagemaker_type
    ]

def get_all_sagemaker_step_types() -> List[str]:
    """Get all unique SageMaker step types."""
    return list(set(info["sagemaker_step_type"] for info in STEP_NAMES.values()))

def validate_sagemaker_step_type(sagemaker_type: str) -> bool:
    """Validate that a SageMaker step type exists in the registry."""
    valid_types = {"Processing", "Training", "Transform", "CreateModel", "RegisterModel"}
    return sagemaker_type in valid_types
```

## Implementation Details

### 1. Registry Structure Update

The complete updated registry structure:

```python
STEP_NAMES = {
    # Base
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "StepBuilderBase",
        "spec_type": "Base",
        "sagemaker_step_type": "Base",  # Special case
        "description": "Base pipeline configuration"
    },

    # Processing Steps
    "Processing": {
        "config_class": "ProcessingStepConfigBase",
        "builder_step_name": "ProcessingStepBuilder",
        "spec_type": "Processing",
        "sagemaker_step_type": "Processing",
        "description": "Base processing step"
    },
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStepBuilder",
        "spec_type": "TabularPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Tabular data preprocessing step"
    },
    "RiskTableMapping": {
        "config_class": "RiskTableMappingConfig",
        "builder_step_name": "RiskTableMappingStepBuilder",
        "spec_type": "RiskTableMapping",
        "sagemaker_step_type": "Processing",
        "description": "Risk table mapping step for categorical features"
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStepBuilder",
        "spec_type": "CurrencyConversion",
        "sagemaker_step_type": "Processing",
        "description": "Currency conversion processing step"
    },
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvalStepBuilder",
        "spec_type": "XGBoostModelEval",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model evaluation step"
    },
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "sagemaker_step_type": "Processing",
        "description": "Calibrates model prediction scores to accurate probabilities"
    },
    "Package": {
        "config_class": "PackageConfig",
        "builder_step_name": "PackageStepBuilder",
        "spec_type": "Package",
        "sagemaker_step_type": "Processing",
        "description": "Model packaging step"
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadStepBuilder",
        "spec_type": "Payload",
        "sagemaker_step_type": "Processing",
        "description": "Payload testing step"
    },
    "CradleDataLoading": {
        "config_class": "CradleDataLoadConfig",
        "builder_step_name": "CradleDataLoadingStepBuilder",
        "spec_type": "CradleDataLoading",
        "sagemaker_step_type": "Processing",
        "description": "Cradle data loading step"
    },

    # Training Steps
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",
        "builder_step_name": "PyTorchTrainingStepBuilder",
        "spec_type": "PyTorchTraining",
        "sagemaker_step_type": "Training",
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step"
    },
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "sagemaker_step_type": "Training",
        "description": "Training step that uses a pretrained model"
    },

    # Model Steps
    "PyTorchModel": {
        "config_class": "PyTorchModelConfig",
        "builder_step_name": "PyTorchModelStepBuilder",
        "spec_type": "PyTorchModel",
        "sagemaker_step_type": "CreateModel",
        "description": "PyTorch model creation step"
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelConfig",
        "builder_step_name": "XGBoostModelStepBuilder",
        "spec_type": "XGBoostModel",
        "sagemaker_step_type": "CreateModel",
        "description": "XGBoost model creation step"
    },

    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStepBuilder",
        "spec_type": "BatchTransform",
        "sagemaker_step_type": "Transform",
        "description": "Batch transform step"
    },

    # Registration Steps
    "Registration": {
        "config_class": "RegistrationConfig",
        "builder_step_name": "RegistrationStepBuilder",
        "spec_type": "Registration",
        "sagemaker_step_type": "RegisterModel",
        "description": "Model registration step"
    },

    # Utility Steps (Special case - these don't create SageMaker steps directly)
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStepBuilder",
        "spec_type": "HyperparameterPrep",
        "sagemaker_step_type": "Utility",  # Special classification
        "description": "Hyperparameter preparation step"
    }
}
```

### 2. Testing Framework Integration

The Universal Builder Test framework will be enhanced to use this classification:

#### Step-Type-Specific Test Variants

```python
class SageMakerStepTypeValidator:
    """Validates step builders based on their SageMaker step type."""
    
    def __init__(self, builder_class: Type[StepBuilderBase]):
        self.builder_class = builder_class
        self.step_name = self._detect_step_name()
        self.sagemaker_step_type = get_sagemaker_step_type(self.step_name)
    
    def validate_step_type_compliance(self) -> List[ValidationViolation]:
        """Validate compliance with SageMaker step type requirements."""
        if self.sagemaker_step_type == "Processing":
            return self._validate_processing_step()
        elif self.sagemaker_step_type == "Training":
            return self._validate_training_step()
        elif self.sagemaker_step_type == "Transform":
            return self._validate_transform_step()
        elif self.sagemaker_step_type == "CreateModel":
            return self._validate_create_model_step()
        elif self.sagemaker_step_type == "RegisterModel":
            return self._validate_register_model_step()
        else:
            return []
    
    def _validate_processing_step(self) -> List[ValidationViolation]:
        """Validate Processing step requirements."""
        violations = []
        
        # Check that create_step returns ProcessingStep
        # Check that _get_inputs returns List[ProcessingInput]
        # Check that _get_outputs returns List[ProcessingOutput]
        # Check processor creation methods
        
        return violations
    
    def _validate_training_step(self) -> List[ValidationViolation]:
        """Validate Training step requirements."""
        violations = []
        
        # Check that create_step returns TrainingStep
        # Check that _get_inputs returns Dict[str, TrainingInput]
        # Check estimator creation methods
        
        return violations
    
    # ... similar methods for other step types
```

#### Enhanced Universal Test Framework

```python
class UniversalStepBuilderTest:
    """Enhanced with SageMaker step type awareness."""
    
    def run_step_type_specific_tests(self) -> Dict[str, Any]:
        """Run tests specific to the SageMaker step type."""
        step_name = self._detect_step_name()
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        if sagemaker_step_type == "Processing":
            return self._run_processing_tests()
        elif sagemaker_step_type == "Training":
            return self._run_training_tests()
        elif sagemaker_step_type == "Transform":
            return self._run_transform_tests()
        elif sagemaker_step_type == "CreateModel":
            return self._run_create_model_tests()
        elif sagemaker_step_type == "RegisterModel":
            return self._run_register_model_tests()
        else:
            return {}
    
    def _run_processing_tests(self) -> Dict[str, Any]:
        """Run Processing-specific tests."""
        return {
            "test_processing_step_creation": self._test_processing_step_creation(),
            "test_processing_inputs": self._test_processing_inputs(),
            "test_processing_outputs": self._test_processing_outputs(),
            "test_processor_configuration": self._test_processor_configuration(),
        }
    
    def _run_training_tests(self) -> Dict[str, Any]:
        """Run Training-specific tests."""
        return {
            "test_training_step_creation": self._test_training_step_creation(),
            "test_training_inputs": self._test_training_inputs(),
            "test_estimator_configuration": self._test_estimator_configuration(),
            "test_hyperparameters": self._test_hyperparameters(),
        }
    
    # ... similar methods for other step types
```

## Benefits

### 1. Automated Testing Enhancement
- **Step-Type-Specific Validation**: Apply appropriate validation rules automatically
- **Targeted Test Cases**: Generate relevant test scenarios for each step type
- **Comprehensive Coverage**: Ensure all SageMaker step requirements are tested

### 2. Improved Developer Experience
- **Clear Classification**: Developers understand what type of SageMaker step their builder creates
- **Targeted Feedback**: Get specific recommendations based on step type
- **Better Documentation**: Auto-generate step-type-specific documentation

### 3. Quality Assurance
- **Compliance Checking**: Ensure builders create the correct SageMaker step type
- **Interface Validation**: Verify input/output handling matches step type requirements
- **Configuration Validation**: Check step-type-specific configuration patterns

### 4. Maintenance and Evolution
- **Easier Refactoring**: Understand impact of changes on specific step types
- **Pattern Recognition**: Identify common patterns within step types
- **Future Extensions**: Easy to add new step types or enhance existing ones

## Implementation Plan

### Phase 1: Registry Enhancement
1. **Update STEP_NAMES**: Add `sagemaker_step_type` field to all entries
2. **Add Helper Functions**: Implement new registry API functions
3. **Update Existing Code**: Ensure backward compatibility
4. **Add Validation**: Validate registry consistency

### Phase 2: Testing Framework Integration
1. **Create Step Type Validators**: Implement step-type-specific validation classes
2. **Enhance Universal Test**: Add step-type-aware testing capabilities
3. **Update Processing Test**: Leverage new classification in existing processing test
4. **Add New Test Variants**: Create training, transform, and model test variants

### Phase 3: Documentation and Tooling
1. **Update Documentation**: Reflect new classification in developer guides
2. **Create Migration Guide**: Help developers understand the changes
3. **Add Tooling**: Create utilities to analyze step types across the codebase
4. **Generate Reports**: Auto-generate step-type-specific reports

### Phase 4: Validation and Rollout
1. **Test All Builders**: Validate all existing step builders with new framework
2. **Fix Issues**: Address any compliance issues discovered
3. **Performance Testing**: Ensure new classification doesn't impact performance
4. **Full Rollout**: Deploy to all environments

## Risks and Mitigation

### Risk 1: Breaking Changes
- **Mitigation**: Maintain backward compatibility, add new fields without removing old ones
- **Testing**: Comprehensive regression testing

### Risk 2: Classification Errors
- **Mitigation**: Careful analysis of each step builder's actual SageMaker step creation
- **Validation**: Cross-reference with actual step creation code

### Risk 3: Maintenance Overhead
- **Mitigation**: Automated validation to catch inconsistencies
- **Documentation**: Clear guidelines for maintaining classification

## Future Enhancements

### 1. Dynamic Step Type Detection
- Automatically detect SageMaker step type from builder implementation
- Validate registry classification against actual behavior

### 2. Step Type Hierarchies
- Support for step type inheritance (e.g., specialized processing steps)
- More granular classification within major step types

### 3. Cross-Step-Type Analysis
- Analyze compatibility between different step types
- Optimize pipeline composition based on step type patterns

### 4. Performance Optimization
- Step-type-specific performance optimizations
- Resource allocation based on step type characteristics

## Conclusion

Adding SageMaker step type classification to the STEP_NAMES registry will significantly enhance the automated testing framework's ability to provide targeted, relevant validation for different types of step builders. This classification creates a foundation for more sophisticated testing, better developer feedback, and improved quality assurance across the entire pipeline system.

The proposed implementation maintains backward compatibility while adding powerful new capabilities that will benefit both current and future development efforts. The step-type-aware testing framework will help ensure that all step builders meet their specific SageMaker requirements and follow appropriate patterns for their step type.

## Reference

- [step_builder_registry_design.md](step_builder_registry_design.md)
- [universal_step_builder_test.md](universal_step_builder_test.md)
- [standardization_rules.md](standardization_rules.md)
