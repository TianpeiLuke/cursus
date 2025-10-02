---
tags:
  - design
  - validation
  - step_builders
  - method_interface
  - registry_integration
keywords:
  - step builder validation
  - method interface validation
  - universal builder rules
  - step type specific rules
  - registry integration
  - step catalog validation
topics:
  - step builder validation
  - method interface compliance
  - validation rulesets
  - registry integration
language: python
date of note: 2025-10-01
implementation_status: DESIGN
---

# Step Builder Validation Rulesets Design

## Executive Summary

This document defines comprehensive validation rulesets for step builders that focus on **method interface compliance** rather than configuration field validation. The approach recognizes that step builders handle their own configuration validation, so the validation system should ensure **structural correctness** - that builders implement the required method interfaces with correct signatures.

### Key Principles

1. **Method-Centric Validation**: Focus on validating method existence and signatures, not config content
2. **Builder Autonomy**: Let step builders handle their own configuration validation
3. **Interface Consistency**: Ensure all builders follow the same structural patterns
4. **Registry Integration**: Design rulesets for easy retrieval via step catalog and registry systems
5. **Step-Type Awareness**: Apply appropriate validation rules based on SageMaker step type

## Problem Statement

Current validation approaches suffer from several issues:

### **❌ Current Problems:**
1. **Config Field Validation is Tedious**: Checking individual config fields is redundant since builders validate themselves
2. **Field Name Variants**: Config field names often differ from SageMaker parameter names
3. **Overly Strict Validations**: Validating implementation details that don't affect functionality
4. **Inconsistent Interfaces**: No systematic way to ensure builders implement required methods
5. **Registry Disconnection**: Validation rules not easily retrievable via step catalog system

### **✅ Proposed Solution:**
Focus validation on **method interface compliance** - ensure builders implement required methods with correct signatures, while letting them handle their own configuration validation internally.

## Validation Strategy Overview

### **Two-Tier Ruleset Structure:**

1. **Universal Builder Rules**: Methods ALL step builders must implement
2. **Step-Type-Specific Rules**: Additional methods required for specific SageMaker step types

### **Registry-Compatible Design:**
- Easy retrieval via step catalog and registry systems
- Automatic step type detection from registry
- Extensible structure for new step types

## Ruleset 1: Universal Builder Method Requirements

All step builders must inherit from `StepBuilderBase` and implement these **universal methods**:

### **Required Universal Methods**

```python
UNIVERSAL_BUILDER_VALIDATION_RULES = {
    "required_methods": {
        "validate_configuration": {
            "signature": "validate_configuration(self) -> None",
            "description": "Validate configuration requirements",
            "return_type": "None",
            "required": True,
            "raises": ["ValueError"],
            "purpose": "Ensure builder configuration is valid before step creation"
        },
        "_get_inputs": {
            "signature": "_get_inputs(self, inputs: Dict[str, Any]) -> Any",
            "description": "Get inputs for the step (step-type specific return type)",
            "return_type": "Any",  # Varies by step type
            "required": True,
            "abstract": True,
            "purpose": "Transform logical inputs to step-specific input format"
        },
        "_get_outputs": {
            "signature": "_get_outputs(self, outputs: Dict[str, Any]) -> Any", 
            "description": "Get outputs for the step (step-type specific return type)",
            "return_type": "Any",  # Varies by step type
            "required": True,
            "abstract": True,
            "purpose": "Transform logical outputs to step-specific output format"
        },
        "create_step": {
            "signature": "create_step(self, **kwargs: Any) -> Step",
            "description": "Create pipeline step",
            "return_type": "Step",
            "required": True,
            "abstract": True,
            "common_kwargs": ["dependencies", "enable_caching", "inputs", "outputs"],
            "purpose": "Create the actual SageMaker pipeline step"
        }
    },
    "inherited_methods": {
        "_get_environment_variables": {
            "signature": "_get_environment_variables(self) -> Dict[str, str]",
            "description": "Create environment variables from script contract",
            "return_type": "Dict[str, str]",
            "inherited_from": "StepBuilderBase",
            "can_override": True,
            "purpose": "Generate environment variables for step execution"
        },
        "_get_job_arguments": {
            "signature": "_get_job_arguments(self) -> Optional[List[str]]",
            "description": "Constructs command-line arguments from script contract",
            "return_type": "Optional[List[str]]",
            "inherited_from": "StepBuilderBase",
            "can_override": True,
            "purpose": "Generate command-line arguments for script execution"
        },
        "_get_cache_config": {
            "signature": "_get_cache_config(self, enable_caching: bool = True) -> CacheConfig",
            "description": "Get cache configuration for step",
            "return_type": "CacheConfig",
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "purpose": "Configure step caching behavior"
        },
        "_generate_job_name": {
            "signature": "_generate_job_name(self, step_type: Optional[str] = None) -> str",
            "description": "Generate standardized job name for SageMaker jobs",
            "return_type": "str",
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "purpose": "Create unique, valid SageMaker job names"
        },
        "_get_step_name": {
            "signature": "_get_step_name(self, include_job_type: bool = True) -> str",
            "description": "Get standard step name from builder class name",
            "return_type": "str",
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "purpose": "Extract step name from builder class for registry lookup"
        }
    },
    "required_constructor_params": {
        "config": {
            "type": "BasePipelineConfig",
            "description": "Step configuration object",
            "required": True,
            "purpose": "Provide step-specific configuration"
        },
        "spec": {
            "type": "Optional[StepSpecification]",
            "description": "Step specification for specification-driven implementation",
            "required": False,
            "purpose": "Enable specification-driven step creation"
        },
        "sagemaker_session": {
            "type": "Optional[PipelineSession]",
            "description": "SageMaker session",
            "required": False,
            "purpose": "Manage AWS SageMaker interactions"
        },
        "role": {
            "type": "Optional[str]",
            "description": "IAM role ARN",
            "required": False,
            "purpose": "Provide AWS execution permissions"
        }
    },
    "validation_rules": {
        "inheritance": {
            "must_inherit_from": "StepBuilderBase",
            "description": "All step builders must inherit from StepBuilderBase"
        },
        "method_signatures": {
            "validate_parameter_names": True,
            "validate_return_types": True,
            "validate_parameter_types": True,
            "description": "Validate method signatures match expected patterns"
        },
        "abstract_methods": {
            "must_implement_all": True,
            "description": "All abstract methods from base class must be implemented"
        }
    }
}
```

### **Universal Method Validation Examples**

```python
# ✅ CORRECT: Proper method implementation
class ExampleStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        """Validate configuration requirements."""
        if not hasattr(self.config, 'required_field'):
            raise ValueError("Missing required_field")
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for processing step."""
        return [ProcessingInput(source=inputs['data'], destination='/opt/ml/processing/input')]
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for processing step."""
        return [ProcessingOutput(source='/opt/ml/processing/output', destination=outputs['result'])]
    
    def create_step(self, **kwargs: Any) -> ProcessingStep:
        """Create processing step."""
        return ProcessingStep(name="example-step", processor=self._create_processor())

# ❌ INCORRECT: Missing required methods
class BadStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        pass
    # Missing _get_inputs, _get_outputs, create_step - VALIDATION ERROR
```

## Ruleset 2: Step-Type-Specific Method Requirements

Each SageMaker step type requires additional specific methods beyond the universal requirements:

### **Step-Type-Specific Rules Registry**

```python
STEP_TYPE_SPECIFIC_VALIDATION_RULES = {
    "TrainingStep": {
        "required_methods": {
            "_create_estimator": {
                "signature": "_create_estimator(self, output_path=None) -> Estimator",
                "description": "Create SageMaker Estimator instance",
                "return_type": "Estimator",  # XGBoost, PyTorch, etc.
                "required": True,
                "purpose": "Create the estimator that defines training job configuration"
            }
        },
        "_get_inputs_return_type": "Dict[str, TrainingInput]",
        "_get_outputs_return_type": "str",  # Output path
        "sagemaker_step_class": "TrainingStep",
        "common_estimator_types": ["XGBoost", "PyTorch", "TensorFlow", "SKLearn"],
        "validation_specifics": {
            "inputs_must_be_training_channels": True,
            "outputs_must_be_s3_path": True,
            "estimator_must_have_role": True
        }
    },
    "ProcessingStep": {
        "required_methods": {
            "_create_processor": {
                "signature": "_create_processor(self) -> Processor",
                "description": "Create SageMaker Processor instance", 
                "return_type": "Processor",
                "required": True,
                "purpose": "Create the processor that defines processing job configuration"
            }
        },
        "_get_inputs_return_type": "List[ProcessingInput]",
        "_get_outputs_return_type": "List[ProcessingOutput]",
        "sagemaker_step_class": "ProcessingStep",
        "common_processor_types": ["ScriptProcessor", "FrameworkProcessor"],
        "validation_specifics": {
            "inputs_must_be_processing_inputs": True,
            "outputs_must_be_processing_outputs": True,
            "processor_must_have_role": True
        }
    },
    "TransformStep": {
        "required_methods": {
            "_create_transformer": {
                "signature": "_create_transformer(self) -> Transformer",
                "description": "Create SageMaker Transformer instance",
                "return_type": "Transformer", 
                "required": True,
                "purpose": "Create the transformer that defines batch transform job configuration"
            }
        },
        "_get_inputs_return_type": "TransformInput",
        "_get_outputs_return_type": "str",  # Output path
        "sagemaker_step_class": "TransformStep",
        "validation_specifics": {
            "inputs_must_be_transform_input": True,
            "outputs_must_be_s3_path": True,
            "transformer_must_have_model": True
        }
    },
    "CreateModelStep": {
        "required_methods": {
            "_create_model": {
                "signature": "_create_model(self) -> Model",
                "description": "Create SageMaker Model instance",
                "return_type": "Model",
                "required": True,
                "purpose": "Create the model that defines model endpoint configuration"
            }
        },
        "_get_inputs_return_type": "CreateModelInput",
        "_get_outputs_return_type": "None",  # No outputs for model creation
        "sagemaker_step_class": "CreateModelStep",
        "validation_specifics": {
            "model_must_have_role": True,
            "model_must_have_image_or_package": True
        }
    },
    "TuningStep": {
        "required_methods": {
            "_create_tuner": {
                "signature": "_create_tuner(self) -> HyperparameterTuner",
                "description": "Create SageMaker HyperparameterTuner instance",
                "return_type": "HyperparameterTuner",
                "required": True,
                "purpose": "Create the tuner that defines hyperparameter tuning job configuration"
            }
        },
        "_get_inputs_return_type": "Dict[str, TrainingInput]",
        "_get_outputs_return_type": "str",  # Output path
        "sagemaker_step_class": "TuningStep",
        "validation_specifics": {
            "tuner_must_have_estimator": True,
            "tuner_must_have_hyperparameter_ranges": True,
            "tuner_must_have_objective_metric": True
        }
    },
    "RegisterModelStep": {
        "required_methods": {
            "_create_model_package": {
                "signature": "_create_model_package(self) -> ModelPackage",
                "description": "Create SageMaker ModelPackage instance",
                "return_type": "ModelPackage",
                "required": True,
                "purpose": "Create the model package for registration"
            }
        },
        "_get_inputs_return_type": "Dict[str, Any]",
        "_get_outputs_return_type": "None",
        "sagemaker_step_class": "RegisterModel",
        "validation_specifics": {
            "model_package_must_have_group": True,
            "model_package_must_have_inference_spec": True
        }
    }
}
```

### **Step-Type-Specific Validation Examples**

```python
# ✅ CORRECT: Training step with required _create_estimator method
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        # Universal method
        pass
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        # Universal method - returns TrainingInput dict for training steps
        return {"train": TrainingInput(s3_data=inputs["training_data"])}
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        # Universal method - returns S3 path for training steps
        return outputs.get("model_output", "/default/path")
    
    def _create_estimator(self, output_path=None) -> XGBoost:
        # Step-type-specific method for TrainingStep
        return XGBoost(
            entry_point="train.py",
            role=self.role,
            instance_type="ml.m5.large"
        )
    
    def create_step(self, **kwargs: Any) -> TrainingStep:
        # Universal method
        estimator = self._create_estimator()
        return TrainingStep(name="xgboost-training", estimator=estimator)

# ✅ CORRECT: Processing step with required _create_processor method
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        # Universal method
        pass
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        # Universal method - returns ProcessingInput list for processing steps
        return [ProcessingInput(source=inputs["raw_data"], destination="/opt/ml/processing/input")]
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        # Universal method - returns ProcessingOutput list for processing steps
        return [ProcessingOutput(source="/opt/ml/processing/output", destination=outputs["processed_data"])]
    
    def _create_processor(self) -> ScriptProcessor:
        # Step-type-specific method for ProcessingStep
        return ScriptProcessor(
            command=["python3"],
            image_uri="python:3.8",
            role=self.role,
            instance_type="ml.m5.large"
        )
    
    def create_step(self, **kwargs: Any) -> ProcessingStep:
        # Universal method
        processor = self._create_processor()
        return ProcessingStep(name="preprocessing", processor=processor)

# ❌ INCORRECT: Training step missing _create_estimator method
class BadTrainingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        pass
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        return {}
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        return "/path"
    
    def create_step(self, **kwargs: Any) -> TrainingStep:
        # Missing _create_estimator method - VALIDATION ERROR
        return TrainingStep(name="bad-training", estimator=None)
```

## Registry-Compatible Validation Structure

### **Complete Validation Registry**

```python
BUILDER_VALIDATION_REGISTRY = {
    "version": "1.0.0",
    "universal_rules": UNIVERSAL_BUILDER_VALIDATION_RULES,
    "step_type_rules": STEP_TYPE_SPECIFIC_VALIDATION_RULES,
    "retrieval_methods": {
        "get_universal_rules": "lambda: BUILDER_VALIDATION_REGISTRY['universal_rules']",
        "get_step_type_rules": "lambda step_type: BUILDER_VALIDATION_REGISTRY['step_type_rules'].get(step_type, {})",
        "get_all_required_methods": "lambda step_type: {**universal_rules['required_methods'], **step_type_rules.get(step_type, {}).get('required_methods', {})}",
        "validate_builder_interface": "lambda builder_class, step_type: validate_builder_methods(builder_class, step_type)"
    },
    "integration_points": {
        "step_catalog": "StepCatalogBuilderValidator",
        "registry_manager": "RegistryBuilderValidator", 
        "universal_tester": "UniversalBuilderTester"
    }
}
```

### **Step Catalog Integration**

```python
class StepCatalogBuilderValidator:
    """Validates step builders using step catalog for step type detection."""
    
    def __init__(self, step_catalog):
        self.step_catalog = step_catalog
        self.validation_rules = BUILDER_VALIDATION_REGISTRY
    
    def validate_builder_by_step_name(self, step_name: str, builder_class: type) -> List[ValidationIssue]:
        """Validate builder using step catalog to determine step type."""
        # Get step info from catalog
        step_info = self.step_catalog.get_step_info(step_name)
        if not step_info:
            return [ValidationIssue("ERROR", "unknown_step", f"Step {step_name} not found in catalog")]
        
        # Get SageMaker step type from registry
        from cursus.registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        # Validate using appropriate ruleset
        return self.validate_builder_interface(builder_class, sagemaker_step_type)
    
    def validate_builder_interface(self, builder_class: type, step_type: str) -> List[ValidationIssue]:
        """Validate builder implements required interface."""
        issues = []
        
        # Check universal methods
        universal_rules = self.validation_rules["universal_rules"]["required_methods"]
        for method_name, method_spec in universal_rules.items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    "ERROR", "missing_universal_method",
                    f"Builder missing required method: {method_name}",
                    f"Implement {method_name} with signature: {method_spec['signature']}"
                ))
        
        # Check step-type-specific methods
        step_rules = self.validation_rules["step_type_rules"].get(step_type, {})
        for method_name, method_spec in step_rules.get("required_methods", {}).items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    "ERROR", "missing_step_type_method", 
                    f"{step_type} builder missing required method: {method_name}",
                    f"Implement {method_name} with signature: {method_spec['signature']}"
                ))
        
        return issues
    
    def get_validation_rules_for_step(self, step_name: str) -> Dict[str, Any]:
        """Get complete validation rules for a specific step."""
        from cursus.registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        universal_rules = self.validation_rules["universal_rules"]
        step_type_rules = self.validation_rules["step_type_rules"].get(sagemaker_step_type, {})
        
        return {
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "universal_rules": universal_rules,
            "step_type_rules": step_type_rules,
            "all_required_methods": {
                **universal_rules["required_methods"],
                **step_type_rules.get("required_methods", {})
            }
        }
```

## Implementation Strategy

### **Phase 1: Method Existence Validation**

```python
class BuilderMethodValidator:
    """Validates that builders implement required methods."""
    
    def validate_builder_methods(self, builder_class: type, step_type: str) -> List[ValidationIssue]:
        """Validate that builder implements required methods."""
        issues = []
        
        # Check universal methods
        universal_rules = BUILDER_VALIDATION_REGISTRY["universal_rules"]["required_methods"]
        for method_name, method_spec in universal_rules.items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="missing_required_method",
                    message=f"Builder missing required method: {method_name}",
                    recommendation=f"Implement {method_name} with signature: {method_spec['signature']}",
                    details={
                        "method_name": method_name,
                        "expected_signature": method_spec['signature'],
                        "purpose": method_spec['purpose']
                    }
                ))
        
        # Check step-type-specific methods
        step_rules = BUILDER_VALIDATION_REGISTRY["step_type_rules"].get(step_type, {})
        for method_name, method_spec in step_rules.get("required_methods", {}).items():
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    severity="ERROR", 
                    category="missing_step_type_method",
                    message=f"{step_type} builder missing required method: {method_name}",
                    recommendation=f"Implement {method_name} with signature: {method_spec['signature']}",
                    details={
                        "step_type": step_type,
                        "method_name": method_name,
                        "expected_signature": method_spec['signature'],
                        "purpose": method_spec['purpose']
                    }
                ))
        
        return issues
```

### **Phase 2: Method Signature Validation**

```python
class MethodSignatureValidator:
    """Validates method signatures match expected patterns."""
    
    def validate_method_signatures(self, builder_class: type, step_type: str) -> List[ValidationIssue]:
        """Validate method signatures match expected patterns."""
        issues = []
        
        # Get all required methods for this step type
        all_methods = self._get_all_required_methods(step_type)
        
        for method_name, method_spec in all_methods.items():
            if hasattr(builder_class, method_name):
                method = getattr(builder_class, method_name)
                signature = inspect.signature(method)
                
                # Validate signature matches expected pattern
                expected_signature = method_spec["signature"]
                if not self._signatures_match(signature, expected_signature):
                    issues.append(ValidationIssue(
                        severity="ERROR",
                        category="invalid_method_signature",
                        message=f"Method {method_name} has incorrect signature",
                        recommendation=f"Update signature to match: {expected_signature}",
                        details={
                            "method_name": method_name,
                            "actual_signature": str(signature),
                            "expected_signature": expected_signature
                        }
                    ))
        
        return issues
    
    def _get_all_required_methods(self, step_type: str) -> Dict[str, Any]:
        """Get all required methods for a step type."""
        universal_methods = BUILDER_VALIDATION_REGISTRY["universal_rules"]["required_methods"]
        step_type_methods = BUILDER_VALIDATION_REGISTRY["step_type_rules"].get(step_type, {}).get("required_methods", {})
        
        return {**universal_methods, **step_type_methods}
    
    def _signatures_match(self, actual_signature: inspect.Signature, expected_signature_str: str) -> bool:
        """Check if actual signature matches expected signature string."""
        # Implementation would parse expected_signature_str and compare with actual_signature
        # This is a simplified version - full implementation would be more robust
        return True  # Placeholder
```

### **Phase 3: Interface Consistency Validation**

```python
class InterfaceConsistencyValidator:
    """Validates that all builders implement consistent interfaces."""
    
    def validate_interface_consistency(self, builder_classes: List[type]) -> List[ValidationIssue]:
        """Validate that all builders implement consistent interfaces."""
        issues = []
        
        # Check that all builders have the same universal methods
        universal_methods = set(BUILDER_VALIDATION_REGISTRY["universal_rules"]["required_methods"].keys())
        
        for builder_class in builder_classes:
            builder_methods = set(method for method in dir(builder_class) 
                                if not method.startswith('__') and callable(getattr(builder_class, method)))
            
            missing_methods = universal_methods - builder_methods
            for missing_method in missing_methods:
                issues.append(ValidationIssue(
                    severity="ERROR",
                    category="interface_inconsistency", 
                    message=f"Builder {builder_class.__name__} missing universal method: {missing_method}",
                    recommendation=f"Implement {missing_method} to maintain interface consistency",
                    details={
                        "builder_class": builder_class.__name__,
                        "missing_method": missing_method,
                        "method_type": "universal"
                    }
                ))
        
        return issues
```

## Validation Configuration

### **Configurable Validation Levels**

```python
STEP_BUILDER_VALIDATION_CONFIG = {
    "validation_levels": {
        "method_existence": {
            "enabled": True,
            "severity": "ERROR",
            "description": "Validate that required methods exist"
        },
        "method_signatures": {
            "enabled": True,
            "severity": "ERROR", 
            "description": "Validate method signatures match expected patterns"
        },
        "interface_consistency": {
            "enabled": True,
            "severity": "WARNING",
            "description": "Validate consistent interfaces across builders"
        },
        "inheritance_validation": {
            "enabled": True,
            "severity": "ERROR",
            "description": "Validate proper inheritance from StepBuilderBase"
        },
        "return_type_validation": {
            "enabled": False,  # Optional - can be strict
            "severity": "WARNING",
            "description": "Validate method return types match specifications"
        }
    },
    "validation_modes": {
        "strict": ["method_existence", "method_signatures", "interface_consistency", "inheritance_validation", "return_type_validation"],
        "standard": ["method_existence", "method_signatures", "interface_consistency", "inheritance_validation"],
        "minimal": ["method_existence", "inheritance_validation"]
    },
    "step_type_specific_validation": {
        "enabled": True,
        "description": "Apply step-type-specific validation rules based on SageMaker step type"
    }
}
```

## Usage Examples

### **Example 1: Validating a Training Step Builder**

```python
# Validate XGBoost training step builder
validator = StepCatalogBuilderValidator(step_catalog)
issues = validator.validate_builder_by_step_name("XGBoostTraining", XGBoostTrainingStepBuilder)

# Expected validation checks:
# ✅ Universal methods: validate_configuration, _get_inputs, _get_outputs, create_step
# ✅ Training-specific method: _create_estimator
# ✅ Method signatures match expected patterns
# ✅ Inherits from StepBuilderBase
```

### **Example 2: Validating a Processing Step Builder**

```python
# Validate tabular preprocessing step builder
validator = StepCatalogBuilderValidator(step_catalog)
issues = validator.validate_builder_by_step_name("TabularPreprocessing", TabularPreprocessingStepBuilder)

# Expected validation checks:
# ✅ Universal methods: validate_configuration, _get_inputs, _get_outputs, create_step
# ✅ Processing-specific method: _create_processor
# ✅ _get_inputs returns List[ProcessingInput]
# ✅ _get_outputs returns List[ProcessingOutput]
```

### **Example 3: Getting Validation Rules for a Step**

```python
# Get complete validation rules for a specific step
validator = StepCatalogBuilderValidator(step_catalog)
rules = validator.get_validation_rules_for_step("XGBoostTraining")

print(f"Step: {rules['step_name']}")
print(f"SageMaker Type: {rules['sagemaker_step_type']}")
print(f"Required Methods: {list(rules['all_required_methods'].keys())}")

# Output:
# Step: XGBoostTraining
# SageMaker Type: TrainingStep
# Required Methods: ['validate_configuration', '_get_inputs', '_get_outputs', 'create_step', '_create_estimator']
```

## Benefits of This Approach

### **✅ Advantages:**

#### **1. Structural Focus**
- **Method-Centric**: Validates architecture, not implementation details
- **Interface Consistency**: Ensures all builders follow same patterns
- **Builder Autonomy**: Let builders handle their own config validation

#### **2. Registry Integration**
- **Easy Retrieval**: Rules easily accessible via step catalog and registry
- **Automatic Detection**: Step type automatically determined from registry
- **Extensible**: Easy to add new step types or modify existing rules

#### **3. Maintainable Validation**
- **Less Brittle**: Not dependent on config field names or implementation details
- **Clear Requirements**: Developers know exactly what methods to implement
- **Targeted Feedback**: Specific recommendations for missing methods

#### **4. Step-Type Awareness**
- **Appropriate Rules**: Different validation for different SageMaker step types
- **Comprehensive Coverage**: Both universal and step-specific requirements
- **Practical Implementation**: Focus on what actually matters for step functionality

### **❌ What This Approach Avoids:**

#### **1. Config Field Validation Tedium**
- **No Field Checking**: Don't validate individual config fields - builders handle this
- **No Field Name Variants**: Don't worry about config vs SageMaker parameter name differences
- **No Implementation Details**: Don't validate script patterns or framework-specific code

#### **2. Overly Strict Validations**
- **No Script Patterns**: Don't validate specific function names like `model_fn`, `predict_fn`
- **No Framework Specifics**: Don't validate XGBoost vs PyTorch implementation details
- **No Container Internals**: Don't validate inference code structure or model serialization

## Migration Strategy

### **From Current Validation to Method-Centric Approach**

#### **Before (Field-Centric Validation)**
```python
# Old approach - validate config fields
def validate_training_step_config(config):
    issues = []
    
    # ❌ TOO DETAILED - Builder handles this
    if not hasattr(config, 'training_instance_type'):
        issues.append("Missing training_instance_type")
    
    if not hasattr(config, 'framework_version'):
        issues.append("Missing framework_version")
    
    # ❌ TOO STRICT - Implementation detail
    if not hasattr(config, 'hyperparameters'):
        issues.append("Missing hyperparameters")
    
    return issues
```

#### **After (Method-Centric Validation)**
```python
# New approach - validate method interface
def validate_training_step_builder(builder_class):
    issues = []
    
    # ✅ CORRECT - Validate structural requirements
    if not hasattr(builder_class, 'validate_configuration'):
        issues.append("Missing validate_configuration method")
    
    if not hasattr(builder_class, '_create_estimator'):
        issues.append("Missing _create_estimator method")
    
    if not hasattr(builder_class, 'create_step'):
        issues.append("Missing create_step method")
    
    return issues
```

### **Migration Steps**

#### **Phase 1: Implement Method Validators**
1. **Create Method Validators**: Implement `BuilderMethodValidator`, `MethodSignatureValidator`
2. **Registry Integration**: Connect validators to step catalog and registry systems
3. **Test Framework**: Update universal builder test to use method validation

#### **Phase 2: Update Existing Validation**
1. **Replace Field Validation**: Remove config field checking from alignment validators
2. **Focus on Interface**: Update validation to check method existence and signatures
3. **Maintain Compatibility**: Ensure existing builders pass new validation

#### **Phase 3: Documentation and Training**
1. **Update Developer Guides**: Document new validation approach
2. **Create Examples**: Provide clear examples of correct builder implementation
3. **Migration Guide**: Help developers understand the changes

## Integration with Existing Systems

### **Alignment Validation Integration**

```python
# Enhanced alignment validator using method-centric approach
class EnhancedBuilderAlignmentValidator:
    """Validates builder alignment using method interface validation."""
    
    def __init__(self, step_catalog):
        self.step_catalog = step_catalog
        self.method_validator = StepCatalogBuilderValidator(step_catalog)
    
    def validate_builder_alignment(self, step_name: str, builder_class: type) -> AlignmentResult:
        """Validate builder alignment using method interface validation."""
        issues = []
        
        # Method interface validation (primary focus)
        method_issues = self.method_validator.validate_builder_by_step_name(step_name, builder_class)
        issues.extend(method_issues)
        
        # Specification alignment (if available)
        if hasattr(builder_class, 'spec') and builder_class.spec:
            spec_issues = self._validate_specification_alignment(builder_class)
            issues.extend(spec_issues)
        
        # Contract alignment (if available)
        if hasattr(builder_class, 'contract') and builder_class.contract:
            contract_issues = self._validate_contract_alignment(builder_class)
            issues.extend(contract_issues)
        
        return AlignmentResult(
            is_valid=len([i for i in issues if i.severity == "ERROR"]) == 0,
            issues=issues,
            validation_approach="method_interface_centric"
        )
```

### **Universal Builder Test Integration**

```python
# Enhanced universal builder test using method validation
class EnhancedUniversalBuilderTest:
    """Universal builder test enhanced with method interface validation."""
    
    def __init__(self, builder_class: Type[StepBuilderBase]):
        self.builder_class = builder_class
        self.step_name = self._detect_step_name()
        self.method_validator = StepCatalogBuilderValidator(step_catalog)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation including method interface checks."""
        results = {}
        
        # Method interface validation (primary)
        results["method_interface"] = self._validate_method_interface()
        
        # Step creation validation
        results["step_creation"] = self._validate_step_creation()
        
        # Configuration validation (builder's responsibility)
        results["configuration"] = self._validate_configuration_handling()
        
        # Integration validation
        results["integration"] = self._validate_integration_capabilities()
        
        return results
    
    def _validate_method_interface(self) -> Dict[str, Any]:
        """Validate method interface compliance."""
        issues = self.method_validator.validate_builder_by_step_name(
            self.step_name, self.builder_class
        )
        
        return {
            "passed": len([i for i in issues if i.severity == "ERROR"]) == 0,
            "issues": issues,
            "validation_type": "method_interface"
        }
```

## Future Enhancements

### **1. Dynamic Method Detection**
```python
# Automatically detect required methods from SageMaker step type
class DynamicMethodDetector:
    """Dynamically detect required methods based on SageMaker step analysis."""
    
    def detect_required_methods(self, sagemaker_step_class: type) -> Dict[str, Any]:
        """Analyze SageMaker step class to determine required builder methods."""
        # Analyze step constructor parameters
        # Determine what builder methods are needed
        # Generate validation rules dynamically
        pass
```

### **2. Method Signature Analysis**
```python
# Advanced signature validation with type checking
class AdvancedSignatureValidator:
    """Advanced method signature validation with type analysis."""
    
    def validate_signature_compatibility(self, method: callable, expected_spec: Dict) -> bool:
        """Validate method signature compatibility with advanced type checking."""
        # Use typing module for advanced type validation
        # Check parameter types, return types, generic types
        # Validate compatibility with SageMaker types
        pass
```

### **3. Performance Optimization**
```python
# Cached validation results for performance
class CachedMethodValidator:
    """Cached method validator for improved performance."""
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_with_cache(self, builder_class: type, step_type: str) -> List[ValidationIssue]:
        """Validate with caching for improved performance."""
        cache_key = f"{builder_class.__name__}:{step_type}"
        
        if cache_key not in self.validation_cache:
            self.validation_cache[cache_key] = self._perform_validation(builder_class, step_type)
        
        return self.validation_cache[cache_key]
```

## Conclusion

This design document establishes a comprehensive **method-centric validation approach** for step builders that:

### **Key Achievements:**

1. **Focuses on What Matters**: Validates method interfaces that actually affect functionality
2. **Reduces Validation Overhead**: Eliminates tedious config field checking
3. **Improves Developer Experience**: Clear requirements and targeted feedback
4. **Enables Registry Integration**: Easy retrieval via step catalog and registry systems
5. **Supports Step-Type Awareness**: Appropriate validation rules for different SageMaker step types

### **Implementation Benefits:**

- **Maintainable**: Less brittle than config field validation
- **Extensible**: Easy to add new step types or modify existing rules
- **Practical**: Based on actual builder implementation patterns
- **Comprehensive**: Covers both universal and step-specific requirements

### **Migration Path:**

The design provides a clear migration path from current field-centric validation to method-centric validation, with backward compatibility and integration with existing systems.

This approach will significantly improve the validation system's effectiveness while reducing maintenance overhead and providing better developer experience.

## References

- [SageMaker Step Validation Requirements Specification](sagemaker_step_validation_requirements_specification.md) - Comprehensive requirements for SageMaker step validation
- [SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md) - Step type classification system for registry integration
- [Universal Step Builder Test](../0_developer_guide/validation_framework_guide.md) - Universal testing framework for step builders
- [Step Builder Base Class](../../src/cursus/core/base/builder_base.py) - Base class defining universal builder interface
- [Step Registry Design](../registry/step_names.py) - Registry system for step name and type management
