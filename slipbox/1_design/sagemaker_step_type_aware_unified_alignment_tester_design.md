---
tags:
  - design
  - validation
  - alignment_tester
  - sagemaker_integration
  - step_type_classification
keywords:
  - unified alignment tester
  - SageMaker step types
  - step type awareness
  - alignment validation
  - training steps
  - processing steps
  - framework detection
  - validation variants
topics:
  - alignment validation framework
  - SageMaker step type integration
  - step builder validation
  - training step support
language: python
date of note: 2025-08-13
---

# SageMaker Step Type-Aware Unified Alignment Tester Design

## Overview

This document presents a comprehensive design for transforming the unified alignment tester into a **SageMaker step type-aware validation framework**. The enhanced system will provide targeted, expert-level validation for each type of SageMaker step while maintaining the proven four-tier alignment validation architecture.

## Problem Statement

The current unified alignment tester achieves **100% success rate** across all 8 processing scripts through a four-tier validation pyramid. However, it was designed primarily for **Processing steps** and lacks the specificity needed for different SageMaker step types across the entire step ecosystem.

### Current Limitations

1. **Single Step Type Focus**: Designed primarily for processing steps using `ProcessingStep` instances
2. **Processing-Centric Patterns**: Script analysis focuses on processing-specific patterns
3. **Limited Framework Awareness**: No detection of training frameworks (XGBoost, PyTorch, etc.)
4. **Generic Validation**: Same validation logic applied to all step types regardless of their unique requirements
5. **Incomplete Step Type Coverage**: Missing validation for 6 out of 7 SageMaker step types

### Complete SageMaker Step Type Ecosystem

Based on the SageMaker step type classification, we need to support **7 distinct step types** with **17 step builders**:

#### **Processing Steps (9 builders)**
- **Step Builders**: TabularPreprocessing, RiskTableMapping, CurrencyConversion, XGBoostModelEval, ModelCalibration, Package, Payload, CradleDataLoading
- **SageMaker Type**: `ProcessingStep` instances using processors (`SKLearnProcessor`, `XGBoostProcessor`)
- **Input/Output**: `ProcessingInput`/`ProcessingOutput` objects with explicit source/destination mapping
- **Data Handling**: Direct file-to-container path mapping
- **Configuration**: Environment variables and job arguments

#### **Training Steps (3 builders)**
- **Step Builders**: XGBoostTraining, PyTorchTraining, DummyTraining
- **SageMaker Type**: `TrainingStep` instances using framework-specific estimators
- **Input/Output**: `TrainingInput` objects with channel-based data organization
- **Data Handling**: Channel-based approach ("train", "val", "test", "config" channels)
- **Configuration**: Hyperparameters uploaded as files to S3 and passed via input channels

#### **CreateModel Steps (2 builders)**
- **Step Builders**: XGBoostModel, PyTorchModel
- **SageMaker Type**: `CreateModelStep` instances for model endpoint creation
- **Input/Output**: Model artifacts input, model endpoint output
- **Data Handling**: Model artifact loading and inference code preparation
- **Configuration**: Container definitions and inference specifications

#### **Transform Steps (1 builder)**
- **Step Builders**: BatchTransform
- **SageMaker Type**: `TransformStep` instances for batch inference
- **Input/Output**: `TransformInput` objects and transform results
- **Data Handling**: Batch processing with model loading for inference
- **Configuration**: Transform job specifications and resource allocation

#### **RegisterModel Steps (1 builder)**
- **Step Builders**: Registration
- **SageMaker Type**: `RegisterModel` instances for model registry
- **Input/Output**: Model artifacts input, registered model output
- **Data Handling**: Model metadata preparation and approval workflow
- **Configuration**: Model package specifications and registry settings

#### **Utility Steps (1 builder)**
- **Step Builders**: HyperparameterPrep
- **SageMaker Type**: No direct SageMaker step (utility functions)
- **Input/Output**: Various inputs, prepared files output
- **Data Handling**: File preparation and parameter generation
- **Configuration**: Preparation specifications and file formatting

#### **Base Steps (2 builders)**
- **Step Builders**: Base, Processing (base classes)
- **SageMaker Type**: Foundation for other step types
- **Input/Output**: Base input/output patterns
- **Data Handling**: Foundation patterns and inheritance structures
- **Configuration**: Base configuration patterns

## Proposed Solution: Maximum Code Reuse with Step Type Awareness

### **Core Architecture Enhancement**

Enhance the existing unified alignment tester with **minimal, additive step type awareness** while preserving 95%+ of existing code:

```
Enhanced Unified Alignment Tester (Existing + Step Type Awareness)
├── UnifiedAlignmentTester (Enhanced, not replaced)
│   ├── Existing 4-tier validation (100% preserved)
│   ├── Step type detection (minimal addition)
│   └── Step type-specific enhancements (additive)
├── Existing Level Testers (Enhanced, not replaced)
│   ├── ScriptContractAlignmentTester + training patterns
│   ├── ContractSpecificationAlignmentTester + step type context
│   ├── SpecificationDependencyAlignmentTester + framework awareness
│   └── BuilderConfigurationAlignmentTester + step type validation
└── Existing Infrastructure (100% reused)
    ├── All existing analyzers, validators, utilities
    ├── All existing data structures (extended, not replaced)
    └── All existing test infrastructure
```

### **Registry-Driven Step Type Detection (Minimal Addition)**

Leverage the existing step registry with a simple detection function:

```python
# Add to existing alignment_utils.py (20 lines)
def detect_step_type_from_registry(script_name: str) -> str:
    """Use existing step registry to determine SageMaker step type"""
    from cursus.steps.registry.step_names import get_sagemaker_step_type, get_canonical_name_from_file_name
    try:
        canonical_name = get_canonical_name_from_file_name(script_name)
        return get_sagemaker_step_type(canonical_name)
    except ValueError:
        return "Processing"  # Default fallback

# Enhance existing UnifiedAlignmentTester (30 lines)
class UnifiedAlignmentTester:
    def __init__(self, ...):
        # All existing initialization remains the same
        self.enable_step_type_awareness = True  # Feature flag
        
    def _run_level1_validation(self, target_scripts):
        # Keep existing logic, just add step type context
        for script_name, result in results.items():
            if self.enable_step_type_awareness:
                step_type = detect_step_type_from_registry(script_name)
                # Add step type context to existing issues
                for issue in result.get('issues', []):
                    if isinstance(issue, dict):
                        issue['step_type'] = step_type
```

## Step Type-Specific Four-Tier Validation

Each SageMaker step type variant implements the same four-tier architecture but with **step type-specific validation logic**:

### **Level 1: Script ↔ Contract Alignment (Step Type-Aware)**

#### **Processing Steps**
- Validate processing script patterns (data loading, transformation, output saving)
- Check environment variable usage and job argument handling
- Verify processor-compatible script structure

#### **Training Steps**
- Validate training script patterns (model training loops, evaluation, artifact saving)
- Check hyperparameter loading from JSON files
- Verify model saving to `/opt/ml/model/`
- Validate evaluation output to `/opt/ml/output/data/`

#### **Transform Steps**
- Validate transform script patterns (batch processing, inference logic)
- Check model loading and inference implementation
- Verify batch processing capabilities

#### **CreateModel Steps**
- Validate model creation patterns (inference code, model loading)
- Check container compatibility and inference script structure
- Verify model artifact handling

### **Level 2: Contract ↔ Specification Alignment (Step Type-Aware)**

#### **Processing Steps**
- Validate `ProcessingInput`/`ProcessingOutput` specifications
- Check processing-specific dependency types
- Verify processing output specifications

#### **Training Steps**
- Validate `TrainingInput` specifications and channel-based data organization
- Check training-specific dependency types (`TRAINING_DATA`, `HYPERPARAMETERS`)
- Verify model artifact output specifications (`MODEL_ARTIFACTS`)
- Validate training-specific property paths

#### **Transform Steps**
- Validate `TransformInput` specifications and batch processing requirements
- Check transform-specific dependency types
- Verify transform output specifications

#### **CreateModel Steps**
- Validate model specifications and container definitions
- Check model artifact dependency requirements
- Verify model creation output specifications

### **Level 3: Specification ↔ Dependencies Alignment (Step Type-Aware)**

#### **Processing Steps**
- Resolve processing data dependencies and output artifacts
- Validate processing-specific semantic keywords
- Check processing dependency chains

#### **Training Steps**
- Resolve training data, hyperparameters, and model artifact dependencies
- Validate training-specific semantic keywords ("training", "model", "estimator", "hyperparameters")
- Check training dependency chains and model artifact flows

#### **Transform Steps**
- Resolve model dependencies and transform data sources
- Validate transform-specific semantic keywords
- Check transform dependency chains

#### **CreateModel Steps**
- Resolve model artifact dependencies and inference requirements
- Validate model creation semantic keywords
- Check model dependency chains

### **Level 4: Builder ↔ Configuration Alignment (Step Type-Aware)**

#### **Processing Steps**
- Validate processor creation (`_create_processor` methods)
- Check `ProcessingStep` instantiation patterns
- Verify processing-specific configuration classes

#### **Training Steps**
- Validate estimator creation (`_create_estimator` methods)
- Check `TrainingStep` instantiation patterns
- Verify training-specific configuration classes
- Validate hyperparameter file preparation (`_prepare_hyperparameters_file`)
- Check training input channel creation (`_create_data_channels_from_source`)

#### **Transform Steps**
- Validate transformer creation (`_create_transformer` methods)
- Check `TransformStep` instantiation patterns
- Verify transform-specific configuration classes

#### **CreateModel Steps**
- Validate model creation (`_create_model` methods)
- Check `CreateModelStep` instantiation patterns
- Verify model-specific configuration classes

## Enhanced Data Structures for Step Type Awareness

### **Step Type-Specific Script Analysis**

```python
@dataclass
class StepTypeAwareScriptAnalysis(ScriptAnalysis):
    """Enhanced script analysis with step type-specific patterns"""
    
    step_type: str                           # Detected SageMaker step type
    step_type_patterns: Dict[str, List[str]] # Step type-specific patterns found
    framework_usage: Optional[str]          # Framework detection (XGBoost, PyTorch, etc.)
    
    # Processing-specific
    processor_patterns: List[str] = field(default_factory=list)
    processing_job_patterns: List[str] = field(default_factory=list)
    
    # Training-specific  
    estimator_patterns: List[str] = field(default_factory=list)
    training_loop_patterns: List[str] = field(default_factory=list)
    model_saving_patterns: List[str] = field(default_factory=list)
    hyperparameter_loading_patterns: List[str] = field(default_factory=list)
    
    # Transform-specific
    transformer_patterns: List[str] = field(default_factory=list)
    batch_processing_patterns: List[str] = field(default_factory=list)
    
    # CreateModel-specific
    model_creation_patterns: List[str] = field(default_factory=list)
    inference_patterns: List[str] = field(default_factory=list)
```

### **Step Type-Specific Validation Issues**

```python
@dataclass
class StepTypeAwareValidationIssue(ValidationIssue):
    """Enhanced validation issue with step type context"""
    
    step_type: str                          # SageMaker step type context
    step_type_category: str                 # Step type-specific issue category
    framework_context: Optional[str]       # Framework-specific context
    reference_examples: List[str]           # Reference implementation examples
    step_type_recommendation: str           # Step type-specific recommendation
```

## Framework-Specific Validation

The system will detect and validate framework-specific patterns:

### **XGBoost Training Validation**

```python
class XGBoostTrainingValidator:
    """XGBoost-specific training validation"""
    
    def validate_xgboost_patterns(self, script_analysis, contract, spec, builder):
        """Comprehensive XGBoost training validation"""
        issues = []
        
        # Level 1: XGBoost script patterns
        issues.extend(self._validate_xgboost_script_patterns(script_analysis))
        
        # Level 2: XGBoost contract/spec alignment  
        issues.extend(self._validate_xgboost_specifications(contract, spec))
        
        # Level 3: XGBoost dependency resolution
        issues.extend(self._validate_xgboost_dependencies(spec))
        
        # Level 4: XGBoost builder configuration
        issues.extend(self._validate_xgboost_builder(builder))
        
        return issues
    
    def _validate_xgboost_script_patterns(self, script_analysis):
        """Validate XGBoost-specific script patterns"""
        issues = []
        
        # Check for XGBoost imports
        xgboost_imports = [imp for imp in script_analysis.imports 
                          if 'xgboost' in imp.module_name.lower()]
        if not xgboost_imports:
            issues.append(StepTypeAwareValidationIssue(
                severity="ERROR",
                category="xgboost_missing_import",
                step_type="Training",
                step_type_category="framework_import",
                framework_context="XGBoost",
                message="XGBoost training script missing XGBoost import",
                recommendation="Add 'import xgboost as xgb' to script",
                reference_examples=["xgboost_training.py"]
            ))
            
        # Check for DMatrix creation patterns
        dmatrix_patterns = [ref for ref in script_analysis.path_references
                           if 'DMatrix' in ref.context]
        if not dmatrix_patterns:
            issues.append(StepTypeAwareValidationIssue(
                severity="WARNING", 
                category="xgboost_missing_dmatrix",
                step_type="Training",
                step_type_category="framework_pattern",
                framework_context="XGBoost",
                message="XGBoost script should use DMatrix for data handling",
                recommendation="Use xgb.DMatrix for training data preparation",
                reference_examples=["xgboost_training.py"]
            ))
            
        return issues
```

### **PyTorch Training Validation**

```python
class PyTorchTrainingValidator:
    """PyTorch-specific training validation"""
    
    def validate_pytorch_patterns(self, script_analysis, contract, spec, builder):
        """Comprehensive PyTorch training validation"""
        issues = []
        
        # Check for PyTorch imports
        pytorch_imports = [imp for imp in script_analysis.imports 
                          if 'torch' in imp.module_name.lower()]
        if not pytorch_imports:
            issues.append(StepTypeAwareValidationIssue(
                severity="ERROR",
                category="pytorch_missing_import",
                step_type="Training",
                step_type_category="framework_import",
                framework_context="PyTorch",
                message="PyTorch training script missing PyTorch import",
                recommendation="Add 'import torch' to script",
                reference_examples=["pytorch_training.py"]
            ))
            
        # Check for model definition patterns
        model_patterns = [ref for ref in script_analysis.path_references
                         if any(pattern in ref.context.lower() 
                               for pattern in ['nn.module', 'model.train', 'model.eval'])]
        if not model_patterns:
            issues.append(StepTypeAwareValidationIssue(
                severity="WARNING",
                category="pytorch_missing_model_patterns",
                step_type="Training",
                step_type_category="framework_pattern",
                framework_context="PyTorch",
                message="PyTorch script should define and use model classes",
                recommendation="Define model class inheriting from nn.Module",
                reference_examples=["pytorch_training.py"]
            ))
            
        return issues
```

## Training Step Alignment Tester Implementation

### **Core Training Step Validator**

```python
class TrainingStepAlignmentTester(UnifiedAlignmentTester):
    """Training step-specific alignment validation"""
    
    def __init__(self):
        super().__init__()
        self.step_type = "Training"
        self.reference_examples = [
            "builder_xgboost_training_step",
            "builder_pytorch_training_step"
        ]
        self.framework_validators = {
            "xgboost": XGBoostTrainingValidator(),
            "pytorch": PyTorchTrainingValidator()
        }
    
    def _validate_level1_training_patterns(self, script_analysis):
        """Training-specific Level 1 validation"""
        issues = []
        
        # Validate training script patterns
        if not script_analysis.training_loop_patterns:
            issues.append(self._create_training_issue(
                "missing_training_loop",
                "Training script should contain model training logic",
                "Add model.fit() or equivalent training loop"
            ))
            
        # Validate hyperparameter loading
        if not script_analysis.hyperparameter_loading_patterns:
            issues.append(self._create_training_issue(
                "missing_hyperparameter_loading", 
                "Training script should load hyperparameters from file",
                "Add hyperparameter loading from /opt/ml/input/data/config/"
            ))
            
        # Validate model saving
        if not script_analysis.model_saving_patterns:
            issues.append(self._create_training_issue(
                "missing_model_saving",
                "Training script should save model artifacts",
                "Add model saving to /opt/ml/model/"
            ))
            
        # Framework-specific validation
        if script_analysis.framework_usage:
            framework_validator = self.framework_validators.get(
                script_analysis.framework_usage.lower()
            )
            if framework_validator:
                issues.extend(framework_validator.validate_script_patterns(script_analysis))
            
        return issues
    
    def _validate_level2_training_specifications(self, contract, spec):
        """Training-specific Level 2 validation"""
        issues = []
        
        # Validate TrainingInput specifications
        for dep_name, dep_spec in spec.dependencies.items():
            if dep_spec.dependency_type == DependencyType.TRAINING_DATA:
                # Validate training data channel specifications
                if not self._validate_training_data_channels(dep_spec):
                    issues.append(self._create_training_issue(
                        "invalid_training_data_spec",
                        f"Training data specification invalid for {dep_name}",
                        "Ensure training data supports train/val/test channels"
                    ))
                    
        # Validate model artifact outputs
        model_outputs = [out for out in spec.outputs.values() 
                        if out.output_type == DependencyType.MODEL_ARTIFACTS]
        if not model_outputs:
            issues.append(self._create_training_issue(
                "missing_model_output",
                "Training specification should define model artifact output",
                "Add MODEL_ARTIFACTS output specification"
            ))
            
        return issues
    
    def _validate_level3_training_dependencies(self, spec):
        """Training-specific Level 3 validation"""
        issues = []
        
        # Validate training-specific semantic keywords
        training_keywords = ["training", "model", "estimator", "hyperparameters", "artifacts"]
        
        for dep_name, dep_spec in spec.dependencies.items():
            if dep_spec.dependency_type == DependencyType.TRAINING_DATA:
                # Check for appropriate semantic keywords
                if not any(keyword in dep_spec.semantic_keywords 
                          for keyword in ["data", "input", "training", "dataset"]):
                    issues.append(self._create_training_issue(
                        "missing_training_data_keywords",
                        f"Training data dependency {dep_name} missing appropriate semantic keywords",
                        "Add semantic keywords like 'training', 'data', 'dataset'"
                    ))
                    
        return issues
    
    def _validate_level4_training_builder(self, builder_analysis):
        """Training-specific Level 4 validation"""
        issues = []
        
        # Validate estimator creation method
        if not builder_analysis.has_method("_create_estimator"):
            issues.append(self._create_training_issue(
                "missing_create_estimator",
                "Training builder should implement _create_estimator method",
                "Add _create_estimator method to create framework estimator"
            ))
            
        # Validate TrainingStep creation
        if not builder_analysis.creates_step_type("TrainingStep"):
            issues.append(self._create_training_issue(
                "wrong_step_type",
                "Training builder should create TrainingStep instances",
                "Ensure create_step method returns TrainingStep"
            ))
            
        # Validate hyperparameter file preparation
        if not builder_analysis.has_method("_prepare_hyperparameters_file"):
            issues.append(self._create_training_issue(
                "missing_hyperparameter_prep",
                "Training builder should implement hyperparameter file preparation",
                "Add _prepare_hyperparameters_file method"
            ))
            
        return issues
    
    def _create_training_issue(self, category, message, recommendation):
        """Create training-specific validation issue"""
        return StepTypeAwareValidationIssue(
            severity="ERROR",
            category=category,
            step_type="Training",
            step_type_category="training_pattern",
            message=message,
            recommendation=recommendation,
            reference_examples=self.reference_examples
        )
```

## Integration with Existing Systems

### **Registry Integration**

The enhanced system leverages existing registry infrastructure:

```python
# Use existing step registry for step type detection
from cursus.steps.registry.step_names import get_sagemaker_step_type

# Use existing step type classification
from cursus.steps.registry.step_type_test_variants import (
    STEP_TYPE_VARIANT_MAP,
    get_step_type_variant,
    get_step_type_requirements
)
```

### **Specification Integration**

The system integrates with existing specification infrastructure:

```python
# Use existing specification loading
from cursus.steps.specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
from cursus.steps.specs.pytorch_training_spec import PYTORCH_TRAINING_SPEC

# Use existing contract loading
from cursus.steps.contracts.xgboost_training_contract import XGBOOST_TRAIN_CONTRACT
from cursus.steps.contracts.pytorch_training_contract import PYTORCH_TRAIN_CONTRACT
```

### **Builder Integration**

The system integrates with existing builder infrastructure:

```python
# Use existing builder classes
from cursus.steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder
from cursus.steps.builders.builder_pytorch_training_step import PyTorchTrainingStepBuilder
```

## Implementation Strategy

### **Phase 1: Core Architecture Enhancement**
1. **Registry Integration**: Integrate with existing step registry for step type detection
2. **Variant Factory**: Implement factory pattern for creating step type-specific testers
3. **Base Variant Class**: Create abstract base class for all step type variants
4. **Data Structure Extensions**: Enhance existing data structures with step type awareness

### **Phase 2: Primary Step Type Variants**
1. **ProcessingStepAlignmentTester**: Implement comprehensive processing step validation
2. **TrainingStepAlignmentTester**: Implement training step validation (immediate priority for XGBoost)
3. **Framework Detection**: Add framework detection and framework-specific validation
4. **Reference Example Integration**: Use existing implementations as validation references

### **Phase 3: Advanced Step Type Variants**
1. **TransformStepAlignmentTester**: Implement transform step validation
2. **CreateModelStepAlignmentTester**: Implement model creation step validation
3. **Remaining Variants**: Implement all 12 SageMaker step type variants
4. **Custom Step Handling**: Handle custom step implementations with basic validation

### **Phase 4: Integration and Optimization**
1. **Unified Orchestration**: Integrate all variants into single unified system
2. **Performance Optimization**: Optimize validation performance across step types
3. **Comprehensive Testing**: Validate against all existing step implementations
4. **Documentation**: Update all design documents with step type-aware architecture

## Expected Outcomes

After implementation, the enhanced unified alignment tester will:

1. **Maintain 100% Success Rate**: Continue perfect validation for all existing scripts
2. **Step Type-Specific Validation**: Provide targeted validation for each SageMaker step type
3. **Framework-Aware Validation**: Detect and validate framework-specific patterns (XGBoost, PyTorch, etc.)
4. **Comprehensive Coverage**: Support all 12 SageMaker step types with appropriate validation
5. **Reference-Driven Quality**: Use existing standardized implementations as validation benchmarks
6. **Extensible Architecture**: Easy addition of new step types or framework-specific validators

## Benefits

### **1. Enhanced Validation Quality**
- **Step Type-Specific Rules**: Each SageMaker step type receives appropriate validation
- **Framework-Aware Validation**: Detects and validates framework-specific implementation patterns
- **Comprehensive Coverage**: Validates all aspects of step implementation from script to builder

### **2. Improved Developer Experience**
- **Targeted Feedback**: Provides step type-specific recommendations and error messages
- **Reference Examples**: Points to existing standardized implementations as examples
- **Framework Guidance**: Offers framework-specific best practices and patterns

### **3. Maintainability and Extensibility**
- **Modular Architecture**: Clear separation of concerns with step type variants
- **Easy Extension**: Simple addition of new step types or framework validators
- **Backward Compatibility**: Existing validation continues to work without modification

### **4. Quality Assurance**
- **Comprehensive Testing**: Ensures all step types meet their specific requirements
- **Pattern Consistency**: Validates against proven reference implementations
- **Early Error Detection**: Catches step type-specific issues during development

## Migration Strategy

### **1. Backward Compatibility**
- Existing unified alignment tester continues to work without modification
- Gradual migration to step type-aware variants
- Fallback to base tester for unknown step types

### **2. Incremental Adoption**
- Teams can adopt enhanced variants incrementally
- No breaking changes to existing validation infrastructure
- Optional enhanced validation features

### **3. Reference Implementation**
- Use existing XGBoost training step as primary reference for training validation
- Leverage existing processing steps as references for processing validation
- Build validation patterns from proven implementations

## Future Enhancements

### **1. Dynamic Framework Detection**
- Automatic detection of frameworks from script imports and usage patterns
- Support for custom frameworks and specialized training approaches
- Framework-specific optimization recommendations

### **2. Advanced Pattern Recognition**
- Machine learning-based pattern recognition for complex validation scenarios
- Automated detection of anti-patterns and code smells
- Intelligent recommendations based on similar implementations

### **3. Integration Enhancements**
- IDE integration for real-time validation feedback
- Automated test generation based on step specifications
- Integration with CI/CD pipelines for continuous validation
