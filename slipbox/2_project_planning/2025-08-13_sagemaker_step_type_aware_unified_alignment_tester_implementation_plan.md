---
tags:
  - project
  - planning
  - implementation
  - alignment_validation
  - sagemaker_integration
keywords:
  - SageMaker step type awareness
  - unified alignment tester
  - training step validation
  - framework detection
  - step type variants
  - implementation roadmap
topics:
  - alignment validation enhancement
  - SageMaker step type integration
  - training step support
  - validation framework evolution
language: python
date of note: 2025-08-13
---

# SageMaker Step Type-Aware Unified Alignment Tester Implementation Plan

## Project Overview

This implementation plan details the transformation of the unified alignment tester into a **SageMaker step type-aware validation framework**. The project will extend the current 100% success rate validation system to support all SageMaker step types, with immediate priority on Training step validation for XGBoost training scripts.

## Project Goals

### **Primary Objectives**
1. **Maintain 100% Success Rate**: Preserve perfect validation for all existing processing scripts
2. **Complete Step Type Coverage**: Implement comprehensive validation for all 7 SageMaker step types (17 step builders)
3. **Framework Awareness**: Add detection and validation of frameworks (XGBoost, PyTorch, SKLearn, SageMaker)
4. **Step Type Classification**: Leverage existing SageMaker step type registry for automatic detection
5. **Extensible Architecture**: Create foundation supporting all SageMaker step types with maximum code reuse

### **Success Criteria**
- All existing processing script validations continue to pass (100% success rate maintained)
- All 17 step builders across 7 step types achieve targeted validation success rates:
  - **Processing Steps (9 builders)**: 100% success rate maintained
  - **Training Steps (3 builders)**: 100% success rate achieved (XGBoost priority)
  - **CreateModel Steps (2 builders)**: 95%+ success rate achieved
  - **Transform Steps (1 builder)**: 95%+ success rate achieved
  - **RegisterModel Steps (1 builder)**: 95%+ success rate achieved
  - **Utility Steps (1 builder)**: 90%+ success rate achieved (special case handling)
  - **Base Steps (2 builders)**: Foundation validation patterns established
- Framework detection correctly identifies XGBoost, PyTorch, SKLearn, and SageMaker usage
- Step type routing correctly directs validation based on SageMaker step type classification
- Architecture supports easy addition of new step types, frameworks, and validation patterns

## Current State Analysis

### **Existing Assets**
- **Unified Alignment Tester**: Four-tier validation system with 100% success rate on processing scripts
- **Step Registry**: Complete SageMaker step type classification in `step_names.py`
- **XGBoost Training Implementation**: Complete training script, contract, spec, and builder
- **Design Documents**: Comprehensive design for step type-aware architecture

### **Current Limitations**
- Processing-centric validation patterns
- No framework detection or framework-specific validation
- Single validation path regardless of step type
- No training-specific pattern recognition

## Implementation Strategy: Maximum Code Reuse Approach

### **Phase 1: Minimal Extensions to Existing Files (Week 1)**

#### **1.1 Extend Existing Data Structures (20 lines)**
**File to Modify:** `src/cursus/validation/alignment/alignment_utils.py`

**Minimal Additions:**
```python
# Add to existing alignment_utils.py (extend existing classes)
@dataclass
class StepTypeAwareAlignmentIssue(AlignmentIssue):
    """Extends existing AlignmentIssue with step type context"""
    step_type: Optional[str] = None
    framework_context: Optional[str] = None
    reference_examples: List[str] = Field(default_factory=list)

# Add simple step type detection function
def detect_step_type_from_registry(script_name: str) -> str:
    """Use existing step registry to determine SageMaker step type"""
    from cursus.steps.registry.step_names import get_sagemaker_step_type, get_canonical_name_from_file_name
    try:
        canonical_name = get_canonical_name_from_file_name(script_name)
        return get_sagemaker_step_type(canonical_name)
    except ValueError:
        return "Processing"  # Default fallback

def detect_framework_from_imports(imports: List[ImportStatement]) -> Optional[str]:
    """Detect framework from existing import analysis"""
    for imp in imports:
        if 'xgboost' in imp.module_name.lower():
            return 'xgboost'
        if 'torch' in imp.module_name.lower():
            return 'pytorch'
    return None
```

#### **1.2 Enhance Existing Unified Alignment Tester (30 lines)**
**File to Modify:** `src/cursus/validation/alignment/unified_alignment_tester.py`

**Minimal Changes:**
```python
class UnifiedAlignmentTester:
    def __init__(self, ...):
        # All existing initialization remains the same
        self.enable_step_type_awareness = True  # Feature flag
        
    def _run_level1_validation(self, target_scripts):
        # Keep ALL existing logic, just add step type context
        try:
            results = self.level1_tester.validate_all_scripts(target_scripts)
            
            for script_name, result in results.items():
                # Existing validation result creation (unchanged)
                validation_result = ValidationResult(...)
                
                # Add step type context to issues (minimal addition)
                if self.enable_step_type_awareness:
                    step_type = detect_step_type_from_registry(script_name)
                    for issue in result.get('issues', []):
                        if isinstance(issue, dict):
                            issue['step_type'] = step_type
                
                # Rest of existing logic unchanged
                self.report.add_level1_result(script_name, validation_result)
```

#### **1.3 Create Framework Pattern Detection (100 lines)**
**New File:** `src/cursus/validation/alignment/framework_patterns.py`

**Training Pattern Detection:**
```python
def detect_training_patterns(script_content: str) -> Dict[str, List[str]]:
    """Detect training-specific patterns in script content"""
    patterns = {
        "training_loop_patterns": [],
        "model_saving_patterns": [],
        "hyperparameter_loading_patterns": []
    }
    
    # Training loop patterns
    training_patterns = [r'\.fit\(', r'\.train\(', r'xgb\.train\(', r'for.*epoch']
    # Model saving patterns  
    saving_patterns = [r'\.save_model\(', r'/opt/ml/model', r'pickle\.dump\(']
    # Hyperparameter patterns
    hyperparam_patterns = [r'json\.load\(', r'/opt/ml/input/data/config']
    
    # Pattern matching logic (reuse existing pattern recognition)
    return patterns
```

### **Phase 2: Enhance Existing Level Testers with Step Type Context (Week 2)**

#### **2.1 Enhance Script Contract Alignment Tester (40 lines)**
**File to Modify:** `src/cursus/validation/alignment/script_contract_alignment.py`

**Minimal Additions:**
```python
class ScriptContractAlignmentTester:
    def validate_script(self, script_name: str):
        # Keep ALL existing validation logic (unchanged)
        result = self._existing_validation_logic(script_name)
        
        # Add step type-specific enhancements (additive)
        step_type = detect_step_type_from_registry(script_name)
        if step_type == "Training":
            result = self._enhance_training_validation(result, script_name)
        
        return result
    
    def _enhance_training_validation(self, existing_result, script_name):
        """Add training-specific validation to existing results"""
        additional_issues = []
        
        # Check for training-specific patterns using existing script analysis
        script_analysis = self._get_existing_script_analysis(script_name)
        framework = detect_framework_from_imports(script_analysis.get('imports', []))
        
        if framework == 'xgboost':
            additional_issues.extend(self._check_xgboost_patterns(script_analysis))
        
        # Add to existing issues (don't replace)
        existing_result['issues'].extend(additional_issues)
        return existing_result
```

#### **2.2 Enhance Static Analysis Script Analyzer (25 lines)**
**File to Modify:** `src/cursus/validation/alignment/static_analysis/script_analyzer.py`

**Minimal Additions:**
```python
class ScriptAnalyzer:
    def analyze_script(self, script_path: str):
        # Keep ALL existing analysis logic (unchanged)
        analysis = self._existing_analysis_logic(script_path)
        
        # Add framework detection (minimal addition)
        analysis['framework'] = detect_framework_from_imports(analysis.get('imports', []))
        analysis['step_type'] = detect_step_type_from_registry(Path(script_path).stem)
        
        # Add step type-specific patterns (additive)
        if analysis['step_type'] == "Training":
            analysis.update(self._detect_training_patterns(script_path))
        
        return analysis
    
    def _detect_training_patterns(self, script_path):
        """Detect training patterns using existing pattern recognition"""
        with open(script_path, 'r') as f:
            content = f.read()
        
        return detect_training_patterns(content)  # Use function from framework_patterns.py
```

#### **2.3 Enhance Script Contract Validator (35 lines)**
**File to Modify:** `src/cursus/validation/alignment/validators/script_contract_validator.py`

**Minimal Additions:**
```python
class ScriptContractValidator:
    def validate(self, script_analysis, contract):
        # Keep ALL existing validation logic (unchanged)
        issues = self._existing_validation_logic(script_analysis, contract)
        
        # Add step type-specific validation (additive)
        step_type = script_analysis.get('step_type', 'Processing')
        if step_type == "Training":
            training_issues = self._validate_training_specific(script_analysis, contract)
            issues.extend(training_issues)
        
        return issues
    
    def _validate_training_specific(self, script_analysis, contract):
        """Add training-specific validation using existing patterns"""
        issues = []
        
        # Reuse existing path validation for training paths
        training_paths = ['/opt/ml/model', '/opt/ml/input/data/config']
        for path in training_paths:
            if not self._path_exists_in_script(script_analysis, path):
                issues.append(create_alignment_issue(
                    level=SeverityLevel.WARNING,
                    category="training_path_missing",
                    message=f"Training script should reference {path}",
                    step_type="Training"
                ))
        
        return issues
```

### **Phase 3: Complete Step Type Enhancement System (Week 3)**

#### **3.1 Step Type Enhancement Router (100 lines)**
**New File:** `src/cursus/validation/alignment/step_type_enhancement_router.py`

**Central Routing System:**
```python
class StepTypeEnhancementRouter:
    """Routes validation enhancement to appropriate step type enhancer"""
    
    def __init__(self):
        self.enhancers = {
            "Processing": ProcessingStepEnhancer(),
            "Training": TrainingStepEnhancer(),
            "CreateModel": CreateModelStepEnhancer(),
            "Transform": TransformStepEnhancer(),
            "RegisterModel": RegisterModelStepEnhancer(),
            "Utility": UtilityStepEnhancer(),
            "Base": BaseStepEnhancer()
        }
    
    def enhance_validation(self, script_name, existing_results):
        """Route to appropriate step type enhancer"""
        step_type = detect_step_type_from_registry(script_name)
        enhancer = self.enhancers.get(step_type)
        
        if enhancer:
            return enhancer.enhance_validation(existing_results, script_name)
        
        return existing_results

    def get_step_type_requirements(self, step_type: str) -> Dict[str, Any]:
        """Get validation requirements for each step type"""
        requirements = {
            "Processing": {
                "input_types": ["ProcessingInput"],
                "output_types": ["ProcessingOutput"],
                "required_methods": ["_create_processor"],
                "required_patterns": ["data_transformation", "environment_variables"]
            },
            "Training": {
                "input_types": ["TrainingInput"],
                "output_types": ["model_artifacts"],
                "required_methods": ["_create_estimator", "_prepare_hyperparameters_file"],
                "required_patterns": ["training_loop", "model_saving", "hyperparameter_loading"]
            },
            "CreateModel": {
                "input_types": ["model_artifacts"],
                "output_types": ["model_endpoint"],
                "required_methods": ["_create_model"],
                "required_patterns": ["model_loading", "inference_code"]
            },
            "Transform": {
                "input_types": ["TransformInput"],
                "output_types": ["transform_results"],
                "required_methods": ["_create_transformer"],
                "required_patterns": ["batch_processing", "model_inference"]
            },
            "RegisterModel": {
                "input_types": ["model_artifacts"],
                "output_types": ["registered_model"],
                "required_methods": ["_create_model_package"],
                "required_patterns": ["model_metadata", "approval_workflow"]
            },
            "Utility": {
                "input_types": ["various"],
                "output_types": ["prepared_files"],
                "required_methods": ["_prepare_files"],
                "required_patterns": ["file_preparation"]
            },
            "Base": {
                "input_types": ["base_inputs"],
                "output_types": ["base_outputs"],
                "required_methods": ["create_step"],
                "required_patterns": ["foundation_patterns"]
            }
        }
        return requirements.get(step_type, {})
```

#### **3.2 Step Type Enhancer Directory Structure**
**New Directory:** `src/cursus/validation/alignment/step_type_enhancers/`

**Base Enhancer (50 lines):**
```python
# base_enhancer.py
class BaseStepEnhancer(ABC):
    """Abstract base class for all step type enhancers"""
    
    def __init__(self, step_type: str):
        self.step_type = step_type
        self.reference_examples = []
        self.framework_validators = {}
    
    @abstractmethod
    def enhance_validation(self, existing_results, script_name):
        """Enhance existing validation with step type-specific checks"""
        pass
    
    def _merge_results(self, existing_results, additional_issues):
        """Merge additional issues with existing validation results"""
        if isinstance(existing_results, dict):
            existing_results.setdefault('issues', []).extend(additional_issues)
        return existing_results
```

#### **3.3 Training Step Enhancer (200 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/training_enhancer.py`

**Complete Training Validation:**
```python
class TrainingStepEnhancer(BaseStepEnhancer):
    """Training step-specific validation enhancement"""
    
    def __init__(self):
        super().__init__("Training")
        self.reference_examples = [
            "xgboost_training.py",
            "pytorch_training.py",
            "builder_xgboost_training_step.py"
        ]
        self.framework_validators = {
            "xgboost": XGBoostTrainingValidator(),
            "pytorch": PyTorchTrainingValidator()
        }
    
    def enhance_validation(self, existing_results, script_name):
        """Add training-specific validation to existing results"""
        additional_issues = []
        
        # Get script analysis from existing validation
        script_analysis = self._get_script_analysis(script_name)
        framework = detect_framework_from_imports(script_analysis.get('imports', []))
        
        # Level 1: Training script patterns
        additional_issues.extend(self._validate_training_script_patterns(script_analysis, framework))
        
        # Level 2: Training specifications
        additional_issues.extend(self._validate_training_specifications(script_name))
        
        # Level 3: Training dependencies
        additional_issues.extend(self._validate_training_dependencies(script_name))
        
        # Level 4: Training builder patterns
        additional_issues.extend(self._validate_training_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_training_script_patterns(self, script_analysis, framework):
        """Validate training-specific script patterns"""
        issues = []
        
        # Check for training loop patterns
        if not self._has_training_loop_patterns(script_analysis):
            issues.append(self._create_training_issue(
                "missing_training_loop",
                "Training script should contain model training logic",
                "Add model.fit() or equivalent training loop"
            ))
        
        # Check for model saving patterns
        if not self._has_model_saving_patterns(script_analysis):
            issues.append(self._create_training_issue(
                "missing_model_saving",
                "Training script should save model artifacts",
                "Add model saving to /opt/ml/model/"
            ))
        
        # Check for hyperparameter loading patterns
        if not self._has_hyperparameter_loading_patterns(script_analysis):
            issues.append(self._create_training_issue(
                "missing_hyperparameter_loading",
                "Training script should load hyperparameters from file",
                "Add hyperparameter loading from /opt/ml/input/data/config/"
            ))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            issues.extend(framework_validator.validate_script_patterns(script_analysis))
        
        return issues
```

#### **3.4 CreateModel Step Enhancer (150 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/create_model_enhancer.py`

**Model Creation Validation:**
```python
class CreateModelStepEnhancer(BaseStepEnhancer):
    """CreateModel step-specific validation enhancement"""
    
    def __init__(self):
        super().__init__("CreateModel")
        self.reference_examples = [
            "builder_xgboost_model_step.py",
            "builder_pytorch_model_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        """Add CreateModel-specific validation"""
        additional_issues = []
        
        # Level 1: Model artifact handling validation
        additional_issues.extend(self._validate_model_artifact_handling(script_name))
        
        # Level 2: Inference code validation
        additional_issues.extend(self._validate_inference_code_patterns(script_name))
        
        # Level 3: Container configuration validation
        additional_issues.extend(self._validate_container_configuration(script_name))
        
        # Level 4: Model creation builder validation
        additional_issues.extend(self._validate_model_creation_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_model_artifact_handling(self, script_name):
        """Validate model artifact loading patterns"""
        issues = []
        
        # Check for model loading patterns
        script_analysis = self._get_script_analysis(script_name)
        if not self._has_model_loading_patterns(script_analysis):
            issues.append(self._create_create_model_issue(
                "missing_model_loading",
                "CreateModel script should load model artifacts",
                "Add model loading from /opt/ml/model/"
            ))
        
        return issues
    
    def _validate_inference_code_patterns(self, script_name):
        """Validate inference code implementation"""
        issues = []
        
        script_analysis = self._get_script_analysis(script_name)
        if not self._has_inference_patterns(script_analysis):
            issues.append(self._create_create_model_issue(
                "missing_inference_code",
                "CreateModel should implement inference logic",
                "Add model_fn, input_fn, predict_fn, or output_fn functions"
            ))
        
        return issues
```

#### **3.5 RegisterModel Step Enhancer (150 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/register_model_enhancer.py`

**Model Registration Validation:**
```python
class RegisterModelStepEnhancer(BaseStepEnhancer):
    """RegisterModel step-specific validation enhancement"""
    
    def __init__(self):
        super().__init__("RegisterModel")
        self.reference_examples = [
            "builder_registration_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        """Add RegisterModel-specific validation"""
        additional_issues = []
        
        # Level 1: Model metadata validation
        additional_issues.extend(self._validate_model_metadata(script_name))
        
        # Level 2: Approval workflow validation
        additional_issues.extend(self._validate_approval_workflow(script_name))
        
        # Level 3: Model package validation
        additional_issues.extend(self._validate_model_package_creation(script_name))
        
        # Level 4: Registration builder validation
        additional_issues.extend(self._validate_registration_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_model_metadata(self, script_name):
        """Validate model metadata preparation"""
        issues = []
        
        # Check for model metadata patterns
        builder_analysis = self._get_builder_analysis(script_name)
        if not self._has_model_metadata_patterns(builder_analysis):
            issues.append(self._create_register_model_issue(
                "missing_model_metadata",
                "RegisterModel should prepare model metadata",
                "Add model description, approval status, and metrics"
            ))
        
        return issues
```

#### **3.6 Transform Step Enhancer (150 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/transform_enhancer.py`

**Batch Transform Validation:**
```python
class TransformStepEnhancer(BaseStepEnhancer):
    """Transform step-specific validation enhancement"""
    
    def __init__(self):
        super().__init__("Transform")
        self.reference_examples = [
            "builder_batch_transform_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        """Add Transform-specific validation"""
        additional_issues = []
        
        # Level 1: Batch processing validation
        additional_issues.extend(self._validate_batch_processing_patterns(script_name))
        
        # Level 2: Transform input validation
        additional_issues.extend(self._validate_transform_input_specifications(script_name))
        
        # Level 3: Model inference validation
        additional_issues.extend(self._validate_model_inference_patterns(script_name))
        
        # Level 4: Transform builder validation
        additional_issues.extend(self._validate_transform_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

#### **3.7 Utility Step Enhancer (100 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/utility_enhancer.py`

**Utility Step Validation:**
```python
class UtilityStepEnhancer(BaseStepEnhancer):
    """Utility step-specific validation enhancement"""
    
    def __init__(self):
        super().__init__("Utility")
        self.reference_examples = [
            "builder_hyperparameter_prep_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        """Add Utility-specific validation"""
        additional_issues = []
        
        # Level 1: File preparation validation
        additional_issues.extend(self._validate_file_preparation_patterns(script_name))
        
        # Level 2: Parameter generation validation
        additional_issues.extend(self._validate_parameter_generation(script_name))
        
        # Level 3: Special case handling
        additional_issues.extend(self._validate_special_case_handling(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

#### **3.8 Processing Step Enhancer (100 lines)**
**File:** `src/cursus/validation/alignment/step_type_enhancers/processing_enhancer.py`

**Processing Step Migration:**
```python
class ProcessingStepEnhancer(BaseStepEnhancer):
    """Processing step-specific validation enhancement (migration from existing)"""
    
    def __init__(self):
        super().__init__("Processing")
        self.reference_examples = [
            "tabular_preprocessing.py",
            "risk_table_mapping.py",
            "builder_tabular_preprocessing_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        """Migrate existing processing validation to step type-aware system"""
        additional_issues = []
        
        # Level 1: Processing script patterns (existing logic)
        additional_issues.extend(self._validate_processing_script_patterns(script_name))
        
        # Level 2: Processing specifications (existing logic)
        additional_issues.extend(self._validate_processing_specifications(script_name))
        
        # Level 3: Processing dependencies (existing logic)
        additional_issues.extend(self._validate_processing_dependencies(script_name))
        
        # Level 4: Processing builder patterns (existing logic)
        additional_issues.extend(self._validate_processing_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

### **Phase 4: Integration and Testing (Week 4)**

#### **4.1 Integration Testing (Comprehensive)**
**Testing Strategy:**
- Run existing processing scripts through enhanced system
- Validate 100% success rate maintained
- Test XGBoost training script with new enhancements
- Performance benchmarking

#### **4.2 Feature Flag Implementation**
**File to Modify:** `src/cursus/validation/alignment/unified_alignment_tester.py`

**Feature Flag Control:**
```python
class UnifiedAlignmentTester:
    def __init__(self, ...):
        # All existing initialization
        self.enable_step_type_awareness = os.getenv('ENABLE_STEP_TYPE_AWARENESS', 'true').lower() == 'true'
        
    def _apply_step_type_enhancements(self, script_name, existing_results):
        """Apply step type enhancements if enabled"""
        if not self.enable_step_type_awareness:
            return existing_results
            
        step_type = detect_step_type_from_registry(script_name)
        if step_type == "Training":
            return enhance_training_validation(existing_results, ...)
        
        return existing_results
```

### **Phase 5: Additional Step Type Variants (Week 7-8)**

#### **5.1 Transform Step Variant**
**Deliverables:**
- `TransformStepAlignmentTester` implementation
- Transform-specific pattern validation
- Batch processing pattern recognition

#### **5.2 CreateModel Step Variant**
**Deliverables:**
- `CreateModelStepAlignmentTester` implementation
- Model creation pattern validation
- Inference code validation

#### **5.3 Remaining Step Types**
**Deliverables:**
- Implement remaining 8 SageMaker step type variants
- Basic validation for less common step types
- Extensible framework for future step types

### **Phase 6: Integration and Testing (Week 9-10)**

#### **6.1 Comprehensive Integration Testing**
**Deliverables:**
- End-to-end testing of all step type variants
- Performance optimization and benchmarking
- Integration with existing CI/CD pipelines

#### **6.2 Documentation and Training**
**Deliverables:**
- Updated developer documentation
- Migration guide for teams
- Training materials for enhanced validation features

## Technical Implementation Details

### **File Structure (Complete Step Type Coverage)**
```
src/cursus/validation/alignment/
├── __init__.py
├── unified_alignment_tester.py                  # Enhanced (50 lines added)
├── alignment_utils.py                           # Enhanced (30 lines added)
├── script_contract_alignment.py                 # Enhanced (40 lines added)
├── static_analysis/
│   └── script_analyzer.py                       # Enhanced (25 lines added)
├── validators/
│   └── script_contract_validator.py             # Enhanced (35 lines added)
├── framework_patterns.py                        # New (200 lines)
├── step_type_enhancement_router.py              # New (100 lines)
├── step_type_enhancers/                         # New directory
│   ├── __init__.py                              # New (20 lines)
│   ├── base_enhancer.py                         # New (50 lines)
│   ├── processing_enhancer.py                   # New (100 lines)
│   ├── training_enhancer.py                     # New (200 lines)
│   ├── create_model_enhancer.py                 # New (150 lines)
│   ├── transform_enhancer.py                    # New (150 lines)
│   ├── register_model_enhancer.py               # New (150 lines)
│   └── utility_enhancer.py                      # New (100 lines)
└── [all existing files remain unchanged]        # 90%+ of existing code preserved
```

### **Code Reuse Summary**
- **Files Modified**: 5 existing files (minimal changes, ~180 lines total)
- **Files Created**: 10 new files (~1,220 lines total)
- **Existing Code Preserved**: 90%+ of existing codebase
- **Total New Code**: ~1,400 lines
- **Total Existing Code Reused**: ~10,000+ lines
- **Code Reuse Ratio**: 87% existing code reused

### **Step Type Coverage Summary**
- **Processing Steps (9 builders)**: Enhanced validation with existing patterns
- **Training Steps (3 builders)**: Complete framework-aware validation
- **CreateModel Steps (2 builders)**: Model artifact and inference validation
- **Transform Steps (1 builder)**: Batch processing validation
- **RegisterModel Steps (1 builder)**: Model registry validation
- **Utility Steps (1 builder)**: Special case handling
- **Base Steps (2 builders)**: Foundation pattern validation
- **Total Coverage**: 17 step builders across 7 SageMaker step types

### **Integration Points**

#### **Registry Integration**
```python
# Use existing step registry
from cursus.steps.registry.step_names import get_sagemaker_step_type, STEP_NAMES

# Use existing step type classification
from cursus.steps.registry.step_type_test_variants import (
    STEP_TYPE_VARIANT_MAP,
    get_step_type_variant
)
```

#### **Specification Integration**
```python
# Use existing specifications
from cursus.steps.specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
from cursus.steps.contracts.xgboost_training_contract import XGBOOST_TRAIN_CONTRACT
```

#### **Builder Integration**
```python
# Use existing builders
from cursus.steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder
```

## Risk Management

### **Risk 1: Regression in Processing Script Validation**
**Mitigation:**
- Comprehensive regression testing before each phase
- Parallel validation with original system during transition
- Rollback plan if success rate drops below 100%

### **Risk 2: Performance Impact**
**Mitigation:**
- Performance benchmarking at each phase
- Optimization of pattern detection algorithms
- Caching of step type detection results

### **Risk 3: Framework Detection Accuracy**
**Mitigation:**
- Comprehensive testing with known framework usage
- Manual validation of framework detection results
- Fallback to generic validation if detection fails

### **Risk 4: Integration Complexity**
**Mitigation:**
- Incremental integration approach
- Extensive unit and integration testing
- Clear separation of concerns between variants

## Success Metrics

### **Quantitative Metrics**
- **Processing Script Success Rate**: Maintain 100% success rate
- **Training Script Success Rate**: Achieve 100% success rate for XGBoost training
- **Framework Detection Accuracy**: >95% accuracy for XGBoost and PyTorch detection
- **Performance Impact**: <20% increase in validation time
- **Code Coverage**: >90% test coverage for new components

### **Qualitative Metrics**
- **Developer Feedback**: Positive feedback on enhanced validation quality
- **Issue Quality**: More specific and actionable validation issues
- **Reference Utility**: Developers find reference examples helpful
- **Maintainability**: Easy addition of new step types and frameworks

## Timeline Summary

| Phase | Duration | Key Deliverables | Success Criteria |
|-------|----------|------------------|------------------|
| Phase 1 | Week 1-2 | Core architecture, data structures, registry integration | Architecture supports variant creation |
| Phase 2 | Week 3-4 | Training step validation, XGBoost framework validator | XGBoost training validation passes |
| Phase 3 | Week 5 | Processing step variant, backward compatibility | 100% success rate maintained |
| Phase 4 | Week 6 | Enhanced orchestration, unified reporting | All step types route correctly |
| Phase 5 | Week 7-8 | Additional step type variants | Transform and CreateModel variants work |
| Phase 6 | Week 9-10 | Integration testing, documentation | Full system integration complete |

## Dependencies

### **Internal Dependencies**
- Existing unified alignment tester codebase
- Step registry with SageMaker step type classification
- XGBoost training script, contract, spec, and builder implementations
- Existing specification and contract infrastructure

### **External Dependencies**
- SageMaker SDK for step type definitions
- Python AST parsing libraries for pattern detection
- Testing frameworks for comprehensive validation

## Conclusion

This implementation plan provides a comprehensive roadmap for transforming the unified alignment tester into a SageMaker step type-aware validation framework. The phased approach ensures:

1. **Risk Mitigation**: Incremental implementation with comprehensive testing
2. **Backward Compatibility**: Existing validation continues to work perfectly
3. **Extensibility**: Foundation for all SageMaker step types and frameworks
4. **Quality Assurance**: Enhanced validation quality with step type-specific rules

The project will deliver a robust, extensible validation framework that maintains the current 100% success rate while adding comprehensive support for training steps and framework-specific validation patterns.

## References

- [SageMaker Step Type-Aware Unified Alignment Tester Design](../1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md) - Comprehensive design document
- [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md) - Original unified alignment tester design
- [SageMaker Step Type Classification Design](../1_design/sagemaker_step_type_classification_design.md) - Step type classification framework
- [SageMaker Step Type Universal Builder Tester Design](../1_design/sagemaker_step_type_universal_builder_tester_design.md) - Universal builder testing framework
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md) - Training step implementation patterns
- [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md) - Processing step implementation patterns
