---
tags:
  - project
  - planning
  - validation_refactoring
  - alignment_optimization
  - unified_tester
  - validation_ruleset
keywords:
  - validation alignment refactoring
  - unified alignment tester
  - validation ruleset configuration
  - step catalog integration
  - registry optimization
topics:
  - validation framework refactoring
  - alignment system optimization
  - unified tester enhancement
  - validation level control
language: python
date of note: 2025-10-01
implementation_status: PHASE_1_COMPLETED
---

# Validation Alignment System Refactoring Plan

## Executive Summary

This plan provides a comprehensive refactoring strategy for `src/cursus/validation/alignment/` based on the **Unified Alignment Tester Validation Ruleset** design and comprehensive analysis findings. The refactoring addresses **significant over-engineering** identified in the current 35-module validation system and implements a **configuration-driven, step-type-aware validation approach**.

### Key Findings

- **Current System**: 35 modules with extensive redundancy and over-engineering
- **Target System**: 8-12 modules with centralized configuration and step-type awareness
- **Module Reduction**: 77% reduction (35 → 8 modules)
- **Code Reduction**: 70% reduction (~10,000 → ~3,000 lines)
- **Performance Improvement**: 90% faster validation through level skipping

### Strategic Impact

- **Eliminates Over-Engineering**: Skip unnecessary validation levels for different step types
- **Centralized Control**: Single configuration file controls all validation behavior
- **Step-Type Awareness**: Different validation behavior for different SageMaker step types
- **Registry Integration**: Leverages existing step catalog and registry systems
- **Performance Optimization**: Dramatic reduction in validation time and complexity

## Current System Analysis

### **Current Architecture Problems**

#### **1. Massive Over-Engineering (35 Modules)**
```
src/cursus/validation/alignment/
├── core/ (5 modules)
│   ├── script_contract_alignment.py
│   ├── contract_spec_alignment.py
│   ├── spec_dependency_alignment.py
│   ├── builder_config_alignment.py (missing)
│   └── validation_orchestrator.py (redundant)
├── analyzer/ (7 modules)
│   ├── script_analyzer.py
│   ├── config_analyzer.py
│   ├── builder_analyzer.py
│   ├── import_analyzer.py
│   ├── path_extractor.py
│   ├── builder_argument_extractor.py
│   └── step_catalog_analyzer.py
├── validators/ (6 modules)
│   ├── script_contract_validator.py
│   ├── contract_spec_validator.py
│   ├── dependency_validator.py
│   ├── property_path_validator.py
│   ├── dependency_classifier.py
│   └── testability_validator.py
├── step_type_enhancers/ (7 modules)
│   ├── base_enhancer.py
│   ├── processing_enhancer.py
│   ├── training_enhancer.py
│   ├── createmodel_enhancer.py
│   ├── transform_enhancer.py
│   ├── registermodel_enhancer.py
│   └── utility_enhancer.py
├── patterns/ (3 modules)
├── reporting/ (3 modules)
├── utils/ (4 modules)
└── unified_alignment_tester.py
```

#### **2. Critical Design Flaws**
- **No Step-Type Awareness**: All steps get same validation regardless of type
- **No Level Control**: Cannot skip unnecessary validation levels
- **Extensive Redundancy**: Multiple modules doing similar validation
- **Missing Method Focus**: Complex alignment logic instead of method interface validation
- **Hardcoded Behavior**: No configuration system for validation rules

#### **3. Performance Issues**
- **Unnecessary Validation**: Script validation for non-script steps
- **Duplicate Analysis**: Multiple analyzers doing similar work
- **Complex Logic**: Over-engineered validation paths
- **No Caching**: Repeated validation operations

## Target Architecture Design

### **New Architecture (8-12 Modules)**
```
src/cursus/validation/alignment/
├── config/
│   └── validation_ruleset.py              # Centralized configuration
├── core/
│   ├── level_validators.py                # Level-specific validation logic
│   ├── script_contract_alignment.py       # Level 1 validation (existing)
│   ├── contract_spec_alignment.py         # Level 2 validation (existing)
│   └── spec_dependency_alignment.py       # Level 3 validation (existing)
├── validators/
│   ├── method_interface_validator.py      # Method interface compliance
│   ├── processing_step_validator.py       # Processing-specific validation
│   ├── training_step_validator.py         # Training-specific validation
│   ├── createmodel_step_validator.py      # CreateModel-specific validation
│   └── transform_step_validator.py        # Transform-specific validation
├── utils/
│   ├── validation_models.py               # Data models and enums
│   └── validation_utils.py                # Utilities and helpers
├── reporting/
│   └── validation_reporter.py             # Reporting and scoring
└── unified_alignment_tester.py            # Enhanced main orchestrator
```

### **Key Architectural Principles**

#### **1. Configuration-Driven Validation**
```python
# Central configuration controls all validation behavior
VALIDATION_RULESETS = {
    "Processing": ValidationRuleset(
        category=StepTypeCategory.SCRIPT_BASED,
        enabled_levels={1, 2, 3, 4},  # Full validation
        level_4_validator_class="ProcessingStepBuilderValidator"
    ),
    "CreateModel": ValidationRuleset(
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={3, 4},  # Skip script/contract validation
        level_4_validator_class="CreateModelStepBuilderValidator"
    ),
    "Base": ValidationRuleset(
        category=StepTypeCategory.EXCLUDED,
        enabled_levels=set(),  # No validation
        skip_reason="Base configurations - no builder to validate"
    )
}
```

#### **2. Step-Type-Aware Validation**
```python
class UnifiedAlignmentTester:
    def run_validation_for_step(self, step_name: str):
        # Get step type from registry
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        # Get validation ruleset
        ruleset = get_validation_ruleset(sagemaker_step_type)
        
        # Skip excluded step types
        if is_step_type_excluded(sagemaker_step_type):
            return {"status": "EXCLUDED", "reason": ruleset.skip_reason}
        
        # Run only enabled validation levels
        for level in ValidationLevel:
            if level in ruleset.enabled_levels:
                self._run_validation_level(step_name, level, ruleset)
```

#### **3. Method Interface Focus**
```python
class MethodInterfaceValidator:
    def validate_builder_interface(self, builder_class: type, step_type: str):
        # Universal method validation
        universal_methods = ["validate_configuration", "_get_inputs", "_get_outputs", "create_step"]
        
        # Step-type-specific method validation
        step_type_methods = self._get_step_type_methods(step_type)
        
        # Simple, focused validation
        return self._validate_methods(builder_class, universal_methods + step_type_methods)
```

## Implementation Plan

### **Phase 1: Configuration System Setup (3 Days) - ✅ COMPLETED**

#### **1.1 Create Validation Ruleset Configuration - ✅ COMPLETED**
**File**: `src/cursus/validation/alignment/config/validation_ruleset.py`

```python
"""Centralized validation ruleset configuration system."""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

class ValidationLevel(Enum):
    SCRIPT_CONTRACT = 1      # Level 1: Script ↔ Contract
    CONTRACT_SPEC = 2        # Level 2: Contract ↔ Specification  
    SPEC_DEPENDENCY = 3      # Level 3: Specification ↔ Dependencies (Universal)
    BUILDER_CONFIG = 4       # Level 4: Builder ↔ Configuration

class StepTypeCategory(Enum):
    SCRIPT_BASED = "script_based"      # Full 4-level validation
    CONTRACT_BASED = "contract_based"   # Skip Level 1, need 2-4
    NON_SCRIPT = "non_script"          # Skip Levels 1-2, need 3-4
    CONFIG_ONLY = "config_only"        # Only Level 4 needed
    EXCLUDED = "excluded"              # No validation needed

@dataclass
class ValidationRuleset:
    step_type: str
    category: StepTypeCategory
    enabled_levels: Set[ValidationLevel]
    level_4_validator_class: Optional[str] = None
    skip_reason: Optional[str] = None
    examples: List[str] = None

# Complete validation ruleset configuration
VALIDATION_RULESETS: Dict[str, ValidationRuleset] = {
    # Script-based steps (Full validation)
    "Processing": ValidationRuleset(...),
    "Training": ValidationRuleset(...),
    
    # Contract-based steps (Skip Level 1)
    "CradleDataLoading": ValidationRuleset(...),
    "MimsModelRegistrationProcessing": ValidationRuleset(...),
    
    # Non-script steps (Skip Levels 1-2)
    "CreateModel": ValidationRuleset(...),
    "Transform": ValidationRuleset(...),
    "RegisterModel": ValidationRuleset(...),
    
    # Configuration-only steps (Only Level 4)
    "Lambda": ValidationRuleset(...),
    
    # Excluded steps (No validation)
    "Base": ValidationRuleset(...),
    "Utility": ValidationRuleset(...)
}

# Configuration API
def get_validation_ruleset(sagemaker_step_type: str) -> Optional[ValidationRuleset]: ...
def is_validation_level_enabled(sagemaker_step_type: str, level: ValidationLevel) -> bool: ...
def get_enabled_validation_levels(sagemaker_step_type: str) -> Set[ValidationLevel]: ...
def is_step_type_excluded(sagemaker_step_type: str) -> bool: ...
```

#### **1.2 Integration with Registry**
```python
# Enhance registry integration
from cursus.registry.step_names import get_sagemaker_step_type

def run_validation_for_step(self, step_name: str):
    # Get step type from registry
    sagemaker_step_type = get_sagemaker_step_type(step_name)
    
    # Get validation configuration
    ruleset = get_validation_ruleset(sagemaker_step_type)
```

#### **1.3 Configuration Validation**
```python
def validate_step_type_configuration() -> List[str]:
    """Validate configuration for consistency issues."""
    issues = []
    
    # Check that excluded steps have no enabled levels
    # Check that Level 3 is enabled for non-excluded steps (universal)
    # Check that all step types have valid categories
    
    return issues
```

**Deliverables:**
- ✅ `validation_ruleset.py` with complete configuration
- ✅ Registry integration functions
- ✅ Configuration validation logic
- ✅ Unit tests for configuration system

**Phase 1 Completion Summary (October 1, 2025):**
- ✅ **Complete Configuration System**: Implemented centralized validation ruleset with 10 SageMaker step types
- ✅ **Step Type Classifications**: Script-based (2), Contract-based (2), Non-script (3), Config-only (1), Excluded (2)
- ✅ **Universal Level 3 Requirement**: All non-excluded steps enforce Spec↔Dependencies validation
- ✅ **Registry Integration**: Seamless integration with cursus registry system with fallback handling
- ✅ **Comprehensive Testing**: 32 test cases with 100% pass rate covering all functionality
- ✅ **Type Safety**: Full enum-based configuration with type hints and validation
- ✅ **API Functions**: 15+ configuration API functions for validation control
- ✅ **Module Structure**: Clean package structure with proper `__init__.py` files
- ✅ **Documentation**: Comprehensive docstrings and configuration examples
- ✅ **Validation**: Configuration consistency validation with error reporting

### **Phase 1.5: Builder Method Validation Rules (2 Days) - ✅ COMPLETED**

#### **1.5.1 Create Universal Builder Validation Rules - ✅ COMPLETED**
**File**: `src/cursus/validation/alignment/config/universal_builder_rules.py`

Based on analysis of actual step builders in the codebase, this module defines the universal validation rules that ALL step builders must implement, regardless of their specific SageMaker step type.

**Key Features:**
- **Universal Methods**: 5 required methods all builders must implement
  - `validate_configuration()` - Step-specific configuration validation
  - `_get_inputs()` - Transform logical inputs to step-specific format
  - `_get_outputs()` - Transform logical outputs to step-specific format  
  - `create_step()` - Create the final SageMaker pipeline step
  - `_get_environment_variables()` - Override base class for step-specific env vars

- **Inherited Methods**: 5 methods inherited from StepBuilderBase
  - Optional overrides: `_get_job_arguments()`
  - Final methods: `_get_cache_config()`, `_generate_job_name()`, `_get_step_name()`, `_get_base_output_path()`

- **Method Categories**: Enum-based categorization for validation control
  - `REQUIRED_ABSTRACT` - Must be implemented by all builders
  - `REQUIRED_OVERRIDE` - Must override base class method
  - `INHERITED_OPTIONAL` - Can optionally override base class
  - `INHERITED_FINAL` - Should not be overridden

- **Implementation Patterns**: Common patterns observed in actual builders
  - Initialization, validation, input processing, output processing, step creation

#### **1.5.2 Create Step-Type-Specific Validation Rules - ✅ COMPLETED**
**File**: `src/cursus/validation/alignment/config/step_type_specific_rules.py`

Based on analysis of actual step builders, this module defines validation rules specific to different SageMaker step types, capturing the unique methods each step type requires.

**Step Types Covered:**
- **Training Steps**: `_create_estimator()` method required
  - Return types: `Dict[str, TrainingInput]` for inputs, `str` for outputs
  - Examples: XGBoostTraining, PyTorchTraining

- **Processing Steps**: `_create_processor()` method required
  - Optional: `_get_job_arguments()` override for command-line args
  - Return types: `List[ProcessingInput]` for inputs, `List[ProcessingOutput]` for outputs
  - Examples: TabularPreprocessing, FeatureEngineering

- **CreateModel Steps**: `_create_model()` method required
  - Optional: `_get_image_uri()` for container image URI generation
  - Return types: `Dict[str, Any]` for inputs, `None` for outputs (handled automatically)
  - Examples: XGBoostModel, PyTorchModel

- **Transform Steps**: `_create_transformer()` method required
- **RegisterModel Steps**: `_create_model_package()` method required
- **Lambda Steps**: `_create_lambda_function()` method required
- **Excluded Types**: Base, Utility (no validation needed)

**Key Features:**
- **Step Type Categories**: Enum-based categorization matching validation levels
- **Method Specifications**: Detailed signatures, return types, implementation patterns
- **Validation Specifics**: Step-type-specific validation requirements
- **API Functions**: 15+ functions for querying step-type-specific rules

#### **1.5.3 Update Configuration Module Integration - ✅ COMPLETED**
**File**: `src/cursus/validation/alignment/config/__init__.py`

Updated the configuration module to expose both universal and step-type-specific validation rules through a unified API.

**New Exports:**
- `UNIVERSAL_BUILDER_VALIDATION_RULES` - Complete universal validation ruleset
- `STEP_TYPE_SPECIFIC_VALIDATION_RULES` - Complete step-type-specific rulesets
- `UniversalMethodCategory` - Enum for method categorization
- 15+ API functions for querying validation rules

**Deliverables:**
- ✅ Universal builder validation rules with 5 required methods
- ✅ Step-type-specific rules for 7 SageMaker step types  
- ✅ Method categorization system with 4 categories
- ✅ Comprehensive API functions for rule queries
- ✅ Integration with existing configuration system
- ✅ Based on actual codebase analysis, not theoretical requirements

### **Phase 2: Core Refactoring (5 Days)**

#### **2.1 Rewrite UnifiedAlignmentTester with Configuration-Driven Design**
**File**: `src/cursus/validation/alignment/unified_alignment_tester.py`

##### **Method Analysis & Cleanup Strategy**

**Current UnifiedAlignmentTester Analysis:**
- **Total Methods**: 32 methods analyzed
- **Keep & Enhance**: 8 core methods (25%)
- **Modify/Simplify**: 12 methods (37.5%)
- **Remove/Replace**: 12 methods (37.5%)

##### **🟢 KEEP & ENHANCE (8 methods) - Core API**
```python
# Essential API methods to preserve with configuration-driven enhancements
def __init__(self, workspace_dirs: List[str], **kwargs)  # Enhanced with validation ruleset
def run_full_validation(self, target_scripts=None, skip_levels=None)  # Configuration-driven level skipping
def validate_specific_script(self, step_name: str)  # Step-type-aware validation
def get_validation_summary(self)  # Enhanced with step-type-aware metrics
def export_report(self, format="json", output_path=None)  # Enhanced with configuration insights
def print_summary(self)  # Enhanced with step type breakdown
def get_critical_issues(self)  # Step-type-aware critical issue analysis
def discover_scripts(self)  # Enhanced with step type classification
```

##### **🟡 MODIFY/SIMPLIFY (12 methods) - Simplified Implementation**
```python
# Methods to simplify or consolidate
def run_level_validation(self, level: int)  # Use configuration-driven approach
def _run_level1_validation(self)  # → Consolidate into _run_validation_level()
def _run_level2_validation(self)  # → Consolidate into _run_validation_level()
def _run_level3_validation(self)  # → Consolidate into _run_validation_level()
def _run_level4_validation(self)  # → Consolidate into _run_validation_level()
def discover_contracts(self)  # → Consolidate into _discover_all_steps()
def discover_specs(self)  # → Consolidate into _discover_all_steps()
def discover_builders(self)  # → Consolidate into _discover_all_steps()
def get_alignment_status_matrix(self)  # Simplify with configuration-aware matrix
def get_step_info_from_catalog(self, step_name: str)  # Keep as utility, enhance with step type info
def get_component_path_from_catalog(self, step_name: str, component_type: str)  # Keep as utility
def validate_cross_workspace_compatibility(self, step_names: List[str])  # Simplify with configuration
```

##### **🔴 REMOVE/REPLACE (12 methods) - Redundant or Over-Engineered**
```python
# Methods to remove or replace with simpler alternatives
def _discover_scripts_with_catalog(self)  # → Remove: Redundant with consolidated discovery
def _discover_scripts_legacy(self)  # → Remove: Legacy fallback no longer needed
def _discover_contracts_with_catalog(self)  # → Remove: Redundant with consolidated discovery
def _discover_specs_with_catalog(self)  # → Remove: Redundant with consolidated discovery
def _discover_builders_with_catalog(self)  # → Remove: Redundant with consolidated discovery
def get_workspace_context(self, step_name: str)  # → Remove: Over-engineered, use simple registry lookup
def get_workspace_validation_summary(self)  # → Remove: Merge into get_validation_summary()
def _add_step_type_context_to_issues(self)  # → Remove: Over-engineered, use simple registry lookup

# Instance variables to remove/replace
self.level1_tester, self.level2_tester, etc.  # → Replace with single self.level_validators
self.step_type_enhancement_router  # → Remove: Over-engineered, use configuration system
self.level3_config  # → Remove: Over-engineered, use validation rulesets
self.enable_step_type_awareness  # → Remove: Always enabled through configuration
```

##### **Enhanced Implementation**
```python
class UnifiedAlignmentTester:
    """Enhanced Unified Alignment Tester with configuration-driven validation."""
    
    def __init__(self, workspace_dirs: List[str], **kwargs):
        self.workspace_dirs = workspace_dirs
        self.validation_config = VALIDATION_RULESETS
        
        # Initialize step catalog - key for discovery methods
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
        
        # Validate configuration on initialization
        config_issues = validate_step_type_configuration()
        if config_issues:
            raise ValueError(f"Configuration issues: {config_issues}")
        
        # Initialize level validators (replaces 4 separate level testers)
        self.level_validators = LevelValidators(workspace_dirs)
        
        # Preserve legacy kwargs for backward compatibility
        self.legacy_kwargs = kwargs
    
    def run_full_validation(self, target_scripts=None, skip_levels=None):
        """Enhanced run_full_validation with configuration-driven approach."""
        if target_scripts:
            results = {}
            for script_name in target_scripts:
                results[script_name] = self.run_validation_for_step(script_name)
            return results
        else:
            return self.run_validation_for_all_steps()
    
    def run_validation_for_step(self, step_name: str) -> Dict[str, Any]:
        """Run validation for a specific step based on its ruleset."""
        # Get step type from registry
        sagemaker_step_type = get_sagemaker_step_type(step_name)
        
        # Get validation ruleset
        ruleset = get_validation_ruleset(sagemaker_step_type)
        
        # Check if step type is excluded
        if is_step_type_excluded(sagemaker_step_type):
            return self._handle_excluded_step(step_name, sagemaker_step_type, ruleset)
        
        # Run enabled validation levels
        return self._run_enabled_validation_levels(step_name, sagemaker_step_type, ruleset)
    
    def run_validation_for_all_steps(self) -> Dict[str, Any]:
        """Run validation for all discovered steps."""
        # Discover all steps using step catalog
        discovered_steps = self._discover_all_steps()
        
        results = {}
        for step_name in discovered_steps:
            results[step_name] = self.run_validation_for_step(step_name)
        
        return results
    
    def _discover_all_steps(self) -> List[str]:
        """Discover all steps using step catalog - key for discovery methods."""
        # Consolidated discovery method (replaces 5 separate discovery methods)
        all_steps = []
        
        # Get all step names from step catalog
        step_names = self.step_catalog.get_all_step_names()
        all_steps.extend(step_names)
        
        # Get builder names (for backward compatibility)
        builder_names = self.step_catalog.get_all_builder_names()
        all_steps.extend(builder_names)
        
        # Get spec names (for comprehensive coverage)
        spec_names = self.step_catalog.get_all_spec_names()
        all_steps.extend(spec_names)
        
        # Remove duplicates and return
        return list(set(all_steps))
    
    def _run_validation_level(self, step_name: str, level: ValidationLevel, ruleset: ValidationRuleset):
        """Run a specific validation level (replaces 4 separate level methods)."""
        if level == ValidationLevel.SCRIPT_CONTRACT:
            return self.level_validators.run_level_1_validation(step_name)
        elif level == ValidationLevel.CONTRACT_SPEC:
            return self.level_validators.run_level_2_validation(step_name)
        elif level == ValidationLevel.SPEC_DEPENDENCY:
            return self.level_validators.run_level_3_validation(step_name)  # Universal
        elif level == ValidationLevel.BUILDER_CONFIG:
            return self.level_validators.run_level_4_validation(step_name, ruleset.level_4_validator_class)
    
    def _run_enabled_validation_levels(self, step_name: str, sagemaker_step_type: str, ruleset: ValidationRuleset):
        """Run all enabled validation levels for a step."""
        results = {
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "category": ruleset.category.value,
            "enabled_levels": [level.value for level in ruleset.enabled_levels],
            "validation_results": {}
        }
        
        for level in ValidationLevel:
            if level in ruleset.enabled_levels:
                try:
                    level_result = self._run_validation_level(step_name, level, ruleset)
                    results["validation_results"][f"level_{level.value}"] = level_result
                except Exception as e:
                    results["validation_results"][f"level_{level.value}"] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
        
        return results
    
    def _handle_excluded_step(self, step_name: str, sagemaker_step_type: str, ruleset: ValidationRuleset):
        """Handle excluded step types."""
        return {
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "status": "EXCLUDED",
            "reason": ruleset.skip_reason,
            "category": ruleset.category.value
        }
    
    # Preserve existing API methods for backward compatibility
    def discover_scripts(self):
        """Discover scripts - maintained for backward compatibility."""
        return self._discover_all_steps()
    
    def get_validation_summary(self):
        """Get validation summary - maintained for backward compatibility."""
        all_results = self.run_validation_for_all_steps()
        return self._generate_summary(all_results)
```

##### **Method Reduction Summary**
- **From 32 methods to 20 methods** (37.5% reduction)
- **Eliminated 5 redundant discovery methods** → 1 consolidated method
- **Consolidated 4 level validation methods** → 1 unified method
- **Removed over-engineered features** (enhancement router, feature flags, complex configs)
- **Preserved all core API methods** with enhanced functionality

#### **2.2 Create Level Validators**
**File**: `src/cursus/validation/alignment/core/level_validators.py`

```python
class LevelValidators:
    """Consolidated validation logic for each level."""
    
    def __init__(self, workspace_dirs: List[str]):
        self.workspace_dirs = workspace_dirs
        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)
    
    def run_level_1_validation(self, step_name: str) -> Dict[str, Any]:
        """Level 1: Script ↔ Contract validation."""
        # Use existing script_contract_alignment logic
        from .script_contract_alignment import ScriptContractAlignment
        alignment = ScriptContractAlignment(workspace_dirs=self.workspace_dirs)
        return alignment.validate_script_contract_alignment(step_name)
    
    def run_level_2_validation(self, step_name: str) -> Dict[str, Any]:
        """Level 2: Contract ↔ Specification validation."""
        # Use existing contract_spec_alignment logic
        from .contract_spec_alignment import ContractSpecAlignment
        alignment = ContractSpecAlignment(workspace_dirs=self.workspace_dirs)
        return alignment.validate_contract_spec_alignment(step_name)
    
    def run_level_3_validation(self, step_name: str) -> Dict[str, Any]:
        """Level 3: Specification ↔ Dependencies validation (Universal)."""
        # Use existing spec_dependency_alignment logic
        from .spec_dependency_alignment import SpecDependencyAlignment
        alignment = SpecDependencyAlignment(workspace_dirs=self.workspace_dirs)
        return alignment.validate_spec_dependency_alignment(step_name)
    
    def run_level_4_validation(self, step_name: str, validator_class: str) -> Dict[str, Any]:
        """Level 4: Builder ↔ Configuration validation (Step-type-specific)."""
        # Use step-type-specific validator
        validator = self._get_step_type_validator(validator_class)
        return validator.validate_builder_config_alignment(step_name)
```

#### **2.3 Create Method Interface Validator**
**File**: `src/cursus/validation/alignment/validators/method_interface_validator.py`

```python
class MethodInterfaceValidator:
    """Validator focusing on method interface compliance."""
    
    def validate_builder_interface(self, builder_class: type, step_type: str) -> List[ValidationIssue]:
        """Validate builder implements required methods."""
        issues = []
        
        # Universal method validation
        universal_methods = ["validate_configuration", "_get_inputs", "_get_outputs", "create_step"]
        for method_name in universal_methods:
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    level="ERROR",
                    message=f"Missing required method: {method_name}",
                    method_name=method_name,
                    step_type=step_type
                ))
        
        # Step-type-specific method validation
        step_type_methods = self._get_step_type_methods(step_type)
        for method_name in step_type_methods:
            if not hasattr(builder_class, method_name):
                issues.append(ValidationIssue(
                    level="ERROR",
                    message=f"Missing {step_type} method: {method_name}",
                    method_name=method_name,
                    step_type=step_type
                ))
        
        return issues
    
    def _get_step_type_methods(self, step_type: str) -> List[str]:
        """Get required methods for a specific step type."""
        step_type_methods = {
            "Training": ["_create_estimator"],
            "Processing": ["_create_processor"],
            "Transform": ["_create_transformer"],
            "CreateModel": ["_create_model"],
            "RegisterModel": ["_create_model_package"]
        }
        return step_type_methods.get(step_type, [])
```

**Deliverables:**
- ✅ `configurable_unified_tester.py` with configuration-driven validation
- ✅ `level_validators.py` with consolidated validation logic
- ✅ `method_interface_validator.py` with method-focused validation
- ✅ Integration with existing alignment modules
- ✅ Comprehensive unit tests

### **Phase 3: Step-Type-Specific Validators (4 Days)**

#### **3.1 Create Step-Type-Specific Validators**

**Processing Step Validator**:
```python
# src/cursus/validation/alignment/validators/processing_step_validator.py
class ProcessingStepBuilderValidator:
    """Validator for Processing step builders."""
    
    def validate_builder_config_alignment(self, step_name: str) -> Dict[str, Any]:
        """Validate Processing step builder configuration."""
        # Processing-specific validation logic
        # - Validate _create_processor() method
        # - Validate ProcessingInput/ProcessingOutput handling
        # - Validate processor configuration
        return self._validate_processing_specific_requirements(step_name)
```

**Training Step Validator**:
```python
# src/cursus/validation/alignment/validators/training_step_validator.py
class TrainingStepBuilderValidator:
    """Validator for Training step builders."""
    
    def validate_builder_config_alignment(self, step_name: str) -> Dict[str, Any]:
        """Validate Training step builder configuration."""
        # Training-specific validation logic
        # - Validate _create_estimator() method
        # - Validate TrainingInput handling
        # - Validate hyperparameter configuration
        return self._validate_training_specific_requirements(step_name)
```

**CreateModel Step Validator**:
```python
# src/cursus/validation/alignment/validators/createmodel_step_validator.py
class CreateModelStepBuilderValidator:
    """Validator for CreateModel step builders."""
    
    def validate_builder_config_alignment(self, step_name: str) -> Dict[str, Any]:
        """Validate CreateModel step builder configuration."""
        # CreateModel-specific validation logic
        # - Validate _create_model() method
        # - Validate model artifact handling
        # - Validate model configuration
        return self._validate_createmodel_specific_requirements(step_name)
```

#### **3.2 Validator Factory**
```python
# src/cursus/validation/alignment/validators/validator_factory.py
class ValidatorFactory:
    """Factory for creating step-type-specific validators."""
    
    @staticmethod
    def get_validator(validator_class: str) -> Any:
        """Get validator instance by class name."""
        validators = {
            "ProcessingStepBuilderValidator": ProcessingStepBuilderValidator,
            "TrainingStepBuilderValidator": TrainingStepBuilderValidator,
            "CreateModelStepBuilderValidator": CreateModelStepBuilderValidator,
            "TransformStepBuilderValidator": TransformStepBuilderValidator,
            "RegisterModelStepBuilderValidator": RegisterModelStepBuilderValidator,
            "LambdaStepBuilderValidator": LambdaStepBuilderValidator
        }
        
        if validator_class not in validators:
            raise ValueError(f"Unknown validator class: {validator_class}")
        
        return validators[validator_class]()
```

**Deliverables:**
- ✅ 5 step-type-specific validator classes
- ✅ Validator factory for dynamic validator creation
- ✅ Step-type-specific validation logic
- ✅ Integration with configuration system
- ✅ Unit tests for each validator

### **Phase 4: Module Consolidation and Cleanup (4 Days)**

#### **4.1 Eliminate Redundant Modules**

**Modules to Remove (20+ modules):**
```python
# Redundant analyzers (5 modules)
MODULES_TO_REMOVE = [
    "analyzer/script_analyzer.py",           # → Use method interface validation
    "analyzer/import_analyzer.py",           # → Not needed for method validation
    "analyzer/path_extractor.py",            # → Not needed for method validation
    "analyzer/builder_argument_extractor.py", # → Not needed
    "analyzer/config_analyzer.py",           # → Consolidated into validators
    
    # Redundant validators (4 modules)
    "validators/script_contract_validator.py",  # → Script validation not needed for many step types
    "validators/contract_spec_validator.py",    # → Contract validation not needed for many step types
    "validators/dependency_classifier.py",      # → Over-engineered dependency logic
    "validators/testability_validator.py",      # → Not core requirement
    
    # Redundant core modules (2 modules)
    "core/validation_orchestrator.py",         # → Redundant with unified_alignment_tester.py
    "core/builder_config_alignment.py",        # → Missing, would be redundant with step-specific validators
    
    # All step type enhancers (7 modules)
    "step_type_enhancers/base_enhancer.py",
    "step_type_enhancers/processing_enhancer.py",
    "step_type_enhancers/training_enhancer.py",
    "step_type_enhancers/createmodel_enhancer.py",
    "step_type_enhancers/transform_enhancer.py",
    "step_type_enhancers/registermodel_enhancer.py",
    "step_type_enhancers/utility_enhancer.py",
    
    # Redundant patterns (2 modules)
    "patterns/framework_patterns.py",          # → Not needed for method validation
    "patterns/pattern_recognizer.py",          # → Over-engineered
    
    # Redundant utilities (2 modules)
    "utils/script_analysis_models.py",         # → Not needed for method validation
    "utils/core_models.py"                     # → Consolidated into validation_models.py
]
```

#### **4.2 Consolidate Remaining Modules**

**Keep and Enhance (8 modules):**
```python
MODULES_TO_KEEP_AND_ENHANCE = [
    # Core modules (4 modules)
    "core/level_validators.py",                 # → Consolidated validation logic
    "core/script_contract_alignment.py",       # → Keep for Level 1 validation
    "core/contract_spec_alignment.py",         # → Keep for Level 2 validation
    "core/spec_dependency_alignment.py",       # → Keep for Level 3 validation (universal)
    
    # Configuration (1 module)
    "config/validation_ruleset.py",            # → New centralized configuration
    
    # Validators (5 modules)
    "validators/method_interface_validator.py", # → New method-focused validation
    "validators/processing_step_validator.py",  # → Processing-specific validation
    "validators/training_step_validator.py",    # → Training-specific validation
    "validators/createmodel_step_validator.py", # → CreateModel-specific validation
    "validators/transform_step_validator.py",   # → Transform-specific validation
    
    # Utils and reporting (3 modules)
    "utils/validation_models.py",              # → Consolidated data models
    "utils/validation_utils.py",               # → Essential utilities only
    "reporting/validation_reporter.py",        # → Consolidated reporting
    
    # Enhanced main orchestrator (1 module)
    "unified_alignment_tester.py"              # → Enhanced with configuration-driven validation
]
```

#### **4.3 Update Import Statements**
```python
# Update all files that import from removed modules
# Replace with imports from consolidated modules

# BEFORE: Importing from removed modules
from cursus.validation.alignment.analyzer.script_analyzer import ScriptAnalyzer
from cursus.validation.alignment.step_type_enhancers.processing_enhancer import ProcessingEnhancer
from cursus.validation.alignment.validators.script_contract_validator import ScriptContractValidator

# AFTER: Importing from consolidated modules
from cursus.validation.alignment.validators.method_interface_validator import MethodInterfaceValidator
from cursus.validation.alignment.validators.processing_step_validator import ProcessingStepBuilderValidator
from cursus.validation.alignment import UnifiedAlignmentTester  # Enhanced with configuration-driven validation
```

**Deliverables:**
- ✅ Remove 20+ redundant modules
- ✅ Update import statements across codebase
- ✅ Update documentation and examples
- ✅ Comprehensive integration testing

### **Phase 5: Testing and Integration (3 Days)**

#### **5.1 Comprehensive Testing Strategy**
```python
# Test configuration system
def test_validation_ruleset_configuration():
    # Test all step types have valid rulesets
    # Test configuration validation logic
    # Test API functions work correctly
    
# Test step-type-aware validation
def test_step_type_aware_validation():
    # Test Processing steps get full validation
    # Test CreateModel steps skip script validation
    # Test Base steps are excluded
    
# Test method interface validation
def test_method_interface_validation():
    # Test universal method validation
    # Test step-type-specific method validation
    # Test validation error reporting
    
# Test backward compatibility
def test_backward_compatibility():
    # Test existing API still works
    # Test same validation results
    # Test parameter compatibility
```

#### **5.2 Performance Testing**
```python
def test_performance_improvements():
    # Measure validation time before/after refactoring
    # Verify 90% improvement for non-script steps
    # Test memory usage reduction
    # Verify no regression in validation quality
```

#### **5.3 Integration Testing**
```python
def test_step_catalog_integration():
    # Test registry integration works
    # Test workspace directory support
    # Test step discovery integration
    
def test_validation_level_control():
    # Test level skipping works correctly
    # Test universal Level 3 validation
    # Test step-type-specific Level 4 validation
```

**Deliverables:**
- ✅ Comprehensive test suite (100+ tests)
- ✅ Performance benchmarks and verification
- ✅ Integration testing with step catalog/registry
- ✅ Backward compatibility verification
- ✅ Documentation updates

## Success Metrics

### **Code Quality Metrics**
- **Module Reduction**: 77% reduction (35 → 8 modules) ✅ Target
- **Lines of Code**: 70% reduction (~10,000 → ~3,000 lines) ✅ Target
- **Cyclomatic Complexity**: Reduced through elimination of complex validation paths
- **Import Simplification**: Cleaner dependencies and module structure
- **Test Coverage**: Maintain or improve test coverage (target: 95%+)

### **Performance Metrics**
- **Validation Time**: 90% faster for non-script steps ✅ Target
- **Memory Usage**: Reduced through elimination of redundant analyzers
- **Cache Efficiency**: Better performance through consolidated validation logic
- **Startup Time**: Faster initialization through simplified module structure

### **Maintainability Metrics**
- **Single Source of Truth**: All validation rules in centralized configuration
- **API Consistency**: Consistent patterns across validation system
- **Documentation**: Clear migration path and usage examples
- **Developer Experience**: Easier to understand and modify validation behavior

### **Functional Metrics**
- **Validation Quality**: No regression in validation accuracy
- **Step Type Coverage**: All SageMaker step types properly handled
- **Error Reporting**: Clear, actionable validation error messages
- **Backward Compatibility**: Existing code continues to work without changes

## Risk Assessment & Mitigation

### **High Risk: Backward Compatibility**
- **Risk**: Breaking existing code that depends on current API
- **Mitigation**: 
  - Comprehensive backward compatibility wrapper
  - Extensive testing of existing usage patterns
  - Gradual migration path with deprecation warnings

### **Medium Risk: Validation Logic Loss**
- **Risk**: Losing important validation logic during consolidation
- **Mitigation**:
  - Careful analysis of all existing validation logic
  - Preserve all unique validation capabilities
  - Comprehensive test coverage to catch regressions

### **Medium Risk: Configuration Complexity**
- **Risk**: Configuration system becoming too complex to maintain
- **Mitigation**:
  - Simple, clear configuration structure
  - Comprehensive validation of configuration consistency
  - Good documentation and examples

### **Low Risk: Performance Regression**
- **Risk**: New system being slower than current system
- **Mitigation**:
  - Performance testing throughout development
  - Benchmarking against current system
  - Optimization focus on critical paths

## Timeline & Resource Allocation

### **Phase 1: Configuration System Setup (3 Days)**
- **Day 1**: Create validation ruleset configuration
- **Day 2**: Registry integration and API functions
- **Day 3**: Configuration validation and testing
- **Resources**: 1 senior developer
- **Deliverables**: Complete configuration system

### **Phase 2: Core Refactoring (5 Days)**
- **Day 1-2**: Create configurable unified tester
- **Day 3**: Create level validators
- **Day 4**: Create method interface validator
- **Day 5**: Integration and testing
- **Resources**: 1 senior developer
- **Deliverables**: Core refactored validation system

### **Phase 3: Step-Type-Specific Validators (4 Days)**
- **Day 1**: Processing and Training validators
- **Day 2**: CreateModel and Transform validators
- **Day 3**: Validator factory and integration
- **Day 4**: Testing and refinement
- **Resources**: 1 developer
- **Deliverables**: Complete step-type-specific validation

### **Phase 4: Module Consolidation and Cleanup (4 Days)**
- **Day 1-2**: Remove redundant modules and update imports
- **Day 3**: Create backward compatibility wrapper
- **Day 4**: Documentation and cleanup
- **Resources**: 1 developer
- **Deliverables**: Clean, consolidated module structure

### **Phase 5: Testing and Integration (3 Days)**
- **Day 1**: Comprehensive testing
- **Day 2**: Performance testing and optimization
- **Day 3**: Integration testing and documentation
- **Resources**: 1 developer + 1 QA engineer
- **Deliverables**: Fully tested and documented system

### **Total Timeline: 19 Days**
- **Total Effort**: 19 developer days + 3 QA days
- **Risk Buffer**: 5 additional days for unexpected issues
- **Total Project Duration**: 27 days (5.4 weeks)

## Migration Strategy

### **Phase 1: Parallel Implementation**
- Implement new system alongside existing system
- No changes to existing API or behavior
- Comprehensive testing of new system

### **Phase 2: Gradual Migration**
- Add configuration-driven validation as option
- Deprecation warnings for old patterns
- Documentation updates with migration examples

### **Phase 3: Full Migration**
- Switch internal implementation to new system
- Remove deprecated code paths
- Complete documentation update

### **Phase 4: Cleanup**
- Remove old modules and redundant code
- Final performance optimization
- Long-term maintenance documentation

## Conclusion

This comprehensive refactoring plan addresses the significant over-engineering identified in the current validation alignment system. The new configuration-driven, step-type-aware approach provides:

- **77% reduction in modules** (35 → 8 modules)
- **70% reduction in code complexity** (~10,000 → ~3,000 lines)
- **90% performance improvement** through validation level skipping
- **Centralized configuration** for all validation behavior
- **Step-type awareness** for optimal validation
- **Method interface focus** for practical validation
- **Registry integration** leveraging existing systems
- **Backward compatibility** for seamless migration

The phased implementation approach ensures minimal risk while maximizing benefits through proven architectural patterns and comprehensive testing.

## Next Steps

1. **Approve refactoring plan** and resource allocation
2. **Begin Phase 1** with configuration system setup
3. **Execute phases sequentially** with thorough testing at each stage
4. **Monitor success metrics** throughout implementation
5. **Document lessons learned** for future refactoring efforts

This refactoring aligns with the broader codebase optimization efforts and provides a foundation for maintainable, efficient validation that scales with the growing complexity of SageMaker step types and validation requirements.

## References

- [Unified Alignment Tester Validation Ruleset](../1_design/unified_alignment_tester_validation_ruleset.md) - Configuration system design
- [Unified Alignment Tester Comprehensive Analysis](../4_analysis/unified_alignment_tester_comprehensive_analysis.md) - Analysis that identified over-engineering
- [Step Builder Validation Rulesets Design](../1_design/step_builder_validation_rulesets_design.md) - Method-centric validation approach
- [SageMaker Step Type Classification Design](../1_design/sagemaker_step_type_classification_design.md) - Step type classification system
- [Factories Redundancy Elimination Plan](2025-10-01_factories_redundancy_elimination_plan.md) - Related refactoring effort
