---
tags:
  - design
  - validation
  - alignment
  - unified_tester
  - configuration
  - ruleset
keywords:
  - validation ruleset configuration
  - unified alignment tester
  - sagemaker step type validation
  - validation level control
  - step type aware validation
topics:
  - validation framework configuration
  - alignment tester optimization
  - sagemaker step type specialization
  - validation level management
language: python
date of note: 2025-10-01
design_status: SPECIFICATION
---

# Unified Alignment Tester - Validation Ruleset Configuration

## Overview

This design specifies a **centralized validation ruleset configuration system** that controls the behavior of the Unified Alignment Tester based on SageMaker step types registered in the registry. The configuration provides flexible control over which validation levels are applied to different step types, enabling optimal validation without over-engineering.

## Problem Statement

Based on the [Unified Alignment Tester Comprehensive Analysis](../4_analysis/unified_alignment_tester_comprehensive_analysis.md), we identified that:

1. **Not all step types need all validation levels**
2. **Level 3 (Spec↔Dependencies) is universal** and step-type independent
3. **Level 1 (Script↔Contract) and Level 2 (Contract↔Spec)** may be skipped for some types
4. **Level 4 (Builder↔Config)** validates different classes/methods depending on the type
5. **Some step types should be excluded** from validation entirely

## Solution: Centralized Validation Ruleset Configuration

### Configuration File Structure

**File Location**: `src/cursus/validation/alignment/config/validation_ruleset.py`

```python
"""
Unified Alignment Tester - Validation Ruleset Configuration

This module defines validation rules for different SageMaker step types,
controlling which validation levels are applied and how they behave.
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

class ValidationLevel(Enum):
    """Validation levels in the alignment tester."""
    SCRIPT_CONTRACT = 1      # Level 1: Script ↔ Contract
    CONTRACT_SPEC = 2        # Level 2: Contract ↔ Specification  
    SPEC_DEPENDENCY = 3      # Level 3: Specification ↔ Dependencies (Universal)
    BUILDER_CONFIG = 4       # Level 4: Builder ↔ Configuration

class StepTypeCategory(Enum):
    """Categories of step types based on validation requirements."""
    SCRIPT_BASED = "script_based"           # Full 4-level validation
    CONTRACT_BASED = "contract_based"       # Skip Level 1, need 2-4
    NON_SCRIPT = "non_script"              # Skip Levels 1-2, need 3-4
    CONFIG_ONLY = "config_only"            # Only Level 4 needed
    EXCLUDED = "excluded"                   # No validation needed

@dataclass
class ValidationRuleset:
    """Validation ruleset for a specific step type."""
    step_type: str
    category: StepTypeCategory
    enabled_levels: Set[ValidationLevel]
    level_4_validator_class: Optional[str] = None  # Step-type-specific Level 4 validator
    skip_reason: Optional[str] = None              # Reason for skipping levels
    examples: List[str] = None                     # Example step names

# ============================================================================
# VALIDATION RULESET CONFIGURATION
# ============================================================================

VALIDATION_RULESETS: Dict[str, ValidationRuleset] = {
    
    # ========================================================================
    # SCRIPT-BASED STEPS (Full 4-Level Validation)
    # ========================================================================
    "Processing": ValidationRuleset(
        step_type="Processing",
        category=StepTypeCategory.SCRIPT_BASED,
        enabled_levels={
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",
        examples=[
            "TabularPreprocessing",
            "CurrencyConversion", 
            "RiskTableMapping",
            "ModelCalibration",
            "XGBoostModelEval"
        ]
    ),
    
    "Training": ValidationRuleset(
        step_type="Training",
        category=StepTypeCategory.SCRIPT_BASED,
        enabled_levels={
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="TrainingStepBuilderValidator",
        examples=[
            "XGBoostTraining",
            "PyTorchTraining",
            "DummyTraining"
        ]
    ),
    
    # ========================================================================
    # CONTRACT-BASED STEPS (Skip Level 1, Need Levels 2-4)
    # ========================================================================
    "CradleDataLoading": ValidationRuleset(
        step_type="CradleDataLoading",
        category=StepTypeCategory.CONTRACT_BASED,
        enabled_levels={
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",  # Uses processing validator
        skip_reason="No script in cursus/steps/scripts",
        examples=["CradleDataLoading"]
    ),
    
    "MimsModelRegistrationProcessing": ValidationRuleset(
        step_type="MimsModelRegistrationProcessing",
        category=StepTypeCategory.CONTRACT_BASED,
        enabled_levels={
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",  # Uses processing validator
        skip_reason="No script in cursus/steps/scripts",
        examples=["MimsModelRegistration"]
    ),
    
    # ========================================================================
    # NON-SCRIPT STEPS (Skip Levels 1-2, Focus on 3-4)
    # ========================================================================
    "CreateModel": ValidationRuleset(
        step_type="CreateModel",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="CreateModelStepBuilderValidator",
        skip_reason="No script or contract - SageMaker model creation",
        examples=["XGBoostModel", "PyTorchModel"]
    ),
    
    "Transform": ValidationRuleset(
        step_type="Transform",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="TransformStepBuilderValidator",
        skip_reason="Uses existing model - no custom script",
        examples=["BatchTransform"]
    ),
    
    "RegisterModel": ValidationRuleset(
        step_type="RegisterModel",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="RegisterModelStepBuilderValidator",
        skip_reason="SageMaker service operation - no custom code",
        examples=["Registration"]
    ),
    
    # ========================================================================
    # CONFIGURATION-ONLY STEPS (Only Level 4 Needed)
    # ========================================================================
    "Lambda": ValidationRuleset(
        step_type="Lambda",
        category=StepTypeCategory.CONFIG_ONLY,
        enabled_levels={
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="LambdaStepBuilderValidator",
        skip_reason="Lambda function - different execution model",
        examples=["LambdaStep"]
    ),
    
    # ========================================================================
    # EXCLUDED STEPS (No Validation Needed)
    # ========================================================================
    "Base": ValidationRuleset(
        step_type="Base",
        category=StepTypeCategory.EXCLUDED,
        enabled_levels=set(),  # No validation levels
        skip_reason="Base configurations - no builder to validate",
        examples=["Base"]
    ),
    
    "Utility": ValidationRuleset(
        step_type="Utility",
        category=StepTypeCategory.EXCLUDED,
        enabled_levels=set(),  # No validation levels
        skip_reason="Special case - doesn't create SageMaker steps directly",
        examples=["HyperparameterPrep"]
    ),
}

# ============================================================================
# CONFIGURATION API
# ============================================================================

def get_validation_ruleset(sagemaker_step_type: str) -> Optional[ValidationRuleset]:
    """Get validation ruleset for a SageMaker step type."""
    return VALIDATION_RULESETS.get(sagemaker_step_type)

def is_validation_level_enabled(sagemaker_step_type: str, level: ValidationLevel) -> bool:
    """Check if a validation level is enabled for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return False
    return level in ruleset.enabled_levels

def get_enabled_validation_levels(sagemaker_step_type: str) -> Set[ValidationLevel]:
    """Get all enabled validation levels for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return set()
    return ruleset.enabled_levels

def get_level_4_validator_class(sagemaker_step_type: str) -> Optional[str]:
    """Get the Level 4 validator class for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return None
    return ruleset.level_4_validator_class

def is_step_type_excluded(sagemaker_step_type: str) -> bool:
    """Check if a step type is excluded from validation."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return False
    return ruleset.category == StepTypeCategory.EXCLUDED

def get_step_types_by_category(category: StepTypeCategory) -> List[str]:
    """Get all step types in a specific category."""
    return [
        step_type for step_type, ruleset in VALIDATION_RULESETS.items()
        if ruleset.category == category
    ]

def get_all_step_types() -> List[str]:
    """Get all configured step types."""
    return list(VALIDATION_RULESETS.keys())

def validate_step_type_configuration() -> List[str]:
    """Validate the configuration for consistency issues."""
    issues = []
    
    # Check that all step types have valid categories
    for step_type, ruleset in VALIDATION_RULESETS.items():
        if not isinstance(ruleset.category, StepTypeCategory):
            issues.append(f"Invalid category for {step_type}: {ruleset.category}")
        
        # Check that excluded steps have no enabled levels
        if ruleset.category == StepTypeCategory.EXCLUDED and ruleset.enabled_levels:
            issues.append(f"Excluded step {step_type} should have no enabled levels")
        
        # Check that Level 3 is enabled for non-excluded steps (universal requirement)
        if (ruleset.category != StepTypeCategory.EXCLUDED and 
            ValidationLevel.SPEC_DEPENDENCY not in ruleset.enabled_levels):
            issues.append(f"Step {step_type} missing universal Level 3 validation")
    
    return issues

# ============================================================================
# INTEGRATION WITH UNIFIED ALIGNMENT TESTER
# ============================================================================

class ConfigurableUnifiedAlignmentTester:
    """Enhanced Unified Alignment Tester with ruleset-based configuration."""
    
    def __init__(self, workspace_dirs: List[str]):
        self.workspace_dirs = workspace_dirs
        self.validation_config = VALIDATION_RULESETS
        
        # Validate configuration on initialization
        config_issues = validate_step_type_configuration()
        if config_issues:
            raise ValueError(f"Configuration issues: {config_issues}")
    
    def run_validation_for_step(self, step_name: str) -> Dict[str, Any]:
        """Run validation for a specific step based on its ruleset."""
        from cursus.registry.step_names import get_sagemaker_step_type
        
        # Get step type from registry
        try:
            sagemaker_step_type = get_sagemaker_step_type(step_name)
        except ValueError:
            return {"error": f"Unknown step name: {step_name}"}
        
        # Get validation ruleset
        ruleset = get_validation_ruleset(sagemaker_step_type)
        if not ruleset:
            return {"error": f"No validation ruleset for step type: {sagemaker_step_type}"}
        
        # Check if step type is excluded
        if is_step_type_excluded(sagemaker_step_type):
            return {
                "step_name": step_name,
                "step_type": sagemaker_step_type,
                "status": "EXCLUDED",
                "reason": ruleset.skip_reason,
                "validation_results": {}
            }
        
        # Run enabled validation levels
        validation_results = {}
        enabled_levels = get_enabled_validation_levels(sagemaker_step_type)
        
        for level in ValidationLevel:
            if level in enabled_levels:
                validation_results[f"level_{level.value}"] = self._run_validation_level(
                    step_name, level, ruleset
                )
            else:
                validation_results[f"level_{level.value}"] = {
                    "status": "SKIPPED",
                    "reason": ruleset.skip_reason or f"Not required for {sagemaker_step_type}"
                }
        
        return {
            "step_name": step_name,
            "step_type": sagemaker_step_type,
            "category": ruleset.category.value,
            "status": "COMPLETED",
            "validation_results": validation_results
        }
    
    def _run_validation_level(self, step_name: str, level: ValidationLevel, ruleset: ValidationRuleset) -> Dict[str, Any]:
        """Run a specific validation level."""
        if level == ValidationLevel.SCRIPT_CONTRACT:
            return self._run_level_1_validation(step_name)
        elif level == ValidationLevel.CONTRACT_SPEC:
            return self._run_level_2_validation(step_name)
        elif level == ValidationLevel.SPEC_DEPENDENCY:
            return self._run_level_3_validation(step_name)  # Universal
        elif level == ValidationLevel.BUILDER_CONFIG:
            return self._run_level_4_validation(step_name, ruleset.level_4_validator_class)
        else:
            return {"status": "ERROR", "message": f"Unknown validation level: {level}"}
    
    def _run_level_1_validation(self, step_name: str) -> Dict[str, Any]:
        """Run Level 1: Script ↔ Contract validation."""
        # Implementation would use existing script_contract_alignment
        return {"status": "SUCCESS", "level": 1, "description": "Script ↔ Contract"}
    
    def _run_level_2_validation(self, step_name: str) -> Dict[str, Any]:
        """Run Level 2: Contract ↔ Specification validation."""
        # Implementation would use existing contract_spec_alignment
        return {"status": "SUCCESS", "level": 2, "description": "Contract ↔ Specification"}
    
    def _run_level_3_validation(self, step_name: str) -> Dict[str, Any]:
        """Run Level 3: Specification ↔ Dependencies validation (Universal)."""
        # Implementation would use existing spec_dependency_alignment
        return {"status": "SUCCESS", "level": 3, "description": "Specification ↔ Dependencies"}
    
    def _run_level_4_validation(self, step_name: str, validator_class: Optional[str]) -> Dict[str, Any]:
        """Run Level 4: Builder ↔ Configuration validation (Step-type-specific)."""
        # Implementation would use step-type-specific validator
        return {
            "status": "SUCCESS", 
            "level": 4, 
            "description": "Builder ↔ Configuration",
            "validator_class": validator_class
        }
    
    def run_validation_for_all_steps(self) -> Dict[str, Any]:
        """Run validation for all steps in the workspace."""
        from cursus.step_catalog.step_catalog import StepCatalog
        
        step_catalog = StepCatalog(workspace_dirs=self.workspace_dirs)
        available_steps = step_catalog.list_available_steps()
        
        results = {}
        summary = {
            "total_steps": len(available_steps),
            "validated": 0,
            "excluded": 0,
            "errors": 0
        }
        
        for step_name in available_steps:
            result = self.run_validation_for_step(step_name)
            results[step_name] = result
            
            if result.get("status") == "EXCLUDED":
                summary["excluded"] += 1
            elif result.get("status") == "COMPLETED":
                summary["validated"] += 1
            else:
                summary["errors"] += 1
        
        return {
            "summary": summary,
            "results": results,
            "configuration": {
                "total_rulesets": len(VALIDATION_RULESETS),
                "categories": {
                    category.value: len(get_step_types_by_category(category))
                    for category in StepTypeCategory
                }
            }
        }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example usage of the configurable validation system."""
    
    # Initialize the configurable tester
    tester = ConfigurableUnifiedAlignmentTester(workspace_dirs=["/path/to/workspace"])
    
    # Check if a specific validation level is enabled for a step type
    if is_validation_level_enabled("Processing", ValidationLevel.SCRIPT_CONTRACT):
        print("Level 1 validation is enabled for Processing steps")
    
    # Get all enabled levels for a step type
    enabled_levels = get_enabled_validation_levels("CreateModel")
    print(f"CreateModel enabled levels: {[level.value for level in enabled_levels]}")
    
    # Run validation for a specific step
    result = tester.run_validation_for_step("TabularPreprocessing")
    print(f"Validation result: {result}")
    
    # Run validation for all steps
    all_results = tester.run_validation_for_all_steps()
    print(f"Summary: {all_results['summary']}")
```

## Implementation Strategy

### Phase 1: Configuration Setup
1. **Create Configuration Module**: Implement `validation_ruleset_config.py`
2. **Integrate with Registry**: Connect with existing `step_names.py` registry
3. **Add Configuration Validation**: Ensure consistency and completeness

### Phase 2: Unified Alignment Tester Enhancement
1. **Modify Existing Tester**: Add configuration-based level control
2. **Implement Level-Specific Validators**: Create step-type-specific Level 4 validators
3. **Add Flexible API**: Support both old and new validation approaches

### Phase 3: Testing and Migration
1. **Test All Step Types**: Verify correct behavior for each category
2. **Performance Testing**: Ensure skipping levels improves performance
3. **Documentation Update**: Update guides and examples

## Benefits

### 1. Flexible Validation Control
- **Configurable Levels**: Turn validation levels on/off per step type
- **Step-Type Awareness**: Different validation behavior for different types
- **Easy Maintenance**: Centralized configuration for all validation rules

### 2. Performance Optimization
- **Skip Unnecessary Validation**: Don't run script validation for non-script steps
- **Targeted Validation**: Focus on relevant validation for each step type
- **Reduced Overhead**: Eliminate redundant validation work

### 3. Clear Separation of Concerns
- **Universal Level 3**: Spec↔Dependencies validation for all non-excluded steps
- **Step-Specific Level 4**: Different builder validation for different step types
- **Excluded Steps**: Clear exclusion of Base and Utility steps

### 4. Maintainability
- **Single Source of Truth**: All validation rules in one place
- **Easy Extension**: Add new step types by adding configuration entries
- **Validation Consistency**: Ensure all step types follow appropriate patterns

## Migration from Current System

### Backward Compatibility
The new system maintains backward compatibility by:
1. **Preserving Existing API**: `UnifiedAlignmentTester` continues to work
2. **Gradual Migration**: New `ConfigurableUnifiedAlignmentTester` runs alongside
3. **Same Validation Logic**: Reuses existing level validation implementations

### Migration Steps
1. **Deploy Configuration**: Add `validation_ruleset_config.py` to codebase
2. **Test New System**: Verify behavior matches current system for enabled levels
3. **Switch Implementation**: Replace internal logic with configuration-based approach
4. **Remove Redundant Code**: Clean up unused validation paths

## Future Enhancements

### 1. Dynamic Configuration
- **Runtime Configuration**: Modify validation rules without code changes
- **Environment-Specific Rules**: Different validation for dev/test/prod
- **User-Configurable**: Allow developers to customize validation levels

### 2. Advanced Validation Rules
- **Conditional Validation**: Enable levels based on step configuration
- **Dependency-Based Rules**: Validation based on step dependencies
- **Custom Validators**: Plugin system for custom validation logic

### 3. Reporting and Analytics
- **Validation Metrics**: Track validation performance and results
- **Rule Usage Analytics**: Understand which rules are most effective
- **Configuration Optimization**: Suggest optimal validation configurations

## Conclusion

The **Validation Ruleset Configuration** provides a flexible, maintainable solution for controlling the Unified Alignment Tester behavior based on SageMaker step types. This approach:

1. **Eliminates Over-Engineering**: Skip unnecessary validation levels
2. **Improves Performance**: Focus validation effort where it matters
3. **Enhances Maintainability**: Centralized, clear configuration
4. **Supports Evolution**: Easy to extend and modify validation rules

The configuration-driven approach aligns perfectly with the analysis findings and provides a practical path forward for optimizing the validation system while maintaining backward compatibility.

## References

- [Unified Alignment Tester Comprehensive Analysis](../4_analysis/unified_alignment_tester_comprehensive_analysis.md) - Analysis that identified the need for flexible validation
- [SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md) - Step type classification system
- [Step Builder Validation Rulesets Design](step_builder_validation_rulesets_design.md) - Method-centric validation approach
- [Alignment Rules](../0_developer_guide/alignment_rules.md) - Current alignment requirements
