---
tags:
  - archive
  - design
  - standardization
  - pydantic_validation
  - registry_system
  - naming_conventions
keywords:
  - StepDefinition model
  - Pydantic validation
  - standardization rules
  - naming conventions
  - registry enforcement
  - SageMaker step types
  - hybrid registry
  - validation framework
topics:
  - standardization enforcement
  - registry validation
  - naming convention compliance
  - step type classification
language: python
date of note: 2025-09-04
---

# StepDefinition Standardization Enforcement Design

## Overview

This design document proposes leveraging the StepDefinition Pydantic model in the hybrid registry system to automatically enforce standardization rules and naming conventions. By embedding validation logic directly into the data model, we can ensure consistent compliance with established standards across all step definitions while providing immediate feedback to developers.

## Problem Statement

The current system has comprehensive standardization rules defined in documentation, but enforcement relies on:

1. **Manual Compliance**: Developers must remember and follow naming conventions
2. **Post-Hoc Validation**: Issues discovered during testing or code review
3. **Inconsistent Enforcement**: Different validation tools with varying coverage
4. **Documentation Drift**: Rules may not be consistently applied as the system evolves

This leads to:
- Naming inconsistencies across components
- Registry entries that don't follow established patterns
- Increased maintenance burden
- Reduced developer productivity due to late-stage error discovery

## Proposed Solution

### 1. Enhanced StepDefinition Model with Comprehensive Validation

Extend the existing StepDefinition Pydantic model to enforce all standardization rules through built-in validation:

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Dict, List, Any, Optional
from enum import Enum
import re


class SageMakerStepType(str, Enum):
    """Valid SageMaker step types from classification design."""
    PROCESSING = "Processing"
    TRAINING = "Training"
    TRANSFORM = "Transform"
    CREATE_MODEL = "CreateModel"
    REGISTER_MODEL = "RegisterModel"
    LAMBDA = "Lambda"
    MIMS_MODEL_REGISTRATION_PROCESSING = "MimsModelRegistrationProcessing"
    CRADLE_DATA_LOADING = "CradleDataLoading"
    BASE = "Base"
    UTILITY = "Utility"


class Framework(str, Enum):
    """Valid framework types."""
    XGBOOST = "xgboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


class JobType(str, Enum):
    """Valid job types following snake_case convention."""
    TRAINING = "training"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    TESTING = "testing"
    INFERENCE = "inference"


class StepDefinition(BaseModel):
    """Enhanced step definition with comprehensive standardization enforcement."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    # Core fields with enhanced validation
    name: str = Field(..., min_length=1, description="Step name identifier (PascalCase)")
    registry_type: RegistryType = Field(..., description="Registry type using enum validation")
    config_class: Optional[str] = Field(None, description="Configuration class name (PascalCase + Config)")
    spec_type: Optional[str] = Field(None, description="Specification type (matches step name)")
    sagemaker_step_type: Optional[SageMakerStepType] = Field(None, description="SageMaker step type classification")
    builder_step_name: Optional[str] = Field(None, description="Builder class name (PascalCase + StepBuilder)")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[Framework] = Field(None, description="Framework used by step")
    job_types: List[JobType] = Field(default_factory=list, description="Supported job types (snake_case)")
    
    # Registry metadata
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    override_source: Optional[str] = Field(None, description="Source of override for tracking")
    priority: int = Field(default=100, description="Resolution priority (lower = higher priority)")
    compatibility_tags: List[str] = Field(default_factory=list, description="Compatibility tags for smart resolution")
    framework_version: Optional[str] = Field(None, description="Framework version for compatibility checking")
    environment_tags: List[str] = Field(default_factory=list, description="Environment compatibility tags")
    conflict_resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.WORKSPACE_PRIORITY, 
        description="Strategy for resolving conflicts using enum validation"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # === STANDARDIZATION RULE ENFORCEMENT ===

    @field_validator('name')
    @classmethod
    def validate_step_name_pascal_case(cls, v: str) -> str:
        """Enforce PascalCase naming for step names per standardization rules."""
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', v):
            raise ValueError(
                f"Step name '{v}' must be PascalCase. "
                f"Examples: 'CradleDataLoading', 'XGBoostTraining', 'PyTorchModel'. "
                f"Counter-examples: 'cradle_data_loading', 'xgboost_training', 'PytorchTraining'"
            )
        return v

    @field_validator('config_class')
    @classmethod
    def validate_config_class_naming(cls, v: Optional[str]) -> Optional[str]:
        """Enforce config class naming convention: PascalCase + 'Config' suffix."""
        if v is not None:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*Config$', v):
                raise ValueError(
                    f"Config class '{v}' must follow pattern: PascalCaseConfig. "
                    f"Examples: 'XGBoostTrainingConfig', 'CradleDataLoadConfig'. "
                    f"Counter-examples: 'XGBoostTrainingConfiguration', 'xgboost_config'"
                )
        return v

    @field_validator('builder_step_name')
    @classmethod
    def validate_builder_naming(cls, v: Optional[str]) -> Optional[str]:
        """Enforce builder class naming convention: PascalCase + 'StepBuilder' suffix."""
        if v is not None:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*StepBuilder$', v):
                raise ValueError(
                    f"Builder class '{v}' must follow pattern: PascalCaseStepBuilder. "
                    f"Examples: 'XGBoostTrainingStepBuilder', 'CradleDataLoadingStepBuilder'. "
                    f"Counter-examples: 'XGBoostTrainer', 'DataLoadingBuilder'"
                )
        return v

    @field_validator('workspace_id')
    @classmethod
    def validate_workspace_id_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate workspace ID follows naming conventions."""
        if v is not None:
            if not re.match(r'^[a-z][a-z0-9_]*$', v):
                raise ValueError(
                    f"Workspace ID '{v}' must be snake_case. "
                    f"Examples: 'project_alpha', 'ml_experiments'. "
                    f"Counter-examples: 'ProjectAlpha', 'ml-experiments'"
                )
        return v

    @model_validator(mode='after')
    def validate_naming_consistency(self) -> 'StepDefinition':
        """Validate consistency between related naming fields."""
        errors = []
        
        # Validate config class matches step name pattern
        if self.config_class and self.name:
            expected_configs = [
                f"{self.name}Config",  # Standard pattern
                f"{self.name.rstrip('ing')}Config"  # Handle -ing endings (e.g., CradleDataLoading -> CradleDataLoadConfig)
            ]
            if self.config_class not in expected_configs:
                errors.append(
                    f"Config class '{self.config_class}' doesn't match expected patterns for step '{self.name}': {expected_configs}"
                )
        
        # Validate builder name matches step name
        if self.builder_step_name and self.name:
            expected_builder = f"{self.name}StepBuilder"
            if self.builder_step_name != expected_builder:
                errors.append(
                    f"Builder name '{self.builder_step_name}' should be '{expected_builder}'"
                )
        
        # Validate spec type matches step name
        if self.spec_type and self.name:
            if self.spec_type != self.name:
                errors.append(
                    f"Spec type '{self.spec_type}' should match step name '{self.name}'"
                )
        
        if errors:
            raise ValueError(f"Naming consistency violations: {'; '.join(errors)}")
        
        return self

    @model_validator(mode='after')
    def validate_step_type_consistency(self) -> 'StepDefinition':
        """Validate consistency between SageMaker step type and expected patterns."""
        errors = []
        
        if self.sagemaker_step_type == SageMakerStepType.PROCESSING:
            # Processing steps should follow specific patterns
            if self.builder_step_name and not self._is_processing_builder():
                errors.append(f"Processing step '{self.name}' should have builder creating ProcessingStep")
        
        elif self.sagemaker_step_type == SageMakerStepType.TRAINING:
            # Training steps should follow specific patterns
            if self.builder_step_name and not self._is_training_builder():
                errors.append(f"Training step '{self.name}' should have builder creating TrainingStep")
        
        elif self.sagemaker_step_type == SageMakerStepType.CREATE_MODEL:
            # CreateModel steps should follow specific patterns
            if self.builder_step_name and not self._is_create_model_builder():
                errors.append(f"CreateModel step '{self.name}' should have builder creating CreateModelStep")
        
        elif self.sagemaker_step_type == SageMakerStepType.TRANSFORM:
            # Transform steps should follow specific patterns
            if self.builder_step_name and not self._is_transform_builder():
                errors.append(f"Transform step '{self.name}' should have builder creating TransformStep")
        
        if errors:
            raise ValueError(f"Step type consistency violations: {'; '.join(errors)}")
        
        return self

    @model_validator(mode='after')
    def validate_registry_consistency(self) -> 'StepDefinition':
        """Validate registry type consistency with workspace context."""
        errors = []
        
        if self.registry_type == RegistryType.WORKSPACE and not self.workspace_id:
            errors.append("Workspace registry type requires workspace_id")
        
        if self.registry_type == RegistryType.CORE and self.workspace_id:
            errors.append("Core registry type should not have workspace_id")
        
        if self.registry_type == RegistryType.OVERRIDE and not self.override_source:
            errors.append("Override registry type should specify override_source")
        
        if errors:
            raise ValueError(f"Registry consistency violations: {'; '.join(errors)}")
        
        return self

    @model_validator(mode='after')
    def validate_against_registry_patterns(self) -> 'StepDefinition':
        """Validate against known STEP_NAMES registry patterns."""
        try:
            from cursus.registry.step_names import STEP_NAMES
            
            errors = []
            
            # If this step exists in STEP_NAMES, validate consistency
            if self.name in STEP_NAMES:
                registry_entry = STEP_NAMES[self.name]
                
                # Validate config class matches registry
                if self.config_class and self.config_class != registry_entry.get("config_class"):
                    errors.append(
                        f"Config class mismatch with registry: expected '{registry_entry.get('config_class')}', got '{self.config_class}'"
                    )
                
                # Validate builder name matches registry
                if self.builder_step_name and self.builder_step_name != registry_entry.get("builder_step_name"):
                    errors.append(
                        f"Builder name mismatch with registry: expected '{registry_entry.get('builder_step_name')}', got '{self.builder_step_name}'"
                    )
                
                # Validate SageMaker step type matches registry
                if self.sagemaker_step_type and str(self.sagemaker_step_type) != registry_entry.get("sagemaker_step_type"):
                    errors.append(
                        f"SageMaker step type mismatch with registry: expected '{registry_entry.get('sagemaker_step_type')}', got '{self.sagemaker_step_type}'"
                    )
            
            if errors:
                raise ValueError(f"Registry consistency violations: {'; '.join(errors)}")
        
        except ImportError:
            # Registry not available, skip validation
            pass
        
        return self

    # Helper methods for step type validation
    def _is_processing_builder(self) -> bool:
        """Check if this is a processing step builder."""
        processing_patterns = [
            'TabularPreprocessing', 'RiskTableMapping', 'CurrencyConversion',
            'XGBoostModelEval', 'ModelCalibration', 'Package', 'Payload',
            'CradleDataLoading'
        ]
        return self.name in processing_patterns

    def _is_training_builder(self) -> bool:
        """Check if this is a training step builder."""
        training_patterns = ['XGBoostTraining', 'PyTorchTraining', 'DummyTraining']
        return self.name in training_patterns

    def _is_create_model_builder(self) -> bool:
        """Check if this is a create model step builder."""
        create_model_patterns = ['XGBoostModel', 'PyTorchModel']
        return self.name in create_model_patterns

    def _is_transform_builder(self) -> bool:
        """Check if this is a transform step builder."""
        transform_patterns = ['BatchTransform']
        return self.name in transform_patterns

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format with validation."""
        from .utils import to_legacy_format
        return to_legacy_format(self)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation rules applied."""
        return {
            'step_name': self.name,
            'naming_rules_applied': [
                'PascalCase step name',
                'Config class naming pattern',
                'Builder class naming pattern',
                'Workspace ID snake_case'
            ],
            'consistency_checks': [
                'Config class matches step name',
                'Builder name matches step name',
                'Spec type matches step name',
                'SageMaker step type consistency',
                'Registry type consistency'
            ],
            'registry_validation': 'Validated against STEP_NAMES registry',
            'enum_validations': [
                'SageMaker step type',
                'Framework type',
                'Job types',
                'Registry type',
                'Resolution strategy'
            ]
        }
```

### 2. Enhanced Validation Error Messages

Provide comprehensive error messages that guide developers toward compliance:

```python
class StandardizationValidationError(ValueError):
    """Enhanced validation error with standardization guidance."""
    
    def __init__(self, message: str, field: str = None, suggestions: List[str] = None):
        self.field = field
        self.suggestions = suggestions or []
        super().__init__(message)
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.suggestions:
            msg += f"\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)
        return msg


def create_naming_error(field: str, value: str, pattern: str, examples: List[str], counter_examples: List[str]) -> StandardizationValidationError:
    """Create a comprehensive naming validation error."""
    message = f"{field} '{value}' doesn't follow the required pattern: {pattern}"
    suggestions = [
        f"Use examples like: {', '.join(examples)}",
        f"Avoid patterns like: {', '.join(counter_examples)}",
        "See standardization_rules.md for complete naming conventions"
    ]
    return StandardizationValidationError(message, field, suggestions)
```

### 3. Integration with Registry System

Enhance the hybrid registry to use StepDefinition validation:

```python
class ValidatedHybridRegistry:
    """Hybrid registry with automatic standardization enforcement."""
    
    def register_step(self, step_definition: Dict[str, Any]) -> StepDefinition:
        """Register a step with automatic validation."""
        try:
            # This will automatically validate all standardization rules
            validated_definition = StepDefinition(**step_definition)
            
            # Store in registry
            self._store_definition(validated_definition)
            
            return validated_definition
            
        except ValidationError as e:
            # Enhance error message with standardization guidance
            enhanced_errors = []
            for error in e.errors():
                field = error.get('loc', ['unknown'])[0]
                message = error.get('msg', 'Validation failed')
                enhanced_errors.append(f"{field}: {message}")
            
            raise StandardizationValidationError(
                f"Step definition validation failed: {'; '.join(enhanced_errors)}",
                suggestions=[
                    "Check naming conventions in standardization_rules.md",
                    "Validate against SageMaker step type classification",
                    "Ensure consistency between related fields"
                ]
            )
    
    def validate_all_definitions(self) -> Dict[str, Any]:
        """Validate all existing definitions against current standards."""
        results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }
        
        for name, definition_data in self._get_all_definitions():
            try:
                validated = StepDefinition(**definition_data)
                results['valid'].append(name)
            except ValidationError as e:
                results['invalid'].append({
                    'name': name,
                    'errors': [error['msg'] for error in e.errors()]
                })
        
        return results
```

### 4. Development Tools and CLI Integration

Provide tools for developers to validate and fix standardization issues:

```python
class StandardizationValidator:
    """CLI tool for validating standardization compliance."""
    
    def validate_step_definition(self, definition_dict: Dict[str, Any]) -> bool:
        """Validate a single step definition."""
        try:
            StepDefinition(**definition_dict)
            print(f"✅ Step definition is compliant with standardization rules")
            return True
        except ValidationError as e:
            print(f"❌ Step definition validation failed:")
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                print(f"  {field}: {error['msg']}")
            return False
    
    def suggest_fixes(self, definition_dict: Dict[str, Any]) -> Dict[str, str]:
        """Suggest fixes for common standardization issues."""
        suggestions = {}
        
        # Check step name
        if 'name' in definition_dict:
            name = definition_dict['name']
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                suggestions['name'] = f"Convert '{name}' to PascalCase (e.g., '{self._to_pascal_case(name)}')"
        
        # Check config class
        if 'config_class' in definition_dict:
            config_class = definition_dict['config_class']
            if not config_class.endswith('Config'):
                suggestions['config_class'] = f"Add 'Config' suffix: '{config_class}Config'"
        
        return suggestions
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase."""
        return ''.join(word.capitalize() for word in re.split(r'[_\-\s]+', text))
```

## Implementation Strategy

### Phase 1: Model Enhancement (Week 1)
1. **Extend StepDefinition Model**: Add comprehensive validation rules
2. **Create Validation Enums**: Define all valid values for constrained fields
3. **Implement Field Validators**: Add individual field validation methods
4. **Add Model Validators**: Implement cross-field consistency checks

### Phase 2: Registry Integration (Week 2)
1. **Update Hybrid Registry**: Integrate StepDefinition validation
2. **Enhance Error Handling**: Provide detailed error messages and suggestions
3. **Add Validation Methods**: Implement registry-wide validation capabilities
4. **Create Migration Tools**: Help existing definitions comply with new standards

### Phase 3: Testing and Validation (Week 3)
1. **Comprehensive Test Suite**: Test all validation rules and edge cases
2. **Integration Testing**: Validate with existing registry system
3. **Performance Testing**: Ensure validation doesn't impact performance
4. **Backward Compatibility**: Ensure existing code continues to work

### Phase 4: Developer Tools and Documentation (Week 4)
1. **CLI Validation Tools**: Create command-line utilities for validation
2. **IDE Integration**: Provide validation feedback in development environments
3. **Documentation Updates**: Update all relevant documentation
4. **Migration Guides**: Help developers adopt new validation system

## Benefits

### 1. Automatic Enforcement
- **Immediate Feedback**: Validation errors caught at model creation time
- **Consistent Standards**: All step definitions automatically comply with rules
- **Reduced Manual Review**: Less need for manual standardization checking
- **Prevention Over Correction**: Issues prevented rather than fixed later

### 2. Enhanced Developer Experience
- **Clear Error Messages**: Specific guidance on how to fix violations
- **Examples and Counter-Examples**: Learn correct patterns from error messages
- **IDE Support**: Real-time validation feedback during development
- **Automated Suggestions**: Tools to automatically fix common issues

### 3. System Quality Improvements
- **Registry Consistency**: All entries follow the same patterns
- **Reduced Technical Debt**: Prevent accumulation of non-standard code
- **Easier Maintenance**: Consistent patterns make code easier to understand
- **Better Tooling**: Standardized patterns enable better automated tools

### 4. Integration with Testing Framework
- **Step-Type-Specific Validation**: Leverage SageMaker step type classification
- **Universal Test Compatibility**: Work seamlessly with existing test framework
- **Automated Compliance Checking**: Include in CI/CD pipelines
- **Quality Gates**: Prevent non-compliant code from being merged

## Integration with Existing Systems

### 1. Hybrid Registry System
The enhanced StepDefinition model integrates seamlessly with the existing hybrid registry:

```python
# Existing registry usage remains the same
registry = UnifiedRegistryManager()

# But now with automatic validation
step_def = {
    "name": "MyNewStep",  # Will be validated for PascalCase
    "config_class": "MyNewStepConfig",  # Will be validated for naming pattern
    "sagemaker_step_type": "Processing"  # Will be validated against enum
}

# This will automatically validate all standardization rules
validated_step = registry.register_step(step_def)
```

### 2. Universal Builder Test Framework
The validation integrates with the testing framework mentioned in the SageMaker step type classification design:

```python
class UniversalStepBuilderTest:
    """Enhanced with StepDefinition validation."""
    
    def test_step_definition_compliance(self):
        """Test that step builder matches standardization rules."""
        step_definition = self._extract_step_definition()
        
        # This automatically validates all naming conventions
        try:
            validated = StepDefinition(**step_definition)
            self.assertTrue(True, "Step definition complies with standardization rules")
        except ValidationError as e:
            self.fail(f"Step definition validation failed: {e}")
```

### 3. Workspace-Aware System
The validation works with the workspace-aware system from the migration plans:

```python
# Workspace-specific validation
workspace_step = {
    "name": "WorkspaceSpecificStep",
    "registry_type": "workspace",
    "workspace_id": "project_alpha",  # Validated for snake_case
    "sagemaker_step_type": "Processing"
}

# Validation ensures workspace consistency
validated_step = StepDefinition(**workspace_step)
```

## Risk Mitigation

### 1. Breaking Changes
**Risk**: Existing step definitions may not comply with new validation rules
**Mitigation**: 
- Gradual rollout with opt-in validation initially
- Migration tools to automatically fix common issues
- Comprehensive testing of existing definitions
- Backward compatibility mode for legacy definitions

### 2. Performance Impact
**Risk**: Validation may slow down registry operations
**Mitigation**:
- Optimize validation logic for performance
- Cache validation results where appropriate
- Profile and benchmark validation performance
- Provide option to disable validation in production if needed

### 3. Developer Adoption
**Risk**: Developers may find new validation rules restrictive
**Mitigation**:
- Clear documentation and examples
- Helpful error messages with suggestions
- Migration tools to ease transition
- Training and support for development teams

## Future Enhancements

### 1. Dynamic Validation Rules
- Load validation rules from configuration files
- Allow workspace-specific validation customizations
- Support for validation rule versioning
- Runtime validation rule updates

### 2. Advanced Pattern Recognition
- Machine learning-based pattern detection
- Automatic suggestion of naming improvements
- Detection of anti-patterns in step definitions
- Intelligent validation rule recommendations

### 3. Integration with External Systems
- Validation against external naming standards
- Integration with corporate governance systems
- Compliance reporting and auditing
- Integration with code review systems

### 4. Enhanced Developer Tools
- Visual validation rule editors
- Real-time validation in web interfaces
- Batch validation and fixing tools
- Validation rule impact analysis

## Testing Strategy

### 1. Unit Testing
```python
class TestStepDefinitionValidation(unittest.TestCase):
    """Comprehensive tests for StepDefinition validation."""
    
    def test_pascal_case_step_names(self):
        """Test PascalCase validation for step names."""
        # Valid names
        valid_names = ["XGBoostTraining", "CradleDataLoading", "PyTorchModel"]
        for name in valid_names:
            step_def = StepDefinition(name=name, registry_type="core")
            self.assertEqual(step_def.name, name)
        
        # Invalid names
        invalid_names = ["xgboost_training", "cradle-data-loading", "pytorchModel"]
        for name in invalid_names:
            with self.assertRaises(ValidationError):
                StepDefinition(name=name, registry_type="core")
    
    def test_config_class_naming(self):
        """Test config class naming validation."""
        # Valid config classes
        valid_configs = ["XGBoostTrainingConfig", "CradleDataLoadConfig"]
        for config in valid_configs:
            step_def = StepDefinition(
                name="TestStep", 
                registry_type="core",
                config_class=config
            )
            self.assertEqual(step_def.config_class, config)
        
        # Invalid config classes
        invalid_configs = ["XGBoostTrainingConfiguration", "xgboost_config"]
        for config in invalid_configs:
            with self.assertRaises(ValidationError):
                StepDefinition(
                    name="TestStep",
                    registry_type="core", 
                    config_class=config
                )
```

### 2. Integration Testing
```python
class TestRegistryIntegration(unittest.TestCase):
    """Test integration with hybrid registry system."""
    
    def test_registry_validation_integration(self):
        """Test that registry uses StepDefinition validation."""
        registry = UnifiedRegistryManager()
        
        # Valid step definition
        valid_step = {
            "name": "TestStep",
            "registry_type": "core",
            "config_class": "TestStepConfig",
            "sagemaker_step_type": "Processing"
        }
        
        # Should succeed
        result = registry.register_step(valid_step)
        self.assertIsInstance(result, StepDefinition)
        
        # Invalid step definition
        invalid_step = {
            "name": "test_step",  # Invalid: not PascalCase
            "registry_type": "core"
        }
        
        # Should fail with validation error
        with self.assertRaises(StandardizationValidationError):
            registry.register_step(invalid_step)
```

## Migration Guide

### 1. Existing Step Definitions
For existing step definitions that may not comply with new validation rules:

```bash
# Check compliance of existing definitions
python -m cursus.cli.validation validate-all-steps

# Get suggestions for fixing issues
python -m cursus.cli.validation suggest-fixes --step-name "YourStepName"

# Automatically fix common issues
python -m cursus.cli.validation auto-fix --step-name "YourStepName" --dry-run
```

### 2. Development Workflow Updates
Update development workflows to include validation:

```yaml
# .github/workflows/validation.yml
name: Standardization Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Validate Step Definitions
        run: python -m cursus.cli.validation validate-all-steps --strict
```

### 3. IDE Integration
Configure IDEs to provide real-time validation feedback:

```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.pylintArgs": [
    "--load-plugins=cursus.validation.pylint_plugin"
  ]
}
```

## Conclusion

The StepDefinition standardization enforcement design provides a comprehensive solution for automatically enforcing naming conventions and standardization rules across the entire registry system. By leveraging Pydantic's powerful validation capabilities, we can ensure consistent compliance with established standards while providing immediate feedback to developers.

Key benefits include:
- **Automatic enforcement** of all standardization rules
- **Enhanced developer experience** with clear error messages and suggestions
- **System quality improvements** through consistent patterns
- **Seamless integration** with existing hybrid registry and testing frameworks

The phased implementation approach ensures minimal disruption to existing workflows while providing powerful new capabilities for maintaining code quality and consistency.

This design builds upon and integrates with several existing systems and plans, creating a cohesive approach to standardization enforcement that will scale with the project's growth and evolution.

## References

### Design Documents
- [standardization_rules.md](standardization_rules.md) - Comprehensive standardization rules and naming conventions
- [sagemaker_step_type_classification_design.md](sagemaker_step_type_classification_design.md) - SageMaker step type classification system
- [workspace_aware_distributed_registry_design.md](workspace_aware_distributed_registry_design.md) - Workspace-aware registry architecture
- [hybrid_registry_standardization_enforcement_design.md](hybrid_registry_standardization_enforcement_design.md) - Registry-level standardization enforcement
- [documentation_yaml_frontmatter_standard.md](documentation_yaml_frontmatter_standard.md) - Documentation metadata standards

### Project Planning Documents
- [2025-09-02_workspace_aware_hybrid_registry_migration_plan.md](../2_project_planning/2025-09-02_workspace_aware_hybrid_registry_migration_plan.md) - Hybrid registry migration implementation plan
- [2025-09-02_workspace_aware_system_refactoring_migration_plan.md](../2_project_planning/2025-09-02_workspace_aware_system_refactoring_migration_plan.md) - Overall system refactoring plan
- [2025-09-04_hybrid_registry_redundancy_reduction_plan.md](../2_project_planning/2025-09-04_hybrid_registry_redundancy_reduction_plan.md) - Registry optimization and redundancy reduction

### Related Components
- `src/cursus/registry/hybrid/models.py` - Current StepDefinition implementation
- `src/cursus/registry/hybrid/manager.py` - Hybrid registry manager
- `src/cursus/validation/` - Validation framework components
- `test/steps/builders/` - Universal builder test framework

### External References
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/) - Pydantic validation framework
- [SageMaker Workflow Steps](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html) - SageMaker step types reference
