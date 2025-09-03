"""
Naming Standard Validator for the Cursus pipeline framework.

This module provides validation for naming conventions as defined in the
standardization rules document. It validates component names, file names,
and ensures consistency with the STEP_NAMES registry.
"""

import re
import inspect
from typing import List, Dict, Any, Optional, Type, Union
from pathlib import Path

from ...core.base.builder_base import StepBuilderBase
from ...core.base.config_base import BasePipelineConfig
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...registry.step_names import STEP_NAMES


class NamingViolation:
    """Represents a naming convention violation."""
    
    def __init__(self, component: str, violation_type: str, message: str, 
                 expected: str = None, actual: str = None, suggestions: List[str] = None):
        self.component = component
        self.violation_type = violation_type
        self.message = message
        self.expected = expected
        self.actual = actual
        self.suggestions = suggestions or []
    
    def __str__(self) -> str:
        result = f"{self.component}: {self.message}"
        if self.expected and self.actual:
            result += f" (Expected: {self.expected}, Actual: {self.actual})"
        if self.suggestions:
            result += f" Suggestions: {', '.join(self.suggestions)}"
        return result


class NamingStandardValidator:
    """
    Validator for naming conventions and standardization rules.
    
    This validator enforces the naming patterns defined in the standardization
    rules document, including:
    - Canonical step names (PascalCase)
    - Config class names (PascalCase + Config suffix)
    - Builder class names (PascalCase + StepBuilder suffix)
    - File naming patterns
    - Registry consistency
    """
    
    def __init__(self):
        """Initialize the naming validator."""
        self.violations = []
    
    def validate_step_specification(self, spec: StepSpecification) -> List[NamingViolation]:
        """
        Validate naming conventions for a step specification.
        
        Args:
            spec: Step specification to validate
            
        Returns:
            List of naming violations found
        """
        violations = []
        
        if not spec:
            violations.append(NamingViolation(
                component="StepSpecification",
                violation_type="missing_spec",
                message="Step specification is None or missing"
            ))
            return violations
        
        # Validate spec type name
        spec_type = getattr(spec, 'step_type', None)
        if spec_type:
            violations.extend(self._validate_canonical_step_name(spec_type, "StepSpecification"))
        
        # Validate logical names in inputs/outputs
        if hasattr(spec, 'inputs') and spec.inputs:
            for input_spec in spec.inputs:
                logical_name = getattr(input_spec, 'logical_name', None)
                if logical_name:
                    violations.extend(self._validate_logical_name(logical_name, f"Input '{logical_name}'"))
        
        if hasattr(spec, 'outputs') and spec.outputs:
            for output_spec in spec.outputs:
                logical_name = getattr(output_spec, 'logical_name', None)
                if logical_name:
                    violations.extend(self._validate_logical_name(logical_name, f"Output '{logical_name}'"))
        
        return violations
    
    def validate_step_builder_class(self, builder_class: Type[StepBuilderBase]) -> List[NamingViolation]:
        """
        Validate naming conventions for a step builder class.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of naming violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Validate builder class name pattern
        violations.extend(self._validate_builder_class_name(class_name))
        
        # Validate registry consistency
        violations.extend(self._validate_registry_consistency(builder_class))
        
        # Validate method naming conventions
        violations.extend(self._validate_method_names(builder_class))
        
        return violations
    
    def validate_config_class(self, config_class: Type[BasePipelineConfig]) -> List[NamingViolation]:
        """
        Validate naming conventions for a configuration class.
        
        Args:
            config_class: Configuration class to validate
            
        Returns:
            List of naming violations found
        """
        violations = []
        class_name = config_class.__name__
        
        # Validate config class name pattern
        violations.extend(self._validate_config_class_name(class_name))
        
        # Validate field naming conventions
        violations.extend(self._validate_config_field_names(config_class))
        
        return violations
    
    def validate_file_naming(self, file_path: Union[str, Path], file_type: str) -> List[NamingViolation]:
        """
        Validate file naming conventions.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (builder, config, spec, contract)
            
        Returns:
            List of naming violations found
        """
        violations = []
        path = Path(file_path)
        filename = path.name
        
        expected_patterns = {
            'builder': r'^builder_[a-z_]+_step\.py$',
            'config': r'^config_[a-z_]+_step\.py$',
            'spec': r'^[a-z_]+_spec\.py$',
            'contract': r'^[a-z_]+_contract\.py$'
        }
        
        if file_type in expected_patterns:
            pattern = expected_patterns[file_type]
            if not re.match(pattern, filename):
                violations.append(NamingViolation(
                    component=f"File '{filename}'",
                    violation_type="file_naming",
                    message=f"File name doesn't match expected pattern for {file_type} files",
                    expected=f"Pattern: {pattern}",
                    actual=filename,
                    suggestions=self._suggest_file_name(filename, file_type)
                ))
        
        return violations
    
    def validate_registry_entry(self, step_name: str, registry_info: Dict[str, Any]) -> List[NamingViolation]:
        """
        Validate a registry entry for naming consistency.
        
        Args:
            step_name: Canonical step name
            registry_info: Registry information dictionary
            
        Returns:
            List of naming violations found
        """
        violations = []
        
        # Validate canonical step name
        violations.extend(self._validate_canonical_step_name(step_name, "Registry Entry"))
        
        # Validate config class name
        config_class = registry_info.get('config_class')
        if config_class:
            violations.extend(self._validate_config_class_name_from_registry(
                config_class, step_name, "Registry Config Class"
            ))
        
        # Validate builder step name
        builder_name = registry_info.get('builder_step_name')
        if builder_name:
            violations.extend(self._validate_builder_class_name_from_registry(
                builder_name, step_name, "Registry Builder Class"
            ))
        
        # Validate spec type
        spec_type = registry_info.get('spec_type')
        if spec_type:
            if spec_type != step_name:
                violations.append(NamingViolation(
                    component="Registry Spec Type",
                    violation_type="spec_type_mismatch",
                    message="Spec type should match canonical step name",
                    expected=step_name,
                    actual=spec_type
                ))
        
        # Validate SageMaker step type
        sagemaker_type = registry_info.get('sagemaker_step_type')
        if sagemaker_type:
            violations.extend(self._validate_sagemaker_step_type(sagemaker_type, step_name))
        
        return violations
    
    def _validate_canonical_step_name(self, name: str, component: str) -> List[NamingViolation]:
        """Validate canonical step name follows PascalCase pattern, allowing job type variants."""
        violations = []
        
        if not name:
            violations.append(NamingViolation(
                component=component,
                violation_type="empty_name",
                message="Canonical step name cannot be empty"
            ))
            return violations
        
        # Check if this is a job type variant (e.g., "TabularPreprocessing_Training")
        valid_job_types = ['Training', 'Testing', 'Validation', 'Calibration']
        is_job_type_variant = False
        base_name = name
        job_type = None
        
        if '_' in name:
            parts = name.split('_')
            if len(parts) == 2:
                potential_base, potential_job_type = parts
                if potential_job_type in valid_job_types:
                    is_job_type_variant = True
                    base_name = potential_base
                    job_type = potential_job_type
        
        if is_job_type_variant:
            # Validate the base name (should be PascalCase and exist in registry)
            if not re.match(r'^[A-Z][A-Za-z0-9]*$', base_name):
                violations.append(NamingViolation(
                    component=component,
                    violation_type="pascal_case_base",
                    message=f"Base step name '{base_name}' in job type variant must be in PascalCase",
                    expected="PascalCase (e.g., TabularPreprocessing, CurrencyConversion)",
                    actual=base_name,
                    suggestions=[self._to_pascal_case(base_name)]
                ))
            
            # Check if base name exists in registry
            if base_name not in STEP_NAMES:
                violations.append(NamingViolation(
                    component=component,
                    violation_type="unknown_base_step",
                    message=f"Base step name '{base_name}' not found in STEP_NAMES registry",
                    actual=base_name,
                    suggestions=list(STEP_NAMES.keys())[:5]  # Show first 5 as suggestions
                ))
            
            # Validate job type capitalization
            if job_type and job_type[0].islower():
                violations.append(NamingViolation(
                    component=component,
                    violation_type="job_type_capitalization",
                    message=f"Job type '{job_type}' should be capitalized",
                    expected=job_type.capitalize(),
                    actual=job_type,
                    suggestions=[job_type.capitalize()]
                ))
        else:
            # Standard validation for non-job-type variants
            # Check PascalCase pattern
            if not re.match(r'^[A-Z][A-Za-z0-9]*$', name):
                violations.append(NamingViolation(
                    component=component,
                    violation_type="pascal_case",
                    message="Canonical step name must be in PascalCase",
                    expected="PascalCase (e.g., CradleDataLoading, XGBoostTraining)",
                    actual=name,
                    suggestions=[self._to_pascal_case(name)]
                ))
            
            # Check for underscores in non-job-type variants
            if '_' in name:
                violations.append(NamingViolation(
                    component=component,
                    violation_type="underscore_in_name",
                    message="Canonical step name should not contain underscores (unless it's a job type variant like StepName_Training)",
                    actual=name,
                    suggestions=[name.replace('_', '')]
                ))
        
        # Common validation for both variants and non-variants
        if name.lower() == name:
            violations.append(NamingViolation(
                component=component,
                violation_type="lowercase_name",
                message="Canonical step name should not be all lowercase",
                actual=name,
                suggestions=[self._to_pascal_case(name)]
            ))
        
        return violations
    
    def _validate_config_class_name(self, class_name: str) -> List[NamingViolation]:
        """Validate config class name follows naming pattern."""
        violations = []
        
        if not class_name.endswith('Config'):
            violations.append(NamingViolation(
                component=f"Config Class '{class_name}'",
                violation_type="config_suffix",
                message="Config class name must end with 'Config'",
                expected=f"{class_name}Config",
                actual=class_name
            ))
        
        # Check PascalCase pattern
        base_name = class_name[:-6] if class_name.endswith('Config') else class_name
        if not re.match(r'^[A-Z][A-Za-z0-9]*$', base_name):
            violations.append(NamingViolation(
                component=f"Config Class '{class_name}'",
                violation_type="pascal_case",
                message="Config class base name must be in PascalCase",
                expected="PascalCase + Config (e.g., CradleDataLoadConfig, XGBoostTrainingConfig)",
                actual=class_name
            ))
        
        return violations
    
    def _validate_config_class_name_from_registry(self, config_class: str, step_name: str, component: str) -> List[NamingViolation]:
        """Validate config class name matches registry patterns."""
        violations = []
        
        # Basic validation
        violations.extend(self._validate_config_class_name(config_class))
        
        # Check consistency with step name
        expected_patterns = [
            f"{step_name}Config",  # Full name + Config
            f"{step_name[:-3]}Config" if step_name.endswith('ing') else None,  # Remove 'ing' + Config
        ]
        expected_patterns = [p for p in expected_patterns if p]  # Remove None values
        
        if config_class not in expected_patterns:
            violations.append(NamingViolation(
                component=component,
                violation_type="config_pattern_mismatch",
                message=f"Config class name doesn't follow expected patterns for step '{step_name}'",
                expected=f"One of: {', '.join(expected_patterns)}",
                actual=config_class
            ))
        
        return violations
    
    def _validate_builder_class_name(self, class_name: str) -> List[NamingViolation]:
        """Validate builder class name follows naming pattern."""
        violations = []
        
        if not class_name.endswith('StepBuilder'):
            violations.append(NamingViolation(
                component=f"Builder Class '{class_name}'",
                violation_type="builder_suffix",
                message="Builder class name must end with 'StepBuilder'",
                expected=f"{class_name}StepBuilder",
                actual=class_name
            ))
        
        # Check PascalCase pattern
        base_name = class_name[:-11] if class_name.endswith('StepBuilder') else class_name
        if not re.match(r'^[A-Z][A-Za-z0-9]*$', base_name):
            violations.append(NamingViolation(
                component=f"Builder Class '{class_name}'",
                violation_type="pascal_case",
                message="Builder class base name must be in PascalCase",
                expected="PascalCase + StepBuilder (e.g., CradleDataLoadingStepBuilder)",
                actual=class_name
            ))
        
        return violations
    
    def _validate_builder_class_name_from_registry(self, builder_name: str, step_name: str, component: str) -> List[NamingViolation]:
        """Validate builder class name matches registry patterns."""
        violations = []
        
        # Basic validation
        violations.extend(self._validate_builder_class_name(builder_name))
        
        # Check consistency with step name
        expected_name = f"{step_name}StepBuilder"
        
        if builder_name != expected_name:
            violations.append(NamingViolation(
                component=component,
                violation_type="builder_pattern_mismatch",
                message=f"Builder class name doesn't match expected pattern for step '{step_name}'",
                expected=expected_name,
                actual=builder_name
            ))
        
        return violations
    
    def _validate_logical_name(self, name: str, component: str) -> List[NamingViolation]:
        """Validate logical name follows snake_case pattern."""
        violations = []
        
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            violations.append(NamingViolation(
                component=component,
                violation_type="snake_case",
                message="Logical name must be in snake_case",
                expected="snake_case (e.g., input_data, model_artifacts)",
                actual=name,
                suggestions=[self._to_snake_case(name)]
            ))
        
        # Check for common anti-patterns
        if name.startswith('_') or name.endswith('_'):
            violations.append(NamingViolation(
                component=component,
                violation_type="underscore_boundary",
                message="Logical name should not start or end with underscore",
                actual=name,
                suggestions=[name.strip('_')]
            ))
        
        if '__' in name:
            violations.append(NamingViolation(
                component=component,
                violation_type="double_underscore",
                message="Logical name should not contain double underscores",
                actual=name,
                suggestions=[re.sub(r'__+', '_', name)]
            ))
        
        return violations
    
    def _validate_method_names(self, builder_class: Type[StepBuilderBase]) -> List[NamingViolation]:
        """Validate method naming conventions."""
        violations = []
        
        for method_name in dir(builder_class):
            if callable(getattr(builder_class, method_name)) and not method_name.startswith('__'):
                # Public methods should be snake_case
                if not method_name.startswith('_'):
                    if not re.match(r'^[a-z][a-z0-9_]*$', method_name):
                        violations.append(NamingViolation(
                            component=f"Method '{method_name}' in {builder_class.__name__}",
                            violation_type="method_snake_case",
                            message="Public method name should be in snake_case",
                            expected="snake_case",
                            actual=method_name,
                            suggestions=[self._to_snake_case(method_name)]
                        ))
        
        return violations
    
    def _validate_config_field_names(self, config_class: Type[BasePipelineConfig]) -> List[NamingViolation]:
        """Validate configuration field naming conventions."""
        violations = []
        
        # Get field names from the class
        if hasattr(config_class, '__fields__'):
            # Pydantic v1 style
            fields = config_class.__fields__.keys()
        elif hasattr(config_class, 'model_fields'):
            # Pydantic v2 style
            fields = config_class.model_fields.keys()
        else:
            # Fallback to class attributes
            fields = [attr for attr in dir(config_class) 
                     if not attr.startswith('__') and not callable(getattr(config_class, attr))]
        
        for field_name in fields:
            # Skip private fields (they have different rules)
            if field_name.startswith('_'):
                continue
                
            # Public fields should be snake_case
            if not re.match(r'^[a-z][a-z0-9_]*$', field_name):
                violations.append(NamingViolation(
                    component=f"Field '{field_name}' in {config_class.__name__}",
                    violation_type="field_snake_case",
                    message="Configuration field name should be in snake_case",
                    expected="snake_case",
                    actual=field_name,
                    suggestions=[self._to_snake_case(field_name)]
                ))
        
        return violations
    
    def _validate_registry_consistency(self, builder_class: Type[StepBuilderBase]) -> List[NamingViolation]:
        """Validate builder class consistency with registry."""
        violations = []
        class_name = builder_class.__name__
        
        # Create reverse mapping from builder names to step names
        reverse_builder_mapping = {
            info["builder_step_name"]: step_name 
            for step_name, info in STEP_NAMES.items()
        }
        
        # Check if class is in reverse mapping
        if class_name in reverse_builder_mapping:
            step_name = reverse_builder_mapping[class_name]
            registry_info = STEP_NAMES.get(step_name, {})
            
            # Validate that registry info is consistent
            expected_builder_name = registry_info.get('builder_step_name')
            if expected_builder_name != class_name:
                violations.append(NamingViolation(
                    component=f"Builder Class '{class_name}'",
                    violation_type="registry_inconsistency",
                    message="Builder class name doesn't match registry entry",
                    expected=expected_builder_name,
                    actual=class_name
                ))
        else:
            # Class not found in registry - this might be intentional for custom builders
            violations.append(NamingViolation(
                component=f"Builder Class '{class_name}'",
                violation_type="registry_missing",
                message="Builder class not found in STEP_NAMES registry",
                suggestions=["Add entry to STEP_NAMES registry or use @register_builder decorator"]
            ))
        
        return violations
    
    def _validate_sagemaker_step_type(self, sagemaker_type: str, step_name: str) -> List[NamingViolation]:
        """Validate SageMaker step type naming."""
        violations = []
        
        valid_types = [
            'Processing', 'Training', 'Transform', 'CreateModel', 'RegisterModel',
            'Lambda', 'MimsModelRegistrationProcessing', 'CradleDataLoading', 'Base'
        ]
        
        if sagemaker_type not in valid_types:
            violations.append(NamingViolation(
                component=f"SageMaker Step Type for '{step_name}'",
                violation_type="invalid_sagemaker_type",
                message="Invalid SageMaker step type",
                expected=f"One of: {', '.join(valid_types)}",
                actual=sagemaker_type
            ))
        
        # Validate naming rule: Step class name minus "Step" suffix
        if sagemaker_type.endswith('Step'):
            violations.append(NamingViolation(
                component=f"SageMaker Step Type for '{step_name}'",
                violation_type="sagemaker_type_suffix",
                message="SageMaker step type should not end with 'Step'",
                expected=sagemaker_type[:-4],
                actual=sagemaker_type,
                suggestions=[sagemaker_type[:-4]]
            ))
        
        return violations
    
    def _suggest_file_name(self, filename: str, file_type: str) -> List[str]:
        """Suggest correct file names based on type."""
        base_name = filename.replace('.py', '')
        
        suggestions = []
        if file_type == 'builder':
            if not base_name.startswith('builder_'):
                suggestions.append(f"builder_{self._to_snake_case(base_name)}_step.py")
            elif not base_name.endswith('_step'):
                suggestions.append(f"{base_name}_step.py")
        elif file_type == 'config':
            if not base_name.startswith('config_'):
                suggestions.append(f"config_{self._to_snake_case(base_name)}_step.py")
            elif not base_name.endswith('_step'):
                suggestions.append(f"{base_name}_step.py")
        elif file_type == 'spec':
            if not base_name.endswith('_spec'):
                suggestions.append(f"{self._to_snake_case(base_name)}_spec.py")
        elif file_type == 'contract':
            if not base_name.endswith('_contract'):
                suggestions.append(f"{self._to_snake_case(base_name)}_contract.py")
        
        return suggestions
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        # Handle snake_case to PascalCase
        if '_' in name:
            parts = name.split('_')
            return ''.join(word.capitalize() for word in parts)
        
        # Handle camelCase to PascalCase
        if name and name[0].islower():
            return name[0].upper() + name[1:]
        
        return name
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Handle PascalCase/camelCase to snake_case
        result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', result)
        return result.lower()
    
    def get_all_violations(self) -> List[NamingViolation]:
        """Get all accumulated violations."""
        return self.violations
    
    def clear_violations(self) -> None:
        """Clear all accumulated violations."""
        self.violations.clear()
    
    def validate_all_registry_entries(self) -> List[NamingViolation]:
        """Validate all entries in the STEP_NAMES registry."""
        violations = []
        
        for step_name, registry_info in STEP_NAMES.items():
            violations.extend(self.validate_registry_entry(step_name, registry_info))
        
        return violations
