"""
Interface Standard Validator for the Cursus pipeline framework.

This module provides validation for interface compliance as defined in the
standardization rules document. It validates that step builders implement
required interfaces correctly.
"""

import inspect
from typing import List, Dict, Any, Optional, Type, Union, get_type_hints
from pathlib import Path

from ...core.base.builder_base import StepBuilderBase
from ...core.base.config_base import BasePipelineConfig
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract


class InterfaceViolation:
    """Represents an interface compliance violation."""
    
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


class InterfaceStandardValidator:
    """
    Validator for interface compliance and standardization rules.
    
    This validator enforces the interface patterns defined in the standardization
    rules document, including:
    - Required method implementation
    - Method signature compliance
    - Inheritance compliance
    - Interface documentation standards
    """
    
    def __init__(self):
        """Initialize the interface validator."""
        self.violations = []
    
    def validate_step_builder_interface(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate complete interface compliance for a step builder class.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of interface violations found
        """
        violations = []
        
        if not builder_class:
            violations.append(InterfaceViolation(
                component="StepBuilder",
                violation_type="missing_class",
                message="Step builder class is None or missing"
            ))
            return violations
        
        # Validate inheritance
        violations.extend(self.validate_inheritance_compliance(builder_class))
        
        # Validate required methods
        violations.extend(self.validate_required_methods(builder_class))
        
        # Validate method signatures
        violations.extend(self.validate_method_signatures(builder_class))
        
        # Validate method documentation
        violations.extend(self.validate_method_documentation(builder_class))
        
        return violations
    
    def validate_inheritance_compliance(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate that the builder class properly inherits from StepBuilderBase.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of inheritance violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Check if class inherits from StepBuilderBase
        if not issubclass(builder_class, StepBuilderBase):
            violations.append(InterfaceViolation(
                component=f"Class '{class_name}'",
                violation_type="inheritance_missing",
                message="Step builder class must inherit from StepBuilderBase",
                expected="class YourStepBuilder(StepBuilderBase)",
                actual=f"class {class_name}(...)",
                suggestions=["Add StepBuilderBase as base class", "Import StepBuilderBase from cursus.core.base.builder_base"]
            ))
        
        # Check method resolution order for proper inheritance
        mro = builder_class.__mro__
        if StepBuilderBase not in mro:
            violations.append(InterfaceViolation(
                component=f"Class '{class_name}'",
                violation_type="inheritance_mro",
                message="StepBuilderBase not found in method resolution order",
                suggestions=["Check inheritance chain", "Ensure proper import of StepBuilderBase"]
            ))
        
        return violations
    
    def validate_required_methods(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate that all required methods are implemented.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of method implementation violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Define required methods based on standardization rules
        required_methods = {
            'validate_configuration': {
                'description': 'Validate the configuration',
                'required': True,
                'signature_hint': 'def validate_configuration(self) -> None'
            },
            '_get_inputs': {
                'description': 'Get inputs for the step',
                'required': True,
                'signature_hint': 'def _get_inputs(self, inputs: Dict[str, Any]) -> List[...]'
            },
            '_get_outputs': {
                'description': 'Get outputs for the step',
                'required': True,
                'signature_hint': 'def _get_outputs(self, outputs: Dict[str, Any]) -> List[...]'
            },
            'create_step': {
                'description': 'Create the SageMaker step',
                'required': True,
                'signature_hint': 'def create_step(self, **kwargs) -> SageMakerStep'
            }
        }
        
        # Check each required method
        for method_name, method_info in required_methods.items():
            if not hasattr(builder_class, method_name):
                violations.append(InterfaceViolation(
                    component=f"Method '{method_name}' in {class_name}",
                    violation_type="method_missing",
                    message=f"Required method '{method_name}' is not implemented",
                    expected=method_info['signature_hint'],
                    suggestions=[
                        f"Implement {method_name} method",
                        f"Method should {method_info['description'].lower()}"
                    ]
                ))
                continue
            
            # Check if method is callable
            method = getattr(builder_class, method_name)
            if not callable(method):
                violations.append(InterfaceViolation(
                    component=f"Method '{method_name}' in {class_name}",
                    violation_type="method_not_callable",
                    message=f"'{method_name}' exists but is not callable",
                    suggestions=["Ensure method is defined as a function, not a property or attribute"]
                ))
        
        return violations
    
    def validate_method_signatures(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate method signatures match expected patterns.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of method signature violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Define expected method signatures
        expected_signatures = {
            'validate_configuration': {
                'params': ['self'],
                'return_type': None,  # Should return None or not specified
            },
            '_get_inputs': {
                'params': ['self', 'inputs'],
                'return_type': 'List',  # Should return List of inputs
            },
            '_get_outputs': {
                'params': ['self', 'outputs'],
                'return_type': 'List',  # Should return List of outputs
            },
            'create_step': {
                'params': ['self'],
                'kwargs': True,  # Should accept **kwargs
                'return_type': 'Step',  # Should return some kind of Step
            }
        }
        
        # Check each method signature
        for method_name, expected in expected_signatures.items():
            if not hasattr(builder_class, method_name):
                continue  # Skip if method doesn't exist (handled by validate_required_methods)
            
            method = getattr(builder_class, method_name)
            if not callable(method):
                continue  # Skip if not callable (handled by validate_required_methods)
            
            try:
                # Get method signature
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                
                # Validate required parameters
                expected_params = expected.get('params', [])
                for i, expected_param in enumerate(expected_params):
                    if i >= len(params):
                        violations.append(InterfaceViolation(
                            component=f"Method '{method_name}' in {class_name}",
                            violation_type="signature_missing_param",
                            message=f"Missing required parameter '{expected_param}'",
                            expected=f"Parameter {i+1}: {expected_param}",
                            actual=f"Only {len(params)} parameters found",
                            suggestions=[f"Add '{expected_param}' parameter to method signature"]
                        ))
                    elif params[i] != expected_param:
                        violations.append(InterfaceViolation(
                            component=f"Method '{method_name}' in {class_name}",
                            violation_type="signature_param_name",
                            message=f"Parameter {i+1} name mismatch",
                            expected=expected_param,
                            actual=params[i],
                            suggestions=[f"Rename parameter to '{expected_param}'"]
                        ))
                
                # Check for **kwargs if required
                if expected.get('kwargs', False):
                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if not has_kwargs:
                        violations.append(InterfaceViolation(
                            component=f"Method '{method_name}' in {class_name}",
                            violation_type="signature_missing_kwargs",
                            message="Method should accept **kwargs",
                            expected=f"def {method_name}(self, **kwargs)",
                            suggestions=["Add **kwargs to method signature"]
                        ))
                
                # Validate return type annotation if available
                return_annotation = sig.return_annotation
                expected_return = expected.get('return_type')
                if expected_return and return_annotation != inspect.Signature.empty:
                    # Basic return type validation (could be enhanced)
                    return_str = str(return_annotation)
                    if expected_return not in return_str:
                        violations.append(InterfaceViolation(
                            component=f"Method '{method_name}' in {class_name}",
                            violation_type="signature_return_type",
                            message="Return type annotation may not match expected pattern",
                            expected=f"Should return {expected_return}",
                            actual=return_str,
                            suggestions=[f"Consider using return type annotation that includes {expected_return}"]
                        ))
                
            except Exception as e:
                violations.append(InterfaceViolation(
                    component=f"Method '{method_name}' in {class_name}",
                    violation_type="signature_analysis_error",
                    message=f"Could not analyze method signature: {str(e)}",
                    suggestions=["Check method definition for syntax errors"]
                ))
        
        return violations
    
    def validate_method_documentation(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate that required methods have proper documentation.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of method documentation violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Methods that should have documentation
        documented_methods = [
            'validate_configuration',
            '_get_inputs',
            '_get_outputs',
            'create_step'
        ]
        
        for method_name in documented_methods:
            if not hasattr(builder_class, method_name):
                continue  # Skip if method doesn't exist
            
            method = getattr(builder_class, method_name)
            if not callable(method):
                continue  # Skip if not callable
            
            # Check for docstring
            docstring = inspect.getdoc(method)
            if not docstring or len(docstring.strip()) == 0:
                violations.append(InterfaceViolation(
                    component=f"Method '{method_name}' in {class_name}",
                    violation_type="documentation_missing",
                    message="Method lacks documentation",
                    suggestions=[
                        "Add docstring to method",
                        "Include description of method purpose",
                        "Document parameters and return values"
                    ]
                ))
                continue
            
            # Basic docstring quality checks
            docstring_lower = docstring.lower()
            
            # Check for parameter documentation if method has parameters
            try:
                sig = inspect.signature(method)
                params = [p for name, p in sig.parameters.items() if name != 'self']
                
                if params and 'args:' not in docstring_lower and 'parameters:' not in docstring_lower:
                    violations.append(InterfaceViolation(
                        component=f"Method '{method_name}' in {class_name}",
                        violation_type="documentation_missing_params",
                        message="Method has parameters but no parameter documentation",
                        suggestions=[
                            "Add 'Args:' or 'Parameters:' section to docstring",
                            "Document each parameter with type and description"
                        ]
                    ))
                
                # Check for return documentation if method has return annotation
                if (sig.return_annotation != inspect.Signature.empty and 
                    'returns:' not in docstring_lower and 'return:' not in docstring_lower):
                    violations.append(InterfaceViolation(
                        component=f"Method '{method_name}' in {class_name}",
                        violation_type="documentation_missing_return",
                        message="Method has return type but no return documentation",
                        suggestions=[
                            "Add 'Returns:' section to docstring",
                            "Document return value type and description"
                        ]
                    ))
                
            except Exception:
                # If signature analysis fails, skip parameter/return checks
                pass
        
        return violations
    
    def validate_class_documentation(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate that the class has proper documentation.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of class documentation violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Check for class docstring
        docstring = inspect.getdoc(builder_class)
        if not docstring or len(docstring.strip()) == 0:
            violations.append(InterfaceViolation(
                component=f"Class '{class_name}'",
                violation_type="class_documentation_missing",
                message="Class lacks documentation",
                suggestions=[
                    "Add class docstring",
                    "Include purpose description",
                    "Add key features and integration points",
                    "Include usage examples"
                ]
            ))
            return violations
        
        # Basic docstring quality checks
        docstring_lower = docstring.lower()
        
        # Check for key documentation elements
        required_elements = {
            'purpose': ['purpose:', 'description:', 'overview:'],
            'example': ['example:', 'usage:', 'examples:'],
        }
        
        for element_name, keywords in required_elements.items():
            if not any(keyword in docstring_lower for keyword in keywords):
                violations.append(InterfaceViolation(
                    component=f"Class '{class_name}'",
                    violation_type=f"class_documentation_missing_{element_name}",
                    message=f"Class documentation missing {element_name} section",
                    suggestions=[
                        f"Add {element_name} section to class docstring",
                        f"Use one of these keywords: {', '.join(keywords)}"
                    ]
                ))
        
        return violations
    
    def get_all_violations(self) -> List[InterfaceViolation]:
        """Get all accumulated violations."""
        return self.violations
    
    def clear_violations(self) -> None:
        """Clear all accumulated violations."""
        self.violations.clear()
    
    def validate_builder_registry_compliance(self, builder_class: Type[StepBuilderBase]) -> List[InterfaceViolation]:
        """
        Validate that the builder class is properly registered or registrable.
        
        Args:
            builder_class: Step builder class to validate
            
        Returns:
            List of registry compliance violations found
        """
        violations = []
        class_name = builder_class.__name__
        
        # Check naming convention for auto-discovery
        if not class_name.endswith('StepBuilder'):
            violations.append(InterfaceViolation(
                component=f"Class '{class_name}'",
                violation_type="registry_naming_convention",
                message="Builder class name should end with 'StepBuilder' for auto-discovery",
                expected=f"{class_name}StepBuilder",
                actual=class_name,
                suggestions=[
                    "Rename class to follow naming convention",
                    "Use @register_builder decorator if custom naming is required"
                ]
            ))
        
        # Check for registration decorator (optional but recommended)
        # This is informational, not a violation
        has_register_decorator = False
        if hasattr(builder_class, '__dict__'):
            # Check if class has been decorated (this is a simplified check)
            # In practice, checking for decorators is complex
            pass
        
        return violations
