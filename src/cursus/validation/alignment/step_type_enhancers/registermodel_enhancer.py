"""
RegisterModel Step Enhancer

RegisterModel step-specific validation enhancement.
Provides validation for model registration patterns, metadata handling, and approval workflows.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer


class RegisterModelStepEnhancer(BaseStepEnhancer):
    """
    RegisterModel step-specific validation enhancement.
    
    Provides validation for:
    - Model metadata validation
    - Approval workflow validation
    - Model package creation validation
    - Registration builder validation
    """
    
    def __init__(self):
        """Initialize the RegisterModel step enhancer."""
        super().__init__("RegisterModel")
        self.reference_examples = [
            "builder_register_model_step.py"
        ]
        self.framework_validators = {}
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Add RegisterModel-specific validation.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with RegisterModel-specific issues
        """
        additional_issues = []
        
        # Get script analysis
        script_analysis = self._get_script_analysis(script_name)
        
        # Level 1: Model metadata validation
        additional_issues.extend(self._validate_model_metadata_patterns(script_analysis, script_name))
        
        # Level 2: Approval workflow validation
        additional_issues.extend(self._validate_approval_workflow_patterns(script_analysis, script_name))
        
        # Level 3: Model package creation validation
        additional_issues.extend(self._validate_model_package_creation(script_name))
        
        # Level 4: Registration builder validation
        additional_issues.extend(self._validate_registration_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_model_metadata_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model metadata handling patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of model metadata validation issues
        """
        issues = []
        
        # Check for model metadata patterns
        if not self._has_model_metadata_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_metadata",
                "RegisterModel should handle model metadata",
                "Add model metadata specification (description, tags, etc.)",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for model metrics patterns
        if not self._has_model_metrics_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_metrics",
                "RegisterModel should include model metrics",
                "Add model performance metrics for registration",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_approval_workflow_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate approval workflow patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of approval workflow validation issues
        """
        issues = []
        
        # Check for approval status patterns
        if not self._has_approval_status_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_approval_status",
                "RegisterModel should handle approval status",
                "Add model approval status configuration",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_model_package_creation(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model package creation.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of model package creation validation issues
        """
        issues = []
        
        # Check if registration specification exists
        spec_path = self._get_registration_spec_path(script_name)
        if not spec_path or not Path(spec_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_registration_specification",
                f"Registration specification not found for {script_name}",
                f"Create registration specification file for {script_name}",
                "INFO",
                {"script": script_name, "expected_spec_path": spec_path}
            ))
        
        return issues
    
    def _validate_registration_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate registration builder patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of registration builder validation issues
        """
        issues = []
        
        # Check if registration builder exists
        builder_path = self._get_registration_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_registration_builder",
                f"Registration builder not found for {script_name}",
                f"Create registration builder file for {script_name}",
                "WARNING",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_model_package_creation_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_model_package_creation_method",
                    "Registration builder should create model package",
                    "Add _create_model_package method to registration builder",
                    "ERROR",
                    {"script": script_name, "builder_path": builder_path}
                ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_model_metadata_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model metadata patterns."""
        metadata_keywords = ['metadata', 'description', 'tags', 'model_name']
        return self._has_pattern_in_analysis(script_analysis, 'functions', metadata_keywords)
    
    def _has_model_metrics_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model metrics patterns."""
        metrics_keywords = ['metrics', 'accuracy', 'performance', 'evaluation']
        return self._has_pattern_in_analysis(script_analysis, 'functions', metrics_keywords)
    
    def _has_approval_status_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has approval status patterns."""
        approval_keywords = ['approval', 'status', 'approved', 'pending']
        return self._has_pattern_in_analysis(script_analysis, 'functions', approval_keywords)
    
    def _has_model_package_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has model package creation patterns."""
        package_keywords = ['_create_model_package', 'ModelPackage', 'register']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', package_keywords)
    
    # Helper methods for file path resolution
    
    def _get_registration_spec_path(self, script_name: str) -> Optional[str]:
        """Get expected registration specification path."""
        base_name = script_name.replace('.py', '').replace('_register', '').replace('_registration', '')
        return f"cursus/steps/specs/{base_name}_registration_spec.py"
    
    def _get_registration_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected registration builder path."""
        base_name = script_name.replace('.py', '').replace('_register', '').replace('_registration', '')
        return f"cursus/steps/builders/builder_{base_name}_registration_step.py"
