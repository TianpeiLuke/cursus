"""
Utility Step Enhancer

Utility step-specific validation enhancement.
Provides validation for utility scripts that handle file preparation, parameter generation, and special cases.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer


class UtilityStepEnhancer(BaseStepEnhancer):
    """
    Utility step-specific validation enhancement.
    
    Provides validation for:
    - File preparation validation
    - Parameter generation validation
    - Special case handling validation
    - Utility builder validation
    """
    
    def __init__(self):
        """Initialize the Utility step enhancer."""
        super().__init__("Utility")
        self.reference_examples = [
            "prepare_hyperparameters.py",
            "prepare_config_files.py"
        ]
        self.framework_validators = {
            "boto3": self._validate_boto3_utility,
            "json": self._validate_json_utility
        }
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Add Utility-specific validation.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with Utility-specific issues
        """
        additional_issues = []
        
        # Get script analysis
        script_analysis = self._get_script_analysis(script_name)
        framework = self._detect_framework_from_script_analysis(script_analysis)
        
        # Level 1: File preparation validation
        additional_issues.extend(self._validate_file_preparation_patterns(script_analysis, script_name))
        
        # Level 2: Parameter generation validation
        additional_issues.extend(self._validate_parameter_generation_patterns(script_analysis, script_name))
        
        # Level 3: Special case handling validation
        additional_issues.extend(self._validate_special_case_handling(script_name))
        
        # Level 4: Utility builder validation
        additional_issues.extend(self._validate_utility_builder(script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(script_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_file_preparation_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate file preparation patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of file preparation validation issues
        """
        issues = []
        
        # Check for file preparation patterns
        if not self._has_file_preparation_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_file_preparation",
                "Utility script should implement file preparation logic",
                "Add file creation, copying, or preparation operations",
                "INFO",
                {"script": script_name}
            ))
        
        # Check for file I/O patterns
        if not self._has_file_io_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_file_io",
                "Utility script should handle file I/O operations",
                "Add file reading, writing, or manipulation operations",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_parameter_generation_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate parameter generation patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of parameter generation validation issues
        """
        issues = []
        
        # Check for parameter generation patterns
        if not self._has_parameter_generation_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_parameter_generation",
                "Utility script should generate parameters or configuration",
                "Add parameter generation, configuration creation, or data preparation",
                "INFO",
                {"script": script_name}
            ))
        
        # Check for JSON/YAML handling
        if not self._has_config_file_handling_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_config_file_handling",
                "Utility script should handle configuration files",
                "Add JSON, YAML, or other configuration file handling",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_special_case_handling(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate special case handling.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of special case handling validation issues
        """
        issues = []
        
        # Check if utility specification exists
        spec_path = self._get_utility_spec_path(script_name)
        if not spec_path or not Path(spec_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_utility_specification",
                f"Utility specification not found for {script_name}",
                f"Create utility specification file for {script_name}",
                "INFO",
                {"script": script_name, "expected_spec_path": spec_path}
            ))
        
        return issues
    
    def _validate_utility_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate utility builder patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of utility builder validation issues
        """
        issues = []
        
        # Check if utility builder exists
        builder_path = self._get_utility_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_utility_builder",
                f"Utility builder not found for {script_name}",
                f"Create utility builder file for {script_name}",
                "INFO",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_file_preparation_builder_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_file_preparation_method",
                    "Utility builder should prepare files",
                    "Add _prepare_files method to utility builder",
                    "WARNING",
                    {"script": script_name, "builder_path": builder_path}
                ))
        
        return issues
    
    def _validate_boto3_utility(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Boto3-specific utility validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of boto3-specific validation issues
        """
        issues = []
        
        # Check for boto3 imports
        if not self._has_pattern_in_analysis(script_analysis, 'imports', ['boto3']):
            issues.append(self._create_step_type_issue(
                "missing_boto3_import",
                "Boto3 utility script should import boto3",
                "Add 'import boto3' to script",
                "INFO",
                {"script": script_name, "framework": "boto3"}
            ))
        
        # Check for AWS service usage
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['client', 'resource', 's3', 'sagemaker']):
            issues.append(self._create_step_type_issue(
                "missing_aws_service_usage",
                "Boto3 utility script should use AWS services",
                "Add AWS service client or resource usage",
                "INFO",
                {"script": script_name, "framework": "boto3"}
            ))
        
        return issues
    
    def _validate_json_utility(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        JSON-specific utility validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of JSON-specific validation issues
        """
        issues = []
        
        # Check for JSON operations
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['json.load', 'json.dump', 'json.loads', 'json.dumps']):
            issues.append(self._create_step_type_issue(
                "missing_json_operations",
                "JSON utility script should use JSON operations",
                "Add JSON loading, dumping, or manipulation operations",
                "INFO",
                {"script": script_name, "framework": "json"}
            ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_file_preparation_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has file preparation patterns."""
        prep_keywords = ['prepare', 'create', 'copy', 'move', 'generate']
        return self._has_pattern_in_analysis(script_analysis, 'functions', prep_keywords)
    
    def _has_file_io_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has file I/O patterns."""
        io_keywords = ['open', 'read', 'write', 'with open', 'file']
        return self._has_pattern_in_analysis(script_analysis, 'functions', io_keywords)
    
    def _has_parameter_generation_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has parameter generation patterns."""
        param_keywords = ['parameters', 'config', 'generate', 'create', 'prepare']
        return self._has_pattern_in_analysis(script_analysis, 'functions', param_keywords)
    
    def _has_config_file_handling_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has configuration file handling patterns."""
        config_keywords = ['json', 'yaml', 'config', 'load', 'dump']
        return self._has_pattern_in_analysis(script_analysis, 'functions', config_keywords)
    
    def _has_file_preparation_builder_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has file preparation patterns."""
        prep_keywords = ['_prepare_files', 'prepare', 'create_files']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', prep_keywords)
    
    # Helper methods for file path resolution
    
    def _get_utility_spec_path(self, script_name: str) -> Optional[str]:
        """Get expected utility specification path."""
        base_name = script_name.replace('.py', '').replace('_utility', '').replace('_util', '')
        return f"cursus/steps/specs/{base_name}_utility_spec.py"
    
    def _get_utility_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected utility builder path."""
        base_name = script_name.replace('.py', '').replace('_utility', '').replace('_util', '')
        return f"cursus/steps/builders/builder_{base_name}_utility_step.py"
