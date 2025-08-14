"""
Base Step Enhancer

Abstract base class for all step type enhancers.
Provides common functionality and interface for step type-specific validation enhancement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..alignment_utils import detect_framework_from_imports


class BaseStepEnhancer(ABC):
    """
    Abstract base class for all step type enhancers.
    
    Provides common functionality for:
    - Step type identification
    - Reference example management
    - Framework validator coordination
    - Result merging and issue creation
    """
    
    def __init__(self, step_type: str):
        """
        Initialize the base enhancer.
        
        Args:
            step_type: The SageMaker step type this enhancer handles
        """
        self.step_type = step_type
        self.reference_examples = []
        self.framework_validators = {}
    
    @abstractmethod
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Enhance existing validation with step type-specific checks.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with additional step type-specific issues
        """
        pass
    
    def _merge_results(self, existing_results, additional_issues):
        """
        Merge additional issues with existing validation results.
        
        Args:
            existing_results: Original validation results (dict or ValidationResult)
            additional_issues: Additional issues to merge
            
        Returns:
            Merged validation results
        """
        # Handle None input
        if existing_results is None:
            if not additional_issues:
                return None
            # Create a basic result structure
            return {
                'issues': additional_issues,
                'success': len(additional_issues) == 0,
                'summary': {'total_issues': len(additional_issues)}
            }
        
        # Handle ValidationResult objects
        if hasattr(existing_results, 'issues'):
            # This is a ValidationResult object
            existing_results.issues.extend(additional_issues)
            return existing_results
        
        # Handle dictionary results
        if isinstance(existing_results, dict):
            existing_results.setdefault('issues', []).extend(additional_issues)
            
            # Update summary statistics if present and summary is a dict
            if 'summary' in existing_results and isinstance(existing_results['summary'], dict):
                summary = existing_results['summary']
                summary['total_issues'] = summary.get('total_issues', 0) + len(additional_issues)
                
                # Update severity counts
                for issue in additional_issues:
                    if hasattr(issue, 'level'):
                        severity = str(issue.level).lower()
                    else:
                        severity = issue.get('severity', 'INFO').lower()
                    severity_key = f'{severity}_count'
                    summary[severity_key] = summary.get(severity_key, 0) + 1
            
            return existing_results
        
        # Handle other types - try to convert to dict
        try:
            result_dict = {'issues': additional_issues}
            if hasattr(existing_results, '__dict__'):
                result_dict.update(existing_results.__dict__)
            return result_dict
        except:
            # Last resort - return a basic structure
            return {
                'issues': additional_issues,
                'original_result': existing_results,
                'success': len(additional_issues) == 0
            }
    
    def _create_step_type_issue(self, category: str, message: str, recommendation: str, 
                               severity: str = 'WARNING', details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a step type-specific validation issue.
        
        Args:
            category: Issue category
            message: Issue description
            recommendation: Recommended action
            severity: Issue severity level
            details: Additional issue details
            
        Returns:
            Formatted validation issue
        """
        issue = {
            'severity': severity,
            'category': category,
            'message': message,
            'recommendation': recommendation,
            'step_type': self.step_type
        }
        
        if details:
            issue['details'] = details
        
        return issue
    
    def _get_script_analysis(self, script_name: str) -> Dict[str, Any]:
        """
        Get script analysis results for the specified script.
        
        This is a placeholder that should be overridden by concrete enhancers
        to integrate with their specific analysis systems.
        
        Args:
            script_name: Name of the script to analyze
            
        Returns:
            Script analysis results
        """
        # Default implementation returns empty analysis
        # Concrete enhancers should override this to integrate with actual analysis
        return {
            'imports': [],
            'functions': [],
            'path_references': [],
            'env_var_accesses': [],
            'argument_definitions': [],
            'file_operations': []
        }
    
    def _get_builder_analysis(self, script_name: str) -> Dict[str, Any]:
        """
        Get builder analysis results for the specified script.
        
        Args:
            script_name: Name of the script to analyze
            
        Returns:
            Builder analysis results
        """
        # Default implementation returns empty analysis
        # Concrete enhancers should override this to integrate with actual analysis
        return {
            'builder_methods': [],
            'step_creation_patterns': [],
            'configuration_patterns': []
        }
    
    def _detect_framework_from_script_analysis(self, script_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Detect framework from script analysis results.
        
        Args:
            script_analysis: Script analysis results
            
        Returns:
            Detected framework name or None
        """
        imports = script_analysis.get('imports', [])
        return detect_framework_from_imports(imports)
    
    def _has_pattern_in_analysis(self, script_analysis: Dict[str, Any], pattern_type: str, pattern_keywords: List[str]) -> bool:
        """
        Check if specific patterns exist in script analysis.
        
        Args:
            script_analysis: Script analysis results
            pattern_type: Type of pattern to check (e.g., 'functions', 'imports')
            pattern_keywords: Keywords to look for
            
        Returns:
            True if pattern is found, False otherwise
        """
        analysis_data = script_analysis.get(pattern_type, [])
        
        if isinstance(analysis_data, list):
            # Check if any item in the list contains the pattern keywords
            for item in analysis_data:
                item_str = str(item).lower()
                if any(keyword.lower() in item_str for keyword in pattern_keywords):
                    return True
        elif isinstance(analysis_data, dict):
            # Check if any value in the dict contains the pattern keywords
            for value in analysis_data.values():
                value_str = str(value).lower()
                if any(keyword.lower() in value_str for keyword in pattern_keywords):
                    return True
        
        return False
    
    def _get_reference_examples(self) -> List[str]:
        """
        Get reference examples for this step type.
        
        Returns:
            List of reference example script names
        """
        return self.reference_examples.copy()
    
    def _validate_against_reference_examples(self, script_name: str, script_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate script against reference examples for this step type.
        
        Args:
            script_name: Name of the script being validated
            script_analysis: Script analysis results
            
        Returns:
            List of validation issues based on reference example comparison
        """
        issues = []
        
        if not self.reference_examples:
            issues.append(self._create_step_type_issue(
                'reference_examples',
                f'No reference examples available for {self.step_type} step type',
                f'Add reference examples for {self.step_type} step validation',
                'INFO'
            ))
        
        return issues
    
    def get_step_type_info(self) -> Dict[str, Any]:
        """
        Get information about this step type enhancer.
        
        Returns:
            Dictionary containing step type information
        """
        return {
            'step_type': self.step_type,
            'reference_examples': self.reference_examples,
            'supported_frameworks': list(self.framework_validators.keys()),
            'enhancer_class': self.__class__.__name__
        }
