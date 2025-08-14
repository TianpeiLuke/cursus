"""
Processing Step Enhancer

Processing step-specific validation enhancement.
Migrates existing processing validation to step type-aware system while maintaining
100% backward compatibility and success rate.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer


class ProcessingStepEnhancer(BaseStepEnhancer):
    """
    Processing step-specific validation enhancement.
    
    Migrates existing processing validation to step type-aware system:
    - Processing script patterns (data transformation, environment variables)
    - Processing specifications alignment
    - Processing dependencies validation
    - Processing builder patterns
    - Framework awareness for processing scripts
    """
    
    def __init__(self):
        """Initialize the processing step enhancer."""
        super().__init__("Processing")
        self.reference_examples = [
            "tabular_preprocessing.py",
            "risk_table_mapping.py",
            "builder_tabular_preprocessing_step.py"
        ]
        self.framework_validators = {
            "pandas": self._validate_pandas_processing,
            "sklearn": self._validate_sklearn_processing
        }
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Migrate existing processing validation to step type-aware system.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with processing-specific context
        """
        additional_issues = []
        
        # Get script analysis from existing validation
        script_analysis = self._get_script_analysis(script_name)
        framework = self._detect_framework_from_script_analysis(script_analysis)
        
        # Level 1: Processing script patterns (existing logic enhanced)
        additional_issues.extend(self._validate_processing_script_patterns(script_analysis, framework, script_name))
        
        # Level 2: Processing specifications (existing logic enhanced)
        additional_issues.extend(self._validate_processing_specifications(script_name))
        
        # Level 3: Processing dependencies (existing logic enhanced)
        additional_issues.extend(self._validate_processing_dependencies(script_name, framework))
        
        # Level 4: Processing builder patterns (existing logic enhanced)
        additional_issues.extend(self._validate_processing_builder(script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(script_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_processing_script_patterns(self, script_analysis: Dict[str, Any], framework: Optional[str], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate processing-specific script patterns.
        
        Args:
            script_analysis: Script analysis results
            framework: Detected framework
            script_name: Name of the script
            
        Returns:
            List of processing pattern validation issues
        """
        issues = []
        
        # Check for data transformation patterns
        if not self._has_data_transformation_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_data_transformation",
                "Processing script should contain data transformation logic",
                "Add data transformation operations (e.g., pandas operations, sklearn transforms)",
                "INFO",
                {"script": script_name, "framework": framework}
            ))
        
        # Check for input data loading patterns
        if not self._has_input_data_loading_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_input_data_loading",
                "Processing script should load input data",
                "Add input data loading from /opt/ml/processing/input/",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/processing/input/"}
            ))
        
        # Check for output data saving patterns
        if not self._has_output_data_saving_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_output_data_saving",
                "Processing script should save processed data",
                "Add output data saving to /opt/ml/processing/output/",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/processing/output/"}
            ))
        
        # Check for environment variable usage
        if not self._has_environment_variable_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_environment_variables",
                "Processing script should use environment variables for configuration",
                "Add environment variable access (e.g., os.environ.get())",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_processing_specifications(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate processing specifications alignment.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of processing specification validation issues
        """
        issues = []
        
        # Check if processing specification exists
        spec_path = self._get_processing_spec_path(script_name)
        if not spec_path or not Path(spec_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_processing_specification",
                f"Processing specification not found for {script_name}",
                f"Create processing specification file for {script_name}",
                "INFO",
                {"script": script_name, "expected_spec_path": spec_path}
            ))
        
        return issues
    
    def _validate_processing_dependencies(self, script_name: str, framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Validate processing dependencies.
        
        Args:
            script_name: Name of the script
            framework: Detected framework
            
        Returns:
            List of processing dependency validation issues
        """
        issues = []
        
        # Check for framework-specific dependencies
        if framework:
            expected_dependencies = self._get_expected_framework_dependencies(framework)
            for dependency in expected_dependencies:
                if not self._has_dependency(script_name, dependency):
                    issues.append(self._create_step_type_issue(
                        "missing_framework_dependency",
                        f"Processing script should declare {framework} dependency: {dependency}",
                        f"Add {dependency} to requirements or imports",
                        "INFO",
                        {"script": script_name, "framework": framework, "dependency": dependency}
                    ))
        
        return issues
    
    def _validate_processing_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate processing builder patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of processing builder validation issues
        """
        issues = []
        
        # Check if processing builder exists
        builder_path = self._get_processing_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_processing_builder",
                f"Processing builder not found for {script_name}",
                f"Create processing builder file for {script_name}",
                "INFO",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_processor_creation_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_processor_creation",
                    "Processing builder should create processor",
                    "Add _create_processor method to processing builder",
                    "WARNING",
                    {"script": script_name, "builder_path": builder_path}
                ))
        
        return issues
    
    def _validate_pandas_processing(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Pandas-specific processing validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of pandas-specific validation issues
        """
        issues = []
        
        # Check for pandas imports
        if not self._has_pattern_in_analysis(script_analysis, 'imports', ['pandas', 'pd']):
            issues.append(self._create_step_type_issue(
                "missing_pandas_import",
                "Pandas processing script should import pandas",
                "Add 'import pandas as pd' to script",
                "INFO",
                {"script": script_name, "framework": "pandas"}
            ))
        
        # Check for DataFrame operations
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['DataFrame', 'pd.read', 'to_csv']):
            issues.append(self._create_step_type_issue(
                "missing_dataframe_operations",
                "Pandas processing script should use DataFrame operations",
                "Add DataFrame creation and manipulation operations",
                "INFO",
                {"script": script_name, "framework": "pandas"}
            ))
        
        return issues
    
    def _validate_sklearn_processing(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Scikit-learn-specific processing validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of sklearn-specific validation issues
        """
        issues = []
        
        # Check for sklearn imports
        if not self._has_pattern_in_analysis(script_analysis, 'imports', ['sklearn', 'scikit-learn']):
            issues.append(self._create_step_type_issue(
                "missing_sklearn_import",
                "Scikit-learn processing script should import sklearn",
                "Add sklearn imports (e.g., from sklearn.preprocessing import StandardScaler)",
                "INFO",
                {"script": script_name, "framework": "sklearn"}
            ))
        
        # Check for preprocessing operations
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['fit_transform', 'transform', 'preprocessing']):
            issues.append(self._create_step_type_issue(
                "missing_preprocessing_operations",
                "Scikit-learn processing script should use preprocessing operations",
                "Add sklearn preprocessing operations (e.g., StandardScaler, LabelEncoder)",
                "INFO",
                {"script": script_name, "framework": "sklearn"}
            ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_data_transformation_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has data transformation patterns."""
        transform_keywords = ['transform', 'process', 'clean', 'filter', 'map', 'apply', 'groupby']
        return self._has_pattern_in_analysis(script_analysis, 'functions', transform_keywords)
    
    def _has_input_data_loading_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has input data loading patterns."""
        input_keywords = ['read_csv', 'read_json', 'load', '/opt/ml/processing/input']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', input_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/processing/input']))
    
    def _has_output_data_saving_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has output data saving patterns."""
        output_keywords = ['to_csv', 'to_json', 'save', 'dump', '/opt/ml/processing/output']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', output_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/processing/output']))
    
    def _has_environment_variable_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has environment variable patterns."""
        env_keywords = ['os.environ', 'getenv', 'environment']
        return self._has_pattern_in_analysis(script_analysis, 'functions', env_keywords)
    
    def _has_processor_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has processor creation patterns."""
        processor_keywords = ['_create_processor', 'Processor', 'SKLearnProcessor', 'ScriptProcessor']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', processor_keywords)
    
    # Helper methods for file path resolution
    
    def _get_processing_spec_path(self, script_name: str) -> Optional[str]:
        """Get expected processing specification path."""
        base_name = script_name.replace('.py', '').replace('_preprocessing', '').replace('_processing', '')
        return f"cursus/steps/specs/{base_name}_processing_spec.py"
    
    def _get_processing_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected processing builder path."""
        base_name = script_name.replace('.py', '').replace('_preprocessing', '').replace('_processing', '')
        return f"cursus/steps/builders/builder_{base_name}_step.py"
    
    def _get_expected_framework_dependencies(self, framework: str) -> List[str]:
        """Get expected dependencies for framework."""
        dependencies = {
            "pandas": ["pandas", "numpy"],
            "sklearn": ["scikit-learn", "pandas", "numpy"],
            "numpy": ["numpy"],
            "scipy": ["scipy", "numpy"]
        }
        return dependencies.get(framework, [])
    
    def _has_dependency(self, script_name: str, dependency: str) -> bool:
        """Check if script has specific dependency."""
        # This is a placeholder - in real implementation, this would check
        # requirements.txt, imports, or other dependency declarations
        return True  # Assume dependency exists for now
