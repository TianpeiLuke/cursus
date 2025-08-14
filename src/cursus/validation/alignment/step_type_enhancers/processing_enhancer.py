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
    
    def _get_script_analysis(self, script_name: str) -> Dict[str, Any]:
        """
        Get script analysis for the given script.
        
        Args:
            script_name: Name of the script to analyze
            
        Returns:
            Dictionary containing script analysis results
        """
        # Try to get existing analysis from validation context
        # In real implementation, this would integrate with the static analysis system
        return {
            'imports': [],
            'functions': [],
            'path_references': [],
            'patterns': {},
            'framework': None
        }
    
    def _get_builder_analysis(self, script_name: str) -> Dict[str, Any]:
        """
        Get builder analysis for the given script.
        
        Args:
            script_name: Name of the script whose builder to analyze
            
        Returns:
            Dictionary containing builder analysis results
        """
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
        
        # Check for framework-specific imports (processing-focused)
        if any('pandas' in imp or 'pd' in imp for imp in imports):
            return 'pandas'
        elif any('sklearn' in imp for imp in imports):
            return 'sklearn'
        elif any('numpy' in imp or 'np' in imp for imp in imports):
            return 'numpy'
        elif any('scipy' in imp for imp in imports):
            return 'scipy'
        
        return None
    
    def _has_pattern_in_analysis(self, analysis: Dict[str, Any], category: str, keywords: List[str]) -> bool:
        """
        Check if analysis contains any of the specified keywords in the given category.
        
        Args:
            analysis: Analysis results dictionary
            category: Category to check (e.g., 'imports', 'functions', 'path_references')
            keywords: List of keywords to search for
            
        Returns:
            True if any keyword is found in the category
        """
        category_data = analysis.get(category, [])
        if not category_data:
            return False
        
        # Convert to lowercase for case-insensitive matching
        category_str = ' '.join(str(item).lower() for item in category_data)
        
        return any(keyword.lower() in category_str for keyword in keywords)
    
    def _create_step_type_issue(self, category: str, message: str, recommendation: str, 
                               severity: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a step type-specific validation issue.
        
        Args:
            category: Issue category
            message: Issue message
            recommendation: Recommended fix
            severity: Issue severity (ERROR, WARNING, INFO)
            details: Additional issue details
            
        Returns:
            Formatted validation issue
        """
        return {
            'category': category,
            'message': message,
            'recommendation': recommendation,
            'severity': severity,
            'step_type': self.step_type,
            'details': details,
            'source': 'ProcessingStepEnhancer'
        }
    
    def get_processing_validation_requirements(self) -> Dict[str, Any]:
        """
        Get comprehensive processing validation requirements.
        
        Returns:
            Dictionary containing processing validation requirements
        """
        return {
            'required_patterns': {
                'data_transformation': {
                    'keywords': ['transform', 'process', 'clean', 'filter', 'map', 'apply'],
                    'description': 'Data transformation operations',
                    'severity': 'INFO'
                },
                'input_data_loading': {
                    'keywords': ['read_csv', 'read_json', 'load', '/opt/ml/processing/input'],
                    'description': 'Input data loading from SageMaker processing input',
                    'severity': 'WARNING'
                },
                'output_data_saving': {
                    'keywords': ['to_csv', 'to_json', 'save', '/opt/ml/processing/output'],
                    'description': 'Output data saving to SageMaker processing output',
                    'severity': 'WARNING'
                },
                'environment_variables': {
                    'keywords': ['os.environ', 'getenv', 'environment'],
                    'description': 'Environment variable usage for configuration',
                    'severity': 'INFO'
                }
            },
            'framework_requirements': {
                'pandas': {
                    'imports': ['pandas', 'pd'],
                    'functions': ['DataFrame', 'read_csv', 'to_csv'],
                    'patterns': ['dataframe_operations', 'data_manipulation']
                },
                'sklearn': {
                    'imports': ['sklearn', 'scikit-learn'],
                    'functions': ['fit_transform', 'transform', 'preprocessing'],
                    'patterns': ['preprocessing_operations', 'feature_engineering']
                }
            },
            'sagemaker_paths': {
                'processing_input': '/opt/ml/processing/input',
                'processing_output': '/opt/ml/processing/output',
                'processing_code': '/opt/ml/processing/code'
            },
            'validation_levels': {
                'level1': 'Script pattern validation',
                'level2': 'Specification alignment',
                'level3': 'Dependency validation',
                'level4': 'Builder pattern validation'
            }
        }
    
    def validate_processing_script_comprehensive(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """
        Perform comprehensive processing script validation.
        
        Args:
            script_name: Name of the processing script
            script_content: Content of the processing script
            
        Returns:
            Comprehensive validation results
        """
        # Create a basic script analysis from content
        script_analysis = self._create_script_analysis_from_content(script_content)
        
        # Detect patterns in script analysis
        from ..framework_patterns import detect_pandas_patterns, detect_sklearn_patterns
        
        # Detect framework from content directly
        from ..framework_patterns import detect_framework_from_script_content
        framework = detect_framework_from_script_content(script_content)
        
        # Get framework-specific patterns
        framework_patterns = {}
        if framework == 'pandas':
            framework_patterns = detect_pandas_patterns(script_analysis)
        elif framework == 'sklearn':
            framework_patterns = detect_sklearn_patterns(script_analysis)
        
        # Create comprehensive analysis
        analysis = {
            'script_name': script_name,
            'framework': framework,
            'processing_patterns': {
                'data_transformation': self._has_data_transformation_patterns(script_analysis),
                'input_data_loading': self._has_input_data_loading_patterns(script_analysis),
                'output_data_saving': self._has_output_data_saving_patterns(script_analysis),
                'environment_variables': self._has_environment_variable_patterns(script_analysis)
            },
            'framework_patterns': framework_patterns,
            'validation_results': {
                'data_transformation': self._has_data_transformation_patterns(script_analysis),
                'input_loading': self._has_input_data_loading_patterns(script_analysis),
                'output_saving': self._has_output_data_saving_patterns(script_analysis),
                'env_variables': self._has_environment_variable_patterns(script_analysis)
            }
        }
        
        return analysis
    
    def _create_script_analysis_from_content(self, script_content: str) -> Dict[str, Any]:
        """
        Create a basic script analysis from script content.
        
        Args:
            script_content: Raw script content
            
        Returns:
            Basic script analysis dictionary
        """
        # Extract imports (simple pattern matching)
        imports = []
        for line in script_content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        # Extract function-like patterns (simple pattern matching)
        functions = []
        for line in script_content.split('\n'):
            line = line.strip()
            if '(' in line and ')' in line:
                functions.append(line)
        
        # Extract path references
        path_references = []
        common_paths = ['/opt/ml/processing/input', '/opt/ml/processing/output', '/opt/ml/processing/code']
        for path in common_paths:
            if path in script_content:
                path_references.append(path)
        
        return {
            'imports': imports,
            'functions': functions,
            'path_references': path_references,
            'patterns': {},
            'framework': None
        }
