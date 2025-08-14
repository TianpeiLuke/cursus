"""
CreateModel Step Enhancer

CreateModel step-specific validation enhancement.
Provides validation for model creation patterns, inference code, and container configuration.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer


class CreateModelStepEnhancer(BaseStepEnhancer):
    """
    CreateModel step-specific validation enhancement.
    
    Provides validation for:
    - Model artifact handling validation
    - Inference code validation
    - Container configuration validation
    - Model creation builder validation
    """
    
    def __init__(self):
        """Initialize the CreateModel step enhancer."""
        super().__init__("CreateModel")
        self.reference_examples = [
            "builder_xgboost_model_step.py",
            "builder_pytorch_model_step.py"
        ]
        self.framework_validators = {
            "xgboost": self._validate_xgboost_model_creation,
            "pytorch": self._validate_pytorch_model_creation
        }
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Add CreateModel-specific validation.
        
        CreateModel steps follow a different pattern than Processing/Training steps:
        - No standalone script validation (inference code is embedded in model artifacts)
        - Focus on builder configuration validation
        - Container image and deployment configuration validation
        - Model artifact structure validation
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated (usually builder name)
            
        Returns:
            Enhanced validation results with CreateModel-specific issues
        """
        additional_issues = []
        
        # CreateModel steps typically don't have scripts, so we focus on builder validation
        builder_analysis = self._get_builder_analysis(script_name)
        framework = self._detect_framework_from_builder_analysis(builder_analysis)
        
        # Level 1: Builder configuration validation (replaces script validation)
        additional_issues.extend(self._validate_builder_configuration(builder_analysis, script_name))
        
        # Level 2: Container and deployment configuration validation
        additional_issues.extend(self._validate_container_deployment_configuration(builder_analysis, script_name))
        
        # Level 3: Model artifact structure validation
        additional_issues.extend(self._validate_model_artifact_structure(script_name, framework))
        
        # Level 4: Model creation builder patterns validation
        additional_issues.extend(self._validate_model_creation_builder_patterns(builder_analysis, script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(builder_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_builder_configuration(self, builder_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate builder configuration patterns (Level 1 for CreateModel).
        
        Args:
            builder_analysis: Builder analysis results
            script_name: Name of the builder script
            
        Returns:
            List of builder configuration validation issues
        """
        issues = []
        
        # Check for model creation method
        if not self._has_model_creation_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_creation_method",
                "CreateModel builder should implement _create_model method",
                "Add _create_model method to builder class",
                "ERROR",
                {"script": script_name}
            ))
        
        # Check for model data configuration
        if not self._has_model_data_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_data_configuration",
                "CreateModel builder should configure model data source",
                "Add model_data parameter configuration in builder",
                "ERROR",
                {"script": script_name}
            ))
        
        # Check for role configuration
        if not self._has_role_configuration_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_role_configuration",
                "CreateModel builder should configure execution role",
                "Add role parameter configuration in builder",
                "WARNING",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_container_deployment_configuration(self, builder_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate container and deployment configuration (Level 2 for CreateModel).
        
        Args:
            builder_analysis: Builder analysis results
            script_name: Name of the builder script
            
        Returns:
            List of container deployment validation issues
        """
        issues = []
        
        # Check for container image specification
        if not self._has_container_image_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_container_image",
                "CreateModel builder should specify container image",
                "Add image_uri parameter in model configuration",
                "ERROR",
                {"script": script_name}
            ))
        
        # Check for instance type configuration
        if not self._has_instance_type_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_instance_type",
                "CreateModel builder should specify instance type for deployment",
                "Add instance_type parameter for model deployment",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for environment variables configuration
        if not self._has_environment_config_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_environment_config",
                "CreateModel builder should configure environment variables",
                "Add environment variables for model deployment",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_model_artifact_structure(self, script_name: str, framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Validate model artifact structure (Level 3 for CreateModel).
        
        Args:
            script_name: Name of the builder script
            framework: Detected framework
            
        Returns:
            List of model artifact structure validation issues
        """
        issues = []
        
        # Check for inference code presence (if not using framework containers)
        if not self._has_inference_code_files(script_name):
            issues.append(self._create_step_type_issue(
                "missing_inference_code",
                "CreateModel should include inference code files",
                "Add inference.py or model_fn implementation",
                "WARNING",
                {"script": script_name, "framework": framework}
            ))
        
        # Check for model artifact dependencies
        if not self._has_model_dependencies_file(script_name):
            issues.append(self._create_step_type_issue(
                "missing_model_dependencies",
                "CreateModel should include requirements.txt for model dependencies",
                "Add requirements.txt file for model dependencies",
                "INFO",
                {"script": script_name}
            ))
        
        # Framework-specific artifact validation
        if framework:
            framework_issues = self._validate_framework_specific_artifacts(script_name, framework)
            issues.extend(framework_issues)
        
        return issues
    
    def _validate_model_creation_builder_patterns(self, builder_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model creation builder patterns (Level 4 for CreateModel).
        
        Args:
            builder_analysis: Builder analysis results
            script_name: Name of the builder script
            
        Returns:
            List of model creation builder pattern validation issues
        """
        issues = []
        
        # Check for step creation patterns
        if not self._has_step_creation_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_step_creation_patterns",
                "CreateModel builder should implement step creation patterns",
                "Add proper step creation and configuration methods",
                "ERROR",
                {"script": script_name}
            ))
        
        # Check for model name generation
        if not self._has_model_name_generation_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_name_generation",
                "CreateModel builder should generate unique model names",
                "Add model name generation logic in builder",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for dependency handling
        if not self._has_dependency_handling_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_dependency_handling",
                "CreateModel builder should handle step dependencies",
                "Add dependency configuration for model creation step",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_xgboost_model_creation(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        XGBoost-specific model creation validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of XGBoost-specific validation issues
        """
        issues = []
        
        # Check for XGBoost model loading
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['xgb.Booster', 'load_model']):
            issues.append(self._create_step_type_issue(
                "missing_xgboost_model_loading",
                "XGBoost CreateModel should load XGBoost model",
                "Add XGBoost model loading (e.g., xgb.Booster(model_file=...))",
                "ERROR",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        return issues
    
    def _validate_pytorch_model_creation(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        PyTorch-specific model creation validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of PyTorch-specific validation issues
        """
        issues = []
        
        # Check for PyTorch model loading
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['torch.load', 'load_state_dict']):
            issues.append(self._create_step_type_issue(
                "missing_pytorch_model_loading",
                "PyTorch CreateModel should load PyTorch model",
                "Add PyTorch model loading (e.g., torch.load())",
                "ERROR",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_model_loading_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model loading patterns."""
        loading_keywords = ['load', 'pickle.load', 'joblib.load', 'torch.load', 'xgb.Booster']
        return self._has_pattern_in_analysis(script_analysis, 'functions', loading_keywords)
    
    def _has_model_path_references(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model path references."""
        return self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/model'])
    
    def _has_inference_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has inference patterns."""
        inference_keywords = ['model_fn', 'input_fn', 'predict_fn', 'output_fn', 'inference']
        return self._has_pattern_in_analysis(script_analysis, 'functions', inference_keywords)
    
    def _has_model_fn_pattern(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model_fn pattern."""
        return self._has_pattern_in_analysis(script_analysis, 'functions', ['model_fn'])
    
    def _has_predict_fn_pattern(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has predict_fn pattern."""
        return self._has_pattern_in_analysis(script_analysis, 'functions', ['predict_fn', 'predict'])
    
    def _has_container_image_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has container image patterns."""
        container_keywords = ['image_uri', 'container', 'image']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', container_keywords)
    
    def _has_model_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has model creation patterns."""
        model_keywords = ['_create_model', 'Model', 'create_model']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', model_keywords)
    
    # New helper methods for CreateModel-specific patterns
    
    def _has_model_data_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has model data configuration patterns."""
        model_data_keywords = ['model_data', 'ModelDataUrl', 'model_artifacts']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', model_data_keywords)
    
    def _has_role_configuration_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has role configuration patterns."""
        role_keywords = ['role', 'execution_role', 'ExecutionRoleArn']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', role_keywords)
    
    def _has_instance_type_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has instance type configuration patterns."""
        instance_keywords = ['instance_type', 'InstanceType', 'ml.']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', instance_keywords)
    
    def _has_environment_config_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has environment configuration patterns."""
        env_keywords = ['environment', 'Environment', 'env_vars']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', env_keywords)
    
    def _has_inference_code_files(self, script_name: str) -> bool:
        """Check if inference code files exist for the model."""
        # This would check for inference.py, model_fn implementations, etc.
        # For now, return True as a placeholder
        return True
    
    def _has_model_dependencies_file(self, script_name: str) -> bool:
        """Check if model dependencies file (requirements.txt) exists."""
        # This would check for requirements.txt in model artifacts
        # For now, return True as a placeholder
        return True
    
    def _validate_framework_specific_artifacts(self, script_name: str, framework: str) -> List[Dict[str, Any]]:
        """Validate framework-specific model artifacts."""
        issues = []
        
        if framework == 'xgboost':
            # Check for XGBoost model file
            issues.append(self._create_step_type_issue(
                "xgboost_model_artifact_check",
                "XGBoost CreateModel should include .model or .json model file",
                "Ensure XGBoost model is saved in proper format",
                "INFO",
                {"script": script_name, "framework": framework}
            ))
        elif framework == 'pytorch':
            # Check for PyTorch model files
            issues.append(self._create_step_type_issue(
                "pytorch_model_artifact_check",
                "PyTorch CreateModel should include .pth or .pt model file",
                "Ensure PyTorch model state dict is saved properly",
                "INFO",
                {"script": script_name, "framework": framework}
            ))
        
        return issues
    
    def _has_step_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has step creation patterns."""
        step_keywords = ['create_step', 'ModelStep', 'CreateModelStep']
        return self._has_pattern_in_analysis(builder_analysis, 'step_creation_patterns', step_keywords)
    
    def _has_model_name_generation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has model name generation patterns."""
        name_keywords = ['model_name', 'ModelName', 'generate_name']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', name_keywords)
    
    def _has_dependency_handling_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has dependency handling patterns."""
        dependency_keywords = ['depends_on', 'DependsOn', 'dependencies']
        return self._has_pattern_in_analysis(builder_analysis, 'configuration_patterns', dependency_keywords)
    
    def _detect_framework_from_builder_analysis(self, builder_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Detect framework from builder analysis results.
        
        Args:
            builder_analysis: Builder analysis results
            
        Returns:
            Detected framework name or None
        """
        builder_methods = builder_analysis.get('builder_methods', [])
        
        # Convert to lowercase for case-insensitive matching
        builder_str = ' '.join(str(method).lower() for method in builder_methods)
        
        # Check for framework-specific patterns in builder
        if 'xgboost' in builder_str or 'xgb' in builder_str:
            return 'xgboost'
        elif 'pytorch' in builder_str or 'torch' in builder_str:
            return 'pytorch'
        elif 'sklearn' in builder_str or 'scikit' in builder_str:
            return 'sklearn'
        elif 'tensorflow' in builder_str or 'tf' in builder_str:
            return 'tensorflow'
        
        return None
    
    # Helper methods for file path resolution
    
    def _get_model_creation_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected model creation builder path."""
        base_name = script_name.replace('.py', '').replace('_model', '')
        return f"cursus/steps/builders/builder_{base_name}_model_step.py"
    
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
        
        # Check for framework-specific imports (model creation focused)
        if any('xgboost' in imp or 'xgb' in imp for imp in imports):
            return 'xgboost'
        elif any('torch' in imp for imp in imports):
            return 'pytorch'
        elif any('sklearn' in imp for imp in imports):
            return 'sklearn'
        elif any('tensorflow' in imp or 'tf' in imp for imp in imports):
            return 'tensorflow'
        
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
            'source': 'CreateModelStepEnhancer'
        }
    
    def get_createmodel_validation_requirements(self) -> Dict[str, Any]:
        """
        Get comprehensive CreateModel validation requirements.
        
        Returns:
            Dictionary containing CreateModel validation requirements
        """
        return {
            'required_patterns': {
                'model_loading': {
                    'keywords': ['load', 'pickle.load', 'joblib.load', 'torch.load', 'xgb.Booster'],
                    'description': 'Model artifact loading from SageMaker model directory',
                    'severity': 'ERROR'
                },
                'inference_functions': {
                    'keywords': ['model_fn', 'input_fn', 'predict_fn', 'output_fn'],
                    'description': 'SageMaker inference handler functions',
                    'severity': 'ERROR'
                },
                'model_path_references': {
                    'keywords': ['/opt/ml/model'],
                    'description': 'References to SageMaker model artifact directory',
                    'severity': 'WARNING'
                },
                'container_configuration': {
                    'keywords': ['image_uri', 'container', 'image'],
                    'description': 'Container image specification for model deployment',
                    'severity': 'WARNING'
                }
            },
            'framework_requirements': {
                'xgboost': {
                    'imports': ['xgboost', 'xgb'],
                    'functions': ['xgb.Booster', 'load_model'],
                    'patterns': ['xgboost_model_loading', 'booster_creation']
                },
                'pytorch': {
                    'imports': ['torch'],
                    'functions': ['torch.load', 'load_state_dict'],
                    'patterns': ['pytorch_model_loading', 'state_dict_loading']
                },
                'sklearn': {
                    'imports': ['sklearn', 'joblib'],
                    'functions': ['joblib.load', 'pickle.load'],
                    'patterns': ['sklearn_model_loading', 'joblib_loading']
                }
            },
            'sagemaker_paths': {
                'model_artifacts': '/opt/ml/model',
                'inference_code': '/opt/ml/code'
            },
            'inference_functions': {
                'model_fn': {
                    'description': 'Load and return the model for inference',
                    'required': True,
                    'signature': 'model_fn(model_dir)'
                },
                'input_fn': {
                    'description': 'Parse and preprocess input data',
                    'required': False,
                    'signature': 'input_fn(request_body, content_type)'
                },
                'predict_fn': {
                    'description': 'Run inference on the model',
                    'required': False,
                    'signature': 'predict_fn(input_data, model)'
                },
                'output_fn': {
                    'description': 'Postprocess and format output data',
                    'required': False,
                    'signature': 'output_fn(prediction, accept)'
                }
            },
            'validation_levels': {
                'level1': 'Model artifact handling validation',
                'level2': 'Inference code validation',
                'level3': 'Container configuration validation',
                'level4': 'Model creation builder validation'
            }
        }
    
    def validate_createmodel_script_comprehensive(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """
        Perform comprehensive CreateModel script validation.
        
        Args:
            script_name: Name of the CreateModel script
            script_content: Content of the CreateModel script
            
        Returns:
            Comprehensive validation results
        """
        # Create a basic script analysis from content
        script_analysis = self._create_script_analysis_from_content(script_content)
        
        # Detect framework from content directly (simple pattern matching)
        framework = self._detect_framework_from_content(script_content)
        
        # Get framework-specific patterns (basic detection)
        framework_patterns = self._detect_framework_patterns_from_content(script_content, framework)
        
        # Create comprehensive analysis
        analysis = {
            'script_name': script_name,
            'framework': framework,
            'createmodel_patterns': {
                'model_loading': self._has_model_loading_patterns(script_analysis),
                'inference_functions': self._has_inference_patterns(script_analysis),
                'model_path_references': self._has_model_path_references(script_analysis),
                'model_fn': self._has_model_fn_pattern(script_analysis),
                'predict_fn': self._has_predict_fn_pattern(script_analysis)
            },
            'framework_patterns': framework_patterns,
            'validation_results': {
                'model_loading': self._has_model_loading_patterns(script_analysis),
                'inference_code': self._has_inference_patterns(script_analysis),
                'model_paths': self._has_model_path_references(script_analysis),
                'framework_specific': bool(framework_patterns)
            }
        }
        
        return analysis
    
    def _detect_framework_from_content(self, content: str) -> Optional[str]:
        """
        Detect framework from script content.
        
        Args:
            content: Script content
            
        Returns:
            Detected framework name or None
        """
        content_lower = content.lower()
        
        # Check for framework-specific patterns
        if 'xgboost' in content_lower or 'xgb' in content_lower:
            return 'xgboost'
        elif 'torch' in content_lower or 'pytorch' in content_lower:
            return 'pytorch'
        elif 'sklearn' in content_lower or 'scikit' in content_lower:
            return 'sklearn'
        elif 'tensorflow' in content_lower or 'tf.' in content_lower:
            return 'tensorflow'
        
        return None
    
    def _detect_framework_patterns_from_content(self, content: str, framework: Optional[str]) -> Dict[str, Any]:
        """
        Detect framework-specific patterns from content.
        
        Args:
            content: Script content
            framework: Detected framework
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {}
        
        if framework == 'xgboost':
            patterns = {
                'model_creation': 'xgb.Booster' in content or 'XGBClassifier' in content,
                'model_loading': 'load_model' in content or 'pickle.load' in content,
                'prediction': 'predict' in content or 'DMatrix' in content
            }
        elif framework == 'pytorch':
            patterns = {
                'model_creation': 'torch.nn' in content or 'nn.Module' in content,
                'model_loading': 'torch.load' in content or 'load_state_dict' in content,
                'prediction': 'forward' in content or 'eval()' in content
            }
        elif framework == 'sklearn':
            patterns = {
                'model_creation': 'sklearn' in content,
                'model_loading': 'joblib.load' in content or 'pickle.load' in content,
                'prediction': 'predict' in content or 'predict_proba' in content
            }
        
        return patterns
    
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
        common_paths = ['/opt/ml/model', '/opt/ml/code']
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
    
    def get_createmodel_best_practices(self) -> Dict[str, Any]:
        """
        Get CreateModel step best practices and recommendations.
        
        Returns:
            Dictionary containing best practices for CreateModel steps
        """
        return {
            'model_loading_best_practices': {
                'use_model_dir_parameter': 'Always use model_dir parameter in model_fn',
                'handle_multiple_files': 'Handle cases where model consists of multiple files',
                'error_handling': 'Add proper error handling for model loading failures',
                'logging': 'Add logging for model loading process'
            },
            'inference_best_practices': {
                'input_validation': 'Validate input data format and content',
                'output_formatting': 'Ensure output is in expected format',
                'error_handling': 'Handle inference errors gracefully',
                'performance': 'Optimize inference performance for production'
            },
            'framework_specific_practices': {
                'xgboost': {
                    'model_loading': 'Use xgb.Booster(model_file=...) for loading',
                    'prediction': 'Use DMatrix for input data if needed',
                    'serialization': 'Save model in XGBoost native format'
                },
                'pytorch': {
                    'model_loading': 'Load both model architecture and weights',
                    'device_handling': 'Handle CPU/GPU device placement',
                    'eval_mode': 'Set model to evaluation mode for inference'
                },
                'sklearn': {
                    'model_loading': 'Use joblib.load for sklearn models',
                    'preprocessing': 'Include preprocessing steps if needed',
                    'feature_names': 'Handle feature name consistency'
                }
            },
            'container_best_practices': {
                'image_selection': 'Choose appropriate base container image',
                'dependencies': 'Include all required dependencies',
                'security': 'Follow security best practices for container images',
                'size_optimization': 'Optimize container image size'
            }
        }
