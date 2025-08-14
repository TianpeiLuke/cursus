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
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with CreateModel-specific issues
        """
        additional_issues = []
        
        # Get script analysis
        script_analysis = self._get_script_analysis(script_name)
        framework = self._detect_framework_from_script_analysis(script_analysis)
        
        # Level 1: Model artifact handling validation
        additional_issues.extend(self._validate_model_artifact_handling(script_analysis, script_name))
        
        # Level 2: Inference code validation
        additional_issues.extend(self._validate_inference_code_patterns(script_analysis, script_name))
        
        # Level 3: Container configuration validation
        additional_issues.extend(self._validate_container_configuration(script_name))
        
        # Level 4: Model creation builder validation
        additional_issues.extend(self._validate_model_creation_builder(script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(script_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_model_artifact_handling(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model artifact loading patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of model artifact handling validation issues
        """
        issues = []
        
        # Check for model loading patterns
        if not self._has_model_loading_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_loading",
                "CreateModel script should load model artifacts",
                "Add model loading from /opt/ml/model/",
                "ERROR",
                {"script": script_name, "expected_path": "/opt/ml/model/"}
            ))
        
        # Check for model artifact path references
        if not self._has_model_path_references(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_path_references",
                "CreateModel script should reference model artifact paths",
                "Add references to /opt/ml/model/ directory",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/model/"}
            ))
        
        return issues
    
    def _validate_inference_code_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate inference code implementation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of inference code validation issues
        """
        issues = []
        
        # Check for inference function patterns
        if not self._has_inference_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_inference_code",
                "CreateModel should implement inference logic",
                "Add model_fn, input_fn, predict_fn, or output_fn functions",
                "ERROR",
                {"script": script_name}
            ))
        
        # Check for model loading function
        if not self._has_model_fn_pattern(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_fn",
                "CreateModel should implement model_fn for model loading",
                "Add model_fn function to load and return the model",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for prediction function
        if not self._has_predict_fn_pattern(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_predict_fn",
                "CreateModel should implement predict_fn for inference",
                "Add predict_fn function to handle model predictions",
                "WARNING",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_container_configuration(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate container configuration.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of container configuration validation issues
        """
        issues = []
        
        # Check for container image specification
        builder_analysis = self._get_builder_analysis(script_name)
        if not self._has_container_image_patterns(builder_analysis):
            issues.append(self._create_step_type_issue(
                "missing_container_image",
                "CreateModel builder should specify container image",
                "Add container image specification in model creation",
                "WARNING",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_model_creation_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model creation builder validation.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of model creation builder validation issues
        """
        issues = []
        
        # Check if model creation builder exists
        builder_path = self._get_model_creation_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_model_creation_builder",
                f"Model creation builder not found for {script_name}",
                f"Create model creation builder file for {script_name}",
                "WARNING",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_model_creation_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_model_creation_method",
                    "Model creation builder should create model",
                    "Add _create_model method to model creation builder",
                    "ERROR",
                    {"script": script_name, "builder_path": builder_path}
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
    
    # Helper methods for file path resolution
    
    def _get_model_creation_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected model creation builder path."""
        base_name = script_name.replace('.py', '').replace('_model', '')
        return f"cursus/steps/builders/builder_{base_name}_model_step.py"
