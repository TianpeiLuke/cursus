"""
Transform Step Enhancer

Transform step-specific validation enhancement.
Provides validation for batch transform patterns, model inference, and transform configuration.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer


class TransformStepEnhancer(BaseStepEnhancer):
    """
    Transform step-specific validation enhancement.
    
    Provides validation for:
    - Batch processing validation
    - Transform input validation
    - Model inference validation
    - Transform builder validation
    """
    
    def __init__(self):
        """Initialize the Transform step enhancer."""
        super().__init__("Transform")
        self.reference_examples = [
            "builder_batch_transform_step.py"
        ]
        self.framework_validators = {
            "xgboost": self._validate_xgboost_transform,
            "pytorch": self._validate_pytorch_transform
        }
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Add Transform-specific validation.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with Transform-specific issues
        """
        additional_issues = []
        
        # Get script analysis
        script_analysis = self._get_script_analysis(script_name)
        framework = self._detect_framework_from_script_analysis(script_analysis)
        
        # Level 1: Batch processing validation
        additional_issues.extend(self._validate_batch_processing_patterns(script_analysis, script_name))
        
        # Level 2: Transform input validation
        additional_issues.extend(self._validate_transform_input_specifications(script_name))
        
        # Level 3: Model inference validation
        additional_issues.extend(self._validate_model_inference_patterns(script_analysis, script_name))
        
        # Level 4: Transform builder validation
        additional_issues.extend(self._validate_transform_builder(script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(script_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_batch_processing_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate batch processing patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of batch processing validation issues
        """
        issues = []
        
        # Check for batch processing patterns
        if not self._has_batch_processing_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_batch_processing",
                "Transform script should implement batch processing logic",
                "Add batch processing for input data transformation",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for input data handling
        if not self._has_input_data_handling_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_input_data_handling",
                "Transform script should handle input data",
                "Add input data loading and processing from /opt/ml/transform/",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/transform/"}
            ))
        
        # Check for output data generation
        if not self._has_output_data_generation_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_output_data_generation",
                "Transform script should generate output data",
                "Add output data generation and saving",
                "WARNING",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_transform_input_specifications(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate transform input specifications.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of transform input specification validation issues
        """
        issues = []
        
        # Check if transform specification exists
        spec_path = self._get_transform_spec_path(script_name)
        if not spec_path or not Path(spec_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_transform_specification",
                f"Transform specification not found for {script_name}",
                f"Create transform specification file for {script_name}",
                "INFO",
                {"script": script_name, "expected_spec_path": spec_path}
            ))
        
        return issues
    
    def _validate_model_inference_patterns(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate model inference patterns.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of model inference validation issues
        """
        issues = []
        
        # Check for model loading patterns
        if not self._has_model_loading_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_loading",
                "Transform script should load model for inference",
                "Add model loading from model artifacts",
                "WARNING",
                {"script": script_name}
            ))
        
        # Check for inference patterns
        if not self._has_inference_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_inference_patterns",
                "Transform script should implement model inference",
                "Add model prediction/inference logic",
                "WARNING",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_transform_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate transform builder patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of transform builder validation issues
        """
        issues = []
        
        # Check if transform builder exists
        builder_path = self._get_transform_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_transform_builder",
                f"Transform builder not found for {script_name}",
                f"Create transform builder file for {script_name}",
                "WARNING",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_transformer_creation_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_transformer_creation",
                    "Transform builder should create transformer",
                    "Add _create_transformer method to transform builder",
                    "ERROR",
                    {"script": script_name, "builder_path": builder_path}
                ))
        
        return issues
    
    def _validate_xgboost_transform(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        XGBoost-specific transform validation.
        
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
                "XGBoost Transform should load XGBoost model",
                "Add XGBoost model loading for inference",
                "WARNING",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        # Check for XGBoost prediction
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['predict', 'xgb.predict']):
            issues.append(self._create_step_type_issue(
                "missing_xgboost_prediction",
                "XGBoost Transform should use XGBoost prediction",
                "Add XGBoost model prediction calls",
                "WARNING",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        return issues
    
    def _validate_pytorch_transform(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        PyTorch-specific transform validation.
        
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
                "PyTorch Transform should load PyTorch model",
                "Add PyTorch model loading for inference",
                "WARNING",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        # Check for PyTorch inference
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['forward', 'eval', 'no_grad']):
            issues.append(self._create_step_type_issue(
                "missing_pytorch_inference",
                "PyTorch Transform should implement inference logic",
                "Add PyTorch model inference with eval() and no_grad()",
                "WARNING",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_batch_processing_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has batch processing patterns."""
        batch_keywords = ['batch', 'transform', 'process', 'iterate', 'loop']
        return self._has_pattern_in_analysis(script_analysis, 'functions', batch_keywords)
    
    def _has_input_data_handling_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has input data handling patterns."""
        input_keywords = ['read', 'load', 'input', '/opt/ml/transform']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', input_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/transform']))
    
    def _has_output_data_generation_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has output data generation patterns."""
        output_keywords = ['write', 'save', 'output', 'result']
        return self._has_pattern_in_analysis(script_analysis, 'functions', output_keywords)
    
    def _has_model_loading_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model loading patterns."""
        loading_keywords = ['load', 'pickle.load', 'joblib.load', 'torch.load', 'xgb.Booster']
        return self._has_pattern_in_analysis(script_analysis, 'functions', loading_keywords)
    
    def _has_inference_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has inference patterns."""
        inference_keywords = ['predict', 'inference', 'forward', 'eval']
        return self._has_pattern_in_analysis(script_analysis, 'functions', inference_keywords)
    
    def _has_transformer_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has transformer creation patterns."""
        transformer_keywords = ['_create_transformer', 'Transformer', 'Transform']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', transformer_keywords)
    
    # Helper methods for file path resolution
    
    def _get_transform_spec_path(self, script_name: str) -> Optional[str]:
        """Get expected transform specification path."""
        base_name = script_name.replace('.py', '').replace('_transform', '')
        return f"cursus/steps/specs/{base_name}_transform_spec.py"
    
    def _get_transform_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected transform builder path."""
        base_name = script_name.replace('.py', '').replace('_transform', '')
        return f"cursus/steps/builders/builder_{base_name}_transform_step.py"
