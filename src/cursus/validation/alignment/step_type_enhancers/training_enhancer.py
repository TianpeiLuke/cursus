"""
Training Step Enhancer

Training step-specific validation enhancement.
Provides comprehensive validation for training scripts including framework-specific patterns,
model saving, hyperparameter loading, and training loop validation.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_enhancer import BaseStepEnhancer
from ..framework_patterns import (
    detect_training_patterns, 
    detect_xgboost_patterns, 
    detect_pytorch_patterns,
    detect_framework_from_script_content
)


class TrainingStepEnhancer(BaseStepEnhancer):
    """
    Training step-specific validation enhancement.
    
    Provides validation for:
    - Training script patterns (training loops, model saving, hyperparameter loading)
    - Framework-specific validation (XGBoost, PyTorch, etc.)
    - Training specifications alignment
    - Training dependencies validation
    - Training builder patterns
    """
    
    def __init__(self):
        """Initialize the training step enhancer."""
        super().__init__("Training")
        self.reference_examples = [
            "xgboost_training.py",
            "pytorch_training.py", 
            "builder_xgboost_training_step.py"
        ]
        self.framework_validators = {
            "xgboost": self._validate_xgboost_training,
            "pytorch": self._validate_pytorch_training
        }
    
    def enhance_validation(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Add training-specific validation to existing results.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with training-specific issues
        """
        additional_issues = []
        
        # Get script analysis from existing validation or create new one
        script_analysis = self._get_script_analysis(script_name)
        framework = self._detect_framework_from_script_analysis(script_analysis)
        
        # Level 1: Training script patterns
        additional_issues.extend(self._validate_training_script_patterns(script_analysis, framework, script_name))
        
        # Level 2: Training specifications
        additional_issues.extend(self._validate_training_specifications(script_name))
        
        # Level 3: Training dependencies
        additional_issues.extend(self._validate_training_dependencies(script_name, framework))
        
        # Level 4: Training builder patterns
        additional_issues.extend(self._validate_training_builder(script_name))
        
        # Framework-specific validation
        if framework and framework in self.framework_validators:
            framework_validator = self.framework_validators[framework]
            additional_issues.extend(framework_validator(script_analysis, script_name))
        
        return self._merge_results(existing_results, additional_issues)
    
    def _validate_training_script_patterns(self, script_analysis: Dict[str, Any], framework: Optional[str], script_name: str) -> List[Dict[str, Any]]:
        """
        Validate training-specific script patterns.
        
        Args:
            script_analysis: Script analysis results
            framework: Detected framework
            script_name: Name of the script
            
        Returns:
            List of training pattern validation issues
        """
        issues = []
        
        # Check for training loop patterns
        if not self._has_training_loop_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_training_loop",
                "Training script should contain model training logic",
                "Add model.fit(), model.train(), or equivalent training loop",
                "WARNING",
                {"script": script_name, "framework": framework}
            ))
        
        # Check for model saving patterns
        if not self._has_model_saving_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_model_saving",
                "Training script should save model artifacts",
                "Add model saving to /opt/ml/model/ directory",
                "ERROR",
                {"script": script_name, "expected_path": "/opt/ml/model/"}
            ))
        
        # Check for hyperparameter loading patterns
        if not self._has_hyperparameter_loading_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_hyperparameter_loading",
                "Training script should load hyperparameters from file",
                "Add hyperparameter loading from /opt/ml/input/data/config/",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/input/data/config/"}
            ))
        
        # Check for training data loading patterns
        if not self._has_training_data_loading_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_training_data_loading",
                "Training script should load training data",
                "Add training data loading from /opt/ml/input/data/train/",
                "WARNING",
                {"script": script_name, "expected_path": "/opt/ml/input/data/train/"}
            ))
        
        # Check for evaluation patterns
        if not self._has_evaluation_patterns(script_analysis):
            issues.append(self._create_step_type_issue(
                "missing_evaluation_patterns",
                "Training script should include model evaluation",
                "Add model evaluation and metrics calculation",
                "INFO",
                {"script": script_name}
            ))
        
        return issues
    
    def _validate_training_specifications(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate training specifications alignment.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of training specification validation issues
        """
        issues = []
        
        # Check if training specification exists
        spec_path = self._get_training_spec_path(script_name)
        if not spec_path or not Path(spec_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_training_specification",
                f"Training specification not found for {script_name}",
                f"Create training specification file for {script_name}",
                "WARNING",
                {"script": script_name, "expected_spec_path": spec_path}
            ))
        
        return issues
    
    def _validate_training_dependencies(self, script_name: str, framework: Optional[str]) -> List[Dict[str, Any]]:
        """
        Validate training dependencies.
        
        Args:
            script_name: Name of the script
            framework: Detected framework
            
        Returns:
            List of training dependency validation issues
        """
        issues = []
        
        # Check for framework-specific dependencies
        if framework:
            expected_dependencies = self._get_expected_framework_dependencies(framework)
            for dependency in expected_dependencies:
                if not self._has_dependency(script_name, dependency):
                    issues.append(self._create_step_type_issue(
                        "missing_framework_dependency",
                        f"Training script should declare {framework} dependency: {dependency}",
                        f"Add {dependency} to requirements or imports",
                        "INFO",
                        {"script": script_name, "framework": framework, "dependency": dependency}
                    ))
        
        return issues
    
    def _validate_training_builder(self, script_name: str) -> List[Dict[str, Any]]:
        """
        Validate training builder patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            List of training builder validation issues
        """
        issues = []
        
        # Check if training builder exists
        builder_path = self._get_training_builder_path(script_name)
        if not builder_path or not Path(builder_path).exists():
            issues.append(self._create_step_type_issue(
                "missing_training_builder",
                f"Training builder not found for {script_name}",
                f"Create training builder file for {script_name}",
                "WARNING",
                {"script": script_name, "expected_builder_path": builder_path}
            ))
        else:
            # Validate builder patterns
            builder_analysis = self._get_builder_analysis(script_name)
            if not self._has_estimator_creation_patterns(builder_analysis):
                issues.append(self._create_step_type_issue(
                    "missing_estimator_creation",
                    "Training builder should create estimator",
                    "Add _create_estimator method to training builder",
                    "ERROR",
                    {"script": script_name, "builder_path": builder_path}
                ))
        
        return issues
    
    def _validate_xgboost_training(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        XGBoost-specific training validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of XGBoost-specific validation issues
        """
        issues = []
        
        # Check for XGBoost imports
        if not self._has_pattern_in_analysis(script_analysis, 'imports', ['xgboost', 'xgb']):
            issues.append(self._create_step_type_issue(
                "missing_xgboost_import",
                "XGBoost training script should import xgboost",
                "Add 'import xgboost as xgb' to script",
                "ERROR",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        # Check for DMatrix usage
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['DMatrix', 'xgb.DMatrix']):
            issues.append(self._create_step_type_issue(
                "missing_dmatrix_usage",
                "XGBoost training script should use DMatrix for data",
                "Convert training data to xgb.DMatrix format",
                "WARNING",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        # Check for XGBoost training call
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['xgb.train', 'train']):
            issues.append(self._create_step_type_issue(
                "missing_xgboost_train_call",
                "XGBoost training script should call xgb.train()",
                "Add xgb.train() call with appropriate parameters",
                "ERROR",
                {"script": script_name, "framework": "xgboost"}
            ))
        
        return issues
    
    def _validate_pytorch_training(self, script_analysis: Dict[str, Any], script_name: str) -> List[Dict[str, Any]]:
        """
        PyTorch-specific training validation.
        
        Args:
            script_analysis: Script analysis results
            script_name: Name of the script
            
        Returns:
            List of PyTorch-specific validation issues
        """
        issues = []
        
        # Check for PyTorch imports
        if not self._has_pattern_in_analysis(script_analysis, 'imports', ['torch', 'pytorch']):
            issues.append(self._create_step_type_issue(
                "missing_pytorch_import",
                "PyTorch training script should import torch",
                "Add 'import torch' to script",
                "ERROR",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        # Check for model definition
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['nn.Module', 'torch.nn']):
            issues.append(self._create_step_type_issue(
                "missing_model_definition",
                "PyTorch training script should define model class",
                "Create model class inheriting from nn.Module",
                "WARNING",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        # Check for optimizer usage
        if not self._has_pattern_in_analysis(script_analysis, 'functions', ['optim', 'optimizer']):
            issues.append(self._create_step_type_issue(
                "missing_optimizer",
                "PyTorch training script should use optimizer",
                "Add optimizer (e.g., torch.optim.Adam) to training loop",
                "WARNING",
                {"script": script_name, "framework": "pytorch"}
            ))
        
        return issues
    
    # Helper methods for pattern detection
    
    def _has_training_loop_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has training loop patterns."""
        training_keywords = ['fit', 'train', 'epoch', 'batch', 'forward', 'backward']
        return self._has_pattern_in_analysis(script_analysis, 'functions', training_keywords)
    
    def _has_model_saving_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has model saving patterns."""
        saving_keywords = ['save', 'dump', 'pickle', 'joblib', 'torch.save', '/opt/ml/model']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', saving_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/model']))
    
    def _has_hyperparameter_loading_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has hyperparameter loading patterns."""
        hp_keywords = ['hyperparameters', 'config', 'params', '/opt/ml/input/data/config']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', hp_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/input/data/config']))
    
    def _has_training_data_loading_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has training data loading patterns."""
        data_keywords = ['read_csv', 'load', 'data', '/opt/ml/input/data/train']
        return (self._has_pattern_in_analysis(script_analysis, 'functions', data_keywords) or
                self._has_pattern_in_analysis(script_analysis, 'path_references', ['/opt/ml/input/data/train']))
    
    def _has_evaluation_patterns(self, script_analysis: Dict[str, Any]) -> bool:
        """Check if script has evaluation patterns."""
        eval_keywords = ['evaluate', 'score', 'metric', 'accuracy', 'loss', 'validation']
        return self._has_pattern_in_analysis(script_analysis, 'functions', eval_keywords)
    
    def _has_estimator_creation_patterns(self, builder_analysis: Dict[str, Any]) -> bool:
        """Check if builder has estimator creation patterns."""
        estimator_keywords = ['_create_estimator', 'Estimator', 'XGBoost', 'PyTorch']
        return self._has_pattern_in_analysis(builder_analysis, 'builder_methods', estimator_keywords)
    
    # Helper methods for file path resolution
    
    def _get_training_spec_path(self, script_name: str) -> Optional[str]:
        """Get expected training specification path."""
        base_name = script_name.replace('.py', '').replace('_training', '')
        return f"cursus/steps/specs/{base_name}_training_spec.py"
    
    def _get_training_builder_path(self, script_name: str) -> Optional[str]:
        """Get expected training builder path."""
        base_name = script_name.replace('.py', '').replace('_training', '')
        return f"cursus/steps/builders/builder_{base_name}_training_step.py"
    
    def _get_expected_framework_dependencies(self, framework: str) -> List[str]:
        """Get expected dependencies for framework."""
        dependencies = {
            "xgboost": ["xgboost", "pandas", "numpy"],
            "pytorch": ["torch", "torchvision", "numpy"],
            "sklearn": ["scikit-learn", "pandas", "numpy"],
            "tensorflow": ["tensorflow", "numpy"]
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
        
        # Check for framework-specific imports
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
            'source': 'TrainingStepEnhancer'
        }
    
    def get_training_validation_requirements(self) -> Dict[str, Any]:
        """
        Get comprehensive training validation requirements.
        
        Returns:
            Dictionary containing training validation requirements
        """
        return {
            'required_patterns': {
                'training_loop': {
                    'keywords': ['fit', 'train', 'epoch', 'batch'],
                    'description': 'Training loop implementation',
                    'severity': 'ERROR'
                },
                'model_saving': {
                    'keywords': ['save', 'dump', 'pickle', '/opt/ml/model'],
                    'description': 'Model artifact saving',
                    'severity': 'ERROR'
                },
                'hyperparameter_loading': {
                    'keywords': ['hyperparameters', 'config', '/opt/ml/input/data/config'],
                    'description': 'Hyperparameter loading from SageMaker',
                    'severity': 'WARNING'
                },
                'data_loading': {
                    'keywords': ['read_csv', 'load', '/opt/ml/input/data/train'],
                    'description': 'Training data loading',
                    'severity': 'WARNING'
                }
            },
            'framework_requirements': {
                'xgboost': {
                    'imports': ['xgboost', 'xgb'],
                    'functions': ['DMatrix', 'xgb.train'],
                    'patterns': ['dmatrix_creation', 'xgboost_training']
                },
                'pytorch': {
                    'imports': ['torch', 'torch.nn'],
                    'functions': ['nn.Module', 'optim', 'loss'],
                    'patterns': ['model_definition', 'training_loop', 'optimizer_usage']
                }
            },
            'sagemaker_paths': {
                'model_output': '/opt/ml/model',
                'hyperparameters': '/opt/ml/input/data/config',
                'training_data': '/opt/ml/input/data/train',
                'validation_data': '/opt/ml/input/data/validation'
            },
            'validation_levels': {
                'level1': 'Script pattern validation',
                'level2': 'Specification alignment',
                'level3': 'Dependency validation',
                'level4': 'Builder pattern validation'
            }
        }
    
    def validate_training_script_comprehensive(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """
        Perform comprehensive training script validation.
        
        Args:
            script_name: Name of the training script
            script_content: Content of the training script
            
        Returns:
            Comprehensive validation results
        """
        # Create a basic script analysis from content
        script_analysis = self._create_script_analysis_from_content(script_content)
        
        # Detect patterns in script analysis
        training_patterns = detect_training_patterns(script_analysis)
        
        # Detect framework from content directly
        framework = detect_framework_from_script_content(script_content)
        
        # Get framework-specific patterns
        framework_patterns = {}
        if framework == 'xgboost':
            framework_patterns = detect_xgboost_patterns(script_analysis)
        elif framework == 'pytorch':
            framework_patterns = detect_pytorch_patterns(script_analysis)
        
        # Create comprehensive analysis
        analysis = {
            'script_name': script_name,
            'framework': framework,
            'training_patterns': training_patterns,
            'framework_patterns': framework_patterns,
            'validation_results': {
                'training_loop': bool(training_patterns.get('has_training_loop')),
                'model_saving': bool(training_patterns.get('has_model_saving')),
                'hyperparameter_loading': bool(training_patterns.get('has_hyperparameter_loading')),
                'data_loading': bool(training_patterns.get('has_data_loading')),
                'evaluation': bool(training_patterns.get('has_evaluation'))
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
        common_paths = ['/opt/ml/model', '/opt/ml/input/data/config', '/opt/ml/input/data/train']
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
