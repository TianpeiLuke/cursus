"""
Step information detection utilities for universal step builder tests.
"""

from typing import Dict, Any, Optional, Type
from ...core.base.builder_base import StepBuilderBase
from ...registry.step_names import STEP_NAMES, get_sagemaker_step_type


class StepInfoDetector:
    """Detects step information from builder classes."""
    
    def __init__(self, builder_class: Type[StepBuilderBase]):
        """
        Initialize detector with builder class.
        
        Args:
            builder_class: The step builder class to analyze
        """
        self.builder_class = builder_class
        self._step_info = None
    
    def detect_step_info(self) -> Dict[str, Any]:
        """
        Detect comprehensive step information from builder class.
        
        Returns:
            Dictionary containing step information
        """
        if self._step_info is None:
            self._step_info = self._analyze_builder_class()
        return self._step_info
    
    def _analyze_builder_class(self) -> Dict[str, Any]:
        """Analyze builder class to extract step information."""
        class_name = self.builder_class.__name__
        
        # Detect step name from class name
        step_name = self._detect_step_name_from_class(class_name)
        
        # Get SageMaker step type from registry
        sagemaker_step_type = get_sagemaker_step_type(step_name) if step_name else None
        
        # Detect framework from class name or methods
        framework = self._detect_framework()
        
        # Detect test pattern
        test_pattern = self._detect_test_pattern(class_name, sagemaker_step_type)
        
        return {
            "builder_class_name": class_name,
            "step_name": step_name,
            "sagemaker_step_type": sagemaker_step_type,
            "framework": framework,
            "test_pattern": test_pattern,
            "is_custom_step": self._is_custom_step(class_name),
            "registry_info": STEP_NAMES.get(step_name, {}) if step_name else {}
        }
    
    def _detect_step_name_from_class(self, class_name: str) -> Optional[str]:
        """Detect step name from builder class name."""
        # Remove common suffixes
        suffixes = ["StepBuilder", "Builder", "Step"]
        base_name = class_name
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Try to find matching step name in registry
        for step_name, info in STEP_NAMES.items():
            builder_step_name = info.get("builder_step_name", "")
            if builder_step_name:
                # Extract base name from builder step name
                builder_base = builder_step_name.replace("StepBuilder", "").replace("Builder", "")
                if builder_base == base_name:
                    return step_name
        
        return None
    
    def _detect_framework(self) -> Optional[str]:
        """Detect framework used by the builder."""
        class_name = self.builder_class.__name__.lower()
        
        # Check for framework indicators in class name
        if "xgboost" in class_name:
            return "xgboost"
        elif "pytorch" in class_name:
            return "pytorch"
        elif "tensorflow" in class_name:
            return "tensorflow"
        elif "sklearn" in class_name:
            return "sklearn"
        
        # Check for framework indicators in methods
        method_names = [method.lower() for method in dir(self.builder_class)]
        method_string = " ".join(method_names)
        
        if "xgboost" in method_string:
            return "xgboost"
        elif "pytorch" in method_string:
            return "pytorch"
        elif "tensorflow" in method_string:
            return "tensorflow"
        elif "sklearn" in method_string:
            return "sklearn"
        
        return None
    
    def _detect_test_pattern(self, class_name: str, sagemaker_step_type: Optional[str]) -> str:
        """Detect test pattern for the builder."""
        # Check for custom step patterns
        if self._is_custom_step(class_name):
            return "custom_step"
        
        # Check for custom package patterns
        framework = self._detect_framework()
        if framework and framework != "sklearn":
            return "custom_package"
        
        # Default to standard pattern
        return "standard"
    
    def _is_custom_step(self, class_name: str) -> bool:
        """Check if this is a custom step implementation."""
        custom_step_indicators = [
            "CradleDataLoading",
            "MimsModelRegistration",
            "Custom"
        ]
        
        return any(indicator in class_name for indicator in custom_step_indicators)
