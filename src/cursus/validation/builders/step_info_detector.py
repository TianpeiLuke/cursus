"""
Step information detection utilities for universal step builder tests.
"""

from typing import Dict, Any, Optional, Type
from pathlib import Path
from ...core.base.builder_base import StepBuilderBase


class StepInfoDetector:
    """Detects step information from builder classes using step catalog."""

    def __init__(self, builder_class: Type[StepBuilderBase]):
        """
        Initialize detector with builder class.

        Args:
            builder_class: The step builder class to analyze
        """
        self.builder_class = builder_class
        self._step_info = None
        self._catalog = None

    def detect_step_info(self) -> Dict[str, Any]:
        """
        Detect comprehensive step information from builder class using step catalog.

        Returns:
            Dictionary containing step information
        """
        if self._step_info is None:
            self._step_info = self._analyze_builder_class_with_catalog()
        return self._step_info

    def _get_step_catalog(self):
        """Get step catalog instance with lazy loading."""
        if self._catalog is None:
            try:
                from ...step_catalog import StepCatalog
                
                # PORTABLE: Package-only discovery (works in all deployment scenarios)
                self._catalog = StepCatalog(workspace_dirs=None)
            except ImportError:
                self._catalog = None
        return self._catalog

    def _analyze_builder_class_with_catalog(self) -> Dict[str, Any]:
        """Analyze builder class using step catalog for step information."""
        class_name = self.builder_class.__name__
        catalog = self._get_step_catalog()
        
        if catalog:
            # Use step catalog to find step information
            step_name = self._find_step_name_via_catalog(class_name, catalog)
            
            if step_name:
                # Get comprehensive step info from catalog
                step_info = catalog.get_step_info(step_name)
                if step_info:
                    # Use catalog's framework detection
                    framework = catalog.detect_framework(step_name)
                    
                    return {
                        "builder_class_name": class_name,
                        "step_name": step_name,
                        "sagemaker_step_type": step_info.sagemaker_step_type,
                        "framework": framework,
                        "test_pattern": self._detect_test_pattern(class_name, step_info.sagemaker_step_type),
                        "is_custom_step": self._is_custom_step(class_name),
                        "registry_info": step_info.registry_data,
                    }
        
        # Fallback to basic analysis if catalog unavailable
        return self._fallback_analysis(class_name)

    def _find_step_name_via_catalog(self, class_name: str, catalog) -> Optional[str]:
        """Find step name using step catalog's builder class information."""
        try:
            # Get all available steps
            available_steps = catalog.list_available_steps()
            
            for step_name in available_steps:
                step_info = catalog.get_step_info(step_name)
                if step_info and step_info.registry_data.get("builder_step_name") == class_name:
                    return step_name
                    
        except Exception:
            pass  # Fall back to basic analysis
            
        return None

    def _fallback_analysis(self, class_name: str) -> Dict[str, Any]:
        """Fallback analysis when step catalog is unavailable."""
        return {
            "builder_class_name": class_name,
            "step_name": None,
            "sagemaker_step_type": None,
            "framework": self._detect_framework_basic(class_name),
            "test_pattern": "standard",
            "is_custom_step": self._is_custom_step(class_name),
            "registry_info": {},
        }

    def _detect_framework_basic(self, class_name: str) -> Optional[str]:
        """Basic framework detection from class name."""
        class_name_lower = class_name.lower()
        
        if "xgboost" in class_name_lower:
            return "xgboost"
        elif "pytorch" in class_name_lower:
            return "pytorch"
        elif "tensorflow" in class_name_lower:
            return "tensorflow"
        elif "sklearn" in class_name_lower:
            return "sklearn"
            
        return None

    def _detect_test_pattern(
        self, class_name: str, sagemaker_step_type: Optional[str]
    ) -> str:
        """Detect test pattern for the builder."""
        # Check for custom step patterns
        if self._is_custom_step(class_name):
            return "custom_step"

        # Check for custom package patterns
        framework = self._detect_framework_basic(class_name)
        if framework and framework != "sklearn":
            return "custom_package"

        # Default to standard pattern
        return "standard"

    def _is_custom_step(self, class_name: str) -> bool:
        """Check if this is a custom step implementation."""
        custom_step_indicators = [
            "CradleDataLoading",
            "MimsModelRegistration",
            "Custom",
        ]

        return any(indicator in class_name for indicator in custom_step_indicators)
