"""
Factory for creating appropriate universal step builder test variants.
"""

from typing import Type, Dict, Any, Optional
from ...core.base.builder_base import StepBuilderBase
from .step_info_detector import StepInfoDetector


class UniversalStepBuilderTestFactory:
    """Factory for creating appropriate test variants based on step type."""
    
    # Variant mapping will be populated as variants are implemented
    VARIANT_MAP = {}
    
    @classmethod
    def _initialize_variants(cls):
        """Initialize built-in variants."""
        if not cls.VARIANT_MAP:
            # Import and register Processing variant
            try:
                from .variants.processing_test import ProcessingStepBuilderTest
                cls.register_variant("Processing", ProcessingStepBuilderTest)
            except ImportError:
                pass  # Variant not available
    
    @classmethod
    def create_tester(
        cls, 
        builder_class: Type[StepBuilderBase], 
        **kwargs
    ) -> 'UniversalStepBuilderTestBase':
        """
        Create appropriate tester variant for the builder class.
        
        Args:
            builder_class: The step builder class to test
            **kwargs: Additional arguments to pass to the tester
            
        Returns:
            Appropriate test variant instance
        """
        # Initialize variants if not already done
        cls._initialize_variants()
        
        # Detect step information
        detector = StepInfoDetector(builder_class)
        step_info = detector.detect_step_info()
        
        # Get SageMaker step type
        sagemaker_step_type = step_info.get("sagemaker_step_type")
        
        # Select appropriate variant
        variant_class = cls._select_variant(sagemaker_step_type, step_info)
        
        # Create and return variant instance
        return variant_class(builder_class, **kwargs)
    
    @classmethod
    def _select_variant(
        cls, 
        sagemaker_step_type: Optional[str], 
        step_info: Dict[str, Any]
    ) -> Type['UniversalStepBuilderTestBase']:
        """
        Select appropriate test variant based on step type and info.
        
        Args:
            sagemaker_step_type: The SageMaker step type
            step_info: Complete step information
            
        Returns:
            Test variant class
        """
        # Check for specific variant implementations
        if sagemaker_step_type and sagemaker_step_type in cls.VARIANT_MAP:
            return cls.VARIANT_MAP[sagemaker_step_type]
        
        # Check for custom step patterns
        if step_info.get("is_custom_step"):
            if "Custom" in cls.VARIANT_MAP:
                return cls.VARIANT_MAP["Custom"]
        
        # Fallback to generic variant (will be implemented later)
        # For now, import and return the base class with a concrete implementation
        from .generic_test import GenericStepBuilderTest
        return GenericStepBuilderTest
    
    @classmethod
    def register_variant(
        cls, 
        step_type: str, 
        variant_class: Type['UniversalStepBuilderTestBase']
    ) -> None:
        """
        Register a new test variant for a step type.
        
        Args:
            step_type: The SageMaker step type
            variant_class: The test variant class
        """
        cls.VARIANT_MAP[step_type] = variant_class
    
    @classmethod
    def get_available_variants(cls) -> Dict[str, Type['UniversalStepBuilderTestBase']]:
        """
        Get all available test variants.
        
        Returns:
            Dictionary mapping step types to variant classes
        """
        return cls.VARIANT_MAP.copy()
    
    @classmethod
    def supports_step_type(cls, step_type: str) -> bool:
        """
        Check if a step type is supported by registered variants.
        
        Args:
            step_type: The SageMaker step type
            
        Returns:
            True if supported, False otherwise
        """
        return step_type in cls.VARIANT_MAP
