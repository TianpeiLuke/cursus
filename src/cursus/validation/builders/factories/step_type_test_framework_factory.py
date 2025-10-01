"""
Step type test framework factory for specialized testing.

This module provides a factory to select appropriate test frameworks based on
SageMaker step types, enabling step-type-specific validation logic.
"""

from typing import Type, Any, Dict


class StepTypeTestFrameworkFactory:
    """Factory to select appropriate test framework based on step type."""
    
    @staticmethod
    def create_tester(builder_class: Type, canonical_name: str, step_catalog, **kwargs) -> Any:
        """Create appropriate test framework based on detected step type."""
        try:
            # Get step info from catalog
            step_info = step_catalog.get_step_info(canonical_name)
            step_type = step_info.registry_data.get('sagemaker_step_type') if step_info else None
            
            # Import and use specialized frameworks based on step type
            if step_type == "Processing":
                try:
                    from ..variants.processing_test import ProcessingStepBuilderTest
                    return ProcessingStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                except ImportError:
                    pass
            elif step_type == "Training":
                try:
                    from ..variants.training_test import TrainingStepBuilderTest
                    return TrainingStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                except ImportError:
                    pass
            elif step_type == "CreateModel":
                try:
                    from ..variants.createmodel_test import CreateModelStepBuilderTest
                    return CreateModelStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                except ImportError:
                    pass
            elif step_type == "Transform":
                try:
                    from ..variants.transform_test import TransformStepBuilderTest
                    return TransformStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                except ImportError:
                    pass
            
            # Fallback to universal test
            from ..core.universal_test import UniversalStepBuilderTest
            return UniversalStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
                
        except Exception as e:
            print(f"Warning: Could not create specialized framework for {canonical_name} ({step_type}): {e}")
            # Fallback to universal test
            from ..core.universal_test import UniversalStepBuilderTest
            return UniversalStepBuilderTest(builder_class, step_name=canonical_name, **kwargs)
    
    @staticmethod
    def get_available_specialized_frameworks() -> Dict[str, Type]:
        """Get list of available specialized test frameworks."""
        frameworks = {}
        
        # Try to import each specialized framework
        try:
            from ..variants.processing_test import ProcessingStepBuilderTest
            frameworks["Processing"] = ProcessingStepBuilderTest
        except ImportError:
            pass
        
        try:
            from ..variants.training_test import TrainingStepBuilderTest
            frameworks["Training"] = TrainingStepBuilderTest
        except ImportError:
            pass
        
        try:
            from ..variants.createmodel_test import CreateModelStepBuilderTest
            frameworks["CreateModel"] = CreateModelStepBuilderTest
        except ImportError:
            pass
        
        try:
            from ..variants.transform_test import TransformStepBuilderTest
            frameworks["Transform"] = TransformStepBuilderTest
        except ImportError:
            pass
        
        return frameworks
    
    @staticmethod
    def get_framework_for_step_type(step_type: str) -> Type:
        """Get the specialized framework class for a specific step type."""
        available_frameworks = StepTypeTestFrameworkFactory.get_available_specialized_frameworks()
        return available_frameworks.get(step_type)
    
    @staticmethod
    def is_specialized_framework_available(step_type: str) -> bool:
        """Check if a specialized framework is available for the step type."""
        return step_type in StepTypeTestFrameworkFactory.get_available_specialized_frameworks()
    
    @staticmethod
    def get_supported_step_types() -> list:
        """Get list of step types that have specialized frameworks available."""
        return list(StepTypeTestFrameworkFactory.get_available_specialized_frameworks().keys())
