"""
Simplified Cursus Validation Framework

This module provides essential validation capabilities through a clean 3-function API,
implementing the simplified integration strategy from the Validation System Complexity Analysis.

Based on the Simplified Universal Step Builder Test Enhancement Plan.
"""

# Import the simplified 3-function API
from .simple_integration import (
    validate_development,
    validate_integration,
    validate_production,
    clear_validation_cache,
    get_validation_statistics,
    # Legacy functions with deprecation warnings
    validate_step_builder,
    validate_step_integration
)

# Import simplified runtime testing components
from .runtime import (
    RuntimeTester,
    ScriptTestResult,
    DataCompatibilityResult
)

# Framework information
__approach__ = "Simplified Integration"
__complexity_reduction__ = "67% integration complexity reduction achieved"

# Export only essential functions
__all__ = [
    # Core 3-function API
    'validate_development',
    'validate_integration', 
    'validate_production',
    
    # Utility functions
    'clear_validation_cache',
    'get_validation_statistics',
    
    # Legacy functions (deprecated)
    'validate_step_builder',
    'validate_step_integration',
    
    # Runtime testing components
    'RuntimeTester',
    'ScriptTestResult',
    'DataCompatibilityResult'
]


# Convenience functions with clear documentation
def validate_builder_development(builder_class: type, **kwargs) -> dict:
    """
    Validate step builder for development workflow.
    
    This is the primary function for development-time validation,
    focusing on implementation quality and standardization compliance.
    
    Args:
        builder_class: Step builder class to validate
        **kwargs: Additional validation arguments
        
    Returns:
        Validation results dictionary
        
    Example:
        >>> from cursus.validation import validate_builder_development
        >>> from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        >>> results = validate_builder_development(TabularPreprocessingStepBuilder)
        >>> print(f"Validation {'passed' if results['passed'] else 'failed'}")
    """
    return validate_development(builder_class, **kwargs)


def validate_script_integration(script_names: list, **kwargs) -> dict:
    """
    Validate script integration for integration workflow.
    
    This is the primary function for integration-time validation,
    focusing on component alignment and cross-layer integration.
    
    Args:
        script_names: List of script names to validate
        **kwargs: Additional validation arguments
        
    Returns:
        Validation results dictionary
        
    Example:
        >>> from cursus.validation import validate_script_integration
        >>> results = validate_script_integration(['tabular_preprocessing'])
        >>> print(f"Integration {'passed' if results['passed'] else 'failed'}")
    """
    return validate_integration(script_names, **kwargs)


def validate_production_readiness(builder_class: type, script_name: str, **kwargs) -> dict:
    """
    Validate production readiness with comprehensive validation.
    
    This is the primary function for production-readiness validation,
    combining both standardization and alignment validation with
    basic correlation analysis.
    
    Args:
        builder_class: Step builder class to validate
        script_name: Associated script name
        **kwargs: Additional validation arguments
        
    Returns:
        Combined validation result with basic correlation
        
    Example:
        >>> from cursus.validation import validate_production_readiness
        >>> from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        >>> result = validate_production_readiness(TabularPreprocessingStepBuilder, 'tabular_preprocessing')
        >>> print(f"Production validation: {result['status']}")
        >>> print(f"Both testers passed: {result['both_passed']}")
    """
    return validate_production(builder_class, script_name, **kwargs)


def get_validation_framework_info() -> dict:
    """
    Get information about the validation framework.
    
    Returns:
        Dictionary with framework information and statistics
    """
    stats = get_validation_statistics()
    
    return {
        'approach': __approach__,
        'complexity_reduction': __complexity_reduction__,
        'api_functions': len(__all__),
        'core_functions': 3,  # validate_development, validate_integration, validate_production
        'statistics': stats,
        'description': 'Simplified validation framework with 3-function API'
    }


# Add convenience aliases for common usage patterns
validate_dev = validate_development
validate_int = validate_integration
validate_prod = validate_production
