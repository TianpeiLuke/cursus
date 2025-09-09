"""
Simple Validation Integration

This module provides essential coordination between Standardization Tester and
Alignment Tester with minimal complexity, implementing the simplified integration
strategy from the Validation System Complexity Analysis.

Based on the Simplified Universal Step Builder Test Enhancement Plan Phase 2.
"""

from typing import Dict, Any, List, Optional


class SimpleValidationCoordinator:
    """
    Minimal coordination between both testers.
    
    This class provides simple coordination between the Standardization Tester
    (Universal Step Builder Test) and Alignment Tester (Unified Alignment Tester)
    without the complexity overhead of rich orchestration patterns.
    """
    
    def __init__(self):
        """Initialize with simple result caching."""
        self.cache = {}  # Simple result caching for performance
        self.stats = {
            'development_validations': 0,
            'integration_validations': 0,
            'production_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def validate_development(self, builder_class: type, **kwargs) -> Dict[str, Any]:
        """
        Development validation using Standardization Tester.
        
        Focuses on implementation quality and step builder pattern compliance.
        
        Args:
            builder_class: Step builder class to validate
            **kwargs: Additional validation arguments
            
        Returns:
            Validation results from standardization tester
        """
        # Cache key for performance
        cache_key = f"dev_{builder_class.__name__}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        self.stats['cache_misses'] += 1
        self.stats['development_validations'] += 1
        
        try:
            # Run standardization validation
            from .builders.universal_test import UniversalStepBuilderTest
            tester = UniversalStepBuilderTest(builder_class, **kwargs)
            results = tester.run_all_tests()
            
            # Add context for clarity
            results['validation_type'] = 'development'
            results['tester'] = 'standardization'
            results['builder_class'] = builder_class.__name__
            
            # Cache results
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            return {
                'validation_type': 'development',
                'tester': 'standardization',
                'builder_class': builder_class.__name__,
                'status': 'error',
                'passed': False,
                'error': str(e),
                'message': f"Development validation failed: {str(e)}"
            }
    
    def validate_integration(self, script_names: List[str], **kwargs) -> Dict[str, Any]:
        """
        Integration validation using Alignment Tester.
        
        Focuses on component alignment and cross-layer integration.
        
        Args:
            script_names: List of script names to validate
            **kwargs: Additional validation arguments
            
        Returns:
            Validation results from alignment tester
        """
        # Cache key for performance
        cache_key = f"int_{'_'.join(sorted(script_names))}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        self.stats['cache_misses'] += 1
        self.stats['integration_validations'] += 1
        
        try:
            # Run alignment validation
            from .alignment.unified_alignment_tester import UnifiedAlignmentTester
            tester = UnifiedAlignmentTester()
            results = tester.run_full_validation(script_names)
            
            # Add context for clarity
            results['validation_type'] = 'integration'
            results['tester'] = 'alignment'
            results['script_names'] = script_names
            
            # Cache results
            self.cache[cache_key] = results
            return results
            
        except Exception as e:
            return {
                'validation_type': 'integration',
                'tester': 'alignment',
                'script_names': script_names,
                'status': 'error',
                'passed': False,
                'error': str(e),
                'message': f"Integration validation failed: {str(e)}"
            }
    
    def validate_production(self, builder_class: type, script_name: str, **kwargs) -> Dict[str, Any]:
        """
        Production validation using both testers with basic correlation.
        
        Runs both standardization and alignment validation with fail-fast approach
        and basic pass/fail correlation.
        
        Args:
            builder_class: Step builder class to validate
            script_name: Associated script name
            **kwargs: Additional validation arguments
            
        Returns:
            Combined validation results with basic correlation
        """
        self.stats['production_validations'] += 1
        
        try:
            # Step 1: Standardization validation (fail-fast)
            std_results = self.validate_development(builder_class, **kwargs)
            
            # Step 2: Check if standardization passes
            std_passed = std_results.get('passed', False)
            if not std_passed:
                return {
                    'status': 'failed_standardization',
                    'validation_type': 'production',
                    'phase': 'standardization',
                    'builder_class': builder_class.__name__,
                    'script_name': script_name,
                    'standardization_results': std_results,
                    'alignment_results': None,
                    'both_passed': False,
                    'message': 'Fix implementation issues before integration testing'
                }
            
            # Step 3: Integration validation
            align_results = self.validate_integration([script_name], **kwargs)
            
            # Step 4: Basic correlation (simple pass/fail)
            align_passed = align_results.get('passed', False)
            both_passed = std_passed and align_passed
            
            # Determine overall status
            if both_passed:
                status = 'passed'
                message = 'Production validation passed - both implementation and integration validated'
            elif std_passed:
                status = 'failed_integration'
                message = 'Implementation quality validated but integration issues found'
            else:
                status = 'failed_both'
                message = 'Both implementation and integration issues found'
            
            return {
                'status': status,
                'validation_type': 'production',
                'phase': 'combined',
                'builder_class': builder_class.__name__,
                'script_name': script_name,
                'standardization_results': std_results,
                'alignment_results': align_results,
                'both_passed': both_passed,
                'standardization_passed': std_passed,
                'alignment_passed': align_passed,
                'correlation': 'basic',
                'message': message
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'validation_type': 'production',
                'phase': 'error',
                'builder_class': builder_class.__name__,
                'script_name': script_name,
                'error': str(e),
                'message': f"Production validation error: {str(e)}"
            }
    
    def clear_cache(self) -> None:
        """Clear validation result cache."""
        self.cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simple validation statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'total_validations': (
                self.stats['development_validations'] + 
                self.stats['integration_validations'] + 
                self.stats['production_validations']
            ),
            'development_validations': self.stats['development_validations'],
            'integration_validations': self.stats['integration_validations'],
            'production_validations': self.stats['production_validations'],
            'cache_hit_rate_percentage': hit_rate,
            'cache_size': len(self.cache)
        }


# Global coordinator instance for simple access
_coordinator = SimpleValidationCoordinator()


# Public API functions - the core 3-function simplified interface
def validate_development(builder_class: type, **kwargs) -> Dict[str, Any]:
    """
    Validate step builder implementation quality.
    
    Uses the Standardization Tester (Universal Step Builder Test) to validate
    that step builder implementations follow correct patterns and standards.
    
    Args:
        builder_class: Step builder class to validate
        **kwargs: Additional validation arguments
        
    Returns:
        Validation results from standardization tester
        
    Example:
        >>> from cursus.validation import validate_development
        >>> from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        >>> results = validate_development(TabularPreprocessingStepBuilder)
        >>> print(f"Development validation {'passed' if results['passed'] else 'failed'}")
    """
    return _coordinator.validate_development(builder_class, **kwargs)


def validate_integration(script_names: List[str], **kwargs) -> Dict[str, Any]:
    """
    Validate component integration and alignment.
    
    Uses the Alignment Tester (Unified Alignment Tester) to validate that
    components align properly across the four-tier architecture.
    
    Args:
        script_names: List of script names to validate
        **kwargs: Additional validation arguments
        
    Returns:
        Validation results from alignment tester
        
    Example:
        >>> from cursus.validation import validate_integration
        >>> results = validate_integration(['tabular_preprocessing'])
        >>> print(f"Integration validation {'passed' if results['passed'] else 'failed'}")
    """
    return _coordinator.validate_integration(script_names, **kwargs)


def validate_production(builder_class: type, script_name: str, **kwargs) -> Dict[str, Any]:
    """
    Validate production readiness with both testers.
    
    Runs both standardization and alignment validation with fail-fast approach
    and basic correlation analysis.
    
    Args:
        builder_class: Step builder class to validate
        script_name: Associated script name
        **kwargs: Additional validation arguments
        
    Returns:
        Combined validation results with basic correlation
        
    Example:
        >>> from cursus.validation import validate_production
        >>> from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        >>> results = validate_production(TabularPreprocessingStepBuilder, 'tabular_preprocessing')
        >>> print(f"Production validation: {results['status']}")
        >>> print(f"Both testers passed: {results['both_passed']}")
    """
    return _coordinator.validate_production(builder_class, script_name, **kwargs)


# Utility functions
def clear_validation_cache() -> None:
    """Clear validation result cache for fresh validation runs."""
    _coordinator.clear_cache()


def get_validation_statistics() -> Dict[str, Any]:
    """Get simple validation statistics."""
    return _coordinator.get_statistics()


# Legacy compatibility functions with deprecation warnings
def validate_step_builder(builder_class: type, **kwargs) -> Dict[str, Any]:
    """
    Legacy function - use validate_development() instead.
    
    Args:
        builder_class: Step builder class to validate
        **kwargs: Additional arguments
        
    Returns:
        Validation results
    """
    import warnings
    warnings.warn(
        "validate_step_builder() is deprecated. Use validate_development() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return validate_development(builder_class, **kwargs)


def validate_step_integration(script_names: List[str], **kwargs) -> Dict[str, Any]:
    """
    Legacy function - use validate_integration() instead.
    
    Args:
        script_names: List of script names to validate
        **kwargs: Additional arguments
        
    Returns:
        Validation results
    """
    import warnings
    warnings.warn(
        "validate_step_integration() is deprecated. Use validate_integration() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return validate_integration(script_names, **kwargs)
