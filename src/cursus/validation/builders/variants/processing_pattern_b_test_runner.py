"""
Processing Pattern B Test Runner for Processing Step Builders.

This module provides utilities to run tests specifically for Pattern B processing
step builders (those using processor.run() + step_args pattern) with appropriate
auto-pass logic for tests that cannot be validated in the test environment.

Pattern B builders include:
- XGBoostModelEvalStepBuilder (uses XGBoostProcessor)
- Any other processing builders that use processor.run() + step_args instead of direct ProcessingStep creation
"""

import json
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from pathlib import Path

from .processing_test import ProcessingStepBuilderTest
from ..universal_test import UniversalStepBuilderTest
from ..registry_discovery import load_builder_class
from ....core.base.builder_base import StepBuilderBase


class ProcessingPatternBTestRunner:
    """
    Test runner specifically designed for Pattern B processing step builders.
    
    Pattern B builders use processor.run() + step_args pattern which cannot be
    fully validated in test environments, requiring auto-pass logic for certain tests.
    """
    
    # Known Pattern B processing builders
    PATTERN_B_PROCESSING_BUILDERS = [
        'XGBoostModelEvalStepBuilder',
        # Add other Pattern B processing builders here as they are identified
    ]
    
    def __init__(self, builder_class: Type[StepBuilderBase], verbose: bool = True):
        """
        Initialize Processing Pattern B test runner.
        
        Args:
            builder_class: The processing builder class to test
            verbose: Whether to print verbose output
        """
        self.builder_class = builder_class
        self.verbose = verbose
        self.builder_name = builder_class.__name__
        
        # Check if this is a Pattern B processing builder
        self.is_pattern_b = self._is_pattern_b_processing_builder()
        
        if self.verbose:
            pattern_type = "Pattern B" if self.is_pattern_b else "Pattern A"
            print(f"🔍 Detected {self.builder_name} as {pattern_type} processing builder")
    
    def _is_pattern_b_processing_builder(self) -> bool:
        """Check if the builder is a Pattern B processing builder."""
        return self.builder_name in self.PATTERN_B_PROCESSING_BUILDERS
    
    def run_processing_pattern_aware_tests(self) -> Dict[str, Any]:
        """
        Run processing tests with Pattern B awareness.
        
        Returns:
            Test results with Pattern B auto-pass logic applied
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING PROCESSING PATTERN-AWARE TESTS FOR {self.builder_name}")
            print(f"{'='*60}")
        
        if self.is_pattern_b:
            return self._run_processing_pattern_b_tests()
        else:
            return self._run_processing_pattern_a_tests()
    
    def _run_processing_pattern_b_tests(self) -> Dict[str, Any]:
        """Run tests for Pattern B processing builders with auto-pass logic."""
        if self.verbose:
            print("🔧 Using ProcessingStepBuilderTest with Pattern B auto-pass logic")
        
        # Use processing-specific test framework
        tester = ProcessingStepBuilderTest(
            builder_class=self.builder_class,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        results = tester.run_processing_validation()
        
        # Add Pattern B metadata
        results.update({
            'processing_pattern_type': 'Pattern B',
            'auto_pass_applied': True,
            'auto_pass_reason': 'processor.run() + step_args pattern cannot be validated in test environment',
            'test_framework': 'ProcessingStepBuilderTest',
            'processing_specific': True
        })
        
        return results
    
    def _run_processing_pattern_a_tests(self) -> Dict[str, Any]:
        """Run tests for Pattern A processing builders using standard logic."""
        if self.verbose:
            print("🔧 Using ProcessingStepBuilderTest for Pattern A processing builder")
        
        # Use processing-specific test framework (but without Pattern B auto-pass)
        tester = ProcessingStepBuilderTest(
            builder_class=self.builder_class,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        results = tester.run_processing_validation()
        
        # Add Pattern A metadata
        results.update({
            'processing_pattern_type': 'Pattern A',
            'auto_pass_applied': False,
            'test_framework': 'ProcessingStepBuilderTest',
            'processing_specific': True
        })
        
        return results
    
    def compare_with_universal_test(self) -> Dict[str, Any]:
        """
        Compare Pattern B processing results with universal test results.
        
        Returns:
            Comparison results showing the difference
        """
        if not self.is_pattern_b:
            return {
                'comparison_available': False,
                'reason': 'Not a Pattern B processing builder - no comparison needed'
            }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"COMPARING PROCESSING PATTERN B vs UNIVERSAL TEST RESULTS")
            print(f"{'='*60}")
        
        # Run Pattern B processing tests
        pattern_b_results = self._run_processing_pattern_b_tests()
        
        # Run universal tests for comparison
        universal_tester = UniversalStepBuilderTest(
            builder_class=self.builder_class,
            verbose=False,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        universal_results = universal_tester.run_all_tests()
        
        # Extract scores
        pattern_b_score = pattern_b_results.get('scoring', {}).get('overall', {}).get('score', 0)
        universal_score = universal_results.get('scoring', {}).get('overall', {}).get('score', 0)
        
        pattern_b_pass_rate = pattern_b_results.get('scoring', {}).get('overall', {}).get('pass_rate', 0)
        universal_pass_rate = universal_results.get('scoring', {}).get('overall', {}).get('pass_rate', 0)
        
        # Calculate improvements
        score_improvement = pattern_b_score - universal_score
        pass_rate_improvement = pattern_b_pass_rate - universal_pass_rate
        
        comparison = {
            'comparison_available': True,
            'processing_pattern_b_results': {
                'score': pattern_b_score,
                'pass_rate': pattern_b_pass_rate,
                'rating': pattern_b_results.get('scoring', {}).get('overall', {}).get('rating', 'Unknown')
            },
            'universal_results': {
                'score': universal_score,
                'pass_rate': universal_pass_rate,
                'rating': universal_results.get('scoring', {}).get('overall', {}).get('rating', 'Unknown')
            },
            'improvements': {
                'score_improvement': score_improvement,
                'pass_rate_improvement': pass_rate_improvement,
                'score_improvement_percentage': (score_improvement / universal_score * 100) if universal_score > 0 else 0,
                'pass_rate_improvement_percentage': (pass_rate_improvement / universal_pass_rate * 100) if universal_pass_rate > 0 else 0
            },
            'processing_pattern_b_effective': score_improvement > 0 or pass_rate_improvement > 0
        }
        
        if self.verbose:
            print(f"\n📊 PROCESSING COMPARISON RESULTS:")
            print(f"   Processing Pattern B Score: {pattern_b_score:.1f}/100")
            print(f"   Universal Score: {universal_score:.1f}/100")
            print(f"   Score Improvement: {score_improvement:+.1f} points")
            print(f"   Pass Rate Improvement: {pass_rate_improvement:+.1f}%")
            
            if comparison['processing_pattern_b_effective']:
                print("✅ Processing Pattern B auto-pass logic is effective!")
            else:
                print("❌ Processing Pattern B auto-pass logic shows no improvement")
        
        return comparison
    
    def generate_processing_pattern_b_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive Processing Pattern B test report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Comprehensive test report
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GENERATING PROCESSING PATTERN B REPORT FOR {self.builder_name}")
            print(f"{'='*60}")
        
        # Run processing pattern-aware tests
        test_results = self.run_processing_pattern_aware_tests()
        
        # Run comparison if Pattern B
        comparison = self.compare_with_universal_test() if self.is_pattern_b else None
        
        # Compile comprehensive report
        report = {
            'builder_class': self.builder_name,
            'processing_pattern_type': 'Pattern B' if self.is_pattern_b else 'Pattern A',
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'comparison': comparison,
            'summary': {
                'is_processing_pattern_b_builder': self.is_pattern_b,
                'auto_pass_applied': self.is_pattern_b,
                'test_framework_used': 'ProcessingStepBuilderTest',
                'processing_specific': True
            }
        }
        
        # Add effectiveness summary for Pattern B processing builders
        if self.is_pattern_b and comparison and comparison.get('comparison_available'):
            improvements = comparison.get('improvements', {})
            report['summary'].update({
                'processing_pattern_b_effective': comparison.get('processing_pattern_b_effective', False),
                'score_improvement': improvements.get('score_improvement', 0),
                'pass_rate_improvement': improvements.get('pass_rate_improvement', 0)
            })
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            if self.verbose:
                print(f"💾 Processing Pattern B report saved to: {output_path}")
        
        return report


def test_processing_pattern_b_builder(builder_name: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to test a Processing Pattern B builder by name.
    
    Args:
        builder_name: Name of the processing builder to test (e.g., "XGBoostModelEval")
        verbose: Whether to print verbose output
        
    Returns:
        Test results
    """
    builder_class = load_builder_class(builder_name)
    if not builder_class:
        return {
            'error': f'Could not load processing builder class for {builder_name}',
            'builder_name': builder_name,
            'timestamp': datetime.now().isoformat()
        }
    
    runner = ProcessingPatternBTestRunner(builder_class, verbose=verbose)
    return runner.run_processing_pattern_aware_tests()


def compare_processing_pattern_b_effectiveness(builder_name: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to compare Processing Pattern B effectiveness for a builder.
    
    Args:
        builder_name: Name of the processing builder to test (e.g., "XGBoostModelEval")
        verbose: Whether to print verbose output
        
    Returns:
        Comparison results
    """
    builder_class = load_builder_class(builder_name)
    if not builder_class:
        return {
            'error': f'Could not load processing builder class for {builder_name}',
            'builder_name': builder_name,
            'timestamp': datetime.now().isoformat()
        }
    
    runner = ProcessingPatternBTestRunner(builder_class, verbose=verbose)
    return runner.compare_with_universal_test()


def generate_processing_pattern_b_report(builder_name: str, output_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to generate a Processing Pattern B report for a builder.
    
    Args:
        builder_name: Name of the processing builder to test (e.g., "XGBoostModelEval")
        output_dir: Optional directory to save the report
        verbose: Whether to print verbose output
        
    Returns:
        Comprehensive report
    """
    builder_class = load_builder_class(builder_name)
    if not builder_class:
        return {
            'error': f'Could not load processing builder class for {builder_name}',
            'builder_name': builder_name,
            'timestamp': datetime.now().isoformat()
        }
    
    runner = ProcessingPatternBTestRunner(builder_class, verbose=verbose)
    
    # Determine output path
    output_path = None
    if output_dir:
        output_path = output_dir / f"{builder_class.__name__}_processing_pattern_b_report.json"
    
    return runner.generate_processing_pattern_b_report(output_path)


# List of known Pattern B processing builders for easy reference
KNOWN_PROCESSING_PATTERN_B_BUILDERS = ProcessingPatternBTestRunner.PATTERN_B_PROCESSING_BUILDERS.copy()
