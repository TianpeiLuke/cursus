"""
Unit tests for cursus.validation.simple_integration module.

Tests the SimpleValidationCoordinator class and public API functions that provide
coordination between Standardization Tester and Alignment Tester with caching,
statistics tracking, and production validation workflows.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import warnings

from cursus.validation.simple_integration import (
    SimpleValidationCoordinator,
    validate_development,
    validate_integration,
    validate_production,
    clear_validation_cache,
    get_validation_statistics,
    validate_step_builder,
    validate_step_integration,
    _coordinator
)


class TestSimpleValidationCoordinator(unittest.TestCase):
    """Test cases for SimpleValidationCoordinator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = SimpleValidationCoordinator()
        
        # Mock builder class
        self.mock_builder = Mock()
        self.mock_builder.__name__ = "TestBuilder"
        
        # Sample validation results
        self.sample_std_results = {
            'passed': True,
            'status': 'success',
            'tests_run': 5,
            'failures': 0,
            'message': 'All tests passed'
        }
        
        self.sample_align_results = {
            'passed': True,
            'status': 'success',
            'alignment_score': 0.95,
            'issues': [],
            'message': 'Alignment validation passed'
        }
    
    def test_init(self):
        """Test coordinator initialization."""
        coordinator = SimpleValidationCoordinator()
        
        self.assertEqual(coordinator.cache, {})
        self.assertEqual(coordinator.stats, {
            'development_validations': 0,
            'integration_validations': 0,
            'production_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        })
    
    @patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest')
    def test_validate_development_success(self, mock_test_class):
        """Test successful development validation."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = self.sample_std_results.copy()
        mock_test_class.return_value = mock_tester
        
        # Run validation
        result = self.coordinator.validate_development(self.mock_builder, test_arg="value")
        
        # Verify results
        self.assertTrue(result['passed'])
        self.assertEqual(result['validation_type'], 'development')
        self.assertEqual(result['tester'], 'standardization')
        self.assertEqual(result['builder_class'], 'TestBuilder')
        self.assertEqual(result['status'], 'success')
        
        # Verify mock calls
        mock_test_class.assert_called_once_with(self.mock_builder, test_arg="value")
        mock_tester.run_all_tests.assert_called_once()
        
        # Verify statistics
        self.assertEqual(self.coordinator.stats['development_validations'], 1)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
        self.assertEqual(self.coordinator.stats['cache_hits'], 0)
    
    @patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest')
    def test_validate_development_error(self, mock_test_class):
        """Test development validation with error."""
        # Setup mock to raise exception
        mock_test_class.side_effect = Exception("Test error")
        
        # Run validation
        result = self.coordinator.validate_development(self.mock_builder)
        
        # Verify error handling
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['validation_type'], 'development')
        self.assertEqual(result['tester'], 'standardization')
        self.assertEqual(result['builder_class'], 'TestBuilder')
        self.assertEqual(result['error'], 'Test error')
        self.assertIn('Development validation failed', result['message'])
    
    @patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest')
    def test_validate_development_caching(self, mock_test_class):
        """Test development validation caching."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = self.sample_std_results.copy()
        mock_test_class.return_value = mock_tester
        
        # First call - should miss cache
        result1 = self.coordinator.validate_development(self.mock_builder)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
        self.assertEqual(self.coordinator.stats['cache_hits'], 0)
        
        # Second call - should hit cache
        result2 = self.coordinator.validate_development(self.mock_builder)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
        self.assertEqual(self.coordinator.stats['cache_hits'], 1)
        
        # Results should be identical
        self.assertEqual(result1, result2)
        
        # Mock should only be called once
        mock_test_class.assert_called_once()
    
    @patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_validate_integration_success(self, mock_tester_class):
        """Test successful integration validation."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_full_validation.return_value = self.sample_align_results.copy()
        mock_tester_class.return_value = mock_tester
        
        script_names = ['script1', 'script2']
        
        # Run validation
        result = self.coordinator.validate_integration(script_names, test_arg="value")
        
        # Verify results
        self.assertTrue(result['passed'])
        self.assertEqual(result['validation_type'], 'integration')
        self.assertEqual(result['tester'], 'alignment')
        self.assertEqual(result['script_names'], script_names)
        self.assertEqual(result['status'], 'success')
        
        # Verify mock calls
        mock_tester_class.assert_called_once()
        mock_tester.run_full_validation.assert_called_once_with(script_names)
        
        # Verify statistics
        self.assertEqual(self.coordinator.stats['integration_validations'], 1)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
    
    @patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_validate_integration_error(self, mock_tester_class):
        """Test integration validation with error."""
        # Setup mock to raise exception
        mock_tester_class.side_effect = Exception("Integration error")
        
        script_names = ['script1']
        
        # Run validation
        result = self.coordinator.validate_integration(script_names)
        
        # Verify error handling
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['validation_type'], 'integration')
        self.assertEqual(result['tester'], 'alignment')
        self.assertEqual(result['script_names'], script_names)
        self.assertEqual(result['error'], 'Integration error')
        self.assertIn('Integration validation failed', result['message'])
    
    @patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_validate_integration_caching(self, mock_tester_class):
        """Test integration validation caching."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_full_validation.return_value = self.sample_align_results.copy()
        mock_tester_class.return_value = mock_tester
        
        script_names = ['script1', 'script2']
        
        # First call - should miss cache
        result1 = self.coordinator.validate_integration(script_names)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
        self.assertEqual(self.coordinator.stats['cache_hits'], 0)
        
        # Second call with same scripts - should hit cache
        result2 = self.coordinator.validate_integration(script_names)
        self.assertEqual(self.coordinator.stats['cache_misses'], 1)
        self.assertEqual(self.coordinator.stats['cache_hits'], 1)
        
        # Results should be identical
        self.assertEqual(result1, result2)
        
        # Different order should still hit cache (sorted)
        result3 = self.coordinator.validate_integration(['script2', 'script1'])
        self.assertEqual(self.coordinator.stats['cache_hits'], 2)
    
    def test_validate_production_both_pass(self):
        """Test production validation when both testers pass."""
        # Mock both validation methods
        std_results = self.sample_std_results.copy()
        align_results = self.sample_align_results.copy()
        
        self.coordinator.validate_development = Mock(return_value=std_results)
        self.coordinator.validate_integration = Mock(return_value=align_results)
        
        # Run production validation
        result = self.coordinator.validate_production(self.mock_builder, 'test_script')
        
        # Verify results
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['validation_type'], 'production')
        self.assertEqual(result['phase'], 'combined')
        self.assertEqual(result['builder_class'], 'TestBuilder')
        self.assertEqual(result['script_name'], 'test_script')
        self.assertTrue(result['both_passed'])
        self.assertTrue(result['standardization_passed'])
        self.assertTrue(result['alignment_passed'])
        self.assertEqual(result['correlation'], 'basic')
        self.assertIn('Production validation passed', result['message'])
        
        # Verify both validations were called
        self.coordinator.validate_development.assert_called_once_with(self.mock_builder)
        self.coordinator.validate_integration.assert_called_once_with(['test_script'])
        
        # Verify statistics
        self.assertEqual(self.coordinator.stats['production_validations'], 1)
    
    def test_validate_production_std_fails(self):
        """Test production validation when standardization fails."""
        # Mock standardization failure
        std_results = {'passed': False, 'status': 'failed', 'message': 'Standard failed'}
        
        self.coordinator.validate_development = Mock(return_value=std_results)
        self.coordinator.validate_integration = Mock()
        
        # Run production validation
        result = self.coordinator.validate_production(self.mock_builder, 'test_script')
        
        # Verify fail-fast behavior
        self.assertEqual(result['status'], 'failed_standardization')
        self.assertEqual(result['validation_type'], 'production')
        self.assertEqual(result['phase'], 'standardization')
        self.assertFalse(result['both_passed'])
        self.assertEqual(result['standardization_results'], std_results)
        self.assertIsNone(result['alignment_results'])
        self.assertIn('Fix implementation issues', result['message'])
        
        # Integration should not be called due to fail-fast
        self.coordinator.validate_integration.assert_not_called()
    
    def test_validate_production_integration_fails(self):
        """Test production validation when integration fails."""
        # Mock standardization pass, integration fail
        std_results = self.sample_std_results.copy()
        align_results = {'passed': False, 'status': 'failed', 'message': 'Alignment failed'}
        
        self.coordinator.validate_development = Mock(return_value=std_results)
        self.coordinator.validate_integration = Mock(return_value=align_results)
        
        # Run production validation
        result = self.coordinator.validate_production(self.mock_builder, 'test_script')
        
        # Verify results
        self.assertEqual(result['status'], 'failed_integration')
        self.assertFalse(result['both_passed'])
        self.assertTrue(result['standardization_passed'])
        self.assertFalse(result['alignment_passed'])
        self.assertIn('Implementation quality validated but integration issues', result['message'])
    
    def test_validate_production_error(self):
        """Test production validation with error."""
        # Mock error in development validation
        self.coordinator.validate_development = Mock(side_effect=Exception("Production error"))
        
        # Run production validation
        result = self.coordinator.validate_production(self.mock_builder, 'test_script')
        
        # Verify error handling
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['validation_type'], 'production')
        self.assertEqual(result['phase'], 'error')
        self.assertEqual(result['builder_class'], 'TestBuilder')
        self.assertEqual(result['script_name'], 'test_script')
        self.assertEqual(result['error'], 'Production error')
        self.assertIn('Production validation error', result['message'])
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some cache entries
        self.coordinator.cache['test1'] = {'result': 'data1'}
        self.coordinator.cache['test2'] = {'result': 'data2'}
        
        self.assertEqual(len(self.coordinator.cache), 2)
        
        # Clear cache
        self.coordinator.clear_cache()
        
        self.assertEqual(len(self.coordinator.cache), 0)
        self.assertEqual(self.coordinator.cache, {})
    
    def test_get_statistics_empty(self):
        """Test statistics with no validations."""
        stats = self.coordinator.get_statistics()
        
        expected = {
            'total_validations': 0,
            'development_validations': 0,
            'integration_validations': 0,
            'production_validations': 0,
            'cache_hit_rate_percentage': 0.0,
            'cache_size': 0
        }
        
        self.assertEqual(stats, expected)
    
    def test_get_statistics_with_data(self):
        """Test statistics with validation data."""
        # Simulate some validations
        self.coordinator.stats.update({
            'development_validations': 5,
            'integration_validations': 3,
            'production_validations': 2,
            'cache_hits': 7,
            'cache_misses': 3
        })
        
        # Add cache entries
        self.coordinator.cache['test1'] = {}
        self.coordinator.cache['test2'] = {}
        
        stats = self.coordinator.get_statistics()
        
        self.assertEqual(stats['total_validations'], 10)  # 5 + 3 + 2
        self.assertEqual(stats['development_validations'], 5)
        self.assertEqual(stats['integration_validations'], 3)
        self.assertEqual(stats['production_validations'], 2)
        self.assertEqual(stats['cache_hit_rate_percentage'], 70.0)  # 7/(7+3) * 100
        self.assertEqual(stats['cache_size'], 2)


class TestPublicAPIFunctions(unittest.TestCase):
    """Test cases for public API functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear global coordinator state
        _coordinator.clear_cache()
        _coordinator.stats = {
            'development_validations': 0,
            'integration_validations': 0,
            'production_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.mock_builder = Mock()
        self.mock_builder.__name__ = "TestBuilder"
    
    @patch.object(_coordinator, 'validate_development')
    def test_validate_development_function(self, mock_method):
        """Test validate_development public function."""
        expected_result = {'passed': True, 'message': 'Success'}
        mock_method.return_value = expected_result
        
        result = validate_development(self.mock_builder, test_arg="value")
        
        self.assertEqual(result, expected_result)
        mock_method.assert_called_once_with(self.mock_builder, test_arg="value")
    
    @patch.object(_coordinator, 'validate_integration')
    def test_validate_integration_function(self, mock_method):
        """Test validate_integration public function."""
        expected_result = {'passed': True, 'message': 'Success'}
        mock_method.return_value = expected_result
        
        script_names = ['script1', 'script2']
        result = validate_integration(script_names, test_arg="value")
        
        self.assertEqual(result, expected_result)
        mock_method.assert_called_once_with(script_names, test_arg="value")
    
    @patch.object(_coordinator, 'validate_production')
    def test_validate_production_function(self, mock_method):
        """Test validate_production public function."""
        expected_result = {'both_passed': True, 'status': 'passed'}
        mock_method.return_value = expected_result
        
        result = validate_production(self.mock_builder, 'test_script', test_arg="value")
        
        self.assertEqual(result, expected_result)
        mock_method.assert_called_once_with(self.mock_builder, 'test_script', test_arg="value")
    
    @patch.object(_coordinator, 'clear_cache')
    def test_clear_validation_cache_function(self, mock_method):
        """Test clear_validation_cache public function."""
        clear_validation_cache()
        mock_method.assert_called_once()
    
    @patch.object(_coordinator, 'get_statistics')
    def test_get_validation_statistics_function(self, mock_method):
        """Test get_validation_statistics public function."""
        expected_stats = {'total_validations': 5, 'cache_hit_rate_percentage': 80.0}
        mock_method.return_value = expected_stats
        
        result = get_validation_statistics()
        
        self.assertEqual(result, expected_stats)
        mock_method.assert_called_once()


class TestLegacyCompatibilityFunctions(unittest.TestCase):
    """Test cases for legacy compatibility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_builder = Mock()
        self.mock_builder.__name__ = "TestBuilder"
    
    @patch('cursus.validation.simple_integration.validate_development')
    def test_validate_step_builder_deprecation(self, mock_validate_dev):
        """Test validate_step_builder shows deprecation warning."""
        expected_result = {'passed': True}
        mock_validate_dev.return_value = expected_result
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = validate_step_builder(self.mock_builder, test_arg="value")
            
            # Check deprecation warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("validate_step_builder() is deprecated", str(w[0].message))
            self.assertIn("Use validate_development() instead", str(w[0].message))
            
            # Check function still works
            self.assertEqual(result, expected_result)
            mock_validate_dev.assert_called_once_with(self.mock_builder, test_arg="value")
    
    @patch('cursus.validation.simple_integration.validate_integration')
    def test_validate_step_integration_deprecation(self, mock_validate_int):
        """Test validate_step_integration shows deprecation warning."""
        expected_result = {'passed': True}
        mock_validate_int.return_value = expected_result
        
        script_names = ['script1', 'script2']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = validate_step_integration(script_names, test_arg="value")
            
            # Check deprecation warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("validate_step_integration() is deprecated", str(w[0].message))
            self.assertIn("Use validate_integration() instead", str(w[0].message))
            
            # Check function still works
            self.assertEqual(result, expected_result)
            mock_validate_int.assert_called_once_with(script_names, test_arg="value")


class TestIntegrationScenarios(unittest.TestCase):
    """Test cases for integration scenarios and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = SimpleValidationCoordinator()
        self.mock_builder = Mock()
        self.mock_builder.__name__ = "IntegrationTestBuilder"
    
    def test_cache_key_generation(self):
        """Test cache key generation for different scenarios."""
        coordinator = SimpleValidationCoordinator()
        
        # Test development cache key
        with patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_test:
            mock_tester = Mock()
            mock_tester.run_all_tests.return_value = {'passed': True}
            mock_test.return_value = mock_tester
            
            # Same builder should use same cache key
            coordinator.validate_development(self.mock_builder)
            coordinator.validate_development(self.mock_builder)
            
            # Should only call once due to caching
            self.assertEqual(mock_test.call_count, 1)
        
        # Test integration cache key with script order
        with patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester') as mock_align:
            mock_tester = Mock()
            mock_tester.run_full_validation.return_value = {'passed': True}
            mock_align.return_value = mock_tester
            
            # Different order should use same cache (sorted)
            coordinator.validate_integration(['script2', 'script1'])
            coordinator.validate_integration(['script1', 'script2'])
            
            # Should only call once due to caching
            self.assertEqual(mock_align.call_count, 1)
    
    def test_statistics_accuracy(self):
        """Test statistics tracking accuracy."""
        coordinator = SimpleValidationCoordinator()
        
        # Mock the validation methods
        with patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest'), \
             patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester'):
            
            # Run various validations
            coordinator.validate_development(self.mock_builder)  # cache miss
            coordinator.validate_development(self.mock_builder)  # cache hit
            coordinator.validate_integration(['script1'])        # cache miss
            coordinator.validate_integration(['script1'])        # cache hit
            
            stats = coordinator.get_statistics()
            
            self.assertEqual(stats['development_validations'], 1)
            self.assertEqual(stats['integration_validations'], 1)
            self.assertEqual(stats['cache_hit_rate_percentage'], 50.0)  # 2 hits out of 4 total
    
    def test_error_resilience(self):
        """Test error handling and resilience."""
        coordinator = SimpleValidationCoordinator()
        
        # Test with various error types
        with patch('cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_test:
            # Import error
            mock_test.side_effect = ImportError("Module not found")
            result = coordinator.validate_development(self.mock_builder)
            self.assertEqual(result['status'], 'error')
            self.assertIn('Module not found', result['error'])
            
            # Runtime error
            mock_test.side_effect = RuntimeError("Runtime issue")
            result = coordinator.validate_development(self.mock_builder)
            self.assertEqual(result['status'], 'error')
            self.assertIn('Runtime issue', result['error'])
    
    def test_production_validation_workflow(self):
        """Test complete production validation workflow."""
        coordinator = SimpleValidationCoordinator()
        
        # Mock successful standardization, failed integration
        std_results = {'passed': True, 'status': 'success'}
        align_results = {'passed': False, 'status': 'failed', 'issues': ['alignment issue']}
        
        coordinator.validate_development = Mock(return_value=std_results)
        coordinator.validate_integration = Mock(return_value=align_results)
        
        result = coordinator.validate_production(self.mock_builder, 'test_script')
        
        # Should proceed through both phases
        self.assertEqual(result['status'], 'failed_integration')
        self.assertTrue(result['standardization_passed'])
        self.assertFalse(result['alignment_passed'])
        self.assertFalse(result['both_passed'])
        
        # Both methods should be called
        coordinator.validate_development.assert_called_once()
        coordinator.validate_integration.assert_called_once()
    
    def test_empty_script_names(self):
        """Test handling of empty script names list."""
        coordinator = SimpleValidationCoordinator()
        
        with patch('cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester') as mock_align:
            mock_tester = Mock()
            mock_tester.run_full_validation.return_value = {'passed': True}
            mock_align.return_value = mock_tester
            
            result = coordinator.validate_integration([])
            
            self.assertEqual(result['script_names'], [])
            mock_tester.run_full_validation.assert_called_once_with([])
    
    def test_large_cache_behavior(self):
        """Test behavior with large cache."""
        coordinator = SimpleValidationCoordinator()
        
        # Simulate large cache
        for i in range(100):
            coordinator.cache[f'test_key_{i}'] = {'result': f'data_{i}'}
        
        stats = coordinator.get_statistics()
        self.assertEqual(stats['cache_size'], 100)
        
        # Clear and verify
        coordinator.clear_cache()
        stats = coordinator.get_statistics()
        self.assertEqual(stats['cache_size'], 0)


if __name__ == '__main__':
    unittest.main()
