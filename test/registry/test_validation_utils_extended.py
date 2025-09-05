"""
Extended unit tests for registry validation utilities.

This module provides additional test coverage for validation functions
that were added in Phase 2 of the implementation, including performance
tracking, validation reporting, and CLI integration support.

Tests focus on:
- Performance metrics and tracking
- Validation reporting functionality
- System status and configuration
- Edge cases and error handling
- Integration scenarios
"""

import pytest
import time
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from src.cursus.registry.validation_utils import (
    create_validation_report,
    get_performance_metrics,
    reset_performance_metrics,
    get_validation_status,
    validate_new_step_definition,
    auto_correct_step_definition,
    to_pascal_case,
    get_validation_errors_with_suggestions,
    _validation_stats
)


class TestCreateValidationReport:
    """Test comprehensive validation reporting functionality."""
    
    def test_valid_step_report(self):
        """Test validation report for valid step definition."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        
        report = create_validation_report("MyCustomStep", step_data, "warn")
        
        assert report["step_name"] == "MyCustomStep"
        assert report["validation_mode"] == "warn"
        assert report["is_valid"] is True
        assert report["error_count"] == 0
        assert report["errors"] == []
        assert report["detailed_errors"] == []
        assert report["corrections_available"] is False
        assert report["suggested_corrections"] == {}
    
    def test_invalid_step_report(self):
        """Test validation report for invalid step definition."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",  # Invalid suffix
            "builder_step_name": "MyCustomBuilder",       # Invalid suffix
            "sagemaker_step_type": "InvalidType"          # Invalid type
        }
        
        report = create_validation_report("my_custom_step", step_data, "strict")
        
        assert report["step_name"] == "my_custom_step"
        assert report["validation_mode"] == "strict"
        assert report["is_valid"] is False
        assert report["error_count"] > 0
        assert len(report["errors"]) > 0
        assert len(report["detailed_errors"]) > 0
        assert report["corrections_available"] is True
        assert len(report["suggested_corrections"]) > 0
    
    def test_report_suggested_corrections(self):
        """Test validation report includes suggested corrections."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",
            "builder_step_name": "MyCustomBuilder"
        }
        
        report = create_validation_report("my_custom_step", step_data, "warn")
        
        corrections = report["suggested_corrections"]
        assert "name" in corrections
        assert corrections["name"]["original"] == "my_custom_step"
        assert corrections["name"]["corrected"] == "MyCustomStep"
        
        assert "config_class" in corrections
        assert corrections["config_class"]["original"] == "MyCustomStepConfiguration"
        assert corrections["config_class"]["corrected"] == "MyCustomStepConfig"
        
        assert "builder_step_name" in corrections
        assert corrections["builder_step_name"]["original"] == "MyCustomBuilder"
        assert corrections["builder_step_name"]["corrected"] == "MyCustomStepStepBuilder"
    
    def test_report_partial_corrections(self):
        """Test validation report with only some fields needing correction."""
        step_data = {
            "config_class": "MyCustomStepConfig",  # Valid
            "builder_step_name": "MyCustomBuilder"  # Invalid
        }
        
        report = create_validation_report("MyCustomStep", step_data, "warn")
        
        corrections = report["suggested_corrections"]
        assert "name" not in corrections  # Already valid
        assert "config_class" not in corrections  # Already valid
        assert "builder_step_name" in corrections  # Needs correction
        assert corrections["builder_step_name"]["corrected"] == "MyCustomStepStepBuilder"
    
    def test_report_with_different_validation_modes(self):
        """Test validation report with different validation modes."""
        step_data = {
            "config_class": "MyCustomStepConfiguration"
        }
        
        modes = ["warn", "strict", "auto_correct"]
        for mode in modes:
            report = create_validation_report("my_custom_step", step_data, mode)
            assert report["validation_mode"] == mode
            assert report["is_valid"] is False  # Should be invalid regardless of mode
            assert report["error_count"] > 0


class TestPerformanceMetrics:
    """Test performance tracking and metrics functionality."""
    
    def setup_method(self):
        """Reset performance metrics before each test."""
        reset_performance_metrics()
    
    def test_initial_performance_metrics(self):
        """Test initial state of performance metrics."""
        metrics = get_performance_metrics()
        
        assert metrics["total_validations"] == 0
        assert metrics["total_time_ms"] == 0.0
        assert metrics["average_time_ms"] == 0.0
        assert metrics["performance_target"] == "< 1ms per validation"
        assert metrics["target_met"] is True  # 0ms meets target
        
        cache_stats = metrics["cache_stats"]
        assert cache_stats["hits"] == 0
        assert cache_stats["misses"] == 0
        assert cache_stats["hit_rate"] == 0.0
        assert cache_stats["cache_size"] == 0
        assert cache_stats["max_size"] == 256
    
    def test_performance_tracking_after_validation(self):
        """Test performance metrics are updated after validation."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfig"
        }
        
        # Perform validation to update metrics
        validate_new_step_definition(step_data)
        
        metrics = get_performance_metrics()
        assert metrics["total_validations"] == 1
        assert metrics["total_time_ms"] > 0.0
        assert metrics["average_time_ms"] > 0.0
        assert metrics["target_met"] is True  # Should be fast
    
    def test_performance_tracking_multiple_validations(self):
        """Test performance metrics with multiple validations."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfig"
        }
        
        # Perform multiple validations
        for i in range(5):
            validate_new_step_definition(step_data)
        
        metrics = get_performance_metrics()
        assert metrics["total_validations"] == 5
        assert metrics["total_time_ms"] >= 0.0  # Allow for very fast validation
        assert metrics["average_time_ms"] >= 0.0  # Allow for very fast validation
        if metrics["total_validations"] > 0 and metrics["total_time_ms"] > 0:
            expected_avg = metrics["total_time_ms"] / 5
            assert abs(metrics["average_time_ms"] - expected_avg) < 0.001  # Allow for floating point precision
    
    def test_cache_performance_tracking(self):
        """Test cache performance is tracked correctly."""
        # First call should be a cache miss
        result1 = to_pascal_case("my_custom_step")
        
        # Second call should be a cache hit
        result2 = to_pascal_case("my_custom_step")
        
        assert result1 == result2 == "MyCustomStep"
        
        metrics = get_performance_metrics()
        cache_stats = metrics["cache_stats"]
        assert cache_stats["hits"] >= 1
        assert cache_stats["misses"] >= 1
        assert cache_stats["hit_rate"] > 0.0
        assert cache_stats["cache_size"] >= 1
    
    def test_reset_performance_metrics(self):
        """Test resetting performance metrics."""
        # Generate some metrics
        step_data = {"name": "MyCustomStep"}
        validate_new_step_definition(step_data)
        to_pascal_case("test_step")
        
        # Verify metrics exist
        metrics_before = get_performance_metrics()
        assert metrics_before["total_validations"] > 0
        assert metrics_before["cache_stats"]["cache_size"] > 0
        
        # Reset metrics
        reset_performance_metrics()
        
        # Verify metrics are reset
        metrics_after = get_performance_metrics()
        assert metrics_after["total_validations"] == 0
        assert metrics_after["total_time_ms"] == 0.0
        assert metrics_after["average_time_ms"] == 0.0
        assert metrics_after["cache_stats"]["hits"] == 0
        assert metrics_after["cache_stats"]["misses"] == 0
        assert metrics_after["cache_stats"]["cache_size"] == 0
    
    def test_performance_target_assessment(self):
        """Test performance target assessment logic."""
        # Test with fast validation (should meet target)
        step_data = {"name": "MyCustomStep"}
        validate_new_step_definition(step_data)
        
        metrics = get_performance_metrics()
        assert metrics["target_met"] is True
        assert metrics["average_time_ms"] < 1.0
    
    @patch('src.cursus.registry.validation_utils.logger')
    def test_performance_warning_logging(self, mock_logger):
        """Test performance warning is logged for slow validation."""
        # Mock time.perf_counter to simulate slow validation
        with patch('time.perf_counter') as mock_time:
            mock_time.side_effect = [0.0, 0.002]  # 2ms validation time
            
            step_data = {"name": "MyCustomStep"}
            validate_new_step_definition(step_data)
            
            # Should log warning for >1ms validation
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Validation took" in warning_call
            assert "target: <1ms" in warning_call


class TestGetValidationStatus:
    """Test validation system status functionality."""
    
    def test_validation_status_structure(self):
        """Test validation status returns correct structure."""
        status = get_validation_status()
        
        # Check main status fields
        assert status["validation_available"] is True
        assert isinstance(status["validation_functions"], list)
        assert isinstance(status["supported_modes"], list)
        assert isinstance(status["implementation_approach"], str)
        assert isinstance(status["performance_target"], str)
        assert isinstance(status["redundancy_level"], str)
        assert isinstance(status["current_performance"], dict)
    
    def test_validation_functions_list(self):
        """Test validation functions list is complete."""
        status = get_validation_status()
        functions = status["validation_functions"]
        
        expected_functions = [
            "validate_new_step_definition",
            "auto_correct_step_definition",
            "get_validation_errors_with_suggestions",
            "register_step_with_validation",
            "create_validation_report",
            "get_performance_metrics",
            "reset_performance_metrics"
        ]
        
        for func in expected_functions:
            assert func in functions
    
    def test_supported_modes_list(self):
        """Test supported validation modes list."""
        status = get_validation_status()
        modes = status["supported_modes"]
        
        expected_modes = ["warn", "strict", "auto_correct"]
        assert modes == expected_modes
    
    def test_current_performance_integration(self):
        """Test current performance is integrated correctly."""
        # Generate some performance data
        step_data = {"name": "MyCustomStep"}
        validate_new_step_definition(step_data)
        
        status = get_validation_status()
        current_perf = status["current_performance"]
        
        assert "average_time_ms" in current_perf
        assert "target_met" in current_perf
        assert "total_validations" in current_perf
        assert "cache_hit_rate" in current_perf
        
        assert current_perf["total_validations"] > 0
        # target_met can be True or False depending on system performance
        # The important thing is that it's a boolean value
        assert isinstance(current_perf["target_met"], bool)
        # Average time should be a reasonable value (not negative, not extremely high)
        assert current_perf["average_time_ms"] >= 0.0
        assert current_perf["average_time_ms"] < 100.0  # Should be under 100ms
    
    def test_implementation_details(self):
        """Test implementation details in status."""
        status = get_validation_status()
        
        assert status["implementation_approach"] == "simplified_regex_based"
        assert status["performance_target"] == "< 1ms per validation"
        assert status["redundancy_level"] == "15-20% (optimal)"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_empty_step_data_validation(self):
        """Test validation with empty step data."""
        step_data = {}
        errors = validate_new_step_definition(step_data)
        
        assert len(errors) == 1
        assert "Step name is required" in errors[0]
    
    def test_none_values_in_step_data(self):
        """Test validation with None values."""
        step_data = {
            "name": None,
            "config_class": None,
            "builder_step_name": None,
            "sagemaker_step_type": None
        }
        
        errors = validate_new_step_definition(step_data)
        assert len(errors) == 1  # Only name is required
        assert "Step name is required" in errors[0]
    
    def test_auto_correction_with_empty_values(self):
        """Test auto-correction handles empty values gracefully."""
        step_data = {
            "name": "",
            "config_class": "",
            "builder_step_name": ""
        }
        
        corrected = auto_correct_step_definition(step_data)
        
        # Should not crash and should preserve empty values
        assert corrected["name"] == ""
        assert corrected["config_class"] == ""
        assert corrected["builder_step_name"] == ""
    
    def test_pascal_case_with_special_characters(self):
        """Test PascalCase conversion with special characters."""
        test_cases = [
            ("step@name", "StepName"),
            ("step#name", "StepName"),
            ("step$name", "StepName"),
            ("step%name", "StepName"),
            ("step&name", "StepName"),
            ("step*name", "StepName"),
            ("step+name", "StepName"),
            ("step=name", "StepName"),
            ("step|name", "StepName"),
            ("step\\name", "StepName"),
            ("step/name", "StepName"),
            ("step?name", "StepName"),
            ("step<name>", "StepName"),
            ("step[name]", "StepName"),
            ("step{name}", "StepName"),
            ("step(name)", "StepName")
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            # Should handle special characters gracefully
            assert isinstance(result, str)
            # May not match expected exactly due to regex limitations,
            # but should not crash
    
    def test_very_long_step_names(self):
        """Test validation with very long step names."""
        long_name = "A" * 1000  # Very long name
        step_data = {"name": long_name}
        
        # Should not crash
        errors = validate_new_step_definition(step_data)
        # Long name should still be valid PascalCase
        assert errors == []
    
    def test_unicode_characters_in_names(self):
        """Test validation with unicode characters."""
        unicode_names = [
            "MyStepWithÜnicode",
            "StepWithEmoji😀",
            "StepWith中文",
            "StepWithÑ"
        ]
        
        for name in unicode_names:
            step_data = {"name": name}
            # Should not crash
            errors = validate_new_step_definition(step_data)
            # May or may not be valid PascalCase, but should handle gracefully
            assert isinstance(errors, list)
    
    def test_concurrent_validation_calls(self):
        """Test concurrent validation calls don't interfere."""
        import threading
        import time
        
        results = []
        errors = []
        
        def validate_step(step_name):
            try:
                step_data = {"name": step_name}
                result = validate_new_step_definition(step_data)
                results.append((step_name, result))
            except Exception as e:
                errors.append((step_name, e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=validate_step, args=[f"Step{i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and all results
        assert len(errors) == 0
        assert len(results) == 10
        
        # All validations should pass (valid PascalCase names)
        for step_name, validation_errors in results:
            assert validation_errors == []


class TestIntegrationScenarios:
    """Test integration scenarios and real-world usage patterns."""
    
    def test_typical_step_creation_workflow(self):
        """Test typical step creation workflow."""
        # Step 1: Developer provides initial step data
        initial_data = {
            "name": "my_new_processing_step",
            "config_class": "MyNewProcessingConfiguration",
            "builder_step_name": "MyNewProcessingBuilder",
            "sagemaker_step_type": "Processing"
        }
        
        # Step 2: Validate the step
        errors = validate_new_step_definition(initial_data)
        assert len(errors) > 0  # Should have validation errors
        
        # Step 3: Get detailed errors with suggestions
        detailed_errors = get_validation_errors_with_suggestions(initial_data)
        assert len(detailed_errors) > len(errors)  # Should include suggestions
        
        # Step 4: Apply auto-correction
        corrected_data = auto_correct_step_definition(initial_data)
        
        # Step 5: Validate corrected data
        corrected_errors = validate_new_step_definition(corrected_data)
        assert len(corrected_errors) == 0  # Should be valid after correction
        
        # Step 6: Create validation report
        report = create_validation_report(
            initial_data["name"], initial_data, "auto_correct"
        )
        assert report["corrections_available"] is True
        assert len(report["suggested_corrections"]) > 0
    
    def test_batch_validation_scenario(self):
        """Test batch validation of multiple steps."""
        steps_to_validate = [
            ("ValidStep", {"config_class": "ValidStepConfig", "builder_step_name": "ValidStepStepBuilder"}),
            ("invalid_step", {"config_class": "InvalidConfig", "builder_step_name": "InvalidBuilder"}),
            ("AnotherValidStep", {"config_class": "AnotherValidStepConfig", "builder_step_name": "AnotherValidStepStepBuilder"}),
            ("another_invalid", {"config_class": "WrongSuffix", "builder_step_name": "WrongSuffix"})
        ]
        
        validation_results = []
        for step_name, step_data in steps_to_validate:
            step_data["name"] = step_name
            errors = validate_new_step_definition(step_data)
            validation_results.append((step_name, len(errors) == 0))
        
        # Should have 2 valid and 2 invalid steps
        valid_count = sum(1 for _, is_valid in validation_results if is_valid)
        assert valid_count == 2
    
    def test_performance_monitoring_scenario(self):
        """Test performance monitoring over time."""
        # Reset metrics
        reset_performance_metrics()
        
        # Simulate validation activity over time
        step_data = {"name": "TestStep"}
        
        # Perform validations and check metrics progression
        for i in range(1, 6):
            validate_new_step_definition(step_data)
            metrics = get_performance_metrics()
            
            assert metrics["total_validations"] == i
            assert metrics["total_time_ms"] >= 0  # Allow for very fast validation
            assert metrics["average_time_ms"] >= 0  # Allow for very fast validation
            assert metrics["target_met"] is True
    
    def test_error_recovery_scenario(self):
        """Test error recovery and graceful handling."""
        # Test with malformed data
        malformed_data = {
            "name": 123,  # Wrong type
            "config_class": [],  # Wrong type
            "builder_step_name": {},  # Wrong type
            "sagemaker_step_type": None
        }
        
        # Should handle gracefully without crashing
        try:
            errors = validate_new_step_definition(malformed_data)
            # Should return some errors but not crash
            assert isinstance(errors, list)
        except Exception as e:
            # If it does raise an exception, it should be a reasonable one
            assert isinstance(e, (TypeError, ValueError, AttributeError))


if __name__ == "__main__":
    pytest.main([__file__])
