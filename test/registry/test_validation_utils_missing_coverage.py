"""
Comprehensive tests for previously untested validation_utils functions.

This test file focuses on the functions that were missing test coverage:
- create_validation_report()
- get_performance_metrics()
- reset_performance_metrics()
- get_validation_status()

Following pytest best practices from COVERAGE_ANALYSIS_GUIDE.md:
- Applied systematic error fixing following pytest best practices
- Comprehensive test coverage with edge cases
- Clear test organization and naming
- Proper mocking and isolation
- Performance validation
"""

import pytest
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import time

from cursus.registry.validation_utils import (
    create_validation_report,
    get_performance_metrics,
    reset_performance_metrics,
    get_validation_status,
    validate_new_step_definition,
    auto_correct_step_definition,
    to_pascal_case,
    _validation_stats,
)


class TestCreateValidationReport:
    """Test create_validation_report function comprehensively."""

    def test_create_validation_report_valid_step(self):
        """Test validation report for valid step definition."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing",
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
        assert "timestamp" in report

    def test_create_validation_report_invalid_step(self):
        """Test validation report for invalid step definition."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",  # Invalid suffix
            "builder_step_name": "MyCustomBuilder",  # Invalid suffix
            "sagemaker_step_type": "InvalidType",  # Invalid type
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

    def test_create_validation_report_with_corrections(self):
        """Test validation report includes suggested corrections."""
        step_data = {
            "config_class": "my_custom_configuration",
            "builder_step_name": "my_custom_builder",
        }

        report = create_validation_report("my_custom_step", step_data, "auto_correct")

        assert report["corrections_available"] is True
        corrections = report["suggested_corrections"]

        # Check that corrections are provided for invalid fields
        assert "name" in corrections
        assert corrections["name"]["original"] == "my_custom_step"
        assert corrections["name"]["corrected"] == "MyCustomStep"

        assert "config_class" in corrections
        assert corrections["config_class"]["original"] == "my_custom_configuration"
        assert corrections["config_class"]["corrected"] == "MyCustomStepConfig"

        assert "builder_step_name" in corrections
        assert corrections["builder_step_name"]["original"] == "my_custom_builder"
        assert corrections["builder_step_name"]["corrected"] == "MyCustomStepStepBuilder"

    def test_create_validation_report_partial_corrections(self):
        """Test validation report with only some fields needing correction."""
        step_data = {
            "config_class": "MyCustomStepConfig",  # Valid
            "builder_step_name": "MyCustomBuilder",  # Invalid
            "sagemaker_step_type": "Processing",  # Valid
        }

        report = create_validation_report("MyCustomStep", step_data, "warn")

        assert report["corrections_available"] is True
        corrections = report["suggested_corrections"]

        # Only builder_step_name should need correction
        assert "builder_step_name" in corrections
        assert corrections["builder_step_name"]["original"] == "MyCustomBuilder"
        assert corrections["builder_step_name"]["corrected"] == "MyCustomStepStepBuilder"

        # Other fields should not be in corrections
        assert "name" not in corrections
        assert "config_class" not in corrections

    def test_create_validation_report_empty_step_data(self):
        """Test validation report with empty step data."""
        step_data = {}

        report = create_validation_report("", step_data, "warn")

        assert report["step_name"] == ""
        assert report["is_valid"] is False
        assert report["error_count"] > 0
        assert any("Step name is required" in error for error in report["errors"])

    def test_create_validation_report_different_modes(self):
        """Test validation report with different validation modes."""
        step_data = {"config_class": "InvalidConfig"}

        modes = ["warn", "strict", "auto_correct"]
        for mode in modes:
            report = create_validation_report("TestStep", step_data, mode)
            assert report["validation_mode"] == mode
            assert isinstance(report["is_valid"], bool)
            assert isinstance(report["error_count"], int)


class TestGetPerformanceMetrics:
    """Test get_performance_metrics function comprehensively."""

    def setup_method(self):
        """Reset performance metrics before each test."""
        reset_performance_metrics()

    def test_get_performance_metrics_initial_state(self):
        """Test performance metrics in initial state."""
        metrics = get_performance_metrics()

        assert metrics["total_validations"] == 0
        assert metrics["total_time_ms"] == 0.0
        assert metrics["average_time_ms"] == 0.0
        assert metrics["performance_target"] == "< 1ms per validation"
        assert metrics["target_met"] is True  # 0.0 < 1.0
        assert "cache_stats" in metrics

        cache_stats = metrics["cache_stats"]
        assert cache_stats["hits"] == 0
        assert cache_stats["misses"] == 0
        assert cache_stats["hit_rate"] == 0.0
        assert cache_stats["cache_size"] == 0
        assert cache_stats["max_size"] == 256  # LRU cache maxsize

    def test_get_performance_metrics_after_validations(self):
        """Test performance metrics after running validations."""
        # Run some validations to populate metrics
        step_data = {"name": "TestStep", "config_class": "TestStepConfig"}
        
        validate_new_step_definition(step_data)
        validate_new_step_definition(step_data)
        validate_new_step_definition(step_data)

        metrics = get_performance_metrics()

        assert metrics["total_validations"] == 3
        assert metrics["total_time_ms"] >= 0.0  # Could be 0.0 for very fast operations
        assert metrics["average_time_ms"] >= 0.0  # Could be 0.0 for very fast operations
        assert isinstance(metrics["target_met"], bool)

    def test_get_performance_metrics_cache_stats(self):
        """Test performance metrics cache statistics."""
        # Use to_pascal_case to populate cache
        to_pascal_case("test_step")
        to_pascal_case("test_step")  # Should be cache hit
        to_pascal_case("another_step")  # Should be cache miss

        metrics = get_performance_metrics()
        cache_stats = metrics["cache_stats"]

        assert cache_stats["hits"] >= 1  # At least one hit from repeated call
        assert cache_stats["misses"] >= 2  # At least two misses from unique calls
        assert cache_stats["cache_size"] >= 2  # At least two items cached
        assert 0.0 <= cache_stats["hit_rate"] <= 1.0

    def test_get_performance_metrics_target_validation(self):
        """Test performance target validation logic."""
        # Test by patching the global stats dictionary
        with patch('cursus.registry.validation_utils._validation_stats', {
            "total_validations": 10,
            "total_time_ms": 5.0,  # 0.5ms average
            "cache_hits": 0,
            "cache_misses": 0,
        }):
            metrics = get_performance_metrics()
            assert metrics["target_met"] is True
            assert metrics["average_time_ms"] == 0.5

        # Test slow validations (target not met)
        with patch('cursus.registry.validation_utils._validation_stats', {
            "total_validations": 10,
            "total_time_ms": 15.0,  # 1.5ms average
            "cache_hits": 0,
            "cache_misses": 0,
        }):
            metrics = get_performance_metrics()
            assert metrics["target_met"] is False
            assert metrics["average_time_ms"] == 1.5

    def test_get_performance_metrics_precision(self):
        """Test performance metrics precision and rounding."""
        # Test with patched stats for precision testing
        with patch('cursus.registry.validation_utils._validation_stats', {
            "total_validations": 3,
            "total_time_ms": 1.23456789,
            "cache_hits": 0,
            "cache_misses": 0,
        }):
            metrics = get_performance_metrics()
            
            # Check rounding to 3 decimal places
            assert metrics["total_time_ms"] == 1.235
            assert metrics["average_time_ms"] == 0.412  # 1.23456789 / 3 rounded


class TestResetPerformanceMetrics:
    """Test reset_performance_metrics function comprehensively."""

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Automatically reset global state before and after each test."""
        reset_performance_metrics()
        yield
        reset_performance_metrics()

    def test_reset_performance_metrics_basic(self):
        """Test basic performance metrics reset."""
        # First, populate some metrics
        step_data = {"name": "TestStep"}
        validate_new_step_definition(step_data)
        to_pascal_case("test_step")

        # Verify metrics are populated
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

    def test_reset_performance_metrics_multiple_calls(self):
        """Test multiple calls to reset_performance_metrics."""
        # Reset multiple times should not cause errors
        reset_performance_metrics()
        reset_performance_metrics()
        reset_performance_metrics()

        metrics = get_performance_metrics()
        assert metrics["total_validations"] == 0
        assert metrics["total_time_ms"] == 0.0

    def test_reset_performance_metrics_cache_clearing(self):
        """Test that reset clears the LRU cache."""
        # Populate cache
        to_pascal_case("test_step_one")
        to_pascal_case("test_step_two")
        to_pascal_case("test_step_one")  # Cache hit

        metrics_before = get_performance_metrics()
        assert metrics_before["cache_stats"]["cache_size"] > 0
        assert metrics_before["cache_stats"]["hits"] > 0

        # Reset and verify cache is cleared
        reset_performance_metrics()

        metrics_after = get_performance_metrics()
        assert metrics_after["cache_stats"]["cache_size"] == 0
        assert metrics_after["cache_stats"]["hits"] == 0
        assert metrics_after["cache_stats"]["misses"] == 0

    def test_reset_performance_metrics_global_state(self):
        """Test that reset affects global validation stats."""
        # Test the behavior of reset function rather than internal state
        # This follows pytest best practice: test behavior, not implementation details
        
        # Start with isolated state to avoid interference from other tests
        with patch('cursus.registry.validation_utils._validation_stats', {
            "total_validations": 0,
            "total_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }):
            # Verify we start with clean state
            initial_metrics = get_performance_metrics()
            assert initial_metrics["total_validations"] == 0
            
            # Populate some stats by running actual validations
            step_data = {"name": "TestStep"}
            validate_new_step_definition(step_data)
            to_pascal_case("test_step")

            # Verify stats are populated
            metrics_before = get_performance_metrics()
            assert metrics_before["total_validations"] > 0

            # Reset should restore to initial state
            reset_performance_metrics()

            # Test the behavior: metrics should be reset to zero
            metrics_after = get_performance_metrics()
            assert metrics_after["total_validations"] == 0
            assert metrics_after["total_time_ms"] == 0.0
            assert metrics_after["cache_stats"]["hits"] == 0
            assert metrics_after["cache_stats"]["misses"] == 0
            assert metrics_after["cache_stats"]["cache_size"] == 0


class TestGetValidationStatus:
    """Test get_validation_status function comprehensively."""

    def setup_method(self):
        """Reset performance metrics before each test."""
        reset_performance_metrics()

    def test_get_validation_status_basic_structure(self):
        """Test basic structure of validation status."""
        status = get_validation_status()

        # Check required top-level keys
        required_keys = [
            "validation_available",
            "validation_functions",
            "supported_modes",
            "implementation_approach",
            "performance_target",
            "redundancy_level",
            "current_performance",
        ]

        for key in required_keys:
            assert key in status, f"Missing required key: {key}"

    def test_get_validation_status_validation_available(self):
        """Test validation_available flag."""
        status = get_validation_status()
        assert status["validation_available"] is True

    def test_get_validation_status_validation_functions(self):
        """Test validation_functions list."""
        status = get_validation_status()
        functions = status["validation_functions"]

        expected_functions = [
            "validate_new_step_definition",
            "auto_correct_step_definition",
            "get_validation_errors_with_suggestions",
            "register_step_with_validation",
            "create_validation_report",
            "get_performance_metrics",
            "reset_performance_metrics",
        ]

        assert isinstance(functions, list)
        for func in expected_functions:
            assert func in functions, f"Missing function: {func}"

    def test_get_validation_status_supported_modes(self):
        """Test supported_modes list."""
        status = get_validation_status()
        modes = status["supported_modes"]

        expected_modes = ["warn", "strict", "auto_correct"]
        assert isinstance(modes, list)
        assert set(modes) == set(expected_modes)

    def test_get_validation_status_implementation_details(self):
        """Test implementation details."""
        status = get_validation_status()

        assert status["implementation_approach"] == "simplified_regex_based"
        assert status["performance_target"] == "< 1ms per validation"
        assert status["redundancy_level"] == "15-20% (optimal)"

    def test_get_validation_status_current_performance(self):
        """Test current_performance section."""
        status = get_validation_status()
        performance = status["current_performance"]

        required_perf_keys = [
            "average_time_ms",
            "target_met",
            "total_validations",
            "cache_hit_rate",
        ]

        for key in required_perf_keys:
            assert key in performance, f"Missing performance key: {key}"

        # Check data types
        assert isinstance(performance["average_time_ms"], (int, float))
        assert isinstance(performance["target_met"], bool)
        assert isinstance(performance["total_validations"], int)
        assert isinstance(performance["cache_hit_rate"], (int, float))
        assert 0.0 <= performance["cache_hit_rate"] <= 1.0

    def test_get_validation_status_after_operations(self):
        """Test validation status after performing operations."""
        # Perform some validations to populate metrics
        step_data = {"name": "TestStep", "config_class": "TestStepConfig"}
        validate_new_step_definition(step_data)
        validate_new_step_definition(step_data)

        status = get_validation_status()
        performance = status["current_performance"]

        assert performance["total_validations"] == 2
        assert performance["average_time_ms"] >= 0.0  # Could be 0.0 for very fast operations

    def test_get_validation_status_performance_integration(self):
        """Test integration with get_performance_metrics."""
        # Run some operations
        to_pascal_case("test_step")
        to_pascal_case("test_step")  # Cache hit
        validate_new_step_definition({"name": "TestStep"})

        status = get_validation_status()
        direct_metrics = get_performance_metrics()

        # Status should match direct metrics
        assert status["current_performance"]["average_time_ms"] == direct_metrics["average_time_ms"]
        assert status["current_performance"]["target_met"] == direct_metrics["target_met"]
        assert status["current_performance"]["total_validations"] == direct_metrics["total_validations"]
        assert status["current_performance"]["cache_hit_rate"] == direct_metrics["cache_stats"]["hit_rate"]

    def test_get_validation_status_consistency(self):
        """Test consistency across multiple calls."""
        status1 = get_validation_status()
        status2 = get_validation_status()

        # Static fields should be identical
        static_fields = [
            "validation_available",
            "validation_functions",
            "supported_modes",
            "implementation_approach",
            "performance_target",
            "redundancy_level",
        ]

        for field in static_fields:
            assert status1[field] == status2[field]


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def setup_method(self):
        """Reset state before each test."""
        reset_performance_metrics()

    def test_performance_tracking_integration(self):
        """Test that performance tracking works across all functions."""
        # Reset to ensure clean state
        reset_performance_metrics()

        # Run validations
        step_data = {"name": "TestStep", "config_class": "TestStepConfig"}
        validate_new_step_definition(step_data)
        create_validation_report("TestStep", step_data)

        # Check metrics are updated
        metrics = get_performance_metrics()
        assert metrics["total_validations"] >= 2  # At least 2 validations

        status = get_validation_status()
        assert status["current_performance"]["total_validations"] >= 2

    def test_cache_performance_across_functions(self):
        """Test cache performance across different functions."""
        reset_performance_metrics()

        # Use to_pascal_case in different contexts
        to_pascal_case("test_step")
        auto_correct_step_definition({"name": "test_step"})  # Should use cache
        create_validation_report("test_step", {})  # Should use cache

        metrics = get_performance_metrics()
        cache_stats = metrics["cache_stats"]

        assert cache_stats["hits"] > 0  # Should have cache hits
        assert cache_stats["hit_rate"] > 0.0

    def test_error_handling_in_performance_tracking(self):
        """Test error handling doesn't break performance tracking."""
        reset_performance_metrics()

        # Test with various invalid inputs
        try:
            validate_new_step_definition(None)
        except (TypeError, AttributeError):
            pass  # Expected to fail

        try:
            create_validation_report("", None)
        except (TypeError, AttributeError):
            pass  # Expected to fail

        # Performance tracking should still work
        metrics = get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "total_validations" in metrics

    def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        reset_performance_metrics()

        # Simulate multiple rapid calls
        for i in range(10):
            validate_new_step_definition({"name": f"TestStep{i}"})
            if i % 3 == 0:
                get_performance_metrics()
            if i % 5 == 0:
                get_validation_status()

        final_metrics = get_performance_metrics()
        assert final_metrics["total_validations"] == 10

    def test_memory_usage_with_large_cache(self):
        """Test behavior with large cache usage."""
        reset_performance_metrics()

        # Generate many unique pascal case conversions
        for i in range(100):
            to_pascal_case(f"test_step_{i}")

        metrics = get_performance_metrics()
        cache_stats = metrics["cache_stats"]

        # Should not exceed max cache size
        assert cache_stats["cache_size"] <= cache_stats["max_size"]
        assert cache_stats["misses"] >= 100  # All should be misses initially
