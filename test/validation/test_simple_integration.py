"""
Pytest tests for cursus.validation.simple_integration module.

Tests the SimpleValidationCoordinator class and public API functions that provide
coordination between Standardization Tester and Alignment Tester with caching,
statistics tracking, and production validation workflows.
"""

import pytest
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
    _coordinator,
)


class TestSimpleValidationCoordinator:
    """Test cases for SimpleValidationCoordinator class."""

    @pytest.fixture
    def coordinator(self):
        """Set up test fixtures."""
        return SimpleValidationCoordinator()

    @pytest.fixture
    def mock_builder(self):
        """Mock builder class."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestBuilder"
        return mock_builder

    @pytest.fixture
    def sample_std_results(self):
        """Sample validation results."""
        return {
            "passed": True,
            "status": "success",
            "tests_run": 5,
            "failures": 0,
            "message": "All tests passed",
        }

    @pytest.fixture
    def sample_align_results(self):
        """Sample alignment results."""
        return {
            "passed": True,
            "status": "success",
            "alignment_score": 0.95,
            "issues": [],
            "message": "Alignment validation passed",
        }

    def test_init(self):
        """Test coordinator initialization."""
        coordinator = SimpleValidationCoordinator()

        assert coordinator.cache == {}
        assert coordinator.stats == {
            "development_validations": 0,
            "integration_validations": 0,
            "production_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @patch("cursus.validation.builders.universal_test.UniversalStepBuilderTest")
    def test_validate_development_success(
        self, mock_test_class, coordinator, mock_builder, sample_std_results
    ):
        """Test successful development validation."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = sample_std_results.copy()
        mock_test_class.return_value = mock_tester

        # Run validation
        result = coordinator.validate_development(mock_builder, test_arg="value")

        # Verify results
        assert result["passed"] is True
        assert result["validation_type"] == "development"
        assert result["tester"] == "standardization"
        assert result["builder_class"] == "TestBuilder"
        assert result["status"] == "success"

        # Verify mock calls
        mock_test_class.assert_called_once_with(mock_builder, test_arg="value")
        mock_tester.run_all_tests.assert_called_once()

        # Verify statistics
        assert coordinator.stats["development_validations"] == 1
        assert coordinator.stats["cache_misses"] == 1
        assert coordinator.stats["cache_hits"] == 0

    @patch("cursus.validation.builders.universal_test.UniversalStepBuilderTest")
    def test_validate_development_error(
        self, mock_test_class, coordinator, mock_builder
    ):
        """Test development validation with error."""
        # Setup mock to raise exception
        mock_test_class.side_effect = Exception("Test error")

        # Run validation
        result = coordinator.validate_development(mock_builder)

        # Verify error handling
        assert result["passed"] is False
        assert result["status"] == "error"
        assert result["validation_type"] == "development"
        assert result["tester"] == "standardization"
        assert result["builder_class"] == "TestBuilder"
        assert result["error"] == "Test error"
        assert "Development validation failed" in result["message"]

    @patch("cursus.validation.builders.universal_test.UniversalStepBuilderTest")
    def test_validate_development_caching(
        self, mock_test_class, coordinator, mock_builder, sample_std_results
    ):
        """Test development validation caching."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_all_tests.return_value = sample_std_results.copy()
        mock_test_class.return_value = mock_tester

        # First call - should miss cache
        result1 = coordinator.validate_development(mock_builder)
        assert coordinator.stats["cache_misses"] == 1
        assert coordinator.stats["cache_hits"] == 0

        # Second call - should hit cache
        result2 = coordinator.validate_development(mock_builder)
        assert coordinator.stats["cache_misses"] == 1
        assert coordinator.stats["cache_hits"] == 1

        # Results should be identical
        assert result1 == result2

        # Mock should only be called once
        mock_test_class.assert_called_once()

    @patch(
        "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
    )
    def test_validate_integration_success(
        self, mock_tester_class, coordinator, sample_align_results
    ):
        """Test successful integration validation."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_full_validation.return_value = sample_align_results.copy()
        mock_tester_class.return_value = mock_tester

        script_names = ["script1", "script2"]

        # Run validation
        result = coordinator.validate_integration(script_names, test_arg="value")

        # Verify results
        assert result["passed"] is True
        assert result["validation_type"] == "integration"
        assert result["tester"] == "alignment"
        assert result["script_names"] == script_names
        assert result["status"] == "success"

        # Verify mock calls
        mock_tester_class.assert_called_once()
        mock_tester.run_full_validation.assert_called_once_with(script_names)

        # Verify statistics
        assert coordinator.stats["integration_validations"] == 1
        assert coordinator.stats["cache_misses"] == 1

    @patch(
        "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
    )
    def test_validate_integration_error(self, mock_tester_class, coordinator):
        """Test integration validation with error."""
        # Setup mock to raise exception
        mock_tester_class.side_effect = Exception("Integration error")

        script_names = ["script1"]

        # Run validation
        result = coordinator.validate_integration(script_names)

        # Verify error handling
        assert result["passed"] is False
        assert result["status"] == "error"
        assert result["validation_type"] == "integration"
        assert result["tester"] == "alignment"
        assert result["script_names"] == script_names
        assert result["error"] == "Integration error"
        assert "Integration validation failed" in result["message"]

    @patch(
        "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
    )
    def test_validate_integration_caching(
        self, mock_tester_class, coordinator, sample_align_results
    ):
        """Test integration validation caching."""
        # Setup mock
        mock_tester = Mock()
        mock_tester.run_full_validation.return_value = sample_align_results.copy()
        mock_tester_class.return_value = mock_tester

        script_names = ["script1", "script2"]

        # First call - should miss cache
        result1 = coordinator.validate_integration(script_names)
        assert coordinator.stats["cache_misses"] == 1
        assert coordinator.stats["cache_hits"] == 0

        # Second call with same scripts - should hit cache
        result2 = coordinator.validate_integration(script_names)
        assert coordinator.stats["cache_misses"] == 1
        assert coordinator.stats["cache_hits"] == 1

        # Results should be identical
        assert result1 == result2

        # Different order should still hit cache (sorted)
        result3 = coordinator.validate_integration(["script2", "script1"])
        assert coordinator.stats["cache_hits"] == 2

    def test_validate_production_both_pass(
        self, coordinator, mock_builder, sample_std_results, sample_align_results
    ):
        """Test production validation when both testers pass."""
        # Mock both validation methods
        std_results = sample_std_results.copy()
        align_results = sample_align_results.copy()

        coordinator.validate_development = Mock(return_value=std_results)
        coordinator.validate_integration = Mock(return_value=align_results)

        # Run production validation
        result = coordinator.validate_production(mock_builder, "test_script")

        # Verify results
        assert result["status"] == "passed"
        assert result["validation_type"] == "production"
        assert result["phase"] == "combined"
        assert result["builder_class"] == "TestBuilder"
        assert result["script_name"] == "test_script"
        assert result["both_passed"] is True
        assert result["standardization_passed"] is True
        assert result["alignment_passed"] is True
        assert result["correlation"] == "basic"
        assert "Production validation passed" in result["message"]

        # Verify both validations were called
        coordinator.validate_development.assert_called_once_with(mock_builder)
        coordinator.validate_integration.assert_called_once_with(["test_script"])

        # Verify statistics
        assert coordinator.stats["production_validations"] == 1

    def test_validate_production_std_fails(self, coordinator, mock_builder):
        """Test production validation when standardization fails."""
        # Mock standardization failure
        std_results = {
            "passed": False,
            "status": "failed",
            "message": "Standard failed",
        }

        coordinator.validate_development = Mock(return_value=std_results)
        coordinator.validate_integration = Mock()

        # Run production validation
        result = coordinator.validate_production(mock_builder, "test_script")

        # Verify fail-fast behavior
        assert result["status"] == "failed_standardization"
        assert result["validation_type"] == "production"
        assert result["phase"] == "standardization"
        assert result["both_passed"] is False
        assert result["standardization_results"] == std_results
        assert result["alignment_results"] is None
        assert "Fix implementation issues" in result["message"]

        # Integration should not be called due to fail-fast
        coordinator.validate_integration.assert_not_called()

    def test_validate_production_integration_fails(
        self, coordinator, mock_builder, sample_std_results
    ):
        """Test production validation when integration fails."""
        # Mock standardization pass, integration fail
        std_results = sample_std_results.copy()
        align_results = {
            "passed": False,
            "status": "failed",
            "message": "Alignment failed",
        }

        coordinator.validate_development = Mock(return_value=std_results)
        coordinator.validate_integration = Mock(return_value=align_results)

        # Run production validation
        result = coordinator.validate_production(mock_builder, "test_script")

        # Verify results
        assert result["status"] == "failed_integration"
        assert result["both_passed"] is False
        assert result["standardization_passed"] is True
        assert result["alignment_passed"] is False
        assert (
            "Implementation quality validated but integration issues"
            in result["message"]
        )

    def test_validate_production_error(self, coordinator, mock_builder):
        """Test production validation with error."""
        # Mock error in development validation
        coordinator.validate_development = Mock(
            side_effect=Exception("Production error")
        )

        # Run production validation
        result = coordinator.validate_production(mock_builder, "test_script")

        # Verify error handling
        assert result["status"] == "error"
        assert result["validation_type"] == "production"
        assert result["phase"] == "error"
        assert result["builder_class"] == "TestBuilder"
        assert result["script_name"] == "test_script"
        assert result["error"] == "Production error"
        assert "Production validation error" in result["message"]

    def test_clear_cache(self, coordinator):
        """Test cache clearing."""
        # Add some cache entries
        coordinator.cache["test1"] = {"result": "data1"}
        coordinator.cache["test2"] = {"result": "data2"}

        assert len(coordinator.cache) == 2

        # Clear cache
        coordinator.clear_cache()

        assert len(coordinator.cache) == 0
        assert coordinator.cache == {}

    def test_get_statistics_empty(self, coordinator):
        """Test statistics with no validations."""
        stats = coordinator.get_statistics()

        expected = {
            "total_validations": 0,
            "development_validations": 0,
            "integration_validations": 0,
            "production_validations": 0,
            "cache_hit_rate_percentage": 0.0,
            "cache_size": 0,
        }

        assert stats == expected

    def test_get_statistics_with_data(self, coordinator):
        """Test statistics with validation data."""
        # Simulate some validations
        coordinator.stats.update(
            {
                "development_validations": 5,
                "integration_validations": 3,
                "production_validations": 2,
                "cache_hits": 7,
                "cache_misses": 3,
            }
        )

        # Add cache entries
        coordinator.cache["test1"] = {}
        coordinator.cache["test2"] = {}

        stats = coordinator.get_statistics()

        assert stats["total_validations"] == 10  # 5 + 3 + 2
        assert stats["development_validations"] == 5
        assert stats["integration_validations"] == 3
        assert stats["production_validations"] == 2
        assert stats["cache_hit_rate_percentage"] == 70.0  # 7/(7+3) * 100
        assert stats["cache_size"] == 2


class TestPublicAPIFunctions:
    """Test cases for public API functions."""

    @pytest.fixture(autouse=True)
    def setup_coordinator(self):
        """Set up test fixtures."""
        # Clear global coordinator state
        _coordinator.clear_cache()
        _coordinator.stats = {
            "development_validations": 0,
            "integration_validations": 0,
            "production_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @pytest.fixture
    def mock_builder(self):
        """Mock builder fixture."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestBuilder"
        return mock_builder

    @patch.object(_coordinator, "validate_development")
    def test_validate_development_function(self, mock_method, mock_builder):
        """Test validate_development public function."""
        expected_result = {"passed": True, "message": "Success"}
        mock_method.return_value = expected_result

        result = validate_development(mock_builder, test_arg="value")

        assert result == expected_result
        mock_method.assert_called_once_with(mock_builder, test_arg="value")

    @patch.object(_coordinator, "validate_integration")
    def test_validate_integration_function(self, mock_method):
        """Test validate_integration public function."""
        expected_result = {"passed": True, "message": "Success"}
        mock_method.return_value = expected_result

        script_names = ["script1", "script2"]
        result = validate_integration(script_names, test_arg="value")

        assert result == expected_result
        mock_method.assert_called_once_with(script_names, test_arg="value")

    @patch.object(_coordinator, "validate_production")
    def test_validate_production_function(self, mock_method, mock_builder):
        """Test validate_production public function."""
        expected_result = {"both_passed": True, "status": "passed"}
        mock_method.return_value = expected_result

        result = validate_production(mock_builder, "test_script", test_arg="value")

        assert result == expected_result
        mock_method.assert_called_once_with(
            mock_builder, "test_script", test_arg="value"
        )

    @patch.object(_coordinator, "clear_cache")
    def test_clear_validation_cache_function(self, mock_method):
        """Test clear_validation_cache public function."""
        clear_validation_cache()
        mock_method.assert_called_once()

    @patch.object(_coordinator, "get_statistics")
    def test_get_validation_statistics_function(self, mock_method):
        """Test get_validation_statistics public function."""
        expected_stats = {"total_validations": 5, "cache_hit_rate_percentage": 80.0}
        mock_method.return_value = expected_stats

        result = get_validation_statistics()

        assert result == expected_stats
        mock_method.assert_called_once()


class TestLegacyCompatibilityFunctions:
    """Test cases for legacy compatibility functions."""

    @pytest.fixture
    def mock_builder(self):
        """Set up test fixtures."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestBuilder"
        return mock_builder

    @patch("cursus.validation.simple_integration.validate_development")
    def test_validate_step_builder_deprecation(self, mock_validate_dev, mock_builder):
        """Test validate_step_builder shows deprecation warning."""
        expected_result = {"passed": True}
        mock_validate_dev.return_value = expected_result

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = validate_step_builder(mock_builder, test_arg="value")

            # Check deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_step_builder() is deprecated" in str(w[0].message)
            assert "Use validate_development() instead" in str(w[0].message)

            # Check function still works
            assert result == expected_result
            mock_validate_dev.assert_called_once_with(mock_builder, test_arg="value")

    @patch("cursus.validation.simple_integration.validate_integration")
    def test_validate_step_integration_deprecation(self, mock_validate_int):
        """Test validate_step_integration shows deprecation warning."""
        expected_result = {"passed": True}
        mock_validate_int.return_value = expected_result

        script_names = ["script1", "script2"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = validate_step_integration(script_names, test_arg="value")

            # Check deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "validate_step_integration() is deprecated" in str(w[0].message)
            assert "Use validate_integration() instead" in str(w[0].message)

            # Check function still works
            assert result == expected_result
            mock_validate_int.assert_called_once_with(script_names, test_arg="value")


class TestIntegrationScenarios:
    """Test cases for integration scenarios and edge cases."""

    @pytest.fixture
    def coordinator(self):
        """Set up test fixtures."""
        return SimpleValidationCoordinator()

    @pytest.fixture
    def mock_builder(self):
        """Mock builder fixture."""
        mock_builder = Mock()
        mock_builder.__name__ = "IntegrationTestBuilder"
        return mock_builder

    def test_cache_key_generation(self, coordinator, mock_builder):
        """Test cache key generation for different scenarios."""
        # Test development cache key
        with patch(
            "cursus.validation.builders.universal_test.UniversalStepBuilderTest"
        ) as mock_test:
            mock_tester = Mock()
            mock_tester.run_all_tests.return_value = {"passed": True}
            mock_test.return_value = mock_tester

            # Same builder should use same cache key
            coordinator.validate_development(mock_builder)
            coordinator.validate_development(mock_builder)

            # Should only call once due to caching
            assert mock_test.call_count == 1

        # Test integration cache key with script order
        with patch(
            "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
        ) as mock_align:
            mock_tester = Mock()
            mock_tester.run_full_validation.return_value = {"passed": True}
            mock_align.return_value = mock_tester

            # Different order should use same cache (sorted)
            coordinator.validate_integration(["script2", "script1"])
            coordinator.validate_integration(["script1", "script2"])

            # Should only call once due to caching
            assert mock_align.call_count == 1

    def test_statistics_accuracy(self, coordinator, mock_builder):
        """Test statistics tracking accuracy."""
        # Mock the validation methods
        with patch(
            "cursus.validation.builders.universal_test.UniversalStepBuilderTest"
        ), patch(
            "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
        ):

            # Run various validations
            coordinator.validate_development(mock_builder)  # cache miss
            coordinator.validate_development(mock_builder)  # cache hit
            coordinator.validate_integration(["script1"])  # cache miss
            coordinator.validate_integration(["script1"])  # cache hit

            stats = coordinator.get_statistics()

            assert stats["development_validations"] == 1
            assert stats["integration_validations"] == 1
            assert stats["cache_hit_rate_percentage"] == 50.0  # 2 hits out of 4 total

    def test_error_resilience(self, coordinator, mock_builder):
        """Test error handling and resilience."""
        # Test with various error types
        with patch(
            "cursus.validation.builders.universal_test.UniversalStepBuilderTest"
        ) as mock_test:
            # Import error
            mock_test.side_effect = ImportError("Module not found")
            result = coordinator.validate_development(mock_builder)
            assert result["status"] == "error"
            assert "Module not found" in result["error"]

            # Runtime error
            mock_test.side_effect = RuntimeError("Runtime issue")
            result = coordinator.validate_development(mock_builder)
            assert result["status"] == "error"
            assert "Runtime issue" in result["error"]

    def test_production_validation_workflow(self, coordinator, mock_builder):
        """Test complete production validation workflow."""
        # Mock successful standardization, failed integration
        std_results = {"passed": True, "status": "success"}
        align_results = {
            "passed": False,
            "status": "failed",
            "issues": ["alignment issue"],
        }

        coordinator.validate_development = Mock(return_value=std_results)
        coordinator.validate_integration = Mock(return_value=align_results)

        result = coordinator.validate_production(mock_builder, "test_script")

        # Should proceed through both phases
        assert result["status"] == "failed_integration"
        assert result["standardization_passed"] is True
        assert result["alignment_passed"] is False
        assert result["both_passed"] is False

        # Both methods should be called
        coordinator.validate_development.assert_called_once()
        coordinator.validate_integration.assert_called_once()

    def test_empty_script_names(self, coordinator):
        """Test handling of empty script names list."""
        with patch(
            "cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester"
        ) as mock_align:
            mock_tester = Mock()
            mock_tester.run_full_validation.return_value = {"passed": True}
            mock_align.return_value = mock_tester

            result = coordinator.validate_integration([])

            assert result["script_names"] == []
            mock_tester.run_full_validation.assert_called_once_with([])

    def test_large_cache_behavior(self, coordinator):
        """Test behavior with large cache."""
        # Simulate large cache
        for i in range(100):
            coordinator.cache[f"test_key_{i}"] = {"result": f"data_{i}"}

        stats = coordinator.get_statistics()
        assert stats["cache_size"] == 100

        # Clear and verify
        coordinator.clear_cache()
        stats = coordinator.get_statistics()
        assert stats["cache_size"] == 0
