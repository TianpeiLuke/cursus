"""
Pytest tests for UniversalStepBuilderTest class.

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Mock at import locations, not definition locations  
3. Match test behavior to actual implementation behavior
4. Use realistic fixtures and data structures
5. Test both success and failure scenarios
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Dict, Any, List, Type
import tempfile
import json

# Import the class under test
from src.cursus.validation.builders.universal_test import UniversalStepBuilderTest


class TestUniversalStepBuilderTestInitialization:
    """Test UniversalStepBuilderTest initialization and setup."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        # Test with defaults
        tester = UniversalStepBuilderTest()
        
        assert tester.workspace_dirs is None
        assert tester.verbose is False
        assert tester.enable_scoring is True
        assert tester.enable_structured_reporting is False

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        workspace_dirs = ["/test/workspace1", "/test/workspace2"]
        
        tester = UniversalStepBuilderTest(
            workspace_dirs=workspace_dirs,
            verbose=True,
            enable_scoring=False,
            enable_structured_reporting=True
        )
        
        assert tester.workspace_dirs == workspace_dirs
        assert tester.verbose is True
        assert tester.enable_scoring is False
        assert tester.enable_structured_reporting is True

    @patch('src.cursus.step_catalog.StepCatalog')
    def test_init_with_step_catalog_available(self, mock_step_catalog_class):
        """Test initialization when StepCatalog is available."""
        mock_catalog_instance = Mock()
        mock_step_catalog_class.return_value = mock_catalog_instance
        
        tester = UniversalStepBuilderTest(workspace_dirs=["/test"])
        
        assert tester.step_catalog_available is True
        assert tester.step_catalog == mock_catalog_instance
        mock_step_catalog_class.assert_called_once_with(workspace_dirs=["/test"])

    @patch('src.cursus.step_catalog.StepCatalog')
    def test_init_with_step_catalog_import_error(self, mock_step_catalog_class):
        """Test initialization when StepCatalog import fails."""
        mock_step_catalog_class.side_effect = ImportError("StepCatalog not available")
        
        tester = UniversalStepBuilderTest()
        
        assert tester.step_catalog_available is False
        assert tester.step_catalog is None

    @patch('src.cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_init_with_alignment_tester_available(self, mock_alignment_tester_class):
        """Test initialization when UnifiedAlignmentTester is available."""
        mock_alignment_instance = Mock()
        mock_alignment_tester_class.return_value = mock_alignment_instance
        
        tester = UniversalStepBuilderTest(workspace_dirs=["/test"])
        
        assert tester.alignment_available is True
        assert tester.alignment_tester == mock_alignment_instance
        mock_alignment_tester_class.assert_called_once_with(workspace_dirs=["/test"])

    @patch('src.cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester')
    def test_init_with_alignment_tester_import_error(self, mock_alignment_tester_class):
        """Test initialization when UnifiedAlignmentTester import fails."""
        mock_alignment_tester_class.side_effect = ImportError("UnifiedAlignmentTester not available")
        
        tester = UniversalStepBuilderTest()
        
        assert tester.alignment_available is False
        assert tester.alignment_tester is None


class TestUniversalStepBuilderTestValidation:
    """Test validation methods of UniversalStepBuilderTest."""

    @pytest.fixture
    def mock_tester_with_catalog(self):
        """Create a tester with mocked step catalog."""
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest()
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            yield tester, mock_catalog

    @pytest.fixture
    def mock_builder_class(self):
        """Create a mock builder class for testing."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        # Add required methods
        mock_builder.create_step = Mock()
        mock_builder.validate_configuration = Mock()
        mock_builder._get_inputs = Mock()
        mock_builder._get_outputs = Mock()
        mock_builder._create_processor = Mock()
        
        return mock_builder

    def test_run_validation_for_step_success(self, mock_tester_with_catalog, mock_builder_class):
        """Test successful validation for a step."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock catalog to return builder class
        mock_catalog.load_builder_class.return_value = mock_builder_class
        
        # Mock alignment tester at the actual import location
        with patch('src.cursus.validation.alignment.unified_alignment_tester.UnifiedAlignmentTester') as mock_alignment_class:
            mock_alignment = Mock()
            mock_alignment.run_validation_for_step.return_value = {
                "step_name": "TestStep",
                "overall_status": "PASSED"
            }
            mock_alignment_class.return_value = mock_alignment
            tester.alignment_tester = mock_alignment
            tester.alignment_available = True
            
            # Mock scoring
            with patch.object(tester, '_calculate_scoring') as mock_scoring:
                mock_scoring.return_value = {"overall": {"score": 95.0, "rating": "Excellent"}}
                
                results = tester.run_validation_for_step("TestStep")
                
                assert results["step_name"] == "TestStep"
                assert results["validation_type"] == "comprehensive_builder_validation"
                assert results["builder_class"] == "TestStepBuilder"
                assert "components" in results
                assert "scoring" in results

    def test_run_validation_for_step_no_builder_found(self, mock_tester_with_catalog):
        """Test validation when no builder class is found."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock catalog to return None (no builder found)
        mock_catalog.load_builder_class.return_value = None
        
        results = tester.run_validation_for_step("NonExistentStep")
        
        assert results["step_name"] == "NonExistentStep"
        assert results["overall_status"] == "ERROR"
        assert "No builder class found" in results["error"]

    def test_run_validation_for_step_with_exception(self, mock_tester_with_catalog, mock_builder_class):
        """Test validation when an exception occurs during validation."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock catalog to return builder class
        mock_catalog.load_builder_class.return_value = mock_builder_class
        
        # Mock _run_comprehensive_validation_for_step to raise exception
        with patch.object(tester, '_run_comprehensive_validation_for_step') as mock_validation:
            mock_validation.side_effect = Exception("Validation failed")
            
            results = tester.run_validation_for_step("TestStep")
            
            assert results["step_name"] == "TestStep"
            assert results["overall_status"] == "ERROR"
            assert "Validation failed" in results["error"]


class TestUniversalStepBuilderTestIntegrationChecks:
    """Test integration capability checking methods."""

    @pytest.fixture
    def mock_builder_class(self):
        """Create a mock builder class with various methods."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        # Add various methods for testing
        mock_builder._get_inputs = Mock()
        mock_builder._get_outputs = Mock()
        mock_builder._get_cache_config = Mock()
        mock_builder.__init__ = Mock()
        
        return mock_builder

    def test_check_dependency_resolution_success(self):
        """Test dependency resolution check with available methods."""
        tester = UniversalStepBuilderTest()
        
        # Create builder with dependency methods
        mock_builder = Mock()
        mock_builder._get_inputs = Mock()
        mock_builder._get_outputs = Mock()
        
        result = tester._check_dependency_resolution(mock_builder)
        
        assert result["passed"] is True
        assert "_get_inputs" in result["found_methods"]
        assert "_get_outputs" in result["found_methods"]

    def test_check_dependency_resolution_no_methods(self):
        """Test dependency resolution check with no dependency methods."""
        tester = UniversalStepBuilderTest()
        
        # Create builder without dependency methods - use spec to control available methods
        mock_builder = Mock(spec=['__name__'])  # Only has __name__, no dependency methods
        mock_builder.__name__ = "TestStepBuilder"
        
        result = tester._check_dependency_resolution(mock_builder)
        
        assert result["passed"] is False
        assert result["found_methods"] == []

    def test_check_cache_configuration_success(self):
        """Test cache configuration check (always passes as it's optional)."""
        tester = UniversalStepBuilderTest()
        
        mock_builder = Mock()
        mock_builder._get_cache_config = Mock()
        
        result = tester._check_cache_configuration(mock_builder)
        
        assert result["passed"] is True
        assert "_get_cache_config" in result["found_methods"]

    def test_check_cache_configuration_no_methods(self):
        """Test cache configuration check with no cache methods (still passes)."""
        tester = UniversalStepBuilderTest()
        
        mock_builder = Mock(spec=[])
        
        result = tester._check_cache_configuration(mock_builder)
        
        assert result["passed"] is True  # Cache is optional
        assert result["found_methods"] == []


class TestUniversalStepBuilderTestStepInstantiation:
    """Test step instantiation structural validation methods."""

    @pytest.fixture
    def mock_tester_with_catalog(self):
        """Create tester with mocked step catalog."""
        # FIXED: Mock at actual import location (Category 1: Conditional Import Mocking)
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest()
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            yield tester, mock_catalog

    def test_check_config_class_exists_success(self, mock_tester_with_catalog):
        """Test config class existence check when config is found."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock builder class
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        # Mock catalog to return config classes
        mock_catalog.discover_config_classes.return_value = {
            "TestConfig": Mock()
        }
        
        result = tester._check_config_class_exists(mock_builder)
        
        assert result["passed"] is True
        assert result["config_class"] == "TestConfig"
        assert result["found_via"] == "step_catalog"

    def test_check_config_class_exists_not_found(self, mock_tester_with_catalog):
        """Test config class existence check when config is not found."""
        tester, mock_catalog = mock_tester_with_catalog
        
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        # Mock catalog to return empty config classes
        mock_catalog.discover_config_classes.return_value = {}
        
        result = tester._check_config_class_exists(mock_builder)
        
        assert result["passed"] is False
        assert result["config_class"] == "TestConfig"
        assert "not found" in result["error"]

    def test_check_config_import_success(self):
        """Test config import check when builder has config parameter."""
        tester = UniversalStepBuilderTest()
        
        # FIXED: Category 13 - Mock the introspection result, not the function itself
        with patch('inspect.signature') as mock_signature:
            mock_sig = Mock()
            mock_sig.parameters.keys.return_value = ['self', 'config', 'other_param']
            mock_signature.return_value = mock_sig
            
            # Create a real class for testing
            class MockBuilderClass:
                def __init__(self, config, other_param=None):
                    pass
            
            result = tester._check_config_import(MockBuilderClass)
            
            assert result["passed"] is True
            assert result["has_config_param"] is True
            assert "config" in result["init_params"]

    def test_check_config_import_no_config_param(self):
        """Test config import check when builder has no config parameter."""
        tester = UniversalStepBuilderTest()
        
        # FIXED: Category 13 - Mock the introspection result, not the function itself
        with patch('inspect.signature') as mock_signature:
            mock_sig = Mock()
            mock_sig.parameters.keys.return_value = ['self', 'other_param']
            mock_signature.return_value = mock_sig
            
            # Create a real class for testing
            class MockBuilderClass:
                def __init__(self, other_param=None):
                    pass
            
            result = tester._check_config_import(MockBuilderClass)
            
            assert result["passed"] is False
            assert result["has_config_param"] is False

    def test_check_input_output_methods_success(self):
        """Test I/O methods check when methods are present."""
        tester = UniversalStepBuilderTest()
        
        mock_builder = Mock()
        mock_builder._get_inputs = Mock()
        mock_builder._get_outputs = Mock()
        
        result = tester._check_input_output_methods(mock_builder)
        
        assert result["passed"] is True
        assert "_get_inputs" in result["found_methods"]
        assert "_get_outputs" in result["found_methods"]

    def test_check_input_output_methods_partial(self):
        """Test I/O methods check when only one method is present."""
        tester = UniversalStepBuilderTest()
        
        # Use spec to control which methods are available
        mock_builder = Mock(spec=['_get_inputs'])  # Only has _get_inputs
        mock_builder._get_inputs = Mock()
        # No _get_outputs method
        
        result = tester._check_input_output_methods(mock_builder)
        
        assert result["passed"] is True  # At least one method present
        assert "_get_inputs" in result["found_methods"]
        assert "_get_outputs" not in result["found_methods"]

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_check_sagemaker_methods_training_step(self, mock_get_step_type):
        """Test SageMaker methods check for Training step."""
        tester = UniversalStepBuilderTest()
        
        mock_get_step_type.return_value = "Training"
        
        mock_builder = Mock()
        mock_builder.__name__ = "XGBoostTrainingStepBuilder"
        mock_builder._create_estimator = Mock()
        
        result = tester._check_sagemaker_methods(mock_builder)
        
        assert result["passed"] is True
        assert result["step_type"] == "Training"
        assert "_create_estimator" in result["found_methods"]

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_check_sagemaker_methods_processing_step(self, mock_get_step_type):
        """Test SageMaker methods check for Processing step."""
        tester = UniversalStepBuilderTest()
        
        mock_get_step_type.return_value = "Processing"
        
        mock_builder = Mock()
        mock_builder.__name__ = "TabularPreprocessingStepBuilder"
        mock_builder._create_processor = Mock()
        
        result = tester._check_sagemaker_methods(mock_builder)
        
        assert result["passed"] is True
        assert result["step_type"] == "Processing"
        assert "_create_processor" in result["found_methods"]

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_check_sagemaker_methods_missing_required_method(self, mock_get_step_type):
        """Test SageMaker methods check when required method is missing."""
        tester = UniversalStepBuilderTest()
        
        mock_get_step_type.return_value = "Training"
        
        mock_builder = Mock(spec=[])  # Empty spec means no methods available
        mock_builder.__name__ = "XGBoostTrainingStepBuilder"
        # No _create_estimator method
        
        result = tester._check_sagemaker_methods(mock_builder)
        
        assert result["passed"] is False
        assert result["step_type"] == "Training"
        assert result["found_methods"] == []


class TestUniversalStepBuilderTestStepCreation:
    """Test step creation capability validation methods."""

    @pytest.fixture
    def mock_tester_with_catalog(self):
        """Create tester with mocked step catalog."""
        # FIXED: Mock at actual import location (Category 1: Conditional Import Mocking)
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest()
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            yield tester, mock_catalog

    def test_check_config_availability_success(self, mock_tester_with_catalog):
        """Test config availability check when config is available."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock config class
        mock_config_class = Mock()
        mock_catalog.discover_config_classes.return_value = {
            "TestStepConfig": mock_config_class
        }
        
        result = tester._check_config_availability("TestStep")
        
        assert result["available"] is True
        assert result["config_class"] == "TestStepConfig"
        assert result["discovery_method"] == "step_catalog"

    def test_check_config_availability_not_found(self, mock_tester_with_catalog):
        """Test config availability check when config is not found."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock empty config classes
        mock_catalog.discover_config_classes.return_value = {}
        
        result = tester._check_config_availability("TestStep")
        
        assert result["available"] is False
        assert result["config_class"] == "TestStepConfig"
        assert "not found" in result["error"]

    def test_check_required_methods_success(self):
        """Test required methods check when all methods are present."""
        tester = UniversalStepBuilderTest()
        
        # Create mock builder with required methods (don't set __init__ directly)
        mock_builder = Mock()
        mock_builder.create_step = Mock()
        mock_builder.validate_configuration = Mock()
        # Don't set __init__ directly - test the behavior, not the magic method
        
        result = tester._check_required_methods(mock_builder)
        
        assert result["has_required_methods"] is True
        assert "create_step" in result["found_required"]
        assert "validate_configuration" in result["found_required"]
        assert result["total_found"] == 2

    def test_check_required_methods_missing(self):
        """Test required methods check when methods are missing."""
        tester = UniversalStepBuilderTest()
        
        mock_builder = Mock(spec=[])  # No methods
        
        result = tester._check_required_methods(mock_builder)
        
        assert result["has_required_methods"] is False
        assert result["found_required"] == []
        assert "create_step" in result["missing_required"]
        assert "validate_configuration" in result["missing_required"]

    def test_check_field_requirements_with_categorization(self, mock_tester_with_catalog):
        """Test field requirements check with field categorization."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock config class with model_fields
        mock_config_class = Mock()
        mock_config_class.model_fields = {
            "author": Mock(),
            "bucket": Mock(),
            "training_entry_point": Mock()
        }
        
        # Mock config instance with categorize_fields method
        mock_config_instance = Mock()
        mock_config_instance.categorize_fields.return_value = {
            "essential": ["author", "bucket", "training_entry_point"],
            "system": [],
            "derived": []
        }
        mock_config_class.model_construct.return_value = mock_config_instance
        
        mock_catalog.discover_config_classes.return_value = {
            "TestStepConfig": mock_config_class
        }
        
        result = tester._check_field_requirements("TestStep", Mock())
        
        assert result["requirements_identifiable"] is True
        assert result["total_fields"] == 3
        assert result["essential_fields"] == ["author", "bucket", "training_entry_point"]
        assert result["identification_method"] == "field_categorization"


class TestUniversalStepBuilderTestStepTypeValidation:
    """Test step type specific validation methods."""

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_run_step_type_specific_validation_processing(self, mock_get_step_type):
        """Test step type validation for Processing steps."""
        tester = UniversalStepBuilderTest()
        mock_get_step_type.return_value = "Processing"
        
        mock_builder = Mock()
        mock_builder._create_processor = Mock()
        mock_builder._get_inputs = Mock()
        mock_builder._get_outputs = Mock()
        
        result = tester._run_step_type_specific_validation("TestStep", mock_builder)
        
        assert result["status"] == "COMPLETED"
        assert result["results"]["step_type"] == "Processing"
        assert result["results"]["step_type_tests"]["processor_methods"]["passed"] is True

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_run_step_type_specific_validation_training(self, mock_get_step_type):
        """Test step type validation for Training steps."""
        tester = UniversalStepBuilderTest()
        mock_get_step_type.return_value = "Training"
        
        mock_builder = Mock()
        mock_builder._create_estimator = Mock()
        
        result = tester._run_step_type_specific_validation("TestStep", mock_builder)
        
        assert result["status"] == "COMPLETED"
        assert result["results"]["step_type"] == "Training"
        assert result["results"]["step_type_tests"]["estimator_methods"]["passed"] is True

    @patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type')
    def test_run_step_type_specific_validation_unknown_type(self, mock_get_step_type):
        """Test step type validation for unknown step types."""
        tester = UniversalStepBuilderTest()
        mock_get_step_type.return_value = "UnknownType"
        
        mock_builder = Mock()
        
        result = tester._run_step_type_specific_validation("TestStep", mock_builder)
        
        assert result["status"] == "COMPLETED"
        assert result["results"]["step_type"] == "UnknownType"
        assert "No specific tests" in result["results"]["step_type_tests"]["note"]


class TestUniversalStepBuilderTestDiscovery:
    """Test step discovery methods."""

    @pytest.fixture
    def mock_tester_with_catalog(self):
        """Create tester with mocked step catalog."""
        # FIXED: Category 1 - Mock at actual import location (conditional import)
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest()
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            yield tester, mock_catalog

    def test_discover_all_steps_with_catalog(self, mock_tester_with_catalog):
        """Test step discovery using step catalog."""
        tester, mock_catalog = mock_tester_with_catalog
        
        # Mock catalog to return step names
        mock_catalog.list_available_steps.return_value = ["Step1", "Step2", "Step3"]
        
        steps = tester._discover_all_steps()
        
        assert len(steps) == 3
        assert "Step1" in steps
        assert "Step2" in steps
        assert "Step3" in steps
        mock_catalog.list_available_steps.assert_called_once()

    @patch('src.cursus.validation.builders.universal_test.STEP_NAMES')
    def test_discover_all_steps_fallback_to_registry(self, mock_step_names):
        """Test step discovery fallback to registry when catalog fails."""
        # Create tester without step catalog
        tester = UniversalStepBuilderTest()
        tester.step_catalog_available = False
        
        # Mock registry step names
        mock_step_names.keys.return_value = ["RegStep1", "RegStep2"]
        
        steps = tester._discover_all_steps()
        
        assert len(steps) == 2
        assert "RegStep1" in steps
        assert "RegStep2" in steps

    def test_get_builder_class_from_catalog_success(self, mock_tester_with_catalog):
        """Test getting builder class from catalog successfully."""
        tester, mock_catalog = mock_tester_with_catalog
        
        mock_builder_class = Mock()
        mock_catalog.load_builder_class.return_value = mock_builder_class
        
        result = tester._get_builder_class_from_catalog("TestStep")
        
        assert result == mock_builder_class
        mock_catalog.load_builder_class.assert_called_once_with("TestStep")

    def test_get_builder_class_from_catalog_not_found(self, mock_tester_with_catalog):
        """Test getting builder class when not found in catalog."""
        tester, mock_catalog = mock_tester_with_catalog
        
        mock_catalog.load_builder_class.return_value = None
        
        result = tester._get_builder_class_from_catalog("NonExistentStep")
        
        assert result is None


class TestUniversalStepBuilderTestUtilityMethods:
    """Test utility and helper methods."""

    def test_determine_overall_status_passed(self):
        """Test overall status determination when all components pass."""
        tester = UniversalStepBuilderTest()
        
        components = {
            "alignment_validation": {"status": "COMPLETED"},
            "integration_testing": {"status": "COMPLETED"},
            "step_creation": {"status": "COMPLETED"}
        }
        
        status = tester._determine_overall_status(components)
        assert status == "PASSED"

    def test_determine_overall_status_issues_found(self):
        """Test overall status determination when issues are found."""
        tester = UniversalStepBuilderTest()
        
        components = {
            "alignment_validation": {"status": "COMPLETED"},
            "integration_testing": {"status": "ISSUES_FOUND"},
            "step_creation": {"status": "COMPLETED"}
        }
        
        status = tester._determine_overall_status(components)
        assert status == "ISSUES_FOUND"

    def test_determine_overall_status_failed(self):
        """Test overall status determination when components fail."""
        tester = UniversalStepBuilderTest()
        
        components = {
            "alignment_validation": {"status": "ERROR"},
            "integration_testing": {"status": "COMPLETED"},
            "step_creation": {"status": "COMPLETED"}
        }
        
        status = tester._determine_overall_status(components)
        assert status == "FAILED"

    def test_generate_validation_summary(self):
        """Test validation summary generation."""
        tester = UniversalStepBuilderTest()
        
        step_results = {
            "Step1": {"overall_status": "PASSED"},
            "Step2": {"overall_status": "ISSUES_FOUND"},
            "Step3": {"overall_status": "FAILED"},
            "Step4": {"overall_status": "PASSED"}
        }
        
        summary = tester._generate_validation_summary(step_results)
        
        assert summary["total_steps"] == 4
        assert summary["passed_steps"] == 2
        assert summary["failed_steps"] == 1
        assert summary["issues_steps"] == 1
        assert summary["pass_rate"] == 50.0

    def test_extract_base_name_with_suffix(self):
        """Test base name extraction from builder class name with suffix."""
        tester = UniversalStepBuilderTest()
        
        base_name = tester._extract_base_name("XGBoostTrainingStepBuilder")
        assert base_name == "XGBoostTraining"

    def test_extract_base_name_without_suffix(self):
        """Test base name extraction from class name without suffix."""
        tester = UniversalStepBuilderTest()
        
        base_name = tester._extract_base_name("SomeClass")
        assert base_name == "SomeClass"


class TestUniversalStepBuilderTestClassMethods:
    """Test class methods and backward compatibility."""

    def test_from_builder_class_method(self):
        """Test from_builder_class class method."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        tester = UniversalStepBuilderTest.from_builder_class(
            builder_class=mock_builder,
            workspace_dirs=["/test"],
            verbose=True
        )
        
        assert tester.builder_class == mock_builder
        assert tester.single_builder_mode is True
        assert tester.workspace_dirs == ["/test"]
        assert tester.verbose is True

    @patch('src.cursus.validation.builders.universal_test.STEP_NAMES')
    def test_test_all_builders_by_type(self, mock_step_names):
        """Test testing all builders by SageMaker step type."""
        # Mock STEP_NAMES registry
        mock_step_names.items.return_value = [
            ("Step1", {"sagemaker_step_type": "Training"}),
            ("Step2", {"sagemaker_step_type": "Processing"}),
            ("Step3", {"sagemaker_step_type": "Training"}),
        ]
        
        # Mock the tester instance and its methods
        with patch.object(UniversalStepBuilderTest, '__init__', return_value=None):
            with patch.object(UniversalStepBuilderTest, 'run_validation_for_step') as mock_validation:
                mock_validation.return_value = {"overall_status": "PASSED"}
                
                results = UniversalStepBuilderTest.test_all_builders_by_type(
                    sagemaker_step_type="Training",
                    verbose=False,
                    enable_scoring=True
                )
                
                # Should test Step1 and Step3 (both Training type)
                assert "Step1" in results
                assert "Step3" in results
                assert "Step2" not in results  # Processing type, not Training


class TestUniversalStepBuilderTestErrorHandling:
    """Test error handling scenarios."""

    def test_run_validation_for_step_scoring_error(self):
        """Test validation when scoring calculation fails."""
        # FIXED: Category 1 - Mock at actual import location (conditional import)
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest(enable_scoring=True)
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            # Mock builder class
            mock_builder = Mock()
            mock_builder.__name__ = "TestStepBuilder"
            mock_catalog.load_builder_class.return_value = mock_builder
            
            # Mock comprehensive validation to succeed
            with patch.object(tester, '_run_comprehensive_validation_for_step') as mock_validation:
                mock_validation.return_value = {
                    "step_name": "TestStep",
                    "components": {},
                    "overall_status": "PASSED"
                }
                
                # Mock scoring to fail
                with patch.object(tester, '_calculate_scoring') as mock_scoring:
                    mock_scoring.side_effect = Exception("Scoring failed")
                    
                    results = tester.run_validation_for_step("TestStep")
                    
                    assert results["step_name"] == "TestStep"
                    assert "scoring_error" in results
                    assert "Scoring failed" in results["scoring_error"]

    def test_check_config_class_exists_exception(self):
        """Test config class existence check when exception occurs."""
        tester = UniversalStepBuilderTest()
        tester.step_catalog_available = True
        tester.step_catalog = Mock()
        
        # Mock catalog to raise exception
        tester.step_catalog.discover_config_classes.side_effect = Exception("Discovery failed")
        
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        result = tester._check_config_class_exists(mock_builder)
        
        assert result["passed"] is False
        assert "Discovery failed" in result["error"]

    def test_check_sagemaker_methods_exception_handling(self):
        """Test SageMaker methods check handles exceptions gracefully."""
        tester = UniversalStepBuilderTest()
        
        # Mock get_sagemaker_step_type to raise exception
        with patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type') as mock_get_type:
            mock_get_type.side_effect = Exception("Step type lookup failed")
            
            mock_builder = Mock()
            mock_builder.__name__ = "TestStepBuilder"
            
            result = tester._check_sagemaker_methods(mock_builder)
            
            # Should pass gracefully even with exception
            assert result["passed"] is True
            # FIXED: Category 14 - Match actual string content from implementation
            assert "Unknown step method validation" in result["note"]


class TestUniversalStepBuilderTestBackwardCompatibility:
    """Test backward compatibility methods."""

    def test_validate_specific_script_compatibility(self):
        """Test validate_specific_script backward compatibility method."""
        with patch.object(UniversalStepBuilderTest, 'run_validation_for_step') as mock_validation:
            mock_validation.return_value = {"step_name": "TestStep", "status": "PASSED"}
            
            tester = UniversalStepBuilderTest()
            result = tester.validate_specific_script("TestStep", skip_levels={1, 2})
            
            assert result["step_name"] == "TestStep"
            mock_validation.assert_called_once_with("TestStep")

    def test_discover_scripts_compatibility(self):
        """Test discover_scripts backward compatibility method."""
        with patch.object(UniversalStepBuilderTest, '_discover_all_steps') as mock_discover:
            mock_discover.return_value = ["Step1", "Step2", "Step3"]
            
            tester = UniversalStepBuilderTest()
            scripts = tester.discover_scripts()
            
            assert scripts == ["Step1", "Step2", "Step3"]
            mock_discover.assert_called_once()

    def test_get_validation_summary_compatibility(self):
        """Test get_validation_summary backward compatibility method."""
        with patch.object(UniversalStepBuilderTest, 'run_full_validation') as mock_full_validation:
            mock_full_validation.return_value = {
                "summary": {
                    "total_steps": 5,
                    "passed_steps": 4,
                    "failed_steps": 1,
                    "pass_rate": 80.0
                }
            }
            
            tester = UniversalStepBuilderTest()
            summary = tester.get_validation_summary()
            
            assert summary["total_steps"] == 5
            assert summary["passed_steps"] == 4
            assert summary["pass_rate"] == 80.0


class TestUniversalStepBuilderTestFullValidation:
    """Test full validation workflow."""

    @pytest.fixture
    def mock_tester_setup(self):
        """Set up tester with all necessary mocks."""
        # FIXED: Category 1 - Mock at actual import location (conditional import)
        with patch('src.cursus.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            tester = UniversalStepBuilderTest()
            tester.step_catalog = mock_catalog
            tester.step_catalog_available = True
            
            yield tester, mock_catalog

    def test_run_full_validation_success(self, mock_tester_setup):
        """Test full validation for all discovered steps."""
        tester, mock_catalog = mock_tester_setup
        
        # Mock step discovery
        mock_catalog.list_available_steps.return_value = ["Step1", "Step2"]
        
        # Mock individual step validation
        with patch.object(tester, 'run_validation_for_step') as mock_step_validation:
            mock_step_validation.side_effect = [
                {"step_name": "Step1", "overall_status": "PASSED"},
                {"step_name": "Step2", "overall_status": "ISSUES_FOUND"}
            ]
            
            results = tester.run_full_validation()
            
            assert results["validation_type"] == "full_builder_validation"
            assert results["total_steps"] == 2
            assert "Step1" in results["step_results"]
            assert "Step2" in results["step_results"]
            assert "summary" in results

    def test_run_full_validation_with_exception(self, mock_tester_setup):
        """Test full validation when individual step validation fails."""
        tester, mock_catalog = mock_tester_setup
        
        # Mock step discovery
        mock_catalog.list_available_steps.return_value = ["Step1", "Step2"]
        
        # Mock individual step validation with exception for one step
        with patch.object(tester, 'run_validation_for_step') as mock_step_validation:
            mock_step_validation.side_effect = [
                {"step_name": "Step1", "overall_status": "PASSED"},
                Exception("Step2 validation failed")
            ]
            
            results = tester.run_full_validation()
            
            assert results["total_steps"] == 2
            assert results["step_results"]["Step1"]["overall_status"] == "PASSED"
            assert results["step_results"]["Step2"]["overall_status"] == "ERROR"
            assert "Step2 validation failed" in results["step_results"]["Step2"]["error"]


class TestUniversalStepBuilderTestScoring:
    """Test scoring integration."""

    def test_calculate_scoring_with_streamlined_scorer(self):
        """Test scoring calculation using StreamlinedStepBuilderScorer."""
        tester = UniversalStepBuilderTest()
        
        validation_results = {
            "step_name": "TestStep",
            "components": {
                "alignment_validation": {"status": "COMPLETED"},
                "integration_testing": {"status": "COMPLETED"}
            }
        }
        
        # FIXED: Category 1 - Mock at actual import location (conditional import)
        with patch('src.cursus.validation.builders.reporting.scoring.StreamlinedStepBuilderScorer') as mock_scorer_class:
            mock_scorer = Mock()
            mock_scorer.generate_report.return_value = {
                "overall": {"score": 95.0, "rating": "Excellent"}
            }
            mock_scorer_class.return_value = mock_scorer
            
            result = tester._calculate_scoring(validation_results)
            
            assert result["overall"]["score"] == 95.0
            assert result["overall"]["rating"] == "Excellent"
            mock_scorer_class.assert_called_once_with(validation_results)

    def test_calculate_basic_scoring_fallback(self):
        """Test basic scoring calculation when StreamlinedStepBuilderScorer fails."""
        tester = UniversalStepBuilderTest()
        
        validation_results = {
            "overall_status": "PASSED",
            "components": {
                "alignment_validation": {"status": "COMPLETED"},
                "integration_testing": {"status": "COMPLETED"},
                "step_creation": {"status": "ISSUES_FOUND"}
            }
        }
        
        # FIXED: Category 1 - Mock at actual import location (conditional import)
        with patch('src.cursus.validation.builders.reporting.scoring.StreamlinedStepBuilderScorer') as mock_scorer_class:
            mock_scorer_class.side_effect = Exception("Scorer failed")
            
            result = tester._calculate_scoring(validation_results)
            
            assert result["overall"]["status"] == "PASSED"
            assert result["components"]["total"] == 3
            assert result["components"]["passed"] == 2
            assert result["scoring_approach"] == "basic_fallback"


class TestUniversalStepBuilderTestExportAndReporting:
    """Test export and reporting functionality."""

    def test_export_results_to_json(self):
        """Test exporting results to JSON format."""
        tester = UniversalStepBuilderTest()
        
        # Mock run_all_tests_with_full_report
        with patch.object(tester, 'run_all_tests_with_full_report') as mock_full_report:
            mock_results = {
                "validation_type": "full_builder_validation",
                "total_steps": 2,
                "step_results": {
                    "Step1": {"overall_status": "PASSED"},
                    "Step2": {"overall_status": "PASSED"}
                }
            }
            mock_full_report.return_value = mock_results
            
            # Test without output path (returns JSON string)
            json_content = tester.export_results_to_json()
            
            assert isinstance(json_content, str)
            parsed_data = json.loads(json_content)
            assert parsed_data["validation_type"] == "full_builder_validation"
            assert parsed_data["total_steps"] == 2

    def test_export_results_to_json_with_file(self):
        """Test exporting results to JSON file."""
        tester = UniversalStepBuilderTest()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_results.json"
            
            # Mock run_all_tests_with_full_report
            with patch.object(tester, 'run_all_tests_with_full_report') as mock_full_report:
                mock_results = {"test": "data"}
                mock_full_report.return_value = mock_results
                
                json_content = tester.export_results_to_json(str(output_path))
                
                # Verify file was created
                assert output_path.exists()
                
                # Verify content
                with open(output_path) as f:
                    saved_data = json.load(f)
                assert saved_data["test"] == "data"

    def test_generate_report_with_reporter(self):
        """Test report generation using StreamlinedBuilderTestReport."""
        tester = UniversalStepBuilderTest()
        
        # Mock validation results
        with patch.object(tester, 'run_validation_for_step') as mock_validation:
            mock_validation.return_value = {
                "step_name": "TestStep",
                "builder_class": "TestStepBuilder",
                "components": {
                    "alignment_validation": {"status": "COMPLETED"},
                    "integration_testing": {"status": "COMPLETED"}
                },
                "scoring": {"overall": {"score": 90.0}}
            }
            
            # Mock get_sagemaker_step_type
            with patch('src.cursus.validation.builders.universal_test.get_sagemaker_step_type') as mock_get_type:
                mock_get_type.return_value = "Training"
                
                # FIXED: Category 1 - Mock at actual import location (conditional import)
                with patch('src.cursus.validation.builders.reporting.builder_reporter.StreamlinedBuilderTestReport') as mock_report_class:
                    mock_report = Mock()
                    mock_report_class.return_value = mock_report
                    
                    result = tester.generate_report("TestStep")
                    
                    assert result == mock_report
                    mock_report_class.assert_called_once_with("TestStep", "TestStepBuilder", "Training")
                    mock_report.add_alignment_results.assert_called_once()
                    mock_report.add_integration_results.assert_called_once()
                    mock_report.add_scoring_data.assert_called_once()

    def test_generate_report_import_error_fallback(self):
        """Test report generation fallback when reporter import fails."""
        tester = UniversalStepBuilderTest()
        
        # Mock validation results
        with patch.object(tester, 'run_validation_for_step') as mock_validation:
            # FIXED: Category 15 - Test Data Issues (following enhanced guide)
            # Use valid step name from registry instead of non-existent "TestStep"
            validation_results = {"step_name": "XGBoostTraining", "status": "PASSED"}
            mock_validation.return_value = validation_results
            
            # FIXED: Category 1 - Mock at actual import location (conditional import)
            with patch('src.cursus.validation.builders.reporting.builder_reporter.StreamlinedBuilderTestReport') as mock_report_class:
                mock_report_class.side_effect = ImportError("Reporter not available")
                
                result = tester.generate_report("XGBoostTraining")
                
                # Should return validation results directly
                assert result == validation_results
