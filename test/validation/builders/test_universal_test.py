"""
Tests for UniversalStepBuilderTest class.

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Mock at import locations, not definition locations  
3. Test actual behavior, not assumed behavior
4. Use implementation-driven test design
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import tempfile
import json

# Import the class under test - CRITICAL: Mock at import location
from src.cursus.validation.builders.universal_test import (
    UniversalStepBuilderTest,
    TestUniversalStepBuilder
)


class TestUniversalStepBuilderTest:
    """Test suite for UniversalStepBuilderTest class."""

    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock step catalog for testing."""
        with patch('src.cursus.validation.builders.universal_test.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            
            # Configure mock catalog methods based on source code analysis
            mock_catalog.list_available_steps.return_value = ["TestStep", "XGBoostTraining"]
            
            # Create mock step info
            mock_step_info = Mock()
            mock_step_info.workspace_id = "core"
            mock_step_info.step_name = "TestStep"
            mock_step_info.registry_data = {"builder_step_name": "TestStepBuilder"}
            mock_step_info.sagemaker_step_type = "Processing"
            
            mock_catalog.get_step_info.return_value = mock_step_info
            
            yield mock_catalog_class, mock_catalog

    @pytest.fixture
    def mock_alignment_tester(self):
        """Create mock alignment tester for testing."""
        with patch('src.cursus.validation.builders.universal_test.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = Mock()
            mock_tester_class.return_value = mock_tester
            
            # Configure alignment validation results based on source code
            mock_tester.run_validation_for_step.return_value = {
                "step_name": "TestStep",
                "validation_type": "alignment_validation",
                "overall_status": "PASSED",
                "validation_results": {
                    "level1_interface": {"status": "PASSED"},
                    "level2_specification": {"status": "PASSED"}
                }
            }
            
            yield mock_tester_class, mock_tester

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            yield workspace_root

    def test_initialization_with_workspace_dirs(self, mock_step_catalog, mock_alignment_tester):
        """Test initialization with workspace directories."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Test initialization - matches source code constructor
        workspace_dirs = ["workspace1", "workspace2"]
        tester = UniversalStepBuilderTest(
            workspace_dirs=workspace_dirs,
            verbose=True,
            enable_scoring=True
        )
        
        # Verify initialization based on source code
        assert tester.workspace_dirs == workspace_dirs
        assert tester.verbose is True
        assert tester.enable_scoring is True
        assert tester.step_catalog_available is True
        assert tester.alignment_available is True
        
        # Verify step catalog was initialized with correct parameters
        mock_catalog_class.assert_called_once_with(workspace_dirs=workspace_dirs)
        mock_tester_class.assert_called_once_with(workspace_dirs=workspace_dirs)

    def test_initialization_without_step_catalog(self, mock_alignment_tester):
        """Test initialization when step catalog is not available."""
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Mock ImportError for step catalog
        with patch('src.cursus.validation.builders.universal_test.StepCatalog', side_effect=ImportError("No module")):
            tester = UniversalStepBuilderTest(verbose=True)
            
            # Verify fallback behavior from source code
            assert tester.step_catalog is None
            assert tester.step_catalog_available is False
            assert tester.alignment_available is True

    def test_initialization_without_alignment_tester(self, mock_step_catalog):
        """Test initialization when alignment tester is not available."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        
        # Mock ImportError for alignment tester
        with patch('src.cursus.validation.builders.universal_test.UnifiedAlignmentTester', side_effect=ImportError("No module")):
            tester = UniversalStepBuilderTest(verbose=True)
            
            # Verify fallback behavior from source code
            assert tester.alignment_tester is None
            assert tester.alignment_available is False
            assert tester.step_catalog_available is True

    def test_run_validation_for_step_with_alignment(self, mock_step_catalog, mock_alignment_tester):
        """Test run_validation_for_step with alignment system available."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Create mock builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestStepBuilder"
        
        # Mock _get_builder_class_from_catalog to return our mock
        tester = UniversalStepBuilderTest()
        
        with patch.object(tester, '_get_builder_class_from_catalog', return_value=mock_builder_class):
            with patch.object(tester, '_run_comprehensive_validation_for_step') as mock_comprehensive:
                mock_comprehensive.return_value = {
                    "step_name": "TestStep",
                    "validation_type": "comprehensive_builder_validation",
                    "overall_status": "PASSED"
                }
                
                result = tester.run_validation_for_step("TestStep")
                
                # Verify result structure matches source code
                assert result["step_name"] == "TestStep"
                assert result["validation_type"] == "comprehensive_builder_validation"
                assert result["overall_status"] == "PASSED"
                
                # Verify methods were called correctly
                mock_comprehensive.assert_called_once_with("TestStep", mock_builder_class)

    def test_run_validation_for_step_no_builder_class(self, mock_step_catalog, mock_alignment_tester):
        """Test run_validation_for_step when no builder class is found."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest()
        
        # Mock _get_builder_class_from_catalog to return None
        with patch.object(tester, '_get_builder_class_from_catalog', return_value=None):
            result = tester.run_validation_for_step("NonExistentStep")
            
            # Verify error handling from source code
            assert result["step_name"] == "NonExistentStep"
            assert result["validation_type"] == "comprehensive_builder_validation"
            assert result["overall_status"] == "ERROR"
            assert "No builder class found" in result["error"]

    def test_run_alignment_validation(self, mock_step_catalog, mock_alignment_tester):
        """Test _run_alignment_validation method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest()
        
        result = tester._run_alignment_validation("TestStep")
        
        # Verify result structure from source code
        assert result["status"] == "COMPLETED"
        assert result["validation_approach"] == "alignment_system"
        assert "results" in result
        assert result["levels_covered"] == ["interface_compliance", "specification_alignment"]
        
        # Verify alignment tester was called
        mock_tester.run_validation_for_step.assert_called_once_with("TestStep")

    def test_run_fallback_core_validation(self, mock_step_catalog, mock_alignment_tester):
        """Test _run_fallback_core_validation method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Create mock builder class with required methods
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestStepBuilder"
        
        # Configure inheritance check - source code uses issubclass
        from src.cursus.core.base.builder_base import StepBuilderBase
        with patch('builtins.issubclass', return_value=True):
            # Configure hasattr checks for required methods
            with patch('builtins.hasattr') as mock_hasattr:
                mock_hasattr.side_effect = lambda obj, attr: attr in ["validate_configuration", "create_step"]
                
                tester = UniversalStepBuilderTest()
                result = tester._run_fallback_core_validation("TestStep", mock_builder_class)
                
                # Verify result structure from source code
                assert result["status"] == "COMPLETED"
                assert result["validation_approach"] == "fallback_core"
                assert "results" in result
                assert result["note"] == "Using fallback validation - alignment system not available"
                
                # Verify inheritance and method checks
                assert result["results"]["inheritance_check"]["passed"] is True
                assert result["results"]["method_checks"]["validate_configuration"]["passed"] is True
                assert result["results"]["method_checks"]["create_step"]["passed"] is True

    def test_test_integration_capabilities(self, mock_step_catalog, mock_alignment_tester):
        """Test _test_integration_capabilities method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestStepBuilder"
        
        tester = UniversalStepBuilderTest()
        
        # Mock the individual check methods
        with patch.object(tester, '_check_dependency_resolution', return_value={"passed": True}):
            with patch.object(tester, '_check_cache_configuration', return_value={"passed": True}):
                with patch.object(tester, '_check_step_instantiation', return_value={"passed": True}):
                    result = tester._test_integration_capabilities("TestStep", mock_builder_class)
                    
                    # Verify result structure from source code
                    assert result["status"] == "COMPLETED"
                    assert "checks" in result
                    assert result["integration_type"] == "capability_validation"
                    
                    # Verify all checks were performed
                    assert "dependency_resolution" in result["checks"]
                    assert "cache_configuration" in result["checks"]
                    assert "step_instantiation" in result["checks"]

    def test_discover_all_steps_with_catalog(self, mock_step_catalog, mock_alignment_tester):
        """Test _discover_all_steps with step catalog available."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest(verbose=True)
        
        steps = tester._discover_all_steps()
        
        # Verify steps were discovered from catalog
        assert steps == ["TestStep", "XGBoostTraining"]
        mock_catalog.list_available_steps.assert_called_once()

    def test_discover_all_steps_fallback_to_registry(self, mock_alignment_tester):
        """Test _discover_all_steps fallback to registry when catalog fails."""
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Mock step catalog to raise exception
        with patch('src.cursus.validation.builders.universal_test.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog
            mock_catalog.list_available_steps.side_effect = Exception("Catalog failed")
            
            # Mock STEP_NAMES registry
            with patch('src.cursus.validation.builders.universal_test.STEP_NAMES', {"Step1": {}, "Step2": {}}):
                tester = UniversalStepBuilderTest(verbose=True)
                steps = tester._discover_all_steps()
                
                # Verify fallback to registry
                assert set(steps) == {"Step1", "Step2"}

    def test_run_full_validation(self, mock_step_catalog, mock_alignment_tester):
        """Test run_full_validation method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest(verbose=True)
        
        # Mock run_validation_for_step to return consistent results
        with patch.object(tester, 'run_validation_for_step') as mock_validate:
            mock_validate.return_value = {
                "step_name": "TestStep",
                "overall_status": "PASSED"
            }
            
            result = tester.run_full_validation()
            
            # Verify result structure from source code
            assert result["validation_type"] == "full_builder_validation"
            assert result["total_steps"] == 2  # TestStep and XGBoostTraining
            assert "step_results" in result
            assert "summary" in result
            
            # Verify each discovered step was validated
            assert mock_validate.call_count == 2
            mock_validate.assert_any_call("TestStep")
            mock_validate.assert_any_call("XGBoostTraining")

    def test_generate_validation_summary(self, mock_step_catalog, mock_alignment_tester):
        """Test _generate_validation_summary method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest()
        
        # Create mock step results
        step_results = {
            "Step1": {"overall_status": "PASSED"},
            "Step2": {"overall_status": "FAILED"},
            "Step3": {"overall_status": "ISSUES_FOUND"}
        }
        
        summary = tester._generate_validation_summary(step_results)
        
        # Verify summary calculation from source code
        assert summary["total_steps"] == 3
        assert summary["passed_steps"] == 1
        assert summary["failed_steps"] == 1
        assert summary["issues_steps"] == 1
        assert summary["pass_rate"] == pytest.approx(33.33, rel=1e-2)

    def test_from_builder_class_method(self, mock_step_catalog, mock_alignment_tester):
        """Test from_builder_class class method for backward compatibility."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Create mock builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "TestStepBuilder"
        
        # Test class method
        tester = UniversalStepBuilderTest.from_builder_class(
            mock_builder_class,
            workspace_dirs=["test_workspace"],
            verbose=True
        )
        
        # Verify backward compatibility attributes from source code
        assert tester.builder_class == mock_builder_class
        assert tester.single_builder_mode is True
        assert tester.workspace_dirs == ["test_workspace"]
        assert tester.verbose is True

    def test_test_all_builders_by_type(self, mock_step_catalog, mock_alignment_tester):
        """Test test_all_builders_by_type class method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        # Mock STEP_NAMES registry with Processing steps
        mock_step_names = {
            "ProcessingStep1": {"sagemaker_step_type": "Processing"},
            "ProcessingStep2": {"sagemaker_step_type": "Processing"},
            "TrainingStep1": {"sagemaker_step_type": "Training"}
        }
        
        with patch('src.cursus.validation.builders.universal_test.STEP_NAMES', mock_step_names):
            # Mock the instance creation and validation
            with patch.object(UniversalStepBuilderTest, 'run_validation_for_step') as mock_validate:
                mock_validate.return_value = {"step_name": "ProcessingStep1", "overall_status": "PASSED"}
                
                results = UniversalStepBuilderTest.test_all_builders_by_type(
                    "Processing",
                    verbose=True,
                    enable_scoring=True
                )
                
                # Verify only Processing steps were tested
                assert "ProcessingStep1" in results
                assert "ProcessingStep2" in results
                assert "TrainingStep1" not in results

    def test_run_all_tests_with_scoring_integration(self, mock_step_catalog, mock_alignment_tester):
        """Test run_all_tests method with scoring integration."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest(enable_scoring=True)
        
        # Mock run_full_validation to return test results
        with patch.object(tester, 'run_full_validation') as mock_full_validation:
            mock_full_validation.return_value = {
                "validation_type": "full_builder_validation",
                "total_steps": 1,
                "step_results": {"TestStep": {"overall_status": "PASSED"}},
                "summary": {"pass_rate": 100.0}
            }
            
            result = tester.run_all_tests(include_scoring=True)
            
            # Verify run_full_validation was called
            mock_full_validation.assert_called_once()
            
            # Verify result structure
            assert "validation_type" in result
            assert "total_steps" in result
            assert "step_results" in result

    def test_export_results_to_json(self, mock_step_catalog, mock_alignment_tester, temp_workspace):
        """Test export_results_to_json method."""
        mock_catalog_class, mock_catalog = mock_step_catalog
        mock_tester_class, mock_tester = mock_alignment_tester
        
        tester = UniversalStepBuilderTest()
        
        # Mock run_all_tests_with_full_report
        with patch.object(tester, 'run_all_tests_with_full_report') as mock_full_report:
            mock_results = {
                "validation_type": "full_builder_validation",
                "total_steps": 1,
                "step_results": {"TestStep": {"overall_status": "PASSED"}}
            }
            mock_full_report.return_value = mock_results
            
            # Test export to file
            output_path = temp_workspace / "test_results.json"
            json_content = tester.export_results_to_json(str(output_path))
            
            # Verify file was created and contains valid JSON
            assert output_path.exists()
            parsed_json = json.loads(json_content)
            assert parsed_json["validation_type"] == "full_builder_validation"
            assert parsed_json["total_steps"] == 1


class TestTestUniversalStepBuilder:
    """Test suite for the unittest compatibility class."""

    def test_refactored_initialization(self):
        """Test that the refactored system initializes correctly."""
        # Mock the dependencies to avoid import errors
        with patch('src.cursus.validation.builders.universal_test.StepCatalog'):
            with patch('src.cursus.validation.builders.universal_test.UnifiedAlignmentTester'):
                # Test new multi-builder mode
                tester = UniversalStepBuilderTest(workspace_dirs=["."], verbose=False)
                
                # Verify initialization - note: single_builder_mode is not set in new constructor
                assert tester.workspace_dirs == ["."]
                assert tester.verbose is False

    def test_from_builder_class_method_compatibility(self):
        """Test the from_builder_class class method for backward compatibility."""
        # Create mock builder class
        from src.cursus.core.base.builder_base import StepBuilderBase
        
        class MockBuilder(StepBuilderBase):
            def validate_configuration(self):
                pass
            def create_step(self):
                pass
        
        with patch('src.cursus.validation.builders.universal_test.StepCatalog'):
            with patch('src.cursus.validation.builders.universal_test.UnifiedAlignmentTester'):
                tester = UniversalStepBuilderTest.from_builder_class(MockBuilder)
                
                # Verify backward compatibility attributes
                assert tester.single_builder_mode is True
                assert tester.builder_class == MockBuilder

    def test_backward_compatibility_methods(self):
        """Test that backward compatibility methods work."""
        with patch('src.cursus.validation.builders.universal_test.StepCatalog'):
            with patch('src.cursus.validation.builders.universal_test.UnifiedAlignmentTester'):
                tester = UniversalStepBuilderTest(workspace_dirs=["."], verbose=False)
                
                # Mock the underlying methods
                with patch.object(tester, '_discover_all_steps', return_value=["TestStep"]):
                    with patch.object(tester, 'run_full_validation', return_value={"summary": {"pass_rate": 100.0}}):
                        # Test discovery method
                        steps = tester.discover_scripts()
                        assert isinstance(steps, list)
                        assert steps == ["TestStep"]
                        
                        # Test summary method
                        summary = tester.get_validation_summary()
                        assert isinstance(summary, dict)
                        assert summary["pass_rate"] == 100.0


# Integration tests
class TestUniversalStepBuilderTestIntegration:
    """Integration tests for UniversalStepBuilderTest."""

    @pytest.fixture
    def real_temp_workspace(self):
        """Create a realistic temporary workspace for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create realistic directory structure
            dev_workspace = workspace_root / "dev1"
            dev_workspace.mkdir(parents=True)
            
            # Create component directories
            for component_type in ["scripts", "contracts", "specs", "builders"]:
                component_dir = dev_workspace / component_type
                component_dir.mkdir()
                
                # Create sample files
                sample_file = component_dir / f"sample_{component_type[:-1]}.py"
                sample_file.write_text(f"# Sample {component_type[:-1]} file")
            
            yield workspace_root

    def test_initialization_integration(self, real_temp_workspace):
        """Test initialization with realistic workspace structure."""
        # Test that initialization works with real workspace structure
        # This tests the integration without mocking everything
        try:
            tester = UniversalStepBuilderTest(
                workspace_dirs=[str(real_temp_workspace)],
                verbose=False
            )
            
            # Basic assertions that don't depend on external systems
            assert tester.workspace_dirs == [str(real_temp_workspace)]
            assert tester.verbose is False
            assert hasattr(tester, 'step_catalog_available')
            assert hasattr(tester, 'alignment_available')
            
        except ImportError:
            # If dependencies are not available, that's expected in test environment
            pytest.skip("Dependencies not available for integration test")

    def test_error_handling_integration(self):
        """Test error handling in realistic scenarios."""
        # Test with invalid workspace directories
        tester = UniversalStepBuilderTest(
            workspace_dirs=["/nonexistent/path"],
            verbose=False
        )
        
        # Should not crash during initialization
        assert tester.workspace_dirs == ["/nonexistent/path"]
        
        # Test validation with non-existent step
        try:
            result = tester.run_validation_for_step("NonExistentStep")
            
            # Should return error result, not crash
            assert isinstance(result, dict)
            assert "step_name" in result
            assert result["step_name"] == "NonExistentStep"
            
        except Exception as e:
            # If it fails, it should be a controlled failure
            assert "NonExistentStep" in str(e) or "No builder class found" in str(e)
