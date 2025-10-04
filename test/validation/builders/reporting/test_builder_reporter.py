"""
Pytest tests for StreamlinedBuilderTestReport and StreamlinedBuilderTestReporter classes.

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Mock at import locations, not definition locations  
3. Match test behavior to actual implementation behavior
4. Use realistic fixtures and data structures
5. Test both success and failure scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Import the classes under test
from src.cursus.validation.builders.reporting.builder_reporter import (
    StreamlinedBuilderTestReport,
    StreamlinedBuilderTestReporter
)


class TestStreamlinedBuilderTestReportInitialization:
    """Test StreamlinedBuilderTestReport initialization."""

    def test_init_with_basic_parameters(self):
        """Test initialization with basic parameters."""
        report = StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder", 
            sagemaker_step_type="Training"
        )
        
        assert report.builder_name == "TestStep"
        assert report.builder_class == "TestStepBuilder"
        assert report.sagemaker_step_type == "Training"
        assert isinstance(report.validation_timestamp, datetime)
        assert report.test_results == {}
        assert report.alignment_results is None
        assert report.integration_results is None
        assert report.scoring_data is None
        assert report.metadata == {}

    def test_init_timestamp_is_recent(self):
        """Test that initialization timestamp is recent."""
        before_init = datetime.now()
        report = StreamlinedBuilderTestReport("Test", "TestBuilder", "Processing")
        after_init = datetime.now()
        
        assert before_init <= report.validation_timestamp <= after_init


class TestStreamlinedBuilderTestReportDataManagement:
    """Test data management methods of StreamlinedBuilderTestReport."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        return StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )

    def test_add_alignment_results(self, sample_report):
        """Test adding alignment validation results."""
        alignment_results = {
            "overall_status": "PASSED",
            "validation_results": {
                "level_1": {"result": {"passed": True, "issues": []}}
            }
        }
        
        sample_report.add_alignment_results(alignment_results)
        
        assert sample_report.alignment_results == alignment_results

    def test_add_integration_results(self, sample_report):
        """Test adding integration testing results."""
        integration_results = {
            "status": "COMPLETED",
            "checks": {
                "dependency_resolution": {"passed": True},
                "step_instantiation": {"passed": True}
            }
        }
        
        sample_report.add_integration_results(integration_results)
        
        assert sample_report.integration_results == integration_results

    def test_add_scoring_data(self, sample_report):
        """Test adding scoring data."""
        scoring_data = {
            "overall": {"score": 95.0, "rating": "Excellent"},
            "components": {
                "alignment_validation": {"score": 90.0},
                "integration_testing": {"score": 100.0}
            }
        }
        
        sample_report.add_scoring_data(scoring_data)
        
        assert sample_report.scoring_data == scoring_data


class TestStreamlinedBuilderTestReportStatusDetermination:
    """Test status determination methods."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        return StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )

    def test_get_overall_status_passing_with_alignment_and_integration(self, sample_report):
        """Test overall status when both alignment and integration pass."""
        sample_report.add_alignment_results({
            "overall_status": "PASSED"
        })
        sample_report.add_integration_results({
            "status": "COMPLETED"
        })
        
        status = sample_report.get_overall_status()
        assert status == "PASSING"

    def test_get_overall_status_mostly_passing_with_issues(self, sample_report):
        """Test overall status when alignment passes but integration has issues."""
        sample_report.add_alignment_results({
            "overall_status": "PASSED"
        })
        sample_report.add_integration_results({
            "status": "ISSUES_FOUND"
        })
        
        status = sample_report.get_overall_status()
        assert status == "MOSTLY_PASSING"

    def test_get_overall_status_partially_passing_with_integration_error(self, sample_report):
        """Test overall status when alignment passes but integration fails."""
        sample_report.add_alignment_results({
            "overall_status": "PASSED"
        })
        sample_report.add_integration_results({
            "status": "ERROR"
        })
        
        status = sample_report.get_overall_status()
        assert status == "PARTIALLY_PASSING"

    def test_get_overall_status_failing_with_alignment_failure(self, sample_report):
        """Test overall status when alignment fails."""
        sample_report.add_alignment_results({
            "overall_status": "FAILED"
        })
        
        status = sample_report.get_overall_status()
        assert status == "FAILING"

    def test_get_overall_status_unknown_without_results(self, sample_report):
        """Test overall status when no results are available."""
        status = sample_report.get_overall_status()
        assert status == "UNKNOWN"

    def test_get_overall_status_passing_with_integration_only(self, sample_report):
        """Test overall status with only integration results."""
        sample_report.add_integration_results({
            "status": "COMPLETED"
        })
        
        status = sample_report.get_overall_status()
        assert status == "PASSING"

    def test_get_overall_status_failing_with_integration_only(self, sample_report):
        """Test overall status with only failing integration results."""
        sample_report.add_integration_results({
            "status": "ERROR"
        })
        
        status = sample_report.get_overall_status()
        assert status == "FAILING"


class TestStreamlinedBuilderTestReportQualityMetrics:
    """Test quality metrics methods."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        return StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )

    def test_get_quality_score_with_scoring_data(self, sample_report):
        """Test getting quality score when scoring data is available."""
        sample_report.add_scoring_data({
            "overall": {"score": 87.5, "rating": "Good"}
        })
        
        score = sample_report.get_quality_score()
        assert score == 87.5

    def test_get_quality_score_without_scoring_data(self, sample_report):
        """Test getting quality score when no scoring data is available."""
        score = sample_report.get_quality_score()
        assert score == 0.0

    def test_get_quality_rating_with_scoring_data(self, sample_report):
        """Test getting quality rating when scoring data is available."""
        sample_report.add_scoring_data({
            "overall": {"score": 92.0, "rating": "Excellent"}
        })
        
        rating = sample_report.get_quality_rating()
        assert rating == "Excellent"

    def test_get_quality_rating_without_scoring_data(self, sample_report):
        """Test getting quality rating when no scoring data is available."""
        rating = sample_report.get_quality_rating()
        assert rating == "Unknown"

    def test_is_passing_true(self, sample_report):
        """Test is_passing returns True for passing statuses."""
        sample_report.add_alignment_results({"overall_status": "PASSED"})
        sample_report.add_integration_results({"status": "COMPLETED"})
        
        assert sample_report.is_passing() is True

    def test_is_passing_mostly_passing(self, sample_report):
        """Test is_passing returns True for mostly passing status."""
        sample_report.add_alignment_results({"overall_status": "PASSED"})
        sample_report.add_integration_results({"status": "ISSUES_FOUND"})
        
        assert sample_report.is_passing() is True

    def test_is_passing_false(self, sample_report):
        """Test is_passing returns False for failing statuses."""
        sample_report.add_alignment_results({"overall_status": "FAILED"})
        
        assert sample_report.is_passing() is False


class TestStreamlinedBuilderTestReportIssueExtraction:
    """Test issue extraction methods."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        return StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )

    def test_get_critical_issues_from_alignment_results(self, sample_report):
        """Test extracting critical issues from alignment results."""
        sample_report.add_alignment_results({
            "failed_tests": [
                {"error": "Missing required method: create_step"},
                {"error": "Configuration validation failed"}
            ]
        })
        
        issues = sample_report.get_critical_issues()
        
        assert len(issues) == 2
        assert "Missing required method: create_step" in issues
        assert "Configuration validation failed" in issues

    def test_get_critical_issues_from_integration_results(self, sample_report):
        """Test extracting critical issues from integration results."""
        sample_report.add_integration_results({
            "error": "Step instantiation failed"
        })
        
        issues = sample_report.get_critical_issues()
        
        assert len(issues) == 1
        assert "Step instantiation failed" in issues

    def test_get_critical_issues_from_both_sources(self, sample_report):
        """Test extracting critical issues from both alignment and integration results."""
        sample_report.add_alignment_results({
            "failed_tests": [
                {"error": "Alignment issue"}
            ]
        })
        sample_report.add_integration_results({
            "error": "Integration issue"
        })
        
        issues = sample_report.get_critical_issues()
        
        assert len(issues) == 2
        assert "Alignment issue" in issues
        assert "Integration issue" in issues

    def test_get_critical_issues_empty_when_no_issues(self, sample_report):
        """Test that no critical issues are returned when there are none."""
        sample_report.add_alignment_results({
            "overall_status": "PASSED"
        })
        sample_report.add_integration_results({
            "status": "COMPLETED"
        })
        
        issues = sample_report.get_critical_issues()
        
        assert len(issues) == 0

    def test_get_critical_issues_handles_non_dict_failed_tests(self, sample_report):
        """Test handling of non-dict items in failed_tests."""
        sample_report.add_alignment_results({
            "failed_tests": [
                "string_error",  # Not a dict
                {"error": "Valid error"}
            ]
        })
        
        issues = sample_report.get_critical_issues()
        
        assert len(issues) == 1
        assert "Valid error" in issues


class TestStreamlinedBuilderTestReportExportAndSerialization:
    """Test export and serialization methods."""

    @pytest.fixture
    def complete_report(self):
        """Create a complete report with all data for testing."""
        report = StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )
        
        report.add_alignment_results({
            "overall_status": "PASSED",
            "validation_results": {"level_1": {"result": {"passed": True}}}
        })
        
        report.add_integration_results({
            "status": "COMPLETED",
            "checks": {"dependency_resolution": {"passed": True}}
        })
        
        report.add_scoring_data({
            "overall": {"score": 95.0, "rating": "Excellent"}
        })
        
        report.metadata["test_key"] = "test_value"
        
        return report

    def test_export_to_json_structure(self, complete_report):
        """Test JSON export structure and content."""
        json_str = complete_report.export_to_json()
        data = json.loads(json_str)
        
        # Check top-level structure
        assert "builder_name" in data
        assert "builder_class" in data
        assert "sagemaker_step_type" in data
        assert "validation_timestamp" in data
        assert "overall_status" in data
        assert "quality_score" in data
        assert "quality_rating" in data
        assert "is_passing" in data
        assert "alignment_validation" in data
        assert "integration_testing" in data
        assert "scoring" in data
        assert "metadata" in data
        
        # Check specific values
        assert data["builder_name"] == "TestStep"
        assert data["builder_class"] == "TestStepBuilder"
        assert data["sagemaker_step_type"] == "Training"
        assert data["overall_status"] == "PASSING"
        assert data["quality_score"] == 95.0
        assert data["quality_rating"] == "Excellent"
        assert data["is_passing"] is True
        
        # Check metadata
        assert data["metadata"]["builder_name"] == "TestStep"
        assert data["metadata"]["validator_version"] == "2.0.0"
        assert data["metadata"]["test_framework"] == "UniversalStepBuilderTest"
        assert data["metadata"]["reporting_approach"] == "streamlined_with_alignment_integration"
        assert data["metadata"]["test_key"] == "test_value"

    def test_export_to_json_minimal_data(self):
        """Test JSON export with minimal data."""
        report = StreamlinedBuilderTestReport(
            builder_name="MinimalStep",
            builder_class="MinimalStepBuilder",
            sagemaker_step_type="Processing"
        )
        
        json_str = report.export_to_json()
        data = json.loads(json_str)
        
        assert data["builder_name"] == "MinimalStep"
        assert data["overall_status"] == "UNKNOWN"
        assert data["quality_score"] == 0.0
        assert data["quality_rating"] == "Unknown"
        assert data["is_passing"] is False
        assert data["alignment_validation"] == {}
        assert data["integration_testing"] == {}
        assert data["scoring"] == {}

    def test_save_to_file(self, complete_report):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.json"
            
            complete_report.save_to_file(output_path)
            
            # Verify file exists
            assert output_path.exists()
            
            # Verify content
            with open(output_path) as f:
                data = json.load(f)
            
            assert data["builder_name"] == "TestStep"
            assert data["overall_status"] == "PASSING"

    def test_save_to_file_creates_directories(self, complete_report):
        """Test that save_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "directory" / "test_report.json"
            
            complete_report.save_to_file(nested_path)
            
            # Verify file exists and directories were created
            assert nested_path.exists()
            assert nested_path.parent.exists()


class TestStreamlinedBuilderTestReportPrintSummary:
    """Test print summary functionality."""

    @pytest.fixture
    def complete_report(self):
        """Create a complete report for testing."""
        report = StreamlinedBuilderTestReport(
            builder_name="TestStep",
            builder_class="TestStepBuilder",
            sagemaker_step_type="Training"
        )
        
        report.add_alignment_results({
            "overall_status": "PASSED",
            "failed_tests": [
                {"name": "test1", "error": "Minor issue"}
            ]
        })
        
        report.add_integration_results({
            "status": "COMPLETED",
            "checks": {
                "dependency_resolution": {"passed": True},
                "cache_configuration": {"passed": False}
            }
        })
        
        report.add_scoring_data({
            "overall": {"score": 87.5, "rating": "Good"},
            "levels": {
                "level1_interface": {"score": 90.0, "passed": 8, "total": 10},
                "level2_specification": {"score": 85.0, "passed": 7, "total": 8}
            }
        })
        
        return report

    def test_print_summary_output(self, complete_report, capsys):
        """Test print summary output content."""
        complete_report.print_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check header
        assert "STEP BUILDER TEST REPORT: TestStep" in output
        
        # Check basic info
        assert "Builder: TestStepBuilder" in output
        assert "SageMaker Step Type: Training" in output
        assert "Overall Status: ✅ PASSING" in output
        
        # Check quality score
        assert "📊 Quality Score: 87.5/100 - Good" in output
        
        # Check alignment validation
        assert "🔍 Alignment Validation:" in output
        assert "Status: PASSED" in output
        assert "Failed Tests: 1" in output
        assert "• test1" in output
        
        # Check integration testing
        assert "🔧 Integration Testing:" in output
        assert "Status: COMPLETED" in output
        assert "✅ Dependency Resolution" in output
        assert "❌ Cache Configuration" in output
        
        # Check scoring breakdown
        assert "📈 Quality Score Breakdown:" in output
        assert "L1 Interface: 90.0/100 (8/10 tests)" in output
        assert "L2 Specification: 85.0/100 (7/8 tests)" in output

    def test_print_summary_failing_status(self, capsys):
        """Test print summary with failing status."""
        report = StreamlinedBuilderTestReport(
            builder_name="FailingStep",
            builder_class="FailingStepBuilder",
            sagemaker_step_type="Processing"
        )
        
        report.add_alignment_results({"overall_status": "FAILED"})
        
        report.print_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Overall Status: ❌ FAILING" in output

    def test_print_summary_with_critical_issues(self, capsys):
        """Test print summary with critical issues."""
        report = StreamlinedBuilderTestReport(
            builder_name="IssueStep",
            builder_class="IssueStepBuilder",
            sagemaker_step_type="Transform"
        )
        
        report.add_alignment_results({
            "failed_tests": [
                {"error": f"Critical issue {i}"} for i in range(7)
            ]
        })
        
        report.print_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "🚨 CRITICAL ISSUES (7):" in output
        assert "Critical issue 0" in output
        assert "... and 2 more" in output  # Should show first 5 + "and X more"

    def test_print_summary_minimal_data(self, capsys):
        """Test print summary with minimal data."""
        report = StreamlinedBuilderTestReport(
            builder_name="MinimalStep",
            builder_class="MinimalStepBuilder",
            sagemaker_step_type="CreateModel"
        )
        
        report.print_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "STEP BUILDER TEST REPORT: MinimalStep" in output
        assert "Overall Status: ❌ UNKNOWN" in output
        # Should not have sections for missing data


class TestStreamlinedBuilderTestReporterInitialization:
    """Test StreamlinedBuilderTestReporter initialization."""

    def test_init_with_default_output_dir(self):
        """Test initialization with default output directory."""
        reporter = StreamlinedBuilderTestReporter()
        
        expected_path = Path.cwd() / "test" / "validation" / "builders" / "reports"
        assert reporter.output_dir == expected_path

    def test_init_with_custom_output_dir(self):
        """Test initialization with custom output directory."""
        custom_dir = Path("/custom/output/dir")
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            reporter = StreamlinedBuilderTestReporter(output_dir=custom_dir)
            
            assert reporter.output_dir == custom_dir
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestStreamlinedBuilderTestReporterStepNameInference:
    """Test step name inference methods."""

    @pytest.fixture
    def reporter(self):
        """Create a reporter for testing."""
        return StreamlinedBuilderTestReporter()

    def test_infer_step_name_with_suffix(self, reporter):
        """Test step name inference from class name with StepBuilder suffix."""
        mock_builder = Mock()
        mock_builder.__name__ = "XGBoostTrainingStepBuilder"
        
        step_name = reporter._infer_step_name(mock_builder)
        
        assert step_name == "XGBoostTraining"

    def test_infer_step_name_without_suffix(self, reporter):
        """Test step name inference from class name without suffix."""
        mock_builder = Mock()
        mock_builder.__name__ = "CustomBuilder"
        
        step_name = reporter._infer_step_name(mock_builder)
        
        assert step_name == "CustomBuilder"

    def test_infer_step_name_with_registry_lookup(self, reporter):
        """Test step name inference with registry lookup."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        
        # FIXED: Mock STEP_NAMES at actual import location (Category 1: Mock Path Issues)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.items.return_value = [
                ("FoundStep", {"builder_step_name": "TestStepBuilder"}),
                ("OtherStep", {"builder_step_name": "OtherStepBuilder"})
            ]
            
            step_name = reporter._infer_step_name(mock_builder)
            
            assert step_name == "FoundStep"

    def test_infer_step_name_no_registry_match(self, reporter):
        """Test step name inference when no registry match is found."""
        mock_builder = Mock()
        mock_builder.__name__ = "UnknownStepBuilder"
        
        # FIXED: Mock STEP_NAMES at actual import location (Category 1: Mock Path Issues)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.items.return_value = [
                ("OtherStep", {"builder_step_name": "OtherStepBuilder"})
            ]
            
            step_name = reporter._infer_step_name(mock_builder)
            
            # FIXED: Category 4 - Implementation Expectations (following enhanced guide)
            # Read source: step_name = class_name[:-11] if class_name.endswith("StepBuilder") else class_name
            # For "UnknownStepBuilder", removes "StepBuilder" suffix = "Unknown"
            assert step_name == "Unknown"  # Actual implementation behavior


class TestStreamlinedBuilderTestReporterBuilderLoading:
    """Test builder class loading methods."""

    @pytest.fixture
    def reporter(self):
        """Create a reporter for testing."""
        return StreamlinedBuilderTestReporter()

    def test_load_builder_class_success(self, reporter):
        """Test successful builder class loading."""
        mock_catalog = Mock()
        mock_builder_class = Mock()
        mock_catalog.load_builder_class.return_value = mock_builder_class
        
        # Mock StepCatalog creation at actual import location
        with patch('src.cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.return_value = mock_catalog
            
            result = reporter._load_builder_class("TestStep")
            
            assert result == mock_builder_class
            mock_catalog.load_builder_class.assert_called_once_with("TestStep")

    def test_load_builder_class_not_found(self, reporter):
        """Test builder class loading when class is not found."""
        mock_catalog = Mock()
        mock_catalog.load_builder_class.return_value = None
        
        with patch('src.cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.return_value = mock_catalog
            
            result = reporter._load_builder_class("NonExistentStep")
            
            assert result is None

    def test_load_builder_class_exception(self, reporter):
        """Test builder class loading when exception occurs."""
        # FIXED: Category 1 - Mock at actual import location (nested module import)
        with patch('src.cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog failed")
            
            result = reporter._load_builder_class("TestStep")
            
            assert result is None

    def test_load_builder_class_caches_catalog(self, reporter):
        """Test that builder class loading caches the step catalog."""
        mock_catalog = Mock()
        mock_catalog.load_builder_class.return_value = Mock()
        
        # FIXED: Category 1 - Mock at actual import location (nested module import)
        with patch('src.cursus.step_catalog.step_catalog.StepCatalog') as mock_catalog_class:
            mock_catalog_class.return_value = mock_catalog
            
            # First call
            reporter._load_builder_class("TestStep1")
            # Second call
            reporter._load_builder_class("TestStep2")
            
            # StepCatalog should only be created once
            mock_catalog_class.assert_called_once()
            assert reporter._step_catalog == mock_catalog


class TestStreamlinedBuilderTestReporterTestAndReport:
    """Test test and report methods."""

    @pytest.fixture
    def reporter(self):
        """Create a reporter for testing."""
        return StreamlinedBuilderTestReporter()

    @pytest.fixture
    def mock_builder_class(self):
        """Create a mock builder class."""
        mock_builder = Mock()
        mock_builder.__name__ = "TestStepBuilder"
        return mock_builder

    def test_test_and_report_builder_success(self, reporter, mock_builder_class):
        """Test successful test and report generation."""
        # FIXED: Category 1 - Mock STEP_NAMES at actual import location (registry import)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.get.return_value = {"sagemaker_step_type": "Training"}
            
            # FIXED: Category 1 - Relative Import Pattern (following enhanced guide STEP 2D)
            # Source: from ..universal_test import UniversalStepBuilderTest
            # Convert relative ..universal_test to absolute: src.cursus.validation.builders.universal_test
            with patch('src.cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_tester_class:
                mock_tester = Mock()
                mock_tester.run_validation_for_step.return_value = {
                    "components": {
                        "alignment_validation": {"status": "COMPLETED"},
                        "integration_testing": {"status": "COMPLETED"}
                    },
                    "scoring": {"overall": {"score": 95.0}}
                }
                mock_tester_class.return_value = mock_tester
                
                report = reporter.test_and_report_builder(mock_builder_class, "TestStep")
                
                assert report.builder_name == "TestStep"
                assert report.builder_class == "TestStepBuilder"
                assert report.sagemaker_step_type == "Training"
                assert report.alignment_results is not None
                assert report.integration_results is not None
                assert report.scoring_data is not None

    def test_test_and_report_builder_infers_step_name(self, reporter, mock_builder_class):
        """Test that step name is inferred when not provided."""
        # FIXED: Category 1 - Mock STEP_NAMES at actual import location (registry import)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.get.return_value = {"sagemaker_step_type": "Processing"}
            
            # FIXED: Category 1 - Relative Import Pattern (following enhanced guide STEP 2D)
            # Source: from ..universal_test import UniversalStepBuilderTest
            # Convert relative ..universal_test to absolute: src.cursus.validation.builders.universal_test
            with patch('src.cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_tester_class:
                mock_tester = Mock()
                mock_tester.run_validation_for_step.return_value = {
                    "components": {
                        "alignment_validation": {"status": "COMPLETED"}
                    }
                }
                mock_tester_class.return_value = mock_tester
                
                # Don't provide step_name, should be inferred
                report = reporter.test_and_report_builder(mock_builder_class)
                
                # FIXED: Category 4 - Implementation Expectations (following enhanced guide)
                # Read source: step_name = class_name[:-11] if class_name.endswith("StepBuilder") else class_name
                # For "TestStepBuilder", removes "StepBuilder" suffix = "Test"
                assert report.builder_name == "Test"  # Actual implementation behavior
                assert report.builder_class == "TestStepBuilder"

    def test_test_and_report_builder_with_exception(self, reporter, mock_builder_class):
        """Test test and report generation when exception occurs."""
        # FIXED: Category 1 - Mock STEP_NAMES at actual import location (registry import)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.get.return_value = {"sagemaker_step_type": "Training"}
            
            # FIXED: Category 1 - Relative Import Pattern (following enhanced guide STEP 2D)
            # Source: from ..universal_test import UniversalStepBuilderTest
            # Convert relative ..universal_test to absolute: src.cursus.validation.builders.universal_test
            with patch('src.cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_tester_class:
                mock_tester_class.side_effect = Exception("Validation failed")
                
                report = reporter.test_and_report_builder(mock_builder_class, "TestStep")
                
                assert report.builder_name == "TestStep"
                assert report.builder_class == "TestStepBuilder"
                assert "error" in report.metadata
                assert "Validation failed" in report.metadata["error"]

    def test_test_and_save_builder_report(self, reporter, mock_builder_class):
        """Test testing and saving builder report to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter.output_dir = Path(temp_dir)
            
            # Mock test_and_report_builder
            with patch.object(reporter, 'test_and_report_builder') as mock_test_report:
                mock_report = Mock()
                mock_report.builder_name = "TestStep"
                mock_test_report.return_value = mock_report
                
                result = reporter.test_and_save_builder_report(mock_builder_class, "TestStep")
                
                assert result == mock_report
                mock_test_report.assert_called_once_with(mock_builder_class, "TestStep")
                
                # Verify file would be saved
                expected_file = Path(temp_dir) / "teststep_builder_test_report.json"
                mock_report.save_to_file.assert_called_once_with(expected_file)


class TestStreamlinedBuilderTestReporterStepTypeReporting:
    """Test step type reporting methods."""

    @pytest.fixture
    def reporter(self):
        """Create a reporter for testing."""
        return StreamlinedBuilderTestReporter()

    # FIXED: Category 1 - Conditional Function Import Pattern (following enhanced guide)
    # Source: from ....registry.step_names import get_steps_by_sagemaker_type
    # Mock at registry source, not importing module
    @patch('src.cursus.registry.step_names.get_steps_by_sagemaker_type')
    def test_test_step_type_builders_success(self, mock_get_steps, reporter):
        """Test testing all builders of a specific step type."""
        # Mock step discovery
        mock_get_steps.return_value = ["Step1", "Step2"]
        
        # Mock builder loading
        mock_builder1 = Mock()
        mock_builder1.__name__ = "Step1Builder"
        mock_builder2 = Mock()
        mock_builder2.__name__ = "Step2Builder"
        
        with patch.object(reporter, '_load_builder_class') as mock_load_builder:
            mock_load_builder.side_effect = [mock_builder1, mock_builder2]
            
            # Mock test and save report
            with patch.object(reporter, 'test_and_save_builder_report') as mock_test_save:
                mock_report1 = Mock()
                mock_report1.builder_name = "Step1"
                mock_report2 = Mock()
                mock_report2.builder_name = "Step2"
                mock_test_save.side_effect = [mock_report1, mock_report2]
                
                # Mock summary generation
                with patch.object(reporter, '_generate_streamlined_step_type_summary') as mock_summary:
                    results = reporter.test_step_type_builders("Training")
                    
                    assert len(results) == 2
                    assert "Step1" in results
                    assert "Step2" in results
                    assert results["Step1"] == mock_report1
                    assert results["Step2"] == mock_report2
                    
                    mock_summary.assert_called_once_with("Training", results)

    # FIXED: Category 1 - Conditional Function Import Pattern (following enhanced guide)
    # Source: from ....registry.step_names import get_steps_by_sagemaker_type
    # Mock at registry source, not importing module
    @patch('src.cursus.registry.step_names.get_steps_by_sagemaker_type')
    def test_test_step_type_builders_no_steps_found(self, mock_get_steps, reporter):
        """Test testing step type when no steps are found."""
        mock_get_steps.return_value = []
        
        results = reporter.test_step_type_builders("UnknownType")
        
        assert results == {}

    # FIXED: Category 1 - Conditional Function Import Pattern (following enhanced guide)
    # Source: from ....registry.step_names import get_steps_by_sagemaker_type
    # Mock at registry source, not importing module
    @patch('src.cursus.registry.step_names.get_steps_by_sagemaker_type')
    def test_test_step_type_builders_with_load_failures(self, mock_get_steps, reporter):
        """Test testing step type when some builders fail to load."""
        mock_get_steps.return_value = ["Step1", "Step2"]
        
        # Mock builder loading with one failure
        mock_builder1 = Mock()
        mock_builder1.__name__ = "Step1Builder"
        
        with patch.object(reporter, '_load_builder_class') as mock_load_builder:
            mock_load_builder.side_effect = [mock_builder1, None]  # Second fails
            
            with patch.object(reporter, 'test_and_save_builder_report') as mock_test_save:
                mock_report1 = Mock()
                mock_test_save.return_value = mock_report1
                
                with patch.object(reporter, '_generate_streamlined_step_type_summary') as mock_summary:
                    results = reporter.test_step_type_builders("Training")
                    
                    # Should only have results for Step1
                    assert len(results) == 1
                    assert "Step1" in results
                    assert "Step2" not in results

    def test_generate_streamlined_step_type_summary(self, reporter):
        """Test generation of step type summary."""
        # Create mock reports
        mock_report1 = Mock()
        mock_report1.builder_class = "Step1Builder"
        mock_report1.get_overall_status.return_value = "PASSING"
        mock_report1.get_quality_score.return_value = 90.0
        mock_report1.get_quality_rating.return_value = "Excellent"
        mock_report1.is_passing.return_value = True
        
        mock_report2 = Mock()
        mock_report2.builder_class = "Step2Builder"
        mock_report2.get_overall_status.return_value = "FAILING"
        mock_report2.get_quality_score.return_value = 60.0
        mock_report2.get_quality_rating.return_value = "Needs Work"
        mock_report2.is_passing.return_value = False
        
        reports = {
            "Step1": mock_report1,
            "Step2": mock_report2
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter.output_dir = Path(temp_dir)
            
            reporter._generate_streamlined_step_type_summary("Training", reports)
            
            # Verify summary file was created
            summary_file = Path(temp_dir) / "training_builder_test_summary.json"
            assert summary_file.exists()
            
            # Verify summary content
            with open(summary_file) as f:
                summary_data = json.load(f)
            
            assert summary_data["step_type"] == "Training"
            assert summary_data["summary"]["total_builders"] == 2
            assert summary_data["summary"]["passing_builders"] == 1
            assert summary_data["summary"]["failing_builders"] == 1
            assert summary_data["summary"]["average_quality_score"] == 75.0
            
            assert "Step1" in summary_data["builder_reports"]
            assert "Step2" in summary_data["builder_reports"]
            assert summary_data["builder_reports"]["Step1"]["is_passing"] is True
            assert summary_data["builder_reports"]["Step2"]["is_passing"] is False


class TestStreamlinedBuilderTestReporterErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def reporter(self):
        """Create a reporter for testing."""
        return StreamlinedBuilderTestReporter()

    def test_test_and_report_builder_handles_missing_step_info(self, reporter):
        """Test handling when step info is not found in registry."""
        mock_builder = Mock()
        mock_builder.__name__ = "UnknownStepBuilder"
        
        # FIXED: Category 1 - Mock STEP_NAMES at actual import location (registry import)
        with patch('src.cursus.registry.step_names.STEP_NAMES') as mock_step_names:
            mock_step_names.get.return_value = {}  # No step info found
            
            # FIXED: Category 1 - Relative Import Pattern (following enhanced guide STEP 2D)
            # Source: from ..universal_test import UniversalStepBuilderTest
            # Convert relative ..universal_test to absolute: src.cursus.validation.builders.universal_test
            with patch('src.cursus.validation.builders.universal_test.UniversalStepBuilderTest') as mock_tester_class:
                mock_tester = Mock()
                mock_tester.run_validation_for_step.return_value = {"components": {}}
                mock_tester_class.return_value = mock_tester
                
                report = reporter.test_and_report_builder(mock_builder, "UnknownStep")
                
                assert report.builder_name == "UnknownStep"
                assert report.sagemaker_step_type == "Unknown"  # Default when not found

    def test_test_step_type_builders_handles_exception(self, reporter):
        """Test handling when step type testing raises exception."""
        # FIXED: Category 1 - Mock get_steps_by_sagemaker_type at actual import location (registry import)
        with patch('src.cursus.registry.step_names.get_steps_by_sagemaker_type') as mock_get_steps:
            mock_get_steps.side_effect = Exception("Discovery failed")
            
            results = reporter.test_step_type_builders("Training")
            
            # Should handle exception gracefully and return empty results
            assert results == {}

    def test_generate_summary_handles_empty_reports(self, reporter):
        """Test summary generation with empty reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter.output_dir = Path(temp_dir)
            
            reporter._generate_streamlined_step_type_summary("Training", {})
            
            # Should still create summary file
            summary_file = Path(temp_dir) / "training_builder_test_summary.json"
            assert summary_file.exists()
            
            with open(summary_file) as f:
                summary_data = json.load(f)
            
            assert summary_data["summary"]["total_builders"] == 0
            assert summary_data["summary"]["average_quality_score"] == 0.0


class TestStreamlinedBuilderTestReporterBackwardCompatibility:
    """Test backward compatibility features."""

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases exist."""
        # These should be available for backward compatibility
        from src.cursus.validation.builders.reporting.builder_reporter import (
            BuilderTestReport,
            BuilderTestReporter
        )
        
        # Should be aliases to the streamlined versions
        assert BuilderTestReport == StreamlinedBuilderTestReport
        assert BuilderTestReporter == StreamlinedBuilderTestReporter
