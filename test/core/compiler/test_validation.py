"""
Unit tests for the validation module.

This module tests the validation and preview classes for the Pipeline API,
ensuring they provide accurate validation results and helpful previews.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from cursus.core.compiler.validation import (
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
    ValidationEngine,
)


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test ValidationResult for a valid configuration."""
        result = ValidationResult(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors={},
            dependency_issues=[],
            warnings=["Minor warning about naming"],
        )

        assert result.is_valid
        summary = result.summary()
        assert "✅ Validation passed" in summary
        assert "with 1 warnings" in summary

    def test_validation_result_invalid(self):
        """Test ValidationResult for an invalid configuration."""
        result = ValidationResult(
            is_valid=False,
            missing_configs=["data_loading", "preprocessing"],
            unresolvable_builders=["unknown_step"],
            config_errors={"training": ["missing required field"]},
            dependency_issues=["circular dependency detected"],
            warnings=[],
        )

        assert not result.is_valid
        summary = result.summary()
        assert "❌ Validation failed" in summary
        assert "2 missing configs" in summary
        assert "1 unresolvable builders" in summary
        assert "1 config errors" in summary
        assert "1 dependency issues" in summary

    def test_detailed_report_valid(self):
        """Test detailed report for valid configuration."""
        result = ValidationResult(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors={},
            dependency_issues=[],
            warnings=["Consider using more descriptive names"],
        )

        report = result.detailed_report()
        assert "✅ Validation passed" in report
        assert "Warnings:" in report
        assert "Consider using more descriptive names" in report

    def test_detailed_report_invalid_with_recommendations(self):
        """Test detailed report for invalid configuration with recommendations."""
        result = ValidationResult(
            is_valid=False,
            missing_configs=["data_loading"],
            unresolvable_builders=["custom_step"],
            config_errors={"training": ["invalid parameter"]},
            dependency_issues=["missing dependency"],
            warnings=[],
        )

        report = result.detailed_report()
        assert "❌ Validation failed" in report
        assert "Missing Configurations:" in report
        assert "- data_loading" in report
        assert "Unresolvable Step Builders:" in report
        assert "- custom_step" in report
        assert "Configuration Errors:" in report
        assert "training:" in report
        assert "- invalid parameter" in report
        assert "Dependency Issues:" in report
        assert "- missing dependency" in report
        assert "Recommendations:" in report
        assert "Add missing configuration instances" in report
        assert "Register missing step builders" in report
        assert "Fix configuration validation errors" in report
        assert "Review DAG structure" in report


class TestResolutionPreview:
    """Tests for the ResolutionPreview dataclass."""

    def test_resolution_preview_display(self):
        """Test the display method of ResolutionPreview."""
        preview = ResolutionPreview(
            node_config_map={
                "data_loading": "CradleDataLoadConfig",
                "preprocessing": "TabularPreprocessingConfig",
                "training": "XGBoostTrainingConfig",
            },
            config_builder_map={
                "CradleDataLoadConfig": "CradleDataLoadingStepBuilder",
                "TabularPreprocessingConfig": "TabularPreprocessingStepBuilder",
                "XGBoostTrainingConfig": "XGBoostTrainingStepBuilder",
            },
            resolution_confidence={
                "data_loading": 1.0,
                "preprocessing": 0.85,
                "training": 0.65,
            },
            ambiguous_resolutions=["training has 2 similar candidates"],
            recommendations=["Consider renaming 'training' for better matching"],
        )

        display = preview.display()
        assert "Resolution Preview" in display
        assert "Node → Configuration Mappings:" in display
        assert "🟢 data_loading → CradleDataLoadConfig (confidence: 1.00)" in display
        assert (
            "🟡 preprocessing → TabularPreprocessingConfig (confidence: 0.85)"
            in display
        )
        assert "🔴 training → XGBoostTrainingConfig (confidence: 0.65)" in display
        assert "Configuration → Builder Mappings:" in display
        assert "✓ CradleDataLoadConfig → CradleDataLoadingStepBuilder" in display
        assert "⚠️  Ambiguous Resolutions:" in display
        assert "training has 2 similar candidates" in display
        assert "💡 Recommendations:" in display
        assert "Consider renaming 'training' for better matching" in display


class TestConversionReport:
    """Tests for the ConversionReport dataclass."""

    def test_conversion_report_summary(self):
        """Test the summary method of ConversionReport."""
        report = ConversionReport(
            pipeline_name="test-pipeline",
            steps=["data_loading", "preprocessing", "training"],
            resolution_details={},
            avg_confidence=0.85,
            warnings=[],
            metadata={},
        )

        summary = report.summary()
        expected = "Pipeline 'test-pipeline' created successfully with 3 steps (avg confidence: 0.85)"
        assert summary == expected

    def test_conversion_report_detailed_report(self):
        """Test the detailed_report method of ConversionReport."""
        resolution_details = {
            "data_loading": {
                "config_type": "CradleDataLoadConfig",
                "builder_type": "CradleDataLoadingStepBuilder",
                "confidence": 1.0,
            },
            "preprocessing": {
                "config_type": "TabularPreprocessingConfig",
                "builder_type": "TabularPreprocessingStepBuilder",
                "confidence": 0.8,
            },
        }

        report = ConversionReport(
            pipeline_name="test-pipeline",
            steps=["data_loading", "preprocessing"],
            resolution_details=resolution_details,
            avg_confidence=0.9,
            warnings=["Low confidence for preprocessing"],
            metadata={"dag_nodes": 2, "dag_edges": 1},
        )

        detailed = report.detailed_report()
        assert "Pipeline Conversion Report" in detailed
        assert "Pipeline Name: test-pipeline" in detailed
        assert "Steps Created: 2" in detailed
        assert "Average Confidence: 0.90" in detailed
        assert "Step Resolution Details:" in detailed
        assert "data_loading:" in detailed
        assert "Config: CradleDataLoadConfig" in detailed
        assert "Builder: CradleDataLoadingStepBuilder" in detailed
        assert "Confidence: 1.00" in detailed
        assert "preprocessing:" in detailed
        assert "Config: TabularPreprocessingConfig" in detailed
        assert "Warnings:" in detailed
        assert "Low confidence for preprocessing" in detailed
        assert "Additional Metadata:" in detailed
        assert "dag_nodes: 2" in detailed
        assert "dag_edges: 1" in detailed


class TestValidationEngine:
    """Tests for the ValidationEngine class."""

    @pytest.fixture
    def engine(self):
        """Set up test fixtures."""
        return ValidationEngine()

    @pytest.fixture
    def mock_config1(self):
        mock_config1 = MagicMock()
        mock_config1.job_type = "training"
        type(mock_config1).__name__ = "XGBoostTrainingConfig"
        return mock_config1

    @pytest.fixture
    def mock_config2(self):
        mock_config2 = MagicMock()
        mock_config2.job_type = "evaluation"
        type(mock_config2).__name__ = "XGBoostModelEvalConfig"
        return mock_config2

    @pytest.fixture
    def available_configs(self, mock_config1, mock_config2):
        return {"training": mock_config1, "evaluation": mock_config2}

    @pytest.fixture
    def config_map(self, mock_config1, mock_config2):
        return {"training": mock_config1, "evaluation": mock_config2}

    @pytest.fixture
    def builder_registry(self):
        return {"XGBoostTraining": MagicMock(), "XGBoostModelEval": MagicMock()}

    @patch("cursus.core.compiler.validation.CONFIG_STEP_REGISTRY")
    @patch("cursus.core.compiler.validation.StepBuilderRegistry")
    def test_validate_dag_compatibility_success(
        self,
        mock_step_builder_registry,
        mock_config_step_registry,
        engine,
        available_configs,
        config_map,
        builder_registry,
        mock_config1,
        mock_config2,
    ):
        """Test successful DAG compatibility validation."""
        # Setup mocks
        mock_config_step_registry.__contains__ = lambda self, x: True
        mock_config_step_registry.__getitem__ = lambda self, x: (
            "XGBoostTraining" if "Training" in x else "XGBoostModelEval"
        )

        mock_step_builder_registry.LEGACY_ALIASES = {}

        dag_nodes = ["training", "evaluation"]

        # Mock config validation
        mock_config1.validate_config = MagicMock()
        mock_config2.validate_config = MagicMock()

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=available_configs,
            config_map=config_map,
            builder_registry=builder_registry,
        )

        assert result.is_valid
        assert result.missing_configs == []
        assert result.unresolvable_builders == []
        assert result.config_errors == {}
        assert result.dependency_issues == []

    def test_validate_dag_compatibility_missing_configs(
        self, engine, available_configs, config_map, builder_registry
    ):
        """Test validation with missing configurations."""
        dag_nodes = ["training", "evaluation", "missing_node"]

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=available_configs,
            config_map=config_map,  # missing_node not in config_map
            builder_registry=builder_registry,
        )

        assert not result.is_valid
        assert "missing_node" in result.missing_configs

    @patch("cursus.core.compiler.validation.CONFIG_STEP_REGISTRY")
    @patch("cursus.core.compiler.validation.StepBuilderRegistry")
    def test_validate_dag_compatibility_unresolvable_builders(
        self, mock_step_builder_registry, mock_config_step_registry, engine
    ):
        """Test validation with unresolvable step builders."""
        # Setup mocks to simulate missing builders
        mock_config_step_registry.__contains__ = (
            lambda self, x: False
        )  # Not in registry
        mock_step_builder_registry.LEGACY_ALIASES = {}

        dag_nodes = ["training"]

        # Mock a config that won't resolve to a builder
        mock_unknown_config = MagicMock()
        mock_unknown_config.job_type = "training"
        type(mock_unknown_config).__name__ = "UnknownConfig"

        config_map = {"training": mock_unknown_config}
        builder_registry = {}  # Empty registry

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"training": mock_unknown_config},
            config_map=config_map,
            builder_registry=builder_registry,
        )

        assert not result.is_valid
        assert len(result.unresolvable_builders) > 0

    def test_validate_dag_compatibility_config_errors(
        self, engine, available_configs, builder_registry, mock_config1
    ):
        """Test validation with configuration validation errors."""
        dag_nodes = ["training"]

        # Mock config that raises validation error
        mock_config1.validate_config = MagicMock(
            side_effect=ValueError("Invalid parameter")
        )

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=available_configs,
            config_map={"training": mock_config1},
            builder_registry=builder_registry,
        )

        assert not result.is_valid
        assert "training" in result.config_errors
        assert "Invalid parameter" in result.config_errors["training"][0]

    @patch("cursus.core.compiler.validation.CONFIG_STEP_REGISTRY")
    @patch("cursus.core.compiler.validation.StepBuilderRegistry")
    def test_validate_dag_compatibility_with_job_type_variants(
        self, mock_step_builder_registry, mock_config_step_registry, engine
    ):
        """Test validation with job type variants."""
        # Setup mocks
        mock_config_step_registry.__contains__ = lambda self, x: True
        mock_config_step_registry.__getitem__ = lambda self, x: "XGBoostTraining"

        mock_step_builder_registry.LEGACY_ALIASES = {}

        dag_nodes = ["training_step"]  # Node with job type pattern

        # Mock config with job type
        mock_config = MagicMock()
        mock_config.job_type = "training"
        type(mock_config).__name__ = "XGBoostTrainingConfig"

        config_map = {"training_step": mock_config}
        builder_registry = {
            "XGBoostTraining_training": MagicMock()
        }  # Builder with job type

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"training_step": mock_config},
            config_map=config_map,
            builder_registry=builder_registry,
        )

        assert result.is_valid

    @patch("cursus.core.compiler.validation.CONFIG_STEP_REGISTRY")
    @patch("cursus.core.compiler.validation.StepBuilderRegistry")
    def test_validate_dag_compatibility_with_legacy_aliases(
        self, mock_step_builder_registry, mock_config_step_registry, engine
    ):
        """Test validation with legacy aliases."""
        # Setup mocks
        mock_config_step_registry.__contains__ = lambda self, x: True
        mock_config_step_registry.__getitem__ = lambda self, x: "Package"

        mock_step_builder_registry.LEGACY_ALIASES = {"Package": "MIMSPackaging"}

        dag_nodes = ["packaging"]

        # Mock config
        mock_config = MagicMock()
        type(mock_config).__name__ = "PackageConfig"

        config_map = {"packaging": mock_config}
        builder_registry = {"MIMSPackaging": MagicMock()}  # Builder under legacy name

        result = engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"packaging": mock_config},
            config_map=config_map,
            builder_registry=builder_registry,
        )

        assert result.is_valid
