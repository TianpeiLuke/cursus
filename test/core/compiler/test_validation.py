"""
Unit tests for the validation module.

This module tests the validation and preview classes for the Pipeline API,
ensuring they provide accurate validation results and helpful previews.
"""

import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from cursus.core.compiler.validation import (
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
    ValidationEngine
)

class TestValidationResult(unittest.TestCase):
    """Tests for the ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test ValidationResult for a valid configuration."""
        result = ValidationResult(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors={},
            dependency_issues=[],
            warnings=["Minor warning about naming"]
        )
        
        self.assertTrue(result.is_valid)
        summary = result.summary()
        self.assertIn("✅ Validation passed", summary)
        self.assertIn("with 1 warnings", summary)

    def test_validation_result_invalid(self):
        """Test ValidationResult for an invalid configuration."""
        result = ValidationResult(
            is_valid=False,
            missing_configs=["data_loading", "preprocessing"],
            unresolvable_builders=["unknown_step"],
            config_errors={"training": ["missing required field"]},
            dependency_issues=["circular dependency detected"],
            warnings=[]
        )
        
        self.assertFalse(result.is_valid)
        summary = result.summary()
        self.assertIn("❌ Validation failed", summary)
        self.assertIn("2 missing configs", summary)
        self.assertIn("1 unresolvable builders", summary)
        self.assertIn("1 config errors", summary)
        self.assertIn("1 dependency issues", summary)

    def test_detailed_report_valid(self):
        """Test detailed report for valid configuration."""
        result = ValidationResult(
            is_valid=True,
            missing_configs=[],
            unresolvable_builders=[],
            config_errors={},
            dependency_issues=[],
            warnings=["Consider using more descriptive names"]
        )
        
        report = result.detailed_report()
        self.assertIn("✅ Validation passed", report)
        self.assertIn("Warnings:", report)
        self.assertIn("Consider using more descriptive names", report)

    def test_detailed_report_invalid_with_recommendations(self):
        """Test detailed report for invalid configuration with recommendations."""
        result = ValidationResult(
            is_valid=False,
            missing_configs=["data_loading"],
            unresolvable_builders=["custom_step"],
            config_errors={"training": ["invalid parameter"]},
            dependency_issues=["missing dependency"],
            warnings=[]
        )
        
        report = result.detailed_report()
        self.assertIn("❌ Validation failed", report)
        self.assertIn("Missing Configurations:", report)
        self.assertIn("- data_loading", report)
        self.assertIn("Unresolvable Step Builders:", report)
        self.assertIn("- custom_step", report)
        self.assertIn("Configuration Errors:", report)
        self.assertIn("training:", report)
        self.assertIn("- invalid parameter", report)
        self.assertIn("Dependency Issues:", report)
        self.assertIn("- missing dependency", report)
        self.assertIn("Recommendations:", report)
        self.assertIn("Add missing configuration instances", report)
        self.assertIn("Register missing step builders", report)
        self.assertIn("Fix configuration validation errors", report)
        self.assertIn("Review DAG structure", report)

class TestResolutionPreview(unittest.TestCase):
    """Tests for the ResolutionPreview dataclass."""

    def test_resolution_preview_display(self):
        """Test the display method of ResolutionPreview."""
        preview = ResolutionPreview(
            node_config_map={
                "data_loading": "CradleDataLoadConfig",
                "preprocessing": "TabularPreprocessingConfig",
                "training": "XGBoostTrainingConfig"
            },
            config_builder_map={
                "CradleDataLoadConfig": "CradleDataLoadingStepBuilder",
                "TabularPreprocessingConfig": "TabularPreprocessingStepBuilder",
                "XGBoostTrainingConfig": "XGBoostTrainingStepBuilder"
            },
            resolution_confidence={
                "data_loading": 1.0,
                "preprocessing": 0.85,
                "training": 0.65
            },
            ambiguous_resolutions=["training has 2 similar candidates"],
            recommendations=["Consider renaming 'training' for better matching"]
        )
        
        display = preview.display()
        self.assertIn("Resolution Preview", display)
        self.assertIn("Node → Configuration Mappings:", display)
        self.assertIn("🟢 data_loading → CradleDataLoadConfig (confidence: 1.00)", display)
        self.assertIn("🟡 preprocessing → TabularPreprocessingConfig (confidence: 0.85)", display)
        self.assertIn("🔴 training → XGBoostTrainingConfig (confidence: 0.65)", display)
        self.assertIn("Configuration → Builder Mappings:", display)
        self.assertIn("✓ CradleDataLoadConfig → CradleDataLoadingStepBuilder", display)
        self.assertIn("⚠️  Ambiguous Resolutions:", display)
        self.assertIn("training has 2 similar candidates", display)
        self.assertIn("💡 Recommendations:", display)
        self.assertIn("Consider renaming 'training' for better matching", display)

class TestConversionReport(unittest.TestCase):
    """Tests for the ConversionReport dataclass."""

    def test_conversion_report_summary(self):
        """Test the summary method of ConversionReport."""
        report = ConversionReport(
            pipeline_name="test-pipeline",
            steps=["data_loading", "preprocessing", "training"],
            resolution_details={},
            avg_confidence=0.85,
            warnings=[],
            metadata={}
        )
        
        summary = report.summary()
        expected = "Pipeline 'test-pipeline' created successfully with 3 steps (avg confidence: 0.85)"
        self.assertEqual(summary, expected)

    def test_conversion_report_detailed_report(self):
        """Test the detailed_report method of ConversionReport."""
        resolution_details = {
            "data_loading": {
                "config_type": "CradleDataLoadConfig",
                "builder_type": "CradleDataLoadingStepBuilder",
                "confidence": 1.0
            },
            "preprocessing": {
                "config_type": "TabularPreprocessingConfig",
                "builder_type": "TabularPreprocessingStepBuilder",
                "confidence": 0.8
            }
        }
        
        report = ConversionReport(
            pipeline_name="test-pipeline",
            steps=["data_loading", "preprocessing"],
            resolution_details=resolution_details,
            avg_confidence=0.9,
            warnings=["Low confidence for preprocessing"],
            metadata={"dag_nodes": 2, "dag_edges": 1}
        )
        
        detailed = report.detailed_report()
        self.assertIn("Pipeline Conversion Report", detailed)
        self.assertIn("Pipeline Name: test-pipeline", detailed)
        self.assertIn("Steps Created: 2", detailed)
        self.assertIn("Average Confidence: 0.90", detailed)
        self.assertIn("Step Resolution Details:", detailed)
        self.assertIn("data_loading:", detailed)
        self.assertIn("Config: CradleDataLoadConfig", detailed)
        self.assertIn("Builder: CradleDataLoadingStepBuilder", detailed)
        self.assertIn("Confidence: 1.00", detailed)
        self.assertIn("preprocessing:", detailed)
        self.assertIn("Config: TabularPreprocessingConfig", detailed)
        self.assertIn("Warnings:", detailed)
        self.assertIn("Low confidence for preprocessing", detailed)
        self.assertIn("Additional Metadata:", detailed)
        self.assertIn("dag_nodes: 2", detailed)
        self.assertIn("dag_edges: 1", detailed)

class TestValidationEngine(unittest.TestCase):
    """Tests for the ValidationEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        
        # Mock configurations
        self.mock_config1 = MagicMock()
        self.mock_config1.job_type = "training"
        type(self.mock_config1).__name__ = "XGBoostTrainingConfig"
        
        self.mock_config2 = MagicMock()
        self.mock_config2.job_type = "evaluation"
        type(self.mock_config2).__name__ = "XGBoostModelEvalConfig"
        
        # Mock available configs
        self.available_configs = {
            "training": self.mock_config1,
            "evaluation": self.mock_config2
        }
        
        # Mock config map
        self.config_map = {
            "training": self.mock_config1,
            "evaluation": self.mock_config2
        }
        
        # Mock builder registry
        self.builder_registry = {
            "XGBoostTraining": MagicMock(),
            "XGBoostModelEval": MagicMock()
        }

    @patch('src.cursus.core.compiler.validation.CONFIG_STEP_REGISTRY')
    @patch('src.cursus.core.compiler.validation.StepBuilderRegistry')
    def test_validate_dag_compatibility_success(self, mock_step_builder_registry, mock_config_step_registry):
        """Test successful DAG compatibility validation."""
        # Setup mocks
        mock_config_step_registry.__contains__ = lambda self, x: True
        mock_config_step_registry.__getitem__ = lambda self, x: "XGBoostTraining" if "Training" in x else "XGBoostModelEval"
        
        mock_step_builder_registry.LEGACY_ALIASES = {}
        
        dag_nodes = ["training", "evaluation"]
        
        # Mock config validation
        self.mock_config1.validate_config = MagicMock()
        self.mock_config2.validate_config = MagicMock()
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=self.available_configs,
            config_map=self.config_map,
            builder_registry=self.builder_registry
        )
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.missing_configs, [])
        self.assertEqual(result.unresolvable_builders, [])
        self.assertEqual(result.config_errors, {})
        self.assertEqual(result.dependency_issues, [])

    def test_validate_dag_compatibility_missing_configs(self):
        """Test validation with missing configurations."""
        dag_nodes = ["training", "evaluation", "missing_node"]
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=self.available_configs,
            config_map=self.config_map,  # missing_node not in config_map
            builder_registry=self.builder_registry
        )
        
        self.assertFalse(result.is_valid)
        self.assertIn("missing_node", result.missing_configs)

    @patch('src.cursus.core.compiler.validation.CONFIG_STEP_REGISTRY')
    @patch('src.cursus.core.compiler.validation.StepBuilderRegistry')
    def test_validate_dag_compatibility_unresolvable_builders(self, mock_step_builder_registry, mock_config_step_registry):
        """Test validation with unresolvable step builders."""
        # Setup mocks to simulate missing builders
        mock_config_step_registry.__contains__ = lambda self, x: False  # Not in registry
        mock_step_builder_registry.LEGACY_ALIASES = {}
        
        dag_nodes = ["training"]
        
        # Mock a config that won't resolve to a builder
        mock_unknown_config = MagicMock()
        mock_unknown_config.job_type = "training"
        type(mock_unknown_config).__name__ = "UnknownConfig"
        
        config_map = {"training": mock_unknown_config}
        builder_registry = {}  # Empty registry
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"training": mock_unknown_config},
            config_map=config_map,
            builder_registry=builder_registry
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.unresolvable_builders) > 0)

    def test_validate_dag_compatibility_config_errors(self):
        """Test validation with configuration validation errors."""
        dag_nodes = ["training"]
        
        # Mock config that raises validation error
        self.mock_config1.validate_config = MagicMock(side_effect=ValueError("Invalid parameter"))
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs=self.available_configs,
            config_map={"training": self.mock_config1},
            builder_registry=self.builder_registry
        )
        
        self.assertFalse(result.is_valid)
        self.assertIn("training", result.config_errors)
        self.assertIn("Invalid parameter", result.config_errors["training"][0])

    @patch('src.cursus.core.compiler.validation.CONFIG_STEP_REGISTRY')
    @patch('src.cursus.core.compiler.validation.StepBuilderRegistry')
    def test_validate_dag_compatibility_with_job_type_variants(self, mock_step_builder_registry, mock_config_step_registry):
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
        builder_registry = {"XGBoostTraining_training": MagicMock()}  # Builder with job type
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"training_step": mock_config},
            config_map=config_map,
            builder_registry=builder_registry
        )
        
        self.assertTrue(result.is_valid)

    @patch('src.cursus.core.compiler.validation.CONFIG_STEP_REGISTRY')
    @patch('src.cursus.core.compiler.validation.StepBuilderRegistry')
    def test_validate_dag_compatibility_with_legacy_aliases(self, mock_step_builder_registry, mock_config_step_registry):
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
        
        result = self.engine.validate_dag_compatibility(
            dag_nodes=dag_nodes,
            available_configs={"packaging": mock_config},
            config_map=config_map,
            builder_registry=builder_registry
        )
        
        self.assertTrue(result.is_valid)

if __name__ == '__main__':
    unittest.main()
