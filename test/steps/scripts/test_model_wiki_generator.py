#!/usr/bin/env python
"""
Pytest tests for model_wiki_generator.py
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import sys

# Add the src directory to the path so we can import the script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from cursus.steps.scripts.model_wiki_generator import (
    DataIngestionManager,
    WikiTemplateEngine,
    ContentGenerator,
    VisualizationIntegrator,
    WikiReportAssembler,
    WikiOutputManager,
    main
)


@pytest.fixture
def test_data_setup():
    """Create test data structure for testing."""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    metrics_dir = os.path.join(temp_dir, "metrics")
    plots_dir = os.path.join(temp_dir, "plots")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(metrics_dir)
    os.makedirs(plots_dir)
    os.makedirs(output_dir)
    
    # Create test metrics data (simulating model_metrics_computation output)
    metrics_report = {
        "timestamp": "2025-09-30T20:30:00Z",
        "data_summary": {
            "total_records": 10000,
            "prediction_columns": ["prob_class_0", "prob_class_1"],
            "has_amount_data": True
        },
        "standard_metrics": {
            "auc_roc": 0.85,
            "average_precision": 0.78,
            "f1_score": 0.72,
            "precision_at_threshold_0.5": 0.75,
            "recall_at_threshold_0.5": 0.69
        },
        "domain_metrics": {
            "dollar_recall": 0.82,
            "count_recall": 0.76,
            "total_abuse_amount": 125000.50,
            "average_abuse_amount": 2500.10
        },
        "performance_insights": [
            "Good discrimination capability (AUC ≥ 0.8)",
            "Model is particularly effective at catching high-value abuse cases"
        ],
        "recommendations": [
            "Consider feature engineering to improve recall",
            "Monitor performance on new data patterns"
        ]
    }
    
    # Save metrics report
    with open(os.path.join(metrics_dir, "metrics_report.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)
    
    # Create basic metrics (simulating xgboost_model_eval output)
    basic_metrics = {
        "auc_roc": 0.85,
        "average_precision": 0.78,
        "f1_score": 0.72
    }
    
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(basic_metrics, f, indent=2)
    
    # Create text summary
    text_summary = """MODEL METRICS COMPUTATION REPORT
==================================================
Generated: 2025-09-30T20:30:00Z

DATA SUMMARY
--------------------
Total Records: 10000
Prediction Columns: prob_class_0, prob_class_1
Has Amount Data: True

STANDARD ML METRICS
--------------------
auc_roc: 0.8500
average_precision: 0.7800
f1_score: 0.7200

DOMAIN-SPECIFIC METRICS
-------------------------
dollar_recall: 0.8200
count_recall: 0.7600
total_abuse_amount: 125000.5000

PERFORMANCE INSIGHTS
--------------------
• Good discrimination capability (AUC ≥ 0.8)
• Model is particularly effective at catching high-value abuse cases

RECOMMENDATIONS
---------------
• Consider feature engineering to improve recall
• Monitor performance on new data patterns
"""
    
    with open(os.path.join(metrics_dir, "metrics_summary.txt"), "w") as f:
        f.write(text_summary)
    
    # Create dummy plot files
    plot_files = [
        "roc_curve.jpg",
        "pr_curve.jpg", 
        "score_distribution.jpg",
        "threshold_analysis.jpg"
    ]
    
    for plot_file in plot_files:
        plot_path = os.path.join(plots_dir, plot_file)
        # Create a dummy image file (just text for testing)
        with open(plot_path, "w") as f:
            f.write(f"Dummy plot data for {plot_file}")
    
    yield temp_dir, metrics_dir, plots_dir, output_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestDataIngestionManager:
    """Test DataIngestionManager functionality."""
    
    def test_load_metrics_data(self, test_data_setup):
        """Test metrics data loading."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        manager = DataIngestionManager()
        metrics_data = manager.load_metrics_data(metrics_dir)
        
        assert "metrics_report" in metrics_data
        assert "basic_metrics" in metrics_data
        assert "text_summary" in metrics_data
        assert metrics_data["metrics_report"]["standard_metrics"]["auc_roc"] == 0.85
        assert metrics_data["basic_metrics"]["auc_roc"] == 0.85
        assert "MODEL METRICS COMPUTATION REPORT" in metrics_data["text_summary"]
    
    def test_load_metrics_data_empty_directory(self):
        """Test metrics data loading with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataIngestionManager()
            metrics_data = manager.load_metrics_data(temp_dir)
            assert metrics_data == {}
    
    def test_discover_visualization_files(self, test_data_setup):
        """Test visualization file discovery."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        manager = DataIngestionManager()
        visualizations = manager.discover_visualization_files(plots_dir)
        
        assert len(visualizations) == 4
        assert "roc_curve" in visualizations
        assert "pr_curve" in visualizations
        assert "score_distribution" in visualizations
        assert "threshold_analysis" in visualizations
        
        # Check structure of visualization entries
        roc_viz = visualizations["roc_curve"]
        assert "path" in roc_viz
        assert "description" in roc_viz
        assert "filename" in roc_viz
        assert roc_viz["filename"] == "roc_curve.jpg"
    
    def test_discover_visualization_files_empty_directory(self):
        """Test visualization discovery with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataIngestionManager()
            visualizations = manager.discover_visualization_files(temp_dir)
            assert visualizations == {}
    
    def test_discover_visualization_files_nonexistent_directory(self):
        """Test visualization discovery with nonexistent directory."""
        manager = DataIngestionManager()
        visualizations = manager.discover_visualization_files("/nonexistent/path")
        assert visualizations == {}


class TestContentGenerator:
    """Test ContentGenerator functionality."""
    
    def test_generate_performance_assessment(self):
        """Test performance assessment generation."""
        generator = ContentGenerator()
        
        assert generator.generate_performance_assessment(0.95) == "excellent"
        assert generator.generate_performance_assessment(0.85) == "good"
        assert generator.generate_performance_assessment(0.75) == "fair"
        assert generator.generate_performance_assessment(0.65) == "poor"
    
    def test_generate_auc_interpretation(self):
        """Test AUC interpretation generation."""
        generator = ContentGenerator()
        
        excellent = generator.generate_auc_interpretation(0.95)
        assert "Excellent discrimination capability" in excellent
        
        good = generator.generate_auc_interpretation(0.85)
        assert "Good discrimination capability" in good
        
        fair = generator.generate_auc_interpretation(0.75)
        assert "Fair discrimination capability" in fair
        
        poor = generator.generate_auc_interpretation(0.65)
        assert "Poor discrimination capability" in poor
    
    def test_generate_ap_interpretation(self):
        """Test Average Precision interpretation generation."""
        generator = ContentGenerator()
        
        excellent = generator.generate_ap_interpretation(0.95)
        assert "Excellent precision-recall performance" in excellent
        
        good = generator.generate_ap_interpretation(0.85)
        assert "Good precision-recall balance" in good
        
        fair = generator.generate_ap_interpretation(0.75)
        assert "Fair precision-recall performance" in fair
        
        poor = generator.generate_ap_interpretation(0.65)
        assert "Poor precision-recall balance" in poor
    
    def test_generate_business_impact_summary(self):
        """Test business impact summary generation."""
        generator = ContentGenerator()
        
        # Test with all metrics
        summary = generator.generate_business_impact_summary(
            dollar_recall=0.82, count_recall=0.76, total_abuse_amount=125000.50
        )
        assert "High dollar recall (82.0%)" in summary
        assert "Moderate count recall (76.0%)" in summary
        assert "$125,000.50" in summary
        
        # Test with no metrics
        summary_empty = generator.generate_business_impact_summary()
        assert "Business impact analysis not available." in summary_empty
        
        # Test with partial metrics
        summary_partial = generator.generate_business_impact_summary(dollar_recall=0.65)
        assert "Low dollar recall (65.0%)" in summary_partial
    
    def test_generate_recommendations_section(self):
        """Test recommendations formatting."""
        generator = ContentGenerator()
        
        # Test with recommendations
        recommendations = ["Improve feature engineering", "Monitor data drift"]
        formatted = generator.generate_recommendations_section(recommendations)
        assert "1. Improve feature engineering" in formatted
        assert "2. Monitor data drift" in formatted
        
        # Test with empty recommendations
        formatted_empty = generator.generate_recommendations_section([])
        assert "No specific recommendations available" in formatted_empty
    
    def test_generate_performance_overview(self):
        """Test performance overview generation."""
        generator = ContentGenerator()
        
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.78
        }
        
        overview = generator.generate_performance_overview(metrics)
        assert "good overall performance" in overview
        assert "AUC-ROC of 0.850" in overview
        assert "Average Precision of 0.780" in overview
        
        # Test with multiclass metrics
        multiclass_metrics = {
            "auc_roc_macro": 0.82,
            "auc_roc_micro": 0.84
        }
        
        overview_mc = generator.generate_performance_overview(multiclass_metrics)
        assert "Macro AUC of 0.820" in overview_mc
        assert "Micro AUC of 0.840" in overview_mc


class TestWikiTemplateEngine:
    """Test WikiTemplateEngine functionality."""
    
    def test_initialization(self):
        """Test template engine initialization."""
        engine = WikiTemplateEngine()
        assert engine.sections is not None
        assert "header" in engine.sections
        assert "summary" in engine.sections
        assert "performance_section" in engine.sections
    
    def test_generate_wiki_content(self):
        """Test wiki content generation."""
        engine = WikiTemplateEngine()
        
        context = {
            "model_name": "Test Model",
            "pipeline_name": "Test Pipeline",
            "model_display_name": "Test Model",
            "region": "US",
            "author": "Test Author",
            "team_alias": "test-team@",
            "contact_email": "test@example.com",
            "cti_classification": "Internal",
            "last_updated": "2025-09-30",
            "model_version": "1.0",
            "model_description": "This is a test model",
            "model_purpose": "perform test classification",
            "auc_score": 0.85,
            "average_precision": 0.78,
            "performance_assessment": "good",
            "auc_interpretation": "Good discrimination capability",
            "ap_interpretation": "Good precision-recall balance",
            "business_impact_summary": "Test business impact",
            "dollar_recall_section": "",
            "count_recall_section": "",
            "performance_overview": "Test performance overview",
            "roc_analysis_section": "",
            "precision_recall_section": "",
            "score_distribution_section": "",
            "threshold_analysis_section": "",
            "multiclass_analysis_section": "",
            "business_impact_details": "Test business impact details",
            "dollar_recall_analysis": "Test dollar recall analysis",
            "count_recall_analysis": "Test count recall analysis",
            "operational_recommendations": "Test operational recommendations",
            "recommendations_formatted": "1. Test recommendation",
            "next_steps": "Test next steps",
            "technical_details": "Test technical details",
            "model_configuration": "Test model configuration",
            "data_information": "Test data information"
        }
        
        wiki_content = engine.generate_wiki_content(context)
        assert "= Test Model =" in wiki_content
        assert "Test Pipeline" in wiki_content
        assert "AUC of 0.850" in wiki_content
        assert "== Summary ==" in wiki_content
        assert "== Model Performance Analysis ==" in wiki_content


class TestVisualizationIntegrator:
    """Test VisualizationIntegrator functionality."""
    
    def test_process_visualizations(self, test_data_setup):
        """Test visualization processing."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        integrator = VisualizationIntegrator(output_dir)
        
        # Create test visualizations data
        visualizations = {
            "roc_curve": {
                "path": os.path.join(plots_dir, "roc_curve.jpg"),
                "description": "ROC Curve Analysis",
                "filename": "roc_curve.jpg"
            }
        }
        
        processed_images = integrator.process_visualizations(visualizations)
        
        assert "roc_curve_image" in processed_images
        assert "roc_curve_description" in processed_images
        
        # Check that image was copied
        image_dir = os.path.join(output_dir, "images")
        assert os.path.exists(image_dir)
        copied_files = os.listdir(image_dir)
        assert len(copied_files) == 1
        assert copied_files[0].startswith("roc_curve_")
    
    def test_generate_plot_description(self, test_data_setup):
        """Test plot description generation."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        integrator = VisualizationIntegrator(output_dir)
        
        # Test known plot type
        description = integrator._generate_plot_description(
            "roc_curve", {"description": "Test description"}
        )
        assert "ROC curve analysis" in description
        assert "discrimination capability" in description
        
        # Test unknown plot type
        description_unknown = integrator._generate_plot_description(
            "unknown_plot", {"description": "Custom description"}
        )
        assert description_unknown == "Custom description"


class TestWikiOutputManager:
    """Test WikiOutputManager functionality."""
    
    def test_save_wiki_documentation(self, test_data_setup):
        """Test wiki documentation saving."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        manager = WikiOutputManager(output_dir)
        
        wiki_content = """= Test Model =

|Pipeline name|Test Pipeline
|Model Name|Test Model

== Summary ==

This is a test model.

* **AUC-ROC**: 0.850 - Good performance
"""
        
        output_files = manager.save_wiki_documentation(
            wiki_content, "Test Model", ["wiki", "html", "markdown"]
        )
        
        assert len(output_files) == 3
        assert "wiki" in output_files
        assert "html" in output_files
        assert "markdown" in output_files
        
        # Check that files were created and have content
        for format_type, file_path in output_files.items():
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                content = f.read()
                assert len(content) > 0
                
                if format_type == "wiki":
                    assert "= Test Model =" in content
                elif format_type == "html":
                    assert "<h1>Test Model</h1>" in content
                elif format_type == "markdown":
                    assert "# Test Model" in content
    
    def test_sanitize_filename(self, test_data_setup):
        """Test filename sanitization."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        manager = WikiOutputManager(output_dir)
        
        # Test various problematic characters
        assert manager._sanitize_filename("Test Model") == "test_model"
        assert manager._sanitize_filename("Model/With\\Slashes") == "model_with_slashes"
        assert manager._sanitize_filename("Model:With*Special?Chars") == "model_with_special_chars"


class TestWikiReportAssembler:
    """Test WikiReportAssembler functionality."""
    
    def test_assemble_complete_report(self, test_data_setup):
        """Test complete report assembly."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        template_engine = WikiTemplateEngine()
        content_generator = ContentGenerator()
        assembler = WikiReportAssembler(template_engine, content_generator)
        
        # Load test data
        manager = DataIngestionManager()
        metrics_data = manager.load_metrics_data(metrics_dir)
        
        processed_images = {
            "roc_curve_image": "roc_curve_20250930.jpg",
            "roc_curve_description": "ROC curve analysis"
        }
        
        environ_vars = {
            "MODEL_NAME": "Test Model",
            "MODEL_USE_CASE": "Test Classification",
            "PIPELINE_NAME": "Test Pipeline",
            "AUTHOR": "Test Author",
            "TEAM_ALIAS": "test-team@",
            "CONTACT_EMAIL": "test@example.com",
            "CTI_CLASSIFICATION": "Internal",
            "REGION": "US",
            "MODEL_VERSION": "1.0",
            "MODEL_DESCRIPTION": "This is a test model for classification tasks.",
            "MODEL_PURPOSE": "perform test classification tasks"
        }
        
        wiki_content = assembler.assemble_complete_report(
            metrics_data, processed_images, environ_vars
        )
        
        assert "= Test Model =" in wiki_content
        assert "Test Pipeline" in wiki_content
        assert "AUC of 0.850" in wiki_content
        assert "good overall performance" in wiki_content
        assert "High dollar recall (82.0%)" in wiki_content


class TestMainFunction:
    """Test main function functionality."""
    
    def test_main_function_basic(self, test_data_setup):
        """Test main function with basic functionality."""
        temp_dir, metrics_dir, plots_dir, output_dir = test_data_setup
        
        # Prepare arguments
        import argparse
        args = argparse.Namespace()
        
        input_paths = {
            "metrics_input": metrics_dir,
            "plots_input": plots_dir
        }
        
        output_paths = {
            "wiki_output": output_dir
        }
        
        environ_vars = {
            "MODEL_NAME": "Test Model",
            "MODEL_USE_CASE": "Test Classification",
            "MODEL_VERSION": "1.0",
            "PIPELINE_NAME": "Test Pipeline",
            "AUTHOR": "Test Author",
            "TEAM_ALIAS": "test-team@",
            "CONTACT_EMAIL": "test@example.com",
            "CTI_CLASSIFICATION": "Internal",
            "REGION": "US",
            "OUTPUT_FORMATS": "wiki,html,markdown",
            "INCLUDE_TECHNICAL_DETAILS": "true",
            "MODEL_DESCRIPTION": "This is a test model for classification tasks.",
            "MODEL_PURPOSE": "perform test classification tasks"
        }
        
        # Run main function
        main(input_paths, output_paths, environ_vars, args)
        
        # Check that output files were created
        output_files = list(Path(output_dir).glob("*.wiki"))
        assert len(output_files) >= 1
        
        output_files_html = list(Path(output_dir).glob("*.html"))
        assert len(output_files_html) >= 1
        
        output_files_md = list(Path(output_dir).glob("*.md"))
        assert len(output_files_md) >= 1
        
        # Check that summary was created
        summary_file = os.path.join(output_dir, "generation_summary.json")
        assert os.path.exists(summary_file)
        
        with open(summary_file, "r") as f:
            summary = json.load(f)
            assert summary["model_name"] == "Test Model"
            assert "wiki" in summary["output_formats"]
            assert "html" in summary["output_formats"]
            assert "markdown" in summary["output_formats"]
        
        # Note: Success markers are only created when script runs as __main__
        # In test context, main() function doesn't create these markers


if __name__ == "__main__":
    pytest.main([__file__])
