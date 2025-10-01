#!/usr/bin/env python
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import re
import shutil
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "METRICS_INPUT_DIR": "/opt/ml/processing/input/metrics",
    "PLOTS_INPUT_DIR": "/opt/ml/processing/input/plots", 
    "OUTPUT_WIKI_DIR": "/opt/ml/processing/output/wiki",
}


class DataIngestionManager:
    """
    Manages loading and parsing of metrics data, configuration, and visualizations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_metrics_data(self, metrics_dir: str) -> Dict[str, Any]:
        """
        Load comprehensive metrics data from metrics computation output.
        Supports both model_metrics_computation and xgboost_model_eval output formats.
        """
        metrics_data = {}
        
        # Load main metrics report (from model_metrics_computation)
        metrics_report_path = os.path.join(metrics_dir, "metrics_report.json")
        if os.path.exists(metrics_report_path):
            with open(metrics_report_path, "r") as f:
                metrics_data["metrics_report"] = json.load(f)
            self.logger.info("Loaded comprehensive metrics report")
        
        # Load basic metrics (from xgboost_model_eval or model_metrics_computation)
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data["basic_metrics"] = json.load(f)
            self.logger.info("Loaded basic metrics")
        
        # Load text summary
        summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                metrics_data["text_summary"] = f.read()
            self.logger.info("Loaded metrics text summary")
        
        return metrics_data
    
    def discover_visualization_files(self, plots_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Discover and catalog visualization files for embedding.
        """
        visualizations = {}
        
        if not os.path.exists(plots_dir):
            self.logger.warning(f"Plots directory not found: {plots_dir}")
            return visualizations
        
        # Standard plot types to look for
        plot_types = {
            "roc_curve": "ROC Curve Analysis",
            "pr_curve": "Precision-Recall Analysis",
            "precision_recall_curve": "Precision-Recall Analysis", 
            "score_distribution": "Score Distribution Analysis",
            "threshold_analysis": "Threshold Analysis",
            "multiclass_roc_curves": "Multi-class ROC Analysis"
        }
        
        for plot_type, description in plot_types.items():
            for ext in [".jpg", ".png", ".jpeg", ".svg"]:
                plot_path = os.path.join(plots_dir, f"{plot_type}{ext}")
                if os.path.exists(plot_path):
                    visualizations[plot_type] = {
                        "path": plot_path,
                        "description": description,
                        "filename": f"{plot_type}{ext}"
                    }
                    self.logger.info(f"Found visualization: {plot_type}")
                    break
        
        # Also look for class-specific plots (multiclass)
        for file_path in Path(plots_dir).glob("class_*_*.jpg"):
            filename = file_path.name
            plot_key = filename.replace(".jpg", "")
            visualizations[plot_key] = {
                "path": str(file_path),
                "description": f"Class-specific analysis: {filename}",
                "filename": filename
            }
            self.logger.info(f"Found class-specific visualization: {plot_key}")
        
        self.logger.info(f"Discovered {len(visualizations)} visualizations")
        return visualizations


class WikiTemplateEngine:
    """
    Template engine for generating wiki documentation from model data.
    """
    
    def __init__(self, template_config: Dict[str, Any] = None):
        self.template_config = template_config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sections = self._load_section_templates()
    
    def _load_section_templates(self) -> Dict[str, str]:
        """Load section templates from configuration."""
        return {
            "header": self._get_header_template(),
            "summary": self._get_summary_template(),
            "performance_section": self._get_performance_section_template(),
            "business_impact_section": self._get_business_impact_section_template(),
            "recommendations_section": self._get_recommendations_section_template(),
            "technical_details_section": self._get_technical_details_section_template()
        }
    
    def _get_header_template(self) -> str:
        """Generate header section template."""
        return """= {model_name} =

|Pipeline name|{pipeline_name}
|Model Name|{model_display_name}
|Region|{region}
|Author|{author}
|Team Alias|{team_alias}
|Point of Contact|{contact_email}
|CTI|{cti_classification}
|Last Updated|{last_updated}
|Model Version|{model_version}
"""

    def _get_summary_template(self) -> str:
        """Generate summary section template."""
        return """
== Summary ==

{model_description}

This model is designed to {model_purpose}. The model achieves an AUC of {auc_score:.3f} and demonstrates {performance_assessment} performance across key metrics.

=== Key Performance Metrics ===

* **AUC-ROC**: {auc_score:.3f} - {auc_interpretation}
* **Average Precision**: {average_precision:.3f} - {ap_interpretation}
{dollar_recall_section}
{count_recall_section}

=== Business Impact ===

{business_impact_summary}
"""

    def _get_performance_section_template(self) -> str:
        """Generate performance analysis section template."""
        return """
== Model Performance Analysis ==

=== Overall Performance ===

{performance_overview}

{roc_analysis_section}

{precision_recall_section}

{score_distribution_section}

{threshold_analysis_section}

{multiclass_analysis_section}
"""

    def _get_business_impact_section_template(self) -> str:
        """Generate business impact section template.""" 
        return """
== Business Impact Analysis ==

{business_impact_details}

{dollar_recall_analysis}

{count_recall_analysis}

{operational_recommendations}
"""

    def _get_recommendations_section_template(self) -> str:
        """Generate recommendations section template."""
        return """
== Recommendations ==

{recommendations_formatted}

=== Next Steps ===

{next_steps}
"""

    def _get_technical_details_section_template(self) -> str:
        """Generate technical details section template."""
        return """
== Technical Details ==

{technical_details}

=== Model Configuration ===

{model_configuration}

=== Data Information ===

{data_information}
"""

    def generate_wiki_content(
        self, 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate complete wiki content from context data.
        """
        # Generate each section
        wiki_sections = []
        for section_name, template in self.sections.items():
            try:
                section_content = template.format(**context)
                wiki_sections.append(section_content)
            except KeyError as e:
                self.logger.warning(f"Missing template variable for {section_name}: {e}")
                # Use fallback content or skip section
                continue
        
        return "\n".join(wiki_sections)


class ContentGenerator:
    """
    Generates intelligent content based on model performance data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_performance_assessment(self, auc_score: float) -> str:
        """Generate performance assessment based on AUC score."""
        if auc_score >= 0.9:
            return "excellent"
        elif auc_score >= 0.8:
            return "good"
        elif auc_score >= 0.7:
            return "fair"
        else:
            return "poor"
    
    def generate_auc_interpretation(self, auc_score: float) -> str:
        """Generate AUC interpretation text."""
        if auc_score >= 0.9:
            return "Excellent discrimination capability, model can reliably distinguish between classes"
        elif auc_score >= 0.8:
            return "Good discrimination capability, model performs well in most scenarios"
        elif auc_score >= 0.7:
            return "Fair discrimination capability, model shows reasonable performance"
        else:
            return "Poor discrimination capability, model may need significant improvement"
    
    def generate_ap_interpretation(self, ap_score: float) -> str:
        """Generate Average Precision interpretation text."""
        if ap_score >= 0.9:
            return "Excellent precision-recall performance"
        elif ap_score >= 0.8:
            return "Good precision-recall balance"
        elif ap_score >= 0.7:
            return "Fair precision-recall performance"
        else:
            return "Poor precision-recall balance, may need improvement"
    
    def generate_business_impact_summary(
        self, 
        dollar_recall: float = None, 
        count_recall: float = None,
        total_abuse_amount: float = None
    ) -> str:
        """Generate business impact summary based on available metrics."""
        impact_statements = []
        
        if dollar_recall is not None:
            if dollar_recall >= 0.8:
                impact_statements.append(f"High dollar recall ({dollar_recall:.1%}) indicates strong financial impact protection")
            elif dollar_recall >= 0.7:
                impact_statements.append(f"Moderate dollar recall ({dollar_recall:.1%}) provides reasonable financial protection")
            else:
                impact_statements.append(f"Low dollar recall ({dollar_recall:.1%}) suggests opportunity for improvement in high-value case detection")
        
        if count_recall is not None:
            if count_recall >= 0.8:
                impact_statements.append(f"High count recall ({count_recall:.1%}) demonstrates effective case detection")
            elif count_recall >= 0.6:
                impact_statements.append(f"Moderate count recall ({count_recall:.1%}) shows reasonable case coverage")
            else:
                impact_statements.append(f"Low count recall ({count_recall:.1%}) indicates potential for improved case detection")
        
        if total_abuse_amount is not None:
            impact_statements.append(f"Model protects against ${total_abuse_amount:,.2f} in potential abuse")
        
        return ". ".join(impact_statements) + "." if impact_statements else "Business impact analysis not available."
    
    def generate_recommendations_section(self, recommendations: List[str]) -> str:
        """Generate formatted recommendations section."""
        if not recommendations:
            return "No specific recommendations available at this time."
        
        formatted_recommendations = []
        for i, rec in enumerate(recommendations, 1):
            formatted_recommendations.append(f"{i}. {rec}")
        
        return "\n".join(formatted_recommendations)
    
    def generate_performance_overview(self, metrics: Dict[str, Any]) -> str:
        """Generate performance overview text."""
        overview_parts = []
        
        # Basic performance summary
        auc = metrics.get("auc_roc", 0)
        ap = metrics.get("average_precision", 0)
        
        overview_parts.append(f"The model demonstrates {self.generate_performance_assessment(auc)} overall performance with an AUC-ROC of {auc:.3f}.")
        
        if ap > 0:
            overview_parts.append(f"Average Precision of {ap:.3f} indicates {self.generate_ap_interpretation(ap).lower()}.")
        
        # Add multiclass summary if applicable
        if "auc_roc_macro" in metrics:
            macro_auc = metrics["auc_roc_macro"]
            micro_auc = metrics["auc_roc_micro"]
            overview_parts.append(f"For multiclass classification: Macro AUC of {macro_auc:.3f} and Micro AUC of {micro_auc:.3f}.")
        
        return " ".join(overview_parts)


class VisualizationIntegrator:
    """
    Handles integration of visualizations into wiki documentation.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_visualizations(
        self, 
        visualizations: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Process and prepare visualizations for wiki embedding.
        Returns mapping of plot types to wiki image references.
        """
        processed_images = {}
        
        for plot_type, plot_info in visualizations.items():
            try:
                # Copy image to output directory
                source_path = plot_info["path"]
                dest_filename = f"{plot_type}_{datetime.now().strftime('%Y%m%d')}.jpg"
                dest_path = os.path.join(self.image_dir, dest_filename)
                
                # Copy image
                shutil.copy2(source_path, dest_path)
                
                # Generate wiki image reference
                processed_images[f"{plot_type}_image"] = dest_filename
                
                # Generate description
                processed_images[f"{plot_type}_description"] = self._generate_plot_description(
                    plot_type, plot_info
                )
                
                self.logger.info(f"Processed visualization: {plot_type}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process visualization {plot_type}: {e}")
                continue
        
        return processed_images
    
    def _generate_plot_description(
        self, 
        plot_type: str, 
        plot_info: Dict[str, str]
    ) -> str:
        """Generate descriptive text for plots."""
        descriptions = {
            "roc_curve": "ROC curve analysis showing the trade-off between true positive rate and false positive rate across different thresholds. Higher AUC values indicate better model discrimination capability.",
            
            "pr_curve": "Precision-Recall curve showing the relationship between precision and recall across different thresholds. This is particularly useful for imbalanced datasets.",
            
            "precision_recall_curve": "Precision-Recall curve showing the relationship between precision and recall across different thresholds. This is particularly useful for imbalanced datasets.",
            
            "score_distribution": "Distribution of prediction scores by class, showing how well the model separates positive and negative cases. Good separation indicates effective discrimination.",
            
            "threshold_analysis": "Analysis of model performance metrics across different decision thresholds, helping identify optimal operating points for different business requirements.",
            
            "multiclass_roc_curves": "ROC curves for each class in multi-class classification, showing per-class discrimination capability and overall model performance."
        }
        
        return descriptions.get(plot_type, plot_info.get("description", "Model performance visualization"))


class WikiReportAssembler:
    """
    Assembles complete wiki reports from generated content and templates.
    """
    
    def __init__(self, template_engine: WikiTemplateEngine, content_generator: ContentGenerator):
        self.template_engine = template_engine
        self.content_generator = content_generator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def assemble_complete_report(
        self,
        metrics_data: Dict[str, Any],
        processed_images: Dict[str, str],
        environ_vars: Dict[str, str]
    ) -> str:
        """
        Assemble complete wiki report from all components.
        """
        # Build comprehensive context
        context = self._build_comprehensive_context(
            metrics_data, processed_images, environ_vars
        )
        
        # Generate wiki content
        wiki_content = self.template_engine.generate_wiki_content(context)
        
        return wiki_content
    
    def _build_comprehensive_context(
        self,
        metrics_data: Dict[str, Any],
        processed_images: Dict[str, str],
        environ_vars: Dict[str, str]
    ) -> Dict[str, Any]:
        """Build comprehensive context for report generation."""
        
        context = {}
        
        # Extract metrics information - try comprehensive report first, then basic metrics
        metrics_source = metrics_data.get("metrics_report", {})
        if not metrics_source:
            # Fallback to basic metrics format
            basic_metrics = metrics_data.get("basic_metrics", {})
            metrics_source = {
                "standard_metrics": basic_metrics,
                "domain_metrics": {},
                "performance_insights": [],
                "recommendations": []
            }
        
        # Standard metrics
        standard_metrics = metrics_source.get("standard_metrics", metrics_data.get("basic_metrics", {}))
        context.update({
            "auc_score": standard_metrics.get("auc_roc", 0),
            "average_precision": standard_metrics.get("average_precision", 0),
        })
        
        # Domain metrics
        domain_metrics = metrics_source.get("domain_metrics", {})
        context.update({
            "dollar_recall": domain_metrics.get("dollar_recall"),
            "count_recall": domain_metrics.get("count_recall"),
            "total_abuse_amount": domain_metrics.get("total_abuse_amount"),
        })
        
        # Add processed images
        context.update(processed_images)
        
        # Environment-based configuration
        context.update({
            "model_name": environ_vars.get("MODEL_NAME", "ML Model"),
            "model_display_name": environ_vars.get("MODEL_NAME", "ML Model"),
            "model_use_case": environ_vars.get("MODEL_USE_CASE", "Machine Learning Model"),
            "pipeline_name": environ_vars.get("PIPELINE_NAME", "ML Pipeline"),
            "region": environ_vars.get("REGION", "Global"),
            "author": environ_vars.get("AUTHOR", "ML Team"),
            "team_alias": environ_vars.get("TEAM_ALIAS", "ml-team@"),
            "contact_email": environ_vars.get("CONTACT_EMAIL", "ml-team@company.com"),
            "cti_classification": environ_vars.get("CTI_CLASSIFICATION", "Internal"),
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
            "model_version": environ_vars.get("MODEL_VERSION", "1.0"),
            "model_description": environ_vars.get("MODEL_DESCRIPTION", f"This is a machine learning model for {environ_vars.get('MODEL_USE_CASE', 'classification tasks')}."),
            "model_purpose": environ_vars.get("MODEL_PURPOSE", "perform classification tasks")
        })
        
        # Generate derived content
        context.update(self._generate_derived_content(context, metrics_source))
        
        return context
    
    def _generate_derived_content(self, context: Dict[str, Any], metrics_source: Dict[str, Any]) -> Dict[str, Any]:
        """Generate derived content from metrics and context."""
        derived = {}
        
        # Performance assessments
        auc_score = context.get("auc_score", 0)
        ap_score = context.get("average_precision", 0)
        
        derived["performance_assessment"] = self.content_generator.generate_performance_assessment(auc_score)
        derived["auc_interpretation"] = self.content_generator.generate_auc_interpretation(auc_score)
        derived["ap_interpretation"] = self.content_generator.generate_ap_interpretation(ap_score)
        
        # Business impact
        derived["business_impact_summary"] = self.content_generator.generate_business_impact_summary(
            dollar_recall=context.get("dollar_recall"),
            count_recall=context.get("count_recall"),
            total_abuse_amount=context.get("total_abuse_amount")
        )
        
        # Performance overview
        standard_metrics = metrics_source.get("standard_metrics", {})
        derived["performance_overview"] = self.content_generator.generate_performance_overview(standard_metrics)
        
        # Recommendations
        recommendations = metrics_source.get("recommendations", [])
        derived["recommendations_formatted"] = self.content_generator.generate_recommendations_section(recommendations)
        
        # Generate sections for visualizations
        derived.update(self._generate_visualization_sections(context))
        
        # Generate optional sections
        derived.update(self._generate_optional_sections(context, metrics_source))
        
        return derived
    
    def _generate_visualization_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization sections for wiki."""
        sections = {}
        
        # ROC Analysis Section
        if "roc_curve_image" in context:
            sections["roc_analysis_section"] = f"""
=== ROC Analysis ===

{context.get('roc_curve_description', 'ROC curve analysis showing model discrimination capability.')}

[[Image:{context['roc_curve_image']}|thumb|ROC Curve showing model discrimination capability]]
"""
        else:
            sections["roc_analysis_section"] = ""
        
        # Precision-Recall Section
        pr_image = context.get("precision_recall_curve_image") or context.get("pr_curve_image")
        if pr_image:
            sections["precision_recall_section"] = f"""
=== Precision-Recall Analysis ===

{context.get('precision_recall_curve_description', context.get('pr_curve_description', 'Precision-Recall curve analysis.'))}

[[Image:{pr_image}|thumb|Precision-Recall curve showing model performance trade-offs]]
"""
        else:
            sections["precision_recall_section"] = ""
        
        # Score Distribution Section
        if "score_distribution_image" in context:
            sections["score_distribution_section"] = f"""
=== Score Distribution ===

{context.get('score_distribution_description', 'Distribution of prediction scores by class.')}

[[Image:{context['score_distribution_image']}|thumb|Distribution of prediction scores by class]]
"""
        else:
            sections["score_distribution_section"] = ""
        
        # Threshold Analysis Section
        if "threshold_analysis_image" in context:
            sections["threshold_analysis_section"] = f"""
=== Threshold Analysis ===

{context.get('threshold_analysis_description', 'Analysis of model performance across different thresholds.')}

[[Image:{context['threshold_analysis_image']}|thumb|Threshold analysis for optimal operating points]]
"""
        else:
            sections["threshold_analysis_section"] = ""
        
        # Multiclass Analysis Section
        if "multiclass_roc_curves_image" in context:
            sections["multiclass_analysis_section"] = f"""
=== Multi-class Analysis ===

{context.get('multiclass_roc_curves_description', 'Multi-class ROC curve analysis.')}

[[Image:{context['multiclass_roc_curves_image']}|thumb|Multi-class ROC curves]]
"""
        else:
            sections["multiclass_analysis_section"] = ""
        
        return sections
    
    def _generate_optional_sections(self, context: Dict[str, Any], metrics_source: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optional sections based on available data."""
        sections = {}
        
        # Dollar recall section
        dollar_recall = context.get("dollar_recall")
        if dollar_recall is not None:
            sections["dollar_recall_section"] = f"* **Dollar Recall**: {dollar_recall:.1%} - Financial impact protection"
            sections["dollar_recall_analysis"] = f"Dollar recall of {dollar_recall:.1%} indicates the model's effectiveness at catching high-value abuse cases."
        else:
            sections["dollar_recall_section"] = ""
            sections["dollar_recall_analysis"] = "Dollar recall analysis not available."
        
        # Count recall section
        count_recall = context.get("count_recall")
        if count_recall is not None:
            sections["count_recall_section"] = f"* **Count Recall**: {count_recall:.1%} - Case detection coverage"
            sections["count_recall_analysis"] = f"Count recall of {count_recall:.1%} shows the model's ability to detect abuse cases overall."
        else:
            sections["count_recall_section"] = ""
            sections["count_recall_analysis"] = "Count recall analysis not available."
        
        # Business impact details
        sections["business_impact_details"] = context.get("business_impact_summary", "Business impact analysis not available.")
        
        # Technical details
        sections["technical_details"] = "Technical details will be populated based on available model configuration and data information."
        sections["model_configuration"] = "Model configuration details not available."
        sections["data_information"] = "Data information not available."
        
        # Operational recommendations
        sections["operational_recommendations"] = "Operational recommendations will be provided based on model performance analysis."
        
        # Next steps
        sections["next_steps"] = "Next steps will be determined based on model performance and business requirements."
        
        return sections


class WikiOutputManager:
    """
    Manages output generation in multiple formats.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_wiki_documentation(
        self, 
        wiki_content: str, 
        model_name: str,
        formats: List[str] = ["wiki", "html", "markdown"]
    ) -> Dict[str, str]:
        """
        Save wiki documentation in multiple formats.
        Returns dictionary of format -> file path mappings.
        """
        output_files = {}
        
        # Generate base filename
        safe_model_name = self._sanitize_filename(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{safe_model_name}_documentation_{timestamp}"
        
        # Save in requested formats
        for format_type in formats:
            try:
                if format_type == "wiki":
                    file_path = self._save_wiki_format(wiki_content, base_filename)
                elif format_type == "html":
                    file_path = self._save_html_format(wiki_content, base_filename)
                elif format_type == "markdown":
                    file_path = self._save_markdown_format(wiki_content, base_filename)
                else:
                    self.logger.warning(f"Unknown output format: {format_type}")
                    continue
                
                output_files[format_type] = file_path
                self.logger.info(f"Saved {format_type} documentation to {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save {format_type} format: {e}")
                continue
        
        return output_files
    
    def _save_wiki_format(self, content: str, base_filename: str) -> str:
        """Save in wiki format."""
        file_path = os.path.join(self.output_dir, f"{base_filename}.wiki")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    
    def _save_html_format(self, content: str, base_filename: str) -> str:
        """Save in HTML format."""
        # Convert wiki markup to HTML
        html_content = self._convert_wiki_to_html(content)
        
        file_path = os.path.join(self.output_dir, f"{base_filename}.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return file_path
    
    def _save_markdown_format(self, content: str, base_filename: str) -> str:
        """Save in Markdown format."""
        # Convert wiki markup to Markdown
        markdown_content = self._convert_wiki_to_markdown(content)
        
        file_path = os.path.join(self.output_dir, f"{base_filename}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return file_path
    
    def _convert_wiki_to_html(self, wiki_content: str) -> str:
        """Convert wiki markup to HTML."""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Model Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        .metric {{ font-weight: bold; color: #2c5aa0; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
        
        # Convert wiki markup to HTML
        html_content = wiki_content
        
        # Convert headers
        html_content = re.sub(r'^= (.*?) =$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^== (.*?) ==$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^=== (.*?) ===$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
        
        # Convert tables
        html_content = self._convert_wiki_tables_to_html(html_content)
        
        # Convert images
        html_content = re.sub(
            r'\[\[Image:(.*?)\|thumb\|(.*?)\]\]',
            r'<div class="image-container"><img src="images/\1" alt="\2"><p class="caption">\2</p></div>',
            html_content
        )
        
        # Convert lists
        html_content = re.sub(r'^\* (.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
        
        # Convert bold text
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
        
        return html_template.format(content=html_content)
    
    def _convert_wiki_to_markdown(self, wiki_content: str) -> str:
        """Convert wiki markup to Markdown."""
        markdown_content = wiki_content
        
        # Convert headers
        markdown_content = re.sub(r'^= (.*?) =$', r'# \1', markdown_content, flags=re.MULTILINE)
        markdown_content = re.sub(r'^== (.*?) ==$', r'## \1', markdown_content, flags=re.MULTILINE)
        markdown_content = re.sub(r'^=== (.*?) ===$', r'### \1', markdown_content, flags=re.MULTILINE)
        
        # Convert images
        markdown_content = re.sub(
            r'\[\[Image:(.*?)\|thumb\|(.*?)\]\]',
            r'![\\2](images/\\1)',
            markdown_content
        )
        
        # Convert tables (basic conversion)
        markdown_content = self._convert_wiki_tables_to_markdown(markdown_content)
        
        return markdown_content
    
    def _convert_wiki_tables_to_html(self, content: str) -> str:
        """Convert wiki table format to HTML tables."""
        lines = content.split('\n')
        html_lines = []
        in_table = False
        
        for line in lines:
            if line.startswith('|') and '|' in line[1:]:
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                
                # Parse table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                row_html = '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
                html_lines.append(row_html)
            else:
                if in_table:
                    html_lines.append('</table>')
                    in_table = False
                html_lines.append(line)
        
        if in_table:
            html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    def _convert_wiki_tables_to_markdown(self, content: str) -> str:
        """Convert wiki table format to Markdown tables."""
        lines = content.split('\n')
        markdown_lines = []
        table_rows = []
        in_table = False
        
        for line in lines:
            if line.startswith('|') and '|' in line[1:]:
                if not in_table:
                    in_table = True
                    table_rows = []
                
                # Parse table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                table_rows.append(cells)
            else:
                if in_table:
                    # Convert accumulated table rows to markdown
                    if table_rows:
                        # Header row
                        markdown_lines.append('| ' + ' | '.join(table_rows[0]) + ' |')
                        markdown_lines.append('| ' + ' | '.join(['---'] * len(table_rows[0])) + ' |')
                        
                        # Data rows
                        for row in table_rows[1:]:
                            markdown_lines.append('| ' + ' | '.join(row) + ' |')
                    
                    in_table = False
                    table_rows = []
                
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized.lower()


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for Model Wiki Generator script.
    Loads metrics data and visualizations, generates wiki documentation, and saves results.

    Args:
        input_paths (Dict[str, str]): Dictionary of input paths
        output_paths (Dict[str, str]): Dictionary of output paths
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
    metrics_input_dir = input_paths.get("metrics_input", input_paths.get("metrics_input_dir"))
    plots_input_dir = input_paths.get("plots_input", input_paths.get("plots_input_dir", metrics_input_dir))
    output_wiki_dir = output_paths.get("wiki_output", output_paths.get("output_wiki_dir"))

    # Extract environment variables
    model_name = environ_vars.get("MODEL_NAME", "ML Model")
    output_formats = environ_vars.get("OUTPUT_FORMATS", "wiki,html,markdown").split(",")
    include_technical_details = environ_vars.get("INCLUDE_TECHNICAL_DETAILS", "true").lower() == "true"

    # Log job info
    logger.info("Running model wiki generator")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output formats: {output_formats}")

    # Ensure output directories exist
    os.makedirs(output_wiki_dir, exist_ok=True)

    logger.info("Starting model wiki generator script")

    # Initialize components
    data_ingestion = DataIngestionManager()
    template_engine = WikiTemplateEngine()
    content_generator = ContentGenerator()
    visualization_integrator = VisualizationIntegrator(output_wiki_dir)
    report_assembler = WikiReportAssembler(template_engine, content_generator)
    output_manager = WikiOutputManager(output_wiki_dir)

    # Load metrics data
    logger.info(f"Loading metrics data from {metrics_input_dir}")
    metrics_data = data_ingestion.load_metrics_data(metrics_input_dir)
    
    if not metrics_data:
        logger.warning("No metrics data found - generating basic documentation")
        metrics_data = {
            "basic_metrics": {"auc_roc": 0.0, "average_precision": 0.0},
            "metrics_report": {
                "standard_metrics": {"auc_roc": 0.0, "average_precision": 0.0},
                "domain_metrics": {},
                "performance_insights": [],
                "recommendations": []
            }
        }

    # Discover and process visualizations
    logger.info(f"Discovering visualizations from {plots_input_dir}")
    visualizations = data_ingestion.discover_visualization_files(plots_input_dir)
    processed_images = visualization_integrator.process_visualizations(visualizations)
    
    logger.info(f"Processed {len(processed_images)//2} visualizations") # Divide by 2 because each viz has image + description

    # Generate comprehensive wiki report
    logger.info("Assembling wiki report")
    wiki_content = report_assembler.assemble_complete_report(
        metrics_data, processed_images, environ_vars
    )

    # Save documentation in multiple formats
    logger.info(f"Saving documentation in formats: {output_formats}")
    output_files = output_manager.save_wiki_documentation(
        wiki_content, model_name, output_formats
    )

    # Log output file locations
    for format_type, file_path in output_files.items():
        logger.info(f"Generated {format_type} documentation: {file_path}")

    # Create summary report
    summary_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "output_formats": list(output_files.keys()),
        "output_files": output_files,
        "visualizations_processed": len(visualizations),
        "metrics_sources": list(metrics_data.keys())
    }
    
    summary_path = os.path.join(output_wiki_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    
    logger.info(f"Generated summary report: {summary_path}")
    logger.info("Model wiki generator script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "metrics_input": CONTAINER_PATHS["METRICS_INPUT_DIR"],
        "plots_input": CONTAINER_PATHS["PLOTS_INPUT_DIR"],
    }

    output_paths = {
        "wiki_output": CONTAINER_PATHS["OUTPUT_WIKI_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        "MODEL_NAME": os.environ.get("MODEL_NAME", "ML Model"),
        "MODEL_USE_CASE": os.environ.get("MODEL_USE_CASE", "Machine Learning Model"),
        "MODEL_VERSION": os.environ.get("MODEL_VERSION", "1.0"),
        "PIPELINE_NAME": os.environ.get("PIPELINE_NAME", "ML Pipeline"),
        "AUTHOR": os.environ.get("AUTHOR", "ML Team"),
        "TEAM_ALIAS": os.environ.get("TEAM_ALIAS", "ml-team@"),
        "CONTACT_EMAIL": os.environ.get("CONTACT_EMAIL", "ml-team@company.com"),
        "CTI_CLASSIFICATION": os.environ.get("CTI_CLASSIFICATION", "Internal"),
        "REGION": os.environ.get("REGION", "Global"),
        "OUTPUT_FORMATS": os.environ.get("OUTPUT_FORMATS", "wiki,html,markdown"),
        "INCLUDE_TECHNICAL_DETAILS": os.environ.get("INCLUDE_TECHNICAL_DETAILS", "true"),
        "MODEL_DESCRIPTION": os.environ.get("MODEL_DESCRIPTION", ""),
        "MODEL_PURPOSE": os.environ.get("MODEL_PURPOSE", "perform classification tasks"),
    }

    try:
        # Call main function with testability parameters
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["wiki_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["wiki_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        import sys
        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("wiki_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        import sys
        sys.exit(1)
