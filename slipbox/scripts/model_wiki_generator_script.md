---
tags:
  - code
  - processing_script
  - documentation_generation
  - wiki_generation
  - model_reporting
keywords:
  - model wiki generator
  - automated documentation
  - multi-format output
  - template engine
  - performance documentation
  - business impact reporting
  - wiki markup
  - HTML generation
  - model registry integration
topics:
  - automated model documentation
  - wiki generation workflows
  - model reporting systems
  - documentation templating
language: python
date of note: 2025-11-18
---

# Model Wiki Generator Script

## Overview

The `model_wiki_generator.py` script is an **automated documentation engine** that transforms model performance metrics and visualizations into comprehensive, human-readable documentation across multiple formats (Wiki, HTML, Markdown). It serves as the final documentation step in ML pipelines, creating standardized model documentation for model registries, knowledge bases, and compliance systems.

The script employs a sophisticated six-component architecture that handles data ingestion, intelligent content generation, visualization integration, template-driven report assembly, and multi-format output. It supports both single model analysis and model comparison workflows with statistical significance testing, making it suitable for production model evaluation, A/B testing documentation, and regulatory compliance reporting.

Key capabilities include intelligent performance assessment, automated business impact analysis, template-driven documentation structure, and seamless integration with upstream metrics computation steps. The script is framework-agnostic, working with outputs from both `model_metrics_computation` and `xgboost_model_eval` scripts.

## Purpose and Major Tasks

### Primary Purpose
Generate comprehensive, professional model documentation from performance metrics and visualizations, creating searchable, structured documentation suitable for model registries, knowledge management systems, and compliance frameworks.

### Major Tasks
1. **Data Ingestion**: Load metrics data from model evaluation outputs including comprehensive metrics reports, basic metrics JSON, text summaries, and discover available visualization files
2. **Visualization Processing**: Copy, optimize, and prepare performance plots (ROC curves, PR curves, score distributions, threshold analysis) for embedding in documentation
3. **Template-Based Generation**: Apply configurable section templates (header, summary, performance analysis, business impact, recommendations, technical details) to structure documentation
4. **Intelligent Content Generation**: Automatically generate performance assessments, AUC/AP interpretations, business impact summaries, and actionable recommendations based on metric values
5. **Model Comparison Analysis**: Detect and document model comparisons including AUC/AP deltas, statistical significance tests (McNemar's, paired t-test, Wilcoxon), correlation analysis, and deployment recommendations
6. **Multi-Format Output**: Export documentation in Wiki markup (MediaWiki), styled HTML with CSS, and GitHub-compatible Markdown formats
7. **Asset Management**: Organize and optimize documentation assets including image processing, file naming with timestamps, and directory structure creation
8. **Metadata Generation**: Create comprehensive generation summaries tracking output formats, visualizations processed, metrics sources, and file locations

## Script Contract

### Entry Point
```
model_wiki_generator.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `metrics_input` | `/opt/ml/processing/input/metrics` | Metrics data directory containing metrics_report.json (comprehensive metrics from model_metrics_computation), metrics.json (basic metrics), and metrics_summary.txt |
| `plots_input` | `/opt/ml/processing/input/plots` | Visualization files directory containing ROC curves, PR curves, score distributions, threshold analysis plots, and multiclass visualizations |

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `wiki_output` | `/opt/ml/processing/output/wiki` | Wiki documentation output directory containing generated documentation files in multiple formats, processed images subdirectory, and generation summary |

### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_NAME` | Name of the model for documentation | `"AbuseDetectionModel"` |

### Optional Environment Variables

#### Model Metadata
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_USE_CASE` | `"Machine Learning Model"` | Description of model use case and application domain |
| `MODEL_VERSION` | `"1.0"` | Model version identifier for tracking |
| `MODEL_DESCRIPTION` | `""` | Custom model description text for summary section |
| `MODEL_PURPOSE` | `"perform classification tasks"` | Custom model purpose description |

#### Documentation Context
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_NAME` | `"ML Pipeline"` | Name of the ML pipeline that produced this model |
| `AUTHOR` | `"ML Team"` | Model author/creator for attribution |
| `TEAM_ALIAS` | `"ml-team@"` | Team email alias for contact |
| `CONTACT_EMAIL` | `"ml-team@company.com"` | Point of contact email address |
| `REGION` | `"Global"` | AWS region or deployment region |
| `CTI_CLASSIFICATION` | `"Internal"` | CTI classification for the model |

#### Output Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_FORMATS` | `"wiki,html,markdown"` | Comma-separated list of output formats to generate |
| `INCLUDE_TECHNICAL_DETAILS` | `"true"` | Include technical details section in documentation |

### Job Arguments
No command-line arguments are expected. All configuration is provided through environment variables.

## Input Data Structure

### Expected Input Format
```
/opt/ml/processing/input/metrics/
├── metrics_report.json          # Comprehensive metrics from model_metrics_computation
├── metrics.json                 # Basic metrics (from xgboost_model_eval or model_metrics_computation)
├── metrics_summary.txt          # Human-readable text summary
└── _SUCCESS                     # Optional success marker

/opt/ml/processing/input/plots/
├── roc_curve.jpg               # ROC curve visualization
├── pr_curve.jpg                # Precision-Recall curve (or precision_recall_curve.jpg)
├── score_distribution.jpg      # Score distribution by class
├── threshold_analysis.jpg      # Threshold analysis plot
├── multiclass_roc_curves.jpg   # Combined multiclass ROC curves (if applicable)
├── class_*_*.jpg               # Per-class visualizations (multiclass)
├── comparison_roc_curves.jpg   # Model comparison ROC curves (comparison mode)
├── comparison_pr_curves.jpg    # Model comparison PR curves (comparison mode)
├── score_scatter_plot.jpg      # Score correlation plot (comparison mode)
├── score_distributions.jpg     # Score distribution comparison (comparison mode)
└── _SUCCESS                    # Optional success marker
```

### Metrics Report JSON Structure (Comprehensive Format)
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "data_summary": {
    "total_samples": 10000,
    "positive_samples": 500,
    "negative_samples": 9500
  },
  "standard_metrics": {
    "auc_roc": 0.85,
    "average_precision": 0.78,
    "f1_score": 0.72,
    "precision": 0.75,
    "recall": 0.70
  },
  "domain_metrics": {
    "dollar_recall": 0.82,
    "count_recall": 0.75,
    "total_abuse_amount": 150000.0
  },
  "performance_insights": [
    "Model shows strong performance on high-value cases",
    "Consider threshold adjustment for better precision-recall balance"
  ],
  "recommendations": [
    "Deploy model to production with 0.5 threshold",
    "Monitor false positive rate in production",
    "Consider retraining with additional features"
  ],
  "visualizations": {
    "roc_curve": "roc_curve.jpg",
    "precision_recall_curve": "pr_curve.jpg"
  }
}
```

### Model Comparison Metrics (Extended Format)
When comparison mode is detected, additional fields are present:
```json
{
  "standard_metrics": {
    "auc_roc": 0.87,
    "new_model_auc": 0.87,
    "previous_model_auc": 0.85,
    "auc_delta": 0.02,
    "auc_lift_percent": 2.35,
    "ap_delta": 0.015,
    "ap_lift_percent": 2.0,
    "pearson_correlation": 0.92,
    "spearman_correlation": 0.91,
    "mcnemar_p_value": 0.001,
    "mcnemar_significant": true,
    "paired_t_p_value": 0.002,
    "paired_t_significant": true,
    "wilcoxon_p_value": 0.003,
    "wilcoxon_significant": true
  }
}
```

### Supported Input Sources
1. **Model Metrics Computation**: Primary source with comprehensive metrics report including standard metrics, domain metrics, performance insights, and recommendations
2. **XGBoost Model Eval**: Fallback source with basic metrics.json format and standard visualization files
3. **Custom Metrics**: Any source providing metrics.json with at least `auc_roc` and `average_precision` fields

## Output Data Structure

### Output Directory Structure
```
/opt/ml/processing/output/wiki/
├── {model_name}_documentation_{timestamp}.wiki    # Wiki markup format
├── {model_name}_documentation_{timestamp}.html    # Styled HTML format
├── {model_name}_documentation_{timestamp}.md      # Markdown format
├── generation_summary.json                         # Generation metadata
├── images/                                         # Processed visualizations
│   ├── roc_curve_20251118.jpg
│   ├── pr_curve_20251118.jpg
│   ├── score_distribution_20251118.jpg
│   ├── threshold_analysis_20251118.jpg
│   ├── comparison_roc_curves_20251118.jpg         # Comparison mode
│   └── comparison_pr_curves_20251118.jpg          # Comparison mode
├── _SUCCESS                                        # Success marker
└── _HEALTH                                         # Health check file
```

### Wiki Markup Output Format
MediaWiki-compatible markup with sections for:
- **Header**: Model metadata table with pipeline name, author, contact, CTI classification
- **Summary**: Model description, key metrics, business impact
- **Performance Analysis**: Overall performance, ROC analysis, precision-recall analysis, score distribution, threshold analysis with embedded visualizations
- **Model Comparison Summary** (if applicable): AUC/AP deltas, statistical significance tests, deployment recommendations
- **Business Impact Analysis**: Dollar recall analysis, count recall analysis, operational recommendations
- **Recommendations**: Numbered actionable recommendations from metrics computation
- **Technical Details**: Model configuration, data information, feature importance

### HTML Output Format
Styled HTML with CSS including:
- Responsive layout with professional styling
- Embedded images with captions
- Formatted tables and lists
- Syntax highlighting for metrics
- Mobile-friendly design

### Markdown Output Format
GitHub-compatible Markdown with:
- Header hierarchy using # symbols
- Markdown tables for structured data
- Image embedding with alt text
- Bullet lists for recommendations
- Code blocks for technical details

### Generation Summary JSON
```json
{
  "timestamp": "2025-01-15T10:35:00Z",
  "model_name": "AbuseDetectionModel",
  "output_formats": ["wiki", "html", "markdown"],
  "output_files": {
    "wiki": "/opt/ml/processing/output/wiki/abusedetectionmodel_documentation_20250115_103500.wiki",
    "html": "/opt/ml/processing/output/wiki/abusedetectionmodel_documentation_20250115_103500.html",
    "markdown": "/opt/ml/processing/output/wiki/abusedetectionmodel_documentation_20250115_103500.md"
  },
  "visualizations_processed": 6,
  "metrics_sources": ["metrics_report", "basic_metrics", "text_summary"]
}
```

## Key Functions and Tasks

### Data Ingestion Component

#### `DataIngestionManager.load_metrics_data(metrics_dir)`
**Purpose**: Load comprehensive metrics data from metrics computation output, supporting both comprehensive and basic formats.

**Algorithm**:
```python
1. Initialize empty metrics_data dictionary
2. Check for comprehensive metrics report (metrics_report.json):
   a. If exists, load JSON and store in metrics_data["metrics_report"]
   b. Log successful load
3. Check for basic metrics (metrics.json):
   a. If exists, load JSON and store in metrics_data["basic_metrics"]
   b. Log successful load
4. Check for text summary (metrics_summary.txt):
   a. If exists, read text content and store in metrics_data["text_summary"]
   b. Log successful load
5. Return metrics_data dictionary with all discovered sources
```

**Parameters**:
- `metrics_dir` (str): Path to metrics input directory

**Returns**: `Dict[str, Any]` - Dictionary containing all discovered metrics data sources

**Complexity**: O(n) where n is total size of metrics files

#### `DataIngestionManager.discover_visualization_files(plots_dir)`
**Purpose**: Discover and catalog visualization files for embedding, including both standard and model comparison plots.

**Algorithm**:
```python
1. Initialize empty visualizations dictionary
2. Check if plots_dir exists, return empty dict if not
3. Define standard_plot_types mapping (roc_curve, pr_curve, etc.)
4. Define comparison_plot_types mapping (comparison_roc_curves, etc.)
5. For each plot_type in combined plot_types:
   a. For each image extension (.jpg, .png, .jpeg, .svg):
      i. Construct plot_path = plots_dir / plot_type + extension
      ii. If plot_path exists:
          - Store plot info with path, description, filename, is_comparison flag
          - Log discovery
          - Break to next plot_type
6. Scan for class-specific plots (class_*_*.jpg pattern):
   a. For each match:
      i. Extract filename and create plot_key
      ii. Store plot info with generated description
      iii. Log discovery
7. Count and log comparison visualizations found
8. Return visualizations dictionary
```

**Parameters**:
- `plots_dir` (str): Path to plots input directory

**Returns**: `Dict[str, Dict[str, str]]` - Dictionary mapping plot types to plot information including path, description, filename, and comparison flag

**Complexity**: O(p * e + c) where p is number of plot types, e is number of extensions, c is number of class-specific plots

### Template Engine Component

#### `WikiTemplateEngine.generate_wiki_content(context)`
**Purpose**: Generate complete wiki content from context data using section templates.

**Algorithm**:
```python
1. Initialize empty wiki_sections list
2. For each (section_name, template) in self.sections.items():
   a. Try to format template with context variables:
      i. section_content = template.format(**context)
      ii. Append section_content to wiki_sections
   b. On KeyError (missing template variable):
      i. Log warning about missing variable
      ii. Continue to next section (skip this section)
3. Join all wiki_sections with newlines
4. Return complete wiki content string
```

**Parameters**:
- `context` (Dict[str, Any]): Template context with all variable values

**Returns**: `str` - Complete wiki markup content

**Complexity**: O(s * v) where s is number of sections, v is average template variables per section

#### `WikiTemplateEngine._build_template_context(metrics_data, config_data)`
**Purpose**: Build comprehensive context dictionary for template rendering from all data sources.

**Algorithm**:
```python
1. Initialize empty context dictionary
2. Extract metrics information:
   a. If metrics_report exists in metrics_data:
      i. Extract standard_metrics (auc_roc, average_precision, etc.)
      ii. Extract domain_metrics (dollar_recall, count_recall, etc.)
      iii. Store in context
3. Extract configuration information:
   a. If config exists in config_data:
      i. Extract pipeline_name, model_name, region, author, etc.
      ii. Store in context
4. Add environment-based overrides:
   a. Get MODEL_NAME, MODEL_USE_CASE from environment
   b. Generate last_updated timestamp
   c. Get MODEL_VERSION from environment
   d. Store all environment overrides in context
5. Generate derived content (call _generate_derived_content)
6. Return comprehensive context dictionary
```

**Parameters**:
- `metrics_data` (Dict[str, Any]): Metrics data from ingestion
- `config_data` (Dict[str, Any]): Configuration data

**Returns**: `Dict[str, Any]` - Comprehensive template context

**Complexity**: O(m + c + d) where m is metrics extraction, c is config extraction, d is derived content generation

### Content Generation Component

#### `ContentGenerator.generate_performance_assessment(auc_score)`
**Purpose**: Generate qualitative performance assessment based on AUC score ranges.

**Algorithm**:
```python
1. If auc_score >= 0.9: return "excellent"
2. Else if auc_score >= 0.8: return "good"
3. Else if auc_score >= 0.7: return "fair"
4. Else: return "poor"
```

**Parameters**:
- `auc_score` (float): AUC-ROC score value between 0.0 and 1.0

**Returns**: `str` - Qualitative assessment ("excellent", "good", "fair", "poor")

**Complexity**: O(1)

#### `ContentGenerator.generate_business_impact_summary(dollar_recall, count_recall, total_abuse_amount)`
**Purpose**: Generate business impact summary based on available domain metrics.

**Algorithm**:
```python
1. Initialize empty impact_statements list
2. If dollar_recall is not None:
   a. If dollar_recall >= 0.8:
      - Append "High dollar recall..." statement
   b. Else if dollar_recall >= 0.7:
      - Append "Moderate dollar recall..." statement
   c. Else:
      - Append "Low dollar recall..." statement
3. If count_recall is not None:
   a. If count_recall >= 0.8:
      - Append "High count recall..." statement
   b. Else if count_recall >= 0.6:
      - Append "Moderate count recall..." statement
   c. Else:
      - Append "Low count recall..." statement
4. If total_abuse_amount is not None:
   - Append "Model protects against $X..." statement
5. Join impact_statements with ". " and add final period
6. If no statements, return default message
7. Return complete business impact summary
```

**Parameters**:
- `dollar_recall` (float, optional): Dollar recall metric value
- `count_recall` (float, optional): Count recall metric value
- `total_abuse_amount` (float, optional): Total abuse amount protected

**Returns**: `str` - Business impact summary text

**Complexity**: O(1)

#### `ContentGenerator.detect_comparison_mode(metrics)`
**Purpose**: Detect if comparison metrics are present in the data to enable comparison-specific documentation sections.

**Algorithm**:
```python
1. Define comparison_indicators list:
   - ["auc_delta", "ap_delta", "pearson_correlation", "spearman_correlation",
      "new_model_auc", "previous_model_auc", "mcnemar_p_value", "paired_t_p_value"]
2. For each indicator in comparison_indicators:
   a. If indicator exists in metrics dictionary:
      - Return True immediately
3. If no indicators found, return False
```

**Parameters**:
- `metrics` (Dict[str, Any]): Metrics dictionary to check

**Returns**: `bool` - True if comparison mode detected, False otherwise

**Complexity**: O(i) where i is number of comparison indicators (constant = 8)

#### `ContentGenerator.generate_comparison_summary(metrics)`
**Purpose**: Generate comprehensive model comparison summary including deltas, correlations, and deployment recommendations.

**Algorithm**:
```python
1. Initialize empty summary_parts list
2. AUC Comparison Analysis:
   a. Extract auc_delta, new_model_auc, prev_model_auc, lift_percent
   b. If auc_delta > 0.01:
      - Append "significant improvement" statement
   c. Else if auc_delta > 0.005:
      - Append "marginal improvement" statement
   d. Else if -0.005 <= auc_delta <= 0.005:
      - Append "similar performance" statement
   e. Else:
      - Append "performance degradation" statement
3. Average Precision Comparison Analysis:
   a. Extract ap_delta, ap_lift_percent
   b. If ap_delta > 0.01:
      - Append "Average Precision improved" statement
   c. Else if ap_delta < -0.01:
      - Append "Average Precision decreased" statement
4. Correlation Analysis:
   a. Extract pearson_correlation
   b. If correlation > 0.9:
      - Append "highly correlated" statement
   c. Else if correlation > 0.7:
      - Append "good correlation" statement
   d. Else if correlation > 0.5:
      - Append "moderate correlation" statement
   e. Else:
      - Append "low correlation" statement
5. Join summary_parts with ". " and add final period
6. Return complete comparison summary
```

**Parameters**:
- `metrics` (Dict[str, Any]): Metrics dictionary with comparison data

**Returns**: `str` - Comprehensive comparison summary text

**Complexity**: O(1)

#### `ContentGenerator.generate_statistical_significance_summary(metrics)`
**Purpose**: Generate summary of statistical significance tests for model comparison.

**Algorithm**:
```python
1. Initialize empty significance_parts list
2. McNemar's Test Analysis:
   a. Extract mcnemar_p_value, mcnemar_significant
   b. If p-value exists:
      i. If mcnemar_significant is True:
         - Append "statistically significant difference" statement
      ii. Else:
         - Append "no significant difference" statement
3. Paired T-Test Analysis:
   a. Extract paired_t_p_value, paired_t_significant
   b. If p-value exists:
      i. If paired_t_significant is True:
         - Append "significant score differences" statement
      ii. Else:
         - Append "no significant score differences" statement
4. Wilcoxon Test Analysis:
   a. Extract wilcoxon_p_value, wilcoxon_significant
   b. If p-value exists and not NaN:
      i. If wilcoxon_significant is True:
         - Append "supports significant differences" statement
      ii. Else:
         - Append "no significant differences" statement
5. Join significance_parts with ". " and add final period
6. If no parts, return default message
7. Return statistical significance summary
```

**Parameters**:
- `metrics` (Dict[str, Any]): Metrics dictionary with statistical test results

**Returns**: `str` - Statistical significance summary text

**Complexity**: O(1)

#### `ContentGenerator.generate_deployment_recommendation(metrics)`
**Purpose**: Generate deployment recommendation based on comparison results combining performance deltas and statistical validation.

**Algorithm**:
```python
1. Extract auc_delta, mcnemar_significant, paired_t_significant from metrics
2. If auc_delta > 0.01 AND (mcnemar_sig OR paired_t_sig):
   - Return "✅ RECOMMENDED FOR DEPLOYMENT" with rationale
3. Else if auc_delta > 0.005:
   - Return "⚠️ CONSIDER FOR DEPLOYMENT" with caution note
4. Else if |auc_delta| <= 0.005:
   - Return "≈ SIMILAR PERFORMANCE" with alternative considerations
5. Else:
   - Return "❌ NOT RECOMMENDED" with degradation warning
```

**Parameters**:
- `metrics` (Dict[str, Any]): Metrics dictionary with comparison data

**Returns**: `str` - Deployment recommendation with visual indicator and rationale

**Complexity**: O(1)

### Visualization Integration Component

#### `VisualizationIntegrator.process_visualizations(visualizations)`
**Purpose**: Process and prepare visualizations for wiki embedding including copying, optimization, and metadata generation.

**Algorithm**:
```python
1. Initialize empty processed_images dictionary
2. For each (plot_type, plot_info) in visualizations.items():
   a. Try:
      i. Extract source_path from plot_info
      ii. Generate dest_filename = f"{plot_type}_{date}.jpg"
      iii. Construct dest_path in self.image_dir
      iv. Copy image from source to destination
      v. Store image reference: processed_images[f"{plot_type}_image"] = dest_filename
      vi. Generate and store description: processed_images[f"{plot_type}_description"] = description
      vii. Log successful processing
   b. On Exception:
      i. Log warning about failed processing
      ii. Continue to next visualization
3. Return processed_images dictionary
```

**Parameters**:
- `visualizations` (Dict[str, Dict[str, str]]): Visualization catalog from discovery

**Returns**: `Dict[str, str]` - Mapping of plot types to image references and descriptions

**Complexity**: O(v) where v is number of visualizations

### Report Assembly Component

#### `WikiReportAssembler.assemble_complete_report(metrics_data, processed_images, environ_vars)`
**Purpose**: Assemble complete wiki report from all components by building context and generating content.

**Algorithm**:
```python
1. Build comprehensive context:
   a. Call _build_comprehensive_context(metrics_data, processed_images, environ_vars)
   b. Returns context dictionary with all template variables
2. Generate wiki content:
   a. Call template_engine.generate_wiki_content(context)
   b. Returns complete wiki markup string
3. Return final wiki content
```

**Parameters**:
- `metrics_data` (Dict[str, Any]): Metrics data from ingestion
- `processed_images` (Dict[str, str]): Processed visualization references
- `environ_vars` (Dict[str, str]): Environment variables

**Returns**: `str` - Complete assembled wiki documentation

**Complexity**: O(s * v + c) where s is sections, v is template variables, c is content generation

#### `WikiReportAssembler._generate_comparison_sections(context, standard_metrics)`
**Purpose**: Generate comparison-specific sections when model comparison data is detected.

**Algorithm**:
```python
1. Initialize empty sections dictionary
2. Detect comparison mode:
   a. Call content_generator.detect_comparison_mode(standard_metrics)
   b. If not comparison mode: return empty comparison sections
3. Generate Model Comparison Summary Section:
   a. Generate comparison summary text
   b. Generate statistical significance summary
   c. Generate deployment recommendation
   d. Format as wiki section with heading and content
4. Generate Comparison Visualizations Section:
   a. Initialize empty comparison_viz_parts list
   b. If comparison_roc_curves_image exists:
      i. Add ROC comparison subsection with image embed
   c. If comparison_pr_curves_image exists:
      i. Add PR comparison subsection with image embed
   d. If score_scatter_plot_image exists:
      i. Add score correlation subsection with image embed
   e. If score_distributions_image exists:
      i. Add score distribution comparison subsection
   f. If individual model images exist (new/previous):
      i. Add individual model performance subsections
   g. Combine all visualization parts into complete section
5. Store both sections in sections dictionary
6. Return sections dictionary
```

**Parameters**:
- `context` (Dict[str, Any]): Template context with processed images
- `standard_metrics` (Dict[str, Any]): Standard metrics including comparison data

**Returns**: `Dict[str, Any]` - Dictionary with comparison-specific section content

**Complexity**: O(v) where v is number of comparison visualizations

### Output Management Component

#### `WikiOutputManager.save_wiki_documentation(wiki_content, model_name, formats)`
**Purpose**: Save wiki documentation in multiple formats with timestamp-based file naming.

**Algorithm**:
```python
1. Initialize empty output_files dictionary
2. Generate base filename:
   a. Sanitize model_name (remove invalid characters)
   b. Generate timestamp string (YYYYMMDD_HHMMSS)
   c. Construct base_filename = f"{safe_model_name}_documentation_{timestamp}"
3. For each format_type in formats:
   a. Try:
      i. If format_type == "wiki":
         - Call _save_wiki_format(wiki_content, base_filename)
      ii. Else if format_type == "html":
         - Call _save_html_format(wiki_content, base_filename)
      iii. Else if format_type == "markdown":
         - Call _save_markdown_format(wiki_content, base_filename)
      iv. Else:
         - Log warning about unknown format
         - Continue to next format
      v. Store file_path in output_files[format_type]
      vi. Log successful save
   b. On Exception:
      i. Log error for format save failure
      ii. Continue to next format
4. Return output_files dictionary
```

**Parameters**:
- `wiki_content` (str): Complete wiki markup content
- `model_name` (str): Model name for file naming
- `formats` (List[str]): List of output formats to generate

**Returns**: `Dict[str, str]` - Mapping of format types to output file paths

**Complexity**: O(f * c) where f is number of formats, c is conversion complexity

## Algorithms and Data Structures

### Wiki Markup to HTML Conversion Algorithm
**Problem**: Convert MediaWiki markup to styled HTML while preserving structure and embedded images.

**Solution Strategy**:
1. Apply regex-based transformations for markup elements
2. Use multi-pass processing for nested structures
3. Embed CSS styling in HTML template
4. Handle tables with special parsing logic

**Algorithm**:
```python
# Multi-pass markup conversion
1. Header Conversion:
   - Replace "= text =" with "<h1>text</h1>"
   - Replace "== text ==" with "<h2>text</h2>"
   - Replace "=== text ===" with "<h3>text</h3>"
2. Table Conversion:
   - Parse wiki table format "|cell1|cell2|"
   - Convert to HTML table structure with <tr><td> tags
   - Apply styling to table elements
3. Image Conversion:
   - Replace "[[Image:file|thumb|caption]]" with HTML img tags
   - Add wrapper divs for styling
   - Include caption as paragraph
4. List Conversion:
   - Replace "* item" with "<li>item</li>"
   - Wrap consecutive list items in <ul> tags
5. Text Formatting:
   - Replace "**text**" with "<strong>text</strong>"
   - Convert line breaks appropriately
6. Template Application:
   - Embed converted content in HTML template with CSS
```

**Complexity**: 
- Time: O(n * p) where n is content length, p is number of passes (constant = 6)
- Space: O(n) for storing converted content

**Key Features**:
- **Multi-pass processing** ensures proper handling of nested structures
- **Regex-based conversion** provides flexible pattern matching
- **CSS embedding** creates self-contained HTML output
- **Error resilience** handles malformed markup gracefully

### Wiki Markup to Markdown Conversion Algorithm
**Problem**: Convert MediaWiki markup to GitHub-compatible Markdown.

**Solution Strategy**:
1. Map wiki syntax to markdown equivalents
2. Handle table conversion with special care
3. Convert image syntax to markdown format
4. Preserve list structure

**Algorithm**:
```python
# Direct syntax mapping conversion
1. Header Conversion:
   - Replace "= text =" with "# text"
   - Replace "== text ==" with "## text"
   - Replace "=== text ===" with "### text"
2. Image Conversion:
   - Replace "[[Image:file|thumb|caption]]" with "![caption](images/file)"
3. Table Conversion:
   - Parse wiki table format
   - Convert to markdown table format with | separators
   - Add header separator row with ---
4. List Preservation:
   - Wiki lists use "*", markdown lists use "*" (same syntax)
   - Preserve list structure during conversion
```

**Complexity**: 
- Time: O(n) where n is content length
- Space: O(n) for storing converted content

**Key Features**:
- **Direct syntax mapping** provides straightforward conversion
- **Table special handling** ensures proper markdown table format
- **Image format preservation** maintains visual content
- **List compatibility** simplifies conversion process

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Load metrics data | O(n) | O(n) | n = total size of metrics files |
| Discover visualizations | O(p * e + c) | O(v) | p = plot types, e = extensions, c = class-specific, v = visualizations found |
| Process visualizations | O(v) | O(v) | v = number of visualizations |
| Generate wiki content | O(s * t) | O(s * t) | s = sections, t = template size |
| Convert to HTML | O(n * p) | O(n) | n = content length, p = conversion passes (6) |
| Convert to Markdown | O(n) | O(n) | n = content length |
| Save all formats | O(f * n) | O(f * n) | f = formats (3), n = content length |

### Processing Mode Characteristics

| Characteristic | Single Model | Model Comparison |
|----------------|--------------|------------------|
| Metrics complexity | Standard | Extended with deltas and statistical tests |
| Visualizations | 4-6 plots | 8-12 plots (includes comparison plots) |
| Content generation | 6 sections | 8 sections (adds comparison sections) |
| Processing time | ~2-5 seconds | ~5-10 seconds |
| Output size | 50-100 KB | 100-200 KB |

## Error Handling

### Error Types

#### Input Validation Errors
- **Missing Metrics Data**: Script creates basic documentation with default values and logs warning about missing comprehensive metrics
- **Missing Visualizations**: Script generates documentation without visualization sections, logs warning about missing plots
- **Invalid Format**: Script attempts to parse available data formats, falls back to basic metrics if comprehensive report unavailable

#### Processing Errors
- **Template Rendering Errors**: Script skips sections with missing template variables, logs warnings, continues with remaining sections
- **Image Processing Errors**: Script logs warnings about failed image processing, continues with other visualizations
- **Format Conversion Errors**: Script attempts all requested formats, logs errors for failed conversions, succeeds if at least one format works

#### Output Errors
- **File System Errors**: Script creates necessary directories automatically, raises exception if write permissions insufficient
- **Invalid Characters in Filenames**: Script sanitizes model names to remove invalid characters, replaces with underscores

### Error Response Structure
```python
# Success markers
{
    "_SUCCESS": "Created after successful completion",
    "_HEALTH": "Contains timestamp: healthy: 2025-01-15T10:35:00Z"
}

# Failure marker
{
    "_FAILURE": "Contains error message: Error: <error description>"
}

# Partial success
{
    "generation_summary.json": {
        "status": "partial",
        "warnings": ["Missing visualization: roc_curve.jpg"],
        "formats_generated": ["wiki", "html"],
        "formats_failed": ["markdown"]
    }
}
```

## Best Practices

### For Production Deployments
1. **Environment Configuration**: Always set MODEL_NAME and other metadata variables for proper attribution and tracking
2. **Format Selection**: Choose output formats based on downstream consumer requirements (wiki for model registries, HTML for presentations, Markdown for documentation repos)
3. **Visualization Availability**: Ensure upstream metrics computation step generates all expected visualizations before running wiki generator
4. **Metadata Completeness**: Provide comprehensive environment variables (AUTHOR, TEAM_ALIAS, CONTACT_EMAIL, etc.) for complete documentation

### For Development and Testing
1. **Start with Basic Metrics**: Test with minimal metrics.json to ensure basic documentation generation works
2. **Add Visualizations Incrementally**: Test visualization integration one plot type at a time
3. **Test Format Conversion**: Verify all output formats render correctly in target systems
4. **Validate Template Variables**: Check that all required template variables are present in context

### For Model Comparison Documentation
1. **Provide Both Models**: Ensure metrics computation provides both new and previous model data
2. **Include Statistical Tests**: Enable statistical significance testing in upstream metrics computation
3. **Review Deployment Recommendations**: Verify deployment recommendations align with business requirements
4. **Document Comparison Context**: Include clear descriptions of what models are being compared

### For Maintenance and Updates
1. **Version Documentation**: Use MODEL_VERSION environment variable to track documentation versions
2. **Archive Old Documentation**: Implement retention policy for wiki documentation files
3. **Monitor Output Sizes**: Track documentation file sizes over time to detect anomalies
4. **Review Generated Content**: Periodically review generated documentation for quality and accuracy

## Example Configurations

### Example 1: Basic Single Model Documentation
```bash
export MODEL_NAME="FraudDetectionModel"
export MODEL_USE_CASE="Credit card fraud detection for payment processing"
export MODEL_VERSION="2.1"
export PIPELINE_NAME="FraudDetection-Production"
export AUTHOR="ML Platform Team"
export TEAM_ALIAS="ml-platform@company.com"
export CONTACT_EMAIL="ml-platform@company.com"
export REGION="us-west-2"
export OUTPUT_FORMATS="wiki,html,markdown"
```

**Use Case**: Standard production model documentation with comprehensive metadata for model registry integration and compliance reporting.

### Example 2: Model Comparison Documentation
```bash
export MODEL_NAME="FraudDetectionModel_v2_vs_v1"
export MODEL_USE_CASE="A/B comparison: Fraud detection model v2.1 vs v2.0"
export MODEL_VERSION="2.1"
export MODEL_DESCRIPTION="Comparison of new fraud detection model (v2.1 with additional features) against previous production model (v2.0) to evaluate performance improvements"
export PIPELINE_NAME="FraudDetection-Comparison"
export AUTHOR="ML Platform Team"
export OUTPUT_FORMATS="html,markdown"
export INCLUDE_TECHNICAL_DETAILS="true"
```

**Use Case**: A/B testing documentation with model comparison analysis, statistical significance testing, and deployment recommendations for production rollout decisions.

### Example 3: Minimal Documentation
```bash
export MODEL_NAME="ExperimentalModel"
export OUTPUT_FORMATS="markdown"
export INCLUDE_TECHNICAL_DETAILS="false"
```

**Use Case**: Quick experimental model documentation with minimal configuration for internal team sharing and rapid iteration.

### Example 4: Compliance Documentation
```bash
export MODEL_NAME="RiskAssessmentModel"
export MODEL_USE_CASE="Financial risk assessment for loan approval decisions"
export MODEL_VERSION="1.0"
export MODEL_DESCRIPTION="Production risk assessment model complying with regulatory requirements for explainability and fairness"
export PIPELINE_NAME="RiskAssessment-Compliance"
export AUTHOR="Risk Analytics Team"
export TEAM_ALIAS="risk-analytics@company.com"
export CONTACT_EMAIL="compliance-ml@company.com"
export CTI_CLASSIFICATION="Confidential - Financial Services"
export REGION="us-east-1"
export OUTPUT_FORMATS="wiki,html,markdown"
export INCLUDE_TECHNICAL_DETAILS="true"
```

**Use Case**: Comprehensive compliance documentation for regulated financial services models with full technical details, business impact analysis, and regulatory attribution.

## Integration Patterns

### Upstream Integration
```
ModelMetricsComputation (or XGBoostModelEval)
   ↓ (outputs: metrics_report.json, metrics.json, visualizations)
   ↓ (S3 paths: metrics_output → metrics_input)
   ↓ (S3 paths: plots_output → plots_input)
ModelWikiGenerator
   ↓ (outputs: documentation files in multiple formats)
```

**Key Dependencies**:
- **Metrics Input**: Requires `metrics_output` from ModelMetricsComputation or `metrics_output` from XGBoostModelEval
- **Plots Input**: Requires `plots_output` from ModelMetricsComputation (or visualizations from XGBoostModelEval)
- **Path Alignment**: Input logical names (`metrics_input`, `plots_input`) must match upstream output logical names

### Downstream Integration
```
ModelWikiGenerator
   ↓ (wiki_output: documentation files)
   ├→ Model Registry (consumes wiki format)
   ├→ Knowledge Base (consumes markdown format)
   ├→ Presentation System (consumes HTML format)
   └→ Compliance System (archives all formats)
```

**Output Consumers**:
- **Model Registry**: Uses wiki format for searchable model catalog
- **Knowledge Base**: Uses markdown for version-controlled documentation
- **Presentation System**: Uses HTML for stakeholder presentations
- **Compliance System**: Archives all formats for regulatory requirements

### Complete Workflow Example
```
1. Training Phase:
   XGBoostTraining → model.tar.gz

2. Evaluation Phase:
   XGBoostModelEval (or ModelMetricsComputation)
   ├→ metrics_output/metrics_report.json
   ├→ metrics_output/metrics.json
   ├→ plots_output/roc_curve.jpg
   ├→ plots_output/pr_curve.jpg
   └→ plots_output/score_distribution.jpg

3. Documentation Phase:
   ModelWikiGenerator
   ├→ wiki_output/model_documentation_20251118.wiki
   ├→ wiki_output/model_documentation_20251118.html
   ├→ wiki_output/model_documentation_20251118.md
   └→ wiki_output/generation_summary.json

4. Distribution Phase:
   ├→ Upload wiki to Model Registry
   ├→ Commit markdown to Documentation Repo
   ├→ Send HTML to Stakeholders
   └→ Archive all formats in Compliance System
```

## Troubleshooting

### Issue: Missing Visualizations in Documentation

**Symptom**: Generated documentation is missing ROC curves, PR curves, or other performance visualizations.

**Common Causes**:
1. **Upstream Step Failure**: Metrics computation step failed to generate visualization files
2. **Path Mismatch**: Visualization files not in expected `plots_input` directory
3. **File Format**: Visualization files in unsupported format (only .jpg, .png, .jpeg, .svg supported)
4. **File Naming**: Visualization files don't match expected naming conventions

**Solution**:
1. Check upstream metrics computation logs for visualization generation
2. Verify plots are in `/opt/ml/processing/input/plots/` directory
3. List files in plots directory to confirm presence and format
4. Check file names match expected patterns: `roc_curve.jpg`, `pr_curve.jpg`, etc.
5. Review script logs for visualization discovery messages

### Issue: Template Rendering Errors

**Symptom**: Generated documentation has missing sections or placeholder text like `{variable_name}`.

**Common Causes**:
1. **Missing Metrics**: Required metrics (auc_roc, average_precision) not present in input data
2. **Missing Environment Variables**: Required environment variable MODEL_NAME not set
3. **Incomplete Metrics Report**: Metrics report missing expected fields (standard_metrics, domain_metrics)

**Solution**:
1. Verify MODEL_NAME environment variable is set
2. Check metrics_report.json contains all expected fields
3. Review script logs for template variable warnings
4. Provide fallback values for optional metrics
5. Ensure upstream metrics computation completed successfully

### Issue: Format Conversion Failures

**Symptom**: Script generates some output formats but not others (e.g., wiki and HTML succeed but markdown fails).

**Common Causes**:
1. **Conversion Logic Error**: Specific format converter has bugs with certain content patterns
2. **File System Issues**: Write permissions or disk space issues for specific format
3. **Character Encoding**: Special characters causing issues in specific format

**Solution**:
1. Check script logs for format-specific error messages
2. Verify write permissions to output directory
3. Test with minimal content to isolate conversion logic issues
4. Check disk space availability
5. Review content for special characters that may need escaping

### Issue: Large Output Files

**Symptom**: Generated documentation files are unexpectedly large (>10 MB).

**Common Causes**:
1. **Unoptimized Images**: Visualization files not optimized for web display
2. **Embedded Data**: Large data tables or text summaries embedded in documentation
3. **Redundant Content**: Duplicate sections or repeated content

**Solution**:
1. Check size of individual visualization files in plots input
2. Consider enabling image optimization (if PIL available)
3. Review generated content for unnecessary duplication
4. Reduce number of class-specific plots for multiclass models
5. Use external links instead of embedding large tables

### Issue: Comparison Mode Not Detected

**Symptom**: Model comparison documentation doesn't include comparison sections despite providing comparison metrics.

**Common Causes**:
1. **Missing Comparison Indicators**: Required comparison fields (auc_delta, new_model_auc) not present in metrics
2. **Wrong Metrics Format**: Comparison metrics in basic_metrics instead of standard_metrics within metrics_report
3. **Field Naming**: Comparison field names don't match expected patterns

**Solution**:
1. Verify upstream metrics computation is in comparison mode
2. Check metrics_report.json contains auc_delta, new_model_auc, previous_model_auc fields
3. Ensure comparison metrics are in standard_metrics section of metrics_report
4. Review comparison mode detection logic indicators
5. Test with known comparison metrics data

## References

### Related Scripts
- [`model_metrics_computation.py`](model_metrics_computation_script.md): Upstream metrics computation providing comprehensive metrics reports and visualizations
- [`xgboost_model_eval.py`](xgboost_model_eval_script.md): Alternative upstream source for basic metrics and visualizations
- [`xgboost_model_inference.py`](xgboost_model_inference_script.md): Inference step that can feed into model evaluation

### Related Documentation
- **Step Builder**: Step builder implementation not yet documented
- **Config Class**: Configuration class not yet documented
- **Contract**: [`src/cursus/steps/contracts/model_wiki_generator_contract.py`](../../src/cursus/steps/contracts/model_wiki_generator_contract.py)
- **Step Specification**: Defined in step builder code

### Related Design Documents
- **[Model Wiki Generator Design](../1_design/model_wiki_generator_design.md)**: Comprehensive design document covering six-component architecture, template system, content generation algorithms, and integration patterns
- **[Model Metrics Computation Design](../1_design/model_metrics_computation_design.md)**: Design for upstream metrics computation including comparison mode and statistical significance testing
- **[Automatic Documentation Generation Design](../1_design/automatic_documentation_generation_design.md)**: Broader design for automated documentation generation across the ML pipeline

### External References
- [MediaWiki Markup Specification](https://www.mediawiki.org/wiki/Help:Formatting): MediaWiki markup syntax reference for wiki format output
- [GitHub Flavored Markdown](https://github.github.com/gfm/): Markdown specification for markdown format output
- [HTML5 Specification](https://html.spec.whatwg.org/): HTML5 standard for HTML format output
