"""
Model Wiki Generator Script Contract

Defines the contract for the model wiki generator script that loads metrics data and visualizations,
generates comprehensive wiki documentation, and creates multi-format model documentation.
"""

from ...core.base.contract_base import ScriptContract

MODEL_WIKI_GENERATOR_CONTRACT = ScriptContract(
    entry_point="model_wiki_generator.py",
    expected_input_paths={
        "metrics_output": "/opt/ml/processing/input/metrics",
        "plots_output": "/opt/ml/processing/input/plots",
    },
    expected_output_paths={
        "wiki_output": "/opt/ml/processing/output/wiki",
    },
    expected_arguments={
        # No expected arguments - configuration comes from environment variables
    },
    required_env_vars=["MODEL_NAME"],
    optional_env_vars={
        "MODEL_USE_CASE": "Machine Learning Model",
        "MODEL_VERSION": "1.0",
        "PIPELINE_NAME": "ML Pipeline",
        "AUTHOR": "ML Team",
        "TEAM_ALIAS": "ml-team@",
        "CONTACT_EMAIL": "ml-team@company.com",
        "CTI_CLASSIFICATION": "Internal",
        "REGION": "Global",
        "OUTPUT_FORMATS": "wiki,html,markdown",
        "INCLUDE_TECHNICAL_DETAILS": "true",
        "MODEL_DESCRIPTION": "",
        "MODEL_PURPOSE": "perform classification tasks",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "matplotlib": ">=3.5.0",
    },
    description="""
    Model wiki generator script that:
    1. Loads metrics data from model metrics computation or xgboost model eval output
    2. Discovers and processes performance visualizations (ROC curves, PR curves, etc.)
    3. Generates intelligent content based on model performance data
    4. Creates comprehensive wiki documentation with business insights
    5. Outputs documentation in multiple formats (Wiki, HTML, Markdown)
    6. Provides automated model documentation for registries and compliance
    
    Input Structure:
    - /opt/ml/processing/input/metrics: Metrics data directory containing:
      - metrics_report.json: Comprehensive metrics report from model_metrics_computation with:
        - timestamp: Generation timestamp
        - data_summary: Data statistics and information
        - standard_metrics: Standard ML metrics (AUC-ROC, precision, recall, F1)
        - domain_metrics: Domain-specific metrics (dollar_recall, count_recall, total_abuse_amount)
        - performance_insights: Generated insights about model performance
        - recommendations: Actionable recommendations for model improvement
        - visualizations: References to generated visualization files
      - metrics.json: Basic metrics in JSON format (from xgboost_model_eval or model_metrics_computation)
      - metrics_summary.txt: Human-readable metrics summary
    - /opt/ml/processing/input/plots: Visualization files directory containing:
      - roc_curve.jpg: ROC curve visualization
      - pr_curve.jpg or precision_recall_curve.jpg: Precision-Recall curve visualization
      - score_distribution.jpg: Score distribution plot
      - threshold_analysis.jpg: Threshold analysis plot
      - multiclass_roc_curves.jpg: Combined multiclass ROC curves (if applicable)
      - class_*_roc_curve.jpg: Per-class ROC curves (multiclass)
      - class_*_pr_curve.jpg: Per-class PR curves (multiclass)
    
    Output Structure:
    - /opt/ml/processing/output/wiki/{model_name}_documentation_{timestamp}.wiki: Wiki format documentation
    - /opt/ml/processing/output/wiki/{model_name}_documentation_{timestamp}.html: HTML format documentation
    - /opt/ml/processing/output/wiki/{model_name}_documentation_{timestamp}.md: Markdown format documentation
    - /opt/ml/processing/output/wiki/images/: Directory containing processed visualization images
      - {plot_type}_{date}.jpg: Processed and optimized visualization images
    - /opt/ml/processing/output/wiki/generation_summary.json: Summary of generation process with:
      - timestamp: Generation timestamp
      - model_name: Model name used for documentation
      - output_formats: List of generated output formats
      - output_files: Dictionary mapping formats to file paths
      - visualizations_processed: Number of visualizations processed
      - metrics_sources: List of metrics data sources used
    
    Environment Variables:
    - MODEL_NAME (required): Name of the model for documentation
    - MODEL_USE_CASE (optional): Description of model use case
    - MODEL_VERSION (optional): Model version identifier
    - PIPELINE_NAME (optional): Name of the ML pipeline
    - AUTHOR (optional): Model author/creator
    - TEAM_ALIAS (optional): Team email alias
    - CONTACT_EMAIL (optional): Point of contact email
    - CTI_CLASSIFICATION (optional): CTI classification for the model
    - REGION (optional): AWS region or deployment region
    - OUTPUT_FORMATS (optional): Comma-separated list of output formats ("wiki,html,markdown")
    - INCLUDE_TECHNICAL_DETAILS (optional): Include technical details section ("true"/"false")
    - MODEL_DESCRIPTION (optional): Custom model description text
    - MODEL_PURPOSE (optional): Custom model purpose description
    
    Features:
    - Multi-Source Input: Compatible with both model_metrics_computation and xgboost_model_eval outputs
    - Intelligent Content Generation: Automatically generates performance assessments and business insights
    - Multi-Format Output: Generates documentation in Wiki, HTML, and Markdown formats
    - Visualization Integration: Embeds performance plots with intelligent descriptions
    - Business Impact Analysis: Includes dollar recall, count recall, and financial impact analysis
    - Template-Driven: Uses configurable templates for consistent documentation structure
    - Comprehensive Reporting: Includes performance metrics, recommendations, and technical details
    - Asset Management: Handles image processing and optimization for web display
    - Error Resilience: Graceful handling of missing data or visualization files
    
    Dependency Alignment:
    - Input Compatibility: Designed to consume output from model_metrics_computation_contract
      - metrics_output logical name matches model_metrics_computation_contract.expected_output_paths["metrics_output"]
      - plots_output logical name matches model_metrics_computation_contract.expected_output_paths["plots_output"]
    - Also compatible with xgboost_model_eval_contract output:
      - metrics_output logical name matches xgboost_model_eval_contract.expected_output_paths["metrics_output"]
      - Can process basic metrics.json and visualization files from xgboost model evaluation
    - Supports graceful degradation when comprehensive metrics are not available
    
    Documentation Structure:
    Generated documentation includes:
    1. **Header Section**: Model metadata, pipeline information, contact details
    2. **Summary Section**: Model description, key performance metrics, business impact
    3. **Performance Analysis**: Detailed performance metrics with visualizations
    4. **Business Impact Analysis**: Dollar recall, count recall, financial impact
    5. **Recommendations**: Actionable recommendations for model improvement
    6. **Technical Details**: Model configuration, data information, feature importance
    
    Format Conversion:
    - **Wiki Format**: MediaWiki markup for wiki systems
    - **HTML Format**: Styled HTML with CSS for web viewing and presentations
    - **Markdown Format**: GitHub-compatible markdown for documentation repositories
    
    Integration Points:
    - **Upstream**: model_metrics_computation, xgboost_model_eval
    - **Downstream**: Model registries, knowledge bases, compliance systems
    - **Parallel**: Can run alongside other documentation or reporting steps
    
    Error Handling:
    - Validates input data structure and provides detailed error messages
    - Gracefully handles missing visualization files or incomplete metrics
    - Creates success/failure markers for pipeline monitoring
    - Comprehensive logging for debugging and monitoring
    """,
)
