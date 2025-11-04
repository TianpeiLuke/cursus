"""
PyTorch Model Evaluation Script Contract

Defines the contract for the PyTorch model evaluation script that loads trained models,
processes evaluation data, and generates performance metrics and visualizations.
"""

from ...core.base.contract_base import ScriptContract

PYTORCH_MODEL_EVAL_CONTRACT = ScriptContract(
    entry_point="pytorch_model_eval.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "processed_data": "/opt/ml/processing/input/eval_data",
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
        "metrics_output": "/opt/ml/processing/output/metrics",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    optional_env_vars={
        "COMPARISON_MODE": "false",
        "PREVIOUS_SCORE_FIELD": "",
        "COMPARISON_METRICS": "all",
        "STATISTICAL_TESTS": "true",
        "COMPARISON_PLOTS": "true",
    },
    framework_requirements={
        "torch": "==2.1.2",
        "torchvision": "==0.16.2",
        "torchaudio": "==2.1.2",
        "transformers": "==4.37.2",
        "lightning": "==2.1.3",
        "lightning-utilities": "==0.10.1",
        "torchmetrics": "==1.7.1",
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "matplotlib": ">=3.5.0",
        "scipy": ">=1.7.0",
        "beautifulsoup4": "==4.12.3",
        "gensim": "==4.3.1",
        "pydantic": "==2.11.2",
        "onnx": "==1.15.0",
        "onnxruntime": "==1.17.0",
    },
    description="""
    PyTorch model evaluation script that:
    1. Loads trained PyTorch model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using text tokenization and tabular processing
    3. Generates predictions and computes performance metrics
    4. Creates ROC and Precision-Recall curve visualizations
    5. Optionally compares performance with previous model scores
    6. Saves predictions, metrics, plots, and comparison reports
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - model.pth: Trained PyTorch model state dict
      - model_artifacts.pth: Model artifacts (config, embeddings, vocab, model_class)
      - hyperparameters.json: Complete model configuration and hyperparameters
      - model.onnx: ONNX exported model (optional)
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV or Parquet files)
      - For comparison mode: must include column with previous model scores
    
    Standard Output Structure:
    - /opt/ml/processing/output/eval/eval_predictions.csv: Model predictions with probabilities
    - /opt/ml/processing/output/metrics/metrics.json: Performance metrics
    - /opt/ml/processing/output/metrics/roc_curve.jpg: ROC curve visualization
    - /opt/ml/processing/output/metrics/pr_curve.jpg: Precision-Recall curve visualization
    - /opt/ml/processing/output/metrics/metrics_summary.txt: Human-readable metrics summary
    
    Additional Output Structure (Comparison Mode):
    - /opt/ml/processing/output/eval/eval_predictions_with_comparison.csv: Enhanced predictions with both models
    - /opt/ml/processing/output/metrics/comparison_report.txt: Executive summary with recommendations
    - /opt/ml/processing/output/metrics/comparison_roc_curves.jpg: Side-by-side ROC comparison
    - /opt/ml/processing/output/metrics/comparison_pr_curves.jpg: Side-by-side PR comparison
    - /opt/ml/processing/output/metrics/score_scatter_plot.jpg: Model score correlation analysis
    - /opt/ml/processing/output/metrics/score_distributions.jpg: 4-panel distribution comparison
    - /opt/ml/processing/output/metrics/new_model_*.jpg: Individual plots for new model
    - /opt/ml/processing/output/metrics/previous_model_*.jpg: Individual plots for previous model
    
    Required Environment Variables:
    - ID_FIELD: Name of the ID column in evaluation data
    - LABEL_FIELD: Name of the label column in evaluation data
    
    Optional Environment Variables (Comparison Mode):
    - COMPARISON_MODE: Enable model comparison functionality (default: "false")
    - PREVIOUS_SCORE_FIELD: Column name containing previous model scores (default: "previous_score")
    - COMPARISON_METRICS: Metrics to compute - "all" or "basic" (default: "all")
    - STATISTICAL_TESTS: Enable statistical significance tests (default: "true")
    - COMPARISON_PLOTS: Enable comparison visualizations (default: "true")
    
    Arguments:
    - job_type: Type of evaluation job to perform (e.g., "evaluation", "validation")
    
    Supported Model Classes:
    - Bimodal models: multimodal_bert, multimodal_cnn, bert, lstm, multimodal_moe, multimodal_gate_fusion, multimodal_cross_attn
    - Trimodal models: trimodal_bert, trimodal_cross_attn_bert, trimodal_gate_fusion_bert
    - Both PyTorch native and ONNX model formats supported
    
    Text Processing Pipeline:
    - Dialogue splitting and HTML normalization
    - Emoji removal and text normalization
    - Dialogue chunking with configurable parameters
    - BERT tokenization with attention masks
    - Support for dual text modalities (primary + secondary)
    
    Tabular Processing Pipeline:
    - Categorical feature encoding using saved mappings
    - Multiclass label processing for non-binary tasks
    - Feature validation and missing value handling
    
    Smart Pipeline Reconstruction:
    - Automatically detects trimodal vs bimodal configuration from saved hyperparameters
    - Reconstructs exact same preprocessing pipelines used during training
    - Chooses appropriate collate function (trimodal vs bimodal) based on model type
    - Supports different tokenizers for primary/secondary text modalities
    
    Comparison Features:
    - Performance delta metrics (AUC-ROC, Average Precision, F1-score improvements)
    - Statistical significance testing (McNemar's test, paired t-test, Wilcoxon test)
    - Correlation analysis between model scores
    - Comprehensive visualizations comparing model performance
    - Automated recommendations for model deployment decisions
    
    Supports both binary and multiclass classification with appropriate metrics for each.
    Binary classification has full comparison functionality; multiclass has limited comparison support.
    
    The script leverages the same inference and metrics computation functions used during
    PyTorch training to ensure consistency between training and evaluation results.
    """,
)
