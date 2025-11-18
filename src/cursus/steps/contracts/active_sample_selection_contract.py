"""
Active Sample Selection Script Contract

Defines the contract for the Active Sample Selection script that intelligently
selects high-value samples from model predictions for Semi-Supervised Learning (SSL)
or Active Learning workflows.
"""

from ...core.base.contract_base import ScriptContract

ACTIVE_SAMPLE_SELECTION_CONTRACT = ScriptContract(
    entry_point="active_sample_selection.py",
    expected_input_paths={
        "evaluation_data": "/opt/ml/processing/input/evaluation_data",
    },
    expected_output_paths={
        "selected_samples": "/opt/ml/processing/output/selected_samples",
        "selection_metadata": "/opt/ml/processing/output/selection_metadata",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        "SELECTION_STRATEGY",
        "USE_CASE",
        "ID_FIELD",
        "LABEL_FIELD",
    ],
    optional_env_vars={
        "SCORE_FIELD": "",
        "OUTPUT_FORMAT": "csv",
        "CONFIDENCE_THRESHOLD": "0.9",
        "MAX_SAMPLES": "0",
        "K_PER_CLASS": "100",
        "UNCERTAINTY_MODE": "margin",
        "BATCH_SIZE": "32",
        "METRIC": "euclidean",
        "RANDOM_SEED": "42",
        "SCORE_FIELD_PREFIX": "prob_class_",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Active sample selection script that intelligently selects high-value samples
    from model predictions for Semi-Supervised Learning or Active Learning workflows.
    
    Two Primary Use Cases:
    
    1. Semi-Supervised Learning (SSL):
       - Goal: Select high-confidence predictions for automatic pseudo-labeling
       - Strategies: confidence_threshold, top_k_per_class
       - Output: High-confidence samples ready for model fine-tuning
    
    2. Active Learning (AL):
       - Goal: Select most informative samples for human labeling
       - Strategies: uncertainty, diversity, badge
       - Output: Uncertain/diverse samples maximizing learning value
    
    Selection Strategies:
    
    SSL Strategies (Confidence-Based):
    - confidence_threshold: Selects samples where max(prob) >= threshold
      * Use when: Simple high-confidence filtering needed
      * Parameters: CONFIDENCE_THRESHOLD (0.5-1.0), MAX_SAMPLES (optional limit)
    
    - top_k_per_class: Selects top-k most confident samples per predicted class
      * Use when: Need balanced pseudo-labeling across classes
      * Parameters: K_PER_CLASS (samples per class)
    
    Active Learning Strategies (Uncertainty/Diversity-Based):
    - uncertainty: Selects samples where model is most uncertain
      * Modes: margin (difference between top-2), entropy (Shannon), least_confidence
      * Use when: Want samples in decision boundary regions
      * Parameters: UNCERTAINTY_MODE, BATCH_SIZE
    
    - diversity: Selects diverse samples using k-center algorithm
      * Use when: Want representative coverage of feature space
      * Parameters: BATCH_SIZE, METRIC (euclidean/cosine)
      * Requires: Embeddings (emb_*) or feature columns
    
    - badge: Combines uncertainty + diversity via gradient embeddings
      * Use when: Want both informative AND diverse samples
      * Parameters: BATCH_SIZE, METRIC
      * Requires: Embeddings/features + probabilities
    
    Input Structure:
    - /opt/ml/processing/input/evaluation_data: Model predictions directory containing:
      - predictions.csv/parquet: Predictions with probability columns
        * Required columns: ID field (configurable), prob_class_0, prob_class_1, ...
        * Optional columns: Label (for validation), features (preserved in output)
        * Alternative score formats: confidence_score, prediction_score, rule_score
        * For diversity/BADGE: emb_0, emb_1, ... (embeddings) or feature columns
    
    Supported Input Sources:
    - XGBoost/LightGBM/PyTorch model inference outputs (prob_class_* format)
    - Bedrock/LLM inference outputs (confidence_score format)
    - Label ruleset execution outputs (rule_score format)
    - Model evaluation outputs (includes predictions with probabilities)
    
    Output Structure:
    - /opt/ml/processing/output/selected_samples: Selected samples directory containing:
      - selected_samples.csv/parquet: Selected samples with metadata
        * Columns: All original columns + selection_score + selection_rank
        * selection_score: Strategy-specific confidence/uncertainty/diversity score
        * selection_rank: Rank within selected batch (1 = highest priority)
      - _SUCCESS: Marker file indicating successful completion
    
    - /opt/ml/processing/output/selection_metadata: Metadata directory containing:
      - selection_metadata.json: Selection metadata including:
        * strategy: Strategy name used
        * use_case: Use case (ssl/active_learning/auto)
        * batch_size: Number of samples selected
        * total_pool_size: Total samples in input pool
        * selected_count: Actual number selected
        * strategy_config: Strategy-specific parameters
        * timestamp: Selection timestamp
        * job_type: Job type identifier
    
    Environment Variables:
    
    Core Parameters (All Use Cases):
    - SELECTION_STRATEGY (required): Strategy name
      * SSL: "confidence_threshold", "top_k_per_class"
      * Active Learning: "uncertainty", "diversity", "badge"
    
    - USE_CASE (optional): Use case validation mode (default: "auto")
      * "ssl": Validates only SSL strategies allowed
      * "active_learning": Validates only AL strategies allowed
      * "auto": No validation (advanced users)
    
    - ID_FIELD (optional): ID column name (default: "id")
    - LABEL_FIELD (optional): Label column name (default: "", unused)
    - OUTPUT_FORMAT (optional): Output format "csv" or "parquet" (default: "csv")
    - RANDOM_SEED (optional): Random seed for reproducibility (default: "42")
    
    Score Field Configuration:
    - SCORE_FIELD (optional): Single score column name for binary/custom scoring
      * If provided, uses this column exclusively
      * Example: "prob_class_1" for binary, "confidence_score" for LLM
    
    - SCORE_FIELD_PREFIX (optional): Prefix for multiple score columns (default: "prob_class_")
      * Used when SCORE_FIELD not specified
      * Finds all columns starting with prefix
      * Example: "prob_class_" matches prob_class_0, prob_class_1, ...
    
    SSL-Specific Parameters:
    - CONFIDENCE_THRESHOLD (optional): Minimum confidence (default: "0.9", range: 0.5-1.0)
    - MAX_SAMPLES (optional): Maximum samples to select (default: "0" = no limit)
    - K_PER_CLASS (optional): Samples per class for top_k_per_class (default: "100")
    
    Active Learning-Specific Parameters:
    - UNCERTAINTY_MODE (optional): Uncertainty mode (default: "margin")
      * Options: "margin", "entropy", "least_confidence"
    - BATCH_SIZE (optional): Number of samples to select (default: "32")
    - METRIC (optional): Distance metric for diversity/BADGE (default: "euclidean")
      * Options: "euclidean", "cosine"
    
    Arguments:
    - job_type: Type of selection job (e.g., "ssl_selection", "active_learning_selection")
    
    Use Case Validation:
    
    When USE_CASE is set to "ssl" or "active_learning", the script validates that
    the selected strategy is appropriate:
    
    - SSL Validation:
      * Allows: confidence_threshold, top_k_per_class
      * Blocks: uncertainty, diversity, badge
      * Rationale: Uncertainty strategies create noisy pseudo-labels
    
    - Active Learning Validation:
      * Allows: uncertainty, diversity, badge
      * Blocks: confidence_threshold, top_k_per_class
      * Rationale: Confidence strategies waste human labeling effort
    
    - Auto Mode:
      * No validation performed
      * User takes responsibility for strategy choice
    
    Strategy Selection Guidelines:
    
    For SSL:
    - Use confidence_threshold when: Simple filtering, single threshold works well
    - Use top_k_per_class when: Need balanced pseudo-labels across classes
    
    For Active Learning:
    - Use uncertainty when: Focus on decision boundary, fast iterations needed
    - Use diversity when: Want representative coverage, have quality embeddings
    - Use badge when: Want best of both, larger batch sizes, computational budget allows
    
    Downstream Integration:
    
    SSL Workflow:
    - ActiveSampleSelection → PseudoLabelMerge → XGBoostTraining (fine-tune)
    - Output provides high-confidence samples ready for training augmentation
    
    Active Learning Workflow:
    - ActiveSampleSelection → Human Labeling Interface → Labeled Data
    - Output provides informative samples for efficient human annotation
    
    Performance Considerations:
    
    - Confidence/Top-K: O(n) complexity, fastest, suitable for large datasets
    - Uncertainty: O(n) complexity, fast, suitable for large datasets
    - Diversity: O(n²) complexity, slower, use for smaller batches (<10K)
    - BADGE: O(n²) complexity, slower, use for smaller batches (<10K)
    
    The output preserves all original columns and adds selection metadata for
    tracking provenance, confidence scores, and selection priorities.
    """,
)
