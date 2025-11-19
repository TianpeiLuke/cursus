"""
Feature Selection Script Contract

Defines the contract for the feature selection script that applies multiple statistical and
machine learning-based feature selection methods for dimensionality reduction and model
performance optimization.
"""

from ...core.base.contract_base import ScriptContract

FEATURE_SELECTION_CONTRACT = ScriptContract(
    entry_point="feature_selection.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input",
        "model_artifacts_input": "/opt/ml/processing/input/model_artifacts",  # Optional for non-training modes
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data",
        "model_artifacts_output": "/opt/ml/processing/output/model_artifacts",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["LABEL_FIELD"],
    optional_env_vars={
        "FEATURE_SELECTION_METHODS": "variance,correlation,mutual_info,rfe",
        "N_FEATURES_TO_SELECT": "10",
        "CORRELATION_THRESHOLD": "0.95",
        "VARIANCE_THRESHOLD": "0.01",
        "RANDOM_STATE": "42",
        "COMBINATION_STRATEGY": "voting",
        "USE_SECURE_PYPI": "false",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Feature selection script that:
    1. Reads processed data from tabular_preprocessing output structure
    2. Applies multiple statistical and ML-based feature selection methods
    3. Combines results using ensemble selection strategies
    4. Maintains folder structure compatibility for seamless pipeline integration
    5. Handles different job types (training vs non-training)
    
    Contract aligned with actual script implementation:
    - Inputs: 
      * input_data (required) - reads from /opt/ml/processing/input
      * model_artifacts_input (optional) - pre-computed artifacts from training job for non-training modes
    - Outputs: 
      * processed_data (primary) - feature-selected train/val/test splits to /opt/ml/processing/output/data
      * model_artifacts_output (metadata) - selected_features.json + feature_scores.csv + feature_selection_report.json to /opt/ml/processing/output/model_artifacts
    - Arguments: job_type (required) - defines processing mode (training/validation/testing)
    
    Job Type Behavior:
    - Training Mode: Runs full feature selection pipeline, fits methods on training data, transforms all splits, 
      saves artifacts using parameter accumulator pattern
    - Non-Training Modes (validation/testing/calibration): Uses pre-computed selected features from training job, 
      copies all existing artifacts from previous steps, skips computation-intensive selection process, 
      applies feature filtering to single split
    
    Script Implementation Details:
    - Reads CSV files from split subdirectories (train/, val/, test/)
    - Supports 9 feature selection methods:
      * Statistical: variance threshold, correlation-based, mutual information, chi-square, F-test
      * ML-based: RFE (RF/SVM/Linear), feature importance (RF/XGBoost/Extra Trees), LASSO, permutation importance
    - Combines methods using 3 ensemble strategies:
      * voting: Count how many methods selected each feature
      * ranking: Average rankings across methods
      * scoring: Normalize and combine scores
    - For training job_type: processes all train/val/test splits, saves selection artifacts
    - For non-training job_types: loads pre-computed selected features, processes only the specified split
    - Outputs feature-selected files maintaining same folder structure as input
    - Generates comprehensive selection metadata and performance reports
    
    Environment Variables:
    - LABEL_FIELD (required): Target column name (standard across framework)
    - FEATURE_SELECTION_METHODS (optional): Comma-separated list of methods to apply
    - N_FEATURES_TO_SELECT (optional): Number of features to select in final ensemble
    - CORRELATION_THRESHOLD (optional): Threshold for removing highly correlated features
    - VARIANCE_THRESHOLD (optional): Threshold for removing low-variance features
    - RANDOM_STATE (optional): Random seed for reproducibility
    - COMBINATION_STRATEGY (optional): Strategy for combining method results
    
    Feature Selection Methods Available:
    - variance: Remove features with low variance
    - correlation: Remove highly correlated features (keep those with higher target correlation)
    - mutual_info: Select features based on mutual information with target
    - chi2: Select features using chi-square test (for non-negative features)
    - f_test: Select features using ANOVA F-test
    - rfe: Recursive Feature Elimination with various estimators
    - importance: Select features based on model feature importance
    - lasso: Select features using LASSO regularization
    - permutation: Select features using permutation importance
    
    Output Artifacts:
    - /opt/ml/processing/output/data/{split}/{split}_processed_data.csv: Feature-selected train/val/test splits
    - /opt/ml/processing/output/model_artifacts/selected_features.json: Metadata about selected features
    - /opt/ml/processing/output/model_artifacts/feature_scores.csv: Detailed scores from all methods
    - /opt/ml/processing/output/model_artifacts/feature_selection_report.json: Performance summary
    - /opt/ml/processing/output/model_artifacts/impute_dict.pkl: Copied from previous step (accumulator pattern)
    - /opt/ml/processing/output/model_artifacts/risk_table_map.pkl: Copied from previous step (accumulator pattern)
    
    Parameter Accumulator Pattern:
    - Copies all existing artifacts from model_artifacts_input to model_artifacts_output
    - Adds its own feature selection artifacts to the accumulated artifacts
    - Enables downstream steps (e.g., xgboost_training) to access all preprocessing parameters
    
    Integration Points:
    - Input compatible with: tabular_preprocessing output, stratified_sampling output
    - Output compatible with: xgboost_training input, other downstream processing steps
    - Maintains SageMaker processing path contracts
    - Follows script testability refactoring patterns for easy unit testing
    
    Performance Characteristics:
    - Handles large datasets efficiently with memory-conscious algorithms
    - Parallel processing support for multiple methods
    - Robust error handling with graceful method failure recovery
    - Comprehensive logging for debugging and monitoring
    - Processing time tracking for performance optimization
    """,
)
