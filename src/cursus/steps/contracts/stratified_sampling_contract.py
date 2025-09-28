"""
Stratified Sampling Script Contract

Defines the contract for the stratified sampling script that applies stratified sampling
with different allocation strategies for handling class imbalance, causal analysis, and variance optimization.
"""

from ...core.base.contract_base import ScriptContract

STRATIFIED_SAMPLING_CONTRACT = ScriptContract(
    entry_point="stratified_sampling.py",
    expected_input_paths={"processed_data": "/opt/ml/processing/input/data"},
    expected_output_paths={"processed_data": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["STRATA_COLUMN"],
    optional_env_vars={
        "SAMPLING_STRATEGY": "balanced",
        "TARGET_SAMPLE_SIZE": "1000",
        "MIN_SAMPLES_PER_STRATUM": "10",
        "VARIANCE_COLUMN": "",
        "RANDOM_STATE": "42",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Stratified sampling script that:
    1. Reads processed data from tabular_preprocessing output structure
    2. Applies stratified sampling with configurable allocation strategies
    3. Maintains folder structure compatibility for seamless pipeline integration
    4. Handles different job types (training vs non-training)
    
    Contract aligned with actual script implementation:
    - Inputs: DATA (required) - reads from /opt/ml/processing/input/data
    - Outputs: processed_data (primary) - writes to /opt/ml/processing/output
    - Arguments: job_type (required) - defines processing mode (training/validation/testing/calibration)
    
    Script Implementation Details:
    - Reads CSV files from split subdirectories (train/, val/, test/)
    - Supports three allocation strategies:
      * balanced: Equal samples per stratum (class imbalance)
      * proportional_min: Proportional with minimum constraints (causal analysis)
      * optimal: Neyman allocation for variance optimization
    - For training job_type: samples train/val splits, copies test unchanged
    - For non-training job_types: samples only the specified split
    - Outputs sampled files maintaining same folder structure as input
    - Preserves test set integrity for training workflows
    
    Environment Variables:
    - STRATA_COLUMN (required): Column name to stratify by
    - SAMPLING_STRATEGY (optional): One of 'balanced', 'proportional_min', 'optimal'
    - TARGET_SAMPLE_SIZE (optional): Total desired sample size per split
    - MIN_SAMPLES_PER_STRATUM (optional): Minimum samples per stratum for statistical power
    - VARIANCE_COLUMN (optional): Column for variance calculation (needed for optimal strategy)
    - RANDOM_STATE (optional): Random seed for reproducibility
    
    Integration Points:
    - Input compatible with: tabular_preprocessing output
    - Output compatible with: xgboost_training input, other downstream processing steps
    - Maintains SageMaker processing path contracts
    """,
)
