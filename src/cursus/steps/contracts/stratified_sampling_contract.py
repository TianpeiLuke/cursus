"""
Stratified Sampling Script Contract

Defines the contract for the stratified sampling script that applies stratified sampling
with different allocation strategies for handling class imbalance, causal analysis, and variance optimization.
"""

from ...core.base.contract_base import ScriptContract

STRATIFIED_SAMPLING_CONTRACT = ScriptContract(
    entry_point="stratified_sampling.py",
    expected_input_paths={"input_data": "/opt/ml/processing/input/data"},
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
        "SAMPLING_MULTIPLIER": "1.0",
        "ALLOW_REPLACEMENT": "false",
        "REFERENCE_COUNTS_JSON": "",
        "SAMPLING_FILTER_COLUMN": "",
        "SAMPLING_FILTER_VALUE": "",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
    },
    description="""
    Stratified sampling script with four allocation strategies and production robustness.

    Strategies:
    - balanced: Equal samples per stratum (class imbalance correction)
    - proportional_min: Proportional allocation with floor constraints (causal analysis)
    - optimal: Neyman variance-weighted allocation (minimizes sampling error)
    - external_proportional: Sample to match external reference distribution with multiplier

    Features:
    - Sampling with replacement when target exceeds available per stratum
    - NaN guard: warns and excludes NaN strata values automatically
    - Empty DataFrame guard: returns empty result gracefully
    - Per-split diagnostics JSON output (requested vs achieved per stratum)
    - Format preservation: reads and writes CSV/TSV/Parquet maintaining input format
    - Reference counts from sidecar file (reference_counts.json) or env var fallback

    Job Type Handling:
    - training: Samples train/val splits, copies test unchanged
    - Other (validation/testing/calibration/sampling): Samples only that split

    Input: /opt/ml/processing/input/data/{split}/{split}_processed_data.{csv|tsv|parquet}
           /opt/ml/processing/input/data/reference_counts.json (optional sidecar)
    Output: /opt/ml/processing/output/{split}/{split}_processed_data.{csv|tsv|parquet}
            /opt/ml/processing/output/{split}/sampling_diagnostics.json

    Environment Variables:
    - STRATA_COLUMN (required): Column name to stratify by
    - SAMPLING_STRATEGY: One of 'balanced', 'proportional_min', 'optimal', 'external_proportional'
    - TARGET_SAMPLE_SIZE: Total desired sample size per split (ignored for external_proportional)
    - MIN_SAMPLES_PER_STRATUM: Minimum samples per stratum for statistical power
    - VARIANCE_COLUMN: Column for variance calculation (optimal strategy)
    - RANDOM_STATE: Random seed for reproducibility
    - SAMPLING_MULTIPLIER: Multiplier for external reference counts (e.g., 5.0)
    - ALLOW_REPLACEMENT: Enable over-sampling with replacement ('true'/'false')
    - REFERENCE_COUNTS_JSON: Fallback JSON when sidecar file absent
    """,
)
