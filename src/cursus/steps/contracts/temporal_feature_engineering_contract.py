"""
Temporal Feature Engineering Script Contract

Defines the contract for the temporal feature engineering script that extracts
comprehensive temporal features from normalized sequence data, combining generic
temporal features with time window aggregations for machine learning models.
"""

from ...core.base.contract_base import ScriptContract

TEMPORAL_FEATURE_ENGINEERING_CONTRACT = ScriptContract(
    entry_point="temporal_feature_engineering.py",
    expected_input_paths={
        "normalized_sequences": "/opt/ml/processing/input/normalized_sequences"
    },
    expected_output_paths={"temporal_feature_tensors": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["SEQUENCE_GROUPING_FIELD", "TIMESTAMP_FIELD", "VALUE_FIELDS"],
    optional_env_vars={
        "FEATURE_TYPES": '["statistical", "temporal", "behavioral"]',
        "CATEGORICAL_FIELDS": '["merchantCategory", "paymentMethod"]',
        "WINDOW_SIZES": "[7, 14, 30, 90]",
        "AGGREGATION_FUNCTIONS": '["mean", "sum", "std", "min", "max", "count"]',
        "LAG_FEATURES": "[1, 7, 14, 30]",
        "EXPONENTIAL_SMOOTHING_ALPHA": "0.3",
        "TIME_UNIT": "days",
        "INPUT_FORMAT": "numpy",
        "OUTPUT_FORMAT": "numpy",
        "ENABLE_DISTRIBUTED_PROCESSING": "false",
        "CHUNK_SIZE": "5000",
        "MAX_WORKERS": "auto",
        "FEATURE_PARALLELISM": "true",
        "CACHE_INTERMEDIATE": "true",
        "ENABLE_VALIDATION": "true",
        "MISSING_VALUE_THRESHOLD": "0.95",
        "CORRELATION_THRESHOLD": "0.99",
        "VARIANCE_THRESHOLD": "0.01",
        "OUTLIER_DETECTION": "true",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "scipy": ">=1.7.0",
    },
    description="""
    Temporal feature engineering script that:
    1. Loads normalized temporal sequences from TemporalSequenceNormalization output
    2. Extracts generic temporal features (statistical, temporal patterns, behavioral)
    3. Computes time window aggregations (rolling, lag, exponential smoothing)
    4. Validates feature quality with comprehensive quality control
    5. Outputs temporal feature tensors for machine learning model consumption
    
    Contract aligned with actual script implementation:
    - Inputs: 
      * normalized_sequences (required) - reads normalized sequence data from /opt/ml/processing/input/normalized_sequences
    - Outputs: temporal_feature_tensors (primary) - writes to /opt/ml/processing/output
    - Arguments: job-type (required) - defines processing mode for different data splits
    
    Script Implementation Details:
    - Supports multi-format input loading (numpy, parquet, csv) matching TemporalSequenceNormalization output
    - Automatic attention mask utilization for proper sequence processing
    - Two core feature engineering operations:
      * GenericTemporalFeaturesOperation - statistical, temporal, and behavioral features
      * TimeWindowAggregationsOperation - rolling windows, lag features, exponential smoothing
    - Comprehensive feature quality control with validation and recommendations
    - Configurable output formats (numpy, parquet, csv) with feature metadata
    - Memory-efficient processing with optional distributed computation
    
    Environment Variable Details:
    - SEQUENCE_GROUPING_FIELD: Field name used to group entities for feature computation (default: "customerId")
    - TIMESTAMP_FIELD: Field name containing temporal information (default: "orderDate")
    - VALUE_FIELDS: JSON array of numerical fields for feature extraction (default: ["transactionAmount", "merchantRiskScore"])
    - CATEGORICAL_FIELDS: JSON array of categorical fields for feature extraction (default: ["merchantCategory", "paymentMethod"])
    - FEATURE_TYPES: JSON array of feature types to extract (default: ["statistical", "temporal", "behavioral"])
    - WINDOW_SIZES: JSON array of time window sizes for aggregations (default: [7, 14, 30, 90])
    - AGGREGATION_FUNCTIONS: JSON array of aggregation functions to apply (default: ["mean", "sum", "std", "min", "max", "count"])
    - LAG_FEATURES: JSON array of lag periods for historical features (default: [1, 7, 14, 30])
    - EXPONENTIAL_SMOOTHING_ALPHA: Alpha parameter for exponential smoothing (default: "0.3")
    - TIME_UNIT: Time unit for window calculations - "days" or "hours" (default: "days")
    - INPUT_FORMAT: Input data format - "numpy", "parquet", or "csv" (default: "numpy")
    - OUTPUT_FORMAT: Output format - "numpy", "parquet", or "csv" (default: "numpy")
    
    Distributed Processing Configuration:
    - ENABLE_DISTRIBUTED_PROCESSING: Enable chunked processing for large datasets (default: "false")
    - CHUNK_SIZE: Chunk size for distributed processing (default: "5000")
    - MAX_WORKERS: Number of parallel workers (default: "auto")
    - FEATURE_PARALLELISM: Enable parallel feature type computation (default: "true")
    - CACHE_INTERMEDIATE: Cache intermediate results for reuse (default: "true")
    
    Quality Control Configuration:
    - ENABLE_VALIDATION: Enable comprehensive feature quality validation (default: "true")
    - MISSING_VALUE_THRESHOLD: Threshold for flagging high missing value features (default: "0.95")
    - CORRELATION_THRESHOLD: Threshold for flagging highly correlated features (default: "0.99")
    - VARIANCE_THRESHOLD: Threshold for flagging low variance features (default: "0.01")
    - OUTLIER_DETECTION: Enable outlier detection in feature distributions (default: "true")
    
    Output Structure:
    - Temporal feature tensors saved in specified format
    - Feature metadata JSON file with feature names and descriptions
    - Quality report JSON file with validation results and recommendations
    - Support for both combined and separate feature matrices
    
    Integration with TemporalSequenceNormalization:
    - Perfect input compatibility with TemporalSequenceNormalization output
    - Supports all output formats from TemporalSequenceNormalization (numpy, parquet, csv)
    - Utilizes attention masks for proper sequence processing
    - Leverages pre-computed time deltas for temporal feature extraction
    - Maintains entity grouping structure for per-entity feature computation
    
    TSA Integration Support:
    - Optimized for fraud detection temporal feature engineering
    - Multi-scale time window analysis for fraud pattern detection
    - Behavioral feature extraction for anomaly detection
    - Configurable feature types for different fraud detection scenarios
    - Quality control optimized for production fraud detection models
    
    Feature Engineering Capabilities:
    - Statistical Features: count, sum, mean, std, min, max, percentiles, skew, kurtosis, CV
    - Temporal Patterns: time delta analysis, frequency, regularity, temporal span, event patterns
    - Behavioral Features: activity concentration (Gini), consistency, trend analysis, volatility
    - Window Aggregations: rolling statistics, lag features, exponential smoothing
    - Categorical Features: unique counts, diversity, mode frequency
    - Quality Metrics: missing value analysis, correlation analysis, variance analysis, outlier detection
    """,
)
