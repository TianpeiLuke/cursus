"""
PyTorch Model Inference Script Contract

Defines the contract for the PyTorch model inference script that loads trained models,
processes evaluation data, and generates predictions without metrics computation.
"""

from ...core.base.contract_base import ScriptContract

PYTORCH_MODEL_INFERENCE_CONTRACT = ScriptContract(
    entry_point="pytorch_inference.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "processed_data": "/opt/ml/processing/input/eval_data",
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    optional_env_vars={
        "OUTPUT_FORMAT": "csv",
        "JSON_ORIENT": "records",
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
        "beautifulsoup4": "==4.12.3",
        "gensim": "==4.3.1",
        "pydantic": "==2.11.2",
        "onnx": "==1.15.0",
        "onnxruntime": "==1.17.0",
    },
    description="""
    PyTorch model inference script that:
    1. Loads trained PyTorch model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using text tokenization and tabular processing
    3. Generates predictions using the trained model
    4. Saves predictions with original data in multiple formats (CSV, Parquet, JSON)
    
    This script focuses solely on inference (prediction generation) without metrics
    computation or visualization, enabling modular pipeline architectures where
    inference results can be cached, reused, and processed by different downstream
    components.
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - model.pth: Trained PyTorch model state dict
      - model_artifacts.pth: Model artifacts (config, embeddings, vocab, model_class)
      - hyperparameters.json: Complete model configuration and hyperparameters
      - model.onnx: ONNX exported model (optional)
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV or Parquet files)
    
    Output Structure:
    - /opt/ml/processing/output/eval/predictions.csv: Model predictions with probabilities (default)
    - /opt/ml/processing/output/eval/predictions.parquet: Parquet format output (if OUTPUT_FORMAT=parquet)
    - /opt/ml/processing/output/eval/predictions.json: JSON format output (if OUTPUT_FORMAT=json)
    
    Output Format:
    The output preserves all original data columns and adds prediction columns:
    - Original columns: All columns from input data (ID, label, features, metadata)
    - Prediction columns: prob_class_0, prob_class_1, ... (probability scores for each class)
    
    Environment Variables:
    - ID_FIELD (required): Name of the ID column in evaluation data
    - LABEL_FIELD (required): Name of the label column in evaluation data
    - OUTPUT_FORMAT (optional): Output format - "csv", "parquet", or "json" (default: "csv")
    - JSON_ORIENT (optional): JSON orientation - "records", "index", "values", "split", "table" (default: "records")
    
    Arguments:
    - job_type: Type of inference job to perform (e.g., "inference", "validation")
    
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
    
    Multi-Format Support:
    - CSV: Human-readable, compatible with existing workflows
    - Parquet: Efficient binary format for large datasets with compression
    - JSON: Universal compatibility with configurable orientations:
      - records: [{column -> value}, ..., {column -> value}]
      - index: {index -> {column -> value}}
      - values: [[row values], [row values], ...]
      - split: {'index': [index], 'columns': [columns], 'data': [values]}
      - table: {'schema': {schema}, 'data': [{row}, {row}, ...]}
    
    Downstream Integration:
    The inference output is designed to be consumed by:
    - Model metrics computation scripts (expects ID + label + prob_class_* columns)
    - Model calibration scripts (expects LABEL_FIELD + prob_class_* columns)
    - Model deployment validation (structured predictions for testing)
    - A/B testing frameworks (predictions with metadata preservation)
    
    Supports both binary and multiclass classification with consistent output format.
    
    The script leverages the same inference functions used during PyTorch training
    and evaluation to ensure consistency across the entire model lifecycle.
    """,
)
