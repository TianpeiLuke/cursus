"""
PyTorch Training Script Contract

Defines the contract for the PyTorch BSM training script that handles multimodal
text and tabular data training using Lightning framework.
"""

from .training_script_contract import TrainingScriptContract

PYTORCH_TRAIN_CONTRACT = TrainingScriptContract(
    entry_point="pytorch_training.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams/hyperparameters.json",
        "model_artifacts_input": "/opt/ml/input/data/model_artifacts_input",  # Optional: Pre-computed preprocessing artifacts
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "evaluation_output": "/opt/ml/output/data",
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables - script uses hyperparameters.json
    ],
    optional_env_vars={
        "SM_CHECKPOINT_DIR": "/opt/ml/checkpoints",
        "USE_SECURE_PYPI": "true",  # Controls PyPI source for package installation (default: secure CodeArtifact)
        "USE_PRECOMPUTED_IMPUTATION": "false",  # If true, uses pre-computed imputation artifacts and skips inline computation
        "USE_PRECOMPUTED_RISK_TABLES": "false",  # If true, uses pre-computed risk table artifacts and skips inline computation
        "USE_PRECOMPUTED_FEATURES": "false",  # If true, uses pre-computed feature selection and skips inline computation
        "REGION": "NA",  # Region identifier (NA/EU/FE) for loading region-specific hyperparameters
    },
    framework_requirements={
        "torch": "==2.1.2",
        "torchvision": "==0.16.2",
        "torchaudio": "==2.1.2",
        "transformers": "==4.37.2",
        "lightning": "==2.1.3",
        "lightning-utilities": "==0.10.1",
        "torchmetrics": "==1.7.1",
        "tensorboard": "==2.16.2",
        "matplotlib": "==3.8.2",
        "scikit-learn": "==1.3.2",
        "pandas": "==2.1.4",
        "pyarrow": "==14.0.2",
        "beautifulsoup4": "==4.12.3",
        "gensim": "==4.3.1",
        "pydantic": "==2.11.2",
        "onnx": "==1.15.0",
        "onnxruntime": "==1.17.0",
        "flask": "==3.0.2",
    },
    description="""
    PyTorch Lightning training script for multimodal BSM (Business Seller Messaging) models that:
    1. Loads and preprocesses multimodal data (text + tabular features)
    2. Supports multiple model architectures (BERT, CNN, LSTM, multimodal variants)
    3. Handles both binary and multiclass classification
    4. Applies text preprocessing pipeline with tokenization and chunking
    5. Performs categorical feature encoding and label processing
    6. Trains using PyTorch Lightning with early stopping and checkpointing
    7. Evaluates on validation and test sets with comprehensive metrics
    8. Exports trained model in multiple formats (PyTorch, ONNX)
    
    Input Structure:
    - /opt/ml/input/data: Root directory containing train/val/test subdirectories
      - /opt/ml/input/data/train: Training data files (.csv, .tsv, .parquet)
      - /opt/ml/input/data/val: Validation data files
      - /opt/ml/input/data/test: Test data files
    - /opt/ml/input/data/model_artifacts_input: Optional directory with pre-computed artifacts
      - /opt/ml/input/data/model_artifacts_input/impute_dict.pkl: Pre-computed imputation parameters
      - /opt/ml/input/data/model_artifacts_input/risk_table_map.pkl: Pre-computed risk tables
      - /opt/ml/input/data/model_artifacts_input/selected_features.json: Pre-computed feature selection
    - /opt/ml/code/hyperparams/hyperparameters.json: Model configuration and hyperparameters
    
    Output Structure:
    - /opt/ml/model: Model artifacts directory
      - /opt/ml/model/model.pth: Trained PyTorch model
      - /opt/ml/model/model_artifacts.pth: Model artifacts (config, embeddings, vocab)
      - /opt/ml/model/model.onnx: ONNX exported model
    - /opt/ml/output/data: Evaluation results directory
      - /opt/ml/output/data/predict_results.pth: Prediction results
      - /opt/ml/output/data/tensorboard_eval/: TensorBoard evaluation logs
    - /opt/ml/checkpoints/: Training checkpoints
    
    Contract aligned with step specification:
    - Inputs: input_path (required), hyperparameters_s3_uri (optional)
    - Outputs: model_output (primary), evaluation_output (secondary)
    
    Pre-Computed Artifact Support:
    - USE_PRECOMPUTED_IMPUTATION=true: Input data already imputed, loads impute_dict.pkl, skips transformation
    - USE_PRECOMPUTED_RISK_TABLES=true: Input data already risk-mapped, loads risk_table_map.pkl, skips transformation
    - USE_PRECOMPUTED_FEATURES=true: Input data already feature-selected, loads selected_features.json, skips selection
    - Default (all false): Computes all preprocessing inline and transforms data
    
    Environment Variables:
    - SM_CHECKPOINT_DIR: SageMaker checkpoint directory (optional)
    - USE_SECURE_PYPI: Controls PyPI source for package installation (default: true for secure CodeArtifact)
    - USE_PRECOMPUTED_IMPUTATION: Use pre-computed imputation artifacts (default: false)
    - USE_PRECOMPUTED_RISK_TABLES: Use pre-computed risk table artifacts (default: false)
    - USE_PRECOMPUTED_FEATURES: Use pre-computed feature selection (default: false)
    
    Hyperparameters (via JSON config):
    - Model architecture: model_class (multimodal_bert, multimodal_cnn, bert, lstm, etc.)
    - Data fields: id_name, text_name, label_name, tab_field_list, cat_field_list
    - Training: batch_size, max_epochs, lr, optimizer, early_stop_patience
    - Text processing: tokenizer, max_sen_len, max_total_chunks
    - Model: num_classes, is_binary, class_weights, hidden_common_dim
    - Advanced: fp16, gradient_clip_val, warmup_steps, reinit_layers
    
    Supported Model Classes:
    - multimodal_bert: BERT + tabular fusion
    - multimodal_cnn: CNN + tabular fusion  
    - multimodal_moe: Mixture of Experts multimodal
    - multimodal_gate_fusion: Gated fusion multimodal
    - multimodal_cross_attn: Cross-attention multimodal
    - bert: Text-only BERT classification
    - lstm: Text-only LSTM classification
    """,
)
