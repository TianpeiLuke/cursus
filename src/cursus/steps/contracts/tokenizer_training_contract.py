"""
Tokenizer Training Script Contract

Defines the contract for the tokenizer training script that trains a custom BPE tokenizer
optimized for customer name data with automatic vocabulary size tuning.
"""

from ...core.base.contract_base import ScriptContract

TOKENIZER_TRAINING_CONTRACT = ScriptContract(
    entry_point="tokenizer_training.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/train",
    },
    expected_output_paths={"model_artifacts_output": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["TEXT_FIELD"],
    optional_env_vars={
        "TARGET_COMPRESSION": "2.5",
        "MIN_FREQUENCY": "25",
        "MAX_VOCAB_SIZE": "50000",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "tokenizers": ">=0.13.0",
    },
    description="""
    Tokenizer training script that trains a Byte Pair Encoding (BPE) tokenizer optimized
    for customer name data with automatic vocabulary size tuning to achieve target compression ratio.
    
    The script uses CompressionBPETokenizer from cursus.processing.tokenizers module, which matches
    the legacy OrderTextTokenizer implementation with improved compression tuning capabilities.
    
    Contract aligned with actual script implementation:
    - Inputs:
      * input_data (required) - reads training texts from /opt/ml/processing/input/train
        Supports parquet format with configurable text field name
    - Outputs: 
      * model_artifacts_output (primary) - writes tokenizer artifacts to /opt/ml/processing/output
        Saves tokenizer.json, vocab.json, and tokenizer_metadata.json
    - Arguments: job_type (required) - defines processing mode (training/validation/testing/calibration)
    
    Script Implementation Details:
    - Loads training texts from parquet files in input directory
    - Trains BPE tokenizer using CompressionBPETokenizer with compression tuning
    - Automatically adjusts vocabulary size to achieve target compression ratio
    - Saves three output artifacts:
      * tokenizer.json - main HuggingFace tokenizer file
      * vocab.json - vocabulary mapping for legacy compatibility
      * tokenizer_metadata.json - metadata including vocab_size, special_tokens, etc.
    
    Environment Variables:
    - TEXT_FIELD (required): Column name containing text data in input parquet
    - TARGET_COMPRESSION (optional, default: "2.5"): Target compression ratio for tokenizer
    - MIN_FREQUENCY (optional, default: "25"): Minimum frequency threshold for BPE merges
    - MAX_VOCAB_SIZE (optional, default: "50000"): Maximum vocabulary size limit
    
    Tokenizer Configuration:
    - Model type: BPE (Byte Pair Encoding)
    - Special tokens: [CLS], [PAD], [UNK], [BOS], [EOS], [MISSING], |
    - Normalizer: NFKC Unicode normalization
    - Pre-tokenizer: Whitespace-based splitting
    - Compression tuning: Automatically adjusts vocab_size to meet target_compression
    
    Output Artifacts:
    - tokenizer.json: HuggingFace Tokenizers format, ready for inference
    - vocab.json: Token-to-ID mapping dictionary
    - tokenizer_metadata.json: Configuration metadata including vocab_size, special_tokens, etc.
    
    References:
    - Legacy implementation: projects/names3risk_legacy/tokenizer.py
    - HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
    - Processing module: src/cursus/processing/tokenizers.py
    """,
)
