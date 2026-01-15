from pydantic import Field
from typing import List, Optional, Dict, Any
from ...core.base.hyperparameters_base import ModelHyperparameters


class LSTM2RiskHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for LSTM2Risk bimodal fraud detection model.

    This class extends the base ModelHyperparameters with LSTM-specific
    architecture parameters needed for the LSTM2Risk model which combines:
    - Bidirectional LSTM for text sequence encoding (names, emails)
    - MLP for tabular feature encoding
    - Bimodal fusion for fraud prediction

    Inherits all base fields including:
    - Data field management (full_field_list, cat_field_list, tab_field_list)
    - Training parameters (lr, batch_size, max_epochs, optimizer)
    - Classification parameters (multiclass_categories, class_weights)
    - Derived properties (input_tab_dim, num_classes, is_binary)

    Example Usage:
    ```python
    hyperparam = LSTM2RiskHyperparameters(
        # Essential fields (Tier 1) - required
        full_field_list=["name", "email", "age", "income", "label"],
        cat_field_list=["name", "email"],
        tab_field_list=["age", "income"],
        id_name="customer_id",
        label_name="label",
        multiclass_categories=[0, 1],

        # LSTM-specific fields (Tier 2) - optional, using defaults
        embedding_size=16,
        hidden_size=128,
        n_embed=4000,
        n_lstm_layers=4,
        dropout_rate=0.2,

        # Can also override base fields
        lr=3e-5,
        batch_size=32,
        max_epochs=5
    )

    # Access derived properties
    print(f"Input tabular dimension: {hyperparam.input_tab_dim}")
    print(f"Number of classes: {hyperparam.num_classes}")
    print(f"Is binary classification: {hyperparam.is_binary}")

    # Serialize for SageMaker
    config = hyperparam.serialize_config()
    ```
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    # For text field specification
    text_name: str = Field(description="Name of the primary text field to be processed")

    # NEW: Track which fields were merged to create text field
    text_source_fields: Optional[List[str]] = Field(
        default=None,
        description="Original field names that were merged to create text_name field. "
        "These fields should be excluded from categorical processing in training "
        "since they no longer exist after preprocessing. Used by pytorch_training to "
        "filter cat_field_list before risk table processing.",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # Override model_class from base to identify this as LSTM2Risk

    model_class: str = Field(
        default="lstm2risk",
        description="Model class identifier for this hyperparameter configuration",
    )

    # For tokenizer settings
    max_sen_len: int = Field(
        default=100,
        description="Maximum sentence length for tokenizer truncation. "
        "This parameter serves dual purpose: "
        "(1) Controls tokenizer truncation during preprocessing "
        "(2) Not directly used in LSTM architecture (unlike Transformer which needs it for position embeddings) "
        "but kept consistent for preprocessing pipeline uniformity. "
        "Default 100 matches legacy behavior and is sufficient for concatenated name fields.",
    )

    fixed_tokenizer_length: bool = Field(
        default=True, description="Use fixed tokenizer length"
    )

    text_input_ids_key: str = Field(
        default="input_ids", description="Key name for input_ids from tokenizer output"
    )

    text_attention_mask_key: str = Field(
        default="attention_mask",
        description="Key name for attention_mask from tokenizer output",
    )

    # Text processing pipeline configuration
    text_processing_steps: List[str] = Field(
        default=[],
        description="Processing steps for text preprocessing pipeline. "
        "For LSTM2Risk, text is concatenated risk scores (e.g., '0.5|0.3|0.8|0.2') "
        "that only need tokenization with custom BPE tokenizer - no dialogue/HTML/emoji cleaning required. "
        "Empty list allows pytorch_training.py to determine appropriate default based on model type.",
    )

    # ===== LSTM-Specific Architecture Parameters (Tier 2) =====
    # These parameters define the LSTM2Risk model architecture

    embedding_size: int = Field(
        default=16,
        gt=0,
        le=512,
        description="Token embedding dimension for text encoding. "
        "Controls the size of learned embeddings for vocabulary tokens. "
        "Larger values capture more semantic information but increase parameters.",
    )

    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Dropout probability for regularization throughout the model. "
        "Applied in LSTM layers, tabular projection, and classifier. "
        "Higher values provide more regularization but may underfit.",
    )

    hidden_size: int = Field(
        default=128,
        gt=0,
        le=1024,
        description="LSTM hidden state dimension. "
        "Bidirectional LSTM outputs 2*hidden_size features. "
        "This dimension is also used for tabular feature projection. "
        "Combined bimodal representation is 4*hidden_size.",
    )

    n_embed: int = Field(
        default=4000,
        gt=0,
        le=100000,
        description="Vocabulary size for token embeddings. "
        "Must match the tokenizer vocabulary size. "
        "Typically determined by BPE tokenizer training.",
    )

    n_lstm_layers: int = Field(
        default=4,
        gt=0,
        le=10,
        description="Number of stacked LSTM layers. "
        "More layers can capture more complex patterns but increase training time. "
        "Dropout is applied between layers when n_lstm_layers > 1.",
    )

    # ===== Training and Optimization Parameters (Tier 2) =====
    # These parameters control the optimization process

    lr_decay: float = Field(default=0.05, description="Learning rate decay")

    momentum: float = Field(
        default=0.9, description="Momentum for SGD optimizer (if SGD is chosen)"
    )

    weight_decay: float = Field(
        default=0.0, description="Weight decay for optimizer (L2 penalty)"
    )

    adam_epsilon: float = Field(default=1e-08, description="Epsilon for Adam optimizer")

    warmup_steps: int = Field(
        default=300,
        gt=0,
        le=1000,
        description="Warmup steps for learning rate scheduler",
    )

    run_scheduler: bool = Field(
        default=True, description="Run learning rate scheduler flag"
    )

    val_check_interval: float = Field(
        default=0.25,
        description="Validation check interval during training (float for fraction of epoch, int for steps)",
    )

    gradient_clip_val: float = Field(
        default=1.0,
        description="Value for gradient clipping to prevent exploding gradients",
    )

    fp16: bool = Field(
        default=False,
        description="Enable 16-bit mixed precision training (requires compatible hardware)",
    )

    use_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage at the cost of ~20% slower training",
    )

    # Early stopping and Checkpointing parameters
    early_stop_metric: str = Field(
        default="val_loss", description="Metric for early stopping"
    )

    early_stop_patience: int = Field(
        default=3, gt=0, le=10, description="Patience for early stopping"
    )

    load_ckpt: bool = Field(default=False, description="Load checkpoint flag")

    # Preprocessing parameters
    smooth_factor: float = Field(
        default=0.0, description="Risk table smoothing factor for categorical encoding"
    )

    count_threshold: int = Field(
        default=0, description="Risk table count threshold for categorical encoding"
    )

    # Text Preprocessing and Tokenization parameters
    text_field_overwrite: bool = Field(
        default=False,
        description="Overwrite text field if it exists (e.g. during feature engineering)",
    )

    # For chunking long texts
    chunk_trancate: bool = Field(
        default=True, description="Chunk truncation flag for long texts"
    )  # Typo 'trancate' kept as per original

    max_total_chunks: int = Field(
        default=3, description="Maximum total chunks for processing long texts"
    )

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include bimodal-specific derived fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add derived fields that should be exposed
        derived_fields = {
            # If you need to expose any derived fields, add them here
        }

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
