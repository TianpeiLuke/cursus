from pydantic import Field, model_validator
from typing import List, Optional, Dict, Any
from .hyperparameters_base import ModelHyperparameters


class Transformer2RiskHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for Transformer2Risk bimodal fraud detection model.

    This class extends the base ModelHyperparameters with Transformer-specific
    architecture parameters needed for the Transformer2Risk model which combines:
    - Transformer encoder with self-attention for text sequence encoding
    - MLP for tabular feature encoding
    - Bimodal fusion for fraud prediction

    Key architectural differences from LSTM2Risk:
    - Uses self-attention mechanism instead of recurrent connections
    - Larger embedding dimensions (128 vs 16) for richer representations
    - Fixed-length sequences with positional embeddings (vs variable-length LSTM)
    - Multi-head attention for parallel attention to different aspects

    Inherits all base fields including:
    - Data field management (full_field_list, cat_field_list, tab_field_list)
    - Training parameters (lr, batch_size, max_epochs, optimizer)
    - Classification parameters (multiclass_categories, class_weights)
    - Derived properties (input_tab_dim, num_classes, is_binary)

    Example Usage:
    ```python
    hyperparam = Transformer2RiskHyperparameters(
        # Essential fields (Tier 1) - required
        full_field_list=["name", "email", "age", "income", "label"],
        cat_field_list=["name", "email"],
        tab_field_list=["age", "income"],
        id_name="customer_id",
        label_name="label",
        multiclass_categories=[0, 1],

        # Transformer-specific fields (Tier 2) - optional, using defaults
        embedding_size=128,
        hidden_size=256,
        n_embed=4000,
        n_blocks=8,
        n_heads=8,
        block_size=100,
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
    # Override model_class from base to identify this as Transformer2Risk

    model_class: str = Field(
        default="transformer2risk",
        description="Model class identifier for this hyperparameter configuration",
    )

    # For tokenizer settings AND model architecture (position embeddings)
    max_sen_len: int = Field(
        default=100,
        description="Maximum sequence length for both tokenizer truncation and model position embeddings. "
        "Controls both data preprocessing (tokenizer) and model architecture (position embedding table size).",
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
        "For Transformer2Risk, text is concatenated risk scores (e.g., '0.5|0.3|0.8|0.2') "
        "that only need tokenization with custom BPE tokenizer - no dialogue/HTML/emoji cleaning required. "
        "Empty list allows pytorch_training.py to determine appropriate default based on model type.",
    )

    # ===== Transformer-Specific Architecture Parameters (Tier 2) =====
    # These parameters define the Transformer2Risk model architecture

    embedding_size: int = Field(
        default=128,
        gt=0,
        le=512,
        description="Token and position embedding dimension. "
        "Significantly larger than LSTM (128 vs 16) since transformers "
        "benefit from higher-dimensional embeddings for effective self-attention. "
        "Must be divisible by n_heads.",
    )

    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Dropout probability for regularization throughout the model. "
        "Applied in attention layers, feedforward networks, tabular projection, "
        "and classifier. Higher values provide more regularization but may underfit.",
    )

    hidden_size: int = Field(
        default=256,
        gt=0,
        le=1024,
        description="Hidden dimension for tabular feature projection. "
        "Text encoder projects embedding_size to 2*hidden_size. "
        "Combined bimodal representation is 4*hidden_size. "
        "Larger than LSTM (256 vs 128) to match increased model capacity.",
    )

    n_embed: int = Field(
        default=4000,
        gt=0,
        le=100000,
        description="Vocabulary size for token embeddings. "
        "Must match the tokenizer vocabulary size. "
        "Typically determined by BPE tokenizer training.",
    )

    n_blocks: int = Field(
        default=8,
        gt=0,
        le=24,
        description="Number of stacked transformer encoder blocks. "
        "Each block contains multi-head self-attention and feedforward network. "
        "More blocks increase model capacity but also computational cost. "
        "Typical range: 6-12 for medium-sized models.",
    )

    n_heads: int = Field(
        default=8,
        gt=0,
        le=16,
        description="Number of attention heads per transformer block. "
        "Must divide embedding_size evenly (head_size = embedding_size / n_heads). "
        "Multiple heads allow model to attend to different representation subspaces. "
        "Common values: 8, 12, 16 for standard architectures.",
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

    @model_validator(mode="after")
    def validate_transformer_hyperparameters(self) -> "Transformer2RiskHyperparameters":
        """Validate transformer-specific constraints."""
        # Call base validator first
        super().validate_dimensions()

        # Validate embedding_size is divisible by n_heads
        if self.embedding_size % self.n_heads != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible by "
                f"n_heads ({self.n_heads}) for multi-head attention. "
                f"Current head_size would be {self.embedding_size / self.n_heads:.2f}"
            )

        return self
