from pydantic import Field, model_validator, PrivateAttr, ConfigDict
from typing import List, Dict, Any, Optional, Union

from ...core.base.hyperparameters_base import ModelHyperparameters


class TriModalHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for tri-modal model training with dual text and tabular modalities.
    Extends ModelHyperparameters to support multiple text inputs.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Override model_class for tri-modal
    model_class: str = Field(
        default="trimodal_bert", 
        description="Model class identifier for tri-modal BERT"
    )
    
    # Dual text field specification
    primary_text_name: str = Field(
        description="Name of the primary text field (e.g., chat conversation)"
    )
    
    secondary_text_name: str = Field(
        description="Name of the secondary text field (e.g., shiptrack events)"
    )
    
    # Backward compatibility field for bi-modal models
    text_name: Optional[str] = Field(
        default=None,
        description="Legacy text field name for backward compatibility with bi-modal models"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # BERT/Text specific fields
    tokenizer: str = Field(
        default="bert-base-cased",
        description="Tokenizer name or path (e.g., from Hugging Face)"
    )
    
    max_sen_len: int = Field(
        default=512, 
        description="Maximum sentence length for tokenizer"
    )
    
    fixed_tokenizer_length: bool = Field(
        default=True, 
        description="Use fixed tokenizer length"
    )
    
    hidden_common_dim: int = Field(
        default=256, 
        description="Common hidden dimension for encoders"
    )
    
    reinit_pooler: bool = Field(
        default=True, 
        description="Reinitialize BERT pooler layer"
    )
    
    reinit_layers: int = Field(
        default=2, 
        description="Number of BERT layers to reinitialize"
    )
    
    # Text processing parameters
    chunk_trancate: bool = Field(
        default=True, 
        description="Chunk truncation flag for long texts"
    )
    
    max_total_chunks: int = Field(
        default=3, 
        description="Maximum total chunks for processing long texts"
    )
    
    # Processing pipeline configuration
    primary_text_processing_steps: List[str] = Field(
        default=["dialogue_splitter", "html_normalizer", "emoji_remover", "text_normalizer", "dialogue_chunker", "tokenizer"],
        description="Processing steps for primary text (e.g., chat with HTML/emoji)"
    )
    
    secondary_text_processing_steps: List[str] = Field(
        default=["dialogue_splitter", "text_normalizer", "dialogue_chunker", "tokenizer"],
        description="Processing steps for secondary text (e.g., structured shiptrack events)"
    )

    # Optional separate tokenizers (fallback to main tokenizer)
    primary_tokenizer: Optional[str] = Field(
        default=None, 
        description="Tokenizer for primary text (falls back to main tokenizer if None)"
    )
    
    secondary_tokenizer: Optional[str] = Field(
        default=None, 
        description="Tokenizer for secondary text (falls back to main tokenizer if None)"
    )
    
    # Optional separate hidden dimensions (fallback to main hidden_common_dim)
    primary_hidden_common_dim: Optional[int] = Field(
        default=None, 
        description="Hidden dimension for primary text encoder (falls back to hidden_common_dim if None)"
    )
    
    secondary_hidden_common_dim: Optional[int] = Field(
        default=None, 
        description="Hidden dimension for secondary text encoder (falls back to hidden_common_dim if None)"
    )
    
    # Separate input keys for each text modality
    primary_text_input_ids_key: str = Field(
        default="input_ids", 
        description="Key name for primary text input_ids"
    )
    
    primary_text_attention_mask_key: str = Field(
        default="attention_mask", 
        description="Key name for primary text attention_mask"
    )
    
    secondary_text_input_ids_key: str = Field(
        default="input_ids", 
        description="Key name for secondary text input_ids"
    )
    
    secondary_text_attention_mask_key: str = Field(
        default="attention_mask", 
        description="Key name for secondary text attention_mask"
    )
    
    # Fusion network configuration
    fusion_hidden_dim: Optional[int] = Field(
        default=None, 
        description="Hidden dimension for fusion network (auto-calculated if None)"
    )
    
    fusion_dropout: float = Field(
        default=0.1, 
        description="Dropout rate for fusion network"
    )
    
    # Optional separate BERT fine-tuning settings
    primary_reinit_pooler: Optional[bool] = Field(
        default=None, 
        description="Reinitialize primary BERT pooler (falls back to reinit_pooler if None)"
    )
    
    primary_reinit_layers: Optional[int] = Field(
        default=None, 
        description="Number of primary BERT layers to reinitialize (falls back to reinit_layers if None)"
    )
    
    secondary_reinit_pooler: Optional[bool] = Field(
        default=None, 
        description="Reinitialize secondary BERT pooler (falls back to reinit_pooler if None)"
    )
    
    secondary_reinit_layers: Optional[int] = Field(
        default=None, 
        description="Number of secondary BERT layers to reinitialize (falls back to reinit_layers if None)"
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _primary_tokenizer_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _secondary_tokenizer_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _trimodal_model_config_dict: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    # Explicitly define the model_config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        protected_namespaces=(),
    )

    @property
    def primary_tokenizer_config(self) -> Dict[str, Any]:
        """Get primary text tokenizer configuration dictionary."""
        if self._primary_tokenizer_config is None:
            self._primary_tokenizer_config = {
                "name": self.primary_tokenizer or self.tokenizer,
                "max_length": self.max_sen_len,
                "fixed_length": self.fixed_tokenizer_length,
                "text_field": self.primary_text_name,
                "input_ids_key": self.primary_text_input_ids_key,
                "attention_mask_key": self.primary_text_attention_mask_key,
            }
        return self._primary_tokenizer_config

    @property
    def secondary_tokenizer_config(self) -> Dict[str, Any]:
        """Get secondary text tokenizer configuration dictionary."""
        if self._secondary_tokenizer_config is None:
            self._secondary_tokenizer_config = {
                "name": self.secondary_tokenizer or self.tokenizer,
                "max_length": self.max_sen_len,
                "fixed_length": self.fixed_tokenizer_length,
                "text_field": self.secondary_text_name,
                "input_ids_key": self.secondary_text_input_ids_key,
                "attention_mask_key": self.secondary_text_attention_mask_key,
            }
        return self._secondary_tokenizer_config

    @property
    def trimodal_model_config_dict(self) -> Dict[str, Any]:
        """Get complete tri-modal model configuration dictionary."""
        if self._trimodal_model_config_dict is None:
            # Get base config from parent's get_config method
            base_config = self.get_config()
            self._trimodal_model_config_dict = {
                **base_config,
                # Tri-modal specific configuration
                "chat_text_name": self.primary_text_name,
                "shiptrack_text_name": self.secondary_text_name,
                "chat_tokenizer": self.primary_tokenizer or self.tokenizer,
                "shiptrack_tokenizer": self.secondary_tokenizer or self.tokenizer,
                "chat_hidden_common_dim": self.primary_hidden_common_dim or self.hidden_common_dim,
                "shiptrack_hidden_common_dim": self.secondary_hidden_common_dim or self.hidden_common_dim,
                "chat_text_input_ids_key": self.primary_text_input_ids_key,
                "chat_text_attention_mask_key": self.primary_text_attention_mask_key,
                "shiptrack_text_input_ids_key": self.secondary_text_input_ids_key,
                "shiptrack_text_attention_mask_key": self.secondary_text_attention_mask_key,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "fusion_dropout": self.fusion_dropout,
                "chat_reinit_pooler": self.primary_reinit_pooler if self.primary_reinit_pooler is not None else self.reinit_pooler,
                "chat_reinit_layers": self.primary_reinit_layers if self.primary_reinit_layers is not None else self.reinit_layers,
                "shiptrack_reinit_pooler": self.secondary_reinit_pooler if self.secondary_reinit_pooler is not None else self.reinit_pooler,
                "shiptrack_reinit_layers": self.secondary_reinit_layers if self.secondary_reinit_layers is not None else self.reinit_layers,
                # Add text processing fields
                "max_sen_len": self.max_sen_len,
                "chunk_trancate": self.chunk_trancate,
                "max_total_chunks": self.max_total_chunks,
            }
        return self._trimodal_model_config_dict

    @model_validator(mode="after")
    def validate_trimodal_hyperparameters(self) -> "TriModalHyperparameters":
        """Validate tri-modal specific hyperparameters and initialize derived fields."""
        # Call parent validator first
        super().validate_dimensions()
        
        # Tri-modal specific validations
        if self.primary_text_name == self.secondary_text_name:
            raise ValueError("primary_text_name and secondary_text_name must be different")
        
        # Initialize derived fields
        self._primary_tokenizer_config = None
        self._secondary_tokenizer_config = None
        self._trimodal_model_config_dict = None
        
        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include tri-modal specific derived fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add tri-modal derived fields that should be exposed
        derived_fields = {
            "primary_tokenizer_config": self.primary_tokenizer_config,
            "secondary_tokenizer_config": self.secondary_tokenizer_config,
            "trimodal_model_config_dict": self.trimodal_model_config_dict,
        }

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
