"""
Models package for Temporal Self-Attention architecture.

This package contains the core model components for the TSA framework.
"""

# Import from categorical embeddings module
from .categorical_embeddings import (
    PyTorchCategoricalTransformer,
    CategoricalEmbeddingLayer,
    MultiCategoricalEmbedding,
    create_pytorch_categorical_transformer,
    create_multi_categorical_embedding
)

from .order_attention_layer import (
    OrderAttentionLayer,
    TimeEncode,
    TimeEncoder,
    FeatureAggregationMLP,
    AttentionLayer
)

from .feature_attention_layer import (
    FeatureAttentionLayer,
    AttentionLayerPreNorm,
    compute_feature_interactions_fm,
    FeatureInteractionLayer,
    MLPBlock
)

from .order_feature_attention_classifier import (
    OrderFeatureAttentionClassifier,
    create_order_feature_attention_classifier
)

from .two_seq_moe_order_feature_attention_classifier import (
    TwoSeqMoEOrderFeatureAttentionClassifier,
    create_two_seq_moe_order_feature_attention_classifier
)

__all__ = [
    # Categorical Embeddings
    "PyTorchCategoricalTransformer",
    "CategoricalEmbeddingLayer",
    "MultiCategoricalEmbedding", 
    "create_pytorch_categorical_transformer",
    "create_multi_categorical_embedding",
    
    # Order Attention Components
    "OrderAttentionLayer",
    "TimeEncode",
    "TimeEncoder", 
    "FeatureAggregationMLP",
    "AttentionLayer",
    
    # Feature Attention Components
    "FeatureAttentionLayer",
    "AttentionLayerPreNorm",
    "compute_feature_interactions_fm",
    "FeatureInteractionLayer",
    "MLPBlock",
    
    # Complete Models
    "OrderFeatureAttentionClassifier",
    "create_order_feature_attention_classifier",
    "TwoSeqMoEOrderFeatureAttentionClassifier", 
    "create_two_seq_moe_order_feature_attention_classifier",
]
