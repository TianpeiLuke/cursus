"""
Models package for Temporal Self-Attention architecture.

This package contains the core model components for the TSA framework.
"""

from .categorical_transformer import (
    CategoricalTransformer,
    PyTorchCategoricalTransformer,
    create_categorical_transformer,
    create_pytorch_categorical_transformer
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

__version__ = "1.0.0"

__all__ = [
    # Categorical Transformer
    "CategoricalTransformer",
    "PyTorchCategoricalTransformer", 
    "create_categorical_transformer",
    "create_pytorch_categorical_transformer",
    
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
