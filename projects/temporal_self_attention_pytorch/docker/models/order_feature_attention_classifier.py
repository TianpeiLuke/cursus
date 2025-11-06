"""
Order Feature Attention Classifier Module for Temporal Self-Attention Model

This module implements the OrderFeatureAttentionClassifier which combines
OrderAttentionLayer and FeatureAttentionLayer for single sequence TSA modeling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .order_attention_layer import OrderAttentionLayer
from .feature_attention_layer import FeatureAttentionLayer, MLPBlock


class OrderFeatureAttentionClassifier(torch.nn.Module):
    """
    Single sequence TSA model with both OrderAttention and FeatureAttention modules.

    This model processes single sequence data through two main components:
    1. OrderAttentionLayer: Processes sequences to learn temporal patterns
    2. FeatureAttentionLayer: Learns interactions between aggregated features

    The model combines embeddings from both components and feeds them through
    a classifier to produce final predictions.
    """

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_classes: int,
        n_embedding: int,
        seq_len: int,
        n_engineered_num_features: int,
        dim_embedding_table: int,
        dim_attn_feedforward: int,
        use_mlp: bool = False,
        num_heads: int = 1,
        dropout: float = 0.1,
        n_layers_order: int = 1,
        n_layers_feature: int = 1,
        emb_tbl_use_bias: bool = True,
        use_moe: bool = True,
        num_experts: int = 5,
        use_time_seq: bool = True,
        return_seq: bool = False,
    ):
        """
        Initialize OrderFeatureAttentionClassifier.

        Args:
            n_cat_features: Number of categorical sequence features
            n_num_features: Number of numerical sequence features
            n_classes: Number of output classes
            n_embedding: Size of sequence embedding table
            seq_len: Sequence length
            n_engineered_num_features: Number of numerical engineered features
            dim_embedding_table: Dimension of embedding table
            dim_attn_feedforward: Dimension of feedforward network inside transformer layer
            use_mlp: Whether to use MLP on numerical features to produce part of the embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            n_layers_order: Number of order attention layers
            n_layers_feature: Number of feature attention layers
            emb_tbl_use_bias: Whether to use bias term in embeddings
            use_moe: Whether to use mixture-of-experts structure for transformer layer
            num_experts: Number of experts to use for MoE layer
            use_time_seq: Whether to use time information
            return_seq: Whether to return sequence of embeddings
        """
        super().__init__()

        # Store parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_classes = n_classes
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embedding_table = dim_embedding_table
        self.dim_attn_feedforward = dim_attn_feedforward
        self.use_mlp = use_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layers_order = n_layers_order
        self.n_layers_feature = n_layers_feature
        self.emb_tbl_use_bias = emb_tbl_use_bias
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.use_time_seq = use_time_seq
        self.return_seq = return_seq

        # Embedding dimensions
        dim_embed = 2 * dim_embedding_table
        self.dim_embed = dim_embed

        # Main embedding tables
        self.embedding = nn.Embedding(
            n_embedding + 2, dim_embedding_table, padding_idx=0
        )

        # Order attention layer
        self.order_attention = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.num_heads,
            self.dropout,
            self.n_layers_order,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
            self.use_time_seq,
            self.return_seq,
        )

        # Engineered features embedding
        self.embedding_engineered = nn.Embedding(
            n_engineered_num_features + 1, dim_embedding_table, padding_idx=0
        )

        # Feature attention layer
        self.feature_attention = FeatureAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.n_engineered_num_features,
            self.dim_embed,
            self.dim_attn_feedforward,
            self.embedding,
            self.embedding_engineered,
            self.num_heads,
            self.dropout,
            self.n_layers_feature,
            self.emb_tbl_use_bias,
            self.use_moe,
            self.num_experts,
        )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(dim_embed)
        self.dropout_layer = nn.Dropout(dropout)

        # Optional MLP for numerical features
        if self.use_mlp:
            self.MLP = MLPBlock(
                self.n_num_features + self.n_engineered_num_features,
                1024,
                dim_embedding_table,
                dropout,
            )
            self.layer_norm_engineered = nn.LayerNorm(dim_embedding_table)

            # Classifier with MLP embeddings
            self.clf = nn.Sequential(
                nn.Linear(dim_embed + dim_embedding_table + dim_embedding_table, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, n_classes),
            )
        else:
            # Classifier without MLP embeddings
            self.clf = nn.Sequential(
                nn.Linear(dim_embed + dim_embedding_table, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, n_classes),
            )

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        x_engineered: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x_cat: Categorical sequence features of shape (batch_size, seq_len, n_cat_features)
            x_num: Numerical sequence features of shape (batch_size, seq_len, n_num_features)
            x_engineered: Numerical engineered features of shape (batch_size, dim_embedding_table)
            time_seq: Time delta sequence
            attn_mask: Attention mask (optional)
            key_padding_mask: Key padding mask (optional)

        Returns:
            Tuple of (scores, ensemble_embeddings)
            - scores: Output class scores of shape (batch_size, n_classes)
            - ensemble: Ensemble embeddings of shape (batch_size, total_embedding_dim)
        """
        # Order attention
        if self.use_time_seq:
            x_order = self.order_attention(
                x_cat, x_num, time_seq, attn_mask, key_padding_mask
            )
        else:
            x_order = self.order_attention(
                x_cat, x_num, None, attn_mask, key_padding_mask
            )

        # Feature attention
        x_feature = self.feature_attention(x_cat, x_num, x_engineered)

        # Optional MLP processing
        if self.use_mlp:
            # Combine numerical and engineered features for MLP
            x_mlp_input = torch.cat([x_num[:, -1, :], x_engineered], dim=-1)
            x_mlp = self.MLP(x_mlp_input)
            x_mlp = self.layer_norm_engineered(x_mlp)

            # Combine all embeddings
            ensemble = torch.cat([x_order, x_feature, x_mlp], dim=-1)
        else:
            # Combine order and feature embeddings
            ensemble = torch.cat([x_order, x_feature], dim=-1)

        # Final classification
        scores = self.clf(ensemble)

        return scores, ensemble

    def get_attention_weights(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        x_engineered: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Extract attention weights from the model for interpretability.

        Args:
            x_cat: Categorical sequence features
            x_num: Numerical sequence features
            x_engineered: Numerical engineered features
            time_seq: Time delta sequence
            attn_mask: Attention mask (optional)
            key_padding_mask: Key padding mask (optional)

        Returns:
            Dictionary containing attention weights from different layers
        """
        attention_weights = {}

        # This would require modifications to the attention layers to return weights
        # For now, return empty dict as placeholder
        return attention_weights

    def get_embeddings(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        x_engineered: torch.Tensor,
        time_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Extract intermediate embeddings from different components.

        Args:
            x_cat: Categorical sequence features
            x_num: Numerical sequence features
            x_engineered: Numerical engineered features
            time_seq: Time delta sequence
            attn_mask: Attention mask (optional)
            key_padding_mask: Key padding mask (optional)

        Returns:
            Dictionary containing embeddings from different components
        """
        embeddings = {}

        # Order attention embeddings
        if self.use_time_seq:
            embeddings["order"] = self.order_attention(
                x_cat, x_num, time_seq, attn_mask, key_padding_mask
            )
        else:
            embeddings["order"] = self.order_attention(
                x_cat, x_num, None, attn_mask, key_padding_mask
            )

        # Feature attention embeddings
        embeddings["feature"] = self.feature_attention(x_cat, x_num, x_engineered)

        # Optional MLP embeddings
        if self.use_mlp:
            x_mlp_input = torch.cat([x_num[:, -1, :], x_engineered], dim=-1)
            embeddings["mlp"] = self.MLP(x_mlp_input)

        return embeddings

    def freeze_embeddings(self):
        """Freeze embedding layers to prevent updates during fine-tuning."""
        self.embedding.weight.requires_grad = False
        self.embedding_engineered.weight.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze embedding layers to allow updates."""
        self.embedding.weight.requires_grad = True
        self.embedding_engineered.weight.requires_grad = True

    def get_model_size(self) -> dict:
        """
        Get model size information.

        Returns:
            Dictionary containing model size statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "embedding_parameters": self.embedding.weight.numel()
            + self.embedding_engineered.weight.numel(),
            "order_attention_parameters": sum(
                p.numel() for p in self.order_attention.parameters()
            ),
            "feature_attention_parameters": sum(
                p.numel() for p in self.feature_attention.parameters()
            ),
            "classifier_parameters": sum(p.numel() for p in self.clf.parameters()),
        }


def create_order_feature_attention_classifier(
    n_cat_features: int,
    n_num_features: int,
    n_classes: int = 2,
    n_embedding: int = 10000,
    seq_len: int = 51,
    n_engineered_num_features: int = 100,
    dim_embedding_table: int = 128,
    dim_attn_feedforward: int = 512,
    use_mlp: bool = False,
    num_heads: int = 8,
    dropout: float = 0.1,
    n_layers_order: int = 2,
    n_layers_feature: int = 2,
    emb_tbl_use_bias: bool = True,
    use_moe: bool = True,
    num_experts: int = 5,
    use_time_seq: bool = True,
    return_seq: bool = False,
) -> OrderFeatureAttentionClassifier:
    """
    Factory function to create OrderFeatureAttentionClassifier with default parameters.

    Args:
        n_cat_features: Number of categorical sequence features
        n_num_features: Number of numerical sequence features
        n_classes: Number of output classes
        n_embedding: Size of sequence embedding table
        seq_len: Sequence length
        n_engineered_num_features: Number of numerical engineered features
        dim_embedding_table: Dimension of embedding table
        dim_attn_feedforward: Dimension of feedforward network
        use_mlp: Whether to use MLP on numerical features
        num_heads: Number of attention heads
        dropout: Dropout probability
        n_layers_order: Number of order attention layers
        n_layers_feature: Number of feature attention layers
        emb_tbl_use_bias: Whether to use bias term in embeddings
        use_moe: Whether to use mixture-of-experts structure
        num_experts: Number of experts for MoE layer
        use_time_seq: Whether to use time information
        return_seq: Whether to return sequence of embeddings

    Returns:
        Configured OrderFeatureAttentionClassifier instance
    """
    return OrderFeatureAttentionClassifier(
        n_cat_features=n_cat_features,
        n_num_features=n_num_features,
        n_classes=n_classes,
        n_embedding=n_embedding,
        seq_len=seq_len,
        n_engineered_num_features=n_engineered_num_features,
        dim_embedding_table=dim_embedding_table,
        dim_attn_feedforward=dim_attn_feedforward,
        use_mlp=use_mlp,
        num_heads=num_heads,
        dropout=dropout,
        n_layers_order=n_layers_order,
        n_layers_feature=n_layers_feature,
        emb_tbl_use_bias=emb_tbl_use_bias,
        use_moe=use_moe,
        num_experts=num_experts,
        use_time_seq=use_time_seq,
        return_seq=return_seq,
    )
