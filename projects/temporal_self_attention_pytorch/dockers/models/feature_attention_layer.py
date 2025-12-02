"""
Feature Attention Layer Module for Temporal Self-Attention Model

This module implements the FeatureAttentionLayer which processes features to learn
interactions between aggregated features from sequences and engineered features.
"""

import torch
import torch.nn as nn
from typing import Optional


class AttentionLayerPreNorm(torch.nn.Module):
    """
    Attention layer with pre-normalization.

    This layer implements the pre-norm variant of the transformer attention
    mechanism, which applies layer normalization before the attention and
    feedforward operations rather than after.
    """

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
    ):
        """
        Initialize AttentionLayerPreNorm.

        Args:
            dim_embed: Embedding dimension
            dim_attn_feedforward: Feedforward network dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_moe: Whether to use mixture of experts
            num_experts: Number of experts for MoE
        """
        super().__init__()

        # parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads

        # main blocks
        self.multi_attn = nn.modules.MultiheadAttention(
            dim_embed, num_heads, dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

        if use_moe:
            # Simplified MoE implementation
            self.feedforward = MixtureOfExperts(
                dim=dim_embed,
                num_experts=num_experts,
                hidden_dim=dim_attn_feedforward,
                second_policy_train="random",
                second_policy_eval="random",
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embed, dim_attn_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_attn_feedforward, dim_embed),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through pre-norm attention layer.

        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Output tensor of shape (seq_len, batch_size, embed_dim)
        """
        # pre norm
        x2 = self.layer_norm1(x)

        # multihead attention
        x2, _ = self.multi_attn(
            x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # add
        x = x + self.dropout1(x2)

        # pre norm
        x2 = self.layer_norm2(x)

        # feedforward
        x2 = self.feedforward(x2)

        # add
        x = x + self.dropout2(x2)

        return x


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer for feature attention.

    This is a simplified implementation of MoE that can be used
    within the feature attention mechanism.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_dim: int,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
    ):
        """
        Initialize MixtureOfExperts.

        Args:
            dim: Input/output dimension
            num_experts: Number of expert networks
            hidden_dim: Hidden dimension for experts
            second_policy_train: Training policy for expert selection
            second_policy_eval: Evaluation policy for expert selection
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # For simplicity, use a single feedforward network
        # In a full implementation, this would have multiple expert networks
        self.expert = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE (simplified)."""
        return self.expert(x)


class FeatureAttentionLayer(torch.nn.Module):
    """
    Feature attention layer for processing feature interactions.

    This layer takes sequence data and engineered features, expands them with
    trainable embedding tables, and aggregates on the order level (takes the last order)
    before feeding to multiple AttentionLayerPreNorm layers.
    """

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_embedding: int,
        n_engineered_num_features: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        embedding_table: nn.Module,
        embedding_table_engineered: nn.Module,
        num_heads: int = 1,
        dropout: float = 0.1,
        n_layers_feature: int = 1,
        emb_tbl_use_bias: bool = True,
        use_moe: bool = True,
        num_experts: int = 5,
    ):
        """
        Initialize FeatureAttentionLayer.

        Args:
            n_cat_features: Number of categorical sequence features
            n_num_features: Number of numerical sequence features
            n_embedding: Size of sequence embedding table
            n_engineered_num_features: Number of numerical engineered features
            dim_embed: Output embedding dimension (should equal 2 * embedding_table_dim)
            dim_attn_feedforward: Dimension of feedforward network inside AttentionLayerPreNorm
            embedding_table: Embedding table for sequence features
            embedding_table_engineered: Embedding table for engineered features
            num_heads: Number of attention heads
            dropout: Dropout rate
            n_layers_feature: Number of attention layers
            emb_tbl_use_bias: Whether to use bias term in embeddings
            use_moe: Whether to use mixture-of-experts structure
            num_experts: Number of experts for MoE layer
        """
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_embedding = n_embedding
        self.n_engineered_num_features = n_engineered_num_features
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads

        # main blocks
        embedding_table_dim = dim_embed // 2
        self.embedding_table_dim = embedding_table_dim

        self.embedding = embedding_table

        # Stack multiple feature attention layers
        self.layer_stack_feature = nn.ModuleList(
            [
                AttentionLayerPreNorm(
                    embedding_table_dim,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                )
                for _ in range(n_layers_feature)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        # Bias terms for embeddings
        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, embedding_table_dim)
            )
            if emb_tbl_use_bias
            else None
        )

        # Engineered features embedding
        self.embedding_engineered = embedding_table_engineered
        self.layer_norm_engineered = nn.LayerNorm(embedding_table_dim)

        if self.n_engineered_num_features > 0:
            self.engineered_emb_tbl_bias = (
                nn.Parameter(
                    torch.randn(self.n_engineered_num_features, embedding_table_dim)
                )
                if emb_tbl_use_bias and self.n_engineered_num_features > 0
                else None
            )

    def forward(
        self, x_cat: torch.Tensor, x_num: torch.Tensor, x_engineered: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through feature attention layer.

        Args:
            x_cat: Categorical sequence features of shape (batch_size, seq_len, n_cat_features)
            x_num: Numerical sequence features of shape (batch_size, seq_len, n_num_features)
            x_engineered: Numerical engineered features of shape (batch_size, dim_embedding_table)

        Returns:
            Output tensor of shape (batch_size, embedding_table_dim)
        """
        if x_engineered is not None:
            x_engineered = x_engineered.float()

        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # Process categorical features
        # (B, L, D) => (B, L, D, E)
        cat_indices = x_cat.int()
        x_cat_all = self.embedding(cat_indices)

        # Process numerical features
        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num_all = self.embedding(num_indices) * (x_num[..., None])

        # Take the last order (current order) for feature attention
        x_cat_last = x_cat_all[:, -1, :, :]  # (B, n_cat_features, embedding_dim)
        x_num_last = x_num_all[:, -1, :, :]  # (B, n_num_features, embedding_dim)

        # Combine categorical and numerical features from last order
        x_last = torch.cat(
            [x_cat_last, x_num_last], dim=1
        )  # (B, n_cat_features + n_num_features, embedding_dim)

        if self.emb_tbl_bias is not None:
            x_last = x_last + self.emb_tbl_bias[None]

        # Process engineered features
        if self.n_engineered_num_features > 0:
            engineered_indices = torch.arange(1, self.n_engineered_num_features + 1).to(
                x_cat.device
            )
            x_engineered_emb = (
                self.embedding_engineered(engineered_indices)
                * (x_engineered[..., None])
            )

            if self.engineered_emb_tbl_bias is not None:
                x_engineered_emb = x_engineered_emb + self.engineered_emb_tbl_bias[None]

            # Add engineered features and a dummy token
            x_last = torch.cat(
                [
                    x_last,
                    x_engineered_emb,
                    self.embedding_engineered(
                        torch.zeros([B, 1]).int().to(x_cat.device)
                    ),
                ],
                dim=1,
            )
        else:
            # Add dummy token even if no engineered features
            x_last = torch.cat(
                [
                    x_last,
                    self.embedding_engineered(
                        torch.zeros([B, 1]).int().to(x_cat.device)
                    ),
                ],
                dim=1,
            )

        # Transpose for attention layer: (num_features, batch_size, embedding_dim)
        x_last = x_last.permute(1, 0, 2)
        x_last = self.layer_norm_engineered(x_last)

        # Apply feature attention layers
        for att_layer_feature in self.layer_stack_feature:
            x_last = att_layer_feature(x_last, None, None)

        # Take the last token (dummy token) as the final feature representation
        x_last = torch.transpose(x_last, 0, 1)[
            :, -1, :
        ]  # (batch_size, embedding_table_dim)

        return x_last


def compute_feature_interactions_fm(feature_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute second-order feature interactions using Factorization Machine.

    This function implements the FM interaction computation that can be used
    within the feature attention mechanism to capture feature interactions.

    Args:
        feature_embeddings: Feature embeddings tensor of shape (batch_size, num_features, embedding_dim)

    Returns:
        Feature interaction tensor of shape (batch_size, embedding_dim)
    """
    # Sum of embeddings
    summed_features_emb = torch.sum(feature_embeddings, dim=-2)
    summed_features_emb_square = torch.square(summed_features_emb)

    # Square of embeddings then sum
    squared_features_emb = torch.square(feature_embeddings)
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)

    # Factorization Machine computation
    fm_interaction = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

    return fm_interaction


class FeatureInteractionLayer(nn.Module):
    """
    Feature interaction layer using various interaction methods.

    This layer can compute feature interactions using different methods
    such as Factorization Machine, bilinear interactions, or attention-based
    interactions.
    """

    def __init__(
        self, embedding_dim: int, interaction_type: str = "fm", num_heads: int = 1
    ):
        """
        Initialize FeatureInteractionLayer.

        Args:
            embedding_dim: Dimension of feature embeddings
            interaction_type: Type of interaction ("fm", "bilinear", "attention")
            num_heads: Number of heads for attention-based interactions
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.interaction_type = interaction_type
        self.num_heads = num_heads

        if interaction_type == "bilinear":
            self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, embedding_dim)
        elif interaction_type == "attention":
            self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, feature_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute feature interactions.

        Args:
            feature_embeddings: Feature embeddings of shape (batch_size, num_features, embedding_dim)

        Returns:
            Feature interactions of shape (batch_size, embedding_dim)
        """
        if self.interaction_type == "fm":
            return compute_feature_interactions_fm(feature_embeddings)
        elif self.interaction_type == "bilinear":
            # Compute pairwise bilinear interactions
            batch_size, num_features, embed_dim = feature_embeddings.shape
            interactions = []

            for i in range(num_features):
                for j in range(i + 1, num_features):
                    interaction = self.bilinear(
                        feature_embeddings[:, i, :], feature_embeddings[:, j, :]
                    )
                    interactions.append(interaction)

            if interactions:
                return torch.stack(interactions, dim=1).mean(dim=1)
            else:
                return torch.zeros(
                    batch_size, embed_dim, device=feature_embeddings.device
                )

        elif self.interaction_type == "attention":
            # Use self-attention for feature interactions
            # Transpose for attention: (num_features, batch_size, embedding_dim)
            x = feature_embeddings.transpose(0, 1)
            attended, _ = self.attention(x, x, x)
            # Return mean of attended features
            return attended.transpose(0, 1).mean(dim=1)

        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")


class MLPBlock(nn.Module):
    """
    Multi-layer perceptron block for feature processing.

    This is a general-purpose MLP that can be used for various
    feature transformation tasks within the feature attention layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize MLPBlock.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout_rate: Dropout rate
            activation: Activation function type
        """
        super().__init__()

        activation_fn = nn.ReLU() if activation == "relu" else nn.GELU()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.mlp(x)
