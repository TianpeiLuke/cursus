"""
Order Attention Layer Module for Temporal Self-Attention Model

This module implements the OrderAttentionLayer which processes sequences to learn
temporal patterns and order-level representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional, Tuple


class TimeEncode(torch.nn.Module):
    """
    Time encoding layer using periodic functions (sinusoids).

    This layer encodes time information with learnable periodic functions,
    similar to positional encoding in transformers but for temporal data.
    """

    def __init__(self, time_dim: int, device=None, dtype=None):
        """
        Initialize TimeEncode layer.

        Args:
            time_dim: Dimension of time encoding
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TimeEncode, self).__init__()

        self.time_dim = time_dim

        self.weight = nn.Parameter(torch.empty((time_dim, 1), **factory_kwargs))
        self.emb_tbl_bias = nn.Parameter(torch.empty(time_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters using Kaiming uniform initialization."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.emb_tbl_bias, -bound, bound)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time encoding.

        Args:
            tt: Time tensor of shape (batch_size, seq_len, 1)

        Returns:
            Time encoded tensor of shape (seq_len, batch_size, time_dim)
        """
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(F.linear(tt, self.weight[1:, :], self.emb_tbl_bias[1:]))
        out1 = F.linear(tt, self.weight[0:1, :], self.emb_tbl_bias[0:1])
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(2)
        t = t.permute(1, 0, 2)

        return t


class TimeEncoder(torch.nn.Module):
    """
    Alternative time encoder using linear and periodic components.

    This is a simpler version of TimeEncode that uses separate linear
    and periodic transformations.
    """

    def __init__(self, time_dim: int):
        """
        Initialize TimeEncoder.

        Args:
            time_dim: Dimension of time encoding
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        self.periodic = nn.Linear(1, time_dim - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time encoding.

        Args:
            tt: Time tensor of shape (batch_size, seq_len, 1)

        Returns:
            Time encoded tensor of shape (seq_len, batch_size, time_dim)
        """
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        t = torch.cat([out1, out2], -1)
        t = t.squeeze(2)
        t = t.permute(1, 0, 2)

        return t


class FeatureAggregationMLP(torch.nn.Module):
    """
    Multi-layer perceptron for feature aggregation.

    This module encodes input features using a progressive dimension reduction
    approach, commonly used for feature-level aggregation in attention mechanisms.
    """

    def __init__(
        self,
        num_feature: int,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.1,
    ):
        """
        Initialize FeatureAggregationMLP.

        Args:
            num_feature: Number of input features
            activation: Activation function type
            dropout_rate: Dropout rate for regularization
        """
        super(FeatureAggregationMLP, self).__init__()

        self.dim_embed = num_feature

        # Progressive dimension reduction
        layers = []
        current_dim = num_feature

        while current_dim > 32:
            next_dim = max(current_dim // 2, 32)
            layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    nn.LeakyReLU() if activation == "leaky_relu" else nn.ReLU(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ]
            )
            current_dim = next_dim

        # Final layer to single output
        layers.append(nn.Linear(current_dim, 1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            Encoded tensor of shape (batch_size, seq_len, 1)
        """
        encode = self.encoder(x)
        return encode


class TemporalMultiheadAttention(nn.Module):
    """
    Temporal multi-head attention mechanism.

    This is a placeholder for the temporal attention mechanism.
    In the actual implementation, this would include time-aware attention
    computations that consider temporal relationships between sequence elements.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize TemporalMultiheadAttention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Use standard multihead attention as base
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        time: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            time: Time encoding tensor
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # For now, use standard attention (temporal logic would be added here)
        return self.attention(
            query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer.

    This is a placeholder for the MoE implementation.
    In practice, this would implement sparse expert routing.
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
        self.expert = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE (simplified)."""
        return self.expert(x)


class AttentionLayer(torch.nn.Module):
    """
    A multi-head attention layer with feedforward network.

    This layer implements the standard transformer attention mechanism
    with optional temporal awareness and mixture of experts.
    """

    def __init__(
        self,
        dim_embed: int,
        dim_attn_feedforward: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_moe: bool = True,
        num_experts: int = 5,
        use_time_seq: bool = True,
    ):
        """
        Initialize AttentionLayer.

        Args:
            dim_embed: Embedding dimension
            dim_attn_feedforward: Feedforward network dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_moe: Whether to use mixture of experts
            num_experts: Number of experts for MoE
            use_time_seq: Whether to use temporal information
        """
        super().__init__()

        # parameters
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.use_time_seq = use_time_seq

        # main blocks
        if self.use_time_seq:
            self.multi_attn = TemporalMultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )
        else:
            self.multi_attn = nn.modules.MultiheadAttention(
                dim_embed, num_heads, dropout=dropout
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

        if use_moe:
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
        time: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through attention layer.

        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            time: Time encoding tensor
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Output tensor of shape (seq_len, batch_size, embed_dim)
        """
        # multihead attention
        if self.use_time_seq and time is not None:
            x2, _ = self.multi_attn(
                x, x, x, time, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        else:
            x2, _ = self.multi_attn(
                x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )

        # add & norm
        x = x + self.dropout1(x2)
        x = self.layer_norm1(x)

        # feedforward
        x2 = self.feedforward(x)

        # add & norm
        x = x + self.dropout2(x2)
        x = self.layer_norm2(x)

        return x


class OrderAttentionLayer(torch.nn.Module):
    """
    Order attention layer for processing sequential data.

    This layer takes sequence data, expands it with trainable embedding tables,
    and aggregates on the feature level before feeding to multiple attention layers.
    The feature aggregation method uses an MLP-like encoder.
    """

    def __init__(
        self,
        n_cat_features: int,
        n_num_features: int,
        n_embedding: int,
        seq_len: int,
        dim_embed: int,
        dim_attn_feedforward: int,
        embedding_table: nn.Module,
        num_heads: int = 1,
        dropout: float = 0.1,
        n_layers_order: int = 1,
        emb_tbl_use_bias: bool = True,
        use_moe: bool = True,
        num_experts: int = 5,
        use_time_seq: bool = True,
        return_seq: bool = False,
    ):
        """
        Initialize OrderAttentionLayer.

        Args:
            n_cat_features: Number of categorical sequence features
            n_num_features: Number of numerical sequence features
            n_embedding: Size of sequence embedding table
            seq_len: Sequence length
            dim_embed: Output embedding dimension (should equal 2 * embedding_table_dim)
            dim_attn_feedforward: Dimension of feedforward network inside AttentionLayer
            embedding_table: Embedding table for sequence features
            num_heads: Number of attention heads
            dropout: Dropout rate
            n_layers_order: Number of attention layers
            emb_tbl_use_bias: Whether to use bias term in embeddings
            use_moe: Whether to use mixture-of-experts structure
            num_experts: Number of experts for MoE layer
            use_time_seq: Whether to use time information
            return_seq: Whether to return sequence of embeddings
        """
        super().__init__()

        # parameters
        self.n_cat_features = n_cat_features
        self.n_num_features = n_num_features
        self.n_embedding = n_embedding
        self.seq_len = seq_len
        self.dim_embed = dim_embed
        self.dim_attn_feedforward = dim_attn_feedforward
        self.num_heads = num_heads
        self.return_seq = return_seq
        self.use_time_seq = use_time_seq

        # main blocks
        self.dummy_order = nn.Parameter(torch.rand(1, dim_embed))

        embedding_table_dim = dim_embed // 2
        self.embedding_table_dim = embedding_table_dim

        self.embedding = embedding_table
        self.layer_norm_feature = nn.LayerNorm(int(embedding_table_dim * 2))

        # stack multiple attention layers
        self.layer_stack = nn.ModuleList(
            [
                AttentionLayer(
                    dim_embed,
                    dim_attn_feedforward,
                    num_heads,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    use_time_seq=use_time_seq,
                )
                for _ in range(n_layers_order)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_embed)

        self.emb_tbl_bias = (
            nn.Parameter(
                torch.randn(n_cat_features + n_num_features, embedding_table_dim)
            )
            if emb_tbl_use_bias
            else None
        )

        self.feature_aggregation_cat = FeatureAggregationMLP(n_cat_features)
        self.feature_aggregation_num = FeatureAggregationMLP(n_num_features)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        time_seq: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through order attention layer.

        Args:
            x_cat: Categorical sequence features of shape (batch_size, seq_len, n_cat_features)
            x_num: Numerical sequence features of shape (batch_size, seq_len, n_num_features)
            time_seq: Time delta sequence
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Output tensor of shape (batch_size, dim_embed) or (batch_size, seq_len, dim_embed)
        """
        B = x_cat.shape[0]  # batch size
        L = x_cat.shape[1]  # sequence length

        # embedding for categorical features
        cat_indices = x_cat.int()
        # (B, L, D, E)
        x_cat_all = self.embedding(cat_indices)

        # Feature aggregation for categorical features
        x_cat = self.feature_aggregation_cat(x_cat_all.permute(0, 1, 3, 2)).squeeze(-1)

        # embedding for numerical features
        num_indices = (
            torch.arange(
                self.n_embedding - self.n_num_features + 1, self.n_embedding + 1
            )
            .repeat(B, L)
            .view(B, L, -1)
            .to(x_cat.device)
        )
        x_num_all = self.embedding(num_indices) * (x_num[..., None])

        # Feature aggregation for numerical features
        x_num = self.feature_aggregation_num(x_num_all.permute(0, 1, 3, 2)).squeeze(-1)

        # Combine categorical and numerical features
        x = torch.cat([x_cat, x_num], dim=-1)

        # Transpose for attention layer (L, B, E)
        x = x.permute(1, 0, 2)
        x = self.layer_norm_feature(x)

        # Add dummy order token
        dummy = self.dummy_order[None].squeeze(1).repeat(B, 1).unsqueeze(1)
        x = torch.cat([x, dummy.permute(1, 0, 2)], dim=0)
        x = self.layer_norm(x)

        # Prepare time sequence if needed
        if self.use_time_seq and time_seq is not None:
            time_seq = torch.cat([time_seq, torch.zeros([B, 1, 1]).to(x.device)], dim=1)
            time_seq = time_seq.permute(1, 0, 2)
        else:
            time_seq = None

        # Apply attention layers
        for att_layer in self.layer_stack:
            x = att_layer(x, time_seq, attn_mask, key_padding_mask)

        # Return final representation
        if not self.return_seq:
            x = torch.transpose(x, 0, 1)[:, -1, :]  # Take last token (dummy order)
        else:
            x = torch.transpose(x, 0, 1)  # Return full sequence

        return x
