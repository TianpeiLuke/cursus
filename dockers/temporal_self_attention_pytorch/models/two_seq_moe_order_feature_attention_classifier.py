"""
Two Sequence MoE Order Feature Attention Classifier Module for Temporal Self-Attention Model

This module implements the TwoSeqMoEOrderFeatureAttentionClassifier which processes
two sequences (customer ID and credit card ID) with a gating mechanism and MoE structure.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .order_attention_layer import OrderAttentionLayer
from .feature_attention_layer import FeatureAttentionLayer


class TwoSeqMoEOrderFeatureAttentionClassifier(torch.nn.Module):
    """
    Two-sequence TSA model with OrderAttention module operating on customerId (cid) and 
    creditCardIds (ccid) sequences, combined with a gate function, and FeatureAttention 
    on current order's sequence and engineered features.
    
    This model processes two different sequence types:
    1. Customer ID (cid) sequences - customer-level transaction history
    2. Credit Card ID (ccid) sequences - card-level transaction history
    
    A gating mechanism determines the relative importance of each sequence type,
    and the model combines their representations for final prediction.
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
        Initialize TwoSeqMoEOrderFeatureAttentionClassifier.
        
        Args:
            n_cat_features: Number of categorical sequence features
            n_num_features: Number of numerical sequence features
            n_classes: Number of output classes
            n_embedding: Size of sequence embedding table
            seq_len: Sequence length
            n_engineered_num_features: Number of numerical engineered features
            dim_embedding_table: Dimension of embedding table
            dim_attn_feedforward: Dimension of feedforward network inside transformer layer
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

        # Main embedding table
        self.embedding = nn.Embedding(n_embedding + 2, dim_embedding_table, padding_idx=0)

        # Order attention layers for both sequences
        self.order_attention_cid = OrderAttentionLayer(
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

        self.order_attention_ccid = OrderAttentionLayer(
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

        # Gate function components
        # Use a separate, smaller embedding table for gate computation
        self.embedding_gate = nn.Embedding(n_embedding + 2, 16, padding_idx=0)
        
        self.gate_emb = OrderAttentionLayer(
            self.n_cat_features,
            self.n_num_features,
            self.n_embedding,
            self.seq_len,
            32,  # Smaller dimension for gate
            128,  # Smaller feedforward dimension
            self.embedding_gate,
            1,  # Single head
            self.dropout,
            1,  # Single layer
            self.emb_tbl_use_bias,
            False,  # No MoE for gate
            1,  # Single expert
            False,  # No time sequence for gate
            False,  # No sequence return for gate
        )
        
        # Gate scoring network
        self.gate_score = nn.Sequential(
            nn.Linear(64, 256),  # 32 * 2 = 64 (cid + ccid embeddings)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # 2 scores for cid and ccid
            nn.Softmax(dim=1),
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

        # Final classifier
        self.clf = nn.Sequential(
            nn.Linear(dim_embed + dim_embedding_table, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.n_classes),
        )

    def forward(
        self,
        x_seq_cat_cid: torch.Tensor,
        x_seq_num_cid: torch.Tensor,
        time_seq_cid: torch.Tensor,
        x_seq_cat_ccid: torch.Tensor,
        x_seq_num_ccid: torch.Tensor,
        time_seq_ccid: torch.Tensor,
        x_engineered: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask_cid: Optional[torch.Tensor] = None,
        key_padding_mask_ccid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-sequence model.
        
        Args:
            x_seq_cat_cid: Categorical sequence features for cid sequence of shape (batch_size, seq_len, n_cat_features)
            x_seq_num_cid: Numerical sequence features for cid sequence of shape (batch_size, seq_len, n_num_features)
            time_seq_cid: Time delta sequence keyed by cid
            x_seq_cat_ccid: Categorical sequence features for ccid sequence of shape (batch_size, seq_len, n_cat_features)
            x_seq_num_ccid: Numerical sequence features for ccid sequence of shape (batch_size, seq_len, n_num_features)
            time_seq_ccid: Time delta sequence keyed by ccid
            x_engineered: Numerical engineered features of shape (batch_size, dim_embedding_table)
            attn_mask: Attention mask (optional)
            key_padding_mask_cid: Key padding mask for cid sequence (optional)
            key_padding_mask_ccid: Key padding mask for ccid sequence (optional)
            
        Returns:
            Tuple of (scores, ensemble_embeddings)
            - scores: Output class scores of shape (batch_size, n_classes)
            - ensemble: Ensemble embeddings of shape (batch_size, total_embedding_dim)
        """
        B, L, D = x_seq_cat_cid.shape

        # Compute gate embeddings for both sequences
        gate_emb_cid = self.gate_emb(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        gate_emb_ccid = self.gate_emb(
            x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid, attn_mask, key_padding_mask_ccid
        )

        # Compute gate scores
        gate_scores_raw = self.gate_score(torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1))
        gate_scores = gate_scores_raw.clone()
        
        # Handle cases where ccid sequence is empty (all padding)
        # If key_padding_mask_ccid indicates all positions are padded, set ccid gate score to 0
        if key_padding_mask_ccid is not None:
            empty_ccid_mask = (torch.sum(key_padding_mask_ccid, dim=1) == self.seq_len)
            gate_scores[empty_ccid_mask, 1] = 0.0
            # Renormalize gate scores
            gate_scores = gate_scores / gate_scores.sum(dim=1, keepdim=True)

        # Determine which samples should use ccid (gate score > threshold)
        ccid_threshold = 0.05
        ccid_keep_idx = (gate_scores[:, 1] > ccid_threshold).nonzero().squeeze(-1).to(x_seq_cat_cid.device)

        # Initialize ccid embeddings with zeros
        x_ccid = torch.zeros([B, self.dim_embed]).to(x_seq_cat_cid.device)

        # Process cid sequence (always processed)
        if self.use_time_seq:
            x_cid = self.order_attention_cid(
                x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
            )
        else:
            x_cid = self.order_attention_cid(
                x_seq_cat_cid, x_seq_num_cid, None, attn_mask, key_padding_mask_cid
            )

        # Process ccid sequence (only for samples with sufficient gate score)
        if len(ccid_keep_idx) > 0:
            if self.use_time_seq:
                x_ccid[ccid_keep_idx, :] = self.order_attention_ccid(
                    x_seq_cat_ccid[ccid_keep_idx, :, :],
                    x_seq_num_ccid[ccid_keep_idx, :, :],
                    time_seq_ccid[ccid_keep_idx, :],
                    attn_mask,
                    key_padding_mask_ccid[ccid_keep_idx, :] if key_padding_mask_ccid is not None else None,
                )
            else:
                x_ccid[ccid_keep_idx, :] = self.order_attention_ccid(
                    x_seq_cat_ccid[ccid_keep_idx, :, :],
                    x_seq_num_ccid[ccid_keep_idx, :, :],
                    None,
                    attn_mask,
                    key_padding_mask_ccid[ccid_keep_idx, :] if key_padding_mask_ccid is not None else None,
                )

        # Feature attention (uses cid sequence as input, representing current order)
        x_feature = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # Combine order embeddings using gate scores
        ensemble_order = torch.einsum("i,ij->ij", gate_scores[:, 0], x_cid) + torch.einsum(
            "i,ij->ij", gate_scores[:, 1], x_ccid
        )
        ensemble_order = self.layer_norm(ensemble_order)

        # Combine order and feature embeddings
        ensemble = torch.cat([ensemble_order, x_feature], dim=-1)
        
        # Final classification
        scores = self.clf(ensemble)

        return scores, ensemble

    def get_gate_scores(
        self,
        x_seq_cat_cid: torch.Tensor,
        x_seq_num_cid: torch.Tensor,
        time_seq_cid: torch.Tensor,
        x_seq_cat_ccid: torch.Tensor,
        x_seq_num_ccid: torch.Tensor,
        time_seq_ccid: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask_cid: Optional[torch.Tensor] = None,
        key_padding_mask_ccid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get gate scores for interpretability.
        
        Args:
            x_seq_cat_cid: Categorical sequence features for cid sequence
            x_seq_num_cid: Numerical sequence features for cid sequence
            time_seq_cid: Time delta sequence keyed by cid
            x_seq_cat_ccid: Categorical sequence features for ccid sequence
            x_seq_num_ccid: Numerical sequence features for ccid sequence
            time_seq_ccid: Time delta sequence keyed by ccid
            attn_mask: Attention mask (optional)
            key_padding_mask_cid: Key padding mask for cid sequence (optional)
            key_padding_mask_ccid: Key padding mask for ccid sequence (optional)
            
        Returns:
            Gate scores tensor of shape (batch_size, 2) where [:, 0] is cid score and [:, 1] is ccid score
        """
        # Compute gate embeddings
        gate_emb_cid = self.gate_emb(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
        )
        gate_emb_ccid = self.gate_emb(
            x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid, attn_mask, key_padding_mask_ccid
        )

        # Compute and return gate scores
        gate_scores = self.gate_score(torch.cat([gate_emb_cid, gate_emb_ccid], dim=-1))
        
        # Handle empty ccid sequences
        if key_padding_mask_ccid is not None:
            empty_ccid_mask = (torch.sum(key_padding_mask_ccid, dim=1) == self.seq_len)
            gate_scores[empty_ccid_mask, 1] = 0.0
            gate_scores = gate_scores / gate_scores.sum(dim=1, keepdim=True)
        
        return gate_scores

    def get_sequence_embeddings(
        self,
        x_seq_cat_cid: torch.Tensor,
        x_seq_num_cid: torch.Tensor,
        time_seq_cid: torch.Tensor,
        x_seq_cat_ccid: torch.Tensor,
        x_seq_num_ccid: torch.Tensor,
        time_seq_ccid: torch.Tensor,
        x_engineered: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask_cid: Optional[torch.Tensor] = None,
        key_padding_mask_ccid: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Extract embeddings from different sequence components.
        
        Returns:
            Dictionary containing embeddings from different components
        """
        embeddings = {}
        
        # CID sequence embeddings
        if self.use_time_seq:
            embeddings['cid'] = self.order_attention_cid(
                x_seq_cat_cid, x_seq_num_cid, time_seq_cid, attn_mask, key_padding_mask_cid
            )
        else:
            embeddings['cid'] = self.order_attention_cid(
                x_seq_cat_cid, x_seq_num_cid, None, attn_mask, key_padding_mask_cid
            )

        # CCID sequence embeddings
        if self.use_time_seq:
            embeddings['ccid'] = self.order_attention_ccid(
                x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid, attn_mask, key_padding_mask_ccid
            )
        else:
            embeddings['ccid'] = self.order_attention_ccid(
                x_seq_cat_ccid, x_seq_num_ccid, None, attn_mask, key_padding_mask_ccid
            )

        # Feature attention embeddings
        embeddings['feature'] = self.feature_attention(x_seq_cat_cid, x_seq_num_cid, x_engineered)

        # Gate scores
        embeddings['gate_scores'] = self.get_gate_scores(
            x_seq_cat_cid, x_seq_num_cid, time_seq_cid,
            x_seq_cat_ccid, x_seq_num_ccid, time_seq_ccid,
            attn_mask, key_padding_mask_cid, key_padding_mask_ccid
        )

        return embeddings

    def freeze_embeddings(self):
        """Freeze embedding layers to prevent updates during fine-tuning."""
        self.embedding.weight.requires_grad = False
        self.embedding_engineered.weight.requires_grad = False
        self.embedding_gate.weight.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze embedding layers to allow updates."""
        self.embedding.weight.requires_grad = True
        self.embedding_engineered.weight.requires_grad = True
        self.embedding_gate.weight.requires_grad = True

    def get_model_size(self) -> dict:
        """
        Get model size information.
        
        Returns:
            Dictionary containing model size statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_parameters': (self.embedding.weight.numel() + 
                                   self.embedding_engineered.weight.numel() + 
                                   self.embedding_gate.weight.numel()),
            'cid_attention_parameters': sum(p.numel() for p in self.order_attention_cid.parameters()),
            'ccid_attention_parameters': sum(p.numel() for p in self.order_attention_ccid.parameters()),
            'feature_attention_parameters': sum(p.numel() for p in self.feature_attention.parameters()),
            'gate_parameters': sum(p.numel() for p in self.gate_emb.parameters()) + sum(p.numel() for p in self.gate_score.parameters()),
            'classifier_parameters': sum(p.numel() for p in self.clf.parameters()),
        }


def create_two_seq_moe_order_feature_attention_classifier(
    n_cat_features: int,
    n_num_features: int,
    n_classes: int = 2,
    n_embedding: int = 10000,
    seq_len: int = 51,
    n_engineered_num_features: int = 100,
    dim_embedding_table: int = 128,
    dim_attn_feedforward: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    n_layers_order: int = 2,
    n_layers_feature: int = 2,
    emb_tbl_use_bias: bool = True,
    use_moe: bool = True,
    num_experts: int = 5,
    use_time_seq: bool = True,
    return_seq: bool = False,
) -> TwoSeqMoEOrderFeatureAttentionClassifier:
    """
    Factory function to create TwoSeqMoEOrderFeatureAttentionClassifier with default parameters.
    
    Args:
        n_cat_features: Number of categorical sequence features
        n_num_features: Number of numerical sequence features
        n_classes: Number of output classes
        n_embedding: Size of sequence embedding table
        seq_len: Sequence length
        n_engineered_num_features: Number of numerical engineered features
        dim_embedding_table: Dimension of embedding table
        dim_attn_feedforward: Dimension of feedforward network
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
        Configured TwoSeqMoEOrderFeatureAttentionClassifier instance
    """
    return TwoSeqMoEOrderFeatureAttentionClassifier(
        n_cat_features=n_cat_features,
        n_num_features=n_num_features,
        n_classes=n_classes,
        n_embedding=n_embedding,
        seq_len=seq_len,
        n_engineered_num_features=n_engineered_num_features,
        dim_embedding_table=dim_embedding_table,
        dim_attn_feedforward=dim_attn_feedforward,
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
