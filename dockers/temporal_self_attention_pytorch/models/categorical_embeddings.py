"""
Categorical Embeddings Module for Temporal Self-Attention Model

This module provides PyTorch-native categorical embedding layers for
neural network integration and end-to-end training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any


class PyTorchCategoricalTransformer(nn.Module):
    """
    PyTorch-native categorical transformer for end-to-end training.
    
    This class provides a PyTorch module that can be integrated into
    neural network architectures for learnable categorical transformations.
    """
    
    def __init__(self, 
                 vocab_sizes: Dict[str, int],
                 embedding_dims: Dict[str, int],
                 column_indices: Dict[str, int],
                 dropout: float = 0.0):
        """
        Initialize PyTorchCategoricalTransformer.
        
        Args:
            vocab_sizes: Dictionary mapping column names to vocabulary sizes
            embedding_dims: Dictionary mapping column names to embedding dimensions
            column_indices: Dictionary mapping column names to column indices
            dropout: Dropout rate for embeddings
        """
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        self.column_indices = column_indices
        
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict()
        for col_name, vocab_size in vocab_sizes.items():
            embed_dim = embedding_dims.get(col_name, min(50, vocab_size // 2))
            self.embeddings[col_name] = nn.Embedding(
                vocab_size, embed_dim, padding_idx=0
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through categorical embeddings.
        
        Args:
            x: Input tensor of categorical indices
            
        Returns:
            Dictionary mapping column names to embedded representations
        """
        embeddings = {}
        
        for col_name, col_idx in self.column_indices.items():
            if col_idx < x.shape[-1]:
                # Extract column and apply embedding
                col_data = x[..., col_idx].long()
                embedded = self.embeddings[col_name](col_data)
                embeddings[col_name] = self.dropout(embedded)
        
        return embeddings
    
    def get_embedding_dim(self, column_name: str) -> int:
        """Get embedding dimension for a specific column."""
        return self.embeddings[column_name].embedding_dim
    
    def get_total_embedding_dim(self) -> int:
        """Get total embedding dimension across all columns."""
        return sum(emb.embedding_dim for emb in self.embeddings.values())


class CategoricalEmbeddingLayer(nn.Module):
    """
    Enhanced categorical embedding layer with advanced features.
    
    This layer provides additional functionality beyond basic embeddings:
    - Learnable positional encodings for sequence-aware embeddings
    - Layer normalization for stable training
    - Configurable initialization strategies
    - Support for embedding regularization
    """
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = 0,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 init_strategy: str = 'normal'):
        """
        Initialize CategoricalEmbeddingLayer.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index for padding token
            max_norm: Maximum norm for embeddings
            norm_type: Type of norm for max_norm
            scale_grad_by_freq: Scale gradients by word frequency
            sparse: Use sparse gradients
            dropout: Dropout rate
            layer_norm: Apply layer normalization
            init_strategy: Initialization strategy ('normal', 'xavier', 'kaiming')
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Create embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
        
        # Initialize embeddings
        self._init_embeddings(init_strategy)
        
        # Optional components
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else nn.Identity()
    
    def _init_embeddings(self, strategy: str):
        """Initialize embedding weights."""
        if strategy == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        elif strategy == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight)
        elif strategy == 'kaiming':
            nn.init.kaiming_uniform_(self.embedding.weight)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        # Zero out padding embedding if specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer.
        
        Args:
            x: Input tensor of categorical indices
            
        Returns:
            Embedded representations
        """
        # Apply embedding
        embedded = self.embedding(x)
        
        # Apply layer normalization
        embedded = self.layer_norm(embedded)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        return embedded


class MultiCategoricalEmbedding(nn.Module):
    """
    Multi-categorical embedding layer for handling multiple categorical features.
    
    This layer efficiently handles multiple categorical features with different
    vocabulary sizes and embedding dimensions.
    """
    
    def __init__(self,
                 feature_configs: Dict[str, Dict[str, Any]],
                 output_strategy: str = 'concat',
                 dropout: float = 0.0):
        """
        Initialize MultiCategoricalEmbedding.
        
        Args:
            feature_configs: Dictionary mapping feature names to embedding configs
                           Format: {"feature_name": {"vocab_size": int, "embedding_dim": int, ...}}
            output_strategy: How to combine embeddings ('concat', 'sum', 'mean', 'dict')
            dropout: Dropout rate applied to final output
        """
        super().__init__()
        
        self.feature_configs = feature_configs
        self.output_strategy = output_strategy
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict()
        for feature_name, config in feature_configs.items():
            self.embeddings[feature_name] = CategoricalEmbeddingLayer(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                padding_idx=config.get('padding_idx', 0),
                dropout=config.get('dropout', 0.0),
                layer_norm=config.get('layer_norm', False),
                init_strategy=config.get('init_strategy', 'normal')
            )
        
        # Output dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Compute output dimension for concat strategy
        if output_strategy == 'concat':
            self.output_dim = sum(config['embedding_dim'] for config in feature_configs.values())
        elif output_strategy in ['sum', 'mean']:
            # All embeddings must have same dimension for sum/mean
            dims = [config['embedding_dim'] for config in feature_configs.values()]
            if len(set(dims)) > 1:
                raise ValueError(f"All embedding dimensions must be equal for {output_strategy} strategy")
            self.output_dim = dims[0]
        else:  # dict strategy
            self.output_dim = None
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through multi-categorical embeddings.
        
        Args:
            x: Dictionary mapping feature names to input tensors
            
        Returns:
            Combined embeddings or dictionary of embeddings
        """
        embeddings = {}
        
        # Apply embeddings to each feature
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in x:
                embeddings[feature_name] = embedding_layer(x[feature_name])
        
        # Combine embeddings based on strategy
        if self.output_strategy == 'dict':
            return {name: self.dropout(emb) for name, emb in embeddings.items()}
        
        elif self.output_strategy == 'concat':
            # Concatenate along last dimension
            combined = torch.cat(list(embeddings.values()), dim=-1)
            return self.dropout(combined)
        
        elif self.output_strategy == 'sum':
            # Sum embeddings
            combined = torch.stack(list(embeddings.values()), dim=0).sum(dim=0)
            return self.dropout(combined)
        
        elif self.output_strategy == 'mean':
            # Mean embeddings
            combined = torch.stack(list(embeddings.values()), dim=0).mean(dim=0)
            return self.dropout(combined)
        
        else:
            raise ValueError(f"Unknown output strategy: {self.output_strategy}")
    
    def get_output_dim(self) -> Optional[int]:
        """Get output dimension (None for dict strategy)."""
        return self.output_dim


def create_pytorch_categorical_transformer(vocab_sizes: Dict[str, int],
                                         embedding_dims: Optional[Dict[str, int]] = None,
                                         column_indices: Optional[Dict[str, int]] = None,
                                         dropout: float = 0.0) -> PyTorchCategoricalTransformer:
    """
    Factory function to create PyTorchCategoricalTransformer instance.
    
    Args:
        vocab_sizes: Dictionary mapping column names to vocabulary sizes
        embedding_dims: Dictionary mapping column names to embedding dimensions
        column_indices: Dictionary mapping column names to column indices
        dropout: Dropout rate for embeddings
        
    Returns:
        Configured PyTorchCategoricalTransformer instance
    """
    if embedding_dims is None:
        embedding_dims = {col: min(50, size // 2) for col, size in vocab_sizes.items()}
    
    if column_indices is None:
        column_indices = {col: i for i, col in enumerate(vocab_sizes.keys())}
    
    return PyTorchCategoricalTransformer(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        column_indices=column_indices,
        dropout=dropout
    )


def create_multi_categorical_embedding(feature_configs: Dict[str, Dict[str, Any]],
                                     output_strategy: str = 'concat',
                                     dropout: float = 0.0) -> MultiCategoricalEmbedding:
    """
    Factory function to create MultiCategoricalEmbedding instance.
    
    Args:
        feature_configs: Dictionary mapping feature names to embedding configs
        output_strategy: How to combine embeddings ('concat', 'sum', 'mean', 'dict')
        dropout: Dropout rate applied to final output
        
    Returns:
        Configured MultiCategoricalEmbedding instance
    """
    return MultiCategoricalEmbedding(
        feature_configs=feature_configs,
        output_strategy=output_strategy,
        dropout=dropout
    )
