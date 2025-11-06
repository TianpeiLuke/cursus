"""
Feature Aggregation Module for Temporal Self-Attention Model

This module handles the aggregation and preprocessing of features,
including categorical encoding, numerical scaling, and engineered feature processing.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Import the improved CategoricalTransformer from models
from ..models.categorical_transformer import (
    CategoricalTransformer,
    PyTorchCategoricalTransformer,
    create_categorical_transformer,
    create_pytorch_categorical_transformer,
)


class FeatureAggregator:
    """
    Handles feature-level aggregation and preprocessing for TSA model.

    This class manages:
    - Categorical feature encoding and embedding preparation
    - Numerical feature scaling and normalization
    - Engineered feature processing
    - Feature interaction computation (Factorization Machine)
    """

    def __init__(self, config_path: str = "/opt/ml/processing/input/config"):
        """
        Initialize FeatureAggregator with configuration.

        Args:
            config_path: Path to configuration files
        """
        self.config_path = config_path

        # Load feature processing configurations
        self._load_feature_configs()

    def _load_feature_configs(self):
        """Load feature processing configurations from files."""
        # Load categorical mappings
        cat_to_index_file = "cat_to_index.json"
        with open(os.path.join(self.config_path, cat_to_index_file), "r") as f:
            self.categorical_map = json.load(f)

        # Handle dot notation in keys
        for key in list(self.categorical_map):
            self.categorical_map[key.replace(".", "__DOT__")] = self.categorical_map[
                key
            ]

        # Load default values
        default_value_dict_file = "default_value_dict.json"
        with open(os.path.join(self.config_path, default_value_dict_file), "r") as f:
            self.default_value_dict = json.load(f)

        # Handle dot notation in keys
        for key in list(self.default_value_dict):
            self.default_value_dict[key.replace(".", "__DOT__")] = (
                self.default_value_dict[key]
            )

        # Load preprocessor parameters
        preprocessor_file = "preprocessor.pkl"
        with open(os.path.join(self.config_path, preprocessor_file), "rb") as f:
            preprocessor = pickle.load(f)

        self.percentile_score_map = preprocessor["bin_map"]
        self.seq_num_scale_ = preprocessor["seq_num_scale_"]
        self.seq_num_min_ = preprocessor["seq_num_min_"]
        self.num_static_scale_ = preprocessor["num_static_scale_"]
        self.num_static_min_ = preprocessor["num_static_min_"]

        # Handle deprecated features (will be removed in future versions)
        self.num_static_scale_ = np.delete(self.num_static_scale_, [266, 267])
        self.num_static_min_ = np.delete(self.num_static_min_, [266, 267])

    def aggregate_categorical_features(
        self,
        categorical_data: np.ndarray,
        feature_names: List[str],
        numerical_cat_vars_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Aggregate and encode categorical features.

        Args:
            categorical_data: Raw categorical feature data
            feature_names: List of feature names
            numerical_cat_vars_indices: Indices of numerical categorical variables

        Returns:
            Encoded categorical features ready for embedding
        """
        # Handle numerical categorical variables (convert to string)
        if numerical_cat_vars_indices:
            for i in numerical_cat_vars_indices:
                if i < categorical_data.shape[1]:
                    for j in range(categorical_data.shape[0]):
                        cur_var = categorical_data[j, i]
                        if cur_var not in ["", "My Text String"]:
                            try:
                                categorical_data[j, i] = str(int(float(cur_var)))
                            except (ValueError, TypeError):
                                pass  # Keep original value if conversion fails

        # Convert to string format for consistent processing
        categorical_data = categorical_data.astype(str)

        # Create categorical transformer
        transformer = CategoricalTransformer(
            categorical_map=self.categorical_map, columns_list=feature_names
        )

        # Transform categorical data
        encoded_data = transformer.transform(categorical_data)

        # Handle None values and convert to integers
        encoded_data[encoded_data == "None"] = "0"
        encoded_data = encoded_data.astype(int)

        return encoded_data

    def aggregate_numerical_features(
        self, numerical_data: np.ndarray, feature_type: str = "sequence"
    ) -> np.ndarray:
        """
        Aggregate and scale numerical features.

        Args:
            numerical_data: Raw numerical feature data
            feature_type: Type of features ("sequence" or "engineered")

        Returns:
            Scaled numerical features
        """
        numerical_data = numerical_data.astype(float)

        if feature_type == "sequence":
            # Apply sequence-specific scaling
            scaled_data = numerical_data * np.array(self.seq_num_scale_) + np.array(
                self.seq_num_min_
            )
        elif feature_type == "engineered":
            # Apply engineered feature scaling
            scaled_data = numerical_data * np.array(self.num_static_scale_) + np.array(
                self.num_static_min_
            )
        else:
            # Default: no scaling
            scaled_data = numerical_data

        return scaled_data

    def aggregate_engineered_features(
        self, input_data: Dict[str, Any], dense_num_vars: List[str]
    ) -> np.ndarray:
        """
        Aggregate engineered (dense) numerical features.

        Args:
            input_data: Dictionary containing feature data
            dense_num_vars: List of dense numerical variable names

        Returns:
            Processed engineered features
        """
        # Extract engineered features
        dense_features = []
        for var in dense_num_vars:
            value = input_data.get(var, "")
            if value in ["", "My Text String"]:
                value = self.default_value_dict.get(var, 0.0)
            try:
                dense_features.append(float(value))
            except (ValueError, TypeError):
                dense_features.append(0.0)

        # Convert to numpy array and exclude last two features (deprecated)
        dense_array = np.array(dense_features)[:-2]

        # Apply scaling
        scaled_features = self.aggregate_numerical_features(
            dense_array.reshape(1, -1), feature_type="engineered"
        )

        return scaled_features.flatten()

    def compute_feature_interactions(
        self, feature_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute second-order feature interactions using Factorization Machine.

        Args:
            feature_embeddings: Feature embeddings tensor of shape (batch_size, num_features, embedding_dim)

        Returns:
            Feature interaction tensor
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

    def create_feature_aggregation_mlp(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout_rate: float = 0.1,
    ) -> nn.Module:
        """
        Create MLP for feature aggregation.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout rate

        Returns:
            Feature aggregation MLP module
        """
        if hidden_dims is None:
            # Default progressive dimension reduction
            hidden_dims = []
            current_dim = input_dim
            while current_dim > output_dim * 8:
                current_dim = current_dim // 2
                hidden_dims.append(current_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def process_feature_batch(
        self,
        df: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: List[str],
        engineered_vars: List[str],
        numerical_cat_vars_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of features.

        Args:
            df: DataFrame containing feature data
            categorical_vars: List of categorical variable names
            numerical_vars: List of numerical variable names
            engineered_vars: List of engineered variable names
            numerical_cat_vars_indices: Indices of numerical categorical variables

        Returns:
            Tuple of (categorical_features, numerical_features, engineered_features)
        """
        categorical_features_list = []
        numerical_features_list = []
        engineered_features_list = []

        for _, row in df.iterrows():
            row_data = row.to_dict()

            # Process categorical features
            cat_data = np.array(
                [row_data.get(var, "") for var in categorical_vars]
            ).reshape(1, -1)
            cat_encoded = self.aggregate_categorical_features(
                cat_data, categorical_vars, numerical_cat_vars_indices
            )
            categorical_features_list.append(cat_encoded.flatten())

            # Process numerical features
            num_data = np.array(
                [
                    float(row_data.get(var, 0.0))
                    if row_data.get(var, "") not in ["", "My Text String"]
                    else 0.0
                    for var in numerical_vars
                ]
            ).reshape(1, -1)
            num_scaled = self.aggregate_numerical_features(
                num_data, feature_type="sequence"
            )
            numerical_features_list.append(num_scaled.flatten())

            # Process engineered features
            eng_features = self.aggregate_engineered_features(row_data, engineered_vars)
            engineered_features_list.append(eng_features)

        # Stack all features
        categorical_features = np.stack(categorical_features_list, axis=0)
        numerical_features = np.stack(numerical_features_list, axis=0)
        engineered_features = np.stack(engineered_features_list, axis=0)

        return categorical_features, numerical_features, engineered_features


class FeatureAggregationMLP(nn.Module):
    """
    Multi-layer perceptron for feature aggregation.

    This module implements a configurable MLP that can be used for:
    - Feature dimension reduction
    - Feature interaction learning
    - Attention weight computation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
    ):
        """
        Initialize FeatureAggregationMLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default progressive dimension reduction
        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim
            while current_dim > output_dim * 8 and current_dim > 32:
                current_dim = max(current_dim // 2, output_dim * 2)
                hidden_dims.append(current_dim)

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation function
            if activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        return self.encoder(x)


def create_feature_aggregator(
    config_path: str = "/opt/ml/processing/input/config",
) -> FeatureAggregator:
    """
    Factory function to create FeatureAggregator instance.

    Args:
        config_path: Path to configuration files

    Returns:
        Configured FeatureAggregator instance
    """
    return FeatureAggregator(config_path=config_path)


def compute_fm_parallel(feature_embedding: torch.Tensor) -> torch.Tensor:
    """
    Compute Factorization Machine interactions in parallel.

    Args:
        feature_embedding: Feature embeddings of shape (batch_size, num_features, embedding_dim)

    Returns:
        FM interaction tensor of shape (batch_size, embedding_dim)
    """
    # Sum of embeddings
    summed_features_emb = torch.sum(feature_embedding, dim=-2)
    summed_features_emb_square = torch.square(summed_features_emb)

    # Square of embeddings then sum
    squared_features_emb = torch.square(feature_embedding)
    squared_sum_features_emb = torch.sum(squared_features_emb, dim=-2)

    # Factorization Machine computation
    fm_interaction = 0.5 * (summed_features_emb_square - squared_sum_features_emb)

    return fm_interaction


# Utility functions for backward compatibility
def arr_from_dict(input_data: Dict[str, Any], var_list: List[str]) -> np.ndarray:
    """Legacy function for array extraction."""
    return np.expand_dims(np.array([input_data[var] for var in var_list]), axis=0)
