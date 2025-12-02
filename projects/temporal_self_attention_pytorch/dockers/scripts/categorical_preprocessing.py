"""
Categorical Preprocessing Module for Temporal Self-Attention Model

This module provides categorical feature preprocessing capabilities for
data pipeline integration and sklearn compatibility.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms categorical features using predefined mappings.

    This class handles:
    - Categorical feature encoding using lookup tables
    - Missing value handling with default mappings
    - String-to-integer conversion for embedding layers
    - PyTorch tensor support for deep learning workflows
    - Efficient vectorized operations
    """

    def __init__(
        self,
        categorical_map: Dict[str, Dict[str, int]],
        columns_list: List[str],
        default_value: int = 0,
        handle_unknown: str = "default",
    ):
        """
        Initialize CategoricalTransformer.

        Args:
            categorical_map: Dictionary mapping feature names to value->index mappings
                           Format: {"feature_name": {"category_value": index, ...}, ...}
            columns_list: List of categorical column names to transform
            default_value: Default value for unknown categories
            handle_unknown: Strategy for handling unknown categories ('default', 'error')

        Raises:
            TypeError: If categorical_map is not dict or columns_list is not list
            ValueError: If handle_unknown is not valid
        """
        if not isinstance(categorical_map, dict):
            raise TypeError(
                f"Categorical Map is not dict type: {type(categorical_map)}. Please send a dict"
            )
        if not isinstance(columns_list, list):
            raise TypeError(
                f"Columns list is not list type: {type(columns_list)}. Please send a list"
            )
        if handle_unknown not in ["default", "error"]:
            raise ValueError(
                f"handle_unknown must be 'default' or 'error', got {handle_unknown}"
            )

        self.categorical_map = categorical_map
        self.columns_list = columns_list
        self.default_value = default_value
        self.handle_unknown = handle_unknown

        # Create index to column mapping for efficient processing
        self.index_column_map = self._get_index_to_column_map(
            categorical_map=self.categorical_map, columns_list=self.columns_list
        )

        # Pre-compute vocabulary sizes for each feature
        self.vocab_sizes = {}
        for col_name, mapping in self.categorical_map.items():
            if col_name in self.columns_list:
                self.vocab_sizes[col_name] = max(mapping.values()) + 1 if mapping else 1

    def _get_index_to_column_map(
        self, categorical_map: Dict[str, Dict[str, int]], columns_list: List[str]
    ) -> Dict[int, str]:
        """
        Create a mapping from column index to column name.

        Args:
            categorical_map: Dictionary of categorical mappings
            columns_list: List of column names

        Returns:
            Dictionary mapping column indices to column names

        Example:
            categorical_map = {"A": {"X": 100}, "D": {"Y": 200}}
            columns_list = ["A", "B", "C", "D", "E", "F"]
            Returns: {0: "A", 3: "D"}
        """
        index_column_map = {}
        for index, column_name in enumerate(columns_list):
            if column_name in categorical_map:
                index_column_map[index] = column_name
        return index_column_map

    def transform(
        self, input_data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform categorical data using predefined mappings.

        Args:
            input_data: Input categorical data array or tensor

        Returns:
            Transformed data with categorical values mapped to integers

        Raises:
            TypeError: If input_data is not np.ndarray or torch.Tensor
            ValueError: If unknown categories are encountered and handle_unknown='error'
        """
        # Handle different input types
        is_tensor = isinstance(input_data, torch.Tensor)
        if is_tensor:
            device = input_data.device
            input_array = input_data.cpu().numpy()
        elif isinstance(input_data, np.ndarray):
            input_array = input_data
        else:
            raise TypeError(
                f"Input data must be np.ndarray or torch.Tensor, got {type(input_data)}"
            )

        # Create copy for transformation
        transformed_data = input_array.copy()

        # Transform each categorical column
        for col_idx, column_name in self.index_column_map.items():
            if col_idx >= transformed_data.shape[1]:
                raise ValueError(
                    f"Column index {col_idx} is out of bounds for input data"
                )

            column_map = self.categorical_map[column_name]

            # Convert column data to string for consistent processing
            column_data = transformed_data[:, col_idx].astype(str)

            # Handle NaN values
            column_data = np.where(column_data == "nan", "", column_data)

            # Vectorized transformation
            if self.handle_unknown == "default":
                # Use default value for unknown categories
                transformed_column = np.vectorize(
                    lambda x: column_map.get(x, self.default_value), otypes=[int]
                )(column_data)
            else:  # handle_unknown == 'error'
                # Check for unknown categories
                unknown_values = set(column_data) - set(column_map.keys())
                if unknown_values:
                    raise ValueError(
                        f"Unknown categories found in column {column_name}: {unknown_values}"
                    )

                transformed_column = np.vectorize(column_map.get, otypes=[int])(
                    column_data
                )

            transformed_data[:, col_idx] = transformed_column

        # Convert back to original type
        if is_tensor:
            return torch.tensor(transformed_data, dtype=torch.long, device=device)
        else:
            return transformed_data.astype(int)

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Optional[np.ndarray] = None):
        """
        Fit method for sklearn compatibility. No-op since mappings are predefined.

        Args:
            X: Input data (unused)
            y: Target data (unused)

        Returns:
            self
        """
        return self

    def fit_transform(
        self, X: Union[np.ndarray, torch.Tensor], y: Optional[np.ndarray] = None
    ):
        """
        Fit and transform in one step.

        Args:
            X: Input data to transform
            y: Target data (unused)

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_vocab_size(self, column_name: str) -> int:
        """
        Get vocabulary size for a specific column.

        Args:
            column_name: Name of the categorical column

        Returns:
            Vocabulary size (max index + 1)

        Raises:
            KeyError: If column_name is not in categorical_map
        """
        if column_name not in self.vocab_sizes:
            raise KeyError(f"Column {column_name} not found in categorical mappings")
        return self.vocab_sizes[column_name]

    def get_all_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for all categorical columns.

        Returns:
            Dictionary mapping column names to vocabulary sizes
        """
        return self.vocab_sizes.copy()

    def inverse_transform(
        self, transformed_data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform encoded data back to original categorical values.

        Args:
            transformed_data: Encoded categorical data

        Returns:
            Original categorical values

        Raises:
            ValueError: If inverse mapping is not possible
        """
        # Handle different input types
        is_tensor = isinstance(transformed_data, torch.Tensor)
        if is_tensor:
            device = transformed_data.device
            input_array = transformed_data.cpu().numpy()
        elif isinstance(transformed_data, np.ndarray):
            input_array = transformed_data
        else:
            raise TypeError(
                f"Input data must be np.ndarray or torch.Tensor, got {type(transformed_data)}"
            )

        # Create inverse mappings
        inverse_maps = {}
        for column_name, mapping in self.categorical_map.items():
            if column_name in self.columns_list:
                inverse_maps[column_name] = {v: k for k, v in mapping.items()}

        # Create copy for inverse transformation
        inverse_data = input_array.copy().astype(object)

        # Inverse transform each categorical column
        for col_idx, column_name in self.index_column_map.items():
            if col_idx >= inverse_data.shape[1]:
                continue

            inverse_map = inverse_maps[column_name]
            column_data = input_array[:, col_idx]

            # Vectorized inverse transformation
            inverse_column = np.vectorize(
                lambda x: inverse_map.get(x, f"UNKNOWN_{x}"), otypes=[object]
            )(column_data)

            inverse_data[:, col_idx] = inverse_column

        # Convert back to original type
        if is_tensor:
            # For tensors, return as string tensor (not commonly used)
            return torch.tensor(inverse_data.astype(str), device=device)
        else:
            return inverse_data

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get output feature names for sklearn compatibility.

        Args:
            input_features: Input feature names (optional)

        Returns:
            List of output feature names
        """
        if input_features is None:
            return self.columns_list.copy()
        return input_features.copy()

    def __repr__(self) -> str:
        """String representation of the transformer."""
        return (
            f"CategoricalTransformer(n_features={len(self.columns_list)}, "
            f"default_value={self.default_value}, "
            f"handle_unknown='{self.handle_unknown}')"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


def create_categorical_transformer(
    categorical_map: Dict[str, Dict[str, int]],
    columns_list: List[str],
    default_value: int = 0,
    handle_unknown: str = "default",
) -> CategoricalTransformer:
    """
    Factory function to create CategoricalTransformer instance.

    Args:
        categorical_map: Dictionary mapping feature names to value->index mappings
        columns_list: List of categorical column names to transform
        default_value: Default value for unknown categories
        handle_unknown: Strategy for handling unknown categories

    Returns:
        Configured CategoricalTransformer instance
    """
    return CategoricalTransformer(
        categorical_map=categorical_map,
        columns_list=columns_list,
        default_value=default_value,
        handle_unknown=handle_unknown,
    )
