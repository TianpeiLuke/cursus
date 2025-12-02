"""
Order Aggregation Module for Temporal Self-Attention Model

This module handles the aggregation and preprocessing of order sequences,
including temporal ordering, sequence padding, and order-level transformations.
"""

import os
import sys
import gc
import time
import glob
import math
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

# Configuration constants
SEP = ";SEP;"
DEFAULT_SEQ_LEN = 51


class OrderAggregator:
    """
    Handles order-level aggregation and sequence processing for TSA model.

    This class manages:
    - Temporal sequence ordering
    - Sequence length normalization (padding/truncation)
    - Order-level feature extraction
    - Time delta computation
    """

    def __init__(
        self,
        seq_len: int = DEFAULT_SEQ_LEN,
        config_path: str = "/opt/ml/processing/input/config",
    ):
        """
        Initialize OrderAggregator with configuration.

        Args:
            seq_len: Target sequence length for padding/truncation
            config_path: Path to configuration files
        """
        self.seq_len = seq_len
        self.config_path = config_path
        self.SEP = SEP

        # Load preprocessing configurations
        self._load_preprocessing_configs()

    def _load_preprocessing_configs(self):
        """Load preprocessing configurations from files."""
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

        # Load default values
        default_value_dict_file = "default_value_dict.json"
        with open(os.path.join(self.config_path, default_value_dict_file), "r") as f:
            self.default_value_dict = json.load(f)

        # Handle dot notation in keys
        for key in list(self.default_value_dict):
            self.default_value_dict[key.replace(".", "__DOT__")] = (
                self.default_value_dict[key]
            )

    def aggregate_order_sequence(
        self,
        input_data: Dict[str, Any],
        seq_cat_otf_vars: List[str],
        seq_cat_vars: List[str],
        seq_num_otf_vars: List[str],
        seq_num_vars: List[str],
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Aggregate order sequence data from input dictionary.

        Args:
            input_data: Dictionary containing order sequence data
            seq_cat_otf_vars: Categorical sequence OTF variable names
            seq_cat_vars: Categorical sequence variable names
            seq_num_otf_vars: Numerical sequence OTF variable names
            seq_num_vars: Numerical sequence variable names

        Returns:
            Tuple of (success_flag, categorical_sequence_matrix, numerical_sequence_matrix)
        """
        try:
            # Check for required orderDate field
            if "orderDate" not in input_data:
                print(f"orderDate missing. Check input. Source: {input_data}")
                return False, None, None

            # Validate input data structure
            if not self._validate_input_data(
                input_data,
                seq_cat_otf_vars,
                seq_cat_vars,
                seq_num_otf_vars,
                seq_num_vars,
            ):
                return False, None, None

            # Check if historical data is available and valid
            no_history_flag = self._check_history_availability(
                input_data, seq_cat_otf_vars
            )

            # Process categorical sequences
            seq_cat_mtx = self._process_categorical_sequences(
                input_data, seq_cat_otf_vars, seq_cat_vars, no_history_flag
            )

            # Process numerical sequences
            seq_num_mtx = self._process_numerical_sequences(
                input_data, seq_num_otf_vars, seq_num_vars, no_history_flag
            )

            return True, seq_cat_mtx, seq_num_mtx

        except Exception as e:
            print(f"Error in order aggregation: {e}")
            return False, None, None

    def _validate_input_data(
        self,
        input_data: Dict[str, Any],
        seq_cat_otf_vars: List[str],
        seq_cat_vars: List[str],
        seq_num_otf_vars: List[str],
        seq_num_vars: List[str],
    ) -> bool:
        """Validate that input data contains all required fields."""
        if not isinstance(input_data, dict):
            return False

        # Check for required sequence variables
        required_vars = (
            seq_cat_otf_vars + seq_cat_vars + seq_num_otf_vars + seq_num_vars
        )
        for var in required_vars:
            if var not in input_data:
                return False

        return True

    def _check_history_availability(
        self, input_data: Dict[str, Any], seq_cat_otf_vars: List[str]
    ) -> bool:
        """Check if historical sequence data is available and valid."""
        # Check if primary sequence field is empty
        primary_field = (
            "payment_risk.bfs_order_cat_seq_by_cid.c_billingaddrlatlongconfidence_seq"
        )
        no_history_flag = input_data.get(primary_field, "") in ["", "My Text String"]

        if not no_history_flag:
            # Validate sequence length consistency
            seq_lengths = []
            for var in seq_cat_otf_vars:
                if var in input_data:
                    seq_len = len(input_data[var].split(self.SEP))
                    seq_lengths.append(seq_len)

            # Check if all sequences have consistent length
            if len(set(seq_lengths)) > 1:
                no_history_flag = True

        return no_history_flag

    def _process_categorical_sequences(
        self,
        input_data: Dict[str, Any],
        seq_cat_otf_vars: List[str],
        seq_cat_vars: List[str],
        no_history_flag: bool,
    ) -> np.ndarray:
        """Process categorical sequence data."""
        # Extract current order categorical data
        seq_cat_vars_lst = self._extract_current_order_data(
            input_data, seq_cat_vars, categorical=True
        )

        if not no_history_flag:
            # Extract historical sequence data
            seq_cat_vars_mtx = self._extract_sequence_data(
                input_data, seq_cat_otf_vars, seq_cat_vars, categorical=True
            )
            # Combine historical and current data
            seq_cat_mtx = np.concatenate(
                [seq_cat_vars_mtx[:, :-2], seq_cat_vars_lst[:, :-2]]
            )
        else:
            # Use only current order data
            seq_cat_mtx = seq_cat_vars_lst[:, :-2]

        # Apply padding to reach target sequence length
        seq_cat_mtx = self._apply_sequence_padding(seq_cat_mtx, no_history_flag)

        return seq_cat_mtx

    def _process_numerical_sequences(
        self,
        input_data: Dict[str, Any],
        seq_num_otf_vars: List[str],
        seq_num_vars: List[str],
        no_history_flag: bool,
    ) -> np.ndarray:
        """Process numerical sequence data."""
        # Extract current order numerical data
        seq_num_vars_lst = self._extract_current_order_data(
            input_data, seq_num_vars, categorical=False
        )

        if not no_history_flag:
            # Extract historical sequence data
            seq_num_vars_mtx = self._extract_sequence_data(
                input_data, seq_num_otf_vars, seq_num_vars, categorical=False
            )
            # Combine historical and current data
            seq_num_mtx = np.concatenate(
                [seq_num_vars_mtx[:, :-1], seq_num_vars_lst[:, :-1]]
            )
        else:
            # Use only current order data
            seq_num_mtx = seq_num_vars_lst[:, :-1]

        # Add keep flag (1 for valid orders, 0 for padding)
        seq_num_mtx = np.concatenate(
            [seq_num_mtx, np.ones((seq_num_mtx.shape[0], 1))], axis=1
        )

        # Apply numerical scaling and transformations
        seq_num_mtx = self._apply_numerical_scaling(seq_num_mtx)

        # Compute time deltas
        seq_num_mtx = self._compute_time_deltas(seq_num_mtx)

        # Apply padding to reach target sequence length
        seq_num_mtx = self._apply_sequence_padding(seq_num_mtx, no_history_flag)

        return seq_num_mtx

    def _extract_current_order_data(
        self, input_data: Dict[str, Any], var_list: List[str], categorical: bool = True
    ) -> np.ndarray:
        """Extract current order data and fill missing values."""
        data_list = []
        for var in var_list:
            value = input_data.get(var, "")
            if value in ["", "My Text String"]:
                value = self.default_value_dict.get(var, "")
            data_list.append(value)

        return np.expand_dims(np.array(data_list), axis=0)

    def _extract_sequence_data(
        self,
        input_data: Dict[str, Any],
        otf_vars: List[str],
        vars: List[str],
        categorical: bool = True,
    ) -> np.ndarray:
        """Extract historical sequence data from OTF variables."""
        sequence_data = []

        for i, otf_var in enumerate(otf_vars):
            var_name = vars[i]
            sequence_str = input_data.get(otf_var, "")

            if sequence_str in ["", "My Text String"]:
                # Use default value if sequence is empty
                default_val = self.default_value_dict.get(var_name, "")
                sequence_values = [default_val]
            else:
                # Split sequence and fill missing values
                sequence_values = []
                for val in sequence_str.split(self.SEP):
                    if val in ["", "My Text String"]:
                        val = self.default_value_dict.get(var_name, "")
                    sequence_values.append(val)

            sequence_data.append(sequence_values)

        # Convert to matrix format (sequence_length x num_features)
        max_len = max(len(seq) for seq in sequence_data)
        matrix = np.zeros((max_len, len(sequence_data)), dtype=object)

        for i, seq in enumerate(sequence_data):
            for j, val in enumerate(seq):
                matrix[j, i] = val

        return matrix

    def _apply_numerical_scaling(self, seq_num_mtx: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to numerical features."""
        seq_num_mtx = seq_num_mtx.astype(float)

        # Apply scaling to all features except the last two (time and keep flag)
        seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(
            self.seq_num_scale_
        ) + np.array(self.seq_num_min_)

        return seq_num_mtx

    def _compute_time_deltas(self, seq_num_mtx: np.ndarray) -> np.ndarray:
        """Compute time deltas relative to the most recent order."""
        # Assuming the second-to-last column contains timestamps
        time_col_idx = -2

        if seq_num_mtx.shape[0] > 0:
            # Compute time differences relative to the last (most recent) order
            latest_time = seq_num_mtx[-1, time_col_idx]
            seq_num_mtx[:, time_col_idx] = latest_time - seq_num_mtx[:, time_col_idx]

            # Cap extremely large time differences
            max_time_diff = 10000000
            seq_num_mtx[:, time_col_idx] = np.clip(
                seq_num_mtx[:, time_col_idx], 0, max_time_diff
            )

        return seq_num_mtx

    def _apply_sequence_padding(
        self, sequence_mtx: np.ndarray, no_history_flag: bool
    ) -> np.ndarray:
        """Apply padding to reach target sequence length."""
        current_len = sequence_mtx.shape[0]

        if no_history_flag:
            # Pad with zeros at the beginning, leaving space for current order
            pad_width = self.seq_len - 1
            padding = ((pad_width, 0), (0, 0))
        else:
            # Pad to reach target length
            pad_width = max(0, self.seq_len - 1 - current_len)
            padding = ((pad_width, 0), (0, 0))

        if pad_width > 0:
            sequence_mtx = np.pad(
                sequence_mtx, padding, mode="constant", constant_values=0
            )

        return sequence_mtx

    def process_order_batch(
        self,
        df: pd.DataFrame,
        seq_cat_otf_vars: List[str],
        seq_cat_vars: List[str],
        seq_num_otf_vars: List[str],
        seq_num_vars: List[str],
        num_workers: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of orders in parallel.

        Args:
            df: DataFrame containing order data
            seq_cat_otf_vars: Categorical sequence OTF variable names
            seq_cat_vars: Categorical sequence variable names
            seq_num_otf_vars: Numerical sequence OTF variable names
            seq_num_vars: Numerical sequence variable names
            num_workers: Number of parallel workers

        Returns:
            Tuple of (categorical_sequences, numerical_sequences)
        """
        # Convert DataFrame rows to JSON strings
        df["json_str"] = df.apply(lambda x: x.to_json(), axis=1)

        print(f"Starting Pool of {num_workers} workers for order aggregation")
        pool = mp.Pool(processes=num_workers)
        results = []

        for row in df.itertuples(index=False):
            input_data = json.loads(row.json_str)
            results.append(
                pool.apply_async(
                    self.aggregate_order_sequence,
                    args=(
                        input_data,
                        seq_cat_otf_vars,
                        seq_cat_vars,
                        seq_num_otf_vars,
                        seq_num_vars,
                    ),
                )
            )

        pool.close()
        pool.join()
        print("Order aggregation pool closed")

        # Collect results
        X_seq_cat_list = []
        X_seq_num_list = []

        for result in results:
            ret, seq_cat_mtx, seq_num_mtx = result.get()
            if ret:
                # Additional validation and filtering
                if np.min(seq_num_mtx[:, -2]) >= 0:  # Valid time deltas
                    X_seq_cat_list.append(seq_cat_mtx)
                    X_seq_num_list.append(seq_num_mtx)

        # Stack results into final arrays
        if X_seq_cat_list:
            X_seq_cat = np.stack(X_seq_cat_list, axis=0)
            X_seq_num = np.stack(X_seq_num_list, axis=0)
        else:
            # Return empty arrays if no valid sequences
            X_seq_cat = np.empty((0, self.seq_len, len(seq_cat_vars) - 2))
            X_seq_num = np.empty(
                (0, self.seq_len, len(seq_num_vars) - 1 + 1)
            )  # +1 for keep flag

        return X_seq_cat, X_seq_num


def create_order_aggregator(
    seq_len: int = DEFAULT_SEQ_LEN, config_path: str = "/opt/ml/processing/input/config"
) -> OrderAggregator:
    """
    Factory function to create OrderAggregator instance.

    Args:
        seq_len: Target sequence length
        config_path: Path to configuration files

    Returns:
        Configured OrderAggregator instance
    """
    return OrderAggregator(seq_len=seq_len, config_path=config_path)


# Utility functions for backward compatibility
def mtx_from_dict_fill_default(
    input_data: Dict[str, Any],
    var_list_otf: List[str],
    var_list: List[str],
    map_dict: Dict[str, Any],
) -> np.ndarray:
    """Legacy function for matrix extraction with default value filling."""
    return np.array(
        [
            [
                map_dict[var_list[i]] if a in ["", "My Text String"] else a
                for a in input_data[var_list_otf[i]].split(SEP)
            ]
            for i in range(len(var_list_otf))
        ]
    ).transpose()


def arr_from_dict_fill_default(
    input_data: Dict[str, Any], var_list: List[str], map_dict: Dict[str, Any]
) -> np.ndarray:
    """Legacy function for array extraction with default value filling."""
    return np.expand_dims(
        np.array(
            [
                map_dict[var_list[i]]
                if input_data[var_list[i]] in ["", "My Text String"]
                else input_data[var_list[i]]
                for i in range(len(var_list))
            ]
        ),
        axis=0,
    )
