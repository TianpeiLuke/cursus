"""
Streaming Numerical Imputation Processor - IterableDataset Support

This processor extends NumericalVariableImputationProcessor to support
fitting from PipelineIterableDataset by accumulating statistics incrementally.

Key Features:
- Single-pass streaming accumulation
- Memory-efficient (no full dataset loading)
- Online mean/median/mode computation
- Same process()/transform() API as base processor
"""

from typing import Union, Optional, Any
import pandas as pd
import numpy as np
import logging
from torch.utils.data import IterableDataset

from .numerical_imputation_processor import NumericalVariableImputationProcessor

# Setup logger
logger = logging.getLogger(__name__)


class StreamingNumericalImputationProcessor(NumericalVariableImputationProcessor):
    """
    Streaming-aware numerical imputation processor.

    Extends NumericalVariableImputationProcessor to support fitting from
    PipelineIterableDataset by accumulating statistics incrementally.

    Uses online algorithms for efficient single-pass statistics computation:
    - Mean: Running sum and count
    - Median: Approximate using sample collection (memory-limited)
    - Mode: Approximate using sample collection (memory-limited)

    Examples:
        >>> # Create streaming processor
        >>> proc = StreamingNumericalImputationProcessor(
        ...     column_name='age',
        ...     strategy='mean'
        ... )
        >>>
        >>> # Fit from streaming dataset
        >>> proc.fit_streaming(train_iterable_dataset)
        >>>
        >>> # Use in pipeline (same API as base processor)
        >>> dataset.add_pipeline('age', proc)
    """

    def __init__(
        self,
        column_name: str,
        imputation_value: Optional[Union[int, float]] = None,
        strategy: Optional[str] = None,
    ):
        """
        Initialize streaming numerical imputation processor.

        Args:
            column_name: Name of the column to impute
            imputation_value: Pre-computed imputation value (for inference)
            strategy: Strategy for fitting ('mean', 'median', 'mode')
        """
        super().__init__(column_name, imputation_value, strategy)
        self.processor_name = "streaming_numerical_imputation_processor"

    def fit_streaming(
        self,
        dataset: IterableDataset,
        max_samples: Optional[int] = None,
        median_sample_limit: int = 100000,
    ) -> "StreamingNumericalImputationProcessor":
        """
        Fit imputation value from streaming dataset.

        Performs single-pass streaming accumulation of statistics.
        For mean: exact computation using running sum/count.
        For median/mode: approximate using sample collection.

        Args:
            dataset: PipelineIterableDataset to stream from
            max_samples: Optional limit on samples processed (for early stopping)
            median_sample_limit: Max samples to keep for median/mode computation
                                (prevents unbounded memory growth)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If strategy is unknown
            RuntimeError: If no valid samples found
        """
        logger.info(
            f"Starting streaming fit for column '{self.column_name}' with strategy '{self.strategy}'"
        )

        # Initialize accumulators based on strategy
        if self.strategy == "mean":
            sum_val = 0.0
            count = 0
        elif self.strategy in ["median", "mode"]:
            values_list = []
            count = 0
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Single-pass accumulation
        samples_processed = 0
        for row in dataset:
            if self.column_name not in row:
                continue

            value = row[self.column_name]

            # Skip missing values
            if pd.isna(value):
                continue

            # Convert to numeric if needed
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert value '{value}' to numeric for column '{self.column_name}'"
                )
                continue

            # Accumulate statistics
            if self.strategy == "mean":
                sum_val += numeric_value
                count += 1
            elif self.strategy in ["median", "mode"]:
                # For median/mode, collect samples up to limit
                if len(values_list) < median_sample_limit:
                    values_list.append(numeric_value)
                count += 1

            samples_processed += 1

            # Early stopping if max_samples specified
            if max_samples and samples_processed >= max_samples:
                logger.info(
                    f"Reached max_samples limit ({max_samples}) for column '{self.column_name}'"
                )
                break

        # Compute final statistic
        if self.strategy == "mean":
            if count == 0:
                logger.warning(
                    f"No valid samples found for column '{self.column_name}', using default 0.0"
                )
                self.imputation_value = 0.0
            else:
                self.imputation_value = float(sum_val / count)
                logger.info(
                    f"Computed mean={self.imputation_value:.4f} from {count} samples"
                )

        elif self.strategy == "median":
            if count == 0:
                logger.warning(
                    f"No valid samples found for column '{self.column_name}', using default 0.0"
                )
                self.imputation_value = 0.0
            else:
                self.imputation_value = float(pd.Series(values_list).median())
                if count > len(values_list):
                    logger.info(
                        f"Computed approximate median={self.imputation_value:.4f} "
                        f"from {len(values_list)} samples (total: {count})"
                    )
                else:
                    logger.info(
                        f"Computed exact median={self.imputation_value:.4f} from {count} samples"
                    )

        elif self.strategy == "mode":
            if count == 0:
                logger.warning(
                    f"No valid samples found for column '{self.column_name}', using default 0.0"
                )
                self.imputation_value = 0.0
            else:
                mode_series = pd.Series(values_list).mode()
                if len(mode_series) > 0:
                    self.imputation_value = float(mode_series[0])
                else:
                    self.imputation_value = 0.0
                if count > len(values_list):
                    logger.info(
                        f"Computed approximate mode={self.imputation_value:.4f} "
                        f"from {len(values_list)} samples (total: {count})"
                    )
                else:
                    logger.info(
                        f"Computed exact mode={self.imputation_value:.4f} from {count} samples"
                    )

        self.is_fitted = True
        logger.info(f"✓ Streaming fit completed for column '{self.column_name}'")

        return self

    def fit(
        self,
        X: Union[pd.Series, pd.DataFrame, IterableDataset],
        y: Optional[pd.Series] = None,
    ) -> "StreamingNumericalImputationProcessor":
        """
        Fit imputation value from data.

        Automatically detects input type and delegates to appropriate method:
        - IterableDataset: uses fit_streaming()
        - Series/DataFrame: uses parent class fit()

        Args:
            X: Series, DataFrame, or IterableDataset
            y: Ignored (for sklearn compatibility)

        Returns:
            self (for method chaining)
        """
        if isinstance(X, IterableDataset):
            return self.fit_streaming(X)
        else:
            return super().fit(X, y)
