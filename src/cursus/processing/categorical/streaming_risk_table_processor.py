"""
Streaming Risk Table Processor - IterableDataset Support

This processor extends RiskTableMappingProcessor to support fitting from
PipelineIterableDataset by accumulating cross-tabulation counts incrementally.

Key Features:
- Single-pass streaming accumulation
- Memory-efficient cross-tabulation
- Exact computation (not approximate)
- Same process()/transform() API as base processor
"""

from typing import Dict, Optional, Union, Any
from collections import defaultdict
import pandas as pd
import numpy as np
import logging
from torch.utils.data import IterableDataset

from .risk_table_processor import RiskTableMappingProcessor

# Setup logger
logger = logging.getLogger(__name__)


class StreamingRiskTableProcessor(RiskTableMappingProcessor):
    """
    Streaming-aware risk table processor.

    Extends RiskTableMappingProcessor to support fitting from
    PipelineIterableDataset by accumulating cross-tabulation counts incrementally.

    Uses online cross-tabulation for exact computation:
    - Maintains category counts for each label value (0, 1)
    - Computes risk ratios with smoothing
    - Memory proportional to number of unique categories (not dataset size)

    Examples:
        >>> # Create streaming processor
        >>> proc = StreamingRiskTableProcessor(
        ...     column_name='customer_type',
        ...     label_name='label',
        ...     smooth_factor=0.1,
        ...     count_threshold=5
        ... )
        >>>
        >>> # Fit from streaming dataset
        >>> proc.fit_streaming(train_iterable_dataset)
        >>>
        >>> # Use in pipeline (same API as base processor)
        >>> dataset.add_pipeline('customer_type', proc)
    """

    def __init__(
        self,
        column_name: str,
        label_name: str,
        smooth_factor: float = 0.0,
        count_threshold: int = 0,
        risk_tables: Optional[Dict] = None,
    ):
        """
        Initialize streaming risk table processor.

        Args:
            column_name: Name of the categorical column
            label_name: Name of the label/target column
            smooth_factor: Smoothing factor (0 to 1)
            count_threshold: Minimum count for category
            risk_tables: Optional pre-computed risk tables
        """
        super().__init__(
            column_name, label_name, smooth_factor, count_threshold, risk_tables
        )
        self.processor_name = "streaming_risk_table_processor"

    def fit_streaming(
        self,
        dataset: IterableDataset,
        max_samples: Optional[int] = None,
    ) -> "StreamingRiskTableProcessor":
        """
        Fit risk tables from streaming dataset.

        Performs single-pass streaming accumulation of cross-tabulation counts.
        This is an EXACT computation (not approximate) since we only track counts.

        Args:
            dataset: PipelineIterableDataset to stream from
            max_samples: Optional limit on samples processed

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If no valid samples found
        """
        logger.info(
            f"Starting streaming fit for column '{self.column_name}' with label '{self.label_name}'"
        )

        # Initialize accumulators for cross-tabulation
        # Format: {category_value: {0: count, 1: count}}
        category_counts = defaultdict(lambda: {0: 0, 1: 0})
        total_positive = 0
        total_count = 0

        # Single-pass accumulation
        samples_processed = 0
        for row in dataset:
            # Check required fields exist
            if self.column_name not in row or self.label_name not in row:
                continue

            cat_value = row[self.column_name]
            label = row[self.label_name]

            # Skip invalid labels (same as base processor fit logic)
            if label == -1 or pd.isna(label):
                continue

            # Convert to string for categorical (consistent with base processor)
            cat_value_str = str(cat_value)

            # Convert label to int
            try:
                label_int = int(label)
                if label_int not in [0, 1]:
                    logger.warning(f"Label value {label_int} not in [0, 1], skipping")
                    continue
            except (ValueError, TypeError):
                logger.warning(f"Could not convert label '{label}' to int, skipping")
                continue

            # Accumulate counts
            category_counts[cat_value_str][label_int] += 1

            if label_int == 1:
                total_positive += 1
            total_count += 1

            samples_processed += 1

            # Early stopping if max_samples specified
            if max_samples and samples_processed >= max_samples:
                logger.info(
                    f"Reached max_samples limit ({max_samples}) for column '{self.column_name}'"
                )
                break

        # Check if we have valid data
        if total_count == 0:
            logger.warning(
                f"No valid samples found for column '{self.column_name}'. "
                "Risk tables will use default_bin=0.5"
            )
            self.risk_tables = {
                "bins": {},
                "default_bin": 0.5,
            }
            self.is_fitted = True
            return self

        # Compute default risk (global positive rate)
        default_risk = float(total_positive / total_count)
        smooth_samples = int(total_count * self.smooth_factor)

        logger.info(
            f"Accumulated {total_count} samples across {len(category_counts)} categories "
            f"for column '{self.column_name}'"
        )
        logger.info(
            f"Default risk: {default_risk:.4f}, Smooth samples: {smooth_samples}"
        )

        # Compute risk for each category with smoothing
        bins = {}
        for cat_value, counts in category_counts.items():
            pos_count = counts[1]
            neg_count = counts[0]
            cat_total = pos_count + neg_count

            # Apply count threshold
            if cat_total >= self.count_threshold:
                # Apply Laplace smoothing
                if cat_total + smooth_samples > 0:
                    smoothed_risk = float(
                        (pos_count + smooth_samples * default_risk)
                        / (cat_total + smooth_samples)
                    )
                    bins[cat_value] = smoothed_risk
                else:
                    bins[cat_value] = default_risk
            else:
                # Below threshold: use default risk
                bins[cat_value] = default_risk

        # Store risk tables
        self.risk_tables = {
            "bins": bins,
            "default_bin": default_risk,
        }

        self.is_fitted = True
        logger.info(
            f"✓ Streaming fit completed for column '{self.column_name}': "
            f"{len(bins)} categories mapped, default_bin={default_risk:.4f}"
        )

        return self

    def fit(
        self, data: Union[pd.DataFrame, IterableDataset]
    ) -> "StreamingRiskTableProcessor":
        """
        Fit risk tables from data.
        
        Automatically detects input type and delegates to appropriate method:
        - IterableDataset: uses fit_streaming()
        - DataFrame: uses parent class fit()
        
        Args:
            data: DataFrame or IterableDataset
        
        Returns:
            self (for method chaining)
        """
        if isinstance(data, IterableDataset):
            return self.fit_streaming(data)
        else:
            return super().fit(data)
