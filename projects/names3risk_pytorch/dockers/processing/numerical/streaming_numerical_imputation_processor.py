"""
Streaming Numerical Imputation Processor - IterableDataset Support

This processor extends NumericalVariableImputationProcessor to support
fitting from PipelineIterableDataset by accumulating statistics incrementally.

Key Features:
- Single-pass streaming accumulation
- Memory-efficient (no full dataset loading)
- Online mean/median/mode computation
- Same process()/transform() API as base processor
- Batch fitting for multiple fields (10x faster than individual fitting)
"""

from typing import Union, Optional, Any, List, Dict
import pandas as pd
import numpy as np
import logging
from torch.utils.data import IterableDataset

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - progress bars will be disabled")

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
        field_names: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        max_samples: Optional[int] = None,
        median_sample_limit: int = 100000,
        show_progress: bool = False,
    ) -> Union["StreamingNumericalImputationProcessor", Dict[str, float]]:
        """
        Fit imputation values from streaming dataset.

        Supports both single-field and multi-field (batch) fitting:
        - Single-field: Uses self.column_name, returns self for chaining
        - Multi-field: Processes all fields in ONE pass, returns Dict[field_name -> imputation_value]

        Args:
            dataset: PipelineIterableDataset to stream from
            field_names: Optional list of fields for batch fitting. If None, uses self.column_name
            strategy: Optional strategy override ('mean', 'median', 'mode'). If None, uses self.strategy
            max_samples: Optional limit on samples processed (for early stopping)
            median_sample_limit: Max samples to keep for median/mode computation
            show_progress: Whether to show progress bar (for batch mode)

        Returns:
            - Single-field mode: self (for method chaining)
            - Multi-field mode: Dict[field_name -> imputation_value]

        Raises:
            ValueError: If strategy is unknown
        """
        # Determine mode: single-field or multi-field
        is_batch_mode = field_names is not None and len(field_names) > 1

        # Use provided parameters or fall back to instance attributes
        strat = strategy or self.strategy

        if is_batch_mode:
            # === BATCH MODE: Fit multiple fields in ONE pass ===
            logger.info(
                f"Starting batch streaming fit for {len(field_names)} fields in ONE pass"
            )
            logger.info(f"Fields: {field_names}")
            logger.info(f"Strategy: {strat}")

            # Initialize accumulators for ALL fields
            field_accumulators = {}
            for field in field_names:
                if strat == "mean":
                    field_accumulators[field] = {"sum": 0.0, "count": 0}
                elif strat in ["median", "mode"]:
                    field_accumulators[field] = {"values": [], "count": 0}
                else:
                    raise ValueError(f"Unknown strategy: {strat}")

            # Single pass through dataset
            samples_processed = 0

            # Create progress bar if requested
            if TQDM_AVAILABLE and show_progress:
                pbar = tqdm(desc="Fitting imputation values", unit=" rows")
            else:
                pbar = None

            try:
                for row in dataset:
                    # Update accumulators for ALL fields
                    for field in field_names:
                        if field not in row:
                            continue

                        value = row[field]
                        if pd.isna(value):
                            continue

                        try:
                            numeric_value = float(value)
                        except (ValueError, TypeError):
                            continue

                        # Accumulate based on strategy
                        if strat == "mean":
                            field_accumulators[field]["sum"] += numeric_value
                            field_accumulators[field]["count"] += 1
                        elif strat in ["median", "mode"]:
                            if (
                                len(field_accumulators[field]["values"])
                                < median_sample_limit
                            ):
                                field_accumulators[field]["values"].append(
                                    numeric_value
                                )
                            field_accumulators[field]["count"] += 1

                    samples_processed += 1
                    if pbar:
                        pbar.update(1)

                    if max_samples and samples_processed >= max_samples:
                        break
            finally:
                if pbar:
                    pbar.close()

            logger.info(f"Processed {samples_processed} rows in single pass")

            # Compute imputation values for each field
            imputation_dict = {}
            for field in field_names:
                acc = field_accumulators[field]

                if strat == "mean":
                    if acc["count"] == 0:
                        logger.warning(
                            f"No valid samples for field '{field}', using default 0.0"
                        )
                        imputation_dict[field] = 0.0
                    else:
                        imputation_dict[field] = float(acc["sum"] / acc["count"])
                        logger.info(
                            f"Field '{field}': mean={imputation_dict[field]:.4f} from {acc['count']} samples"
                        )

                elif strat == "median":
                    if acc["count"] == 0:
                        logger.warning(
                            f"No valid samples for field '{field}', using default 0.0"
                        )
                        imputation_dict[field] = 0.0
                    else:
                        imputation_dict[field] = float(
                            pd.Series(acc["values"]).median()
                        )
                        approx = (
                            " (approx)" if acc["count"] > len(acc["values"]) else ""
                        )
                        logger.info(
                            f"Field '{field}': median={imputation_dict[field]:.4f}{approx} from {acc['count']} samples"
                        )

                elif strat == "mode":
                    if acc["count"] == 0:
                        logger.warning(
                            f"No valid samples for field '{field}', using default 0.0"
                        )
                        imputation_dict[field] = 0.0
                    else:
                        mode_series = pd.Series(acc["values"]).mode()
                        imputation_dict[field] = (
                            float(mode_series[0]) if len(mode_series) > 0 else 0.0
                        )
                        approx = (
                            " (approx)" if acc["count"] > len(acc["values"]) else ""
                        )
                        logger.info(
                            f"Field '{field}': mode={imputation_dict[field]:.4f}{approx} from {acc['count']} samples"
                        )

            logger.info(f"✓ Batch fit completed for {len(field_names)} fields")
            return imputation_dict

        else:
            # === SINGLE-FIELD MODE: Fit one field (backward compatible) ===
            field = field_names[0] if field_names else self.column_name
            logger.info(
                f"Starting streaming fit for column '{field}' with strategy '{strat}'"
            )

            # Initialize accumulators
            if strat == "mean":
                sum_val = 0.0
                count = 0
            elif strat in ["median", "mode"]:
                values_list = []
                count = 0
            else:
                raise ValueError(f"Unknown strategy: {strat}")

            # Single-pass accumulation
            samples_processed = 0
            for row in dataset:
                if field not in row:
                    continue

                value = row[field]
                if pd.isna(value):
                    continue

                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    continue

                if strat == "mean":
                    sum_val += numeric_value
                    count += 1
                elif strat in ["median", "mode"]:
                    if len(values_list) < median_sample_limit:
                        values_list.append(numeric_value)
                    count += 1

                samples_processed += 1
                if max_samples and samples_processed >= max_samples:
                    break

            # Compute final statistic
            if strat == "mean":
                if count == 0:
                    logger.warning(f"No valid samples for '{field}', using default 0.0")
                    self.imputation_value = 0.0
                else:
                    self.imputation_value = float(sum_val / count)
                    logger.info(
                        f"Computed mean={self.imputation_value:.4f} from {count} samples"
                    )

            elif strat == "median":
                if count == 0:
                    logger.warning(f"No valid samples for '{field}', using default 0.0")
                    self.imputation_value = 0.0
                else:
                    self.imputation_value = float(pd.Series(values_list).median())
                    approx = " (approx)" if count > len(values_list) else ""
                    logger.info(
                        f"Computed median={self.imputation_value:.4f}{approx} from {count} samples"
                    )

            elif strat == "mode":
                if count == 0:
                    logger.warning(f"No valid samples for '{field}', using default 0.0")
                    self.imputation_value = 0.0
                else:
                    mode_series = pd.Series(values_list).mode()
                    self.imputation_value = (
                        float(mode_series[0]) if len(mode_series) > 0 else 0.0
                    )
                    approx = " (approx)" if count > len(values_list) else ""
                    logger.info(
                        f"Computed mode={self.imputation_value:.4f}{approx} from {count} samples"
                    )

            self.is_fitted = True
            logger.info(f"✓ Streaming fit completed for column '{field}'")
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
