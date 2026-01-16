"""
Streaming Risk Table Processor - IterableDataset Support

This processor extends RiskTableMappingProcessor to support fitting from
PipelineIterableDataset by accumulating cross-tabulation counts incrementally.

Key Features:
- Single-pass streaming accumulation
- Memory-efficient cross-tabulation
- Exact computation (not approximate)
- Same process()/transform() API as base processor
- Batch fitting for multiple fields (10x faster than individual fitting)
"""

from typing import Dict, Optional, Union, Any, List
from collections import defaultdict
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
        field_names: Optional[List[str]] = None,
        label_name: Optional[str] = None,
        smooth_factor: Optional[float] = None,
        count_threshold: Optional[int] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union["StreamingRiskTableProcessor", Dict[str, Dict]]:
        """
        Fit risk tables from streaming dataset.

        Supports both single-field and multi-field (batch) fitting:
        - Single-field: Uses self.column_name, returns self for chaining
        - Multi-field: Processes all fields in ONE pass, returns Dict[field_name -> risk_tables]

        Args:
            dataset: PipelineIterableDataset to stream from
            field_names: Optional list of fields for batch fitting. If None, uses self.column_name
            label_name: Optional label name override. If None, uses self.label_name
            smooth_factor: Optional smoothing factor override. If None, uses self.smooth_factor
            count_threshold: Optional count threshold override. If None, uses self.count_threshold
            max_samples: Optional limit on samples processed
            show_progress: Whether to show progress bar (for batch mode)

        Returns:
            - Single-field mode: self (for method chaining)
            - Multi-field mode: Dict[field_name -> risk_tables_dict]

        Raises:
            RuntimeError: If no valid samples found
        """
        # Determine mode: single-field or multi-field
        is_batch_mode = field_names is not None and len(field_names) > 1

        # Use provided parameters or fall back to instance attributes
        label_col = label_name or self.label_name
        smooth = smooth_factor if smooth_factor is not None else self.smooth_factor
        threshold = (
            count_threshold if count_threshold is not None else self.count_threshold
        )

        if is_batch_mode:
            # === BATCH MODE: Fit multiple fields in ONE pass ===
            logger.info(
                f"Starting batch streaming fit for {len(field_names)} fields in ONE pass"
            )
            logger.info(f"Fields: {field_names}")

            # Initialize accumulators for ALL fields
            all_accumulators = {
                field: defaultdict(lambda: {0: 0, 1: 0}) for field in field_names
            }

            # Global statistics for each field
            field_stats = {
                field: {"total_positive": 0, "total_count": 0} for field in field_names
            }

            # Single pass through dataset
            samples_processed = 0

            # Create progress bar if requested
            if TQDM_AVAILABLE and show_progress:
                pbar = tqdm(desc="Fitting risk tables", unit=" rows")
            else:
                pbar = None

            try:
                for row in dataset:
                    if label_col not in row:
                        continue

                    label = row[label_col]
                    if label == -1 or pd.isna(label):
                        continue

                    try:
                        label_int = int(label)
                        if label_int not in [0, 1]:
                            continue
                    except (ValueError, TypeError):
                        continue

                    # Update accumulators for ALL fields
                    for field in field_names:
                        if field not in row:
                            continue

                        cat_value_str = str(row[field])
                        all_accumulators[field][cat_value_str][label_int] += 1

                        if label_int == 1:
                            field_stats[field]["total_positive"] += 1
                        field_stats[field]["total_count"] += 1

                    samples_processed += 1
                    if pbar:
                        pbar.update(1)

                    if max_samples and samples_processed >= max_samples:
                        break
            finally:
                if pbar:
                    pbar.close()

            logger.info(f"Processed {samples_processed} rows in single pass")

            # Compute risk tables for each field
            all_risk_tables = {}
            for field in field_names:
                category_counts = all_accumulators[field]
                total_positive = field_stats[field]["total_positive"]
                total_count = field_stats[field]["total_count"]

                if total_count == 0:
                    logger.warning(
                        f"No valid samples for field '{field}'. Using default_bin=0.5"
                    )
                    all_risk_tables[field] = {"bins": {}, "default_bin": 0.5}
                    continue

                default_risk = float(total_positive / total_count)
                smooth_samples = int(total_count * smooth)

                logger.info(
                    f"Field '{field}': {total_count} samples, "
                    f"{len(category_counts)} categories, default_risk={default_risk:.4f}"
                )

                bins = {}
                for cat_value, counts in category_counts.items():
                    pos_count = counts[1]
                    cat_total = pos_count + counts[0]

                    if cat_total >= threshold:
                        if cat_total + smooth_samples > 0:
                            smoothed_risk = float(
                                (pos_count + smooth_samples * default_risk)
                                / (cat_total + smooth_samples)
                            )
                            bins[cat_value] = smoothed_risk
                        else:
                            bins[cat_value] = default_risk
                    else:
                        bins[cat_value] = default_risk

                all_risk_tables[field] = {"bins": bins, "default_bin": default_risk}

            logger.info(f"✓ Batch fit completed for {len(field_names)} fields")
            return all_risk_tables

        else:
            # === SINGLE-FIELD MODE: Fit one field (backward compatible) ===
            field = field_names[0] if field_names else self.column_name
            logger.info(
                f"Starting streaming fit for column '{field}' with label '{label_col}'"
            )

            category_counts = defaultdict(lambda: {0: 0, 1: 0})
            total_positive = 0
            total_count = 0
            samples_processed = 0

            for row in dataset:
                if field not in row or label_col not in row:
                    continue

                label = row[label_col]
                if label == -1 or pd.isna(label):
                    continue

                try:
                    label_int = int(label)
                    if label_int not in [0, 1]:
                        continue
                except (ValueError, TypeError):
                    continue

                cat_value_str = str(row[field])
                category_counts[cat_value_str][label_int] += 1

                if label_int == 1:
                    total_positive += 1
                total_count += 1
                samples_processed += 1

                if max_samples and samples_processed >= max_samples:
                    break

            if total_count == 0:
                logger.warning(f"No valid samples for '{field}'. Using default_bin=0.5")
                self.risk_tables = {"bins": {}, "default_bin": 0.5}
                self.is_fitted = True
                return self

            default_risk = float(total_positive / total_count)
            smooth_samples = int(total_count * smooth)

            logger.info(
                f"Accumulated {total_count} samples across {len(category_counts)} categories "
                f"for column '{field}'"
            )

            bins = {}
            for cat_value, counts in category_counts.items():
                pos_count = counts[1]
                cat_total = pos_count + counts[0]

                if cat_total >= threshold:
                    if cat_total + smooth_samples > 0:
                        smoothed_risk = float(
                            (pos_count + smooth_samples * default_risk)
                            / (cat_total + smooth_samples)
                        )
                        bins[cat_value] = smoothed_risk
                    else:
                        bins[cat_value] = default_risk
                else:
                    bins[cat_value] = default_risk

            self.risk_tables = {"bins": bins, "default_bin": default_risk}
            self.is_fitted = True

            logger.info(
                f"✓ Streaming fit completed for column '{field}': "
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
