"""
Label Ruleset Execution Script

Applies validated rulesets to processed data to generate classification labels.
Supports train/val/test splits and provides comprehensive execution statistics.

Key Features:
- Field availability validation at execution time
- Priority-based rule evaluation (first match wins)
- Comprehensive statistics tracking
- Fail-safe error handling
- Support for multiple job types (training, validation, testing, calibration)

Usage:
    python label_ruleset_execution.py --job-type training
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RulesetFieldValidator:
    """Validates field availability in actual data at execution time."""

    def validate_fields(self, ruleset: dict, data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates all field references exist in actual data.

        This is an EXECUTION-TIME validator that checks:
        - All required fields exist in DataFrame
        - All fields used in rules exist in DataFrame
        - Field null percentages (data quality check)

        Args:
            ruleset: Validated ruleset configuration
            data_df: Actual DataFrame to check

        Returns:
            Dictionary with validation results:
            - valid: bool
            - missing_fields: List[str]
            - warnings: List[str]
        """
        result = {"valid": True, "missing_fields": [], "warnings": []}

        field_config = ruleset.get("field_config", {})
        required_fields = set(field_config.get("required_fields", []))

        rules = ruleset.get("ruleset", [])

        # Extract all field references from rules
        used_fields = set()
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            fields = self._extract_fields_from_conditions(rule.get("conditions", {}))
            used_fields.update(fields)

        # Check field availability in data
        available_fields = set(data_df.columns)

        # Check required fields exist
        missing_required = required_fields - available_fields
        if missing_required:
            result["valid"] = False
            result["missing_fields"].extend(list(missing_required))
            logger.error(f"Required fields missing in data: {missing_required}")

        # Check used fields exist
        missing_used = used_fields - available_fields
        if missing_used:
            result["valid"] = False
            result["missing_fields"].extend(list(missing_used))
            logger.error(f"Fields used in rules but not in data: {missing_used}")

        # Check for high null percentages
        for field in used_fields & available_fields:
            null_pct = data_df[field].isnull().sum() / len(data_df)
            if null_pct > 0.5:
                result["warnings"].append(
                    f"Field '{field}' has {null_pct:.1%} null values"
                )
                logger.warning(f"Field '{field}' has {null_pct:.1%} null values")

        return result

    def _extract_fields_from_conditions(self, condition: dict) -> List[str]:
        """Recursively extract all field names from a condition."""
        fields = []

        if "all_of" in condition:
            for subcond in condition["all_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "any_of" in condition:
            for subcond in condition["any_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "none_of" in condition:
            for subcond in condition["none_of"]:
                fields.extend(self._extract_fields_from_conditions(subcond))
        elif "field" in condition:
            fields.append(condition["field"])

        return fields


class RuleEngine:
    """
    Evaluates validated rules against data rows to produce labels.
    Extended for multilabel support.

    Optimized for:
    - Batch processing (vectorized where possible)
    - Priority-based evaluation (first match wins)
    - Efficient condition checking
    - Minimal memory footprint
    - Multilabel sparse representation
    """

    def __init__(self, validated_ruleset: dict):
        """
        Initialize rule engine with multilabel support.

        Args:
            validated_ruleset: Pre-validated ruleset from RulesetGenerator
        """
        # Extract configuration
        self.label_config = validated_ruleset["label_config"]
        self.field_config = validated_ruleset["field_config"]
        self.ruleset = validated_ruleset["ruleset"]
        self.metadata = validated_ruleset.get("metadata", {})

        # Filter to enabled rules only (already sorted by priority)
        self.active_rules = [r for r in self.ruleset if r.get("enabled", True)]

        # Determine label type
        self.label_type = self.label_config.get("output_label_type", "binary")

        # Get output column names (normalize to list)
        output_label_name = self.label_config["output_label_name"]
        if isinstance(output_label_name, str):
            # Single-label: string → list of one
            self.output_columns = [output_label_name]
        else:
            # Multilabel: already a list
            self.output_columns = output_label_name

        # Multilabel-specific configuration
        self.sparse_representation = self.label_config.get(
            "sparse_representation", True
        )

        # Common configuration
        self.default_label = self.label_config["default_label"]
        self.evaluation_mode = self.label_config.get("evaluation_mode", "priority")

        # Statistics tracking (per column)
        self.rule_match_counts = {
            col: {r["rule_id"]: 0 for r in self.active_rules}
            for col in self.output_columns
        }
        self.default_label_counts = {col: 0 for col in self.output_columns}
        self.total_evaluated = 0

    def evaluate_row(self, row: pd.Series):
        """
        Evaluate rules against a single row.

        Returns:
            - Single-label mode: int or str (label value)
            - Multilabel mode: Dict[str, Any] (column → value mapping)
        """
        self.total_evaluated += 1

        if self.label_type in ["binary", "multiclass"]:
            return self._evaluate_row_single_label(row)
        else:
            return self._evaluate_row_multilabel(row)

    def _evaluate_row_single_label(self, row: pd.Series):
        """Evaluate rules for single-label mode (backward compatible)."""
        # Single column name (output_columns is list of one)
        output_col = self.output_columns[0]

        for rule in self.active_rules:
            try:
                if self._evaluate_conditions(rule["conditions"], row):
                    rule_id = rule["rule_id"]
                    output_label = rule["output_label"]

                    # output_label should be int/str for single-label
                    self.rule_match_counts[output_col][rule_id] += 1
                    return output_label
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
                continue

        self.default_label_counts[output_col] += 1
        return self.default_label

    def _evaluate_row_multilabel(self, row: pd.Series) -> Dict[str, Any]:
        """Evaluate rules for multilabel mode with sparse representation."""
        import numpy as np

        # Initialize all columns with NaN (sparse) or default (dense)
        if self.sparse_representation:
            result = {col: np.nan for col in self.output_columns}
        else:
            # Handle per-column default_label
            if isinstance(self.default_label, dict):
                result = {col: self.default_label[col] for col in self.output_columns}
            else:
                result = {col: self.default_label for col in self.output_columns}

        # Evaluate rules in priority order
        for rule in self.active_rules:
            try:
                if not self._evaluate_conditions(rule["conditions"], row):
                    continue

                # Rule matched - get output
                output_label = rule.get("output_label")

                if isinstance(output_label, dict):
                    # Multilabel: dict mapping column → value
                    for col, value in output_label.items():
                        if col not in result:
                            continue

                        # Get per-column default for comparison
                        if isinstance(self.default_label, dict):
                            col_default = self.default_label.get(col)
                        else:
                            col_default = self.default_label

                        # Only set if not already set (priority order)
                        if pd.isna(result[col]) or result[col] == col_default:
                            result[col] = value
                            self.rule_match_counts[col][rule["rule_id"]] += 1

            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
                continue

        # Fill remaining NaN columns with default if dense mode
        for col in result:
            if pd.isna(result[col]):
                self.default_label_counts[col] += 1
                if not self.sparse_representation:
                    # Handle per-column default_label
                    if isinstance(self.default_label, dict):
                        result[col] = self.default_label[col]
                    else:
                        result[col] = self.default_label

        return result

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate rules for entire DataFrame.

        Returns:
            DataFrame with label column(s) added
        """
        if self.label_type in ["binary", "multiclass"]:
            # Single column result (backward compatible)
            output_col = self.output_columns[0]
            df[output_col] = df.apply(self.evaluate_row, axis=1)
            return df

        else:
            # Multi-column result (multilabel)
            results = df.apply(self.evaluate_row, axis=1, result_type="expand")

            # Add all label columns to original dataframe
            for col in self.output_columns:
                df[col] = results[col]

            return df

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics with multilabel support."""
        if self.label_type in ["binary", "multiclass"]:
            # Single-label statistics (backward compatible)
            output_col = self.output_columns[0]

            return {
                "total_evaluated": self.total_evaluated,
                "rule_match_counts": self.rule_match_counts[output_col],
                "default_label_count": self.default_label_counts[output_col],
                "rule_match_percentages": {
                    rule_id: (count / self.total_evaluated * 100)
                    if self.total_evaluated > 0
                    else 0
                    for rule_id, count in self.rule_match_counts[output_col].items()
                },
                "default_label_percentage": (
                    self.default_label_counts[output_col] / self.total_evaluated * 100
                    if self.total_evaluated > 0
                    else 0
                ),
            }

        else:
            # Multilabel statistics (per column)
            stats = {
                "label_type": "multilabel",
                "total_evaluated": self.total_evaluated,
                "per_column_statistics": {},
            }

            for col in self.output_columns:
                col_stats = {
                    "rule_match_counts": self.rule_match_counts[col],
                    "default_label_count": self.default_label_counts[col],
                    "rule_match_percentages": {
                        rule_id: (count / self.total_evaluated * 100)
                        if self.total_evaluated > 0
                        else 0
                        for rule_id, count in self.rule_match_counts[col].items()
                    },
                    "default_label_percentage": (
                        self.default_label_counts[col] / self.total_evaluated * 100
                        if self.total_evaluated > 0
                        else 0
                    ),
                }
                stats["per_column_statistics"][col] = col_stats

            return stats

    def _evaluate_conditions(self, conditions: dict, row: pd.Series) -> bool:
        """
        Recursively evaluate nested conditions.

        Args:
            conditions: Condition dictionary with logical operators
            row: DataFrame row as Series

        Returns:
            Boolean indicating whether conditions are satisfied
        """
        # Handle logical operators
        if "all_of" in conditions:
            return all(
                self._evaluate_conditions(cond, row) for cond in conditions["all_of"]
            )

        elif "any_of" in conditions:
            return any(
                self._evaluate_conditions(cond, row) for cond in conditions["any_of"]
            )

        elif "none_of" in conditions:
            return not any(
                self._evaluate_conditions(cond, row) for cond in conditions["none_of"]
            )

        # Handle leaf condition (field comparison)
        else:
            return self._evaluate_leaf_condition(conditions, row)

    def _evaluate_leaf_condition(self, condition: dict, row: pd.Series) -> bool:
        """
        Evaluate a single leaf condition (field comparison).

        Args:
            condition: Single condition with field, operator, value
            row: DataFrame row as Series

        Returns:
            Boolean indicating whether condition is satisfied
        """
        field = condition["field"]
        operator = condition["operator"]
        expected_value = condition["value"]

        # Get actual value from row
        if field not in row.index:
            return False

        actual_value = row[field]

        # Handle null values
        if pd.isna(actual_value):
            if operator == "is_null":
                return True
            elif operator == "is_not_null":
                return False
            else:
                return False  # Null doesn't match comparisons

        # Apply operator
        return self._apply_operator(operator, actual_value, expected_value)

    def _apply_operator(self, operator: str, actual: Any, expected: Any) -> bool:
        """Apply comparison operator."""

        # Comparison operators
        if operator == "equals":
            return actual == expected
        elif operator == "not_equals":
            return actual != expected
        elif operator == ">":
            return float(actual) > float(expected)
        elif operator == ">=":
            return float(actual) >= float(expected)
        elif operator == "<":
            return float(actual) < float(expected)
        elif operator == "<=":
            return float(actual) <= float(expected)

        # Collection operators
        elif operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected

        # String operators
        elif operator == "contains":
            return str(expected) in str(actual)
        elif operator == "not_contains":
            return str(expected) not in str(actual)
        elif operator == "starts_with":
            return str(actual).startswith(str(expected))
        elif operator == "ends_with":
            return str(actual).endswith(str(expected))
        elif operator == "regex_match":
            import re

            return bool(re.search(expected, str(actual)))

        # Null operators
        elif operator == "is_null":
            return False  # Already handled null case
        elif operator == "is_not_null":
            return True  # Already handled null case

        else:
            raise ValueError(f"Unsupported operator: {operator}")


def _detect_file_format(file_path: Path) -> str:
    """
    Detect file format based on extension.

    Args:
        file_path: Path to file

    Returns:
        File format: 'csv', 'tsv', or 'parquet'
    """
    suffix = file_path.suffix.lower()
    if suffix in [".csv", ".csv.gz"]:
        return "csv"
    elif suffix in [".tsv", ".tsv.gz"]:
        return "tsv"
    elif suffix in [".parquet", ".pq"]:
        return "parquet"
    else:
        # Default to CSV
        return "csv"


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Read DataFrame from file, automatically detecting format.

    Args:
        file_path: Path to data file

    Returns:
        DataFrame
    """
    file_format = _detect_file_format(file_path)

    if file_format == "csv":
        return pd.read_csv(file_path)
    elif file_format == "tsv":
        return pd.read_csv(file_path, sep="\t")
    elif file_format == "parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _write_dataframe(df: pd.DataFrame, file_path: Path, file_format: str):
    """
    Write DataFrame to file in specified format.

    Args:
        df: DataFrame to write
        file_path: Output file path
        file_format: Format to write ('csv', 'tsv', 'parquet')
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "csv":
        df.to_csv(file_path, index=False)
    elif file_format == "tsv":
        df.to_csv(file_path, sep="\t", index=False)
    elif file_format == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for ruleset execution.

    Supports multiple file formats: CSV, TSV, Parquet (auto-detected).

    Args:
        input_paths: Dictionary with keys:
            - "validated_ruleset": Path to validated ruleset JSON
            - "input_data": Directory with train/val/test splits
        output_paths: Dictionary with keys:
            - "processed_data": Directory for output with labels
            - "execution_report": Path for execution statistics
            - "rule_match_statistics": Optional path for detailed statistics
        environ_vars: Environment variables
        job_args: Command line arguments (job_type)
        logger: Optional logger function

    Returns:
        Dictionary of processed DataFrames by split name
    """
    log = logger or print

    # 1. Load validated ruleset from directory
    ruleset_dir = Path(input_paths["validated_ruleset"])
    ruleset_path = ruleset_dir / "validated_ruleset.json"
    with open(ruleset_path, "r") as f:
        validated_ruleset = json.load(f)

    log(f"[INFO] Loaded validated ruleset v{validated_ruleset.get('version')}")
    log(f"[INFO] Rules: {validated_ruleset['metadata']['enabled_rules']} enabled")

    # 2. Initialize field validator
    field_validator = RulesetFieldValidator()
    log(f"[INFO] Initialized field validator")

    # 3. Initialize rule engine
    rule_engine = RuleEngine(validated_ruleset)
    log(f"[INFO] Initialized rule engine")

    # 4. Determine splits to process
    input_dir = Path(input_paths["input_data"])
    output_dir = Path(output_paths["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if job_args.job_type == "training":
        splits = ["train", "val", "test"]
    else:
        splits = [job_args.job_type]

    # 5. Process each split
    processed_splits = {}
    split_statistics = {}

    # Get preferred format from environment (optional)
    preferred_format = environ_vars.get("PREFERRED_INPUT_FORMAT", "").lower()
    if preferred_format and preferred_format not in ["csv", "tsv", "parquet"]:
        log(f"[WARNING] Invalid PREFERRED_INPUT_FORMAT '{preferred_format}', ignoring")
        preferred_format = ""

    for split_name in splits:
        log(f"[INFO] Processing {split_name} split...")

        # Load data - try multiple formats
        split_dir = input_dir / split_name
        if not split_dir.exists():
            log(f"[WARNING] Split directory not found: {split_dir}")
            continue

        # Find any data file with supported extensions (no filename assumptions)
        data_file = None
        input_format = None

        # Define supported extensions with their priority order
        supported_extensions = [".csv", ".parquet", ".tsv", ".pq", ".csv.gz", ".tsv.gz"]

        # If preferred format specified, reorder to check it first
        if preferred_format:
            format_extensions = {
                "csv": [".csv", ".csv.gz"],
                "tsv": [".tsv", ".tsv.gz"],
                "parquet": [".parquet", ".pq"],
            }

            preferred_exts = format_extensions.get(preferred_format, [])
            # Put preferred extensions first, then others
            other_exts = [
                ext for ext in supported_extensions if ext not in preferred_exts
            ]
            supported_extensions = preferred_exts + other_exts
            log(
                f"[INFO] Looking for '{preferred_format}' format first for {split_name}"
            )

        # Search for files with supported extensions
        found_files = []
        for ext in supported_extensions:
            matching_files = list(split_dir.glob(f"*{ext}"))
            if matching_files:
                found_files.extend(matching_files)
                # Use first match from this extension
                data_file = matching_files[0]
                input_format = _detect_file_format(data_file)

                if len(matching_files) > 1:
                    log(
                        f"[WARNING] Multiple {ext} files found in {split_dir}, using: {data_file.name}"
                    )
                else:
                    log(
                        f"[INFO] Found data file: {data_file.name} (format: {input_format})"
                    )
                break

        if data_file is None:
            log(
                f"[WARNING] No data file with supported extensions found in {split_dir}"
            )
            log(f"[WARNING] Supported extensions: {', '.join(supported_extensions)}")
            continue

        df = _read_dataframe(data_file)
        log(f"[INFO] Loaded {split_name}: {df.shape} (format: {input_format})")

        # Validate field availability in data
        validation_result = field_validator.validate_fields(validated_ruleset, df)
        if not validation_result["valid"]:
            error_msg = f"Field validation failed for {split_name}: {validation_result['missing_fields']}"
            log(f"[ERROR] {error_msg}")
            if environ_vars.get("FAIL_ON_MISSING_FIELDS", "true").lower() == "true":
                raise ValueError(error_msg)
            else:
                log(f"[WARNING] Skipping {split_name} due to validation failure")
                continue

        # Log warnings if any
        for warning in validation_result.get("warnings", []):
            log(f"[WARNING] {warning}")

        log(f"[INFO] Field validation passed for {split_name}")

        # Apply rules to generate labels
        df = rule_engine.evaluate_batch(df)

        # Compute label distribution (handles both single-label and multilabel)
        label_dist = {}
        for col in rule_engine.output_columns:
            if col in df.columns:
                label_dist[col] = df[col].value_counts().to_dict()

        if rule_engine.label_type in ["binary", "multiclass"]:
            # Single-label: flatten dict
            label_dist = label_dist.get(rule_engine.output_columns[0], {})

        log(f"[INFO] {split_name} label distribution: {label_dist}")

        # Save statistics
        split_statistics[split_name] = {
            "total_rows": len(df),
            "label_distribution": label_dist,
            "execution_stats": rule_engine.get_statistics(),
        }

        # Reset engine statistics for next split
        rule_engine.rule_match_counts = {
            col: {r["rule_id"]: 0 for r in rule_engine.active_rules}
            for col in rule_engine.output_columns
        }
        rule_engine.default_label_counts = {
            col: 0 for col in rule_engine.output_columns
        }
        rule_engine.total_evaluated = 0

        # Save labeled data in same format as input
        output_split_dir = output_dir / split_name
        output_split_dir.mkdir(exist_ok=True)

        # Determine output extension based on input format
        if input_format == "csv":
            output_file = output_split_dir / f"{split_name}_processed_data.csv"
        elif input_format == "tsv":
            output_file = output_split_dir / f"{split_name}_processed_data.tsv"
        elif input_format == "parquet":
            output_file = output_split_dir / f"{split_name}_processed_data.parquet"
        else:
            output_file = output_split_dir / f"{split_name}_processed_data.csv"

        _write_dataframe(df, output_file, input_format)
        log(f"[INFO] Saved {output_file} (format: {input_format})")

        processed_splits[split_name] = df

    # 6. Save execution report
    execution_report = {
        "ruleset_version": validated_ruleset.get("version"),
        "ruleset_timestamp": validated_ruleset.get("generated_timestamp"),
        "execution_timestamp": datetime.now().isoformat(),
        "label_config": validated_ruleset["label_config"],
        "split_statistics": split_statistics,
        "total_rules_evaluated": validated_ruleset["metadata"]["enabled_rules"],
    }

    report_dir = Path(output_paths["execution_report"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "execution_report.json"
    with open(report_path, "w") as f:
        json.dump(execution_report, f, indent=2)
    log(f"[INFO] Saved execution report: {report_path}")

    # 7. Save detailed rule match statistics in execution_report folder
    stats_path = report_dir / "rule_match_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(split_statistics, f, indent=2)
    log(f"[INFO] Saved rule match statistics: {stats_path}")

    log("[INFO] Ruleset execution complete")
    return processed_splits


if __name__ == "__main__":
    import sys
    import traceback
    import os

    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Execute validated rulesets on processed data to generate labels"
        )
        parser.add_argument(
            "--job-type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="Job type: training (all splits), validation, testing, or calibration",
        )

        args = parser.parse_args()

        # Set up paths using container paths
        input_paths = {
            "validated_ruleset": "/opt/ml/processing/input/validated_ruleset",
            "input_data": "/opt/ml/processing/input/data",
        }

        output_paths = {
            "processed_data": "/opt/ml/processing/output/processed_data",
            "execution_report": "/opt/ml/processing/output/execution_report",
        }

        # Get configuration from environment variables
        environ_vars = {
            "FAIL_ON_MISSING_FIELDS": os.environ.get("FAIL_ON_MISSING_FIELDS", "true"),
            "ENABLE_RULE_MATCH_TRACKING": os.environ.get(
                "ENABLE_RULE_MATCH_TRACKING", "true"
            ),
            "ENABLE_PROGRESS_LOGGING": os.environ.get(
                "ENABLE_PROGRESS_LOGGING", "true"
            ),
            "PREFERRED_INPUT_FORMAT": os.environ.get("PREFERRED_INPUT_FORMAT", ""),
        }

        # Configure detailed logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Log key parameters
        logger.info("Starting ruleset execution with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(
            f"  Fail on Missing Fields: {environ_vars['FAIL_ON_MISSING_FIELDS']}"
        )
        logger.info(
            f"  Rule Match Tracking: {environ_vars['ENABLE_RULE_MATCH_TRACKING']}"
        )
        logger.info(f"  Progress Logging: {environ_vars['ENABLE_PROGRESS_LOGGING']}")

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        total_splits = len(result)
        logger.info(
            f"Ruleset execution completed successfully. Processed {total_splits} split(s)"
        )
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error in ruleset execution script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
