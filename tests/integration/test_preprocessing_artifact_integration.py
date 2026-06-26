"""
Integration Tests for Phase 5: Bidirectional Artifact Flow

Tests verify:
1. Script → Processor workflow (training to inference)
2. Processor → Script workflow (experimentation to production)
3. Artifact format compatibility
4. End-to-end integration
"""

import pytest
import pandas as pd
import numpy as np
import pickle as pkl
import json
import tempfile
import subprocess
import sys
from pathlib import Path

from cursus.processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)
from cursus.processing.categorical.risk_table_processor import (
    RiskTableMappingProcessor,
)


class TestScriptToProcessorWorkflow:
    """Test Script → Processor workflow (Training → Inference)."""

    def test_load_numerical_imputation_artifacts(self):
        """Test loading missing_value_imputation.py script artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate script output (missing_value_imputation.py format)
            script_output = {
                "age": 30.5,
                "income": 55000.0,
                "credit_score": 680.0,
                "loan_amount": 25000.0,
            }

            # Save in script format
            impute_dict_file = Path(tmpdir) / "impute_dict.pkl"
            with open(impute_dict_file, "wb") as f:
                pkl.dump(script_output, f)

            # Load with processor factory method
            processors = NumericalVariableImputationProcessor.from_script_artifacts(
                tmpdir
            )

            # Verify all processors created correctly
            assert len(processors) == 4
            assert set(processors.keys()) == {
                "age",
                "income",
                "credit_score",
                "loan_amount",
            }

            # Verify each processor is fitted with correct value
            for col, expected_value in script_output.items():
                proc = processors[col]
                assert proc.is_fitted
                assert proc.column_name == col
                assert proc.get_imputation_value() == expected_value

            # Verify processors can process values
            assert processors["age"].process(None) == 30.5
            assert processors["age"].process(25.0) == 25.0
            assert processors["income"].process(np.nan) == 55000.0

    def test_load_risk_table_artifacts(self):
        """Test loading risk_table_mapping.py script artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate script output (risk_table_mapping.py format)
            script_output = {
                "payment_method": {
                    "bins": {"CC": 0.15, "DC": 0.08, "ACH": 0.03},
                    "default_bin": 0.10,
                    "varName": "payment_method",
                    "type": "categorical",
                    "mode": "categorical",
                },
                "category": {
                    "bins": {"A": 0.25, "B": 0.12, "C": 0.05},
                    "default_bin": 0.18,
                    "varName": "category",
                    "type": "categorical",
                    "mode": "categorical",
                },
            }

            # Save in script format
            risk_table_file = Path(tmpdir) / "risk_table_map.pkl"
            with open(risk_table_file, "wb") as f:
                pkl.dump(script_output, f)

            # Load with processors (manual creation for now, could add factory later)
            with open(risk_table_file, "rb") as f:
                risk_tables = pkl.load(f)

            processors = {}
            for var_name, risk_table_data in risk_tables.items():
                # Extract core structure (ignore extra metadata)
                risk_tables_core = {
                    "bins": risk_table_data["bins"],
                    "default_bin": risk_table_data["default_bin"],
                }

                proc = RiskTableMappingProcessor(
                    column_name=var_name,
                    label_name="target",
                    risk_tables=risk_tables_core,
                )
                processors[var_name] = proc

            # Verify all processors created correctly
            assert len(processors) == 2
            assert set(processors.keys()) == {"payment_method", "category"}

            # Verify processors can process values
            assert processors["payment_method"].process("CC") == 0.15
            assert processors["payment_method"].process("Unknown") == 0.10  # default
            assert processors["category"].process("A") == 0.25

    def test_complete_training_to_inference_pipeline(self):
        """Test complete workflow from training artifacts to inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Simulate training phase (script outputs)
            impute_dict = {"age": 32.5, "income": 55000.0, "score": 0.75}

            risk_tables = {
                "payment_method": {
                    "bins": {"CC": 0.15, "DC": 0.08},
                    "default_bin": 0.10,
                    "varName": "payment_method",
                    "type": "categorical",
                    "mode": "categorical",
                }
            }

            # Save artifacts
            (Path(tmpdir) / "impute_dict.pkl").write_bytes(pkl.dumps(impute_dict))
            (Path(tmpdir) / "risk_table_map.pkl").write_bytes(pkl.dumps(risk_tables))

            # Step 2: Inference phase - load all processors
            num_processors = NumericalVariableImputationProcessor.from_script_artifacts(
                tmpdir
            )

            with open(Path(tmpdir) / "risk_table_map.pkl", "rb") as f:
                risk_data = pkl.load(f)

            cat_processors = {}
            for var_name, risk_table_data in risk_data.items():
                risk_tables_core = {
                    "bins": risk_table_data["bins"],
                    "default_bin": risk_table_data["default_bin"],
                }
                proc = RiskTableMappingProcessor(
                    column_name=var_name,
                    label_name="target",
                    risk_tables=risk_tables_core,
                )
                cat_processors[var_name] = proc

            # Step 3: Process a record (simulating real-time inference)
            record = {
                "age": None,  # Will be imputed
                "income": 60000.0,  # Not missing
                "score": np.nan,  # Will be imputed
                "payment_method": "CC",  # Will be risk-mapped
            }

            # Apply numerical imputation
            for col in ["age", "income", "score"]:
                if col in num_processors:
                    record[col] = num_processors[col].process(record[col])

            # Apply risk mapping
            for col in ["payment_method"]:
                if col in cat_processors:
                    record[col] = cat_processors[col].process(record[col])

            # Verify processed record
            assert record["age"] == 32.5  # Imputed
            assert record["income"] == 60000.0  # Unchanged
            assert record["score"] == 0.75  # Imputed
            assert record["payment_method"] == 0.15  # Risk-mapped

    def test_handles_missing_artifact_files_gracefully(self):
        """Test graceful error handling when artifact files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No files created - should raise appropriate errors

            with pytest.raises(FileNotFoundError, match="impute_dict.pkl not found"):
                NumericalVariableImputationProcessor.from_script_artifacts(tmpdir)


class TestProcessorToScriptWorkflow:
    """Test Processor → Script workflow (Experimentation → Production)."""

    def test_save_numerical_imputation_in_script_format(self):
        """Test saving processor artifacts in script-compatible format."""
        # Fit processors (simulating experimentation phase)
        df = pd.DataFrame(
            {
                "age": [25, 30, None, 40, 35],
                "income": [50000, None, 60000, 70000, 55000],
                "score": [0.8, 0.6, 0.9, None, 0.7],
            }
        )

        processors = {}
        for col in ["age", "income", "score"]:
            proc = NumericalVariableImputationProcessor(
                column_name=col, strategy="mean"
            )
            proc.fit(df[col])
            processors[col] = proc

        # Save in script-compatible format
        impute_dict = {
            col: proc.get_imputation_value() for col, proc in processors.items()
        }

        # Verify format matches script output
        assert isinstance(impute_dict, dict)
        assert set(impute_dict.keys()) == {"age", "income", "score"}
        assert all(isinstance(v, float) for v in impute_dict.values())

        # Verify values are correct
        assert impute_dict["age"] == 32.5  # mean of [25, 30, 40, 35]
        assert impute_dict["income"] == pytest.approx(58750.0)  # mean of non-null
        assert impute_dict["score"] == pytest.approx(0.75)  # mean of non-null

    def test_save_and_reload_in_script_format(self):
        """Test roundtrip: Processor → Script Format → Processor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Phase 1: Fit processors
            df = pd.DataFrame(
                {"age": [10, 20, 30, 40], "income": [1000, 2000, 3000, 4000]}
            )

            processors_original = {}
            for col in ["age", "income"]:
                proc = NumericalVariableImputationProcessor(
                    column_name=col, strategy="mean"
                )
                proc.fit(df[col])
                processors_original[col] = proc

            # Phase 2: Save in script format
            impute_dict = {
                col: proc.get_imputation_value()
                for col, proc in processors_original.items()
            }

            impute_dict_file = Path(tmpdir) / "impute_dict.pkl"
            with open(impute_dict_file, "wb") as f:
                pkl.dump(impute_dict, f)

            # Phase 3: Load as if in script
            with open(impute_dict_file, "rb") as f:
                loaded_dict = pkl.load(f)

            # Verify script can use the artifacts
            assert loaded_dict["age"] == 25.0
            assert loaded_dict["income"] == 2500.0

            # Phase 4: Create new processors from loaded dict
            processors_reloaded = (
                NumericalVariableImputationProcessor.from_imputation_dict(loaded_dict)
            )

            # Verify reloaded processors match original
            for col in ["age", "income"]:
                assert (
                    processors_reloaded[col].get_imputation_value()
                    == processors_original[col].get_imputation_value()
                )

    def test_processor_artifacts_usable_by_script_logic(self):
        """Test that processor artifacts can be used by script-like logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save processors
            processors = {
                "age": NumericalVariableImputationProcessor(
                    "age", imputation_value=30.0
                ),
                "income": NumericalVariableImputationProcessor(
                    "income", imputation_value=50000.0
                ),
            }

            # Save in script format
            impute_dict = {
                col: proc.get_imputation_value() for col, proc in processors.items()
            }

            with open(Path(tmpdir) / "impute_dict.pkl", "wb") as f:
                pkl.dump(impute_dict, f)

            # Simulate script loading and using the artifacts
            with open(Path(tmpdir) / "impute_dict.pkl", "rb") as f:
                loaded_dict = pkl.load(f)

            # Script would apply imputation like this:
            df = pd.DataFrame({"age": [25, None, 35], "income": [45000, 55000, None]})

            for col, impute_value in loaded_dict.items():
                df[col] = df[col].fillna(impute_value)

            # Verify script logic works
            assert df["age"].tolist() == [25, 30.0, 35]
            assert df["income"].tolist() == [45000, 55000, 50000.0]


class TestArtifactFormatCompatibility:
    """Test artifact format compatibility between scripts and processors."""

    def test_impute_dict_format_identical(self):
        """Test that impute_dict format is identical between script and processor."""
        df = pd.DataFrame({"age": [10, 20, 30, 40], "income": [1000, 2000, 3000, 4000]})

        # Create via processor
        processors = {}
        for col in ["age", "income"]:
            proc = NumericalVariableImputationProcessor(
                column_name=col, strategy="mean"
            )
            proc.fit(df[col])
            processors[col] = proc

        processor_dict = {
            col: proc.get_imputation_value() for col, proc in processors.items()
        }

        # Simulate script output (same logic)
        script_dict = {"age": df["age"].mean(), "income": df["income"].mean()}

        # Verify formats match
        assert set(processor_dict.keys()) == set(script_dict.keys())
        for col in processor_dict.keys():
            assert processor_dict[col] == script_dict[col]

    def test_risk_table_format_compatibility(self):
        """Test that risk table format is compatible between script and processor."""
        # Script format (from risk_table_mapping.py)
        script_format = {
            "payment_method": {
                "bins": {"CC": 0.15, "DC": 0.08},
                "default_bin": 0.10,
                "varName": "payment_method",  # Extra metadata
                "type": "categorical",  # Extra metadata
                "mode": "categorical",  # Extra metadata
            }
        }

        # Processor format (core structure only)
        processor_format = {"bins": {"CC": 0.15, "DC": 0.08}, "default_bin": 0.10}

        # Verify processor can extract core from script format
        script_data = script_format["payment_method"]
        extracted_core = {
            "bins": script_data["bins"],
            "default_bin": script_data["default_bin"],
        }

        assert extracted_core == processor_format

        # Verify processor works with extracted core
        proc = RiskTableMappingProcessor(
            column_name="payment_method",
            label_name="target",
            risk_tables=extracted_core,
        )

        assert proc.process("CC") == 0.15
        assert proc.process("DC") == 0.08
        assert proc.process("Unknown") == 0.10

    def test_pickle_format_consistency(self):
        """Test that pickle serialization is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with processor
            test_dict = {"age": 30.0, "income": 50000.0}

            proc_file = Path(tmpdir) / "processor_output.pkl"
            with open(proc_file, "wb") as f:
                pkl.dump(test_dict, f)

            # Load and verify
            with open(proc_file, "rb") as f:
                loaded = pkl.load(f)

            assert loaded == test_dict
            assert type(loaded["age"]) == float
            assert type(loaded["income"]) == float

    def test_json_sidecar_compatibility(self):
        """Test JSON sidecar files for human readability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = NumericalVariableImputationProcessor(
                column_name="age", imputation_value=30.0, strategy="mean"
            )

            proc.save_imputation_value(tmpdir)

            # Verify JSON file is human-readable
            json_file = Path(tmpdir) / "age_impute_value.json"
            assert json_file.exists()

            with open(json_file, "r") as f:
                json_data = json.load(f)

            assert json_data["column_name"] == "age"
            assert json_data["imputation_value"] == 30.0
            assert json_data["strategy"] == "mean"

            # JSON should be pretty-printed (has indentation)
            json_text = json_file.read_text()
            assert "\n" in json_text  # Multi-line
            assert "  " in json_text  # Indented


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""

    def test_full_training_to_inference_cycle(self):
        """Test complete cycle: Training → Artifacts → Inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ===== TRAINING PHASE =====
            # Create training data
            train_df = pd.DataFrame(
                {
                    "age": [25, 30, None, 40, 35, 28, None, 45],
                    "income": [50000, None, 60000, 70000, 55000, 52000, 58000, None],
                    "score": [0.8, 0.6, 0.9, None, 0.7, 0.75, None, 0.85],
                    "payment_method": [
                        "CC",
                        "DC",
                        "CC",
                        "ACH",
                        "CC",
                        "DC",
                        "ACH",
                        "CC",
                    ],
                    "category": ["A", "B", "A", "C", "B", "A", "C", "B"],
                    "target": [1, 0, 1, 0, 1, 0, 1, 0],
                }
            )

            # Fit numerical imputation
            num_processors = {}
            for col in ["age", "income", "score"]:
                proc = NumericalVariableImputationProcessor(
                    column_name=col, strategy="mean"
                )
                proc.fit(train_df[col])
                num_processors[col] = proc

            # Fit risk tables
            cat_processors = {}
            for col in ["payment_method", "category"]:
                proc = RiskTableMappingProcessor(
                    column_name=col, label_name="target", smooth_factor=0.0
                )
                proc.fit(train_df)
                cat_processors[col] = proc

            # Save in script format
            impute_dict = {
                col: proc.get_imputation_value() for col, proc in num_processors.items()
            }

            risk_tables = {
                col: {
                    "bins": proc.get_risk_tables()["bins"],
                    "default_bin": proc.get_risk_tables()["default_bin"],
                    "varName": col,
                    "type": "categorical",
                    "mode": "categorical",
                }
                for col, proc in cat_processors.items()
            }

            # Save artifacts
            with open(Path(tmpdir) / "impute_dict.pkl", "wb") as f:
                pkl.dump(impute_dict, f)

            with open(Path(tmpdir) / "risk_table_map.pkl", "wb") as f:
                pkl.dump(risk_tables, f)

            # ===== INFERENCE PHASE =====
            # Load processors from artifacts
            loaded_num_procs = (
                NumericalVariableImputationProcessor.from_script_artifacts(tmpdir)
            )

            with open(Path(tmpdir) / "risk_table_map.pkl", "rb") as f:
                loaded_risk_data = pkl.load(f)

            loaded_cat_procs = {}
            for var_name, risk_table_data in loaded_risk_data.items():
                risk_tables_core = {
                    "bins": risk_table_data["bins"],
                    "default_bin": risk_table_data["default_bin"],
                }
                proc = RiskTableMappingProcessor(
                    column_name=var_name,
                    label_name="target",
                    risk_tables=risk_tables_core,
                )
                loaded_cat_procs[var_name] = proc

            # Process new records
            test_records = [
                {
                    "age": None,
                    "income": 65000.0,
                    "score": np.nan,
                    "payment_method": "CC",
                    "category": "A",
                },
                {
                    "age": 33.0,
                    "income": None,
                    "score": 0.82,
                    "payment_method": "DC",
                    "category": "B",
                },
            ]

            for record in test_records:
                # Apply numerical imputation
                for col in ["age", "income", "score"]:
                    if col in loaded_num_procs:
                        record[col] = loaded_num_procs[col].process(record[col])

                # Apply risk mapping
                for col in ["payment_method", "category"]:
                    if col in loaded_cat_procs:
                        record[col] = loaded_cat_procs[col].process(record[col])

            # Verify all values are processed
            for record in test_records:
                for col in ["age", "income", "score", "payment_method", "category"]:
                    assert record[col] is not None
                    assert not pd.isna(record[col])
                    # Numerical columns should be numeric
                    if col in ["age", "income", "score"]:
                        assert isinstance(record[col], (int, float))
                    # Risk-mapped columns should be float
                    elif col in ["payment_method", "category"]:
                        assert isinstance(record[col], float)

    def test_experimentation_to_production_cycle(self):
        """Test cycle: Notebook Experimentation → Artifacts → Production Script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ===== EXPERIMENTATION PHASE (Notebook) =====
            # Small sample data for experimentation
            sample_df = pd.DataFrame(
                {
                    "age": [25, 30, 35],
                    "income": [50000, 60000, 70000],
                    "score": [0.7, 0.8, 0.9],
                }
            )

            # Fit processors
            processors = {}
            for col in sample_df.columns:
                proc = NumericalVariableImputationProcessor(
                    column_name=col, strategy="mean"
                )
                proc.fit(sample_df[col])
                processors[col] = proc

            # Save for production
            impute_dict = {
                col: proc.get_imputation_value() for col, proc in processors.items()
            }

            with open(Path(tmpdir) / "impute_dict.pkl", "wb") as f:
                pkl.dump(impute_dict, f)

            # ===== PRODUCTION PHASE (Script) =====
            # Load artifacts (simulating script loading)
            with open(Path(tmpdir) / "impute_dict.pkl", "rb") as f:
                production_dict = pkl.load(f)

            # Apply to production data (simulating script logic)
            production_df = pd.DataFrame(
                {
                    "age": [None, 28, None, 42],
                    "income": [55000, None, 65000, None],
                    "score": [None, 0.75, None, 0.88],
                }
            )

            for col, impute_value in production_dict.items():
                production_df[col] = production_df[col].fillna(impute_value)

            # Verify no missing values
            assert production_df.isna().sum().sum() == 0

            # Verify imputation values are from experimentation
            assert production_df.loc[0, "age"] == 30.0  # Mean from sample
            assert production_df.loc[1, "income"] == 60000.0  # Mean from sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
