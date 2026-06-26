"""
Tests for NumericalVariableImputationProcessor - Single Column Architecture

Tests verify:
1. Base Processor inheritance and compatibility
2. Single-column processing
3. Pipeline chaining with >> operator
4. Artifact loading from script output
5. Backward compatibility
"""

import pytest
import pandas as pd
import numpy as np
import pickle as pkl
import json
from pathlib import Path
import tempfile
import warnings

from cursus.processing.numerical.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)
from cursus.processing.processors import Processor


class TestNumericalVariableImputationProcessorArchitecture:
    """Test proper inheritance and architecture."""

    def test_extends_base_processor(self):
        """Verify processor extends base Processor class."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )
        assert isinstance(proc, Processor)

    def test_has_process_method(self):
        """Verify process() method exists and has correct signature."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        # Should handle single value
        result = proc.process(None)
        assert result == 30.0

        result = proc.process(25.0)
        assert result == 25.0

    def test_callable_via_base_class(self):
        """Verify processor can be called via __call__ (inherited)."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        # __call__ should delegate to process()
        result = proc(None)
        assert result == 30.0

    def test_pipeline_chaining(self):
        """Verify processors can be chained with >> operator."""
        proc1 = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )
        proc2 = NumericalVariableImputationProcessor(
            column_name="income", imputation_value=50000.0
        )

        # Should be chainable
        pipeline = proc1 >> proc2
        assert pipeline is not None


class TestSingleColumnPattern:
    """Test single-column architecture."""

    def test_requires_column_name(self):
        """Column name is required parameter."""
        with pytest.raises(ValueError, match="column_name must be a non-empty string"):
            NumericalVariableImputationProcessor(column_name="", strategy="mean")

    def test_single_column_context(self):
        """Processor tracks its column."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")
        assert proc.column_name == "age"

    def test_fit_on_series(self):
        """Can fit on pandas Series (single column)."""
        data = pd.Series([10, 20, 30, None, 50])

        proc = NumericalVariableImputationProcessor(
            column_name="test_col", strategy="mean"
        )
        proc.fit(data)

        assert proc.is_fitted
        assert proc.imputation_value == 27.5  # (10+20+30+50)/4

    def test_fit_on_dataframe_extracts_column(self):
        """Can fit on DataFrame by extracting column."""
        df = pd.DataFrame(
            {"age": [10, 20, 30, None, 50], "income": [1000, 2000, 3000, 4000, 5000]}
        )

        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")
        proc.fit(df)

        assert proc.is_fitted
        assert proc.imputation_value == 27.5

    def test_process_single_value(self):
        """Process method handles single values."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        # Missing value
        assert proc.process(None) == 30.0
        assert proc.process(np.nan) == 30.0

        # Non-missing value
        assert proc.process(25.0) == 25.0
        assert proc.process(35) == 35


class TestFitStrategies:
    """Test different imputation strategies."""

    def test_mean_strategy(self):
        """Mean strategy calculates correct value."""
        data = pd.Series([10, 20, 30, 40])

        proc = NumericalVariableImputationProcessor(column_name="test", strategy="mean")
        proc.fit(data)

        assert proc.imputation_value == 25.0

    def test_median_strategy(self):
        """Median strategy calculates correct value."""
        data = pd.Series([10, 20, 30, 40, 50])

        proc = NumericalVariableImputationProcessor(
            column_name="test", strategy="median"
        )
        proc.fit(data)

        assert proc.imputation_value == 30.0

    def test_mode_strategy(self):
        """Mode strategy calculates correct value."""
        data = pd.Series([10, 20, 20, 30, 40])

        proc = NumericalVariableImputationProcessor(column_name="test", strategy="mode")
        proc.fit(data)

        assert proc.imputation_value == 20.0

    def test_handles_all_nan(self):
        """Handles case where all values are NaN."""
        data = pd.Series([None, np.nan, None])

        proc = NumericalVariableImputationProcessor(column_name="test", strategy="mean")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proc.fit(data)

        assert proc.imputation_value == 0.0


class TestTransformMethod:
    """Test transform method with different input types."""

    def test_transform_series(self):
        """Transform pandas Series."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        data = pd.Series([10, None, 30, np.nan, 50])
        result = proc.transform(data)

        assert result.tolist() == [10, 30.0, 30, 30.0, 50]

    def test_transform_dataframe(self):
        """Transform DataFrame (modifies only target column)."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        df = pd.DataFrame({"age": [10, None, 30], "income": [1000, 2000, 3000]})

        result = proc.transform(df)

        assert result["age"].tolist() == [10, 30.0, 30]
        assert result["income"].tolist() == [1000, 2000, 3000]

    def test_transform_single_value(self):
        """Transform single value (delegates to process)."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        assert proc.transform(None) == 30.0
        assert proc.transform(25.0) == 25.0


class TestGettersAndSetters:
    """Test getter/setter methods."""

    def test_get_imputation_value(self):
        """Can get fitted imputation value."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        assert proc.get_imputation_value() == 30.0

    def test_get_imputation_value_requires_fit(self):
        """Getting value before fit raises error."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")

        with pytest.raises(RuntimeError, match="has not been fitted"):
            proc.get_imputation_value()

    def test_set_imputation_value(self):
        """Can set imputation value."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")

        proc.set_imputation_value(35.0)

        assert proc.is_fitted
        assert proc.imputation_value == 35.0

    def test_set_imputation_value_validates(self):
        """Setting non-numeric value raises error."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")

        with pytest.raises(ValueError, match="must be numeric"):
            proc.set_imputation_value("not a number")

    def test_get_params_deprecated(self):
        """get_params() raises deprecation warning."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", imputation_value=30.0
        )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            params = proc.get_params()

        assert params["column_name"] == "age"
        assert params["imputation_value"] == 30.0


class TestFactoryMethods:
    """Test factory methods for creating processors."""

    def test_from_imputation_dict(self):
        """Create processors from dictionary."""
        impute_dict = {"age": 30.0, "income": 50000.0, "score": 0.5}

        processors = NumericalVariableImputationProcessor.from_imputation_dict(
            impute_dict
        )

        assert len(processors) == 3
        assert "age" in processors
        assert processors["age"].column_name == "age"
        assert processors["age"].imputation_value == 30.0
        assert processors["age"].is_fitted

    def test_from_imputation_dict_validates(self):
        """Factory validates input."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            NumericalVariableImputationProcessor.from_imputation_dict("not a dict")

    def test_from_script_artifacts(self):
        """Load processors from script output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock script output
            impute_dict = {"age": 30.0, "income": 50000.0}

            impute_dict_file = Path(tmpdir) / "impute_dict.pkl"
            with open(impute_dict_file, "wb") as f:
                pkl.dump(impute_dict, f)

            # Load with factory
            processors = NumericalVariableImputationProcessor.from_script_artifacts(
                tmpdir
            )

            assert len(processors) == 2
            assert processors["age"].imputation_value == 30.0
            assert processors["income"].imputation_value == 50000.0

    def test_from_script_artifacts_file_not_found(self):
        """Raises error if file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="impute_dict.pkl not found"):
                NumericalVariableImputationProcessor.from_script_artifacts(tmpdir)


class TestScriptCompatibility:
    """Test compatibility with script artifacts."""

    def test_loads_script_format(self):
        """Can load script output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate script output (missing_value_imputation.py)
            script_output = {"feature1": 10.5, "feature2": 20.3, "feature3": 30.7}

            pkl_file = Path(tmpdir) / "impute_dict.pkl"
            with open(pkl_file, "wb") as f:
                pkl.dump(script_output, f)

            # Load with processor
            processors = NumericalVariableImputationProcessor.from_script_artifacts(
                tmpdir
            )

            # Verify all processors created correctly
            assert len(processors) == 3
            for col, proc in processors.items():
                assert proc.column_name == col
                assert proc.imputation_value == script_output[col]
                assert proc.is_fitted

    def test_processor_output_matches_script_format(self):
        """Processor can save in script-compatible format."""
        # Create processors
        processors = {
            "age": NumericalVariableImputationProcessor("age", imputation_value=30.0),
            "income": NumericalVariableImputationProcessor(
                "income", imputation_value=50000.0
            ),
        }

        # Extract dict (script format)
        impute_dict = {
            col: proc.get_imputation_value() for col, proc in processors.items()
        }

        # Verify format matches script output
        assert isinstance(impute_dict, dict)
        assert impute_dict["age"] == 30.0
        assert impute_dict["income"] == 50000.0


class TestErrorHandling:
    """Test error handling and validation."""

    def test_requires_column_name(self):
        """Requires valid column name."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            NumericalVariableImputationProcessor(column_name="", strategy="mean")

    def test_requires_value_or_strategy(self):
        """Requires either imputation_value or strategy."""
        with pytest.raises(ValueError, match="Either imputation_value or strategy"):
            NumericalVariableImputationProcessor(column_name="age")

    def test_process_before_fit_raises_error(self):
        """Processing before fit raises error."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")

        with pytest.raises(RuntimeError, match="must be fitted"):
            proc.process(None)

    def test_transform_before_fit_raises_error(self):
        """Transforming before fit raises error."""
        proc = NumericalVariableImputationProcessor(column_name="age", strategy="mean")

        with pytest.raises(RuntimeError, match="must be fitted"):
            proc.transform(pd.Series([1, 2, 3]))

    def test_unknown_strategy_raises_error(self):
        """Unknown strategy raises error."""
        proc = NumericalVariableImputationProcessor(
            column_name="age", strategy="unknown"
        )

        data = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown strategy"):
            proc.fit(data)

    def test_column_not_in_dataframe_raises_error(self):
        """Error if column not in DataFrame."""
        proc = NumericalVariableImputationProcessor(
            column_name="missing_col", strategy="mean"
        )

        df = pd.DataFrame({"age": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in DataFrame"):
            proc.fit(df)


class TestArtifactPersistence:
    """Test save/load artifact persistence methods."""

    def test_save_imputation_value(self):
        """Can save imputation value to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = NumericalVariableImputationProcessor(
                column_name="age", imputation_value=30.0
            )

            proc.save_imputation_value(tmpdir)

            # Verify files created
            pkl_file = Path(tmpdir) / "age_impute_value.pkl"
            json_file = Path(tmpdir) / "age_impute_value.json"

            assert pkl_file.exists()
            assert json_file.exists()

            # Verify pickle content
            with open(pkl_file, "rb") as f:
                loaded_val = pkl.load(f)
            assert loaded_val == 30.0

            # Verify JSON content
            with open(json_file, "r") as f:
                json_data = json.load(f)
            assert json_data["column_name"] == "age"
            assert json_data["imputation_value"] == 30.0

    def test_save_requires_fit(self):
        """Cannot save before fitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = NumericalVariableImputationProcessor(
                column_name="age", strategy="mean"
            )

            with pytest.raises(RuntimeError, match="Cannot save before fitting"):
                proc.save_imputation_value(tmpdir)

    def test_load_imputation_value_from_directory(self):
        """Can load imputation value from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save first
            proc1 = NumericalVariableImputationProcessor(
                column_name="age", imputation_value=30.0
            )
            proc1.save_imputation_value(tmpdir)

            # Load in new processor
            proc2 = NumericalVariableImputationProcessor(
                column_name="age", strategy="mean"
            )
            proc2.load_imputation_value(tmpdir)

            assert proc2.is_fitted
            assert proc2.imputation_value == 30.0

    def test_load_imputation_value_from_file(self):
        """Can load imputation value from specific file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save first
            proc1 = NumericalVariableImputationProcessor(
                column_name="age", imputation_value=30.0
            )
            proc1.save_imputation_value(tmpdir)

            # Load from specific file
            proc2 = NumericalVariableImputationProcessor(
                column_name="age", strategy="mean"
            )
            pkl_file = Path(tmpdir) / "age_impute_value.pkl"
            proc2.load_imputation_value(pkl_file)

            assert proc2.is_fitted
            assert proc2.imputation_value == 30.0

    def test_load_file_not_found(self):
        """Loading non-existent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = NumericalVariableImputationProcessor(
                column_name="age", strategy="mean"
            )

            with pytest.raises(FileNotFoundError, match="not found"):
                proc.load_imputation_value(tmpdir)

    def test_load_validates_value(self):
        """Loading invalid value raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid pickle file
            pkl_file = Path(tmpdir) / "age_impute_value.pkl"
            with open(pkl_file, "wb") as f:
                pkl.dump("not a number", f)

            proc = NumericalVariableImputationProcessor(
                column_name="age", strategy="mean"
            )

            with pytest.raises(ValueError, match="must be numeric"):
                proc.load_imputation_value(tmpdir)

    def test_save_load_roundtrip(self):
        """Complete save/load roundtrip preserves value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fit and save
            data = pd.Series([10, 20, 30, 40])
            proc1 = NumericalVariableImputationProcessor(
                column_name="test", strategy="mean"
            )
            proc1.fit(data)
            original_value = proc1.get_imputation_value()
            proc1.save_imputation_value(tmpdir)

            # Load in new processor
            proc2 = NumericalVariableImputationProcessor(
                column_name="test",
                strategy="median",  # Different strategy
            )
            proc2.load_imputation_value(tmpdir)

            # Values should match
            assert proc2.get_imputation_value() == original_value
            assert proc2.get_imputation_value() == 25.0

    def test_save_multiple_processors(self):
        """Can save multiple processors to same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processors = {
                "age": NumericalVariableImputationProcessor(
                    "age", imputation_value=30.0
                ),
                "income": NumericalVariableImputationProcessor(
                    "income", imputation_value=50000.0
                ),
                "score": NumericalVariableImputationProcessor(
                    "score", imputation_value=0.75
                ),
            }

            # Save all
            for proc in processors.values():
                proc.save_imputation_value(tmpdir)

            # Verify all files created
            for col in processors.keys():
                pkl_file = Path(tmpdir) / f"{col}_impute_value.pkl"
                json_file = Path(tmpdir) / f"{col}_impute_value.json"
                assert pkl_file.exists()
                assert json_file.exists()

    def test_load_multiple_processors(self):
        """Can load multiple processors from same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple processors
            original_values = {"age": 30.0, "income": 50000.0, "score": 0.75}

            for col, val in original_values.items():
                proc = NumericalVariableImputationProcessor(col, imputation_value=val)
                proc.save_imputation_value(tmpdir)

            # Load all
            loaded_processors = {}
            for col in original_values.keys():
                proc = NumericalVariableImputationProcessor(col, strategy="mean")
                proc.load_imputation_value(tmpdir)
                loaded_processors[col] = proc

            # Verify all loaded correctly
            for col, proc in loaded_processors.items():
                assert proc.is_fitted
                assert proc.get_imputation_value() == original_values[col]


class TestRealWorldUsage:
    """Test real-world usage patterns."""

    def test_training_inference_workflow(self):
        """Complete training to inference workflow."""
        # Training phase
        train_df = pd.DataFrame(
            {
                "age": [25, 30, None, 40, 35],
                "income": [50000, None, 60000, 70000, 55000],
            }
        )

        # Fit processors
        age_proc = NumericalVariableImputationProcessor(
            column_name="age", strategy="mean"
        )
        age_proc.fit(train_df["age"])

        income_proc = NumericalVariableImputationProcessor(
            column_name="income", strategy="median"
        )
        income_proc.fit(train_df["income"])

        # Save in script format
        impute_dict = {
            "age": age_proc.get_imputation_value(),
            "income": income_proc.get_imputation_value(),
        }

        # Inference phase - load from dict
        processors = NumericalVariableImputationProcessor.from_imputation_dict(
            impute_dict
        )

        # Process single record
        record = {"age": None, "income": 65000}
        age_imputed = processors["age"].process(record["age"])
        income_imputed = processors["income"].process(record["income"])

        assert age_imputed == 32.5  # Mean of [25, 30, 40, 35]
        assert income_imputed == 65000  # Not missing

    def test_dataset_pipeline_pattern(self):
        """Pattern for adding to dataset pipelines."""
        # Load from script artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            impute_dict = {"age": 30.0, "income": 50000.0}
            pkl_file = Path(tmpdir) / "impute_dict.pkl"

            with open(pkl_file, "wb") as f:
                pkl.dump(impute_dict, f)

            processors = NumericalVariableImputationProcessor.from_script_artifacts(
                tmpdir
            )

            # Pattern: Add each processor to dataset
            # for col, proc in processors.items():
            #     dataset.add_pipeline(col, proc)

            # Verify processors are ready
            assert len(processors) == 2
            assert all(proc.is_fitted for proc in processors.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
