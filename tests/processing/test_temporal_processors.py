"""
Unit tests for the temporal preprocessing processors (TSA sequence prep):
``TimeDeltaProcessor``, ``SequencePaddingProcessor``, ``SequenceOrderingProcessor``,
``TemporalMaskProcessor``.

These modules import cleanly (numpy/pandas only) but had zero coverage. They carry real logic —
padding/truncation strategies, argument guards, the fitted-state contract, and mask formats — so
these tests both lift coverage and pin the fail-fast guards the code deliberately added.
"""

import numpy as np
import pandas as pd
import pytest

from cursus.processing.temporal.time_delta_processor import TimeDeltaProcessor
from cursus.processing.temporal.sequence_padding_processor import SequencePaddingProcessor
from cursus.processing.temporal.sequence_ordering_processor import SequenceOrderingProcessor
from cursus.processing.temporal.temporal_mask_processor import TemporalMaskProcessor


class TestTimeDeltaProcessor:
    def test_rejects_unknown_reference_strategy(self):
        with pytest.raises(ValueError):
            TimeDeltaProcessor(reference_strategy="custom")

    def test_missing_reference_field_raises_keyerror(self):
        p = TimeDeltaProcessor(reference_field="ts")
        with pytest.raises(KeyError):
            p.fit({"other": [1, 2, 3]})

    def test_empty_reference_field_raises(self):
        p = TimeDeltaProcessor(reference_field="ts")
        with pytest.raises(ValueError):
            p.fit({"ts": []})

    def test_process_requires_fit(self):
        p = TimeDeltaProcessor(reference_field="ts")
        with pytest.raises(RuntimeError):
            p.process({"ts": [1, 2, 3]})

    def test_most_recent_delta(self):
        p = TimeDeltaProcessor(reference_field="ts", output_field="delta", max_delta=None)
        p.fit({"ts": [10, 20, 30]})
        assert p.reference_time == 30
        out = p.process({"ts": [10, 20, 30]})
        # deltas = reference_time - t
        assert out["delta"] == [20, 10, 0]

    def test_first_delta_strategy(self):
        p = TimeDeltaProcessor(reference_strategy="first", reference_field="ts", max_delta=None)
        p.fit({"ts": [10, 20, 30]})
        assert p.reference_time == 10

    def test_max_delta_caps_values(self):
        p = TimeDeltaProcessor(reference_field="ts", output_field="delta", max_delta=15)
        p.fit({"ts": [0, 20, 30]})  # reference_time = 30
        out = p.process({"ts": [0, 20, 30]})  # raw deltas 30,10,0 -> capped at 15
        assert max(out["delta"]) <= 15


class TestSequencePaddingProcessor:
    def test_rejects_nonpositive_target_length(self):
        with pytest.raises(ValueError):
            SequencePaddingProcessor(target_length=0)

    def test_rejects_negative_axis(self):
        with pytest.raises(ValueError):
            SequencePaddingProcessor(target_length=5, axis=-1)

    def test_process_requires_fit(self):
        with pytest.raises(RuntimeError):
            SequencePaddingProcessor(target_length=3).process([1, 2])

    def test_pad_list_pre(self):
        p = SequencePaddingProcessor(target_length=5, padding_strategy="pre", padding_value=0).fit(None)
        assert p.process([1, 2, 3]) == [0, 0, 1, 2, 3]

    def test_pad_list_post(self):
        p = SequencePaddingProcessor(target_length=5, padding_strategy="post", padding_value=9).fit(None)
        assert p.process([1, 2, 3]) == [1, 2, 3, 9, 9]

    def test_truncate_list_post_keeps_first(self):
        p = SequencePaddingProcessor(target_length=2, truncation_strategy="post").fit(None)
        assert p.process([1, 2, 3, 4]) == [1, 2]

    def test_truncate_list_pre_keeps_last(self):
        p = SequencePaddingProcessor(target_length=2, truncation_strategy="pre").fit(None)
        assert p.process([1, 2, 3, 4]) == [3, 4]

    def test_exact_length_is_unchanged(self):
        p = SequencePaddingProcessor(target_length=3).fit(None)
        assert p.process([1, 2, 3]) == [1, 2, 3]

    def test_numpy_pad_pre(self):
        p = SequencePaddingProcessor(target_length=4, padding_strategy="pre", padding_value=0).fit(None)
        out = p.process(np.array([[1, 1], [2, 2]]))
        assert out.shape == (4, 2)
        assert np.array_equal(out[:2], np.zeros((2, 2)))  # pre-padding rows are zero

    def test_unsupported_input_type_raises(self):
        p = SequencePaddingProcessor(target_length=3).fit(None)
        with pytest.raises(ValueError):
            p.process("not a sequence")


class TestSequenceOrderingProcessor:
    def test_process_requires_fit(self):
        with pytest.raises(RuntimeError):
            SequenceOrderingProcessor().process(pd.DataFrame({"orderDate": [1]}))

    def test_dataframe_sorted_ascending(self):
        p = SequenceOrderingProcessor(sort_field="orderDate", sort_order="ascending").fit(None)
        df = pd.DataFrame({"orderDate": [30, 10, 20], "v": ["c", "a", "b"]})
        out = p.process(df)
        assert list(out["orderDate"]) == [10, 20, 30]
        assert list(out["v"]) == ["a", "b", "c"]

    def test_dataframe_sorted_descending(self):
        p = SequenceOrderingProcessor(sort_order="descending").fit(None)
        out = p.process(pd.DataFrame({"orderDate": [10, 30, 20]}))
        assert list(out["orderDate"]) == [30, 20, 10]

    def test_dataframe_missing_sort_field_raises(self):
        p = SequenceOrderingProcessor(sort_field="ts").fit(None)
        with pytest.raises(ValueError):
            p.process(pd.DataFrame({"other": [1, 2]}))

    def test_numpy_1d_raises_clear_error(self):
        p = SequenceOrderingProcessor().fit(None)
        with pytest.raises(ValueError):
            p.process(np.array([1, 2, 3]))  # must be 2D

    def test_numpy_2d_sorted_by_last_column(self):
        p = SequenceOrderingProcessor(sort_order="ascending").fit(None)
        arr = np.array([[1, 30], [2, 10], [3, 20]])
        out = p.process(arr)
        assert list(out[:, -1]) == [10, 20, 30]

    def test_unsupported_input_type_raises(self):
        p = SequenceOrderingProcessor().fit(None)
        with pytest.raises(ValueError):
            p.process("nope")


class TestTemporalMaskProcessor:
    def test_rejects_bad_output_format(self):
        with pytest.raises(ValueError):
            TemporalMaskProcessor(output_format="hex")

    def test_process_requires_fit(self):
        with pytest.raises(RuntimeError):
            TemporalMaskProcessor().process([1, 0, 2])

    def test_list_boolean_mask(self):
        p = TemporalMaskProcessor(padding_value=0, output_format="boolean").fit(None)
        assert p.process([1, 0, 2, 0]) == [True, False, True, False]

    def test_list_int_mask(self):
        p = TemporalMaskProcessor(padding_value=0, output_format="int").fit(None)
        assert p.process([1, 0, 2]) == [1, 0, 1]

    def test_numpy_2d_rows_any_nonpad(self):
        p = TemporalMaskProcessor(padding_value=0, output_format="boolean").fit(None)
        arr = np.array([[0, 0], [0, 5], [0, 0]])
        out = p.process(arr)
        assert list(out) == [False, True, False]

    def test_float_format(self):
        p = TemporalMaskProcessor(padding_value=0, output_format="float").fit(None)
        out = p.process(np.array([1, 0, 3]))
        assert out.dtype == float
        assert list(out) == [1.0, 0.0, 1.0]

    def test_causal_mask_is_lower_triangular(self):
        p = TemporalMaskProcessor().fit(None)
        m = p.create_causal_mask(3)
        assert m.shape == (3, 3)
        assert np.array_equal(m, np.tril(np.ones((3, 3))))

    def test_unsupported_input_type_raises(self):
        p = TemporalMaskProcessor().fit(None)
        with pytest.raises(ValueError):
            p.process("nope")
