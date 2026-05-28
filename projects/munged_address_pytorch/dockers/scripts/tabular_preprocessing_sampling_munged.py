"""
Custom TabularPreprocessing script for Munged Address — Sampling Phase.

Reads bad addresses (DATA) + good addresses (DATA_SECONDARY),
deduplicates on (saddr, marketplaceId), adds __cohort__ column,
and emits reference_counts.json for downstream StratifiedSampling.

Source: FZ 29d16b (adapted from 04_process_address.ipynb)
"""

# TODO: Implement — see FZ 29d16b for full script design
# Placeholder for Sprint 2, Task 2.1


def main(input_paths, output_paths, environ_vars, job_args, logger=None):
    raise NotImplementedError(
        "tabular_preprocessing_sampling_munged.py — not yet implemented"
    )
