"""
Batch-C migration parity helper (not a test module — a callable gate used during S3 migration).

Given a builder module name + a config kwargs builder, compare a candidate (current, possibly
migrated) builder's session-independent constituents — get_inputs ProcessingInputs, get_outputs
ProcessingOutputs, get_script_path, job arguments — against a snapshot captured from the
pre-migration builder source at git HEAD~ (or a frozen expectation). Used by migrate-and-verify.

This is intentionally NOT collected by pytest (underscore prefix); it's imported ad hoc.
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")


def _stable(v):
    return repr(v.expr) if hasattr(v, "expr") else str(v)


def processing_input_sig(pi):
    return (pi.input_name, _stable(pi.source), _stable(pi.destination))


def processing_output_sig(po):
    return (po.output_name, _stable(po.source), _stable(po.destination))


def constituents(builder, sample_inputs, sample_outputs):
    """Capture the session-independent constituents a Processing step is assembled from."""
    out = {}
    ins = builder._get_inputs(dict(sample_inputs))
    out["inputs"] = sorted(processing_input_sig(p) for p in ins)
    outs = builder._get_outputs(dict(sample_outputs))
    out["outputs"] = sorted(processing_output_sig(p) for p in outs)
    out["script_path"] = builder.config.get_script_path()
    out["job_args"] = (
        builder._get_job_arguments() if hasattr(builder, "_get_job_arguments") else None
    )
    return out
