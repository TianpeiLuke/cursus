"""
Step utilities module.

Home for shared, step-adjacent helpers that are NOT per-step artifacts (no
``.step.yaml`` row, not discovered by the catalog). Introduced as part of the
FZ 31 folder simplification, which emptied the per-step data folders:

- ``S3PathHandler`` (from ``s3_utils``) — S3 path parsing/joining used by the
  synthesized step builders.

Import it from here (``cursus.steps.utils``); the former
``cursus.steps.builders`` re-export was removed with that folder.
"""

from .s3_utils import S3PathHandler

__all__ = ["S3PathHandler"]
