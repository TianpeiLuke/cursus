"""
Top-level pytest configuration for the cursus test suite.

Offline heavy-dependency shims (torch / xgboost)
------------------------------------------------
Several script modules import heavy ML libraries at MODULE level — e.g.
``cursus.steps.scripts.xgboost_training`` does ``import xgboost as xgb`` and
``cursus.processing.numerical`` pulls in ``streaming_numerical_imputation_processor``
which does ``import torch``. The tests that exercise those modules ``@patch`` the heavy
symbols (``xgb.Booster``, ``xgb.DMatrix``, …) and never call the real library, but the
bare ``import`` still has to succeed for the test module to be COLLECTED. When the suite
runs offline (``PYTHONNOUSERSITE=1`` in the Brazil build, where torch/xgboost are not in
the version set) those imports raise ``ModuleNotFoundError`` and pytest reports a
*collection error*, not a skip.

This conftest installs lightweight stand-ins in ``sys.modules`` for any heavy lib that is
genuinely absent, BEFORE test modules are imported — so the patched tests collect and run
offline, while a real install (when present) is left untouched.

Subtlety preserved from the original per-file shim (FZ 31e1d3g3 Phase E test hardening):
``scipy``'s ``array_api_compat`` probes ``issubclass(cls, sys.modules["torch"].Tensor)``.
A bare ``MagicMock`` makes ``.Tensor`` a Mock (not a class), which raises
"issubclass() arg 2 must be a class" in UNRELATED tests run later in the same process
(the ``sys.modules`` entry leaks process-wide). Giving the fake module real class
attributes for the names array-API libraries probe keeps ``issubclass`` valid.
"""

import sys
from unittest.mock import MagicMock


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    stub = MagicMock()
    # Real class/attr objects for the names scipy / array-API libs introspect, so a
    # leaked stub never breaks issubclass()/dtype checks in unrelated tests.
    stub.Tensor = type("Tensor", (), {})
    stub.bool = bool
    sys.modules["torch"] = stub

    # Several processing modules do `from torch.utils.data import IterableDataset` /
    # `from torch.utils.data._utils.collate import default_collate` / `from torch.nn.utils.rnn
    # import pad_sequence`. `from torch.<sub> import X` requires each dotted package to be a
    # real sys.modules entry — a flat MagicMock alone raises "torch is not a package". Register
    # the submodule chain as MagicMock modules so the from-imports resolve to mocks.
    for submodule in (
        "torch.utils",
        "torch.utils.data",
        "torch.utils.data._utils",
        "torch.utils.data._utils.collate",
        "torch.nn",
        "torch.nn.utils",
        "torch.nn.utils.rnn",
    ):
        sys.modules[submodule] = MagicMock()


def _install_xgboost_stub() -> None:
    if "xgboost" in sys.modules:
        return
    # A plain MagicMock: xgb.Booster / xgb.DMatrix must be CALLABLE with arbitrary args
    # (some tests construct them without patching, e.g. xgb.DMatrix(X, label=y)). Unlike
    # torch.Tensor (which scipy probes via issubclass and so needs a real class), nothing
    # introspects xgboost's classes, so the auto-callable Mock is the correct stand-in —
    # this mirrors the original per-file torch shim.
    sys.modules["xgboost"] = MagicMock()


# Install at import time (conftest is imported before any test module is collected).
_install_torch_stub()
_install_xgboost_stub()
