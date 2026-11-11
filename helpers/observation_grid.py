"""
Observation tensors ``X`` for seqid / param RNNs: one numeric pad token, no NaN slabs.

This module is NumPy-only so lightweight scripts (e.g. ``scripts/convert_real_pkl_to_npy.py``)
can use it without importing TensorFlow via ``data_prep``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Must match default ``mask_value`` in :func:`helpers.data_prep.get_features_from_dataframe`
# and ``trial_pad_value`` in :class:`helpers.rnn.paramRNN.paramRNN`.
DEFAULT_OBSERVATION_MASK_VALUE: float = -1.0


def finalize_observation_tensor(
    arr: np.ndarray,
    *,
    mask_value: float = DEFAULT_OBSERVATION_MASK_VALUE,
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Coerce dtype and replace any non-finite values with ``mask_value`` in model-input
    feature arrays ``(N, S, T, F)`` (or per-agent ``(S, T, F)``).

    Use as the last step before saving **X**; do not use for label grids that rely on
    NaN + ``torch.isfinite`` for loss masking.
    """
    out = np.asarray(arr, dtype=dtype, order="C")
    if np.isfinite(out).all():
        return out
    out = out.copy()
    fill = np.dtype(dtype).type(mask_value)
    out[~np.isfinite(out)] = fill
    return out
