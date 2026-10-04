"""Port of `poliastro._math.linalg` -- see `src/mojopoliastro/linalg.mojo`."""

import numpy as np

from .._lib import addr, f64, fn

__all__ = ["norm"]


def norm(arr):
    """Euclidean norm of a 3-vector, or row-wise norm of an (..., 3) array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError("norm expects a trailing axis of length 3")
    flat = a.reshape(-1, 3)
    n = flat.shape[0]
    out = np.empty(n, dtype=np.float64)
    fn("mp_norm")(addr(f64(flat)), addr(out), n)
    out = out.reshape(a.shape[:-1])
    return float(out) if a.ndim == 1 else out
