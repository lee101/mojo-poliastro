"""Port of `poliastro.core.propagation.vallado`."""

import numpy as np

from ..._dispatch import Batch
from ..._lib import addr, f64, fn

__all__ = ["vallado"]


def vallado(k, r0, v0, tof, numiter):
    r"""Universal-variable Lagrange coefficients ``(f, g, fdot, gdot)``.

    Returns ``0`` for a converged solve, ``-3`` when `numiter` is exhausted
    (upstream raises ``RuntimeError`` there).
    """
    a = np.asarray(r0, dtype=np.float64)
    b_ = np.asarray(v0, dtype=np.float64)
    t = np.asarray(tof, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-1], b_.shape[:-1], t.shape)
    scalar = not shape
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    r = f64(np.broadcast_to(a, shape + (3,)).reshape(-1))
    v = f64(np.broadcast_to(b_, shape + (3,)).reshape(-1))
    tt = f64(np.broadcast_to(t, shape).reshape(-1))
    out = np.empty((n, 4), dtype=np.float64)
    codes = np.empty(n, dtype=np.int64)
    status = fn("mp_vallado_uni")(
        float(k), addr(r), addr(v), addr(tt), int(numiter), addr(out), addr(codes), n
    )
    if status == -3:
        raise RuntimeError("Maximum number of iterations reached")
    out = out.reshape(shape + (4,))
    if scalar:
        return tuple(out.reshape(4).tolist())
    return tuple(out[..., j] for j in range(4))
